"""Model providers for video-intelligence: Gemini (google-genai) and OpenRouter (OpenAI-compatible chat completions).

A ProviderChain tries providers in order. Every provider makes at most MAX_ATTEMPTS model requests per logical
request (e.g. "pass 1"), then the chain falls over to the next provider, so a busy server cannot burn the Gemini
free-tier daily quota. A provider that reports a daily quota is marked dead for the rest of the run.

Callers pass build(provider) -> dict(video=Path, fps=float|None, prompt=str, audio=Path|None) so each provider can get
its own video variant (e.g. OpenRouter gets a slowed-down, timecoded detail clip because it cannot set sampling fps).
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MAX_ATTEMPTS = 2           # model requests per logical request per provider
TRANSIENT_WAIT_S = 8       # one short wait between attempts on 429/5xx/disconnects
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_GEN_URL = "https://openrouter.ai/api/v1/generation"
DEFAULT_OPENROUTER_MODEL = "qwen/qwen3.8-omni-flash"
OR_MAX_VIDEO_BYTES = 12 * 1024 * 1024   # re-encode above this (base64 adds ~33%)
OR_MAX_LONG_EDGE = 1280

TRANSIENT = ("Server disconnected", "RemoteProtocolError", "ReadTimeout", "ConnectError", "timed out", "429", "500",
             "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED", "overloaded",
             "high demand", "rate limit", "Rate limit")


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


class ProviderError(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind  # daily | transient | rejected | parse | config


def parse_json_loose(text: str):
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        return json.loads(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("no JSON object found")


def classify(msg: str) -> str:
    if "PerDay" in msg or "per day" in msg.lower():
        return "daily"
    if any(code in msg for code in TRANSIENT):
        return "transient"
    return "rejected"


class RunStats:
    """Counts every model request made in this run (successful or not), per provider."""

    def __init__(self):
        self.requests: list[dict] = []

    def add(self, provider: str, model: str, label: str, attempt: int, outcome: str, usage: dict | None = None):
        row = {"n": len(self.requests) + 1, "provider": provider, "model": model, "label": label,
               "attempt": attempt, "outcome": outcome}
        if usage:
            row["usage"] = usage
        self.requests.append(row)
        log(f"[req #{row['n']}] {provider}:{model} {label} attempt {attempt} -> {outcome}")

    def summary(self) -> dict:
        per: dict[str, int] = {}
        cost = 0.0
        for r in self.requests:
            per[r["provider"]] = per.get(r["provider"], 0) + 1
            cost += float((r.get("usage") or {}).get("cost_usd") or 0)
        return {"total_requests": len(self.requests), "per_provider": per,
                "openrouter_cost_usd": round(cost, 6), "requests": self.requests}


# ----------------------------------------------------------------------------- Gemini

def gemini_usage(response) -> dict:
    u = getattr(response, "usage_metadata", None)
    if not u:
        return {}
    out = {}
    for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count"):
        v = getattr(u, k, None)
        if v is not None:
            out[k] = v
    for d in getattr(u, "prompt_tokens_details", None) or []:
        mod = getattr(getattr(d, "modality", None), "name", str(getattr(d, "modality", "")))
        out[f"prompt_{mod.lower()}_tokens"] = getattr(d, "token_count", None)
    return out


class GeminiProvider:
    name = "gemini"
    max_attempts = MAX_ATTEMPTS

    def __init__(self, client, types, model: str, make_part, stats: RunStats):
        self.client, self.types, self.model, self.make_part, self.stats = client, types, model, make_part, stats
        self.dead = False
        self.uploads: list[str] = []
        self._parts: dict = {}

    def _part(self, video: Path, fps):
        key = (str(video), fps)
        if key not in self._parts:
            part, up = self.make_part(video, fps)
            if up:
                self.uploads.append(up)
            self._parts[key] = part
        return self._parts[key]

    def generate(self, label: str, schema: dict | None, req: dict):
        t = self.types
        part = self._part(req["video"], req.get("fps"))
        if schema is not None:
            configs = [("schema", t.GenerateContentConfig(response_mime_type="application/json",
                                                         response_json_schema=schema, temperature=0.2)),
                       ("json-mime", t.GenerateContentConfig(response_mime_type="application/json", temperature=0.2))]
        else:
            configs = [("text", None)]
        ci, last = 0, None
        for attempt in range(1, self.max_attempts + 1):
            cname, config = configs[min(ci, len(configs) - 1)]
            try:
                kw = {"config": config} if config is not None else {}
                resp = self.client.models.generate_content(
                    model=self.model, contents=t.Content(parts=[part, t.Part(text=req["prompt"])]), **kw)
                usage = gemini_usage(resp)
                text = resp.text or ""
                if schema is None:
                    self.stats.add(self.name, self.model, label, attempt, "ok", usage)
                    return text, {"provider": self.name, "model": self.model, "config": cname, "usage": usage}
                try:
                    data = parse_json_loose(text)
                except ValueError as error:
                    self.stats.add(self.name, self.model, label, attempt, f"parse-fail ({error})", usage)
                    last = ProviderError("parse", str(error))
                    ci += 1
                    continue
                self.stats.add(self.name, self.model, label, attempt, "ok", usage)
                return data, {"provider": self.name, "model": self.model, "config": cname, "usage": usage}
            except Exception as error:
                msg = str(error)
                kind = classify(msg)
                self.stats.add(self.name, self.model, label, attempt, f"{kind}: {msg[:90]}")
                last = ProviderError(kind, msg[:300])
                if kind == "daily":
                    self.dead = True
                    raise ProviderError("daily", f"{self.model}: daily free-tier quota exhausted") from error
                if kind == "rejected":
                    ci += 1  # e.g. schema too complex -> plain JSON mode
                elif attempt < self.max_attempts:
                    time.sleep(TRANSIENT_WAIT_S)
        raise last or ProviderError("rejected", "no attempts made")

    def cleanup(self, keep: bool = False):
        if keep:
            return
        for name in self.uploads:
            try:
                self.client.files.delete(name=name)
                log("[info] Deleted temporary Gemini upload.")
            except Exception as error:
                log(f"[warning] Could not delete temporary upload: {error}")


# ----------------------------------------------------------------------------- OpenRouter

def _probe_wh(path: Path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=False).stdout.strip()
    try:
        w, h = (int(x) for x in out.split(",")[:2])
        return w, h
    except ValueError:
        return 0, 0


def _has_audio(path: Path) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=False).stdout.strip()
    return bool(out)


class OpenRouterProvider:
    name = "openrouter"
    max_attempts = MAX_ATTEMPTS

    def __init__(self, model: str, stats: RunStats, timeout: int = 300, send_audio: bool = True):
        self.model, self.stats, self.timeout, self.send_audio = model, stats, timeout, send_audio
        self.key = os.environ.get("OPENROUTER_API_KEY", "")
        self.dead = not self.key
        self.tmp = tempfile.TemporaryDirectory(prefix="vi-or-")
        self._cache: dict = {}
        self.json_schema_ok = True  # flips to False if the endpoint rejects response_format json_schema

    def _video_payload(self, video: Path) -> tuple[str, dict]:
        """Data URL for the video; re-encode (H.264, long edge <= 1280, crf 26) only when too big."""
        key = ("v", str(video))
        if key in self._cache:
            return self._cache[key]
        w, h = _probe_wh(video)
        src, note = video, {"reencoded": False}
        if video.stat().st_size > OR_MAX_VIDEO_BYTES or max(w, h) > OR_MAX_LONG_EDGE:
            out = Path(self.tmp.name) / f"{video.stem}_or.mp4"
            scale = (f"scale='if(gt(iw,ih),min({OR_MAX_LONG_EDGE},iw),-2)':'if(gt(iw,ih),-2,min({OR_MAX_LONG_EDGE},ih))'")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vf", scale,
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-c:a", "aac", "-b:a", "96k",
                            str(out)], check=True)
            src, note = out, {"reencoded": True}
        note["bytes"] = src.stat().st_size
        url = "data:video/mp4;base64," + base64.b64encode(src.read_bytes()).decode()
        self._cache[key] = (url, note)
        return url, note

    def _audio_payload(self, video: Path) -> str | None:
        key = ("a", str(video))
        if key in self._cache:
            return self._cache[key]
        data = None
        if _has_audio(video):
            out = Path(self.tmp.name) / f"{video.stem}_audio.mp3"
            res = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vn",
                                  "-ac", "1", "-ar", "24000", "-b:a", "64k", str(out)], capture_output=True, check=False)
            if res.returncode == 0:
                data = base64.b64encode(out.read_bytes()).decode()
        self._cache[key] = data
        return data

    def _post(self, payload: dict) -> dict:
        import httpx
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                   "X-Title": "video-intelligence"}
        r = httpx.post(OPENROUTER_URL, headers=headers, json=payload, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} {r.text[:400]}")
        data = r.json()
        if data.get("error"):
            err = data["error"]
            raise RuntimeError(f"{err.get('code')} {str(err.get('message'))[:400]}")
        return data

    def _generation_cost(self, gen_id: str) -> dict:
        """Authoritative cost from /api/v1/generation (not a model request; free). Best effort."""
        import httpx
        for wait in (2, 4, 8):
            time.sleep(wait)
            try:
                r = httpx.get(OPENROUTER_GEN_URL, params={"id": gen_id},
                              headers={"Authorization": f"Bearer {self.key}"}, timeout=30)
                if r.status_code == 200:
                    d = r.json().get("data", {})
                    keep = ("total_cost", "native_tokens_prompt", "native_tokens_completion", "native_tokens_reasoning",
                            "tokens_prompt", "tokens_completion", "provider_name", "generation_time", "latency")
                    return {k: d.get(k) for k in keep if k in d}
            except Exception:
                pass
        return {}

    def generate(self, label: str, schema: dict | None, req: dict):
        url, vnote = self._video_payload(req["video"])
        audio = self._audio_payload(req["video"]) if (self.send_audio and req.get("audio", True)) else None
        last = None
        for attempt in range(1, self.max_attempts + 1):
            prompt = req["prompt"]
            content = [{"type": "video_url", "video_url": {"url": url}}]
            if audio:
                content.append({"type": "input_audio", "input_audio": {"data": audio, "format": "mp3"}})
                prompt = ("(The second attachment is the same clip's soundtrack as audio, for sound design and "
                          "speech; the video may or may not carry audio for you.)\n\n" + prompt)
            payload = {"model": self.model, "messages": [{"role": "user", "content": content}],
                       "temperature": 0.2, "max_tokens": 32000, "usage": {"include": True}}
            mode = "text"
            if schema is not None:
                if self.json_schema_ok and attempt == 1:
                    payload["response_format"] = {"type": "json_schema",
                                                  "json_schema": {"name": "analysis", "strict": False, "schema": schema}}
                    mode = "json_schema"
                else:  # JSON-in-prompt + loose parse
                    prompt += ("\n\nReturn ONE JSON object only (no prose, no code fence) that follows this JSON "
                               "schema exactly:\n" + json.dumps(schema, separators=(",", ":")))
                    payload["response_format"] = {"type": "json_object"}
                    mode = "json-in-prompt"
            content.append({"type": "text", "text": prompt})
            try:
                data = self._post(payload)
            except Exception as error:
                msg = str(error)
                kind = classify(msg)
                if kind == "rejected" and mode == "json_schema":
                    self.json_schema_ok = False
                self.stats.add(self.name, self.model, label, attempt, f"{kind} [{mode}]: {msg[:120]}")
                last = ProviderError(kind, msg[:300])
                if kind == "transient" and attempt < self.max_attempts:
                    time.sleep(TRANSIENT_WAIT_S)
                continue
            u = data.get("usage") or {}
            usage = {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                     "total_tokens": u.get("total_tokens"), "cost_usd": u.get("cost"),
                     "prompt_details": u.get("prompt_tokens_details"),
                     "completion_details": u.get("completion_tokens_details"), "generation_id": data.get("id")}
            if data.get("id"):
                gen = self._generation_cost(data["id"])
                if gen:
                    usage["generation"] = gen
                    if usage["cost_usd"] is None and gen.get("total_cost") is not None:
                        usage["cost_usd"] = gen["total_cost"]
            choice = (data.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or ""
            info = {"provider": self.name, "model": self.model, "config": mode, "usage": usage,
                    "finish_reason": choice.get("finish_reason"), "video_payload": vnote, "audio_part": bool(audio)}
            if schema is None:
                self.stats.add(self.name, self.model, label, attempt, "ok", usage)
                return text, info
            try:
                parsed = parse_json_loose(text)
                self.stats.add(self.name, self.model, label, attempt, f"ok [{mode}]", usage)
                return parsed, info
            except ValueError as error:
                self.stats.add(self.name, self.model, label, attempt,
                               f"parse-fail [{mode}] ({error}; finish={choice.get('finish_reason')})", usage)
                last = ProviderError("parse", str(error))
        raise last or ProviderError("rejected", "no attempts made")

    def cleanup(self, keep: bool = False):
        self.tmp.cleanup()


# ----------------------------------------------------------------------------- chain

class ProviderChain:
    def __init__(self, providers: list, stats: RunStats):
        self.providers, self.stats = providers, stats

    @property
    def names(self) -> list[str]:
        return [f"{p.name}:{p.model}" for p in self.providers]

    def generate(self, label: str, schema: dict | None, build):
        errors = []
        for p in self.providers:
            if p.dead:
                continue
            try:
                log(f"[info] {label}: {p.name}:{p.model} ...")
                return p.generate(label, schema, build(p))
            except ProviderError as error:
                errors.append(f"{p.name}:{p.model} {error.kind}: {str(error)[:200]}")
                if error.kind == "transient":
                    p.max_attempts = 1  # still busy later in this run? one probe only, then fall over
                log(f"[warning] {label}: {p.name}:{p.model} gave up ({error.kind}); trying next provider")
        raise RuntimeError(f"{label}: all providers failed -> " + " | ".join(errors) if errors
                           else f"{label}: no usable provider (check API keys / --provider)")

    def cleanup(self, keep: bool = False):
        for p in self.providers:
            p.cleanup(keep)


def build_chain(args, stats: RunStats, make_gemini_part=None, gemini_client=None, gemini_types=None) -> ProviderChain:
    want = getattr(args, "provider", "auto")
    providers: list = []
    if want in ("auto", "gemini") and gemini_client is not None:
        providers.append(GeminiProvider(gemini_client, gemini_types, args.model, make_gemini_part, stats))
        fb = getattr(args, "fallback_model", "")
        if fb and fb != args.model:
            providers.append(GeminiProvider(gemini_client, gemini_types, fb, make_gemini_part, stats))
    if want in ("auto", "openrouter") and os.environ.get("OPENROUTER_API_KEY"):
        providers.append(OpenRouterProvider(args.openrouter_model, stats, timeout=max(args.timeout, 300),
                                            send_audio=not getattr(args, "no_audio_part", False)))
    return ProviderChain(providers, stats)
