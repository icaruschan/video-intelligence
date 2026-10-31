#!/usr/bin/env python3
"""Local, deterministic measurements for edit mode (ffmpeg/ffprobe + numpy).

Everything returned here is tagged basis="measured" and is passed to Gemini as ground
truth. Nothing in this module calls a network service.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    import numpy as np
except ImportError:  # numpy is optional; onsets/palette are skipped without it
    np = None

SCENE_SCALE_WIDTH = 320
AUDIO_SR = 22050


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} not found on PATH; edit mode needs ffmpeg for local measurements")


def _fraction(value: str | None) -> float | None:
    if not value or value in ("0/0", "N/A"):
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        return float(num) / float(den) if float(den) else None
    return float(value)


def probe(path: Path) -> dict:
    res = _run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)])
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {res.stderr.strip()[:300]}")
    data = json.loads(res.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    if not video:
        raise RuntimeError("No video stream found")
    width, height = int(video.get("width", 0)), int(video.get("height", 0))
    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    if abs(rotation) in (90, 270):
        width, height = height, width
    fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    ratio = width / height if height else 0
    if abs(ratio - 9 / 16) < 0.02:
        aspect = "9:16"
    elif abs(ratio - 16 / 9) < 0.02:
        aspect = "16:9"
    elif abs(ratio - 1) < 0.02:
        aspect = "1:1"
    elif abs(ratio - 4 / 5) < 0.02:
        aspect = "4:5"
    else:
        aspect = f"{width}:{height}"
    return {
        "duration_s": round(duration, 3),
        "width": width,
        "height": height,
        "aspect": aspect,
        "fps": round(fps, 3) if fps else None,
        "video_codec": video.get("codec_name"),
        "pix_fmt": video.get("pix_fmt"),
        "has_audio": audio is not None,
        "audio_codec": audio.get("codec_name") if audio else None,
        "audio_channels": audio.get("channels") if audio else None,
        "size_mb": round(float(data.get("format", {}).get("size", 0)) / 1024 / 1024, 2),
    }


def scene_scores(path: Path) -> list[tuple[float, float]]:
    """Per-frame ffmpeg scene-change score (0..1) as (time_s, score)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "scores.txt"
        vf = f"scale={SCENE_SCALE_WIDTH}:-2,select='gte(scene,0)',metadata=print:key=lavfi.scene_score:file={out}"
        res = _run(["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error", "-i", str(path),
                    "-vf", vf, "-an", "-f", "null", "-"])
        if res.returncode != 0 or not out.exists():
            raise RuntimeError(f"ffmpeg scene detection failed: {res.stderr.strip()[:300]}")
        scores: list[tuple[float, float]] = []
        current_t = None
        for line in out.read_text().splitlines():
            if line.startswith("frame:"):
                for token in line.split():
                    if token.startswith("pts_time:"):
                        current_t = float(token.split(":", 1)[1])
            elif line.startswith("lavfi.scene_score=") and current_t is not None:
                scores.append((current_t, float(line.split("=", 1)[1])))
        return scores


def detect_cuts(scores: list[tuple[float, float]], duration: float, threshold: float,
                soft_threshold: float, min_gap: float) -> dict:
    """Hard cuts = score >= threshold; nearby hits within min_gap merge (flashes, 2-frame cuts).
    Soft events = soft_threshold..threshold (dissolves, whips, big motion, flashes)."""
    def cluster(hits: list[tuple[float, float]]) -> list[dict]:
        clusters: list[list[tuple[float, float]]] = []
        for t, s in hits:
            if clusters and t - clusters[-1][-1][0] <= min_gap:
                clusters[-1].append((t, s))
            else:
                clusters.append([(t, s)])
        events = []
        for c in clusters:
            t_peak, s_peak = max(c, key=lambda x: x[1])
            events.append({"t": round(c[0][0], 3), "t_peak": round(t_peak, 3), "score": round(s_peak, 3),
                           "frames": len(c), "span_s": round(c[-1][0] - c[0][0], 3)})
        return events

    # Ignore the very last frames (end-of-stream artefacts give score 1.0)
    usable = [(t, s) for t, s in scores if t < duration - 0.1] if duration else scores
    hard = cluster([(t, s) for t, s in usable if s >= threshold])
    hard_times = [e["t"] for e in hard]
    soft = [e for e in cluster([(t, s) for t, s in usable if soft_threshold <= s < threshold])
            if all(abs(e["t"] - h) > min_gap for h in hard_times)]

    bounds = [0.0] + hard_times + [duration]
    shots = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        if end - start <= 0:
            continue
        shots.append({"index": i + 1, "start": round(start, 3), "end": round(end, 3), "dur": round(end - start, 3)})
    durs = [s["dur"] for s in shots]
    durs_sorted = sorted(durs)
    median = durs_sorted[len(durs_sorted) // 2] if durs_sorted else 0
    return {
        "threshold": threshold,
        "soft_threshold": soft_threshold,
        "min_gap_s": min_gap,
        "hard_cuts": hard,
        "soft_events": soft,
        "shots": shots,
        "shot_count": len(shots),
        "avg_shot_s": round(sum(durs) / len(durs), 3) if durs else None,
        "median_shot_s": round(median, 3) if durs else None,
        "min_shot_s": round(min(durs), 3) if durs else None,
        "max_shot_s": round(max(durs), 3) if durs else None,
        "cuts_per_sec": round(len(hard) / duration, 3) if duration else None,
        "cuts_per_10s": round(len(hard) / duration * 10, 2) if duration else None,
    }


def motion_curve(scores: list[tuple[float, float]], duration: float, window: float = 1.0) -> list[dict]:
    """Mean scene score per window: a rough 'visual change energy' curve (includes motion, not just cuts)."""
    if not scores or not duration:
        return []
    buckets: dict[int, list[float]] = {}
    for t, s in scores:
        if t < duration - 0.1:
            buckets.setdefault(int(t // window), []).append(s)
    return [{"t": round(k * window, 2), "energy": round(sum(v) / len(v), 4), "max": round(max(v), 3)}
            for k, v in sorted(buckets.items())]


def densest_window(scores: list[tuple[float, float]], cuts: list[dict], duration: float, length: float,
                   target: str = "motion") -> dict | None:
    """Pick the window for the high-fps detail pass.
    target='motion' (default): most visual change *inside shots* (frames near hard cuts excluded), i.e. where
      graphics, captions, zooms and slides animate. Cut times are already measured exactly, so this is
      where extra frames buy the most.
    target='cuts': most hard cuts + overall change."""
    if duration <= length * 1.2:
        return {"start": 0.0, "end": round(duration, 3), "target": target, "reason": "clip shorter than window"}
    cut_times = [c["t"] for c in cuts]
    near_cut = lambda ts: any(-0.05 <= ts - c <= 0.1 for c in cut_times)
    best = None
    t = 0.0
    while t + length <= duration - 0.1:
        n_cuts = sum(1 for c in cut_times if t <= c < t + length)
        if target == "cuts":
            vals = [s for (ts, s) in scores if t <= ts < t + length]
            score = n_cuts + (sum(vals) / len(vals) if vals else 0) * 20
        else:
            vals = [s for (ts, s) in scores if t <= ts < t + length and not near_cut(ts)]
            score = sum(vals) / len(vals) if vals else 0
        if best is None or score > best[0]:
            best = (score, t, n_cuts)
        t += 0.5
    if not best:
        return None
    return {"start": round(best[1], 3), "end": round(best[1] + length, 3), "cuts_in_window": best[2], "target": target,
            "reason": "most within-shot visual change (cut frames excluded)" if target == "motion"
            else "highest cut count + visual-change energy", "score": round(best[0], 4)}


def loudness(path: Path) -> dict:
    res = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn",
                "-af", "ebur128=peak=true", "-f", "null", "-"])
    text = res.stderr
    out: dict = {}
    summary = text.split("Summary:")[-1] if "Summary:" in text else ""
    for line in summary.splitlines():
        line = line.strip()
        for key, name in (("I:", "integrated_lufs"), ("LRA:", "loudness_range_lu"), ("Peak:", "true_peak_dbtp")):
            if line.startswith(key):
                try:
                    out[name] = float(line.split()[1])
                except (IndexError, ValueError):
                    pass
    # Momentary loudness (400 ms window) per ~0.5s for an energy curve. (Short-term S uses a 3 s window and reads
    # -120 LUFS for the first 3 s of every file, which models mistook for silence.)
    curve = []
    for line in text.splitlines():
        if "Parsed_ebur128" in line and " t:" in line and " M:" in line:
            try:
                t = float(line.split(" t:")[1].split()[0])
                s = float(line.split(" M:")[1].split()[0])
                if t >= 0.4 and abs(t * 2 - round(t * 2)) < 0.051:
                    curve.append({"t": round(t, 1), "momentary_lufs": s})
            except (IndexError, ValueError):
                continue
    out["momentary_curve_0p5s"] = curve
    return out


def silences(path: Path, noise_db: float = -35, min_dur: float = 0.25) -> list[dict]:
    res = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn",
                "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}", "-f", "null", "-"])
    spans, start = [], None
    for line in res.stderr.splitlines():
        if "silence_start:" in line:
            start = float(line.split("silence_start:")[1].split()[0])
        elif "silence_end:" in line and start is not None:
            end = float(line.split("silence_end:")[1].split()[0])
            spans.append({"start": round(start, 3), "end": round(end, 3), "dur": round(end - start, 3)})
            start = None
    return spans


def _decode_audio(path: Path):
    res = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1",
                          "-ar", str(AUDIO_SR), "-f", "f32le", "-"], capture_output=True, check=False)
    if res.returncode != 0 or not res.stdout:
        return None
    return np.frombuffer(res.stdout, dtype=np.float32)


def onsets(path: Path, max_onsets: int = 200) -> dict:
    """Spectral-flux onset detection (numpy only). Rough: catches transients like hits, pops, clicks,
    drum hits and plosives; it cannot tell SFX from voice or music. earshot will replace this."""
    if np is None:
        return {"available": False, "reason": "numpy not installed"}
    y = _decode_audio(path)
    if y is None or len(y) < AUDIO_SR // 2:
        return {"available": False, "reason": "no decodable audio"}
    n_fft, hop = 1024, 256
    frames = 1 + (len(y) - n_fft) // hop
    if frames <= 2:
        return {"available": False, "reason": "audio too short"}
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    window = np.hanning(n_fft).astype(np.float32)
    mag = np.abs(np.fft.rfft(y[idx] * window, axis=1))
    logmag = np.log1p(10 * mag)
    flux = np.maximum(0, np.diff(logmag, axis=0)).sum(axis=1)
    flux = np.concatenate([[0], flux])
    rms = np.sqrt((y[idx] ** 2).mean(axis=1))
    # adaptive threshold: local median + k * local MAD-ish spread
    w = 16
    padded = np.pad(flux, (w, w), mode="edge")
    local_med = np.array([np.median(padded[i:i + 2 * w + 1]) for i in range(len(flux))])
    thresh = local_med + 0.8 * flux.std()
    min_sep = int(0.08 * AUDIO_SR / hop)  # 80 ms
    peaks, last = [], -min_sep
    for i in range(1, len(flux) - 1):
        if flux[i] > thresh[i] and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1] and i - last >= min_sep:
            peaks.append(i)
            last = i
    strength = flux[peaks] / (flux.max() or 1) if peaks else np.array([])
    events = [{"t": round(float(p * hop / AUDIO_SR), 3), "strength": round(float(s), 3),
               "rms_db": round(float(20 * np.log10(rms[p] + 1e-9)), 1)} for p, s in zip(peaks, strength)]
    strongest = sorted(events, key=lambda e: -e["strength"])[:max_onsets]
    return {"available": True, "method": "spectral flux, 80ms min separation", "count": len(events),
            "per_sec": round(len(events) / (len(y) / AUDIO_SR), 2),
            "events": sorted(strongest, key=lambda e: e["t"])}


def cut_audio_sync(cuts: list[dict], onset_events: list[dict], window_ms: float = 80,
                   strong_quantile: float = 0.6) -> dict:
    """For each hard cut, the nearest *strong* audio onset within +/- window_ms
    (negative offset = sound lands before the cut). Also reports a chance baseline: the same
    hit ratio for cut times shifted by off-beat offsets, so a high ratio can be judged honestly."""
    if not onset_events:
        return {"available": False}
    strengths = sorted(o["strength"] for o in onset_events)
    floor = strengths[int(len(strengths) * strong_quantile)] if strengths else 0
    strong = [o for o in onset_events if o["strength"] >= floor]

    def hit_ratio(times: list[float]) -> float:
        if not times:
            return 0.0
        return sum(1 for t in times if any(abs(o["t"] - t) * 1000 <= window_ms for o in strong)) / len(times)

    rows, hits = [], 0
    for c in cuts:
        near = [o for o in strong if abs(o["t"] - c["t"]) * 1000 <= window_ms]
        if near:
            o = min(near, key=lambda o: abs(o["t"] - c["t"]))
            hits += 1
            rows.append({"cut_t": c["t"], "onset_t": o["t"], "offset_ms": round((o["t"] - c["t"]) * 1000),
                         "onset_strength": o["strength"]})
        else:
            rows.append({"cut_t": c["t"], "onset_t": None, "offset_ms": None})
    cut_times = [c["t"] for c in cuts]
    shifts = (0.37, -0.43, 0.61, -0.77, 1.13)
    baseline = sum(hit_ratio([t + d for t in cut_times]) for d in shifts) / len(shifts) if cut_times else None
    offsets = [r["offset_ms"] for r in rows if r["offset_ms"] is not None]
    ratio = round(hits / len(cuts), 2) if cuts else None
    verdict = None
    if ratio is not None and baseline is not None:
        verdict = ("cuts track audio transients well above chance" if ratio >= baseline + 0.25 else
                   "cuts track audio transients somewhat above chance" if ratio >= baseline + 0.1 else
                   "no clear cut-to-transient sync beyond chance")
    return {"window_ms": window_ms, "strong_onsets_used": len(strong), "cuts_with_onset": hits,
            "cuts_total": len(cuts), "ratio": ratio,
            "chance_baseline_ratio": round(baseline, 2) if baseline is not None else None, "verdict": verdict,
            "median_offset_ms": sorted(offsets)[len(offsets) // 2] if offsets else None, "pairs": rows,
            "note": "Rough. Onsets include speech/music transients; an onset near a cut is not proof of an SFX."}


def palette(path: Path, duration: float, samples: int = 24, k: int = 8) -> dict:
    """Dominant colours across evenly sampled frames (quantised histogram, 4 bits/channel)."""
    if np is None:
        return {"available": False, "reason": "numpy not installed"}
    fps = max(samples / duration, 0.05) if duration else 1
    w, h = 48, 86
    res = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf",
                          f"fps={fps:.4f},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=False)
    if res.returncode != 0 or not res.stdout:
        return {"available": False, "reason": "frame decode failed"}
    px = np.frombuffer(res.stdout, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    q = px >> 4
    keys = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    counts = np.bincount(keys, minlength=4096)
    order = np.argsort(-counts)
    colours, taken = [], []
    for key in order:
        if counts[key] == 0 or len(colours) >= k:
            break
        mask = keys == key
        mean = px[mask].mean(axis=0)
        if any(np.abs(mean - t).sum() < 40 for t in taken):
            continue
        taken.append(mean)
        r, g, b = (int(round(v)) for v in mean)
        colours.append({"hex": f"#{r:02X}{g:02X}{b:02X}", "share": round(float(counts[key] / len(keys)), 3)})
    luma = (0.2126 * px[:, 0] + 0.7152 * px[:, 1] + 0.0722 * px[:, 2])
    sat = (px.max(axis=1) - px.min(axis=1))
    return {"available": True, "frames_sampled": int(len(px) // (w * h)), "dominant": colours,
            "mean_luma_0_255": round(float(luma.mean()), 1), "dark_pixel_share": round(float((luma < 40).mean()), 3),
            "bright_pixel_share": round(float((luma > 215).mean()), 3),
            "mean_saturation_0_255": round(float(sat.mean()), 1),
            "note": "Bin-averaged colours of whole frames (footage + graphics); roles are not assigned here."}


def measure(path: Path, threshold: float = 0.3, soft_threshold: float = 0.15, min_gap: float = 0.12,
            detail_window: float = 6.0, detail_target: str = "motion") -> dict:
    require_ffmpeg()
    meta = probe(path)
    duration = meta["duration_s"]
    scores = scene_scores(path)
    cuts = detect_cuts(scores, duration, threshold, soft_threshold, min_gap)
    result: dict = {
        "basis": "measured",
        "tooling": "ffprobe/ffmpeg scene score (scaled 320px), ebur128, silencedetect, numpy spectral flux",
        "media": meta,
        "cuts": cuts,
        "visual_change_per_sec": motion_curve(scores, duration),
        "densest_window": densest_window(scores, cuts["hard_cuts"], duration, detail_window, detail_target),
        "densest_cut_window": densest_window(scores, cuts["hard_cuts"], duration, detail_window, "cuts"),
        "palette": palette(path, duration),
    }
    if meta["has_audio"]:
        result["loudness"] = loudness(path)
        result["silences"] = silences(path)
        ons = onsets(path)
        result["onsets"] = ons
        if ons.get("available"):
            result["cut_audio_sync"] = cut_audio_sync(cuts["hard_cuts"], ons["events"])
    else:
        result["audio"] = "no audio stream"
    # TODO(earshot): when Fortune un-pauses earshot, merge its sound-event timeline here, e.g.
    #   result["earshot"] = earshot_events(path)   # SFX labels, per-event ms timing, levels vs voice
    # and prefer it over `onsets`/`cut_audio_sync` in the prompt and the Sound design section.
    result["earshot"] = None
    return result


def compact_for_prompt(m: dict, max_onsets: int = 60) -> dict:
    """Prompt-sized copy of the measurements: drops per-shot list (derivable from cuts), thins the
    loudness curve to 1 s, keeps only the strongest onsets and the cut/onset pairs that matched."""
    c = json.loads(json.dumps(m))
    c["cuts"]["hard_cuts"] = [{"t": e["t"], "score": e["score"]} for e in c["cuts"]["hard_cuts"]]
    c["cuts"]["soft_events"] = [{"t": e["t"], "score": e["score"]} for e in c["cuts"]["soft_events"]]
    c["cuts"]["shots"] = [[s["start"], s["dur"]] for s in c["cuts"]["shots"]]
    c["visual_change_per_sec"] = [[v["t"], v["energy"]] for v in c.get("visual_change_per_sec", [])]
    if "loudness" in c:
        c["loudness"]["momentary_curve_1s"] = [[p["t"], p["momentary_lufs"]] for p in
                                                 c["loudness"].pop("momentary_curve_0p5s", []) if float(p["t"]).is_integer()]
    if c.get("onsets", {}).get("events"):
        strongest = sorted(c["onsets"]["events"], key=lambda e: -e["strength"])[:max_onsets]
        c["onsets"]["events"] = [[e["t"], e["strength"]] for e in sorted(strongest, key=lambda e: e["t"])]
        c["onsets"]["events_note"] = f"strongest {len(strongest)} only, as [t, strength]"
    if c.get("cut_audio_sync", {}).get("pairs"):
        c["cut_audio_sync"]["pairs"] = [[p["cut_t"], p["offset_ms"]] for p in c["cut_audio_sync"]["pairs"] if p["offset_ms"] is not None]
    c.pop("palette", None) if not c.get("palette", {}).get("available") else None
    return c


if __name__ == "__main__":
    import sys
    print(json.dumps(measure(Path(sys.argv[1])), indent=2))
