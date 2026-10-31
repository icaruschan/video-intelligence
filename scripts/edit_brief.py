"""Deterministic brief builder for edit mode (no model calls).

Turns measured.json + frame measurements + the model analysis (pass 1 + detail pass) into a build-ready brief:
  - STYLE RECIPE (portable rules) and BEAT MAP (% of runtime, one still per beat)
  - contradictions reconciled with an explicit winner; every value carries `source`
  - sound honesty (no 'no SFX needed' unless measured; unseparable SFX = unclear)
  - frame-measured entry timing -> GSAP eases; persistent overlays; in-place builds
  - VO pace from word timings (wpm, pause distribution, cut-vs-word alignment)
  - fonts mapped to free installed files with licence; px at 1080x1920
  - asset list (reference material is study-only), do-not list tagged [core-style]/[ref-specific]
  - QA checklist as ranges/ratios, never absolute counts
  - `cooked` block shaped like ai-video-editor-cooked style.json (field map in README)

Sources (the `source` field on every value):
  measured     ffmpeg/numpy on the file (cuts, loudness, silences, onsets, palette)
  frame        numpy on decoded frames (overlays, in-place builds, element entry timing)
  detail-pass  model, high-fps detail window with burned-in timecode (wins inside its window)
  pass1        model, whole clip
  inference    derived/interpretation (rule of thumb, house style)
  unclear      cannot be determined offline
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
from pathlib import Path

BUILD_W, BUILD_H = 1080, 1920
WORD_TOL_S = 0.08

# ----------------------------------------------------------------------------- small helpers


def _q(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    v = sorted(vals)
    k = (len(v) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 3)


def V(value, source: str, note: str | None = None) -> dict:
    d = {"value": value, "source": source}
    if note:
        d["note"] = note
    return d


def _slot(section: dict, name: str) -> tuple[str | None, str | None]:
    sp = (section or {}).get("spec")
    if isinstance(sp, dict):
        x = sp.get(name) or {}
        return x.get("value"), x.get("basis")
    for x in sp or []:
        if x.get("slot") == name:
            return x.get("value"), x.get("basis")
    return None, None


def _pass1(basis: str | None) -> str:
    # model basis 'measured' means 'copied from the MEASURED block'; keep the provenance honest
    return "pass1 (copied measured)" if basis == "measured" else "pass1"


def px_note(text: str) -> str:
    """Append px at 1080x1920 for every 'N% ... height/width' found in a value string."""
    if not isinstance(text, str):
        return text
    out = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)(?:\s*[-–]\s*(\d+(?:\.\d+)?))?\s*%\s*(?:of\s*)?(frame\s*)?(height|width|h\b|w\b|H\b|W\b)",
                         text):
        a = float(m.group(1))
        b = float(m.group(2)) if m.group(2) else None
        dim = BUILD_H if m.group(4).lower().startswith("h") else BUILD_W
        if b is not None:
            out.append(f"{m.group(0)} ≈ {round(a * dim / 100)}–{round(b * dim / 100)} px")
        else:
            out.append(f"{m.group(0)} ≈ {round(a * dim / 100)} px")
    return text + (f"  [@1080x1920: {'; '.join(out)}]" if out else "")


# ----------------------------------------------------------------------------- pacing


def shot_stats(measured: dict) -> dict:
    cuts = measured["cuts"]
    durs = [s["dur"] for s in cuts["shots"]]
    dur = measured["media"]["duration_s"]
    ct = [c["t"] for c in cuts["hard_cuts"]]
    win = min(10.0, dur)
    counts = []
    t = 0.0
    while t + win <= dur + 1e-6:
        counts.append(sum(1 for c in ct if t <= c < t + win) * 10.0 / win)
        t += 1.0
    return {
        "shot_len_s": {"p10": _q(durs, .1), "p25": _q(durs, .25), "p50": _q(durs, .5), "p75": _q(durs, .75),
                       "p90": _q(durs, .9), "min": min(durs), "max": max(durs), "source": "measured"},
        "cuts_per_10s": {"overall": cuts["cuts_per_10s"],
                         "band_p25_p75": [_q(counts, .25), _q(counts, .75)],
                         "peak": max(counts) if counts else None,
                         "method": f"sliding {win:g} s windows, 1 s step", "source": "measured"},
        "first_cut_pct": round(100 * ct[0] / dur, 1) if ct else None,
    }


def beat_map(analysis: dict, measured: dict, out_dir: Path | None, video: Path | None, frames_mod=None) -> list[dict]:
    dur = measured["media"]["duration_s"]
    cuts = [c["t"] for c in measured["cuts"]["hard_cuts"]]
    beats = (analysis or {}).get("structure_pacing", {}).get("beats") or []
    out = []
    for i, b in enumerate(beats, 1):
        s, e = b.get("start_s"), b.get("end_s")
        if not isinstance(s, (int, float)) or not isinstance(e, (int, float)) or e <= s:
            continue
        inside = [c for c in cuts if s < c < e]
        shots_in = [x["dur"] for x in measured["cuts"]["shots"] if x["start"] >= s - 0.05 and x["end"] <= e + 0.05]
        label = b.get("label") or f"beat {i}"
        rec = {"n": i, "label": label, "start_s": s, "end_s": e,
               "start_pct": round(100 * s / dur, 1), "end_pct": round(100 * e / dur, 1),
               "share_pct": round(100 * (e - s) / dur, 1), "cuts_inside": len(inside),
               "cuts_per_10s": round(len(inside) * 10 / (e - s), 1), "median_shot_s": _q(shots_in, .5),
               "description": b.get("description"),
               "source": "boundaries pass1 (snapped to measured cuts where within 0.25 s); % and cut stats measured"}
        snapped = [c for c in cuts if abs(c - s) <= 0.25]
        if snapped:
            rec["start_s"] = snapped[0]
            rec["start_pct"] = round(100 * snapped[0] / dur, 1)
        if out_dir and video and frames_mod:
            t = min(e - 0.04, rec["start_s"] + min(0.5, (e - rec["start_s"]) / 2))
            slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:32]
            name = f"beat{i:02d}_t{t:06.2f}s_{slug}.jpg"
            p = frames_mod.save_still(video, t, out_dir / "beats" / name, width=540)
            rec["still"] = f"beats/{name}" if p else None
            rec["still_t_s"] = round(t, 3)
        out.append(rec)
    return out


# ----------------------------------------------------------------------------- VO / words


def _speech_words(detail: dict | None) -> list[dict]:
    if not detail:
        return []
    ws = [w for w in detail.get("caption_word_timing", []) if isinstance(w.get("t_s"), (int, float)) and w.get("word")]
    # overlay text (several 'words' at the same instant, or multi-word strings) is not speech
    times = [round(w["t_s"], 2) for w in ws]
    return [w for w in ws if times.count(round(w["t_s"], 2)) == 1 and len(str(w["word"]).split()) <= 2]


def vo_stats(measured: dict, analysis: dict, detail: dict | None) -> dict:
    words = _speech_words(detail)
    res: dict = {"proxy": "caption word first-appearance times (detail pass) stand in for spoken word onsets; "
                          "captions in this style track speech closely. No forced alignment / ASR ran.",
                 "source": "detail-pass words; maths deterministic"}
    if len(words) >= 4:
        t = [w["t_s"] for w in words]
        span = t[-1] - t[0]
        gaps = [round(b - a, 3) for a, b in zip(t, t[1:])]
        res.update({
            "window_s": [detail["window"]["start"], detail["window"]["end"]] if detail.get("window") else None,
            "words": len(words),
            "wpm": round((len(words) - 1) / span * 60) if span > 0 else None,
            "gap_s": {"p25": _q(gaps, .25), "p50": _q(gaps, .5), "p75": _q(gaps, .75), "p90": _q(gaps, .9),
                      "max": max(gaps)},
            "phrase_gaps_ge_0p3s": [g for g in gaps if g >= 0.3],
            "phrase_breaks_per_10s": round(10 * sum(1 for g in gaps if g >= 0.3) / span, 1) if span else None,
            "gap_note": "onset-to-onset; a gap includes the previous word's spoken length, so it is not pure silence. "
                        "< 0.3 s = inside a phrase, >= 0.3 s = phrase break",
        })
        lo, hi = t[0] - 0.3, t[-1] + 0.3
        cuts = [c["t"] for c in measured["cuts"]["hard_cuts"] if lo <= c["t"] <= hi]
        rows, within = [], 0
        before = after = 0
        for c in cuts:
            nearest = min(t, key=lambda x: abs(x - c))
            off = round((c - nearest) * 1000)          # + = cut lands after the word onset
            rows.append({"cut_s": c, "nearest_word_s": nearest, "offset_ms": off})
            if abs(off) <= WORD_TOL_S * 1000:
                within += 1
            elif off < 0:
                before += 1
            else:
                after += 1
        offs = [abs(r["offset_ms"]) for r in rows]
        res["cut_vs_word"] = {
            "cuts_in_window": len(cuts),
            "within_80ms_pct": round(100 * within / len(cuts), 1) if cuts else None,
            "earlier_than_word_pct": round(100 * before / len(cuts), 1) if cuts else None,
            "later_than_word_pct": round(100 * after / len(cuts), 1) if cuts else None,
            "abs_offset_ms_p50": _q(offs, .5),
            "rows": rows,
            "limit": "word times are caption-appearance times read by the model at ~8-10 fps effective (±60-120 ms); "
                     "treat ±80 ms alignment as indicative",
        }
    else:
        res["note"] = "fewer than 4 word timings available; VO pace not computed"
    # whole-clip rough pace from pass-1 caption examples (lower bound: examples are a sample)
    ex = [e for e in (analysis or {}).get("captions", {}).get("examples", []) if isinstance(e.get("t_s"), (int, float))]
    if len(ex) >= 4:
        n = sum(len(str(e.get("text", "")).split()) for e in ex)
        span = ex[-1]["t_s"] - ex[0]["t_s"]
        res["pass1_caption_wpm_lower_bound"] = V(round(n / span * 60) if span else None, "pass1",
                                                 "words in pass-1 caption examples / their span; examples are a sample, so this is a floor")
    sil = measured.get("silences") or []
    res["measured_silences"] = V(sil, "measured", "silencedetect < -35 dB >= 0.25 s; a music bed masks VO pauses, so "
                                                  "pauses between words are rarely 'silent'")
    return res


# ----------------------------------------------------------------------------- captions reconciliation


def reconcile_captions(analysis: dict, detail: dict | None, frames: dict | None) -> dict:
    cp = (analysis or {}).get("captions", {})
    out: dict = {}
    conflicts: list[dict] = []
    for slot in ("position_pct_from_top", "lines_per_caption", "words_per_line", "font_and_size", "colour_and_stroke_shadow",
                 "highlight_style", "entry_animation", "entry_duration_ms", "easing", "stagger_ms_per_word", "exit_animation",
                 "sync_to_speech"):
        v, b = _slot(cp, slot)
        if v is not None:
            out[slot] = V(px_note(v) if slot in ("font_and_size", "position_pct_from_top") else v, _pass1(b))
    words = _speech_words(detail)
    if detail and len(words) >= 3:
        w0, w1 = detail["window"]["start"], detail["window"]["end"]
        t = [w["t_s"] for w in words]
        gaps = [b - a for a, b in zip(t, t[1:])]
        stag = [g for g in gaps if g < 0.3]
        if stag:
            med = round(_q(stag, .5) * 1000)
            dv = V(f"word-by-word build, median {med} ms between words inside a phrase "
                   f"(p25 {round(_q(stag, .25) * 1000)} – p75 {round(_q(stag, .75) * 1000)} ms, n={len(stag)})",
                   "detail-pass", f"window {w0}-{w1} s; detail pass wins inside its window")
            p1 = out.get("stagger_ms_per_word")
            if p1 and not re.search(r"\b1[0-4]\d\s*ms|word[- ]by[- ]word", str(p1["value"]), re.I):
                conflicts.append({"field": "captions.stagger_ms_per_word", "pass1": p1["value"], "detail_pass": dv["value"],
                                  "winner": "detail-pass inside its window; pass1 kept as 'outside window, unverified'"})
                out["stagger_ms_per_word_outside_window"] = dict(p1, note="pass1, outside the detail window, unverified")
            out["stagger_ms_per_word"] = dv
        durs = [e.get("duration_ms") for e in detail.get("events", []) if e.get("action") == "caption"
                and isinstance(e.get("duration_ms"), (int, float)) and e["duration_ms"] > 0]
        if durs:
            dv = V(f"~{round(_q(durs, .5))} ms per word entry (detail pass estimate, ±1 sample)", "detail-pass")
            p1 = out.get("entry_duration_ms")
            if p1:
                conflicts.append({"field": "captions.entry_duration_ms", "pass1": p1["value"], "detail_pass": dv["value"],
                                  "winner": "detail-pass inside its window"})
            out["entry_duration_ms"] = dv
        # word timing disagreement: pass-1 examples vs detail words inside the window
        rows = []
        for e in cp.get("examples", []):
            ts = e.get("t_s")
            if not isinstance(ts, (int, float)) or not (w0 <= ts <= w1):
                continue
            first = str(e.get("text", "")).split()[:1]
            if not first:
                continue
            key = re.sub(r"\W", "", first[0].lower())
            cand = [w for w in words if re.sub(r"\W", "", str(w["word"]).lower()) == key]
            if cand:
                best = min(cand, key=lambda w: abs(w["t_s"] - ts))
                if abs(best["t_s"] - ts) >= 0.08:
                    rows.append({"text": e.get("text"), "pass1_t_s": ts, "detail_t_s": best["t_s"],
                                 "delta_ms": round((ts - best["t_s"]) * 1000)})
        if rows:
            conflicts.append({"field": "caption word times (section 6 vs detail pass)", "rows": rows,
                              "winner": "detail-pass (read from burned-in timecode); pass-1 times inside the window are "
                                        "phrase starts snapped to cuts"})
    if frames and frames.get("entries_summary", {}).get("n"):
        s = frames["entries_summary"]
        out["measured_entries"] = V(f"{s['n']} settled entries on still footage: p25/p50/p75 {s['entry_ms_p25']}/"
                                    f"{s['entry_ms_p50']}/{s['entry_ms_p75']} ms; {s['hard_pops_le_2_frames']} hard pops "
                                    f"(<= 2 frames)", "frame")
    return {"spec": out, "conflicts": conflicts}


# ----------------------------------------------------------------------------- sound honesty

_SFX_NEG = re.compile(
    r"(no|not|don'?t|do not|never|avoid|without|skip|zero)\b[^.;]{0,60}\b(sfx|sound effects?|whoosh|swoosh sound|pops?\b)"
    r"|\b(sfx|sound effects?)\b[^.;]{0,40}\b(not needed|unnecessary|none|no need|not required|absent|not detected|not used|n.t used|not present|not audible)"
    r"|no isolated non-speech sfx", re.I)


def sound_honesty(analysis: dict, measured: dict) -> dict:
    """Strip/annotate SFX-negative claims: offline measurement cannot separate SFX from VO/music (earshot paused)."""
    hits = []

    def scan(path: str, text):
        if isinstance(text, str) and _SFX_NEG.search(text):
            hits.append({"where": path, "text": text})

    a = analysis or {}
    for i, cv in enumerate(a.get("edit_brief", {}).get("copy_values", [])):
        scan(f"edit_brief.copy_values[{i}] {cv.get('parameter')}", cv.get("value"))
    for i, dn in enumerate(a.get("edit_brief", {}).get("do_not", [])):
        scan(f"edit_brief.do_not[{i}]", dn if isinstance(dn, str) else dn.get("rule"))
    sd = a.get("sound_design", {})
    sp = sd.get("spec")
    items = sp.items() if isinstance(sp, dict) else [(x.get("slot"), x) for x in sp or []]
    for k, x in items:
        scan(f"sound_design.spec.{k}", (x or {}).get("value"))
    for i, f in enumerate(sd.get("facts", [])):
        scan(f"sound_design.facts[{i}]", f.get("value"))
    for i, f in enumerate(a.get("motion_language", {}).get("facts", [])):
        scan(f"motion_language.facts[{i}]", f.get("value"))
    onsets = measured.get("onsets", {})
    status = {
        "sfx_presence": V("unclear", "unclear",
                          "offline measurement cannot separate SFX from voice/music: onsets are any transient; earshot "
                          "sound-event detection is paused"),
        "sfx_events_from_model": V(sd.get("sfx_events") or [], "pass1",
                                   "model listening; unverified (empty list != 'no SFX')"),
        "onset_rate_per_s": V(onsets.get("per_sec"), "measured", "speech + music + SFX transients mixed"),
        "cut_audio_sync": V({k: measured.get("cut_audio_sync", {}).get(k) for k in ("ratio", "chance_baseline_ratio", "verdict",
                                                                                   "median_offset_ms")}, "measured"),
        "house_style": V("Fortune's edits use SFX throughout: plan whooshes on transitions, pops/clicks on text and "
                         "graphic entries, risers into beats, impacts on punchlines. The reference does not override "
                         "this unless SFX absence is MEASURED.", "inference"),
        "removed_claims": hits,
    }
    return status


def _sfx_clean(text: str) -> str:
    return "SFX: unclear in the reference (not separable offline) — follow house style (SFX throughout)" \
        if isinstance(text, str) and _SFX_NEG.search(text) else text


# ----------------------------------------------------------------------------- fonts

FONT_DIR = Path("/usr/share/fonts/truetype/sand-box/google")
HF_CACHE = Path.home() / ".cache/hyperframes/fonts"
FONT_RULES = [
    # (regex on the model's guess, family, style, licence, why)
    (r"helvetica|inter\b|grotesk|grotesque|neue haas|sf pro|arial|geometric sans|neo-grotesque|sans", "Inter", "normal",
     "OFL-1.1", "closest free neo-grotesque; variable 100-900"),
    (r"tiktok", "TikTok Sans", "normal", "OFL-1.1", "HyperFrames cache"),
    (r"canela|playfair|didone|high-contrast serif|serif italic|italic serif|italic", "Playfair Display", "italic", "OFL-1.1",
     "high-contrast italic serif; alt: Instrument Serif Italic (OFL)"),
    (r"instrument serif", "Instrument Serif", "italic", "OFL-1.1", "exact free family"),
    (r"garamond|caslon|old-style", "EB Garamond", "normal", "OFL-1.1", "old-style serif"),
    (r"marker|handwrit|hand-lettered|script", "Permanent Marker", "normal", "Apache-2.0",
     "marker hand; alt: Caveat (OFL) for a lighter pen hand"),
    (r"mono|code|typewriter", "JetBrains Mono", "normal", "OFL-1.1", "HyperFrames code-editorial preset"),
]


def _font_files(family: str, style: str) -> list[str]:
    d = FONT_DIR / family
    files = []
    if d.exists():
        files = sorted(str(p) for p in d.iterdir() if p.suffix in (".ttf", ".otf"))
        if style == "italic":
            files = [f for f in files if "Italic" in f] or files
        else:
            files = [f for f in files if "Italic" not in f] or files
    cache = HF_CACHE / family.lower().replace(" ", "-")
    if cache.exists():
        files += [str(next(iter(sorted(cache.glob("*.woff2")))))] if list(cache.glob("*.woff2")) else []
    if family == "JetBrains Mono":
        files += [str(p) for p in Path("/workspace/video-editor/skills/hyperframes/hyperframes/assets/frame-presets/code-editorial/fonts").glob("JetBrainsMono-*.woff2")]
    return files


def map_fonts(analysis: dict) -> list[dict]:
    out = []
    for f in (analysis or {}).get("typography_layout", {}).get("fonts", []):
        guess = str(f.get("font_guess", ""))
        if re.search(r"\bprop\b|in-clip|baked|not font", guess + " " + str(f.get("role", "")), re.I):
            out.append({"role": f.get("role"), "reference_guess": guess, "build_with": None,
                        "note": "part of the footage/prop, not a font to set", "source": "pass1"})
            continue
        rule = next((r for r in FONT_RULES if re.search(r[0], guess, re.I)), None)
        size = f.get("size_pct_of_frame_height")
        rec = {"role": f.get("role"), "reference_guess": guess, "weight": f.get("weight"), "case": f.get("case"),
               "size": px_note(f"{size} of frame height") if size else None, "colour": f.get("colour_hex"),
               "source": "pass1 guess -> mapped by rule (inference)"}
        if rule:
            files = _font_files(rule[1], rule[2])
            rec.update({"build_with": rule[1], "style": rule[2], "licence": rule[3], "why": rule[4],
                        "installed_files": files, "installed": bool(files),
                        "css": f"font-family: '{rule[1]}'; font-style: {rule[2]};"})
        else:
            rec.update({"build_with": None, "note": "no rule matched; pick manually"})
        out.append(rec)
    return out


# ----------------------------------------------------------------------------- eases

def gsap_from_text(text: str) -> str | None:
    t = str(text or "").lower()
    m = re.search(r"(back|elastic|power[1-4]|expo|sine|circ|quad|cubic)\.(out|in|inout)(\([\d.]+\))?", t)
    if m:
        return m.group(0).replace("inout", "inOut")
    for pat, ease in ((r"spring|overshoot|bounce", "back.out(1.7)"), (r"expo", "expo.out"), (r"ease-in-out|ease in out", "power2.inOut"),
                      (r"ease-out|ease out|decelerat", "power2.out"), (r"ease-in|ease in|gravity", "power2.in"),
                      (r"linear|constant", "none")):
        if re.search(pat, t):
            return ease
    return None


# ----------------------------------------------------------------------------- assets / do-not

ASSET_CATS = [
    ("faces / ECU portraits", r"face|eye|ecu|portrait|man\b|woman|glasses|beard"),
    ("hands / writing / desk", r"hand|writ|pen|paper|notebook|desk|highlighter"),
    ("props / objects", r"typewriter|book|envelope|wax|stamp|prop|mic|keyboard|laptop"),
    ("screen recordings / UI", r"screen|word |ui|document|app|browser"),
    ("lifestyle / action B-roll", r"run|dog|bike|cyclist|baseball|sport|lake|car|convertible"),
    ("stickies / paper graphics", r"sticky|note"),
]


def assets(analysis: dict, frames: dict | None) -> list[dict]:
    a = analysis or {}
    out: list[dict] = []
    for x in a.get("assets", []) or []:      # new schema field (model)
        out.append({**x, "source": "pass1", "study_only": True})
    seen = {}
    for b in a.get("broll", {}).get("placements", []):
        c = str(b.get("content", ""))
        cat = next((n for n, rx in ASSET_CATS if re.search(rx, c, re.I)), "other footage")
        seen.setdefault(cat, []).append(f"{b.get('start_s')}s {c[:60]}")
    for cat, ex in seen.items():
        out.append({"kind": "b-roll category", "item": cat, "count_in_ref": len(ex), "examples": ex[:4],
                    "to_do": "source own/licensed footage of this kind", "source": "pass1", "study_only": True})
    for e in a.get("elements_assets", {}).get("elements", []):
        el = str(e.get("element", ""))
        kind = "prop (physical)" if re.search(r"prop|typewriter|envelope|wax|sticky|paper", el, re.I) else "graphic element (build)"
        out.append({"kind": kind, "item": el[:90], "t_in_s": e.get("t_in_s"), "motion_in": e.get("motion_in"),
                    "to_do": "build in HyperFrames" if kind.startswith("graphic") else "shoot/source own",
                    "source": "pass1", "study_only": True})
    for o in (frames or {}).get("overlays", []):
        if o["kind"].startswith("persistent"):
            out.append({"kind": "graphic element (build)", "item": f"persistent overlay {o['id']} ({o['zone']})",
                        "t_in_s": o["t_start_s"], "px_1080x1920": o["px_1080x1920"], "still": o.get("crop_still"),
                        "to_do": "rebuild as a HyperFrames layer above the cut track", "source": "frame", "study_only": True})
    dbs = (frames or {}).get("dynamic_badges", [])
    if dbs:
        out.append({"kind": "graphic element (build)",
                    "item": f"{len(dbs)} in-place builds (timer/counter pill, word-by-word builds, number reveals): "
                            + ", ".join(f"{o['id']}@{o['t_start_s']}s" for o in dbs),
                    "stills": [o.get("crop_still") for o in dbs],
                    "to_do": "rebuild as HyperFrames text/graphic layers; see 13d crops", "source": "frame", "study_only": True})
    return out


_REF_SPECIFIC = re.compile(r"\d+(\.\d+)?\s*s\b|step\s*\d|'[^']+'|\"[^\"]+\"|swoosh|typewriter|logo|wax|carousel|"
                           r"sticky|cta|\b\d{4}\b|red\b|brand", re.I)


def tag_do_not(analysis: dict) -> list[dict]:
    out = []
    for x in (analysis or {}).get("edit_brief", {}).get("do_not", []):
        if isinstance(x, dict):
            rule, scope, src = x.get("rule", ""), x.get("scope"), "pass1"
        else:
            rule, scope, src = str(x), None, "pass1 (scope tagged by rule)"
        if scope not in ("core-style", "ref-specific"):
            scope = "ref-specific" if _REF_SPECIFIC.search(rule) else "core-style"
        if _SFX_NEG.search(rule):
            out.append({"rule": rule, "scope": "removed", "source": src,
                        "note": "removed by sound-honesty: SFX absence is not measured; house style uses SFX"})
            continue
        out.append({"rule": rule, "scope": scope, "source": src})
    return out


# ----------------------------------------------------------------------------- QA checklist (ranges only)

def qa_checklist(stats: dict, beats: list[dict], vo: dict, frames: dict | None, caps: dict) -> list[dict]:
    sl, cr = stats["shot_len_s"], stats["cuts_per_10s"]
    q = [
        {"check": f"Median shot length {round(sl['p50'] * 0.85, 2)}–{round(sl['p50'] * 1.15, 2)} s "
                  f"(ref p50 {sl['p50']} s, ±15%); p25–p75 inside {sl['p25']}–{sl['p75']} s ±20%", "source": "measured"},
        {"check": f"Cuts per 10 s in every 10 s window within {cr['band_p25_p75'][0]}–{cr['band_p25_p75'][1]} for most of "
                  f"the runtime (ref overall {cr['overall']}); peak ≤ {cr['peak']}", "source": "measured"},
        {"check": f"First cut at ≤ {max(5, round((stats['first_cut_pct'] or 5) * 1.3))}% of runtime", "source": "measured"},
    ]
    if beats:
        q.append({"check": "Beat boundaries within ±5% of runtime of the beat map: " +
                           ", ".join(f"{b['label']} {b['start_pct']}–{b['end_pct']}%" for b in beats), "source": "measured"})
        q.append({"check": "Pacing contour: the fastest beat and the slowest beat are the same ones as in the reference "
                           f"(fastest: {max(beats, key=lambda b: b['cuts_per_10s'])['label']}, slowest: "
                           f"{min(beats, key=lambda b: b['cuts_per_10s'])['label']})", "source": "measured"})
    if vo.get("wpm"):
        q.append({"check": f"VO pace {round(vo['wpm'] * 0.9)}–{round(vo['wpm'] * 1.1)} wpm (ref ~{vo['wpm']}, ±10%)",
                  "source": "detail-pass words"})
    cvw = vo.get("cut_vs_word") or {}
    if cvw.get("within_80ms_pct") is not None:
        q.append({"check": f"≥ {max(0, round(cvw['within_80ms_pct'] - 15))}% of cuts land within ±80 ms of a word onset "
                           f"(ref {cvw['within_80ms_pct']}% in the detail window)", "source": "detail-pass words"})
    s = (frames or {}).get("entries_summary") or {}
    if s.get("n"):
        ents = (frames or {}).get("entries", [])
        stg = [e for e in ents if e.get("kind") == "staggered build"]
        spr = [e for e in ents if e["overshoot_pct"] > 0]
        bits = []
        if stg:
            tr = [e["travel"] for e in stg if e.get("travel")]
            bits.append(f"staggered card builds: 2nd part +{min(e['stagger_ms'] for e in stg)}–{max(e['stagger_ms'] for e in stg)} ms "
                        f"after the 1st (±40 ms)" + (f", rise {min(t['px_1080x1920'] for t in tr)}–{max(t['px_1080x1920'] for t in tr)} px "
                        f"over {min(t['ms'] for t in tr)}–{max(t['ms'] for t in tr)} ms {tr[0]['ease']}" if tr else ""))
        if spr:
            bits.append(f"springy entries {min(e['overshoot_pct'] for e in spr)}–{max(e['overshoot_pct'] for e in spr)}% overshoot (±3 pts)")
        else:
            bits.append("no overshoot (no back.out)")
        txt = "; ".join(bits)
        q.append({"check": f"Text/graphic entries: ≥ {round(70 * s['hard_pops_le_2_frames'] / s['n'])}% are hard pops ≤ 2 frames "
                           f"(ref {s['hard_pops_le_2_frames']}/{s['n']}); {txt}", "source": "frame"})
    st = caps["spec"].get("stagger_ms_per_word")
    if st:
        q.append({"check": f"Caption build matches: {st['value']}", "source": st["source"]})
    for o in (frames or {}).get("overlays", []):
        if o["kind"].startswith("persistent"):
            q.append({"check": f"Persistent overlay {o['id']} sits at {o['px_1080x1920']} px (±2% of frame) and rides over "
                               "the cut without re-animating", "source": "frame"})
    q.append({"check": "SFX present on transitions and on text/graphic pops (house style); none clipped; VO stays on top",
              "source": "inference"})
    q.append({"check": "Nothing from the reference (footage, logos, props shots, text) appears in the build", "source": "inference"})
    return q


# ----------------------------------------------------------------------------- the brief


def build_brief(measured: dict, analysis: dict, detail: dict | None, frames: dict | None,
                out_dir: Path | None = None, video: Path | None = None, frames_mod=None) -> dict:
    a = analysis or {}
    stats = shot_stats(measured)
    beats = beat_map(a, measured, out_dir, video, frames_mod)
    vo = vo_stats(measured, a, detail)
    caps = reconcile_captions(a, detail, frames)
    sound = sound_honesty(a, measured)
    fonts = map_fonts(a)
    eb = a.get("edit_brief", {})

    # transitions vocabulary
    tv: dict[str, int] = {}
    for e in a.get("transitions", {}).get("events", []):
        k = str(e.get("type", "unclear")).lower()
        tv[k] = tv.get(k, 0) + 1
    n_tr = sum(tv.values()) or 1
    # emphasis rule
    emph = next((cv for cv in eb.get("copy_values", []) if re.search(r"emphasis|highlight", str(cv.get("parameter")), re.I)), None)
    hv, hb = _slot(a.get("captions", {}), "highlight_style")
    # motion
    fs = (frames or {}).get("entries_summary") or {}
    entries = (frames or {}).get("entries") or []
    springy = [e for e in entries if e["overshoot_pct"] > 0]
    ml = a.get("motion_language", {})
    p1_ease, p1_b = _slot(ml, "gsap_ease_equivalent")
    p1_os, _ = _slot(ml, "overshoot_pct")
    contradictions = list(caps["conflicts"])
    staggered = [e for e in entries if e.get("kind") == "staggered build"]
    if p1_os and entries:
        contradictions.append({"field": "motion.overshoot", "pass1": p1_os,
                               "frame": (f"{', '.join(str(e['overshoot_pct']) + '%' for e in springy)} on "
                                         f"{', '.join(str(e['t_start_s']) + 's' for e in springy)}" if springy else
                                         "0% on every settled entry (ink size constant); "
                                         f"{len(staggered)} staggered fade+rise builds, rest hard pops"),
                               "winner": "frame measurement"})
    if any("short-term" in json.dumps(x) for x in (a.get("sound_design", {}),)):
        contradictions.append({"field": "sound_design (loudness facts)", "pass1": "quotes short-term (3 s) loudness",
                               "measured": "re-measured with momentary (400 ms) loudness; short-term smears the first seconds",
                               "winner": "measured (momentary)"})
    if sound["removed_claims"]:
        contradictions.append({"field": "sound (SFX)", "pass1": [h["text"] for h in sound["removed_claims"]],
                               "measured": "SFX presence cannot be measured offline", "winner": "unclear -> house style"})

    recipe = {
        "cut_rate_band": V(f"{stats['cuts_per_10s']['band_p25_p75'][0]}–{stats['cuts_per_10s']['band_p25_p75'][1]} cuts per 10 s "
                           f"(p25–p75 of sliding 10 s windows; overall {stats['cuts_per_10s']['overall']}, peak "
                           f"{stats['cuts_per_10s']['peak']})", "measured"),
        "shot_length_s": V({k: stats["shot_len_s"][k] for k in ("p25", "p50", "p75")}, "measured",
                           f"p10 {stats['shot_len_s']['p10']} / p90 {stats['shot_len_s']['p90']} s"),
        "caption_system": caps["spec"],
        "emphasis_rule": V(px_note(emph["value"]) if emph else hv, _pass1(emph.get("basis") if emph else hb)),
        "transition_vocabulary": V({k: f"{round(100 * v / n_tr)}%" for k, v in sorted(tv.items(), key=lambda kv: -kv[1])},
                                   "pass1 types; hard-cut times measured"),
        "colour_logic": V({"roles": a.get("colour", {}).get("roles", []), "grade": a.get("colour", {}).get("grade")},
                          "pass1 (hex roles marked measured come from the measured palette)"),
        "motion": {
            "entries_ms_p25_p50_p75": V([fs.get("entry_ms_p25"), fs.get("entry_ms_p50"), fs.get("entry_ms_p75")], "frame")
            if fs.get("n") else V("not measurable offline", "unclear"),
            "hard_pop_share": V(f"{fs.get('hard_pops_le_2_frames')}/{fs.get('n')}", "frame") if fs.get("n") else None,
            "springy_entries": [V({"t_s": e["t_start_s"], "frames": e["entry_frames"], "ms": e["entry_ms"],
                                   "overshoot_pct": e["overshoot_pct"], "gsap": e["ease"]["gsap"],
                                   "px_1080x1920": e["px_1080x1920"]}, "frame") for e in springy],
            "staggered_builds": [V({"t_s": e["t_start_s"], "parts": e["parts"], "stagger_ms": e["stagger_ms"],
                                    "travel": e.get("travel"), "px_1080x1920": e["px_1080x1920"],
                                    "gsap_sketch": _gsap_stagger(e)}, "frame") for e in staggered],
            "pass1_ease": V(p1_ease, _pass1(p1_b), "model estimate; frame-measured eases win where they exist"),
            "gsap_default": V(springy[0]["ease"]["gsap"] if springy else ("none" if fs.get("n") else gsap_from_text(p1_ease)),
                              "frame" if fs.get("n") else "pass1",
                              "text pops: none (cut-on); cards: see staggered_builds / springy_entries"),
        },
        "vo_pace": V({k: vo.get(k) for k in ("wpm", "gap_s", "phrase_breaks_per_10s")}, "detail-pass words",
                     vo.get("proxy")),
        "sound": {k: sound[k] for k in ("sfx_presence", "house_style", "cut_audio_sync")},
        "typography": [{k: f.get(k) for k in ("role", "build_with", "style", "weight", "size", "licence")} for f in fonts],
    }
    copy_values = []
    for cv in eb.get("copy_values", []):
        val = _sfx_clean(px_note(cv.get("value")))
        copy_values.append({"parameter": cv.get("parameter"), "value": val,
                            "source": "unclear" if val != px_note(cv.get("value")) else _pass1(cv.get("basis")),
                            "portable": not bool(_REF_SPECIFIC.search(str(cv.get("parameter")) + " " + str(cv.get("value"))))})
    ref_notes = [c for c in eb.get("checklist", []) if re.search(r"\b\d+\s*(shots|cuts)\b", str(c))]
    brief = {
        "schema": "video-intelligence/brief@1",
        "video": str(video) if video else None,
        "canvas": {"width": BUILD_W, "height": BUILD_H, "fps": measured["media"]["fps"],
                   "ref_duration_s": measured["media"]["duration_s"], "source": "measured"},
        "sources_legend": {"measured": "ffmpeg/numpy on the file", "frame": "numpy on decoded frames",
                           "detail-pass": "model, timecoded high-fps window (wins inside its window)",
                           "pass1": "model, whole clip", "inference": "derived / house style", "unclear": "cannot tell"},
        "style_recipe": recipe,
        "beat_map": beats,
        "contradictions": contradictions,
        "frame_measurements": {k: (frames or {}).get(k) for k in ("entries_summary", "entries", "overlays", "dynamic_badges",
                                                                  "entries_limits", "overlay_limits")},
        "vo": vo,
        "sound": sound,
        "fonts": fonts,
        "copy_values": copy_values,
        "assets": assets(a, frames),
        "do_not": tag_do_not(a),
        "qa_checklist": qa_checklist(stats, beats, vo, frames, caps),
        "ref_specific_notes": ref_notes,
        "pacing_stats": stats,
    }
    brief["cooked"] = cooked_style(brief, measured, a)
    return brief


def _gsap_stagger(e: dict) -> str:
    tr = e.get("travel") or {}
    p = e.get("parts") or []
    if not p:
        return ""
    first = (f'gsap.from(part1, {{opacity: 0, y: {tr.get("px_1080x1920", 0)}, duration: {tr.get("ms", p[0]["ms"]) / 1000}, '
             f'ease: "{tr.get("ease", "power1.out")}"}})' if tr else
             f'gsap.from(part1, {{opacity: 0, duration: {p[0]["ms"] / 1000}, ease: "power1.out"}})')
    rest = "; ".join(f'gsap.from(part{k + 1}, {{opacity: 0, duration: {q["ms"] / 1000}, ease: "power1.out", '
                     f'delay: {q["offset_ms"] / 1000}}})' for k, q in enumerate(p[1:], 1))
    return first + ("; " + rest if rest else "")


def cooked_style(brief: dict, measured: dict, a: dict) -> dict:
    """Same field names as ai-video-editor-cooked creator-teardown style.json where the data exists."""
    vo, st = brief["vo"], brief["pacing_stats"]
    fm = brief["frame_measurements"]
    cap = brief["style_recipe"]["caption_system"]
    y = re.findall(r"(\d+(?:\.\d+)?)", str((cap.get("position_pct_from_top") or {}).get("value", "")))
    # graphics.entrances: aggregated groups, most used first (contract enum: cut|slide|scale|fade|mask)
    groups: dict = {}
    for e in fm.get("entries") or []:
        tr = e.get("travel")
        kind = ("cut" if e["entry_frames"] <= 2 else "scale" if e["overshoot_pct"] > 0 else
                "slide" if tr else "fade")
        g = groups.setdefault((kind, e["ease"]["gsap"]), {"kind": kind, "ease": e["ease"]["gsap"], "durs": [], "ov": [],
                                                           "dist": [], "dir": None, "n": 0})
        g["n"] += 1
        g["durs"].append((tr or {}).get("ms", e["entry_ms"]) / 1000 if kind == "slide" else e["entry_ms"] / 1000)
        g["ov"].append(e["overshoot_pct"] / 100)
        if tr:
            g["dist"].append(abs(tr["px_1080x1920"]) / 19.2)
            g["dir"] = "bottom" if tr["direction"] == "up" else "top"
    n_ent = sum(g["n"] for g in groups.values()) or 1
    ent = []
    for g in sorted(groups.values(), key=lambda g: -g["n"]):
        row = {"kind": g["kind"], "ease": g["ease"], "duration_s": round(float(np.median(g["durs"])), 3),
               "overshoot": round(float(np.median(g["ov"])), 3), "fade": g["kind"] in ("fade", "slide"),
               "share_pct": round(100 * g["n"] / n_ent), "n": g["n"]}
        if g["kind"] == "slide":
            row.update({"from": g["dir"], "distance_pct": round(float(np.median(g["dist"])), 2)})
        ent.append(row)
    # pace.cut_kinds: % per contract kind, from pass1 transition types (cut times are measured)
    kmap = ((r"dissolve|cross ?fade|fade", "dissolve"), (r"whip", "whip"), (r"jump", "jump"),
            (r"zoom|punch", "zoom-punch"), (r"match", "match"), (r"mask|wipe|slide|panel", "mask"), (r"cut", "hard"))
    ck: dict[str, int] = {}
    for e in a.get("transitions", {}).get("events", []):
        t = str(e.get("type", "")).lower()
        k = next((v for pat, v in kmap if re.search(pat, t)), None)
        if k:
            ck[k] = ck.get(k, 0) + 1
    nck = sum(ck.values()) or 1
    cap_text = " ".join(str((cap.get(k) or {}).get("value", "")) for k in ("stagger_ms_per_word", "entry_animation"))
    anim = ("word_highlight" if re.search(r"highlight", cap_text, re.I) else "slide" if re.search(r"slide", cap_text, re.I)
            else "pop" if re.search(r"pop|word|build|scale", cap_text, re.I) else "none")
    sfx = brief["sound"]
    return {
        "pace": {"wpm": vo.get("wpm"), "max_pause_s": (vo.get("gap_s") or {}).get("max"),
                 "cuts_per_10s": st["cuts_per_10s"]["overall"], "median_shot_s": st["shot_len_s"]["p50"],
                 "cut_kinds": {k: round(100 * v / nck) for k, v in sorted(ck.items(), key=lambda kv: -kv[1])}},
        "captions": {"present": a.get("captions", {}).get("present"),
                     "y_pct": round(sum(float(v) for v in y[:2]) / len(y[:2]), 1) if y else None,
                     "animation": anim,
                     "entrance": (cap.get("entry_animation") or {}).get("value"),
                     "stagger_note": (cap.get("stagger_ms_per_word") or {}).get("value"),
                     "ease_s": None, "source": "edit-mode brief"},
        "graphics": {"entrances": ent,
                     "layout": {"overlays": [{"id": o["id"], "box_pct": o["pct"], "zone": o["zone"]}
                                             for o in (fm.get("overlays") or []) if o["kind"].startswith("persistent")]}},
        "sound": {"sfx_per_min": None, "kinds": None, "on_cut_pct": None,
                  "note": "unclear offline (earshot paused); see sound.house_style",
                  "music": {"present_pct": None, "level_db": None},
                  "clip_integrated_lufs": measured.get("loudness", {}).get("integrated_lufs")},
        "hook": {"first_cut_s": measured["cuts"]["hard_cuts"][0]["t"] if measured["cuts"]["hard_cuts"] else None,
                 "first_graphic_s": min([e["t_start_s"] for e in fm.get("entries") or []] +
                                        [d["t_start_s"] for d in fm.get("dynamic_badges") or []] or [None]) if
                 (fm.get("entries") or fm.get("dynamic_badges")) else None,
                 "first_caption_s": next((e.get("t_s") for e in a.get("captions", {}).get("examples", [])), None)},
        "_sources": {"pace.wpm": "detail-pass caption word times (proxy)", "pace.cut_kinds": "pass1 transition types, % of events",
                     "pace.cuts_per_10s/median_shot_s": "measured", "captions": "edit-mode brief (detail pass wins)",
                     "graphics.entrances": "frame-measured (numbers never from a model)",
                     "graphics.layout": "frame-measured persistent overlays", "sound": "unclear offline; level_db under voice not separable",
                     "hook": "measured cuts / frame entries / pass1 caption examples"},
    }


# ----------------------------------------------------------------------------- markdown


def _v(x) -> str:
    if isinstance(x, dict) and "value" in x:
        val = x["value"]
        val = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else str(val)
        return f"{val} _[{x['source']}]_" + (f" — {x['note']}" if x.get("note") else "")
    return json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)


def render_brief_md(b: dict) -> str:
    L = []
    r = b["style_recipe"]
    L.append("## 13. Edit brief for Scott (deterministic; every value tagged with its source)\n")
    L.append("Sources: **measured** ffmpeg/numpy · **frame** numpy on frames · **detail-pass** model, timecoded window "
             "(wins inside its window) · **pass1** model, whole clip · **inference** · **unclear**\n")
    L.append("### 13a. STYLE RECIPE (portable rules)\n")
    L.append(f"- **Cut-rate band:** {_v(r['cut_rate_band'])}")
    L.append(f"- **Shot length p25/p50/p75 (s):** {_v(r['shot_length_s'])}")
    L.append(f"- **Emphasis rule:** {_v(r['emphasis_rule'])}")
    L.append(f"- **Transition vocabulary:** {_v(r['transition_vocabulary'])}")
    L.append(f"- **VO pace:** {_v(r['vo_pace'])}")
    L.append("- **Caption system:**")
    for k, v in r["caption_system"].items():
        L.append(f"  - {k}: {_v(v)}")
    L.append("- **Motion (frame-measured first):**")
    for k, v in r["motion"].items():
        if isinstance(v, list):
            for x in v:
                L.append(f"  - {k}: {_v(x)}")
        elif v:
            L.append(f"  - {k}: {_v(v)}")
    L.append("- **Colour logic:** " + ", ".join(f"{x.get('role')} {x.get('hex')}" for x in r["colour_logic"]["value"]["roles"])
             + f"; grade: {r['colour_logic']['value'].get('grade')} _[{r['colour_logic']['source']}]_")
    L.append("- **Sound:**")
    for k, v in r["sound"].items():
        L.append(f"  - {k}: {_v(v)}")
    L.append("- **Typography (free installed equivalents):**")
    for f in b["fonts"]:
        if f.get("build_with"):
            L.append(f"  - {f['role']}: **{f['build_with']}** {f.get('style')} ({f.get('licence')}; "
                     f"{'installed' if f.get('installed') else 'NOT installed'}) for ref '{f['reference_guess']}' · {f.get('size')}")
        else:
            L.append(f"  - {f['role']}: {f.get('note')}")
    L.append("")
    L.append("### 13b. BEAT MAP (% of runtime)\n")
    L.append("| # | Beat | Start % | End % | Share % | Cuts/10 s | Median shot s | Still |")
    L.append("|---|---|---|---|---|---|---|---|")
    for x in b["beat_map"]:
        st = f"![]({x['still']})" if x.get("still") else ""
        L.append(f"| {x['n']} | {x['label']} ({x['start_s']}–{x['end_s']} s) | {x['start_pct']} | {x['end_pct']} | "
                 f"{x['share_pct']} | {x['cuts_per_10s']} | {x['median_shot_s']} | {st} |")
    L.append("\n_Boundaries: pass1 snapped to measured cuts; percentages and cut stats measured._\n")

    fm = b["frame_measurements"]
    L.append("### 13c. Measured timing (frames → ms → GSAP)\n")
    s = fm.get("entries_summary") or {}
    if s.get("n"):
        L.append(f"{s['n']} settled entries · p25/p50/p75 {s['entry_ms_p25']}/{s['entry_ms_p50']}/{s['entry_ms_p75']} ms · "
                 f"hard pops (≤2 frames) {s['hard_pops_le_2_frames']} · eases {s['ease_counts']} _[frame]_\n")
        L.append("| Start s | Settle s | Frames | ms | Kind | Overshoot % | Travel | GSAP | Zone | Box @1080x1920 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for e in fm["entries"]:
            p = e["px_1080x1920"]
            tr = e.get("travel")
            trs = f"{tr['direction']} {abs(tr['px_1080x1920'])} px / {tr['ms']} ms `{tr['ease']}`" if tr else "–"
            kind = e.get("kind", "")
            if e.get("parts"):
                kind += ": " + ", ".join(f"+{q['offset_ms']} ms ({q['ms']} ms)" for q in e["parts"])
            L.append(f"| {e['t_start_s']} | {e['t_settle_s']} | {e['entry_frames']} | {e['entry_ms']} | {kind} | "
                     f"{e['overshoot_pct']} | {trs} | `{e['ease']['gsap']}` | {e['zone']} | x{p['x']} y{p['y']} w{p['w']} h{p['h']} |")
        for e in fm["entries"]:
            sk = _gsap_stagger(e) if e.get("parts") else ""
            if sk:
                L.append(f"\n- {e['t_start_s']} s GSAP sketch: `{sk}`")
        L.append(f"\n_Limits: {fm.get('entries_limits')}_\n")
    L.append("### 13d. Persistent overlays & in-place builds\n")
    rows = (fm.get("overlays") or []) + (fm.get("dynamic_badges") or [])
    if rows:
        L.append("| id | Kind | Span s | Zone | % box (x,y,w,h) | px @1080x1920 | Crop |")
        L.append("|---|---|---|---|---|---|---|")
        for o in rows:
            p, q = o["pct"], o["px_1080x1920"]
            L.append(f"| {o['id']} | {o['kind']} | {o['t_start_s']}–{o['t_end_s']} | {o['zone']} | "
                     f"{p['x']},{p['y']},{p['w']},{p['h']} | {q['x']},{q['y']},{q['w']},{q['h']} | "
                     + (f"![]({o['crop_still']})" if o.get("crop_still") else "") + " |")
        L.append(f"\n_Limits: {fm.get('overlay_limits')}_\n")
    else:
        L.append("_none found_\n")
    vo = b["vo"]
    L.append("### 13e. VO measurement\n")
    if vo.get("wpm"):
        c = vo.get("cut_vs_word") or {}
        L.append(f"- Window {vo.get('window_s')} s: {vo['words']} words, **{vo['wpm']} wpm**; gaps p25/p50/p75/p90 "
                 f"{vo['gap_s']['p25']}/{vo['gap_s']['p50']}/{vo['gap_s']['p75']}/{vo['gap_s']['p90']} s, max {vo['gap_s']['max']} s; "
                 f"phrase breaks (≥0.3 s onset gaps): {vo['phrase_gaps_ge_0p3s']} = {vo['phrase_breaks_per_10s']} per 10 s "
                 f"_[detail-pass words]_ ({vo['gap_note']})")
        L.append(f"- Cuts vs word onsets: {c.get('cuts_in_window')} cuts; **{c.get('within_80ms_pct')}% within ±80 ms**, "
                 f"{c.get('earlier_than_word_pct')}% earlier, {c.get('later_than_word_pct')}% later; median |offset| "
                 f"{c.get('abs_offset_ms_p50')} ms. _{c.get('limit')}_")
    if vo.get("pass1_caption_wpm_lower_bound"):
        L.append(f"- Whole clip: {_v(vo['pass1_caption_wpm_lower_bound'])}")
    L.append(f"- {_v(vo['measured_silences'])}\n- _{vo['proxy']}_\n")
    L.append("### 13f. Values to copy\n")
    L.append("| Parameter | Value | Source | Portable |")
    L.append("|---|---|---|---|")
    for cv in b["copy_values"]:
        L.append(f"| {cv['parameter']} | {cv['value']} | {cv['source']} | {'yes' if cv['portable'] else 'ref-specific'} |")
    L.append("\n### 13g. Assets to source/build (reference material is STUDY-ONLY)\n")
    for x in b["assets"]:
        L.append(f"- [{x['kind']}] {x['item']}" + (f" — {x.get('count_in_ref')} shots in ref" if x.get("count_in_ref") else "")
                 + (f" · {x['to_do']}" if x.get("to_do") else "") + f" _[{x['source']}, study-only]_")
    L.append("\n### 13h. Do not\n")
    for x in b["do_not"]:
        L.append(f"- **[{x['scope']}]** {x['rule']}" + (f" — _{x['note']}_" if x.get("note") else ""))
    L.append("\n### 13i. QA checklist (ranges/ratios)\n")
    for x in b["qa_checklist"]:
        L.append(f"- [ ] {x['check']} _[{x['source']}]_")
    if b.get("ref_specific_notes"):
        L.append("\n_Reference-specific counts (do not copy as targets): " + "; ".join(b["ref_specific_notes"]) + "_")
    L.append("\n### 13j. Contradictions reconciled\n")
    if not b["contradictions"]:
        L.append("_none found_")
    for c in b["contradictions"]:
        L.append(f"- **{c['field']}** → winner: {c['winner']}")
        for k in ("pass1", "detail_pass", "frame", "measured"):
            if k in c:
                L.append(f"  - {k}: {_v(c[k])}")
        for row in c.get("rows", []):
            L.append(f"  - '{row['text']}': pass1 {row['pass1_t_s']} s vs detail {row['detail_t_s']} s (Δ {row['delta_ms']} ms)")
    L.append("")
    return "\n".join(L)
