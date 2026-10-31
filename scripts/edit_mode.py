"""Edit / craft analysis mode: schema, prompts, detail pass, reconciliation and Markdown rendering.

Pipeline (see run_edit_mode):
  1. edit_measure.measure()  -> local ground truth (basis = measured)
  2. Pass 1: whole clip at 2-5 fps, structured JSON against EDIT_SCHEMA, measured block in prompt
  3. Pass 2: ffmpeg-cut densest-cut window at ~10 fps -> frame-level motion timing (DETAIL_SCHEMA)
  4. Reconcile model-reported cut/transition times against measured cuts
  5. Write analysis.json + report.md
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASIS = ["measured", "estimated", "inference", "unclear"]

FACT = {
    "type": "object",
    "properties": {
        "aspect": {"type": "string", "description": "What is being described, e.g. 'zoom speed on punch-ins'"},
        "value": {"type": "string", "description": "The value; use numbers + units wherever possible"},
        "basis": {"type": "string", "enum": BASIS},
        "timestamps": {"type": "string", "description": "MM:SS.s or ranges where observed; '' if global"},
    },
    "required": ["aspect", "value", "basis"],
}
FACTS = {"type": "array", "items": FACT}


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or list(props)}


def _arr(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "array", "items": _obj(props, required)}


S = {"type": "string"}
N = {"type": "number"}
B = {"type": "string", "enum": BASIS}
def _slots(*names: str) -> dict:
    """Spec slots as a compact array of {slot, value, basis}; slot names are an enum. (An object with one
    required property per slot is clearer but exceeds Gemini's schema-complexity limit.) Coverage of every
    slot is requested in the prompt and checked in code (see fill_missing_slots)."""
    return {"type": "array", "items": {"type": "object", "properties": {
        "slot": {"type": "string", "enum": list(names)}, "value": S, "basis": B}, "required": ["slot", "value", "basis"]}}


SPEC_SLOTS = {
    "camera_framing": ("subject_placement_9x16", "headroom_pct", "punch_ins_scale_pct", "zoom_speed", "zoom_frequency",
                       "crop_reframe_changes", "shake_handheld", "safe_zone_compliance"),
    "broll": ("trigger_pattern", "typical_duration_s", "presentation_mix", "entry_style", "exit_style", "share_of_runtime_pct"),
    "typography_layout": ("font_pair", "weights", "case_style", "headline_size_pct_h", "body_size_pct_h", "line_spacing",
                          "letter_spacing", "alignment", "side_margin_pct_w", "vertical_position_pct_h", "grid_notes"),
    "captions": ("position_pct_from_top", "lines_per_caption", "words_per_line", "font_and_size", "colour_and_stroke_shadow",
                 "highlight_style", "entry_animation", "entry_duration_ms", "easing", "stagger_ms_per_word", "exit_animation",
                 "sync_to_speech"),
    "motion_language": ("easing_family", "gsap_ease_equivalent", "overshoot_pct", "entry_duration_ms", "exit_duration_ms",
                        "stagger_ms", "motion_blur", "scale_range", "travel_distance_pct", "hold_time_s", "overall_energy"),
    "colour": ("background", "primary_text", "accent", "highlight", "contrast", "saturation", "warmth"),
    "sound_design": ("music_genre_energy", "music_level_vs_voice", "ducking", "sfx_density", "sfx_palette",
                     "sfx_to_cut_sync", "voice_character"),
}

EDIT_SCHEMA = _obj({
    "overview": _obj({
        "format": S, "length": S, "platform_guess": S, "feel_one_line": S, "facts": FACTS,
    }),
    "structure_pacing": _obj({
        "hook_0_3s": S,
        "beats": _arr({"start_s": N, "end_s": N, "label": S, "description": S, "basis": B}),
        "breathing_points": _arr({"t_s": N, "description": S, "basis": B}),
        "ending_cta": S,
        "facts": FACTS,
    }),
    "camera_framing": _obj({"spec": _slots(*SPEC_SLOTS["camera_framing"]), "facts": FACTS}),
    "broll": _obj({
        "spec": _slots(*SPEC_SLOTS["broll"]),
        "placements": _arr({"start_s": N, "end_s": N, "content": S, "trigger": S,
                            "presentation": {"type": "string", "enum": ["full-screen", "pip", "split", "overlay", "card", "other", "unclear"]},
                            "entry": S, "exit": S, "basis": B}),
        "facts": FACTS,
    }),
    "typography_layout": _obj({
        "spec": _slots(*SPEC_SLOTS["typography_layout"]),
        "fonts": _arr({"role": S, "font_guess": S, "weight": S, "case": S,
                       "size_pct_of_frame_height": S, "colour_hex": S, "basis": B}),
        "facts": FACTS,
    }),
    "captions": _obj({
        "present": {"type": "boolean"},
        "spec": _slots(*SPEC_SLOTS["captions"]),
        "examples": _arr({"t_s": N, "text": S, "highlighted_word": S}),
        "facts": FACTS,
    }),
    "motion_language": _obj({"spec": _slots(*SPEC_SLOTS["motion_language"]), "facts": FACTS}),
    "elements_assets": _obj({
        "elements": _arr({"t_in_s": N, "t_out_s": N, "element": S, "motion_in": S, "motion_out": S,
                          "layering_depth": S, "basis": B}),
        "facts": FACTS,
    }),
    "transitions": _obj({
        "events": _arr({"t_s": N, "type": S, "duration_ms": N, "purpose": S, "basis": B}),
        "facts": FACTS,
    }),
    "colour": _obj({
        "spec": _slots(*SPEC_SLOTS["colour"]),
        "roles": _arr({"role": S, "hex": S, "basis": B}),
        "grade": S, "texture_grain": S, "facts": FACTS,
    }),
    "sound_design": _obj({
        "music": S, "ducking": S,
        "spec": _slots(*SPEC_SLOTS["sound_design"]),
        "sfx_events": _arr({"t_s": N, "sound": S, "linked_visual_event": S, "level_vs_voice": S, "basis": B}),
        "facts": FACTS,
    }),
    "retention_devices": _arr({"t_s": N, "device": S, "description": S, "basis": B}),
    "cut_reconciliation": _arr({"measured_t_s": N, "what_happens": S, "is_real_cut": {"type": "string", "enum": ["yes", "no", "unclear"]}}),
    "missed_by_measurement": _arr({"t_s": N, "event": S}),
    "edit_brief": _obj({
        "copy_values": _arr({"parameter": S, "value": S, "basis": B}),
        "do_not": {"type": "array", "items": S},
        "checklist": {"type": "array", "items": S},
    }),
})

DETAIL_SCHEMA = _obj({
    "summary": S,
    "shot_inventory": _arr({"start_s": N, "end_s": N, "content": S}),
    "events": _arr({
        "t_start_s": N, "t_end_s": N, "element": S,
        "action": {"type": "string", "enum": ["enter", "exit", "move", "scale", "rotate", "transition",
                                              "caption", "camera", "speed-ramp", "other"]},
        "duration_ms": N, "easing": S, "overshoot": S, "stagger_ms": S, "motion_blur": S,
        "scale_from_to": S, "position_from_to": S, "notes": S, "basis": B,
    }, ["t_start_s", "element", "action", "duration_ms", "easing", "basis"]),
    "caption_word_timing": _arr({"t_s": N, "word": S}),
})

GROUNDING = """Rules:
1. Report only what the video supports. Do not turn likely context into fact.
2. Never invent a creator, speaker name, narrator, brand, product, dialogue, visible text, or voiceover.
3. Quote visible words or speech verbatim only when you can discern them.
4. 'unclear' is a valid answer for any field. Accuracy beats a complete-looking report.
5. Every value carries a basis:
   - measured: copied from the MEASURED block below (never use this for your own estimates)
   - estimated: your own visual/audio estimate of a number (durations, sizes, hex colours, offsets)
   - inference: interpretation of intent, feel, purpose or strategy
   - unclear: you cannot tell
6. Font names are always guesses: write them as 'looks like X (or similar geometric sans)' with basis estimated.
"""

EDIT_PROMPT = """You are a senior short-form video editor and motion designer reverse-engineering a REFERENCE video
so another editor (who builds in HyperFrames: HTML + GSAP, default 9:16, 1080x1920) can recreate its look and feel
from exact numbers. Analyse the CRAFT (edit, motion, type, captions, colour, sound), not the topic.

{grounding}
MEASURED block (ground truth from ffmpeg/numpy; trust it over your own perception for timing/counts):
- Times are seconds from the start. Hard cuts = ffmpeg scene score >= {threshold}. Soft events are smaller visual
  changes (dissolves, whips, flashes, big motion, or graphics changing over the same footage).
- Use measured cut times when you place beats, B-roll and transitions. If you see a cut the measurement missed,
  list it in missed_by_measurement. For EVERY measured hard cut, add a cut_reconciliation row describing what
  actually happens there (a cut to B-roll, a graphic change, a flash, a false positive ...).
- onsets / cut_audio_sync are rough audio transients, not labelled SFX. Use them only as hints for sound design.

```json
{measured}
```

Section guidance (be specific; numbers + units; seconds as decimals; durations in ms):
- overview: format (talking head / voiceover + B-roll / screen recording / motion graphics ...), length, platform guess, one-line feel (4-6 adjectives).
- structure_pacing: hook 0-3s (what is seen/heard, what earns the next second); beats with start/end; breathing points (holds, pauses); ending/CTA. Facts should include avg shot length and cuts/sec copied from MEASURED.
- camera_framing: subject placement in the 9:16 frame (% from top, head room, centre/offset), punch-ins/zooms (scale % and speed), crop/reframe changes, shake/handheld, safe zones (does text avoid top 10% / bottom 20% / right rail?).
- broll: when it appears (on keywords, on pauses, on a rhythm), durations, full-screen vs PiP vs card, entry/exit motion.
- typography_layout: font pair (guesses), weights, case, sizes as % of frame height, colours, grid, margins (% of frame width), alignment.
- captions: layout (position, lines, words per line), highlight style (colour/box/scale), motion: entry type, duration ms, easing, stagger ms per word, exit.
- motion_language: easing family (spring / snappy expo-out / smooth sine), overshoot %, typical entry and exit durations ms, stagger ms, motion blur yes/no, scale ranges, GSAP-style ease suggestions (label estimated).
- elements_assets: UI cards, icons, stickers, emoji, arrows, shapes: how each moves and its layering (in front of/behind subject, shadow, blur, glass).
- transitions: each type with time, duration ms, frequency, purpose.
- colour: palette as roles (background, primary text, accent, highlight, secondary) with hex estimates (MEASURED palette is whole-frame dominant colours; use it to anchor hexes), grade (contrast, warmth, saturation), texture/grain.
- sound_design: music energy/genre feel, ducking under voice, SFX per visual event (whoosh on transition, pop on caption ...), level vs voice, sync to cuts.
- retention_devices: pattern interrupts, open loops, callbacks, progress cues, payoff timing.
- edit_brief: the exact values Scott should copy (fps, cut rhythm, caption spec, ease curves, durations, colours, sizes), a do_not list, and a QA checklist.

Depth requirements (only where the video supports it; never pad with invented detail):
- Every section's 'spec' array must contain ONE row for EVERY allowed slot name: give a concrete value with units, or 'unclear' (basis unclear). Never leave it vague
  when the frames show it (e.g. 'bottom third, ~68% from top', not 'lower area').
- Then 3-10 extra facts each for camera_framing, typography_layout, captions, motion_language, colour, sound_design.
- List EVERY B-roll placement, transition, on-screen element/asset and caption style change you can see, not a sample.
- captions.examples: one row per distinct caption/text card (up to 25).
- edit_brief.copy_values: 12-25 concrete parameters (durations ms, ease names, sizes %, hexes, cut rhythm, caption spec).
- basis 'measured' is ONLY for values copied from the MEASURED block. A beat whose boundaries come from measured cuts but
  whose description is yours is 'estimated'. Font sizes, alignment, hexes you read off frames are 'estimated'.

Return JSON only, matching the schema.
"""

DETAIL_PROMPT = """This clip is a {length:.1f}s window ({start:.2f}s to {end:.2f}s of the full video) sampled at {fps} fps.
A black strip has been ADDED below the original frame. It shows the ABSOLUTE timestamp of each frame in the full video
as 't=SS.sss' and the frame number. The strip is not part of the video design: ignore it for layout, colour and
typography. Read times from the strip and return ABSOLUTE seconds (as shown in the strip).

{grounding}
Hard cuts in this window are already measured exactly (absolute s): {cuts}. Do NOT list cuts as events.
Shots in this window (absolute s): {shots}

Task: describe the MOTION inside these shots, as a motion designer would need to rebuild it in GSAP.
1. shot_inventory: for each listed shot, what is on screen (read the strip to make sure you match the right shot).
2. events: every element entry/exit/move/scale/rotate (caption words, title text, cards, icons, B-roll windows,
   camera zooms/pans, speed ramps). For each: absolute start/end, duration ms, easing (e.g. 'expo-out ~power4.out',
   'spring ~8% overshoot', 'linear'), overshoot, stagger ms between words/items, motion blur, scale from/to and
   position from/to (as % of frame).
3. caption_word_timing: when each caption word first appears (absolute s), if captions are present.
Anything faster than one sampled frame ({frame_ms:.0f} ms) cannot be seen: say 'under {frame_ms:.0f} ms' rather than guessing.
Return JSON only, matching the schema.
"""


def default_edit_fps(duration: float) -> float:
    if duration <= 30:
        return 5
    if duration <= 60:
        return 3
    if duration <= 90:
        return 2
    return 1


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


def usage_dict(response) -> dict:
    u = getattr(response, "usage_metadata", None)
    if not u:
        return {}
    out = {}
    for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count"):
        v = getattr(u, k, None)
        if v is not None:
            out[k] = v
    details = getattr(u, "prompt_tokens_details", None) or []
    for d in details:
        mod = getattr(getattr(d, "modality", None), "name", str(getattr(d, "modality", "")))
        out[f"prompt_{mod.lower()}_tokens"] = getattr(d, "token_count", None)
    return out


TRANSIENT = ("Server disconnected", "RemoteProtocolError", "ReadTimeout", "429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED", "overloaded")


def generate_json(client, types, model: str, parts: list, schema: dict, label: str, retries: int = 2,
                  transient_retries: int = 5):
    """Structured output with response_json_schema; fall back to plain JSON mode + loose parse.
    Transient API errors (429/5xx, 'high demand') back off exponentially without burning a config."""
    attempts = [
        ("schema", types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=schema,
                                               temperature=0.2)),
        ("json-mime", types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)),
    ]
    last_error = None
    usage_total: list[dict] = []
    for name, config in attempts:
        parse_failures = transient = 0
        while parse_failures < retries and transient <= transient_retries:
            try:
                print(f"[info] {label}: requesting structured JSON ({name})...", file=sys.stderr)
                resp = client.models.generate_content(model=model, contents=types.Content(parts=parts), config=config)
                usage_total.append(usage_dict(resp))
                return parse_json_loose(resp.text), {"config": name, "calls": len(usage_total), "usage": usage_total}
            except ValueError as error:  # JSON parse problem -> retry same config
                last_error = error
                parse_failures += 1
                print(f"[warning] {label}: could not parse JSON ({error}); retrying", file=sys.stderr)
            except Exception as error:
                last_error = error
                msg = str(error)
                if "PerDay" in msg or "per day" in msg.lower():
                    raise RuntimeError(f"{label}: daily Gemini quota exhausted; try tomorrow or use a paid key") from error
                if any(code in msg for code in TRANSIENT):
                    transient += 1
                    wait = min(10 * 2 ** (transient - 1), 90)
                    print(f"[warning] {label}: transient API error ({msg[:80]}...); waiting {wait}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                print(f"[warning] {label}: API rejected request ({msg[:300]}); trying next config", file=sys.stderr)
                break  # e.g. schema rejected -> next config
        if transient > transient_retries:
            break  # service unavailable; switching config will not help
    raise RuntimeError(f"{label} failed: {str(last_error)[:300]}")


def _font_file() -> str | None:
    try:
        out = subprocess.run(["fc-match", "-f", "%{file}", "monospace"], capture_output=True, text=True, check=False)
        return out.stdout.strip() or None
    except FileNotFoundError:
        return None


def generate_json_with_fallback(client, types, args, parts, schema, label):
    try:
        return generate_json(client, types, args.model, parts, schema, label)
    except RuntimeError as error:
        fb = getattr(args, "fallback_model", None)
        daily = "daily Gemini quota" in str(error)
        if not fb or fb == args.model or not (daily or any(c in str(error) for c in TRANSIENT)):
            raise
        why = "daily free-tier quota used up (quotas are per model)" if daily else "overloaded"
        print(f"[warning] {label}: {args.model} {why}; falling back to {fb}", file=sys.stderr)
        data, info = generate_json(client, types, fb, parts, schema, f"{label} (fallback {fb})")
        info["model"] = fb
        return data, info


def cut_segment(path: Path, start: float, end: float, out_dir: Path, timecode: bool = True) -> Path:
    """Accurate re-encoded segment. With timecode=True, pads a black strip under the frame and burns the
    ABSOLUTE source time + frame number into it (original pixels untouched), so the model reads time instead
    of estimating it."""
    out = out_dir / f"detail_{start:.2f}-{end:.2f}.mp4"
    vf = []
    font = _font_file() if timecode else None
    if timecode and font:
        vf = ["-vf", (f"pad=iw:ih+ih/14:0:0:black,drawtext=fontfile='{font}':fontcolor=white:fontsize=h/30:"
                      f"x=(w-tw)/2:y=h-(h/14)/2-th/2-th/4:"
                      f"text='t=%{{eif\\:trunc(t+{start:.3f})\\:d}}.%{{eif\\:mod((t+{start:.3f})*1000\\,1000)\\:d\\:3}}s  f=%{{frame_num}}'")]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(path),
           "-t", f"{end - start:.3f}", *vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-c:a", "aac", "-b:a", "128k", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        if vf:  # timecode overlay failed (fonts/filters) -> plain segment
            print(f"[warning] timecode overlay failed ({res.stderr[:150]}); sending plain segment", file=sys.stderr)
            return cut_segment(path, start, end, out_dir, timecode=False)
        raise RuntimeError(f"ffmpeg segment cut failed: {res.stderr[:300]}")
    return out


def reconcile(measured: dict, analysis: dict, tol: float = 0.25) -> dict:
    """How well do model-reported transition/cut times line up with measured hard cuts?"""
    cuts = [c["t"] for c in measured["cuts"]["hard_cuts"]]
    model_times = [e.get("t_s") for e in analysis.get("transitions", {}).get("events", []) if isinstance(e.get("t_s"), (int, float))]
    model_times += [b.get("start_s") for b in analysis.get("broll", {}).get("placements", []) if isinstance(b.get("start_s"), (int, float))]
    model_times = sorted(set(round(t, 2) for t in model_times))
    rows = []
    for t in model_times:
        nearest = min(cuts, key=lambda c: abs(c - t)) if cuts else None
        rows.append({"model_t": t, "nearest_measured_cut": nearest,
                     "delta_s": round(t - nearest, 3) if nearest is not None else None,
                     "match": nearest is not None and abs(t - nearest) <= tol})
    rec = analysis.get("cut_reconciliation", [])
    covered = {round(r.get("measured_t_s", -1), 2) for r in rec}
    return {
        "tolerance_s": tol,
        "model_event_times_checked": len(rows),
        "model_events_matching_a_measured_cut": sum(1 for r in rows if r["match"]),
        "measured_cuts": len(cuts),
        "measured_cuts_explained_by_model": sum(1 for c in cuts if any(abs(c - x) <= 0.05 for x in covered)),
        "model_says_false_positive": [r.get("measured_t_s") for r in rec if r.get("is_real_cut") == "no"],
        "model_extra_events": analysis.get("missed_by_measurement", []),
        "rows": rows,
    }


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(text))]


def fill_missing_slots(analysis: dict) -> dict:
    """Normalise spec arrays into ordered dicts and mark any slot the model skipped."""
    missing = {}
    for sec_name, names in SPEC_SLOTS.items():
        sec = analysis.get(sec_name)
        if not isinstance(sec, dict):
            continue
        rows = sec.get("spec") or []
        got = {}
        if isinstance(rows, list):
            for r in rows:
                if isinstance(r, dict) and r.get("slot") in names and r["slot"] not in got:
                    got[r["slot"]] = {"value": r.get("value"), "basis": r.get("basis")}
        elif isinstance(rows, dict):
            got = {k: v for k, v in rows.items() if isinstance(v, dict)}
        miss = [n for n in names if n not in got]
        for n in miss:
            got[n] = {"value": "not returned by model", "basis": "unclear"}
        sec["spec"] = {n: got[n] for n in names}
        if miss:
            missing[sec_name] = miss
    return missing


def audit_basis(measured_block: dict, analysis: dict) -> list[dict]:
    """Downgrade 'measured' labels the model cannot justify. A fact is kept as measured only if a number in its
    value appears in the MEASURED block (or a hex matches the measured palette). Timed rows tagged measured keep
    that label only if their time matches a measured cut (+-0.05 s); otherwise they become 'estimated'."""
    blob = json.dumps(measured_block)
    measured_nums = set()
    for x in _numbers(blob):
        measured_nums.add(round(x, 2))
        measured_nums.add(round(x, 1))
        measured_nums.add(round(x))
    hexes = {h.upper() for h in re.findall(r"#[0-9A-Fa-f]{6}", blob)}
    cuts = [0.0] + [c["t"] for c in measured_block.get("cuts", {}).get("hard_cuts", [])]
    changes = []

    def check_fact(path: str, f: dict, text: str):
        if f.get("basis") != "measured":
            return
        hx = {h.upper() for h in re.findall(r"#[0-9A-Fa-f]{6}", text)}
        nums = [n for n in _numbers(text) if not (1 <= n <= 9 and ":" in text)]  # ignore '9:16'-style ratios
        ok = (hx and hx <= hexes) or any(round(n, 2) in measured_nums or round(n, 1) in measured_nums for n in nums if n not in (0, 1))
        if not ok:
            f["basis"] = "estimated"
            f["basis_note"] = "downgraded from 'measured' (value not found in measured data)"
            changes.append({"where": path, "value": text[:80]})

    def check_timed(path: str, row: dict, keys: tuple[str, ...]):
        if row.get("basis") != "measured":
            return
        times = [row[k] for k in keys if isinstance(row.get(k), (int, float))]
        if times and all(any(abs(t - c) <= 0.05 for c in cuts) or t >= measured_block["media"]["duration_s"] - 0.1 for t in times):
            row["basis"] = "measured timing"
        else:
            row["basis"] = "estimated"
            row["basis_note"] = "downgraded: times do not match measured cuts"
        changes.append({"where": path, "to": row["basis"]})

    for sec_name, sec in analysis.items():
        if isinstance(sec, dict):
            for f in sec.get("facts", []) or []:
                check_fact(f"{sec_name}.facts", f, str(f.get("value", "")))
            for list_name, keys in (("beats", ("start_s", "end_s")), ("placements", ("start_s", "end_s")),
                                    ("elements", ("t_in_s", "t_out_s")), ("events", ("t_s",)),
                                    ("breathing_points", ("t_s",)), ("sfx_events", ("t_s",))):
                for row in sec.get(list_name, []) or []:
                    check_timed(f"{sec_name}.{list_name}", row, keys)
            for k, slot in (sec.get("spec") or {}).items():
                if isinstance(slot, dict):
                    check_fact(f"{sec_name}.spec.{k}", slot, str(slot.get("value", "")))
            for row in sec.get("roles", []) or []:
                check_fact(f"{sec_name}.roles", row, str(row.get("hex", "")))
            for row in sec.get("fonts", []) or []:
                if row.get("basis") == "measured":
                    row["basis"] = "estimated"
                    changes.append({"where": f"{sec_name}.fonts", "to": "estimated"})
            for row in sec.get("copy_values", []) or []:
                check_fact(f"{sec_name}.copy_values", row, str(row.get("value", "")))
        elif isinstance(sec, list) and sec_name == "retention_devices":
            for row in sec:
                check_timed(sec_name, row, ("t_s",))
    return changes


# ----------------------------------------------------------------------------- rendering

def _fmt_t(t) -> str:
    if not isinstance(t, (int, float)):
        return str(t or "")
    m, s = divmod(float(t), 60)
    return f"{int(m):02d}:{s:04.1f}"


def _cell(v) -> str:
    return str(v if v is not None else "").replace("|", "\\|").replace("\n", " ")


def _table(rows: list[dict], cols: list[tuple[str, str]], times: tuple[str, ...] = ()) -> str:
    if not rows:
        return "_none reported_\n"
    out = "| " + " | ".join(h for _, h in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        out += "| " + " | ".join(_cell(_fmt_t(r.get(k)) if k in times else r.get(k)) for k, _ in cols) + " |\n"
    return out


def _spec(section: dict) -> str:
    spec = section.get("spec") or {}
    if not spec:
        return ""
    rows = [{"k": k.replace("_", " "), "v": (v or {}).get("value"), "b": (v or {}).get("basis")} for k, v in spec.items()]
    return _table(rows, [("k", "Spec"), ("v", "Value"), ("b", "Basis")])


def _facts(section: dict) -> str:
    return _spec(section) + _table(section.get("facts", []), [("aspect", "Aspect"), ("value", "Value"), ("basis", "Basis"), ("timestamps", "When")])


def render_report(path: Path, result: dict) -> str:
    m = result["measured"]
    a = result.get("analysis", {})
    d = result.get("detail_pass")
    rc = result.get("reconciliation", {})
    media = m["media"]
    cuts = m["cuts"]
    L = []
    L.append(f"# Edit analysis: {path.name}\n")
    fb = [v.get("model") for v in result["run"].get("usage", {}).values() if isinstance(v, dict) and v.get("model")]
    L.append(f"Model: `{result['run']['model']}`" + (f" (fallback used: {', '.join(fb)})" if fb else "") + f" · pass-1 fps {result['run']['fps']} · generated {result['run']['generated_at']}  ")
    L.append("Every number is tagged **measured** (ffmpeg/numpy on this file), **estimated** (model's visual/audio estimate), "
             "**inference** (interpretation) or **unclear**.\n")

    L.append("## 0. Measured ground truth (verbatim, basis = measured)\n")
    L.append(f"- Media: {media['width']}x{media['height']} ({media['aspect']}), {media['fps']} fps, {media['duration_s']} s, "
             f"audio: {'yes' if media['has_audio'] else 'no'}")
    L.append(f"- Hard cuts (scene >= {cuts['threshold']}): **{len(cuts['hard_cuts'])}** · shots {cuts['shot_count']} · "
             f"avg shot {cuts['avg_shot_s']} s · median {cuts['median_shot_s']} s · min {cuts['min_shot_s']} s · max {cuts['max_shot_s']} s · "
             f"{cuts['cuts_per_sec']} cuts/s ({cuts['cuts_per_10s']} per 10 s)")
    L.append(f"- Cut times (s): {', '.join(str(c['t']) for c in cuts['hard_cuts']) or 'none'}")
    L.append(f"- Soft visual events (scene {cuts['soft_threshold']}-{cuts['threshold']}) (s): {', '.join(str(c['t']) for c in cuts['soft_events']) or 'none'}")
    dw = m.get("densest_window")
    if dw:
        L.append(f"- Densest window: {dw['start']}-{dw['end']} s ({dw.get('cuts_in_window', '?')} cuts)")
    if m.get("loudness"):
        lo = m["loudness"]
        L.append(f"- Loudness: {lo.get('integrated_lufs')} LUFS integrated · LRA {lo.get('loudness_range_lu')} LU · true peak {lo.get('true_peak_dbtp')} dBTP")
    if "silences" in m:
        L.append(f"- Silences (< -35 dB, >= 0.25 s): {', '.join('%s-%s' % (x['start'], x['end']) for x in m['silences']) or 'none'}")
    if m.get("onsets", {}).get("available"):
        L.append(f"- Audio onsets: {m['onsets']['count']} ({m['onsets']['per_sec']}/s, {m['onsets']['method']})")
    if m.get("cut_audio_sync", {}).get("cuts_total"):
        cs = m["cut_audio_sync"]
        L.append(f"- Cut↔audio transient sync (±{cs['window_ms']} ms, strong onsets): {cs['cuts_with_onset']}/{cs['cuts_total']} "
                 f"= {cs['ratio']} vs chance {cs['chance_baseline_ratio']} → {cs['verdict']}; median offset {cs['median_offset_ms']} ms")
    pal = m.get("palette", {})
    if pal.get("available"):
        L.append(f"- Dominant colours (whole frame): {', '.join('%s %.0f%%' % (c['hex'], c['share'] * 100) for c in pal['dominant'])} · "
                 f"mean luma {pal['mean_luma_0_255']} · mean saturation {pal['mean_saturation_0_255']}")
    L.append("- earshot sound-event timeline: _not integrated yet (paused)_\n")
    L.append("- Shot lengths (s, in order): " + ", ".join(str(sh["dur"]) for sh in cuts["shots"]))
    L.append("- Full measurements (verbatim, incl. per-second visual-change curve, loudness curve, onsets, cut/onset pairs): "
             "`analysis.json` → `measured`\n")
    if result.get("basis_audit"):
        L.append(f"_Basis audit: {len(result['basis_audit'])} model labels of 'measured' were checked/corrected "
                 "('measured timing' = times match measured cuts, description is the model's)._\n")

    if not a:
        L.append("_Model analysis missing (see errors in analysis.json)._\n")
        return "\n".join(L)

    ov = a.get("overview", {})
    L.append("## 1. Overview\n")
    L.append(f"- Format: {ov.get('format')}\n- Length: {ov.get('length')}\n- Platform: {ov.get('platform_guess')}\n- Feel: **{ov.get('feel_one_line')}**\n")
    L.append(_facts(ov))

    sp = a.get("structure_pacing", {})
    L.append("## 2. Structure & pacing\n")
    L.append(f"**Hook (0-3 s):** {sp.get('hook_0_3s')}\n")
    L.append(_table(sp.get("beats", []), [("start_s", "Start"), ("end_s", "End"), ("label", "Beat"), ("description", "What happens"), ("basis", "Basis")], ("start_s", "end_s")))
    L.append("**Breathing points**\n")
    L.append(_table(sp.get("breathing_points", []), [("t_s", "When"), ("description", "Description"), ("basis", "Basis")], ("t_s",)))
    L.append(f"**Ending / CTA:** {sp.get('ending_cta')}\n")
    L.append(_facts(sp))

    L.append("## 3. Camera & framing\n")
    L.append(_facts(a.get("camera_framing", {})))

    br = a.get("broll", {})
    L.append("## 4. B-roll\n")
    L.append(_table(br.get("placements", []), [("start_s", "In"), ("end_s", "Out"), ("content", "Content"), ("trigger", "Trigger"),
                                                ("presentation", "Presentation"), ("entry", "Entry"), ("exit", "Exit"), ("basis", "Basis")], ("start_s", "end_s")))
    L.append(_facts(br))

    ty = a.get("typography_layout", {})
    L.append("## 5. Typography & layout\n")
    L.append(_table(ty.get("fonts", []), [("role", "Role"), ("font_guess", "Font (guess)"), ("weight", "Weight"), ("case", "Case"),
                                          ("size_pct_of_frame_height", "Size % of H"), ("colour_hex", "Colour"), ("basis", "Basis")]))
    L.append(_facts(ty))

    cp = a.get("captions", {})
    L.append("## 6. Captions\n")
    L.append(f"Present: {cp.get('present')}\n")
    L.append(_table(cp.get("examples", []), [("t_s", "When"), ("text", "Text"), ("highlighted_word", "Highlighted")], ("t_s",)))
    L.append(_facts(cp))

    L.append("## 7. Motion language\n")
    L.append(_facts(a.get("motion_language", {})))
    if d and d.get("events"):
        L.append(f"### Frame-level detail pass ({d['window']['start']}-{d['window']['end']} s at {d['fps']} fps, window = {d['window'].get('reason')}; "
                 "times are absolute, read from a burned-in timecode strip)\n")
        if d.get("summary"):
            L.append(d["summary"] + "\n")
        if d.get("shot_inventory"):
            L.append("**Shots in window:** " + "; ".join(f"{_fmt_t(x.get('start_s'))}-{_fmt_t(x.get('end_s'))} {x.get('content')}" for x in d["shot_inventory"]) + "\n")
        L.append(_table(d["events"], [("t_start_s", "Start"), ("t_end_s", "End"), ("element", "Element"), ("action", "Action"),
                                      ("duration_ms", "ms"), ("easing", "Easing"), ("overshoot", "Overshoot"), ("stagger_ms", "Stagger"),
                                      ("scale_from_to", "Scale"), ("position_from_to", "Position"), ("motion_blur", "Blur"), ("basis", "Basis")],
                        ("t_start_s", "t_end_s")))
        if d.get("caption_word_timing"):
            L.append("**Caption word timing (detail pass)**: " + ", ".join(f"{_fmt_t(w.get('t_s'))} {w.get('word')}" for w in d["caption_word_timing"]) + "\n")
    elif result.get("detail_error"):
        L.append(f"_Detail pass failed: {result['detail_error']}_\n")

    el = a.get("elements_assets", {})
    L.append("## 8. Elements & assets\n")
    L.append(_table(el.get("elements", []), [("t_in_s", "In"), ("t_out_s", "Out"), ("element", "Element"), ("motion_in", "Motion in"),
                                             ("motion_out", "Motion out"), ("layering_depth", "Layering / depth"), ("basis", "Basis")], ("t_in_s", "t_out_s")))
    L.append(_facts(el))

    tr = a.get("transitions", {})
    L.append("## 9. Transitions\n")
    L.append(_table(tr.get("events", []), [("t_s", "When"), ("type", "Type"), ("duration_ms", "ms"), ("purpose", "Purpose"), ("basis", "Basis")], ("t_s",)))
    L.append(_facts(tr))

    co = a.get("colour", {})
    L.append("## 10. Colour\n")
    L.append(_table(co.get("roles", []), [("role", "Role"), ("hex", "Hex"), ("basis", "Basis")]))
    L.append(f"- Grade: {co.get('grade')}\n- Texture / grain: {co.get('texture_grain')}\n")
    L.append(_facts(co))

    sd = a.get("sound_design", {})
    L.append("## 11. Sound design\n")
    L.append("> Agents cannot hear precisely; SFX labels come from the model, timing anchors from measured onsets. "
             "earshot will replace this section's timing once enabled.\n")
    L.append(f"- Music: {sd.get('music')}\n- Ducking: {sd.get('ducking')}\n")
    L.append(_table(sd.get("sfx_events", []), [("t_s", "When"), ("sound", "Sound"), ("linked_visual_event", "Visual event"),
                                               ("level_vs_voice", "Level vs voice"), ("basis", "Basis")], ("t_s",)))
    L.append(_facts(sd))

    L.append("## 12. Retention devices\n")
    L.append(_table(a.get("retention_devices", []), [("t_s", "When"), ("device", "Device"), ("description", "Description"), ("basis", "Basis")], ("t_s",)))

    eb = a.get("edit_brief", {})
    L.append("## 13. Edit brief for Scott\n")
    L.append("### Values to copy\n")
    L.append(_table(eb.get("copy_values", []), [("parameter", "Parameter"), ("value", "Value"), ("basis", "Basis")]))
    L.append("### Do not\n" + "\n".join(f"- {x}" for x in eb.get("do_not", [])) + "\n")
    L.append("### QA checklist\n" + "\n".join(f"- [ ] {x}" for x in eb.get("checklist", [])) + "\n")

    L.append("## Appendix: measured vs model reconciliation\n")
    if rc:
        L.append(f"- Model transition/B-roll times checked: {rc['model_event_times_checked']}, within ±{rc['tolerance_s']} s of a measured cut: "
                 f"{rc['model_events_matching_a_measured_cut']}")
        L.append(f"- Measured cuts the model explained: {rc['measured_cuts_explained_by_model']}/{rc['measured_cuts']}")
        L.append(f"- Measured cuts the model calls false positives: {rc['model_says_false_positive'] or 'none'}")
        L.append(f"- Events the model saw that measurement missed: " +
                 (", ".join(f"{_fmt_t(e.get('t_s'))} {e.get('event')}" for e in rc['model_extra_events']) or "none") + "\n")
    L.append(_table(a.get("cut_reconciliation", []), [("measured_t_s", "Measured cut"), ("is_real_cut", "Real cut?"), ("what_happens", "What happens")], ("measured_t_s",)))
    u = result["run"].get("usage", {})
    L.append(f"\n---\nAPI usage: {json.dumps(u)}\n")
    return "\n".join(L)


# ----------------------------------------------------------------------------- orchestration

def run_edit_mode(args, client, types, path: Path, make_video_part, measure_mod) -> tuple[dict, str]:
    print("[info] Measuring locally (ffprobe / scene detection / loudness / onsets / palette)...", file=sys.stderr)
    measured = measure_mod.measure(path, threshold=args.scene_threshold, detail_window=args.detail_window,
                                   detail_target=args.detail_target)
    if args.detail_start is not None:
        start = max(0.0, min(args.detail_start, measured["media"]["duration_s"] - 1.0))
        measured["densest_window"] = {"start": round(start, 3),
                                      "end": round(min(start + args.detail_window, measured["media"]["duration_s"]), 3),
                                      "target": "manual", "reason": "set with --detail-start"}
    duration = measured["media"]["duration_s"]
    if duration > 90 and not args.allow_long:
        raise SystemExit(f"ERROR: edit mode is tuned for short-form (<= 90 s); this clip is {duration:.0f} s. "
                         "Re-run with --allow-long (uses 1 fps) or trim the clip.")
    fps = args.fps or default_edit_fps(duration)
    block = measure_mod.compact_for_prompt(measured)
    prompt = EDIT_PROMPT.format(grounding=GROUNDING, threshold=args.scene_threshold, measured=json.dumps(block, separators=(",", ":")))
    if args.prompt:
        prompt += f"\n\n## Additional user request\n{args.prompt}\n"

    result: dict = {
        "run": {"mode": "edit", "model": args.model, "fps": fps, "video": str(path),
                "generated_at": time.strftime("%Y-%m-%d %H:%M %Z"), "usage": {}},
        "measured": measured,
        "measured_prompt_block": block,
    }
    uploads: list[str] = []
    try:
        part, up = make_video_part(path, fps)
        if up:
            uploads.append(up)
        print(f"[info] Pass 1: whole clip at {fps} fps with {args.model}...", file=sys.stderr)
        analysis, info = generate_json_with_fallback(client, types, args, [part, types.Part(text=prompt)], EDIT_SCHEMA, "pass 1")
        result["analysis"] = analysis
        result["run"]["usage"]["pass1"] = info
        result["missing_spec_slots"] = fill_missing_slots(analysis)
        result["basis_audit"] = audit_basis(block, analysis)
        result["reconciliation"] = reconcile(measured, analysis)

        dw = measured.get("densest_window")
        if not args.no_detail and dw and dw["end"] - dw["start"] >= 1.0:
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    seg = cut_segment(path, dw["start"], dw["end"], Path(tmp), timecode=not args.no_timecode)
                    if args.keep_detail_clip:
                        out_dir = Path(args.out) if args.out else Path("output") / path.stem
                        out_dir.mkdir(parents=True, exist_ok=True)
                        (out_dir / seg.name).write_bytes(seg.read_bytes())
                    dfps = min(args.detail_fps, measured["media"]["fps"] or args.detail_fps)
                    in_win = [c["t"] for c in measured["cuts"]["hard_cuts"] if dw["start"] <= c["t"] < dw["end"]]
                    shots = [[max(sh["start"], dw["start"]), min(sh["end"], dw["end"])] for sh in measured["cuts"]["shots"]
                             if sh["end"] > dw["start"] and sh["start"] < dw["end"]]
                    dprompt = DETAIL_PROMPT.format(length=dw["end"] - dw["start"], start=dw["start"], end=dw["end"], fps=dfps,
                                                   grounding=GROUNDING, cuts=in_win or "none", shots=shots,
                                                   frame_ms=1000 / dfps)
                    dpart, dup = make_video_part(seg, dfps)
                    if dup:
                        uploads.append(dup)
                    print(f"[info] Pass 2: detail window {dw['start']}-{dw['end']} s at {dfps} fps ({dw.get('target')})...", file=sys.stderr)
                    detail, dinfo = generate_json_with_fallback(client, types, args, [dpart, types.Part(text=dprompt)], DETAIL_SCHEMA, "pass 2")
                    # sanity: times must fall inside the window (model was asked for absolute seconds)
                    lo, hi = dw["start"] - 0.2, dw["end"] + 0.2
                    bad = [e for e in detail.get("events", []) if isinstance(e.get("t_start_s"), (int, float))
                           and not lo <= e["t_start_s"] <= hi]
                    if bad and len(bad) > len(detail.get("events", [])) / 2:
                        # model answered relative to clip start after all -> shift
                        for e in detail.get("events", []):
                            for k in ("t_start_s", "t_end_s"):
                                if isinstance(e.get(k), (int, float)):
                                    e[k] = round(e[k] + dw["start"], 3)
                        for w in detail.get("caption_word_timing", []):
                            if isinstance(w.get("t_s"), (int, float)):
                                w["t_s"] = round(w["t_s"] + dw["start"], 3)
                        detail["time_fix"] = "model returned clip-relative times; shifted by window start"
                    for e in detail.get("events", []):
                        if e.get("basis") == "measured":
                            e["basis"] = "estimated"
                    detail["window"] = dw
                    detail["fps"] = dfps
                    result["detail_pass"] = detail
                    result["run"]["usage"]["pass2"] = dinfo
                except Exception as error:  # detail pass is best-effort
                    result["detail_error"] = str(error)[:500]
                    print(f"[warning] Detail pass failed: {error}", file=sys.stderr)
    except Exception as error:
        result["error"] = str(error)[:1000]
        print(f"[error] {error}", file=sys.stderr)
    finally:
        result["_uploads"] = uploads
    report = render_report(path, result)
    return result, report
