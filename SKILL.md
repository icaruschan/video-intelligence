---
name: video-intelligence
description: Watch and analyze a local video with Gemini's multimodal understanding. Produces grounded general, creative, repurposing, or edit/craft (motion + measured cut timing) reports with timestamps, verbatim on-screen text, dialogue, and clear confidence boundaries. Use for UGC, TikToks, ads, tutorials, webinars, meetings, and screen recordings.
argument-hint: <path/to/video.mp4> [--mode general|creative|repurpose|edit] [--out DIR] [--prompt "..."] [--fps N] [--model ...]
disable-model-invocation: true
allowed-tools: Bash, Read
---

# Video Intelligence

Analyze a local video using Gemini's native multimodal video understanding.

## Prerequisites

- Python 3.10+
- `ffmpeg` / `ffprobe` on PATH (edit mode measures locally) and `numpy` (edit mode onsets + palette)
- `google-genai` installed:
  ```bash
  pip install -r requirements.txt
  ```
- `GEMINI_API_KEY` set in your environment:
  ```bash
  export GEMINI_API_KEY="your-api-key"
  ```
- Optional backup: `OPENROUTER_API_KEY` (OpenRouter, default model `qwen/qwen3.8-omni-flash`). With `--provider auto`
  (default) Gemini is tried first and OpenRouter takes over on a daily-quota 429, repeated 503s or disconnects.

## How to Invoke

Run `analyze_video.py` passing the target video path:

```bash
python scripts/analyze_video.py <path/to/video.mp4> [--mode general|creative|repurpose|edit] [--out DIR]

# Edit / craft spec of a short reference (writes output/<stem>/report.md + analysis.json)
python scripts/analyze_video.py ref.mp4 --mode edit
```

### Windows (PowerShell):
```powershell
python scripts\analyze_video.py <path\to\video.mp4> --mode creative
```

## Modes

1. **`general`** (Default):
   Evidence-based video breakdown: verified summary, timeline beats, audio/speech transcript, confirmed visual details (people, brands, products, UI), and notable moments.
2. **`creative`**:
   Designed for creators, UGC, and performance ads: general report plus Hook, Intended Audience, Angle, Tension, Payoff, CTA, and Reusable Transferable Patterns. Strategy claims are explicitly labelled as **Inference**.
3. **`repurpose`**:
   Content inventory map of claims, demonstrations, and quotes, followed by 5–8 actionable clip, post, or format repurposing recommendations grounded in specific timestamps.
4. **`edit`**:
   Reverse-engineers the CRAFT of a short-form reference so an editor can recreate its look and feel from numbers.
   - Measures locally first (ffprobe, ffmpeg scene detection, EBU R128 loudness, silences, spectral-flux audio onsets,
     dominant colours) and feeds the results to Gemini as ground truth.
   - Pass 1: whole clip at 2–5 fps (by length) → structured JSON against a schema: overview, structure & pacing,
     camera & framing, B-roll, typography & layout, captions, motion language, elements & assets, transitions,
     colour, sound design, retention devices, edit brief (values to copy / do-not / checklist), plus a
     reconciliation row for every measured cut.
   - Pass 2: the window with the most within-shot motion at ~10 fps, with an absolute timecode strip burned under
     the frame, for frame-level motion timing (durations, easing, overshoot, stagger, caption word timing).
   - Every number is labelled `measured` / `measured timing` / `estimated` / `inference` / `unclear`; model
     claims of "measured" that are not in the measured data are downgraded automatically.
   - Writes `report.md` + `analysis.json` to `--out` (default `./output/<video-stem>/`). Short-form only by default
     (≤ 90 s; `--allow-long` to override).

## Supported Flags

- `--mode <general|creative|repurpose|edit>`: Choose report framework (default: `general`).
- `--out <dir>`: Write `report.md` there (edit mode also writes `analysis.json`; default `./output/<stem>/`).
- `--prompt "<text>"`: Add an extra instruction or specific question for Gemini.
- `--fps <N>`: Sample frames per second (e.g. `--fps 2` for fast cuts or small text).
- `--model <model-id>`: Gemini model to use (default: `gemini-3.5-flash`).
- `--timeout <seconds>`: Processing timeout for large video uploads via Files API (default: `300`).
- `--keep-upload`: Retain the uploaded file on Gemini Files API instead of auto-deleting.

Edit-mode flags: `--scene-threshold 0.3` (hard-cut score), `--detail-window 6`, `--detail-fps 10`,
`--detail-target motion|cuts`, `--detail-start S`, `--no-detail`, `--no-timecode`, `--keep-detail-clip`,
`--fallback-model <gemini-model>` (optional second Gemini model, off by default), `--openrouter-slowmo 4`,
`--allow-long`, `--measure-only` (no API call; writes `measured.json`).

Provider flags (all modes): `--provider auto|gemini|openrouter` (default `auto`), `--openrouter-model <id>`,
`--no-audio-part`. Max 2 model requests per pass per provider, then fall over; the request count and OpenRouter
cost are logged at the end of the run and stored in `analysis.json → run.requests`.

Cost per edit run on a 42 s clip (2 requests):
- gemini-3.5-flash: ≈ 22k tokens pass 1 + ≈ 7k pass 2 (free tier: 20 requests/day per model).
- qwen/qwen3.8-omni-flash via OpenRouter: ≈ $0.028 (35k in / 31k out incl. 14k reasoning for pass 1, 16k / 12k for
  pass 2), ~9 min wall time.

## Accuracy Guardrails

- Never invent speaker names, creators, brands, dialogue, visible text, or voiceover.
- Preserve uncertainty: state "unclear" or "not discernible" when appropriate.
- Quote visible text and dialogue only when legible or clearly audible.
- In `creative` mode, strictly distinguish verified direct observations from strategic interpretations.
- In `edit` mode, measured data beats model perception: cut times, shot lengths, loudness and onsets come from
  ffmpeg/numpy; the model explains each measured cut and lists cuts it thinks measurement missed.
- Sound design timing is rough (onsets cannot tell SFX from speech/music). TODO: merge earshot's sound-event
  timeline at the `earshot` hook in `scripts/edit_measure.py` once enabled.
