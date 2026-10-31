# Video Intelligence 🎥🧠

An AI agent skill and CLI for grounded, multimodal video understanding powered by Gemini.

Extract verified timelines, verbatim on-screen text, audio transcripts, creative deconstructions (hook, angle, tension, payoff, CTA), and content repurposing plans from any video.

---

## 🚀 Setup for Grok Bot

Grok Bot environments run in a persistent cloud container (`/workspace`). To give your Grok Bot video intelligence capabilities:

### Step 1: Clone into Grok Bot's Workspace
Tell Grok Bot or run in its terminal:
```bash
cd /workspace
git clone https://github.com/icaruschan/video-intelligence.git
cd video-intelligence
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # ffmpeg must be installed for edit mode
```

### Step 2: Set your Gemini API Key
Set your Gemini API key in Grok Bot's environment:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```
*(Get a free key from [Google AI Studio](https://aistudio.google.com/apikey). You can also add this line to `/workspace/.bashrc` so it persists across sessions).*

Optional backup provider (recommended: the Gemini free tier allows only ~20 requests/day per model):
```bash
export OPENROUTER_API_KEY="your-openrouter-key"   # used for qwen/qwen3.8-omni-flash by default
```

### Step 3: Tell Grok Bot How to Use It
Paste this prompt into Grok Bot:
> *"You now have the **Video Intelligence** tool installed in `/workspace/video-intelligence`. Read `/workspace/video-intelligence/SKILL.md`. Whenever I give you a video or ask you to watch, analyze, deconstruct, or repurpose a video, run:*
> ```bash
> python3 /workspace/video-intelligence/scripts/analyze_video.py <video-path> --mode <general|creative|repurpose>
> ```
> *and present the grounded Markdown report to me."*

---

## 💻 Setup for Codex & Claude Code

To install directly into Codex or Claude Code:

```bash
# For Codex
bash install_skill.sh codex

# For Claude Code
bash install_skill.sh claude
```

---

## 🛠️ Modes & Features

| Mode | Purpose | Key Output Sections |
| :--- | :--- | :--- |
| **`general`** (Default) | Objective video breakdown | Summary, Timeline beats, Audio/Dialogue transcript, Confirmed visual elements, Notable moments. |
| **`creative`** | UGC, ads, and creator teardowns | Beat map, Audio transcript, Hook, Audience, Angle, Tension, Payoff, CTA, Reusable patterns. |
| **`repurpose`** | Content recycling & formatting | Content inventory, Audio transcript, 5–8 Timestamped clip & format opportunities. |
| **`edit`** | Recreate a reference's look & feel | Measured cuts/shots/loudness/onsets/palette + 13-section craft spec (pacing, framing, B-roll, type, captions, motion, elements, transitions, colour, sound, retention, edit brief) as `report.md` + `analysis.json`. |

---

## 📖 CLI Usage

```bash
# Basic general analysis
python scripts/analyze_video.py path/to/video.mp4

# Creative breakdown for a UGC ad
python scripts/analyze_video.py path/to/ad.mp4 --mode creative

# Repurpose webinar or demo into short-form clips
python scripts/analyze_video.py path/to/webinar.mp4 --mode repurpose

# High-framerate sampling for fast cuts or small on-screen text
python scripts/analyze_video.py path/to/video.mp4 --fps 2

# Edit / craft analysis of a short-form reference (ffmpeg + numpy required)
python scripts/analyze_video.py path/to/reference.mp4 --mode edit            # -> output/reference/
python scripts/analyze_video.py path/to/reference.mp4 --mode edit --measure-only   # local numbers only, no API

# Force the OpenRouter backup (e.g. Gemini daily quota used up)
python3 scripts/analyze_video.py ref.mp4 --mode edit --provider openrouter

# Save any mode's report to a folder
python scripts/analyze_video.py path/to/video.mp4 --mode creative --out output/my-video

# Asking a specific question about the video
python scripts/analyze_video.py path/to/video.mp4 --prompt "What pricing plan is shown on screen?"
```

---

## 🎬 How `edit` mode stays accurate

1. **Measure first** (`scripts/edit_measure.py`): ffprobe metadata, ffmpeg scene-score cuts (merged within 120 ms; soft events 0.15–0.3), shot lengths and cuts/sec, per-second visual-change curve, EBU R128 loudness, silences, spectral-flux onsets, cut↔onset sync with a chance baseline, dominant colours.
2. **Pass 1** (`scripts/edit_mode.py`): whole clip at 5/3/2 fps (≤30/≤60/≤90 s) with the measurements in the prompt, structured JSON via `response_json_schema` (falls back to JSON mode + tolerant parsing). Required spec slots per section so nothing comes back empty.
3. **Pass 2**: the window with the most within-shot motion, re-encoded with an absolute timecode strip under the frame, at ~10 fps, for frame-level motion timing.
4. **Audit**: model "measured" labels not backed by the measured data are downgraded; model event times are reconciled against measured cuts.
5. **Providers** (`scripts/providers.py`, `--provider auto|gemini|openrouter`, default `auto`): Gemini first, then OpenRouter
   (`--openrouter-model`, default `qwen/qwen3.8-omni-flash`) on a daily-quota 429, repeated 503s or disconnects.
   Hard cap: at most 2 model requests per pass per provider; a provider that hit its daily quota is skipped for the
   rest of the run, and one that gave up on 503s gets a single probe on the next pass. Every request is counted and
   logged (`[req #n]`) and summarised in `analysis.json → run.requests`, including the OpenRouter cost in USD.
6. **OpenRouter specifics:** the video goes as a base64 `video_url` data URL (re-encoded only if > 12 MB or long edge
   > 1280 px), the soundtrack is also attached as an `input_audio` mp3 (`--no-audio-part` to skip), and structured
   output uses `response_format: json_schema` (falls back to JSON-in-prompt + tolerant parsing). OpenRouter has no fps
   control, so pass 2 sends the timecoded detail clip slowed down `--openrouter-slowmo 4`× (strip still shows real
   source time) so the model's own ~2 fps sampling sees ~8 fps of source time.

Sound timing is the weak spot (no agent can truly listen yet); `edit_measure.py` has an `earshot` hook for a future sound-event timeline.

## 🛡️ Grounding & Anti-Hallucination Guardrails

- **Zero Invention**: Never hallucinate speaker names, brand identities, or dialogues not supported by audio/visuals.
- **Explicit Uncertainty**: Labels moments with low clarity as "unclear" or "not discernible".
- **Observation vs. Strategy**: In `creative` mode, all strategic judgments are strictly labelled as **Inference**.

---

## 🔗 Companion Tool

For downloading videos directly from URLs (YouTube, TikTok, X) and extracting frame scrubbers via `yt-dlp` and `ffmpeg`, see [`bradautomates/claude-video`](https://github.com/bradautomates/claude-video) (`/watch`).

---

## License

MIT
