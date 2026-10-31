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
pip install -r requirements.txt
```

### Step 2: Set your Gemini API Key
Set your Gemini API key in Grok Bot's environment:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```
*(Get a free key from [Google AI Studio](https://aistudio.google.com/apikey). You can also add this line to `/workspace/.bashrc` so it persists across sessions).*

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

# Asking a specific question about the video
python scripts/analyze_video.py path/to/video.mp4 --prompt "What pricing plan is shown on screen?"
```

---

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
