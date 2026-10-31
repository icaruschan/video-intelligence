---
name: video-intelligence
description: Watch and analyze a local video with Gemini's multimodal understanding. Produces grounded general, creative, or repurposing reports with timestamps, verbatim on-screen text, dialogue, and clear confidence boundaries. Use for UGC, TikToks, ads, tutorials, webinars, meetings, and screen recordings.
argument-hint: <path/to/video.mp4> [--mode general|creative|repurpose] [--prompt "..."] [--fps N] [--model ...]
disable-model-invocation: true
allowed-tools: Bash, Read
---

# Video Intelligence

Analyze a local video using Gemini's native multimodal video understanding.

## Prerequisites

- Python 3.10+
- `google-genai` installed:
  ```bash
  pip install -r requirements.txt
  ```
- `GEMINI_API_KEY` set in your environment:
  ```bash
  export GEMINI_API_KEY="your-api-key"
  ```

## How to Invoke

Run `analyze_video.py` passing the target video path:

```bash
python scripts/analyze_video.py <path/to/video.mp4> [--mode general|creative|repurpose]
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

## Supported Flags

- `--mode <general|creative|repurpose>`: Choose report framework (default: `general`).
- `--prompt "<text>"`: Add an extra instruction or specific question for Gemini.
- `--fps <N>`: Sample frames per second (e.g. `--fps 2` for fast cuts or small text).
- `--model <model-id>`: Gemini model to use (default: `gemini-3.5-flash`).
- `--timeout <seconds>`: Processing timeout for large video uploads via Files API (default: `300`).
- `--keep-upload`: Retain the uploaded file on Gemini Files API instead of auto-deleting.

## Accuracy Guardrails

- Never invent speaker names, creators, brands, dialogue, visible text, or voiceover.
- Preserve uncertainty: state "unclear" or "not discernible" when appropriate.
- Quote visible text and dialogue only when legible or clearly audible.
- In `creative` mode, strictly distinguish verified direct observations from strategic interpretations.
