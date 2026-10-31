#!/usr/bin/env python3
"""Analyze a local video with Gemini and print a grounded Markdown report."""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys
import time
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


DEFAULT_MODEL = "gemini-3.5-flash"
INLINE_LIMIT_BYTES = 18 * 1024 * 1024
SUPPORTED_EXTENSIONS = {
    ".mp4", ".mov", ".webm", ".avi", ".mpeg", ".mpg", ".flv", ".wmv", ".3gpp", ".3gp"
}
MIME_OVERRIDES = {
    ".mov": "video/mov", ".avi": "video/avi", ".flv": "video/x-flv", ".mpg": "video/mpg",
    ".wmv": "video/wmv", ".3gpp": "video/3gpp", ".3gp": "video/3gpp",
}

MODE_INSTRUCTIONS = {
    "general": """## Report format
## Verified summary
Write 2 to 4 sentences covering only what is clearly supported by the video.

## Timeline
Use timestamped bullets. For every distinct beat, include visible activity and exact on-screen text only when legible.

## Audio and dialogue
Give a timestamped transcript only for speech you can discern. Otherwise accurately say whether audio is absent, silent, music-only, ambient, or unclear.

## Confirmed visual details
List people, products, interfaces, graphics, visible text, and brands only when identifiable.

## Notable moments
List 3 to 7 memorable timestamped events grounded in the video.""",
    "creative": """## Report format
## Verified summary
Write 2 to 4 sentences covering only what is clearly supported by the video.

## Timeline and beat map
Use timestamped bullets. Identify the visible or audible event in each beat and quote exact on-screen text only when legible.

## Audio and dialogue
Give a timestamped transcript only for speech you can discern. Otherwise accurately say whether audio is absent, silent, music-only, ambient, or unclear.

## Creative read
For Hook, Intended audience, Angle, Tension, Payoff, and CTA: state the evidence first. Label any strategic interpretation as **Inference**. Do not claim performance or intent as fact.

## Reusable patterns
Give 3 to 5 practical patterns. Tie each to a timestamp and say how someone could adapt the mechanism without copying the source's exact words or assets.

## Confirmed visual details
List people, products, interfaces, graphics, visible text, and brands only when identifiable.""",
    "repurpose": """## Report format
## Verified summary
Write 2 to 4 sentences covering only what is clearly supported by the video.

## Content inventory
Make a timestamped map of claims, demonstrations, stories, proof, visual changes, and quotations that are actually present.

## Audio and dialogue
Give a timestamped transcript only for speech you can discern. Otherwise accurately say whether audio is absent, silent, music-only, ambient, or unclear.

## Repurpose opportunities
Suggest 5 to 8 clips, posts, or format changes. Every suggestion must point to a source timestamp and distinguish source fact from your recommendation.

## Confirmed visual details
List people, products, interfaces, graphics, visible text, and brands only when identifiable.""",
}

BASE_PROMPT = """Analyze the attached video with accuracy as the primary objective.

Rules:
1. Report only what the video supports. Do not turn likely context into fact.
2. Never invent a creator, speaker name, narrator, brand, product, dialogue, visible text, or voiceover.
3. Use exact timestamps in MM:SS. If timing is approximate, write "about MM:SS".
4. Quote visible words or speech verbatim only when you can discern them. Mark partial text as partial and unreadable text as unreadable.
5. Treat silence as a valid result. Do not fill gaps with imagined narration.
6. Separate observations from interpretation. Where the report asks for an inference, label it exactly as **Inference**.
7. Be concrete and compact. Accuracy is more valuable than a complete-looking report.

"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", help="Path to the local video")
    parser.add_argument("--mode", choices=MODE_INSTRUCTIONS, default="general")
    parser.add_argument("--prompt", help="Additional question or instruction for Gemini")
    parser.add_argument("--fps", type=float, help="Frames sampled per second; use 2+ for fast cuts")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--timeout", type=int, default=300, help="Files API processing timeout in seconds")
    parser.add_argument("--keep-upload", action="store_true", help="Do not delete an uploaded Gemini file")
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def validate_video(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        fail(f"File not found: {path}")
    if not path.is_file():
        fail(f"Not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        fail(f"Unsupported file type '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    return path


def mime_type_for(path: Path) -> str:
    return MIME_OVERRIDES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "video/mp4"


def wait_for_active_file(client: genai.Client, file_name: str, timeout: int):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        uploaded = client.files.get(name=file_name)
        state = getattr(getattr(uploaded, "state", None), "name", str(getattr(uploaded, "state", "")))
        if state == "ACTIVE":
            return uploaded
        if state == "FAILED":
            fail("Gemini could not process the uploaded video.")
        print("[info] Gemini is processing the video...", file=sys.stderr)
        time.sleep(5)
    fail(f"Gemini did not finish processing within {timeout} seconds.")


def video_part(client: genai.Client, path: Path, fps: float | None, timeout: int):
    mime_type = mime_type_for(path)
    metadata = types.VideoMetadata(fps=fps) if fps else None
    if path.stat().st_size <= INLINE_LIMIT_BYTES:
        print(f"[info] Sending {path.stat().st_size / 1024 / 1024:.1f} MB inline...", file=sys.stderr)
        return types.Part(inline_data=types.Blob(data=path.read_bytes(), mime_type=mime_type), video_metadata=metadata), None

    print(f"[info] Uploading {path.stat().st_size / 1024 / 1024:.1f} MB through Gemini Files API...", file=sys.stderr)
    uploaded = client.files.upload(file=str(path), config={"mime_type": mime_type})
    active = wait_for_active_file(client, uploaded.name, timeout)
    return types.Part(file_data=types.FileData(file_uri=active.uri, mime_type=active.mime_type or mime_type), video_metadata=metadata), active.name


def build_prompt(mode: str, extra_prompt: str | None) -> str:
    prompt = BASE_PROMPT + MODE_INSTRUCTIONS[mode]
    if extra_prompt:
        prompt += f"\n\n## Additional user request\n{extra_prompt}\n"
    return prompt


def main() -> None:
    args = parse_args()
    if genai is None or types is None:
        fail("Missing dependency: google-genai. Install it with: python3 -m pip install google-genai")
    if args.fps is not None and args.fps <= 0:
        fail("--fps must be greater than zero.")
    if args.timeout <= 0:
        fail("--timeout must be greater than zero.")
    if not os.getenv("GEMINI_API_KEY"):
        fail("GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com/apikey and export it in your shell.")

    path = validate_video(args.video_path)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    uploaded_name = None
    try:
        part, uploaded_name = video_part(client, path, args.fps, args.timeout)
        print(f"[info] Analyzing with {args.model} in {args.mode} mode...", file=sys.stderr)
        response = client.models.generate_content(
            model=args.model,
            contents=types.Content(parts=[part, types.Part(text=build_prompt(args.mode, args.prompt))]),
        )
        print(response.text or "(Gemini returned no text.)")
    finally:
        if uploaded_name and not args.keep_upload:
            try:
                client.files.delete(name=uploaded_name)
                print("[info] Deleted temporary Gemini upload.", file=sys.stderr)
            except Exception as error:  # Cleanup must not hide the analysis result.
                print(f"[warning] Could not delete temporary upload: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
