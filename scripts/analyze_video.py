#!/usr/bin/env python3
"""Analyze a local video with Gemini (or OpenRouter as backup) and print a grounded Markdown report.

Modes: general / creative / repurpose (Markdown) and edit (measured + structured craft spec:
analysis.json + report.md; see scripts/edit_mode.py)."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import providers  # noqa: E402

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
    parser.add_argument("--mode", choices=[*MODE_INSTRUCTIONS, "edit"], default="general",
                        help="general | creative | repurpose | edit (craft/motion spec for recreating the look)")
    parser.add_argument("--prompt", help="Additional question or instruction for Gemini")
    parser.add_argument("--fps", type=float, help="Frames sampled per second; use 2+ for fast cuts")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--timeout", type=int, default=300, help="Files API processing timeout in seconds")
    parser.add_argument("--keep-upload", action="store_true", help="Do not delete an uploaded Gemini file")
    parser.add_argument("--provider", choices=["auto", "gemini", "openrouter"], default="auto",
                        help="auto (default): Gemini first, then OpenRouter on daily quota / repeated 503 / disconnects")
    parser.add_argument("--openrouter-model", default=providers.DEFAULT_OPENROUTER_MODEL,
                        help=f"OpenRouter model (default: {providers.DEFAULT_OPENROUTER_MODEL}); needs OPENROUTER_API_KEY")
    parser.add_argument("--no-audio-part", action="store_true",
                        help="OpenRouter: do not attach the soundtrack as a separate audio input")
    parser.add_argument("--out", help="Output directory (edit mode default: ./output/<video-stem>/; other modes: print only)")
    edit = parser.add_argument_group("edit mode")
    edit.add_argument("--scene-threshold", type=float, default=0.3, help="ffmpeg scene score for a hard cut (default 0.3)")
    edit.add_argument("--detail-window", type=float, default=6.0, help="Seconds for the dense-cut detail pass (default 6)")
    edit.add_argument("--detail-fps", type=float, default=10.0, help="fps for the detail pass (default 10, capped at source fps)")
    edit.add_argument("--fallback-model", default="",
                      help="Optional second Gemini model tried before OpenRouter (own daily quota). Off by default")
    edit.add_argument("--openrouter-slowmo", type=float, default=4.0,
                      help="OpenRouter detail pass: slow the timecoded clip this many times so the model samples it densely (default 4; 1 = off)")
    edit.add_argument("--no-detail", action="store_true", help="Skip the second, high-fps detail pass")
    edit.add_argument("--detail-target", choices=["motion", "cuts"], default="motion",
                      help="Detail window: most within-shot motion (default) or most cuts")
    edit.add_argument("--detail-start", type=float, help="Force the detail window start (seconds)")
    edit.add_argument("--no-timecode", action="store_true", help="Do not burn a timecode strip into the detail clip")
    edit.add_argument("--keep-detail-clip", action="store_true", help="Save the (timecoded) detail clip in the output dir")
    edit.add_argument("--allow-long", action="store_true", help="Allow clips over 90 s in edit mode (1 fps)")
    edit.add_argument("--measure-only", action="store_true",
                      help="Edit mode: only local measurements (no API call): measured.json, frames.json, overlays/, brief.json")
    edit.add_argument("--from-json", metavar="ANALYSIS_JSON",
                      help="Edit mode: re-measure the video and rebuild report.md/brief.json from a saved analysis.json (no API call)")
    edit.add_argument("--no-frames", action="store_true", help="Skip frame measurements (overlays, entry timing, stills)")
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


def write_outputs(out_dir: Path, report: str, data: dict | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(report)
    if data is not None:
        (out_dir / "analysis.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float))
        if data.get("brief"):
            (out_dir / "brief.json").write_text(json.dumps(data["brief"], indent=2, ensure_ascii=False, default=float))
    print(f"[info] Wrote {out_dir / 'report.md'}" + (f" and {out_dir / 'analysis.json'}" if data is not None else ""), file=sys.stderr)


def main() -> None:
    args = parse_args()
    if args.fps is not None and args.fps <= 0:
        fail("--fps must be greater than zero.")
    if args.timeout <= 0:
        fail("--timeout must be greater than zero.")
    path = validate_video(args.video_path)

    if args.mode == "edit" and args.from_json:
        import edit_measure
        import edit_mode
        result, report = edit_mode.rerender_from_json(args, path, Path(args.from_json), edit_measure)
        out_dir = Path(args.out) if args.out else Path("output") / (path.stem + "-rerender")
        write_outputs(out_dir, report, result)
        print(report)
        print("[info] Model requests this run: 0 (offline re-render)", file=sys.stderr)
        return

    if args.mode == "edit" and args.measure_only:
        import edit_measure
        import edit_mode
        data = edit_measure.measure(path, threshold=args.scene_threshold, detail_window=args.detail_window,
                                    detail_target=args.detail_target)
        out_dir = Path(args.out) if args.out else Path("output") / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "measured.json").write_text(json.dumps(data, indent=2))
        res = {"measured": data}
        edit_mode.postprocess(res, path, out_dir, frames=not args.no_frames)
        if res.get("frames"):
            (out_dir / "frames.json").write_text(json.dumps(res["frames"], indent=2, default=float))
        (out_dir / "brief.json").write_text(json.dumps(res["brief"], indent=2, ensure_ascii=False, default=float))
        print(json.dumps(data, indent=2))
        print(f"[info] Wrote measured.json, frames.json, brief.json (measured-only) in {out_dir}", file=sys.stderr)
        return

    client = None
    if args.provider in ("auto", "gemini"):
        if genai is None or types is None:
            if args.provider == "gemini":
                fail("Missing dependency: google-genai. Install it with: python3 -m pip install google-genai")
        elif os.getenv("GEMINI_API_KEY"):
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        elif args.provider == "gemini":
            fail("GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com/apikey and export it in your shell.")
    if args.provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        fail("OPENROUTER_API_KEY is not set.")

    stats = providers.RunStats()
    chain = providers.build_chain(args, stats, make_gemini_part=lambda p, fps: video_part(client, p, fps, args.timeout),
                                  gemini_client=client, gemini_types=types)
    if not chain.providers:
        fail("No provider available: set GEMINI_API_KEY and/or OPENROUTER_API_KEY.")
    print(f"[info] Provider order: {' -> '.join(chain.names)}", file=sys.stderr)

    try:
        if args.mode == "edit":
            import edit_measure
            import edit_mode
            result, report = edit_mode.run_edit_mode(args, chain, path, edit_measure)
            out_dir = Path(args.out) if args.out else Path("output") / path.stem
            write_outputs(out_dir, report, result)
            print(report)
            _log_requests(stats)
            if result.get("error"):
                raise SystemExit(2)
            return

        prompt = build_prompt(args.mode, args.prompt)
        print(f"[info] Analyzing in {args.mode} mode...", file=sys.stderr)
        text, info = chain.generate(args.mode, None, lambda p: {"video": path, "fps": args.fps, "prompt": prompt})
        text = text or "(model returned no text.)"
        print(text)
        if args.out:
            write_outputs(Path(args.out), text + f"\n\n---\nAnswered by {info['provider']}:{info['model']}\n", None)
        _log_requests(stats)
    finally:
        chain.cleanup(keep=args.keep_upload)


def _log_requests(stats) -> None:
    s = stats.summary()
    cost = f", OpenRouter cost ${s['openrouter_cost_usd']:.4f}" if s["openrouter_cost_usd"] else ""
    print(f"[info] Model requests this run: {s['total_requests']} {s['per_provider']}{cost}", file=sys.stderr)

if __name__ == "__main__":
    main()
