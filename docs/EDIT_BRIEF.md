# Edit brief (`brief.json`) — workflow and field mapping

`--mode edit` writes `report.md`, `analysis.json` and `brief.json` (+ `beats/` keyframes and `overlays/` crops).
`brief.json` is the build-ready part: every value is `{"value", "source", "note"}` where `source` is one of
`measured` (ffmpeg/numpy), `frame` (frame measurement in `edit_frames.py`), `detail-pass` (pass 2, wins inside its
window), `pass1` (model, unverified outside the detail window), `inference`, `unclear`.

## Workflow

1. **Reference** — pick a short-form reference (≤ 90 s). Reference footage, logos and people are *study-only*.
2. **Edit-mode analysis** — `python scripts/analyze_video.py ref.mp4 --mode edit --out output/<name>`
   (no quota? `--measure-only` for the local numbers; re-render an old run with `--from-json old/analysis.json`, 0 model calls).
3. **Brief** — read `report.md` §13a–13j / `brief.json`: style recipe (portable), beat map (% of runtime), contradictions,
   measured timing, overlays, VO pace, fonts, values to copy, assets, do-not list, QA ranges.
4. **Storyboard check with Fortune** — beat map + `beats/` stills + new script; agree what is copied
   (`[core-style]`) and what is dropped (`[ref-specific]`) before building.
5. **Build** in HyperFrames / cooked — feed `brief.cooked` to style-edit, GSAP strings to the motion composition.
6. **Check against spec** — run edit mode on the render and tick `qa_checklist` (ranges/ratios, not exact copies).

## brief.json → cooked `style.json` (creator-teardown contract)

| brief.json | style.json | Notes |
|---|---|---|
| `cooked.pace.wpm` | `pace.wpm` | detail-pass caption word times (proxy until forced alignment) |
| `cooked.pace.max_pause_s` | `pace.max_pause_s` | max onset gap in the detail window |
| `cooked.pace.cuts_per_10s` / `median_shot_s` | same | measured |
| `cooked.pace.cut_kinds` | `pace.cut_kinds` | % per kind (`hard` / `jump` / `zoom-punch` / `whip` / `match` / `mask` / `dissolve`) from pass-1 types |
| `cooked.captions.{present,y_pct,animation,entrance,ease_s,source}` | `captions.*` | `animation` is the contract enum `pop` / `none` / `slide` / `word_highlight`; free text in `entrance` / `stagger_note` |
| `cooked.graphics.entrances[]` | `graphics.entrances[]` | grouped, most used first: `kind` (`cut`/`slide`/`scale`/`fade`), `ease` (GSAP), `duration_s`, `overshoot` (0.06 = 6 %), `fade`, `from`, `distance_pct`, `share_pct`, `n` — numbers frame-measured |
| `cooked.graphics.layout.overlays[]` | `graphics.layout` (extra key) | persistent overlay boxes in % of frame |
| `cooked.sound.*` | `sound.*` | `null` = unclear offline (earshot paused); house style says SFX throughout |
| `cooked.hook.{first_cut_s,first_graphic_s,first_caption_s}` | `hook.*` | measured cut / frame entry / pass-1 caption |

## brief.json → HyperFrames composition

| brief.json | HyperFrames / GSAP |
|---|---|
| `beat_map[].start_s` / `end_s` (and `start_pct`) | clip `data-start` / `data-duration` per beat (scale % × new runtime) |
| `style_recipe.shot_length_s` p25/p50/p75, `cut_rate_band` | shot durations inside each beat |
| `frame_measurements.entries[].ease.gsap` + `entry_ms` | `tl.from(el, {duration: ms/1000, ease: "<gsap>"})` |
| `style_recipe.motion.staggered_builds[].gsap_sketch` | ready-made `gsap.from(...)` lines (opacity + y rise, part delays) |
| `frame_measurements.overlays[].px_1080x1920` / `pct` | absolute `left/top/width/height` px at 1080×1920 |
| `fonts[].css` / `fonts[].installed_files` | `@font-face` / font-family; licence in `fonts[].licence` |
| `copy_values[]` (`portable: true`) | composition constants; `portable: false` = this reference only |
| `assets[]` (`study_only`) | shot list for new footage; never reuse reference material |
| `do_not[]` (`[core-style]` / `[ref-specific]`) | linter-style checklist for the build |

## How the frame measurements work (all local, deterministic)

- **Entries** (180 px wide): a localized region changes then settles inside one shot → frames × 40 ms (25 fps).
  ≤ 2 frames = hard pop (`none`). Staggered builds are split by per-pixel first-change frame. Overshoot only from a
  scale/move whose area grows past its rest size → `back.out(s)` (s inverted from the overshoot). Vertical travel
  from the ink centre → px at 1920 + least-squares fit to GSAP curves (`power1-4.out`, `expo.out`, `sine.*`, `linear`).
- **Persistent overlays** (360 px): stable edges shared by the mid frames of consecutive shots, ±1 px shift-tolerant;
  spans only cross verified cuts. Middle-zone spans < 1.5 s are captions held over a cut, not overlays.
- **In-place builds**: a fixed slot whose content changes while the rest of the frame is still (timer pills,
  counters, word builds).
- **VO**: wpm and gaps from detail-pass caption appearance times (no ASR); cut-vs-word offsets ±80 ms.
- `scripts/selftest_edit.py` checks the helpers offline.
