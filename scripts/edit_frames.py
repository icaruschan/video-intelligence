"""Deterministic frame-level measurements for edit mode (numpy + ffmpeg only, no model calls).

- read_frames: whole clip at source fps, downscaled (default 180x320 for 9:16), RGB uint8.
- persistent_overlays: regions whose pixels stay constant ACROSS cuts (badges, timer pills, lower-thirds,
  watermarks, captions that ride over a cut). bbox, exact frame span, crop still.
- element_entries: graphics/text that animate in inside a shot and then settle: first change -> settle frame
  count (ms), scale/position overshoot from the per-frame bbox, best-fit GSAP ease.
- save_stills: full-resolution keyframes (beats/, overlays/).
All outputs carry basis "measured" (frame-measured); limits are written into each record.
"""

from __future__ import annotations

import math
import subprocess
from collections import deque
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------- IO

def read_frames(path: Path, width: int, height: int, scale_w: int = 180) -> tuple[np.ndarray, int, int]:
    sw = scale_w
    sh = int(round(height * sw / width / 2) * 2)
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={sw}:{sh}:flags=area", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8)
    n = frames.size // (sw * sh * 3)
    return frames[: n * sw * sh * 3].reshape(n, sh, sw, 3), sw, sh


def save_still(path: Path, t: float, out: Path, crop: tuple[int, int, int, int] | None = None,
               width: int | None = None) -> Path | None:
    """crop = (x, y, w, h) in SOURCE pixels. Writes jpg/png via ffmpeg."""
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = []
    if crop:
        x, y, w, h = crop
        vf.append(f"crop={w}:{h}:{x}:{y}")
    if width:
        vf.append(f"scale={width}:-2")
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(t, 0):.3f}", "-i", str(path), "-frames:v", "1",
           *(["-vf", ",".join(vf)] if vf else []), "-q:v", "3", str(out)]
    return out if subprocess.run(cmd, capture_output=True).returncode == 0 and out.exists() else None


# ----------------------------------------------------------------------------- helpers

def _gray(f: np.ndarray) -> np.ndarray:
    return (f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114).astype(np.float32)


def _edges(g: np.ndarray, thr: float = 40.0) -> np.ndarray:
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    return np.hypot(gx, gy) > thr


def _dilate(m: np.ndarray, r: int = 1) -> np.ndarray:
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def _components(mask: np.ndarray, min_px: int) -> list[dict]:
    """4-connected components (pure python BFS; masks are small)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    comps = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        pts = []
        while q:
            y, x = q.popleft()
            pts.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        if len(pts) >= min_px:
            py, px = zip(*pts)
            comps.append({"x0": min(px), "y0": min(py), "x1": max(px) + 1, "y1": max(py) + 1, "px": len(pts)})
    return comps


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def _frame_index(t: float, fps: float, n: int) -> int:
    return int(min(max(round(t * fps), 0), n - 1))


def _box_out(box, sw, sh, src_w, src_h) -> dict:
    x0, y0, x1, y1 = (int(v) for v in box)
    fx, fy = x0 / sw, y0 / sh
    fw, fh = (x1 - x0) / sw, (y1 - y0) / sh
    return {
        "pct": {"x": round(fx * 100, 1), "y": round(fy * 100, 1), "w": round(fw * 100, 1), "h": round(fh * 100, 1)},
        "px_source": {"x": round(fx * src_w), "y": round(fy * src_h), "w": round(fw * src_w), "h": round(fh * src_h)},
        "px_1080x1920": {"x": round(fx * 1080), "y": round(fy * 1920), "w": round(fw * 1080), "h": round(fh * 1920)},
    }


def _zone(box, sh) -> str:
    cy = (box[1] + box[3]) / 2 / sh
    if cy < 0.2:
        return "top (badge / timer / label zone)"
    if cy > 0.68:
        return "bottom (lower-third / CTA zone)"
    return "middle (caption / graphic zone)"


# ----------------------------------------------------------------------------- persistent overlays

def _group_lines(boxes: list[tuple], gap: int) -> list[tuple]:
    """Union boxes that overlap horizontally and sit within `gap` px vertically (stacked text lines, pill + label)."""
    boxes = [tuple(int(v) for v in b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                xo = min(a[2], b[2]) - max(a[0], b[0])
                vg = max(a[1], b[1]) - min(a[3], b[3])
                if xo > 0 and vg < gap:
                    boxes[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return boxes


def persistent_overlays(frames: np.ndarray, fps: float, shots: list[dict], src_w: int, src_h: int,
                        diff_thr: float = 14.0) -> list[dict]:
    """Edge pixels that are identical in the middle frames of two consecutive shots survive a cut -> overlay.
    Each candidate is then extended frame by frame (template match on its edge pixels) to an exact span."""
    n, sh, sw, _ = frames.shape
    grays = {}

    def g(i):
        if i not in grays:
            grays[i] = _gray(frames[i])
        return grays[i]

    cands = []
    for a, b in zip(shots, shots[1:]):
        ia = _frame_index((a["start"] + a["end"]) / 2, fps, n)
        ib = _frame_index((b["start"] + b["end"]) / 2, fps, n)
        if ia == ib:
            continue
        ga, gb = g(ia), g(ib)
        stable = _edges(ga) & _edges(gb) & (np.abs(ga - gb) < diff_thr)
        # whole-frame stability (two identical cards) is not an overlay
        if stable.mean() > 0.08:
            continue
        r = max(2, sw // 72)   # fuse letters of one text line (~5 px at 360 w)
        pair_boxes = [(c["x0"], c["y0"], c["x1"], c["y1"]) for c in _components(_dilate(stable, r), 60)]
        for box in pair_boxes:
            area = (box[2] - box[0]) * (box[3] - box[1]) / (sw * sh)
            if area > 0.3 or (box[2] - box[0]) < 6 or (box[3] - box[1]) < 3:
                continue
            core = stable[box[1]:box[3], box[0]:box[2]]
            if core.sum() < 25:
                continue
            full = np.zeros_like(stable)
            ink = (((ga > 200) & (gb > 200)) | ((ga < 45) & (gb < 45))) & (np.abs(ga - gb) < diff_thr)
            full[box[1]:box[3], box[0]:box[2]] = core | (ink[box[1]:box[3], box[0]:box[2]] &
                                                         _dilate(stable, 1)[box[1]:box[3], box[0]:box[2]])
            cands.append({"box": box, "ref": ib, "cut_t": b["start"], "core": full, "pair_boxes": pair_boxes})

    # merge candidates that are the same region over consecutive cuts
    merged: list[dict] = []
    for c in cands:
        for m in merged:
            if _iou(m["box"], c["box"]) > 0.3 and abs(m["ref"] - c["ref"]) < fps * 12:
                m["box"] = (min(m["box"][0], c["box"][0]), min(m["box"][1], c["box"][1]),
                            max(m["box"][2], c["box"][2]), max(m["box"][3], c["box"][3]))
                m["cuts"].append(c["cut_t"])
                m["core"] |= c["core"] if m["ref"] == c["ref"] else False
                break
        else:
            merged.append({"box": c["box"], "ref": c["ref"], "cuts": [c["cut_t"]], "core": c["core"].copy(),
                           "pair_boxes": c["pair_boxes"]})

    out = []
    for m in merged:
        x0, y0, x1, y1 = m["box"]
        ref = g(m["ref"])[y0:y1, x0:x1]
        emask = m["core"][y0:y1, x0:x1]       # only pixels that survived the cut = the overlay itself
        eref = _edges(g(m["ref"]))[y0:y1, x0:x1] & emask
        if emask.sum() < 15:
            continue

        H, Wd = g(0).shape

        def match(i):
            # tolerate +-1 px drift (scaling/subpixel moves of the overlay or re-encode)
            best = None
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ya, yb, xa, xb = y0 + dy, y1 + dy, x0 + dx, x1 + dx
                    if ya < 0 or xa < 0 or yb > H or xb > Wd:
                        continue
                    d = np.abs(g(i)[ya:yb, xa:xb] - ref)
                    best = d if best is None else np.minimum(best, d)
            v = best[emask]
            if float(np.median(v)) >= diff_thr:
                return False
            # the overlay's own edges must be there too (white text vs a blank white card is not a match)
            cur_e = _dilate(_edges(g(i)), 1)[y0:y1, x0:x1]
            return eref.sum() == 0 or (cur_e & eref).sum() / eref.sum() >= 0.6

        verified = {round(c, 3) for c in m["cuts"]}
        cut_frames = {_frame_index(s["start"], fps, n): round(s["start"], 3) for s in shots[1:]}

        def ok_step(i_from, i_to):
            # only cross a cut whose stability was verified on the shot-pair mid frames
            c = cut_frames.get(max(i_from, i_to))
            return c is None or c in verified

        def extend(i, step):
            miss = 0
            last = i
            while 0 <= i + step < n and ok_step(i, i + step):
                i += step
                if match(i):
                    last, miss = i, 0
                else:
                    miss += 1
                    if miss > 1:      # tolerate one dropped frame (compression / flash)
                        break
            return last

        i0 = extend(m["ref"], -1)
        i1 = extend(m["ref"], 1)
        t0, t1 = i0 / fps, (i1 + 1) / fps
        crossed = [s["start"] for s in shots[1:] if t0 < s["start"] < t1]
        if not crossed or t1 - t0 < 0.3:
            continue
        # report box: the matched component plus stacked lines from the same verified cut pair
        rep = m["box"]
        for gb_ in _group_lines(m["pair_boxes"], gap=int(sh * 0.035)):
            if gb_[0] <= rep[0] and gb_[1] <= rep[1] and gb_[2] >= rep[2] and gb_[3] >= rep[3]:
                rep = gb_
                break
        if (rep[2] - rep[0]) * (rep[3] - rep[1]) > 6 * (x1 - x0) * (y1 - y0):
            rep = m["box"]      # grouping swallowed background; keep the matched component
        m["box"] = rep
        zone = _zone(m["box"], sh)
        kind = ("caption/text held across a cut (likely not a permanent overlay)"
                if zone.startswith("middle") and t1 - t0 < 1.5 else
                "persistent overlay (badge / lower-third / watermark / pill)")
        rec = {"kind": kind, "t_start_s": round(t0, 3), "t_end_s": round(t1, 3), "duration_s": round(t1 - t0, 3),
               "frames": i1 - i0 + 1, "cuts_crossed": [round(c, 3) for c in crossed],
               "zone": _zone(m["box"], sh), **_box_out(m["box"], sw, sh, src_w, src_h),
               "basis": "measured",
               "method": "edge pixels unchanged across >=1 cut, span extended frame-by-frame (median |diff| < "
                         f"{diff_thr:g} on edge pixels); +-1 px drift and 1 dropped frame tolerated, own edges >= 60% present; spans only cross cuts verified on shot-pair mid frames"}
        out.append(rec)
    # de-duplicate overlapping records (same region, overlapping spans)
    out.sort(key=lambda r: (r["t_start_s"], -r["duration_s"]))
    final: list[dict] = []
    for r in out:
        b = (r["pct"]["x"], r["pct"]["y"], r["pct"]["x"] + r["pct"]["w"], r["pct"]["y"] + r["pct"]["h"])
        dup = False
        for f in final:
            fb = (f["pct"]["x"], f["pct"]["y"], f["pct"]["x"] + f["pct"]["w"], f["pct"]["y"] + f["pct"]["h"])
            if _iou(b, fb) > 0.4 and r["t_start_s"] < f["t_end_s"] and f["t_start_s"] < r["t_end_s"]:
                f["t_start_s"] = min(f["t_start_s"], r["t_start_s"])
                f["t_end_s"] = max(f["t_end_s"], r["t_end_s"])
                f["duration_s"] = round(f["t_end_s"] - f["t_start_s"], 3)
                f["cuts_crossed"] = sorted(set(f["cuts_crossed"]) | set(r["cuts_crossed"]))
                dup = True
                break
        if not dup:
            final.append(r)
    for i, r in enumerate(final, 1):
        r["id"] = f"ov{i:02d}"
    return final


def dynamic_badges(frames: np.ndarray, fps: float, shots: list[dict], src_w: int, src_h: int,
                   diff_thr: float = 14.0, win_s: float = 0.6) -> list[dict]:
    """In-place builds INSIDE one shot (rolling counters, timer pills, word-by-word text builds, number reveals): inside an otherwise STILL frame (<= 6% of pixels change), a small
    box (0.1-5% of frame) keeps a fixed outline/label (edges stable in >= 80% of frame pairs) while its inside
    changes in >= 30% of frame pairs. Sliding 0.6 s windows inside each shot; neighbours merged."""
    n, sh, sw, _ = frames.shape
    W = max(6, int(round(win_s * fps)))
    hits = []
    for s in shots:
        a, b = _frame_index(s["start"], fps, n) + 1, _frame_index(s["end"], fps, n) - 1
        for w0 in range(a, max(a, b - W) + 1, max(2, W // 3)):
            idx = list(range(w0, min(w0 + W, b + 1)))
            if len(idx) < 6:
                continue
            gs = [_gray(frames[i]) for i in idx]
            S = np.zeros((sh, sw), np.int16)
            C = np.zeros((sh, sw), np.int16)
            pc = []
            for k in range(len(gs) - 1):
                d = np.abs(gs[k + 1] - gs[k])
                S += (_edges(gs[k]) & _edges(gs[k + 1]) & (d < diff_thr))
                ch = d > diff_thr * 2
                C += ch
                pc.append(ch)
            pairs = len(gs) - 1
            if (C > 0).mean() > 0.06:
                continue
            stable = S >= pairs * 0.8
            for c in _components(_dilate(C > 0, max(2, sw // 72)), min_px=20):
                box = (c["x0"], c["y0"], c["x1"], c["y1"])
                area = (box[2] - box[0]) * (box[3] - box[1]) / (sw * sh)
                if not 0.001 <= area <= 0.05:
                    continue
                x0, y0, x1, y1 = box
                pad = max(2, sw // 90)
                outline = stable[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad].sum()
                active_pairs = sum(1 for ch in pc if ch[y0:y1, x0:x1].mean() > 0.02)
                if outline < 20 or active_pairs < pairs * 0.3:
                    continue
                hits.append({"box": box, "t0": idx[0] / fps, "t1": (idx[-1] + 1) / fps, "shot": s["start"]})
    merged: list[dict] = []
    for h in hits:
        for m in merged:
            if m["shot"] == h["shot"] and _iou(m["box"], h["box"]) > 0.2 and h["t0"] <= m["t1"] + 0.1:
                m["t1"] = max(m["t1"], h["t1"])
                m["box"] = (min(m["box"][0], h["box"][0]), min(m["box"][1], h["box"][1]),
                            max(m["box"][2], h["box"][2]), max(m["box"][3], h["box"][3]))
                break
        else:
            merged.append(dict(h))
    out = []
    for i, m in enumerate(merged, 1):
        out.append({"id": f"db{i:02d}", "t_start_s": round(m["t0"], 3), "t_end_s": round(m["t1"], 3),
                    "kind": "in-place build: fixed slot whose content changes while the frame is still "
                            "(timer/counter pill, word-by-word text build, number reveal)",
                    "zone": _zone(m["box"], sh), **_box_out(m["box"], sw, sh, src_w, src_h), "basis": "measured",
                    "shot_start_s": m["shot"],
                    "method": "sliding 0.6 s windows: frame <= 6% changing, box outline stable >= 80% of pairs, "
                              "inside changing >= 30% of pairs"})
    return out


# ----------------------------------------------------------------------------- element entries

def _gsap_curves():
    def back_out(s):
        return lambda p: 1 + (s + 1) * (p - 1) ** 3 + s * (p - 1) ** 2
    return {
        "none": None,
        "linear": lambda p: p,
        "sine.out": lambda p: math.sin(p * math.pi / 2),
        "power1.out": lambda p: 1 - (1 - p) ** 2,
        "power2.out": lambda p: 1 - (1 - p) ** 3,
        "power3.out": lambda p: 1 - (1 - p) ** 4,
        "power4.out": lambda p: 1 - (1 - p) ** 5,
        "expo.out": lambda p: 1.0 if p >= 1 else 1 - 2 ** (-10 * p),
        "power2.inOut": lambda p: 2 * p * p if p < 0.5 else 1 - (-2 * p + 2) ** 2 / 2,
        "sine.inOut": lambda p: -(math.cos(math.pi * p) - 1) / 2,
        "_back": back_out,
    }


def _back_s_for_overshoot(ov: float) -> float:
    """GSAP back.out(s) peak overshoot -> s (numeric inverse)."""
    best, bs = 9.9, 1.7
    for s10 in range(2, 60):
        s = s10 / 10
        f = _gsap_curves()["_back"](s)
        peak = max(f(i / 200) for i in range(201)) - 1
        if abs(peak - ov) < best:
            best, bs = abs(peak - ov), s
    return round(bs, 1)


def fit_ease(progress: list[float], overshoot: float) -> dict:
    """progress: 0..1 samples, one per frame AFTER the first change (frame 0 = untouched is excluded), last = settled."""
    k = len(progress)
    if k <= 2:
        return {"gsap": "none", "note": f"appears within {k} frame(s): hard pop / cut-on", "rmse": None}
    if overshoot >= 0.03:
        s = _back_s_for_overshoot(overshoot)
        return {"gsap": f"back.out({s})", "note": f"{overshoot * 100:.0f}% size overshoot measured", "rmse": None}
    curves = {k_: v for k_, v in _gsap_curves().items() if v and not k_.startswith("_")}
    xs = [(i + 1) / k for i in range(k)]
    scored = []
    for name, f in curves.items():
        err = math.sqrt(sum((f(x) - p) ** 2 for x, p in zip(xs, progress)) / k)
        scored.append((err, name))
    scored.sort()
    return {"gsap": scored[0][1], "rmse": round(scored[0][0], 3),
            "runner_up": scored[1][1], "note": f"least-squares fit over {k} frames"}


def element_entries(frames: np.ndarray, fps: float, shots: list[dict], src_w: int, src_h: int,
                    window: tuple[float, float] | None = None, change_thr: float = 22.0,
                    max_entry_s: float = 1.2) -> list[dict]:
    """Inside each shot: a localized region starts changing, then goes static while holding new content
    (= an element animated in and settled). Footage motion rarely settles inside max_entry_s and is dropped.
    Limits: one region per episode, 1-frame resolution (1000/fps ms), fades and moves are both 'entries'."""
    n, sh, sw, _ = frames.shape
    fms = 1000.0 / fps
    out = []
    for s in shots:
        a = _frame_index(s["start"], fps, n) + 1          # skip the cut frame itself
        b = min(_frame_index(s["end"], fps, n) - 1, n - 1)
        if window and (s["end"] <= window[0] or s["start"] >= window[1]):
            continue
        if b - a < 3:
            continue
        g = np.stack([_gray(frames[i]) for i in range(a - 1, b + 1)])  # g[0] = frame before a
        d = np.abs(np.diff(g, axis=0)) > change_thr                     # d[j] = change into frame a+j
        active = d.reshape(len(d), -1).mean(1)
        i = 0
        while i < len(d):
            if active[i] < 0.0015 or active[i] > 0.5:   # nothing, or whole-frame change (flash/footage)
                i += 1
                continue
            j = i
            region = d[i].copy()
            while j + 1 < len(d) and active[j + 1] >= 0.0015 and (j - i) * fms < max_entry_s * 1000:
                j += 1
                region |= d[j]
            # settled: next 2 frames quiet inside the region
            rb = _dilate(region, 2)
            quiet = all(j + q < len(d) and (d[j + q] & rb).mean() < 0.0008 for q in (1, 2))
            ys, xs = np.nonzero(region)
            if quiet and len(ys) > 20:
                box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                area = (box[2] - box[0]) * (box[3] - box[1]) / (sw * sh)
                if area < 0.6:
                    pre = g[i]                     # frame before the change
                    fin = g[j + 1]                 # settled frame
                    x0, y0, x1, y1 = box
                    tot = np.abs(fin[y0:y1, x0:x1] - pre[y0:y1, x0:x1]).mean() + 1e-6
                    prog, heights, widths, cys = [], [], [], []
                    for f in range(i, j + 2):
                        cur = g[f][y0:y1, x0:x1]
                        prog.append(float(1 - np.abs(fin[y0:y1, x0:x1] - cur).mean() / tot))
                        m = np.abs(cur - pre[y0:y1, x0:x1]) > change_thr
                        my, mx = np.nonzero(m)
                        if len(my):
                            heights.append(my.max() - my.min() + 1)
                            widths.append(mx.max() - mx.min() + 1)
                            cys.append((my.max() + my.min()) / 2)
                    prog = [min(max(p, 0.0), 1.0) for p in prog]
                    # staggered build? cluster each pixel's first-change frame inside the episode
                    reg = region[y0:y1, x0:x1]
                    stack = d[i:j + 1, y0:y1, x0:x1]
                    first = np.where(stack.any(0), stack.argmax(0), -1)[reg]
                    first = first[first >= 0]
                    parts = []
                    if len(first):
                        hist = np.bincount(first, minlength=j - i + 2).astype(float)
                        sm = hist[:-1] + hist[1:]                   # 2-frame smoothing: fades spread over frames
                        onsets = [k for k in range(len(sm)) if sm[k] >= 0.10 * len(first)
                                  and (k == 0 or sm[k] >= sm[k - 1])]
                        for k in onsets:
                            if not parts or k - parts[-1]["onset"] >= 3:
                                parts.append({"onset": k})
                        for pi, part in enumerate(parts):
                            k0 = part["onset"]
                            k1 = parts[pi + 1]["onset"] - 1 if pi + 1 < len(parts) else j - i
                            fm = np.where(stack.any(0), stack.argmax(0), -1)
                            sel = reg & (fm >= k0) & (fm <= k1)
                            last = k0
                            for k in range(k0, j - i + 1):
                                if (stack[k] & sel).sum() > 0.05 * max(sel.sum(), 1):
                                    last = k
                            part.update({"offset_ms": round(k0 * fms), "frames": last - k0 + 1,
                                         "ms": round((last - k0 + 1) * fms)})
                            del part["onset"]
                    staggered = len(parts) > 1
                    # ink travel: track the vertical centre of 'ink' (pixels far from the settled background)
                    bg = float(np.median(fin[y0:y1, x0:x1]))
                    iy = []
                    for f in range(i, j + 2):
                        ink = np.abs(g[f][y0:y1, x0:x1] - bg) > 60
                        rows = np.nonzero(ink.any(1))[0]
                        iy.append((rows.min() + rows.max()) / 2 if len(rows) else None)
                    iy = [v for v in iy if v is not None]
                    travel = (iy[0] - iy[-1]) if len(iy) > 2 else 0.0
                    travel_ease = None
                    if abs(travel) >= 1.0:
                        tp = [(iy[0] - v) / travel for v in iy]
                        moving = [k for k in range(1, len(iy)) if abs(iy[k] - iy[k - 1]) > 0.01]
                        tf = (moving[-1] + 1) if moving else len(iy)
                        travel_ease = {"px_1080x1920": round(travel * 1920 / sh), "frames": tf,
                                       "ms": round(tf * fms), "ease": fit_ease(tp[1:tf + 1], 0.0)["gsap"],
                                       "resolution_px_1080": round(1920 / sh, 1),
                                       "direction": "up" if travel > 0 else "down"}
                    # fade vs scale/move: does the changed area GROW over the entry, or is it full size from frame 1?
                    growth = (heights[0] / max(heights[-1], 1), widths[0] / max(widths[-1], 1)) if heights else (1, 1)
                    kind = "hard pop" if j - i + 1 <= 2 else "staggered build" if staggered else (
                        "fade/colour (full size from first frame)" if min(growth) >= 0.85 else "scale/move")
                    ov = 0.0
                    if kind == "scale/move" and len(heights) >= 3 and heights[-1] > 3:
                        ov = max(0.0, max(max(heights[:-1]) / heights[-1], max(widths[:-1]) / max(widths[-1], 1)) - 1)
                        ov = ov if ov < 0.6 else 0.0   # bigger 'overshoot' = shape changed, not a spring
                    frames_n = j - i + 1
                    t0 = (a + i) / fps
                    if staggered:
                        ease = {"gsap": "none" if all(pp["frames"] <= 2 for pp in parts) else "power1.out",
                                "note": "staggered build: " + ", ".join(f"+{pp['offset_ms']} ms ({pp['frames']} fr)"
                                                                        for pp in parts) +
                                        "; per-part ease from frame count (<= 2 frames = none)", "rmse": None}
                    else:
                        ease = fit_ease(prog[1:], ov)
                    out.append({
                        "t_start_s": round(t0, 3), "t_settle_s": round((a + j + 1) / fps, 3),
                        "entry_frames": frames_n, "entry_ms": round(frames_n * fms),
                        "resolution_ms": round(fms), "overshoot_pct": round(ov * 100, 1), "kind": kind,
                        "parts": parts if staggered else None,
                        "stagger_ms": (parts[1]["offset_ms"] - parts[0]["offset_ms"]) if staggered else None,
                        "ease": ease, "zone": _zone(box, sh), **_box_out(box, sw, sh, src_w, src_h),
                        "travel": travel_ease,
                        "basis": "measured", "shot_start_s": s["start"]})
            i = j + 1
    return out


def summarize_entries(entries: list[dict]) -> dict:
    if not entries:
        return {"n": 0}
    ms = sorted(e["entry_ms"] for e in entries)
    eases: dict[str, int] = {}
    for e in entries:
        eases[e["ease"]["gsap"]] = eases.get(e["ease"]["gsap"], 0) + 1
    ovs = [e["overshoot_pct"] for e in entries if e["overshoot_pct"] > 0]
    q = lambda p: ms[min(len(ms) - 1, int(round(p * (len(ms) - 1))))]
    st = [e for e in entries if e.get("kind") == "staggered build"]
    return {"n": len(entries), "entry_ms_p25": q(0.25), "entry_ms_p50": q(0.5), "entry_ms_p75": q(0.75),
            "kinds": {k: sum(1 for e in entries if e.get("kind") == k) for k in {e.get("kind") for e in entries}},
            "staggered_builds": [{"t_s": e["t_start_s"], "parts": e["parts"]} for e in st],
            "with_overshoot": len(ovs), "overshoot_pct_median": (sorted(ovs)[len(ovs) // 2] if ovs else 0),
            "ease_counts": dict(sorted(eases.items(), key=lambda kv: -kv[1])),
            "hard_pops_le_2_frames": sum(1 for e in entries if e["entry_frames"] <= 2)}


# ----------------------------------------------------------------------------- orchestration

def run_frames(path: Path, measured: dict, out_dir: Path | None = None) -> dict:
    """All frame-level measurements for one clip. Writes overlay crop stills to out_dir/overlays/ when given.
    Entries are measured at 180 px wide (motion is coarse, keeps it fast); overlays/badges at 360 px (small text)."""
    media = measured["media"]
    W, H, fps = media["width"], media["height"], media["fps"] or 25.0
    shots = measured["cuts"]["shots"]
    res: dict = {"basis": "measured", "tooling": "ffmpeg rawvideo + numpy (deterministic, no model)"}
    fr, sw, sh = read_frames(path, W, H, scale_w=180)
    entries = element_entries(fr, fps, shots, W, H)
    res["entries"] = entries
    res["entries_summary"] = summarize_entries(entries)
    res["entries_limits"] = (f"1-frame resolution ({1000 / fps:.0f} ms); one region per change episode; only entries that "
                             "settle inside 1.2 s over otherwise-still footage are counted, so word pops on moving footage "
                             "or during continuous motion are missed; staggered builds are split by each pixel's first-change frame (2-frame "
                             "smoothing, parts >= 3 frames apart); overshoot is only measured on single-part entries whose changed "
                             "area grows (scale/move), never on fades; travel = vertical centre of ink vs the settled background, "
                             f"resolution ~{1920 / 320:.0f} px @1920 at the 180 px analysis width")
    del fr
    fr, sw, sh = read_frames(path, W, H, scale_w=360)
    ovs = persistent_overlays(fr, fps, shots, W, H)
    dbs = dynamic_badges(fr, fps, shots, W, H)
    del fr
    if out_dir:
        for r in ovs + dbs:
            p = r["px_source"]
            pad = int(0.02 * W)
            crop = (max(0, p["x"] - pad), max(0, p["y"] - pad),
                    min(W - max(0, p["x"] - pad), p["w"] + 2 * pad), min(H - max(0, p["y"] - pad), p["h"] + 2 * pad))
            crop = (crop[0], crop[1], crop[2] - crop[2] % 2, crop[3] - crop[3] % 2)
            t = (r["t_start_s"] + r["t_end_s"]) / 2
            name = f"{r['id']}_t{r['t_start_s']:06.2f}-{r['t_end_s']:06.2f}s.jpg"
            s1 = save_still(path, t, out_dir / "overlays" / name, crop=crop)
            s2 = save_still(path, t, out_dir / "overlays" / name.replace(".jpg", "_context.jpg"), width=360)
            r["crop_still"] = f"overlays/{name}" if s1 else None
            r["context_still"] = f"overlays/{s2.name}" if s2 else None
    res["overlays"] = ovs
    res["dynamic_badges"] = dbs
    res["overlay_limits"] = ("cross-cut overlays need >= 1 cut where the shot-pair mid frames share the overlay's edges; "
                             "elements that live inside one shot (e.g. a timer pill on a title card) are reported by the "
                             "dynamic-badge pass only when the rest of the frame is still and the inside keeps changing")
    return res
