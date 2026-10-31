#!/usr/bin/env python3
"""Offline self-test for edit mode helpers (no model calls, no video needed).
Run: .venv/bin/python scripts/selftest_edit.py"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import edit_brief as B  # noqa: E402
import edit_frames as F  # noqa: E402
import edit_mode as M  # noqa: E402


def check(cond, msg):
    if not cond:
        raise SystemExit(f"FAIL: {msg}")
    print(f"ok  {msg}")


def schema_depth(x, d=0):
    if isinstance(x, dict):
        here = d + (1 if x.get("type") == "object" else 0)
        return max([schema_depth(v, here) for v in x.values()] + [here])
    if isinstance(x, list):
        return max([schema_depth(v, d) for v in x] + [d])
    return d


def main():
    # px conversion at 1080x1920
    check("1037 px" in B.px_note("54% of height"), "px_note: 54% of height -> 1037 px @1920")
    # ease text -> GSAP
    check(B.gsap_from_text("spring with overshoot") == "back.out(1.7)", "gsap_from_text spring -> back.out(1.7)")
    # sound honesty
    for t in ("no SFX needed", "SFX are not used", "avoid whoosh sounds"):
        check(B._SFX_NEG.search(t), f"sound honesty catches: {t!r}")
    check(not B._SFX_NEG.search("whoosh on every cut"), "sound honesty keeps positive SFX claims")
    # ease fit recovers exact GSAP curves (samples are frames after the first change)
    t = np.linspace(0, 1, 13)[1:]
    for want, f in (("power1.out", lambda p: 1 - (1 - p) ** 2), ("power2.out", lambda p: 1 - (1 - p) ** 3),
                    ("linear", lambda p: p), ("sine.out", lambda p: math.sin(p * math.pi / 2)),
                    ("expo.out", lambda p: 1 - 2 ** (-10 * p))):
        got = F.fit_ease([f(x) for x in t], 0.0)["gsap"]
        check(got == want, f"fit_ease recovers {want} (got {got})")
    check(F.fit_ease([1.0, 1.0], 0.0)["gsap"] == "none", "<= 2 frames -> none (hard pop)")
    check(F._back_s_for_overshoot(0.10) == 1.7, "10% overshoot -> back.out(1.7)")
    # Gemini schema stays flat
    d = schema_depth(M.EDIT_SCHEMA)
    check(d <= 3, f"EDIT_SCHEMA object nesting depth {d} <= 3")
    props = M.EDIT_SCHEMA["properties"]
    dn = props["edit_brief"]["properties"]["do_not"]["items"]["properties"]
    check("assets" in props and set(dn) == {"rule", "scope"}, "schema has assets + do_not {rule, scope}")
    print("all offline checks passed")


if __name__ == "__main__":
    main()
