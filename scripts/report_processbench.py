#!/usr/bin/env python
"""§9.3's report, read back out of `processbench.json`.

The eval prints one JSON blob per subset as it goes, which scrolls past in tmux and buries
the two numbers that decide how to read the rest. This renders the same file as the table
§9.3 asks for, and nothing here recomputes anything -- it is a view.

**Read tau first, before any F1.** §9.2 makes the fitted threshold a free check on the whole
pipeline: training puts good steps at `Delta ~ -log gamma` and pushes `Delta_{z+1} >= m`, so
a healthy tau lands near their midpoint (0.347 at `discount 0.5`, `margin_steps 2.0`). A tau
far above that is not a threshold choice, it is tau climbing to dodge a good-step tail that
no phase-1 term bounds from above (§7.12) -- and the F1 underneath it is then a measurement
of that, not of the goal head.

The skyline rows are printed LAST and under their own banner. §5.1 measured that the gold
answer alone solves one entire half of the F1 exactly, on all four subsets, so a skyline is
never a reported result -- its only job is splitting "the goal head is the bottleneck" from
"the metric never learned correctness" (§9.5).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
NATURAL_TAU = 0.347          # (m - (-log gamma)) / 2 at discount 0.5, margin_steps 2.0 (§7.6.4)


def _row(name: str, m: dict) -> str:
    return (
        f"  {name:<24s} {m['acc_error']:>9.3f} {m['acc_correct']:>11.3f} {m['f1']:>7.3f}"
        f" {int(m['n_error']):>7d} {int(m['n_correct']):>8d}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="render processbench.json as §9.3's table")
    parser.add_argument(
        "path",
        nargs="?",
        default="runs/phase1/phase2/final/processbench.json",
        help="the file the eval wrote (default: the phase-2 checkpoint's)",
    )
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"no such file: {path}\nrun: bash scripts/eval_processbench.sh <phase2-ckpt>")
        return 2
    results = json.loads(path.read_text())

    print(f"\ncheckpoint: {results.get('checkpoint', '?')}")

    # ---- tau, and what it says (§9.2) -------------------------------------------------
    cal = results.get("calibration")
    if cal:
        tau, expected = cal["calibration/tau"], cal["calibration/expected_tau"]
        print(
            f"\ntau = {tau:.4f}   (natural midpoint {expected:.4f})"
            f"\n  fitted on held-out Math-Shepherd val questions, never on ProcessBench (§9.2)"
            f"\n  val F1 {cal['calibration/f1']:.4f}   sensitivity +/-0.1 -> "
            f"{cal['calibration/sensitivity']:.4f}"
        )
        # MULTIPLICATIVE, not additive. `expected` is 0.347; an additive slack of 1.0 lets a
        # 3.4x overshoot print "the margin held" -- which is exactly what tau=1.1685 did on
        # 2026-07-30. The quantity has no natural additive scale; the ratio is the reading.
        if tau > 2.0 * expected:
            print(
                f"  ** tau is {tau / max(expected, 1e-9):.1f}x the natural midpoint. It is "
                "climbing to dodge a\n     good-step Delta tail (§7.12, diagnostic #14), and "
                "TPR dies with it. The F1\n     below is bounded by that, NOT by the goal head."
                "\n     Read good/above_target_fraction and #14's p90/p99 for THIS run before"
                "\n     touching the goal head."
            )
        elif abs(tau) < 0.05:
            print(
                "  ** tau ~ 0: Delta_{z+1} only reached ~-log gamma, so the second step of "
                "margin\n     never landed. §9.2 says drop margin_steps to 1.0 rather than "
                "fight it."
            )
        else:
            print("  ** tau is near the midpoint -- the ruler and the margin both held.")

    header = (
        f"  {'subset':<24s} {'acc_error':>9s} {'acc_correct':>11s} {'F1':>7s}"
        f" {'n_err':>7s} {'n_ok':>8s}"
    )

    # ---- the reported result (§9.3) ---------------------------------------------------
    print("\nRESULT -- quasimetric Delta-d with the goal head. The only scoring path.")
    print(header)
    f1s = []
    for subset in SUBSETS:
        if subset not in results:
            continue
        m = results[subset]
        print(_row(subset, m))
        f1s.append(m["f1"])
        over = (m.get("counters") or {}).get("over_length", 0)
        if over:
            print(f"  {'':<24s} ({int(over)} over eval.max_len, predicted -1)")
        # Locked #5: the gap between these two IS the measurement of the contamination.
        if "leak_split" in m:
            for label, part in (("leaked (587)", "leaked"), ("clean (413)", "clean")):
                print(_row(f"  math, {label}", m["leak_split"][part]))
    if f1s:
        print(f"\n  mean F1 over {len(f1s)} subsets: {sum(f1s) / len(f1s):.3f}")
        print(
            "  (report PER SUBSET -- §5's distribution shift is real: our training solutions "
            "are\n   ~5 steps from Mistral-class generators, olympiadbench/omnimath are 8-9 "
            "from Qwen2.5-72B.)"
        )

    # ---- the skyline, last and labelled (§9.5) ----------------------------------------
    sky = {k: v for k, v in results.items() if k.endswith("_SKYLINE_not_a_result")}
    if sky:
        print("\nSKYLINE -- NOT A RESULT (§9.5). Reference-solution goals; the gold answer")
        print("alone solves one whole half of this metric exactly (§5.1).")
        print(header)
        for key, m in sky.items():
            print(_row(key.replace("_SKYLINE_not_a_result", ""), m))
        print(
            "\n  skyline >> result  -> the geometry works, the GOAL HEAD is the bottleneck"
            "\n  both bad           -> the metric never learned correctness (§1.2's old project)"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
