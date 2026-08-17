#!/usr/bin/env python
"""Read a finished phase-1 run against the guards that were set for it, before any GPU.

    python scripts/run_report.py runs/phase1_nce_temp_relu2
    python scripts/run_report.py runs/phase1_nce_temp_relu2 --baseline runs/phase1

**Every number here is a WINDOW MEAN over the last 20% of logged steps, not a single point.**
Each of these is batch-noisy at +/-0.3 and §16.23's two-run table was built the same way; a
last-row reading would be a coin flip on most of them.

The verdicts are deliberately coarse and every row prints its own target, because the one
thing this file's history says over and over is that a guard rendering a verdict is a guard
that can render the wrong one (B11, B12, B13, §10.1.1's `< 0.3`). Read the numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run without `pip install -e .`

import argparse
import json
import math

# (key, target, how to read it) -- targets are from CLAUDE.md, cited per row.
GUARDS = [
    ("backup/delta_mean", 0.69315,
     "§16.23 THE RULER. Must SETTLE at -log gamma, not pass through it. Decayed 0.86 -> 0.49 "
     "on BOTH previous runs; zeta 0.1 is the test of that."),
    ("invariance/residual_diagonal", 0.26,
     "§7.12 / §17. Where L_good's cost lands, and relu_squared is unbounded. Compare to the "
     "MEASURED lambda_good=0 baseline 0.263 -- NOT to §7.12's 0.15, which was calibrated "
     "against a simulated 0.098 and fires on the baseline run itself."),
    ("probe14/delta_good_of_correct/frac_above_natural", 0.05,
     "§7.12 relu_squared's TARGET. Was 0.34 at step 750 under relu, then regressed to ~0.16 "
     "mid-run. THE number this form change exists to move."),
    ("probe14/delta_good_of_correct/p99", None,
     "§7.12 the tail itself. Was 2.43 mid-run under relu. Read this, never the mean -- the "
     "mean was -0.412 while the tail ran away."),
    ("probe02/delta_good_mean", -0.69315,
     "§9.9's judge for the mask. Was -0.32 against a target of -0.693."),
    ("probe03/gap", 1.8,
     "§7.12's guard: bad-step Delta minus good-step Delta. Below ~1.8 means the error signal "
     "is flattening (§16.3, diagnostic #3)."),
    ("probe14/delta_boundary/mean", 1.38629,
     "§7.6.4: Delta_{z+1} should reach m = margin_steps * -log gamma."),
    ("nce/accuracy_within_question", None,
     "§9.10.3 THE INTERPRETABILITY GAUGE, and it is THIS key -- not the categorical one. "
     "Scored below against chance 1/(1+negatives_same_question). §9.8.2 has tau=1.0 reaching "
     "10.6x by step 1460 on runs/phase1; far below that means the mask result cannot be read."),
    ("nce/categorical_accuracy_backward", None,
     "phase1: 0.2836. Chance is 1/R over the WHOLE pool, ~10x tighter than the row above -- "
     "the two multiples are not comparable (§3.3, B12's shape a fourth time)."),
    ("nce/argmax_in_nearer_set", None,
     "§9.9.6, computed on RAW logits so it reads through the mask. Was 0.369 (195x its null) "
     "on the previous checkpoint."),
    ("good/above_target_fraction", None,
     "§10 #18. Reached 1.0000 by step 50 of the probe with lambda_good half-ramped."),
    ("train/grad_norm", 1.0,
     "§14. PRE-clip, against train.grad_clip. Far above it all run means the clip stopped "
     "being a guard and became an LR rescale -- raise it rather than leave it binding."),
]


def window(records, key, frac=0.2):
    vals = [(r["step"], r[key]) for r in records if key in r and r[key] is not None]
    if not vals:
        return None, None, 0
    cut = vals[-1][0] - (vals[-1][0] - vals[0][0]) * frac
    tail = [v for s, v in vals if s >= cut]
    return sum(tail) / len(tail), vals, len(tail)


def trend(vals):
    """First-half mean -> last-half mean. Says DECAYING vs SETTLING, which is the whole
    question for the ruler (§16.23) and the tail (§7.12)."""
    if len(vals) < 4:
        return None
    half = len(vals) // 2
    a = sum(v for _, v in vals[:half]) / half
    b = sum(v for _, v in vals[half:]) / (len(vals) - half)
    return a, b


def load(run_dir: Path):
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        raise SystemExit(f"no metrics at {path}")
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    records = [r for r in records if "step" in r]
    # RunLogger opens append-mode, so a directory can hold a probe AND the real run
    # concatenated (that is how runs/phase1/metrics.jsonl has steps 1/10/20 twice). Keep the
    # LAST monotone series -- restarts show up as the step counter going backwards.
    starts = [i for i in range(1, len(records)) if records[i]["step"] <= records[i - 1]["step"]]
    if starts:
        print(f"  [note] {len(starts) + 1} series in metrics.jsonl (append-mode); "
              f"reading the last, from record {starts[-1]}")
        records = records[starts[-1]:]
    return records


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="read a finished phase-1 run against its guards")
    p.add_argument("run_dir")
    p.add_argument("--baseline", default=None, help="a previous run to diff against")
    p.add_argument("--window", type=float, default=0.2)
    args = p.parse_args(argv)

    records = load(Path(args.run_dir))
    base = load(Path(args.baseline)) if args.baseline else None
    last = records[-1]["step"]
    print(f"\n{args.run_dir}: {len(records)} logged points, last step {last}, "
          f"window = last {args.window:.0%}\n")

    for key, target, note in GUARDS:
        mean, vals, n = window(records, key, args.window)
        if mean is None:
            print(f"  {key:<48} -- not logged --")
            continue
        tr = trend(vals)
        line = f"  {key:<48} {mean:+9.4f}"
        if target is not None:
            line += f"   target {target:+7.4f}"
        if tr:
            line += f"   trend {tr[0]:+.3f} -> {tr[1]:+.3f}"
        if base is not None:
            bmean, _, _ = window(base, key, args.window)
            if bmean is not None:
                line += f"   baseline {bmean:+.4f}"
        print(line)
        print(f"      {note}")

    # the two that need arithmetic rather than a level.
    #
    # B12 AGAIN, FIXED 2026-08-04 (§3.3): this block used to multiply
    # `categorical_accuracy_backward` (chance 1/R ~ 1/348) by `negatives_per_column` and compare
    # the product against 10.6x -- a multiple §9.8.2 measured on `accuracy_within_question`
    # (chance 1/(1+n_same) ~ 1/32). Different statistic, different pool, tenfold different
    # chance level, and the `< 4.0` abort inherited the error: phase1_nce_temp_relu2 printed
    # "4.5x" and passed a guard it was never eligible for, where the like-for-like number was
    # 3.1x against 10.8x. Each statistic is now scored against ITS OWN chance level and the
    # 10.6x reference sits only on the row it was measured on.
    within, _, _ = window(records, "nce/accuracy_within_question", args.window)
    n_same, _, _ = window(records, "nce/negatives_same_question", args.window)
    multiple = None
    if within is not None and n_same is not None and n_same > 0:
        chance = 1.0 / (1.0 + n_same)
        multiple = within / chance
        print(f"\n  nce/accuracy_within_question = {within:.4f} = {multiple:.1f}x chance "
              f"(chance 1/{1.0 + n_same:.0f}).  THE INTERPRETABILITY GAUGE: tau=1.0 reached "
              f"10.6x by step 1460 on runs/phase1 (§9.8.2, same statistic, same pool).")
        if multiple < 6.0:
            print("  ** well below phase1's 10.6x: (1) is muted or the geometry contracted, and"
                  "\n     the MASK result in this run is not interpretable (§9.10.3, §2.4). **")

    acc, _, _ = window(records, "nce/categorical_accuracy_backward", args.window)
    negs, _, _ = window(records, "nce/negatives_per_column", args.window)
    if acc is not None and negs:
        print(f"  nce/categorical_accuracy_backward = {acc:.4f} = {acc * (negs + 1):.0f}x "
              f"chance (chance 1/{negs + 1:.0f}, the WHOLE pool).  phase1: 0.2836. "
              f"Do NOT compare this multiple against the 10.6x above -- different pool.")

    ruler, rvals, _ = window(records, "backup/delta_mean", args.window)
    if ruler is not None and trend(rvals):
        a, b = trend(rvals)
        verdict = ("DECAYING -- §16.23 recurs, zeta 0.1 did not fix it"
                   if b < a - 0.05 and b < 0.6 else
                   "settled near target" if abs(b - 0.69315) < 0.15 else
                   "off target, read the curve")
        print(f"\n  ruler: {a:+.3f} -> {b:+.3f} against 0.693  ->  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
