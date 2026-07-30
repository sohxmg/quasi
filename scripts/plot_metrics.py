#!/usr/bin/env python
"""Summarise `metrics.jsonl` -- every loss curve separately (§10 #11).

Dependency-free on purpose: matplotlib is not in requirements.txt, so this prints ASCII
sparklines and writes a wide CSV you can plot with whatever you like.

What to look at, in order:
  * `probe14/*` -- the three-way Delta histogram. **The single best predictor of
    ProcessBench F1** (§7.6.6). A positive tail on `delta_good_of_correct` is F1 leaking
    with no loss training against it.
  * `probe02/delta_good_mean` against `-log gamma` (-0.693 at discount 0.5). The old
    project's was 108x off and nobody noticed.
  * `backup/loss` -- expected NEGATIVE. Watch plateau and NaN, not sign.
  * `nce/logit_std` -- ~0 with the loss pinned at log(R) is bug B10a.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run without `pip install -e .`

import argparse
import csv

from feynman_prm.diagnostics.logging import read_metrics

BLOCKS = " .:-=+*#%@"

HEADLINE = [
    "loss/total",
    "nce/loss",
    "nce/logit_std",
    "invariance/residual_diagonal",
    "backup/loss",
    "backup/linear_branch_fraction",
    "step/loss",
    "step/delta_mean",
    "good/loss",
    "good/delta_mean",
    "good/above_target_fraction",
    "probe02/delta_good_mean",
    "probe03/delta_bad_mean",
    "probe03/gap",
    # The tail, not the mean (§7.12): the mean read +0.240 for a whole run while a third of
    # good steps sat above tau and F1 capped at 0.456.
    "probe14/delta_good_of_correct/frac_above_natural",
    "probe14/delta_good_of_correct/p99",
    "probe14/delta_good_of_correct/positive_fraction",
    "probe01/distinct_goal_ratio",
    "probe04/symmetric_share",
]


def sparkline(values: list[float]) -> str:
    finite = [v for v in values if v == v and abs(v) != float("inf")]
    if not finite:
        return "(no data)"
    lo, hi = min(finite), max(finite)
    span = (hi - lo) or 1.0
    return "".join(BLOCKS[min(int((v - lo) / span * (len(BLOCKS) - 1)), len(BLOCKS) - 1)]
                   if v == v else "?" for v in values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="summarise metrics.jsonl")
    parser.add_argument("path", help="runs/<name>/metrics.jsonl")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--keys", nargs="*", default=None)
    args = parser.parse_args(argv)

    records = read_metrics(args.path)
    if not records:
        print("no records")
        return 1
    keys = args.keys or [k for k in HEADLINE if any(k in r for r in records)]

    width = max(len(k) for k in keys)
    print(f"{len(records)} logged points, steps {records[0]['step']}..{records[-1]['step']}\n")
    for key in keys:
        series = [r.get(key, float("nan")) for r in records]
        finite = [v for v in series if v == v]
        first, last = (finite[0], finite[-1]) if finite else (float("nan"), float("nan"))
        print(f"{key:<{width}}  {sparkline(series)}  {first:+.4f} -> {last:+.4f}")

    if args.csv:
        all_keys = sorted({k for r in records for k in r})
        with Path(args.csv).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(records)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
