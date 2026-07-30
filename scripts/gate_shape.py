#!/usr/bin/env python
"""Why `gate/recall_at_1` moved, and whether the goal head still has a target.

`goal_gate.py` reports three numbers that can disagree, and on the phase-1 checkpoint they
do: `ratio` **improved** 0.598 -> 0.303, `auc` stayed flat (0.897 -> 0.906, i.e. at the
untrained bar of §10.1.1), and `recall@1` **halved** 0.638 -> 0.280 with
`questions_fully_scattered` doubling 0.32 -> 0.62. A mean ratio and a nearest-neighbour
recall cannot both be right about "did clustering improve", so one of them is measuring
something else. This script says which.

Two questions, neither of which the gate can answer:

**1. Is the recall collapse a LEVEL problem or a VARIANCE problem?**
`recall@1` is decided by the *left tail* of the across-question distribution racing the
*body* of the within-question one -- with ~520 competing across-pairs per terminal, the
nearest impostor is drawn from the across distribution's ~0.2nd percentile, not its mean.
So a geometry can widen its mean gap (which `ratio` and `auc` reward) and still lose every
nearest-neighbour race, provided the variance widened faster. That is the signature to
check, because it is the same signature as the ruler decay in §16.23: `backup/delta_mean`
falling to 0.49 means the per-step scale is unanchored, and an unanchored scale inflates
spread rather than position. `probe14`'s `std` tripling over the same window is the third
view of it.

`hub_share` separates a third possibility: a handful of terminals collapsing to one point
would win every nearest-neighbour race by being close to *everything*, which destroys
recall without touching either distribution's shape.

**2. Can a question-conditioned goal head beat a constant?**
This is the question §10.1 was always a proxy for, and it can be asked directly on the same
cached terminals -- **without the goal head existing**, which is the point of the phase split
(§7.7). Phase 2 minimises `mean_c [ d(pred_q, t_c) + d(t_c, pred_q) ]`, so:

    floor = mean over q of the best per-question prediction   <- a perfect head
    blind = the best SINGLE prediction for every question     <- diagnostic #6's failure mode

Both are restricted to predicting an *observed* terminal (the medoid), so `floor` is an
upper bound on what a head predicting a free point in R^D could reach and the comparison is
conservative in the direction that matters. **`floor / blind` is the headroom the head has.**
Near 1.0 means the head can do no better than a global anchor and will learn one -- root
cause D by a different route, and the actual "STOP AND REDESIGN" condition. Well below 1.0
means the target exists and `recall@1` is not measuring the thing that decides phase 2.

Neither answer is F1. `scripts/val_f1.py` is F1: it substitutes a sibling correct terminal
of the same question for `g_q` and fits tau, which is the §9.5 skyline and the only path
from any of this to a reported number.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run without `pip install -e .`

import argparse
import json
import math

import torch

from goal_gate import add_gate_args, cache_terminals   # noqa: E402  (after the sys.path insert)


def _q(x: torch.Tensor, qs: list[float]) -> dict[str, float]:
    ts = torch.tensor(qs, device=x.device, dtype=x.dtype)
    return {f"p{q * 100:g}": float(v) for q, v in zip(qs, torch.quantile(x, ts))}


@torch.no_grad()
def shape(psi_t: torch.Tensor, qidx: torch.Tensor, distance, step_cost: float) -> dict:
    n = len(psi_t)
    d = distance(psi_t[:, None, :], psi_t[None, :, :]).float()
    same = qidx[:, None] == qidx[None, :]
    eye = torch.eye(n, dtype=torch.bool, device=d.device)

    w = d[same & ~eye].flatten()
    a = d[~same].flatten()

    # ---- 1. level vs variance ---------------------------------------------------------
    # The nearest impostor is an order statistic, not a mean: with `across_per_terminal`
    # competitors the winner sits at about the 1/k quantile of the across distribution.
    across_per_terminal = len(a) / n
    out = {
        "within/mean": float(w.mean()),
        "within/std": float(w.std()),
        "within/in_ruler_units": float(w.mean()) / step_cost,
        "across/mean": float(a.mean()),
        "across/std": float(a.std()),
        # Cohen's d. `ratio` and `auc` both read this; `recall@1` does not.
        "separation/cohens_d": float(
            (a.mean() - w.mean()) / torch.sqrt(0.5 * (a.var() + w.var()))
        ),
        "separation/across_competitors_per_terminal": across_per_terminal,
    }
    out.update({f"within/{k}": v for k, v in _q(w, [0.5, 0.9, 0.99]).items()})
    out.update({f"across/{k}": v for k, v in _q(a, [0.001, 0.01, 0.05, 0.5]).items()})

    # THE comparison recall@1 actually runs. If the across left tail sits under the within
    # body, nearest-neighbour recall is lost no matter how far apart the means are.
    out["separation/across_p0.1_minus_within_p50"] = out["across/p0.1"] - out["within/p50"]

    # ---- 2. hubness -------------------------------------------------------------------
    # A few terminals collapsed onto one point would be everyone's nearest neighbour and
    # would sink recall@1 without moving either distribution.
    nearest = d.masked_fill(eye, float("inf")).argmin(dim=1)
    counts = torch.bincount(nearest, minlength=n).float()
    top = counts.sort(descending=True).values
    out["hub/max_times_nearest"] = float(top[0])
    out["hub/top1pct_share"] = float(top[: max(1, n // 100)].sum() / n)
    out["hub/expected_uniform_share"] = max(1, n // 100) / n

    # ---- 3. the goal head's headroom (§7.7) -------------------------------------------
    # L_goal is symmetrised, so score a candidate by d(pred, t) + d(t, pred).
    #
    # A candidate must NEVER be scored against itself: `d(x, x) = 0` lets a terminal win by
    # predicting itself, which no head can do, and at ~2.6 terminals per question that cheat
    # is worth a third of the floor. Masking the diagonal is what makes `floor` and `blind`
    # the same measurement asked at two scopes.
    sym = (d + d.t()).masked_fill(eye, float("nan"))

    def score(cols: torch.Tensor) -> torch.Tensor:
        """Mean symmetrised cost of every candidate (rows of `sym`) against `cols`."""
        return sym[:, cols].nanmean(dim=1)

    def best(x: torch.Tensor) -> torch.Tensor:
        """`min` ignoring the masked diagonal (torch < 2.14 has no `nanmin`)."""
        return x.nan_to_num(nan=float("inf")).min()

    per_q = torch.unique(qidx)
    cols_of = {int(q): torch.nonzero(qidx == q, as_tuple=True)[0] for q in per_q.tolist()}

    floors, blind_per_candidate = [], []
    for q in per_q.tolist():
        cols = cols_of[q]
        s = score(cols)
        # Restricted to that question's own terminals: a head predicting a free point in R^D
        # could only do better, so this OVERSTATES the floor and understates the headroom.
        floors.append(best(s[cols]))
        blind_per_candidate.append(s)
    floor = float(torch.stack(floors).mean())

    # The best question-BLIND constant, scored the same way: ONE predictor for every
    # question, which is exactly what diagnostic #6 fires on.
    blind = float(best(torch.stack(blind_per_candidate).mean(dim=0)))

    out["goal_head/floor_per_question"] = floor
    out["goal_head/floor_question_blind"] = blind
    out["goal_head/headroom_ratio"] = floor / blind if blind else float("nan")
    out["goal_head/floor_in_ruler_units"] = floor / (2 * step_cost)   # symmetrised -> 2 legs
    return out


def main(argv: list[str] | None = None) -> int:
    parser = add_gate_args(
        argparse.ArgumentParser(description="the shape behind gate/recall_at_1")
    )
    args = parser.parse_args(argv)

    model, psi_t, qidx, cfg = cache_terminals(args)
    step_cost = -math.log(cfg.discount)
    out = shape(psi_t, qidx, model.distance, step_cost)
    out["untrained_baseline"] = args.untrained
    print(json.dumps(out, indent=2))

    print(
        f"\nrecall@1's actual race: across p0.1 = {out['across/p0.1']:.2f}  vs  "
        f"within p50 = {out['within/p50']:.2f}   (gap "
        f"{out['separation/across_p0.1_minus_within_p50']:+.2f}; negative = impostors win)"
        f"\nspread:  within {out['within/mean']:.2f} +- {out['within/std']:.2f}   "
        f"across {out['across/mean']:.2f} +- {out['across/std']:.2f}   "
        f"cohen's d {out['separation/cohens_d']:.2f}"
        f"\nhubness: top 1% of terminals win {out['hub/top1pct_share']:.1%} of the "
        f"nearest-neighbour slots (uniform would be "
        f"{out['hub/expected_uniform_share']:.1%})"
        f"\ngoal head: per-question floor {out['goal_head/floor_per_question']:.2f}  vs  "
        f"question-blind {out['goal_head/floor_question_blind']:.2f}  ->  headroom ratio "
        f"{out['goal_head/headroom_ratio']:.3f}"
        f"\n           ({out['goal_head/floor_in_ruler_units']:.1f} steps of irreducible "
        f"target ambiguity, at -log gamma = {step_cost:.3f})"
    )
    # The gate's own verdict is on auc (§10.1.1); this script gates on the one condition that
    # makes phase 2 pointless -- a head that cannot beat a constant.
    return 0 if out["goal_head/headroom_ratio"] < 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
