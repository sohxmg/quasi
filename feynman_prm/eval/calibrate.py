"""Threshold calibration (§9.2). tau is fit on the held-out Math-Shepherd VALIDATION
questions, NEVER on ProcessBench.

The fit is also a free check on (5) L_step. Training puts good steps at `Delta ~= -0.693`
and pushes `Delta_{z+1} >= m = 1.386`, so the fitted tau should land near their midpoint,
**0.347** at discount 0.5:

    tau ~= 0.347  -> the margin took
    tau ~= 0      -> the model only reached Delta_{z+1} ~= 0.693, i.e. the second step of
                     margin never landed. Drop margin_steps to 1.0 (§16.19) rather than
                     fighting it.
    tau far off   -> read diagnostic #14's histogram before touching anything.

This is a check, not a constraint: tau is still whatever maximises validation F1. If F1 is
very sensitive to tau, say so -- it means the good-step / bad-step separation is weak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ..config import Config
from ..diagnostics.logging import Progress
from .metrics import processbench_metrics
from ..utils.indexing import predicted_label_from_deltas


@dataclass
class Calibration:
    tau: float
    f1: float
    curve: list[dict[str, float]]
    expected_tau: float
    sensitivity: float          # F1 range within +/-0.1 of the chosen tau

    def summary(self) -> dict[str, float]:
        return {
            "calibration/tau": self.tau,
            "calibration/f1": self.f1,
            "calibration/expected_tau": self.expected_tau,
            "calibration/sensitivity": self.sensitivity,
        }


def natural_tau(cfg: Config) -> float:
    """The midpoint implied by the ruler and the margin: (m + (-log gamma))/2 - (-log gamma)
    = (m - (-log gamma)) / 2. 0.347 at discount 0.5 / margin_steps 2.0 (§7.6.4)."""
    return (cfg.step_margin - cfg.neg_log_gamma) / 2.0


@torch.no_grad()
def score_validation(model, rows, cfg: Config, device, pad_id: int) -> tuple[list, list]:
    """Score held-out Math-Shepherd validation trajectories the way §9.1 scores ProcessBench.

    The gold label is `z` for an incorrect trajectory and -1 for a correct one -- the same
    convention ProcessBench uses -- so the identical F1 rule applies and tau transfers.
    """
    from ..data.collate import collate

    deltas: list[list[float]] = []
    labels: list[int] = []
    pending = []

    def flush() -> None:
        if not pending:
            return
        batch = collate(pending, pad_id=pad_id).to(device)
        reps = model(batch)
        h_s0 = reps.h_states.index_select(0, batch.traj_state_offset)
        goals = model.goal_head(h_s0)
        for b in range(batch.n_traj):
            T = int(batch.traj_T[b])
            offset = int(batch.traj_state_offset[b])
            states = reps.psi[offset : offset + T + 1]
            d = model.distance(states, goals[b].expand_as(states))
            deltas.append((d[1:] - d[:-1]).float().cpu().tolist())
        progress.advance(len(pending))
        pending.clear()

    # ~18k held-out sequences (2,000 val questions x 9.18): minutes of silence before
    # ProcessBench is even touched, and tau comes out of it (§9.2).
    progress = Progress("calibrate/val", len(rows))
    for row in rows:
        pending.append(row)
        labels.append(int(row.z) if not row.correct else -1)
        if len(pending) >= cfg.eval.batch_sequences:
            flush()
    flush()
    return deltas, labels


def calibrate_tau(
    deltas: Sequence[Sequence[float]],
    labels: Sequence[int],
    cfg: Config,
    grid: Sequence[float] | None = None,
) -> Calibration:
    """Pick tau maximising validation F1 under the ProcessBench rule."""
    if grid is None:
        flat = np.concatenate([np.asarray(d) for d in deltas if len(d)]) if deltas else np.zeros(1)
        lo, hi = float(np.quantile(flat, 0.01)), float(np.quantile(flat, 0.99))
        grid = np.unique(np.concatenate([np.linspace(lo, hi, 201), [0.0, natural_tau(cfg)]]))

    curve = []
    for tau in grid:
        preds = [predicted_label_from_deltas(d, float(tau)) for d in deltas]
        metrics = processbench_metrics(preds, labels)
        curve.append({"tau": float(tau), **metrics})

    best = max(curve, key=lambda row: row["f1"])
    near = [row["f1"] for row in curve if abs(row["tau"] - best["tau"]) <= 0.1]
    return Calibration(
        tau=best["tau"],
        f1=best["f1"],
        curve=curve,
        expected_tau=natural_tau(cfg),
        sensitivity=max(near) - min(near) if near else 0.0,
    )
