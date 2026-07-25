"""(3) L_T -- temporal invariance, the LINEX backup (§7.4, `tmd.py:107-122`).

**This is the ruler.** L_NCE only teaches ranking; L_T sets the scale: one good step shrinks
the distance by exactly `-log gamma = 0.693` (at discount 0.5). Distances then read in units
of *steps remaining*, which is what makes the single global eval threshold tau mean the same
thing on every question -- and (5) L_step is expressed in those same units.

    delta[r,c] = Dist[r,c] - Next[r,c]              Next is ALWAYS detached (tmd.py:113)
    mask       = delta > t                          t = clip_t_steps * (-log gamma)
    div        = where(mask, delta, gamma*exp(where(mask, t, delta)) - Dist)

Minimiser: `d/dDist [gamma*exp(Dist - Next) - Dist] = 0  =>  Dist = Next - log gamma`, i.e.
arriving at s_i cut the distance to g by exactly one step's worth. The value at that minimum
is `1 - Dist`, so **L_T is expected to be NEGATIVE. Watch plateau and NaN, not sign.**

Two things not to "simplify":

* The **double `where`** is exponent clipping, and it is load-bearing because `torch.where`
  evaluates the discarded branch in the backward pass -- an `inf` there poisons the gradient
  even though the value is thrown away. A fixture with `delta = 200` must give finite loss
  AND finite gradients (§15).
* The two branches are discontinuous in value *and* slope at `delta == t`. Do not smooth
  them. Branch B subtracts **`Dist`**, not `delta`.

All pairs, not just matched pairs (§7.4.1): `Dist` and `Next` are full (source x goal)
matrices, so this is ~60,000 backup terms per step, not ~348. It costs nothing -- L_NCE
already built `Dist` -- and it is the point of the loss: calibrating only matched pairs lets
the model learn a different scale per question ("one step = 0.36 on algebra, 5.0 on
geometry"), which satisfies a matched-only loss perfectly and breaks a single global tau.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..config import Config


def temporal_loss(
    Dist: Tensor,
    Next: Tensor,
    pos_row: Tensor,
    SQ: Tensor,
    cfg: Config,
) -> tuple[Tensor, dict[str, float]]:
    """Returns (loss, diagnostics). `Dist` here is `Matrices.Dist_backup`."""
    gamma = cfg.discount
    t = cfg.clip_t                       # scale-free: clip_t_steps * (-log gamma), §7.4.3
    dw = cfg.losses.backup.diag_backup
    rho = cfg.losses.backup.goal_scope_ratio

    delta = Dist - Next                  # Next is already detached (matrix.py)
    mask = delta > t
    delta_clipped = torch.where(mask, torch.full_like(delta, t), delta)
    div = torch.where(mask, delta, gamma * torch.exp(delta_clipped) - Dist)

    # tmd.py:120-121 broadcasts the matched-pair divergence across every column of its row,
    # so after the mean it reduces exactly to a (1-dw)/dw mix in which the matched set is
    # counted inside BOTH terms (§7.4.2). Reproduce by mixing, not by partitioning.
    cols = torch.arange(Dist.shape[1], device=Dist.device)
    matched = div[pos_row, cols]

    all_mean = div.mean()
    same_q = div[SQ]
    same_q_mean = same_q.mean() if same_q.numel() else all_mean
    non_matched = rho * all_mean + (1.0 - rho) * same_q_mean
    loss = (1.0 - dw) * non_matched + dw * matched.mean()

    with torch.no_grad():
        cross = div[~SQ]
        info = {
            "backup/loss": float(loss),
            "backup/delta_mean": float(delta.mean()),
            "backup/dist_mean": float(Dist.mean()),
            "backup/target_step_cost": cfg.neg_log_gamma,
            # Diagnostic #15: the LINEX guard should fire RARELY. If it climbs, raise
            # clip_t_steps -- do not change `discount` (§7.4.3).
            "backup/linear_branch_fraction": float(mask.float().mean()),
            # Diagnostic #13: logged separately and ALWAYS, whatever goal_scope_ratio is.
            # Cross-question pairs ask to shrink a distance to an unreachable goal (§7.4.2);
            # if that term dominates or diverges, lower rho.
            "backup/div_same_question": float(same_q.mean()) if same_q.numel() else 0.0,
            "backup/div_cross_question": float(cross.mean()) if cross.numel() else 0.0,
            "backup/div_matched": float(matched.mean()),
            "backup/terms": float(div.numel()),          # R*C, ~60k -- NOT R
        }
    return loss, info


def backup_target(cfg: Config) -> float:
    """`Dist = Next - log gamma` at the minimum: what diagnostic #2 compares against."""
    return cfg.neg_log_gamma
