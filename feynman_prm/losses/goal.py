"""L_goal -- the goal head, PHASE 2 ONLY (§7.7, locked #13/#14/#15).

    pred_q    = goal_head(h_{s_0}^q)
    targets_q = { psi(s_T^c) : c a correct trajectory of q }        FROZEN

    L_goal = mean over c of [ d(pred_q, psi(s_T^c)) + d(psi(s_T^c), pred_q) ]

The mean OF DISTANCES, never a distance to a mean -- a latent-space centroid over 30k
terminals collapses onto the population mean and the distance becomes an atypicality
detector. That is old root cause D and there is no centroid anywhere in this codebase.

**Both directions, summed.** The distance is one-way, so a single direction would let the
guess drift to somewhere *reachable from* the real ending rather than *being* it.

`.detach()` is structural here: phase 1 has no goal head at all, and phase 2 freezes the
backbone, psi and phi, so the degenerate joint optimum (psi squashes every ending onto one
point, the head predicts that point, loss -> 0, every state identical) is unreachable. The
collapse test in §15 is kept anyway -- it documents why the phase split exists.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..model.distances import Distance


def goal_loss(
    pred: Tensor,
    targets: Tensor,
    target_example: Tensor,
    distance: Distance,
) -> tuple[Tensor, dict[str, float]]:
    """`pred` (N, D) one prediction per question; `targets` (M, D) frozen terminals;
    `target_example` (M,) which question each terminal belongs to."""
    p = pred.index_select(0, target_example)
    forward = distance(p, targets)
    backward = distance(targets, p)
    loss = (forward + backward).mean()

    with torch.no_grad():
        info = {
            "goal/loss": float(loss),
            "goal/d_pred_to_target": float(forward.mean()),
            "goal/d_target_to_pred": float(backward.mean()),
            # Diagnostic #6: near-zero variance means the head learned a constant, i.e. a
            # global anchor rather than a question-conditioned goal.
            "goal/pred_variance": float(pred.var(dim=0).mean()) if pred.shape[0] > 1 else 0.0,
        }
    return loss, info


def terminal_spread_ratio(
    psi_terminals: Tensor, question_index: Tensor, distance: Distance
) -> dict[str, float]:
    """§10.1's go/no-go gate (diagnostic #5). Run it on a SHORT run, before GPU hours.

        within = mean over q of mean_{j!=k} d(psi(s_T^j), psi(s_T^k))    same question
        across = mean over q!=q' of d(psi(s_T^q), psi(s_T^q'))
        ratio  = within / across

    ratio < ~0.3  -> correct endings cluster by question, the goal head has a well-defined
                     target, proceed.
    ratio -> 1    -> endings do not cluster and the goal head CANNOT work however well it is
                     trained. Fall back to the goal-free asymmetry score (§9.4) -- NOT to a
                     reference goal, which is a skyline (§5.1).
    """
    with torch.no_grad():
        d = distance(psi_terminals[:, None, :], psi_terminals[None, :, :])
        same = question_index[:, None] == question_index[None, :]
        eye = torch.eye(len(psi_terminals), dtype=torch.bool, device=d.device)
        within_mask = same & ~eye
        across_mask = ~same
        within = float(d[within_mask].mean()) if within_mask.any() else float("nan")
        across = float(d[across_mask].mean()) if across_mask.any() else float("nan")
    return {
        "gate/within_question_terminal_spread": within,
        "gate/across_question_terminal_spread": across,
        "gate/ratio": within / across if across else float("nan"),
    }
