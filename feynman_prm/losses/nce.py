"""(1) L_NCE -- the contrastive loss (§7.2).

    logits[r, c] = -Dist[r, c] / tau
    L_NCE = cross-entropy over SOURCE ROWS within each goal column, at r = pos_row[c]

**Softmax over sources per goal, not over goals per source.** That is TMD's *backward* NCE:
`tmd.py:96-98` applies the cross-entropy to `logits.T`, so optax normalises over sources
within each goal column. The spec's own annotation fixes the same direction -- negatives
"keep the goal, change (s_i, a_i)", drawn from the same trajectory at other states and from
other trajectories, "correct or incorrect soln". That the negative pool explicitly includes
incorrect solutions is a correctness signal already present in the spec.

`tau = 1.0` is a DOCUMENTED DIVERGENCE from TMD's `1/sqrt(512) = 22.6` (`tmd.py:92`). At
22.6 our O(1-10) distances become O(0.05-0.5) logits, i.e. a near-uniform softmax -- the
exact signature of old bug B10a. It is a float knob: raise it if `logit_std` blows up (§7.2).

At initialisation L_NCE should sit at chance, `log(R) ~= log(348) = 5.85`. Pinned there
**with `logit_std ~= 0`** is bug B10a and means the fp32 cast is not effective at runtime;
pinned there with `logit_std > 0` and `pos ~= neg` is geometry collapse -- check goal
duplication (§14's stuck-NCE table).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def nce_loss(
    Dist: Tensor,
    pos_row: Tensor,
    temperature: float = 1.0,
    mask_same_traj: bool = False,
    row_traj: Tensor | None = None,
    goal_traj: Tensor | None = None,
    SQ: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Returns (loss, diagnostics). `Dist` is the shared (R, C) matrix from matrix.py."""
    R, C = Dist.shape
    logits = -Dist / temperature                       # (R, C)

    if mask_same_traj:
        # locked #12 leaves this OFF: we want the negatives, and masking removes only ~6 of
        # ~347 (§16.4). Flip it to true if diagnostic #7's within-trajectory spread -> 0.
        if row_traj is None or goal_traj is None:
            raise ValueError("nce_mask_same_traj needs row_traj and goal_traj")
        same = row_traj[:, None] == goal_traj[None, :]          # (R, C)
        same[pos_row, torch.arange(C, device=Dist.device)] = False   # never mask the positive
        logits = logits.masked_fill(same, float("-inf"))

    loss = F.cross_entropy(logits.t(), pos_row)        # normalise over rows, per column

    with torch.no_grad():
        cols = torch.arange(C, device=Dist.device)
        finite = torch.isfinite(logits)
        pos_logit = logits[pos_row, cols]
        neg_mask = torch.ones_like(logits, dtype=torch.bool)
        neg_mask[pos_row, cols] = False
        neg_mask &= finite
        predicted = logits.argmax(dim=0)                # per column, over sources
        info = {
            "nce/loss": float(loss),
            "nce/chance": math.log(R),                  # ~5.85 at R = 348
            "nce/logit_std": float(logits[finite].std()),
            "nce/logits_pos": float(pos_logit.mean()),
            "nce/logits_neg": float(logits[neg_mask].mean()),
            "nce/pos_dist": float(Dist[pos_row, cols].mean()),
            "nce/neg_dist": float(Dist[neg_mask].mean()),
            # The direction the LOSS optimises (sources within a column). TMD's own
            # categorical_accuracy measures the forward direction instead (tmd.py:127) while
            # its loss normalises over sources (tmd.py:97); we report what is optimised.
            "nce/categorical_accuracy_backward": float((predicted == pos_row).float().mean()),
            "nce/negatives_per_column": float(R - 1),
        }

        # The same-question split -- the L_NCE counterpart of diagnostic #13, and the only way
        # to read `nce/loss` correctly. The pool is ~R/Q same-question rows plus ~R(1-1/Q)
        # cross-question ones, and those are two different problems: cross-question separation
        # is what L_NCE is for, while same-question rows include states that are GENUINELY
        # closer to the goal than the sampled positive (§16.4 -- goal at s_6, positive at s_3,
        # and s_5 sits between them). So a chunk of the same-question mass is unlearnable by
        # construction, and the loss floors at log(1 + same-question negatives) even with
        # cross-question solved perfectly. Read `loss_cross_question` and
        # `accuracy_within_question` -- NOT `nce/loss` -- to tell "still learning" from
        # "at the floor". A falling `nce/loss` with Q rising is the task getting easier, not
        # the model getting better.
        if SQ is not None:
            same_q_neg = SQ & neg_mask
            cross_q_neg = (~SQ) & neg_mask
            n_same = same_q_neg.sum(dim=0).float().mean()
            info["nce/negatives_same_question"] = float(n_same)
            info["nce/floor_same_question"] = float(torch.log1p(n_same))
            if same_q_neg.any():
                info["nce/logits_neg_same_question"] = float(logits[same_q_neg].mean())
            if cross_q_neg.any():
                info["nce/logits_neg_cross_question"] = float(logits[cross_q_neg].mean())
            # The loss with same-question negatives removed: how much of the residual is
            # cross-question discrimination that is still genuinely unsolved.
            info["nce/loss_cross_question"] = float(
                F.cross_entropy(logits.masked_fill(same_q_neg, float("-inf")).t(), pos_row)
            )
            # Ranking WITHIN the question, which is what L_T's ruler has to supply and what
            # more training can actually move. Chance is 1 / (1 + negatives_same_question).
            within = logits.masked_fill(~SQ, float("-inf"))
            info["nce/accuracy_within_question"] = float(
                (within.argmax(dim=0) == pos_row).float().mean()
            )
    return loss, info
