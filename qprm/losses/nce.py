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
    return loss, info
