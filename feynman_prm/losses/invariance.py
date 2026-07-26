"""(2) L_I -- action invariance (§7.3, `tmd.py:100-105`).

    L_I = (1/R) * sum_r  d( psi(s_{row_src[r]}) , phi_r )

**Elementwise, each state against its own action only** -- TMD's own setting, and verified
genuinely elementwise there (shape `(e, B)`, no outer product). Argument order matters:
`d(psi(s), phi(s,a)) = 0` does NOT imply `phi = psi(s)` in a quasimetric; it means
"phi(s,a) is reachable from s at zero cost".

Weight 1.0, NOT under zeta: `tmd.py:124` is
`contrastive + action_invariance + zeta * backup`. The old project put L_I under zeta=0.1,
making it 10x weaker than TMD's own setting, and its residual never got below 0.43 (§7.0).

The two grid modes exist ONLY so §16.3's failure can be reproduced. They drive
`phi(s,a) -> psi(s)`, which tightens the triangle bound to an equality and pins Delta at
`-log gamma` good step or bad -- which breaks (5) L_step outright (§7.6.5). `grid_max_actions`
is hardcoded here because it is unreachable from the default configuration (PLAN 'Config').
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..model.distances import Distance

GRID_MAX_ACTIONS = 64  # caps the two reproduce-the-failure modes only


def invariance_loss(
    psi: Tensor,
    phi: Tensor,
    row_src: Tensor,
    row_question: Tensor,
    distance: Distance,
    mode: str = "diagonal",
    stopgrad_phi: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Returns (loss, diagnostics). `row_question` is the question index of each source row."""
    psi_s = psi.index_select(0, row_src)                       # (R, D)
    phi_used = phi.detach() if stopgrad_phi else phi           # tmd.py:100-103, default False

    if mode == "diagonal":
        residual = distance(psi_s, phi_used)                   # (R,) elementwise
        loss = residual.mean()
    elif mode in ("grid_within_question", "grid_batch"):
        R = phi_used.shape[0]
        if R > GRID_MAX_ACTIONS:
            idx = torch.randperm(R, generator=generator, device=phi_used.device)[:GRID_MAX_ACTIONS]
        else:
            idx = torch.arange(R, device=phi_used.device)
        grid = distance(psi_s[:, None, :], phi_used[idx][None, :, :])     # (R, A)
        if mode == "grid_within_question":
            keep = row_question[:, None] == row_question[idx][None, :]
            loss = (grid * keep).sum() / keep.sum().clamp(min=1)
        else:
            loss = grid.mean()
        residual = distance(psi_s, phi_used)
    else:
        raise ValueError(f"unknown action_invariance mode {mode!r}")

    with torch.no_grad():
        info = {
            "invariance/loss": float(loss),
            # Diagnostic #9: the old project's residual plateaued at 0.43; the target is 0.
            "invariance/residual_diagonal": float(residual.mean()),
        }
    return loss, info
