"""Assemble the phase-1 loss (§7.0).

    L = lambda_NCE*L_NCE + lambda_I*L_I + zeta*L_T + lambda_CF*L_CF + lambda_step*L_step

**zeta weights the BACKUP ONLY.** `tmd.py:124` is
`contrastive_loss + action_invariance_loss + zeta * backup_loss` -- action invariance sits at
1.0. `get_config`'s own comment (`tmd.py:362`) claims zeta weights invariance too; the code
wins. The old project used `L_NCE + zeta*(L_I + L_T)` with zeta=0.1, making L_I 10x weaker
than TMD's own setting, and its L_I residual never got below 0.43.

Every weight here is UNVALIDATED (§16.8): these terms were never designed to be additive and
their gradient magnitudes are uncharacterised. Every curve is logged separately from step 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from ..config import Config
from ..data.collate import Batch
from ..model.distances import Distance
from .counterfactual import counterfactual_loss
from .invariance import invariance_loss
from .matrix import Matrices
from .nce import nce_loss
from .step import step_loss
from .temporal import temporal_loss


@dataclass
class Phase1Loss:
    total: Tensor
    terms: dict[str, Tensor] = field(default_factory=dict)
    info: dict[str, float] = field(default_factory=dict)


def phase1_loss(
    psi: Tensor,
    phi: Tensor,
    batch: Batch,
    matrices: Matrices,
    distance: Distance,
    cfg: Config,
    goal_traj: Optional[Tensor] = None,
    cf: Optional[tuple[Tensor, Tensor, Tensor]] = None,
) -> Phase1Loss:
    """`cf` is (phi_variants, variant_example, variant_kind) or None."""
    losses = cfg.losses

    l_nce, nce_info = nce_loss(
        matrices.Dist,
        matrices.pos_row,
        temperature=losses.nce_temperature,
        mask_same_traj=cfg.sampling.nce_mask_same_traj,
        row_traj=batch.row_traj,
        goal_traj=goal_traj,
    )
    l_i, inv_info = invariance_loss(
        psi,
        phi,
        batch.row_src,
        batch.traj_qid[batch.row_traj],
        distance,
        mode=losses.action_invariance.mode,
        stopgrad_phi=losses.stopgrad_phi_invariance,
    )
    l_t, backup_info = temporal_loss(
        matrices.Dist_backup, matrices.Next, matrices.pos_row, matrices.SQ, cfg
    )
    l_step, step_info = step_loss(matrices.D_term, batch, matrices.terminal_traj, cfg)

    if cf is not None and losses.lambda_cf > 0:
        l_cf, cf_info = counterfactual_loss(*cf, distance=distance)
    else:
        l_cf, cf_info = matrices.Dist.sum() * 0.0, {"cf/loss": 0.0}

    total = (
        losses.lambda_nce * l_nce
        + losses.lambda_i * l_i
        + losses.zeta * l_t          # zeta on the backup ONLY (tmd.py:124)
        + losses.lambda_cf * l_cf
        + losses.lambda_step * l_step
    )

    info: dict[str, float] = {"loss/total": float(total)}
    for part in (nce_info, inv_info, backup_info, step_info, cf_info):
        info.update(part)
    return Phase1Loss(
        total=total,
        terms={"nce": l_nce, "invariance": l_i, "backup": l_t, "cf": l_cf, "step": l_step},
        info=info,
    )


def expected_init_values(
    cfg: Config, n_rows: int, mean_dist: Optional[float] = None
) -> dict[str, float]:
    """§18's initialisation probe. Compute these; do not eyeball them.

    * `nce`  -- chance is log(R): ~5.85 at R = 348. Pinned there WITH `logit_std ~= 0` is bug
      B10a; pinned there with `logit_std > 0` and pos ~= neg is geometry collapse (§14).
    * `step` -- EXACT: Delta_{z+1} ~= 0 at init, so `-log sigma(-m) = log(1 + e^m)` = ln 5 =
      1.6094 at discount 0.5 / margin_steps 2, and 1.1120 at the 0.7 fallback. If it is not
      1.609 the margin or the z indexing is wrong.
    * `backup` -- at init `delta ~= 0`, so `div ~= gamma - Dist`. **§18 states this as
      "starts positive, ~= gamma", which holds only while the mean distance is below gamma.**
      Xavier-initialised heads at latent 512 produce distances of O(1), so a NEGATIVE L_T at
      step 1 is not by itself evidence the backup is broken. Pass `mean_dist`
      (`backup/dist_mean`) and the prediction becomes `gamma - mean_dist`, which is the
      quantity to check. What matters either way is that it FALLS and does not plateau or
      NaN (§7.4).
    """
    import math

    return {
        "nce": math.log(n_rows) if n_rows > 0 else float("nan"),
        "step": math.log1p(math.exp(cfg.step_margin)),
        "backup": cfg.discount if mean_dist is None else cfg.discount - mean_dist,
        "invariance": float("nan"),  # depends on the random head init, no closed form
    }
