"""Assemble the phase-1 loss (§7.0).

    L = lambda_NCE*L_NCE + lambda_I*L_I + zeta*L_T + lambda_CF*L_CF + lambda_step*L_step
        + lambda_good*L_good + lambda_term*L_term

**zeta weights the BACKUP ONLY.** `tmd.py:124` is
`contrastive_loss + action_invariance_loss + zeta * backup_loss` -- action invariance sits at
1.0. `get_config`'s own comment (`tmd.py:362`) claims zeta weights invariance too; the code
wins. The old project used `L_NCE + zeta*(L_I + L_T)` with zeta=0.1, making L_I 10x weaker
than TMD's own setting, and its L_I residual never got below 0.43.

Every weight here is UNVALIDATED (§16.8): these terms were never designed to be additive and
their gradient magnitudes are uncharacterised. Every curve is logged separately from step 1.

(6) `L_good` is the §7.12 addition and it ships ON at `lambda_good = 1.0` (signed off
2026-07-28, §16.21), ramped over `good_loss.warmup_steps`. The gauge for it is
`invariance/residual_diagonal`, NOT `probe03/gap`: its cost is paid by (2) L_I (measured
0.098 -> 0.260 at lambda 1.0) while probe03/gap RISES with lambda_good. `--set
losses.lambda_good=0.0` makes it inert -- the term is still computed and logged every step,
because `good/above_target_fraction` is a diagnostic worth having whether or not it is being
trained, and `0.0 * L_good` is an exact zero.

(7) `L_term` is the §7.13 / §16.26 addition and it ships **OFF at `lambda_term = 0.0`**. It is
computed unconditionally for the same reason (6) is: `term/within_question_terminal_spread` is
the statistic §16.26 says decides whether the term is worth turning on, and it is worth
plotting per step whether or not the term is being trained. `L_CF` is the one term still
gated -- its DATA is deferred, so there is nothing to compute when `cf is None`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from ..config import Config
from ..data.collate import Batch
from ..model.distances import Distance
from .counterfactual import _empty_info as _empty_cf_info
from .counterfactual import counterfactual_loss
from .good import good_loss
from .invariance import invariance_loss
from .matrix import Matrices
from .nce import nce_loss
from .step import step_loss
from .temporal import temporal_loss
from .terminal_class import terminal_class_loss


@dataclass
class Phase1Loss:
    total: Tensor
    terms: dict[str, Tensor] = field(default_factory=dict)
    info: dict[str, float] = field(default_factory=dict)


def good_warmup_scale(cfg: Config, step: Optional[int]) -> float:
    """Linear ramp on lambda_good over the first `warmup_steps` optimizer steps (§7.12).

    **relu is the reason this exists.** Every other term tapers as it approaches its target;
    relu applies its FULL gradient the whole time it is violated and then stops dead. So
    L_good hits at full strength from step 1 -- during the ~100 steps when L_I has not yet
    closed the psi/phi gap and L_T is still riding the LINEX LINEAR branch (§7.4.3), i.e.
    when the ruler L_good's target `c` is expressed in does not exist yet.

    The measured cost of L_good is paid by L_I (0.098 -> 0.260 at lambda 1.0, §7.12), and the
    first 100 steps are exactly when L_I is largest and least able to absorb it. Ramping
    removes that overlap for ~7% of a 1,464-step run.

    `step=None` means "no schedule" -- tests and the init probe see the full weight.
    """
    warmup = cfg.losses.good_loss.warmup_steps
    if step is None or warmup <= 0:
        return 1.0
    return min(1.0, step / warmup)


def phase1_loss(
    psi: Tensor,
    phi: Tensor,
    batch: Batch,
    matrices: Matrices,
    distance: Distance,
    cfg: Config,
    goal_traj: Optional[Tensor] = None,
    cf: Optional[tuple[Tensor, Tensor, Tensor]] = None,
    step: Optional[int] = None,
) -> Phase1Loss:
    """`cf` is (phi_variants, variant_example, variant_kind) or None.

    `step` is the completed optimizer-step count, used ONLY for (6) L_good's warmup ramp
    (§7.12). None disables the ramp, which is what tests and the init probe want.
    """
    losses = cfg.losses

    l_nce, nce_info = nce_loss(
        matrices.Dist,
        matrices.pos_row,
        temperature=losses.nce_temperature,
        mask_same_traj=cfg.sampling.nce_mask_same_traj,
        mask_nearer_same_traj=cfg.sampling.nce_mask_nearer_same_traj,
        mask_same_question_correct=cfg.sampling.nce_mask_same_question_correct,
        mask_sibling_correct_late=cfg.sampling.nce_mask_sibling_correct_late,
        sibling_late_margin=cfg.sampling.nce_sibling_late_margin,
        row_traj=batch.row_traj,
        goal_traj=goal_traj,
        row_step=batch.row_step,
        goal_step=matrices.goal_step,
        goal_is_terminal=matrices.goal_is_terminal,
        # steps remaining to this row's OWN terminal, for the sibling-late mask (§9.9.5)
        row_steps_to_end=batch.traj_T[batch.row_traj] - batch.row_step,
        row_correct=batch.traj_correct[batch.row_traj],
        SQ=matrices.SQ,
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
    # (6) L_good reads D_term_good, which IS D_term unless good_loss.detach_goal (§7.12).
    # Computed unconditionally: at lambda_good = 0 it contributes an exact zero and its
    # diagnostics are the ones that decide whether to turn it on.
    l_good, good_info = good_loss(matrices.D_term_good, batch, matrices.terminal_traj, cfg)
    # The ramp scales the WEIGHT, never the term: `terms["good"]` and every good/* diagnostic
    # stay unscaled, so the §18 sandwich and the probe panel read the same quantity at every
    # step of the warmup.
    lambda_good_eff = losses.lambda_good * good_warmup_scale(cfg, step)
    good_info["good/lambda_effective"] = lambda_good_eff

    # (7) L_term reads the terminal psi -- the same tensor D_term's columns come from, not a
    # fresh head, and no extra LM forward (§7.13). Computed at every lambda_term for the same
    # reason (6) is: term/within_question_terminal_spread is §16.26's own gauge.
    # `tau` is passed EXPLICITLY and comes from the config: §7.13 requires whoever raises
    # `lambda_term` to pick it in the same change, and a defaulted argument is a pick nobody
    # made. It stays 1.0 -- (7)'s denominator is ~4.9 candidates, so sqrt(512) would flatten
    # the softmax onto log(c-1+w) and act as an off switch rather than a temperature.
    l_term, term_info = terminal_class_loss(
        psi, batch, distance, temperature=losses.lambda_term_temperature
    )

    if cf is not None and losses.lambda_cf > 0:
        # `tau` is passed EXPLICITLY and comes from the config, exactly as (7)'s does.
        # It was passed NOWHERE before 2026-08-18 -- this call inherited
        # counterfactual_loss's 1.0 default, which is how (4) entered a gradient at a
        # temperature nobody picked. See config/default.yaml for why it is 0.1.
        l_cf, cf_info = counterfactual_loss(
            *cf, distance=distance, temperature=losses.lambda_cf_temperature
        )
    else:
        # The FULL key set, not just `cf/loss`. `counterfactual.py:_empty_info` exists
        # because a diagnostic that vanishes on a degenerate batch cannot be plotted, and
        # this branch used to defeat it -- at lambda_cf = 0, or on a batch where nothing
        # attached, every other cf/* key disappeared from metrics.jsonl mid-run.
        l_cf, cf_info = matrices.Dist.sum() * 0.0, _empty_cf_info()

    total = (
        losses.lambda_nce * l_nce
        + losses.lambda_i * l_i
        + losses.zeta * l_t          # zeta on the backup ONLY (tmd.py:124)
        + losses.lambda_cf * l_cf
        + losses.lambda_step * l_step
        + lambda_good_eff * l_good        # §7.12, ramped over warmup_steps
        + losses.lambda_term * l_term     # §7.13, 0.0 by default -- an exact zero
    )

    info: dict[str, float] = {"loss/total": float(total.detach())}
    for part in (nce_info, inv_info, backup_info, step_info, good_info, cf_info, term_info):
        info.update(part)
    return Phase1Loss(
        total=total,
        terms={
            "nce": l_nce,
            "invariance": l_i,
            "backup": l_t,
            "cf": l_cf,
            "step": l_step,
            "good": l_good,
            "term": l_term,
        },
        info=info,
    )


def expected_init_values(
    cfg: Config,
    n_rows: int,
    mean_dist: Optional[float] = None,
    mean_delta: Optional[float] = None,
    step_delta: float = 0.0,
    linear_fraction: Optional[float] = None,
    term_chance: Optional[float] = None,
) -> dict[str, float]:
    """§18's initialisation probe. Compute these; do not eyeball them.

    * `nce`  -- chance is log(R): ~5.85 at R = 348. Pinned there WITH `logit_std ~= 0` is bug
      B10a; pinned there with `logit_std > 0` and pos ~= neg is geometry collapse (§14).
    * `step` -- `log(1 + exp(m - Delta))`, i.e. ln 5 = 1.6094 at discount 0.5 / margin_steps 2
      **only where `Delta_{z+1} = 0`.** That is exact on a fixture (tests/test_step_loss.py) and
      it is the default here, but it is NOT what the real model gives at init: `psi` is a random
      map over anisotropic LM hiddens and `h_{s_0}` -- the prompt-only state, which is `psi_z`
      for the 45.4% of trajectories with z = 0 (§4.2.1) -- sits well away from mid-solution
      hiddens, so `Delta_{z+1}` starts NEGATIVE and L_step starts above 1.61. **How far above
      depends on the batch's z mix** and is not a constant: an all-z=0 fixture gives
      `Delta = -2.86` and `L_step = 4.26`, while the first real training batch -- mixed z, real
      prompts -- measured `Delta = -0.44` and `L_step = 2.08` (2026-07-27). Pass the measured
      `step/delta_mean` as `step_delta`; the prediction was 1.98 against that 2.08, the gap
      being Jensen (softplus is convex, so mean(f) >= f(mean)). The exactness claim belongs to the margin arithmetic, not
      to the initialisation value: what pins `m` and the `z` indexing is
      tests/test_step_loss.py.
    * `backup` -- **`delta` is NOT ~0 at init, and assuming it was is what made this probe
      predict -10.53 against an actual of +8760.29.** psi and phi are independent networks, so
      `Dist` starts at the unrelated-latent value ~11 while `Next` is psi against itself on
      anisotropic LM hidden states and collapses to ~1.3: `delta ~ 9.8` (§7.4.3). Pass BOTH
      `mean_dist` (`backup/dist_mean`) and `mean_delta` (`backup/delta_mean`) and the
      prediction follows whichever LINEX branch `delta` actually selects. With `delta > t` --
      which is the expected state at step 0 -- that is just `delta` itself, a number of order
      10, and `linear_branch_fraction` should read ~1.0 alongside it.

      Only once L_I has closed the psi/phi gap does the exponential branch take over and the
      old `gamma - mean_dist` reading apply. What matters either way is that it FALLS and does
      not plateau or NaN (§7.4).

    * `invariance` -- no closed form for the *trained* value, but at init it is the same
      unrelated-latent distance `Dist` starts at, so `L_I ~= backup/dist_mean ~ 11`. L_I
      arriving near 11 is the geometry working as expected, not a fault.

    * `good` -- **deliberately `nan`. There is no level to predict** (§7.12, §18). L_good is
      `mean f(Delta - c)` over ~600 good transitions against a *negative* `c`, so its init
      value is a property of the random psi's Delta distribution and nothing else. What IS
      checkable is the sandwich, and note it runs OPPOSITE to L_step's because every `f`
      (`relu`, `relu_squared`, `softplus`) is INCREASING in Delta:

          f(good/delta_min - c)  <=  L_good  <=  f(good/delta_max - c)

      `losses.good.good_bounds` applies the configured form; do not reimplement it with a
      hardcoded `relu`, which would fire on a correct `relu_squared` run.

      **A lower bound of exactly 0 is legitimate** -- it is what "every good step already sits
      at or below target" looks like, not a dead term. Predicting a level here and then
      "correcting" the code to hit it is the §7.4.3 / §7.6.7 mistake for the third time.

    * `term` -- chance, exactly as `nce` is, but chance here is **not a constant**: a query of
      question `q` sees `(c_q - 1)` positives and `w_q` negatives in one softmax, so a flat
      `d` gives `mean_q log(c_q - 1 + w_q)` over the questions with `c_q >= 2`, and the
      `min(4,k_c)/min(3,k_i)` counts are ragged (§8.1). `terminal_class_loss` computes it on
      the batch and logs it as `term/chance`; pass that value here. `L_term` sitting AT it is
      the correct reading at step 0 and says nothing is wrong -- the term ships at
      `lambda_term = 0.0` (§7.13), so it is a diagnostic and is expected to stay near chance
      for the whole run unless someone turns it on.
    """
    import math

    if mean_delta is None:
        backup = cfg.discount if mean_dist is None else cfg.discount - mean_dist
    elif linear_fraction is not None and 0.05 < linear_fraction < 0.95:
        # **Both LINEX branches are populated, so no mean-based prediction is possible.**
        # Measured on the first real run: `mean_delta = 9.12` with `linear_fraction = 0.52`,
        # and this returned 9.12 against an actual 3.55. The mean is dominated by the terms
        # ABOVE the clip while the terms below it sit far under `t` -- delta is bimodal at
        # init, so `f(mean)` is not `mean(f)` by a factor of ~3. Return nan rather than a
        # confident wrong number; the checks that matter (finite, falling, and
        # `linear_branch_fraction -> 0` within ~100 steps) are logged either way.
        backup = float("nan")
    elif mean_delta > cfg.clip_t:
        backup = mean_delta  # linear branch: div = delta
    else:
        backup = cfg.discount * math.exp(mean_delta) - (mean_dist or 0.0)

    return {
        "nce": math.log(n_rows) if n_rows > 0 else float("nan"),
        "step": math.log1p(math.exp(cfg.step_margin - step_delta)),
        "backup": backup,
        "invariance": mean_dist if mean_dist is not None else float("nan"),
        # lambda_cf is 0 until the counterfactual data exists (§2 #4), so the term is a
        # structural zero, not a prediction. Listed so the probe's printout has no nan in it.
        "cf": 0.0,
        # NOT a structural zero -- L_good is computed at every lambda_good, including 0.0.
        # nan means "no prediction is possible", and the probe asserts the sandwich instead.
        "good": float("nan"),
        # Same: computed at every lambda_term, including the shipped 0.0 (§7.13). The
        # prediction is the batch's own chance level, which only the batch knows.
        "term": term_chance if term_chance is not None else float("nan"),
    }
