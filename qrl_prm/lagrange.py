"""QRL's dual variables. Ported from
`quasimetric-rl/quasimetric_rl/modules/utils.py:162-199` (`softplus_inv_float`, `GradMul`,
`grad_mul`) and `.../quasimetric_critic/losses/local_constraint.py:48-63` (how they combine).

**What a Lagrange multiplier is doing here, and why it is not a lambda.** Feynman-PRM's loss
set weights each term with a fixed number chosen by hand; §16.8 says plainly that those
weights are unvalidated and that the terms were never designed to be additive. QRL removes
the choice: the objective is *maximize distances*, and each thing we know must stay small is
a CONSTRAINT with its own multiplier, trained by gradient ASCENT on the same scalar the
primal descends. The multiplier finds its own weight, and it is a readable diagnostic while
it does -- a multiplier that climbs without its violation falling means the constraint cannot
be satisfied, which for the CF constraint means the CF data contradicts itself.

Three mechanics, all of them load-bearing:

* **softplus parameterisation.** The trained scalar is `raw`, and the multiplier is
  `softplus(raw)`, so it is positive by construction. An unconstrained multiplier could go
  negative and silently invert the constraint into a reward for violating it.
* **`grad_mul(x, -1)`.** Identity in the FORWARD pass, negation in the BACKWARD pass. This is
  what makes one `loss.backward()` do minimax: the primal parameters see
  `+lambda * violation` and descend it, while `raw` sees `-violation` and therefore ASCENDS
  the violation -- exactly `max_lambda min_theta`. Writing `-lambda * violation` instead
  would flip the primal's sign too, which trains the model to violate its own constraints.
* **`softplus_inv_float` at init.** `init_lagrange_*` is stated as the MULTIPLIER's value, so
  the raw scalar has to be its inverse-softplus. Storing 0.01 raw would start the multiplier
  at `softplus(0.01) = 0.698`, 70x the intended value, and nothing downstream would say so.

The multipliers deliberately live OUTSIDE `FeynmanPRM`. `model/backbone.py:300`'s
`param_groups` sweeps every trainable non-LoRA parameter into the "heads" group at
`lr_heads = 3e-4` on the cosine schedule -- so a multiplier registered on the model would
join the primal optimiser, ride the primal's LR decay, and be updated in the WRONG direction
by an optimiser that never saw the sign flip. They get their own AdamW at a constant
`lagrange_lr` (QRL's `losses/__init__.py:47`), which is also why `assert_phase1_trainable`
still passes unchanged: `qrl_prm` adds no parameter to the model at all.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def softplus_inv_float(y: float) -> float:
    """`utils.py:162-168`, verbatim including the threshold.

    `softplus` is numerically the identity above 20 (torch's own documented threshold), so
    its inverse is too; below that it is `log(expm1(y))`.
    """
    threshold: float = 20.0
    if y > threshold:
        return y
    return float(np.log(np.expm1(y)))


class GradMul(torch.autograd.Function):
    """`utils.py:176-193`, verbatim. Identity forward, `mult`-scaled backward."""

    @staticmethod
    def forward(ctx, x: Tensor, mult: Union[float, Tensor]) -> Tensor:
        ctx.mult_is_tensor = isinstance(mult, torch.Tensor)
        if ctx.mult_is_tensor:
            assert not mult.requires_grad
            ctx.save_for_backward(mult)
        else:
            ctx.mult = mult
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.mult_is_tensor:
            (mult,) = ctx.saved_tensors
        else:
            mult = ctx.mult
        return grad_output * mult, None


def grad_mul(x: Tensor, mult: Union[float, Tensor]) -> Tensor:
    """`utils.py:196-199`, verbatim. `mult == 0` short-circuits to a detach."""
    if not isinstance(mult, torch.Tensor) and mult == 0:
        return x.detach()
    return GradMul.apply(x, mult)


class LagrangeMultiplier(nn.Module):
    """One dual variable: `softplus(raw)`, gradient-reversed on the way back.

    `forward(violation)` returns the term that goes into the total loss. The violation must
    already be `E[penalty] - epsilon^2` -- the sign of that scalar is what the multiplier
    integrates, so a caller that forgets to subtract the target trains a multiplier that can
    only ever climb.
    """

    def __init__(self, init_value: float = 0.01):
        super().__init__()
        if init_value <= 0.0:
            raise ValueError(
                f"init_value must be > 0, got {init_value}: softplus_inv is undefined at 0 "
                "and a multiplier pinned at exactly 0 has no gradient path into the primal."
            )
        self.init_value = float(init_value)
        self.raw = nn.Parameter(
            torch.tensor(softplus_inv_float(init_value), dtype=torch.float32)
        )

    @property
    def value(self) -> Tensor:
        """The multiplier itself, WITH grad. Use `.detach()` for logging -- `float()` on a
        requires_grad tensor warns, and this is read every log step."""
        return F.softplus(self.raw)

    def forward(self, violation: Tensor) -> Tensor:
        # grad_mul(-1) is the whole minimax mechanism -- see the module docstring.
        return grad_mul(self.value, -1.0) * violation

    def extra_repr(self) -> str:
        return f"init_value={self.init_value:g}, value={float(self.value.detach()):.6f}"


class LagrangeMultipliers(nn.Module):
    """The three dual variables of this run, in one module so the second optimiser is built
    from `.parameters()` and cannot miss one.

    **Three, not two, and the third is a SPLIT rather than an addition.** `local` is the
    observed-transition constraint at k = 1 (`losses/local_constraint.py`, unchanged) and
    `path` is the observed-sub-path constraint at `2 <= k <= path_max_gap`. The two pair sets
    are DISJOINT, so nothing is counted twice -- which is what makes two multipliers legal
    here. A `path` set that included k = 1 would be the same constraint under two multipliers
    whose ratio nothing determines, and that is the merge this split undoes.

    **Why they were split back apart.** Both terms are means of squared deviations, and the
    constraint is ONE-SIDED, so a mean divides by pairs that contribute exactly zero. Measured
    at step 1 of probe `0lcrduzl` against `1wbpyf2g` on the identical seed-42 batch: the 489
    adjacent pairs carried 969.4 of the 1,195.7 total squared deviation (81%) and received
    489/3608 = 13.5% of the weight. `local_dist_mean` went 2.263 -> 3.009 over 20 steps where
    the same multiplier on the k = 1 constraint alone had taken it 2.263 -> 1.390. The
    "N cancels" argument for merging is right at EQUILIBRIUM -- `(2N/S) * 2*lambda*eps / N` --
    and wrong in the TRANSIENT, where slack pairs are 0 in the numerator and full weight in N.

    `state_dict()` / `load_state_dict()` are what the checkpoint payload carries: a resumed
    run that restarted its multipliers at their init values would spend the first ~100 steps
    re-climbing to wherever they had settled, on a primal that is already trained -- a
    transient that looks exactly like a bad resume and is not one.

    **`raw_values()` has changed key set twice now: `{local, cf}` -> `{path, cf}` -> `{local,
    path, cf}`.** A resume path must FAIL on either older key set rather than fall back to it.
    The dangerous one is the middle spelling, because it uses a name this class still holds
    while meaning something else: `path` was the equilibrium of a mean over EVERY
    same-trajectory pair, and `path` here is the equilibrium of a mean over `k >= 2` only,
    under a `local` that now takes the k = 1 rows. The two are numerically close by the
    N-cancels argument above, which is exactly what would make a silent fallback hard to spot.

    The three inits are SEPARATE arguments and there is no shared default, which is deliberate.
    QRL starts all of its multipliers at 0.01 (`local_constraint.py:31`) because it trains for
    2e5 steps; at the ~1,464 this run gets, a multiplier cannot reach its equilibrium in time
    and the primal spends the first 40% of training under a push term with nothing opposing it.
    `qrl_prm/config.py` carries the measurement behind each of the three values.
    """

    def __init__(self, init_local: float, init_path: float, init_cf: float):
        super().__init__()
        self.local = LagrangeMultiplier(init_local)
        self.path = LagrangeMultiplier(init_path)
        self.cf = LagrangeMultiplier(init_cf)

    def values(self) -> dict[str, float]:
        return {
            "qrl/lagrange_local": float(self.local.value.detach()),
            "qrl/lagrange_path": float(self.path.value.detach()),
            "qrl/lagrange_cf": float(self.cf.value.detach()),
        }

    def raw_values(self) -> dict[str, float]:
        """The RAW scalars, for the checkpoint payload. Saved raw rather than post-softplus
        so a reload is exact rather than round-tripped through `softplus_inv`."""
        return {
            "local": float(self.local.raw.detach()),
            "path": float(self.path.raw.detach()),
            "cf": float(self.cf.raw.detach()),
        }
