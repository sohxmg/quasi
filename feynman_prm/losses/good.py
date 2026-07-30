"""(6) L_good -- the false-positive loss on GOOD steps (§7.12, added 2026-07-28).

    L_good = mean over good transitions of  relu( Delta_i - c )
    c      = -margin_steps * (-log gamma) = -0.69315 at discount 0.5, margin_steps 1.0
             **NEGATIVE.** It is the target Delta of a good step, not a slack allowance.

**Why this term exists.** Eval reads `Delta_i = d(psi_i, g) - d(psi_{i-1}, g)` and a good step
is supposed to sit at `-log gamma = -0.693`. At step 750 of the first full run the measured
mean over good steps of CORRECT trajectories was **+0.240**, half of them positive, with
`P(Delta > 0.347) = 0.34`. tau then has to climb to 2.39 to dodge the false alarms, TPR dies,
and F1 caps at 0.456.

The reason is an exact identity, not a training accident:

    Delta_i = -(-log gamma) - A_i - B_i
    A_i     = d(psi_{i-1}, g) - d(phi_i, g)                 L_I slack, goal-projected
    B_i     = d(phi_i, g) - d(psi_i, g) - (-log gamma)      L_T Bellman residual

`L_I` drives `d(psi_{i-1}, phi_i) -> 0`, and at that optimum the MRN's own geometry forces
`d(phi_i, g) >= d(psi_{i-1}, g)`, i.e. `-A_i >= 0` **with no upper bound**. So the loss set
had a one-sided hole: nothing constrained a good step's Delta FROM ABOVE. Confirmed
structural on a free-latent probe -- `L_I` at 0.0024 and the positive tail still there.

**Scope** (§6.1's index convention, `matrix.step_deltas`):

    correct trajectories     every i          trained
    incorrect trajectories   i <= z           trained    (still on-track)
    incorrect trajectories   i == z+1         EXCLUDED   -- (5) L_step's pair, opposite sign
    incorrect trajectories   i >  z+1         EXCLUDED   -- no loss defines Delta there

**relu, not softplus, and this is not a style preference.** `Delta >= -0.693` is a hard floor
implied by `L_I` + `L_T`, so pushing below `c` costs one of those two terms. `softplus` still
applies half its gradient AT the target and never reaches zero, so it keeps pushing:

    form       P(Delta > tau)   ruler   -B      Delta mean
    base            0.229       1.201   -0.97    +0.240     (no L_good)
    softplus        0.000       1.283   -1.42    -1.556     overshoots, breaks L_T
    relu            0.000       1.222   -0.95    -0.834     clean

(SIMULATED ON FREE LATENTS -- see §7.12. Same seed and data across the three rows, but not
measured on the model.) `relu` stops at `c`; that is the whole point of choosing it.

**This is the one place in the loss set where the sandwich runs the other way.** `relu` is
INCREASING in Delta, so `relu(delta_min - c) <= L_good <= relu(delta_max - c)`, and the lower
bound is legitimately 0 whenever every good step already sits at or below target. Do not
predict a level for it at init (§18).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import Config
from ..data.collate import Batch
from .matrix import step_deltas


def _empty(delta: Tensor, cfg: Config) -> tuple[Tensor, dict[str, float]]:
    """Masked, not crashed: no correct terminal in the batch (no goal column), or no good
    transition against one. Keeps a zero in the graph so backward still works."""
    zero = delta.sum() * 0.0
    return zero, {
        "good/loss": 0.0,
        "good/terms": 0.0,
        "good/delta_mean": 0.0,
        "good/delta_min": 0.0,
        "good/delta_max": 0.0,
        "good/margin": cfg.good_margin,
        "good/above_target_fraction": 0.0,
    }


def good_loss(
    D_term: Tensor,
    batch: Batch,
    terminal_traj: Tensor,
    cfg: Config,
) -> tuple[Tensor, dict[str, float]]:
    """Returns (loss, diagnostics).

    `D_term` is `Matrices.D_term_good` -- the same tensor as `Matrices.D_term` unless
    `good_loss.detach_goal` is set, in which case the terminal side is stop-gradded.
    """
    c = cfg.good_margin                       # NEGATIVE. -1 * (-log gamma) by default.
    form = cfg.losses.good_loss.form

    deltas = step_deltas(D_term, batch, terminal_traj)
    mask = deltas.good(cfg.losses.good_loss.include_incorrect_prefix)
    delta = deltas.delta[mask]
    if delta.numel() == 0:
        return _empty(deltas.delta, cfg)

    excess = delta - c
    per_term = F.relu(excess) if form == "relu" else F.softplus(excess)
    loss = per_term.mean()

    with torch.no_grad():
        info = {
            "good/loss": float(loss),
            # R x T_c restricted to same-question, other-trajectory terminals: ~600 at the
            # §8.1.1 layout, an order of magnitude over (5) L_step's ~64 pairs. That asymmetry
            # is NOT a drowning risk: both losses are means, so 10x the terms makes each term
            # 10x weaker rather than the sum 10x stronger, and psi_{z+1} -- the state detection
            # rides on -- is outside this mask entirely. Measured, probe03/gap RISES with
            # lambda_good (§7.12). The gauge for this term is invariance/residual_diagonal.
            # Read THIS number rather than the estimate.
            "good/terms": float(delta.numel()),
            "good/delta_mean": float(delta.mean()),
            # The sandwich is INCREASING in Delta (relu), the opposite of L_step's:
            #   relu(delta_min - c) <= L_good <= relu(delta_max - c)
            # and a lower bound of exactly 0 is a legitimate reading, not a dead term (§18).
            "good/delta_min": float(delta.min()),
            "good/delta_max": float(delta.max()),
            "good/margin": c,
            # THE number this term exists to move. It is not the mean: at step 750 the mean
            # was +0.240 while a THIRD of good steps sat above the natural tau, and it is the
            # tail that sets where tau has to go (§7.12). probe14's quantiles are the finer
            # version of this line.
            "good/above_target_fraction": float((delta > c).float().mean()),
        }
    return loss, info
