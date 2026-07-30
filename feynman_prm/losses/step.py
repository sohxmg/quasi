"""(5) L_step -- the correctness loss, at the first-error boundary (§7.6, locked #3b).

    L_step = -log sigma( d(psi_{z+1}, g) - d(psi_z, g) - m )
    m      = margin_steps * (-log gamma) = 1.38629 at discount 0.5

**z IS 0-BASED (§6.1).** `labels[z]` is the bad step and it takes `s_z -> s_{z+1}`, so
`psi_z` is the LAST GOOD state and `psi_{z+1}` the FIRST BROKEN one. The pair is
`(psi_z, psi_{z+1})`. `psi_{z-1}` is wrong twice over: it is a *good* state, and it does not
exist for the 45.4% of incorrect trajectories with z = 0.

That mistake is invisible in every loss curve -- training converges, Delta separates, and
`predicted_label` comes out `z-1` on EVERY errored sample, so `acc_error -> 0` and F1
collapses through the harmonic mean. tests/test_step_loss.py pins it.

`g` is `psi(s_T)` of a CORRECT trajectory in the same batch -- a real terminal, never a
prediction, never a centroid, never a geometric-sampler column. So the term carrying
correctness queries an ending-like goal 100% of the time at any `discount` (§7.6.3), which
is what made `discount = 0.5` affordable.

Equivalently this is Bradley-Terry on `Delta_{z+1}` -- the exact statistic §9.1 thresholds.
No other phase-1 loss touches it, and §7.6.2's gradient table shows it is the only term that
trains psi AS A SOURCE.

`ln 5` **is the FIXTURE value, not the model's** (corrected 2026-07-27, §7.6.7). It is
`-log sigma(-m) = 1.6094`, which needs `Delta_{z+1} = 0` -- true only where every distance is
equal, which tests/test_step_loss.py pins exactly and a real random psi does not give: LM
hiddens are strongly anisotropic and `h_{s_0}` is the most atypical of them, so `psi_0` starts
far from mid-solution states and `Delta_{z+1}` starts NEGATIVE wherever z = 0 (45.4% of
incorrect trajectories). Measured: `Delta = -2.86, L_step = 4.26` on an all-z=0 fixture,
`Delta = -0.44, L_step = 2.08` on the first real batch. **There is no single init level to
quote and the earlier "exact, not approximate" claim was about the wrong quantity.**

What is exact is the sandwich, which needs no tolerance because softplus is decreasing in
Delta:

    softplus(m - step/delta_max)  <=  L_step  <=  softplus(m - step/delta_min)

and the margin arithmetic `m = margin_steps * (-log gamma)` plus the z indexing -- which are
what the fixture actually tests. (6) L_good's sandwich runs the OTHER way, because relu is
increasing in Delta; see §7.12.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import Config
from ..data.collate import Batch


def _incorrect_trajectories(batch: Batch, cfg: Config) -> Tensor:
    keep = (~batch.traj_correct) & (batch.traj_z >= 0)
    if cfg.losses.step_loss.exclude_recovery:
        # §16.15: the 1.48% of False -> True trajectories. z is still the FIRST False, which
        # matches ProcessBench semantics; this flag drops them from L_step specifically if
        # diagnostic #17 says the label noise is hurting.
        keep = keep & (~batch.traj_recovery)
    return torch.nonzero(keep, as_tuple=False).flatten()


def _empty(D_term: Tensor) -> tuple[Tensor, dict[str, float]]:
    """Masked, not crashed: no z exists (fully correct trajectory) or no g exists (question
    with no correct trajectory). Keeps a zero in the graph so backward still works."""
    zero = D_term.sum() * 0.0
    return zero, {
        "step/pairs": 0.0,
        "step/distinct_z": 0.0,
        "step/loss": 0.0,
        "step/delta_min": 0.0,
        "step/delta_max": 0.0,
    }


def step_loss(
    D_term: Tensor,
    batch: Batch,
    terminal_traj: Tensor,
    cfg: Config,
) -> tuple[Tensor, dict[str, float]]:
    """Returns (loss, diagnostics).

    `D_term[s, t] = d(psi_s, psi(s_T^t))` over all states x correct terminals (matrix.py).
    """
    m = cfg.step_margin
    losers = _incorrect_trajectories(batch, cfg)
    if losers.numel() == 0 or terminal_traj.numel() == 0:
        return _empty(D_term)

    q_loser = batch.traj_qid[losers]                       # (B_i,)
    q_term = batch.traj_qid[terminal_traj]                 # (T_c,)
    pair = q_loser[:, None] == q_term[None, :]             # (B_i, T_c)
    li, ti = torch.nonzero(pair, as_tuple=True)
    if li.numel() == 0:
        return _empty(D_term)

    traj = losers[li]
    offset = batch.traj_state_offset[traj]
    z = batch.traj_z[traj]
    T = batch.traj_T[traj]
    assert bool((z + 1 <= T).all()), "z+1 must exist: z indexes labels, so z <= T-1 (§6.1)"

    pairing = cfg.losses.step_loss.pairing
    if pairing == "boundary":
        delta = D_term[offset + z + 1, ti] - D_term[offset + z, ti]
        loss = -F.logsigmoid(delta - m).mean()
    elif pairing == "position_corrected":
        delta, loss = _position_corrected(D_term, batch, traj, ti, m, cfg)
    elif pairing == "same_index":
        delta, loss = _same_index(D_term, batch, traj, ti, terminal_traj, m)
    else:
        raise ValueError(f"unknown step_loss.pairing {pairing!r}")

    if delta.numel() == 0:
        return _empty(D_term)

    with torch.no_grad():
        info = {
            "step/loss": float(loss),
            "step/pairs": float(delta.numel()),                       # expect ~64 (§8.1.1)
            # Diagnostic #17: distinct z, i.e. the number of incorrect trajectories, is the
            # count that matters -- the pairs reuse the same psi_z against several goals.
            "step/distinct_z": float(losers.numel()),                 # expect ~28
            "step/delta_mean": float(delta.mean()),
            # softplus is decreasing in delta, so the loss is sandwiched:
            #   softplus(m - delta_max) <= L_step <= softplus(m - delta_min)
            # That is an exact bound with no tolerance in it, and it is what the §18 init probe
            # checks against instead of assuming Delta ~ 0 (tests/test_gpu.py).
            "step/delta_min": float(delta.min()),
            "step/delta_max": float(delta.max()),
            "step/delta_at_margin_fraction": float((delta > m).float().mean()),
            "step/margin": m,
            "step/z_zero_fraction": float((batch.traj_z[losers] == 0).float().mean()),
            "step/recovery_fraction": float(batch.traj_recovery.float().mean()),
        }
    return loss, info


def _position_corrected(
    D_term: Tensor, batch: Batch, traj: Tensor, ti: Tensor, m: float, cfg: Config
) -> tuple[Tensor, Tensor]:
    """§7.10, OFF by default. All (good i, bad j) pairs with the ruler subtracted:

        -log sigma( d_j - d_i + (j-i)*(-log gamma) - m )

    The correction is MANDATORY, not cosmetic: without it the demand grows with (j-i) and
    fights L_T (§7.6.1). Note `boundary(k) == position_corrected(k+1)` at j = i+1, so
    switching pairing must bump `margin_steps` 2.0 -> 3.0 or the boundary pair silently
    loses one step of margin.
    """
    step_cost = cfg.neg_log_gamma
    deltas = []
    for n in range(traj.numel()):
        b = int(traj[n])
        col = int(ti[n])
        offset = int(batch.traj_state_offset[b])
        z = int(batch.traj_z[b])
        T = int(batch.traj_T[b])
        good = torch.arange(0, z + 1, device=D_term.device)
        bad = torch.arange(z + 1, T + 1, device=D_term.device)
        d_good = D_term[offset + good, col]
        d_bad = D_term[offset + bad, col]
        gap = (bad[None, :] - good[:, None]).to(d_good.dtype) * step_cost
        deltas.append((d_bad[None, :] - d_good[:, None] + gap).reshape(-1))
    delta = torch.cat(deltas) if deltas else D_term.new_zeros(0)
    return delta, -F.logsigmoid(delta - m).mean()


def _same_index(
    D_term: Tensor, batch: Batch, traj: Tensor, ti: Tensor, terminal_traj: Tensor, m: float
) -> tuple[Tensor, Tensor]:
    """§7.10, OFF by default. Step i of an incorrect trajectory against step i of a CORRECT
    one, for i >= z+1:

        -log sigma( d(psi_i^l, g) - d(psi_i^w, g) - m )

    Position cancels exactly, so there is no ruler correction and no assumption that L_T has
    converged -- the correct trajectory is its own positional control. Worth ~4.5 examples
    per incorrect trajectory (~128/batch vs 64) at no extra forward passes, and it is OFF
    because it is the only term that would compare states across two different solutions
    (§7.10). `margin_steps` carries over unchanged.
    """
    deltas = []
    for n in range(traj.numel()):
        b = int(traj[n])
        col = int(ti[n])
        winner = int(terminal_traj[col])
        z = int(batch.traj_z[b])
        T_l = int(batch.traj_T[b])
        T_w = int(batch.traj_T[winner])
        top = min(T_l, T_w)                       # cannot compare step 6 against a 4-step solution
        if z + 1 > top:
            continue                              # ~3.5% of incorrect trajectories get nothing
        idx = torch.arange(z + 1, top + 1, device=D_term.device)
        d_l = D_term[int(batch.traj_state_offset[b]) + idx, col]
        d_w = D_term[int(batch.traj_state_offset[winner]) + idx, col]
        deltas.append(d_l - d_w)
    delta = torch.cat(deltas) if deltas else D_term.new_zeros(0)
    if delta.numel() == 0:
        return delta, D_term.sum() * 0.0
    return delta, -F.logsigmoid(delta - m).mean()
