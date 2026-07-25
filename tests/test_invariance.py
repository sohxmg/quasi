"""§15's (2) L_I tests (§7.3, `tmd.py:100-105`)."""

from __future__ import annotations

import torch

from feynman_prm.losses.invariance import invariance_loss
from feynman_prm.model.distances import Distance


def test_diagonal_drives_the_residual_below_0_1():
    """`d(psi(s), phi(s,a)) -> 0` under L_I alone, in DIAGONAL mode."""
    torch.manual_seed(0)
    dist = Distance("full_mrn", 8)
    psi = torch.randn(6, 512)
    phi = torch.randn(6, 512, requires_grad=True)
    row_src = torch.arange(6)
    row_q = torch.zeros(6, dtype=torch.long)

    opt = torch.optim.Adam([phi], lr=0.1)
    for _ in range(300):
        loss, info = invariance_loss(psi, phi, row_src, row_q, dist, mode="diagonal")
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert info["invariance/residual_diagonal"] < 0.1


def test_all_three_modes_run_and_differ_when_actions_span_questions():
    """The grid modes assert that a step from one question is a free move in another -- ~88%
    of grid entries at Q=8. They exist only to reproduce §16.3's failure."""
    torch.manual_seed(0)
    dist = Distance("full_mrn", 8)
    psi = torch.randn(8, 512)
    phi = torch.randn(8, 512)
    row_src = torch.arange(8)
    row_q = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    values = {
        mode: float(invariance_loss(psi, phi, row_src, row_q, dist, mode=mode)[0])
        for mode in ("diagonal", "grid_within_question", "grid_batch")
    }
    assert len(set(values.values())) == 3, values


def test_stopgrad_phi_blocks_gradient_to_phi_only():
    dist = Distance("full_mrn", 8)
    psi = torch.randn(4, 512, requires_grad=True)
    phi = torch.randn(4, 512, requires_grad=True)
    loss, _ = invariance_loss(
        psi, phi, torch.arange(4), torch.zeros(4, dtype=torch.long), dist, stopgrad_phi=True
    )
    loss.backward()
    assert phi.grad is None
    assert psi.grad is not None


def test_lambda_i_sits_at_1_0_not_under_zeta(cfg):
    """tmd.py:124 is `contrastive + action_invariance + zeta * backup`. The old project used
    `L_NCE + zeta*(L_I + L_T)` with zeta=0.1, making L_I 10x weaker than TMD's own setting,
    and its residual never got below 0.43."""
    assert cfg.losses.lambda_i == 1.0
    assert cfg.losses.zeta == 0.05
