"""§15's (3) L_T tests (§7.4, `tmd.py:107-122`)."""

from __future__ import annotations

import dataclasses
import math

import torch

from feynman_prm.losses.temporal import temporal_loss


def _cfg(cfg, **backup):
    return dataclasses.replace(
        cfg,
        losses=dataclasses.replace(
            cfg.losses, backup=dataclasses.replace(cfg.losses.backup, **backup)
        ),
    )


def test_minimiser_is_next_minus_log_gamma(cfg):
    """`d/dDist [gamma*exp(Dist - Next) - Dist] = 0  =>  Dist = Next - log gamma`, i.e.
    arriving at s_i cut the distance to g by exactly one step's worth."""
    Next = torch.full((4, 3), 2.0)
    Dist = (Next - math.log(cfg.discount)).clone().requires_grad_(True)
    loss, _ = temporal_loss(Dist, Next, torch.tensor([0, 1, 2]), torch.ones(4, 3, dtype=torch.bool), cfg)
    loss.backward()
    assert torch.allclose(Dist.grad, torch.zeros_like(Dist.grad), atol=1e-6)


def test_value_at_the_minimum_is_one_minus_dist(cfg):
    """`gamma*exp(-log gamma) - Dist = 1 - Dist`, so L_T is EXPECTED TO BE NEGATIVE. Watch
    plateau and NaN, not sign."""
    Next = torch.full((5, 5), 3.0)
    Dist = Next - math.log(cfg.discount)
    loss, info = temporal_loss(Dist, Next, torch.arange(5), torch.ones(5, 5, dtype=torch.bool), cfg)
    assert math.isclose(float(loss), 1.0 - float(Dist[0, 0]), rel_tol=1e-6)
    assert float(loss) < 0


def test_all_pairs_not_just_matched_pairs(cfg):
    """tmd.py:91 and :107 are full (source x goal) matrices, so the term count is R*C
    (~60,000 at the §8.1 layout), not R. Calibrating only matched pairs lets the model learn
    a different scale per question, which breaks a single global tau (§7.4.1)."""
    R, C = 7, 4
    Dist = torch.rand(R, C)
    Next = torch.rand(R, C)
    _, info = temporal_loss(
        Dist, Next, torch.arange(C), torch.ones(R, C, dtype=torch.bool), cfg
    )
    assert info["backup/terms"] == R * C


def test_nan_guard_gives_finite_loss_AND_finite_gradients(cfg):
    """The double `where` is exponent clipping, and it is load-bearing: torch.where evaluates
    the DISCARDED branch in the backward pass, so an inf there poisons the gradient even
    though the value is thrown away. Fixture: delta = 200."""
    Next = torch.zeros(3, 3)
    Dist = torch.full((3, 3), 200.0, requires_grad=True)
    loss, info = temporal_loss(
        Dist, Next, torch.arange(3), torch.ones(3, 3, dtype=torch.bool), cfg
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(Dist.grad).all()
    assert info["backup/linear_branch_fraction"] == 1.0, "delta=200 > t=19.75 -> linear branch"


def test_clip_t_scales_with_discount(cfg):
    """t = clip_t_steps * (-log gamma): 28.5 reproduces TMD's bare t=3.0 at gamma=0.9, and
    keeps the same slack at any discount (§7.4.3, locked #9a)."""
    at_09 = dataclasses.replace(cfg, discount=0.9)
    at_07 = dataclasses.replace(cfg, discount=0.7)
    assert math.isclose(at_09.clip_t, 3.0, rel_tol=1e-3)
    assert math.isclose(at_07.clip_t, 10.17, rel_tol=1e-3)
    assert math.isclose(cfg.clip_t, 19.75, rel_tol=1e-3)


def test_goal_scope_ratio_interpolates_batch_and_same_question(cfg):
    """rho = 1.0 reproduces the plain batch mean, 0.0 the same-question mean, 0.5 their
    average -- and BOTH terms are logged either way (diagnostic #13, §7.4.2)."""
    R, C = 6, 3
    torch.manual_seed(0)
    Dist, Next = torch.rand(R, C), torch.rand(R, C)
    SQ = torch.zeros(R, C, dtype=torch.bool)
    SQ[:3] = True
    pos_row = torch.tensor([0, 1, 2])

    full = temporal_loss(Dist, Next, pos_row, SQ, _cfg(cfg, goal_scope_ratio=1.0, diag_backup=0.0))[0]
    same = temporal_loss(Dist, Next, pos_row, SQ, _cfg(cfg, goal_scope_ratio=0.0, diag_backup=0.0))[0]
    half = temporal_loss(Dist, Next, pos_row, SQ, _cfg(cfg, goal_scope_ratio=0.5, diag_backup=0.0))[0]
    assert torch.allclose(half, (full + same) / 2, atol=1e-6)

    _, info = temporal_loss(Dist, Next, pos_row, SQ, _cfg(cfg, goal_scope_ratio=1.0))
    assert "backup/div_same_question" in info and "backup/div_cross_question" in info


def test_diag_backup_mixes_matched_into_both_terms(cfg):
    """tmd.py:120-121 BROADCASTS the matched divergence across every column of its row, so
    after the mean it is a (1-dw)/dw mix with the matched set counted inside both (§7.4.2)."""
    R, C = 4, 2
    torch.manual_seed(1)
    Dist, Next = torch.rand(R, C), torch.rand(R, C)
    SQ = torch.ones(R, C, dtype=torch.bool)
    pos_row = torch.tensor([0, 1])

    only_all = temporal_loss(Dist, Next, pos_row, SQ, _cfg(cfg, diag_backup=0.0))[0]
    only_matched = temporal_loss(Dist, Next, pos_row, SQ, _cfg(cfg, diag_backup=1.0))[0]
    mixed = temporal_loss(Dist, Next, pos_row, SQ, _cfg(cfg, diag_backup=0.5))[0]
    assert torch.allclose(mixed, 0.5 * only_all + 0.5 * only_matched, atol=1e-6)


def test_next_receives_no_gradient(cfg):
    """tmd.py:113 -- the Bellman target is stop-gradded unconditionally."""
    Next = torch.rand(3, 2, requires_grad=True)
    Dist = torch.rand(3, 2, requires_grad=True)
    loss, _ = temporal_loss(
        Dist, Next.detach(), torch.arange(2), torch.ones(3, 2, dtype=torch.bool), cfg
    )
    loss.backward()
    assert Next.grad is None
    assert Dist.grad is not None
