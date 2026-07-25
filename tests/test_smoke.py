"""The CPU smoke test (PLAN step 3, §15): every loss finite and backward on random hiddens,
no model, no GPU; and same seed -> identical losses.

This also exercises the full assembly path -- collate -> goals -> matrices -> phase1_loss ->
probes -- which is everything except the backbone forward.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from feynman_prm.data.collate import collate
from feynman_prm.data.goals import sample_goals
from feynman_prm.diagnostics.probes import asymmetry_score, batch_probes
from feynman_prm.losses.matrix import build_matrices
from feynman_prm.losses.total import expected_init_values, phase1_loss
from feynman_prm.model.distances import Distance
from conftest import synthetic_row

T, F = True, False


def _batch(seed=0, n_questions=6):
    rows = []
    rng = np.random.default_rng(seed)
    for q in range(n_questions):
        for _ in range(int(rng.integers(1, 4))):
            rows.append(synthetic_row(f"q{q}", [T] * int(rng.integers(2, 7))))
        for _ in range(int(rng.integers(1, 4))):
            T_len = int(rng.integers(2, 7))
            z = int(rng.integers(0, T_len))
            labels = [True] * z + [False] * (T_len - z)
            rows.append(synthetic_row(f"q{q}", labels))
    return collate(rows, pad_id=0)


def _run(cfg, seed=0):
    torch.manual_seed(seed)
    batch = _batch(seed)
    goals = sample_goals(batch, cfg.discount, np.random.default_rng(seed))
    D = cfg.heads.latent_dim
    psi = torch.randn(batch.n_states, D, requires_grad=True) * 0.1
    phi = torch.randn(batch.n_rows, D, requires_grad=True) * 0.1
    psi.retain_grad()
    phi.retain_grad()
    distance = Distance(cfg.distance.variant, cfg.distance.components)
    matrices = build_matrices(psi, phi, batch, goals, distance, cfg)
    out = phase1_loss(psi, phi, batch, matrices, distance, cfg, goal_traj=goals.goal_traj)
    return batch, goals, psi, phi, distance, matrices, out


def test_every_loss_is_finite_and_backward_works(cfg):
    batch, goals, psi, phi, distance, matrices, out = _run(cfg)
    for name, term in out.terms.items():
        assert torch.isfinite(term), f"{name} is not finite"
    out.total.backward()
    assert torch.isfinite(psi.grad).all()
    assert torch.isfinite(phi.grad).all()
    assert float(psi.grad.abs().sum()) > 0, "psi must receive gradient (§7.6.2)"
    assert float(phi.grad.abs().sum()) > 0


def test_probes_all_compute(cfg):
    batch, goals, psi, phi, distance, matrices, _ = _run(cfg)
    probes = batch_probes(psi, phi, batch, goals, matrices, distance, cfg)
    for key in (
        "probe01/distinct_goal_ratio",
        "probe02/target_good_step_delta",
        "probe03/gap",
        "probe04/symmetric_share",
        "probe12/off_over_on",
        "probe14/delta_boundary/mean",
        "probe16/goal_is_terminal_fraction",
    ):
        assert key in probes, key
    assert probes["probe02/target_good_step_delta"] == pytest.approx(-0.69315, rel=1e-4)
    assert asymmetry_score(psi, batch, distance)["probe09_4/irreversibility_mean"] is not None


def test_same_seed_gives_identical_losses(cfg):
    a = _run(cfg, seed=3)[-1]
    b = _run(cfg, seed=3)[-1]
    assert float(a.total) == float(b.total)
    for key in a.terms:
        assert float(a.terms[key]) == float(b.terms[key])


def test_different_seed_gives_different_losses(cfg):
    assert float(_run(cfg, seed=1)[-1].total) != float(_run(cfg, seed=2)[-1].total)


def test_init_values_are_in_the_expected_neighbourhood(cfg):
    """§18: L_NCE ~ log(R); L_step is EXACT at ln 5 when Delta ~ 0; L_T starts positive."""
    batch, goals, psi, phi, distance, matrices, out = _run(cfg)
    expected = expected_init_values(cfg, matrices.n_rows, out.info["backup/dist_mean"])
    assert expected["step"] == pytest.approx(1.6094, rel=1e-4)
    # random latents at 0.1 scale are near-uniform, so NCE should sit close to chance
    assert float(out.terms["nce"]) == pytest.approx(expected["nce"], rel=0.2)
    # §18 says L_T "starts positive, ~= gamma". That reading assumes delta ~= 0 AND a mean
    # distance below gamma; with random latents neither holds exactly (Jensen also makes
    # E[exp(delta)] > exp(E[delta])). What survives at any init is `div = gamma*exp(delta) -
    # Dist`, so the loss sits below gamma whenever distances are O(1). The exact minimiser
    # and the `1 - Dist` value are pinned in tests/test_temporal.py, not here.
    assert torch.isfinite(out.terms["backup"])
    assert float(out.terms["backup"]) < cfg.discount
    assert expected["backup"] == pytest.approx(cfg.discount - out.info["backup/dist_mean"])


def test_zeta_weights_the_backup_only(cfg):
    """tmd.py:124 -- action invariance sits at 1.0, not under zeta (§7.0)."""
    batch, goals, psi, phi, distance, matrices, out = _run(cfg)
    manual = (
        cfg.losses.lambda_nce * out.terms["nce"]
        + cfg.losses.lambda_i * out.terms["invariance"]
        + cfg.losses.zeta * out.terms["backup"]
        + cfg.losses.lambda_cf * out.terms["cf"]
        + cfg.losses.lambda_step * out.terms["step"]
    )
    assert torch.allclose(out.total, manual, atol=1e-6)


def test_cf_is_inert_at_lambda_zero(cfg):
    """§7.10/§15: a run with defaults produces bit-identical losses whether or not the
    deferred code paths exist."""
    out = _run(cfg)[-1]
    assert float(out.terms["cf"]) == 0.0
    assert float(out.info["cf/loss"]) == 0.0
