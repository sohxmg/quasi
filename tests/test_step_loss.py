"""§15's (5) L_step tests -- "the highest-value test in this file" (§7.6, locked #3b).

z is 0-BASED. The pair is (psi_z, psi_{z+1}): psi_z is the LAST GOOD state, psi_{z+1} the
FIRST BROKEN one. Training on Delta_z instead predicts z-1 on every errored sample, which
zeroes acc_error and collapses F1 through the harmonic mean while every loss curve still
looks healthy.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from feynman_prm.data.collate import collate, state_index_of
from feynman_prm.data.goals import correct_terminal_columns
from feynman_prm.losses.step import step_loss
from conftest import synthetic_row

T, F = True, False


def _fixture(labels_incorrect, labels_correct=(True, True), qid="q"):
    rows = [synthetic_row(qid, list(labels_correct)), synthetic_row(qid, list(labels_incorrect))]
    batch = collate(rows, pad_id=0)
    terminal_states, terminal_traj = correct_terminal_columns(batch)
    return batch, terminal_traj, terminal_states


def test_init_value_is_exactly_ln5(cfg):
    """§18: at init Delta_{z+1} ~= 0, so L_step = -log sigma(-m) = log(1 + e^m) = ln 5 =
    1.6094 at discount 0.5 / margin_steps 2. EXACT, not approximate -- if it is not 1.609 the
    margin or the z indexing is wrong."""
    batch, terminal_traj, _ = _fixture([T, T, F, F])
    D_term = torch.full((batch.n_states, int(batch.traj_correct.sum())), 3.0)
    loss, info = step_loss(D_term, batch, terminal_traj, cfg)
    assert math.isclose(float(loss), math.log(5.0), rel_tol=1e-6)
    assert math.isclose(float(loss), 1.6094379, rel_tol=1e-6)
    assert math.isclose(info["step/margin"], math.log(4.0), rel_tol=1e-6)


def test_init_value_at_the_070_fallback(cfg):
    """1.112 at the sanctioned fallback (§7.8's consequence table)."""
    at_07 = dataclasses.replace(cfg, discount=0.7)
    batch, terminal_traj, _ = _fixture([T, F])
    D_term = torch.full((batch.n_states, 1), 2.0)
    loss, _ = step_loss(D_term, batch, terminal_traj, at_07)
    assert math.isclose(float(loss), 1.1120, rel_tol=1e-3)


def test_reads_d_z_plus_1_minus_d_z_and_never_touches_d_1(cfg):
    """labels = [T, T, F, F, F, F] => z = 2 => the loss reads d_3 - d_2 and MUST NOT touch
    d_1. Checked through the gradient, which is where a wrong index would show up."""
    batch, terminal_traj, _ = _fixture([T, T, F, F, F, F])
    D_term = torch.zeros(batch.n_states, 1, requires_grad=True)
    loss, _ = step_loss(D_term, batch, terminal_traj, cfg)
    loss.backward()

    incorrect = 1
    z = int(batch.traj_z[incorrect])
    assert z == 2
    touched = {
        i for i in range(batch.n_states) if float(D_term.grad[i, 0].abs()) > 0
    }
    assert touched == {
        state_index_of(batch, incorrect, z),        # psi_z, last good
        state_index_of(batch, incorrect, z + 1),    # psi_{z+1}, first broken
    }
    assert state_index_of(batch, incorrect, 1) not in touched, "d_1 must be untouched"
    assert state_index_of(batch, incorrect, z - 1) not in touched, "psi_{z-1} is a GOOD state"


def test_z_zero_is_finite_and_differentiable(cfg):
    """45.4% of incorrect trajectories have z = 0 (§4.2.1) -- not an edge case. The pair is
    (psi_0, psi_1); psi_0 is the prompt-only state and always exists."""
    batch, terminal_traj, _ = _fixture([F, F])
    D_term = torch.randn(batch.n_states, 1, requires_grad=True)
    loss, info = step_loss(D_term, batch, terminal_traj, cfg)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(D_term.grad).all()
    assert info["step/z_zero_fraction"] == 1.0


def test_sign_swapping_the_pair_flips_the_loss(cfg):
    batch, terminal_traj, _ = _fixture([T, F, F])
    incorrect, z = 1, 1
    s_z = state_index_of(batch, incorrect, z)
    s_z1 = state_index_of(batch, incorrect, z + 1)

    good = torch.zeros(batch.n_states, 1)
    good[s_z1] = 3.0            # the broken state is FAR from the goal: satisfied
    bad = torch.zeros(batch.n_states, 1)
    bad[s_z] = 3.0              # the last good state is far instead: violated
    assert float(step_loss(good, batch, terminal_traj, cfg)[0]) < float(
        step_loss(bad, batch, terminal_traj, cfg)[0]
    )


def test_matches_a_reference_bradley_terry_implementation(cfg):
    batch, terminal_traj, _ = _fixture([T, F, F])
    torch.manual_seed(0)
    D_term = torch.randn(batch.n_states, 1)
    loss, _ = step_loss(D_term, batch, terminal_traj, cfg)

    incorrect, z = 1, 1
    delta = D_term[state_index_of(batch, incorrect, z + 1), 0] - D_term[
        state_index_of(batch, incorrect, z), 0
    ]
    reference = -torch.log(torch.sigmoid(delta - cfg.step_margin))
    assert torch.allclose(loss, reference, atol=1e-6)


def test_margin_scales_with_discount(cfg):
    """A test that hardcodes 1.386 would not catch a revert to gamma = 0.7, where m must
    become 0.713 (§15)."""
    assert math.isclose(cfg.step_margin, cfg.losses.step_loss.margin_steps * -math.log(cfg.discount))
    at_07 = dataclasses.replace(cfg, discount=0.7)
    assert math.isclose(at_07.step_margin, 0.7133, rel_tol=1e-3)


def test_masked_not_crashed_when_no_z_or_no_goal_exists(cfg):
    """Fully correct trajectories have no z; a question with no correct trajectory has no g."""
    rows = [synthetic_row("q", [T, T]), synthetic_row("q", [T, T, T])]
    batch = collate(rows, pad_id=0)
    _, terminal_traj = correct_terminal_columns(batch)
    D_term = torch.randn(batch.n_states, 2, requires_grad=True)
    loss, info = step_loss(D_term, batch, terminal_traj, cfg)
    assert float(loss) == 0.0 and info["step/pairs"] == 0.0
    loss.backward()                      # still differentiable, just zero

    rows = [synthetic_row("q", [T, F]), synthetic_row("q", [F, F])]
    batch = collate(rows, pad_id=0)
    _, terminal_traj = correct_terminal_columns(batch)
    D_term = torch.randn(batch.n_states, 0, requires_grad=True)
    loss, info = step_loss(D_term, batch, terminal_traj, cfg)
    assert info["step/pairs"] == 0.0


def test_goal_is_a_terminal_of_a_DIFFERENT_correct_trajectory(cfg):
    """§7.6.3: g is psi(s_T) of a CORRECT trajectory in the batch -- never a prediction,
    never a centroid, never a geometric-sampler goal column."""
    batch, terminal_traj, terminal_states = _fixture([T, F])
    for col, traj in enumerate(terminal_traj.tolist()):
        assert bool(batch.traj_correct[traj])
        assert int(terminal_states[col]) == int(batch.traj_terminal[traj])
        assert int(batch.state_step[terminal_states[col]]) == int(batch.traj_T[traj])


def test_pairs_and_distinct_z_are_counted_separately(cfg):
    """Diagnostic #17: pairs are k_c * k_i but the k_c goals all compare against the SAME
    psi_z. Distinct z -- the number of incorrect trajectories -- is what matters."""
    rows = [
        synthetic_row("q", [T, T]),
        synthetic_row("q", [T, T, T]),
        synthetic_row("q", [T, F]),
        synthetic_row("q", [F, F]),
    ]
    batch = collate(rows, pad_id=0)
    _, terminal_traj = correct_terminal_columns(batch)
    D_term = torch.randn(batch.n_states, 2)
    _, info = step_loss(D_term, batch, terminal_traj, cfg)
    assert info["step/pairs"] == 4.0            # 2 correct x 2 incorrect
    assert info["step/distinct_z"] == 2.0       # ...but only 2 independent gradients


def test_exclude_recovery_drops_the_1_48_percent(cfg):
    """§16.15: z is still the FIRST False (ProcessBench semantics); this flag only removes
    those trajectories from L_step."""
    rows = [synthetic_row("q", [T, T]), synthetic_row("q", [T, F, T])]
    batch = collate(rows, pad_id=0)
    _, terminal_traj = correct_terminal_columns(batch)
    D_term = torch.randn(batch.n_states, 1)
    assert bool(batch.traj_recovery[1])

    on = dataclasses.replace(
        cfg,
        losses=dataclasses.replace(
            cfg.losses, step_loss=dataclasses.replace(cfg.losses.step_loss, exclude_recovery=True)
        ),
    )
    assert step_loss(D_term, batch, terminal_traj, cfg)[1]["step/pairs"] == 1.0
    assert step_loss(D_term, batch, terminal_traj, on)[1]["step/pairs"] == 0.0


@pytest.mark.parametrize("pairing", ["boundary", "position_corrected", "same_index"])
def test_all_three_pairings_run(cfg, pairing):
    batch, terminal_traj, _ = _fixture([T, T, F, F], labels_correct=[T, T, T, T])
    D_term = torch.randn(batch.n_states, 1, requires_grad=True)
    variant = dataclasses.replace(
        cfg,
        losses=dataclasses.replace(
            cfg.losses, step_loss=dataclasses.replace(cfg.losses.step_loss, pairing=pairing)
        ),
    )
    loss, info = step_loss(D_term, batch, terminal_traj, variant)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(D_term.grad).all()


def test_boundary_k_equals_position_corrected_k_plus_one_at_j_eq_i_plus_1(cfg):
    """§7.10/§15: the two forms coincide on the boundary pair only when their margins differ
    by exactly -log gamma, so switching `pairing` to position_corrected must move
    margin_steps 2.0 -> 3.0 or the boundary pair silently loses one step of margin."""
    batch, terminal_traj, _ = _fixture([F])          # z = 0, T = 1: good {0}, bad {1} only
    torch.manual_seed(0)
    D_term = torch.randn(batch.n_states, 1)

    def variant(pairing, margin_steps):
        return dataclasses.replace(
            cfg,
            losses=dataclasses.replace(
                cfg.losses,
                step_loss=dataclasses.replace(
                    cfg.losses.step_loss, pairing=pairing, margin_steps=margin_steps
                ),
            ),
        )

    boundary = step_loss(D_term, batch, terminal_traj, variant("boundary", 2.0))[0]
    corrected = step_loss(D_term, batch, terminal_traj, variant("position_corrected", 3.0))[0]
    assert torch.allclose(boundary, corrected, atol=1e-6)

    mismatched = step_loss(D_term, batch, terminal_traj, variant("position_corrected", 2.0))[0]
    assert not torch.allclose(boundary, mismatched, atol=1e-4)


def test_same_index_yields_more_terms_than_boundary(cfg):
    """~4.5 examples per incorrect trajectory instead of 1 (§7.10), at no extra forward
    passes -- but OFF by decision, because it is the only term that would compare states
    across two different solutions."""
    batch, terminal_traj, _ = _fixture([T, F, F, F], labels_correct=[T, T, T, T])
    D_term = torch.randn(batch.n_states, 1)
    same = dataclasses.replace(
        cfg,
        losses=dataclasses.replace(
            cfg.losses,
            step_loss=dataclasses.replace(cfg.losses.step_loss, pairing="same_index"),
        ),
    )
    assert step_loss(D_term, batch, terminal_traj, same)[1]["step/pairs"] > step_loss(
        D_term, batch, terminal_traj, cfg
    )[1]["step/pairs"]
