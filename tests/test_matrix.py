"""§15's shared-matrix tests (§7.1)."""

from __future__ import annotations

import numpy as np
import torch

from feynman_prm.data.goals import sample_goals
from feynman_prm.losses.matrix import build_matrices


def _matrices(cfg, batch, psi, phi, distance, seed=0):
    goals = sample_goals(batch, cfg.discount, np.random.default_rng(seed))
    return goals, build_matrices(psi, phi, batch, goals, distance, cfg)


def test_dist_is_one_tensor_shared_by_nce_and_backup(cfg, small_batch, random_reps, distance):
    """Assert IDENTITY, not equality: TMD builds it once (tmd.py:91) and reuses it at :115,
    and the all-pairs form of L_T is free only because the tensor is the same object."""
    psi, phi = random_reps
    _, m = _matrices(cfg, small_batch, psi, phi, distance)
    assert m.Dist_backup is m.Dist


def test_stopgrad_psi_backup_creates_a_second_tensor(cfg, small_batch, random_reps, distance):
    """tmd.py:111-112 RECOMPUTES dist with the goal side detached, after the contrastive loss
    has used the attached one. The flag must not change what L_NCE reads."""
    import dataclasses

    psi, phi = random_reps
    backup = dataclasses.replace(cfg.losses.backup, stopgrad_psi_backup=True)
    losses = dataclasses.replace(cfg.losses, backup=backup)
    cfg2 = dataclasses.replace(cfg, losses=losses)
    _, m = _matrices(cfg2, small_batch, psi, phi, distance)
    assert m.Dist_backup is not m.Dist
    assert torch.allclose(m.Dist_backup, m.Dist), "same values, different graph"


def test_matrix_is_rectangular_and_pos_row_replaces_the_diagonal(
    cfg, small_batch, random_reps, distance
):
    psi, phi = random_reps
    goals, m = _matrices(cfg, small_batch, psi, phi, distance)
    assert m.Dist.shape == (small_batch.n_rows, goals.n_goals)
    assert m.n_rows != m.n_goals, "rectangular: incorrect-trajectory rows have no goal column"
    for c in range(goals.n_goals):
        r = int(m.pos_row[c])
        assert bool(small_batch.traj_correct[small_batch.row_traj[r]]), (
            "goal columns come only from CORRECT trajectories (§7.1)"
        )


def test_sq_mask_is_exactly_same_question(cfg, small_batch, random_reps, distance):
    psi, phi = random_reps
    goals, m = _matrices(cfg, small_batch, psi, phi, distance)
    row_q = small_batch.traj_qid[small_batch.row_traj]
    goal_q = small_batch.traj_qid[goals.goal_traj]
    assert torch.equal(m.SQ, row_q[:, None] == goal_q[None, :])
    # SQ INCLUDES the matched pairs, exactly as TMD's diagonal sits inside its full matrix.
    for c in range(goals.n_goals):
        assert bool(m.SQ[int(m.pos_row[c]), c])


def test_next_is_detached_unconditionally(cfg, small_batch, random_reps, distance):
    """tmd.py:113: the Bellman target carries no gradient in either stopgrad setting."""
    psi, phi = random_reps
    _, m = _matrices(cfg, small_batch, psi, phi, distance)
    assert not m.Next.requires_grad
    assert m.Dist.requires_grad


def test_d_term_covers_all_states_x_correct_terminals(cfg, small_batch, random_reps, distance):
    psi, phi = random_reps
    _, m = _matrices(cfg, small_batch, psi, phi, distance)
    n_correct = int(small_batch.traj_correct.sum())
    assert m.D_term.shape == (small_batch.n_states, n_correct)
    assert m.terminal_states.tolist() == [
        int(small_batch.traj_terminal[b])
        for b in range(small_batch.n_traj)
        if bool(small_batch.traj_correct[b])
    ]
