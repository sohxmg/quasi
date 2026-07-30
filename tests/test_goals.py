"""§15's goal-sampler tests (§7.1, `datasets.py:263-264`)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

from feynman_prm.data.collate import collate, state_index_of
from feynman_prm.data.goals import sample_geometric_offsets, sample_goals
from conftest import synthetic_row

T, F = True, False


def test_offsets_are_at_least_one():
    """`np.random.geometric` returns >= 1; `torch.distributions.Geometric` returns >= 0. The
    port must not silently introduce a zero offset (a goal equal to the source state)."""
    rng = np.random.default_rng(0)
    offsets = sample_geometric_offsets(rng, 10_000, 0.5)
    assert offsets.min() >= 1
    assert abs(offsets.mean() - 2.0) < 0.1, "E[geometric(p=0.5)] = 2 at discount 0.5"


def test_offset_base_is_the_SOURCE_state_index():
    """TMD's `idxs` is the observations index of s_t -> s_{t+1} (datasets.py:264). Our row
    phi_i is s_{i-1} -> s_i, so the base is i-1: offset = 1 means "the goal is s_i", which is
    what makes Next = d(x,x) = 0 and the backup target -log gamma consistent."""
    batch = collate([synthetic_row("q", [T, T, T, T])], pad_id=0)

    class OneRng:
        def geometric(self, p, size):
            return np.ones(size, dtype=np.int64)

    goals = sample_goals(batch, 0.5, OneRng())
    for c in range(goals.n_goals):
        r = int(goals.pos_row[c])
        i = int(batch.row_step[r])
        assert int(goals.goal_state[c]) == state_index_of(batch, 0, i), (
            "offset 1 from base i-1 lands on s_i, the row's own destination"
        )
        assert int(goals.goal_state[c]) == int(batch.row_dst[r])


def test_goals_are_clamped_to_the_terminal():
    batch = collate([synthetic_row("q", [T, T])], pad_id=0)

    class BigRng:
        def geometric(self, p, size):
            return np.full(size, 99, dtype=np.int64)

    goals = sample_goals(batch, 0.5, BigRng())
    assert bool(goals.is_terminal.all())
    assert torch.all(goals.goal_state == batch.traj_terminal[0])


def test_goal_columns_come_only_from_correct_trajectories():
    """Source rows from incorrect trajectories are NEGATIVE-ONLY -- no goal of their own and
    no positive column. That is why Dist is rectangular and has no diagonal (§7.1)."""
    rows = [synthetic_row("q", [T, T]), synthetic_row("q", [T, F, F])]
    batch = collate(rows, pad_id=0)
    goals = sample_goals(batch, 0.5, np.random.default_rng(0))
    assert goals.n_goals == 2, "only the correct trajectory's 2 rows produce columns"
    for traj in goals.goal_traj.tolist():
        assert bool(batch.traj_correct[traj])


def test_discount_reaches_BOTH_consumers(cfg):
    """The single key moves the sampled goal offsets AND the backup target -log gamma (and
    hence m and t). A test that changed `discount` and asserted only one of them moved would
    have caught the old two-key split (§7.8, §15)."""
    batch = collate([synthetic_row("q", [T] * 8)], pad_id=0)
    low = sample_goals(batch, 0.5, np.random.default_rng(0)).offsets.float().mean()
    high = sample_goals(batch, 0.9, np.random.default_rng(0)).offsets.float().mean()
    assert float(high) > float(low), "a larger discount draws goals further ahead"

    at_09 = dataclasses.replace(cfg, discount=0.9)
    assert at_09.neg_log_gamma < cfg.neg_log_gamma
    assert at_09.step_margin < cfg.step_margin
    # clip_t also falls with a larger discount, but via log(gain/gamma), NOT as a multiple of
    # neg_log_gamma -- and by far less: 3.689 -> 3.101, where the step cost drops 6.5x (§7.4.3).
    assert at_09.clip_t < cfg.clip_t
    assert math.isclose(at_09.backup_gain, cfg.backup_gain, rel_tol=1e-9)


def test_goal_type_mix_is_logged(cfg):
    """Diagnostic #16: the realised fraction of goals that are endings should match §4.5 for
    the chosen discount (41.1% at 0.5, 55.0% at 0.7). Eval queries an ending EVERY time --
    that gap is §16.2's train/eval mismatch."""
    rows = [synthetic_row("q", [T] * 6) for _ in range(40)]
    batch = collate(rows, pad_id=0)
    goals = sample_goals(batch, 0.5, np.random.default_rng(0))
    fraction = float(goals.is_terminal.float().mean())
    assert 0.25 < fraction < 0.6, fraction
