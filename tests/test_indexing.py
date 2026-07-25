"""§15's index-convention tests -- the highest-value tests in the repo (§6.1).

Every off-by-one in this project lives in these four functions.
"""

from __future__ import annotations

import pytest

from feynman_prm.data.collate import state_index_of
from feynman_prm.utils.indexing import (
    first_error_index,
    has_recovery,
    predicted_label_from_deltas,
    trajectory_is_correct,
)

T, F = True, False


def test_completions_k_maps_to_s_k_to_s_k_plus_1(small_batch):
    """`completions[k]` is the step that takes s_k -> s_{k+1} (§6.1), round-tripped."""
    for r in range(small_batch.n_rows):
        i = int(small_batch.row_step[r])          # 1-based step index
        traj = int(small_batch.row_traj[r])
        # the row for step k = i-1 departs from s_{i-1} and lands in s_i
        assert int(small_batch.row_src[r]) == state_index_of(small_batch, traj, i - 1)
        assert int(small_batch.row_dst[r]) == state_index_of(small_batch, traj, i)


def test_z_is_zero_based_and_selects_the_boundary_pair():
    labels = [T, T, F, F, F, F]
    z = first_error_index(labels)
    assert z == 2, "z is the index of the first False, 0-based"
    # psi_z is the LAST GOOD state, psi_{z+1} the FIRST BROKEN one.
    assert labels[z] is F and labels[z - 1] is T
    # z+1 always exists: z indexes labels, so z <= T-1 and s_{z+1} <= s_T.
    assert z + 1 <= len(labels)


def test_z_zero_is_common_not_an_edge_case():
    """45.4% of incorrect trajectories have z = 0 (§4.2.1). psi_{z-1} does not exist."""
    assert first_error_index([F, F]) == 0
    assert first_error_index([T, T, T]) is None


def test_recovery_detection():
    assert has_recovery([T, F, T]) is True          # the 1.48% (§4.2, §16.15)
    assert has_recovery([T, T, F, F]) is False
    assert trajectory_is_correct([T, T]) is True
    assert trajectory_is_correct([T, F]) is False


@pytest.mark.parametrize(
    "deltas,tau,expected",
    [
        ([-0.7, -0.7, 2.0, -0.7], 0.347, 2),   # Delta_3 fires -> steps[2] is the bad step
        ([-0.7, -0.7, -0.7], 0.347, -1),       # nothing crosses tau -> -1
        ([5.0], 0.347, 0),                     # the very first step is wrong
        ([], 0.347, -1),                       # an over-length sample scores nothing
    ],
)
def test_delta_to_predicted_label(deltas, tau, expected):
    """`i* = first i with Delta_i > tau; predicted_label = i* - 1` (§9.1), with `deltas`
    0-based over i = 1..T so the answer is just the list index."""
    assert predicted_label_from_deltas(deltas, tau) == expected


def test_training_target_round_trips_to_the_right_prediction():
    """The round trip that makes the off-by-one fatal (§15).

    A model trained to spike at Delta_{z+1} produces i* = z+1, and predicted_label = i*-1 = z
    -- correct. Training on Delta_z instead gives predicted_label = z-1, wrong on EVERY
    errored sample, which zeroes acc_error and collapses F1 through the harmonic mean while
    every loss curve still looks healthy.
    """
    labels = [T, T, F, F]
    z = first_error_index(labels)
    trained_right = [-0.7] * len(labels)
    trained_right[z] = 2.0          # deltas[z] IS Delta_{z+1}
    assert predicted_label_from_deltas(trained_right, 0.347) == z

    trained_wrong = [-0.7] * len(labels)
    trained_wrong[z - 1] = 2.0      # spiking at Delta_z
    assert predicted_label_from_deltas(trained_wrong, 0.347) == z - 1 != z
