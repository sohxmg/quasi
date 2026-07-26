"""Index conventions. Every off-by-one in this project lives here (§6.1).

    [prompt] SEP [step 1] SEP [step 2] SEP ... [step T] SEP
              ^            ^            ^                ^
             s_0          s_1          s_2              s_T

    completions[k] / labels[k]  is the step that takes  s_k -> s_{k+1}     (k 0-based)
    z = first k with labels[k] == False                                    (0-based)
      => psi_z     is the LAST GOOD state
      => psi_{z+1} is the FIRST BROKEN state
    phi_i = phi(h_{i-1}, act_emb_i)  for i = 1..T   (source state s_{i-1}, lands in s_i)

    Delta_i = d_i - d_{i-1}  for i = 1..T  is the cost of steps[i-1], so
    i* = first i with Delta_i > tau   =>   predicted_label = i* - 1   (or -1)

NEVER write psi_{z-1}: it is a *good* state, and it does not exist for the 45.4% of
incorrect trajectories with z = 0. A model trained on Delta_z instead of Delta_{z+1}
predicts z-1 on every errored sample, which zeroes acc_error and collapses F1 through
the harmonic mean while every loss curve still looks healthy (§7.6).
"""

from __future__ import annotations

from typing import Optional, Sequence


def first_error_index(labels: Sequence[bool]) -> Optional[int]:
    """z: the first False label, 0-based. None if the trajectory is fully correct.

    §16.15: the 1.48% of trajectories with a False -> True recovery still take their FIRST
    False, which matches ProcessBench semantics.
    """
    for k, ok in enumerate(labels):
        if not ok:
            return k
    return None


def has_recovery(labels: Sequence[bool]) -> bool:
    """True if some False is followed by a True (§4.2: 1.48% of trajectories)."""
    seen_false = False
    for ok in labels:
        if not ok:
            seen_false = True
        elif seen_false:
            return True
    return False


def trajectory_is_correct(labels: Sequence[bool]) -> bool:
    """`all(labels) == labels[-1]` holds for 100.0% of math-shepherd rows (§4.2), but we
    compute `all()` rather than relying on it."""
    return bool(len(labels) > 0 and all(labels))


def predicted_label_from_deltas(deltas: Sequence[float], tau: float) -> int:
    """§9.1 steps 5-6. `deltas[j]` is Delta_{j+1}, i.e. the cost of steps[j].

    Returns the 0-based index of the first step whose Delta exceeds tau, or -1 if none
    does. Because Delta_i is the cost of steps[i-1], the first i with Delta_i > tau maps to
    predicted_label = i - 1 -- which, with `deltas` 0-based over i = 1..T, is just j.
    """
    for j, delta in enumerate(deltas):
        if delta > tau:
            return j
    return -1


def state_index(traj_state_offset: int, i: int) -> int:
    """Flat index of state s_i of a trajectory whose s_0 sits at `traj_state_offset`."""
    return traj_state_offset + i
