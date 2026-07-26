"""The geometric goal sampler (§7.1, ported from `datasets.py:252-281`).

    offsets        = np.random.geometric(p = 1 - discount)      # >= 1
    traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)

Two things that are easy to get wrong and are both load-bearing:

1. **`np.random.geometric` returns >= 1; `torch.distributions.Geometric` returns >= 0.**
   We use numpy, exactly as TMD does. If you ever port this to torch, add 1.

2. **The offset base is the SOURCE state index.** TMD's `idxs` is the `observations` index
   of transition `s_t -> s_{t+1}` (`datasets.py:264`). Our row phi_i is `s_{i-1} -> s_i`, so
   the base is **i-1, not i** -- which is what makes `offset = 1` mean "the goal is s_i", so
   `Next = d(x,x) = 0` and the backup target `-log gamma` are consistent. This reproduces
   §4.5's measured "goal = next state 61.5%" at discount 0.5.

Only source rows of CORRECT trajectories get a goal column. Rows of incorrect trajectories
are negative-only, which is why `Dist` is rectangular and has no diagonal (§7.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .collate import Batch


@dataclass
class GoalIndex:
    goal_state: Tensor    # (C,) flat state index of each goal column
    pos_row: Tensor       # (C,) the source row that sampled column c
    goal_traj: Tensor     # (C,) trajectory of column c
    goal_step: Tensor     # (C,) i of s_i within its trajectory
    offsets: Tensor       # (C,) the drawn geometric offsets, for diagnostics
    is_terminal: Tensor   # (C,) bool: the goal is the trajectory's ending (diagnostic #16)

    @property
    def n_goals(self) -> int:
        return int(self.goal_state.numel())

    def to(self, device) -> "GoalIndex":
        return GoalIndex(
            **{
                k: (v.to(device, non_blocking=True) if isinstance(v, Tensor) else v)
                for k, v in self.__dict__.items()
            }
        )


def sample_geometric_offsets(
    rng: np.random.Generator, n: int, discount: float
) -> np.ndarray:
    """`np.random.geometric(p = 1 - discount)`, values in [1, inf) (`datasets.py:263`)."""
    return rng.geometric(p=1.0 - discount, size=n)


def sample_goals(batch: Batch, discount: float, rng: np.random.Generator) -> GoalIndex:
    """One goal column per source row of a correct trajectory."""
    row_traj = batch.row_traj.numpy()
    row_step = batch.row_step.numpy()
    traj_correct = batch.traj_correct.numpy()
    traj_T = batch.traj_T.numpy()
    offsets_all = batch.traj_state_offset.numpy()

    src_rows = np.nonzero(traj_correct[row_traj])[0]
    if src_rows.size == 0:
        empty = torch.zeros(0, dtype=torch.long)
        return GoalIndex(empty, empty, empty, empty, empty, torch.zeros(0, dtype=torch.bool))

    trajs = row_traj[src_rows]
    base = row_step[src_rows] - 1                      # i-1: the SOURCE state index
    offsets = sample_geometric_offsets(rng, src_rows.size, discount)
    goal_step = np.minimum(base + offsets, traj_T[trajs])
    goal_state = offsets_all[trajs] + goal_step

    return GoalIndex(
        goal_state=torch.as_tensor(goal_state, dtype=torch.long),
        pos_row=torch.as_tensor(src_rows, dtype=torch.long),
        goal_traj=torch.as_tensor(trajs, dtype=torch.long),
        goal_step=torch.as_tensor(goal_step, dtype=torch.long),
        offsets=torch.as_tensor(offsets, dtype=torch.long),
        is_terminal=torch.as_tensor(goal_step == traj_T[trajs], dtype=torch.bool),
    )


def same_question_mask(batch: Batch, goals: GoalIndex) -> Tensor:
    """`SQ[r, c] = question(source r) == question(goal c)` (§7.1). Shape (R, C), bool.

    Used by L_T's `goal_scope_ratio` and by diagnostic #13. Note SQ INCLUDES the matched
    pairs, exactly as TMD's diagonal sits inside its full matrix (`tmd.py:121`).
    """
    row_q = batch.traj_qid[batch.row_traj]         # (R,)
    goal_q = batch.traj_qid[goals.goal_traj]       # (C,)
    return row_q[:, None] == goal_q[None, :]


def correct_terminal_columns(batch: Batch) -> tuple[Tensor, Tensor]:
    """(state indices of the terminals of CORRECT trajectories, their trajectory indices).

    These are `D_term`'s columns and L_step's goals `g` (§7.6.3): real terminals taken from
    the batch, never a prediction, never a centroid, never a geometric-sampler column.
    """
    correct = torch.nonzero(batch.traj_correct, as_tuple=False).flatten()
    return batch.traj_terminal[correct], correct
