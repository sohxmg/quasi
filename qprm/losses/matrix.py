"""The three distance matrices, built once each (§7.1, PLAN 'Core design decisions' 3).

| tensor                            | shape          | grad | consumers                        |
|-----------------------------------|----------------|------|----------------------------------|
| Dist[r,c]   = d(phi_r, psi(g_c))  | R x C ~ 348x172| yes  | (1) L_NCE and (3) L_T -- THE SAME|
|                                   |                |      | tensor object, asserted by identity|
| Next[r,c]   = d(psi(s_r), psi(g_c))| R x C         | no   | (3) L_T only. Unconditionally    |
|                                   |                |      | detached (tmd.py:113), so no_grad|
|                                   |                |      | also halves its activation cost  |
| D_term[s,t] = d(psi_s, psi(s_T^t))| S x T_c ~404x28| yes  | (5) L_step, diagnostics #2/#3/#14,|
|                                   |                |      | the §10.1 gate                   |

`Dist` is RECTANGULAR and has NO DIAGONAL: source rows from incorrect trajectories are
negative-only -- they have no goal of their own and no positive column. Every "diagonal" in
`tmd.py` becomes a `pos_row` gather here. **Never call `torch.diagonal` on Dist** (grep test).

`D_term` is not in CLAUDE.md; it is what makes the design cheap. L_step and the three-way
Delta histogram (diagnostic #14, "the single best predictor of ProcessBench F1") read the
same small matrix, and it is the EVAL-shaped query: Delta against a real correct terminal.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..config import Config
from ..data.collate import Batch
from ..data.goals import GoalIndex, correct_terminal_columns, same_question_mask
from ..model.distances import Distance


@dataclass
class Matrices:
    Dist: Tensor            # (R, C) with grad -- shared by (1) and (3)
    Dist_backup: Tensor     # (R, C) -- Dist itself unless stopgrad_psi_backup (tmd.py:111-112)
    Next: Tensor            # (R, C) detached
    D_term: Tensor          # (S, T_c) with grad
    pos_row: Tensor         # (C,) the source row that sampled goal column c
    SQ: Tensor              # (R, C) bool: same-question
    terminal_states: Tensor # (T_c,) state index of each correct terminal
    terminal_traj: Tensor   # (T_c,) trajectory index of each correct terminal

    @property
    def n_rows(self) -> int:
        return int(self.Dist.shape[0])

    @property
    def n_goals(self) -> int:
        return int(self.Dist.shape[1])

    @property
    def matched(self) -> Tensor:
        """`div[pos_row[c], c]` for every c -- the "diagonal" of a matrix that has none."""
        return torch.arange(self.n_goals, device=self.Dist.device)


def build_matrices(
    psi: Tensor,
    phi: Tensor,
    batch: Batch,
    goals: GoalIndex,
    distance: Distance,
    cfg: Config,
) -> Matrices:
    """Build Dist, Next and D_term. One call per micro-batch."""
    psi_g = psi.index_select(0, goals.goal_state)                # (C, D)
    Dist = distance(phi[:, None, :], psi_g[None, :, :])          # (R, C)

    if cfg.losses.backup.stopgrad_psi_backup:
        # tmd.py:111-112 recomputes `dist` with the GOAL side detached, AFTER the contrastive
        # loss has used the attached one. So the flag creates a second tensor; it does not
        # change the one L_NCE reads.
        Dist_backup = distance(phi[:, None, :], psi_g.detach()[None, :, :])
    else:
        Dist_backup = Dist

    with torch.no_grad():
        psi_next = psi.index_select(0, batch.row_dst)             # (R, D) the state phi_r lands in
        Next = distance(psi_next[:, None, :], psi_g[None, :, :])  # (R, C)

    terminal_states, terminal_traj = correct_terminal_columns(batch)
    psi_term = psi.index_select(0, terminal_states)                # (T_c, D)
    D_term = distance(psi[:, None, :], psi_term[None, :, :])       # (S, T_c)

    return Matrices(
        Dist=Dist,
        Dist_backup=Dist_backup,
        Next=Next,
        D_term=D_term,
        pos_row=goals.pos_row,
        SQ=same_question_mask(batch, goals),
        terminal_states=terminal_states,
        terminal_traj=terminal_traj,
    )
