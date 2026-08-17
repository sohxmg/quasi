"""The three distance matrices, built once each (§7.1, PLAN 'Core design decisions' 3).

| tensor                            | shape          | grad | consumers                        |
|-----------------------------------|----------------|------|----------------------------------|
| Dist[r,c]   = d(phi_r, psi(g_c))  | R x C ~ 348x172| yes  | (1) L_NCE and (3) L_T -- THE SAME|
|                                   |                |      | tensor object, asserted by identity|
| Next[r,c]   = d(psi(s_r), psi(g_c))| R x C         | no   | (3) L_T only. Unconditionally    |
|                                   |                |      | detached (tmd.py:113), so no_grad|
|                                   |                |      | also halves its activation cost  |
| D_term[s,t] = d(psi_s, psi(s_T^t))| S x T_c ~404x28| yes  | (5) L_step, (6) L_good,          |
|                                   |                |      | diagnostics #2/#3/#14, §10.1 gate |
| D_term_good[s,t]                  | S x T_c        | yes  | (6) L_good only. D_term itself   |
|                                   |                |      | unless good_loss.detach_goal     |

`Dist` is RECTANGULAR and has NO DIAGONAL: source rows from incorrect trajectories are
negative-only -- they have no goal of their own and no positive column. Every "diagonal" in
`tmd.py` becomes a `pos_row` gather here. **Never call `torch.diagonal` on Dist** (grep test).

`D_term` is not in CLAUDE.md; it is what makes the design cheap. L_step, L_good and the
three-way Delta histogram (diagnostic #14, "the single best predictor of ProcessBench F1")
read the same small matrix, and it is the EVAL-shaped query: Delta against a real correct
terminal.

`step_deltas` below is the ONE place that turns `D_term` into `Delta_i`. It was extracted
from `diagnostics/probes.py` on 2026-07-28 when (6) `L_good` (§7.12) needed the same tensor
WITH a gradient: the probe panel detaches it, the loss does not. Anything that computes
`D_term[row_dst] - D_term[row_src]` by hand is a second definition of the eval statistic and
will drift from the one the loss trains -- there must not be one.
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
    D_term_good: Tensor     # (S, T_c) -- D_term itself unless good_loss.detach_goal (§7.12)
    pos_row: Tensor         # (C,) the source row that sampled goal column c
    goal_step: Tensor       # (C,) i of s_i within its trajectory -- L_NCE's nearer-set (§9.9.2)
    goal_is_terminal: Tensor # (C,) bool: the goal is its trajectory's ending (§9.9.5)
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

    if cfg.losses.good_loss.detach_goal:
        # Same shape as `stopgrad_psi_backup` above: a SECOND tensor, so (5) L_step and the
        # probes keep reading the attached one. §16.17's leak runs the other way for (6)
        # L_good -- it pulls d(psi_i, g) DOWN, and with g attached the cheapest way to do
        # that is to drag the correct terminals onto the trajectory, which is exactly what
        # phase 2's goal head must later predict. OFF by default (§7.12).
        D_term_good = distance(psi[:, None, :], psi_term.detach()[None, :, :])
    else:
        D_term_good = D_term

    return Matrices(
        Dist=Dist,
        Dist_backup=Dist_backup,
        Next=Next,
        D_term=D_term,
        D_term_good=D_term_good,
        pos_row=goals.pos_row,
        goal_step=goals.goal_step,
        goal_is_terminal=goals.is_terminal,
        SQ=same_question_mask(batch, goals),
        terminal_states=terminal_states,
        terminal_traj=terminal_traj,
    )


# --------------------------------------------------------------------------------------
# Delta over D_term -- the eval-shaped query, defined ONCE (§7.12)
# --------------------------------------------------------------------------------------


@dataclass
class StepDeltas:
    """`Delta[r, t] = d(psi_i, g_t) - d(psi_{i-1}, g_t)` for source row r = (i-1 -> i).

    Columns are restricted by `valid` to correct terminals of the SAME question that are
    NOT the row's own trajectory: a trajectory's distance to its own ending runs to
    `d(x, x) = 0` and would put a spurious spike in every statistic computed here.

    The four masks partition the valid entries by §6.1's index convention, and they are the
    same four diagnostic #14 splits three ways plus the post-error tail:

        good_correct    every i of a CORRECT trajectory
        good_incorrect  i <= z of an INCORRECT one          (still on-track)
        boundary        i == z+1                            (5) L_step's own pair
        post_error      i >  z+1                            (already broken)

    (6) `L_good` (§7.12) trains `good_correct | good_incorrect`. It must NOT touch
    `boundary` -- that is L_step's pair and the two terms pull it in opposite directions --
    and it must not touch `post_error`, where no loss defines what Delta should be.
    """

    delta: Tensor           # (R, T_c)
    valid: Tensor           # (R, T_c) bool
    good_correct: Tensor    # (R, T_c) bool
    good_incorrect: Tensor  # (R, T_c) bool
    boundary: Tensor        # (R, T_c) bool
    post_error: Tensor      # (R, T_c) bool

    def detach(self) -> "StepDeltas":
        """For the probe panel. The masks are bool and carry no grad already."""
        return StepDeltas(
            delta=self.delta.detach(),
            valid=self.valid,
            good_correct=self.good_correct,
            good_incorrect=self.good_incorrect,
            boundary=self.boundary,
            post_error=self.post_error,
        )

    def good(self, include_incorrect_prefix: bool = True) -> Tensor:
        """(6) L_good's scope. `False` drops the on-track prefix of wrong solutions."""
        if include_incorrect_prefix:
            return self.good_correct | self.good_incorrect
        return self.good_correct


def step_deltas(D_term: Tensor, batch: Batch, terminal_traj: Tensor) -> StepDeltas:
    """The single definition of `Delta_i` over `D_term`. Grad flows unless the caller cuts it.

    `probes.batch_probes` calls this and `.detach()`s; `losses/good.py` calls it and keeps
    the graph. Before 2026-07-28 the probe computed it inline and nothing else could reach
    it -- which is how a `+0.240` mean on good steps sat unopposed by any loss for a whole
    run (§7.12).
    """
    row_q = batch.traj_qid[batch.row_traj]                      # (R,)
    term_q = batch.traj_qid[terminal_traj]                      # (T_c,)
    valid = (row_q[:, None] == term_q[None, :]) & (
        batch.row_traj[:, None] != terminal_traj[None, :]
    )
    delta = D_term[batch.row_dst] - D_term[batch.row_src]       # (R, T_c)

    row_correct = batch.traj_correct[batch.row_traj]
    z = batch.traj_z[batch.row_traj]
    i = batch.row_step
    return StepDeltas(
        delta=delta,
        valid=valid,
        good_correct=valid & row_correct[:, None],
        good_incorrect=valid & (~row_correct & (i <= z))[:, None],
        boundary=valid & (~row_correct & (i == z + 1))[:, None],
        post_error=valid & (~row_correct & (i > z + 1))[:, None],
    )
