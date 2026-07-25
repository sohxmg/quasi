"""The §10 diagnostics. Build these on day one.

The old project's failure was invisible for a full training cycle because none of these
existed: it was a MODELLING failure, not a plumbing failure. Probes #9, #10, #13, #15 and
#17 are computed inside the losses that own the tensors; everything else is here.

**#14 is the one to watch** -- the three-way Delta histogram is what predicts ProcessBench
F1 (§7.6.6). L_step never sees a correct trajectory, so L_T alone suppresses false
positives; a positive Delta tail on good steps of CORRECT trajectories is F1 leaking with no
loss training against it, and that is the trigger for a §7.10 pairing expansion.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..config import Config
from ..data.collate import Batch
from ..data.goals import GoalIndex
from ..losses.matrix import Matrices
from ..model.distances import Distance


def _pearson(a: Tensor, b: Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    if a.numel() < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if float(denom) else float("nan")


@torch.no_grad()
def batch_probes(
    psi: Tensor,
    phi: Tensor,
    batch: Batch,
    goals: GoalIndex,
    matrices: Matrices,
    distance: Distance,
    cfg: Config,
) -> dict[str, float]:
    out: dict[str, float] = {}

    # ---- #1 distinct goals / realised negatives -------------------------------------
    C = goals.n_goals
    distinct = int(torch.unique(goals.goal_state).numel()) if C else 0
    out["probe01/goal_columns"] = float(C)
    out["probe01/distinct_goals"] = float(distinct)
    out["probe01/distinct_goal_ratio"] = float(distinct / C) if C else 0.0
    out["probe01/negatives_per_column"] = float(max(batch.n_rows - 1, 0))
    out["probe01/questions_in_batch"] = float(batch.n_questions)       # Q, expect ~12.9
    out["probe01/sequences_in_batch"] = float(batch.n_traj)
    out["probe01/padding_fraction"] = batch.padding_fraction

    # ---- #16 realised goal-type mix --------------------------------------------------
    # Should match §4.5 for the chosen discount: 41.1% endings at 0.5, 55.0% at 0.7. Eval
    # queries an ending EVERY time, so this gap is §16.2's train/eval mismatch.
    out["probe16/goal_is_terminal_fraction"] = (
        float(goals.is_terminal.float().mean()) if C else 0.0
    )
    out["probe16/goal_offset_mean"] = float(goals.offsets.float().mean()) if C else 0.0

    # ---- Delta over D_term: the eval-shaped query ------------------------------------
    # Delta[r, t] = d(psi_{i}, g_t) - d(psi_{i-1}, g_t), for terminal columns of the same
    # question that do not belong to the row's own trajectory (a trajectory's distance to
    # its own terminal ends at d(x,x) and would distort the histogram).
    D_term = matrices.D_term
    if D_term.numel() and batch.n_rows:
        row_q = batch.traj_qid[batch.row_traj]                      # (R,)
        term_q = batch.traj_qid[matrices.terminal_traj]             # (T_c,)
        valid = (row_q[:, None] == term_q[None, :]) & (
            batch.row_traj[:, None] != matrices.terminal_traj[None, :]
        )
        delta = D_term[batch.row_dst] - D_term[batch.row_src]       # (R, T_c)

        row_correct = batch.traj_correct[batch.row_traj]
        z = batch.traj_z[batch.row_traj]
        i = batch.row_step
        good_correct = valid & row_correct[:, None]
        good_incorrect = valid & (~row_correct & (i <= z))[:, None]
        boundary = valid & (~row_correct & (i == z + 1))[:, None]
        post_error = valid & (~row_correct & (i > z + 1))[:, None]

        def stat(mask: Tensor, name: str) -> None:
            vals = delta[mask]
            out[f"{name}/n"] = float(vals.numel())
            out[f"{name}/mean"] = float(vals.mean()) if vals.numel() else float("nan")
            out[f"{name}/std"] = float(vals.std()) if vals.numel() > 1 else float("nan")
            out[f"{name}/positive_fraction"] = (
                float((vals > 0).float().mean()) if vals.numel() else float("nan")
            )

        # #2 good steps vs the target -log gamma; #3 the gap to bad steps; #14 the three-way
        # histogram. The old project's #2 was 108x off and nobody noticed.
        stat(good_correct, "probe14/delta_good_of_correct")
        stat(good_incorrect, "probe14/delta_good_of_incorrect")
        stat(boundary, "probe14/delta_boundary")
        stat(post_error, "probe14/delta_post_error")
        out["probe02/target_good_step_delta"] = -cfg.neg_log_gamma
        good_all = delta[good_correct | good_incorrect]
        bad_all = delta[boundary]
        out["probe02/delta_good_mean"] = float(good_all.mean()) if good_all.numel() else float("nan")
        out["probe03/delta_bad_mean"] = float(bad_all.mean()) if bad_all.numel() else float("nan")
        # #3: THIS GAP IS THE SIGNAL. Collapsing toward zero means the error signal is being
        # flattened (§16.3).
        out["probe03/gap"] = (
            float(bad_all.mean() - good_all.mean())
            if good_all.numel() and bad_all.numel()
            else float("nan")
        )

        # ---- #7 within-trajectory distance spread ------------------------------------
        # If this heads to zero with masking off, states are being squashed (§16.4) and
        # nce_mask_same_traj should be flipped to true.
        d_valid = D_term[batch.row_dst] * valid
        spreads = []
        for b in range(batch.n_traj):
            rows = torch.nonzero(batch.row_traj == b, as_tuple=False).flatten()
            if rows.numel() < 2:
                continue
            cols = torch.nonzero(valid[rows[0]], as_tuple=False).flatten()
            if cols.numel() == 0:
                continue
            spreads.append(D_term[batch.row_dst[rows]][:, cols].std(dim=0).mean())
        out["probe07/within_trajectory_spread"] = (
            float(torch.stack(spreads).mean()) if spreads else float("nan")
        )
        del d_valid

        # ---- #8 corr(distance, ||psi||) ----------------------------------------------
        # r > 0.9 means the goal contributes nothing and the score is a norm in disguise --
        # root cause D recreated.
        norms = psi.index_select(0, batch.row_dst).norm(dim=-1)
        first_col = torch.argmax(valid.float(), dim=1)
        has_col = valid.any(dim=1)
        if bool(has_col.any()):
            d_first = D_term[batch.row_dst, first_col][has_col]
            out["probe08/corr_distance_psi_norm"] = _pearson(d_first, norms[has_col])

    # ---- #4 symmetric / asymmetric share ---------------------------------------------
    if goals.n_goals:
        # On the MATCHED pairs (phi_r against its own goal), which is where the reported
        # distance actually lives -- the full R x C grid would give the same ratio at 100x
        # the cost.
        psi_g = psi.index_select(0, goals.goal_state)
        out["probe04/symmetric_share"] = distance.symmetric_share(
            phi.index_select(0, goals.pos_row), psi_g
        )

    # ---- #12 on-track vs off-track states in incorrect trajectories -------------------
    # Expect ~1:2 (0.65M on-track vs 1.35M off-track, §4.2). Also replaces the dropped
    # `sampling.hard_negatives_post_error` knob: log the realised ratio instead.
    incorrect_states = ~batch.traj_correct[batch.state_traj]
    z_state = batch.traj_z[batch.state_traj]
    on_track = incorrect_states & (batch.state_step <= z_state)
    off_track = incorrect_states & (batch.state_step > z_state)
    n_on, n_off = float(on_track.sum()), float(off_track.sum())
    out["probe12/on_track_states"] = n_on
    out["probe12/off_track_states"] = n_off
    out["probe12/off_over_on"] = n_off / n_on if n_on else float("nan")

    return out


@torch.no_grad()
def asymmetry_score(psi: Tensor, batch: Batch, distance: Distance) -> dict[str, float]:
    """§9.4, DIAGNOSTIC ONLY (§16.10) -- never reported without a decision.

        irreversibility_i = d(psi_i, psi_{i-1}) - d(psi_{i-1}, psi_i)

    Large positive = that step was hard to undo = likely the error. No goal, no reference,
    no generator. It is the direct test of the asymmetry claim that justifies a quasimetric
    over a cosine similarity in the first place -- but nothing in L_NCE/L_I/L_T supervises
    reverse distances, so it is emergent, and full_mrn at D=512 is ~73% symmetric.
    """
    back = distance(psi[batch.row_dst], psi[batch.row_src])
    forward = distance(psi[batch.row_src], psi[batch.row_dst])
    irr = back - forward
    correct = batch.traj_correct[batch.row_traj]
    z = batch.traj_z[batch.row_traj]
    is_error = (~correct) & (batch.row_step == z + 1)
    return {
        "probe09_4/irreversibility_mean": float(irr.mean()),
        "probe09_4/irreversibility_error_step": float(irr[is_error].mean())
        if bool(is_error.any())
        else float("nan"),
        "probe09_4/irreversibility_good_step": float(irr[~is_error].mean())
        if bool((~is_error).any())
        else float("nan"),
    }
