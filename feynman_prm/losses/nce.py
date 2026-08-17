"""(1) L_NCE -- the contrastive loss (§7.2).

    logits[r, c] = -Dist[r, c] / tau
    L_NCE = cross-entropy over SOURCE ROWS within each goal column, at r = pos_row[c]

**Softmax over sources per goal, not over goals per source.** That is TMD's *backward* NCE:
`tmd.py:96-98` applies the cross-entropy to `logits.T`, so optax normalises over sources
within each goal column. The spec's own annotation fixes the same direction -- negatives
"keep the goal, change (s_i, a_i)", drawn from the same trajectory at other states and from
other trajectories, "correct or incorrect soln". That the negative pool explicitly includes
incorrect solutions is a correctness signal already present in the spec.

`tau = 1.0` is a DOCUMENTED DIVERGENCE from TMD's `1/sqrt(512) = 22.6` (`tmd.py:92`). At
22.6 our O(1-10) distances become O(0.05-0.5) logits, i.e. a near-uniform softmax -- the
exact signature of old bug B10a. It is a float knob: raise it if `logit_std` blows up (§7.2).

At initialisation L_NCE should sit at chance, `log(R) ~= log(348) = 5.85`. Pinned there
**with `logit_std ~= 0`** is bug B10a and means the fp32 cast is not effective at runtime;
pinned there with `logit_std > 0` and `pos ~= neg` is geometry collapse -- check goal
duplication (§14's stuck-NCE table).

**The loss expression is NOT changed by any mask here.** `F.cross_entropy(logits.t(),
pos_row)` is the whole objective, exactly as in `tmd.py:96-98`. A soft/multi-positive target
over the trajectory prefix was designed and REJECTED on 2026-08-04 (§9.9.4): its optimum is
`d(phi_i, psi_j) = (j-i)*(-log gamma)`, which is (3) `L_T`'s ruler -- and `L_T` already
supplies that, on this same matrix. `L_NCE` does not need to learn the ruler; it needs to
stop CONTRADICTING it. That is a mask, not a new objective.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def nce_loss(
    Dist: Tensor,
    pos_row: Tensor,
    temperature: float = 1.0,
    mask_same_traj: bool = False,
    mask_nearer_same_traj: bool = False,
    mask_same_question_correct: bool = False,
    mask_sibling_correct_late: bool = False,
    sibling_late_margin: int = 1,
    row_traj: Tensor | None = None,
    goal_traj: Tensor | None = None,
    row_step: Tensor | None = None,
    goal_step: Tensor | None = None,
    goal_is_terminal: Tensor | None = None,
    row_steps_to_end: Tensor | None = None,
    row_correct: Tensor | None = None,
    SQ: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Returns (loss, diagnostics). `Dist` is the shared (R, C) matrix from matrix.py."""
    R, C = Dist.shape
    logits = -Dist / temperature                       # (R, C)
    cols = torch.arange(C, device=Dist.device)
    excluded = torch.zeros_like(logits, dtype=torch.bool)

    # The NEARER set: same trajectory as the goal, strictly later than the positive, at or
    # before the goal. Built unconditionally because `nce/argmax_in_nearer_set` is the
    # pre-flight diagnostic for §9.9.2 and has to be readable with every mask OFF.
    nearer: Tensor | None = None
    if row_traj is not None and goal_traj is not None and row_step is not None \
            and goal_step is not None:
        pos_step = row_step[pos_row]                            # (C,)
        nearer = (
            (row_traj[:, None] == goal_traj[None, :])
            & (row_step[:, None] > pos_step[None, :])
            & (row_step[:, None] <= goal_step[None, :])
        )

    if mask_same_traj:
        # locked #12 leaves this OFF: we want the negatives, and masking removes only ~6 of
        # ~347 (§16.4). Flip it to true if diagnostic #7's within-trajectory spread -> 0.
        # **This is the BLUNT switch and it is the wrong one** -- it also drops the rows
        # EARLIER than the positive, which are honest hard negatives and the only thing
        # teaching within-solution progress. Prefer `mask_nearer_same_traj` (§9.9.2).
        if row_traj is None or goal_traj is None:
            raise ValueError("nce_mask_same_traj needs row_traj and goal_traj")
        excluded |= row_traj[:, None] == goal_traj[None, :]     # (R, C)

    if mask_nearer_same_traj:
        # **§16.4's false negative, and it is the one that runs the loss (§9.9.2).**
        # Goal at s_6, sampled positive phi_3: rows phi_4, phi_5, phi_6 are on the SAME
        # trajectory and are all NEARER to s_6 than phi_3 is -- phi_6 lands ON the goal.
        # `F.cross_entropy` marks all three wrong, so the loss pushes the last state of a
        # solution away from that solution's own ending. Measured: 41.8% of columns carry at
        # least one (mean 1.65, max 11); ~0.64 rows per column, and they are the CLOSEST rows
        # in the pool, so their share of the softmax mass is far above their share of the count.
        #
        # Rows EARLIER than the positive stay: "phi_3 is nearer the ending than phi_1" is
        # true, useful, and the only within-solution gradient there is. That is the whole
        # difference from `mask_same_traj` above.
        #
        # **It also dissolves the duplicate-column contradiction, for free.** 29.6% of goal
        # columns are byte-identical to another column with a DIFFERENT positive
        # (`probe01/distinct_goal_ratio` = 0.704, and a simulation of the sampler reproduces
        # 0.703 -- it is entirely the geometric draw plus the terminal clip, §9.9.3). Column A
        # (goal s_6, positive phi_2) and column B (goal s_6, positive phi_5) currently assert
        # opposite orderings of phi_2 vs phi_5. With this mask A cannot vote on phi_5 at all,
        # B still can, and B's ordering is the one L_T agrees with.
        if nearer is None:
            raise ValueError(
                "nce_mask_nearer_same_traj needs row_traj, goal_traj, row_step, goal_step"
            )
        excluded |= nearer

    if mask_same_question_correct:
        # **FALSE NEGATIVES.** A goal column is sampled from a CORRECT trajectory of question
        # q, and the negative pool contains source rows from q's OTHER correct trajectories.
        # Two correct solutions to one problem end in the same place, so their late states are
        # LEGITIMATELY near each other's goals, and pushing them apart teaches the opposite.
        # §10.1.1 wrote the mechanism down in advance -- "nothing in the loss set pulls two
        # correct terminals of one question together" -- and the phase-2 `gate_before_fit`
        # event measured it happening: gate/recall_at_1 0.618 untrained -> 0.276 trained,
        # 62% of questions fully scattered, while gate/auc stood still at 0.904 -> 0.9065.
        #
        # Same-question INCORRECT rows are NOT masked. They are the correctness signal §7.2's
        # annotation explicitly asks for ("different trajectory, correct or incorrect soln"),
        # and they are not false negatives -- an incorrect solution genuinely does not end
        # where the correct one does.
        #
        # This is a targeted exception to locked #12, not a reversal of it, and it needs
        # sign-off. Cost is ~half of `nce/negatives_same_question` (~15 of ~350, about 4%).
        #
        # **OVER-BROAD, by §16.25's own caveat (a).** Sibling B's EARLY states are legitimately
        # far from A's terminal and the ruler says how far; only B's LATE states are false
        # negatives. `mask_sibling_correct_late` below is the position-aware version and is
        # the one to run first (§9.9.5).
        if SQ is None or row_correct is None:
            raise ValueError("nce_mask_same_question_correct needs SQ and row_correct")
        excluded |= SQ & row_correct[:, None]

    if mask_sibling_correct_late:
        # §16.25(a)'s "principled version", written out. Only the LATE states of a sibling
        # CORRECT trajectory, and only against a TERMINAL goal column -- which is 32.9% of
        # columns (`probe16/goal_is_terminal_fraction`) and 100% of the query type eval makes
        # (§7.7: the goal head predicts psi(s_T)).
        #
        # `sibling_late_margin` counts steps remaining to that row's OWN terminal, so margin 1
        # keeps {phi_T, phi_{T-1}}. Everything earlier stays a negative at the distance L_T
        # prices it at. Cost is far below the blunt mask's 4.3%.
        missing = [
            n for n, v in (
                ("SQ", SQ), ("row_correct", row_correct), ("row_traj", row_traj),
                ("goal_traj", goal_traj), ("goal_is_terminal", goal_is_terminal),
                ("row_steps_to_end", row_steps_to_end),
            ) if v is None
        ]
        if missing:
            raise ValueError(f"nce_mask_sibling_correct_late needs {', '.join(missing)}")
        late = row_steps_to_end <= sibling_late_margin              # (R,)
        excluded |= (
            SQ
            & (row_traj[:, None] != goal_traj[None, :])             # sibling, not own trajectory
            & row_correct[:, None]
            & late[:, None]
            & goal_is_terminal[None, :]
        )

    n_excluded = 0.0
    if bool(excluded.any()):
        excluded[pos_row, cols] = False                 # never mask the positive
        n_excluded = float(excluded.sum(dim=0).float().mean())
        logits = logits.masked_fill(excluded, float("-inf"))

    loss = F.cross_entropy(logits.t(), pos_row)        # normalise over rows, per column

    with torch.no_grad():
        cols = torch.arange(C, device=Dist.device)
        finite = torch.isfinite(logits)
        pos_logit = logits[pos_row, cols]
        neg_mask = torch.ones_like(logits, dtype=torch.bool)
        neg_mask[pos_row, cols] = False
        neg_mask &= finite
        predicted = logits.argmax(dim=0)                # per column, over sources
        info = {
            "nce/loss": float(loss),
            "nce/chance": math.log(R),                  # ~5.85 at R = 348
            "nce/logit_std": float(logits[finite].std()),
            "nce/logits_pos": float(pos_logit.mean()),
            "nce/logits_neg": float(logits[neg_mask].mean()),
            "nce/pos_dist": float(Dist[pos_row, cols].mean()),
            "nce/neg_dist": float(Dist[neg_mask].mean()),
            # The direction the LOSS optimises (sources within a column). TMD's own
            # categorical_accuracy measures the forward direction instead (tmd.py:127) while
            # its loss normalises over sources (tmd.py:97); we report what is optimised.
            "nce/categorical_accuracy_backward": float((predicted == pos_row).float().mean()),
            "nce/negatives_per_column": float(R - 1),
            # The pool actually used. Equal to negatives_per_column with no masking; read
            # THIS one when either mask is on, and read the pair to price what a mask cost.
            "nce/negatives_effective": float(R - 1) - n_excluded,
            "nce/negatives_masked": n_excluded,
        }

        # **The pre-flight for §9.9.2, computed on the RAW logits with every mask off.**
        # `categorical_accuracy_backward` says 65% of columns pick the wrong row; this says
        # WHERE those picks land. If a large share of them land in the nearer set, §16.4's
        # false negative is confirmed as taking the softmax mass and `mask_nearer_same_traj`
        # is aimed at the right thing. If it is ~0 the mass is going somewhere else and the
        # mask will not move anything -- find out where before spending a run.
        if nearer is not None:
            raw_pred = (-Dist / temperature).argmax(dim=0)          # pre-mask
            wrong = raw_pred != pos_row
            info["nce/nearer_set_size"] = float(nearer.sum(dim=0).float().mean())
            info["nce/columns_with_nearer"] = float(nearer.any(dim=0).float().mean())
            if bool(wrong.any()):
                hit = nearer[raw_pred[wrong], cols[wrong]]
                info["nce/argmax_in_nearer_set"] = float(hit.float().mean())

        # The same-question split -- the L_NCE counterpart of diagnostic #13, and the only way
        # to read `nce/loss` correctly. The pool is ~R/Q same-question rows plus ~R(1-1/Q)
        # cross-question ones, and those are two different problems: cross-question separation
        # is what L_NCE is for, while same-question rows include states that are GENUINELY
        # closer to the goal than the sampled positive (§16.4 -- goal at s_6, positive at s_3,
        # and s_5 sits between them). So a chunk of the same-question mass is unlearnable by
        # construction, and the loss floors at log(1 + same-question negatives) even with
        # cross-question solved perfectly. Read `loss_cross_question` and
        # `accuracy_within_question` -- NOT `nce/loss` -- to tell "still learning" from
        # "at the floor". A falling `nce/loss` with Q rising is the task getting easier, not
        # the model getting better.
        if SQ is not None:
            same_q_neg = SQ & neg_mask
            cross_q_neg = (~SQ) & neg_mask
            n_same = same_q_neg.sum(dim=0).float().mean()
            info["nce/negatives_same_question"] = float(n_same)
            info["nce/floor_same_question"] = float(torch.log1p(n_same))
            if same_q_neg.any():
                info["nce/logits_neg_same_question"] = float(logits[same_q_neg].mean())
            if cross_q_neg.any():
                info["nce/logits_neg_cross_question"] = float(logits[cross_q_neg].mean())
            # The loss with same-question negatives removed: how much of the residual is
            # cross-question discrimination that is still genuinely unsolved.
            info["nce/loss_cross_question"] = float(
                F.cross_entropy(logits.masked_fill(same_q_neg, float("-inf")).t(), pos_row)
            )
            # Ranking WITHIN the question, which is what L_T's ruler has to supply and what
            # more training can actually move. Chance is 1 / (1 + negatives_same_question).
            within = logits.masked_fill(~SQ, float("-inf"))
            info["nce/accuracy_within_question"] = float(
                (within.argmax(dim=0) == pos_row).float().mean()
            )
    return loss, info
