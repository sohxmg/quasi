"""L_goal -- the goal head, PHASE 2 ONLY (§7.7, locked #13/#14/#15).

    pred_q    = goal_head(h_{s_0}^q)
    targets_q = { psi(s_T^c) : c a correct trajectory of q }        FROZEN

    L_goal = mean over c of [ d(pred_q, psi(s_T^c)) + d(psi(s_T^c), pred_q) ]

The mean OF DISTANCES, never a distance to a mean -- a latent-space centroid over 30k
terminals collapses onto the population mean and the distance becomes an atypicality
detector. That is old root cause D and there is no centroid anywhere in this codebase.

**Both directions, summed.** The distance is one-way, so a single direction would let the
guess drift to somewhere *reachable from* the real ending rather than *being* it.

`.detach()` is structural here: phase 1 has no goal head at all, and phase 2 freezes the
backbone, psi and phi, so the degenerate joint optimum (psi squashes every ending onto one
point, the head predicts that point, loss -> 0, every state identical) is unreachable. The
collapse test in §15 is kept anyway -- it documents why the phase split exists.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..model.distances import Distance


def goal_loss(
    pred: Tensor,
    targets: Tensor,
    target_example: Tensor,
    distance: Distance,
) -> tuple[Tensor, dict[str, float]]:
    """`pred` (N, D) one prediction per question; `targets` (M, D) frozen terminals;
    `target_example` (M,) which question each terminal belongs to."""
    p = pred.index_select(0, target_example)
    forward = distance(p, targets)
    backward = distance(targets, p)
    loss = (forward + backward).mean()

    with torch.no_grad():
        info = {
            "goal/loss": float(loss),
            "goal/d_pred_to_target": float(forward.mean()),
            "goal/d_target_to_pred": float(backward.mean()),
            # Diagnostic #6: near-zero variance means the head learned a constant, i.e. a
            # global anchor rather than a question-conditioned goal.
            "goal/pred_variance": float(pred.var(dim=0).mean()) if pred.shape[0] > 1 else 0.0,
        }
    return loss, info


def terminal_spread_ratio(
    psi_terminals: Tensor, question_index: Tensor, distance: Distance
) -> dict[str, float]:
    """§10.1's go/no-go gate (diagnostic #5). Run it on a SHORT run, before GPU hours.

        within = mean over q of mean_{j!=k} d(psi(s_T^j), psi(s_T^k))    same question
        across = mean over q!=q' of d(psi(s_T^q), psi(s_T^q'))
        ratio  = within / across

    ratio < ~0.3  -> correct endings cluster by question, the goal head has a well-defined
                     target, proceed.
    ratio -> 1    -> endings do not cluster and the goal head CANNOT work however well it is
                     trained. Fall back to the goal-free asymmetry score (§9.4) -- NOT to a
                     reference goal, which is a skyline (§5.1).
    """
    with torch.no_grad():
        d = distance(psi_terminals[:, None, :], psi_terminals[None, :, :])
        same = question_index[:, None] == question_index[None, :]
        eye = torch.eye(len(psi_terminals), dtype=torch.bool, device=d.device)
        within_mask = same & ~eye
        across_mask = ~same
        within = float(d[within_mask].mean()) if within_mask.any() else float("nan")
        across = float(d[across_mask].mean()) if across_mask.any() else float("nan")
    return {
        "gate/within_question_terminal_spread": within,
        "gate/across_question_terminal_spread": across,
        "gate/ratio": within / across if across else float("nan"),
    }


def terminal_separability(
    psi_terminals: Tensor, question_index: Tensor, distance: Distance
) -> dict[str, float]:
    """The §10.1 gate's question, asked as a SEPARABILITY question rather than a ratio.

    `terminal_spread_ratio` compares two MEANS, and in 512 dimensions that is a weak proxy
    for what the goal head actually needs. Concentration of measure makes both distance
    distributions extremely tight around their means, so a modest gap between the means is
    already total separation. Simulated on isotropic Gaussian clusters at this file's own
    shape (200 questions x ~2.7 terminals, D=512, full_mrn):

        cluster noise   ratio   AUC    recall@1
             0.4        0.370   1.000    1.00
             0.6        0.514   1.000    1.00
             0.8        0.625   1.000    1.00
             1.0        0.708   1.000    1.00
             1.3        0.794   1.000    1.00

    i.e. **ratio 0.62 is already perfect same-question retrieval.** §10.1's "< 0.3" was never
    derived from anything -- it is the one number in CLAUDE.md with no entry in §17 -- and it
    sits far inside the regime where the head has a perfectly well-defined target. The null
    is what the ratio pins down well: iid terminals with no question structure give **1.000**
    to three decimals, so the ratio's job is detecting "no signal at all", not gating.

    What the goal head needs is that a question's terminals be closer to EACH OTHER than to
    other questions' terminals, which is a ranking property:

        auc      = P( d(same-question pair) < d(different-question pair) )
        recall@1 = fraction of terminals whose nearest OTHER terminal shares its question

    Read `auc`. Chance is 0.5 and it is scale-free, so unlike the ratio it needs no null run
    to interpret. `recall@1` is the harsher version -- it is what a nearest-terminal lookup
    would score -- and it degrades first when a minority of questions are well clustered and
    the rest are at chance, which is the failure mode a mean ratio cannot see at all.

    **The table above is isotropic Gaussian clusters and the real geometry is not that.**
    Measured on the first phase-1 checkpoint: `ratio 0.599` came with `auc 0.875` and
    `recall@1 0.625`, where the simulation predicted 1.000/1.00. Clean clusters cannot produce
    that combination -- at `auc 0.875` with 531 competing across-pairs per terminal, an
    independent-distance model puts `recall@1` near zero, so getting 0.625 means the misses
    are structured rather than spread. Hence `questions_fully_clustered` /
    `questions_fully_scattered`: they separate "every question is moderately noisy" (a level
    problem -- the head still has a target everywhere, and eval's `Delta` differencing in §9.1
    absorbs a constant goal error) from "a subset of questions has no target at all" (a
    coverage problem -- the head works on that subset and eval degrades per question). The two
    call for opposite responses and the mean cannot distinguish them.
    """
    with torch.no_grad():
        n = len(psi_terminals)
        d = distance(psi_terminals[:, None, :], psi_terminals[None, :, :])
        same = question_index[:, None] == question_index[None, :]
        eye = torch.eye(n, dtype=torch.bool, device=d.device)
        w = d[same & ~eye].flatten().float()
        a = d[~same].flatten().float()
        if not len(w) or not len(a):
            return {"gate/auc": float("nan"), "gate/recall_at_1": float("nan")}

        # Exact AUC by rank, not by sampling: for each within-pair, how many across-pairs
        # exceed it. |a| is ~285k here, so the O(|w| log|a|) form is the affordable one.
        a_sorted = a.sort().values
        greater = len(a_sorted) - torch.searchsorted(a_sorted, w, right=True)
        auc = float(greater.float().mean() / len(a_sorted))

        masked = d.masked_fill(eye, float("inf"))
        nearest = masked.argmin(dim=1)
        hit = (question_index[nearest] == question_index).float()
        recall = float(hit.mean())

        # Is the miss rate spread evenly over questions, or concentrated in a subset?
        # A mean cannot tell those apart and they call for opposite responses.
        qs = torch.unique(question_index)
        per_q = torch.stack([hit[question_index == q].mean() for q in qs])
        clustered = float((per_q == 1.0).float().mean())
        scattered = float((per_q == 0.0).float().mean())

    return {
        "gate/auc": auc,
        "gate/recall_at_1": recall,
        # Diagnostic #5b: the shape of the miss rate, not its level.
        "gate/questions_fully_clustered": clustered,
        "gate/questions_fully_scattered": scattered,
        "gate/per_question_recall_std": float(per_q.std()) if len(per_q) > 1 else 0.0,
        "gate/within_pairs": len(w),
        "gate/across_pairs": len(a),
    }
