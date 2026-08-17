"""(7) L_term -- multi-positive counterfactual loss over ONE QUESTION'S terminals (§7.13,
§16.26). BUILT NOW, `lambda_term = 0.0`. **It ships at zero.**

This is `L_CF` (§7.5.1) with a different notion of "variant" and NOTHING else. `L_CF` asks
*"these are the same ACTION"*; `L_term` asks *"these are the same ENDING"*:

    equivalence class C_q = { psi(s_T^c) : c a CORRECT trajectory of question q }
    negatives       N_q   = { psi(s_T^w) : w an INCORRECT trajectory of question q }

and hands `(phi, group, kind)` straight to `counterfactual_loss` with `group = traj_qid`.
There is no second loss here, no `mode=` flag on the first one, and no copy of the SupCon
arithmetic -- the loss was already agnostic to what a "variant" is (a flat `(V, D)` tensor, a
group id per row, a kind per row). **Do not fork it.** §7.5.1's two REQUIREMENT properties are
requirements here too and they carry over unchanged because the code does:

* **Positives stay in the DENOMINATOR.** All correct solutions of one question are ONE class,
  not `|P|` independent pull-together problems. `dL_q/ds_p = softmax_p - 1/|P|`, so collapsing
  one correct terminal onto another while the rest stay far RAISES the loss.
* **NEGATIVES ARE NEVER QUERIES.** Two solutions that are wrong in different ways end at
  different wrong answers; nothing may pull them together. `tests/test_terminal_class.py`
  pins this the same way `tests/test_counterfactual.py` does -- permuting the negatives
  leaves the loss bit-identical.

**Why it exists (§16.26).** Phase 2 asks `goal_head(h_{s_0})` to predict ONE vector per
question, so for that target to be well-posed a question's correct terminals must be one
point. §10.1.1 wrote down in advance that *nothing in the loss set pulls two correct terminals
of one question together*, and §9.8.3 measured the consequence: `gate/recall_at_1` 0.618
untrained -> 0.276 trained, `within_question_terminal_spread` 2.905 across 525 terminals.
Every mask in §9.9 says *"stop pushing these apart"*; this is the only term that says *"pull
these together."*

**THIS TERM IS A HYPOTHESIS, NOT A FIX**, and §16.26 states the order of work plainly: the
§9.9 NCE masks are judged FIRST, because they are free and unambiguously correct, and **if
`within_question_terminal_spread` and `gate/recall_at_1` do not move under the masks they will
not move under this either.** Building the term does not override that order.

**Which latent, and what it costs.** The terminal `psi` -- `psi[batch.traj_terminal]`, the
same tensor `D_term`'s columns are gathered from (`losses/matrix.py`), not a fresh head.
`psi` is attached on this path (the only terminal-side `.detach()` in the codebase is
`good_loss.detach_goal`, which builds a SECOND tensor `D_term_good` and leaves `D_term`
alone -- §7.12), so nothing needed undetaching. **ZERO EXTRA LM FORWARDS:** `traj_qid`,
`traj_correct` and `traj_terminal` are already built by `data/collate.py` and every terminal
is already in `psi`; this term is an index_select plus a (Q_queries x width) distance, where
width is the largest per-question trajectory count (<= 7 under §8.1's `min(4,k_c)+min(3,k_i)`
caps). Confirmed by reading `collate.py` before writing this file: the `question_index` per
sequence was ALREADY carried through as `Batch.traj_qid`, so none had to be added.

**The shortcut you must read before turning this on.** Every correct trajectory of a question
ends with the same printed answer (`The answer is: 60`) and the incorrect ones mostly do not,
so the encoder can solve this loss by clustering on the final number -- learning to match a
printed string, not to judge reasoning, which transfers to nothing because a PRM scores
*unfinished* solutions. That is §7.5.6's lexical shortcut one level up.
`diagnostics/terminal_shortcut.py` measures its AVAILABILITY on the raw text with a fixed
chance level of 0.5; §7.13.1 records the value. It is REPORTED, not gated on -- §7.5.6
records what gating on an unmeasured rate costs.

**`tau` WAS a plain argument at 1.0 and is now `losses.lambda_term_temperature` (2026-08-15),
because the weight was raised and this file's own rule is that whoever raises it PICKS `tau` in
the same change rather than inheriting an unexamined default. A defaulted argument nobody has
to type is exactly what "unexamined" means, so it became a key the moment it mattered.**

**The pick is 1.0**, and it is §7.5.13's argument on a smaller denominator: a query here scores
`c_q - 1 + w_q` candidates, ~4.9 at §8.1's caps against (4)'s ~8. §9.10.1 measured
distance-space spread at 0.31 -> 0.76, so at `tau = sqrt(512)` the logits span ~0.034 across ~5
candidates -- a flat softmax pinned at `log(c-1+w)`, i.e. an off switch, not a temperature.
`sqrt(512)` has provenance for (1) alone (`tmd.py:92`) and there is no TMD counterpart to this
term to inherit from. `tau` and `lambda` are the same knob (gradient ~ `lambda/tau`), so moving
one is moving the other (§9.10 is what bundling costs).
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..data.collate import Batch
from ..model.distances import Distance
from .counterfactual import counterfactual_loss

# The `cf/*` key set, renamed. The two losses ARE the same loss, so the diagnostics are the
# same diagnostics and this mapping is the whole difference. Any key added to
# `counterfactual.py` must be added here too -- `tests/test_terminal_class.py` asserts the
# mapping covers `counterfactual_loss`'s output exactly, in both the populated and the empty
# path, so a new `cf/*` key cannot appear without a `term/*` twin.
_CF_TO_TERM = {
    "cf/loss": "term/loss",
    # `counterfactual_loss` computes this generically as mean over kept groups of
    # log(|denominator|), which for (7) IS `terminal_class_chance`'s
    # mean over questions with c >= 2 of log(c - 1 + w) -- the same number by construction.
    # The explicit `info["term/chance"] = terminal_class_chance(...)` below therefore
    # overwrites this with an identical value and stays the authority (it is separately
    # tested, and it is derived from the counts rather than from the padded mask).
    "cf/chance": "term/chance",
    "cf/examples": "term/questions",
    "cf/positive_distance": "term/positive_distance",
    "cf/negative_distance": "term/negative_distance",
    "cf/positives_per_example": "term/positives_per_question",
    "cf/negatives_per_example": "term/negatives_per_question",
}


def terminal_class_chance(n_correct: Tensor, n_incorrect: Tensor) -> float:
    """The value `term/loss` takes when `d` carries no preference at all -- the level to read
    `term/loss` against, and it is NOT a constant.

    A query of question `q` scores `(c_q - 1)` positives and `w_q` negatives in one softmax,
    so a flat softmax gives `L_q = log(c_q - 1 + w_q)`; every query of a question gets the
    same value, and the loss means over questions. Hence

        chance = mean over questions with c >= 2 of  log(c - 1 + w)

    which moves with the batch's ragged `min(4,k_c)/min(3,k_i)` counts (§8.1) -- unlike
    `L_NCE`'s `log(R)`, there is no single number to quote. Logging it beside `term/loss` is
    the §7.5.6 discipline: a statistic is only readable against its own chance level.
    """
    keep = n_correct >= 2
    if not bool(keep.any()):
        return 0.0
    candidates = (n_correct[keep] - 1 + n_incorrect[keep]).to(torch.float64)
    return float(torch.log(candidates).mean())


def terminal_class_index(batch: Batch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """`(group, kind, n_correct, n_incorrect)` for the batch's terminals, one row per
    TRAJECTORY; the two counts are per QUESTION.

    `group` is `batch.traj_qid` -- the grouping is by QUESTION, never by trajectory, so two
    solutions of different questions can never share a class however they interleave in the
    batch.

    `kind` follows `counterfactual_loss`'s convention: **0** for the first correct trajectory
    of each question in batch order, **1** for its other correct trajectories, **2** for every
    incorrect one. Which correct terminal gets slot 0 is arbitrary and must not matter -- the
    anchor is a query like any other class member, so the loss is symmetric in the class
    (pinned by a test).

    Questions with **fewer than 2 correct terminals** -- `n_correct < 2`, which the caller
    counts and logs as `term/questions_skipped_single_correct` -- have nothing to pull
    together: the class is a singleton, its only query has `|P| = 0`, and
    `counterfactual_loss` drops it rather than crashing. That drop is relied on, so it is
    COUNTED and logged rather than left silent (§14: a guard that fails toward "healthy" is
    worse than no guard). A question with ZERO correct terminals -- which §8.1's carry rule
    permits for a partially included question -- lands in the same count: it contributes no
    query at all, and its incorrect terminals sit in the batch pushing against nothing.
    """
    device = batch.traj_qid.device
    q = batch.traj_qid.long()
    if q.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty.clone(), empty.clone(), empty.clone()

    n_questions = int(q.max()) + 1
    kind = torch.full_like(q, 2)

    correct_rows = torch.nonzero(batch.traj_correct, as_tuple=False).flatten()
    n_correct = torch.bincount(q[correct_rows], minlength=n_questions)
    n_incorrect = torch.bincount(q, minlength=n_questions) - n_correct
    if correct_rows.numel():
        # Rank each correct trajectory within its question, in batch order. Same vectorised
        # slot assignment `counterfactual_loss` uses; a Python loop over Q is the wrong shape.
        q_correct = q[correct_rows]
        order = torch.argsort(q_correct, stable=True)
        starts = torch.cumsum(n_correct, dim=0) - n_correct
        rank = torch.empty_like(q_correct)
        rank[order] = torch.arange(q_correct.numel(), device=device) - starts[q_correct[order]]
        kind[correct_rows] = torch.where(rank == 0, 0, 1)

    return q, kind, n_correct, n_incorrect


def terminal_class_loss(
    psi: Tensor,
    batch: Batch,
    distance: Distance,
    temperature: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """SupCon over each question's correct terminals, against its incorrect ones.

    Thin by design: build `(phi, group, kind)` and call `counterfactual_loss`. `psi` is the
    full `(S, D)` state latent; the terminals are gathered with `batch.traj_terminal`.
    """
    psi_terminal = psi.index_select(0, batch.traj_terminal)          # (B, D)
    group, kind, n_correct, n_incorrect = terminal_class_index(batch)
    loss, cf_info = counterfactual_loss(psi_terminal, group, kind, distance, temperature)

    info = {_CF_TO_TERM[key]: value for key, value in cf_info.items()}
    info["term/questions_skipped_single_correct"] = float(int((n_correct < 2).sum()))
    info["term/chance"] = terminal_class_chance(n_correct, n_incorrect)
    # §16.26 names this as the statistic the term is supposed to move (and `gate/recall_at_1`
    # as the one that decides whether it worked). It is the SAME quantity
    # `losses/goal.py:terminal_spread_ratio` reports as `gate/within_question_terminal_spread`
    # -- the mean over ordered same-question correct-terminal pairs -- restricted to this
    # batch's terminals, which makes it identical to `term/positive_distance` by construction:
    # the class IS the correct terminals and `pos_mask` IS the within-question pairs, both
    # orderings. Logged under both names so the §16.26 statistic is plottable per step during
    # phase 1 (the gate version only runs in phase 2 / `scripts/goal_gate.py`), and pinned
    # equal by a test so the identity cannot drift.
    info["term/within_question_terminal_spread"] = info["term/positive_distance"]
    return loss, info
