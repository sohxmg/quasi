"""Does (7) `L_term` have a printed-answer shortcut? (§7.13.1, §16.26.)

**Read this before raising `lambda_term` above 0.0.**

`L_term` (§7.13) makes every correct terminal of a question one equivalence class and every
incorrect terminal a negative. In Math-Shepherd every correct solution ends by printing the
same answer -- `... The answer is: 85` -- and the incorrect ones mostly print a different one
or, when the solution was truncated, print none at all. **So the encoder can solve `L_term` by
reading the final number and clustering on it: learning to match a printed string, not to
judge reasoning.** That transfers to nothing. A PRM scores *unfinished* solutions, where no
answer has been printed yet, which is exactly the regime the shortcut says nothing about.

This is §7.5.6's lexical shortcut one level up -- there the danger was `act_emb`'s token
overlap ranking a CF positive above a CF negative; here it is the printed answer ranking a
correct sibling above an incorrect one -- and it is measured the same way: **a rank statistic
with a chance level of 0.5, reported in both directions.**

    positive pair = (correct, correct) of one question      -- L_term pulls these together
    negative pair = (correct, incorrect) of one question     -- L_term pushes these apart

    auc = P( s(positive pair) > s(negative pair) ),  ties at 0.5

**Why chance is exactly 0.5, at any number of pairs.** If the surface score `s` is independent
of the pair's label, then `P(s_pos > s_neg) = P(s_neg > s_pos)`, and those two plus
`P(s_pos = s_neg)` sum to 1; the tie-at-half convention gives
`auc = P(s_pos > s_neg) + 0.5*P(s_pos = s_neg) = 0.5`. That holds for ANY counts and any
distribution of `s`, which is the whole reason a rank statistic replaced a hit rate whose
chance level moved with the counts (§7.5.6). **A deviation in EITHER direction is a finding**;
1.0 means the shortcut fully determines the class structure and 0.0 means it determines it
inverted, which would be just as exploitable.

Three statistics, all on raw text, no model:

* `answer_match_auc` -- `s = 1` if the two solutions print the same final answer, else 0. How
  available the shortcut is. **This is the headline number.**
* `masked_overlap_auc` -- `s` = token overlap between the two full solution texts with the
  answer span DELETED. How much of the class structure survives once the shortcut is removed
  (the brief's "strip the final answer span and report what survives").
* `unmasked_overlap_auc` -- the same with the span left in, as the control. The pair
  (unmasked, masked) is what makes the masked number readable: a drop toward 0.5 says the
  surface structure WAS the answer.

**What this does and does not measure.** It measures the shortcut's AVAILABILITY in the data,
not whether `psi` takes it -- exactly as §7.5.6's AUC measures the CF data and not the
encoder. Measuring EXPLOITATION needs a trained checkpoint scored twice, once with the answer
span masked out of the input, and `gate/recall_at_1` compared across the two; that is the
follow-up and it is not run here. **Report this number; do not gate on it.** §7.5.6 records
what gating on an unmeasured rate costs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

# Math-Shepherd's terminal marker, verified on the train split 2026-08-08: a correct solution
# ends `... The answer is: 85`, and a truncated one ends without the phrase at all. The
# ABSENCE is itself a shortcut -- "does this terminal contain `The answer is:`" separates many
# correct endings from incorrect ones on its own -- so it is reported separately rather than
# quietly folded in.
_ANSWER = re.compile(r"The answer is:\s*(?P<answer>.*?)\s*$", re.IGNORECASE | re.DOTALL)

# The same regex split `scripts/generate_counterfactuals.py:_FallbackTokenizer` uses: keep
# operators, split digits individually. Much closer to what `act_emb` averages than
# `normalise_step`, which deletes every operator. NOT the Qwen tokenizer, and the report says
# so wherever it prints (§7.5.6 ran on this same fallback for the same reason).
_TOKEN = re.compile(r"\d|[A-Za-z]+|[^\sA-Za-z0-9]")


def final_answer(solution_steps: Sequence[str]) -> str | None:
    """The printed answer of a solution, or None if it never printed one."""
    if not solution_steps:
        return None
    match = _ANSWER.search(solution_steps[-1])
    if match is None:
        return None
    answer = match.group("answer").strip()
    return answer or None


def strip_answer_span(text: str) -> str:
    """`text` with `The answer is: ...` and everything after it removed."""
    return _ANSWER.sub("", text).strip()


def solution_text(solution_steps: Sequence[str]) -> str:
    """What the terminal state attends over: the whole solution, not just its last step.

    Under §6.1's one-sequence-per-trajectory construction `s_T` sits at the final separator
    and attends over the question and every step, so the overlap statistics are computed on
    the whole thing. (The question itself is shared by every trajectory of a question and
    would be a constant added to both sides of every pair, so it is left out.)
    """
    return "\n".join(solution_steps)


def token_set(text: str) -> frozenset[str]:
    """Bag of tokens, computed ONCE per trajectory and reused across every pair it joins."""
    return frozenset(_TOKEN.findall(text))


def token_overlap(a: frozenset[str] | str, b: frozenset[str] | str) -> float:
    """Jaccard over the regex fallback tokenizer. Accepts raw text or a prepared token set."""
    sa = token_set(a) if isinstance(a, str) else a
    sb = token_set(b) if isinstance(b, str) else b
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def rank_auc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """`P(positive > negative)`, ties at 0.5. None when either side is empty.

    Same statistic and same tie convention as `scripts/generate_counterfactuals.py:rank_auc`
    -- the two must be read against the same chance level, and §7.5.6's numbers came from that
    one. It is reimplemented rather than imported because `scripts/` is not a package and the
    CF generator is out of scope to edit; `tests/test_terminal_shortcut.py` pins the two
    against each other on random inputs so they cannot drift.

    **Computed by midranks, NOT by the double loop.** The CF generator scores tens of examples
    and the O(P*N) form is free there; this scores every within-question pair of the whole
    train split -- ~1e5 positives against ~2e5 negatives, i.e. 2e10 comparisons, which does
    not finish. The Mann-Whitney identity

        auc = (sum of the positives' midranks - P(P+1)/2) / (P*N)

    gives the identical value in O((P+N) log(P+N)), ties included: a tie group occupying ranks
    `start+1 .. start+count` contributes `start + (count+1)/2` to each of its members, which is
    exactly the 0.5 the double loop credits each tied comparison.
    """
    if not positives or not negatives:
        return None
    import numpy as np

    n_pos, n_neg = len(positives), len(negatives)
    combined = np.concatenate(
        [np.asarray(positives, dtype=np.float64), np.asarray(negatives, dtype=np.float64)]
    )
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    midranks = (starts + (counts + 1) / 2.0)[inverse]
    rank_sum = float(midranks[:n_pos].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@dataclass(frozen=True)
class ShortcutReport:
    """Every field is a measurement. `None` means "not computable on this sample", never 0."""

    questions: int                       # questions contributing at least one pair of each kind
    positive_pairs: int
    negative_pairs: int

    answer_match_auc: float | None       # chance 0.5 -- THE HEADLINE
    answer_match_rate_positive: float | None   # P(same printed answer | correct, correct)
    answer_match_rate_negative: float | None   # P(same printed answer | correct, incorrect)

    masked_overlap_auc: float | None     # chance 0.5 -- what survives with the answer removed
    unmasked_overlap_auc: float | None   # chance 0.5 -- the control

    prints_answer_correct: float | None      # fraction of CORRECT solutions printing one
    prints_answer_incorrect: float | None    # fraction of INCORRECT solutions printing one

    tokenizer: str

    def lines(self) -> list[str]:
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        return [
            f"questions {self.questions}  positive_pairs {self.positive_pairs}  "
            f"negative_pairs {self.negative_pairs}",
            f"answer_match_auc      {fmt(self.answer_match_auc)}   (chance 0.500)",
            f"  same answer | (correct, correct)    {fmt(self.answer_match_rate_positive)}",
            f"  same answer | (correct, incorrect)  {fmt(self.answer_match_rate_negative)}",
            f"unmasked_overlap_auc  {fmt(self.unmasked_overlap_auc)}   (chance 0.500)",
            f"masked_overlap_auc    {fmt(self.masked_overlap_auc)}   (chance 0.500)",
            f"prints an answer: correct {fmt(self.prints_answer_correct)}  "
            f"incorrect {fmt(self.prints_answer_incorrect)}",
            f"tokenizer: {self.tokenizer}",
        ]


def terminal_shortcut_report(questions) -> ShortcutReport:
    """Score the shortcut over `questions`, each a `data.math_shepherd.Question`.

    Only questions with >= 2 correct and >= 1 incorrect trajectory contribute: those are
    exactly the questions `L_term` scores at all (§7.13 -- fewer than 2 correct is skipped and
    counted), so the statistic is measured on the population the loss actually sees rather
    than on the dataset as a whole.
    """
    answer_pos: list[float] = []
    answer_neg: list[float] = []
    masked_pos: list[float] = []
    masked_neg: list[float] = []
    plain_pos: list[float] = []
    plain_neg: list[float] = []
    printed_correct: list[float] = []
    printed_incorrect: list[float] = []
    n_questions = 0

    for question in questions:
        correct = [t for t in question.trajectories if t.correct]
        incorrect = [t for t in question.trajectories if not t.correct]
        printed_correct += [float(final_answer(t.steps) is not None) for t in correct]
        printed_incorrect += [float(final_answer(t.steps) is not None) for t in incorrect]
        if len(correct) < 2 or not incorrect:
            continue
        n_questions += 1

        def features(trajectory):
            """Tokenise ONCE per trajectory; a trajectory joins many pairs."""
            text = solution_text(trajectory.steps)
            return (
                final_answer(trajectory.steps),
                token_set(text),
                token_set(strip_answer_span(text)),
            )

        c = [features(t) for t in correct]
        w = [features(t) for t in incorrect]

        for i, (a_ans, a_text, a_masked) in enumerate(c):
            for b_ans, b_text, b_masked in c[i + 1:]:
                # A missing answer never MATCHES, not even another missing answer: two
                # solutions that both stopped early have not agreed on anything.
                answer_pos.append(float(a_ans is not None and a_ans == b_ans))
                plain_pos.append(token_overlap(a_text, b_text))
                masked_pos.append(token_overlap(a_masked, b_masked))
            for b_ans, b_text, b_masked in w:
                answer_neg.append(float(a_ans is not None and a_ans == b_ans))
                plain_neg.append(token_overlap(a_text, b_text))
                masked_neg.append(token_overlap(a_masked, b_masked))

    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return ShortcutReport(
        questions=n_questions,
        positive_pairs=len(answer_pos),
        negative_pairs=len(answer_neg),
        answer_match_auc=rank_auc(answer_pos, answer_neg),
        answer_match_rate_positive=mean(answer_pos),
        answer_match_rate_negative=mean(answer_neg),
        masked_overlap_auc=rank_auc(masked_pos, masked_neg),
        unmasked_overlap_auc=rank_auc(plain_pos, plain_neg),
        prints_answer_correct=mean(printed_correct),
        prints_answer_incorrect=mean(printed_incorrect),
        tokenizer="fallback-regex (NOT Qwen -- install transformers for the real statistic)",
    )
