"""(7) `L_term`'s printed-answer shortcut statistic (§7.13.1). CPU, no model, no network.

The statistic decides whether anyone should train on `L_term`, so what is pinned here is the
things that would make it lie: the chance level, the tie convention, and the direction.
"""

from __future__ import annotations

import importlib.util
import random
import sys

import pytest

from feynman_prm.data.math_shepherd import Question, Trajectory
from feynman_prm.diagnostics.terminal_shortcut import (
    final_answer,
    rank_auc,
    strip_answer_span,
    terminal_shortcut_report,
    token_overlap,
)
from conftest import REPO_ROOT

T, F = True, False


def _question(qid: str, solutions):
    """`solutions` is [(labels, steps), ...]."""
    return Question(
        qid=qid,
        prompt=f"prompt {qid}",
        trajectories=[Trajectory(steps=tuple(steps), labels=tuple(labels))
                      for labels, steps in solutions],
    )


def _solution(answer: str | None, body: str = "Step 1: work", n_steps: int = 1):
    steps = [f"{body} {i}" for i in range(n_steps)]
    steps.append("Step last: so we are done." + (f" The answer is: {answer}" if answer else ""))
    return steps


# ---------------------------------------------------------------------- the pieces

def test_final_answer_reads_the_last_step_only():
    assert final_answer(["Step 1: The answer is: 3", "Step 2: done. The answer is: 7"]) == "7"
    assert final_answer(["Step 1: The answer is: 3", "Step 2: no answer here"]) is None
    assert final_answer([]) is None
    # LaTeX answers are common in the split and must survive intact.
    assert final_answer(["... The answer is: \\frac{1}{2}"]) == "\\frac{1}{2}"


def test_strip_answer_span_removes_the_phrase_and_everything_after_it():
    assert strip_answer_span("Step 5: 15 + 20 = 35. The answer is: 35") == "Step 5: 15 + 20 = 35."
    assert strip_answer_span("no answer here") == "no answer here"


def test_token_overlap_keeps_the_operators_and_splits_the_digits():
    """`normalise_step` deletes `* + - / =`, which are precisely the tokens `act_emb` sees
    (§7.5.6). This tokenizer keeps them."""
    assert token_overlap("2 + 2 = 4", "2 + 2 = 4") == 1.0
    assert token_overlap("2 + 2 = 4", "2 - 2 = 0") < 1.0
    assert token_overlap("", "anything") == 0.0


# ------------------------------------------------------------------------ rank_auc

def test_rank_auc_matches_the_cf_generators_double_loop():
    """The two statistics must be read against the same chance level and the same tie
    convention, so they must be the same number. This one is computed by midranks because the
    O(P*N) form does not finish on the whole train split (§7.13.1)."""
    _spec = importlib.util.spec_from_file_location(
        "generate_counterfactuals", REPO_ROOT / "scripts" / "generate_counterfactuals.py"
    )
    gen = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = gen
    _spec.loader.exec_module(gen)

    rng = random.Random(0)
    for trial in range(30):
        # Deliberately coarse values so there are MANY ties -- the tie convention is the part
        # a midrank implementation gets wrong, and the headline statistic is binary.
        pos = [rng.choice([0.0, 0.5, 1.0]) for _ in range(rng.randint(1, 12))]
        neg = [rng.choice([0.0, 0.5, 1.0]) for _ in range(rng.randint(1, 12))]
        assert rank_auc(pos, neg) == pytest.approx(gen.rank_auc(pos, neg), abs=1e-12), trial


def test_rank_auc_chance_is_a_half_and_the_ends_are_reachable():
    assert rank_auc([1.0, 1.0], [0.0, 0.0]) == 1.0
    assert rank_auc([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert rank_auc([1.0, 0.0], [1.0, 0.0]) == 0.5      # identical distributions -> exactly 0.5
    assert rank_auc([0.5], [0.5]) == 0.5                # all ties -> exactly 0.5
    assert rank_auc([], [1.0]) is None
    assert rank_auc([1.0], []) is None


# ------------------------------------------------------------------------- the report

def test_a_perfect_shortcut_reads_one():
    """Every correct solution prints the same answer, every incorrect one prints a different
    one: the class structure is fully determined by the final number."""
    questions = [
        _question("q1", [
            (( T, T), _solution("60")),
            (( T, T), _solution("60")),
            (( T, F), _solution("41")),
        ]),
    ]
    report = terminal_shortcut_report(questions)
    assert report.answer_match_auc == 1.0
    assert report.answer_match_rate_positive == 1.0
    assert report.answer_match_rate_negative == 0.0
    assert report.positive_pairs == 1 and report.negative_pairs == 2


def test_no_shortcut_reads_a_half():
    """The chance level is not a claim about the data -- it is a property of the statistic, so
    it must come out exactly on a fixture where the answer carries no information."""
    questions = [
        _question("q1", [
            ((T, T), _solution("60")),
            ((T, T), _solution("41")),      # correct, different printed answer
            ((T, F), _solution("60")),      # incorrect, the SAME printed answer
            ((T, F), _solution("41")),
        ]),
    ]
    report = terminal_shortcut_report(questions)
    # 1 positive pair (60 vs 41 -> no match) and 4 negative pairs, 2 of which match.
    assert report.answer_match_rate_positive == 0.0
    assert report.answer_match_rate_negative == 0.5
    # AUC = 0.5 + 0.5*(p - n) = 0.5 + 0.5*(0 - 0.5) = 0.25 -- BELOW chance, and a deviation
    # below chance is a finding exactly as a deviation above it is (§7.5.6).
    assert report.answer_match_auc == pytest.approx(0.25)


def test_the_auc_is_the_half_plus_half_the_rate_gap():
    """`auc = 0.5 + 0.5*(p - n)` for a BINARY score with ties at 0.5, so the headline number
    and the two rates beside it cannot disagree. Pinned because the report prints all three
    and a reader will check them against each other."""
    questions = [
        _question("q1", [
            ((T, T), _solution("7")), ((T, T), _solution("7")), ((T, T), _solution("9")),
            ((T, F), _solution("7")), ((T, F), _solution("4")),
        ]),
    ]
    report = terminal_shortcut_report(questions)
    expected = 0.5 + 0.5 * (
        report.answer_match_rate_positive - report.answer_match_rate_negative
    )
    assert report.answer_match_auc == pytest.approx(expected)


def test_a_missing_answer_never_matches_another_missing_answer():
    """Two solutions that both stopped early have not agreed on anything. Counting them as a
    match would move the headline AUC in the direction that says "no shortcut here", which is
    the direction a guard must never fail toward (§14)."""
    questions = [
        _question("q1", [
            ((T, T), _solution(None)),
            ((T, T), _solution(None)),
            ((T, F), _solution(None)),
        ]),
    ]
    report = terminal_shortcut_report(questions)
    assert report.answer_match_rate_positive == 0.0
    assert report.answer_match_rate_negative == 0.0
    assert report.prints_answer_correct == 0.0
    assert report.prints_answer_incorrect == 0.0


def test_masking_removes_the_answer_from_the_overlap_statistic():
    """The masked/unmasked pair is what makes "how much survives" readable. Here the ONLY
    thing distinguishing the correct siblings from the incorrect one is the printed answer,
    so masking must take the overlap AUC down to chance."""
    body = "Step 1: identical prose in every solution"
    questions = [
        _question("q1", [
            ((T, T), [body, "Step 2: done. The answer is: 60"]),
            ((T, T), [body, "Step 2: done. The answer is: 60"]),
            ((T, F), [body, "Step 2: done. The answer is: 41"]),
        ]),
    ]
    report = terminal_shortcut_report(questions)
    assert report.unmasked_overlap_auc == 1.0
    assert report.masked_overlap_auc == 0.5


def test_only_questions_the_loss_actually_scores_are_counted():
    """`L_term` skips questions with fewer than 2 correct terminals (§7.13), so the statistic
    is measured on the population the loss sees, not on the dataset as a whole."""
    questions = [
        _question("q1", [((T, T), _solution("1")), ((T, F), _solution("2"))]),   # 1 correct
        _question("q2", [((T, T), _solution("3")), ((T, T), _solution("3"))]),   # 0 incorrect
        _question("q3", [((T, T), _solution("5")), ((T, T), _solution("5")),
                         ((T, F), _solution("6"))]),
    ]
    report = terminal_shortcut_report(questions)
    assert report.questions == 1
    assert report.positive_pairs == 1 and report.negative_pairs == 2
    # ...but the "does it print an answer at all" rates are over EVERY trajectory seen, which
    # is a property of the dataset rather than of the loss's population.
    assert report.prints_answer_correct == 1.0


def test_an_empty_sample_reports_none_not_zero():
    """`None` means "not computable", and 0.0 is a legitimate value of every AUC here. A
    statistic that reports 0.0 for "no data" is a statistic that reads as a strong finding
    when nothing was measured at all (§0 rule 5)."""
    report = terminal_shortcut_report([])
    assert report.questions == 0
    assert report.answer_match_auc is None
    assert report.masked_overlap_auc is None
    assert report.prints_answer_correct is None
    assert all("n/a" in line or "n/a" not in line for line in report.lines())  # renders
