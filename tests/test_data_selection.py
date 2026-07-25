"""§15's data tests: selection, splitting, the SHA, and branch-point mining (§8.2, §8.3)."""

from __future__ import annotations

import pytest

from feynman_prm.data.branch_points import mine, normalise_step, read_jsonl, write_jsonl
from feynman_prm.data.math_shepherd import (
    Question,
    Trajectory,
    build_questions,
    dataset_stats,
    selection_sha,
    split_questions,
)

T, F = True, False


def _row(prompt, steps, labels):
    return {"prompt": prompt, "completions": steps, "labels": labels}


def test_empty_label_rows_are_dropped():
    """15 train rows have empty labels (§4.7); CRM filters them too."""
    questions, counters = build_questions(
        [
            _row("q1", ["a"], [T]),
            _row("q2", [], []),
            _row("q3", ["a", "b"], [T]),      # length mismatch
        ]
    )
    assert counters["dropped_empty_labels"] == 1
    assert counters["dropped_length_mismatch"] == 1
    assert len(questions) == 1


def test_questions_group_by_exact_prompt():
    questions, _ = build_questions(
        [_row("same", ["a"], [T]), _row("same", ["b"], [F]), _row("other", ["c"], [T])]
    )
    assert len(questions) == 2
    assert questions[0].n_correct == 1 and questions[0].n_incorrect == 1
    assert questions[0].trainable and not questions[1].trainable


def test_stats_reproduce_the_measured_quantities():
    questions, _ = build_questions(
        [
            _row("q1", ["a", "b"], [T, T]),
            _row("q1", ["a", "b", "c"], [T, F, F]),
            _row("q2", ["a"], [T]),
        ]
    )
    stats = dataset_stats(questions)
    assert stats["questions"] == 2
    assert stats["trainable_questions"] == 1
    assert stats["all_labels_equals_last_label"] == 1.0   # 100.0% in real data (§4.2)
    assert stats["first_error_index_mean"] == 1.0
    assert stats["questions_ge2_correct"] == 0.0          # §16.16 / §4.2.1


def test_split_is_question_level_with_no_overlap():
    questions = [
        Question(
            qid=f"q{i}",
            prompt=f"p{i}",
            trajectories=[Trajectory(("a",), (T,)), Trajectory(("b",), (F,))],
        )
        for i in range(20)
    ]
    train, val = split_questions(questions, n_val_questions=5, n_questions=10, seed=42)
    assert len(val) == 5 and len(train) == 10
    assert not ({q.qid for q in train} & {q.qid for q in val})


def test_selection_sha_is_stable_and_order_independent():
    """R10: log the ACTUAL selected id list, not the seed that implies it. The old project
    documented a "seed 42 shuffle" for two years that turned out not to exist."""
    assert selection_sha(["b", "a", "c"]) == selection_sha(["a", "b", "c"])
    assert selection_sha(["a", "b"]) != selection_sha(["a", "b", "c"])


def test_split_is_reproducible_from_the_seed():
    questions = [
        Question(qid=f"q{i}", prompt=f"p{i}",
                 trajectories=[Trajectory(("a",), (T,)), Trajectory(("b",), (F,))])
        for i in range(20)
    ]
    a = split_questions(questions, 5, 10, seed=42)
    b = split_questions(questions, 5, 10, seed=42)
    assert selection_sha([q.qid for q in a[0]]) == selection_sha([q.qid for q in b[0]])


def test_branch_point_mining_finds_correctness_disagreement(tmp_path):
    """§8.3 / §4.4: ~105k branch points, ~63k with one correct and one incorrect
    continuation -- "same state, right move vs wrong move"."""
    question = Question(
        qid="q",
        prompt="p",
        trajectories=[
            Trajectory(("Step 1: shared", "Step 2: good"), (T, T)),
            Trajectory(("Step 1: shared", "Step 2: bad"), (T, F)),
        ],
    )
    points, counters = mine([question])
    assert counters["branching_nodes"] == 1
    assert len(points) == 1
    point = points[0]
    assert point.depth == 1 and point.correct_step == "Step 2: good"
    assert point.incorrect_step == "Step 2: bad"

    path = tmp_path / "branch_points.jsonl"
    write_jsonl(points, path)
    assert read_jsonl(path) == points


def test_normalisation_strips_calculator_annotations():
    assert normalise_step("So 2+2 = <<2+2=4>>4.") == normalise_step("so 2+2 = 4")


def test_no_branch_point_when_both_continuations_agree():
    question = Question(
        qid="q",
        prompt="p",
        trajectories=[
            Trajectory(("shared", "good a"), (T, T)),
            Trajectory(("shared", "good b"), (T, T)),
        ],
    )
    points, counters = mine([question])
    assert counters["branching_nodes"] == 1 and len(points) == 0


def test_split_refuses_when_the_pool_is_too_small():
    questions = [
        Question(qid="q", prompt="p",
                 trajectories=[Trajectory(("a",), (T,)), Trajectory(("b",), (F,))])
    ]
    with pytest.raises(ValueError, match="cannot hold out"):
        split_questions(questions, n_val_questions=5, n_questions=1, seed=0)
