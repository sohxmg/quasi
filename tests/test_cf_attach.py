"""§7.5.3 option (b): CF examples attach to prefixes already in the main batch.

What these pin is the JOIN, not the loss -- `tests/test_counterfactual.py` owns the SupCon
arithmetic. The property that matters here and that nothing else checks: the state a variant
is bound to is `s_{i-1}` of the anchor's own prefix, so `L_CF` compares
`phi(h(q + steps[:i]), act_emb(variant))` across variants. Bind it to the wrong state and
every distance is still finite, the loss still falls, and no curve shows it.
"""

from __future__ import annotations

import numpy as np
import pytest

from feynman_prm.data.cf_attach import CFContext
from feynman_prm.data.collate import SequenceRow, collate
from feynman_prm.data.counterfactual import CounterfactualExample
from feynman_prm.data.prefix_hash import prefix_hash, prefix_hashes

QUESTION = "Janet has 3 apples and buys 5 more. How many?"
STEPS = ["Step 1: she starts with 3", "Step 2: she buys 5", "Step 3: 3 + 5 = 8"]


class _Tok:
    """Deterministic, non-empty, and no `transformers` dependency -- the whole suite runs
    on CPU with no model (`data/collate.py`'s docstring)."""

    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 100 + 1 for c in text][:16] or [1]}


def _row(question=QUESTION, steps=STEPS, qid="q1", correct=True, with_hash=True):
    """A row whose index arrays are DERIVED from `steps`, so `prefix_hash` (T+1) and
    `state_pos` (T+1) cannot drift apart in the fixture -- `collate` rejects that, and a
    fixture that hard-codes both is a test that passes for the wrong reason."""
    T = len(steps)
    # prompt(3 tok) SEP | step(4 tok) SEP | ... -- positions by arithmetic, as in tokenize.py
    state_pos = np.array([3 + 5 * i for i in range(T + 1)])
    span_start = np.array([4 + 5 * i for i in range(T)])
    span_end = span_start + 4
    return SequenceRow(
        qid=qid,
        input_ids=np.arange(1, int(state_pos[-1]) + 2),
        state_pos=state_pos,
        span_start=span_start,
        span_end=span_end,
        correct=correct,
        z=-1 if correct else T - 1,
        prefix_hash=prefix_hashes(question, steps) if with_hash else None,
    )


def _example(step_index=2, n_pos=2, n_neg=3):
    return CounterfactualExample(
        question=QUESTION,
        steps=tuple(STEPS),
        step_index=step_index,
        positive_rewrites=tuple(f"positive rewrite {i}" for i in range(n_pos)),
        negative_rewrites=tuple(f"negative rewrite {i}" for i in range(n_neg)),
    )


def _ctx(examples, max_examples=12):
    return CFContext(examples, _Tok(), 0, max_examples)


def test_variants_bind_to_the_prefix_state_not_the_terminal():
    """Rewriting step 3 must attach at `s_2` -- the state after steps 1 and 2.

    This is the §7.5.3-(b) claim in one assert. `s_2` is flat state index 2 of the first
    trajectory, and binding to `s_3` (the terminal) or `s_0` would leave the loss perfectly
    well-defined and perfectly wrong.
    """
    batch = collate([_row()], pad_id=0)
    attached = _ctx([_example(step_index=2)]).attach(batch, np.random.default_rng(0))

    assert attached is not None
    assert attached.variant_state.tolist() == [2] * 6
    # anchor, then every positive, then every negative -- counterfactual_loss's convention
    assert attached.variant_kind.tolist() == [0, 1, 1, 2, 2, 2]
    assert attached.variant_example.tolist() == [0] * 6


@pytest.mark.parametrize("step_index,expected_state", [(0, 0), (1, 1), (2, 2)])
def test_the_bound_state_tracks_step_index(step_index, expected_state):
    batch = collate([_row()], pad_id=0)
    attached = _ctx([_example(step_index=step_index)]).attach(batch, np.random.default_rng(0))
    assert attached.variant_state.tolist() == [expected_state] * 6


def test_a_sibling_trajectory_sharing_the_prefix_is_an_equally_valid_host():
    """Two trajectories that agree on steps 1-2 give the SAME `h_1`, so a CF example
    rewriting step 2 may bind to either. The join takes the first deterministically -- what
    must not happen is a miss just because the generator sampled the other one."""
    other = ["Step 1: she starts with 3", "Step 2: she buys 5", "Step 3: WRONG"]
    batch = collate([_row(steps=other, correct=False), _row()], pad_id=0)

    attached = _ctx([_example(step_index=2)]).attach(batch, np.random.default_rng(0))
    assert attached is not None
    # s_2 of the FIRST trajectory, whose prefix is identical through step 2
    assert attached.variant_state.tolist() == [2] * 6


def test_an_absent_prefix_attaches_nothing_rather_than_guessing():
    batch = collate([_row(question="a completely different question")], pad_id=0)
    assert _ctx([_example()]).attach(batch, np.random.default_rng(0)) is None


def test_a_parquet_without_prefix_hash_degrades_instead_of_crashing():
    """§14: a stale `sequences.parquet` must not take a run down at micro-batch 1. It
    trains exactly as it did before the CF path existed."""
    batch = collate([_row(with_hash=False)], pad_id=0)
    assert batch.state_prefix_hash.tolist() == [0, 0, 0, 0]
    assert _ctx([_example()]).attach(batch, np.random.default_rng(0)) is None


def test_a_prefix_hash_of_the_wrong_length_is_rejected_loudly():
    """Built from a different step list than the sequence was tokenised from, `prefix_hash`
    is indexed by state and would bind CF examples to the wrong states. Silent is the one
    thing it must not be."""
    row = _row()
    bad = SequenceRow(
        qid=row.qid,
        input_ids=row.input_ids,
        state_pos=row.state_pos,
        span_start=row.span_start,
        span_end=row.span_end,
        correct=row.correct,
        z=row.z,
        prefix_hash=prefix_hashes(QUESTION, STEPS[:2]),   # T+1 = 3, but the row has T = 3
    )
    with pytest.raises(ValueError, match="prefix_hash has 3 entries"):
        collate([bad], pad_id=0)


def test_the_per_batch_cap_bounds_the_variant_count():
    """A batch holding many CF-covered questions must not swing L_CF's magnitude."""
    steps = [[f"Step 1: q{n}", f"Step 2: q{n}", f"Step 3: q{n}"] for n in range(10)]
    rows = [_row(question=f"question {n}", steps=s, qid=f"q{n}") for n, s in enumerate(steps)]
    examples = [
        CounterfactualExample(
            question=f"question {n}",
            steps=tuple(s),
            step_index=2,
            positive_rewrites=("p0", "p1"),
            negative_rewrites=("n0", "n1", "n2"),
        )
        for n, s in enumerate(steps)
    ]
    batch = collate(rows, pad_id=0)

    attached = _ctx(examples, max_examples=3).attach(batch, np.random.default_rng(0))
    assert attached.info["cf/examples_attached"] == 3.0
    assert attached.info["cf/examples_eligible"] == 10.0
    assert len(set(attached.variant_example.tolist())) == 3


def test_variant_example_ids_are_contiguous_from_zero():
    """`counterfactual_loss` sizes its padded view with `variant_example.max() + 1` and
    indexes rows by it, so a gap left by a dropped example would allocate a phantom class
    whose only query has `|P| = 0`. It is dropped there rather than crashing, which is
    exactly why the gap would be invisible."""
    rows = [
        _row(question=f"question {n}", steps=[f"Step 1: q{n}", f"Step 2: q{n}"], qid=f"q{n}")
        for n in range(3)
    ]
    examples = [
        CounterfactualExample(
            question=f"question {n}",
            steps=(f"Step 1: q{n}", f"Step 2: q{n}"),
            step_index=1,
            positive_rewrites=("p0",),
            negative_rewrites=("n0", "n1"),
        )
        for n in range(3)
    ]
    batch = collate(rows, pad_id=0)
    attached = _ctx(examples).attach(batch, np.random.default_rng(0))

    seen = sorted(set(attached.variant_example.tolist()))
    assert seen == list(range(len(seen)))


def test_prefix_hash_separates_an_internal_newline_from_a_step_boundary():
    """§4.7: 13.9% of solutions contain a step with an internal newline. `["a\\nb"]` and
    `["a", "b"]` are different trajectories and must not share a prefix identity."""
    assert prefix_hash(QUESTION, ["a\nb"]) != prefix_hash(QUESTION, ["a", "b"])


def test_prefix_hashes_are_aligned_with_state_pos():
    hashes = prefix_hashes(QUESTION, STEPS)
    assert len(hashes) == len(STEPS) + 1              # s_0 .. s_T
    assert hashes[0] == prefix_hash(QUESTION, [])     # s_0 is the prompt alone
    assert hashes[2] == prefix_hash(QUESTION, STEPS[:2])


def test_a_stale_parquet_is_a_hard_error_once_lambda_cf_is_nonzero():
    """The guard added with `lambda_cf` 0.0 -> 1.0 on 2026-08-15.

    A `sequences.parquet` written before that date has no `prefix_hash` column, so nothing
    attaches. At `lambda_cf = 0` that is exactly right -- the run is bit-identical to the
    baseline. At a NONZERO weight it is the §14 failure this repo keeps paying for: (4) is
    weighted, fed and logged while training on nothing, and every other curve looks healthy.
    `cf/attach_rate` would show it, three hours in; the launch must show it in one second.

    Grep, because the failure is the absence of a line of code (§15, the B11 precedent).
    """
    import re
    from pathlib import Path

    code = (Path(__file__).resolve().parents[1] / "feynman_prm" / "train.py").read_text()
    stripped = re.sub(r"#.*", "", code)
    assert "cfg.losses.lambda_cf > 0" in stripped, (
        "the stale-parquet check must be gated on a NONZERO lambda_cf -- at 0.0 a parquet "
        "with no prefix_hash is legitimate and must still launch"
    )
    guard = stripped[stripped.index("cfg.losses.lambda_cf > 0"):]
    assert "prefix_hash is not None" in guard and "raise AssertionError" in guard, (
        "the check must RAISE, not warn: a warning in a launch log is what B12 was"
    )


def test_the_stale_parquet_guard_reads_the_train_rows_not_the_cf_examples():
    """The hashes that can go missing are the PARQUET's, not the CF file's. `cf_attach`
    computes its side from `(question, steps[:i])` in memory every launch, so it can never be
    stale -- checking that side would be a guard that cannot fire (§14, B12)."""
    row = SequenceRow(
        qid="q", input_ids=np.zeros(4, dtype=np.int64), state_pos=np.array([0, 3]),
        span_start=np.array([1]), span_end=np.array([3]), correct=True, z=1, recovery=False,
    )
    assert row.prefix_hash is None, "a row built without one defaults to None, not to zeros"


# ---------------------------------------------------------------------------------------
# §8.2's split guard. It stopped the 2026-08-16 launch over 4 of 27,114 examples, and the
# thing it named -- val leakage, or a different selection -- was neither.
# ---------------------------------------------------------------------------------------


def _cf_example(question: str) -> CounterfactualExample:
    return CounterfactualExample(
        question=question, steps=list(STEPS), step_index=1,
        positive_rewrites=["p"], negative_rewrites=["n1", "n2", "n3"],
    )


def _qid(question: str) -> str:
    from feynman_prm.data.math_shepherd import question_id

    return question_id(question)


def test_a_cf_example_on_a_val_question_is_fatal_at_any_count():
    """The §8.2 hazard, and the only one: phase 1 training on a held-out question is silent
    leakage that no curve shows, so one example is enough."""
    from feynman_prm.data.cf_attach import select_cf_examples_for_train

    examples = [_cf_example(f"q{i}") for i in range(100)]
    train = {_qid(f"q{i}") for i in range(1, 100)}
    val = {_qid("q0")}
    with pytest.raises(AssertionError, match="VAL question"):
        select_cf_examples_for_train(examples, train, val)


def test_a_question_absent_from_BOTH_splits_is_dropped_and_counted_not_fatal():
    """A question whose every trajectory was dropped at tokenisation (§4.6) is in the train
    SELECTION and absent from the train ROWS. Its CF examples cannot attach -- `attach_cf`
    already counts that miss -- and calling it leakage stops a legitimate launch."""
    from feynman_prm.data.cf_attach import select_cf_examples_for_train

    examples = [_cf_example(f"q{i}") for i in range(1000)]
    train = {_qid(f"q{i}") for i in range(4, 1000)}
    kept, info = select_cf_examples_for_train(examples, train, val_qids=set())

    assert len(kept) == 996
    assert info["examples_dropped_question_absent"] == 4
    assert info["questions_absent"] == 4
    assert len(info["questions_absent_sample"]) == 4


def test_too_many_absent_questions_IS_fatal_a_different_selection_cannot_be_a_handful():
    from feynman_prm.data.cf_attach import select_cf_examples_for_train

    examples = [_cf_example(f"q{i}") for i in range(100)]
    train = {_qid(f"q{i}") for i in range(50)}
    with pytest.raises(AssertionError, match="different selection"):
        select_cf_examples_for_train(examples, train, val_qids=set())


def test_the_clean_case_keeps_everything_and_reports_zero_drops():
    from feynman_prm.data.cf_attach import select_cf_examples_for_train

    examples = [_cf_example(f"q{i}") for i in range(20)]
    train = {_qid(f"q{i}") for i in range(20)}
    kept, info = select_cf_examples_for_train(examples, train, {_qid("other")})
    assert len(kept) == 20
    assert info["examples_dropped_question_absent"] == 0 and info["questions_absent"] == 0


def test_leakage_is_checked_BEFORE_the_absent_tolerance():
    """A corpus that is both leaky and largely absent must report the leak: the tolerance is
    about a tokenisation drop, and no count of absent questions excuses a val question."""
    from feynman_prm.data.cf_attach import select_cf_examples_for_train

    examples = [_cf_example(f"q{i}") for i in range(10)]
    with pytest.raises(AssertionError, match="VAL question"):
        select_cf_examples_for_train(examples, set(), {_qid("q0")})
