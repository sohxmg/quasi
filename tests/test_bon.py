"""BoN harness tests. CPU-only, no model, no GPU.

Everything here exists to protect the FAIRNESS of the CRM comparison rather than the
correctness of any number. Four things have to hold or the two rows of the table stop being
about the reward models:

  * the step segmentation is CRM's, character for character;
  * the N ladder subsets the pool the way CRM's `split_query` does;
  * the argmax and its tie-break are CRM's `best_of_n`;
  * the vendored grader is byte-identical to CRM's but for three import lines.

The aggregator tests are ordinary unit tests on arithmetic, plus the one structural claim
§9.1 makes about `d_0` being constant within a question.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from feynman_prm.eval.bon import (
    AGGREGATOR_BY_NAME,
    AGGREGATORS,
    SAMPLE_NUMS,
    best_of_n,
    split_response_into_steps,
    take_first_n,
)

CRM_EVAL = Path(__file__).resolve().parent.parent.parent / "CRM" / "BoN" / "eval"
VENDORED = Path(__file__).resolve().parent.parent / "feynman_prm" / "eval" / "crm_grader"


# ---------------------------------------------------------------------------------------
# 1. the ladder is CRM's
# ---------------------------------------------------------------------------------------


def test_sample_nums_are_crms():
    """The paper reports these five. Extending the ladder makes a column with nothing to
    compare against; shortening it drops one that has."""
    assert SAMPLE_NUMS == (8, 16, 32, 64, 128)


# ---------------------------------------------------------------------------------------
# 2. step segmentation
# ---------------------------------------------------------------------------------------


def crm_split(response: str) -> list[str]:
    """`eval_BoN.py:213-219`, copied here so the test compares against the real thing and
    not against our description of it."""
    steps = re.split(r"Step \d+:", response)
    return [f"Step {i + 1}: {s.strip()}" for i, s in enumerate(steps) if s.strip()]


RESPONSES = [
    "Step 1: First add 2 and 3.\nStep 2: The answer is 5.\n#### 5",
    "Let me think about it.\nStep 1: add\nStep 2: mul\nStep 3: The answer is 6.",
    "Step 1: only one step. The answer is \\boxed{7}",
    "no steps at all, the answer is 3",
    "Step 1: a\n\nStep 2:\nStep 3: b",          # an empty fragment in the MIDDLE
]


@pytest.mark.parametrize("response", RESPONSES)
def test_crm_verbatim_numbering_is_bit_identical_to_crm(response):
    assert list(split_response_into_steps(response, "crm_verbatim")) == crm_split(response)


@pytest.mark.parametrize("response", RESPONSES)
def test_segmentation_is_identical_under_both_numberings(response):
    """The numbering flag may change the "Step N: " prefix and NOTHING else. If it changed
    the segmentation, the two models would be scoring different decompositions and the
    comparison would be void."""
    strip = lambda steps: [re.sub(r"^Step \d+: ", "", s) for s in steps]  # noqa: E731
    assert strip(split_response_into_steps(response, "one_based")) == strip(
        split_response_into_steps(response, "crm_verbatim")
    )


def test_crm_numbering_is_off_by_one_on_the_common_case():
    """Not a bug in our transcription -- a bug in CRM's file, and the reason the flag exists.
    A response beginning "Step 1:" splits into a leading EMPTY fragment; the filter drops it
    but `enumerate` has already counted it, so every step is renumbered one too high."""
    response = "Step 1: add\nStep 2: mul"
    assert split_response_into_steps(response, "crm_verbatim") == (
        "Step 2: add", "Step 3: mul",
    )
    assert split_response_into_steps(response, "one_based") == (
        "Step 1: add", "Step 2: mul",
    )


def test_one_based_numbering_matches_math_shepherd_format():
    """§4.7: 99.98% of Math-Shepherd steps start with "Step N: ", correctly numbered from 1.
    That is the format both models trained on, which is why it is the default."""
    steps = split_response_into_steps("Step 1: a\nStep 2: b\nStep 3: c", "one_based")
    assert [s.split(":")[0] for s in steps] == ["Step 1", "Step 2", "Step 3"]


def test_response_with_no_step_markers_still_yields_one_fragment():
    """It must not vanish: a dropped candidate would turn best_of_128 into best_of_127."""
    assert len(split_response_into_steps("no markers here", "one_based")) == 1


# ---------------------------------------------------------------------------------------
# 3. selection semantics
# ---------------------------------------------------------------------------------------


def crm_split_query(completions, n, num_samples):
    """`eval_BoN.py:35-41`, copied. Note `load_queries` sets every `logprobs` to 0."""
    out = []
    for idx in range(int(len(completions) / num_samples)):
        samples = [s for s in completions if s["idx"] == idx]
        samples = sorted(samples, key=lambda x: x["logprobs"], reverse=True)
        out.append(samples[:n])
    return out


def test_take_first_n_reproduces_crm_split_query():
    """CRM sorts by a constant key, and Python's sort is stable, so it keeps file order. If
    that ever stopped being true the N ladder would stop being nested and the accuracy curve
    would be comparing different pools at different N."""
    completions = [{"idx": 0, "rank": r, "logprobs": 0} for r in range(128)]
    for n in SAMPLE_NUMS:
        crm = [c["rank"] for c in crm_split_query(completions, n, 128)[0]]
        assert crm == take_first_n(n, 128)


def test_the_ladder_is_nested():
    for small, large in zip(SAMPLE_NUMS, SAMPLE_NUMS[1:]):
        assert set(take_first_n(small, 128)) <= set(take_first_n(large, 128))


def test_best_of_n_matches_crm_argmax_and_tie_break():
    """CRM sorts by reward descending and takes [0]; a stable sort gives the EARLIEST
    candidate on a tie. `np.argmax` has the same rule, which is what makes the -inf score for
    an over-length candidate safe -- an all-unscorable pool falls back to candidate 0, the
    no-reranking baseline, rather than to an arbitrary index."""
    scores = np.array([
        [1.0, 3.0, 2.0, 3.0],       # tie at the max -> earliest wins
        [-np.inf] * 4,              # whole pool unscorable -> candidate 0
        [0.0, -1.0, -2.0, 5.0],     # the winner is outside the n=2 window
    ])
    assert best_of_n(scores, 4).tolist() == [1, 0, 3]
    assert best_of_n(scores, 2).tolist() == [1, 0, 0]

    for n in (2, 4):
        for q, row in enumerate(scores):
            crm = sorted(
                [{"reward": v, "rank": i} for i, v in enumerate(row[:n])],
                key=lambda x: x["reward"],
                reverse=True,
            )[0]["rank"]
            assert crm == best_of_n(scores, n)[q]


def test_best_of_n_only_looks_at_the_first_n():
    scores = np.array([[0.0, 0.0, 0.0, 100.0]])
    assert best_of_n(scores, 2).tolist() == [0]
    assert best_of_n(scores, 4).tolist() == [3]


# ---------------------------------------------------------------------------------------
# 4. aggregators
# ---------------------------------------------------------------------------------------


def test_every_aggregator_prefers_the_solution_with_smaller_deltas():
    """The sign convention: higher is better, and a solution whose every step costs less must
    win under all six. A flipped sign is invisible in a loss curve and inverts every BoN
    number, which is the §7.12 `good/margin` failure in a new place."""
    good = np.array([-0.7, -0.7, -0.7])
    bad = np.array([-0.7, 2.0, -0.7])
    for agg in AGGREGATORS:
        assert agg.fn(good, 3.0, 0.347) > agg.fn(bad, 3.0, 0.347), agg.name


def test_neg_final_distance_ranks_identically_to_neg_sum_delta_within_a_question():
    """The claim in `neg_final_distance`'s own docstring: d_0 depends only on the prompt, so
    within one question it is a constant offset shared by all 128 candidates. Two aggregators
    that must never disagree -- if they do, d_0 is varying within a question, which means
    h_{s_0} is, which contradicts §7.7."""
    rng = np.random.default_rng(0)
    d0 = 4.2
    pools = [rng.normal(size=rng.integers(3, 12)) for _ in range(64)]
    by_distance = [AGGREGATOR_BY_NAME["neg_final_distance"].fn(d, d0, 0.3) for d in pools]
    by_sum = [AGGREGATOR_BY_NAME["neg_sum_delta"].fn(d, d0, 0.3) for d in pools]
    assert np.argsort(by_distance).tolist() == np.argsort(by_sum).tolist()


def test_log_survival_is_the_log_of_crms_product_form():
    """CRM scores prod_i (1 - h_i) with h_i = sigma(logit_i) (`eval_BoN.py:243-248`). Read
    h_i = sigma(Delta_i - tau) and this aggregator is the log of exactly that product."""
    deltas = np.array([-0.7, 0.2, 1.5])
    tau = 0.347
    h = 1.0 / (1.0 + np.exp(-(deltas - tau)))
    expected = np.log(np.prod(1.0 - h))
    assert AGGREGATOR_BY_NAME["log_survival"].fn(deltas, 0.0, tau) == pytest.approx(expected)


def test_neg_sum_excess_is_zero_when_every_step_is_at_or_below_the_ruler():
    """The hinge property ⑥ L_good was chosen for (§7.12): exactly zero below the target, so
    a long correct solution is not penalised for being long."""
    deltas = np.full(20, -0.693)
    assert AGGREGATOR_BY_NAME["neg_sum_excess"].fn(deltas, 0.0, 0.347) == 0.0


def test_summed_aggregators_prefer_longer_solutions_and_the_mean_does_not():
    """The reason `neg_avg_delta` is in the set. On a trained ruler every good step costs
    -0.693, so a sum REWARDS length: an 8-step correct solution outscores a 4-step one purely
    for having more steps. The gap between the two rows in the report is how much of any BoN
    gain is really a length preference."""
    short, long_ = np.full(4, -0.693), np.full(8, -0.693)
    assert AGGREGATOR_BY_NAME["neg_sum_delta"].fn(long_, 0.0, 0.347) > (
        AGGREGATOR_BY_NAME["neg_sum_delta"].fn(short, 0.0, 0.347)
    )
    assert AGGREGATOR_BY_NAME["neg_avg_delta"].fn(long_, 0.0, 0.347) == pytest.approx(
        AGGREGATOR_BY_NAME["neg_avg_delta"].fn(short, 0.0, 0.347)
    )


def test_neg_max_delta_is_the_processbench_detection_statistic():
    """§9.6.1: flagging is `max_t Delta_t > tau`. Ranking by -max Delta is that statistic used
    as an order rather than a threshold, so the two evals share a scoring statistic."""
    deltas = np.array([-0.7, 3.0, -0.7])
    assert AGGREGATOR_BY_NAME["neg_max_delta"].fn(deltas, 0.0, 0.347) == pytest.approx(-3.0)


def test_aggregator_names_are_unique():
    assert len({a.name for a in AGGREGATORS}) == len(AGGREGATORS)


# ---------------------------------------------------------------------------------------
# 5. the vendored grader
# ---------------------------------------------------------------------------------------


PERMITTED_EDITS = {
    "eval_normalizer.py": 1,   # from eval_PQM_grader -> from .eval_grader
    "eval_utils.py": 2,        # two flat imports -> relative
    "eval_grader.py": 0,       # untouched
}


@pytest.mark.skipif(not CRM_EVAL.exists(), reason="../CRM not checked out next to this repo")
@pytest.mark.parametrize("name", sorted(PERMITTED_EDITS))
def test_vendored_grader_differs_from_crm_only_in_its_imports(name):
    """The comparison rests on both rows being judged by the same code. A drifted grader --
    one bug fixed, one normalisation improved -- would move our number and not CRM's, and
    would look exactly like a modelling result."""
    original = (CRM_EVAL / name).read_text().splitlines()
    vendored = (VENDORED / name).read_text().splitlines()
    assert len(original) == len(vendored), f"{name}: line count changed"
    differing = [i for i, (a, b) in enumerate(zip(original, vendored)) if a != b]
    assert len(differing) == PERMITTED_EDITS[name], f"{name}: lines {differing} differ"
    for i in differing:
        assert "import" in vendored[i], f"{name}:{i + 1} is not an import line: {vendored[i]!r}"


@pytest.mark.skipif(not CRM_EVAL.exists(), reason="../CRM not checked out next to this repo")
def test_eval_grader_is_byte_identical():
    a = hashlib.sha256((CRM_EVAL / "eval_grader.py").read_bytes()).hexdigest()
    b = hashlib.sha256((VENDORED / "eval_grader.py").read_bytes()).hexdigest()
    assert a == b


def test_gsm8k_grader_is_importable_and_matches_crms_last_number_rule():
    """`eval_utils.py:23-28` extracts the LAST number in the response. Pinned because it is
    lenient in both directions and it is what CRM's published gsm8k numbers were produced by
    -- so it must not be "fixed" here."""
    from feynman_prm.eval.crm_grader import eval_gsm8k

    responses = [{"response": "Step 1: 2+3 is 5.\nThe answer is 5"},
                 {"response": "Step 1: the answer is 4"}]
    acc, per_item, extracted = eval_gsm8k(responses, answers=["5", "5"], is_extract=True)
    assert per_item == [True, False]
    assert extracted == [5, 4]
    assert acc == pytest.approx(50.0)


# ---------------------------------------------------------------------------------------
# 6. the whole non-GPU path: candidate file -> selection -> grading -> table
# ---------------------------------------------------------------------------------------


def build_fixture(tmp_path, n_questions=3, n_candidates=128):
    """A candidate file shaped exactly like CRM's: a list aligned index-for-index with the
    reference set, each entry `{"question": ..., "responses": [{"text": ...}, ...]}`.

    Candidate 0 of every question is WRONG and candidate `q + 1` is RIGHT, so the correct
    answer sits inside the best_of_8 window and the baselines are known in advance:
    first_candidate = 0%, oracle = 100% at every N, mean = 1/128.
    """
    # GSM-Plus `answer` is the bare numeric answer, and CRM grades gsm8k with
    # `is_extract=True`, i.e. `gold = eval(answer)`. A "#### n" string here would silently
    # fall through to a string comparison and grade everything wrong.
    reference = [{"question": f"what is {q} plus one?", "answer": str(q + 1)}
                 for q in range(n_questions)]
    payload = []
    for q in range(n_questions):
        responses = [{"text": f"Step 1: guess.\nThe answer is {999 + i}"}
                     for i in range(n_candidates)]
        responses[q + 1] = {"text": f"Step 1: add one.\nThe answer is {q + 1}"}
        payload.append({"question": reference[q]["question"], "responses": responses})
    path = tmp_path / "gsm8k-plus-fixture-128.json"
    path.write_text(json.dumps(payload))
    return path, reference


def test_load_candidates_aligns_and_rejects_a_mismatched_reference(tmp_path):
    from feynman_prm.eval.bon import load_candidates

    path, reference = build_fixture(tmp_path)
    pool = load_candidates(path, "gsm8k", reference)
    assert len(pool) == len(reference)
    assert len(pool[0]) == 128
    assert pool[1][2].rank == 2 and pool[1][2].idx == 1

    shuffled = [reference[1], reference[0], reference[2]]
    with pytest.raises(AssertionError, match="does not match the reference set"):
        load_candidates(path, "gsm8k", shuffled)

    with pytest.raises(AssertionError, match="reference set"):
        load_candidates(path, "gsm8k", reference[:2])


def test_short_pool_is_rejected_rather_than_silently_becoming_best_of_fewer(tmp_path):
    """A question with 100 responses would make `best_of_128` a `best_of_100` and flatter the
    model against a paper number computed over 128."""
    from feynman_prm.eval.bon import load_candidates

    path, reference = build_fixture(tmp_path)
    payload = json.loads(path.read_text())
    payload[1]["responses"] = payload[1]["responses"][:100]
    path.write_text(json.dumps(payload))
    with pytest.raises(AssertionError, match="ladder tops out"):
        load_candidates(path, "gsm8k", reference)


def test_end_to_end_table_on_a_fixture_with_a_known_answer(tmp_path):
    """Drives everything downstream of the model: a scorer that ranks the known-good candidate
    first must produce 100% at every N, and the baselines must come out at their known values.

    Scores are injected rather than computed -- the model is the only part of the pipeline
    this file cannot exercise on CPU, and it is the part the GPU eval measures anyway."""
    from feynman_prm.eval.bon import AGGREGATORS, PoolGrader, ScoredPool, evaluate_pool, load_candidates

    path, reference = build_fixture(tmp_path)
    pool = load_candidates(path, "gsm8k", reference)
    n_q, n_c = len(pool), len(pool[0])

    perfect = np.zeros((n_q, n_c))
    for q in range(n_q):
        perfect[q, q + 1] = 1.0          # the correct candidate, always inside the n=8 window
    scored = ScoredPool(
        scores={a.name: perfect.copy() for a in AGGREGATORS},
        n_steps=np.ones((n_q, n_c), dtype=np.int64),
        counters={"over_length": 0.0, "over_length_fraction": 0.0},
    )
    grader = PoolGrader("gsm8k", pool, reference)
    table = evaluate_pool(scored, grader, n_c, grade_all=True)

    assert table["baseline_first_candidate"] == pytest.approx(0.0)
    assert table["baseline_mean_candidate"] == pytest.approx(100.0 / 128)
    for n in SAMPLE_NUMS:
        assert table["oracle"][f"best_of_{n}"] == pytest.approx(100.0)
        for agg in AGGREGATORS:
            assert table[agg.name][f"best_of_{n}"] == pytest.approx(100.0), (agg.name, n)


def test_an_all_unscorable_pool_falls_back_to_the_no_reranking_baseline(tmp_path):
    """The -inf convention, end to end. Every candidate over max_len must leave the question
    scored as candidate 0 -- i.e. exactly the no-reward-model baseline -- and must never drop
    the question or shrink the pool."""
    from feynman_prm.eval.bon import AGGREGATORS, PoolGrader, ScoredPool, evaluate_pool, load_candidates

    path, reference = build_fixture(tmp_path)
    pool = load_candidates(path, "gsm8k", reference)
    n_q, n_c = len(pool), len(pool[0])
    scored = ScoredPool(
        scores={a.name: np.full((n_q, n_c), -np.inf) for a in AGGREGATORS},
        n_steps=np.zeros((n_q, n_c), dtype=np.int64),
        counters={"over_length": float(n_q * n_c), "over_length_fraction": 1.0},
    )
    table = evaluate_pool(scored, PoolGrader("gsm8k", pool, reference), n_c, grade_all=False)
    for agg in AGGREGATORS:
        for n in SAMPLE_NUMS:
            assert table[agg.name][f"best_of_{n}"] == table["baseline_first_candidate"]


def test_grader_memoises_so_every_table_reads_one_verdict_per_candidate(tmp_path):
    """30 selection sets per file (6 aggregators x 5 N) overlap heavily. Without the cache the
    math subsets would re-run sympy on answers already judged -- and, worse, two tables could
    in principle disagree about the same candidate if the grader were ever non-deterministic."""
    from feynman_prm.eval.bon import PoolGrader, load_candidates

    path, reference = build_fixture(tmp_path)
    pool = load_candidates(path, "gsm8k", reference)
    grader = PoolGrader("gsm8k", pool, reference)

    calls = []
    original = grader._judge
    grader._judge = lambda keys: (calls.append(len(keys)), original(keys))[1]

    picks = np.array([1, 2, 3])
    assert grader.grade(picks) == [True, True, True]
    assert grader.grade(picks) == [True, True, True]
    assert calls == [3], "the second call must be served entirely from the cache"
