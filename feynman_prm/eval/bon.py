"""Best-of-N eval, built to be comparable with CRM's published BoN numbers.

Locked #7 says "ProcessBench only. Hooks left for BoN. Do not build them now." **The human
lifted that on 2026-08-05 and asked for BoN specifically, to put this model next to CRM's
paper numbers.** This file is that hook, cashed.

A BoN comparison is only a comparison if everything except the reranker is held fixed. Five
things are shared with `../CRM/BoN/eval/eval_BoN.py` and each one is load-bearing:

  1. **The candidate files.** The same four `*-128.json` sampled-response files CRM scores.
     We generate nothing. If the pool differs, nothing downstream is comparable.
  2. **The N ladder** -- `SAMPLE_NUMS = [8, 16, 32, 64, 128]`, and the same "first n in file
     order" subsetting (`split_query`, see `take_first_n` below for why sorting by a constant
     is exactly that).
  3. **The selection rule** -- argmax reward, ties to the earliest candidate (`best_of_n`).
  4. **The step segmentation** -- CRM's `re.split(r"Step \\d+:", response)`, character for
     character. Both models therefore see the same steps, and so the same number of scoring
     positions.
  5. **The grader** -- CRM's own, vendored byte-identical (`crm_grader/__init__.py`). Same
     judge on both rows of the table.

What is NOT shared, and must not be:

  * **The score.** CRM aggregates `prod_i (1 - sigma(logit_i))`, a survival probability off a
    token-classification head. We have no such head (locked #3a) -- one scoring path, the
    quasimetric distance. §9.1's per-step statistic is `Delta_i = d_i - d_{i-1}`, and BoN needs
    one number per SOLUTION, so an aggregator over `Delta` is required. There is no single
    obviously-right choice, so this file computes SIX (see `AGGREGATORS`) and reports all of
    them -- and picks the headline one on held-out Math-Shepherd val, never on the BoN files.
    That is §9.2's rule for `tau`, applied to the aggregator, and for the same reason: a
    number chosen on the test set is not a result.
  * **The sequence layout.** We insert a separator after the prompt so `s_0` exists (§6.1);
    CRM has no `s_0`. That is an architecture difference, not a handicap either way.
  * **The step-prefix NUMBERING.** See `split_response_into_steps` -- CRM's re-prefixing is
    off by one and we default to the correct numbering. The segmentation is identical either
    way; only the literal "Step N: " text differs, and ours matches what each model trained on.

**What to expect, written down before the run so it cannot be rationalised after it.** §9.7
measured the within-solution rank of the true error at 0.34-0.42 against a chance 0.5, and
§9.8.1 measured that swapping in ANOTHER QUESTION's goal retains 103% of that signal. A score
whose goal argument carries no information is unlikely to rerank well. BoN is also an easier
task than ProcessBench -- it needs only "is this solution good", never "which step broke" --
so it is genuinely possible for it to look better than F1 0.241 did. Both readings are
available in advance; report the number either way.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import torch

from ..config import Config
from ..data.collate import SequenceRow, collate
from ..data.tokenize import EmptyStep, SequenceTooLong, build_sequence, sep_token_id
from ..diagnostics.logging import Progress
from .calibrate import natural_tau

# CRM's ladder, verbatim (`eval_BoN.py:21`). Do not extend it -- the paper reports these five.
SAMPLE_NUMS: tuple[int, ...] = (8, 16, 32, 64, 128)

_STEP_SPLIT = re.compile(r"Step \d+:")


# --------------------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------------------


@dataclass
class Candidate:
    """One sampled response for one question. `idx` is the question, `rank` the position of
    this response inside that question's list -- and `rank` is what the N ladder cuts on."""

    idx: int
    rank: int
    prompt: str
    response: str
    steps: tuple[str, ...]
    empty: bool = False          # the response contained no "Step N:" at all


def split_response_into_steps(response: str, numbering: str = "one_based") -> tuple[str, ...]:
    """CRM's segmentation (`eval_BoN.py:213-219`), with the numbering bug made a flag.

    The segmentation itself is reproduced exactly: `re.split(r"Step \\d+:", response)`, empty
    fragments dropped. Every model sees the same steps.

    **CRM's re-prefixing is off by one and it is not a transcription error here -- it is in
    their file.** `enumerate` runs over the UNFILTERED fragments while the filter drops the
    leading empty one, so a response that begins "Step 1:" -- which is the overwhelming
    majority -- comes back as `["Step 2: ...", "Step 3: ..."]`:

        >>> re.split(r"Step \\d+:", "Step 1: add\\nStep 2: mul\\n")
        ['', ' add\\n', ' mul\\n']                      # fragment 0 is empty and is dropped
        >>> [f"Step {i+1}: {s.strip()}" for i, s in enumerate(_) if s.strip()]
        ['Step 2: add', 'Step 3: mul']                  # ...but i does not renumber

    `one_based` (the default) numbers the KEPT steps 1..T, which is what Math-Shepherd rows
    look like (§4.7: 99.98% of its steps start with "Step N: ") and therefore what both models
    were trained on. `crm_verbatim` reproduces the shift.

    This is the one place the two harnesses can legitimately diverge, so it is a flag rather
    than a decision, and `--step-numbering crm_verbatim` measures what it is worth. Judge the
    difference on the BoN accuracy it produces; if it is large, say so in the write-up, because
    then the comparison is partly a measurement of prompt formatting.
    """
    fragments = _STEP_SPLIT.split(response)
    if numbering == "crm_verbatim":
        return tuple(
            f"Step {i + 1}: {frag.strip()}" for i, frag in enumerate(fragments) if frag.strip()
        )
    if numbering != "one_based":
        raise ValueError(f"unknown step numbering {numbering!r}")
    kept = [frag.strip() for frag in fragments if frag.strip()]
    return tuple(f"Step {i + 1}: {frag}" for i, frag in enumerate(kept))


def load_reference_dataset(
    data_name: str,
    gsm8k_reference: str = "qintongli/GSM-Plus",
    math_reference: Optional[str] = None,
) -> list[dict]:
    """The gold-answer source, per CRM's `load_reference_dataset` (`eval_BoN.py:56-62`).

    CRM's shipped version takes no arguments yet is CALLED with two keyword arguments in the
    same file (`:66-70`, `:186-190`), so it raises `TypeError` on the first line of both its
    callers -- their published script cannot run as committed. The signature here is the one
    the call sites want; the bodies are theirs.

    gsm8k -> GSM-Plus `testmini`, keyed `question` / `answer`.
    math  -> MATH-500 test, keyed `problem` / `solution`. A local `test.jsonl` path if given
             (CRM's `MATH_REFERENCE`), else the HF copy `skyline.py` already joins against.
    """
    if data_name == "gsm8k":
        from datasets import load_dataset

        return list(load_dataset(gsm8k_reference)["testmini"])

    if math_reference and Path(math_reference).exists():
        with open(math_reference) as handle:
            return [json.loads(line) for line in handle if line.strip()]

    from datasets import load_dataset

    return list(load_dataset("HuggingFaceH4/MATH-500", split="test"))


def load_candidates(
    data_file: str | Path,
    data_name: str,
    reference: Sequence[dict],
    numbering: str = "one_based",
    max_candidates: int = max(SAMPLE_NUMS),
) -> list[list[Candidate]]:
    """Read one `*-128.json` and align it to the reference dataset, question by question.

    The alignment asserts are CRM's (`eval_BoN.py:196-203`) and they are the reason this is
    safe to compare: the candidate file carries its own question text, so a silently reordered
    or differently-sized reference set fails here rather than producing a plausible number.
    """
    raw = json.loads(Path(data_file).read_text())
    if len(raw) != len(reference):
        raise AssertionError(
            f"{data_file}: {len(raw)} questions but the reference set has {len(reference)}. "
            "The BoN files are index-aligned to the reference dataset; a mismatch means the "
            "wrong reference (GSM-Plus testmini for gsm8k, MATH-500 test for math)."
        )

    gold_key = "question" if data_name == "gsm8k" else "problem"
    out: list[list[Candidate]] = []
    for idx, (row, gold) in enumerate(zip(raw, reference)):
        if row["question"].strip() != gold[gold_key].strip():
            raise AssertionError(
                f"{data_file}: question {idx} does not match the reference set.\n"
                f"  candidates: {row['question'][:120]!r}\n  reference:  {gold[gold_key][:120]!r}"
            )
        responses = row["responses"][:max_candidates]
        if len(responses) < max(SAMPLE_NUMS):
            raise AssertionError(
                f"{data_file}: question {idx} has {len(responses)} responses, and the ladder "
                f"tops out at {max(SAMPLE_NUMS)}. A short question would make best_of_128 a "
                "best_of_fewer and quietly flatter the model."
            )
        group = []
        for rank, response in enumerate(responses):
            text = response["text"]
            steps = split_response_into_steps(text, numbering)
            empty = not steps
            if empty:
                # CRM's fallback (`eval_BoN.py:217-219`): a response with no "Step N:" marker
                # still has to occupy its slot in the pool, so it gets one placeholder step
                # and whatever score that earns. Dropping it would change N.
                steps = ("\n",)
            group.append(
                Candidate(idx=idx, rank=rank, prompt=row["question"], response=text,
                          steps=steps, empty=empty)
            )
        out.append(group)
    return out


# --------------------------------------------------------------------------------------
# aggregators: Delta_1..Delta_T (+ d_0) -> one number per solution, higher is better
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Aggregator:
    name: str
    fn: Callable[[np.ndarray, float, float], float]
    why: str


def _neg_final_distance(deltas: np.ndarray, d0: float, tau: float) -> float:
    return -(d0 + float(deltas.sum()))


def _neg_sum_delta(deltas: np.ndarray, d0: float, tau: float) -> float:
    return -float(deltas.sum())


# "avg", not "mean", and the name is load-bearing: `tests/test_grep_invariants.py`'s root
# cause D guard bans the substring `g_mean` package-wide, and `neg_mean_delta` contains it.
# The guard is right and must not be relaxed -- it is what stops a latent-space centroid
# reappearing. This is a mean of SCALARS, which is fine; the ban is on `distance to a mean`.
def _neg_avg_delta(deltas: np.ndarray, d0: float, tau: float) -> float:
    return -float(deltas.mean())


def _neg_max_delta(deltas: np.ndarray, d0: float, tau: float) -> float:
    return -float(deltas.max())


def _neg_sum_excess(deltas: np.ndarray, d0: float, tau: float) -> float:
    return -float(np.maximum(deltas - tau, 0.0).sum())


def _log_survival(deltas: np.ndarray, d0: float, tau: float) -> float:
    # log prod_i sigma(-(Delta_i - tau)) = -sum_i softplus(Delta_i - tau), in a stable form.
    x = deltas - tau
    return -float(np.logaddexp(0.0, x).sum())


AGGREGATORS: tuple[Aggregator, ...] = (
    Aggregator(
        "log_survival", _log_survival,
        "the structural mirror of CRM's own aggregation. CRM scores prod_i (1 - h_i) with "
        "h_i = sigma(logit_i) a per-step error probability; read h_i = sigma(Delta_i - tau) "
        "-- tau is exactly the point §9.1 calls a step bad -- and this is log of the same "
        "product. The closest thing to scoring both models the same way.",
    ),
    Aggregator(
        "neg_max_delta", _neg_max_delta,
        "-max_i Delta_i. `max Delta > tau` is the detection half of §9.1 verbatim (§9.6.1), "
        "so this is the ProcessBench flagging statistic used as a ranking rather than a "
        "threshold. It is the aggregator with the most measurement behind it: §9.6.3 puts its "
        "detection AUC at 0.639 on gsm8k and ~0.53 on the hard subsets.",
    ),
    Aggregator(
        "neg_sum_excess", _neg_sum_excess,
        "-sum_i relu(Delta_i - tau). The hinge form of log_survival -- steps at or below the "
        "ruler cost exactly nothing, so a long correct solution is not penalised for being "
        "long. Same shape as ⑥ L_good's own penalty (§7.12), which is what the model was "
        "trained to keep at zero.",
    ),
    Aggregator(
        "neg_final_distance", _neg_final_distance,
        "-d(psi_T, g_q): how far the FINISHED solution is from the goal. The quasimetric's "
        "own solution-level readout and the only aggregator here that uses d rather than "
        "Delta. **Within a question it ranks identically to -sum_i Delta_i**, because d_0 "
        "depends only on the prompt (§7.7: h_{s_0} is identical across a question's "
        "candidates under causal attention) and so is a constant offset shared by all 128. "
        "Reported separately anyway because its LEVEL is comparable across questions.",
    ),
    Aggregator(
        "neg_sum_delta", _neg_sum_delta,
        "-sum_i Delta_i. Kept as the explicit check on the sentence above: it must produce "
        "BoN accuracies bit-identical to neg_final_distance, and `tests/test_bon.py` asserts "
        "the ranking equivalence. If the two ever disagree, d_0 is varying within a question "
        "and something upstream is wrong.",
    ),
    Aggregator(
        "neg_avg_delta", _neg_avg_delta,
        "-mean_i Delta_i. The length control. BoN candidates for one question vary in step "
        "count, and every summed aggregator above grows with T -- on a trained ruler each "
        "good step contributes -log gamma = -0.693, so a LONGER correct solution scores "
        "BETTER under a sum. This one removes that. The gap between it and neg_sum_delta is "
        "a direct measurement of how much of any BoN gain is really a length preference.",
    ),
)

AGGREGATOR_BY_NAME = {a.name: a for a in AGGREGATORS}


def aggregate_all(deltas: np.ndarray, d0: float, tau: float) -> dict[str, float]:
    if deltas.size == 0:
        return {a.name: -math.inf for a in AGGREGATORS}
    return {a.name: a.fn(deltas, d0, tau) for a in AGGREGATORS}


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------


@dataclass
class ScoredPool:
    """Per-candidate scores for one file, plus the counters that say whether to trust them."""

    scores: dict[str, np.ndarray]       # aggregator -> (n_questions, n_candidates) float64
    n_steps: np.ndarray                 # (n_questions, n_candidates) int64, 0 if unscorable
    counters: dict[str, float] = field(default_factory=dict)


@torch.no_grad()
def score_pool(
    model,
    tokenizer,
    pool: Sequence[Sequence[Candidate]],
    cfg: Config,
    device,
    tau: float,
    max_len: int,
    batch_sequences: int,
    max_padded_tokens: int,
    label: str = "bon",
) -> ScoredPool:
    """Score every candidate of every question. One forward per candidate (§3).

    **An over-length candidate is scored -inf, never dropped.** ProcessBench drops and counts
    (`processbench.py:150-152`), which is right there -- a dropped sample predicts -1 and is
    still scored by the metric. Here dropping would remove a member of the pool and turn
    best_of_128 into best_of_127 on that question, i.e. it would change the task. -inf leaves
    it in the pool, last, and it is only ever selected if the whole pool is over-length.
    """
    sep_id = sep_token_id(tokenizer, cfg.data.sep_token)
    pad_id = tokenizer.pad_token_id
    n_q, n_c = len(pool), len(pool[0])

    scores = {a.name: np.full((n_q, n_c), -np.inf, dtype=np.float64) for a in AGGREGATORS}
    n_steps = np.zeros((n_q, n_c), dtype=np.int64)
    counters = {"over_length": 0.0, "empty_response": 0.0, "unscorable": 0.0, "scored": 0.0}

    pending: list[tuple[int, int, SequenceRow]] = []
    pending_max_len = 0
    progress = Progress(f"bon/score/{label}", n_q * n_c)

    def flush() -> None:
        nonlocal pending_max_len
        if not pending:
            return
        batch = collate([row for _, _, row in pending], pad_id=pad_id).to(device)
        reps = model(batch)
        h_s0 = reps.h_states.index_select(0, batch.traj_state_offset)
        goals = model.goal_head(h_s0)
        for b, (qi, ci, _) in enumerate(pending):
            T = int(batch.traj_T[b])
            offset = int(batch.traj_state_offset[b])
            states = reps.psi[offset : offset + T + 1]
            d = model.distance(states, goals[b].expand_as(states)).float().cpu().numpy()
            deltas = d[1:] - d[:-1]
            for name, value in aggregate_all(deltas, float(d[0]), tau).items():
                scores[name][qi, ci] = value
            n_steps[qi, ci] = T
            counters["scored"] += 1
        progress.advance(len(pending))
        pending.clear()
        pending_max_len = 0

    for qi, group in enumerate(pool):
        for ci, cand in enumerate(group):
            if cand.empty:
                counters["empty_response"] += 1
            try:
                seq = build_sequence(
                    tokenizer,
                    cand.prompt,
                    list(cand.steps),
                    sep_id,
                    prompt_format=cfg.data.prompt_format,
                    max_len=max_len,
                    add_prefix=False,          # the splitter already wrote "Step N: " (locked #8)
                )
            except SequenceTooLong:
                counters["over_length"] += 1
                counters["unscorable"] += 1
                progress.advance(1)
                continue
            except (EmptyStep, ValueError):
                counters["unscorable"] += 1
                progress.advance(1)
                continue

            row = SequenceRow(
                qid=str(qi),
                input_ids=np.asarray(seq.input_ids, dtype=np.int64),
                state_pos=np.asarray(seq.state_pos, dtype=np.int64),
                span_start=np.asarray([s for s, _ in seq.step_spans], dtype=np.int64),
                span_end=np.asarray([e for _, e in seq.step_spans], dtype=np.int64),
                correct=True,
                z=-1,
            )
            # Two budgets, as in §8.1.2: a batch costs len(batch) x its longest row, and with
            # 128 candidates per question the long ones cluster. Close on whichever binds.
            new_max = max(pending_max_len, len(seq.input_ids))
            if pending and (
                len(pending) >= batch_sequences
                or (len(pending) + 1) * new_max > max_padded_tokens
            ):
                flush()
                new_max = len(seq.input_ids)
            pending.append((qi, ci, row))
            pending_max_len = new_max
    flush()

    total = float(n_q * n_c)
    counters["over_length_fraction"] = counters["over_length"] / total if total else 0.0
    return ScoredPool(scores=scores, n_steps=n_steps, counters=counters)


# --------------------------------------------------------------------------------------
# selection: CRM's semantics, exactly
# --------------------------------------------------------------------------------------


def take_first_n(n: int, n_candidates: int) -> list[int]:
    """CRM's `split_query` (`eval_BoN.py:35-41`) reduced to what it actually does.

    It sorts each question's candidates by `logprobs` descending and keeps the first `n`. But
    `load_queries` (`:207`) sets `"logprobs": 0` on every single candidate, so the key is
    constant, and Python's sort is stable -- the result is the first `n` in FILE order, for
    every n. `tests/test_bon.py` asserts the two forms agree on a shuffled fixture.

    This matters more than it looks: it means the N ladder is nested (the best_of_8 pool is a
    subset of the best_of_16 pool), so accuracy is expected to be roughly monotone in N and a
    non-monotone curve is a signal, not noise in the subsetting.
    """
    return list(range(min(n, n_candidates)))


def best_of_n(scores: np.ndarray, n: int) -> np.ndarray:
    """Pick one candidate index per question from the first `n`, argmax score.

    CRM sorts by reward descending and takes `[0]` (`eval_BoN.py:44-50`). Python's sort is
    stable, so a tie goes to the EARLIEST candidate -- `np.argmax` has the same tie rule, and
    that equivalence is what makes the -inf over-length score safe: a question whose whole
    pool is unscorable falls back to candidate 0, which is the no-reranking baseline.
    """
    return np.asarray([row[:n].argmax() for row in scores], dtype=np.int64)


# --------------------------------------------------------------------------------------
# grading, through CRM's judge
# --------------------------------------------------------------------------------------


class PoolGrader:
    """Grades (question, candidate) pairs through CRM's grader, memoised.

    Memoisation is not an optimisation detail. Six aggregators x five N values means 30
    selection sets per file, and they overlap heavily -- without the cache the math subsets
    would run `math_equal` (sympy, with a `signal.alarm` per call) tens of thousands of times
    for answers it has already judged. With it, each distinct candidate is judged once and
    every table in the output reads the same verdict for it.
    """

    def __init__(self, data_name: str, pool: Sequence[Sequence[Candidate]], reference: Sequence[dict]):
        self.data_name = data_name
        self.pool = pool
        self.reference = reference
        self._cache: dict[tuple[int, int], bool] = {}

    def grade(self, picks: Sequence[int]) -> list[bool]:
        """`picks[q]` is the chosen candidate rank for question q. Returns per-question
        correctness, in question order."""
        wanted = [(q, int(c)) for q, c in enumerate(picks) if (q, int(c)) not in self._cache]
        if wanted:
            self._judge(wanted)
        return [self._cache[(q, int(c))] for q, c in enumerate(picks)]

    def grade_all(self) -> np.ndarray:
        """(n_questions, n_candidates) correctness. The oracle and mean-candidate baselines
        need it; it is the expensive call on the math subsets and is behind a flag."""
        every = [(q, c) for q in range(len(self.pool)) for c in range(len(self.pool[0]))]
        todo = [key for key in every if key not in self._cache]
        if todo:
            self._judge(todo)
        return np.asarray(
            [[self._cache[(q, c)] for c in range(len(self.pool[0]))] for q in range(len(self.pool))],
            dtype=bool,
        )

    def _judge(self, keys: Sequence[tuple[int, int]]) -> None:
        from .crm_grader import eval_gsm8k, eval_math_prm

        responses = [{"response": self.pool[q][c].response} for q, c in keys]
        if self.data_name == "gsm8k":
            for q, _ in keys:
                # CRM's own alignment assert (`eval_BoN.py:88-91`), kept per grading batch.
                assert self.pool[q][0].prompt.strip() == self.reference[q]["question"].strip()
            answers = [self.reference[q]["answer"] for q, _ in keys]
            _, correct, _ = eval_gsm8k(responses, answers=answers, is_extract=True)
        else:
            problems = [
                {"solution": self.reference[q]["solution"], "question": self.reference[q]["problem"]}
                for q, _ in keys
            ]
            _, correct, _ = eval_math_prm(responses, all_problems=problems, is_extract=False)
        for key, ok in zip(keys, correct):
            self._cache[key] = bool(ok)


# --------------------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------------------


def accuracy(correct: Iterable[bool]) -> float:
    values = list(correct)
    return 100.0 * sum(values) / len(values) if values else 0.0


def evaluate_pool(
    scored: ScoredPool,
    grader: PoolGrader,
    n_candidates: int,
    grade_all: bool,
) -> dict:
    """BoN accuracy for every aggregator at every N, plus the three baselines.

    **Read the baselines first.** `first_candidate` is what you get with no reward model at
    all, and every BoN number in the table has to be read as a delta against it -- CRM's
    paper reports the same quantity, so it is also the check that this harness reproduces
    their setup at all. If `first_candidate` here does not match the paper's no-RM row, the
    candidate files or the grader differ and NOTHING else in the table is comparable.
    """
    results: dict = {}
    n_questions = len(grader.pool)

    picks_first = np.zeros(n_questions, dtype=np.int64)
    results["baseline_first_candidate"] = accuracy(grader.grade(picks_first))

    for agg in AGGREGATORS:
        table = {}
        for n in SAMPLE_NUMS:
            if n > n_candidates:
                continue
            picks = best_of_n(scored.scores[agg.name], n)
            table[f"best_of_{n}"] = accuracy(grader.grade(picks))
        results[agg.name] = table

    if grade_all:
        grid = grader.grade_all()
        results["baseline_mean_candidate"] = 100.0 * float(grid.mean())
        results["oracle"] = {
            f"best_of_{n}": 100.0 * float(grid[:, :n].any(axis=1).mean())
            for n in SAMPLE_NUMS
            if n <= n_candidates
        }
    return results


# --------------------------------------------------------------------------------------
# aggregator selection -- on Math-Shepherd val, NEVER on the BoN files (§9.2's rule)
# --------------------------------------------------------------------------------------


@torch.no_grad()
def select_aggregator_on_val(model, rows, cfg: Config, device, pad_id: int, tau: float) -> dict:
    """Pick the headline aggregator on held-out validation QUESTIONS.

    §9.2 fits `tau` on Math-Shepherd val and never on ProcessBench, for the obvious reason.
    An aggregator is the same kind of free parameter -- six of them, evaluated on the test set,
    is six chances to be lucky -- so it gets the same treatment.

    The val proxy is a BoN task built out of the data we already have: each validation question
    contributes its correct and incorrect trajectories as a candidate pool, and the score is
    the fraction of questions whose TOP-RANKED trajectory is correct. Same selection rule as
    BoN, same argmax, same tie-break.

    Two ways it is NOT the BoN task, both stated because they bound what this choice is worth:
      * the pool is 2-9 Math-Shepherd trajectories, not 128 sampled ones, so it never exercises
        the tail of a 128-wide pool;
      * "correct" is Math-Shepherd's per-step labels, not a graded final answer.
    It is a legal, leak-free way to break a six-way tie. It is not a prediction of the BoN
    number, and `--aggregator NAME` overrides it if you would rather fix the choice by hand.
    """
    by_question: dict[str, list[tuple[bool, float, np.ndarray]]] = {}
    pending: list[SequenceRow] = []

    def flush() -> None:
        if not pending:
            return
        batch = collate(pending, pad_id=pad_id).to(device)
        reps = model(batch)
        h_s0 = reps.h_states.index_select(0, batch.traj_state_offset)
        goals = model.goal_head(h_s0)
        for b, row in enumerate(pending):
            T = int(batch.traj_T[b])
            offset = int(batch.traj_state_offset[b])
            states = reps.psi[offset : offset + T + 1]
            d = model.distance(states, goals[b].expand_as(states)).float().cpu().numpy()
            by_question.setdefault(row.qid, []).append(
                (bool(row.correct), float(d[0]), d[1:] - d[:-1])
            )
        progress.advance(len(pending))
        pending.clear()

    progress = Progress("bon/select/val", len(rows))
    for row in rows:
        pending.append(row)
        if len(pending) >= cfg.eval.batch_sequences:
            flush()
    flush()

    # Only questions with both a correct and an incorrect trajectory can be got wrong.
    usable = {
        qid: entries
        for qid, entries in by_question.items()
        if any(c for c, _, _ in entries) and any(not c for c, _, _ in entries)
    }
    if not usable:
        raise RuntimeError(
            "no validation question has both a correct and an incorrect trajectory, so no "
            "aggregator can be told apart from any other. Check that sequences.parquet holds "
            "a 'val' split (scripts/prepare_data.py) before using --aggregator auto."
        )
    summary: dict = {
        "n_val_questions": len(usable),
        "mean_pool_size": float(np.mean([len(v) for v in usable.values()])),
        # The score a coin flip over each question's own pool would get. This is what an
        # aggregator has to beat to have done anything at all.
        "baseline_random": float(
            np.mean([sum(c for c, _, _ in v) / len(v) for v in usable.values()])
        ),
        "selection_accuracy": {},
    }
    for agg in AGGREGATORS:
        hits = 0
        for entries in usable.values():
            values = [agg.fn(deltas, d0, tau) for _, d0, deltas in entries]
            # `max` returns the FIRST maximum, which is `best_of_n`'s tie rule (np.argmax).
            best = max(range(len(values)), key=values.__getitem__)
            hits += int(entries[best][0])
        summary["selection_accuracy"][agg.name] = hits / len(usable)

    summary["chosen"] = max(
        summary["selection_accuracy"], key=lambda name: summary["selection_accuracy"][name]
    )
    return summary


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def resolve_tau(checkpoint: Path, cfg: Config, override: Optional[float]) -> tuple[float, str]:
    """tau enters three of the six aggregators. Take the one the reported eval used.

    Order: an explicit --tau; then the checkpoint's own `processbench.json`, which stores the
    tau `calibrate.py` fit on Math-Shepherd val; then the natural midpoint, LOUDLY, because
    §9.2 measured the fitted tau at 1.1685 against a natural 0.3466 on this checkpoint -- a
    3.37x gap, so the fallback is not a near-substitute for the fit.
    """
    if override is not None:
        return float(override), "--tau"
    report = checkpoint / "processbench.json"
    if report.exists():
        payload = json.loads(report.read_text())
        value = payload.get("calibration", {}).get("calibration/tau")
        if value is not None:
            return float(value), f"{report}"
    print(
        f"!! no fitted tau found ({report} absent or has no calibration block). Falling back "
        f"to the natural midpoint {natural_tau(cfg):.4f}. §9.2 measured the FITTED tau at "
        "3.37x that on runs/phase1/phase2/final, so this is a different threshold, not a "
        "rounding of the same one. Pass --tau explicitly, or run the ProcessBench eval first.",
        flush=True,
    )
    return natural_tau(cfg), "natural_tau (NOT fitted)"


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ..data.math_shepherd import read_sequences_parquet
    from ..model.backbone import load_backbone_with_adapter, load_tokenizer, read_hidden_size
    from ..model.wrapper import FeynmanPRM
    from ..utils.checkpoint import load_config_from_checkpoint, load_heads

    parser = argparse.ArgumentParser(description="Best-of-N eval, comparable with CRM's")
    parser.add_argument("--checkpoint", required=True, help="a PHASE-2 checkpoint (goal head)")
    parser.add_argument("--data-file", required=True, help="one CRM *-128.json candidate file")
    parser.add_argument("--data-name", required=True, choices=["gsm8k", "math"])
    parser.add_argument("--save-file", default=None)
    parser.add_argument("--gsm8k-reference", default="qintongli/GSM-Plus")
    parser.add_argument("--math-reference", default=None, help="MATH-500 test.jsonl (else HF)")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument(
        "--aggregator", default="auto",
        help="'auto' picks it on Math-Shepherd val (§9.2's rule); or name one of "
             + ", ".join(a.name for a in AGGREGATORS),
    )
    parser.add_argument("--step-numbering", default="one_based",
                        choices=["one_based", "crm_verbatim"])
    parser.add_argument("--max-len", type=int, default=None,
                        help="default cfg.eval.max_len. Over-length candidates score -inf, "
                             "they are never dropped -- dropping would change N.")
    parser.add_argument("--batch-sequences", type=int, default=None)
    parser.add_argument("--max-padded-tokens", type=int, default=32768)
    parser.add_argument("--grade-all-candidates", action="store_true",
                        help="also grade all 128 per question, for the mean-candidate and "
                             "oracle (pass@N) baselines. Cheap on gsm8k (a regex), slow on "
                             "math (sympy per call).")
    parser.add_argument("--allow-degraded-grader", action="store_true",
                        help="proceed on math even though sympy's LaTeX parser is unusable. "
                             "The result JSON records it; the number is not comparable.")
    args = parser.parse_args(argv)

    # Before the model loads, not after the scoring pass: a grader that silently lost
    # parse_latex marks LaTeX answers wrong that CRM's accepted (crm_grader/__init__.py).
    from .crm_grader import assert_grader_environment, latex_parser_available

    if not args.allow_degraded_grader:
        assert_grader_environment(args.data_name)

    ckpt = Path(args.checkpoint)
    cfg = load_config_from_checkpoint(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(cfg)
    backbone = load_backbone_with_adapter(cfg, ckpt / "adapter")
    model = FeynmanPRM(cfg, read_hidden_size(cfg.model.name), backbone=backbone, with_goal_head=True)
    # No allow_missing: a checkpoint with no goal head must abort, not score through a fresh
    # random one (§9.3.2's audit, link 2).
    load_heads(model, ckpt)
    model.to(device).eval()

    tau, tau_source = resolve_tau(ckpt, cfg, args.tau)
    max_len = args.max_len if args.max_len is not None else cfg.eval.max_len
    batch_sequences = args.batch_sequences or cfg.eval.batch_sequences
    print(f"tau = {tau:.4f} (from {tau_source}); max_len = {max_len}", flush=True)

    reference = load_reference_dataset(args.data_name, args.gsm8k_reference, args.math_reference)
    pool = load_candidates(args.data_file, args.data_name, reference, args.step_numbering)
    n_candidates = len(pool[0])
    print(f"{args.data_file}: {len(pool)} questions x {n_candidates} candidates", flush=True)

    results: dict = {
        "checkpoint": str(ckpt),
        "data_file": str(args.data_file),
        "data_name": args.data_name,
        "n_questions": len(pool),
        "n_candidates": n_candidates,
        "tau": tau,
        "tau_source": tau_source,
        "step_numbering": args.step_numbering,
        "max_len": max_len,
        "sample_nums": list(SAMPLE_NUMS),
        # Recorded per run, not assumed: it is the one environment fact that silently moves a
        # math number without moving anything the model did.
        "latex_parser_available": latex_parser_available(),
    }

    # ---- the headline aggregator, chosen on val. Never on this file. -------------------
    if args.aggregator == "auto":
        val_rows = read_sequences_parquet(Path(cfg.data.dir) / "sequences.parquet", split="val")
        selection = select_aggregator_on_val(model, val_rows, cfg, device, tokenizer.pad_token_id, tau)
        results["val_selection"] = selection
        primary = selection["chosen"]
        print(
            f"aggregator chosen on val: {primary} "
            f"(val selection accuracy {selection['selection_accuracy'][primary]:.4f} against a "
            f"random baseline of {selection['baseline_random']:.4f})",
            flush=True,
        )
    else:
        if args.aggregator not in AGGREGATOR_BY_NAME:
            parser.error(f"unknown aggregator {args.aggregator!r}")
        primary = args.aggregator
    results["primary_aggregator"] = primary

    scored = score_pool(
        model, tokenizer, pool, cfg, device, tau,
        max_len=max_len,
        batch_sequences=batch_sequences,
        max_padded_tokens=args.max_padded_tokens,
        label=Path(args.data_file).stem,
    )
    results["counters"] = scored.counters
    if scored.counters["over_length_fraction"] > 0.01:
        print(
            f"!! {scored.counters['over_length']:.0f} of {len(pool) * n_candidates} candidates "
            f"({100 * scored.counters['over_length_fraction']:.2f}%) exceed max_len {max_len} "
            "and are ranked last. Above ~1% that is enough to move a BoN number on its own -- "
            "raise --max-len and re-run before quoting this.",
            flush=True,
        )

    grader = PoolGrader(args.data_name, pool, reference)
    results["accuracy"] = evaluate_pool(scored, grader, n_candidates, args.grade_all_candidates)

    print(json.dumps({"primary": primary, **results["accuracy"].get(primary, {}),
                      "baseline_first_candidate": results["accuracy"]["baseline_first_candidate"]},
                     indent=2), flush=True)

    out_path = Path(args.save_file) if args.save_file else ckpt / f"bon_{Path(args.data_file).stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {out_path}", flush=True)

    # Same discipline as `deltas.npz` (§9.3's rationale): scoring is the only expensive part,
    # and every re-aggregation, every N, and every future aggregator is a pure function of
    # these arrays. Discarding them turns each of those into another GPU run.
    npz_path = out_path.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        n_steps=scored.n_steps,
        **{f"score/{name}": array for name, array in scored.scores.items()},
    )
    print(f"wrote {npz_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
