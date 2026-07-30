"""Batch composition: the 56-sequence budget, the padded-token cap, the caps, and
length-grouped ordering (§8.1).

TWO budgets, and a batch is closed when it would break either: a count of SEQUENCES
(`sequences_per_micro_batch`, 56) and a count of PADDED TOKENS (`max_padded_tokens`,
32768, i.e. `len(batch) x max_len`). The token cap is a DIVERGENCE from §8.1's
sequences-only budget, measured and adopted 2026-07-27 -- see `_fits` for why.

Per question take `min(4, k_c)` correct and `min(3, k_i)` incorrect -- CAPS, not quotas:
only 20.0% / 29.5% of questions can fill them and the median question has 2 of each
(§4.2.1). Q is whatever fills the budget: ~12.9 in §8.1.1, 11.8 measured under the token
cap (`scripts/batch_report.py`).

**The rule for a question that does not fit** (§8.1 does not state it; PLAN 'Core design
decisions' 4 decides it): if the next question's full allocation does not fit, emit the
batch short and carry that question to the next batch. No question is ever split -- a
partially included question can end up with 0 correct or 0 incorrect trajectories, which
silently produces goal-less rows and zero L_step pairs.

One epoch = one visit per selected question, shuffled with the logged seed. A visit samples
`min(cap, available)`, so a question with 10 correct solutions contributes 4 and the other
6 are never seen that epoch: §2 #1's "take all trajectories" applies to dataset SELECTION,
not to what the sampler consumes (§8.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import Config
from .collate import SequenceRow

# HF's group_by_length scheme: sort inside megabatches of ~50 batches, not globally.
MEGABATCH_BATCHES = 50


@dataclass
class QuestionSlot:
    qid: str
    correct: tuple[int, ...]    # row indices into the row table
    incorrect: tuple[int, ...]

    def allocation(self, cap_correct: int, cap_incorrect: int) -> int:
        """How many sequences this question contributes: min(cap, available) each way."""
        return min(cap_correct, len(self.correct)) + min(cap_incorrect, len(self.incorrect))


def build_question_slots(rows: Sequence[SequenceRow]) -> list[QuestionSlot]:
    """Group row indices by question, splitting correct from incorrect."""
    correct: dict[str, list[int]] = {}
    incorrect: dict[str, list[int]] = {}
    order: list[str] = []
    for idx, row in enumerate(rows):
        if row.qid not in correct:
            correct[row.qid], incorrect[row.qid] = [], []
            order.append(row.qid)
        (correct if row.correct else incorrect)[row.qid].append(idx)
    return [
        QuestionSlot(qid=q, correct=tuple(correct[q]), incorrect=tuple(incorrect[q]))
        for q in order
    ]


def expected_sequences_per_question(slots: Sequence[QuestionSlot], cfg: Config) -> float:
    """E[min(4,k_c) + min(3,k_i)] -- measured, 4.33 in §4.2.1.

    NEVER k_c + k_i: that is the 9.18 assumption that made the run 2.1x short (§8.2).
    """
    if not slots:
        return 0.0
    caps = (cfg.sampling.max_correct_per_question, cfg.sampling.max_incorrect_per_question)
    return sum(s.allocation(*caps) for s in slots) / len(slots)


def planned_optimizer_steps(n_batches: int, grad_accum: int) -> int:
    """§11.1. Print this at startup and fail below 300 -- the old regression ran 106 steps."""
    return n_batches // grad_accum


def _allocate(slot: QuestionSlot, cfg: Config, rng: np.random.Generator) -> list[int]:
    """Sample this question's rows for one visit, without replacement."""
    take_c = min(cfg.sampling.max_correct_per_question, len(slot.correct))
    take_i = min(cfg.sampling.max_incorrect_per_question, len(slot.incorrect))
    chosen = []
    if take_c:
        chosen += [slot.correct[i] for i in rng.choice(len(slot.correct), take_c, replace=False)]
    if take_i:
        chosen += [
            slot.incorrect[i] for i in rng.choice(len(slot.incorrect), take_i, replace=False)
        ]
    return chosen


def _fits(n_seqs: int, max_len: int, budget: int, token_cap: int) -> bool:
    """Both budgets. The token cap exists because the sequence budget is the wrong unit.

    A batch costs `len(batch) x max_len` on the GPU, not `len(batch)`. With
    `group_by_length` on, that decouples the two hard: the median batch is 19,423 padded
    tokens and the largest is 57,344 (56 sequences all at `data.max_len`), because row
    lengths are heavily left-skewed -- p50 248, p90 514, only 0.7% above 896. The 57,344
    bucket OOMed the PLAN 4a probe on a 15.46 GiB card; PLAN 4a's "peak 9->12 GB" was
    computed off the MEAN batch and never covered the tail.

    Lowering the sequence budget instead (SETUP.md:344) shrinks all 1,832 batches to fix
    the 312 that are too big, and what it spends is L_NCE's negative pool: 28 sequences
    halves R from 364 to 175 and Q from 12.6 to 6.0, taking L_step's within-question pairs
    with it. The token cap costs 5% of the mean pool (364 -> 344) and NOTHING at the 10th
    percentile (177 -> 176), because the thin-pool batches are the SHORT ones -- few tokens
    per row means few steps per row means small R -- while the oversized batches are long
    sequences, which carry the LARGEST pools. The cap trims only where there is pool to
    spare. That is structural, not a lucky draw on this dataset, and it is the whole reason
    the two knobs are not interchangeable. Numbers: `scripts/batch_report.py`.

    32768 leaves ~4 GiB of headroom on a 16 GiB card. Raise it only against a re-run of
    that script: 40960 measures ~13.4 GiB, which is inside the error bar of an estimate
    calibrated on a single observed OOM.
    """
    return n_seqs <= budget and n_seqs * max_len <= token_cap


def epoch_batches(
    rows: Sequence[SequenceRow],
    slots: Sequence[QuestionSlot],
    cfg: Config,
    epoch: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """One epoch of micro-batches, each a list of row indices, under both budgets (`_fits`).

    With `group_by_length` the padding fraction falls from ~60% to ~10% (PLAN 4a), peak
    memory moves to the longest bucket, and batches become length-homogeneous -- which is
    mildly good for L_NCE (length stops being a shortcut cue) and mildly worse for gradient
    diversity. `group_by_length: false` reproduces the spec-literal order.

    The token cap binds on ~17% of batches and leaves the other 83% exactly as the sequence
    budget alone would have made them, so the §8.1.1 counts hold on the bulk of the epoch
    and shift only where memory forces it. `batch_stats` logs the realised distribution --
    read it, do not assume 56.
    """
    budget = cfg.sampling.sequences_per_micro_batch
    token_cap = cfg.sampling.max_padded_tokens
    allocations = [(_allocate(slot, cfg, rng)) for slot in slots]
    order = list(rng.permutation(len(slots)))

    if cfg.sampling.group_by_length:
        order = _length_grouped_order(order, allocations, rows, budget)

    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0                 # the padded width of `current`, which the cap is against
    for qi in order:
        alloc = allocations[qi]
        if not alloc:
            continue
        alloc_max = max(rows[i].length for i in alloc)
        if not _fits(len(current) + len(alloc), max(current_max, alloc_max), budget, token_cap):
            if current:
                batches.append(current)
            current, current_max = [], 0
        # A question whose own allocation breaks a budget is still emitted whole, in a batch
        # of its own: no question is ever split (PLAN 'Core design decisions' 4), and the
        # config asserts max_padded_tokens is large enough that this cannot happen.
        current.extend(alloc)
        current_max = max(current_max, alloc_max)
    if current:
        batches.append(current)

    if cfg.sampling.group_by_length:
        # Batch ORDER is shuffled so length is not correlated with training step, while the
        # length homogeneity inside a batch (the thing that removes the padding) is kept.
        batches = [batches[i] for i in rng.permutation(len(batches))]
    return batches


def _length_grouped_order(
    order: Sequence[int],
    allocations: Sequence[Sequence[int]],
    rows: Sequence[SequenceRow],
    budget: int,
) -> list[int]:
    """Sort questions by their longest allocated sequence inside megabatches of ~50 batches.

    Sorting on the MAX (not the mean) is what matters: padding is to the batch max, so the
    longest member of a question sets the cost of putting that question in a batch.
    """
    q_len = [
        max((rows[i].length for i in alloc), default=0) for alloc in allocations
    ]
    q_alloc = [len(alloc) for alloc in allocations]
    target = MEGABATCH_BATCHES * budget

    grouped: list[int] = []
    mega: list[int] = []
    filled = 0
    for qi in order:
        mega.append(qi)
        filled += q_alloc[qi]
        if filled >= target:
            grouped.extend(sorted(mega, key=lambda q: q_len[q], reverse=True))
            mega, filled = [], 0
    grouped.extend(sorted(mega, key=lambda q: q_len[q], reverse=True))
    return grouped


def longest_batch_index(batches: Sequence[Sequence[int]], rows: Sequence[SequenceRow]) -> int:
    """The batch whose padded shape is largest: n_sequences x max_length.

    train.py runs this one FIRST as a memory probe, so an OOM on the longest length-grouped
    bucket surfaces in 30 seconds instead of three hours in (PLAN 4a).
    """
    def cost(batch: Sequence[int]) -> int:
        return len(batch) * max(rows[i].length for i in batch)

    return max(range(len(batches)), key=lambda b: cost(batches[b]))


def batch_stats(batches: Sequence[Sequence[int]], rows: Sequence[SequenceRow]) -> dict[str, float]:
    """Realised composition, for the launch log and diagnostic #17."""
    if not batches:
        return {}
    seqs = [len(b) for b in batches]
    padded = sum(len(b) * max(rows[i].length for i in b) for b in batches)
    real = sum(rows[i].length for b in batches for i in b)
    questions = [len({rows[i].qid for i in b}) for b in batches]
    incorrect = [sum(not rows[i].correct for i in b) for b in batches]
    pairs = []
    for b in batches:
        by_q: dict[str, list[int]] = {}
        for i in b:
            by_q.setdefault(rows[i].qid, []).append(i)
        pairs.append(
            sum(
                sum(rows[i].correct for i in idxs) * sum(not rows[i].correct for i in idxs)
                for idxs in by_q.values()
            )
        )
    return {
        "n_batches": len(batches),
        "sequences_total": sum(seqs),
        "sequences_per_batch_mean": sum(seqs) / len(batches),
        "questions_per_batch_mean": sum(questions) / len(batches),   # Q, expect ~12.9
        "distinct_z_per_batch_mean": sum(incorrect) / len(batches),  # expect ~28
        "step_pairs_per_batch_mean": sum(pairs) / len(batches),      # expect ~64
        "padding_fraction": 1.0 - real / max(padded, 1),
        "max_padded_tokens": max(len(b) * max(rows[i].length for i in b) for b in batches),
    }


def steps_report(n_sequences: int, cfg: Config, n_batches: int | None = None) -> dict[str, float]:
    """§11.1's arithmetic, for the launch assert.

    Pass `n_batches` when the epoch has already been built: the sequence budget is only an
    UPPER bound on a batch now that `max_padded_tokens` also binds (~51 of 56 realised), so
    the budget-only form under-counts the steps a run will actually take. The budget-only
    form is kept for the planning path, which has no batches yet.
    """
    if n_batches is not None:
        steps = planned_optimizer_steps(n_batches, cfg.train.grad_accum)
    else:
        per_step = cfg.sampling.sequences_per_micro_batch * cfg.train.grad_accum
        steps = math.floor(n_sequences / per_step)
    return {
        "sequences_per_epoch": n_sequences,
        "optimizer_steps": steps,
        "warmup_steps": round(cfg.train.warmup_ratio * steps),
    }
