"""Batch composition: the 56-sequence budget, the caps, and length-grouped ordering (§8.1).

The budget is a fixed number of SEQUENCES, not of questions. Per question take
`min(4, k_c)` correct and `min(3, k_i)` incorrect -- CAPS, not quotas: only 20.0% / 29.5%
of questions can fill them and the median question has 2 of each (§4.2.1). Q is whatever
fills the budget, ~12.9 measured (§8.1.1).

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


def epoch_batches(
    rows: Sequence[SequenceRow],
    slots: Sequence[QuestionSlot],
    cfg: Config,
    epoch: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """One epoch of micro-batches, each a list of row indices (<= the sequence budget).

    With `group_by_length` the padding fraction falls from ~60% to ~10% (PLAN 4a), peak
    memory moves to the longest bucket, and batches become length-homogeneous -- which is
    mildly good for L_NCE (length stops being a shortcut cue) and mildly worse for gradient
    diversity. `group_by_length: false` reproduces the spec-literal order; Q and every
    §8.1.1 count are unchanged either way.
    """
    budget = cfg.sampling.sequences_per_micro_batch
    allocations = [(_allocate(slot, cfg, rng)) for slot in slots]
    order = list(rng.permutation(len(slots)))

    if cfg.sampling.group_by_length:
        order = _length_grouped_order(order, allocations, rows, budget)

    batches: list[list[int]] = []
    current: list[int] = []
    for qi in order:
        alloc = allocations[qi]
        if not alloc:
            continue
        if len(current) + len(alloc) > budget:
            if current:
                batches.append(current)
            current = []
        current.extend(alloc)
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


def steps_report(n_sequences: int, cfg: Config) -> dict[str, float]:
    """§11.1's arithmetic, for the launch assert."""
    per_step = cfg.sampling.sequences_per_micro_batch * cfg.train.grad_accum
    steps = math.floor(n_sequences / per_step)
    return {
        "sequences_per_epoch": n_sequences,
        "optimizer_steps": steps,
        "warmup_steps": round(cfg.train.warmup_ratio * steps),
    }
