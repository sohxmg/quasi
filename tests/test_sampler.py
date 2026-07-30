"""§15's batch-composition and step-count tests (§8.1, §8.2, §11.1)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from feynman_prm.data.sampler import (
    batch_stats,
    build_question_slots,
    epoch_batches,
    expected_sequences_per_question,
    longest_batch_index,
    planned_optimizer_steps,
    steps_report,
)
from conftest import synthetic_row

T, F = True, False


def _rows(spec: list[tuple[str, int, int]]):
    """spec = [(qid, n_correct, n_incorrect), ...]"""
    rows = []
    for qid, n_c, n_i in spec:
        rows += [synthetic_row(qid, [T, T]) for _ in range(n_c)]
        rows += [synthetic_row(qid, [T, F]) for _ in range(n_i)]
    return rows


def _sized(cfg, rows, **sampling):
    """A config whose `data.max_len` matches these synthetic rows, so the token cap's
    one-question floor (§8.1) is checked against the fixture and not against the real 1024."""
    return dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, max_len=max(r.length for r in rows)),
        sampling=dataclasses.replace(cfg.sampling, group_by_length=False, **sampling),
    )


def test_caps_are_caps_not_quotas(cfg):
    """§4.2.1: only 20.0% of questions have >=4 correct and the median has 2, so a hard quota
    would discard most of the dataset. E[seqs] = min(4,k_c) + min(3,k_i) = 4.33, NEVER
    k_c + k_i, which is the 9.18 assumption that made the run 2.1x short (§8.2)."""
    slots = build_question_slots(_rows([("a", 1, 1), ("b", 10, 10), ("c", 2, 2)]))
    assert [s.allocation(4, 3) for s in slots] == [2, 7, 4]
    assert expected_sequences_per_question(slots, cfg) == pytest.approx((2 + 7 + 4) / 3)


def test_short_questions_are_not_dropped(cfg):
    slots = build_question_slots(_rows([("a", 1, 1)]))
    rows = _rows([("a", 1, 1)])
    batches = epoch_batches(rows, slots, cfg, 0, np.random.default_rng(0))
    assert sum(len(b) for b in batches) == 2


def test_a_question_is_never_split_across_batches(cfg):
    """PLAN 4: a partially included question can end up with 0 correct or 0 incorrect, which
    silently produces goal-less rows and zero L_step pairs. Emit the batch SHORT instead."""
    small = dataclasses.replace(
        cfg, sampling=dataclasses.replace(cfg.sampling, sequences_per_micro_batch=8,
                                          group_by_length=False)
    )
    rows = _rows([(f"q{i}", 4, 3) for i in range(5)])     # 7 sequences each
    slots = build_question_slots(rows)
    batches = epoch_batches(rows, slots, small, 0, np.random.default_rng(0))
    for batch in batches:
        by_q: dict[str, int] = {}
        for i in batch:
            by_q[rows[i].qid] = by_q.get(rows[i].qid, 0) + 1
        for qid, count in by_q.items():
            assert count == 7, f"{qid} was split across batches"
        assert len(batch) <= 8


def test_every_question_is_visited_once_per_epoch(cfg):
    rows = _rows([(f"q{i}", 2, 2) for i in range(30)])
    slots = build_question_slots(rows)
    batches = epoch_batches(rows, slots, cfg, 0, np.random.default_rng(0))
    seen = [rows[i].qid for batch in batches for i in batch]
    assert len(set(seen)) == 30
    assert len(seen) == 30 * 4


def test_group_by_length_reduces_padding(cfg):
    """~60% -> ~10% padding, estimated 4h -> 2h/epoch (PLAN 4a). Q and every §8.1.1 count are
    unchanged either way -- only the flavour mix per batch changes."""
    rows = []
    for i in range(40):
        length = 2 if i % 2 == 0 else 20
        rows.append(synthetic_row(f"q{i}", [T, T], step_len=length))
        rows.append(synthetic_row(f"q{i}", [T, F], step_len=length))
    slots = build_question_slots(rows)

    grouped = dataclasses.replace(
        cfg, sampling=dataclasses.replace(cfg.sampling, sequences_per_micro_batch=8,
                                          group_by_length=True)
    )
    spec_order = dataclasses.replace(
        grouped, sampling=dataclasses.replace(grouped.sampling, group_by_length=False)
    )
    pad_grouped = batch_stats(
        epoch_batches(rows, slots, grouped, 0, np.random.default_rng(0)), rows
    )["padding_fraction"]
    pad_plain = batch_stats(
        epoch_batches(rows, slots, spec_order, 0, np.random.default_rng(0)), rows
    )["padding_fraction"]
    assert pad_grouped < pad_plain


def test_longest_batch_is_identifiable_for_the_memory_probe(cfg):
    """Peak memory moves to the longest bucket (~9 -> ~12 GB of 16), so train.py runs that
    batch FIRST: an OOM then surfaces in 30 seconds instead of three hours in (PLAN 4a)."""
    rows = [synthetic_row("a", [T, T], step_len=2), synthetic_row("b", [T, T], step_len=50)]
    batches = [[0], [1]]
    assert longest_batch_index(batches, rows) == 1


def test_the_token_cap_bounds_the_padded_shape(cfg):
    """The budget the PLAN 4a probe actually cares about: a batch costs len(batch) x max_len,
    not len(batch). The 2026-07-26 OOM was 56 sequences all at data.max_len = 57,344 tokens
    on a 15.46 GiB card, which the sequence budget alone permits."""
    rows = [synthetic_row(f"q{i}", [T, T] if k < 4 else [T, F], step_len=2 + 10 * (i % 3))
            for i in range(20) for k in range(7)]
    slots = build_question_slots(rows)
    cap = 12 * max(r.length for r in rows)

    capped = _sized(cfg, rows, max_padded_tokens=cap)
    batches = epoch_batches(rows, slots, capped, 0, np.random.default_rng(0))
    assert max(len(b) for b in batches) < capped.sampling.sequences_per_micro_batch  # cap binds
    for batch in batches:
        assert len(batch) * max(rows[i].length for i in batch) <= cap
    assert sum(len(b) for b in batches) == len(rows)      # nothing is dropped


def test_the_token_cap_leaves_short_batches_alone(cfg):
    """Why the cap and not a smaller sequence budget: it must bind ONLY where memory forces
    it, so L_NCE's negative pool survives on the 83% of batches that were never too big."""
    rows = _rows([(f"q{i}", 4, 3) for i in range(20)])
    slots = build_question_slots(rows)
    budget = cfg.sampling.sequences_per_micro_batch
    # Exactly non-binding: no batch can reach the sequence budget AND exceed this.
    generous = _sized(cfg, rows, max_padded_tokens=budget * max(r.length for r in rows))
    enormous = _sized(cfg, rows, max_padded_tokens=100 * budget * max(r.length for r in rows))

    batches = epoch_batches(rows, slots, generous, 0, np.random.default_rng(0))
    assert batches == epoch_batches(rows, slots, enormous, 0, np.random.default_rng(0))
    assert max(len(b) for b in batches) == budget         # the sequence budget still bound


def test_the_token_cap_cannot_be_set_below_one_question(cfg):
    """No question is ever split (PLAN 4), so a cap under one question's worst case would be
    exceeded by single-question batches and would bound nothing. Fail at config time."""
    from feynman_prm.config import ConfigError

    with pytest.raises(ConfigError, match="max_padded_tokens"):
        dataclasses.replace(
            cfg, sampling=dataclasses.replace(cfg.sampling, max_padded_tokens=1024)
        )


def test_optimizer_step_arithmetic_matches_section_11_1(cfg):
    """23,000 questions x 4.33 seqs = ~99,600 sequences; / (56 x 2) = ~889 steps, 27 warmup.
    A test that pins the LR schedule without pinning the step count is the hole that produced
    the 106-step run."""
    report = steps_report(round(23_000 * 4.33), cfg)
    assert report["optimizer_steps"] == pytest.approx(889, abs=2)
    assert report["warmup_steps"] == pytest.approx(27, abs=1)

    # The regression, so it stays recognisable: 9.18 seqs/question with grad_accum 8.
    old = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, grad_accum=8))
    assert steps_report(round(11_000 * 9.18 * 0.47), old)["optimizer_steps"] < 300


def test_min_optimizer_steps_guard_is_wired(cfg):
    from feynman_prm.train import MIN_OPTIMIZER_STEPS

    assert MIN_OPTIMIZER_STEPS == 300
    assert planned_optimizer_steps(212, 2) == 106      # the exact number §11.1 records
    assert planned_optimizer_steps(212, 2) < MIN_OPTIMIZER_STEPS


def test_batch_stats_report_the_diagnostics(cfg):
    rows = _rows([(f"q{i}", 4, 3) for i in range(30)])
    slots = build_question_slots(rows)
    stats = batch_stats(epoch_batches(rows, slots, cfg, 0, np.random.default_rng(0)), rows)
    assert stats["questions_per_batch_mean"] == pytest.approx(8.0, abs=0.5)   # 56/7
    assert stats["distinct_z_per_batch_mean"] > 0
    assert stats["step_pairs_per_batch_mean"] > stats["distinct_z_per_batch_mean"]
