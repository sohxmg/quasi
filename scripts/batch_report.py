#!/usr/bin/env python
"""What shape are the micro-batches actually, and what would a token cap cost?

The PLAN 4a memory probe OOMed on the longest length-grouped bucket: 56 sequences all at
`data.max_len` = 57,344 padded tokens, on a 15.46 GiB card. PLAN 4a's "peak mem 9->12 GB"
was computed off the MEAN batch; `group_by_length` moved the peak onto the all-1024 bucket
and the estimate never covered it.

Before changing the budget, measure. The choice is between shrinking EVERY batch (lower
`sequences_per_micro_batch`, the SETUP.md:344 remedy) and shrinking only the batches that
are actually too big (cap `len(batch) x max_len`). Which is right depends on how heavy the
long tail is -- if most batches sit far below the peak, a sequence cut pays globally for a
tail problem, and the thing it spends is L_NCE's negative pool (~347 rows, §8.1.1, the
number `wrapper.py` calls the structural win of the one-forward design).

    python scripts/batch_report.py
    python scripts/batch_report.py --caps 24576,32768,40960 --seq-budgets 28,36,40

Everything except the last section is EXACT: batch shapes and pool sizes are counted from
the real parquet under the real sampler RNG. The VRAM column is an extrapolation and is
labelled as one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run without `pip install -e .`

import argparse
import dataclasses
from typing import Sequence

import numpy as np

from feynman_prm.config import Config, load_config
from feynman_prm.data.collate import SequenceRow
from feynman_prm.data.math_shepherd import read_sequences_parquet
from feynman_prm.data.sampler import (
    _allocate,
    _length_grouped_order,
    build_question_slots,
    planned_optimizer_steps,
)
from feynman_prm.utils.seeding import epoch_rng

# ---- the VRAM extrapolation ---------------------------------------------------------
# Calibrated on the observed failure, NOT on first principles: a from-first-principles sum
# (28 x 1536 stored per token + 3 x 8960 recomputed) predicts ~10.8 GiB at 57,344 tokens and
# the run died needing more than 15.46, so the principled model underestimates by ~6 GiB and
# is useless for picking a cap. The linear fit below is anchored on the one hard measurement
# there is, and it is a LOWER bound: the OOM happened partway through backward, so true peak
# demand at 57,344 tokens is >= 16.7 GiB, not == it.
OOM_TOKENS = 57_344
OOM_DEMAND_GIB = 16.7          # 15.46 total capacity + the 1.91 GiB alloc that failed, less
                               # the 0.2 GiB still free at the time
WEIGHTS_GIB = 3.1              # 1.54B params bf16, resident regardless of batch shape
OPTIMIZER_GIB = 0.6            # AdamW fp32 moments + grads for 22.4M trainable params. NOT
                               # yet allocated when the probe runs, so the probe fitting is
                               # not enough -- step 1 needs this too.

# The bf16 -> fp32 activation cast PEFT inserts at every adapted projection, per token.
# `get_peft_model(autocast_adapter_dtype=True)` (the default, and what load_backbone takes)
# holds the LoRA weights in fp32 while the base is bf16, so `lora/layer.py` casts each input
# to fp32 and the cast is retained for backward. Input widths of the 7 targets:
#   q,k,v,o,gate,up = 1536 each, down = 8960 (intermediate_size)
LORA_INPUT_ELEMS_PER_TOKEN = 6 * 1536 + 8960
FP32_CAST_EXTRA_BYTES = LORA_INPUT_ELEMS_PER_TOKEN * 2     # fp32 (4B) minus bf16 (2B)


def gib(n_bytes: float) -> float:
    return n_bytes / 2**30


def estimate_peak_gib(tokens: int, with_optimizer: bool = True) -> float:
    per_token = (OOM_DEMAND_GIB - WEIGHTS_GIB) * 2**30 / OOM_TOKENS
    total = WEIGHTS_GIB + gib(tokens * per_token)
    return total + (OPTIMIZER_GIB if with_optimizer else 0.0)


def fp32_cast_gib(tokens: int) -> float:
    """How much of the peak is the fp32 adapter cast, one decoder layer live at a time."""
    return gib(tokens * FP32_CAST_EXTRA_BYTES)


# ---- batching ------------------------------------------------------------------------


def pack(
    order: Sequence[int],
    allocations: Sequence[Sequence[int]],
    rows: Sequence[SequenceRow],
    seq_budget: int,
    token_cap: int | None,
) -> list[list[int]]:
    """The sampler's packing loop with an optional `len(batch) x max_len` cap.

    Mirrors `sampler.epoch_batches` exactly when `token_cap is None`, including PLAN 'Core
    design decisions' 4: a question that does not fit is carried whole to the next batch,
    never split.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for qi in order:
        alloc = allocations[qi]
        if not alloc:
            continue
        alloc_max = max(rows[i].length for i in alloc)
        n = len(current) + len(alloc)
        wide = max(current_max, alloc_max)
        over_seqs = n > seq_budget
        over_tokens = token_cap is not None and n * wide > token_cap
        if over_seqs or over_tokens:
            if current:
                batches.append(current)
            current, current_max = [], 0
        current.extend(alloc)
        current_max = max(current_max, alloc_max)
    if current:
        batches.append(current)
    return batches


def epoch_inputs(rows, slots, cfg: Config, seq_budget: int):
    """The allocations and length-grouped order for epoch 0, under the run's own seed."""
    rng = epoch_rng(cfg.run.seed, 0)
    sized = dataclasses.replace(
        cfg, sampling=dataclasses.replace(cfg.sampling, sequences_per_micro_batch=seq_budget)
    )
    allocations = [_allocate(slot, sized, rng) for slot in slots]
    order = list(rng.permutation(len(slots)))
    if cfg.sampling.group_by_length:
        order = _length_grouped_order(order, allocations, rows, seq_budget)
    return allocations, order


# ---- measurement ----------------------------------------------------------------------


def padded_tokens(batch: Sequence[int], rows: Sequence[SequenceRow]) -> int:
    return len(batch) * max(rows[i].length for i in batch)


def nce_rows(batch: Sequence[int], rows: Sequence[SequenceRow]) -> int:
    """R, the (state, action) pool L_NCE normalises over -- one row per step (§8.1.1).

    The negatives a source competes against are R-1 of these, which is the ~347 in
    `nce.py`'s "log(R) ~= log(348) = 5.85" init check.
    """
    return sum(rows[i].n_steps for i in batch)


def goal_cols(batch: Sequence[int], rows: Sequence[SequenceRow]) -> int:
    """C, the goal columns: one per source row of a CORRECT trajectory (goals.py:62)."""
    return sum(rows[i].n_steps for i in batch if rows[i].correct)


def describe(batches, rows, cfg: Config, label: str) -> dict:
    toks = np.array([padded_tokens(b, rows) for b in batches])
    seqs = np.array([len(b) for b in batches])
    pool = np.array([nce_rows(b, rows) for b in batches])
    cols = np.array([goal_cols(b, rows) for b in batches])
    qs = np.array([len({rows[i].qid for i in b}) for b in batches])
    real = sum(rows[i].length for b in batches for i in b)
    steps = planned_optimizer_steps(len(batches), cfg.train.grad_accum) * cfg.train.epochs
    return {
        "label": label,
        "n_batches": len(batches),
        "opt_steps": steps,
        "seqs_mean": seqs.mean(),
        "tok_p50": np.percentile(toks, 50),
        "tok_p99": np.percentile(toks, 99),
        "tok_max": toks.max(),
        "pad_frac": 1.0 - real / toks.sum(),
        "pool_mean": pool.mean(),
        "pool_p10": np.percentile(pool, 10),
        "cols_mean": cols.mean(),
        "q_mean": qs.mean(),
        "peak_gib": estimate_peak_gib(int(toks.max())),
        "toks": toks,
        "pool": pool,
    }


def print_row(d: dict) -> None:
    print(
        f"  {d['label']:<26} {d['n_batches']:>6} {d['opt_steps']:>7} {d['seqs_mean']:>7.1f}"
        f" {d['tok_p50']:>9,.0f} {d['tok_max']:>9,.0f} {d['pad_frac']:>7.1%}"
        f" {d['pool_mean']:>8.0f} {d['pool_p10']:>7.0f} {d['q_mean']:>6.1f} {d['peak_gib']:>8.1f}"
    )


HEADER = (
    f"  {'configuration':<26} {'batches':>6} {'steps':>7} {'seqs':>7} {'tok p50':>9}"
    f" {'tok max':>9} {'pad':>7} {'pool R':>8} {'R p10':>7} {'Q':>6} {'~GiB':>8}"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--set", action="append", default=[], help="key.path=value override")
    parser.add_argument("--caps", default="24576,32768,40960,49152",
                        help="candidate len(batch) x max_len caps")
    parser.add_argument("--seq-budgets", default="28,36,40",
                        help="candidate sequences_per_micro_batch, for comparison")
    parser.add_argument("--capacity", type=float, default=15.46, help="GiB of the target card")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.set)
    rows = read_sequences_parquet(Path(cfg.data.dir) / "sequences.parquet", split="train")
    slots = build_question_slots(rows)
    budget = cfg.sampling.sequences_per_micro_batch

    # ---- 1. the row length distribution ----------------------------------------------
    lens = np.array([r.length for r in rows])
    steps_per_row = np.array([r.n_steps for r in rows])
    print(f"\n=== rows: {len(rows):,} sequences, max_len {cfg.data.max_len} ===")
    for p in (50, 75, 90, 95, 99, 100):
        print(f"  length p{p:<3} {np.percentile(lens, p):>7.0f}")
    print(f"  mean length {lens.mean():.0f}   mean steps/row {steps_per_row.mean():.2f}")
    print("\n  length histogram (fraction of rows, and of the sequence mass above each bin):")
    edges = list(range(0, cfg.data.max_len + 1, 128))
    hist, _ = np.histogram(lens, bins=edges + [cfg.data.max_len + 1])
    tail = 1.0
    for lo, n in zip(edges, hist):
        bar = "#" * int(60 * n / hist.max())
        print(f"   {lo:>5}-{lo + 127:<5} {n:>7,} {n / len(rows):>6.1%}  >= {tail:>5.1%}  {bar}")
        tail -= n / len(rows)

    # ---- 2. the batching as configured ------------------------------------------------
    allocations, order = epoch_inputs(rows, slots, cfg, budget)
    base = describe(pack(order, allocations, rows, budget, None), rows, cfg, f"{budget} seqs (current)")

    print(f"\n=== how heavy is the tail? (current: {budget} seqs, grad_accum {cfg.train.grad_accum}) ===")
    toks = base["toks"]
    for p in (50, 75, 90, 95, 99, 100):
        print(f"  padded tokens p{p:<3} {np.percentile(toks, p):>9,.0f}"
              f"   ~{estimate_peak_gib(int(np.percentile(toks, p))):>5.1f} GiB")
    for cap in (24576, 32768, 40960, 49152):
        n = int((toks > cap).sum())
        print(f"  batches over {cap:>6,} tok: {n:>6,}  {n / len(toks):>6.1%}"
              f"   ({(toks[toks > cap].sum() - cap * n) / max(toks.sum(), 1):>5.1%} of all padding above the cap)")

    # ---- 3. the two families of fix ---------------------------------------------------
    print("\n=== token cap (keeps the 56-seq budget; only oversized buckets shrink) ===")
    print(HEADER)
    print_row(base)
    for cap in [int(c) for c in args.caps.split(",")]:
        batches = pack(order, allocations, rows, budget, cap)
        print_row(describe(batches, rows, cfg, f"cap {cap:,} tok"))

    print("\n=== sequence cut (SETUP.md:344; shrinks every batch, pool included) ===")
    print(HEADER)
    print_row(base)
    for b in [int(s) for s in args.seq_budgets.split(",")]:
        alloc_b, order_b = epoch_inputs(rows, slots, cfg, b)
        batches = pack(order_b, alloc_b, rows, b, None)
        # grad_accum must rise to hold the effective batch; report at the accum that keeps
        # sequences-per-optimizer-step closest to the current budget x grad_accum.
        target = budget * cfg.train.grad_accum
        accum = max(1, round(target / b))
        scaled = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, grad_accum=accum))
        d = describe(batches, rows, scaled, f"{b} seqs / accum {accum}")
        print_row(d)
        print(f"  {'':<26} effective {b * accum} seqs/step (current {target})")

    # ---- 4. the other lever: the fp32 adapter cast ------------------------------------
    print("\n=== the fp32 LoRA cast (independent of batch shape) ===")
    print("  The allocation that actually failed was 1.91 GiB, which is exactly")
    print(f"  57,344 tok x 8,960 (intermediate_size) x 4 B = {gib(57344 * 8960 * 4):.2f} GiB --")
    print("  down_proj's input cast to fp32. load_backbone takes PEFT's default")
    print("  autocast_adapter_dtype=True, so the adapters are fp32 under a bf16 base and")
    print("  every adapted projection retains an fp32 copy of its input for backward.")
    print(f"\n  {'batch tokens':>13} {'fp32 cast':>11} {'bf16 would be':>14} {'saved':>8}")
    for t in (int(base["tok_max"]), 40960, 32768, 28672):
        cast = fp32_cast_gib(t)
        print(f"  {t:>13,} {cast * 2:>10.2f}G {cast:>13.2f}G {cast:>7.2f}G")
    print("\n  Saving is per decoder layer with one layer live at a time under checkpointing.")

    print(f"\n=== reading this (card: {args.capacity} GiB) ===")
    print("  The ~GiB column is an extrapolation from the one observed OOM and is a LOWER")
    print("  bound; treat anything within ~2 GiB of capacity as not proven to fit. The")
    print("  batch, pool R and Q columns are exact counts.")
    print("  'R p10' is the 10th-percentile negative pool: how thin L_NCE gets on its")
    print("  worst batches, which is what a sequence cut spends and a token cap does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
