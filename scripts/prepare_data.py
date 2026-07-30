#!/usr/bin/env python
"""Preprocess Math-Shepherd once: select, split, tokenise, mine branch points (§18.1).

Prints §4.1/§4.2's measured counts as a SELF-CHECK -- 422,422 rows -> 45,989 questions ->
40,247 trainable -> 36.6% correct. If these do not reproduce, the dataset changed under us
and every number in CLAUDE.md §4 is suspect.

Also prints the **real tokenised length distribution**, which replaces §4.6's chars/3.4
estimate -- the only estimated number in CLAUDE.md (§17).

Rows longer than `data.max_len` are DROPPED and counted, never truncated (§4.6): truncation
would silently drop trailing separators and shorten T, which is the analogue of the failure
CRM avoids at `train_RM.py:47-49`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run without `pip install -e .`

import argparse
import json

import numpy as np

from feynman_prm.config import load_config
from feynman_prm.data.branch_points import mine, write_jsonl
from feynman_prm.data.math_shepherd import (
    EXPECTED_TRAIN,
    build_questions,
    dataset_stats,
    iter_hf_rows,
    selection_sha,
    split_questions,
    write_sequences_parquet,
)
from feynman_prm.data.tokenize import EmptyStep, SequenceTooLong, build_sequence, sep_token_id
from feynman_prm.model.backbone import load_tokenizer


def tokenise_split(questions, tokenizer, sep_id, cfg, split_name):
    rows, dropped = [], {"too_long": 0, "empty_step": 0}
    for question in questions:
        for traj in question.trajectories:
            try:
                seq = build_sequence(
                    tokenizer,
                    question.prompt,
                    list(traj.steps),
                    sep_id,
                    prompt_format=cfg.data.prompt_format,
                    max_len=cfg.data.max_len,
                )
            except SequenceTooLong:
                dropped["too_long"] += 1
                continue
            except EmptyStep:
                dropped["empty_step"] += 1
                continue
            rows.append(
                {
                    "qid": question.qid,
                    "split": split_name,
                    "correct": traj.correct,
                    "z": traj.z,
                    "recovery": traj.recovery,
                    "n_steps": seq.n_steps,
                    "length": seq.length,
                    "input_ids": np.asarray(seq.input_ids, dtype=np.int32),
                    "state_pos": np.asarray(seq.state_pos, dtype=np.int32),
                    "span_start": np.asarray([s for s, _ in seq.step_spans], dtype=np.int32),
                    "span_end": np.asarray([e for _, e in seq.step_spans], dtype=np.int32),
                }
            )
    return rows, dropped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess Math-Shepherd")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None, help="dev: only read N rows")
    parser.add_argument("--skip-leak-check", action="store_true",
                        help="skip the ProcessBench-math contamination flags (needs the benchmark)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)

    out = Path(cfg.data.dir)
    out.mkdir(parents=True, exist_ok=True)

    print("loading trl-lib/math_shepherd ...", flush=True)
    rows_iter = iter_hf_rows(split="train")
    if args.limit:
        import itertools

        rows_iter = itertools.islice(rows_iter, args.limit)
    questions, counters = build_questions(rows_iter)
    stats = dataset_stats(questions)

    print(json.dumps({"counters": counters, "stats": {k: round(v, 4) for k, v in stats.items()}},
                     indent=2), flush=True)
    if args.limit is None:
        for key, expected in (
            ("questions", EXPECTED_TRAIN["questions"]),
            ("trainable_questions", EXPECTED_TRAIN["trainable_questions"]),
        ):
            if int(stats[key]) != expected:
                print(f"  !! {key}: {int(stats[key])} != §4's measured {expected}", flush=True)

    train_q, val_q = split_questions(
        questions, cfg.data.n_val_questions, cfg.data.n_questions, cfg.run.seed
    )
    overlap = {q.qid for q in train_q} & {q.qid for q in val_q}
    assert not overlap, f"{len(overlap)} questions in both train and val"

    tokenizer = load_tokenizer(cfg)
    sep_id = sep_token_id(tokenizer, cfg.data.sep_token)   # asserts it is ONE token (§6.1)

    all_rows, dropped_total = [], {"too_long": 0, "empty_step": 0}
    for name, qs in (("train", train_q), ("val", val_q)):
        rows, dropped = tokenise_split(qs, tokenizer, sep_id, cfg, name)
        all_rows.extend(rows)
        for k, v in dropped.items():
            dropped_total[k] += v
        print(f"{name}: {len(qs)} questions -> {len(rows)} sequences, dropped {dropped}", flush=True)

    write_sequences_parquet(all_rows, out / "sequences.parquet")

    lengths = np.array([r["length"] for r in all_rows])
    length_stats = {
        "median": float(np.median(lengths)),
        "p90": float(np.quantile(lengths, 0.90)),
        "p95": float(np.quantile(lengths, 0.95)),
        "p99": float(np.quantile(lengths, 0.99)),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "over_max_len_dropped": dropped_total["too_long"],
        "over_max_len_fraction": dropped_total["too_long"] / max(len(all_rows), 1),
    }
    print("REAL tokenised lengths (replaces §4.6's chars/3.4 estimate):",
          json.dumps(length_stats, indent=2), flush=True)

    print("mining branch points (§8.3) ...", flush=True)
    points, branch_counters = mine(train_q)
    write_jsonl(points, out / "branch_points.jsonl")
    print(json.dumps(branch_counters, indent=2), flush=True)

    if not args.skip_leak_check:
        # §4.3(b) / locked #5: 587 of ProcessBench-math's 1000 problems are in math-shepherd
        # train. We KEEP them and report math F1 split, so eval needs the per-sample flag and
        # only this script has the training prompts. Problem-level leakage, not
        # solution-level: any PRM trained on Math-Shepherd has it, CRM included.
        try:
            from feynman_prm.eval.processbench import load_processbench
            from feynman_prm.eval.skyline import normalise_problem

            train_prompts = {normalise_problem(q.prompt) for q in questions}
            leak = {
                s.id: normalise_problem(s.problem) in train_prompts
                for s in load_processbench("math")
            }
            (out / "processbench_math_leak.json").write_text(json.dumps(leak))
            print(f"ProcessBench-math leak: {sum(leak.values())}/{len(leak)} "
                  "(§4.3b measured 587/1000)", flush=True)
        except Exception as exc:  # offline box, or the benchmark is not cached
            print(f"  !! leak check skipped: {exc}", flush=True)

    selection = {
        "selection_sha_train": selection_sha([q.qid for q in train_q]),
        "selection_sha_val": selection_sha([q.qid for q in val_q]),
        "n_train_questions": len(train_q),
        "n_val_questions": len(val_q),
        "n_sequences": len(all_rows),
        # ~9.18. This is what is ON DISK -- every trajectory of every selected question, per
        # §2 #1. It is NOT what a training epoch consumes: the sampler takes
        # min(4, k_c) + min(3, k_i) = 4.33 per question (§8.1's caps), so one epoch is ~99.6k
        # sequences and ~889 optimizer steps, not 211k and 1,885. The key is named for the
        # disk on purpose -- reading 9.18 as the epoch rate is exactly the mistake that
        # produced the 106-step run (§8.2, §11.1). train.py derives its step count from the
        # realised `epoch_batches`, never from this number.
        "trajectories_per_question_on_disk": len(all_rows) / max(len(train_q) + len(val_q), 1),
        "sampler_sequences_per_question_expected": 4.33,
        "dataset_stats": stats,
        "length_stats": length_stats,
        "branch_points": branch_counters,
        "dropped": dropped_total,
        "config": {"seed": cfg.run.seed, "max_len": cfg.data.max_len,
                   "prompt_format": cfg.data.prompt_format, "model": cfg.model.name},
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=2, default=float))
    # R10: log the ACTUAL selected id list, not the seed that implies it. The old project
    # documented a "seed 42 shuffle" for two years that turned out not to exist.
    (out / "train_questions.txt").write_text("\n".join(sorted(q.qid for q in train_q)))
    (out / "val_questions.txt").write_text("\n".join(sorted(q.qid for q in val_q)))
    print(json.dumps({k: selection[k] for k in
                      ("selection_sha_train", "selection_sha_val", "n_sequences",
                       "trajectories_per_question_on_disk",
                       "sampler_sequences_per_question_expected")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
