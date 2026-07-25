"""ProcessBench scoring (§9.1). CRM has no ProcessBench code at all, so this is from scratch.

    1. build the sequence per §6.1, ADDING "Step N: " prefixes (locked #8)
    2. one forward -> hiddens at s_0 .. s_T
    3. g_q = goal_head(h_{s_0})
    4. d_i = d(psi_i, g_q)                       for i = 0..T
    5. Delta_i = d_i - d_{i-1}                   for i = 1..T
    6. i* = first i with Delta_i > tau;  predicted_label = i* - 1, else -1

**Why Delta and not d directly:** ProcessBench is change-point detection, not calibrated
scoring. Any error in `g_q` that acts like a roughly constant offset drops out of the
difference. First-order only -- d is nonlinear in g -- but it buys real robustness against
exactly the failure that killed the old project.

Over-length samples are NOT truncated (truncation would drop trailing separators and shorten
T). They are counted and predicted -1; §PLAN's rule is that >1% of a subset over-length is a
hard failure, so the count is asserted rather than silently absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ..config import Config
from ..data.collate import SequenceRow, collate
from ..data.tokenize import SequenceTooLong, build_sequence, sep_token_id
from ..utils.indexing import predicted_label_from_deltas
from .metrics import processbench_metrics, split_metrics

BENCHMARK = "Qwen/ProcessBench"


@dataclass
class Sample:
    id: str
    problem: str
    steps: tuple[str, ...]
    label: int                    # 0-based index of the first erroneous step, or -1
    final_answer_correct: bool
    generator: str = ""


def load_processbench(subset: str) -> list[Sample]:
    from datasets import load_dataset

    ds = load_dataset(BENCHMARK, split=subset)
    return [
        Sample(
            id=str(row["id"]),
            problem=row["problem"],
            steps=tuple(row["steps"]),
            label=int(row["label"]),
            final_answer_correct=bool(row["final_answer_correct"]),
            generator=str(row.get("generator", "")),
        )
        for row in ds
    ]


@torch.no_grad()
def score_samples(
    model,
    tokenizer,
    samples: Sequence[Sample],
    cfg: Config,
    device,
    goal_fn: Optional[Callable[[Tensor, Sequence[Sample]], Tensor]] = None,
) -> tuple[list[list[float]], dict[str, float]]:
    """Returns (per-sample Delta lists, counters). An over-length sample gets an empty Delta
    list, which predicts -1.

    `goal_fn(h_s0, batch_samples) -> (B, D)` overrides the goal head; the skyline passes the
    reference-solution terminal that way (§9.5).
    """
    sep_id = sep_token_id(tokenizer, cfg.data.sep_token)
    pad_id = tokenizer.pad_token_id
    counters = {"over_length": 0, "scored": 0}

    deltas: list[list[float]] = [[] for _ in samples]
    pending: list[tuple[int, SequenceRow]] = []

    def flush() -> None:
        if not pending:
            return
        idxs = [i for i, _ in pending]
        batch = collate([r for _, r in pending], pad_id=pad_id).to(device)
        reps = model(batch)
        if goal_fn is None:
            if model.goal_head is None:
                raise RuntimeError(
                    "no goal head: phase 2 must run before ProcessBench (locked #13). A "
                    "reference-solution goal is a skyline, not a substitute (§5.1)."
                )
            h_s0 = reps.h_states.index_select(0, batch.traj_state_offset)
            goals = model.goal_head(h_s0)
        else:
            h_s0 = reps.h_states.index_select(0, batch.traj_state_offset)
            goals = goal_fn(h_s0, [samples[i] for i in idxs])

        for b, sample_idx in enumerate(idxs):
            T = int(batch.traj_T[b])
            offset = int(batch.traj_state_offset[b])
            states = reps.psi[offset : offset + T + 1]                      # (T+1, D)
            d = model.distance(states, goals[b].expand_as(states))          # (T+1,)
            deltas[sample_idx] = (d[1:] - d[:-1]).float().cpu().tolist()    # Delta_1..Delta_T
            counters["scored"] += 1
        pending.clear()

    for i, sample in enumerate(samples):
        try:
            seq = build_sequence(
                tokenizer,
                sample.problem,
                list(sample.steps),
                sep_id,
                prompt_format=cfg.data.prompt_format,
                max_len=cfg.eval.max_len,
                add_prefix=True,                    # locked #8
            )
        except (SequenceTooLong, ValueError):
            counters["over_length"] += 1
            continue
        pending.append(
            (
                i,
                SequenceRow(
                    qid=sample.id,
                    input_ids=np.asarray(seq.input_ids, dtype=np.int64),
                    state_pos=np.asarray(seq.state_pos, dtype=np.int64),
                    span_start=np.asarray([s for s, _ in seq.step_spans], dtype=np.int64),
                    span_end=np.asarray([e for _, e in seq.step_spans], dtype=np.int64),
                    correct=True,
                    z=-1,
                ),
            )
        )
        if len(pending) >= cfg.eval.batch_sequences:
            flush()
    flush()
    return deltas, counters


def predict(deltas: Sequence[Sequence[float]], tau: float) -> list[int]:
    return [predicted_label_from_deltas(d, tau) for d in deltas]


def evaluate_subset(
    deltas: Sequence[Sequence[float]],
    samples: Sequence[Sample],
    tau: float,
    leaked: Optional[Sequence[bool]] = None,
) -> dict:
    predictions = predict(deltas, tau)
    labels = [s.label for s in samples]
    result = {"tau": tau, **processbench_metrics(predictions, labels)}
    if leaked is not None:
        result["leak_split"] = split_metrics(predictions, labels, leaked)
    return result


def main(argv: list[str] | None = None) -> int:
    """The reported eval (§9.3): one scoring path, the quasimetric Delta-d with the goal head.

    There is no second scoring head to report against (locked #3a). The skyline runs on the
    1,400 joinable samples and is LABELLED as a skyline (§9.5).
    """
    import argparse
    import json
    from pathlib import Path

    from ..data.math_shepherd import read_sequences_parquet
    from ..model.backbone import load_backbone_with_adapter, load_tokenizer, read_hidden_size
    from ..model.wrapper import FeynmanPRM
    from ..utils.checkpoint import load_config_from_checkpoint, load_heads
    from .calibrate import calibrate_tau, score_validation
    from .skyline import joinable_count, load_reference_solutions, reference_goal_fn

    parser = argparse.ArgumentParser(description="ProcessBench eval")
    parser.add_argument("--checkpoint", required=True, help="a PHASE-2 checkpoint (goal head)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--tau", type=float, default=None, help="skip calibration")
    args = parser.parse_args(argv)

    ckpt = Path(args.checkpoint)
    cfg = load_config_from_checkpoint(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(cfg)
    backbone = load_backbone_with_adapter(cfg, ckpt / "adapter")
    model = FeynmanPRM(cfg, read_hidden_size(cfg.model.name), backbone=backbone, with_goal_head=True)
    load_heads(model, ckpt)
    model.to(device).eval()

    results: dict = {"checkpoint": str(ckpt)}

    # ---- tau, on held-out VAL QUESTIONS. Never on ProcessBench (§9.2). ----------------
    if args.tau is None:
        val_rows = read_sequences_parquet(Path(cfg.data.dir) / "sequences.parquet", split="val")
        deltas, labels = score_validation(model, val_rows, cfg, device, tokenizer.pad_token_id)
        calibration = calibrate_tau(deltas, labels, cfg)
        tau = calibration.tau
        results["calibration"] = calibration.summary()
        results["calibration"]["curve"] = calibration.curve
        print(
            f"tau = {tau:.4f} (natural midpoint {calibration.expected_tau:.4f}); "
            f"val F1 {calibration.f1:.4f}, sensitivity +/-0.1 -> {calibration.sensitivity:.4f}",
            flush=True,
        )
    else:
        tau = args.tau

    leak_path = Path(cfg.data.dir) / "processbench_math_leak.json"
    leak_map = json.loads(leak_path.read_text()) if leak_path.exists() else {}

    for subset in cfg.eval.subsets:
        samples = load_processbench(subset)
        deltas, counters = score_samples(model, tokenizer, samples, cfg, device)
        assert_truncation_budget(counters, len(samples), subset)
        leaked = [bool(leak_map.get(s.id, False)) for s in samples] if subset == "math" else None
        results[subset] = evaluate_subset(deltas, samples, tau, leaked=leaked)
        results[subset]["counters"] = counters
        print(f"{subset}: {json.dumps(results[subset], default=float)}", flush=True)

        if cfg.eval.skyline and subset in ("gsm8k", "omnimath", "math"):
            references = load_reference_solutions(subset)
            if references:
                goal_fn = reference_goal_fn(model, tokenizer, cfg, device, references)
                sky_deltas, _ = score_samples(
                    model, tokenizer, samples, cfg, device, goal_fn=goal_fn
                )
                results[f"{subset}_SKYLINE_not_a_result"] = {
                    **evaluate_subset(sky_deltas, samples, tau),
                    "joinable": joinable_count(samples, references),
                    "note": "SKYLINE (§9.5). The gold answer alone solves half the metric "
                            "exactly (§5.1), so this is never a reported result.",
                }

    out_path = Path(args.out) if args.out else ckpt / "processbench.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {out_path}", flush=True)
    return 0


def assert_truncation_budget(counters: dict[str, float], n_samples: int, subset: str) -> None:
    over = counters.get("over_length", 0)
    if n_samples and over / n_samples > 0.01:
        raise AssertionError(
            f"{subset}: {over}/{n_samples} samples exceed eval.max_len -- over the 1% budget. "
            "Raise eval.max_len; do NOT truncate (it drops trailing separators and shortens T)."
        )


if __name__ == "__main__":
    raise SystemExit(main())
