"""ProcessBench + Math-Shepherd val F1 for a PQM checkpoint, under Feynman-PRM's protocol.

**The one idea that makes the whole Feynman eval stack reusable: PQM's per-step score enters
as `-r_i`.** Feynman's `Delta_i` is "higher = worse"; PQM's reward is "higher = better".
Negating makes the two the same object, so `predicted_label_from_deltas`, `processbench_metrics`,
`split_metrics` and the tau sweep all apply UNCHANGED, and the `deltas.npz` schema is
identical -- which means `scripts/analyze_deltas.py` and `scripts/error_rank.py` run on a PQM
checkpoint for free.

Nothing under `feynman_prm/eval/` is touched. This driver imports its PURE parts --
`load_processbench`, `Sample`, `evaluate_subset`, `pack_deltas`, `assert_truncation_budget`,
`processbench_metrics` -- and writes its own scoring loop, so no file that produced a reported
Feynman number changes at all.

    1. tau on the held-out 2,000 VAL questions (§9.2 -- NEVER on ProcessBench)
    2. the four subsets, `add_prefix=True` (locked #8), `assert_truncation_budget` at 1%,
       the math leak split (locked #5)
    3. writes processbench.json (+ a `pqm` block), deltas.npz and val_f1.json

> `scripts/report_processbench.py`'s tau verdict line is calibrated against FEYNMAN's ruler
> (`natural_tau = (m - (-log gamma))/2 = 0.347`) and is MEANINGLESS for this checkpoint. The
> verdict printed here is against PQM's own anchor, `zeta/2` on the negated scale. Do not read
> the shared script's line for a PQM run.

    python -m pqm_baseline.eval_processbench --checkpoint runs/pqm_zeta4/final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feynman_prm.config import Config
from feynman_prm.data.collate import SequenceRow, collate
from feynman_prm.data.math_shepherd import read_sequences_parquet
from feynman_prm.data.tokenize import SequenceTooLong, build_sequence, sep_token_id
from feynman_prm.diagnostics.logging import Progress
from feynman_prm.eval.metrics import processbench_metrics
from feynman_prm.eval.processbench import (
    Sample,
    assert_truncation_budget,
    evaluate_subset,
    load_processbench,
    pack_deltas,
)
from feynman_prm.model.backbone import load_backbone_with_adapter, load_tokenizer, read_hidden_size
from feynman_prm.utils.checkpoint import load_config_from_checkpoint
from feynman_prm.utils.indexing import predicted_label_from_deltas

from .config import load_pqm_config_from_checkpoint
from .model import PQMValueModel, load_value_head


# ---------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------


@torch.no_grad()
def score_rows(model, rows: Sequence[SequenceRow], cfg: Config, device, label: str) -> list[list[float]]:
    """`-r_1 .. -r_T` for every pre-tokenised row, in row order.

    `r_i` is read at the state step `i` LANDS IN (`s_i`), which is the identical index path
    training uses (`loss.rewards_at_steps`). `s_0` carries no step and is not scored.
    """
    deltas: list[list[float]] = [[] for _ in rows]
    pending: list[tuple[int, SequenceRow]] = []
    progress = Progress(f"score/{label}", len(rows))

    def flush() -> None:
        if not pending:
            return
        batch = collate([r for _, r in pending], pad_id=model.pad_id).to(device)
        rewards = model(batch)
        for b, (idx, _) in enumerate(pending):
            T = int(batch.traj_T[b])
            offset = int(batch.traj_state_offset[b])
            r = rewards[offset + 1 : offset + T + 1]                  # r_1 .. r_T
            deltas[idx] = (-r).float().cpu().tolist()                 # higher = worse
        progress.advance(len(pending))
        pending.clear()

    for i, row in enumerate(rows):
        pending.append((i, row))
        if len(pending) >= cfg.eval.batch_sequences:
            flush()
    flush()
    return deltas


@torch.no_grad()
def score_samples(model, tokenizer, samples: Sequence[Sample], cfg: Config, device, label: str):
    """ProcessBench samples -> (deltas, counters). Tokenised through `build_sequence` with
    `add_prefix=True` (locked #8) at `eval.max_len`; an over-length sample is NOT truncated
    (that would drop trailing separators and shorten T) -- it is counted and predicts -1."""
    sep_id = sep_token_id(tokenizer, cfg.data.sep_token)
    counters = {"over_length": 0, "scored": 0}
    keep: list[tuple[int, SequenceRow]] = []

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
        keep.append(
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

    scored = score_rows(model, [r for _, r in keep], cfg, device, label)
    counters["scored"] = len(keep)
    deltas: list[list[float]] = [[] for _ in samples]
    for (i, _), d in zip(keep, scored):
        deltas[i] = d
    return deltas, counters


# ---------------------------------------------------------------------------------------
# tau, on val. NEVER on ProcessBench (§9.2)
# ---------------------------------------------------------------------------------------


def sweep_tau(
    deltas: Sequence[Sequence[float]],
    labels: Sequence[int],
    cfg: Config,
    natural_tau: float,
    grid: Sequence[float] | None = None,
) -> dict:
    """Pick tau maximising val F1 under the ProcessBench rule.

    Not `eval.calibrate.calibrate_tau`: that one reports `natural_tau(cfg)`, Feynman's
    `(m - (-log gamma))/2`, which is a statement about a ruler this checkpoint does not have.
    The sweep itself is the same, and the natural value is forced into the grid so the check
    below is exact rather than nearest-gridpoint.
    """
    if grid is None:
        flat = np.concatenate([np.asarray(d) for d in deltas if len(d)]) if deltas else np.zeros(1)
        lo, hi = float(np.quantile(flat, 0.01)), float(np.quantile(flat, 0.99))
        grid = np.unique(np.concatenate([np.linspace(lo, hi, 201), [0.0, natural_tau]]))

    curve = []
    for tau in grid:
        preds = [
            predicted_label_from_deltas(d, float(tau), cfg.eval.localisation_rule) for d in deltas
        ]
        curve.append({"tau": float(tau), **processbench_metrics(preds, labels)})

    best = max(curve, key=lambda row: row["f1"])
    near = [row["f1"] for row in curve if abs(row["tau"] - best["tau"]) <= 0.1]
    return {
        "calibration/tau": best["tau"],
        "calibration/f1": best["f1"],
        "calibration/acc_error": best["acc_error"],
        "calibration/acc_correct": best["acc_correct"],
        "calibration/expected_tau": natural_tau,
        "calibration/sensitivity": (max(near) - min(near)) if near else 0.0,
        "curve": curve,
    }


def tau_verdict(tau: float, natural: float) -> str:
    """PQM's own check, against `zeta/2`. A CHECK, not a constraint (§9.2's spirit).

    MULTIPLICATIVE, never additive -- §14's B12 is exactly the additive version of this line
    printing "the margin held" over a 3.4x overshoot.
    """
    if natural <= 0:
        return "no natural tau (zeta must be > 0)"
    ratio = tau / natural
    if 0.5 <= ratio <= 2.0:
        return (
            f"tau/natural = {ratio:.2f} -- the loss's absolute anchors took. A single global "
            f"threshold means the same thing on every question."
        )
    return (
        f"** tau/natural = {ratio:.2f}. PQM's ranking loss anchors positives above 0 and "
        f"negatives below -zeta, so on the negated scale their midpoint is +zeta/2 = "
        f"{natural:.3f}. A fitted tau far from it means those absolute anchors did NOT take, "
        f"and a global threshold is not doing what it looks like it is doing -- read "
        f"pqm/frac_pos_above_0 and pqm/frac_neg_below_neg_zeta for this run before reading "
        f"the F1 below."
    )


# ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ProcessBench + val F1 for a PQM checkpoint")
    parser.add_argument("--checkpoint", required=True, help="e.g. runs/pqm_zeta4/final")
    parser.add_argument("--out", default=None)
    parser.add_argument("--tau", type=float, default=None, help="skip calibration")
    parser.add_argument("--val-limit", type=int, default=None,
                        help="score only the first N val trajectories (debugging)")
    args = parser.parse_args(argv)

    ckpt = Path(args.checkpoint)
    cfg = load_config_from_checkpoint(ckpt)
    pqm = load_pqm_config_from_checkpoint(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(cfg)
    backbone = load_backbone_with_adapter(cfg, ckpt / "adapter")
    model = PQMValueModel(cfg, pqm, read_hidden_size(cfg.model.name), backbone=backbone)
    load_value_head(model, ckpt)
    model.pad_id = tokenizer.pad_token_id
    model.to(device).eval()

    results: dict = {
        "checkpoint": str(ckpt),
        "pqm": {
            "zeta": pqm.zeta,
            "loss_type": pqm.loss_type,
            "head_init": pqm.head_init,
            "label_source": pqm.label_source,
            "natural_tau_delta": pqm.natural_tau_delta,
            "note": (
                "PQM (Li & Li, ICLR 2025) re-implemented under Feynman-PRM's matched "
                "conditions -- NOT PQM's published numbers. Its own paper reports Best-of-N "
                "on a 7B full finetune and never reports ProcessBench; the localisation rule "
                "here is OURS, applied identically to both rows of the table."
            ),
        },
    }

    # ---- tau, on the held-out VAL questions. Never on ProcessBench (§9.2) ---------------
    if args.tau is None:
        val_rows = read_sequences_parquet(Path(cfg.data.dir) / "sequences.parquet", split="val")
        if args.val_limit is not None:
            val_rows = val_rows[: args.val_limit]
        # `calibrate.score_validation`'s convention exactly: gold label is `z` for an
        # incorrect trajectory and -1 for a correct one, which is ProcessBench's own.
        val_labels = [int(r.z) if not r.correct else -1 for r in val_rows]
        val_deltas = score_rows(model, val_rows, cfg, device, "val")
        calibration = sweep_tau(val_deltas, val_labels, cfg, pqm.natural_tau_delta)
        tau = calibration["calibration/tau"]
        results["calibration"] = calibration
        print(
            f"tau = {tau:.4f} (natural zeta/2 = {pqm.natural_tau_delta:.4f}); "
            f"val F1 {calibration['calibration/f1']:.4f}, sensitivity +/-0.1 -> "
            f"{calibration['calibration/sensitivity']:.4f}",
            flush=True,
        )
        print("  " + tau_verdict(tau, pqm.natural_tau_delta), flush=True)

        val_path = ckpt / "val_f1.json"
        val_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(ckpt),
                    "trajectories": len(val_rows),
                    "questions": len({r.qid for r in val_rows}),
                    **{k: v for k, v in calibration.items() if k != "curve"},
                    "note": (
                        "The Math-Shepherd val F1. The COMPARABLE Feynman number is the "
                        "goal-head `calibration/f1` in that run's "
                        "phase2/final/processbench.json (0.5900 for abl_cf_only, 0.5872 for "
                        "phase1_nce_temp_relu2) -- NOT scripts/val_f1.py's 0.5615, which "
                        "substitutes a real terminal for the goal and is a ceiling by its own "
                        "docstring (§9.5)."
                    ),
                },
                indent=2,
            )
        )
        print(f"wrote {val_path}", flush=True)
    else:
        tau = args.tau

    # ---- the four subsets ---------------------------------------------------------------
    leak_path = Path(cfg.data.dir) / "processbench_math_leak.json"
    leak_map = json.loads(leak_path.read_text()) if leak_path.exists() else {}

    raw: dict[str, np.ndarray] = {"tau": np.asarray(tau, dtype=np.float64)}
    f1s = []
    for subset in cfg.eval.subsets:
        samples = load_processbench(subset)
        deltas, counters = score_samples(model, tokenizer, samples, cfg, device, subset)
        assert_truncation_budget(counters, len(samples), subset)
        leaked = [bool(leak_map.get(s.id, False)) for s in samples] if subset == "math" else None
        results[subset] = evaluate_subset(
            deltas, samples, tau, leaked=leaked, rule=cfg.eval.localisation_rule
        )
        results[subset]["counters"] = counters
        f1s.append(results[subset]["f1"])
        print(f"{subset}: {json.dumps(results[subset], default=float)}", flush=True)

        flat, lengths = pack_deltas(deltas)
        raw[f"{subset}/flat"] = flat
        raw[f"{subset}/lengths"] = lengths
        raw[f"{subset}/labels"] = np.asarray([s.label for s in samples], dtype=np.int64)
        raw[f"{subset}/final_answer_correct"] = np.asarray(
            [s.final_answer_correct for s in samples], dtype=bool
        )
        if leaked is not None:
            raw[f"{subset}/leaked"] = np.asarray(leaked, dtype=bool)

    if f1s:
        results["mean_f1"] = sum(f1s) / len(f1s)
        print(f"\nmean F1 over {len(f1s)} subsets: {results['mean_f1']:.4f}", flush=True)

    out_path = Path(args.out) if args.out else ckpt / "processbench.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {out_path}", flush=True)

    npz_path = out_path.with_name("deltas.npz")
    np.savez_compressed(npz_path, **raw)
    print(f"wrote {npz_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
