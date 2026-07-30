#!/usr/bin/env python
"""Run the §10 diagnostic panel over a phase-1 checkpoint and dump it as JSON.

`metrics.jsonl` already holds every §10 probe for the run that produced the checkpoint, so
**read that first** -- it costs no GPU. This script exists for the two things the JSONL cannot
give you:

  1. probes added AFTER the run (the `nce/*_question` split, §16.4's floor), which no existing
     log line contains;
  2. a matched `--untrained` null on the SAME batches. §10.1.1 is the standing lesson here --
     `< 0.3` and `ln 5` were both thresholds set by intuition against an assumed baseline, and
     both were wrong. A level alone is not a measurement. The gate passes an untrained model at
     auc 0.904; assume nothing else is different.

No gradients, no optimizer, ~2 min for 40 batches. It does not touch the checkpoint.

    python scripts/diagnose_checkpoint.py --checkpoint runs/phase1/step750 --batches 40
    python scripts/diagnose_checkpoint.py --checkpoint runs/phase1/step750 --untrained \
        --out runs/phase1/diagnose_untrained.json
    python scripts/diagnose_checkpoint.py --checkpoint runs/phase1/step750 --split val
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run without `pip install -e .`

import argparse
import json
import math
from statistics import mean, pstdev

import torch

from feynman_prm.data.math_shepherd import read_sequences_parquet
from feynman_prm.data.sampler import batch_stats, build_question_slots, epoch_batches
from feynman_prm.diagnostics.probes import asymmetry_score, batch_probes
from feynman_prm.model.backbone import (
    load_backbone_with_adapter,
    load_tokenizer,
    read_hidden_size,
)
from feynman_prm.model.wrapper import FeynmanPRM
from feynman_prm.train import run_micro_batch
from feynman_prm.utils.checkpoint import load_config_from_checkpoint, load_heads
from feynman_prm.utils.seeding import epoch_rng, goal_rng, seed_everything


def _backbone(cfg, ckpt: Path, untrained: bool):
    if not untrained:
        return load_backbone_with_adapter(cfg, ckpt / "adapter")

    from transformers import AutoModel

    base = AutoModel.from_pretrained(
        cfg.model.name,
        dtype=torch.bfloat16 if cfg.train.bf16 else torch.float32,
        attn_implementation=cfg.model.attn_implementation,
    )
    for p in base.parameters():
        p.requires_grad_(False)
    return base.eval()


@torch.no_grad()
def confusion_structure(
    Dist: torch.Tensor, pos_row: torch.Tensor, SQ: torch.Tensor, row_correct: torch.Tensor
) -> dict:
    """WHICH rows beat the positive, and whether they are the SAME rows in every column.

    `nce/loss_cross_question` says a chunk of cross-question rows sit closer to a goal than
    that goal's own source. There are two very different reasons for that and they need
    opposite fixes:

      * HUB COLLAPSE -- a few source rows land somewhere central and are close to EVERY goal.
        In an MRN the asymmetric half is `max(relu(x - y))`, which is ~0 whenever the source
        is elementwise below the goal, so a small-coordinate row is cheap to reach from
        nowhere and near everything. Signature: `top5pct_rows_share_of_beats` -> 1.0.
      * GENUINE CONFUSION -- different rows beat different columns, i.e. the metric simply has
        not separated those questions yet. Signature: the share stays near the ~0.05-0.10 a
        uniform spread gives.

    Only the second one is fixed by more training or more data.
    """
    R, C = Dist.shape
    cols = torch.arange(C, device=Dist.device)
    beats = Dist < Dist[pos_row, cols][None, :]
    beats[pos_row, cols] = False

    per_row = beats.sum(dim=1).float()
    total = per_row.sum().clamp(min=1.0)
    k = max(1, int(round(0.05 * R)))
    return {
        "confusion/rows_beating_positive": float(beats.sum(dim=0).float().mean()),
        "confusion/beats_same_question": float((beats & SQ).sum()) / C,
        "confusion/beats_cross_question": float((beats & ~SQ).sum()) / C,
        "confusion/top5pct_rows_share_of_beats": float(per_row.topk(k).values.sum() / total),
        "confusion/rows_that_never_beat_fraction": float((per_row == 0).float().mean()),
        # Are the confusable rows error states? Post-error states are 2:1 over-represented
        # (§4.2) and L_step pushes them AWAY from their own goal -- which says nothing about
        # where they land relative to everyone else's.
        "confusion/beats_from_incorrect_rows": float(
            (per_row * (~row_correct).float()).sum() / total
        ),
        "confusion/incorrect_row_fraction": float((~row_correct).float().mean()),
    }


def _summarise(samples: dict[str, list[float]]) -> dict[str, dict]:
    out = {}
    for key, values in sorted(samples.items()):
        finite = [v for v in values if math.isfinite(v)]
        if not finite:
            out[key] = {"n": 0}
            continue
        ordered = sorted(finite)
        out[key] = {
            "mean": mean(finite),
            "std": pstdev(finite) if len(finite) > 1 else 0.0,
            "p10": ordered[int(0.10 * (len(ordered) - 1))],
            "p90": ordered[int(0.90 * (len(ordered) - 1))],
            "n": len(finite),
        }
    return out


def _report(summary: dict[str, dict]) -> list[str]:
    """The handful of comparisons that are worth reading first. Every line is a number against
    the number it has to be judged against -- never a level on its own."""

    def m(key):
        entry = summary.get(key)
        return entry["mean"] if entry and entry.get("n") else float("nan")

    lines = []
    floor, nce, cross = m("nce/floor_same_question"), m("nce/loss"), m("nce/loss_cross_question")
    n_same = m("nce/negatives_same_question")
    lines.append("L_NCE -- is the residual same-question ambiguity, or real error? (§16.4)")
    lines.append(f"  nce/loss                    {nce:8.4f}   vs chance {m('nce/chance'):.3f}")
    lines.append(f"  nce/floor_same_question     {floor:8.4f}   = log(1 + {n_same:.1f})")
    lines.append(f"  nce/loss_cross_question     {cross:8.4f}   -> ~0 means cross-question SOLVED")
    lines.append(
        f"  nce/accuracy_within_question{m('nce/accuracy_within_question'):8.4f}   "
        f"vs chance {1.0 / (1.0 + n_same):.4f}"
    )
    if math.isfinite(nce) and math.isfinite(floor):
        head = nce - floor
        verdict = (
            "AT THE FLOOR -- more data cannot move nce/loss; only within-question ranking can"
            if head < 0.25
            else "headroom above the floor -- cross-question separation is still unfinished"
        )
        lines.append(f"  headroom above floor        {head:+8.4f}   {verdict}")

    lines.append("")
    lines.append("The ruler, and the signal it carries (§10 #2, #3, #14)")
    lines.append(
        f"  probe02/delta_good_mean     {m('probe02/delta_good_mean'):8.4f}   "
        f"target {m('probe02/target_good_step_delta'):.4f}"
    )
    lines.append(f"  probe03/delta_bad_mean      {m('probe03/delta_bad_mean'):8.4f}")
    lines.append(
        f"  probe03/gap                 {m('probe03/gap'):8.4f}   "
        "THE SIGNAL -- collapsing to 0 means the error signal is flattened"
    )
    for key in (
        "probe14/delta_good_of_correct/mean",
        "probe14/delta_good_of_correct/positive_fraction",
        "probe14/delta_good_of_correct/frac_above_natural",
        "probe14/delta_good_of_correct/p90",
        "probe14/delta_good_of_correct/p99",
        "probe14/delta_good_of_incorrect/mean",
        "probe14/delta_boundary/mean",
        "good/above_target_fraction",
        "good/delta_mean",
    ):
        if key in summary:
            lines.append(f"  {key:<44}{m(key):8.4f}")
    lines.append(
        "  ^ a POSITIVE tail on good steps of CORRECT trajectories is F1 leaking with no loss"
    )
    lines.append("    training against it -- READ frac_above_natural AND p99, NOT the mean.")
    lines.append("    At step 750 the mean was +0.240 and frac_above_natural was 0.34, which")
    lines.append("    drove tau to 2.39 and capped F1 at 0.456. The fix is (6) L_good (§7.12),")
    lines.append("    not a §7.10 pairing expansion: the bound on a good step's Delta is")
    lines.append("    one-sided by construction, and more L_step terms do not close it.")

    lines.append("")
    lines.append("Who beats the positive -- hub collapse, or genuine confusion?")
    share = m("confusion/top5pct_rows_share_of_beats")
    lines.append(f"  confusion/rows_beating_positive      {m('confusion/rows_beating_positive'):8.4f}")
    lines.append(f"    of which same-question             {m('confusion/beats_same_question'):8.4f}")
    lines.append(f"    of which cross-question            {m('confusion/beats_cross_question'):8.4f}")
    lines.append(
        f"  top 5% of rows hold this many beats  {share:8.4f}   "
        + ("HUB COLLAPSE -- a few rows are near every goal" if share > 0.30 else
           "spread out -- genuine confusion, which training can fix")
    )
    lines.append(
        f"  beats from INCORRECT-traj rows       {m('confusion/beats_from_incorrect_rows'):8.4f}"
        f"   vs their share of rows {m('confusion/incorrect_row_fraction'):.4f}"
    )

    lines.append("")
    lines.append("Everything else that has a target")
    for key, note in (
        ("invariance/loss", "#9: target 0; the old project plateaued at 0.43"),
        ("backup/loss", "expected NEGATIVE; watch for plateau and NaN, not sign"),
        ("backup/linear_branch_fraction", "#15: ~1.0 at init, ~0 within ~100 steps"),
        ("backup/div_same_question", "#13: vs cross-question, the rho decision (§7.4.2)"),
        ("backup/div_cross_question", "#13"),
        ("step/loss", "softplus(m - step/delta_mean); check the sandwich, not a level"),
        ("step/delta_mean", "must climb toward m; still negative late = phi ignores its action"),
        ("step/distinct_z", "#17: expect ~28. 21 means the sampler is on 2c+1i"),
        ("good/loss", "#18: relu(Delta - c) over good steps; sandwich, not a level (§7.12)"),
        ("good/margin", "c -- MUST BE NEGATIVE (-0.693 at discount 0.5)"),
        ("good/above_target_fraction", "the number lambda_good exists to move"),
        ("probe01/questions_in_batch", "Q -- sets the L_NCE floor above"),
        ("probe01/distinct_goal_ratio", "#1: << 1 means root cause B (goal collapse) is back"),
        ("probe04/symmetric_share", "#4: if asymmetry is a minority, do not claim it drives it"),
        ("probe07/within_trajectory_spread", "#7: -> 0 means states are squashed (§16.4)"),
        ("probe08/corr_distance_psi_norm", "#8: r > 0.9 means the goal contributes nothing"),
        ("probe16/goal_is_terminal_fraction", "#16: expect ~0.41 at discount 0.5"),
    ):
        if key in summary:
            lines.append(f"  {key:<28}{m(key):8.4f}   {note}")
    return lines


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="the §10 panel over a phase-1 checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--out", default=None, help="JSON path (default: <checkpoint>/diagnose.json)")
    parser.add_argument(
        "--untrained",
        action="store_true",
        help="NULL BASELINE on the same batches: base backbone, no adapter, random-init heads. "
        "Run it. A level without its baseline is what §10.1.1 exists to warn about.",
    )
    args = parser.parse_args(argv)

    ckpt = Path(args.checkpoint)
    cfg = load_config_from_checkpoint(ckpt)
    seed_everything(cfg.run.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(cfg)
    model = FeynmanPRM(
        cfg,
        read_hidden_size(cfg.model.name),
        backbone=_backbone(cfg, ckpt, args.untrained),
        with_goal_head=False,
    )
    if not args.untrained:
        load_heads(model, ckpt)
    model.pad_id = tokenizer.pad_token_id
    model.to(device).eval()

    rows = read_sequences_parquet(Path(cfg.data.dir) / "sequences.parquet", split=args.split)
    slots = build_question_slots(rows)
    # Epoch 0 with the run's own seed, so these are batches the run actually saw (on `train`).
    batches = epoch_batches(rows, slots, cfg, 0, epoch_rng(cfg.run.seed, 0))
    batches = batches[: args.batches]

    samples: dict[str, list[float]] = {}
    for micro, batch_rows in enumerate(batches):
        batch, goals, reps, matrices, out = run_micro_batch(
            model, rows, batch_rows, cfg, device, goal_rng(cfg.run.seed, 0, micro)
        )
        # phase1_loss returns the LOSS diagnostics only. The §10 panel -- probes 1,2,3,4,7,8,
        # 12,14,16, which includes the ruler and the three-way delta histogram that predicts
        # F1 -- lives in batch_probes, and train.py calls it separately in the logging block.
        # Leaving it out here is what printed `nan` across the entire ruler section.
        extra = dict(
            batch_probes(reps.psi, reps.phi, batch, goals, matrices, model.distance, cfg)
        )
        extra.update(asymmetry_score(reps.psi, batch, model.distance))
        extra.update(
            confusion_structure(
                matrices.Dist,
                matrices.pos_row,
                matrices.SQ,
                batch.traj_correct[batch.row_traj],
            )
        )
        for key, value in {**out.info, **extra}.items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            samples.setdefault(key, []).append(value)
        for key, value in out.terms.items():
            samples.setdefault(f"terms/{key}", []).append(float(value))
        if (micro + 1) % 10 == 0:
            print(f"  {micro + 1}/{len(batches)} batches", flush=True)

    summary = _summarise(samples)
    payload = {
        "checkpoint": str(ckpt),
        "untrained": args.untrained,
        "split": args.split,
        "batches": len(batches),
        "batch_stats": batch_stats(batches, rows),
        "discount": cfg.discount,
        "neg_log_gamma": cfg.neg_log_gamma,
        "summary": summary,
    }
    out_path = Path(args.out) if args.out else ckpt / "diagnose.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))

    print()
    print("=" * 88)
    print(f"{ckpt}   split={args.split}   batches={len(batches)}"
          f"{'   UNTRAINED NULL' if args.untrained else ''}")
    print("=" * 88)
    print("\n".join(_report(summary)))
    print()
    print(f"full panel ({len(summary)} keys) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
