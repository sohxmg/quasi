#!/usr/bin/env python
"""§9.9.6's pre-flight, on a TRAINED checkpoint -- the gate for every `L_NCE` mask.

    python scripts/nce_preflight.py --checkpoint runs/phase1/final
    python scripts/nce_preflight.py --checkpoint runs/phase1/final --batches 8

`nce/categorical_accuracy_backward` says ~65% of goal columns pick the wrong source row.
**This says WHERE those wrong picks land.** If a large share fall in the "nearer set" -- rows
on the goal's own trajectory, later than the sampled positive and at or before the goal, which
the ruler says are genuinely CLOSER (§16.4, §9.9.2) -- then `nce_mask_nearer_same_traj` is
aimed at the thing taking the softmax mass. If the share is at chance, the mask will be inert
and the mass is going somewhere else: find out where before spending a run.

**IT MUST BE READ AGAINST ITS NULL, AND THE NULL IS NOT ZERO.** If the argmax were uniform
over the R rows it would land in the nearer set with probability `nearer_set_size / R` ~= 0.002.
So on an UNTRAINED model -- where `categorical_accuracy_backward` is 0 and the argmax is
effectively random -- this statistic reads 0.000 and means nothing at all: the expected COUNT
of hits over ~200 columns is ~0.3. Reading a 20-step probe's 0.000 as "the mask is inert" is
the §10.1.1 mistake again (a threshold quoted against an assumed baseline). This script exists
so the number is taken on trained representations, where the argmax lands near the goal and
the statistic can discriminate.

Everything here runs with EVERY MASK OFF regardless of the config: `nce_loss` computes the
pre-flight on the raw logits, so the reading is a property of the checkpoint, not of the
config it is run under. `tau` cancels out of an argmax entirely.

Read-only: no optimizer, no checkpoint written, one forward per batch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run without `pip install -e .`

import argparse
import math

import numpy as np
import torch

from feynman_prm.data.collate import collate
from feynman_prm.data.goals import sample_goals
from feynman_prm.data.math_shepherd import read_sequences_parquet
from feynman_prm.data.sampler import build_question_slots, epoch_batches
from feynman_prm.losses.matrix import build_matrices
from feynman_prm.losses.nce import nce_loss
from feynman_prm.model.backbone import load_backbone_with_adapter, load_tokenizer, read_hidden_size
from feynman_prm.model.wrapper import FeynmanPRM
from feynman_prm.utils.checkpoint import load_config_from_checkpoint, load_heads
from feynman_prm.utils.seeding import epoch_rng, goal_rng, seed_everything

KEYS = ("nce/argmax_in_nearer_set", "nce/columns_with_nearer", "nce/nearer_set_size",
        "nce/categorical_accuracy_backward", "nce/negatives_per_column",
        "nce/loss", "nce/chance", "nce/logit_std", "nce/pos_dist", "nce/neg_dist",
        "nce/loss_cross_question", "nce/negatives_same_question")


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="§9.9.6's L_NCE pre-flight on a trained checkpoint")
    p.add_argument("--checkpoint", required=True, help="a PHASE-1 checkpoint directory")
    p.add_argument("--batches", type=int, default=6)
    p.add_argument("--untrained", action="store_true",
                   help="NULL BASELINE: base backbone, no adapter, random-init heads. The "
                        "reading is only interpretable against this -- see the module docstring")
    args = p.parse_args(argv)

    ckpt = Path(args.checkpoint)
    cfg = load_config_from_checkpoint(ckpt)
    seed_everything(cfg.run.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(cfg)
    if args.untrained:
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(
            cfg.model.name,
            dtype=torch.bfloat16 if cfg.train.bf16 else torch.float32,
            attn_implementation=cfg.model.attn_implementation,
        ).eval()
        for q in backbone.parameters():
            q.requires_grad_(False)
    else:
        backbone = load_backbone_with_adapter(cfg, ckpt / "adapter")

    model = FeynmanPRM(cfg, read_hidden_size(cfg.model.name), backbone=backbone)
    if not args.untrained:
        load_heads(model, ckpt)
    model.pad_id = tokenizer.pad_token_id
    model = model.to(device).eval()

    rows = read_sequences_parquet(Path(cfg.data.dir) / "sequences.parquet", split="train")
    slots = build_question_slots(rows)
    batches = epoch_batches(rows, slots, cfg, 0, epoch_rng(cfg.run.seed, 0))

    print(f"\ncheckpoint {ckpt}"
          f"{'  [UNTRAINED NULL]' if args.untrained else ''}"
          f"\n{len(batches)} batches available, reading {args.batches}\n")

    totals: dict[str, list[float]] = {k: [] for k in KEYS}
    for i, batch_rows in enumerate(batches[: args.batches]):
        batch_cpu = collate([rows[j] for j in batch_rows], pad_id=model.pad_id)
        goals = sample_goals(batch_cpu, cfg.discount, goal_rng(cfg.run.seed, 0, i)).to(device)
        batch = batch_cpu.to(device)
        reps = model(batch)
        m = build_matrices(reps.psi, reps.phi, batch, goals, model.distance, cfg)
        # EVERY MASK OFF: the pre-flight is a property of the checkpoint, not of the config.
        _, info = nce_loss(
            m.Dist, m.pos_row, temperature=cfg.losses.nce_temperature,
            row_traj=batch.row_traj, goal_traj=goals.goal_traj,
            row_step=batch.row_step, goal_step=m.goal_step,
            goal_is_terminal=m.goal_is_terminal, SQ=m.SQ,
            row_steps_to_end=batch.traj_T[batch.row_traj] - batch.row_step,
            row_correct=batch.traj_correct[batch.row_traj],
        )
        for k in KEYS:
            if k in info:
                totals[k].append(float(info[k]))
        print(f"  batch {i}: R={m.n_rows:4d} C={m.n_goals:4d}  "
              f"argmax_in_nearer={info.get('nce/argmax_in_nearer_set', float('nan')):.4f}  "
              f"acc={info['nce/categorical_accuracy_backward']:.4f}")

    print("\n  mean over batches")
    for k in KEYS:
        if totals[k]:
            print(f"    {k:<40} {np.mean(totals[k]):+.5f}")

    # ---- the verdict, against the null it has to be read against ----------------------
    hit = float(np.mean(totals["nce/argmax_in_nearer_set"]))
    nset = float(np.mean(totals["nce/nearer_set_size"]))
    R = float(np.mean(totals["nce/negatives_per_column"])) + 1.0
    acc = float(np.mean(totals["nce/categorical_accuracy_backward"]))
    null = nset / R
    print(f"\n  null (argmax uniform over R): {null:.5f}     measured: {hit:.5f}"
          f"     ratio: {hit / null if null else float('nan'):.1f}x")
    print(f"  categorical_accuracy_backward: {acc:.4f}   (chance {1.0 / R:.5f})")

    if acc < 5.0 / R:
        print("\n  ** INCONCLUSIVE. The argmax is at chance, so it is landing at random and this"
              "\n     statistic cannot discriminate -- 0.000 here is the NULL, not a finding."
              "\n     Run it on a TRAINED phase-1 checkpoint. **")
    elif hit > 5.0 * null:
        print("\n  ** The wrong picks concentrate in the nearer set."
              "\n     nce_mask_nearer_same_traj is aimed at what is taking the softmax mass. **")
    else:
        print("\n  ** At or near the null on a trained checkpoint: the mass is going somewhere"
              "\n     ELSE. The mask will be inert. Find out where before spending a run"
              "\n     (§9.9.6). **")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
