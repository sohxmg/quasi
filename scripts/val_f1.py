#!/usr/bin/env python
"""The F1 CEILING of a phase-1 checkpoint, on held-out val questions, with no goal head.

§9.2 fits `tau` on val Math-Shepherd questions under the ProcessBench rule. That normally
needs `goal_head(h_{s_0})`, which does not exist until phase 2 -- so this script substitutes a
**real terminal of another correct trajectory of the same question**, exactly the §9.5 skyline
substitution, and for the same reason: it splits one failure mode into two.

    F1 here is BAD                 -> the geometry never learned correctness. No goal head can
                                      rescue it, and phase 2 is wasted GPU time.
    F1 here is GOOD, phase 2 bad   -> the goal head is the bottleneck, which is fixable.

So this is a **ceiling, not a result** -- it is handed a real ending that eval will not have,
and §5.1's rule applies: never report it as a number for the method. It exists to decide
whether to spend the next GPU hours on phase 2 or on the phase-1 loss set.

The three numbers to read are `acc_error`, `acc_correct` and their harmonic mean. They fail
for different reasons and averaging them hides which (§7.6.6): `L_step` trains the error half
and NOTHING trains the clean half.

    python scripts/val_f1.py --checkpoint runs/phase1/step750
    python scripts/val_f1.py --checkpoint runs/phase1/step750 --untrained
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run without `pip install -e .`

import argparse
import json
from collections import defaultdict

import torch

from feynman_prm.data.collate import collate
from feynman_prm.data.math_shepherd import read_sequences_parquet
from feynman_prm.eval.calibrate import calibrate_tau, natural_tau
from feynman_prm.eval.metrics import processbench_metrics
from feynman_prm.model.backbone import (
    load_backbone_with_adapter,
    load_tokenizer,
    read_hidden_size,
)
from feynman_prm.model.wrapper import FeynmanPRM
from feynman_prm.utils.checkpoint import load_config_from_checkpoint, load_heads
from feynman_prm.utils.indexing import predicted_label_from_deltas
from feynman_prm.utils.seeding import seed_everything


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
def _psi_terminals_and_states(model, rows, pad_id, device, batch_sequences):
    """psi at every state, and psi(s_T), for each row. One forward per chunk."""
    states, terminals = [], []
    for start in range(0, len(rows), batch_sequences):
        chunk = rows[start : start + batch_sequences]
        batch = collate(chunk, pad_id=pad_id).to(device)
        reps = model(batch)
        for b in range(batch.n_traj):
            T = int(batch.traj_T[b])
            offset = int(batch.traj_state_offset[b])
            psi = reps.psi[offset : offset + T + 1].float().cpu()
            states.append(psi)
            terminals.append(psi[-1])
    return states, terminals


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="val F1 ceiling with real terminal goals")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--questions", type=int, default=400)
    parser.add_argument("--batch-sequences", type=int, default=16)
    parser.add_argument("--untrained", action="store_true", help="null baseline")
    parser.add_argument("--out", default=None)
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

    rows = read_sequences_parquet(Path(cfg.data.dir) / "sequences.parquet", split="val")
    by_question = defaultdict(list)
    for row in rows:
        by_question[row.qid].append(row)
    # A question needs >= 2 correct trajectories: one to supply the goal and one to be scored
    # by it. Scoring a trajectory against its OWN terminal is d(x, x) = 0 at the end and would
    # flatter the clean half exactly where it is weakest.
    usable = [
        (qid, group)
        for qid, group in sorted(by_question.items())
        if sum(r.correct for r in group) >= 2
    ][: args.questions]
    if not usable:
        raise SystemExit("no val question has 2 correct trajectories -- widen --questions")

    flat = [row for _, group in usable for row in group]
    states, terminals = _psi_terminals_and_states(
        model, flat, tokenizer.pad_token_id, device, args.batch_sequences
    )
    index = {id(row): i for i, row in enumerate(flat)}

    deltas, labels = [], []
    for _, group in usable:
        correct_rows = [r for r in group if r.correct]
        for row in group:
            goal_source = next((c for c in correct_rows if c is not row), None)
            if goal_source is None:
                continue
            psi = states[index[id(row)]].to(device)
            goal = terminals[index[id(goal_source)]].to(device)
            d = model.distance(psi, goal.expand_as(psi)).float().cpu()
            deltas.append((d[1:] - d[:-1]).tolist())
            labels.append(-1 if row.correct else int(row.z))

    calibration = calibrate_tau(deltas, labels, cfg)
    at_best = processbench_metrics(
        [predicted_label_from_deltas(d, calibration.tau) for d in deltas], labels
    )
    at_natural = processbench_metrics(
        [predicted_label_from_deltas(d, natural_tau(cfg)) for d in deltas], labels
    )

    payload = {
        "checkpoint": str(ckpt),
        "untrained": args.untrained,
        "questions": len(usable),
        "trajectories": len(deltas),
        "tau": calibration.tau,
        "natural_tau": calibration.expected_tau,
        "sensitivity": calibration.sensitivity,
        "at_best_tau": at_best,
        "at_natural_tau": at_natural,
        "curve": calibration.curve,
    }
    # The null must NOT land on the trained run's path. It is the same checkpoint dir and the
    # same default filename, so `--untrained` silently overwrote the curve it exists to be
    # compared against -- the one artifact the comparison needs both halves of.
    default_name = "val_f1_untrained.json" if args.untrained else "val_f1.json"
    out_path = Path(args.out) if args.out else ckpt / default_name
    out_path.write_text(json.dumps(payload, indent=2, default=float))

    print()
    print("=" * 80)
    print(f"{ckpt}   val   {len(usable)} questions, {len(deltas)} trajectories"
          f"{'   UNTRAINED NULL' if args.untrained else ''}")
    print("=" * 80)
    print("CEILING, not a result: this is handed a real terminal that eval never has (§9.5).")
    print()
    print(f"  tau (fitted)        {calibration.tau:8.4f}   natural {calibration.expected_tau:.4f}")
    print(f"  tau sensitivity     {calibration.sensitivity:8.4f}   F1 range within +/-0.1 of tau")
    for name, metrics in (("at fitted tau", at_best), ("at natural tau", at_natural)):
        print(f"  {name}")
        print(f"    acc_error         {metrics['acc_error']:8.4f}   (n={metrics['n_error']:.0f})"
              "   trained by L_step")
        print(f"    acc_correct       {metrics['acc_correct']:8.4f}   (n={metrics['n_correct']:.0f})"
              "   trained by NOTHING -- rests on L_T (§7.6.6)")
        print(f"    F1                {metrics['f1']:8.4f}")
    print()
    print(f"curve ({len(calibration.curve)} taus) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
