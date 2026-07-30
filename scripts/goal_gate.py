#!/usr/bin/env python
"""§10.1's goal-head go/no-go gate, on a phase-1 checkpoint (milestone §18.1, PLAN step 5).

    within = mean over q of mean_{j!=k} d(psi(s_T^j), psi(s_T^k))    same question
    across = mean over q!=q' of d(psi(s_T^q), psi(s_T^q'))
    ratio  = within / across

    ratio -> 1   -> STOP AND REDESIGN before spending GPU hours. The goal head cannot work
                    no matter how it is trained; the fallback is the goal-free asymmetry
                    score (§9.4), NOT a reference goal (that is a skyline, §5.1).

**READ `gate/auc`, NOT `gate/ratio`.** §10.1's "< 0.3 -> proceed" was never derived -- it is
the one number in CLAUDE.md with no §17 provenance entry -- and it is far stricter than what
the goal head needs. In 512 dimensions a mean-distance ratio of **0.62 is already 100%
same-question retrieval** (the simulation is tabulated in `losses/goal.py:terminal_separability`).
The ratio's null is what it does pin down: iid terminals with no question structure give
**1.000** to three decimals, so the ratio detects "no signal at all" and nothing finer.
`--untrained` reruns the whole thing on the base model with random heads to show that null on
the real data rather than in simulation.

Cheap: it needs only terminals, and there are 3.36 correct solutions per question on average
(§4.2), which is enough to estimate it.
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
from feynman_prm.losses.goal import terminal_separability, terminal_spread_ratio
from feynman_prm.model.backbone import load_backbone_with_adapter, load_tokenizer, read_hidden_size
from feynman_prm.model.wrapper import FeynmanPRM
from feynman_prm.utils.checkpoint import load_config_from_checkpoint, load_heads


def _backbone(cfg, ckpt: Path, args):
    """The trained adapter, or the bare base model for the `--untrained` null."""
    if not args.untrained:
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


def add_gate_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The flags every probe that reads cached terminals needs. Shared so a second probe
    cannot silently cache a *different* set of terminals than the gate ran on."""
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--questions", type=int, default=200)
    parser.add_argument("--batch-sequences", type=int, default=16)
    parser.add_argument(
        "--untrained",
        action="store_true",
        help="NULL BASELINE: base backbone, no adapter, random-init psi. The gate's numbers "
        "are only interpretable against this -- a ratio is a comparison and the checkpoint "
        "alone gives you one side of it.",
    )
    return parser


@torch.no_grad()
def cache_terminals(args):
    """(model, psi_terminals (N,D), question_index (N,)) -- psi(s_T) of every correct
    trajectory of the first `--questions` questions that have >= 2 of them."""
    ckpt = Path(args.checkpoint)
    cfg = load_config_from_checkpoint(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(cfg)
    model = FeynmanPRM(cfg, read_hidden_size(cfg.model.name), backbone=_backbone(cfg, ckpt, args))
    if not args.untrained:
        load_heads(model, ckpt)
    model.to(device).eval()

    rows = read_sequences_parquet(Path(cfg.data.dir) / "sequences.parquet", split="train")
    by_question = defaultdict(list)
    for row in rows:
        if row.correct:
            by_question[row.qid].append(row)
    # Only questions with >=2 correct solutions can contribute a within-question pair.
    qids = [q for q in sorted(by_question) if len(by_question[q]) >= 2][: args.questions]

    terminals, question_index = [], []
    pending = []

    def flush():
        if not pending:
            return
        batch = collate([r for _, r in pending], pad_id=tokenizer.pad_token_id).to(device)
        reps = model(batch)
        for b, (qi, _) in enumerate(pending):
            terminals.append(reps.psi[int(batch.traj_terminal[b])].float().cpu())
            question_index.append(qi)
        pending.clear()

    for qi, qid in enumerate(qids):
        for row in by_question[qid][:4]:
            pending.append((qi, row))
            if len(pending) >= args.batch_sequences:
                flush()
    flush()

    return (
        model,
        torch.stack(terminals).to(device),
        torch.as_tensor(question_index).to(device),
        cfg,
    )


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    parser = add_gate_args(argparse.ArgumentParser(description="the §10.1 goal-head gate"))
    args = parser.parse_args(argv)

    model, psi_t, qidx, _ = cache_terminals(args)
    qids = torch.unique(qidx)

    gate = terminal_spread_ratio(psi_t, qidx, model.distance)
    gate.update(terminal_separability(psi_t, qidx, model.distance))
    gate["gate/questions"] = len(qids)
    gate["gate/terminals"] = len(psi_t)
    gate["gate/untrained_baseline"] = args.untrained
    print(json.dumps(gate, indent=2))

    # The gate is on AUC, not on the ratio. Chance is 0.5 and it is scale-free; the "< 0.3"
    # ratio rule was never derived and rejects representations with perfect retrieval.
    auc = gate["gate/auc"]
    verdict = "PROCEED" if auc > 0.9 else "INVESTIGATE" if auc > 0.7 else "STOP AND REDESIGN (§10.1)"
    print(
        f"\nauc      = {auc:.3f}  (chance 0.5)   ->  {verdict}"
        f"\nrecall@1 = {gate['gate/recall_at_1']:.3f}  (chance ~{1 / max(len(qids), 1):.3f})"
        f"\nratio    = {gate['gate/ratio']:.3f}  (no-structure null is 1.000; NOT gated on -- see"
        " losses/goal.py:terminal_separability)"
    )
    return 0 if auc > 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
