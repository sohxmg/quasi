#!/usr/bin/env python
"""§10.1's goal-head go/no-go gate, on a phase-1 checkpoint (milestone §18.1, PLAN step 5).

    within = mean over q of mean_{j!=k} d(psi(s_T^j), psi(s_T^k))    same question
    across = mean over q!=q' of d(psi(s_T^q), psi(s_T^q'))
    ratio  = within / across

    ratio < 0.3  -> PROCEED. Correct endings cluster by question, so the goal head has a
                    well-defined target.
    ratio -> 1   -> STOP AND REDESIGN before spending GPU hours. The goal head cannot work
                    no matter how it is trained; the fallback is the goal-free asymmetry
                    score (§9.4), NOT a reference goal (that is a skyline, §5.1).

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
from feynman_prm.losses.goal import terminal_spread_ratio
from feynman_prm.model.backbone import load_backbone_with_adapter, load_tokenizer, read_hidden_size
from feynman_prm.model.wrapper import FeynmanPRM
from feynman_prm.utils.checkpoint import load_config_from_checkpoint, load_heads


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="the §10.1 goal-head gate")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--questions", type=int, default=200)
    parser.add_argument("--batch-sequences", type=int, default=16)
    args = parser.parse_args(argv)

    ckpt = Path(args.checkpoint)
    cfg = load_config_from_checkpoint(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(cfg)
    backbone = load_backbone_with_adapter(cfg, ckpt / "adapter")
    model = FeynmanPRM(cfg, read_hidden_size(cfg.model.name), backbone=backbone)
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

    gate = terminal_spread_ratio(
        torch.stack(terminals).to(device), torch.as_tensor(question_index).to(device), model.distance
    )
    gate["gate/questions"] = len(qids)
    gate["gate/terminals"] = len(terminals)
    print(json.dumps(gate, indent=2))
    verdict = "PROCEED" if gate["gate/ratio"] < 0.3 else "STOP AND REDESIGN (§10.1)"
    print(f"\nratio = {gate['gate/ratio']:.3f}  ->  {verdict}")
    return 0 if gate["gate/ratio"] < 0.3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
