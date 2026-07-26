"""Phase 2: fit the goal head alone, on FROZEN cached vectors (§7.7, locked #15).

Phase 1 trains {LoRA, psi, phi} and has no goal head at all. Phase 2 freezes everything and
fits `goal_head` on precomputed vectors, so `.detach()` is structural rather than remembered
and the head cannot corrupt the metric through `h_{s_0}`.

Cheap, because everything is cacheable:

* `h_{s_0}` is the hidden at the separator after the prompt and, under causal attention,
  depends only on the prompt -- so it is identical across all trajectories of a question.
  **One ~150-token forward per question**, not one full-sequence forward per row.
* `psi(s_T^c)` is fixed once psi and the backbone are frozen.

The §10.1 gate runs on those same cached terminals BEFORE the head is fitted: if the
within/across ratio is near 1, the goal head cannot work no matter how it is trained, and you
learn that in minutes rather than after a full cycle.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data.collate import collate
from .data.math_shepherd import read_sequences_parquet
from .data.tokenize import sep_token_id
from .diagnostics.logging import RunLogger
from .losses.goal import goal_loss, terminal_spread_ratio
from .model.backbone import assert_phase2_trainable, load_backbone_with_adapter, load_tokenizer
from .model.wrapper import FeynmanPRM
from .utils.checkpoint import load_config_from_checkpoint, load_heads, save_checkpoint
from .utils.seeding import seed_everything


@torch.no_grad()
def build_cache(model, tokenizer, rows, cfg: Config, device, batch_sequences: int = 16):
    """(h_s0 per question, psi terminals, question index per terminal, qids)."""
    sep_token_id(tokenizer, cfg.data.sep_token)   # load-time assert: SEP is exactly one id (§6.1)
    pad_id = tokenizer.pad_token_id

    by_question: dict[str, list] = defaultdict(list)
    for row in rows:
        if row.correct:
            by_question[row.qid].append(row)
    qids = sorted(by_question)
    qindex = {q: i for i, q in enumerate(qids)}

    # ---- h_{s_0}: one prompt-only forward per question --------------------------------
    # The prompt text is not stored in sequences.parquet, so s_0's ids are recovered from
    # the stored sequence: everything up to and including the first separator IS the prompt
    # prefix, by construction of the one sequence builder (§6.1).
    h_s0 = torch.zeros(len(qids), model.hidden_size, dtype=torch.float32)
    pending: list[tuple[int, np.ndarray, int]] = []

    def flush_prompts() -> None:
        if not pending:
            return
        L = max(len(ids) for _, ids, _ in pending)
        input_ids = np.full((len(pending), L), pad_id, dtype=np.int64)
        mask = np.zeros((len(pending), L), dtype=np.int64)
        flat = np.zeros(len(pending), dtype=np.int64)
        for b, (_, ids, pos) in enumerate(pending):
            input_ids[b, : len(ids)] = ids
            mask[b, : len(ids)] = 1
            flat[b] = b * L + pos
        h = model.encode_states(
            torch.from_numpy(input_ids).to(device),
            torch.from_numpy(mask).to(device),
            torch.from_numpy(flat).to(device),
        )
        for b, (qi, _, _) in enumerate(pending):
            h_s0[qi] = h[b].float().cpu()
        pending.clear()

    for qid in qids:
        row = by_question[qid][0]
        s0 = int(row.state_pos[0])
        pending.append((qindex[qid], row.input_ids[: s0 + 1], s0))
        if len(pending) >= batch_sequences:
            flush_prompts()
    flush_prompts()

    # ---- psi(s_T^c) for up to max_terminals_per_question correct trajectories ---------
    terminals, terminal_question = [], []
    pending_rows: list[tuple[int, object]] = []

    def flush_terminals() -> None:
        if not pending_rows:
            return
        batch = collate([r for _, r in pending_rows], pad_id=pad_id).to(device)
        reps = model(batch)
        for b, (qi, _) in enumerate(pending_rows):
            terminals.append(reps.psi[int(batch.traj_terminal[b])].float().cpu())
            terminal_question.append(qi)
        pending_rows.clear()

    cap = cfg.goal_head.max_terminals_per_question
    for qid in qids:
        for row in by_question[qid][:cap]:
            pending_rows.append((qindex[qid], row))
            if len(pending_rows) >= batch_sequences:
                flush_terminals()
    flush_terminals()

    return (
        h_s0,
        torch.stack(terminals) if terminals else torch.zeros(0, model.cfg.heads.latent_dim),
        torch.as_tensor(terminal_question, dtype=torch.long),
        qids,
    )


def fit_goal_head(model, cache, cfg: Config, device, logger: RunLogger) -> None:
    h_s0, terminals, terminal_question, _ = cache
    h_s0 = h_s0.to(device)
    terminals = terminals.to(device)
    terminal_question = terminal_question.to(device)

    optimizer = torch.optim.AdamW(model.goal_head.parameters(), lr=cfg.goal_head.lr, foreach=False)
    n = terminals.shape[0]
    for epoch in range(cfg.goal_head.epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, cfg.goal_head.batch_size):
            idx = perm[start : start + cfg.goal_head.batch_size]
            pred = model.goal_head(h_s0)                     # (Q, D)
            loss, info = goal_loss(
                pred, terminals[idx], terminal_question[idx], model.distance
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        logger.log(epoch, {"goal/epoch": epoch, **info}, console=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Feynman-PRM phase 2 (goal head)")
    parser.add_argument("--checkpoint", required=True, help="phase-1 checkpoint directory")
    parser.add_argument("--out", default=None, help="where to write the phase-2 checkpoint")
    args = parser.parse_args(argv)

    ckpt = Path(args.checkpoint)
    cfg = load_config_from_checkpoint(ckpt)
    seed_everything(cfg.run.seed)
    out_dir = Path(args.out) if args.out else ckpt.parent / "phase2"
    logger = RunLogger(out_dir.parent, out_dir.name, cfg.log.wandb, cfg.log.wandb_project, cfg.to_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg)
    backbone = load_backbone_with_adapter(cfg, ckpt / "adapter")
    from .model.backbone import read_hidden_size

    model = FeynmanPRM(cfg, read_hidden_size(cfg.model.name), backbone=backbone, with_goal_head=True)
    load_heads(model, ckpt)
    model.to(device)
    model.freeze_for_phase2()
    logger.event("phase2/trainable", assert_phase2_trainable(model))

    rows = read_sequences_parquet(Path(cfg.data.dir) / "sequences.parquet", split="train")
    cache = build_cache(model, tokenizer, rows, cfg, device)
    torch.save(
        {"h_s0": cache[0], "terminals": cache[1], "terminal_question": cache[2], "qids": cache[3]},
        out_dir / "cache.pt",
    )

    gate = terminal_spread_ratio(cache[1].to(device), cache[2].to(device), model.distance)
    logger.event("phase2/gate_before_fit", gate)
    if gate["gate/ratio"] > 0.3:
        print(
            f"[gate] within/across = {gate['gate/ratio']:.3f} > 0.3 (§10.1). Correct endings "
            "do not cluster tightly by question; the goal head's target is weak. Proceeding, "
            "but read §10.1 before spending more GPU hours -- the fallback is the goal-free "
            "asymmetry score (§9.4), NOT a reference goal (§5.1).",
            flush=True,
        )

    fit_goal_head(model, cache, cfg, device, logger)
    logger.event("phase2/gate_after_fit", terminal_spread_ratio(
        cache[1].to(device), cache[2].to(device), model.distance
    ))
    save_checkpoint(out_dir / "final", model, cfg, tokenizer, step=cfg.goal_head.epochs)
    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
