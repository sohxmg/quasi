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
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data.collate import collate
from .data.math_shepherd import read_sequences_parquet
from .data.tokenize import sep_token_id
from .diagnostics.logging import Progress, RunLogger
from .losses.goal import goal_loss, terminal_separability, terminal_spread_ratio
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
    cap = cfg.goal_head.max_terminals_per_question
    n_terminals = sum(min(cap, len(by_question[q])) for q in qids)
    print(
        f"[cache] {len(rows):,} train sequences -> {len(qids):,} questions with a correct "
        f"solution, {n_terminals:,} terminals to encode (cap {cap}/question).\n"
        f"[cache] Both passes are backbone forwards and this is the slow part of phase 2 "
        f"-- expect tens of minutes, not seconds.",
        flush=True,
    )

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
        prompt_progress.advance(len(pending))
        pending.clear()

    prompt_progress = Progress("cache/h_s0", len(qids))
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
        terminal_progress.advance(len(pending_rows))
        pending_rows.clear()

    terminal_progress = Progress("cache/psi_terminals", n_terminals)
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


GATE_MAX_QUESTIONS = 200
GATE_MAX_TERMINALS_PER_QUESTION = 4


def gate_subsample(
    terminals: torch.Tensor,
    terminal_question: torch.Tensor,
    max_questions: int = GATE_MAX_QUESTIONS,
    per_question: int = GATE_MAX_TERMINALS_PER_QUESTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The §10.1 gate is an ESTIMATE and must be run on a subsample, not the whole cache.

    `terminal_spread_ratio` and `terminal_separability` both materialise the FULL pairwise
    matrix -- `psi[:, None, :]` against `psi[None, :, :]`, so `O(N^2 * D)`. That is fine at
    `scripts/goal_gate.py`'s default 200 questions (~525 terminals, ~0.5 GiB) and impossible
    on the phase-2 cache, which holds every correct terminal of every selected question:

        N ~ 95,000  ->  N^2 * (D/2) * 4 B  =  **8,621 GiB**, the observed OOM

    §10.1 sizes the estimate itself -- "3.36 correct solutions per question on average, which
    is enough" -- so this is the shape the gate was designed for, not a degraded version of it.

    Deterministic (first `max_questions` in cache order, first `per_question` terminals of
    each), and restricted to questions with >= 2 terminals since a singleton contributes no
    within-question pair.
    """
    from collections import Counter

    labels = terminal_question.tolist()
    counts = Counter(labels)
    eligible: set[int] = set()
    for q in labels:
        if counts[q] >= 2 and q not in eligible and len(eligible) < max_questions:
            eligible.add(q)

    taken: dict[int, int] = defaultdict(int)
    chosen = []
    for i, q in enumerate(labels):
        if q in eligible and taken[q] < per_question:
            taken[q] += 1
            chosen.append(i)
    idx = torch.as_tensor(chosen, dtype=torch.long)
    return terminals[idx], terminal_question[idx]


def fit_goal_head(model, cache, cfg: Config, device, logger: RunLogger) -> None:
    h_s0, terminals, terminal_question, _ = cache
    h_s0 = h_s0.to(device)
    terminals = terminals.to(device)
    terminal_question = terminal_question.to(device)

    optimizer = torch.optim.AdamW(model.goal_head.parameters(), lr=cfg.goal_head.lr, foreach=False)
    n_all = terminals.shape[0]

    # ---- HELD-OUT QUESTIONS (added 2026-08-04) --------------------------------------------
    # `goal/loss` fell 7.469 -> 4.491 over 20 epochs and was STILL FALLING, which is the usual
    # reason to train longer. It is not a reason on its own: the head is ~1.6M params fitting
    # one target per question, there was no held-out number, and eval queries questions it has
    # never seen. Raising `epochs` on a train curve alone is choosing a stopping point on the
    # only series that cannot tell learning from memorising.
    #
    # The split is by QUESTION, not by terminal -- holding out one of a question's terminals
    # while training on its siblings leaks h_s0 and measures nothing eval cares about. This is
    # a slice on a cached tensor, so it costs nothing and the training set shrinks by ~6%.
    n_val_q = int(cfg.goal_head.val_questions)
    n_q = int(h_s0.shape[0])
    if n_val_q > 0 and n_q > n_val_q:
        g = torch.Generator(device="cpu").manual_seed(cfg.run.seed)
        val_q = torch.randperm(n_q, generator=g)[:n_val_q].to(device)
        is_val_q = torch.zeros(n_q, dtype=torch.bool, device=device)
        is_val_q[val_q] = True
        val_rows = is_val_q[terminal_question]
    else:
        val_rows = torch.zeros(n_all, dtype=torch.bool, device=device)
    train_idx = (~val_rows).nonzero(as_tuple=True)[0]
    val_idx = val_rows.nonzero(as_tuple=True)[0]

    terminals_val, tq_val = terminals[val_idx], terminal_question[val_idx]
    terminals, terminal_question = terminals[train_idx], terminal_question[train_idx]
    n = terminals.shape[0]
    if n == 0:
        # The empty cache used to reach the log line below with `info` never assigned and die
        # on an UnboundLocalError 20 frames from the cause. Say what is actually wrong.
        raise RuntimeError(
            "no cached terminals: sequences.parquet's train split has no correct trajectories. "
            "Re-run scripts/prepare_data.py (§8.2 -- the selection SHA moves with n_questions)."
        )

    batches_per_epoch = math.ceil(n / cfg.goal_head.batch_size)
    # §11.1's discipline, which phase 1 asserts and phase 2 never did: a schedule is
    # meaningless without a step count, and this one is `ceil(n / batch_size) * epochs` --
    # small enough to be worth printing before it runs rather than inferring from wall clock.
    logger.event(
        "phase2/schedule",
        {
            "questions": int(h_s0.shape[0]),
            "terminals": n_all,
            "terminals_train": n,
            "terminals_val": int(terminals_val.shape[0]),
            "val_questions": n_val_q,
            "batch_size": cfg.goal_head.batch_size,
            "batches_per_epoch": batches_per_epoch,
            "epochs": cfg.goal_head.epochs,
            "optimizer_steps": batches_per_epoch * cfg.goal_head.epochs,
            "lr": cfg.goal_head.lr,
        },
    )

    step = 0
    best = {"epoch": -1, "val": float("inf")}
    for epoch in range(cfg.goal_head.epochs):
        perm = torch.randperm(n, device=device)
        # Epoch MEANS, weighted by terminals per batch. The last minibatch's `info` was what
        # this used to log, and a single 512-terminal draw is far too noisy to read a trend
        # off -- which mattered because 20 epochs is the entire curve.
        totals: dict[str, float] = defaultdict(float)
        seen = 0
        t0 = time.time()
        for start in range(0, n, cfg.goal_head.batch_size):
            idx = perm[start : start + cfg.goal_head.batch_size]
            pred = model.goal_head(h_s0)                     # (Q, D)
            loss, info = goal_loss(
                pred, terminals[idx], terminal_question[idx], model.distance
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            for key, value in info.items():
                totals[key] += value * len(idx)
            seen += len(idx)

        # The number `epochs` should be chosen on. Held-out QUESTIONS, so it is the same
        # generalisation the eval asks for, and it is one forward over a frozen tensor.
        val_info: dict[str, float] = {}
        if terminals_val.shape[0]:
            with torch.no_grad():
                _, vi = goal_loss(
                    model.goal_head(h_s0), terminals_val, tq_val, model.distance
                )
            val_info = {
                "goal/val_loss": vi["goal/loss"],
                "goal/val_d_pred_to_target": vi["goal/d_pred_to_target"],
                "goal/val_gap": vi["goal/loss"] - totals["goal/loss"] / seen,
            }
            if vi["goal/loss"] < best["val"]:
                best = {"epoch": epoch, "val": vi["goal/loss"]}

        logger.log(
            epoch,
            {
                "goal/epoch": epoch,
                "goal/optimizer_step": step,
                "goal/batches": batches_per_epoch,
                "goal/terminals_seen": seen,
                "goal/seconds": time.time() - t0,
                **{key: total / seen for key, total in totals.items()},
                **val_info,
            },
            console=True,
        )

    # NOT early stopping, and deliberately not: the checkpoint saved is the LAST epoch, as it
    # always was. This only reports where val bottomed, so `epochs` is set from a measurement
    # instead of from a train curve. If `best_epoch` is far below `epochs - 1`, lower the key.
    if best["epoch"] >= 0:
        logger.event(
            "phase2/val_best",
            {
                "best_epoch": best["epoch"],
                "best_val_loss": best["val"],
                "epochs_run": cfg.goal_head.epochs,
                "note": "checkpoint is the LAST epoch, not the best -- set goal_head.epochs "
                        "to best_epoch + 1 and refit if they differ. Refit is seconds with "
                        "--from-cache.",
            },
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Feynman-PRM phase 2 (goal head)")
    parser.add_argument("--checkpoint", required=True, help="phase-1 checkpoint directory")
    parser.add_argument("--out", default=None, help="where to write the phase-2 checkpoint")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="discard an existing phase-2 checkpoint at --out. See the guard below.",
    )
    parser.add_argument(
        "--from-cache", action="store_true",
        help="reuse <out>/cache.pt instead of rebuilding it. The cache is psi(s_T) and h_s0 "
             "on a FROZEN backbone (locked #15), so it is a pure function of the phase-1 "
             "checkpoint -- refitting the head at a different `epochs` costs seconds, not the "
             "~75 min the cache does. Invalid if the phase-1 checkpoint or sequences.parquet "
             "changed; it is not fingerprinted, so do not reuse one across checkpoints.",
    )
    parser.add_argument(
        "--set", action="append", default=[],
        help="key.path=value override, as in train.sh. **Phase 2 reads its config from the "
             "PHASE-1 CHECKPOINT's saved config.yaml, not from config/default.yaml** -- the "
             "phase-1 config is the one that produced psi and phi, so it has to be. That "
             "means editing config/default.yaml does NOTHING here, which cost one 20-epoch "
             "fit on 2026-08-04 that was meant to be 100. Use this flag, e.g. "
             "--set goal_head.epochs=13.",
    )
    args = parser.parse_args(argv)

    ckpt = Path(args.checkpoint)
    cfg = load_config_from_checkpoint(ckpt)
    if args.set:
        from .config import apply_override, config_from_dict
        import yaml as _yaml

        raw = _yaml.safe_load((ckpt / "config.yaml").read_text())
        for assignment in args.set:
            apply_override(raw, assignment)
        cfg = config_from_dict(raw)
        print(f"[config] from {ckpt / 'config.yaml'} + {len(args.set)} override(s): "
              f"{', '.join(args.set)}", flush=True)
    else:
        print(f"[config] from {ckpt / 'config.yaml'} "
              f"(NOT config/default.yaml -- use --set to override; "
              f"goal_head.epochs = {cfg.goal_head.epochs})", flush=True)
    seed_everything(cfg.run.seed)
    out_dir = Path(args.out) if args.out else ckpt.parent / "phase2"

    # B13/B14's guard, which phase 1 has had since 2026-08-04 and phase 2 never did.
    # `out_dir` defaults to `<phase-1 run>/phase2`, so a second phase-2 fit on the SAME phase-1
    # checkpoint silently overwrites the first -- and for `runs/phase1/final` that is the
    # checkpoint every reported number in the project was measured on (§9.3.1's F1 table, §9.6,
    # §9.7, §9.8), plus its `deltas.npz` and the `gate_before_fit` event. A different phase-1
    # run writes to its own directory and never collides; the hazard is refitting the same one.
    # Keyed on `heads.pt` per B14 -- `config.yaml` is a sidecar that proves nothing was trained.
    existing = out_dir / "final" / "heads.pt"
    if existing.exists() and not args.overwrite:
        raise SystemExit(
            f"\n{out_dir / 'final'} already holds a trained goal head ({existing}).\n"
            f"Refitting would overwrite it, and for runs/phase1 that is the checkpoint every\n"
            f"reported number was measured on. Pass --out <somewhere else>, or --overwrite if\n"
            f"discarding it is really the intent.\n"
        )
    logger = RunLogger(out_dir.parent, out_dir.name, cfg.log.wandb, cfg.log.wandb_project, cfg.to_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg)
    backbone = load_backbone_with_adapter(cfg, ckpt / "adapter")
    from .model.backbone import read_hidden_size

    model = FeynmanPRM(cfg, read_hidden_size(cfg.model.name), backbone=backbone, with_goal_head=True)
    # A phase-1 checkpoint carries psi and phi and NOTHING ELSE -- the goal head does not exist
    # until here (§7.7, locked #15), so its absence is the expected state and not the §14 trap
    # the guard is for. Named explicitly so a phase-2 checkpoint that lost a TRAINED goal head
    # still fails loudly everywhere else.
    loaded = load_heads(model, ckpt, allow_missing=("goal_head.",))
    # COUNTS, never the lists. `missing` is every backbone parameter -- heads.pt holds only
    # HEAD_PREFIXES by design, so load_state_dict reports the whole 1.5B backbone as missing
    # and `**loaded` put ~340 layer names on the console.
    logger.event(
        "phase2/heads_loaded",
        {
            "phase1_step": loaded["step"],
            "freshly_initialised": loaded["freshly_initialised"],
            "n_missing_non_head": len(loaded["missing"]),
            "n_unexpected": len(loaded["unexpected"]),
        },
    )
    model.to(device)
    model.freeze_for_phase2()
    logger.event("phase2/trainable", assert_phase2_trainable(model))

    # ~300k rows rebuilt into SequenceRow objects one at a time: a minute of silence on its
    # own, before the backbone has done anything.
    cache_path = out_dir / "cache.pt"
    if args.from_cache:
        if not cache_path.exists():
            raise SystemExit(f"--from-cache: no cache at {cache_path}")
        blob = torch.load(cache_path, map_location="cpu")
        cache = (blob["h_s0"], blob["terminals"], blob["terminal_question"], blob["qids"])
        print(f"[cache] reused {cache_path} "
              f"({cache[0].shape[0]:,} questions, {cache[1].shape[0]:,} terminals)", flush=True)
    else:
        parquet = Path(cfg.data.dir) / "sequences.parquet"
        print(f"[cache] reading {parquet} (train split)...", flush=True)
        t_read = time.time()
        rows = read_sequences_parquet(parquet, split="train")
        print(f"[cache] read {len(rows):,} rows in {time.time() - t_read:.1f}s", flush=True)

        cache = build_cache(model, tokenizer, rows, cfg, device)
        torch.save(
            {"h_s0": cache[0], "terminals": cache[1],
             "terminal_question": cache[2], "qids": cache[3]},
            cache_path,
        )
        print(f"[cache] wrote {cache_path}", flush=True)

    psi_t, qidx = gate_subsample(cache[1], cache[2])
    gate = terminal_spread_ratio(psi_t.to(device), qidx.to(device), model.distance)
    gate.update(terminal_separability(psi_t.to(device), qidx.to(device), model.distance))
    gate["gate/terminals"] = len(psi_t)
    logger.event("phase2/gate_before_fit", gate)
    # READ `auc`, NOT `ratio` (§10.1.1). The "< 0.3" rule was never derived -- it is the one
    # number in CLAUDE.md with no §17 provenance entry -- and at D=512 a ratio of 0.62 is
    # already 100% same-question retrieval. Report against the measured untrained baseline
    # (auc 0.904, recall@1 0.618) rather than a constant, because a ratio is a comparison.
    print(
        f"[gate] auc = {gate['gate/auc']:.3f}  recall@1 = {gate['gate/recall_at_1']:.3f}  "
        f"ratio = {gate['gate/ratio']:.3f}   on {len(psi_t)} terminals / "
        f"{len(torch.unique(qidx))} questions."
        "\n[gate] The untrained baseline measured auc 0.904 / recall@1 0.618 (§10.1.1) -- "
        "same-question terminals are similar before ANY training, because the solutions share "
        "the question text. Compare against `scripts/goal_gate.py --untrained`, not against a "
        "fixed threshold.",
        flush=True,
    )

    fit_goal_head(model, cache, cfg, device, logger)
    # There is deliberately no `gate_after_fit`. The gate reads only `psi(s_T)` and the
    # distance, both FROZEN in phase 2 (locked #15), and the cache is a fixed tensor -- so it
    # is provably the same number as `gate_before_fit`, at another O(N^2) to find that out.
    # Phase 2 moves `goal_head`, and what moved is `goal/loss` and `goal/pred_variance`.
    save_checkpoint(out_dir / "final", model, cfg, tokenizer, step=cfg.goal_head.epochs)
    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
