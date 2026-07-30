"""Phase 1: train {LoRA, psi, phi} on (1) L_NCE, (2) L_I, (3) L_T, (4) L_CF, (5) L_step.

There is no goal head in this phase (§7.7) and no value head anywhere (§7.9).

Order of operations at launch, all of which have a §14 or §11.1 entry behind them:

    1. resolve the config strictly            (bug B4)
    2. build the epoch's batches and ASSERT the optimizer-step count   (§11.1: the old
       arithmetic silently produced a 106-step run and it would have completed)
    3. build the model and ASSERT the trainable set is exactly {LoRA, psi, phi}   (§14)
    4. build the cosine schedule and ASSERT it steps                  (bug B6)
    5. run the LONGEST batch of the epoch first, as a memory probe    (PLAN 4a)
    6. check the initialisation values against §18 on the first micro-batch

Run it under tmux. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` first (§13).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .config import Config, load_config
from .data.collate import collate
from .data.goals import sample_goals
from .data.math_shepherd import read_sequences_parquet
from .data.sampler import (
    batch_stats,
    build_question_slots,
    epoch_batches,
    expected_sequences_per_question,
    longest_batch_index,
    planned_optimizer_steps,
    steps_report,
)
from .diagnostics.logging import RunLogger
from .diagnostics.probes import asymmetry_score, batch_probes
from .losses.matrix import build_matrices
from .losses.total import expected_init_values, phase1_loss
from .model.backbone import (
    assert_phase1_trainable,
    load_backbone,
    load_tokenizer,
    param_groups,
    read_hidden_size,
)
from .model.wrapper import FeynmanPRM
from .utils.checkpoint import save_checkpoint
from .utils.seeding import epoch_rng, goal_rng, probe_rng, seed_everything

MIN_OPTIMIZER_STEPS = 300  # §11.1: below this the schedule is meaningless. Cut n_questions,
                           # NEVER raise grad_accum alone -- that is the knob that hid the
                           # 106-step regression.


def build_scheduler(optimizer, total_steps: int, cfg: Config):
    """Cosine with `warmup_ratio` warmup. Bug B6 was a scheduler that was never built at all,
    so the caller asserts the LR actually moves."""
    warmup = max(1, round(cfg.train.warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        if cfg.train.schedule == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_micro_batch(model, rows, batch_rows, cfg: Config, device, rng, step: int = 0):
    """One micro-batch: collate -> goals -> forward -> matrices -> loss."""
    # Goals are drawn on the CPU batch (the sampler is numpy, and there are no DataLoader
    # workers to seed), then both move to the device together.
    batch_cpu = collate([rows[i] for i in batch_rows], pad_id=model.pad_id)
    goals = sample_goals(batch_cpu, cfg.discount, rng).to(device)
    batch = batch_cpu.to(device)
    reps = model(batch)
    matrices = build_matrices(reps.psi, reps.phi, batch, goals, model.distance, cfg)
    out = phase1_loss(
        reps.psi,
        reps.phi,
        batch,
        matrices,
        model.distance,
        cfg,
        goal_traj=goals.goal_traj,
        step=step,          # (6) L_good's warmup ramp only (§7.12)
    )
    return batch, goals, reps, matrices, out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Feynman-PRM phase 1")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--set", action="append", default=[], help="key.path=value override")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--allow-short-run", action="store_true",
                        help="skip the >=300 optimizer-step assert (smoke tests only)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.set)
    if args.max_steps is not None:
        cfg = load_config(args.config, args.set + [f"train.max_steps={args.max_steps}"])
    seed_everything(cfg.run.seed)

    logger = RunLogger(
        cfg.run.out_dir, cfg.run.name, cfg.log.wandb, cfg.log.wandb_project, cfg.to_dict()
    )
    cfg.save(Path(cfg.run.out_dir) / cfg.run.name / "config.resolved.yaml")

    # ---- data -----------------------------------------------------------------------
    rows = read_sequences_parquet(Path(cfg.data.dir) / "sequences.parquet", split="train")
    slots = build_question_slots(rows)
    rng = epoch_rng(cfg.run.seed, 0)
    batches = epoch_batches(rows, slots, cfg, 0, rng)
    stats = batch_stats(batches, rows)
    report = steps_report(stats["sequences_total"], cfg, n_batches=len(batches))
    steps_total = planned_optimizer_steps(len(batches), cfg.train.grad_accum) * cfg.train.epochs

    logger.event(
        "launch/data",
        {
            "questions": len(slots),
            "sequences_per_question": round(expected_sequences_per_question(slots, cfg), 3),
            "optimizer_steps": steps_total,
            "warmup_steps": report["warmup_steps"],
            **{k: round(v, 4) for k, v in stats.items()},
        },
    )
    if steps_total < MIN_OPTIMIZER_STEPS and cfg.train.max_steps is None and not args.allow_short_run:
        raise AssertionError(
            f"optimizer_steps = {steps_total} < {MIN_OPTIMIZER_STEPS} (§11.1). If this prints "
            "~106, the n_questions/grad_accum regression is back: sequences per question is "
            "min(4,k_c)+min(3,k_i) = 4.33, NOT 9.18. Cut n_questions, not grad_accum."
        )

    # ---- model ----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg)
    hidden_size = read_hidden_size(cfg.model.name)   # from config.json, never a doc (§13)
    backbone = load_backbone(cfg)
    model = FeynmanPRM(cfg, hidden_size, backbone=backbone, with_goal_head=False)
    model.pad_id = tokenizer.pad_token_id
    model.to(device)
    model.train()   # must precede the memory probe: eval mode disables gradient checkpointing

    trainable = assert_phase1_trainable(model, cfg)   # §14: exactly {LoRA, psi, phi}
    logger.event(
        "launch/model",
        {
            "hidden_size": hidden_size,
            "trainable_tensors": trainable,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "device": str(device),
            "attn_implementation": cfg.model.attn_implementation,
        },
    )

    optimizer = torch.optim.AdamW(
        param_groups(model, cfg),
        betas=tuple(cfg.train.betas),
        weight_decay=cfg.train.weight_decay,
        foreach=False,        # bug B9: the foreach path allocates a large fp32 transient
    )                         # torch AdamW, never FusedAdam (bug B8)
    scheduler = build_scheduler(optimizer, max(steps_total, 1), cfg)
    # LambdaLR's constructor already applies lr_lambda(0), which is 0.0 under warmup -- so this
    # is NOT the base LR. The B6 guard tracks the range over the whole run instead of comparing
    # this against the final value; see the block after the loop.
    lr_before = optimizer.param_groups[0]["lr"]
    lr_min_seen = lr_max_seen = lr_before

    # ---- the longest-batch memory probe (PLAN 4a) ------------------------------------
    probe_idx = longest_batch_index(batches, rows)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _, _, _, _, probe_out = run_micro_batch(
        model, rows, batches[probe_idx], cfg, device, probe_rng(cfg.run.seed)
    )
    probe_out.total.backward()
    optimizer.zero_grad(set_to_none=True)
    logger.event(
        "launch/memory_probe",
        {
            "batch_index": probe_idx,
            "sequences": len(batches[probe_idx]),
            "max_length": max(rows[i].length for i in batches[probe_idx]),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)
            if device.type == "cuda"
            else None,
            "loss": float(probe_out.total),
        },
    )

    # ---- train ----------------------------------------------------------------------
    step = 0
    checked_init = False
    stop = False
    for epoch in range(cfg.train.epochs):
        if epoch > 0:
            batches = epoch_batches(rows, slots, cfg, epoch, epoch_rng(cfg.run.seed, epoch))
        for micro, batch_rows in enumerate(batches):
            batch, goals, reps, matrices, out = run_micro_batch(
                model, rows, batch_rows, cfg, device, goal_rng(cfg.run.seed, epoch, micro),
                step=step,
            )
            if not torch.isfinite(out.total):
                raise RuntimeError(f"non-finite loss at epoch {epoch} micro {micro}: {out.info}")
            (out.total / cfg.train.grad_accum).backward()

            if not checked_init:
                expected = expected_init_values(
                    cfg,
                    matrices.n_rows,
                    out.info["backup/dist_mean"],
                    out.info["backup/delta_mean"],
                    out.info["step/delta_mean"],
                    out.info["backup/linear_branch_fraction"],
                )
                logger.event(
                    "launch/init_values",
                    {
                        "expected": {k: round(v, 4) for k, v in expected.items()},
                        "actual": {k: round(float(v), 4) for k, v in out.terms.items()},
                        "note": (
                            "L_NCE ~ log(R); L_step is softplus(m - Delta) at the MEASURED "
                            "Delta, not ln 5 -- Delta_{z+1} starts negative because psi_0 (the "
                            "prompt-only state) is atypical, so ~4 at init is expected and 1.61 "
                            "is the Delta = 0 fixture value only (§7.6); L_T at init is the "
                            "psi/phi representation gap delta ~ 10 riding the LINEX LINEAR "
                            "branch, so linear_branch_fraction ~ 1.0 and L_I ~ dist_mean ~ 11 "
                            "are both correct here (§7.4.3). L_NCE pinned at log(R) with "
                            "logit_std ~= 0 is bug B10a. L_good has NO predicted level (nan on "
                            "purpose) -- check the sandwich relu(good/delta_min - c) <= L_good "
                            "<= relu(good/delta_max - c), which runs OPPOSITE to L_step's, and "
                            "note a lower bound of exactly 0 is legitimate (§7.12)."
                        ),
                        "step_delta_mean": round(out.info["step/delta_mean"], 4),
                        # §7.12: c must be NEGATIVE. It is derived, but it is printed at
                        # launch because the sign is the one thing about this term that no
                        # downstream curve would reveal if it were wrong.
                        "good_margin": round(cfg.good_margin, 4),
                        "lambda_good": cfg.losses.lambda_good,
                        "good_delta_mean": round(out.info["good/delta_mean"], 4),
                        "good_above_target_fraction": round(
                            out.info["good/above_target_fraction"], 4
                        ),
                        "logit_std": round(out.info.get("nce/logit_std", float("nan")), 5),
                        "clip_t": round(cfg.clip_t, 4),
                        "linear_branch_fraction": round(
                            out.info["backup/linear_branch_fraction"], 4
                        ),
                    },
                )
                # §7.12/§18. relu is INCREASING in Delta, so this sandwich is the mirror of
                # L_step's and it is exact -- the only tolerance is fp rounding on the mean.
                # A lower bound of 0 is legitimate (every good step already at or below c);
                # a violation means L_good is not the relu of its own logged deltas, which is
                # a wrong `c`, a wrong sign on `c`, or a wrong scope mask.
                if out.info["good/terms"] > 0:
                    c = cfg.good_margin
                    lo = max(out.info["good/delta_min"] - c, 0.0)
                    hi = max(out.info["good/delta_max"] - c, 0.0)
                    good_now = float(out.terms["good"])
                    assert lo - 1e-4 <= good_now <= hi + 1e-4, (
                        f"L_good {good_now:.6f} outside relu sandwich [{lo:.6f}, {hi:.6f}] at "
                        f"c = {c:.6f} (§7.12). Check the sign of good_margin first."
                    )
                checked_init = True

            if (micro + 1) % cfg.train.grad_accum == 0:
                # TMD has no gradient clipping (optax.adam bare, tmd.py:341) and does not need
                # it: its psi/phi sit on a small MLP with no backbone underneath. Here the same
                # gradient also lands on LoRA inside a 1.5B model, and the LINEX is an exp() --
                # one bad micro-batch is enough. The PRE-clip norm is logged every optimizer
                # step so the clip stays auditable (§7.4.3): if train/grad_norm sits far above
                # train.grad_clip for the whole run, the clip has stopped being a guard and is
                # acting as an LR rescale -- raise it, do not leave it silently binding.
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.train.grad_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                lr_now = optimizer.param_groups[0]["lr"]
                lr_min_seen = min(lr_min_seen, lr_now)
                lr_max_seen = max(lr_max_seen, lr_now)

                if step % cfg.train.log_every == 0 or step == 1:
                    metrics = dict(out.info)
                    metrics["train/grad_norm"] = float(grad_norm)
                    metrics["train/grad_clipped"] = float(float(grad_norm) > cfg.train.grad_clip)
                    metrics.update(
                        batch_probes(
                            reps.psi, reps.phi, batch, goals, matrices, model.distance, cfg
                        )
                    )
                    # §9.4 / §16.10: DIAGNOSTIC ONLY, never reported without a decision. It is
                    # the direct test of the asymmetry claim that justifies a quasimetric over a
                    # cosine similarity, and nothing in L_NCE/L_I/L_T supervises the reverse
                    # distance -- so it is emergent, and it has to be watched from day one to be
                    # worth anything later.
                    metrics.update(asymmetry_score(reps.psi, batch, model.distance))
                    metrics["lr/backbone"] = optimizer.param_groups[0]["lr"]
                    metrics["lr/heads"] = optimizer.param_groups[-1]["lr"]
                    logger.log(step, metrics, console=True)
                if step % cfg.train.save_every == 0:
                    save_checkpoint(
                        Path(cfg.run.out_dir) / cfg.run.name / f"step{step}",
                        model, cfg, tokenizer, step=step,
                    )
                if cfg.train.max_steps is not None and step >= cfg.train.max_steps:
                    stop = True
                    break
        if stop:
            break

    # ---- the B6 guard, and the checkpoint it is not allowed to eat ---------------------
    # Read the LR over the WHOLE run, never start-vs-end. Two facts make start-vs-end fire on
    # every run that FINISHES: LambdaLR's constructor applies lr_lambda(0), which is 0.0 under
    # warmup, and a completed cosine ends at 0.5*(1+cos(pi)) = 0.0 exactly. So `lr_end ==
    # lr_before` is 0.0 == 0.0 -- it passed only on runs cut short by --max-steps, i.e. exactly
    # the ones whose checkpoints do not matter. It fired on 2026-07-27 after 971 good steps and
    # took the final checkpoint with it, because the save sat BELOW the raise. It now sits
    # above: a diagnostic must never destroy the artifact it is diagnosing.
    lr_stuck = (
        step > 0 and cfg.train.schedule != "constant" and lr_max_seen <= lr_min_seen
    )

    save_checkpoint(
        Path(cfg.run.out_dir) / cfg.run.name / "final", model, cfg, tokenizer, step=step
    )
    logger.event(
        "done",
        {"optimizer_steps": step, "lr_min": lr_min_seen, "lr_max": lr_max_seen},
    )
    logger.close()

    if lr_stuck:
        raise AssertionError(
            f"the LR never moved -- bug B6 (no scheduler) is back. It held {lr_max_seen:g} for "
            f"all {step} optimizer steps. The final checkpoint was written first and is intact."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
