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
import dataclasses
import math
import warnings
from pathlib import Path

import torch

from .config import Config, load_config
from .data.cf_attach import CFContext, select_cf_examples_for_train
from .data.collate import collate
from .data.counterfactual import read_cf_glob
from .data.goals import sample_goals
from .data.math_shepherd import read_sequences_parquet
from .data.sequence_cache import question_ids
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
from .losses.good import good_bounds
from .losses.matrix import build_matrices
from .losses.total import expected_init_values, phase1_loss
from .model.backbone import (
    assert_phase1_trainable,
    load_backbone,
    load_backbone_resume,
    load_tokenizer,
    param_groups,
    read_hidden_size,
)
from .model.wrapper import FeynmanPRM
from .utils.checkpoint import load_config_from_checkpoint, load_heads, save_checkpoint
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


def run_micro_batch(model, rows, batch_rows, cfg: Config, device, rng, step: int = 0, cf_ctx=None):
    """One micro-batch: collate -> goals -> attach CF -> forward -> matrices -> loss.

    `cf_ctx` is the §7.5.3-(b) attachment context (`CFContext` or None). When present, the
    CF examples whose prefix is already in THIS batch are attached and their phi is computed
    off this batch's own `h_states` -- one forward, one backward, one optimizer step for the
    main losses and L_CF together, which is what "in the main batch only" means. With no
    context, or when nothing attaches, `cf` is None and (4) contributes an exact zero.
    """
    # Goals are drawn on the CPU batch (the sampler is numpy, and there are no DataLoader
    # workers to seed), then both move to the device together.
    batch_cpu = collate([rows[i] for i in batch_rows], pad_id=model.pad_id)
    goals = sample_goals(batch_cpu, cfg.discount, rng).to(device)
    # The join runs on the CPU batch: it reads `state_prefix_hash` and builds index tensors,
    # so doing it after `.to(device)` would move ints to the GPU only to read them back.
    attached = cf_ctx.attach(batch_cpu, rng) if cf_ctx is not None else None
    batch = batch_cpu.to(device)
    reps = model(batch)
    cf = None
    if attached is not None:
        attached = attached.to(device)
        cf = (model.cf_phi(reps.h_states, attached), attached.variant_example, attached.variant_kind)
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
        cf=cf,
    )
    if attached is not None:
        out.info.update(attached.info)
    return batch, goals, reps, matrices, out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Feynman-PRM phase 1")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--set", action="append", default=[], help="key.path=value override")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--allow-short-run", action="store_true",
                        help="skip the >=300 optimizer-step assert (smoke tests only)")
    parser.add_argument("--overwrite", action="store_true",
                        help="write into a run directory that already holds a checkpoint")
    parser.add_argument("--resume", default=None, metavar="CHECKPOINT_DIR",
                        help="continue a killed run from a stepN checkpoint: restores the "
                             "weights, replays the LR schedule to that step, and consumes only "
                             "the batches that step never saw. Optimizer moments restart at "
                             "zero -- see the block at the resume site for what that costs.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.set)
    if args.max_steps is not None:
        cfg = load_config(args.config, args.set + [f"train.max_steps={args.max_steps}"])
    seed_everything(cfg.run.seed)

    # **A probe writes to its own directory.** `--max-steps 20` runs the same code path and
    # ends with the same `save_checkpoint(.../final)`, so before 2026-08-04 the documented
    # workflow -- probe, read the launch blocks, then launch for real -- left a 20-step
    # `final/` sitting exactly where the real run wanted to write. Suffixing is better than
    # exempting: the probe's checkpoint stays on disk to be inspected, its wandb run stops
    # sharing a name with the real one, and the collision cannot happen in either direction.
    if cfg.train.max_steps is not None:
        cfg = dataclasses.replace(
            cfg, run=dataclasses.replace(cfg.run, name=f"{cfg.run.name}_probe")
        )
        print(f"[probe] max_steps={cfg.train.max_steps} -> writing to {cfg.run.name}/ "
              f"(disposable; the guard does not apply)", flush=True)

    # A run directory is not resumable and never has been -- a loss-set change invalidates the
    # checkpoint (§7.12), so every experiment gets its own `run.name`. What this guards is the
    # accident: launching with the PREVIOUS name still in the yaml silently overwrites the
    # checkpoint the last set of reported numbers came from, and the only symptom is that
    # `runs/<name>/final` now answers to different weights than the numbers quoted against it.
    # Nothing downstream can detect that, so it is checked here, before the GPU is touched.
    # Probes are exempt: they are disposable by construction and re-probing is routine.
    # Keyed on `heads.pt`, not `config.yaml`: the config is a sidecar that proves nothing was
    # trained, while `heads.pt` IS the artifact -- save_checkpoint refuses to write it empty
    # (§14: the stock PEFT path drops the heads silently), so its presence means real weights.
    run_dir = Path(cfg.run.out_dir) / cfg.run.name
    existing = sorted(
        p.name for p in run_dir.glob("*") if p.is_dir() and (p / "heads.pt").exists()
    )
    if existing and cfg.train.max_steps is None and not args.overwrite and not args.resume:
        raise SystemExit(
            f"{run_dir} already holds checkpoint(s): {', '.join(existing)}.\n"
            f"Change `run.name` in {args.config} (one directory per loss-set change), or pass "
            f"--overwrite if you really mean to discard them.\n"
            f"If these came from a --max-steps probe, they are disposable: rm -rf {run_dir}"
        )

    logger = RunLogger(
        cfg.run.out_dir, cfg.run.name, cfg.log.wandb, cfg.log.wandb_project, cfg.to_dict()
    )
    cfg.save(Path(cfg.run.out_dir) / cfg.run.name / "config.resolved.yaml")

    # ---- data -----------------------------------------------------------------------
    sequences_path = Path(cfg.data.dir) / "sequences.parquet"
    rows = read_sequences_parquet(sequences_path, split="train")
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

    # ---- resume: what is restored, and the one thing that is not ----------------------
    # A stepN checkpoint holds the LoRA adapter, the psi/phi heads and the step number. Three of
    # the four things a resume needs are therefore recoverable EXACTLY, and none of them are
    # approximations:
    #
    #   * the weights          -- straight off the checkpoint;
    #   * the LR               -- `build_scheduler` is a pure function of (step, steps_total),
    #                             and `steps_total` here is the FULL plan, not the remainder,
    #                             so replaying it `resume_step` times lands on exactly the LR
    #                             an uninterrupted run would be holding at that step;
    #   * the data position    -- `epoch_batches` is seeded off `cfg.run.seed` alone, so the
    #                             batch ORDER is identical run to run and "the batches step N
    #                             never saw" is just `batches[resume_step * grad_accum:]`.
    #                             `goal_rng` is keyed on (seed, epoch, micro), so skipping by
    #                             index leaves every surviving micro-batch bit-identical to what
    #                             it would have been.
    #
    # THE FOURTH IS THE ADAM MOMENTS, AND THEY ARE GONE. save_checkpoint does not write
    # optimizer state, so m and v restart at zero and the first steps after a resume are taken
    # on bias-corrected estimates built from one gradient. This is the honest cost of the
    # feature and it is not hidden: what bounds it is `betas = [0.9, 0.95]`, a second-moment
    # half-life of ~14 steps, so the estimates are re-converged within ~40 of the ~700 steps a
    # mid-run resume has left. A run with beta2 = 0.999 would NOT be safe to resume this way --
    # its ~700-step memory is the same order as the remainder, and the transient would be a
    # confound rather than a blip. If that ever changes, this comment is the thing to re-read.
    resume_step = 0
    if args.resume:
        resume_dir = Path(args.resume)
        if not (resume_dir / "heads.pt").exists():
            raise SystemExit(f"--resume {resume_dir}: no heads.pt there")
        resume_step = int(
            torch.load(resume_dir / "heads.pt", map_location="cpu", weights_only=False)["step"]
        )
        if resume_step <= 0:
            raise SystemExit(f"--resume {resume_dir}: checkpoint reports step {resume_step}")

        # The loss set is the whole reason run directories are not shared (§7.12), and a resume
        # is the one code path that reads weights trained under one config into a process
        # configured by another. Silently continuing a lambda_cf=2.0 checkpoint under
        # lambda_cf=1.0 would produce a curve that is a blend of two experiments and looks like
        # neither. Compared here, before the GPU is touched, and fatal.
        prev = load_config_from_checkpoint(resume_dir)
        drift = {
            k: (a, b)
            for k, (a, b) in {
                f.name: (getattr(prev.losses, f.name, None), getattr(cfg.losses, f.name))
                for f in dataclasses.fields(cfg.losses)
                if isinstance(getattr(cfg.losses, f.name), (int, float))
            }.items()
            if a != b
        }
        if drift:
            raise SystemExit(
                f"--resume {resume_dir} was trained under a different loss set: "
                + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in sorted(drift.items()))
                + ".\nResuming across a loss-set change blends two experiments into one curve. "
                "Launch a fresh run.name instead."
            )
        if prev.run.seed != cfg.run.seed:
            raise SystemExit(
                f"--resume {resume_dir}: seed {prev.run.seed} -> {cfg.run.seed}. The batch order "
                f"is a function of the seed, so the skip would drop the wrong batches."
            )

    # ---- model ----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg)
    hidden_size = read_hidden_size(cfg.model.name)   # from config.json, never a doc (§13)
    backbone = load_backbone_resume(cfg, resume_dir / "adapter") if args.resume \
        else load_backbone(cfg)
    model = FeynmanPRM(cfg, hidden_size, backbone=backbone, with_goal_head=False)
    model.pad_id = tokenizer.pad_token_id
    if args.resume:
        # The adapter came back with the backbone; the heads are the other half and they are
        # the half §14's LoRA trap loses. `load_heads` raises if heads.pt carries no head
        # parameters, so a resume cannot quietly restart psi/phi from random init under a
        # trained backbone -- which would look like a loss spike and read as a bad resume.
        load_heads(model, resume_dir)
    model.to(device)
    model.train()   # must precede the memory probe: eval mode disables gradient checkpointing

    # ---- (4) L_CF's data, attached to the main batch (§7.5.3-(b)) ---------------------
    # Built after the tokenizer and before the probe, so the memory probe measures a batch
    # that CARRIES CF variants. A probe that skipped them would under-measure peak VRAM by
    # exactly the tensor this change adds.
    cf_ctx = None
    if cfg.data.cf_glob and cfg.data.cf_max_per_batch > 0:
        cf_examples = read_cf_glob(cfg.data.cf_glob)
        # §8.2: a CF example on a VAL question would be silent leakage into phase 1 -- the one
        # failure this path could cause that no curve would show -- so it is asserted rather
        # than assumed. It needs the VAL qids to say that, not just the train ones: a question
        # whose every trajectory was dropped at tokenisation (§4.6) is in the selection and
        # absent from the rows, and charging that to leakage is what stopped the 2026-08-16
        # launch over 4 of 27,114 examples. `select_cf_examples_for_train` separates them.
        kept, cf_split_info = select_cf_examples_for_train(
            cf_examples,
            {r.qid for r in rows},
            question_ids(sequences_path, split="val"),
        )
        if cf_split_info["examples_dropped_question_absent"]:
            print(
                f"[cf] {cf_split_info['examples_dropped_question_absent']} of "
                f"{len(cf_examples)} CF examples sit on "
                f"{cf_split_info['questions_absent']} question(s) that are in NO split of "
                f"sequences.parquet -- every trajectory of those questions was dropped at "
                f"tokenisation (§4.6), so there is no prefix to attach to. Dropped, and in "
                f"the launch/cf_data event.",
                flush=True,
            )
        # A parquet written before 2026-08-15 has no `prefix_hash` column. `cf_attach` never
        # matches on a missing hash, so it degrades to "nothing attaches" -- which is exactly
        # right while lambda_cf is 0 and exactly WRONG once it is not: (4) would then be
        # weighted, fed, logged and training on nothing, with every other curve healthy. That
        # is §14's whole family (B11, B12), so at a nonzero weight it is a HARD ERROR at
        # launch rather than a number to notice in `cf/attach_rate` three hours in.
        if cfg.losses.lambda_cf > 0:
            n_hashed = sum(r.prefix_hash is not None for r in rows)
            if n_hashed == 0:
                raise AssertionError(
                    f"losses.lambda_cf = {cfg.losses.lambda_cf} but not one of {len(rows)} "
                    f"rows in {cfg.data.dir}/sequences.parquet carries a `prefix_hash` "
                    f"column, so NOTHING would attach and (4) would train on nothing while "
                    f"every other curve looked healthy (§7.5.13). Re-run "
                    f"`python scripts/prepare_data.py`, or set losses.lambda_cf=0.0."
                )
            if n_hashed < len(rows):
                raise AssertionError(
                    f"{len(rows) - n_hashed} of {len(rows)} rows carry no `prefix_hash` -- "
                    f"the parquet is a MIX of pre- and post-2026-08-15 writes, which silently "
                    f"biases which CF examples can attach. Re-run scripts/prepare_data.py."
                )
        cf_ctx = CFContext(kept, tokenizer, tokenizer.pad_token_id, cfg.data.cf_max_per_batch)
        logger.event(
            "launch/cf_data",
            {
                "examples": len(kept),
                "distinct_prefixes": len(cf_ctx.index),
                "max_per_batch": cfg.data.cf_max_per_batch,
                "lambda_cf": cfg.losses.lambda_cf,
                # The §7.5.13 ceiling is 100% and the seeded draw measures 91.6%, so this is
                # the number `cf/attach_rate` is read against. Far below it means the join is
                # broken, not that the data is thin.
                "rows_with_prefix_hash": sum(r.prefix_hash is not None for r in rows),
                "rows": len(rows),
                # The split audit: leakage is fatal above, so what lands here is the count of
                # examples whose question has no rows at all. Logged rather than left silent
                # because the drop is relied on (§14).
                **cf_split_info,
            },
        )

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
    if resume_step:
        # `steps_total` is the FULL plan and is unchanged by resuming, so the cosine this
        # replays is the same curve the original run was riding -- stepping it `resume_step`
        # times lands on the LR an uninterrupted run would hold at that step, warmup included.
        # Advanced by calling scheduler.step() rather than by setting last_epoch, because
        # LambdaLR only writes the LR into the param groups from inside step().
        # torch warns when scheduler.step() precedes the first optimizer.step(). That heuristic
        # is about the ORDER inside a training loop and this is a deliberate fast-forward before
        # the loop starts, so it fires `resume_step` times and says nothing true. Suppressed
        # here and nowhere wider: `tests/test_resume.py` pins the LR this produces.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*lr_scheduler\.step\(\).*")
            for _ in range(resume_step):
                scheduler.step()

    lr_before = optimizer.param_groups[0]["lr"]
    lr_min_seen = lr_max_seen = lr_before

    # ---- the longest-batch memory probe (PLAN 4a) ------------------------------------
    probe_idx = longest_batch_index(batches, rows)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _, _, _, _, probe_out = run_micro_batch(
        model, rows, batches[probe_idx], cfg, device, probe_rng(cfg.run.seed), cf_ctx=cf_ctx
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
    step = resume_step
    checked_init = False
    stop = False

    # Every optimizer step consumed exactly `grad_accum` micro-batches, so the boundary is
    # exact: batches[:skip_micro] are the ones the checkpoint has already been trained on.
    skip_micro = resume_step * cfg.train.grad_accum
    if resume_step:
        if skip_micro >= len(batches) * cfg.train.epochs:
            raise SystemExit(
                f"--resume at step {resume_step} consumes {skip_micro} micro-batches but the "
                f"plan only has {len(batches) * cfg.train.epochs}: nothing left to train."
            )
        logger.event(
            "train/resume",
            {
                "checkpoint": str(resume_dir),
                "resume_step": resume_step,
                "steps_total": steps_total,
                "steps_remaining": steps_total - resume_step,
                "micro_batches_skipped": skip_micro,
                "micro_batches_remaining": len(batches) - skip_micro,
                "lr_at_resume": lr_before,
                # Read this against the ~14-step beta2 half-life, not as an error bar: the
                # moments are the one piece of state a stepN checkpoint cannot return.
                "optimizer_moments": "RESET TO ZERO (not saved in checkpoints)",
                "beta2": cfg.train.betas[1],
            },
        )
        print(
            f"[resume] {resume_dir} @ step {resume_step}/{steps_total} | "
            f"lr {lr_before:.3e} | skipping {skip_micro} micro-batches | "
            f"{steps_total - resume_step} optimizer steps left",
            flush=True,
        )

    for epoch in range(cfg.train.epochs):
        if epoch > 0:
            batches = epoch_batches(rows, slots, cfg, epoch, epoch_rng(cfg.run.seed, epoch))
        for micro, batch_rows in enumerate(batches):
            # Skipped by INDEX, never by consuming and discarding: `goal_rng` is keyed on
            # (seed, epoch, micro), so the surviving micro-batches are bit-identical to the
            # ones an uninterrupted run would have seen here.
            if epoch == 0 and micro < skip_micro:
                continue
            batch, goals, reps, matrices, out = run_micro_batch(
                model, rows, batch_rows, cfg, device, goal_rng(cfg.run.seed, epoch, micro),
                step=step, cf_ctx=cf_ctx,
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
                    # (7) L_term's chance level is a property of the batch's ragged
                    # min(4,k_c)/min(3,k_i) counts, not a constant (§7.13).
                    out.info["term/chance"],
                )
                logger.event(
                    "launch/init_values",
                    {
                        # On a resume these are TRAINED values, not init values: the §18
                        # comparison against `expected` does not apply and the event says so.
                        "resumed_from_step": resume_step or None,
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
                            "purpose) -- check the sandwich f(good/delta_min - c) <= L_good "
                            "<= f(good/delta_max - c) in the CONFIGURED form (good_form "
                            "below), which runs OPPOSITE to L_step's, and note a lower bound "
                            "of exactly 0 is legitimate (§7.12)."
                        ),
                        "step_delta_mean": round(out.info["step/delta_mean"], 4),
                        # §7.12: c must be NEGATIVE. It is derived, but it is printed at
                        # launch because the sign is the one thing about this term that no
                        # downstream curve would reveal if it were wrong.
                        "good_margin": round(cfg.good_margin, 4),
                        "good_form": cfg.losses.good_loss.form,
                        "lambda_good": cfg.losses.lambda_good,
                        # tau_NCE: 22.63 = sqrt(512) is TMD's own scaling (tmd.py:92) and
                        # reverses §7.2's divergence. It divides logit_std by the same factor,
                        # so read `logit_std * nce_temperature` against B10a, never logit_std.
                        "nce_temperature": cfg.losses.nce_temperature,
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
                # §7.12/§18. Every L_good form is INCREASING in Delta, so this sandwich is the
                # mirror of L_step's and it is exact -- the only tolerance is fp rounding on
                # the mean. A lower bound of 0 is legitimate (every good step already at or
                # below c); a violation means L_good is not `f` of its own logged deltas,
                # which is a wrong `c`, a wrong sign on `c`, or a wrong scope mask.
                # `good_bounds` applies the CONFIGURED form -- a relu-shaped bound here would
                # fire on a correct relu_squared run, which is the B11/B12 failure mode.
                if out.info["good/terms"] > 0:
                    c = cfg.good_margin
                    lo, hi = good_bounds(
                        out.info["good/delta_min"], out.info["good/delta_max"], cfg
                    )
                    good_now = float(out.terms["good"])
                    assert lo - 1e-4 <= good_now <= hi + 1e-4, (
                        f"L_good {good_now:.6f} outside the "
                        f"{cfg.losses.good_loss.form} sandwich [{lo:.6f}, {hi:.6f}] at "
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
