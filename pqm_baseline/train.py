"""PQM's loss and head, trained under Feynman-PRM's EXACT conditions.

Same parquet, same 34,650-question selection, same seed, same batch stream, same optimizer
steps, same LoRA config, same schedule. `feynman_prm/train.py`'s launch discipline is
mirrored line for line and for the same reasons -- every one of them has a §14 or §11.1 entry
behind it:

    1. resolve BOTH configs strictly                              (bug B4)
    2. build the epoch's batches through the IDENTICAL call and ASSERT the step count
       (§11.1: the old arithmetic silently produced a 106-step run and it would have
       completed). `launch/data` carries the same keys as a Feynman run's, so the two events
       diff line for line -- that is the matched-data proof, and it is free
    3. ASSERT the trainable set is exactly {LoRA, value_head}     (§14)
    4. build the cosine schedule and ASSERT it steps              (bug B6)
    5. run the LONGEST batch of the epoch first, as a memory probe (PLAN 4a)
    6. check the initialisation value against its CLOSED FORM on the first micro-batch (§18)

There is no goal sampling, no counterfactual attachment and no phase 2: PQM scores directly
from this checkpoint.

Run it under tmux. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` first (§13) --
`pqm_baseline/train.sh` does both.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch

from feynman_prm.config import Config, load_config
from feynman_prm.data.collate import collate
from feynman_prm.data.math_shepherd import read_sequences_parquet
from feynman_prm.data.sampler import (
    batch_stats,
    build_question_slots,
    epoch_batches,
    expected_sequences_per_question,
    longest_batch_index,
    planned_optimizer_steps,
    steps_report,
)
from feynman_prm.diagnostics.logging import RunLogger
from feynman_prm.model.backbone import (
    load_backbone,
    load_tokenizer,
    param_groups,
    read_hidden_size,
)
from feynman_prm.train import MIN_OPTIMIZER_STEPS, build_scheduler
from feynman_prm.utils.checkpoint import save_checkpoint
from feynman_prm.utils.seeding import epoch_rng, seed_everything

from .config import PQM_YAML, PQMConfig, load_pqm_config, split_overrides
from .loss import (
    build_padded,
    loss_at_zero_rewards,
    pqm_diagnostics,
    pqm_ranking_loss,
    summarise_labels,
)
from .model import VALUE_HEAD_PREFIXES, PQMValueModel, assert_pqm_trainable

INIT_LOSS_TOLERANCE = 1e-4


def save_pqm_checkpoint(out_dir, model, cfg: Config, pqm: PQMConfig, tokenizer=None, step=None):
    """`save_checkpoint` with the value head named as the head, plus PQM's own knobs.

    Going through the shared function rather than around it is deliberate: §14's LoRA trap 3
    is that the stock PEFT save writes the adapter and silently drops the trained head, and a
    second save path here would be a second place for that to happen. `pqm.yaml` sits beside
    `config.yaml` so `eval_processbench.py` can reconstruct BOTH halves of the run.
    """
    path = save_checkpoint(
        out_dir, model, cfg, tokenizer=tokenizer, step=step, prefixes=VALUE_HEAD_PREFIXES
    )
    pqm.save(path / "pqm.yaml")
    return path


def run_micro_batch(model, rows, batch_rows, pqm: PQMConfig, device):
    """One micro-batch: collate -> forward -> flat-to-padded -> PQM's ranking loss.

    No goals, no CF attachment, no distance matrices. The batch COMPOSITION is Feynman's
    (the question-grouped sampler, unchanged) even though PQM's loss is per-sequence and
    batch composition does not enter its objective at all -- only gradient noise. That is the
    point: an identical batch stream is one fewer difference between the two rows.
    """
    batch = collate([rows[i] for i in batch_rows], pad_id=model.pad_id).to(device)
    rewards = model(batch)                                   # (S,) fp32
    rewards_pad, labels_pad, has_neg = build_padded(batch, rewards)
    loss = pqm_ranking_loss(rewards_pad, labels_pad, pqm.zeta, has_neg)
    return batch, rewards, rewards_pad, labels_pad, has_neg, loss


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PQM baseline, matched to Feynman-PRM")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--pqm-config", default=str(PQM_YAML))
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="key.path=value override. `pqm.*` goes to pqm.yaml, everything else to "
             "config/default.yaml",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--allow-short-run", action="store_true",
                        help="skip the >=300 optimizer-step assert (smoke tests only)")
    parser.add_argument("--overwrite", action="store_true",
                        help="write into a run directory that already holds a checkpoint")
    args = parser.parse_args(argv)

    feynman_overrides, pqm_overrides = split_overrides(args.set)
    if args.max_steps is not None:
        feynman_overrides = feynman_overrides + [f"train.max_steps={args.max_steps}"]
    cfg = load_config(args.config, feynman_overrides)
    pqm = load_pqm_config(args.pqm_config, pqm_overrides)
    seed_everything(cfg.run.seed)

    # **The naming trap, checked before the GPU is touched.** `losses.zeta` is Feynman's (3)
    # L_T backup weight -- 0.05 TMD-faithful, 0.1 in the shipped runs -- and PQM's ζ is a
    # reward offset of 4 that lives in `pqm.yaml`. `losses.zeta` is not read by anything in
    # this file, so `--set losses.zeta=4` would be silently inert, which is old bug B4's
    # shape. A value at PQM's scale can only be that mistake; the guard cannot fire on any
    # legitimate Feynman setting.
    if cfg.losses.zeta >= 1.0:
        raise SystemExit(
            f"losses.zeta = {cfg.losses.zeta} is Feynman's (3) L_T BACKUP WEIGHT (0.05/0.1), "
            f"not PQM's zeta. Nothing in pqm_baseline/ reads it. Did you mean "
            f"`--set pqm.zeta={cfg.losses.zeta:g}`?"
        )

    # A probe writes to its own directory (§14 B13): `--max-steps 20` runs the identical code
    # path and ends with the identical `save_checkpoint(.../final)`, so without the suffix the
    # documented workflow -- probe, read the launch blocks, then launch for real -- leaves a
    # 20-step `final/` exactly where the real run wants to write.
    if cfg.train.max_steps is not None:
        cfg = dataclasses.replace(
            cfg, run=dataclasses.replace(cfg.run, name=f"{cfg.run.name}_probe")
        )
        print(f"[probe] max_steps={cfg.train.max_steps} -> writing to {cfg.run.name}/ "
              f"(disposable; the guard does not apply)", flush=True)

    # §14 B14: a run directory holding a `heads.pt` is not overwritten without --overwrite.
    # Keyed on `heads.pt`, not `config.yaml`: the config is a sidecar that proves nothing was
    # trained, while `heads.pt` IS the artifact.
    run_dir = Path(cfg.run.out_dir) / cfg.run.name
    existing = sorted(
        p.name for p in run_dir.glob("*") if p.is_dir() and (p / "heads.pt").exists()
    )
    if existing and cfg.train.max_steps is None and not args.overwrite:
        raise SystemExit(
            f"{run_dir} already holds checkpoint(s): {', '.join(existing)}.\n"
            f"Pass a different --set run.name=..., or --overwrite if you really mean to "
            f"discard them.\nIf these came from a --max-steps probe, they are disposable: "
            f"rm -rf {run_dir}"
        )

    logger = RunLogger(
        cfg.run.out_dir, cfg.run.name, cfg.log.wandb, cfg.log.wandb_project,
        {**cfg.to_dict(), "pqm": pqm.to_dict()},
    )
    cfg.save(run_dir / "config.resolved.yaml")
    pqm.save(run_dir / "pqm.resolved.yaml")
    logger.event(
        "launch/config",
        {
            "pqm/zeta": pqm.zeta,
            "pqm/loss_type": pqm.loss_type,
            "pqm/head_init": pqm.head_init,
            "pqm/head_dropout": pqm.head_dropout,
            "pqm/label_source": pqm.label_source,
            "pqm/natural_tau_reward": pqm.natural_tau_reward,
            "pqm/natural_tau_delta": pqm.natural_tau_delta,
            # Logged side by side ON PURPOSE: these two are different quantities with the
            # same Greek letter, in different files (see the guard above).
            "feynman/losses.zeta": cfg.losses.zeta,
            "note": "pqm.zeta is PQM's reward offset; losses.zeta is Feynman's (3) L_T "
                    "backup weight. Unrelated.",
        },
    )

    # ---- data. THE IDENTICAL CALL -----------------------------------------------------
    # `epoch_batches` reads only `sampling.*` and the row table, so with the same parquet and
    # the same seed this is byte-identical to a Feynman run's batch stream. `launch/data`
    # below carries the same keys, so the two events diff line for line.
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

    # ---- model ------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg)
    hidden_size = read_hidden_size(cfg.model.name)   # from config.json, never a doc (§13)
    backbone = load_backbone(cfg)                    # VERBATIM: identical LoRA + checkpointing
    model = PQMValueModel(cfg, pqm, hidden_size, backbone=backbone)
    model.pad_id = tokenizer.pad_token_id
    model.to(device)
    model.train()   # must precede the memory probe: eval mode disables gradient checkpointing

    trainable = assert_pqm_trainable(model)          # §14: exactly {LoRA, value_head}
    logger.event(
        "launch/model",
        {
            "hidden_size": hidden_size,
            "trainable_tensors": trainable,           # expect {lora: 392, value_head: 2}
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "device": str(device),
            "attn_implementation": cfg.model.attn_implementation,
            "head": "Dropout -> Linear(hidden_size, 1), fp32 "
                    "(Process_Q_Model/value_model.py:22-59)",
        },
    )

    optimizer = torch.optim.AdamW(
        param_groups(model, cfg),                     # LoRA 9e-6, the fresh head 3e-4 (§11)
        betas=tuple(cfg.train.betas),
        weight_decay=cfg.train.weight_decay,
        foreach=False,        # bug B9: the foreach path allocates a large fp32 transient
    )                         # torch AdamW, never FusedAdam (bug B8)
    scheduler = build_scheduler(optimizer, max(steps_total, 1), cfg)
    lr_before = optimizer.param_groups[0]["lr"]
    lr_min_seen = lr_max_seen = lr_before

    # ---- the longest-batch memory probe (PLAN 4a) --------------------------------------
    # Expect LESS than Feynman's ~11.5 GiB: no psi/phi MLPs, no R x C distance matrix, no CF
    # variants. If it is not less, something is holding a tensor this loss does not need.
    probe_idx = longest_batch_index(batches, rows)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _, _, _, _, _, probe_loss = run_micro_batch(model, rows, batches[probe_idx], pqm, device)
    probe_loss.backward()
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
            "loss": float(probe_loss),
        },
    )

    # ---- train -------------------------------------------------------------------------
    step = 0
    checked_init = False
    stop = False
    for epoch in range(cfg.train.epochs):
        if epoch > 0:
            batches = epoch_batches(rows, slots, cfg, epoch, epoch_rng(cfg.run.seed, epoch))
        for micro, batch_rows in enumerate(batches):
            batch, rewards, rewards_pad, labels_pad, has_neg, loss = run_micro_batch(
                model, rows, batch_rows, pqm, device
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch {epoch} micro {micro}: the ranking loss "
                    f"evaluates exp(r + zeta), and fp32 exp overflows above r ~= 84. "
                    f"reward max {float(rewards.max()):.3f}, min {float(rewards.min()):.3f}"
                )
            (loss / cfg.train.grad_accum).backward()

            if not checked_init:
                # §18. With `head_init: zero` every reward is EXACTLY 0 here, so the loss has
                # a closed form in (n_pos, n_neg, has_neg) per trajectory and this is an
                # ASSERT, not an eyeball -- §7.4.3/§7.6.7's lesson is that an assumed init
                # value is how two regressions got through.
                expected = loss_at_zero_rewards(labels_pad, pqm.zeta)
                actual = float(loss)
                logger.event(
                    "launch/init_values",
                    {
                        "pqm/loss": round(actual, 6),
                        "pqm/loss_at_zero_rewards": round(expected, 6),
                        "abs_error": abs(actual - expected),
                        "head_init": pqm.head_init,
                        "reward_min": float(rewards.min()),
                        "reward_max": float(rewards.max()),
                        "reward_std": float(rewards.std()),
                        **summarise_labels(labels_pad),
                        "note": (
                            "At head_init=zero the loss IS the closed form and reward_std is "
                            "exactly 0; the assert below is exact. At head_init=default it is "
                            "only the chance anchor -- read reward_max instead, because the "
                            "loss evaluates exp(r + zeta) and fp32 exp overflows above "
                            "r ~= 84 on Qwen's massive-activation channels."
                        ),
                    },
                )
                if pqm.head_init == "zero":
                    assert abs(actual - expected) < INIT_LOSS_TOLERANCE, (
                        f"init loss {actual:.6f} != the closed form {expected:.6f} at "
                        f"zeta={pqm.zeta} (§18). Either the head is not zero-initialised, or "
                        f"the flat -> padded mapping is wrong -- check `row_dst` and "
                        f"`row_step - 1` before anything else."
                    )
                    assert float(rewards.std()) == 0.0, (
                        f"head_init=zero but reward_std = {float(rewards.std()):.3e}. The "
                        f"value head is not zero, so the closed form above is not the anchor "
                        f"it claims to be."
                    )
                checked_init = True

            if (micro + 1) % cfg.train.grad_accum == 0:
                # The PRE-clip norm is logged every optimizer step so the clip stays auditable
                # (§7.4.3): if train/grad_norm sits far above train.grad_clip for the whole
                # run, the clip has stopped being a guard and is acting as an LR rescale.
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
                    metrics = pqm_diagnostics(
                        batch, rewards, rewards_pad, labels_pad, has_neg, loss, pqm.zeta
                    )
                    metrics["loss/total"] = float(loss)
                    metrics["train/grad_norm"] = float(grad_norm)
                    metrics["train/grad_clipped"] = float(float(grad_norm) > cfg.train.grad_clip)
                    metrics["lr/backbone"] = optimizer.param_groups[0]["lr"]
                    metrics["lr/heads"] = optimizer.param_groups[-1]["lr"]
                    logger.log(step, metrics, console=True)
                if step % cfg.train.save_every == 0:
                    save_pqm_checkpoint(
                        run_dir / f"step{step}", model, cfg, pqm, tokenizer, step=step
                    )
                if cfg.train.max_steps is not None and step >= cfg.train.max_steps:
                    stop = True
                    break
        if stop:
            break

    # ---- the B6 guard, and the checkpoint it is not allowed to eat ----------------------
    # Read the LR over the WHOLE run, never start-vs-end: LambdaLR's constructor applies
    # lr_lambda(0) = 0.0 under warmup and a completed cosine ends at exactly 0.0, so
    # start-vs-end passes only on runs cut short by --max-steps. The save sits ABOVE the
    # raise: a diagnostic must never destroy the artifact it is diagnosing.
    lr_stuck = step > 0 and cfg.train.schedule != "constant" and lr_max_seen <= lr_min_seen

    save_pqm_checkpoint(run_dir / "final", model, cfg, pqm, tokenizer, step=step)
    logger.event("done", {"optimizer_steps": step, "lr_min": lr_min_seen, "lr_max": lr_max_seen})
    logger.close()

    if lr_stuck:
        raise AssertionError(
            f"the LR never moved -- bug B6 (no scheduler) is back. It held {lr_max_seen:g} for "
            f"all {step} optimizer steps. The final checkpoint was written first and is intact."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
