#!/usr/bin/env python
"""Take a couple of real optimizer steps and check the plumbing end to end.

This is NOT a training run and NOT a substitute for the launch asserts in `train.py`. It is
the "does the algorithm actually do anything" check: gradients reach every trainable tensor,
the optimizer moves them, the checkpoint on disk contains them, and reloading gives back
bitwise the same numbers.

    python scripts/one_step_check.py                         # 3 optimizer steps, real batches
    python scripts/one_step_check.py --steps 1               # see the two notes below
    python scripts/one_step_check.py --memory-probe          # + the longest-batch probe
    python scripts/one_step_check.py --set sampling.sequences_per_micro_batch=8   # cheaper

**Why the default is 3 steps and not 1.** Two independent facts, and neither is a fault:

1. `build_scheduler` is a `LambdaLR` whose warmup lambda is `step / warmup`, and
   `LambdaLR.__init__` applies `lr_lambda(0) = 0.0` -- so the FIRST `optimizer.step()` of any
   run happens at LR exactly 0 and moves nothing. Same fact §14's B11 entry is about, seen
   from the other end.
2. **`lora_A`'s gradient is IDENTICALLY ZERO while its paired `lora_B` is still zero.** The
   adapter computes `y = B @ A @ x`, so `dL/dA = B^T (dL/dy) x^T`, which is `0` at `B = 0` --
   and PEFT initialises `B` to exactly zero, which is what makes the adapter a no-op at init.
   `A` cannot receive a gradient until `B` has taken a non-zero step.

So: step 1 moves nothing (LR 0); step 2 moves `lora_B` and the heads; step 3 is the first at
which `lora_A` has a non-zero gradient and moves. `--steps` below 3 reports the checks it
cannot reach as SKIP with the reason, never as a failure. 196 all-zero `lora_A` gradients
(28 layers x 7 projections) is fact 2, not a plumbing bug -- it is what a CORRECT LoRA looks
like on the first backward.

What it asserts, in order:

    1. the phase-1 trainable set is exactly {LoRA, psi, phi}                       §14
    2. the loss and every term is finite
    3. every trainable tensor gets a finite gradient, and a non-zero one everywhere the
       math allows one -- `psi.`/`phi.`/`lora_B` unconditionally, `lora_A` once `B != 0`.
       A zero grad on `psi.`/`phi.` is §14's LoRA trap 2 (heads un-frozen before PEFT
       wrapped them); a zero grad on `lora_B` means nothing reached the backbone at all
    4. the optimizer moved every trainable tensor, reported per group
    5. `lora_B` is non-zero -- it initialises to exactly zero, so this is the one check that
       cannot pass by accident
    6. `heads.pt` contains EVERY trainable non-LoRA tensor, and the adapter file contains
       EVERY trainable LoRA tensor -- §14's trap 3, the failure where the stock PEFT save
       path writes the adapter and silently drops the trained heads
    7. reloading `heads.pt` into a fresh model reproduces the tensors bitwise
    8. the resolved config and tokenizer are next to them

Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run without `pip install -e .`

import argparse

import torch

from feynman_prm.config import load_config
from feynman_prm.data.math_shepherd import read_sequences_parquet
from feynman_prm.data.sampler import (
    batch_stats,
    build_question_slots,
    epoch_batches,
    longest_batch_index,
)
from feynman_prm.model.backbone import (
    assert_phase1_trainable,
    classify_trainable,
    load_backbone,
    load_tokenizer,
    param_groups,
    read_hidden_size,
    trainable_parameter_names,
)
from feynman_prm.model.wrapper import FeynmanPRM
from feynman_prm.train import build_scheduler, run_micro_batch
from feynman_prm.utils.checkpoint import head_state_dict, load_heads
from feynman_prm.utils.checkpoint import save_checkpoint
from feynman_prm.utils.seeding import epoch_rng, goal_rng, probe_rng, seed_everything


class Checks:
    """A pass/fail ledger, so one failure does not hide the ones after it."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def record(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        print(f"  [{status:4}] {name}" + (f"  --  {detail}" if detail else ""), flush=True)

    def ok(self, name: str, detail: str = "") -> None:
        self.record("PASS", name, detail)

    def fail(self, name: str, detail: str = "") -> None:
        self.record("FAIL", name, detail)

    def skip(self, name: str, detail: str = "") -> None:
        self.record("SKIP", name, detail)

    def check(self, condition: bool, name: str, detail: str = "") -> bool:
        (self.ok if condition else self.fail)(name, detail)
        return bool(condition)

    @property
    def failures(self) -> list[tuple[str, str, str]]:
        return [r for r in self.rows if r[0] == "FAIL"]


def group_of(name: str) -> str:
    """`lora_A` and `lora_B` are separate groups on purpose -- they do NOT come alive at the
    same step, and lumping them together is what makes a correct LoRA look broken (see the
    header). Everything else is bucketed the way §14's trainability assert buckets it."""
    if "lora_A" in name:
        return "lora_A"
    if "lora_B" in name:
        return "lora_B"
    if name.startswith("psi.") or ".psi." in name:
        return "psi"
    if name.startswith("phi.") or ".phi." in name:
        return "phi"
    return "other"


def adapter_key(name: str) -> str:
    """Normalise a LoRA parameter name so the in-memory and on-disk forms compare.

    In memory:  backbone.base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight
    On disk:    base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight

    The prefixes differ by how many `model.` hops PEFT and the wrapper each add, so key on
    the suffix from `layers.` -- unique per tensor, and stable across PEFT versions.
    """
    stripped = name.replace(".default.", ".")
    idx = stripped.find("layers.")
    return stripped[idx:] if idx >= 0 else stripped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="one-step training plumbing check")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--set", action="append", default=[], help="key.path=value override")
    parser.add_argument("--steps", type=int, default=3,
                        help="optimizer steps to take (default 3: step 1 runs at LR 0, step 2 "
                             "moves lora_B and the heads, step 3 is the first that can move "
                             "lora_A -- see the module docstring)")
    parser.add_argument("--out", default=None,
                        help="checkpoint dir (default <run.out_dir>/one_step_check)")
    parser.add_argument("--memory-probe", action="store_true",
                        help="also run the longest batch of the epoch first (PLAN 4a)")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    args = parser.parse_args(argv)

    checks = Checks()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg.run.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out) if args.out else Path(cfg.run.out_dir) / "one_step_check"

    # ---- data ------------------------------------------------------------------------
    print("\n== data ==", flush=True)
    parquet = Path(cfg.data.dir) / "sequences.parquet"
    if not parquet.exists():
        print(f"{parquet} does not exist -- run `python scripts/prepare_data.py` first "
              f"(and re-run it whenever data.n_questions changes, §8.2)", file=sys.stderr)
        return 1
    rows = read_sequences_parquet(parquet, split="train")
    slots = build_question_slots(rows)
    batches = epoch_batches(rows, slots, cfg, 0, epoch_rng(cfg.run.seed, 0))
    stats = batch_stats(batches, rows)
    needed = args.steps * cfg.train.grad_accum
    print(f"  questions {len(slots)}  batches {len(batches)}  "
          f"{ {k: round(v, 3) for k, v in stats.items()} }", flush=True)
    if not checks.check(len(batches) >= needed, "enough micro-batches",
                        f"need {needed} (= {args.steps} steps x grad_accum {cfg.train.grad_accum}), "
                        f"have {len(batches)}"):
        return 1

    # ---- model -----------------------------------------------------------------------
    print("\n== model ==", flush=True)
    tokenizer = load_tokenizer(cfg)
    hidden_size = read_hidden_size(cfg.model.name)
    model = FeynmanPRM(cfg, hidden_size, backbone=load_backbone(cfg), with_goal_head=False)
    model.pad_id = tokenizer.pad_token_id
    model.to(device)
    model.train()

    try:
        buckets = assert_phase1_trainable(model, cfg)
        checks.ok("trainable set is exactly {LoRA, psi, phi} (§14)", str(buckets))
    except AssertionError as exc:
        checks.fail("trainable set is exactly {LoRA, psi, phi} (§14)", str(exc))

    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    n_params = sum(p.numel() for p in trainable.values())
    print(f"  device {device}  hidden_size {hidden_size}  "
          f"trainable tensors {len(trainable)}  params {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(
        param_groups(model, cfg),
        betas=tuple(cfg.train.betas),
        weight_decay=cfg.train.weight_decay,
        foreach=False,
    )
    scheduler = build_scheduler(optimizer, max(args.steps, 1), cfg)

    if args.memory_probe:
        print("\n== longest-batch memory probe (PLAN 4a) ==", flush=True)
        probe_idx = longest_batch_index(batches, rows)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        *_, probe_out = run_micro_batch(
            model, rows, batches[probe_idx], cfg, device, probe_rng(cfg.run.seed)
        )
        probe_out.total.backward()
        optimizer.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
        checks.check(torch.isfinite(probe_out.total).item(), "longest batch is finite",
                     f"batch {probe_idx}, {len(batches[probe_idx])} seqs, "
                     f"loss {float(probe_out.total):.4f}, peak {peak:.3f} GiB")

    # ---- the steps -------------------------------------------------------------------
    print(f"\n== {args.steps} optimizer step(s) ==", flush=True)
    before = {n: p.detach().clone() for n, p in trainable.items()}
    ever_grad = {n: False for n in trainable}       # over ALL steps, not just the first
    lrs_used: list[float] = []
    micro = 0
    for step in range(args.steps):
        for _ in range(cfg.train.grad_accum):
            _, _, _, _, out = run_micro_batch(
                model, rows, batches[micro], cfg, device, goal_rng(cfg.run.seed, 0, micro)
            )
            finite = torch.isfinite(out.total).item()
            terms = {k: round(float(v), 4) for k, v in out.terms.items()}
            checks.check(finite, f"micro {micro} loss finite",
                         f"total {float(out.total):.4f}  terms {terms}")
            if not finite:
                return 1
            (out.total / cfg.train.grad_accum).backward()
            micro += 1

        # Gradients, before the optimizer consumes them.
        for name, param in trainable.items():
            if param.grad is not None and param.grad.any():
                ever_grad[name] = True
        if step == 0:
            missing = sorted(n for n, p in trainable.items() if p.grad is None)
            nonfinite = sorted(n for n, p in trainable.items()
                               if p.grad is not None and not torch.isfinite(p.grad).all())
            checks.check(not missing, "every trainable tensor has a gradient",
                         f"{len(missing)} missing: {missing[:4]}" if missing else
                         f"{len(trainable)} tensors")
            checks.check(not nonfinite, "every gradient is finite",
                         f"{len(nonfinite)} non-finite: {nonfinite[:4]}" if nonfinite else "")

        grad_norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), cfg.train.grad_clip)
        lr = optimizer.param_groups[0]["lr"]
        lrs_used.append(float(lr))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"  step {step + 1}: lr_backbone {lr:.3e}  lr_heads "
              f"{optimizer.param_groups[-1]['lr']:.3e}  pre-clip grad_norm {float(grad_norm):.4f}",
              flush=True)

    # ---- gradients and movement, per group -------------------------------------------
    # `live_steps` is the number of optimizer steps that ran at a non-zero LR. It is what
    # decides which of these checks are even reachable:
    #     >= 1  -> lora_B and the heads can have moved
    #     >= 2  -> lora_A can have moved (it needs B != 0 during a backward first)
    # See the header. Below those counts the checks report SKIP, not FAIL.
    live_steps = sum(1 for lr in lrs_used if lr > 0.0)
    # The first live step is what moves B off zero, so the FIRST backward that can see B != 0
    # is the one after it -- which only happens if a second live step exists. Two live steps
    # is therefore the bar for both of lora_A's checks, and `b_alive` is NOT the right test:
    # at --steps 2, B moves on the last step and no backward ever sees it.
    lora_a_reachable = live_steps >= 2

    groups: dict[str, list[str]] = {}
    for name in trainable:
        groups.setdefault(group_of(name), []).append(name)

    print("\n== gradients ==", flush=True)
    for group in sorted(groups):
        names = groups[group]
        dead = sorted(n for n in names if not ever_grad[n])
        if group == "lora_A" and not lora_a_reachable:
            checks.skip(
                "lora_A gradients are non-zero",
                f"all {len(names)} are exactly zero, which is CORRECT: dL/dA = B^T.(dL/dy).x^T "
                f"= 0 while lora_B is still at its zero init. Needs --steps >= 3.",
            )
            continue
        checks.check(
            not dead,
            f"{group} gradients are non-zero",
            f"{len(dead)} never got one: {dead[:4]}"
            + ("  (§14's LoRA trap 2)" if group in ("psi", "phi") else "")
            if dead else f"{len(names)} tensors",
        )

    print("\n== weight movement ==", flush=True)
    needed_steps = {"lora_A": 2, "lora_B": 1, "psi": 1, "phi": 1, "other": 1}
    for group in sorted(groups):
        names = groups[group]
        deltas = [(trainable[n].detach() - before[n]).abs().max().item() for n in names]
        stuck = [n for n, d in zip(names, deltas) if d == 0.0]
        detail = f"{len(names)} tensors, max|delta| {max(deltas):.3e}"
        need = needed_steps.get(group, 1)
        if live_steps < need:
            reason = (f"only {live_steps} step(s) ran at a non-zero LR; this group needs "
                      f"{need} (step 1 is always at LR 0"
                      + (", and lora_A needs lora_B to move first" if group == "lora_A" else "")
                      + ")")
            checks.skip(f"{group} weights moved", f"{detail}  --  {reason}")
            continue
        checks.check(not stuck, f"{group} weights moved",
                     f"{len(stuck)} stuck: {stuck[:4]}" if stuck else detail)

    lora_b = {n: p for n, p in trainable.items() if "lora_B" in n}
    dead_b = sorted(n for n, p in lora_b.items() if not p.detach().any())
    if live_steps < 1:
        checks.skip("lora_B left its zero init", "every step ran at LR 0")
    else:
        checks.check(
            bool(lora_b) and not dead_b,
            "lora_B left its zero init",
            f"{len(dead_b)} of {len(lora_b)} still exactly zero: {dead_b[:4]}" if dead_b
            else f"all {len(lora_b)} non-zero -- gradient really reached the backbone",
        )

    checks.check(
        len(set(lrs_used)) > 1 or (lrs_used and lrs_used[-1] != 0.0),
        "the scheduler moved the LR (bug B6)",
        f"lrs {['%.3e' % v for v in lrs_used]}",
    )

    # ---- save ------------------------------------------------------------------------
    print(f"\n== checkpoint -> {out_dir} ==", flush=True)
    saved = save_checkpoint(out_dir, model, cfg, tokenizer, step=args.steps)
    for rel in ("heads.pt", "config.yaml", "adapter", "tokenizer"):
        path = saved / rel
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() \
            else (path.stat().st_size if path.exists() else 0)
        checks.check(path.exists(), f"{rel} written", f"{size / 2**20:.2f} MiB")

    heads_now = head_state_dict(model)
    trainable_names = trainable_parameter_names(model)
    lora_names = classify_trainable(trainable_names)["lora"]
    head_trainable = sorted(trainable_names - lora_names)
    dropped = [n for n in head_trainable if n not in heads_now]
    checks.check(
        not dropped,
        "heads.pt holds every trainable non-LoRA tensor (§14 trap 3)",
        f"{len(dropped)} dropped: {dropped[:6]}" if dropped
        else f"{len(head_trainable)} of {len(heads_now)} saved head tensors are trainable",
    )

    try:
        from peft import load_peft_weights

        on_disk = {adapter_key(k): v for k, v in load_peft_weights(str(saved / "adapter")).items()}
        missing_lora = [n for n in sorted(lora_names) if adapter_key(n) not in on_disk]
        checks.check(
            not missing_lora,
            "the adapter holds every trainable LoRA tensor",
            f"{len(missing_lora)} missing: {missing_lora[:4]}" if missing_lora
            else f"{len(lora_names)} tensors",
        )
        mismatched = [
            n for n in sorted(lora_names)
            if adapter_key(n) in on_disk
            and not torch.equal(on_disk[adapter_key(n)].cpu().float(),
                                trainable[n].detach().cpu().float())
        ]
        checks.check(
            not mismatched,
            "the adapter on disk equals the trained LoRA weights",
            f"{len(mismatched)} differ: {mismatched[:4]}" if mismatched else "bitwise",
        )
    except Exception as exc:                                  # noqa: BLE001 - report, don't crash
        checks.fail("the adapter on disk equals the trained LoRA weights", f"{type(exc).__name__}: {exc}")

    # ---- reload ----------------------------------------------------------------------
    print("\n== reload ==", flush=True)
    try:
        fresh = FeynmanPRM(cfg, hidden_size, backbone=None, with_goal_head=False)
        info = load_heads(fresh, saved)
        reloaded = fresh.state_dict()
        differing = [
            k for k, v in heads_now.items()
            if k in reloaded and not torch.equal(reloaded[k].cpu().float(), v.cpu().float())
        ]
        absent = [k for k in heads_now if k not in reloaded]
        checks.check(
            not differing and not absent,
            "reloaded heads are bitwise identical",
            f"{len(differing)} differ, {len(absent)} absent" if (differing or absent)
            else f"{len(heads_now)} tensors, step {info['step']}",
        )
    except Exception as exc:                                  # noqa: BLE001
        checks.fail("reloaded heads are bitwise identical", f"{type(exc).__name__}: {exc}")

    # ---- verdict ---------------------------------------------------------------------
    passed = sum(1 for r in checks.rows if r[0] == "PASS")
    skipped = sum(1 for r in checks.rows if r[0] == "SKIP")
    print(f"\n{'=' * 70}")
    if checks.failures:
        print(f"FAILED  --  {len(checks.failures)} of {len(checks.rows)} checks:")
        for _, name, detail in checks.failures:
            print(f"  - {name}  --  {detail}")
        return 1
    print(f"OK  --  {passed} checks passed"
          + (f", {skipped} skipped" if skipped else "")
          + f".  checkpoint at {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
