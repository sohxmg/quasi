"""§9.7.7 -- the cross-question goal swap. World A vs World B, in one eval run, no training.

**The question.** §9.7.3 measured that swapping the goal head's predicted goal for the
terminal of the GOLD REFERENCE SOLUTION moves the within-solution rank by +0.0035 (0.2 sigma)
on `gsm8k`. So a better goal head cannot help. But both goals tested are goals **of the same
question**, so two worlds survive and nothing to date separates them:

    (A)  d(psi, g) depends on g only through a term that cancels in Delta. The goal channel
         is dead outright -- root cause D (§1.2). An ARCHITECTURE problem.
    (B)  the goal places you in the right QUESTION but carries no WITHIN-question resolution.
         A TRAINING problem: the loss set never had to resolve within a question.

**The test.** Score each sample with ANOTHER QUESTION'S goal.

    rank unchanged (~0.39, r high vs the base path)  ->  (A). The goal is ignored entirely.
    rank -> 0.5    (signal destroyed)                ->  (B). The goal carries the question,
                                                             and within-question resolution is
                                                             what the next run must target.

**Why the rank and not F1.** The rank is paired within one solution, so per-solution scale,
per-solution offset, length and tau all cancel exactly and the null is exactly 0.500 (§9.7.1).
tau never enters, so this run needs no calibration and borrows none.

**Two passes, and pass A is a self-check.** Pass A runs the reported goal-head path through a
pass-through `goal_fn` that also records each sample's goal; its Deltas must reproduce the
reported `deltas.npz` bit for bit (`--compare`). Pass B replays the identical batching with a
seeded derangement of those same goal vectors. The ONLY thing that differs between the two
passes is which question's goal each sample is scored against.

NOTHING HERE IS A REPORTED RESULT (§9.3). It is a diagnostic on an existing checkpoint.

    python scripts/cross_question_goal.py --checkpoint runs/phase1/phase2/final
    python scripts/error_rank.py runs/phase1/phase2/final/deltas_crossq.npz

For a ~5 minute first read, `--subsets gsm8k` alone is enough to see which world this is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from feynman_prm.eval.processbench import (
    Sample,
    load_processbench,
    pack_deltas,
    score_samples,
)
from feynman_prm.model.backbone import (
    load_backbone_with_adapter,
    load_tokenizer,
    read_hidden_size,
)
from feynman_prm.model.wrapper import FeynmanPRM
from feynman_prm.utils.checkpoint import load_config_from_checkpoint, load_heads


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation with no fixed point, so no sample is ever scored against its own goal.

    Rejection-sampled; the probability a random permutation is a derangement tends to 1/e, so
    this terminates in a handful of draws at any n we run.
    """
    if n < 2:
        raise ValueError("a cross-question swap needs at least 2 samples")
    while True:
        perm = rng.permutation(n)
        if not (perm == np.arange(n)).any():
            return perm


def rank(delta: np.ndarray, z: int) -> float | None:
    """§9.7.1's statistic, duplicated from `error_rank.py` so this script prints a verdict
    without a second invocation. `error_rank.py` remains the authority."""
    T = len(delta)
    if T < 2 or not (0 <= z < T):
        return None
    others = np.delete(delta, z)
    return float(((others > delta[z]).sum() + 0.5 * (others == delta[z]).sum()) / (T - 1))


def ranks(deltas: list[list[float]], labels: np.ndarray) -> dict[int, float]:
    out: dict[int, float] = {}
    for i, (d, z) in enumerate(zip(deltas, labels)):
        if int(z) < 0:
            continue
        r = rank(np.asarray(d, dtype=np.float64), int(z))
        if r is not None:
            out[i] = r
    return out


def verdict(base: dict[int, float], swapped: dict[int, float], subset: str) -> None:
    common = sorted(set(base) & set(swapped))
    if len(common) < 10:
        print(f"\n--- {subset}: too few paired samples ({len(common)}), skipping")
        return
    a = np.array([base[i] for i in common])
    b = np.array([swapped[i] for i in common])
    diff = b - a
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    r = float(np.corrcoef(a, b)[0, 1])
    # Distance from chance is the effect size that matters: the base path sits at 0.34-0.42
    # against a null of exactly 0.500, and the question is how much of that survives the swap.
    kept = (0.5 - b.mean()) / (0.5 - a.mean()) if abs(0.5 - a.mean()) > 1e-9 else float("nan")

    print(f"\n--- {subset}: own goal vs ANOTHER QUESTION'S goal, PAIRED on {len(common)}")
    print(f"    own goal {a.mean():.4f}   cross-question {b.mean():.4f}"
          f"   diff {diff.mean():+.4f} +/- {se:.4f}")
    print(f"    signal above the 0.500 null: {0.5 - a.mean():.4f} -> {0.5 - b.mean():.4f}"
          f"   ({kept:.0%} retained)")
    print(f"    per-sample correlation of the two paths: r = {r:.3f}")

    if kept > 0.7 and r > 0.7:
        print("    >> WORLD A. The goal argument is inert -- another question's goal scores")
        print("       the same samples the same way. Delta is not measuring distance-to-goal")
        print("       (root cause D, §1.2). No loss reweighting reaches this; the goal has to")
        print("       enter the score architecturally.")
    elif kept < 0.3:
        print("    >> WORLD B. The goal carries the QUESTION and the signal dies without it,")
        print("       but §9.7.3 showed a better goal buys nothing WITHIN a question. The")
        print("       next run targets within-question resolution: §7.10 `same_index`")
        print("       pairing plus §16.23's ruler. This is a training problem, not an")
        print("       architecture one.")
    else:
        print("    >> NEITHER CLEANLY. Partial retention -- the goal carries some")
        print("       within-question information and it is weak. Read `kept` and r together")
        print("       and do NOT collapse this to A or B in the writeup.")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="a PHASE-2 checkpoint (goal head)")
    ap.add_argument("--subsets", default="gsm8k,math,olympiadbench,omnimath")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", type=Path, default=None,
                    help="the reported deltas.npz; asserts pass A reproduces it exactly")
    args = ap.parse_args(argv)

    ckpt = Path(args.checkpoint)
    cfg = load_config_from_checkpoint(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(cfg)
    backbone = load_backbone_with_adapter(cfg, ckpt / "adapter")
    model = FeynmanPRM(cfg, read_hidden_size(cfg.model.name), backbone=backbone,
                       with_goal_head=True)
    load_heads(model, ckpt)                     # no allow_missing -- §9.3.2 link 2
    model.to(device).eval()

    reported = np.load(args.compare) if args.compare else None
    raw: dict[str, np.ndarray] = {}
    summary: dict[str, dict[str, float]] = {}

    for subset in [s.strip() for s in args.subsets.split(",") if s.strip()]:
        samples = load_processbench(subset)
        labels = np.asarray([s.label for s in samples], dtype=np.int64)

        # ---- pass A: the reported path, recording each sample's own goal ----------------
        cache: dict[str, Tensor] = {}

        def spy(h_s0: Tensor, batch: Sequence[Sample]) -> Tensor:
            g = model.goal_head(h_s0)
            for b, sample in enumerate(batch):
                cache[sample.id] = g[b].detach().to("cpu")
            return g

        base_deltas, _ = score_samples(model, tokenizer, samples, cfg, device,
                                       goal_fn=spy, label=f"{subset}-own")

        if reported is not None and f"{subset}/flat" in reported.files:
            flat, lengths = pack_deltas(base_deltas)
            if not (np.array_equal(lengths, reported[f"{subset}/lengths"])
                    and np.allclose(flat, reported[f"{subset}/flat"], atol=1e-5)):
                raise SystemExit(
                    f"pass A does not reproduce the reported {subset} Deltas. The checkpoint "
                    f"or the scoring path has moved; fix that before reading pass B."
                )
            print(f"{subset}: pass A reproduces the reported Deltas", flush=True)

        # ---- pass B: identical batching, another question's goal ------------------------
        ids = sorted(cache)                      # over-length samples never reach goal_fn
        perm = derangement(len(ids), np.random.default_rng(args.seed))
        swap = {ids[i]: cache[ids[perm[i]]] for i in range(len(ids))}

        def crossq(h_s0: Tensor, batch: Sequence[Sample]) -> Tensor:
            g = torch.stack([swap[s.id] for s in batch])
            return g.to(device=h_s0.device, dtype=h_s0.dtype)

        swap_deltas, _ = score_samples(model, tokenizer, samples, cfg, device,
                                       goal_fn=crossq, label=f"{subset}-crossq")

        for name, d in ((subset, base_deltas), (f"{subset}-crossq", swap_deltas)):
            flat, lengths = pack_deltas(d)
            raw[f"{name}/flat"] = flat
            raw[f"{name}/lengths"] = lengths
            raw[f"{name}/labels"] = labels

        base_r, swap_r = ranks(base_deltas, labels), ranks(swap_deltas, labels)
        verdict(base_r, swap_r, subset)
        summary[subset] = {
            "n_paired": len(set(base_r) & set(swap_r)),
            "rank_own_goal": float(np.mean(list(base_r.values()))),
            "rank_cross_question": float(np.mean(list(swap_r.values()))),
        }

    out = Path(args.out) if args.out else ckpt / "deltas_crossq.npz"
    np.savez_compressed(out, **raw)
    (out.with_suffix(".json")).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}\nwrote {out.with_suffix('.json')}")
    print("nothing above is a reported result (§9.3) -- it decomposes §9.3.1's F1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
