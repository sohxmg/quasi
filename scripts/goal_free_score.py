"""§9.4 + §9.7.7 -- how much localisation survives with the goal REMOVED, not replaced.

**Why this exists.** The cross-question swap (`scripts/cross_question_goal.py`) returned
`gsm8k` rank 0.3871 with the sample's own goal and 0.3839 with an unrelated question's goal --
**103% of the signal retained, 0.13 sigma** -- while the per-sample correlation was only
r = 0.477. Read together those two say the goal argument DOES enter Delta (r is not 1, so it
is not cancelling) and yet carries NO information (the rank does not move when it is wrong).
That is neither §9.7.7's World A nor its World B. It says the localisation ability measured in
§9.7.2 is a property of the psi TRAJECTORY and not of any state-goal relationship.

**The test.** Score each step with no goal at any point in the computation:

    fwd    d(psi_{i-1}, psi_i)        the forward step cost. Big = an expensive step.
    rev    d(psi_i, psi_{i-1})        the same step measured backwards
    asym   fwd - rev                  §9.4's irreversibility score. Large positive = hard to
                                      undo = the direct test of the ASYMMETRY claim that
                                      justifies a quasimetric over a plain similarity, and
                                      §9.4 has asked for it since day one.
    l2     ||psi_i - psi_{i-1}||_2    plain Euclidean displacement. THE CONTROL, and the
                                      harshest one: no MRN, no asymmetry, no goal, no learned
                                      distance structure at all beyond psi itself.

Read against the goal-head path's own rank on the same samples, paired:

    fwd/asym ~ the goal-head rank   ->  the goal contributes nothing and §9.4 becomes the
                                        headline score instead of a footnote. Cheaper, needs
                                        no phase 2, and honest about what it measures.
    l2 ~ the goal-head rank         ->  **the quasimetric contributes nothing either.** The
                                        whole score reduces to "this step moved psi a lot",
                                        and §1's two motivating properties (triangle
                                        inequality, asymmetry) are unsupported on this
                                        checkpoint. This is the result that would redirect
                                        the project rather than the run.
    all well above 0.5 / near 0.5   ->  the goal IS load-bearing after all and the
                                        cross-question result needs re-examining first.

NOTHING HERE IS A REPORTED RESULT (§9.3). It decomposes §9.3.1's F1 on an existing checkpoint.

    python scripts/goal_free_score.py --checkpoint runs/phase1/phase2/final \
        --subsets gsm8k --compare runs/phase1/phase2/final/deltas.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from cross_question_goal import ranks                      # scripts/ is on sys.path
from feynman_prm.eval.processbench import (
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


def variants(model) -> dict[str, callable]:
    """`states` is (T+1, D) = psi_0 .. psi_T; each returns (T,), one score per step.

    `d(from, to)` -- state first, goal second, everywhere (§6.5). Here `from` is the earlier
    state for `fwd`, so `fwd_i` is the cost of the step that produced psi_i.
    """
    def fwd(states: Tensor) -> Tensor:
        return model.distance(states[:-1], states[1:])

    def rev(states: Tensor) -> Tensor:
        return model.distance(states[1:], states[:-1])

    return {
        "fwd": fwd,
        "rev": rev,
        "asym": lambda s: fwd(s) - rev(s),
        "l2": lambda s: (s[1:] - s[:-1]).norm(dim=-1),
    }


def compare(base: dict[int, float], other: dict[int, float], name: str) -> dict[str, float]:
    common = sorted(set(base) & set(other))
    a = np.array([base[i] for i in common])
    b = np.array([other[i] for i in common])
    diff = b - a
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
    r = float(np.corrcoef(a, b)[0, 1]) if len(common) > 2 else float("nan")
    # The signal is the distance from the 0.500 null, not the rank itself.
    kept = (0.5 - b.mean()) / (0.5 - a.mean()) if abs(0.5 - a.mean()) > 1e-9 else float("nan")
    print(f"    {name:<5} rank {b.mean():.4f}   vs goal head {a.mean():.4f}"
          f"   diff {diff.mean():+.4f} +/- {se:.4f}"
          f"   signal retained {kept:6.0%}   r = {r:+.3f}")
    return {"rank": float(b.mean()), "retained": float(kept), "r": r,
            "diff": float(diff.mean()), "se": float(se), "n": len(common)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--subsets", default="gsm8k,math,olympiadbench,omnimath")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", type=Path, default=None,
                    help="the reported deltas.npz; asserts the goal-head pass reproduces it")
    args = ap.parse_args(argv)

    ckpt = Path(args.checkpoint)
    cfg = load_config_from_checkpoint(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(cfg)
    backbone = load_backbone_with_adapter(cfg, ckpt / "adapter")
    model = FeynmanPRM(cfg, read_hidden_size(cfg.model.name), backbone=backbone,
                       with_goal_head=True)
    load_heads(model, ckpt)
    model.to(device).eval()

    reported = np.load(args.compare) if args.compare else None
    raw: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}

    for subset in [s.strip() for s in args.subsets.split(",") if s.strip()]:
        samples = load_processbench(subset)
        labels = np.asarray([s.label for s in samples], dtype=np.int64)

        base_deltas, _ = score_samples(model, tokenizer, samples, cfg, device,
                                       label=f"{subset}-goalhead")
        if reported is not None and f"{subset}/flat" in reported.files:
            flat, lengths = pack_deltas(base_deltas)
            if not (np.array_equal(lengths, reported[f"{subset}/lengths"])
                    and np.allclose(flat, reported[f"{subset}/flat"], atol=1e-5)):
                raise SystemExit(f"the goal-head pass does not reproduce the reported "
                                 f"{subset} Deltas -- fix that before reading anything below")
            print(f"{subset}: goal-head pass reproduces the reported Deltas", flush=True)

        base_r = ranks(base_deltas, labels)
        flat, lengths = pack_deltas(base_deltas)
        raw[f"{subset}/flat"], raw[f"{subset}/lengths"] = flat, lengths
        raw[f"{subset}/labels"] = labels

        print(f"\n--- {subset}: goal-free scores, PAIRED against the goal head "
              f"({len(base_r)} errored samples)")
        summary[subset] = {}
        for name, fn in variants(model).items():
            d, _ = score_samples(model, tokenizer, samples, cfg, device,
                                 state_score_fn=fn, label=f"{subset}-{name}")
            summary[subset][name] = compare(base_r, ranks(d, labels), name)
            flat, lengths = pack_deltas(d)
            raw[f"{subset}-{name}/flat"], raw[f"{subset}-{name}/lengths"] = flat, lengths
            raw[f"{subset}-{name}/labels"] = labels

        l2, fwd = summary[subset]["l2"], summary[subset]["fwd"]
        if l2["retained"] > 0.8:
            print("    >> THE CONTROL MATCHES. Plain Euclidean displacement localises as well")
            print("       as the full pipeline. On this checkpoint the goal, the MRN and the")
            print("       asymmetry are all contributing nothing measurable -- the score is a")
            print("       'this step moved psi a lot' detector. Redirect, do not retune.")
        elif fwd["retained"] > 0.8:
            print("    >> The goal contributes nothing but the learned distance does. §9.4's")
            print("       goal-free score is the honest headline and needs no phase 2.")
        else:
            print("    >> The goal IS load-bearing here. Re-examine the cross-question result")
            print("       before acting on it.")

    out = Path(args.out) if args.out else ckpt / "deltas_goalfree.npz"
    np.savez_compressed(out, **raw)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}\nwrote {out.with_suffix('.json')}")
    print("nothing above is a reported result (§9.3) -- it decomposes §9.3.1's F1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
