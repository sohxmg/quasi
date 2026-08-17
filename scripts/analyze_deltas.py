#!/usr/bin/env python
"""Everything about `tau` that `processbench.json` cannot answer, from `deltas.npz`.

`processbench.json` keeps four floats per subset. Those four floats cannot distinguish:

  * a model that never flags an errored solution   from one that flags it at the wrong step
  * a real result                                  from what an uninformative Delta scores
    under the same first-crossing rule
  * a separation failure                           from a tau that failed to transfer
  * the model's judgement                          from the rule's bias toward short solutions

All four are pure functions of the raw Delta arrays, which `eval/processbench.py` now writes
to `deltas.npz` next to the json. This script runs on a laptop in seconds. No GPU.

    python scripts/analyze_deltas.py runs/phase1/phase2/final/deltas.npz

NOTHING HERE IS A REPORTED RESULT. The tau sweep in particular fits a threshold ON
ProcessBench, which §9.2 forbids for the headline number -- it is here to decompose the loss
into "separation" and "threshold transfer", and it is labelled accordingly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run without `pip install -e .`

from feynman_prm.eval.metrics import harmonic_mean, processbench_metrics
from feynman_prm.utils.indexing import predicted_label_from_deltas

SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


def load(path: Path, key: str) -> tuple[list[np.ndarray], np.ndarray] | None:
    z = np.load(path)
    if f"{key}/lengths" not in z:
        return None
    flat, lengths = z[f"{key}/flat"], z[f"{key}/lengths"]
    out, i = [], 0
    for n in lengths.tolist():
        out.append(flat[i : i + n])
        i += n
    return out, z[f"{key}/labels"]


def metrics_at(deltas: list[np.ndarray], labels: np.ndarray, tau: float) -> dict:
    preds = [predicted_label_from_deltas(d.tolist(), tau) for d in deltas]
    return processbench_metrics(preds, labels.tolist())


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random `pos` scores above a random `neg`), ties at 0.5. Chance 0.5."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    both = np.concatenate([pos, neg])
    order = both.argsort(kind="mergesort")
    ranks = np.empty(len(both), dtype=np.float64)
    ranks[order] = np.arange(1, len(both) + 1, dtype=np.float64)
    # average ranks over ties
    srt = both[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def permutation_null(
    deltas: list[np.ndarray], labels: np.ndarray, tau: float, n_perm: int, seed: int
) -> tuple[float, float]:
    """F1 of an uninformative Delta under the SAME rule, at the SAME tau.

    Delta vectors are permuted among samples of IDENTICAL length, so solution length, the
    marginal Delta distribution, and the gold-label distribution are all preserved exactly --
    only the sample-specific association is destroyed. What survives is the coincidence that
    a first-crossing rule predicts early indices and gold first errors ARE early (§5's
    median 2-3). That coincidence is worth real acc_error, and it is not a result.
    """
    rng = np.random.default_rng(seed)
    lengths = np.asarray([len(d) for d in deltas])
    buckets = [np.flatnonzero(lengths == n) for n in np.unique(lengths)]
    f1s = []
    for _ in range(n_perm):
        shuffled = list(deltas)
        for idx in buckets:
            if len(idx) < 2:
                continue
            for src, dst in zip(idx, rng.permutation(idx)):
                shuffled[dst] = deltas[src]
        f1s.append(metrics_at(shuffled, labels, tau)["f1"])
    return float(np.mean(f1s)), float(np.std(f1s))


def robust_rescale(deltas: list[np.ndarray], mode: str, floor: float) -> list[np.ndarray]:
    """Per-SOLUTION normalisation of Delta. §9.1 chose Delta over d because "any error in g_q
    that acts like a roughly constant offset drops out of the difference" -- true, and only
    first order. A goal error that acts like a SCALE, d(.,g_q) ~ lambda*d(.,g*), does not drop
    out: it multiplies every Delta in that solution by lambda, while tau is global. Solutions
    whose lambda > 1 then fire spuriously and set tau for everyone.

    Dividing by a per-solution robust scale is the second-order version of the same argument.
    The median is a good-step baseline estimate: even in an errored ProcessBench solution most
    steps are good (first error at index 2-3 of 8-9), so it is not contaminated by the error.

    Costs nothing and needs no retraining -- but only helps if the good-step tail is BETWEEN
    solutions rather than within them, which `variance_split` below is what decides.
    """
    out = []
    for d in deltas:
        if len(d) < 3:
            out.append(d)
            continue
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med))) * 1.4826
        scale = max(mad, floor)
        out.append((d - med) / scale if mode == "center_scale" else d / scale)
    return out


def variance_split(deltas: list[np.ndarray], labels: np.ndarray) -> tuple[float, float]:
    """Variance of good-step Delta decomposed into between-solution and within-solution.

    Between-dominant -> a per-solution offset/scale is setting tau, and `robust_rescale`
    removes it for free. Within-dominant -> the tail is genuinely per-step and only the
    training fix (a tail-weighted L_good, §7.12) touches it.
    """
    good = []
    for i, d in enumerate(deltas):
        if len(d) < 3:
            continue
        g = d if labels[i] == -1 else d[: max(labels[i], 1)]   # steps before the first error
        if len(g) >= 2:
            good.append(g)
    if len(good) < 2:
        return float("nan"), float("nan")
    means = np.array([g.mean() for g in good])
    within = float(np.mean([g.var() for g in good]))
    between = float(means.var())
    return between, within


def report(name: str, deltas: list[np.ndarray], labels: np.ndarray, tau: float, args) -> None:
    scored = [i for i, d in enumerate(deltas) if len(d)]
    err = np.array([i for i in scored if labels[i] != -1])
    ok = np.array([i for i in scored if labels[i] == -1])
    at_tau = metrics_at(deltas, labels, tau)

    print(f"\n=== {name} " + "=" * (58 - len(name)))
    print(f"  at tau={tau:.4f}:  acc_error {at_tau['acc_error']:.3f}   "
          f"acc_correct {at_tau['acc_correct']:.3f}   F1 {at_tau['f1']:.3f}")

    # -- 1. is this above what noise scores under the same rule? ------------------------
    mu, sd = permutation_null(deltas, labels, tau, args.n_perm, args.seed)
    ratio = at_tau["f1"] / mu if mu > 0 else float("inf")
    verdict = "AT THE NULL" if ratio < 1.25 else ("weak" if ratio < 2 else "clear signal")
    print(f"  permutation null F1: {mu:.3f} +/- {sd:.3f}  ->  observed is {ratio:.2f}x  [{verdict}]")

    # -- 2. detection vs localisation ---------------------------------------------------
    # Threshold-free: does the largest Delta in a solution separate errored from clean?
    peak_err = np.array([deltas[i].max() for i in err]) if len(err) else np.zeros(0)
    peak_ok = np.array([deltas[i].max() for i in ok]) if len(ok) else np.zeros(0)
    print(f"  detection AUC (max Delta, errored vs clean, threshold-free): "
          f"{auc(peak_err, peak_ok):.3f}   [chance 0.500]")

    flag_err = float(np.mean([deltas[i].max() > tau for i in err])) if len(err) else float("nan")
    flag_ok = float(np.mean([deltas[i].max() > tau for i in ok])) if len(ok) else float("nan")
    print(f"  flag rate at tau:  errored {flag_err:.3f}   clean {flag_ok:.3f}"
          f"   (errored <= clean means the sign is inverted)")

    flagged = [i for i in err if deltas[i].max() > tau]
    if flagged:
        preds = np.array([predicted_label_from_deltas(deltas[i].tolist(), tau) for i in flagged])
        gold = labels[flagged]
        print(f"  localisation | flagged ({len(flagged)}):  exact "
              f"{np.mean(preds == gold):.3f}   within +/-1 {np.mean(np.abs(preds - gold) <= 1):.3f}"
              f"   [1/T chance ~ {1 / np.mean([len(deltas[i]) for i in flagged]):.3f}]")

    # -- 3. how much of the loss is tau failing to transfer? ----------------------------
    pool = np.concatenate([deltas[i] for i in scored])
    grid = np.unique(np.quantile(pool, np.linspace(0.01, 0.999, 240)))
    curve = [(t, metrics_at(deltas, labels, float(t))["f1"]) for t in grid]
    best_tau, best_f1 = max(curve, key=lambda r: r[1])
    print(f"  ORACLE tau {best_tau:.4f} -> F1 {best_f1:.3f}   NOT A RESULT (fit on ProcessBench,"
          f" §9.2 forbids it)\n    the gap {best_f1 - at_tau['f1']:+.3f} is threshold transfer;"
          f" the rest of the gap to a real PRM is separation")

    # -- 4. is acc_correct just measuring solution length? ------------------------------
    if len(ok):
        per_step = np.concatenate([deltas[i] for i in ok])
        r = float(np.mean(per_step > tau))
        Tbar = float(np.mean([len(deltas[i]) for i in ok]))
        print(f"  clean half: per-step crossing rate {r:.4f}, mean T {Tbar:.2f}"
              f"  ->  (1-r)^T = {(1 - r) ** Tbar:.3f} vs observed acc_correct "
              f"{at_tau['acc_correct']:.3f}")
        edges = [0, 5, 7, 9, 99]
        parts = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [i for i in ok if lo < len(deltas[i]) <= hi]
            if len(sel) >= 20:
                parts.append(f"T in ({lo},{hi}]: {np.mean([deltas[i].max() <= tau for i in sel]):.3f}"
                             f" (n={len(sel)})")
        if parts:
            print("    acc_correct by length -> " + "  ".join(parts))

    # -- 5. can a per-solution rescale fix tau for free? --------------------------------
    between, within = variance_split(deltas, labels)
    if between == between:      # not nan
        share = between / (between + within)
        print(f"  good-step Delta variance: between-solution {between:.3f}, within {within:.3f}"
              f"  -> between is {share:.0%}")
        for mode in ("scale", "center_scale"):
            resc = robust_rescale(deltas, mode, args.mad_floor)
            pool = np.concatenate([resc[i] for i in scored])
            grid = np.unique(np.quantile(pool, np.linspace(0.01, 0.999, 240)))
            f1 = max(metrics_at(resc, labels, float(t))["f1"] for t in grid)
            print(f"    rescale[{mode:12s}] oracle F1 {f1:.3f}")
        print("    (oracle tau on both sides, so this compares SEPARATION, not thresholds. If"
              "\n     a rescale wins, refit tau on Math-Shepherd val under the same rescale --"
              "\n     that stays inside §9.2. If between-solution share is small it cannot help.)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="threshold-free diagnostics on the raw Deltas")
    p.add_argument("path", nargs="?", default="runs/phase1/phase2/final/deltas.npz")
    p.add_argument("--tau", type=float, default=None, help="default: the tau stored in the npz")
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mad-floor", type=float, default=0.05)
    args = p.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"no such file: {path}\nre-run the eval -- it now writes deltas.npz beside the json")
        return 2
    z = np.load(path)
    tau = args.tau if args.tau is not None else float(z["tau"])

    print(f"\n{path}   tau = {tau:.4f}")
    print("nothing below is a reported result (§9.3). These decompose the F1 already reported.")

    f1s = []
    for subset in SUBSETS:
        loaded = load(path, subset)
        if loaded is None:
            continue
        report(subset, *loaded, tau, args)
        f1s.append(metrics_at(loaded[0], loaded[1], tau)["f1"])

    # The skyline, at ITS OWN operating point. §9.5's dichotomy is only decidable this way:
    # result and skyline use different goal sources, so a shared tau compares operating
    # points, not separations. The AUC line is the comparison that means something.
    for subset in SUBSETS:
        loaded = load(path, f"{subset}-skyline")
        if loaded is None:
            continue
        report(f"{subset}-skyline (NOT A RESULT, §9.5)", *loaded, tau, args)

    if f1s:
        print(f"\nmean F1 over {len(f1s)} subsets at the fitted tau: {sum(f1s) / len(f1s):.3f}")
    print("\nRead the detection AUC first. If it is at chance the goal head is irrelevant and")
    print("§9.5's second branch holds. If it is well above chance the geometry does separate")
    print("and the loss is in localisation and threshold transfer, which is a rule problem.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
