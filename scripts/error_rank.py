"""§9.6.10 #1 -- the within-solution rank of the true error's Delta.

    rank_s = ( #{j != z: Delta_j > Delta_z} + 0.5 * #ties ) / (T_s - 1)

for every ERRORED sample s, where z is the gold first-error index (0-based over steps)
and Delta_z is that step's own cost (§6.1: Delta_i is the cost of steps[i-1], so the
0-based Delta array is indexed by z directly).

**This is the one statistic in §9.6 that is paired WITHIN a solution.** Every per-solution
nuisance cancels exactly:

    scale    rank is invariant to Delta -> a*Delta   (a > 0)
    offset   rank is invariant to Delta -> Delta + b
    length   rank is normalised to [0, 1] by construction
    tau      never enters -- no threshold is applied

so the null is **exactly 0.5** under exchangeability, distribution-free, with no permutation
test needed. That is what makes it decisive where detection AUC and F1 are not: those two
confound the geometry with the §9.1 decision rule, and this does not.

Read it as:

    ~0.5        the geometry does not know where errors are. No reweighting of the loss set
                reaches this -- not zeta, not relu^2 on L_good, not tau_NCE. The lever is
                more correctness supervision (§7.10 `same_index`, or L_CF).
    0.2 - 0.3   real ordering signal that `max_t Delta_t > tau` is throwing away. The first
                fix is free: a within-solution eval statistic + a tau refit on Math-Shepherd
                val (§9.2-legal). Only then a run.

Also reports, because both are cheap and both are missing from `analyze_deltas.py`:

  * P(argmax == z), i.e. localisation with detection removed entirely -- the ceiling on
    `P(exact | flagged)` if every errored sample were flagged. Against a within-solution
    shuffle null, which preserves T and the Delta marginals exactly.
  * mean T on the errored half vs the clean half (§9.6.10 #2). Every length statistic in
    `analyze_deltas.py` is computed on the clean half only, so the two have never been
    checked for comparability -- if they differ, part of the AUC is a length artifact.

NOTHING HERE IS A REPORTED RESULT. It decomposes the F1 already reported in §9.3.1.

    python scripts/error_rank.py runs/phase1/phase2/final/deltas.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def unpack(flat: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    """Inverse of `processbench.pack_deltas`. Zero-length rows come back as empty arrays."""
    out, off = [], 0
    for n in lengths:
        out.append(flat[off:off + n].astype(np.float64))
        off += int(n)
    return out


def error_rank(delta: np.ndarray, z: int) -> float | None:
    """Normalised rank of delta[z] among its own solution, 0 = largest, 0.5 = chance."""
    T = len(delta)
    if T < 2 or not (0 <= z < T):
        return None
    x = delta[z]
    others = np.delete(delta, z)
    greater = float((others > x).sum())
    ties = float((others == x).sum())
    return (greater + 0.5 * ties) / (T - 1)


def shuffle_null_hit_rate(deltas, zs, rng, reps: int = 200) -> float:
    """P(argmax == z) when each solution's own Delta vector is permuted within itself.

    Preserves T and every Delta marginal; destroys only the position<->label pairing. This
    is the right null for the argmax hit rate -- `1/T` is not (§9.6.9(a): errors are not
    uniform in position and neither is the argmax of a random vector)."""
    hits = 0
    for _ in range(reps):
        for d, z in zip(deltas, zs):
            hits += int(rng.permutation(d).argmax() == z)
    return hits / (reps * len(deltas))


def ranks_by_index(deltas: list[np.ndarray], labels: np.ndarray) -> dict[int, float]:
    """Per-sample rank, keyed by sample index so two scoring paths can be paired."""
    out: dict[int, float] = {}
    for i, (d, z) in enumerate(zip(deltas, labels)):
        r = error_rank(d, int(z)) if int(z) >= 0 else None
        if r is not None:
            out[i] = r
    return out


def paired_paths(base: dict[int, float], sky: dict[int, float], name: str,
                 other: str = "skyline") -> None:
    """Goal head vs reference-solution goal, PAIRED on the same samples.

    tau never enters the rank and every per-solution scale/offset cancels, so this is the
    one comparison of the two scoring paths that carries no handicap (§9.5.1's unrefit-tau
    caveat, and §9.6.6's measured +0.080 / +0.055, are both inapplicable here).

    **Read the correlation, not just the mean difference.** Equal means could be two
    independently-mediocre paths. A high per-sample r means they are making the SAME call on
    the SAME samples -- i.e. Delta is invariant to its own goal argument, which is a much
    stronger and more damning statement (§1.2 root cause D)."""
    common = sorted(set(base) & set(sky))
    if len(common) < 10:
        return
    a = np.array([base[i] for i in common])
    b = np.array([sky[i] for i in common])
    diff = b - a
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    r = float(np.corrcoef(a, b)[0, 1])
    sigma = f"{abs(diff.mean()) / se:.1f} sigma" if se > 0 else "identical"
    print(f"\n--- {name}: goal head vs {other}, PAIRED on {len(common)} samples")
    print(f"    goal head {a.mean():.4f}   {other} {b.mean():.4f}"
          f"   diff {diff.mean():+.4f} +/- {se:.4f}  ({sigma})")
    print(f"    per-sample correlation of the two paths: r = {r:.3f}")
    # **r is the verdict, not the mean difference.** At n in the hundreds a systematic shift
    # of 0.005 clears 2 sigma while being substantively nothing, so a mean-driven rule would
    # go silent exactly where the finding is strongest. r asks the question that matters:
    # do the two goals produce the SAME per-sample call?
    if r > 0.7:
        print("    >> THE GOAL ARGUMENT IS BEING IGNORED. Swapping a predicted goal for the")
        print("       gold reference terminal leaves the per-sample call unchanged, so Delta")
        print("       is not measuring distance-to-goal (§1.2 root cause D). A better goal")
        print("       head cannot help. Check probe08/corr_distance_psi_norm and")
        print("       nce/accuracy_within_question to confirm from the phase-1 logs.")
        if abs(diff.mean()) > 2 * se:
            print(f"       (the {diff.mean():+.4f} mean shift is significant but substantively")
            print("        negligible at this n -- read r, not the sigma)")
    elif abs(diff.mean()) <= 2 * se + 1e-12:
        print("    >> Means match but the paths DISAGREE per sample: two independently weak")
        print("       goals, NOT proof the goal channel is dead. Weaker conclusion.")
    else:
        print("    >> The paths differ in both mean and per-sample call -- the goal argument")
        print("       does carry information, and one path is genuinely better.")


def leak_split(ranks: dict[int, float], leaked: np.ndarray,
               deltas: list[np.ndarray], labels: np.ndarray) -> None:
    """§2 #5 on the rank instead of on F1 -- no rule confounds, so it is the cleaner test.

    §9.3.1 measured the F1 split at +0.047 and read it as 'the metric is not keying on the
    problem'. F1 mixes in detection, tau and length, all of which are noise-dominated here
    and wash the split out; the rank does not. If the ordering signal is partly memorisation
    it shows up here and nowhere else.

    Length is reported alongside because it is the one confound the rank does NOT remove:
    leaked and clean math problems could simply differ in T, and shorter solutions are easier
    to localise in. Read the T rows before believing the split."""
    idx = sorted(ranks)
    a_i = [i for i in idx if leaked[i]]
    b_i = [i for i in idx if not leaked[i]]
    if len(a_i) < 10 or len(b_i) < 10:
        return
    a = np.array([ranks[i] for i in a_i])
    b = np.array([ranks[i] for i in b_i])
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    print(f"\n--- math leak split on the RANK (§2 #5, no rule confounds)")
    print(f"    rank    leaked {a.mean():.4f} (n={len(a)})   clean {b.mean():.4f} (n={len(b)})"
          f"   diff {a.mean() - b.mean():+.4f} +/- {se:.4f}"
          f"  ({abs(a.mean() - b.mean()) / se:.1f} sigma)")
    print(f"    signal above the 0.5 null:  leaked {0.5 - a.mean():.4f}"
          f"   clean {0.5 - b.mean():.4f}"
          f"   ratio {(0.5 - a.mean()) / (0.5 - b.mean()):.2f}x")

    ha = float(np.mean([deltas[i].argmax() == int(labels[i]) for i in a_i]))
    hb = float(np.mean([deltas[i].argmax() == int(labels[i]) for i in b_i]))
    print(f"    argmax  leaked {ha:.4f}   clean {hb:.4f}   diff {ha - hb:+.4f}")

    ta = np.array([len(deltas[i]) for i in a_i], dtype=float)
    tb = np.array([len(deltas[i]) for i in b_i], dtype=float)
    print(f"    mean T  leaked {ta.mean():.2f}   clean {tb.mean():.2f}"
          f"   -- {'CONFOUNDED, the split may be length' if abs(ta.mean() - tb.mean()) > 0.5 else 'comparable, so the split is not length'}")

    # ---- the length confound, stratified away ---------------------------------------
    # E[rank] is MATHEMATICALLY independent of T -- rank is a mean over (T-1) comparisons
    # and each comparison has the same expectation however many there are -- so a length
    # gap cannot shift it mechanically. But T correlates with DIFFICULTY, and leaked
    # problems here are 1.1 steps shorter. So the split could be "easier problems" rather
    # than "remembered problems". Comparing only inside matched-length buckets removes it.
    print("\n    stratified by T (removes the difficulty-via-length confound):")
    strata, num, den = [(2, 5), (6, 7), (8, 9), (10, 99)], 0.0, 0.0
    for lo, hi in strata:
        sa = np.array([ranks[i] for i in a_i if lo <= len(deltas[i]) <= hi])
        sb = np.array([ranks[i] for i in b_i if lo <= len(deltas[i]) <= hi])
        if len(sa) < 10 or len(sb) < 10:
            print(f"      T {lo:>2}-{hi:<2}  leaked n={len(sa):<4} clean n={len(sb):<4}"
                  "  too thin to read")
            continue
        d = sa.mean() - sb.mean()
        w = 1.0 / (sa.var(ddof=1) / len(sa) + sb.var(ddof=1) / len(sb))
        num, den = num + w * d, den + w
        print(f"      T {lo:>2}-{hi:<2}  leaked {sa.mean():.4f} (n={len(sa):<4})"
              f" clean {sb.mean():.4f} (n={len(sb):<4}) diff {d:+.4f}")
    if den > 0:
        pooled, pse = num / den, np.sqrt(1.0 / den)
        print(f"      POOLED within-length diff {pooled:+.4f} +/- {pse:.4f}"
              f"  ({abs(pooled) / pse:.1f} sigma)   [unstratified was"
              f" {a.mean() - b.mean():+.4f}]")
        if pooled < -2 * pse:
            print("    >> SURVIVES stratification. The leak advantage is NOT length, so part")
            print("       of the ordering signal is MEMORISATION, not geometry. The honest")
            print("       uncontaminated number is the `clean` column, and this OVERTURNS")
            print("       §9.3.1's reading of the +0.047 F1 split -- F1 hid it, the rank does not.")
        else:
            print("    >> DOES NOT survive stratification. The raw split was length/difficulty,")
            print("       not memorisation. §9.3.1's reading stands and finding 3 is withdrawn.")
    elif a.mean() < b.mean() - 2 * se:
        print("    >> Raw split is large but every stratum is too thin to confirm it.")
        print("       Treat memorisation as SUSPECTED, not established.")


def report(name: str, deltas: list[np.ndarray], labels: np.ndarray, rng) -> None:
    err = [(d, int(z)) for d, z in zip(deltas, labels) if z >= 0 and len(d) >= 2 and z < len(d)]
    ok = [d for d, z in zip(deltas, labels) if z < 0 and len(d)]
    if not err:
        print(f"\n=== {name} — no usable errored samples")
        return

    ed, ez = [d for d, _ in err], [z for _, z in err]
    ranks = np.array([error_rank(d, z) for d, z in err], dtype=np.float64)
    n = len(ranks)
    mean, se = ranks.mean(), ranks.std(ddof=1) / np.sqrt(n)

    hit = float(np.mean([d.argmax() == z for d, z in err]))
    null_hit = shuffle_null_hit_rate(ed, ez, rng)

    print(f"\n=== {name} " + "=" * max(0, 55 - len(name)))
    print(f"  errored samples used: {n}")
    print(f"  WITHIN-SOLUTION RANK of the true error's Delta:  {mean:.4f} +/- {se:.4f}"
          f"   [null EXACTLY 0.500, 0 = error is the largest step]")
    z_score = (0.5 - mean) / se if se > 0 else float("nan")
    print(f"    {z_score:+.1f} sigma from chance  (positive = errors rank ABOVE their own"
          f" solution's other steps, which is the direction that helps)")
    print(f"  top-decile / quartile of ranks:  p10 {np.quantile(ranks, .10):.3f}"
          f"   p25 {np.quantile(ranks, .25):.3f}   median {np.median(ranks):.3f}")
    multiple = f" -> {hit / null_hit:.2f}x" if null_hit > 0 else ""
    print(f"  P(argmax == z), detection removed:  {hit:.4f}"
          f"   [within-solution shuffle null {null_hit:.4f}{multiple}]")
    if ok:
        te = np.array([len(d) for d in ed], dtype=float)
        to = np.array([len(d) for d in ok], dtype=float)
        print(f"  mean T:  errored {te.mean():.2f} (n={len(te)})"
              f"   clean {to.mean():.2f} (n={len(to)})"
              f"   {'COMPARABLE' if abs(te.mean() - to.mean()) < 0.5 else 'DIFFERENT'}"
              " -- if different, part of the detection AUC is a length artifact")

    if mean > 0.45:
        print("  >> AT THE NULL. The geometry does not localise errors. No loss-weight change")
        print("     (zeta, relu^2 on L_good, tau_NCE) reaches this -- see the module docstring.")
    elif mean < 0.35:
        print("  >> REAL SIGNAL the `max_t Delta_t > tau` rule is discarding. The within-solution")
        print("     statistic + a val-side tau refit is free and comes BEFORE any retrain.")
    else:
        print("  >> Marginal. Read p25 and the argmax multiple before committing a run.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.npz)
    rng = np.random.default_rng(args.seed)
    subsets = sorted({k.split("/")[0] for k in z.files})

    print(f"{args.npz}")
    print("nothing below is a reported result (§9.3). These decompose the F1 already reported.")
    per_subset: dict[str, dict[int, float]] = {}
    for s in subsets:
        if f"{s}/flat" not in z or f"{s}/labels" not in z:
            continue
        deltas, labels = unpack(z[f"{s}/flat"], z[f"{s}/lengths"]), z[f"{s}/labels"]
        report(s, deltas, labels, rng)
        per_subset[s] = ranks_by_index(deltas, labels)

    for s in sorted(per_subset):
        # `-crossq` is §9.7.7's cross-question swap (`scripts/cross_question_goal.py`), which
        # is a DIFFERENT question's goal, not a better one. Read its r the same way.
        for suffix in ("skyline", "crossq", "fwd", "rev", "asym", "l2"):
            if f"{s}-{suffix}" in per_subset:
                paired_paths(per_subset[s], per_subset[f"{s}-{suffix}"], s, other=suffix)

    if "math" in per_subset and "math/leaked" in z.files:
        md, ml = unpack(z["math/flat"], z["math/lengths"]), z["math/labels"]
        leak_split(per_subset["math"], z["math/leaked"], md, ml)

    print("\nThe rank is the number to act on. It is paired within solution, so T, tau, and")
    print("every per-solution scale and offset cancel -- unlike AUC and F1, which confound the")
    print("geometry with the §9.1 decision rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
