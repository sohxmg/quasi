"""The side-by-side table for the paper: QRL+CF against the Feynman-PRM rows and the PQM row.

Nothing here recomputes anything -- it is a view over `processbench.json` files.

    python -m qrl_prm.report --qrl runs/qrl_iqe/phase2/final

With no `--baseline`, the three shipped rows are used by default:

    runs/abl_cf_only/phase2/final        mean F1 0.2599, goal-head val F1 0.5900
    runs/cf_lam2_tau005/phase2/final     mean F1 0.2611, goal-head val F1 0.5954
    runs/pqm_zeta4/final                 mean F1 0.2682, goal-head val F1 0.5766

**The val-F1 column is the trap this file exists to close** (inherited from
`pqm_baseline/report.py`). The comparable number is `calibration/f1` inside each run's own
`processbench.json` -- NOT `scripts/val_f1.py`'s 0.5615, which substitutes a real terminal for
the goal (the §9.5 skyline substitution) and whose own docstring calls it a ceiling.

**Read the QRL row against BOTH neighbours before drawing a conclusion.** It changes the
objective *and*, deliberately, the head (IQE vs full_mrn -- README.md §2), so a gap against
`abl_cf_only` is a gap against the pair of changes. The `--set distance.variant=full_mrn`
control run is what separates them, and if it exists it should be passed as another
`--baseline`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
NAME_W = 40
COL_W = 15

DEFAULT_BASELINES = (
    "runs/abl_cf_only/phase2/final",
    "runs/cf_lam2_tau005/phase2/final",
    "runs/pqm_zeta4/final",
)

FOOTER = """
NOTES, and all of them belong in the paper:

  * The QRL row is OUR adaptation of QRL's constrained objective (Wang & Isola, ICLR 2023 --
    global push + Lagrangian local constraint, quasimetric-rl/.../losses/) to chain-of-thought
    process reward, under Feynman-PRM's exact training conditions: same parquet, same
    34,650-question selection, same seed, same batch stream, same goal sampler, same CF
    corpus, same optimizer steps, same eval. It is NOT a published QRL number: QRL is an
    offline/online goal-reaching RL method on continuous control and reports no ProcessBench.

  * The COUNTERFACTUAL INVARIANCE CONSTRAINT is not QRL's. It is this project's term,
    expressed in QRL's form: an equivalence class of meaning-preserving rewrites is held
    inside a ball of radius epsilon_cf around the true arrived state, in both directions, by
    its own Lagrange multiplier.

  * The QRL row's HEAD DIFFERS from every other row: IQE where the baselines are full_mrn
    (decided 2026-08-25). The comparison therefore moves the objective and the head together.
    `--set distance.variant=full_mrn` is the one-line control; pass it as a --baseline if it
    has been run.

  * ONE RUN EACH, no seed replicate. Quote the gap, not a ranking, unless it is large.
"""


def _load(path: Path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "processbench.json"
    if not p.exists():
        raise SystemExit(f"no such file: {p}")
    return json.loads(p.read_text())


def _val_f1(blob: dict) -> float | None:
    cal = blob.get("calibration")
    return float(cal["calibration/f1"]) if cal and "calibration/f1" in cal else None


def _tau(blob: dict) -> float | None:
    cal = blob.get("calibration")
    if cal and "calibration/tau" in cal:
        return float(cal["calibration/tau"])
    for subset in SUBSETS:
        if subset in blob:
            return float(blob[subset]["tau"])
    return None


def _mean_f1(blob: dict) -> float | None:
    f1s = [float(blob[s]["f1"]) for s in SUBSETS if s in blob]
    return sum(f1s) / len(f1s) if f1s else None


def _row(name: str, blob: dict) -> str:
    cells = [
        f"{float(blob[s]['f1']):>{COL_W}.4f}" if s in blob else f"{'--':>{COL_W}s}"
        for s in SUBSETS
    ]

    def fmt(value: float | None) -> str:
        return f"{value:>{COL_W}.4f}" if value is not None else f"{'--':>{COL_W}s}"

    return (
        f"  {name:<{NAME_W}s}"
        + "".join(cells)
        + fmt(_mean_f1(blob))
        + fmt(_val_f1(blob))
        + fmt(_tau(blob))
    )


def _leak_row(name: str, blob: dict) -> list[str]:
    """Locked #5: the math subset splits 587 leaked / 413 clean. Report BOTH halves."""
    split = (blob.get("math") or {}).get("leak_split")
    if not split:
        return []
    return [
        f"  {name:<{NAME_W}s} leaked(587) F1 {split['leaked']['f1']:.4f}   "
        f"clean(413) F1 {split['clean']['f1']:.4f}   "
        f"gap {split['leaked']['f1'] - split['clean']['f1']:+.4f}"
    ]


def _label(path: Path) -> str:
    """`runs/<name>/phase2/final` -> `<name>`."""
    parts = path.parts
    return parts[1] if len(parts) > 1 and parts[0] == "runs" else str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QRL+CF vs the shipped rows, side by side")
    parser.add_argument("--qrl", required=True, help="runs/qrl_iqe/phase2/final")
    parser.add_argument(
        "--baseline",
        action="append",
        default=None,
        help="a run's eval directory -- repeatable, one row each. Defaults to "
             f"{', '.join(DEFAULT_BASELINES)}",
    )
    parser.add_argument("--qrl-name", default=None, help="row label for the QRL run")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="runs/qrl_iqe -- read its qrl.resolved.yaml and last metrics line for the "
             "constraint footer (the two curves that say whether the objective took)",
    )
    args = parser.parse_args(argv)

    qrl_path = Path(args.qrl)
    qrl_blob = _load(qrl_path)
    baselines = [
        (Path(p), _load(Path(p)))
        for p in (args.baseline if args.baseline is not None else DEFAULT_BASELINES)
    ]
    qrl_name = args.qrl_name or f"QRL+CF ({_label(qrl_path)}), matched"

    header = (
        f"  {'':<{NAME_W}s}"
        + "".join(f"{s:>{COL_W}s}" for s in SUBSETS)
        + f"{'mean':>{COL_W}s}{'val F1':>{COL_W}s}{'tau':>{COL_W}s}"
    )
    print("\nProcessBench F1, and the Math-Shepherd val F1 tau was fitted on")
    print(header)
    for path, blob in baselines:
        print(_row(_label(path), blob))
    print(_row(qrl_name, qrl_blob))

    leak_lines: list[str] = []
    for path, blob in baselines:
        leak_lines += _leak_row(_label(path), blob)
    leak_lines += _leak_row(qrl_name, qrl_blob)
    if leak_lines:
        print("\nmath leak split (locked #5) -- report BOTH halves, never the leaked one alone")
        print("\n".join(leak_lines))

    # ---- the deltas, spelled out so nobody has to subtract in their head ---------------
    qrl_mean, qrl_val = _mean_f1(qrl_blob), _val_f1(qrl_blob)
    if qrl_mean is not None:
        print("\ndelta vs each baseline (mean ProcessBench F1 / goal-head val F1)")
        for path, blob in baselines:
            base_mean, base_val = _mean_f1(blob), _val_f1(blob)
            dv = (
                f"{qrl_val - base_val:+.4f}"
                if qrl_val is not None and base_val is not None
                else "--"
            )
            dm = f"{qrl_mean - base_mean:+.4f}" if base_mean is not None else "--"
            print(f"  vs {_label(path):<{NAME_W - 3}s} {dm:>10s}   {dv:>10s}")

    _constraint_footer(args.run_dir)
    print(FOOTER)
    return 0


def _constraint_footer(run_dir: str | None) -> None:
    """The two curves that say whether the objective took, read off the run's last metrics line.

    `qrl/local_dist_mean` is THE ruler and should sit near `step_cost`; that is the direct
    answer to IMPLEMENTATION.md §9's decaying ruler. `qrl/path_ratio_mean` is the same
    quantity at k >= 2, so the two are read together: the ruler near 1.0 with the ratio far
    above it means the metric satisfies the adjacent steps and still lets observed sub-paths
    blow out -- which is the leak the k >= 2 constraint exists to price, and the reason its
    multiplier is separate. `qrl/lagrange_cf` climbing while
    `qrl/cf_sq_dev` did not fall means the CF corpus contradicts itself -- the dual variable
    is the data-quality detector, and it belongs next to the F1 numbers, not in a separate
    conversation.
    """
    if not run_dir:
        return
    import yaml

    path = Path(run_dir)
    cfg_path = path / "qrl.resolved.yaml"
    metrics_path = path / "metrics.jsonl"
    if not metrics_path.exists():
        print(f"\n[report] no metrics.jsonl under {path}; skipping the constraint footer")
        return
    last = None
    for line in metrics_path.read_text().splitlines():
        if line.strip():
            last = json.loads(line)
    if not last:
        return
    knobs = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    step_cost = knobs.get("step_cost")
    print(f"\nconstraints at the last logged step ({last.get('step')})")
    for key, target in (
        ("qrl/local_dist_mean", step_cost),
        ("qrl/local_over_cost_frac", 0.0),
        ("qrl/local_violation", 0.0),
        ("qrl/lagrange_local", None),
        ("qrl/path_ratio_mean", step_cost),
        ("qrl/path_violation", 0.0),
        ("qrl/lagrange_path", None),
        ("qrl/cf_sq_dev", (knobs.get("_derived") or {}).get("cf_target")),
        ("qrl/cf_violation", 0.0),
        ("qrl/cf_p95", knobs.get("epsilon_cf")),
        ("qrl/lagrange_cf", None),
        ("qrl/push_dist_mean", None),
        ("qrl/push_saturated_frac", None),
        ("qrl/pos_neg_push_dist_mean", None),
        ("qrl/pos_neg_push_gap", None),
    ):
        if key not in last:
            continue
        tail = f"   (target {target:g})" if isinstance(target, (int, float)) else ""
        print(f"  {key:<28s} {float(last[key]):>12.5f}{tail}")


if __name__ == "__main__":
    raise SystemExit(main())
