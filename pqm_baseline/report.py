"""The side-by-side table for the paper: PQM (ours, matched) against Feynman-PRM.

Nothing here recomputes anything -- it is a view over two `processbench.json` files.

    python -m pqm_baseline.report --pqm runs/pqm_zeta4/final \
                                  --feynman runs/abl_cf_only/phase2/final

**The val-F1 column is the trap this file exists to close.** The comparable Feynman number is
the goal-head val F1 recorded as `calibration/f1` inside each run's
`phase2/final/processbench.json` -- **0.5900** for `abl_cf_only`, **0.5872** for
`phase1_nce_temp_relu2`. It is NOT `scripts/val_f1.py`'s 0.5615: that script substitutes a
real terminal for the goal (the §9.5 skyline substitution) and its own docstring calls it a
ceiling, not a result. This reads `calibration/f1`.

**The PQM row is our re-implementation under matched conditions, not PQM's published
numbers.** The footer says so; keep it wherever the table is quoted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
NAME_W = 38        # "Feynman-PRM (phase1_nce_temp_relu2)" is 35
COL_W = 15         # "olympiadbench" is 13

FOOTER = """
NOTES, and all three belong in the paper:

  * The PQM row is OUR re-implementation of PQM's head (Dropout -> Linear(H,1)) and objective
    (the Q-ranking loss of Process_Q_Model/train_main.py:61-78, ported verbatim) under
    Feynman-PRM's exact training conditions -- same parquet, same 34,650-question selection,
    same seed, same batch stream, same optimizer steps, same eval. It is NOT PQM's published
    number: that paper trains deepseek-math-7b-base full-finetune on 8 GPUs for 2 epochs over
    the whole Math-Shepherd corpus and reports Best-of-N.

  * PQM never reports ProcessBench, so the localisation rule -- "the first step whose reward
    falls below tau" -- is OURS. It is the same protocol Feynman-PRM is scored under, which is
    what makes the comparison fair, but it must be described as ours.

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


def _row(name: str, blob: dict) -> str:
    cells = []
    f1s = []
    for subset in SUBSETS:
        if subset in blob:
            f1 = float(blob[subset]["f1"])
            f1s.append(f1)
            cells.append(f"{f1:>{COL_W}.4f}")
        else:
            cells.append(f"{'--':>{COL_W}s}")
    mean = f"{sum(f1s) / len(f1s):>{COL_W}.4f}" if f1s else f"{'--':>{COL_W}s}"
    val = _val_f1(blob)
    val_s = f"{val:>{COL_W}.4f}" if val is not None else f"{'--':>{COL_W}s}"
    tau = _tau(blob)
    tau_s = f"{tau:>{COL_W}.4f}" if tau is not None else f"{'--':>{COL_W}s}"
    return f"  {name:<{NAME_W}s}" + "".join(cells) + mean + val_s + tau_s


def _leak_row(name: str, blob: dict) -> list[str]:
    """Locked #5: the math subset split 587 leaked / 413 clean, for both rows."""
    split = (blob.get("math") or {}).get("leak_split")
    if not split:
        return []
    return [
        f"  {name:<{NAME_W}s} leaked(587) F1 {split['leaked']['f1']:.4f}   "
        f"clean(413) F1 {split['clean']['f1']:.4f}   "
        f"gap {split['leaked']['f1'] - split['clean']['f1']:+.4f}"
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PQM vs Feynman-PRM, side by side")
    parser.add_argument("--pqm", required=True, help="runs/pqm_zeta4/final")
    parser.add_argument(
        "--feynman",
        action="append",
        required=True,
        help="runs/<name>/phase2/final -- repeatable, one row each",
    )
    parser.add_argument("--pqm-name", default="PQM (zeta=4), matched")
    args = parser.parse_args(argv)

    pqm = _load(Path(args.pqm))
    feynman = [(Path(p), _load(Path(p))) for p in args.feynman]

    header = (
        f"  {'':<{NAME_W}s}"
        + "".join(f"{s:>{COL_W}s}" for s in SUBSETS)
        + f"{'mean':>{COL_W}s}{'val F1':>{COL_W}s}{'tau':>{COL_W}s}"
    )
    print("\nProcessBench F1, and the Math-Shepherd val F1 tau was fitted on")
    print(header)
    for path, blob in feynman:
        name = path.parts[1] if len(path.parts) > 1 else str(path)
        print(_row(f"Feynman-PRM ({name})", blob))
    print(_row(args.pqm_name, pqm))

    leak_lines = []
    for path, blob in feynman:
        name = path.parts[1] if len(path.parts) > 1 else str(path)
        leak_lines += _leak_row(f"Feynman-PRM ({name})", blob)
    leak_lines += _leak_row(args.pqm_name, pqm)
    if leak_lines:
        print("\nmath leak split (locked #5) -- report BOTH halves, never the leaked one alone")
        print("\n".join(leak_lines))

    block = pqm.get("pqm") or {}
    if block:
        natural = block.get("natural_tau_delta")
        tau = _tau(pqm)
        print(
            f"\nPQM: zeta = {block.get('zeta')}, loss_type = {block.get('loss_type')}, "
            f"head_init = {block.get('head_init')}, label_source = {block.get('label_source')}"
        )
        if natural is not None and tau is not None:
            ratio = tau / natural if natural else float("nan")
            print(
                f"     fitted tau {tau:.4f} against PQM's natural zeta/2 = {natural:.4f} "
                f"(ratio {ratio:.2f}) -- a CHECK, not a constraint"
            )
    print(FOOTER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
