"""Put our Best-of-N numbers next to CRM's published ones.

    python scripts/report_bon.py runs/phase1/phase2/final/bon

Reads every `bon_*.json` written by `feynman_prm.eval.bon` and the transcribed paper values
in `bon_reference/crm_paper.json`, and prints three blocks per candidate file:

  1. **The comparison**, ours against every published method, at N = 8..128.
  2. **The harness check** -- our `baseline_first_candidate` against the paper's. This is the
     one row that does not involve a reward model, so it is the only row that can tell you
     whether the two harnesses are looking at the same thing. Read it before block 1.
  3. **All six aggregators**, ours only, so the headline choice can be seen in context. The
     headline is whichever `--aggregator auto` picked on Math-Shepherd val (§9.2's rule); the
     other five are printed because hiding them would make the choice look luckier than it is.

Nothing here recomputes an accuracy. It is a renderer over two JSON files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent.parent / "bon_reference" / "crm_paper.json"
COLUMNS = ["best_of_8", "best_of_16", "best_of_32", "best_of_64", "best_of_128"]
LABEL_WIDTH = 28


def cell(value) -> str:
    return "--" if value is None else f"{float(value):.2f}"


def row(label: str, values: dict | None) -> None:
    values = values or {}
    print(f"  {label:<{LABEL_WIDTH}}" + "".join(f"{cell(values.get(c)):>9}" for c in COLUMNS))


def delta_row(label: str, ours: dict | None, theirs: dict | None) -> None:
    ours, theirs = ours or {}, theirs or {}
    cells = []
    for column in COLUMNS:
        a, b = ours.get(column), theirs.get(column)
        cells.append("--" if a is None or b is None else f"{float(a) - float(b):+.2f}")
    print(f"  {label:<{LABEL_WIDTH}}" + "".join(f"{c:>9}" for c in cells))


def header(title: str) -> None:
    print(f"\n{title}")
    print("  " + "-" * (LABEL_WIDTH + 9 * len(COLUMNS)))
    print(f"  {'N':<{LABEL_WIDTH}}" + "".join(f"{c.removeprefix('best_of_'):>9}" for c in COLUMNS))


def has_numbers(block: dict) -> bool:
    return any(
        isinstance(v, dict) and any(x is not None for x in v.values()) for v in block.values()
    ) or block.get("baseline_first_candidate") is not None


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    results_dir = Path(argv[1])
    files = sorted(results_dir.glob("bon_*.json"))
    if not files:
        print(f"no bon_*.json under {results_dir}")
        return 1

    reference = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else {}
    published = reference.get("files", {})
    any_reference = any(has_numbers(b) for b in published.values() if isinstance(b, dict))
    print(f"published reference: {reference.get('source', '(none)')}")

    for path in files:
        payload = json.loads(path.read_text())
        stem = Path(payload["data_file"]).stem
        acc = payload["accuracy"]
        primary = payload["primary_aggregator"]
        ours = acc.get(primary, {})
        theirs = published.get(stem, {}) if isinstance(published.get(stem), dict) else {}

        header(
            f"{stem}  ({payload['n_questions']} questions x {payload['n_candidates']} candidates,"
            f" tau={payload['tau']:.4f} from {payload['tau_source']})"
        )
        for method, table in theirs.items():
            if isinstance(table, dict):
                row(f"{method} (published)", table)
        row(f"feynman-prm [{primary}]", ours)
        for method, table in theirs.items():
            if isinstance(table, dict):
                delta_row(f"  delta vs {method}", ours, table)

        # -- block 2: the harness check, and it comes before any conclusion --------------
        ours_base = acc.get("baseline_first_candidate")
        theirs_base = theirs.get("baseline_first_candidate")
        print(f"\n  no-reward-model baseline: ours {cell(ours_base)}, published {cell(theirs_base)}")
        if theirs_base is None:
            print("    ^ NOT TRANSCRIBED. Until it is, any difference in the rows above cannot "
                  "be attributed to the reward model rather than to the harness.")
        elif ours_base is not None and abs(float(ours_base) - float(theirs_base)) > 1.0:
            print("    ^ MORE THAN 1 POINT APART. The candidate pool or the grader differs, so "
                  "the comparison above is NOT valid. Check the *-128.json files and that the "
                  "reference set is GSM-Plus testmini / MATH-500 test.")
        if "baseline_mean_candidate" in acc:
            print(f"  mean over all candidates: {acc['baseline_mean_candidate']:.2f}")
        if "oracle" in acc:
            row("  oracle (pass@N, upper bound)", acc["oracle"])

        counters = payload.get("counters", {})
        if counters.get("over_length", 0):
            print(f"  !! {counters['over_length']:.0f} candidates over max_len "
                  f"({100 * counters['over_length_fraction']:.2f}%), ranked last, not dropped")
        if counters.get("empty_response", 0):
            print(f"  !! {counters['empty_response']:.0f} responses contained no 'Step N:' "
                  "marker and were scored as a single placeholder step (CRM does the same)")

        # -- block 3: every aggregator ---------------------------------------------------
        header(f"{stem} -- all six aggregators (ours only; * is the headline)")
        for name, table in acc.items():
            if isinstance(table, dict) and name != "oracle":
                row(("* " if name == primary else "  ") + name, table)

        selection = payload.get("val_selection")
        if selection:
            chosen = selection["chosen"]
            print(f"\n  headline chosen on Math-Shepherd val, never on this file "
                  f"({selection['n_val_questions']} questions, mean pool "
                  f"{selection['mean_pool_size']:.1f}): {chosen} at "
                  f"{selection['selection_accuracy'][chosen]:.4f}, against a random-pick "
                  f"baseline of {selection['baseline_random']:.4f}")

    if not any_reference:
        print(f"\n!! {REFERENCE} holds no numbers yet, so only our column is real. Transcribe "
              "CRM's published table into it -- percent, 0-100 -- and re-run. Start with "
              "`baseline_first_candidate`: it is the row that validates the harness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
