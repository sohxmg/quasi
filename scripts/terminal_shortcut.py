#!/usr/bin/env python
"""Measure (7) `L_term`'s printed-answer shortcut on real Math-Shepherd questions (§7.13.1).

    python scripts/terminal_shortcut.py --questions 2000

Streams the train split, groups by question exactly as training does
(`data/math_shepherd.build_questions`), and reports the three AUCs. Chance is 0.500 for all
three; a deviation in either direction is the finding (§7.5.6).

**This is a measurement, not a gate.** Nothing in the training path reads it. Write the value
into CLAUDE.md labelled MEASURED with the date, per §17 -- and if it could not be run, write it
in as UNMEASURED with the reason. Never a fabricated number (§0 rule 5).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feynman_prm.data.math_shepherd import build_questions, iter_hf_rows      # noqa: E402
from feynman_prm.diagnostics.terminal_shortcut import (                      # noqa: E402
    terminal_shortcut_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=2000,
                        help="stop after this many DISTINCT questions (0 = the whole split)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--rows", type=int, default=0,
                        help="hard cap on rows streamed; 0 = no cap")
    args = parser.parse_args(argv)

    rows = iter_hf_rows(split=args.split)
    if args.rows:
        rows = itertools.islice(rows, args.rows)
    questions, counters = build_questions(rows)
    if args.questions:
        # First-seen order, which is the dataset's own order -- NOT a sample of the selection
        # `prepare_data.py` makes. Say so in anything this number is written into.
        questions = questions[: args.questions]

    report = terminal_shortcut_report(questions)
    print(f"[shortcut] split={args.split} rows={counters['rows_raw']} "
          f"questions_seen={len(questions)}")
    for line in report.lines():
        print(f"[shortcut] {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
