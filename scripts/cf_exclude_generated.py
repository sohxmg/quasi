#!/usr/bin/env python3
"""Build a run's items file as its slice MINUS every anchor already generated anywhere.

    python3 scripts/cf_exclude_generated.py            # the OpenAI run; build, then verify
    python3 scripts/cf_exclude_generated.py --check    # verify only, touch nothing

    # the GEMINI run, which inherited the Codex slice after Codex went down (2026-08-10):
    python3 scripts/cf_exclude_generated.py \
        --slice data/cf/cf_items_70k_cx.jsonl \
        --out   data/cf/cf_items_70k_cx_gm.jsonl \
        --self  data/cf/cf70k_gm.responses.jsonl

The defaults are the OpenAI run's because it was the first caller; every path is an argument
and the second caller changed none of the logic. `--self` is the one that MUST move with the
caller: it names the response file this campaign will itself write, which is `--resume`'s job
to read and therefore the one file that must not be held out.

WHY THIS IS NEEDED AT ALL, GIVEN split_cf_items.py. That script makes the three CAMPAIGN
slices disjoint, and they are. It cannot see anything that generated INSIDE a slice under a
different `--out`, and there are two such cases:

  * PILOTS. `cf_oai_pilot_low` (20 anchors) and `cf_oai_probe_medium` (2) were run before the
    campaign existed, drew from what is now the OpenAI slice, and wrote to their OWN `--out`.
    `--resume` matches custom_ids inside ONE response file, so the campaign is blind to them
    and would pay for those 22 anchors a second time. MEASURED 2026-08-10: 22 of the 3,000
    oai anchors are in that state.
  * A HANDOVER. Codex went down on 2026-08-10 having generated **1,440** of its 20,000-anchor
    slice into `cf70k_cx.jsonl.responses.jsonl`, and Gemini took the slice over under a new
    `--out`. Same blindness, 65x the cost. MEASURED: 20,000 -> 18,560 to generate.
  * A HANDOVER INTO A LIVE CAMPAIGN, which is the 2026-08-21 case and the one that reads
    oddly. OpenAI exhausted its 3,000-anchor slice and took the back 20,000 of what was left
    of bharatcode's, so its `--slice` is now the UNION of both blocks
    (`cf_items_oai_src.jsonl`). The 2,980 anchors it had already generated are in `--self`, so
    they are NOT held out: they stay in the items file and `--resume` skips them, which is why
    the file is 22,980 and the night's counter says 20,000 to go. Holding them out here would
    work too, and would make every night's "N to go" disagree with the ones before it.

WHY EXCLUDE AND NOT RE-GENERATE THEM. `L_CF` keys on the ANCHOR. A second rewrite set for an
anchor that already has one is a duplicate training row, not a second opinion -- the same
argument split_cf_items.py's docstring makes for why two endpoints must not share anchors.
The 22 are already spent; buying them again buys nothing.

WHY A NEW FILE AND NOT AN EDIT OF THE SLICE. `cf_items_70k_oai.jsonl` is what
`split_cf_items.py --check` validates every night, and the nightly scripts refuse to run if
that check fails. Rewriting it in place would either break the check or, worse, quietly
change what the check is checking. This writes a SEPARATE file that is a strict subset,
copied LINE FOR LINE so every `custom_id` means exactly what it meant in the slice --
subsetting preserves ids, re-sampling does not (§7.5, and split_cf_items.py's docstring).

RUN IT ONCE, BEFORE A CAMPAIGN'S FIRST NIGHT. Re-running it later is safe but pointless: once
the campaign is live its own `.responses.jsonl` is what `--resume` reads, and shrinking the
items file underneath a running campaign only makes the "N to go" counters harder to read.
The exception is a second handover -- if Codex ever restarts inside the Gemini slice, `--check`
will fail (correctly) and the file has to be re-cut before either side runs again.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

# The OpenAI run's, because it was the first caller -- but UPDATED 2026-08-21 to the union
# slice it reads now. Leaving them at the exhausted `cf_items_70k_oai.jsonl` ->
# `cf_items_70k_oai_luna.jsonl` pair would mean a bare `python3 scripts/cf_exclude_generated.py`
# silently rebuilt the file the campaign STOPPED reading, and reported success doing it.
SLICE = "data/cf/cf_items_oai_src.jsonl"
OUT = "data/cf/cf_items_oai.jsonl"
# Every response file in data/cf, EXCEPT the one this campaign will itself write -- that one is
# `--resume`'s job, and reading it here would make the items file shrink every night.
RESPONSES_GLOB = "data/cf/*responses.jsonl"
SELF = "data/cf/cf70k_oai.responses.jsonl"


def generated_ids(glob_pat: str, exclude: str) -> dict[str, str]:
    """custom_id -> the file that already holds a NON-NULL response for it.

    Non-null is the same test `_already_done` uses. A row with `response: null` is a FAILED
    request, not a generated anchor: it cost nothing usable and must stay eligible.
    """
    seen: dict[str, str] = {}
    for path in sorted(glob.glob(glob_pat)):
        if Path(path).resolve() == Path(exclude).resolve():
            continue
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue        # half-written final line; same tolerance as _already_done
                if row.get("response") is not None and row.get("custom_id"):
                    seen.setdefault(row["custom_id"], Path(path).name)
    return seen


def check(slice_path: Path, out_path: Path, burned: dict[str, str]) -> int:
    if not out_path.exists():
        print(f"[exclude] {out_path} does not exist -- run this script with no --check first",
              file=sys.stderr)
        return 1
    slice_ids = [json.loads(l)["custom_id"] for l in slice_path.open() if l.strip()]
    out_ids = [json.loads(l)["custom_id"] for l in out_path.open() if l.strip()]
    bad = 0
    if len(set(out_ids)) != len(out_ids):
        print("[exclude] duplicate custom_ids in the output", file=sys.stderr)
        bad = 1
    if not set(out_ids) <= set(slice_ids):
        stray = sorted(set(out_ids) - set(slice_ids))[:5]
        print(f"[exclude] output is NOT a subset of the slice, e.g. {stray}", file=sys.stderr)
        bad = 1
    leaked = sorted(set(out_ids) & set(burned))
    if leaked:
        print(f"[exclude] {len(leaked)} already-generated anchors are still in the items "
              f"file, e.g. {leaked[:5]}", file=sys.stderr)
        bad = 1
    if bad:
        return 1
    print(f"[exclude] OK  {out_path.name} {len(out_ids):,} anchors "
          f"({out_ids[0]}..{out_ids[-1]}), a strict subset of {slice_path.name} "
          f"({len(slice_ids):,})")
    print(f"[exclude] {len(set(slice_ids) & set(burned))} already-generated anchors held out; "
          f"none of them survive into the items file")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slice", default=SLICE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--responses-glob", default=RESPONSES_GLOB)
    ap.add_argument("--self", dest="self_path", default=SELF)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    slice_path, out_path = Path(args.slice), Path(args.out)
    if not slice_path.exists():
        print(f"[exclude] {slice_path} does not exist -- run split_cf_items.py first",
              file=sys.stderr)
        return 1

    burned = generated_ids(args.responses_glob, args.self_path)
    if args.check:
        return check(slice_path, out_path, burned)

    kept, dropped = [], []
    with slice_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            cid = json.loads(line)["custom_id"]
            (dropped if cid in burned else kept).append((cid, line))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(line for _, line in kept))    # line for line, ids untouched
    print(f"[exclude] {len(kept) + len(dropped):,} anchors in {slice_path.name} -> "
          f"{len(kept):,} to generate, {len(dropped):,} held out as already generated")
    for cid, _ in dropped[:25]:
        print(f"[exclude]   held out {cid}  (already in {burned[cid]})")
    if len(dropped) > 25:
        print(f"[exclude]   ... and {len(dropped) - 25} more")
    return check(slice_path, out_path, burned)


if __name__ == "__main__":
    raise SystemExit(main())
