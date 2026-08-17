#!/usr/bin/env python3
"""Split the 70k anchor file into per-endpoint slices that CANNOT overlap.

    python3 scripts/split_cf_items.py                    # build the slices, then verify
    python3 scripts/split_cf_items.py --check            # verify only, touch nothing

WHY A SPLIT AND NOT TWO SAMPLES. Two endpoints generating the same anchor is money spent
twice for one training row -- `L_CF` keys on the anchor, so the second copy is a duplicate,
not a second opinion. Re-running `sample` to get a second file does NOT prevent that: `sample`
draws from the same 21,750-question pool, and its `cf%06d` ids are assigned BY POSITION, so
two independently sampled files collide on ids while disagreeing about what those ids mean.
Slicing one file preserves every id exactly as generated and makes the disjointness a property
of the arithmetic rather than of a seed.

WHY THE TAIL GOES TO THE PAID ENDPOINT. `generate --resume` walks the file IN ORDER, so the
front is what a long-running campaign consumes first. bharatcode gets the front (~34 nights of
it at the measured 4.53 ok/min, §7.5.6) and the small paid slice comes off the back, where the
free campaign would not arrive for months even if the split were removed tomorrow.

THIS RUNS ONCE, BEFORE NIGHT 1, AND NEVER AGAIN. Re-slicing mid-campaign with a different
`--openai-anchors` moves the boundary, and every id that crosses it is one an endpoint has
already paid for and will not skip -- `--resume` matches on `custom_id` within one output
file and cannot see the other's. `--check` is safe at any time and is what the nightly
scripts call.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE = "data/cf/cf_items_70k.jsonl"
# **THE BACK 20,000 OF THIS SLICE ARE DRIVEN BY GEMINI SINCE 2026-08-14** (cf027000..cf046999),
# because bharatcode stopped being run 2,201 anchors in. THE SLICE FILE DID NOT MOVE and must
# not -- it is still what `--check` validates, and both halves are a line-for-line cut of it,
# so every custom_id is unchanged. bharatcode reads `cf_items_70k_bc_keep.jsonl`
# (cf000000..cf026999) and Gemini reads the tail as part of `cf_items_gm.jsonl`. The cut is off
# the BACK for the same reason the OpenAI slice is: `--resume` walks in order, so the front is
# what bharatcode consumes first and the two campaigns cannot meet. `cf_nightly.sh` re-proves
# the cut against Gemini's items file every night.
BHARATCODE = "data/cf/cf_items_70k_bc.jsonl"
CODEX = "data/cf/cf_items_70k_cx.jsonl"
OPENAI = "data/cf/cf_items_70k_oai.jsonl"

# 3,000 anchors against a 2,000,000-token grant that buys ~500 of them at ~4,000 tokens each
# (§7.5.6). The file is deliberately the SLACK and the budget the CONSTRAINT: an items file
# sized exactly to one grant would have to be re-sliced -- the one operation this script says
# not to do -- the moment a second grant arrives.
OPENAI_ANCHORS = 3000

# 20,000 for the Codex CLI, MEASURED 2026-08-10 and sized from the measurement rather than
# from a guess: 18.5 ok/min at concurrency 24 is ~8,300 anchors in a 7.5 h night, so a slice
# that a good night could exhaust is a slice that has to be re-cut -- the one operation the
# docstring above says not to do. Same logic as OPENAI_ANCHORS: THE FILE IS THE SLACK. What
# bounds a Codex night is `--stop-after`, and what bounds the campaign is how many nights
# there are.
#
# It comes out of BHARATCODE's tail and NOT out of the OpenAI slice, whose ids
# (cf067000..cf069999) must not move -- that campaign is budgeted and about to run. bharatcode
# needs ~34 nights to reach cf047000 at its measured 4.53 ok/min, so the anchors handed over
# here are ones it would not have touched for a month.
#
# **THE CODEX SLICE IS DRIVEN BY GEMINI SINCE 2026-08-10** (`scripts/cf_nightly_gemini.sh`),
# because Codex went down 1,440 anchors in. The SLICE DID NOT MOVE and must not -- re-cutting
# is the one operation this file says not to do, and the boundary is what keeps bharatcode and
# OpenAI out of it. What changed is only which endpoint reads it, and Gemini reads
# `cf_items_70k_cx_gm.jsonl`: this slice minus the 1,440 Codex already generated, built by
# `cf_exclude_generated.py`. If Codex comes back it needs a slice of its own or a fresh
# handover -- restarting it here would put it back on anchors Gemini is now working through.
CODEX_ANCHORS = 20000


def custom_ids(path: Path) -> list[str]:
    with path.open() as fh:
        return [json.loads(line)["custom_id"] for line in fh if line.strip()]


def check(*slices: Path) -> int:
    """Every slice against every other. PAIRWISE, not against a running union -- a union check
    reports `A & (B|C)` and cannot say WHICH pair overlaps, and on three files that is the only
    thing worth knowing."""
    missing = [p for p in slices if not p.exists()]
    if missing:
        print(f"[split] missing slice(s): {', '.join(str(p) for p in missing)}",
              file=sys.stderr)
        return 1
    ids = {p: custom_ids(p) for p in slices}
    bad = 0
    for p, vals in ids.items():
        if len(set(vals)) != len(vals):
            print(f"[split] {p.name} contains duplicate custom_ids", file=sys.stderr)
            bad = 1
        print(f"[split] OK  {p.name} {len(vals):,} anchors ({vals[0]}..{vals[-1]})")
    for i, p in enumerate(slices):
        for q in slices[i + 1:]:
            shared = set(ids[p]) & set(ids[q])
            if shared:
                print(f"[split] OVERLAP: {p.name} and {q.name} share {len(shared):,} "
                      f"custom_ids, e.g. {sorted(shared)[:5]}", file=sys.stderr)
                bad = 1
    if bad:
        return 1
    print(f"[split] the {len(slices)} slices share no custom_id")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--bharatcode-out", default=BHARATCODE)
    ap.add_argument("--codex-out", default=CODEX)
    ap.add_argument("--openai-out", default=OPENAI)
    ap.add_argument("--codex-anchors", type=int, default=CODEX_ANCHORS)
    ap.add_argument("--openai-anchors", type=int, default=OPENAI_ANCHORS)
    ap.add_argument("--check", action="store_true", help="verify the slices, write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite existing slices")
    args = ap.parse_args(argv)

    bc = Path(args.bharatcode_out)
    cx = Path(args.codex_out)
    oai = Path(args.openai_out)
    if args.check:
        return check(bc, cx, oai)

    source = Path(args.source)
    if not source.exists():
        print(f"[split] {source} does not exist -- build it once with `sample` first",
              file=sys.stderr)
        return 1
    live = [p for p in (bc, cx, oai) if p.exists()]
    # The `--out` each nightly script passes, spelled out rather than derived from the item
    # name: the bharatcode pair is `cf_items_70k_bc.jsonl` -> `cf70k.jsonl` and does NOT follow
    # the pattern the other two do. A clever transform here would look right and check the
    # wrong file, which on this guard means reporting "nothing spent" over a live campaign.
    outs = {bc: "data/cf/cf70k.jsonl",
            cx: "data/cf/cf70k_cx.jsonl",
            oai: "data/cf/cf70k_oai.jsonl"}
    spent = [p for p in (bc, cx, oai) if Path(f"{outs[p]}.responses.jsonl").exists()]
    if spent and not args.force:
        print(f"[split] REFUSING: responses already exist for {', '.join(p.name for p in spent)}"
              f". Re-slicing moves the boundary under a campaign that has already PAID for ids "
              f"on the other side of it, and `--resume` matches on custom_id within one output "
              f"file and cannot see the others. Re-cutting here silently drops or duplicates "
              f"every id that crosses.", file=sys.stderr)
        return check(bc, cx, oai)
    if live and not args.force:
        print(f"[split] {', '.join(p.name for p in live)} already exist and no responses were "
              f"found beside them. Pass --force to re-cut.", file=sys.stderr)
        return check(bc, cx, oai)

    lines = source.read_text().splitlines(keepends=True)
    lines = [line for line in lines if line.strip()]
    paid = args.codex_anchors + args.openai_anchors
    if paid >= len(lines):
        print(f"[split] --codex-anchors + --openai-anchors = {paid:,} >= {len(lines):,} anchors",
              file=sys.stderr)
        return 1
    # OpenAI keeps the LAST `openai_anchors` lines, byte for byte, so cf067000..cf069999 mean
    # exactly what they meant before Codex existed. Codex takes the block in front of it.
    oai_cut = len(lines) - args.openai_anchors
    cx_cut = oai_cut - args.codex_anchors
    bc.parent.mkdir(parents=True, exist_ok=True)
    bc.write_text("".join(lines[:cx_cut]))
    cx.write_text("".join(lines[cx_cut:oai_cut]))
    oai.write_text("".join(lines[oai_cut:]))
    print(f"[split] {len(lines):,} anchors -> {cx_cut:,} bharatcode + "
          f"{args.codex_anchors:,} codex + {args.openai_anchors:,} openai")
    return check(bc, cx, oai)


if __name__ == "__main__":
    raise SystemExit(main())
