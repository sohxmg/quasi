#!/usr/bin/env python3
"""Generate counterfactual rewrites through the Codex CLI instead of an HTTP endpoint.

    python3 scripts/generate_via_codex.py --items data/cf/cf_items_70k_cx.jsonl \
        --out data/cf/cf70k_cx.jsonl --concurrency 4 --resume --stop-after 7.5

WHY THIS EXISTS. gpt-5-nano at `reasoning_effort=low` is the wrong model for this workload and
the pilot says so: 29 of 52 discards are positives that CHANGED THE RESULT, and only 2 of those
29 are symbolically equal to the anchor (MEASURED 2026-08-10, sympy over
`cf_oai_pilot_low.discarded.jsonl`). The rewrites are genuinely wrong, not merely spelled
differently, so the lever is model capability and not the validator. `medium` cannot be used to
buy that capability -- it spends the entire 8,000-token budget on reasoning and returns empty
content, 0 of 2 items (§7.5.7). A stronger model reached through a subscription CLI is the only
way to raise the ceiling without raising the bill.

WHY ONE-SHOT `codex exec` AND NOT SPAWNED AGENTS. An agent is a tool loop over a workspace, and
this task has no workspace: the prompt is fixed, the schema is fixed, nothing is read and
nothing is written. Every anchor is independent, so an agent per anchor pays for a session, a
context and a tool loop to do one completion. `codex exec` IS the one-shot form, and running N
of them under a thread pool is the same parallelism a spawned fleet would give with none of the
per-agent overhead and -- the part that matters -- a return value this script controls the
shape of rather than a report written in prose.

WHAT IT WRITES. `<out>.responses.jsonl`, in the SAME envelope `generate` writes, so validation
is the existing code path and costs nothing:

    python3 scripts/generate_counterfactuals.py generate \
        --items <the same items file> --out <out> --replay <out>.responses.jsonl

That separation is the point. This file is only a transport; every judgement about whether a
rewrite is usable stays in `validate()`, where the other two endpoints are judged, so the three
datasets remain comparable.

THE PROMPT IS BYTE-IDENTICAL to the HTTP path -- `SYSTEM_PROMPT` and `user_message` are
imported, not copied. A reworded prompt here would make a yield difference between endpoints
unreadable: model, or wording?
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_counterfactuals import (  # noqa: E402
    SYSTEM_PROMPT,
    _already_done,
    read_items,
    response_schema,
    user_message,
)


def build_prompt(item: dict, positives: int, negatives: int) -> str:
    """The HTTP path sends two messages; `codex exec` takes one string. Same bytes, joined."""
    return f"{SYSTEM_PROMPT}\n\n{user_message(item, positives, negatives)}"


def codex_argv(args: argparse.Namespace, cwd: str, schema: str, last: str) -> list[str]:
    """Every flag here was read out of `codex exec --help` on codex-cli 0.147.0, not guessed.

    `-C <cwd>` points every worker at its OWN EMPTY DIRECTORY. Two reasons, and the second is
    the one that is easy to miss. (a) Concurrent agents sharing a working directory overwrite
    each other's files -- the standard worktree hazard. It cannot bite here because the task
    writes nothing, but "cannot bite" resting on a sandbox flag being spelled correctly is a
    coincidence, not a property; one empty dir per item makes it a property.
    (b) A worker started inside THIS repo can read it: `data/cf/`, this file, `CLAUDE.md`. The
    prompt asks for JSON and nothing else, so every turn spent looking around is latency and
    quota spent on nothing, and a model that finds the anchor's own trajectory on disk is being
    handed the answer it is supposed to derive.

    `--output-schema` is the CLI's equivalent of the HTTP path's `response_format:
    {type: json_schema}`, and it is the SAME `response_schema()` -- imported, not restated, so
    the three endpoints cannot drift into validating different shapes. `negotiate_response_
    format` exists on the HTTP side because a weak endpoint may refuse the schema; here it is
    a local file and either the CLI accepts it or the run stops on request one.

    `-o/--output-last-message` writes the final message to a file, so the JSON is read from
    somewhere the CLI's own progress output cannot reach. `extract_json` would cope with
    preamble on stdout, but "cope" is the wrong contract for 3,000 paid requests.

    `--ephemeral` stops each anchor persisting a session under `~/.codex/sessions`. At 3,000
    anchors that is 3,000 session directories recording a task with no history worth resuming.

    `--sandbox read-only` is belt to `-C`'s braces.
    """
    argv = [args.codex_bin, "exec", "-C", cwd, "--skip-git-repo-check",
            "--sandbox", "read-only", "--ephemeral",
            "--output-schema", schema, "-o", last, "--color", "never"]
    if args.model:
        argv += ["-m", args.model]
    return argv + args.codex_arg


def one(item: dict, args: argparse.Namespace) -> dict:
    """Run one anchor. Returns a row in `generate`'s response envelope, success or not.

    A failure row carries `response: null` and a status naming the cause, because that is what
    `_already_done` keys on: a null response is NOT done, so `--resume` retries it, and a later
    success supersedes it in `_stream_responses`. Writing a fabricated empty success here would
    make the item permanently skipped and silently absent from the dataset -- a guard failing
    toward healthy (§14 B11/B12).
    """
    prompt = build_prompt(item, args.positives, args.negatives)
    try:
        # One empty directory per anchor, removed on the way out -- see codex_argv. It costs a
        # mkdir against a request that takes tens of seconds, which is why this is per-item
        # rather than one shared dir: nothing has to reason about whether sharing is safe.
        with tempfile.TemporaryDirectory(prefix="cf-codex-") as cwd:
            schema = Path(cwd) / "schema.json"
            schema.write_text(json.dumps(response_schema()))
            last = Path(cwd) / "last.txt"
            proc = subprocess.run(
                codex_argv(args, cwd, str(schema), str(last)),
                input=prompt,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                cwd=cwd,
            )
            # Read INSIDE the context manager: the directory goes away on exit.
            written = last.read_text() if last.exists() else ""
    except subprocess.TimeoutExpired:
        return {"custom_id": item["custom_id"], "response": None, "status": "codex_timeout"}
    except FileNotFoundError:
        raise SystemExit(f"[codex] {args.codex_bin!r} not found on PATH")
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
        return {
            "custom_id": item["custom_id"],
            "response": None,
            "status": f"codex_exit_{proc.returncode}",
            "stderr": tail[0][:500],
        }
    # Prefer the `-o` file; fall back to stdout so a CLI that drops the flag still works rather
    # than reporting every item empty -- `extract_json` handles preamble either way.
    text = written or proc.stdout
    if not text.strip():
        return {"custom_id": item["custom_id"], "response": None, "status": "codex_empty"}
    return {
        "custom_id": item["custom_id"],
        "response": {
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {},
        },
        "status": "ok",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--positives", type=int, default=3)
    p.add_argument("--negatives", type=int, default=6)
    p.add_argument("--concurrency", type=int, default=4,
                   help="Codex sessions are heavy and subscription-rate-limited; start low.")
    p.add_argument("--timeout", type=float, default=300.0, help="seconds per anchor")
    p.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN", "codex"))
    p.add_argument("--model", default="", help="passed to `codex exec -m`; empty = its default")
    p.add_argument("--codex-arg", action="append", default=[],
                   help="extra argv for `codex exec`, repeatable")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=float, default=0.0,
                   help="hours; stops SUBMITTING then drains what is in flight")
    p.add_argument("--limit", type=int, default=0, help="first N pending anchors; 0 = all")
    args = p.parse_args()

    items = read_items(args.items)
    responses = Path(f"{args.out}.responses.jsonl")
    done = _already_done(responses) if args.resume else set()
    pending = [it for it in items if it["custom_id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"[codex] {len(items):,} anchors, {len(done):,} already done, "
          f"{len(pending):,} to run at concurrency {args.concurrency}")
    if not pending:
        return 0

    responses.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.stop_after * 3600 if args.stop_after else 0.0
    lock = threading.Lock()
    start = time.time()
    counts = {"ok": 0, "fail": 0}

    # Append-and-flush per response, same contract as `generate`: a run killed at any instant
    # loses at most the row in flight, and `--resume` picks up from the file rather than from
    # anything held in memory.
    with responses.open("a") as fh, ThreadPoolExecutor(args.concurrency) as pool:
        futures = {}
        for item in pending:
            if deadline and time.time() > deadline:
                print("[codex] --stop-after reached; draining what is in flight")
                break
            futures[pool.submit(one, item, args)] = item
            # A bounded submission window keeps --stop-after honest: an unbounded submit loop
            # queues every remaining anchor up front and the deadline can only stop a queue
            # that is still being fed.
            while len(futures) >= args.concurrency * 2:
                _drain(futures, fh, lock, counts, start, len(pending))
        while futures:
            _drain(futures, fh, lock, counts, start, len(pending))

    print(f"[codex] {counts['ok']:,} ok / {counts['fail']:,} failed in "
          f"{time.time() - start:.0f}s -> {responses}")
    print(f"[codex] validate for free:\n"
          f"  python3 scripts/generate_counterfactuals.py generate \\\n"
          f"    --items {args.items} --out {args.out} --replay {responses}")
    return 0


def _drain(futures: dict, fh, lock, counts: dict, start: float, total: int) -> None:
    """Retire every future that has finished, writing each row as it is retired."""
    done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
    for fut in done:
        futures.pop(fut, None)
        row = fut.result()
        with lock:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            counts["ok" if row["response"] is not None else "fail"] += 1
            n = counts["ok"] + counts["fail"]
            if n % 5 == 0 or n == total:
                rate = n / max(time.time() - start, 1e-9) * 60
                print(f"[codex] {n}/{total}  {rate:.1f}/min  {counts['fail']} failed")


if __name__ == "__main__":
    raise SystemExit(main())
