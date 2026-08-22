"""Bring `runs/cf_lam2_tau005/` home, put the analysis files on GitHub, then destroy the volumes.

    python modal_cf_lam2_tau005_finish.py             # fetch -> verify -> commit+push -> delete
    python modal_cf_lam2_tau005_finish.py --no-delete # everything except the delete
    python modal_cf_lam2_tau005_finish.py --verify    # verify what is already local, touch nothing

The ORDER is the whole point of this file and it is enforced rather than documented: the
volumes are deleted only after (a) every required artifact is verified present on this
machine and (b) `git push` has actually succeeded. Either check failing means the delete does
not happen and the volumes are still there to try again. A $9 run with no output is the
failure mode this is written against.

WHAT GOES TO GITHUB, AND WHAT DOES NOT. `runs/` is in .gitignore, so the analysis files are
force-added by explicit path -- never `git add runs/`. Weights are NOT pushed: heads.pt is
16 MB per checkpoint, phase2/cache.pt is 409 MB, and there are six checkpoints. All of it is
downloaded and stays on this machine; only the ~1.5 MB that answers "how good was the model"
goes to the remote.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import modal

RUN_NAME = "cf_lam2_tau005"
VOLUME = "feynman-runs"
VOLUME_NAMES = ["feynman-hf-cache", "feynman-processed", "feynman-runs"]
REPO_ROOT = Path(__file__).parent.resolve()

# Phase 1 alone must produce these.
REQUIRED_PHASE1 = [
    "config.resolved.yaml",
    "events.jsonl",
    "metrics.jsonl",
    "final/heads.pt",
    "final/config.yaml",
    "final/val_f1.json",
]

# ProcessBench -- the eval that was asked for -- must produce these.
REQUIRED_EVAL = [
    "phase2/final/processbench.json",
    "phase2/final/heads.pt",
]

# Small, textual, and the only thing that answers "how good was it". Missing entries are
# skipped, so a val-F1-gated run still commits what it has.
COMMIT_PATHS = [
    "config.resolved.yaml",
    "events.jsonl",
    "metrics.jsonl",
    "final/config.yaml",
    "final/val_f1.json",
    "phase2/events.jsonl",
    "phase2/metrics.jsonl",
    "phase2/final/processbench.json",
    "phase2/final/deltas.npz",
    "RESULT.md",
]


def download(dest_root: Path) -> int:
    vol = modal.Volume.from_name(VOLUME)
    dest = dest_root / RUN_NAME
    dest.mkdir(parents=True, exist_ok=True)

    total = 0
    for entry in vol.listdir(f"/{RUN_NAME}", recursive=True):
        if getattr(entry.type, "value", entry.type) != 1:   # 1 == FILE
            continue
        rel = entry.path.lstrip("/")
        if rel.startswith(f"{RUN_NAME}/"):
            rel = rel[len(RUN_NAME) + 1:]
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)

        size = getattr(entry, "size", None)
        if out.exists() and size is not None and out.stat().st_size == size:
            print(f"  = {rel}  ({size / 1e6:.2f} MB, already local)")
            total += size
            continue

        with out.open("wb") as fh:
            for chunk in vol.read_file(entry.path):
                fh.write(chunk)
        got = out.stat().st_size
        total += got
        print(f"  < {rel}  ({got / 1e6:.2f} MB)")

    return total


def summarise(dest: Path) -> dict:
    """The headline numbers, read off the artifacts rather than off the log stream."""
    out: dict = {"run": RUN_NAME}

    val = dest / "final/val_f1.json"
    if val.exists():
        d = json.loads(val.read_text())
        out["val_f1_at_best_tau"] = d.get("at_best_tau", {}).get("f1")
        out["val_f1_tau"] = d.get("tau")

    pb = dest / "phase2/final/processbench.json"
    if pb.exists():
        d = json.loads(pb.read_text())
        subsets = {
            k: v["f1"] for k, v in d.items()
            if isinstance(v, dict) and "f1" in v
            and k != "calibration" and not k.endswith("SKYLINE_not_a_result")
        }
        out["processbench"] = {k: round(v, 4) for k, v in subsets.items()}
        if subsets:
            out["processbench_mean_f1"] = round(sum(subsets.values()) / len(subsets), 4)
        out["processbench_tau"] = d.get("calibration", {}).get("calibration/tau")
        math = d.get("math", {}).get("leak_split")
        if math:
            out["math_clean_f1"] = round(math["clean"]["f1"], 4)
            out["math_leaked_f1"] = round(math["leaked"]["f1"], 4)

    ev = dest / "events.jsonl"
    if ev.exists():
        for line in ev.read_text().splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("event") == "wandb/run":
                out["wandb_url"] = rec["url"]
    return out


def write_result_md(dest: Path, summary: dict) -> None:
    """A human-readable header for the commit, next to the JSON it is read from."""
    pb = summary.get("processbench", {})
    lines = [
        f"# {RUN_NAME}",
        "",
        "```",
        "bash scripts/train_cloud.sh --set losses.zeta=0.2 \\",
        "    --set losses.nce_temperature=22.627417 \\",
        "    --set losses.lambda_term=0.0 \\",
        "    --set losses.lambda_cf=2.0 \\",
        "    --set losses.lambda_cf_temperature=0.05 \\",
        "    --set run.name=cf_lam2_tau005",
        "```",
        "",
        "A100 40 GB on Modal, PROFILE=match (batch shape is `config/default.yaml`'s, so every",
        "loss statistic is comparable to the other 56-sequence runs).",
        "",
        "TWO-DELTA against `runs/abl_cf_only`, which is identical on every other axis",
        "(zeta 0.2, lambda_term 0.0, nce_temperature sqrt(512)). Both deltas are on (4) L_CF:",
        "lambda_cf 1.0 -> 2.0 and lambda_cf_temperature 0.1 -> 0.05. Per the config, tau and",
        "lambda are the same knob (gradient ~ lambda/tau), so the EFFECTIVE weight on (4) is",
        "4x the baseline's, not 2x -- read any L_CF difference against that, not against 2x.",
        "",
        "`lambda_cf_temperature = 0.05` is the value `config/default.yaml` explicitly rejected",
        "as saturating. THE TELL IS `cf/loss - cf/chance`, never raw `cf/loss`: near -0.02 the",
        "change did nothing, -1.09 or beyond means tau overshot and 0.2 is the fallback.",
        "",
        "## Phase-1 ceiling (held-out Math-Shepherd val)",
        "",
        f"- val F1 at best tau: **{summary.get('val_f1_at_best_tau')}**  (tau = {summary.get('val_f1_tau')})",
        "",
        "## ProcessBench",
        "",
        f"- tau = {summary.get('processbench_tau')}",
        "",
        "| subset | F1 |",
        "|---|---|",
    ]
    for k, v in pb.items():
        lines.append(f"| {k} | {v} |")
    lines += [
        f"| **mean** | **{summary.get('processbench_mean_f1')}** |",
        "",
    ]
    if "math_clean_f1" in summary:
        lines += [
            f"Math subset, leak split: clean **{summary['math_clean_f1']}**, "
            f"leaked {summary['math_leaked_f1']}.",
            "",
        ]
    lines += [
        "Skyline rows in `processbench.json` are labelled and are never a reported result",
        "(the gold answer alone solves half the metric).",
        "",
    ]
    if summary.get("wandb_url"):
        lines += [f"wandb: {summary['wandb_url']}", ""]
    lines += [
        "## What is here and what is not",
        "",
        "Committed: the resolved config, both metrics/events streams, `val_f1.json`,",
        "`processbench.json`, `deltas.npz`. NOT committed: the LoRA adapters, `heads.pt`,",
        "the tokenizers and `phase2/cache.pt` -- ~1 GB of weights that live on the box the",
        "run was fetched to.",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
    ]
    (dest / "RESULT.md").write_text("\n".join(lines))


def verify(dest_root: Path) -> tuple[bool, bool]:
    """(phase-1 ok, eval ok)."""
    dest = dest_root / RUN_NAME
    missing_p1 = [a for a in REQUIRED_PHASE1 if not (dest / a).exists()]
    missing_eval = [a for a in REQUIRED_EVAL if not (dest / a).exists()]

    print(f"\n--- verify {dest} ---")
    if missing_p1:
        print(f"  MISSING (phase 1): {missing_p1}")
        print("  Phase 1 itself did not come back intact. NOT SAFE TO DELETE.")
        return False, False
    print(f"  phase 1 artifacts: all {len(REQUIRED_PHASE1)} present")

    ckpts = sorted(p.name for p in dest.glob("step*") if p.is_dir())
    print(f"  mid-run checkpoints: {len(ckpts)} {ckpts}")

    if missing_eval:
        print(f"  missing (ProcessBench): {missing_eval}")
        print("  Either the chain is still running, or the val-F1 gate stopped it.")
        print("  Check `modal app list` for a RUNNING app before concluding anything.")
        return True, False
    print(f"  ProcessBench artifacts: all {len(REQUIRED_EVAL)} present")
    return True, True


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ git {' '.join(args)}")
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=check,
                          capture_output=True, text=True)


def commit_and_push(dest_root: Path, summary: dict) -> bool:
    """Force-add the analysis files (runs/ is gitignored) and push. True only on a real push."""
    dest = dest_root / RUN_NAME
    paths = [f"runs/{RUN_NAME}/{p}" for p in COMMIT_PATHS if (dest / p).exists()]
    if not paths:
        print("  nothing to commit")
        return False

    print(f"\n--- committing {len(paths)} files ---")
    for p in paths:
        print(f"  + {p}  ({(REPO_ROOT / p).stat().st_size / 1e6:.2f} MB)")

    # -f because runs/ is in .gitignore. Only these explicit paths -- never `git add runs/`,
    # which would stage a gigabyte of weights.
    _git("add", "-f", *paths)

    staged = _git("diff", "--cached", "--name-only").stdout.strip()
    if not staged:
        print("  already committed, nothing staged")
    else:
        f1 = summary.get("processbench_mean_f1")
        val = summary.get("val_f1_at_best_tau")
        msg = (
            f"{RUN_NAME}: ProcessBench mean F1 {f1}, phase-1 val F1 {val}\n\n"
            "zeta=0.2, nce_temperature=22.627417, lambda_cf=2.0, "
            "lambda_cf_temperature=0.05, lambda_term=0.0.\n"
            "Ran on a Modal A100 40 GB at PROFILE=match. Analysis files only -- the\n"
            "adapters, heads and phase2/cache.pt stayed on the fetching machine.\n"
        )
        _git("commit", "-m", msg)

    branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    push = _git("push", "origin", branch, check=False)
    print(push.stdout + push.stderr)
    if push.returncode != 0:
        print("  PUSH FAILED. The volumes are NOT being deleted.")
        return False

    # Belt and braces: confirm the remote actually has this commit before anything is
    # destroyed. A push that "succeeded" into the wrong branch is not a backup.
    local = _git("rev-parse", "HEAD").stdout.strip()
    remote = _git("rev-parse", f"origin/{branch}", check=False).stdout.strip()
    if local != remote:
        print(f"  local {local[:8]} != origin/{branch} {remote[:8]}. NOT deleting.")
        return False
    print(f"  pushed: origin/{branch} @ {local[:8]}")
    return True


def delete_volumes() -> None:
    print("\n--- deleting volumes ---")
    for v in VOLUME_NAMES:
        r = subprocess.run([sys.executable, "-m", "modal", "volume", "delete", v, "--yes"],
                           capture_output=True, text=True)
        print(f"  {v}: {'deleted' if r.returncode == 0 else r.stderr.strip()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="verify local files, change nothing")
    ap.add_argument("--no-delete", action="store_true", help="fetch + commit, keep the volumes")
    args = ap.parse_args()

    dest_root = REPO_ROOT / "runs"
    dest = dest_root / RUN_NAME

    if not args.verify:
        print(f"=== downloading {VOLUME}:/{RUN_NAME} -> {dest} ===")
        total = download(dest_root)
        print(f"\n{total / 1e6:.1f} MB total")

    p1_ok, eval_ok = verify(dest_root)
    if not p1_ok:
        return 1

    summary = summarise(dest)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))

    if args.verify:
        return 0 if eval_ok else 1

    write_result_md(dest, summary)
    pushed = commit_and_push(dest_root, summary)

    if not eval_ok:
        print("\nProcessBench did not come back. Volumes KEPT -- the chain may still be live.")
        return 1
    if not pushed:
        print("\nPush did not land. Volumes KEPT.")
        return 1
    if args.no_delete:
        print("\n--no-delete: volumes kept.")
        return 0

    delete_volumes()
    print("\nDone. GPU released when the container exited; volumes are gone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
