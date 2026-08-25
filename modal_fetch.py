"""Pull `runs/nce_tau4/` off the Modal volume onto this machine, then say whether it is safe
to delete the volumes.

    python modal_fetch.py              # download + verify
    python modal_fetch.py --verify     # verify what is already local, download nothing

Run from the repo root. This is deliberately a SEPARATE script from `modal_app.py` rather than
a step inside its local entrypoint: the training function is launched with `modal run --detach`
so that a dropped shell cannot kill four paid hours, and a detached launch means the local
entrypoint's post-processing never runs. Downloading has to be something that can be run
later, and more than once, against a run that has already finished.

EVERYTHING under the run directory comes back, mid-run `step*/` checkpoints included. They are
~100 MB each and there are five or six of them, which is nothing against a volume that is
about to be destroyed -- and §9.7's rank tables were measured on a `step750`, so "only final/
matters" is empirically false for this project.

Nothing here deletes anything. The delete commands are PRINTED, and only when every required
artifact is verified present locally, because the ordering is the only thing between a $9 run
and a $9 run with no output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import modal

RUN_NAME = "nce_tau4"
VOLUME = "feynman-runs"
VOLUME_NAMES = ["feynman-hf-cache", "feynman-processed", "feynman-runs"]

# Phase 1 alone must produce these. If the val-F1 gate in modal_app.py stopped the chain,
# phase 2 and ProcessBench legitimately will not exist -- so they are checked separately.
REQUIRED_PHASE1 = [
    "config.resolved.yaml",
    "events.jsonl",
    "metrics.jsonl",
    "final/heads.pt",
    "final/config.yaml",
    "final/val_f1.json",
]

REQUIRED_FULL = [
    "phase2/final/processbench.json",
    "phase2/final/deltas.npz",
    "phase2/events.jsonl",
    "final/goal_gate_mask.txt",
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


def verify(dest_root: Path) -> bool:
    dest = dest_root / RUN_NAME
    missing_p1 = [a for a in REQUIRED_PHASE1 if not (dest / a).exists()]
    missing_full = [a for a in REQUIRED_FULL if not (dest / a).exists()]

    print(f"\n--- verify {dest} ---")
    if missing_p1:
        print(f"  MISSING (phase 1): {missing_p1}")
        print("  NOT SAFE TO DELETE. Phase 1 itself did not come back intact.")
        return False
    print(f"  phase 1 artifacts: all {len(REQUIRED_PHASE1)} present")

    ckpts = sorted(p.name for p in dest.glob("step*") if p.is_dir())
    print(f"  mid-run checkpoints: {len(ckpts)} {ckpts}")

    if missing_full:
        # Phase 1 being complete is NOT a licence to delete. The chain runs phase 2 and the
        # evals in the same container afterwards, so "phase 1 present, phase 2 absent" is the
        # normal state for the ~50 minutes the run is still going -- and deleting there would
        # destroy the volume out from under a live job. Only two things distinguish that from
        # a legitimately-short chain, and neither can be guessed, so this refuses instead.
        print(f"\n  missing (phase 2 / eval): {missing_full}")
        print("  NOT SAFE TO DELETE YET. Either the chain is still running, or the val-F1")
        print("  gate stopped it. Check `modal app list` for a RUNNING app, and check")
        print("  final/val_f1.json's at_best_tau.f1 against the 0.35 floor. Re-run this")
        print("  script once the app has stopped.")
        return False

    print(f"  phase 2 + eval artifacts: all {len(REQUIRED_FULL)} present")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="verify local files, download nothing")
    args = ap.parse_args()

    dest_root = Path(__file__).parent.resolve() / "runs"

    if not args.verify:
        print(f"=== downloading {VOLUME}:/{RUN_NAME} -> {dest_root / RUN_NAME} ===")
        total = download(dest_root)
        print(f"\n{total / 1e6:.1f} MB total")

    ok = verify(dest_root)

    if ok:
        print("\nSafe to delete. The GPU container is already gone; this frees the volumes:")
        for v in VOLUME_NAMES:
            print(f"    modal volume delete {v} --yes")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
