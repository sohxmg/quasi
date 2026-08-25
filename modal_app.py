"""Run the `nce_tau4` phase-1 run + its full eval chain on one rented Modal A100, then leave
nothing behind.

    modal run --detach modal_app.py

WHY THIS FILE EXISTS AND WHAT IT IS NOT. It is not a new way to configure training. Every
knob still lives in `config/default.yaml` and every override is still a `--set` on the command
line, exactly as `scripts/train_cloud.sh` documents -- this file *shells out* to that script
with the same argv the human would have typed on a Lightning box. If the two ever disagree,
`scripts/train_cloud.sh` is right and this is the bug.

    bash scripts/train_cloud.sh --set losses.nce_temperature=4.0 \
        --set losses.zeta=0.2 --set losses.lambda_term=0.0 --set run.name=nce_tau4

PROFILE stays `match` (the default), so the batch shape is `config/default.yaml`'s and every
loss statistic is comparable to `runs/abl_cf_only` -- which is the point of the run. That
baseline is `losses.zeta=0.2, losses.lambda_term=0.0, losses.nce_temperature=22.627`, so this
is a ONE-delta comparison on tau_NCE, which is §9.13.7's "still open": tau_NCE = 22.627
starves (1) on BOTH sides of every comparison in §9.12-§9.13, and (1) has never been read
against a live gradient.

---------------------------------------------------------------------------------------
COST -- the whole design is subordinate to this
---------------------------------------------------------------------------------------

A100 40 GB at $2.10/h. Chosen on arithmetic, not preference: `runs/phase1_cf_term_taucf01`
ran this exact batch shape at PROFILE=match on a rented A100 40 GB in 2.13 h, so the phase-1
wall clock is measured rather than guessed. H100 at $3.95/h needs a 1.88x speedup to break
even and a gradient-checkpointed 1.5B LoRA on sdpa gets ~1.6-1.7x; L40S is 1.4x slower at
0.93x the price; A10/L4 are 2.5-3.5x slower at 0.52/0.38x. A100 40 GB is the minimum of
rate x hours, and it is also the card the baseline was measured on.

Three structural savings, each worth naming because each is a thing this file does
differently from "just run it on a GPU box":

  1. `prepare_data.py` RUNS ON A CPU CONTAINER. It builds sequences.parquet from a tokenizer
     and never touches a GPU. Ten minutes of A100 is $0.35; ten minutes of 4 CPU cores is
     $0.05. The same container also pre-pulls the Qwen weights into the HF cache volume, so
     the A100 never spends billed seconds on a 3.1 GB download.

  2. TRAIN AND ALL FOUR EVALS RUN IN ONE CONTAINER. Five `.remote()` calls would re-pay
     container start, image pull and backbone load five times. They are one function that
     runs the stages in sequence, committing the runs volume after each so a crash in stage 4
     never costs stage 1.

  3. THE VAL-F1 GATE (stage 2, and it runs SECOND for this reason). `scripts/val_f1.py`'s
     docstring is explicit that a bad ceiling means "phase 2 is wasted GPU time". If F1 comes
     back under `SKIP_PHASE2_BELOW` the run stops there and saves ~1.65 h = ~$3.90. The
     threshold is deliberately far below the 0.560 benchmark so that only a COLLAPSE trips
     it -- a merely disappointing number still gets the full chain, because a disappointing
     number is a result and needs its ProcessBench row.

Storage bills $0.00 regardless: Modal volumes are $0.09/GiB/mo against 1 TiB/mo free and this
uses ~5 GiB. The volumes are still deleted at the end, because that is what was asked for.

---------------------------------------------------------------------------------------
WHAT COMES BACK, AND WHAT IS THEN DESTROYED
---------------------------------------------------------------------------------------

`runs/nce_tau4/` is downloaded to this machine before anything is deleted, and the delete step
REFUSES to run if the download did not land the files it expects (see `_REQUIRED_ARTIFACTS`).
That ordering is the only thing standing between a $10 run and a $10 run with no output, so it
is checked rather than assumed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------------------

APP_NAME = "feynman-nce-tau4"
RUN_NAME = "nce_tau4"

LOCAL_REPO = Path(__file__).parent.resolve()
REPO = "/root/quasi"          # the clone root, per pqm_baseline/RUNBOOK.md -- NOT feynman_prm/
HF_CACHE = "/cache/hf"

# The argv of `scripts/train_cloud.sh`, verbatim. PROFILE is left at its default (`match`).
TRAIN_ARGS = [
    "--set", "losses.nce_temperature=4.0",
    "--set", "losses.zeta=0.2",
    "--set", "losses.lambda_term=0.0",
    "--set", f"run.name={RUN_NAME}",
]

# §9.13.1's benchmark is 0.560 and abl_cf_only's ceiling is in that region. This is not a
# "did it improve" threshold -- it is a "did the geometry learn correctness at all" floor, and
# a run that clears it gets the full chain no matter how it compares.
SKIP_PHASE2_BELOW = 0.35

# The delete step will not run unless all of these came back. Relative to runs/nce_tau4/.
_REQUIRED_ARTIFACTS = [
    "config.resolved.yaml",
    "events.jsonl",
    "metrics.jsonl",
    "final/heads.pt",
    "final/val_f1.json",
]

# ---------------------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------------------
#
# The default PyPI torch wheel is correct HERE and requirements.txt's cu128-index note is not:
# that note is a Kratos constraint (sm_120 Blackwell has no other wheel) and an A100 is sm_80.
# pqm_baseline/RUNBOOK.md §0 says the same thing in the same words.
#
# `add_local_dir` with copy=False (the default) attaches the code at container START rather
# than baking it into an image layer, so editing a script does not trigger a torch rebuild.

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_requirements(str(LOCAL_REPO / "requirements.txt"))
    .env(
        {
            "HF_HOME": HF_CACHE,
            # §13, every GPU shell. train_cloud.sh sets this too; belt and braces, because a
            # ragged allocator without it fragments into a false OOM.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .workdir(REPO)
    .add_local_dir(
        LOCAL_REPO,
        REPO,
        # Everything here is either rebuilt on the box, mounted as a volume, or irrelevant to
        # training. `data/cf_train/` is NOT in this list and must not be: it is the 52 MB
        # cf_glob points at (§7.5.13) and lambda_cf = 1.0 trains on nothing without it.
        ignore=[
            ".git/**",
            ".venv/**",
            ".pytest_cache/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "runs/**",              # volume
            "data/processed/**",    # volume
            "data/cf/**",           # 426 MB campaign directory; the snapshot is what ships
            "eval_data/**",         # 452 MB, BoN only -- not this chain
            "bon_reference/**",
            ".env",                 # secrets travel as a modal.Secret, never in the image
            "secret.txt",
        ],
    )
)

app = modal.App(APP_NAME)

# ---------------------------------------------------------------------------------------
# Volumes and secrets
# ---------------------------------------------------------------------------------------

hf_vol = modal.Volume.from_name("feynman-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("feynman-processed", create_if_missing=True)
runs_vol = modal.Volume.from_name("feynman-runs", create_if_missing=True)

VOLUME_NAMES = ["feynman-hf-cache", "feynman-processed", "feynman-runs"]

VOLUMES = {
    HF_CACHE: hf_vol,
    f"{REPO}/data/processed": data_vol,
    f"{REPO}/runs": runs_vol,
}

# HF_TOKEN (Math-Shepherd and Qwen2.5-Math-1.5B are both gated pulls) and WANDB_API_KEY
# (RunLogger RAISES if wandb is asked for and missing -- logging.py:80).
secret = modal.Secret.from_name("feynman-prm")


# ---------------------------------------------------------------------------------------
# Helpers, container-side
# ---------------------------------------------------------------------------------------


def _run(cmd: list[str], stage: str) -> None:
    """Run one stage with its output streamed to the Modal log, and fail loudly."""
    print(f"\n{'=' * 78}\n[{stage}] {' '.join(cmd)}\n{'=' * 78}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO)
    dt = time.time() - t0
    print(f"[{stage}] exit={result.returncode} in {dt / 60:.1f} min", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"stage '{stage}' failed with exit code {result.returncode}")


class _Committer:
    """Commit the runs volume every `every` seconds.

    Modal commits a volume when the function returns, which is exactly the wrong time here:
    the function returns after four hours, and a container that dies at hour three would
    otherwise take phase 1 with it. `metrics.jsonl` and `events.jsonl` are appended and
    flushed per step (logging.py), so a periodic commit means a crash costs at most `every`
    seconds of curve rather than the whole run.
    """

    def __init__(self, volume, every: int = 300):
        self._volume = volume
        self._every = every
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.wait(self._every):
            try:
                self._volume.commit()
            except Exception as exc:  # a failed commit must never kill a 4-hour run
                print(f"[commit] non-fatal: {exc}", flush=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)
        self._volume.commit()
        return False


# ---------------------------------------------------------------------------------------
# Stage 0 -- CPU. Build data/processed, warm the HF cache. No GPU is billed here.
# ---------------------------------------------------------------------------------------


@app.function(
    image=image,
    volumes=VOLUMES,
    secrets=[secret],
    cpu=(4.0, 8.0),
    memory=(8192, 32768),
    timeout=60 * 60,
)
def prepare() -> dict:
    """`python scripts/prepare_data.py` plus a Qwen prefetch, on cores that cost $0.05/h."""
    selection = Path(REPO) / "data/processed/selection.json"

    if selection.exists():
        payload = json.loads(selection.read_text())
        print(f"[prepare] data/processed already built: {payload}", flush=True)
    else:
        _run([sys.executable, "scripts/prepare_data.py"], "prepare_data")
        data_vol.commit()
        payload = json.loads(selection.read_text())

    # THE MATCHED-DATA CHECK (RUNBOOK §1). If this is not the selection the Feynman runs used,
    # the one-delta comparison against abl_cf_only is not matched and the run is pointless --
    # so it fails here, on a CPU container, rather than four A100-hours later.
    n_train = payload.get("n_train_questions")
    n_val = payload.get("n_val_questions")
    if (n_train, n_val) != (34650, 2000):
        raise RuntimeError(
            f"selection mismatch: n_train={n_train} n_val={n_val}, expected 34650/2000. "
            "The comparison against runs/abl_cf_only would not be matched."
        )
    print(f"[prepare] selection_sha_train={payload.get('selection_sha_train')}", flush=True)

    # Pre-pull the backbone so the A100 never waits on a 3.1 GB download.
    import yaml
    from huggingface_hub import snapshot_download

    model_name = yaml.safe_load(open(f"{REPO}/config/default.yaml"))["model"]["name"]
    print(f"[prepare] prefetching {model_name} into {HF_CACHE}", flush=True)
    snapshot_download(model_name, token=os.environ.get("HF_TOKEN"))
    hf_vol.commit()

    return payload


# ---------------------------------------------------------------------------------------
# Stage 1-5 -- GPU. One container, five stages, commit between each.
# ---------------------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes=VOLUMES,
    secrets=[secret],
    cpu=(4.0, 12.0),
    memory=(16384, 65536),
    timeout=8 * 60 * 60,   # measured chain is ~4.2 h; this is headroom, not an expectation
)
def train_and_eval(skip_phase2_below: float = SKIP_PHASE2_BELOW) -> dict:
    import torch

    name = torch.cuda.get_device_name(0)
    print(f"[gpu] {name}  bf16={torch.cuda.is_bf16_supported()}  torch={torch.__version__}",
          flush=True)
    if not torch.cuda.is_bf16_supported():
        # The whole stack is bf16; a pre-Ampere card cannot run this at all (RUNBOOK §0).
        raise RuntimeError(f"{name} does not support bf16")

    run_dir = f"runs/{RUN_NAME}"
    final = f"{run_dir}/final"
    phase2_final = f"{run_dir}/phase2/final"
    summary: dict = {"gpu": name, "stages": {}}

    with _Committer(runs_vol):
        # --- 1. phase 1. The human's command, unmodified. ---------------------------------
        t0 = time.time()
        _run(["bash", "scripts/train_cloud.sh", *TRAIN_ARGS], "1/train_cloud")
        summary["stages"]["train"] = round((time.time() - t0) / 3600, 2)
        runs_vol.commit()

        # --- 2. the val-F1 ceiling, and the gate on spending another 1.65 h ----------------
        _run([sys.executable, "scripts/val_f1.py", "--checkpoint", final], "2/val_f1")
        runs_vol.commit()

        val_f1 = json.loads((Path(REPO) / final / "val_f1.json").read_text())
        # scripts/val_f1.py writes {"at_best_tau": {"f1": ...}, "at_natural_tau": {...}}.
        # It is NOT the flat "calibration/f1" key that processbench.json and pqm_baseline use
        # -- reading that one silently yields None, and a None gate is a gate that never
        # fires. It cost nothing on 2026-08-20 (the fallthrough runs phase 2, which was the
        # right call at 0.5533) but the failure mode is "spends the $3.90 it was written to
        # save", so it is a real bug and not a cosmetic one.
        f1 = val_f1.get("at_best_tau", {}).get("f1")
        summary["val_f1"] = f1
        print(f"[gate] phase-1 ceiling F1 = {f1} (benchmark 0.560, floor {skip_phase2_below})",
              flush=True)

        if f1 is not None and f1 < skip_phase2_below:
            # val_f1.py's docstring: "F1 here is BAD -> ... phase 2 is wasted GPU time."
            print(
                f"[gate] STOPPING. {f1:.4f} < {skip_phase2_below} means the geometry never "
                "learned correctness and no goal head rescues it. Phase 2 + ProcessBench "
                "skipped, ~1.65 h of A100 (~$3.90) not spent. Phase 1 is on the volume and "
                "comes home as normal.",
                flush=True,
            )
            summary["stopped_at"] = "val_f1_gate"
            return summary

        # --- 3. phase 2, the goal head on frozen cached vectors ---------------------------
        t0 = time.time()
        _run(
            [sys.executable, "-m", "feynman_prm.train_goal_head", "--checkpoint", final],
            "3/goal_head",
        )
        summary["stages"]["phase2"] = round((time.time() - t0) / 3600, 2)
        runs_vol.commit()

        # --- 4. ProcessBench, on the phase-2 checkpoint (locked #13) ----------------------
        t0 = time.time()
        _run(["bash", "scripts/eval_processbench.sh", phase2_final], "4/processbench")
        summary["stages"]["processbench"] = round((time.time() - t0) / 3600, 2)
        runs_vol.commit()

        # --- 5. the masked gate. §9.13.7 asks for this on the lambda_term = 0 cell --------
        # Recall that survives masking is structure; recall that collapses under it was the
        # printed answer string. This run IS lambda_term = 0, so it is that control cell.
        gate_out = Path(REPO) / final / "goal_gate_mask.txt"
        print(f"\n{'=' * 78}\n[5/goal_gate] --mask-answer -> {gate_out}\n{'=' * 78}", flush=True)
        with gate_out.open("w") as fh:
            rc = subprocess.run(
                [sys.executable, "scripts/goal_gate.py", "--checkpoint", final, "--mask-answer"],
                cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
            ).returncode
        print(gate_out.read_text()[-4000:], flush=True)
        if rc != 0:
            # Non-fatal by choice: stages 1-4 are the run and they are already on the volume.
            print(f"[5/goal_gate] exit={rc} -- NOT failing the run, 1-4 are complete.",
                  flush=True)
        summary["stages"]["goal_gate"] = "ok" if rc == 0 else f"exit {rc}"
        runs_vol.commit()

    try:
        pb = json.loads((Path(REPO) / phase2_final / "processbench.json").read_text())
        summary["processbench_mean_f1"] = pb.get("mean_f1")
    except Exception:
        pass

    return summary


# ---------------------------------------------------------------------------------------
# Local orchestration: prepare -> train+eval -> download -> verify. Delete is separate.
# ---------------------------------------------------------------------------------------


def _download_run(dest_root: Path) -> list[Path]:
    """Pull runs/<RUN_NAME>/ off the volume onto this machine."""
    dest = dest_root / RUN_NAME
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for entry in runs_vol.listdir(f"/{RUN_NAME}", recursive=True):
        # FileEntry.type 1 == FILE; directories are created implicitly by mkdir below.
        if getattr(entry.type, "value", entry.type) != 1:
            continue
        rel = entry.path.lstrip("/")
        rel = rel[len(RUN_NAME) + 1:] if rel.startswith(RUN_NAME + "/") else rel
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as fh:
            for chunk in runs_vol.read_file(entry.path):
                fh.write(chunk)
        written.append(out)
        print(f"  <- {rel}  ({out.stat().st_size / 1e6:.2f} MB)")
    return written


@app.local_entrypoint()
def main(skip_phase2_below: float = SKIP_PHASE2_BELOW):
    print(f"\n=== stage 0: prepare (CPU) ===")
    selection = prepare.remote()
    print(f"    selection ok: {selection.get('selection_sha_train')}")

    print(f"\n=== stages 1-5: train + eval (A100 40 GB, ~4.2 h) ===")
    summary = train_and_eval.remote(skip_phase2_below=skip_phase2_below)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))

    print(f"\n=== downloading runs/{RUN_NAME}/ ===")
    dest_root = LOCAL_REPO / "runs"
    _download_run(dest_root)

    missing = [a for a in _REQUIRED_ARTIFACTS if not (dest_root / RUN_NAME / a).exists()]
    if missing:
        print(f"\n!! NOT SAFE TO DELETE -- missing locally: {missing}")
        print("!! The volumes are intact. Re-run the download before cleaning up.")
        return

    print(f"\nAll artifacts landed in {dest_root / RUN_NAME}")
    print("Clean up (GPU is already released; this frees the volumes):")
    for v in VOLUME_NAMES:
        print(f"    modal volume delete {v} --yes")
