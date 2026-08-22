"""Run the `cf_lam2_tau005` phase-1 run + ProcessBench on one rented Modal A100.

    modal run --detach modal_cf_lam2_tau005.py

Same machinery as `modal_nce1_masked.py` / `modal_app.py` -- read `modal_app.py`'s header for
the cost arithmetic and the "why A100 40 GB" argument. Only the argv differs:

    bash scripts/train_cloud.sh --set losses.zeta=0.2 \
        --set losses.nce_temperature=22.627417 \
        --set losses.lambda_term=0.0 \
        --set losses.lambda_cf=2.0 \
        --set losses.lambda_cf_temperature=0.05 \
        --set run.name=cf_lam2_tau005

`nce_temperature` goes BACK to sqrt(512) = 22.627417, which is `config/default.yaml`'s own
value; it is passed explicitly anyway so the intent is on the command line and not merely in
the resolved config. With zeta 0.2 and lambda_term 0.0 that makes the phase-1 setup identical
to `runs/abl_cf_only`, so this is a TWO-DELTA run against that baseline and both deltas are
on (4) L_CF: lambda_cf 1.0 -> 2.0 and lambda_cf_temperature 0.1 -> 0.05.

TWO THINGS THE CONFIG SAYS ABOUT THIS ARGV, RECORDED HERE BECAUSE THEY ARE NOT OBVIOUS FROM
THE COMMAND LINE. Neither is a reason not to run it -- both values were asked for explicitly
-- but both change how the result must be read.

  1. THE TWO KNOBS COMPOUND. `config/default.yaml` on both `lambda_cf_temperature` and
     `lambda_term_temperature`: "tau and lambda are the same knob (gradient ~ lambda/tau), so
     whoever moves this is moving lambda_cf by the reciprocal." Halving tau (0.1 -> 0.05) is
     already a 2x on (4)'s gradient; doubling lambda_cf on top makes it **4x**, not 2x. If
     the intent was 2x total, tau should stay at 0.1.

  2. tau = 0.05 IS THE VALUE THE CONFIG EXPLICITLY REJECTED. Verbatim: "0.05 was rejected: it
     saturates (p_pos -> 1, loss -> -log(1) territory) if the separation keeps growing, and a
     saturated softmax has no gradient left." The measured spread it was reasoned from is
     cf/positive_distance 0.056 vs cf/negative_distance 0.119 at step ~1090 of wc5byua1.

     THE TELL IS `cf/loss - cf/chance`, NEVER RAW `cf/loss` -- same config block. At tau = 0.1
     it was expected to sit somewhere useful; "if it runs to -1.09 or beyond, tau overshot"
     and 0.2 is the documented fallback, "if it stays near -0.02, this change did nothing."
     At tau = 0.05 the overshoot case is the one to watch, and it is visible within ~100
     steps, which is ~$0.40 of A100 rather than the whole run.

PROFILE stays `match`, so the batch shape is `config/default.yaml`'s and every loss statistic
is comparable to the other 56-sequence runs.
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

APP_NAME = "feynman-cf-lam2-tau005"
RUN_NAME = "cf_lam2_tau005"

LOCAL_REPO = Path(__file__).parent.resolve()
REPO = "/root/quasi"
HF_CACHE = "/cache/hf"

# The argv of `scripts/train_cloud.sh`, verbatim. PROFILE is left at its default (`match`).
TRAIN_ARGS = [
    "--set", "losses.zeta=0.2",
    "--set", "losses.nce_temperature=22.627417",
    "--set", "losses.lambda_term=0.0",
    "--set", "losses.lambda_cf=2.0",
    "--set", "losses.lambda_cf_temperature=0.05",
    "--set", f"run.name={RUN_NAME}",
]

# Not a "did it improve" threshold -- a "did the geometry learn correctness at all" floor.
# Under it, phase 2 and ProcessBench are wasted GPU time (scripts/val_f1.py's docstring).
SKIP_PHASE2_BELOW = 0.35

# The delete step will not run unless all of these came back. Relative to runs/nce1_masked/.
_REQUIRED_ARTIFACTS = [
    "config.resolved.yaml",
    "events.jsonl",
    "metrics.jsonl",
    "final/heads.pt",
    "final/val_f1.json",
    "phase2/final/processbench.json",
]

# ---------------------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_requirements(str(LOCAL_REPO / "requirements.txt"))
    .env(
        {
            "HF_HOME": HF_CACHE,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .workdir(REPO)
    .add_local_dir(
        LOCAL_REPO,
        REPO,
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

secret = modal.Secret.from_name("feynman-prm")


# ---------------------------------------------------------------------------------------
# Helpers, container-side
# ---------------------------------------------------------------------------------------


def _run(cmd: list[str], stage: str) -> None:
    print(f"\n{'=' * 78}\n[{stage}] {' '.join(cmd)}\n{'=' * 78}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO)
    dt = time.time() - t0
    print(f"[{stage}] exit={result.returncode} in {dt / 60:.1f} min", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"stage '{stage}' failed with exit code {result.returncode}")


class _Committer:
    """Commit the runs volume every `every` seconds.

    Modal commits a volume when the function returns, which is the wrong time here: the
    function returns after four hours and a container that dies at hour three would otherwise
    take phase 1 with it. `metrics.jsonl` and `events.jsonl` are appended and flushed per step,
    so a periodic commit means a crash costs at most `every` seconds of curve.
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
# Stage 0 -- CPU. Build data/processed, warm the HF cache, and check the argv parses.
# ---------------------------------------------------------------------------------------


@app.function(
    image=image,
    volumes=VOLUMES,
    secrets=[secret],
    cpu=(4.0, 8.0),
    memory=(8192, 32768),
    timeout=90 * 60,
)
def prepare() -> dict:
    """`prepare_data.py` plus a Qwen prefetch, on cores that cost $0.05/h."""
    # THE ARGV CHECK, FIRST AND CHEAPEST. config/default.yaml is strict-parsed, so a
    # misspelled `--set` key is a hard error -- and finding that out on the A100 costs
    # container start plus a 3.1 GB model load before it throws. Four CPU seconds here.
    sys.path.insert(0, REPO)
    from feynman_prm.config import load_config

    overrides = [TRAIN_ARGS[i + 1] for i in range(0, len(TRAIN_ARGS), 2)]
    cfg = load_config(f"{REPO}/config/default.yaml", overrides)
    print(
        f"[prepare/argv] zeta={cfg.losses.zeta} tau_nce={cfg.losses.nce_temperature} "
        f"mask_sibling_correct_late={cfg.sampling.nce_mask_sibling_correct_late} "
        f"lambda_term={cfg.losses.lambda_term} run={cfg.run.name}",
        flush=True,
    )

    selection = Path(REPO) / "data/processed/selection.json"

    if selection.exists():
        payload = json.loads(selection.read_text())
        print(f"[prepare] data/processed already built: {payload}", flush=True)
    else:
        _run([sys.executable, "scripts/prepare_data.py"], "prepare_data")
        data_vol.commit()
        payload = json.loads(selection.read_text())

    # THE MATCHED-DATA CHECK. If this is not the selection the other Feynman runs used, no
    # comparison against them is matched -- so it fails here, on a CPU container.
    n_train = payload.get("n_train_questions")
    n_val = payload.get("n_val_questions")
    if (n_train, n_val) != (34650, 2000):
        raise RuntimeError(
            f"selection mismatch: n_train={n_train} n_val={n_val}, expected 34650/2000."
        )
    print(f"[prepare] selection_sha_train={payload.get('selection_sha_train')}", flush=True)

    import yaml
    from huggingface_hub import snapshot_download

    model_name = yaml.safe_load(open(f"{REPO}/config/default.yaml"))["model"]["name"]
    print(f"[prepare] prefetching {model_name} into {HF_CACHE}", flush=True)
    snapshot_download(model_name, token=os.environ.get("HF_TOKEN"))
    hf_vol.commit()

    return payload


# ---------------------------------------------------------------------------------------
# Stages 1-4 -- GPU. One container, four stages, commit between each.
# ---------------------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes=VOLUMES,
    secrets=[secret],
    cpu=(4.0, 12.0),
    memory=(16384, 65536),
    timeout=8 * 60 * 60,   # measured chain is ~4.2 h; headroom, not an expectation
)
def train_and_eval(skip_phase2_below: float = SKIP_PHASE2_BELOW) -> dict:
    import torch

    name = torch.cuda.get_device_name(0)
    print(f"[gpu] {name}  bf16={torch.cuda.is_bf16_supported()}  torch={torch.__version__}",
          flush=True)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{name} does not support bf16")

    root = Path(REPO)
    run_dir = f"runs/{RUN_NAME}"
    final = f"{run_dir}/final"
    phase2_final = f"{run_dir}/phase2/final"
    summary: dict = {"gpu": name, "run": RUN_NAME, "stages": {}}

    with _Committer(runs_vol):
        # --- 1. phase 1. The human's command, unmodified. ---------------------------------
        if (root / final / "heads.pt").exists():
            print(f"[1/train_cloud] SKIPPED: {final}/heads.pt is already on the volume.",
                  flush=True)
            summary["stages"]["train"] = "skipped (already done)"
        else:
            t0 = time.time()
            _run(["bash", "scripts/train_cloud.sh", *TRAIN_ARGS], "1/train_cloud")
            summary["stages"]["train"] = round((time.time() - t0) / 3600, 2)
            runs_vol.commit()

        # The wandb dashboard URL, persisted by RunLogger into events.jsonl. Surfaced here so
        # it is in the returned summary and not only somewhere in a four-hour log stream.
        try:
            for line in (root / run_dir / "events.jsonl").read_text().splitlines():
                rec = json.loads(line)
                if rec.get("event") == "wandb/run":
                    summary["wandb_url"] = rec["url"]
        except Exception:
            pass

        # --- 2. the val-F1 ceiling, and the gate on spending another ~1.65 h --------------
        if not (root / final / "val_f1.json").exists():
            _run([sys.executable, "scripts/val_f1.py", "--checkpoint", final], "2/val_f1")
            runs_vol.commit()
        else:
            print("[2/val_f1] SKIPPED: val_f1.json already on the volume.", flush=True)

        val_f1 = json.loads((root / final / "val_f1.json").read_text())
        # val_f1.py writes {"at_best_tau": {"f1": ...}, ...}. It is NOT the flat
        # "calibration/f1" key processbench.json uses -- reading that one yields None, and a
        # None gate is a gate that never fires.
        f1 = val_f1.get("at_best_tau", {}).get("f1")
        summary["val_f1"] = f1
        print(f"[gate] phase-1 ceiling F1 = {f1} (benchmark 0.560, floor {skip_phase2_below})",
              flush=True)

        if f1 is not None and f1 < skip_phase2_below:
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
        # ProcessBench refuses to score a phase-1 checkpoint (§7.3 of SETUP.md), so this is
        # not optional even though the goal head is not itself the result.
        if (root / phase2_final / "heads.pt").exists():
            print("[3/goal_head] SKIPPED: phase2/final already on the volume.", flush=True)
            summary["stages"]["phase2"] = "skipped (already done)"
        else:
            t0 = time.time()
            _run(
                [sys.executable, "-m", "feynman_prm.train_goal_head", "--checkpoint", final],
                "3/goal_head",
            )
            summary["stages"]["phase2"] = round((time.time() - t0) / 3600, 2)
            runs_vol.commit()

        # --- 4. ProcessBench, on the phase-2 checkpoint (locked #13). THE result. ----------
        if (root / phase2_final / "processbench.json").exists():
            print("[4/processbench] SKIPPED: processbench.json already on the volume.",
                  flush=True)
            summary["stages"]["processbench"] = "skipped (already done)"
        else:
            t0 = time.time()
            _run(["bash", "scripts/eval_processbench.sh", phase2_final], "4/processbench")
            summary["stages"]["processbench"] = round((time.time() - t0) / 3600, 2)
            runs_vol.commit()

    # The headline numbers, into the returned summary and the log. processbench.json has no
    # "mean_f1" key -- the four subsets are top-level dicts -- so it is computed here rather
    # than read, which is the bug modal_app.py shipped (it reported None every time).
    try:
        pb = json.loads((root / phase2_final / "processbench.json").read_text())
        subsets = {
            k: v["f1"] for k, v in pb.items()
            if isinstance(v, dict) and "f1" in v and not k.endswith("SKYLINE_not_a_result")
            and k != "calibration"
        }
        summary["processbench_f1"] = {k: round(v, 4) for k, v in subsets.items()}
        if subsets:
            summary["processbench_mean_f1"] = round(sum(subsets.values()) / len(subsets), 4)
        summary["processbench_tau"] = pb.get("calibration", {}).get("calibration/tau")
        print(f"\n[result] {json.dumps(summary, indent=2)}", flush=True)
    except Exception as exc:
        print(f"[result] could not summarise processbench.json: {exc}", flush=True)

    return summary


# ---------------------------------------------------------------------------------------
# Local orchestration. With `--detach` this entrypoint's post-processing does NOT run once
# the shell is gone, which is the whole point -- fetching and deleting live in
# modal_nce1_masked_finish.py so they can be run later, and more than once.
# ---------------------------------------------------------------------------------------


@app.local_entrypoint()
def main(skip_phase2_below: float = SKIP_PHASE2_BELOW):
    print("\n=== stage 0: prepare (CPU) ===")
    selection = prepare.remote()
    print(f"    selection ok: {selection.get('selection_sha_train')}")

    print("\n=== stages 1-4: train + ProcessBench (A100 40 GB, ~4.2 h) ===")
    summary = train_and_eval.remote(skip_phase2_below=skip_phase2_below)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print("\nNow run:  python modal_nce1_masked_finish.py")
