#!/usr/bin/env bash
# Fetch the Best-of-N candidate files that scripts/eval_bon.sh scores.
#
#   bash scripts/fetch_bon_data.sh            # -> ./eval_data
#   DATA_DIR=/mnt/eval_data bash scripts/fetch_bon_data.sh
#
# WHAT THESE ARE. PQM's Best-of-N test corpus (Process Reward Model with Q-Value Rankings,
# arXiv:2410.11287, ICLR 2025): 128 sampled solutions per question over GSM-Plus testmini and
# MATH-500 test, from MetaMath-Mistral-7B and MuggleMath-13B. CRM reuses it wholesale, which
# is the only reason our numbers can sit next to theirs.
#
# THEY ARE NOT IN PQM's GIT REPO. github.com/WindyLee0822/Process_Q_Model ships code only --
# its README points at the HF model repo below, under `eval_data/`. Cloning the repo gets you
# nothing; this script is the step people miss.
#
# WHY THIS IS A SCRIPT AND NOT COMMITTED DATA. 451 MB across four files, two of them over
# GitHub's 100 MB per-file hard block. Same call as `data/cf/` in .gitignore, and stronger:
# this corpus is PQM's, it is published, and it is one command away. A fetcher in git beats
# 451 MB in git.
#
# DO NOT REGENERATE THEM with PQM's sample_testset.py. Sampling our own 128 candidates would
# make every number incomparable with CRM's published table.
set -euo pipefail

DATA_DIR="${DATA_DIR:-eval_data}"
REPO="${PQM_REPO:-Windy0822/PQM}"

FILES=(
    "gsm8k-plus-metamath-mistral-128.json"
    "math-metamath-mistral-128.json"
    "gsm8k-plus-muggle-128.json"
    "math-muggle-128.json"
)

mkdir -p "${DATA_DIR}"

python - "${DATA_DIR}" "${REPO}" "${FILES[@]}" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

data_dir, repo, *files = sys.argv[1:]
for name in files:
    target = Path(data_dir) / name
    if target.exists():
        print(f"[have] {target} ({target.stat().st_size / 1e6:.0f} MB)", flush=True)
        continue
    print(f"[get ] {name} from {repo}", flush=True)
    # `eval_data/` is the path INSIDE the HF repo; we flatten it into DATA_DIR because that is
    # where eval_bon.sh looks. local_dir keeps the bytes here rather than in the HF cache, so
    # one copy on disk, not two.
    path = hf_hub_download(
        repo_id=repo,
        filename=f"eval_data/{name}",
        repo_type="model",
        local_dir=data_dir,
    )
    Path(path).replace(target)
    print(f"[done] {target} ({target.stat().st_size / 1e6:.0f} MB)", flush=True)

nested = Path(data_dir) / "eval_data"
if nested.is_dir() and not any(nested.iterdir()):
    nested.rmdir()
PY

echo
echo "eval_data ready at ${DATA_DIR}:"
ls -lh "${DATA_DIR}"
echo
echo "next:  bash scripts/eval_bon.sh <phase2-checkpoint> "
echo "       (DATA_DIR=${DATA_DIR} if you changed it above)"
