#!/usr/bin/env bash
# Phase 1. RUN THIS UNDER TMUX -- tmux is mandatory for every GPU run (§13).
#
#   tmux new -s feynman
#   bash scripts/train.sh                     # full run, ~889 optimizer steps
#   bash scripts/train.sh --max-steps 20      # the short GPU probe (PLAN step 4)
#
# The launch sequence prints, in order: the step-count assert (§11.1), the trainability
# assert (§14), the longest-batch memory probe (PLAN 4a), and the initialisation values
# against §18. Read all four before walking away.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # §13, every GPU shell

CONFIG="${CONFIG:-config/default.yaml}"
exec python -m feynman_prm.train --config "$CONFIG" "$@"
