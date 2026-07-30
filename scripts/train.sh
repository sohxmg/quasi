#!/usr/bin/env bash
# Phase 1. RUN THIS UNDER TMUX -- tmux is mandatory for every GPU run (§13).
#
#   tmux new -s feynman
#   bash scripts/train.sh                     # full run, ~1,460 optimizer steps, 3-6 h
#                                             # PREREQUISITE: prepare_data.py must have been
#                                             # run at the CURRENT n_questions (34650).
#   bash scripts/train.sh --max-steps 20      # the short GPU probe (PLAN step 4)
#   bash scripts/train.sh --set losses.lambda_good=0.0    # (6) L_good OFF but still
#                                                         # logged -- §7.12, §16.21
#
# The launch sequence prints, in order: the step-count assert (§11.1), the trainability
# assert (§14), the longest-batch memory probe (PLAN 4a), and the initialisation values
# against §18 -- which now include L_good's relu sandwich and the SIGN of its margin (c must
# print NEGATIVE). Read all four before walking away.
#
# (6) L_good is ON at lambda_good = 1.0 (signed off 2026-07-28, §16.21). It is the term that
# bounds a GOOD step's Delta from above, which nothing else in the loss set does (§7.12's
# identity). ALWAYS TRAIN FRESH: a loss-set change invalidates the checkpoint, so step750 is
# not resumable under it. Setting lambda_good=0.0 leaves the term computed and logged.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # §13, every GPU shell

CONFIG="${CONFIG:-config/default.yaml}"
exec python -m feynman_prm.train --config "$CONFIG" "$@"
