#!/usr/bin/env bash
# ProcessBench eval. Needs a PHASE-2 checkpoint: the goal head is the only sanctioned goal
# at eval (locked #13); a reference-solution goal is a labelled skyline, not a result (§5.1).
#
#   bash scripts/eval_processbench.sh runs/phase1/phase2/final
#
# Prints per subset: acc_error, acc_correct, F1; the math subset split 587 leaked vs 413
# clean (locked #5); and the skyline on the 1,400 joinable samples, labelled as a skyline.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT="${1:?usage: eval_processbench.sh <phase2-checkpoint-dir> [--tau X]}"
shift || true
exec python -m feynman_prm.eval.processbench --checkpoint "$CHECKPOINT" "$@"
