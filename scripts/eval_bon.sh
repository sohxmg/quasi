#!/usr/bin/env bash
# Best-of-N eval over CRM's four candidate files, so the numbers sit next to CRM's paper's.
# The file list, the dataset mapping and the N ladder are ../CRM/BoN/eval_PQM_our.sh's.
#
#   DATA_DIR=/path/to/bon/eval_data \
#   MATH_REFERENCE=/path/to/MATH-500/test.jsonl \
#   bash scripts/eval_bon.sh runs/phase1/phase2/final
#
# Needs a PHASE-2 checkpoint: the goal head is the only sanctioned goal at eval (locked #13).
# Run scripts/eval_processbench.sh first if you can -- it writes the fitted tau into
# processbench.json and three of the six aggregators read it. Without it the harness falls
# back to the natural midpoint and says so loudly (§9.2 measured the two 3.37x apart).
#
# MATH_REFERENCE is optional; without it the math subsets join HuggingFaceH4/MATH-500 from
# the hub, which is the same copy skyline.py already uses (§5.2).
#
# WHERE DATA_DIR COMES FROM. The four *-128.json files are not ours and are not CRM's either:
# they are PQM's Best-of-N test corpus (Process Reward Model with Q-Value Rankings,
# arXiv:2410.11287) -- 128 sampled solutions per question over GSM-Plus and MATH-500, from
# MetaMath-Mistral-7B and MuggleMath-13B, which is exactly what the filenames say. CRM reuses
# it wholesale, and its own tree says so: the driver is named `eval_PQM_our.sh` and
# `eval_normalizer.py:242` still imports from `eval_PQM_grader`. Get the files from the PQM
# release and point DATA_DIR at them.
#
# DO NOT REGENERATE THEM. Sampling our own 128 candidates would make every number here
# incomparable with CRM's published table, which is the only reason this eval exists.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT="${1:?usage: eval_bon.sh <phase2-checkpoint-dir> [extra args passed through]}"
shift || true

DATA_DIR="${DATA_DIR:-eval_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT}/bon}"
GSM8K_REFERENCE="${GSM8K_REFERENCE:-qintongli/GSM-Plus}"
MATH_REFERENCE="${MATH_REFERENCE:-}"

FILES=(
    "gsm8k-plus-metamath-mistral-128.json"
    "math-metamath-mistral-128.json"
    "gsm8k-plus-muggle-128.json"
    "math-muggle-128.json"
)

mkdir -p "${OUTPUT_DIR}"

for FILE in "${FILES[@]}"; do
    DATA_FILE="${DATA_DIR}/${FILE}"
    if [[ ! -f "${DATA_FILE}" ]]; then
        echo "!! missing ${DATA_FILE} -- skipping. These are CRM's sampled-response files;"
        echo "   we do not generate candidates, because a different pool is not a comparison."
        continue
    fi

    case "${FILE}" in
        gsm8k*) DATA_NAME="gsm8k" ;;
        math*)  DATA_NAME="math" ;;
        *)      echo "unknown dataset in filename: ${FILE}"; continue ;;
    esac

    # The oracle / mean-candidate baselines need all 128 graded per question. That is a regex
    # on gsm8k and sympy-with-a-timeout on math, so it is on for one and off for the other.
    EXTRA=()
    [[ "${DATA_NAME}" == "gsm8k" ]] && EXTRA+=(--grade-all-candidates)

    echo "=== ${FILE} (${DATA_NAME}) ==="
    python -m feynman_prm.eval.bon \
        --checkpoint "${CHECKPOINT}" \
        --data-name "${DATA_NAME}" \
        --data-file "${DATA_FILE}" \
        --save-file "${OUTPUT_DIR}/bon_${FILE%.json}.json" \
        --gsm8k-reference "${GSM8K_REFERENCE}" \
        ${MATH_REFERENCE:+--math-reference "${MATH_REFERENCE}"} \
        "${EXTRA[@]}" "$@"
done

echo
echo "=== comparison against CRM's published numbers ==="
python scripts/report_bon.py "${OUTPUT_DIR}"
