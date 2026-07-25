#!/usr/bin/env bash
# CPU smoke test: no model, no GPU, no HuggingFace download (PLAN step 3).
#
# Every loss finite and backward on random hiddens; same seed -> identical losses.
# This is the check to run before touching the GPU box.
set -euo pipefail

python -m pytest tests/test_smoke.py -q -m "not gpu"
echo
echo "full CPU suite:"
python -m pytest tests/ -q -m "not gpu"
