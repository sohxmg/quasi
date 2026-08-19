#!/usr/bin/env bash
# PQM baseline. RUN THIS UNDER TMUX -- tmux is mandatory for every GPU run (§13).
#
#   tmux new -s pqm
#   bash pqm_baseline/train.sh --max-steps 20      # the probe -> runs/pqm_zeta4_probe/
#   bash pqm_baseline/train.sh                     # ~1,460 optimizer steps
#
# PREREQUISITE, and it is the whole point of this baseline: `data/processed/sequences.parquet`
# must be the SAME FILE the Feynman runs read, at `n_questions: 34650`. Check it before
# launching -- if `data/processed/selection.json`'s `selection_sha_train` does not match the
# one those runs used, the comparison is not matched and nothing else matters:
#
#   python -c "import json;print(json.load(open('data/processed/selection.json'))['selection_sha_train'])"
#
# WHAT THIS RUNS: PQM's head and objective under Feynman-PRM's exact conditions. Same parquet,
# same selection, same seed, same batch stream, same optimizer steps, same eval protocol.
# `config/default.yaml` is read UNCHANGED and never edited -- `run.name` comes through --set,
# and PQM's own knobs live in `pqm_baseline/config/pqm.yaml`.
#
# READ FOUR THINGS OFF THE PROBE'S `events.jsonl` BEFORE THE REAL LAUNCH:
#
#   launch/data          `questions`, `sequences_total` and `optimizer_steps` (~1,460) must be
#                        IDENTICAL to runs/phase1_nce_temp_relu2/events.jsonl's. That is the
#                        matched-data proof and it is free:
#                          diff <(jq -c 'select(.event=="launch/data")' runs/pqm_zeta4_probe/events.jsonl) \
#                               <(jq -c 'select(.event=="launch/data")' runs/phase1_nce_temp_relu2/events.jsonl)
#   launch/model         trainable_tensors == {lora: 392, value_head: 2}. Nothing else.
#   launch/init_values   pqm/loss EQUALS the analytic pqm/loss_at_zero_rewards (asserted to
#                        1e-4 at head_init=zero) and reward_std is exactly 0.
#   launch/memory_probe  peak VRAM below the card. Expect LESS than Feynman's ~11.5 GiB --
#                        no psi/phi MLPs, no R x C distance matrix, no CF variants.
#
# DURING THE RUN, two curves decide whether it is worth finishing (§10):
#   pqm/reward_gap                  MUST OPEN. Eval thresholds exactly this separation.
#   pqm/frac_neg_below_neg_zeta     must rise -- the loss's lower absolute anchor taking hold.
# Flat by step ~300 means the head or the lr is wrong and the rest of the run is wasted.
#
# ZETA: `pqm.zeta` (this baseline's, 4) is NOT `losses.zeta` (Feynman's (3) L_T backup weight,
# 0.05/0.1). train.py prints both at launch and refuses to start if losses.zeta is set to
# something at PQM's scale. A zeta sweep was offered and declined; if the number looks bad,
#   bash pqm_baseline/train.sh --set pqm.zeta=8 --set run.name=pqm_zeta8
# is a one-line re-run and it is the first thing to try before drawing a conclusion.
#
# AFTER THE RUN:
#   python -m pqm_baseline.eval_processbench --checkpoint runs/pqm_zeta4/final
#   python -m pqm_baseline.report --pqm runs/pqm_zeta4/final \
#                                 --feynman runs/abl_cf_only/phase2/final
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # §13, every GPU shell

CONFIG="${CONFIG:-config/default.yaml}"
PQM_CONFIG="${PQM_CONFIG:-pqm_baseline/config/pqm.yaml}"
RUN_NAME="${RUN_NAME:-pqm_zeta4}"

# `run.name` is supplied here rather than in config/default.yaml so that file stays the
# Feynman configuration, byte for byte. A later `--set run.name=...` in "$@" wins: overrides
# are applied in order.
exec python -m pqm_baseline.train \
    --config "$CONFIG" \
    --pqm-config "$PQM_CONFIG" \
    --set "run.name=${RUN_NAME}" \
    "$@"
