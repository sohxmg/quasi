#!/usr/bin/env bash
# QRL + counterfactual invariance, matched to Feynman-PRM.
# RUN THIS UNDER TMUX -- tmux is mandatory for every GPU run (§13).
#
#   tmux new -s qrl
#   bash qrl_prm/train.sh --max-steps 20      # the probe -> runs/qrl_iqe_probe/
#   bash qrl_prm/train.sh                     # ~1,464 optimizer steps
#
# TARGET MACHINE: kratoss, ONE 5070 Ti 16 GB. Local only -- there is no Modal wiring for this
# run and there is not meant to be.
#
# PREREQUISITE, and it is the whole point of a matched run: `data/processed/sequences.parquet`
# must be the SAME FILE the baseline runs read, at `n_questions: 34650`. Check it before
# launching -- if `data/processed/selection.json`'s `selection_sha_train` does not match the
# one those runs used, the comparison is not matched and nothing else matters:
#
#   python -c "import json;print(json.load(open('data/processed/selection.json'))['selection_sha_train'])"
#
# WHAT THIS RUNS: QRL's constrained objective -- maximize distances everywhere (global push),
# subject to (a) observed transitions cost about one step and (b) a counterfactual equivalence
# class is one point. Both constraints carry their own learned Lagrange multiplier. NONE of
# Feynman-PRM's phase-1 losses are computed: no L_NCE, L_I, L_T, L_CF, L_step, L_good, L_term.
# `config/default.yaml` is read UNCHANGED and never edited -- `run.name` and `distance.variant`
# come through --set, and QRL's own knobs live in `qrl_prm/config/qrl.yaml`.
#
# READ FIVE THINGS OFF THE PROBE'S `events.jsonl` BEFORE THE REAL LAUNCH:
#
#   launch/data          `questions`, `sequences_total` and `optimizer_steps` (~1,464) must be
#                        IDENTICAL to the baselines'. That is the matched-data proof, and it
#                        is free:
#                          diff <(jq -c 'select(.event=="launch/data")' runs/qrl_iqe_probe/events.jsonl) \
#                               <(jq -c 'select(.event=="launch/data")' runs/abl_cf_only/events.jsonl)
#   launch/model         trainable_tensors is {lora: 392, psi: .., distance: 1} and phi: 0 --
#                        `distance: 1` is IQE's learned alpha and it MUST be there at
#                        variant=iqe. `lagrange_params: 2`.
#   launch/cf_data       `examples` ~41.4k on the 2026-08-25 snapshot (the corpus GROWS
#                        between runs -- data/cf_train/MANIFEST.md has the history; the three
#                        filenames in data.cf_glob never change, only their contents) and
#                        `rows_with_prefix_hash == rows`. If the second is not an equality the
#                        run aborts, by design. `examples_dropped_question_absent` should be a
#                        handful; a leaked VAL question is fatal at any count (§8.2).
#                        NOTE cf_max_per_batch=12 now BINDS (~14.1 eligible/batch at 41,380),
#                        so cf/attach_rate reads below 1.0 as a CAP effect, not a broken join.
#   launch/init_values   the two CONSTRAINT terms equal lambda*violation exactly (asserted);
#                        `push` is at or above its Jensen bound (asserted); and
#                        `push_saturated_frac` is NOT near 1.0 -- if it is, softplus_offset is
#                        below the untrained distance scale and the push term has no gradient.
#   launch/memory_probe  BOTH peaks under the card's 16 GB. `peak_vram_gb_with_probes` is the
#                        real high-water mark -- it includes the every-10-steps diagnostic
#                        panel. IQE is heavier per pair than MRN. If it does not fit:
#                          bash qrl_prm/train.sh --set qrl.push_chunk_cols=32
#                        which splits the (S, C) push matrix by goal columns and keeps the
#                        mean exact.
#
# DURING THE RUN, three curves decide whether it is worth finishing:
#   qrl/local_dist_mean   -> 1.0. THE RULER, and the direct answer to IMPLEMENTATION.md §9's
#                         decaying `backup/delta_mean`. Watch it against that history.
#   qrl/lagrange_local, qrl/lagrange_cf   must RISE then STABILISE. A multiplier that climbs
#                         monotonically for the whole run while its violation does not fall
#                         means the constraint cannot be satisfied -- for lambda_cf that means
#                         the CF corpus contradicts itself, and the dual variable is the
#                         data-quality detector that says so.
#   qrl/push_saturated_frac   rising towards 1.0 means the objective has run out of the
#                         softplus's sloped region and stopped maximising anything.
#
# AFTER THE RUN -- phase 2 and eval are the UNCHANGED feynman entry points, because the
# checkpoint is format-identical (same FeynmanPRM, same head names):
#   python -m feynman_prm.train_goal_head --checkpoint runs/qrl_iqe/final
#   python -m feynman_prm.eval.processbench --checkpoint runs/qrl_iqe/phase2/final
#   python -m qrl_prm.report --qrl runs/qrl_iqe/phase2/final --run-dir runs/qrl_iqe
#
# The `natural_tau = 0.347` line the eval prints is RULER-BASED (-log gamma) and is merely
# informational for a QRL checkpoint, whose ruler is step_cost = 1.0. The 203-point sweep on
# the 2,000 held-out val questions is what decides tau.
#
# ABLATIONS, one line each:
#   bash qrl_prm/train.sh --set qrl.cf_encode_max_tokens=8192 --set run.name=qrl_cheap  # smaller CF fwd
#   bash qrl_prm/train.sh --set qrl.cf_neg_push_weight=0 --set run.name=qrl_nonegpush
#   bash qrl_prm/train.sh --set distance.variant=full_mrn --set run.name=qrl_mrn  # head control
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # §13, every GPU shell

CONFIG="${CONFIG:-config/default.yaml}"
QRL_CONFIG="${QRL_CONFIG:-qrl_prm/config/qrl.yaml}"
RUN_NAME="${RUN_NAME:-qrl_iqe}"
VARIANT="${VARIANT:-iqe}"

# `run.name` and `distance.variant` are supplied here rather than in config/default.yaml so
# that file stays the Feynman configuration, byte for byte. A later `--set ...` in "$@" wins:
# overrides are applied in order.
exec python -m qrl_prm.train \
    --config "$CONFIG" \
    --qrl-config "$QRL_CONFIG" \
    --set "run.name=${RUN_NAME}" \
    --set "distance.variant=${VARIANT}" \
    "$@"
