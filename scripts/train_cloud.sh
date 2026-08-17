#!/usr/bin/env bash
# Launch phase 1 on a RENTED card (Lightning AI), WITHOUT editing config/default.yaml.
#
# Everything this script does, it does with `--set` overrides on the command line. That is
# deliberate and it is the whole point of the file: `config/default.yaml` stays the Kratos
# (RTX 5070 Ti, 16 GB) config, `bash scripts/train.sh` keeps meaning exactly what it has always
# meant, and nothing here can follow you back to the box you do most of your runs on.
#
#   RUN THIS UNDER TMUX. Same rule as scripts/train.sh (§13) -- more so on a rented box, where
#   a dropped websocket to the Studio kills a foreground process.
#
#   tmux new -s feynman
#   bash scripts/train_cloud.sh --set losses.zeta=0.2 --set run.name=phase1_cf_term_taucf01
#
# ---------------------------------------------------------------------------------------
# PROFILE -- the only knob, and the default is the CONSERVATIVE one on purpose
# ---------------------------------------------------------------------------------------
#
#   PROFILE=match   (DEFAULT)  identical batch composition to Kratos. The rented card buys
#                              WALL CLOCK and nothing else. Every loss statistic -- R, Q,
#                              nce/chance = log(R), the optimizer-step count, the §18 init
#                              values -- is bit-for-bit the same quantity the 0.560-val-F1
#                              benchmark (phase1_nce_temp_relu2) was measured on.
#
#   PROFILE=big                2x the micro-batch, grad_accum 2 -> 1. Uses the extra VRAM.
#                              **IT CHANGES THE LOSS.** See the block below before choosing it.
#
#   PROFILE=big3               3x, for a 40 GB card only. Same caveats, more of them.
#
# WHY `match` IS THE DEFAULT, AND WHY "USE THE 40 GB" IS NOT FREE.
#
# There is no lever on this codebase that converts VRAM into speed at a FIXED loss. The one
# that usually exists -- turning gradient checkpointing off -- does not fit: Qwen2.5-1.5B has
# intermediate_size 8960, so the un-checkpointed MLP activations alone are ~1 GB per layer at
# 32k tokens x 28 layers, i.e. tens of GB past any card here. `model.gradient_checkpointing`
# stays true on every profile.
#
# So the only way to spend VRAM is a bigger micro-batch, and the micro-batch IS the loss:
# every phase-1 term is a comparison between rows of ONE batch (§8.1). Doubling
# `sequences_per_micro_batch` doubles L_NCE's negative pool R (~348 -> ~700), which moves
# `nce/chance = log(R)` from ~5.85 to ~6.55 and makes ①'s task materially harder. That is a
# real change, not a scaling artifact -- and this particular run is a controlled TWO-delta
# comparison (tau_cf 1.0->0.1, zeta 0.1->0.2) against a benchmark measured at 56 sequences.
# A third, unlogged delta in the batch shape is exactly §9.10's confound.
#
# `big` does hold two things fixed that would otherwise drift, so the confound is one axis and
# not three:
#   * grad_accum 2 -> 1, so the OPTIMIZER batch stays 112 sequences per update and the step
#     count stays ~1,460 (§11.1's assert still reads what it always read).
#   * cf_max_per_batch 12 -> 24, so ④ still attaches ~9.3 examples per 56 sequences rather
#     than having the cap silently bind and shrink L_CF's per-sequence contribution.
# `term/chance` and `cf/chance` are per-question and per-example, so they do NOT move with the
# batch size. `nce/chance` does. Read it as a difference either way (diagnostic #19).
#
# ---------------------------------------------------------------------------------------
# WHICH CARD
# ---------------------------------------------------------------------------------------
#
# **A100 40 GB, not L4.** L4 is ~4x cheaper per hour and it is the wrong buy for this run:
# 24 GB against Kratos's 16 GB is +8 GB, and the run already fits in ~11.5 GiB, so the extra
# memory buys nothing at PROFILE=match. What you would be paying for is speed, and an L4 is
# not obviously faster than the card you already have -- 300 GB/s of memory bandwidth and a
# 72 W board power against the 5070 Ti's 896 GB/s. A gradient-checkpointed 1.5B backbone is
# bandwidth- and compute-bound, so the plausible outcome on L4 is "cheaper per hour, more
# hours, no faster wall clock than Kratos". The A100's 1555 GB/s and 312 TFLOPS bf16 are a
# real step up on both axes.
#
# This is an argument from the cards' specifications, not a benchmark of THIS code on THEM.
# The honest check costs ~10 minutes and is in the launch sequence below: run the 20-step
# probe and compare its s/step against Kratos's. If the L4 is within ~20% of the 5070 Ti,
# take the L4 and the 4x saving; if it is 2x slower, it is not cheaper.
#
set -euo pipefail

PROFILE="${PROFILE:-match}"

# §13, every GPU shell. `expandable_segments` matters MORE at PROFILE=big: the token cap makes
# batch shapes ragged, and a ragged allocator without it fragments into a false OOM.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# NO TF32 KNOB, and that is a decision rather than an omission. The only fp32 matmuls in this
# model are the psi/phi MLPs and the distance (§6.3, mandatory, bug B10a) -- a rounding error
# of the total FLOPs next to a bf16 1.5B backbone -- so TF32 would buy approximately nothing
# and would put a numerics change next to the one tensor this project is deliberate about.
# `flash_attention_2` IS installable on sm_80/sm_89 unlike on Kratos's sm_120, and is also not
# taken: config/default.yaml prices it at ~2% of FLOPs at max_len 1024 (PLAN 4a), which does
# not pay for a new dependency mid-run.

case "$PROFILE" in
  match)
    # Nothing. This is the point: no batch-shape override at all, so the resolved config is
    # config/default.yaml exactly as Kratos reads it.
    PROFILE_SETS=()
    ;;
  big)
    PROFILE_SETS=(
      --set sampling.sequences_per_micro_batch=112
      --set sampling.max_padded_tokens=65536
      --set train.grad_accum=1
      --set data.cf_max_per_batch=24
    )
    ;;
  big3)
    PROFILE_SETS=(
      --set sampling.sequences_per_micro_batch=168
      --set sampling.max_padded_tokens=98304
      --set train.grad_accum=1
      --set data.cf_max_per_batch=36
    )
    ;;
  *)
    echo "PROFILE must be one of: match | big | big3   (got '$PROFILE')" >&2
    exit 2
    ;;
esac

echo "=============================================================================="
echo " train_cloud.sh   PROFILE=$PROFILE"
if [ "${#PROFILE_SETS[@]}" -eq 0 ]; then
  echo "   batch shape: UNCHANGED from config/default.yaml (56 seqs / 32768 tok / accum 2)"
  echo "   -> every loss statistic is comparable to runs/phase1_nce_temp_relu2 (0.560 val F1)"
else
  echo "   batch shape overridden:"
  printf '     %s %s\n' "${PROFILE_SETS[@]}"
  echo "   -> L_NCE's pool R roughly x$([ "$PROFILE" = big ] && echo 2 || echo 3); nce/chance = log(R) MOVES."
  echo "      Read nce/loss - nce/chance, never nce/loss (diagnostic #19), and do NOT compare"
  echo "      raw NCE curves against a 56-sequence run."
fi
echo "   PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "=============================================================================="
echo

exec bash "$(dirname "$0")/train.sh" "${PROFILE_SETS[@]+"${PROFILE_SETS[@]}"}" "$@"
