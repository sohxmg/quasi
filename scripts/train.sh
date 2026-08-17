#!/usr/bin/env bash
# Phase 1. RUN THIS UNDER TMUX -- tmux is mandatory for every GPU run (§13).
#
#   tmux new -s feynman
#   bash scripts/train.sh                     # full run, ~1,460 optimizer steps, 3-6 h
#                                             # PREREQUISITE: prepare_data.py must have been
#                                             # run at the CURRENT n_questions (34650).
#   bash scripts/train.sh --max-steps 20      # the short GPU probe (PLAN step 4). Writes to
#                                             # runs/<run.name>_probe/ -- a probe is disposable
#                                             # by construction, so it gets its own directory
#                                             # and never collides with the real run.
#
# ON A RENTED CARD, USE `scripts/train_cloud.sh` INSTEAD (2026-08-18). It is a thin wrapper
# that adds `--set` overrides for a bigger GPU and then execs this file, so THIS file and
# `config/default.yaml` stay the Kratos (16 GB) configuration and nothing there follows you
# back. Its default profile changes the batch shape NOT AT ALL -- read its header for why
# "use the extra VRAM" is not free on this loss set.
#
# WHAT THIS RUN CARRIES (2026-08-15) -- TWO changes, bundled deliberately:
#
#   lambda_cf    0.0 -> 1.0   (4) L_CF enters a gradient for the FIRST time. Locked #4's two
#                             gates are cleared: 27,114 examples on disk (§7.5.2) and the
#                             main-batch attach path wired (§7.5.13). tau stays 1.0.
#   lambda_term  0.0 -> 1.0   (7) L_term, §16.26's root-cause candidate -- the only term in the
#                             set that says "pull these together" rather than "stop pushing
#                             these apart". tau is now losses.lambda_term_temperature, 1.0,
#                             picked in this change as §7.13 requires.
#
# THE BUNDLE IS A CHOSEN CONFOUND, NOT AN OVERSIGHT. The human was offered the sequenced pair
# (one change each, ~8h) and chose the bundle (~4h) for the faster signal. So §16.27's
# complaint about the 2026-08-04 run applies here too: a move in F1 or gate/recall_at_1 CANNOT
# be assigned to (4) or (7) from this run alone. The separating re-runs, each one line:
#   bash scripts/train.sh --set losses.zeta=0.1 --set losses.lambda_term=0.0 --set run.name=abl_cf_only
#   bash scripts/train.sh --set losses.zeta=0.1 --set losses.lambda_cf=0.0   --set run.name=abl_term_only
# Both weights are exact reverts at 0.0 -- (4) is skipped in total.py and (7) is still computed
# and logged -- so either ablation reproduces the other term's contribution exactly.
#
# LAUNCH IT AGAINST THE 2026-08-04 BASELINE, WHICH MEANS PASSING ZETA:
#
#   bash scripts/train.sh --set losses.zeta=0.1
#
# `runs/phase1_nce_temp_relu2/config.resolved.yaml` records zeta = 0.1, and it got there from
# the COMMAND LINE -- this file still ships the TMD-faithful 0.05 (§11, deliberately). Omit the
# override and the bundle is THREE changes instead of two. Everything else -- tau_NCE =
# sqrt(512), L_good relu_squared, nce_mask_nearer_same_traj -- is already the shipped default.
#
# BEFORE THE FIRST LAUNCH AFTER 2026-08-15: `python scripts/prepare_data.py` MUST have been
# re-run. A sequences.parquet written before that date has no `prefix_hash` column, every
# state hashes to 0, `cf_attach` never matches on 0, and lambda_cf = 1.0 trains on NOTHING
# while every log line stays healthy (§7.5.13). train.py now REFUSES to launch in that state
# rather than let it run; the in-run tell is `cf/attach_rate`, which must sit near 0.916.
#
# THE PARQUET IS NO LONGER READ IN THIS PROCESS (2026-08-16, §14 B15). `munmap_chunk():
# invalid pointer` aborted the launch on the free() of the pandas/pyarrow objects, in a
# process that also holds torch. `feynman_prm/data/sequence_cache.py` converts
# sequences.parquet ONCE in a child interpreter with no torch in it; every reader then loads
# `sequences.cache.npz` with numpy alone. It happens automatically on the first launch after
# a new parquet -- to do it up front, and to see the failure on its own rather than inside a
# launch, run:
#
#   python -m feynman_prm.data.sequence_cache --parquet data/processed/sequences.parquet
#
# If THAT aborts natively too, pandas+pyarrow are broken in the env independently of torch
# (mixed conda-forge/pip numpy+pyarrow is the usual cause) -- fix the env, do not work around
# it. `FEYNMAN_SEQUENCE_CACHE=0 bash scripts/train.sh` restores the old in-process read.
#
# AFTER THE RUN, (7) OBLIGES ONE MORE MEASUREMENT -- post-hoc, no training time (§7.13.1):
#
#   python scripts/goal_gate.py --checkpoint runs/phase1_cf_term/final --mask-answer
#
# Every correct solution of a question prints the same final answer and incorrect siblings
# essentially never do (answer_match_auc 0.927 against a chance of 0.500), so (7) can be solved
# by clustering on the printed NUMBER -- which transfers to nothing, because a PRM scores
# UNFINISHED solutions. That flag scores the gate twice on the IDENTICAL rows, with and without
# the answer span in the input, and prints both. Recall that survives masking is structure;
# recall that collapses under it was the string. An improvement reported without this pair does
# not distinguish the two, and §16.26 asks for the run to be designed to separate them.
#
# ---------------------------------------------------------------------------------------
# WHAT THE PREVIOUS RUN CARRIED (2026-08-04) -- three changes at once, and a real confound:
#
#   1. tau_NCE  1.0 -> sqrt(512) = 22.627   TMD's own scaling (tmd.py:92), reversing §7.2's
#                                           documented divergence. Flattens the L_NCE softmax
#                                           and shrinks its gradient on the distances ~22.6x,
#                                           so it is a REWEIGHTING of (1) as much as a
#                                           temperature. Watch nce/loss against nce/chance:
#                                           pinned at log(R) with pos ~= neg means revert.
#                                           READ logit_std x tau, never logit_std -- the
#                                           scale-free quantity is the DISTANCE spread.
#   2. L_good   relu -> relu_squared        The linear hinge moved the bulk and lost the tail
#                                           (Delta mean +0.392 -> -0.412 while
#                                           frac_above_natural regressed 0.070 -> 0.16 and
#                                           delta_max hit 7.58, §7.12). The square prices a
#                                           violator by how far out it is. NOT MEASURED at
#                                           this form anywhere.
#   3. nce_mask_nearer_same_traj  ON        §16.4 / §9.9.2, open for 11 days. Goal at s_6 with
#                                           positive phi_3 marks phi_4/phi_5/phi_6 wrong even
#                                           though they are NEARER the goal. 41.8% of columns,
#                                           and it also dissolves the 29.6% duplicate-column
#                                           contradiction (probe01/distinct_goal_ratio 0.704).
#
# Three changes cannot be attributed to one. The separating re-runs, each one line:
#   bash scripts/train.sh --set losses.nce_temperature=1.0            --set run.name=abl_tau1
#   bash scripts/train.sh --set losses.good_loss.form=relu            --set run.name=abl_relu
#   bash scripts/train.sh --set sampling.nce_mask_nearer_same_traj=false --set run.name=abl_nomask
#   bash scripts/train.sh --set losses.lambda_good=0.0                --set run.name=abl_nogood
# (lambda_good=0.0 leaves (6) computed and logged -- inert, not absent.)
# The (4) ablation is the same shape:
#   bash scripts/train.sh --set losses.zeta=0.1 --set losses.lambda_cf=0.0 --set run.name=abl_nocf
# (lambda_cf=0.0 skips the term in total.py and the total is bit-identical to the 2026-08-04
# run's, with the full cf/* key set still logged -- so this reproduces the baseline exactly
# rather than approximating it.)
#
# ONE DIRECTORY PER LOSS-SET CHANGE. `run.name` is `phase1_cf_term`;
# `runs/phase1_nce_temp_relu2/` is the 2026-08-04 run and the baseline this one is read
# against, and `runs/phase1/` is the older one every §9.3.1 / §9.7 number was measured on.
# train.py now
# ABORTS before touching the GPU if the target directory already holds a checkpoint -- change
# `run.name`, or pass --overwrite if discarding it is really the intent. `--max-steps` runs are
# exempt and land in `<run.name>_probe/`, so probing never blocks the run it is probing for.
#
# The launch sequence prints, in order: the step-count assert (§11.1), the trainability
# assert (§14), the longest-batch memory probe (PLAN 4a), and the initialisation values
# against §18 -- which include L_good's sandwich IN THE CONFIGURED FORM and the SIGN of its
# margin (c must print NEGATIVE). Read all four before walking away.
#
# ALWAYS TRAIN FRESH: a loss-set change invalidates the checkpoint, so nothing here is
# resumable from a previous run's step*/.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # §13, every GPU shell

CONFIG="${CONFIG:-config/default.yaml}"
exec python -m feynman_prm.train --config "$CONFIG" "$@"
