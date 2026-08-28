"""QRL's constrained objective, trained under Feynman-PRM's EXACT conditions.

Same parquet, same 34,650-question selection, same seed, same batch stream, same goal
sampler, same CF corpus and the same seeded attach draw, same LoRA config, same optimizer
steps, same schedule. `feynman_prm/train.py`'s launch discipline is mirrored line for line and
for the same reasons -- every one of them has a §14 or §11.1 entry behind it:

    1. resolve BOTH configs strictly                              (bug B4)
    2. build the epoch's batches through the IDENTICAL call and ASSERT the step count
       (§11.1: the old arithmetic silently produced a 106-step run and it would have
       completed). `launch/data` carries the same keys as a Feynman run's, so the two events
       diff line for line -- that is the matched-data proof, and it is free
    3. ASSERT the trainable set is exactly {LoRA, psi} (+ IQE's alpha), and that `phi` is
       NOT in it (§14, and see the phi note below)
    4. build the cosine schedule and ASSERT it steps              (bug B6)
    5. run the LONGEST batch of the epoch first, as a memory probe (PLAN 4a)
    6. check every term against its CLOSED FORM on the first micro-batch, computed from
       that batch's own MEASURED means                             (§18)

**The second optimizer is the one thing here with no counterpart upstream.** QRL's dual
variables are trained by gradient ASCENT on the same scalar the primal descends
(`lagrange.py`), at a CONSTANT lr of 1e-2 with betas (0.9, 0.999) -- QRL's
`losses/__init__.py:47`. They must not join the primal's AdamW: `param_groups` would put them
in the "heads" group at 3e-4 on the cosine schedule, so they would decay to a standstill
exactly when the constraints start binding. They are zeroed and stepped at the SAME
grad-accum boundary as the main optimizer, so a dual step sees the gradient of the same two
micro-batches the primal step does.

**`phi` is frozen and untrained here, and that is the whole point of the variant.** The
environment is deterministic, so an arrived state is `psi(prefix + step)` and is READ rather
than predicted -- `loss.py`'s header has the argument, `cf_encode.py` is the machinery. So
nothing in this objective calls `model.phi`, `assert_qrl_phase1_trainable` refuses to start if
it is trainable, and the checkpoint carries it at its random initialisation with
`qrl_phi_untrained: True` beside it. Phase 2 freezes phi anyway and eval never reads it
(`eval/processbench.py:6` scores `d(psi_i, g_q)`), so the untrained head costs a `heads.pt` key
and nothing else -- but the key must still be WRITTEN, because `utils/checkpoint.py:88-91`
fails a load whose head prefixes are missing.

Phase 2 and eval are the UNCHANGED feynman entry points -- the checkpoint is format-identical
(same `FeynmanPRM`, same head names):

    python -m feynman_prm.train_goal_head --checkpoint runs/qrl_iqe/final
    python -m feynman_prm.eval.processbench --checkpoint runs/qrl_iqe/phase2/final

Run it under tmux. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` first (§13) --
`qrl_prm/train.sh` does both.
"""

from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import os
import signal
from pathlib import Path

import torch

from feynman_prm.config import Config, load_config
from feynman_prm.data.cf_attach import select_cf_examples_for_train
from feynman_prm.data.collate import collate
from feynman_prm.data.counterfactual import read_cf_glob
from feynman_prm.data.goals import sample_goals
from feynman_prm.data.math_shepherd import read_sequences_parquet
from feynman_prm.data.sampler import (
    batch_stats,
    build_question_slots,
    epoch_batches,
    expected_sequences_per_question,
    longest_batch_index,
    planned_optimizer_steps,
    steps_report,
)
from feynman_prm.data.sequence_cache import question_ids
from feynman_prm.diagnostics.logging import RunLogger
from feynman_prm.diagnostics.probes import asymmetry_score, batch_probes
from feynman_prm.data.tokenize import sep_token_id
from feynman_prm.losses.matrix import build_matrices
from feynman_prm.model.backbone import (
    classify_trainable,
    load_backbone,
    load_tokenizer,
    param_groups,
    read_hidden_size,
    trainable_parameter_names,
)
from feynman_prm.model.wrapper import FeynmanPRM
from feynman_prm.train import MIN_OPTIMIZER_STEPS, build_scheduler
from feynman_prm.utils.checkpoint import HEAD_PREFIXES, head_state_dict, save_checkpoint
from feynman_prm.utils.seeding import epoch_rng, goal_rng, probe_rng, seed_everything

from .cf_encode import CFEncodeContext, empty_encode_info, encode_cf_psi
from .config import QRL_YAML, QRLConfig, load_qrl_config, split_overrides
from .lagrange import LagrangeMultipliers
from .loss import expected_init_values, qrl_loss

# The constraint terms are `lambda * violation` and `grad_mul` is the identity forward, so
# their closed form is EXACT -- this is fp32 rounding, not a fudge factor. The push term is
# checked one-sided (Jensen); see `expected_init_values`.
INIT_TOLERANCE = 1e-4




def largest_push_batch_index(batches, rows) -> int:
    """The batch whose `(S, C)` PUSH MATRIX is largest: states x goal columns.

    **This is NOT the same batch as `longest_batch_index`, and the gap is the whole reason
    this function exists.** `longest_batch_index` maximises `n_sequences x max_length`, which
    is the BACKBONE activation cost -- the right probe for every Feynman run, because every
    tensor those losses add is `R x C` with R and C both small.

    QRL adds an `(S, C)` push matrix that does not depend on sequence LENGTH at all: `S` is
    the state count and `C` is the number of goal columns, so a batch of many SHORT sequences
    beats a batch of few long ones by a wide margin. Probing only the long batch
    under-measures the peak by exactly the tensor this objective adds.

    **Measured 2026-08-25 on the first probe run:** `longest_batch_index` picked a 32-sequence
    batch with a 13,756-pair push matrix and reported 12.15 GB; the allocator then hit the
    wall (43 MB free of 16.6 GB) on a later, shorter batch a few steps in. Both are probed
    now, and `launch/memory_probe` reports both.

    `C` is estimated as the source rows of CORRECT trajectories, which is exactly what
    `sample_goals` draws one goal column per (`data/goals.py`) -- so this is the realised
    shape, not a bound.
    """
    def cost(batch) -> int:
        states = sum(rows[i].n_steps + 1 for i in batch)
        goals = sum(rows[i].n_steps for i in batch if rows[i].correct)
        return states * goals

    return max(range(len(batches)), key=lambda b: cost(batches[b]))


def assert_qrl_phase1_trainable(model, cfg: Config) -> dict[str, int]:
    """§14's guard, restated for a loss set with no `phi` in it.

    `feynman_prm/model/backbone.py::assert_phase1_trainable` requires `phi` to be trainable,
    because every Feynman phase-1 term reads it. **Here the opposite is required.** Nothing in
    `qrl_prm/` calls `model.phi` -- an arrived state is `psi(prefix + step)`, read rather than
    predicted -- so a trainable `phi` would sit in the optimizer's "heads" group at lr 3e-4
    collecting zero gradient, and `launch/model`'s `trainable_params` would report ~1.6M
    parameters that are not being trained. Frozen and asserted instead, so the fact is on the
    record rather than merely true.

    The same argument covers `action_pool` under `heads.action_pool: attention`: pooling exists
    to build an ACTION embedding for `phi`, and with `phi` gone there is no action to pool.
    Frozen by the caller and refused here.
    """
    buckets = classify_trainable(trainable_parameter_names(model))
    problems = []
    if not buckets["lora"]:
        problems.append("no LoRA parameters are trainable")
    if not buckets["psi"]:
        problems.append("psi is not trainable (did the heads get un-frozen AFTER PEFT wrapped?)")
    if buckets["phi"]:
        problems.append(
            "phi IS trainable -- nothing in qrl_prm/ computes it, so it would sit in the "
            "optimizer collecting no gradient while launch/model reported it as trained. "
            "The arrived state is read, not predicted (loss.py, cf_encode.py)"
        )
    if buckets["action_pool"]:
        problems.append(
            "action_pool has trainable params -- it pools an ACTION embedding for phi, and "
            "there is no phi here"
        )
    if buckets["goal_head"]:
        problems.append("goal_head is trainable in phase 1 -- it must not exist until phase 2 (§7.7)")
    if buckets["distance"] and cfg.distance.variant != "iqe":
        problems.append(f"distance has trainable params but variant={cfg.distance.variant}")
    if buckets["other"]:
        problems.append(f"unexpected trainable parameters: {sorted(buckets['other'])[:8]}")
    if problems:
        raise AssertionError("qrl phase-1 trainability assert failed: " + "; ".join(problems))
    return {k: len(v) for k, v in buckets.items()}


def check_init_values(qrl: QRLConfig, terms, info: dict, expected: dict) -> None:
    """§18's asserts, extracted so `tests/test_qrl.py` runs the SAME code the launch runs.

    Two shapes of check, and the difference is not cosmetic:

    * the CONSTRAINT terms are `lambda * violation` and `grad_mul` is the identity in the
      forward pass, so their closed form is an EQUALITY. A mismatch means the multiplier is
      not `softplus(raw)` -- the `softplus_inv` at init is the thing that goes wrong -- or the
      term is not built from the violation being logged;
    * the PUSH terms are convex transforms, so by Jensen `mean(f(d)) >= f(mean(d))` and only
      the ONE-SIDED check is exact. Asserting equality there would fire on a correct run whose
      distances have any spread at all, which is the B11/B12 failure mode (a guard that fires
      on healthy runs stops being read).
    """
    actual = {k: float(v) for k, v in terms.items()}
    for name in ("local", "path", "cf"):
        assert abs(actual[name] - expected[name]) <= max(
            INIT_TOLERANCE, INIT_TOLERANCE * abs(expected[name])
        ), (
            f"{name} term {actual[name]:.8f} != lambda * violation {expected[name]:.8f} "
            f"(§18). grad_mul is the identity in the forward pass, so this is exact -- a "
            f"mismatch means the multiplier is not softplus(raw) (check softplus_inv at init) "
            f"or the violation is not the one being logged."
        )
    for name in ("push", "neg_push", "pos_neg_push"):
        assert actual[name] >= expected[name] - INIT_TOLERANCE, (
            f"{name} {actual[name]:.8f} is BELOW its Jensen lower bound "
            f"{expected[name]:.8f} (§18). softplus is convex, so "
            f"mean(softplus(offset - d)) >= softplus(offset - mean(d)) always. Below it means "
            f"the term is not the transform of its own logged distances -- check that offset "
            f"and beta actually reach F.softplus."
        )
    assert info["qrl/push_saturated_frac"] < 0.99, (
        f"{info['qrl/push_saturated_frac']:.3f} of (state, goal) pairs are already past "
        f"softplus_offset = {qrl.softplus_offset} at INIT, where the transform is flat and the "
        f"push term has no gradient at all. The objective would be two constraints and nothing "
        f"to maximise. Raise qrl.softplus_offset."
    )


def save_qrl_checkpoint(out_dir, model, cfg: Config, qrl: QRLConfig, lagrange, tokenizer=None,
                        step=None):
    """`save_checkpoint` at the DEFAULT `HEAD_PREFIXES`, plus QRL's knobs and dual variables.

    The default prefixes already include `"distance."` (`utils/checkpoint.py:26`), which is
    what carries IQE's learned `alpha_raw` -- the launch guard below asserts that rather than
    trusting it, because a distance parameter dropped at save time would make the phase-2
    checkpoint answer to a DIFFERENT metric than the one phase 1 trained, and nothing
    downstream could tell.

    The multipliers are NOT model parameters (see `lagrange.py`), so they ride in the payload
    `extra` instead. They are saved raw (pre-softplus) so a reload is exact.

    **`phi.*` is written at its RANDOM INITIALISATION and `qrl_phi_untrained` says so.** It has
    to be written: `utils/checkpoint.py:88-91` fails any load whose `HEAD_PREFIXES` are missing,
    so dropping the prefix would make this checkpoint unloadable by phase 2 and by every eval.
    It has to be flagged: an untrained head that is byte-for-byte where a trained one lives is
    exactly the shape of thing someone reads a number off two months later. Phase 2 freezes
    `phi` and eval never calls it, so nothing downstream is wrong -- only unlabelled.
    """
    path = save_checkpoint(
        out_dir, model, cfg, tokenizer=tokenizer, step=step,
        extra={"qrl_lagrange": lagrange.raw_values(), "qrl_phi_untrained": True},
    )
    qrl.save(path / "qrl.yaml")
    return path


def run_micro_batch(model, rows, batch_rows, cfg: Config, qrl: QRLConfig, lagrange, device, rng,
                    cf_ctx=None):
    """One micro-batch: collate -> goals -> attach + encode CF -> forward -> QRL loss.

    The RNG consumption order is IDENTICAL to `feynman_prm/train.py::run_micro_batch` -- goals
    first, then the CF attach draw -- and `CFEncodeContext.choose` replicates `attach_cf`'s
    draw call for call, so at the same seed this run sees bit-identical batches, bit-identical
    goal columns and bit-identical CF selections to every baseline. That is not decoration: it
    is what makes the objective the only difference between the rows.

    **There are TWO LM forwards here**, and the second is the cost of dropping `phi`: the main
    batch, and the CF variant sequences. `encode_cf_psi` is where an arrived state stops being
    predicted and starts being read.

    `cf` carries `variant_state` as well as `psi_v` and the two id tensors, because a CF
    NEGATIVE is pushed against its host trajectory's own question's goal columns (loss.py §3).
    """
    batch_cpu = collate([rows[i] for i in batch_rows], pad_id=model.pad_id)
    goals = sample_goals(batch_cpu, cfg.discount, rng).to(device)
    enc = cf_ctx.attach(batch_cpu, rng) if cf_ctx is not None else None
    batch = batch_cpu.to(device)
    reps = model(batch)
    cf = None
    if enc is not None:
        enc = enc.to(device)
        cf = (
            encode_cf_psi(model, enc),          # THE second forward -- cf_encode.py
            enc.variant_state,
            enc.variant_example,
            enc.variant_kind,
        )
    out = qrl_loss(reps.psi, batch, goals, model.distance, qrl, lagrange, cf=cf)
    # The `cf/*` key set is logged on EVERY micro-batch, including the ones where nothing
    # attached or the budget dropped everything -- a diagnostic that disappears on a degenerate
    # batch cannot be plotted (`losses/counterfactual.py::_empty_info`'s rule).
    out.info.update(enc.info if enc is not None else empty_encode_info())
    return batch, goals, reps, out


@torch.no_grad()
def comparability_probes(reps, batch, goals, model, cfg: Config) -> dict[str, float]:
    """The Feynman diagnostic panel, computed on a QRL checkpoint under `no_grad`.

    This calls `build_matrices` and `batch_probes` VERBATIM rather than re-deriving anything.
    Diagnostic #14 -- the three-way Delta histogram over `D_term` -- is "the single best
    predictor of ProcessBench F1" (§7.6.6), and it is only worth having if the QRL row's
    `probe14/*` numbers mean exactly what `abl_cf_only`'s and `pqm_zeta4`'s do. A second
    definition here would drift from the one every other run logged, which is precisely the
    trap `losses/matrix.py::step_deltas` was extracted to close.

    **`psi[row_dst]` is passed where those two take `phi`, and that is the substitution this
    whole variant is about**, not a stand-in: `phi_r` is Feynman-PRM's PREDICTION of the state
    row `r` lands in, and `psi[row_dst]` is that state. `Dist[r,c] = d(psi(s_r), psi(g_c))` is
    then the eval-aligned quantity (`eval/processbench.py:6`), and `Next` -- which
    `build_matrices` computes as exactly that, detached -- becomes its detached twin. Only
    `D_term` feeds probe14, and `D_term` never read `phi` at all, so the diagnostic that
    matters is unaffected either way. `probe04/symmetric_share` is the one number whose
    meaning moves, and it moves onto the pair eval actually queries.

    None of these tensors touch the loss: no QRL term reads `Dist`, `Next` or `D_term`. They
    are built under `no_grad` on log steps only, so they cost transient memory and no graph.
    """
    psi_next = reps.psi.index_select(0, batch.row_dst)
    matrices = build_matrices(reps.psi, psi_next, batch, goals, model.distance, cfg)
    out = batch_probes(reps.psi, psi_next, batch, goals, matrices, model.distance, cfg)
    out.update(asymmetry_score(reps.psi, batch, model.distance))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QRL + CF variant, matched to Feynman-PRM")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--qrl-config", default=str(QRL_YAML))
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="key.path=value override. `qrl.*` goes to qrl.yaml, everything else to "
             "config/default.yaml",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--allow-short-run", action="store_true",
                        help="skip the >=300 optimizer-step assert (smoke tests only)")
    parser.add_argument("--overwrite", action="store_true",
                        help="write into a run directory that already holds a checkpoint")
    args = parser.parse_args(argv)

    feynman_overrides, qrl_overrides = split_overrides(args.set)

    # **Bug B4's exact shape, caught before the GPU is touched.** `config/default.yaml`'s
    # `losses:` block is Feynman-PRM's fixed-weight loss set, and NOT ONE KEY OF IT IS READ
    # ANYWHERE IN `qrl_prm/` -- QRL's weights are learned dual variables. So
    # `--set losses.lambda_cf=2` would run to completion, change nothing, and produce a curve
    # someone would later read as "lambda_cf 2 under QRL". Refused rather than ignored.
    # A `losses.*` value left at its yaml default is fine: it is inert and nobody typed it.
    stray = [o for o in feynman_overrides if o.split("=", 1)[0].strip().startswith("losses.")]
    if stray:
        raise SystemExit(
            "these overrides are silently INERT under qrl_prm/ -- nothing here reads "
            f"`losses.*`: {', '.join(stray)}.\n"
            "QRL has no fixed lambdas: the three weights are LEARNED Lagrange multipliers "
            "(qrl.init_lagrange_local, qrl.init_lagrange_path, qrl.init_lagrange_cf, "
            "qrl.lagrange_lr). Did you mean a "
            "`qrl.*` knob? "
            f"Legal: {', '.join(sorted(f.name for f in dataclasses.fields(QRLConfig)))}"
        )

    if args.max_steps is not None:
        feynman_overrides = feynman_overrides + [f"train.max_steps={args.max_steps}"]
    cfg = load_config(args.config, feynman_overrides)
    qrl = load_qrl_config(args.qrl_config, qrl_overrides)
    seed_everything(cfg.run.seed)

    # A probe writes to its own directory (§14 B13): `--max-steps 20` runs the identical code
    # path and ends with the identical `save_checkpoint(.../final)`, so without the suffix the
    # documented workflow -- probe, read the launch blocks, then launch for real -- leaves a
    # 20-step `final/` exactly where the real run wants to write.
    if cfg.train.max_steps is not None:
        cfg = dataclasses.replace(
            cfg, run=dataclasses.replace(cfg.run, name=f"{cfg.run.name}_probe")
        )
        print(f"[probe] max_steps={cfg.train.max_steps} -> writing to {cfg.run.name}/ "
              f"(disposable; the guard does not apply)", flush=True)

    # §14 B14: a run directory holding a `heads.pt` is not overwritten without --overwrite.
    # Keyed on `heads.pt`, not `config.yaml`: the config is a sidecar that proves nothing was
    # trained, while `heads.pt` IS the artifact.
    run_dir = Path(cfg.run.out_dir) / cfg.run.name
    existing = sorted(
        p.name for p in run_dir.glob("*") if p.is_dir() and (p / "heads.pt").exists()
    )
    if existing and cfg.train.max_steps is None and not args.overwrite:
        raise SystemExit(
            f"{run_dir} already holds checkpoint(s): {', '.join(existing)}.\n"
            f"Pass a different --set run.name=..., or --overwrite if you really mean to "
            f"discard them.\nIf these came from a --max-steps probe, they are disposable: "
            f"rm -rf {run_dir}"
        )

    logger = RunLogger(
        cfg.run.out_dir, cfg.run.name, cfg.log.wandb, cfg.log.wandb_project,
        {**cfg.to_dict(), "qrl": qrl.to_dict()},
    )
    cfg.save(run_dir / "config.resolved.yaml")
    qrl.save(run_dir / "qrl.resolved.yaml")
    logger.event(
        "launch/config",
        {
            **{f"qrl/{k}": v for k, v in qrl.to_dict().items()},
            "qrl/local_target": qrl.local_target,
            "qrl/path_target": qrl.path_target,
            "qrl/cf_target": qrl.cf_target,
            "distance/variant": cfg.distance.variant,
            "distance/components": cfg.distance.components,
            # The DELIBERATE divergence, printed where it cannot be missed. Every baseline row
            # (abl_cf_only, cf_lam2_tau005, pqm_zeta4) trained under full_mrn; the decided QRL
            # run uses IQE, so the comparison moves the objective AND the head at once.
            # `--set distance.variant=full_mrn` is the one-line control (README.md §2).
            "note_head": (
                "distance.variant=iqe is a DELIBERATE divergence from the full_mrn baselines "
                "(user's call, 2026-08-25). --set distance.variant=full_mrn is the control."
                if cfg.distance.variant == "iqe"
                else f"distance.variant={cfg.distance.variant}: this is NOT the decided QRL "
                     "configuration (iqe). Fine as a control run; say so in the report."
            ),
            # Logged side by side ON PURPOSE: `losses.*` exists in the resolved config because
            # config/default.yaml is read unchanged, and it is read by NOTHING here.
            "note_losses": (
                "config/default.yaml's `losses:` block is present in config.resolved.yaml "
                "because the shared config is read unchanged. NOT ONE KEY OF IT IS READ by "
                "qrl_prm/: there is no L_NCE, L_I, L_T, L_CF, L_step, L_good or L_term in "
                "this run. The two weights are learned Lagrange multipliers."
            ),
        },
    )

    # ---- data. THE IDENTICAL CALL -----------------------------------------------------
    # `epoch_batches` reads only `sampling.*` and the row table, so with the same parquet and
    # the same seed this is byte-identical to a Feynman run's batch stream. `launch/data`
    # below carries the same keys, so the two events diff line for line.
    sequences_path = Path(cfg.data.dir) / "sequences.parquet"
    rows = read_sequences_parquet(sequences_path, split="train")
    slots = build_question_slots(rows)
    rng = epoch_rng(cfg.run.seed, 0)
    batches = epoch_batches(rows, slots, cfg, 0, rng)
    stats = batch_stats(batches, rows)
    report = steps_report(stats["sequences_total"], cfg, n_batches=len(batches))
    steps_total = planned_optimizer_steps(len(batches), cfg.train.grad_accum) * cfg.train.epochs

    logger.event(
        "launch/data",
        {
            "questions": len(slots),
            "sequences_per_question": round(expected_sequences_per_question(slots, cfg), 3),
            "optimizer_steps": steps_total,
            "warmup_steps": report["warmup_steps"],
            **{k: round(v, 4) for k, v in stats.items()},
        },
    )
    if steps_total < MIN_OPTIMIZER_STEPS and cfg.train.max_steps is None and not args.allow_short_run:
        raise AssertionError(
            f"optimizer_steps = {steps_total} < {MIN_OPTIMIZER_STEPS} (§11.1). If this prints "
            "~106, the n_questions/grad_accum regression is back: sequences per question is "
            "min(4,k_c)+min(3,k_i) = 4.33, NOT 9.18. Cut n_questions, not grad_accum."
        )

    # ---- model ------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg)
    hidden_size = read_hidden_size(cfg.model.name)   # from config.json, never a doc (§13)
    backbone = load_backbone(cfg)                    # VERBATIM: identical LoRA + checkpointing
    model = FeynmanPRM(cfg, hidden_size, backbone=backbone, with_goal_head=False)
    model.pad_id = tokenizer.pad_token_id
    model.to(device)
    model.train()   # must precede the memory probe: eval mode disables gradient checkpointing

    lagrange = LagrangeMultipliers(
        qrl.init_lagrange_local, qrl.init_lagrange_path, qrl.init_lagrange_cf
    ).to(device)

    # ---- the CF corpus. THE IDENTICAL PATH ---------------------------------------------
    # Ported from `feynman_prm/train.py` with one change: there, the `prefix_hash` guard is
    # conditional on `losses.lambda_cf > 0`; here the CF constraint is not optional -- it is
    # one of the two constraints the objective is defined by -- so the guard is unconditional.
    cf_ctx = None
    if cfg.data.cf_glob and cfg.data.cf_max_per_batch > 0:
        cf_examples = read_cf_glob(cfg.data.cf_glob)
        # §8.2: a CF example on a VAL question would be silent leakage -- the one failure this
        # path could cause that no curve would show -- so it is asserted rather than assumed.
        kept, cf_split_info = select_cf_examples_for_train(
            cf_examples,
            {r.qid for r in rows},
            question_ids(sequences_path, split="val"),
        )
        if cf_split_info["examples_dropped_question_absent"]:
            print(
                f"[cf] {cf_split_info['examples_dropped_question_absent']} of "
                f"{len(cf_examples)} CF examples sit on "
                f"{cf_split_info['questions_absent']} question(s) that are in NO split of "
                f"sequences.parquet -- every trajectory of those questions was dropped at "
                f"tokenisation (§4.6), so there is no prefix to attach to. Dropped, and in "
                f"the launch/cf_data event.",
                flush=True,
            )
        n_hashed = sum(r.prefix_hash is not None for r in rows)
        if n_hashed == 0:
            raise AssertionError(
                f"not one of {len(rows)} rows in {cfg.data.dir}/sequences.parquet carries a "
                f"`prefix_hash` column, so NOTHING would attach and the CF CONSTRAINT -- one "
                f"of the two constraints this objective is defined by -- would train on "
                f"nothing while every other curve looked healthy (§7.5.13). Re-run "
                f"`python scripts/prepare_data.py`."
            )
        if n_hashed < len(rows):
            raise AssertionError(
                f"{len(rows) - n_hashed} of {len(rows)} rows carry no `prefix_hash` -- the "
                f"parquet is a MIX of pre- and post-2026-08-15 writes, which silently biases "
                f"which CF examples can attach. Re-run scripts/prepare_data.py."
            )
        cf_ctx = CFEncodeContext(
            kept,
            tokenizer,
            tokenizer.pad_token_id,
            cfg.data.cf_max_per_batch,
            # THE SAME ARGUMENTS `scripts/prepare_data.py` TOKENISED THE ROWS WITH. A variant
            # sequence and a trajectory row must be built the same way or `psi(prefix +
            # variant)` is not a point of the space `psi(s_i)` lives in, and nothing
            # downstream could tell.
            sep_id=sep_token_id(tokenizer, cfg.data.sep_token),
            prompt_format=cfg.data.prompt_format,
            max_len=cfg.data.max_len,
            max_tokens=qrl.cf_encode_max_tokens,
        )
        logger.event(
            "launch/cf_data",
            {
                "examples": len(kept),
                "distinct_prefixes": len(cf_ctx.index),
                "max_per_batch": cfg.data.cf_max_per_batch,
                "cf_encode_max_tokens": qrl.cf_encode_max_tokens,
                "note_cost": (
                    "data.cf_max_per_batch is now a COST knob, not a free one: every variant "
                    "is a full sequence through the backbone (qrl_prm/cf_encode.py), not an "
                    "embedding lookup and an MLP. Read cf/encode_padded_tokens against "
                    "qrl.cf_encode_max_tokens and launch/memory_probe's cf_encode fields."
                ),
                # The cap arithmetic, computed rather than remembered: at more than
                # `max_per_batch` eligible examples per micro-batch on average, a growing
                # fraction is simply not drawn in a given epoch. That is coverage, not a
                # defect -- but it must be on the record for the run, because a corpus refresh
                # changes it and nothing else would say so.
                "examples_per_micro_batch_raw": round(len(kept) / max(len(batches), 1), 2),
                "cap_binds": bool(len(kept) / max(len(batches), 1) > cfg.data.cf_max_per_batch),
                # The §7.5.13 ceiling is 100% and the seeded draw measures 91.6%. NOTE that
                # `cf_max_per_batch` now BINDS: it was sized on 27,114 examples (~9.3/batch)
                # and the 2026-08-25 snapshot is 41,380 (~14.1/batch), so `cf/attach_rate`
                # sits below 1.0 as a CAP effect. Read it against
                # `cf/examples_eligible`, not against 1.0, before calling the join broken.
                "rows_with_prefix_hash": n_hashed,
                "rows": len(rows),
                "epsilon_cf": qrl.epsilon_cf,
                "cf_neg_push_weight": qrl.cf_neg_push_weight,
                "cf_pos_neg_push_weight": qrl.cf_pos_neg_push_weight,
                **cf_split_info,
            },
        )
    else:
        raise SystemExit(
            f"data.cf_glob = {cfg.data.cf_glob!r}, data.cf_max_per_batch = "
            f"{cfg.data.cf_max_per_batch}: there is no CF data, so the CF INVARIANCE "
            "CONSTRAINT -- one of the two constraints this objective is defined by -- has "
            "nothing to constrain. Run the push + local-constraint ablation deliberately if "
            "that is what you want, but not by an empty glob."
        )

    # ---- freeze what this objective does not compute ----------------------------------
    # `FeynmanPRM.__init__` always builds `phi` and `action_pool`; nothing here calls either.
    # Frozen BEFORE `param_groups` runs, so they never enter the optimizer at all, and then
    # asserted rather than trusted -- an un-frozen head would show up as trained parameters in
    # `launch/model` and as ~1.6M AdamW states on the card, both for nothing.
    for module in (model.phi, model.action_pool):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    trainable = assert_qrl_phase1_trainable(model, cfg)   # {LoRA, psi} (+ alpha), and NO phi

    # §14's LoRA trap 3, checked at LAUNCH rather than discovered at eval. `HEAD_PREFIXES`
    # includes `"distance."`, so IQE's learned `alpha_raw` is saved -- but a checkpoint that
    # silently dropped it would load a DIFFERENT metric in phase 2 (alpha back at sigmoid(0)
    # = 0.5) and every downstream number would be wrong with nothing to see. One line here
    # instead of trusting a default two files away.
    saved_heads = head_state_dict(model)
    if cfg.distance.variant == "iqe":
        assert "distance.alpha_raw" in saved_heads, (
            f"distance.variant=iqe but `distance.alpha_raw` is not in the saved head state "
            f"dict at prefixes {HEAD_PREFIXES}. IQE's alpha is a LEARNED parameter "
            f"(model/distances.py:126); dropping it would make phase 2 and eval read a "
            f"different metric than phase 1 trained, silently."
        )

    logger.event(
        "launch/model",
        {
            "hidden_size": hidden_size,
            # expect {lora: 392, psi: ..}, and phi: 0 -- see the module docstring
            "trainable_tensors": trainable,
            "note_phi": (
                "phi is FROZEN and UNTRAINED under qrl_prm/: the environment is deterministic, "
                "so an arrived state is psi(prefix + step) and is read, not predicted. It is "
                "still saved (heads.pt must carry every HEAD_PREFIX or the load fails) and the "
                "payload carries qrl_phi_untrained: True. Phase 2 freezes it and eval scores "
                "d(psi_i, g_q), so nothing downstream reads it."
            ),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            # The fairness ledger's "new params" line: 2 scalars, plus IQE's alpha which is
            # already inside trainable_params above.
            "lagrange_params": sum(p.numel() for p in lagrange.parameters()),
            "distance_params": sum(p.numel() for p in model.distance.parameters()),
            "saved_head_tensors": len(saved_heads),
            "device": str(device),
            "attn_implementation": cfg.model.attn_implementation,
        },
    )

    optimizer = torch.optim.AdamW(
        param_groups(model, cfg),                     # LoRA 9e-6, the heads 3e-4 (§11)
        betas=tuple(cfg.train.betas),
        weight_decay=cfg.train.weight_decay,
        foreach=False,        # bug B9: the foreach path allocates a large fp32 transient
    )                         # torch AdamW, never FusedAdam (bug B8)
    scheduler = build_scheduler(optimizer, max(steps_total, 1), cfg)

    # QRL's dual optimizer (`losses/__init__.py:47`): its OWN AdamW, lr 1e-2, betas
    # (0.9, 0.999), weight_decay 0, and NO scheduler -- constant for the whole run. The betas
    # differ from the primal's (0.9, 0.95) because this is QRL's spec, not an oversight.
    dual_optimizer = torch.optim.AdamW(
        lagrange.parameters(), lr=qrl.lagrange_lr, betas=(0.9, 0.999), weight_decay=0.0,
        foreach=False,
    )
    lr_before = optimizer.param_groups[0]["lr"]
    lr_min_seen = lr_max_seen = lr_before

    # ---- the memory probes (PLAN 4a), and there are TWO of them --------------------
    # `longest_batch_index` maximises n_sequences x max_length -- the BACKBONE cost, and the
    # only probe a Feynman run needs. QRL adds an (S, C) push matrix that ignores sequence
    # length entirely, so the batch that maximises IT is a different one (see
    # `largest_push_batch_index`). Probing only the first under-measured the real peak by the
    # exact tensor this objective adds, and the allocator found the wall a few steps in.
    #
    # Each probe also runs the comparability panel, because that panel runs every `log_every`
    # steps on top of a step that has already peaked -- a probe that skipped it would
    # under-report the run's high-water mark a second way.
    probes = {"longest_sequences": longest_batch_index(batches, rows),
              "largest_push_matrix": largest_push_batch_index(batches, rows)}
    probe_report: dict[str, dict] = {}
    for label, idx in probes.items():
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        p_batch, p_goals, p_reps, probe_out = run_micro_batch(
            model, rows, batches[idx], cfg, qrl, lagrange, device, probe_rng(cfg.run.seed),
            cf_ctx=cf_ctx,
        )
        probe_out.total.backward()
        peak_loss = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
        comparability_probes(p_reps, p_batch, p_goals, model, cfg)
        peak_all = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
        optimizer.zero_grad(set_to_none=True)
        dual_optimizer.zero_grad(set_to_none=True)
        probe_report[label] = {
            "batch_index": idx,
            "sequences": len(batches[idx]),
            "max_length": max(rows[i].length for i in batches[idx]),
            "states": int(p_batch.n_states),
            "goal_columns": int(p_goals.n_goals),
            "push_pairs": int(probe_out.info["qrl/push_pairs"]),
            # The SECOND forward, reported beside the first: a probe that measured only the
            # main batch would under-report the peak by exactly the tensor this variant adds,
            # which is the same mistake `largest_push_batch_index` exists to stop making.
            "cf_encode_sequences": int(probe_out.info["cf/encode_sequences"]),
            "cf_encode_padded_tokens": int(probe_out.info["cf/encode_padded_tokens"]),
            "cf_encode_max_len": int(probe_out.info["cf/encode_max_len"]),
            "cf_examples_dropped_budget": int(probe_out.info["cf/examples_dropped_budget"]),
            "peak_vram_gb": round(peak_loss, 3) if peak_loss is not None else None,
            "peak_vram_gb_with_probes": round(peak_all, 3) if peak_all is not None else None,
            "loss": float(probe_out.total),
        }
        del p_batch, p_goals, p_reps, probe_out
        if device.type == "cuda":
            torch.cuda.empty_cache()

    peaks = [r["peak_vram_gb_with_probes"] for r in probe_report.values() if r["peak_vram_gb_with_probes"]]
    logger.event(
        "launch/memory_probe",
        {
            **probe_report,
            "worst_peak_vram_gb": max(peaks) if peaks else None,
            "push_chunk_cols": qrl.push_chunk_cols,
            "cf_encode_max_tokens": qrl.cf_encode_max_tokens,
            "note": (
                "READ `worst_peak_vram_gb`, not either probe alone. These two are DIFFERENT "
                "batches: `longest_sequences` maximises the backbone activation cost, "
                "`largest_push_matrix` maximises the (S, C) push matrix QRL adds. Both peaks "
                "now include a SECOND LM forward over the CF variant sequences -- read "
                "`cf_encode_padded_tokens` against `sampling.max_padded_tokens` to see how "
                "much of the peak it is. Three knobs, in the order to reach for them: "
                "`--set qrl.cf_encode_max_tokens=8192` trims the CF forward per batch; "
                "`--set qrl.push_chunk_cols=32` splits the push matrix by goal columns and "
                "keeps the mean exact (it lowers the transient peak of the distance's "
                "internals -- IQE materialises several (S, C, D/k, 2k) fp32 tensors -- and "
                "does NOT shrink the autograd graph); `data.cf_max_per_batch` is last, "
                "because it trims the corpus's coverage across the whole epoch rather than "
                "one batch's cost."
            ),
        },
    )

    # ---- the stall watchdog ------------------------------------------------------------
    # 2026-08-26: a run wedged mid-micro-batch burning one core at 100% with ZERO syscalls,
    # and Ctrl-C could not touch it -- SIGINT is only delivered between bytecode ops, so a
    # process inside one long C/Rust call never sees it. The only recoverable evidence was a
    # stack, and `py-spy`/`gdb` both need ptrace, which we do not have on the box.
    #
    # `dump_traceback_later` runs off a DEDICATED C thread, so it fires regardless of the GIL
    # or of what extension is running. Re-armed at the top of every micro-batch (each call
    # cancels the pending timer), so it only fires when ONE micro-batch overruns -- not on a
    # slow run. `exit=False`: dump and keep going, in case it is merely slow.
    #
    # `faulthandler.register(SIGUSR1)` is the manual version: `kill -USR1 <pid>` for a stack
    # on demand. It has SIGINT's limitation and may not land mid-C-call, which is exactly why
    # the timer above exists as well and is not redundant with it.
    watchdog_s = float(os.environ.get("QRL_WATCHDOG_S", "300"))
    if watchdog_s > 0:
        faulthandler.enable()
        try:
            faulthandler.register(signal.SIGUSR1)
        except (AttributeError, ValueError):
            pass  # not POSIX, or no stderr to write to
        print(f"[watchdog] arming at {watchdog_s:.0f}s per micro-batch; "
              f"`kill -USR1 {os.getpid()}` for a stack on demand. "
              f"QRL_WATCHDOG_S=0 disables.", flush=True)

    # ---- train -------------------------------------------------------------------------
    step = 0
    checked_init = False
    stop = False
    for epoch in range(cfg.train.epochs):
        if epoch > 0:
            batches = epoch_batches(rows, slots, cfg, epoch, epoch_rng(cfg.run.seed, epoch))
        for micro, batch_rows in enumerate(batches):
            if watchdog_s > 0:
                # Re-arm: cancels the pending timer and starts a fresh one, so the deadline is
                # per-micro-batch. A dump here names the exact line the step wedged on.
                faulthandler.dump_traceback_later(watchdog_s, repeat=False, exit=False)
            batch, goals, reps, out = run_micro_batch(
                model, rows, batch_rows, cfg, qrl, lagrange, device,
                goal_rng(cfg.run.seed, epoch, micro), cf_ctx=cf_ctx,
            )
            if not torch.isfinite(out.total):
                raise RuntimeError(f"non-finite loss at epoch {epoch} micro {micro}: {out.info}")
            (out.total / cfg.train.grad_accum).backward()

            if not checked_init:
                # §18. Every prediction below is a function of a mean THIS micro-batch
                # measured -- never a constant. The two constraint terms are exact (grad_mul
                # is the identity forward, so the term IS lambda * violation) and are
                # asserted as equalities; the push terms are convex transforms and are
                # asserted one-sided by Jensen. §7.4.3/§7.6.7's lesson is that an assumed
                # init value is how two regressions got through.
                expected = expected_init_values(qrl, lagrange, out.info)
                actual = {k: float(v) for k, v in out.terms.items()}
                logger.event(
                    "launch/init_values",
                    {
                        "expected": {k: round(v, 6) for k, v in expected.items()},
                        "actual": {k: round(v, 6) for k, v in actual.items()},
                        "push_dist_mean": round(out.info["qrl/push_dist_mean"], 4),
                        "push_saturated_frac": round(out.info["qrl/push_saturated_frac"], 4),
                        "local_dist_mean": round(out.info["qrl/local_dist_mean"], 4),
                        "local_over_cost_frac": round(
                            out.info["qrl/local_over_cost_frac"], 4
                        ),
                        "local_transitions": out.info["qrl/local_transitions"],
                        "local_violation": round(out.info["qrl/local_violation"], 6),
                        "path_dist_mean": round(out.info["qrl/path_dist_mean"], 4),
                        "path_ratio_mean": round(out.info["qrl/path_ratio_mean"], 4),
                        "path_gap_max": out.info["qrl/path_gap_max"],
                        "path_pairs": out.info["qrl/path_pairs"],
                        "path_violation": round(out.info["qrl/path_violation"], 6),
                        "cf_violation": round(out.info["qrl/cf_violation"], 6),
                        "cf_pairs": out.info["qrl/cf_pairs"],
                        "cf_positives": out.info["qrl/cf_positives"],
                        "cf_anchor_missing": out.info["qrl/cf_anchor_missing"],
                        "cf_encode_sequences": out.info["cf/encode_sequences"],
                        "pos_neg_push_dist_mean": round(
                            out.info["qrl/pos_neg_push_dist_mean"], 4
                        ),
                        "pos_neg_push_pairs": out.info["qrl/pos_neg_push_pairs"],
                        "pos_neg_push_gap": round(out.info["qrl/pos_neg_push_gap"], 4),
                        "lagrange_local": round(out.info["qrl/lagrange_local"], 6),
                        "lagrange_path": round(out.info["qrl/lagrange_path"], 6),
                        "lagrange_cf": round(out.info["qrl/lagrange_cf"], 6),
                        "note": (
                            "the push terms are LOWER bounds (softplus is convex: "
                            "mean(f) >= f(mean)); the constraint terms are EXACT. At init psi "
                            "is a random map over anisotropic LM hiddens, so "
                            "local_dist_mean starts far above step_cost = 1 and "
                            "local_violation starts strongly POSITIVE -- that is correct and "
                            "it is why lambda_local is initialised at its destination rather "
                            "than climbed to. local_transitions must equal the transition "
                            "count. path_violation starts NEAR ZERO by contrast: on an "
                            "untrained psi the distance barely grows with the gap, so the "
                            "k >= 2 rows are mostly slack and lambda_path sits armed rather "
                            "than working -- that is the regime the split exists to keep out "
                            "of the k = 1 mean. push_saturated_frac near 1.0 at init would "
                            "mean softplus_offset is below the untrained distance scale and "
                            "the push term has no gradient; raise the offset, do not proceed."
                        ),
                    },
                )
                check_init_values(qrl, out.terms, out.info, expected)
                checked_init = True

            if (micro + 1) % cfg.train.grad_accum == 0:
                # The PRE-clip norm is logged every optimizer step so the clip stays auditable
                # (§7.4.3): if train/grad_norm sits far above train.grad_clip for the whole
                # run, the clip has stopped being a guard and is acting as an LR rescale.
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.train.grad_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                # The dual step, at the SAME boundary and on the SAME accumulated gradient.
                # NOT clipped: the multipliers are two scalars whose whole job is to track the
                # constraint, and clipping the primal's norm has nothing to say about them.
                dual_optimizer.step()
                dual_optimizer.zero_grad(set_to_none=True)
                step += 1
                lr_now = optimizer.param_groups[0]["lr"]
                lr_min_seen = min(lr_min_seen, lr_now)
                lr_max_seen = max(lr_max_seen, lr_now)

                if step % cfg.train.log_every == 0 or step == 1:
                    metrics = dict(out.info)
                    # Post-step values, so the logged multiplier is the one the NEXT batch
                    # will be weighted by. lambda_cf climbing while `qrl/cf_sq_dev` does not
                    # fall is the CF corpus contradicting itself -- the dual variable is the
                    # data-quality detector.
                    metrics.update(lagrange.values())
                    metrics["train/grad_norm"] = float(grad_norm)
                    metrics["train/grad_clipped"] = float(float(grad_norm) > cfg.train.grad_clip)
                    metrics.update(comparability_probes(reps, batch, goals, model, cfg))
                    metrics["lr/backbone"] = optimizer.param_groups[0]["lr"]
                    metrics["lr/heads"] = optimizer.param_groups[-1]["lr"]
                    metrics["lr/lagrange"] = dual_optimizer.param_groups[0]["lr"]
                    logger.log(step, metrics, console=True)
                if step % cfg.train.save_every == 0:
                    save_qrl_checkpoint(
                        run_dir / f"step{step}", model, cfg, qrl, lagrange, tokenizer, step=step
                    )
                if cfg.train.max_steps is not None and step >= cfg.train.max_steps:
                    stop = True
                    break
        if stop:
            break

    # Disarmed before the final save: a checkpoint write is legitimately slow and a dump here
    # would be noise on a healthy run.
    if watchdog_s > 0:
        faulthandler.cancel_dump_traceback_later()

    # ---- the B6 guard, and the checkpoint it is not allowed to eat ----------------------
    # Read the LR over the WHOLE run, never start-vs-end: LambdaLR's constructor applies
    # lr_lambda(0) = 0.0 under warmup and a completed cosine ends at exactly 0.0, so
    # start-vs-end passes only on runs cut short by --max-steps. The save sits ABOVE the
    # raise: a diagnostic must never destroy the artifact it is diagnosing.
    lr_stuck = step > 0 and cfg.train.schedule != "constant" and lr_max_seen <= lr_min_seen

    save_qrl_checkpoint(run_dir / "final", model, cfg, qrl, lagrange, tokenizer, step=step)
    logger.event(
        "done",
        {
            "optimizer_steps": step,
            "lr_min": lr_min_seen,
            "lr_max": lr_max_seen,
            **lagrange.values(),
        },
    )
    logger.close()

    if lr_stuck:
        raise AssertionError(
            f"the LR never moved -- bug B6 (no scheduler) is back. It held {lr_max_seen:g} for "
            f"all {step} optimizer steps. The final checkpoint was written first and is intact."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
