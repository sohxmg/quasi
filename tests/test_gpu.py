"""GPU tests (§15's marked set). These are the ONLY tests that touch CUDA, the real
Qwen2.5-Math-1.5B-Instruct download, `transformers` and `peft`.

    pytest tests/test_gpu.py -m gpu -v -s        # on the RTX 5070 Ti, inside tmux

They exist because the CPU suite deliberately cannot see any of this: it runs on random
hiddens with no model, so it cannot catch a dtype seam, a PEFT freezing mistake, a
checkpointing interaction, or a memory figure. Everything here is a claim made somewhere in
CLAUDE.md/PLAN.md that has never been executed.

`-s` is worth passing: the memory and timing tests print measurements that replace the
estimates in IMPLEMENTATION.md §8 risk 10.

Order matters if something fails: run `test_environment_*` first. A wrong torch build or
driver makes every later failure a red herring (§13).
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

pytest.importorskip("transformers")
pytest.importorskip("peft")

from feynman_prm.config import load_config                                  # noqa: E402
from feynman_prm.data.collate import SequenceRow, collate                   # noqa: E402
from feynman_prm.data.goals import sample_goals                             # noqa: E402
from feynman_prm.data.tokenize import build_sequence, sep_token_id          # noqa: E402
from feynman_prm.losses.matrix import build_matrices                        # noqa: E402
from feynman_prm.losses.total import expected_init_values, phase1_loss      # noqa: E402
from feynman_prm.model.backbone import (                                    # noqa: E402
    LORA_TARGET_MODULES,
    assert_phase1_trainable,
    assert_phase2_trainable,
    load_backbone,
    load_backbone_with_adapter,
    load_tokenizer,
    param_groups,
    read_hidden_size,
)
from feynman_prm.model.wrapper import FeynmanPRM                            # noqa: E402
from conftest import REPO_ROOT                                              # noqa: E402


# ======================================================================================
# fixtures
# ======================================================================================


@pytest.fixture(scope="module")
def gpu_cfg():
    """Module-scoped copy of the shipped config (the `cfg` fixture is function-scoped)."""
    return load_config(REPO_ROOT / "config" / "default.yaml")


@pytest.fixture(scope="module")
def tokenizer(gpu_cfg):
    return load_tokenizer(gpu_cfg)


@pytest.fixture(scope="module")
def model(gpu_cfg, tokenizer):
    """Phase-1 model: Qwen2Model + LoRA + fresh psi/phi, on CUDA. Shared across tests, so no
    test in this file may mutate its parameter set -- the phase-2 test builds its own."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    backbone = load_backbone(gpu_cfg)
    m = FeynmanPRM(gpu_cfg, read_hidden_size(gpu_cfg.model.name), backbone=backbone)
    m.pad_id = tokenizer.pad_token_id
    return m.cuda()


def make_rows(tokenizer, cfg, specs, add_prefix=False) -> list[SequenceRow]:
    """specs = [(qid, prompt, steps, labels), ...] -> pre-tokenised rows, through the ONE
    sequence builder."""
    sep_id = sep_token_id(tokenizer, cfg.data.sep_token)
    rows = []
    for qid, prompt, steps, labels in specs:
        seq = build_sequence(
            tokenizer, prompt, list(steps), sep_id,
            prompt_format=cfg.data.prompt_format, add_prefix=add_prefix,
        )
        z = next((k for k, ok in enumerate(labels) if not ok), -1)
        rows.append(
            SequenceRow(
                qid=qid,
                input_ids=np.asarray(seq.input_ids, dtype=np.int64),
                state_pos=np.asarray(seq.state_pos, dtype=np.int64),
                span_start=np.asarray([s for s, _ in seq.step_spans], dtype=np.int64),
                span_end=np.asarray([e for _, e in seq.step_spans], dtype=np.int64),
                correct=all(labels),
                z=z,
            )
        )
    return rows


def toy_specs(n_questions=2):
    """A minimal batch with >=1 correct and >=1 incorrect per question (§8.2)."""
    specs = []
    for q in range(n_questions):
        specs.append((
            f"q{q}", f"What is {q}+2?",
            [f"Step 1: {q} + 2 = {q + 2}", f"Step 2: The answer is {q + 2}"],
            [True, True],
        ))
        specs.append((
            f"q{q}", f"What is {q}+2?",
            [f"Step 1: {q} + 2 = {q + 3}", f"Step 2: The answer is {q + 3}"],
            [False, False],
        ))
    return specs


def forward_batch(model, cfg, rows, seed=0):
    """collate -> goals -> forward -> matrices -> loss, exactly as train.py does it."""
    batch_cpu = collate(rows, pad_id=model.pad_id)
    goals = sample_goals(batch_cpu, cfg.discount, np.random.default_rng(seed)).to("cuda")
    batch = batch_cpu.to("cuda")
    reps = model(batch)
    matrices = build_matrices(reps.psi, reps.phi, batch, goals, model.distance, cfg)
    out = phase1_loss(reps.psi, reps.phi, batch, matrices, model.distance, cfg,
                      goal_traj=goals.goal_traj)
    return batch, goals, reps, matrices, out


# ======================================================================================
# A. environment preflight -- run these first (§13)
# ======================================================================================


def test_environment_is_blackwell_with_a_cu128_torch():
    """RTX 5070 Ti is sm_120, and cu128 wheels start at torch 2.7 -- torch 2.6 (CRM's pin)
    has no cu128 build and cannot run on this card at all."""
    assert torch.cuda.is_available(), "no CUDA device"
    major, minor = torch.cuda.get_device_capability()
    print(f"\ndevice: {torch.cuda.get_device_name()}  sm_{major}{minor}  "
          f"torch {torch.__version__} (cuda {torch.version.cuda})")
    assert (major, minor) >= (12, 0), f"expected sm_120 (Blackwell), got sm_{major}{minor}"
    assert torch.__version__ >= "2.8", "PLAN pins torch>=2.8 from the cu128 index"
    assert torch.version.cuda and torch.version.cuda.startswith("12.8")


def test_environment_has_enough_vram():
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"total VRAM: {total:.1f} GB")
    assert total > 15.0, "the memory budget in PLAN assumes a 16 GB card"


def test_expandable_segments_is_set():
    """§13 asks for `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in every GPU shell.
    Not fatal, but a fragmented allocator is a common cause of a late OOM."""
    import os

    if "expandable_segments" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        pytest.skip("PYTORCH_CUDA_ALLOC_CONF not set -- see SETUP.md; scripts/*.sh set it")


def test_hidden_size_comes_from_config_json(gpu_cfg):
    """§13: verify it from the downloaded config.json every time the model is re-downloaded,
    never from a doc. A shape error in psi/phi almost always means this."""
    assert read_hidden_size(gpu_cfg.model.name) == 1536


def test_separator_is_one_token_with_the_real_tokenizer(gpu_cfg, tokenizer):
    """§6.1 / CRM `eval_BoN.py:118-120`. The separator is the position a state is read at, so
    it has to be a single id."""
    assert sep_token_id(tokenizer, gpu_cfg.data.sep_token) is not None
    assert tokenizer.pad_token_id is not None


def test_step_with_an_internal_newline_keeps_its_states(gpu_cfg, tokenizer):
    """§4.7's hazard on the REAL tokenizer: 13.9% of solutions contain a step with an internal
    newline, and scanning the ids would put a state inside the step."""
    sep_id = sep_token_id(tokenizer, gpu_cfg.data.sep_token)
    steps = ["Step 1: first line\nsecond line", "Step 2: done"]
    seq = build_sequence(tokenizer, "What is 2+2?", steps, sep_id)
    assert len(seq.state_pos) == 3
    scanned = [i for i, t in enumerate(seq.input_ids) if t == sep_id]
    assert len(scanned) > len(seq.state_pos), "the in-step newline really does produce a SEP id"
    start, end = seq.step_spans[0]
    assert any(seq.input_ids[p] == sep_id for p in range(start, end))


def test_both_prompt_formats_build(gpu_cfg, tokenizer):
    """`prompt_format: raw` is CRM-faithful; `chat` is the one-line flip (§6.1 is silent on
    this, so both paths are exercised)."""
    sep_id = sep_token_id(tokenizer, gpu_cfg.data.sep_token)
    raw = build_sequence(tokenizer, "What is 2+2?", ["Step 1: 4"], sep_id, prompt_format="raw")
    chat = build_sequence(tokenizer, "What is 2+2?", ["Step 1: 4"], sep_id, prompt_format="chat")
    assert chat.length > raw.length
    assert chat.input_ids[chat.state_pos[0]] == sep_id


# ======================================================================================
# B. model construction and the trainability asserts (§14)
# ======================================================================================


def test_phase1_trainability_assert(gpu_cfg, model):
    """§14's launch guard: the trainable set is EXACTLY {LoRA, psi, phi}. A value_head or a
    goal_head here means someone reintroduced §7.9 or §7.7."""
    counts = assert_phase1_trainable(model, gpu_cfg)
    print(f"\ntrainable tensors: {counts}")
    assert counts["lora"] > 0 and counts["psi"] > 0 and counts["phi"] > 0
    assert counts["goal_head"] == 0 and counts["other"] == 0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable/1e6:.2f}M of {total/1e6:.1f}M")
    assert trainable < 0.05 * total, "LoRA + two small heads, not a full fine-tune"


def test_lora_is_on_the_seven_projections_and_never_on_a_head(model):
    """§14's LoRA trap 1: an adapter on a frozen random-init head produces garbage."""
    adapted = {
        n.split(".lora_")[0].split(".")[-1]
        for n, _ in model.named_parameters()
        if "lora_" in n
    }
    assert adapted == set(LORA_TARGET_MODULES), adapted
    assert not any("lora_" in n and (".psi." in n or ".phi." in n or "goal_head" in n)
                   for n, _ in model.named_parameters())


def test_heads_are_trainable_after_peft_wrapped_the_backbone(model):
    """§14's LoRA trap 2: PEFT freezes every non-LoRA parameter, so heads built BEFORE the
    wrap would silently train through frozen random weights."""
    assert model.psi.net.out.weight.requires_grad
    assert model.phi.net.out.weight.requires_grad


def test_param_groups_give_two_learning_rates(gpu_cfg, model):
    groups = param_groups(model, gpu_cfg)
    by_name = {g["name"]: g["lr"] for g in groups}
    assert by_name == {"backbone_lora": gpu_cfg.train.lr_backbone, "heads": gpu_cfg.train.lr_heads}


# ======================================================================================
# C. the forward pass (§6, PLAN 'Core design decisions' 2)
# ======================================================================================


def test_one_forward_produces_psi_phi_act_and_gradients(gpu_cfg, model, tokenizer):
    """Milestone §18.3: gradients reach the backbone, psi and phi; shapes are right; Dist is
    one tensor shared by (1) and (3)."""
    rows = make_rows(tokenizer, gpu_cfg, toy_specs())
    batch, goals, reps, matrices, out = forward_batch(model, gpu_cfg, rows)

    assert reps.psi.shape == (batch.n_states, gpu_cfg.heads.latent_dim)
    assert reps.phi.shape == (batch.n_rows, gpu_cfg.heads.latent_dim)
    assert reps.act_emb.shape == (batch.n_rows, model.hidden_size)
    assert matrices.Dist_backup is matrices.Dist, "built once, read by (1) and (3)"
    assert not matrices.Next.requires_grad, "tmd.py:113 -- unconditionally detached"

    out.total.backward()
    lora = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in lora)
    assert model.psi.net.out.weight.grad is not None
    assert model.phi.net.out.weight.grad is not None
    model.zero_grad(set_to_none=True)


def test_bf16_backbone_with_fp32_heads_and_fp32_distances(gpu_cfg, model, tokenizer):
    """§6.3 + bug B10a. The backbone is bf16, the heads are fp32, and the distance casts to
    fp32 regardless. In bf16 the small logit DIFFERENCES round to equal, the softmax goes
    exactly uniform and the gradient is zero -- `logit_std ~= 0` is the signature."""
    rows = make_rows(tokenizer, gpu_cfg, toy_specs())
    _, _, reps, matrices, out = forward_batch(model, gpu_cfg, rows)

    assert next(model.backbone.parameters()).dtype == torch.bfloat16
    assert model.head_dtype == torch.float32
    assert reps.psi.dtype == torch.float32
    assert matrices.Dist.dtype == torch.float32
    print(f"\nlogit_std {out.info['nce/logit_std']:.5f}  "
          f"pos {out.info['nce/logits_pos']:.4f}  neg {out.info['nce/logits_neg']:.4f}")
    assert out.info["nce/logit_std"] > 1e-3, "bug B10a: the fp32 cast is not effective at runtime"
    model.zero_grad(set_to_none=True)


def test_gradient_checkpointing_works_with_inputs_embeds(gpu_cfg, model, tokenizer):
    """`enable_input_require_grads()` is REQUIRED when the forward is given `inputs_embeds`
    instead of `input_ids` -- which is how the action embedding comes out free (PLAN 2).
    Without it the checkpointed graph is cut and no backbone gradient arrives."""
    assert gpu_cfg.model.gradient_checkpointing
    rows = make_rows(tokenizer, gpu_cfg, toy_specs())
    _, _, reps, _, out = forward_batch(model, gpu_cfg, rows)
    out.total.backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_" in n and p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
    model.zero_grad(set_to_none=True)


def test_right_padding_does_not_change_the_states(gpu_cfg, model, tokenizer):
    """Padding is on the RIGHT and the states are absolute positions, so under causal
    attention trailing pad tokens cannot influence any real position. This is what lets a
    length-homogeneous batch drop the mask entirely (PLAN 4a)."""
    specs = [
        ("q", "What is 2+2?", ["Step 1: 2 + 2 = 4", "Step 2: done"], [True, True]),
        ("q", "What is 2+2?", ["Step 1: " + "a long padded step " * 20, "Step 2: done"],
         [True, True]),
    ]
    rows = make_rows(tokenizer, gpu_cfg, specs)

    alone = collate([rows[0]], pad_id=model.pad_id).to("cuda")
    assert alone.attention_mask is None, "an unpadded batch needs no mask at all"
    padded = collate(rows, pad_id=model.pad_id).to("cuda")
    assert padded.attention_mask is not None

    with torch.no_grad():
        h_alone = model(alone).h_states
        h_padded = model(padded).h_states[: alone.n_states]
    diff = (h_alone - h_padded).abs().max()
    print(f"\nmax |h_alone - h_padded| = {float(diff):.4f} (bf16 kernel tolerance)")
    assert float(diff) < 0.05


def test_h_s0_from_a_prompt_only_forward_equals_the_full_sequence(gpu_cfg, model, tokenizer):
    """The claim phase 2's whole cost model rests on (§7.7): under causal attention `h_{s_0}`
    depends only on `prompt + SEP`, so it is identical across every trajectory of a question
    and ONE ~150-token forward per QUESTION suffices instead of one per row."""
    rows = make_rows(tokenizer, gpu_cfg, toy_specs(1))
    row = rows[0]
    s0 = int(row.state_pos[0])
    ids = torch.from_numpy(row.input_ids)[None].cuda()
    pos = torch.tensor([s0]).cuda()

    full = model.encode_states(ids, None, pos).float()
    short = model.encode_states(ids[:, : s0 + 1], None, pos).float()
    diff = (full - short).abs().max()
    print(f"\nmax |h_s0 full - h_s0 short| = {float(diff):.5f}")
    assert float(diff) < 0.05


# ======================================================================================
# D. the losses at initialisation, on the real model (§18)
# ======================================================================================


def test_initialisation_values_hit_section_18(gpu_cfg, model, tokenizer):
    """§18's probe. L_NCE at chance is log(R). L_step is EXACTLY ln 5 only when Delta = 0,
    which holds for a fixture and only approximately for real random heads -- the exact form
    is pinned in tests/test_step_loss.py; here we check the neighbourhood and, more
    importantly, that nothing is NaN and no deleted term reappeared."""
    rows = make_rows(tokenizer, gpu_cfg, toy_specs(4))
    _, _, _, matrices, out = forward_batch(model, gpu_cfg, rows)
    expected = expected_init_values(gpu_cfg, matrices.n_rows, out.info["backup/dist_mean"])

    print("\n" + "\n".join(
        f"  {k:<12} actual {float(v):+.4f}   expected {expected.get(k, float('nan')):+.4f}"
        for k, v in out.terms.items()
    ))
    for name, term in out.terms.items():
        assert torch.isfinite(term), name
    assert float(out.terms["nce"]) == pytest.approx(expected["nce"], rel=0.3)
    assert float(out.terms["step"]) == pytest.approx(math.log(5.0), abs=0.8)
    assert set(out.terms) == {"nce", "invariance", "backup", "cf", "step"}, (
        "a new term appeared -- there is no L_BT, no L_CRM and no L_correct (§7.9)"
    )
    assert float(out.terms["cf"]) == 0.0, "lambda_cf is 0 until the data exists"
    model.zero_grad(set_to_none=True)


def test_batch_composition_at_the_real_budget(gpu_cfg, model, tokenizer):
    """§8.1.1's counts, realised on the real tokenizer at the real 56-sequence budget:
    Q ~ 12.9, R ~ 348, C ~ 172, 64 pairs, ~28 distinct z. The absolute values depend on the
    synthetic step count here; what is asserted is the STRUCTURE -- C < R, distinct z equals
    the number of incorrect trajectories, and pairs > distinct z."""
    specs = []
    for q in range(13):
        for _ in range(2):
            specs.append((f"q{q}", f"Question {q}: what is the value?",
                          [f"Step {i+1}: reasoning line {i}" for i in range(6)], [True] * 6))
        for _ in range(2):
            specs.append((f"q{q}", f"Question {q}: what is the value?",
                          [f"Step {i+1}: reasoning line {i}" for i in range(6)],
                          [True, True, False, False, False, False]))
    rows = make_rows(tokenizer, gpu_cfg, specs)
    batch, goals, _, matrices, out = forward_batch(model, gpu_cfg, rows)

    print(f"\nQ={batch.n_questions} R={matrices.n_rows} C={matrices.n_goals} "
          f"pairs={out.info['step/pairs']:.0f} distinct_z={out.info['step/distinct_z']:.0f} "
          f"L_T terms={out.info['backup/terms']:.0f} padding={batch.padding_fraction:.1%}")
    assert matrices.n_goals < matrices.n_rows, "goal columns come only from correct rows"
    assert out.info["step/distinct_z"] == float(int((~batch.traj_correct).sum()))
    assert out.info["step/pairs"] > out.info["step/distinct_z"]
    assert out.info["backup/terms"] == matrices.n_rows * matrices.n_goals
    model.zero_grad(set_to_none=True)


# ======================================================================================
# E. optimisation: the schedule, NaNs, and the memory probe
# ======================================================================================


def test_three_optimizer_steps_move_the_loss_without_nan(gpu_cfg, model, tokenizer):
    from feynman_prm.train import build_scheduler

    rows = make_rows(tokenizer, gpu_cfg, toy_specs(4))
    optimizer = torch.optim.AdamW(
        param_groups(model, gpu_cfg), betas=tuple(gpu_cfg.train.betas),
        weight_decay=gpu_cfg.train.weight_decay, foreach=False,   # bugs B8, B9
    )
    scheduler = build_scheduler(optimizer, total_steps=10, cfg=gpu_cfg)

    losses, lrs = [], []
    for step in range(3):
        _, _, _, _, out = forward_batch(model, gpu_cfg, rows, seed=step)
        assert torch.isfinite(out.total), out.info
        out.total.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(out.total))
        lrs.append(optimizer.param_groups[0]["lr"])

    print(f"\nlosses {['%.4f' % v for v in losses]}   lrs {['%.2e' % v for v in lrs]}")
    assert all(math.isfinite(v) for v in losses)
    assert lrs[0] != lrs[-1], "bug B6: the scheduler must actually step"
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert torch.isfinite(param).all(), name


def test_lr_schedule_arithmetic(gpu_cfg):
    """§11.1: 3% warmup then cosine to zero, over the ASSERTED step count. A test that pins
    the LR schedule without pinning the step count is the hole that produced the 106-step
    run."""
    from feynman_prm.data.sampler import steps_report
    from feynman_prm.train import MIN_OPTIMIZER_STEPS, build_scheduler

    report = steps_report(round(23_000 * 4.33), gpu_cfg)
    assert report["optimizer_steps"] == pytest.approx(889, abs=2)
    assert report["optimizer_steps"] > MIN_OPTIMIZER_STEPS

    total = int(report["optimizer_steps"])
    param = torch.nn.Parameter(torch.zeros(1, device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=1.0, foreach=False)
    scheduler = build_scheduler(optimizer, total, gpu_cfg)
    seen = []
    for _ in range(total):
        seen.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    warmup = int(report["warmup_steps"])
    assert seen[0] < seen[warmup], "the warmup ramps up"
    assert seen[warmup] == pytest.approx(1.0, abs=0.02), "peak at the end of warmup"
    assert seen[-1] < 0.01, "cosine decays to ~0"


def test_memory_probe_at_the_full_batch_shape(gpu_cfg, model, tokenizer):
    """PLAN 'Environment and budget': ~9 GB at 56 x ~600 tokens, rising to ~12 GB on the
    longest length-grouped bucket, of 16 GB. **These are estimates; this test is what
    replaces them with a measurement.** train.py runs the same probe on the longest batch of
    the epoch BEFORE step 1, so an OOM surfaces in 30 seconds instead of three hours in."""
    specs = []
    for q in range(8):
        long_step = "Step {i}: " + "consider the following intermediate quantity carefully " * 6
        for k in range(7):
            steps = [long_step.format(i=i + 1) for i in range(6)]
            labels = [True] * 6 if k < 4 else [True, True, False, False, False, False]
            specs.append((f"q{q}", f"Question number {q}: " + "context " * 30, steps, labels))
    rows = make_rows(tokenizer, gpu_cfg, specs)[: gpu_cfg.sampling.sequences_per_micro_batch]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    batch, _, _, _, out = forward_batch(model, gpu_cfg, rows)
    out.total.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    peak = torch.cuda.max_memory_allocated() / 2**30

    print(f"\n{batch.n_traj} sequences x {batch.seq_len} tokens "
          f"({batch.n_real_tokens} real, {batch.padding_fraction:.1%} padding)")
    print(f"peak VRAM {peak:.2f} GB   fwd+bwd {elapsed:.2f} s")
    print(f"=> at grad_accum {gpu_cfg.train.grad_accum}, ~889 steps is "
          f"~{889 * gpu_cfg.train.grad_accum * elapsed / 3600:.1f} h/epoch")
    model.zero_grad(set_to_none=True)
    assert peak < 15.0, "over budget on a 16 GB card"


# ======================================================================================
# F. checkpointing (§14's LoRA trap 3, PLAN 'Core design decisions' 7)
# ======================================================================================


def test_save_resume_round_trip_reproduces_the_representations(gpu_cfg, model, tokenizer, tmp_path):
    """Save adapter + heads, rebuild a fresh model from disk, and get the SAME psi. This is
    the check the old project did not have: the stock PEFT save path writes the adapter only
    and silently drops the trained heads."""
    from feynman_prm.utils.checkpoint import load_config_from_checkpoint, load_heads, save_checkpoint

    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, model, gpu_cfg, tokenizer, step=1)
    assert (ckpt / "heads.pt").exists() and (ckpt / "adapter").exists()
    assert (ckpt / "config.yaml").exists() and (ckpt / "tokenizer").exists()
    assert load_config_from_checkpoint(ckpt) == gpu_cfg

    rows = make_rows(tokenizer, gpu_cfg, toy_specs(1))
    batch = collate(rows, pad_id=model.pad_id).to("cuda")
    with torch.no_grad():
        before = model(batch).psi.clone()

    restored = FeynmanPRM(
        gpu_cfg, read_hidden_size(gpu_cfg.model.name),
        backbone=load_backbone_with_adapter(gpu_cfg, ckpt / "adapter"),
    )
    load_heads(restored, ckpt)
    restored.pad_id = model.pad_id
    restored = restored.cuda()
    with torch.no_grad():
        after = restored(batch).psi

    diff = (before - after).abs().max()
    print(f"\nmax |psi_saved - psi_restored| = {float(diff):.6f}")
    assert float(diff) < 1e-2
    del restored
    torch.cuda.empty_cache()


def test_merge_and_unload_produces_a_standalone_backbone(gpu_cfg, tmp_path):
    """scripts/export_merged.py's core step. Kept separate from the save path so mid-run
    saves stay ~100 MB instead of 3.1 GB."""
    from peft import PeftModel

    backbone = load_backbone(gpu_cfg)
    assert isinstance(backbone, PeftModel)
    merged = backbone.merge_and_unload()
    assert not any("lora_" in n for n, _ in merged.named_parameters())
    del backbone, merged
    torch.cuda.empty_cache()


# ======================================================================================
# G. phase 2 -- the goal head on frozen representations (§7.7)
# ======================================================================================


def test_phase2_freezes_everything_except_the_goal_head(gpu_cfg, tokenizer):
    """Locked #15. Builds its own model because it mutates requires_grad."""
    m = FeynmanPRM(
        gpu_cfg, read_hidden_size(gpu_cfg.model.name),
        backbone=load_backbone(gpu_cfg), with_goal_head=True,
    ).cuda()
    m.pad_id = tokenizer.pad_token_id
    m.freeze_for_phase2()
    counts = assert_phase2_trainable(m)
    assert counts["goal_head"] > 0
    assert counts["lora"] == 0 and counts["psi"] == 0 and counts["phi"] == 0

    # ...and fitting it cannot move psi, which is what makes `.detach()` structural.
    # TWO correct trajectories per question, so the §10.1 gate has a within-question pair to
    # measure at all (a question with one correct solution contributes nothing to `within`).
    specs = []
    for q in range(2):
        for v in range(2):
            specs.append((f"q{q}", f"What is {q}+2?",
                          [f"Step 1: {q} + 2 = {q + 2}", f"Step 2: therefore {q + 2}" + " ok" * v],
                          [True, True]))
        specs.append((f"q{q}", f"What is {q}+2?",
                      [f"Step 1: {q} + 2 = {q + 3}", f"Step 2: therefore {q + 3}"], [False, False]))
    rows = make_rows(tokenizer, gpu_cfg, specs)
    batch = collate(rows, pad_id=m.pad_id).to("cuda")
    with torch.no_grad():
        reps = m(batch)
        psi_before = reps.psi.clone()
        # ONE h_{s_0} per QUESTION, not per trajectory: under causal attention it depends
        # only on prompt + SEP, so it is identical across a question's rows (§7.7).
        qids = batch.traj_qid.tolist()
        q_first = torch.tensor([qids.index(q) for q in range(batch.n_questions)], device="cuda")
        h_s0 = reps.h_states.index_select(0, batch.traj_state_offset.index_select(0, q_first)).clone()
        # ...and the targets are the terminals of CORRECT trajectories only.
        correct = torch.nonzero(batch.traj_correct, as_tuple=False).flatten()
        terminals = reps.psi.index_select(0, batch.traj_terminal.index_select(0, correct)).clone()
        target_question = batch.traj_qid.index_select(0, correct)

    from feynman_prm.losses.goal import goal_loss, terminal_spread_ratio

    optimizer = torch.optim.AdamW(m.goal_head.parameters(), lr=gpu_cfg.goal_head.lr, foreach=False)
    first = last = None
    for _ in range(20):
        loss, info = goal_loss(m.goal_head(h_s0), terminals, target_question, m.distance)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        first = float(loss) if first is None else first
        last = float(loss)
    print(f"\nL_goal {first:.4f} -> {last:.4f}   pred_var {info['goal/pred_variance']:.5f}")
    assert last < first, "the head should fit its frozen targets"

    with torch.no_grad():
        assert torch.equal(psi_before, m(batch).psi), "phase 2 cannot move psi"

    gate = terminal_spread_ratio(terminals, target_question, m.distance)
    print(f"gate ratio {gate['gate/ratio']:.3f} "
          f"(within {gate['gate/within_question_terminal_spread']:.3f}, "
          f"across {gate['gate/across_question_terminal_spread']:.3f})")
    assert math.isfinite(gate["gate/ratio"])
    del m
    torch.cuda.empty_cache()


# ======================================================================================
# H. eval scoring (§9.1) -- no benchmark download needed
# ======================================================================================


def test_processbench_scoring_path(gpu_cfg, tokenizer):
    """§9.1 end to end on hand-built samples: build with "Step N: " prefixes ADDED (locked
    #8), one forward, g_q = goal_head(h_{s_0}), d_i, Delta_i, then i* - 1.

    Uses fabricated Samples so the test needs no ProcessBench download; the real benchmark
    join is exercised by scripts/eval_processbench.sh.
    """
    from feynman_prm.eval.processbench import Sample, evaluate_subset, predict, score_samples

    m = FeynmanPRM(
        gpu_cfg, read_hidden_size(gpu_cfg.model.name),
        backbone=load_backbone(gpu_cfg, trainable=False), with_goal_head=True,
    ).cuda().eval()
    m.pad_id = tokenizer.pad_token_id

    samples = [
        Sample(id="a", problem="What is 2+2?",
               steps=("2 + 2 = 4", "So the answer is 4"), label=-1, final_answer_correct=True),
        Sample(id="b", problem="What is 3+5?",
               steps=("3 + 5 = 9", "So the answer is 9"), label=0, final_answer_correct=False),
    ]
    deltas, counters = score_samples(m, tokenizer, samples, gpu_cfg, torch.device("cuda"))
    print(f"\ncounters {counters}\ndeltas {[[round(d, 3) for d in row] for row in deltas]}")

    assert counters["scored"] == 2 and counters["over_length"] == 0
    for row, sample in zip(deltas, samples):
        assert len(row) == len(sample.steps), "Delta_1..Delta_T, one per step (§6.1)"
    predictions = predict(deltas, tau=0.347)
    assert all(-1 <= p < len(samples[0].steps) for p in predictions)
    result = evaluate_subset(deltas, samples, tau=0.347)
    assert set(result) >= {"tau", "acc_error", "acc_correct", "f1"}
    del m
    torch.cuda.empty_cache()


def test_eval_refuses_to_score_without_a_goal_head(gpu_cfg, model, tokenizer):
    """Locked #13: the goal head is the only sanctioned goal at eval. A reference-solution
    goal is a labelled skyline, not a substitute (§5.1)."""
    from feynman_prm.eval.processbench import Sample, score_samples

    assert model.goal_head is None
    with pytest.raises(RuntimeError, match="no goal head"):
        score_samples(
            model, tokenizer,
            [Sample(id="a", problem="q", steps=("s1",), label=-1, final_answer_correct=True)],
            gpu_cfg, torch.device("cuda"),
        )
