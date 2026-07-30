"""Backbone: Qwen2 + LoRA, and the launch-time trainability asserts (§6.2, §14).

`AutoModel` (Qwen2Model), **not** `AutoModelForCausalLM`: this is token classification over
hidden states, there is no LM head and no vocab-sized logits anywhere. Qwen's vocab is ~151k,
so that is a large memory saving (§6.2).

Three LoRA traps, all from §14, all of which fail silently if undone:
  1. Never put a LoRA adapter on a head -- an adapter on a frozen random-init head is garbage.
  2. Un-freeze the heads AFTER PEFT wraps the model: PEFT freezes every non-LoRA parameter.
  3. Do not rely on PEFT's save path -- it writes the adapter only and drops the trained
     heads. `utils/checkpoint.py` owns saving and asserts the head state dict is non-empty.

`transformers >= 4.56` takes `dtype=`, not `torch_dtype=` (deprecated, removed in v5). CRM
uses the old name; we do not.

Blackwell note (§14): if `CUBLAS_STATUS_NOT_SUPPORTED` appears inside SDPA `o_proj` on the
first forward, cast the model to fp32 after load and before `.cuda()`. Do NOT try
`torch.backends.cuda.preferred_blas_library("cublaslt")` -- it gets past `o_proj` and then
throws `invalid resource handle` at a layernorm.
"""

from __future__ import annotations

import torch

from ..config import Config

# The 7 Qwen2 attention + MLP projections. NEVER a head (§6.2).
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def read_hidden_size(model_name: str) -> int:
    """From the downloaded `config.json`, never from a doc (§13).

    A shape error in psi/phi almost always means this number was assumed.
    """
    from transformers import AutoConfig

    return int(AutoConfig.from_pretrained(model_name).hidden_size)


def load_tokenizer(cfg: Config):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_backbone(cfg: Config, trainable: bool = True):
    """Load Qwen2Model, wrap in LoRA, enable gradient checkpointing. Returns the PEFT model."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel

    dtype = torch.bfloat16 if cfg.train.bf16 else torch.float32
    model = AutoModel.from_pretrained(
        cfg.model.name,
        dtype=dtype,                                  # NOT torch_dtype= (deprecated)
        attn_implementation=cfg.model.attn_implementation,
    )

    if not trainable:
        for p in model.parameters():
            p.requires_grad_(False)
        return model.eval()

    lora = LoraConfig(
        r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.alpha,
        lora_dropout=cfg.model.lora.dropout,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    # `autocast_adapter_dtype` is left at PEFT's default True, so the LoRA weights are fp32
    # under a bf16 base and `lora/layer.py` casts every adapted projection's INPUT to fp32,
    # retained for backward. Across the 7 targets that is 6*1536 + 8960 = 18,176 elements
    # per token at double width: ~1.1 GiB at a 32k-token batch, ~1.9 GiB at 57k, one decoder
    # layer live at a time under checkpointing. It is what the 1.91 GiB allocation in the
    # 2026-07-26 OOM was (57,344 x 8,960 x 4 B, down_proj's input, exactly).
    # A MEMORY LEVER HELD IN RESERVE, not a bug: passing autocast_adapter_dtype=False buys
    # that back and would make a 40960 token cap comfortable. Not taken, because
    # sampling.max_padded_tokens=32768 already leaves ~4 GiB free and this is a numerics
    # change on 22.4M trainable params in a repo that is deliberate about fp32 (§6.3 heads,
    # bug B10a distances). Flip it only with a §18 init-value check.
    model = get_peft_model(model, lora)

    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Required for gradient checkpointing when the forward is given `inputs_embeds`
        # instead of `input_ids` (which is how the action embedding comes out free, PLAN 2).
        model.enable_input_require_grads()

    # `from_pretrained` returns the model in EVAL mode, and `get_peft_model` wraps it in a fresh
    # nn.Module whose own `.training` is True -- so `peft_model.training` reads True while all 28
    # Qwen2DecoderLayers underneath read False. Nothing propagates train mode upward or downward
    # at wrap time. See arm_gradient_checkpointing for what that silently costs.
    model.train()

    if cfg.model.gradient_checkpointing:
        arm_gradient_checkpointing(model)
    return model


def arm_gradient_checkpointing(model: torch.nn.Module) -> int:
    """Assert gradient checkpointing will ACTUALLY run, and arm any layer that would not.

    `GradientCheckpointingLayer.__call__` (transformers/modeling_layers.py) reads TWO things,
    and neither of them is the flag on the Qwen2Model that `gradient_checkpointing_enable()`
    is easiest to verify through:

        if self.gradient_checkpointing and self.training:      # <- both, per LAYER
            return self._gradient_checkpointing_func(...)
        return super().__call__(*args, **kwargs)               # <- silent full-activation path

    When either is False the fallback is taken per layer, nothing warns, gradients still flow
    normally, and the ONLY symptom is ~10x the activation memory. It cost one CUDA OOM in
    test_memory_probe_at_the_full_batch_shape; on a longer sequence bucket it would have cost
    three hours of a run instead. `model.training` is the trap: the PEFT wrapper reports True
    while every decoder layer under it is in eval mode straight out of `from_pretrained`.

    Returns the number of checkpointable layers. Raises if there are none, or if any is in eval
    mode -- that one is not silently fixable here, because a caller that put the backbone in
    eval mode may have meant it.
    """
    from transformers.modeling_layers import GradientCheckpointingLayer

    layers = [m for m in model.modules() if isinstance(m, GradientCheckpointingLayer)]
    if not layers:
        raise RuntimeError(
            "no GradientCheckpointingLayer found under the backbone -- gradient checkpointing "
            "cannot be enabled, and the §8.1 batch shape will not fit in 16 GB"
        )

    in_eval = [l for l in layers if not l.training]
    if in_eval:
        raise RuntimeError(
            f"{len(in_eval)} of {len(layers)} decoder layers are in EVAL mode, so gradient "
            f"checkpointing will not run and activations will not fit in 16 GB "
            f"(model.training reads {model.training} -- that is the PEFT wrapper, not the "
            f"layers). Call model.train() before training."
        )

    func = next(
        (f for m in model.modules() if (f := getattr(m, "_gradient_checkpointing_func", None))),
        None,
    )
    for layer in layers:
        if getattr(layer, "gradient_checkpointing", False):
            continue
        if getattr(layer, "_gradient_checkpointing_func", None) is None:
            if func is None:
                raise RuntimeError(
                    "gradient_checkpointing_enable() left no _gradient_checkpointing_func to "
                    "copy onto the decoder layers"
                )
            layer._gradient_checkpointing_func = func
        layer.gradient_checkpointing = True
    return len(layers)


def load_backbone_with_adapter(cfg: Config, adapter_dir):
    """Reload a phase-1 backbone from a saved adapter, frozen. Used by phase 2 and eval."""
    from peft import PeftModel
    from transformers import AutoModel

    dtype = torch.bfloat16 if cfg.train.bf16 else torch.float32
    base = AutoModel.from_pretrained(
        cfg.model.name, dtype=dtype, attn_implementation=cfg.model.attn_implementation
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()   # explicit: the PEFT wrapper's own `.training` defaults to True


def trainable_parameter_names(module: torch.nn.Module) -> set[str]:
    return {name for name, p in module.named_parameters() if p.requires_grad}


def classify_trainable(names: set[str]) -> dict[str, set[str]]:
    """Bucket trainable parameter names for the phase asserts."""
    buckets: dict[str, set[str]] = {
        "lora": set(),
        "psi": set(),
        "phi": set(),
        "goal_head": set(),
        "action_pool": set(),
        "distance": set(),
        "other": set(),
    }
    for name in names:
        if "lora_" in name:
            buckets["lora"].add(name)
        elif name.startswith("psi.") or ".psi." in name:
            buckets["psi"].add(name)
        elif name.startswith("phi.") or ".phi." in name:
            buckets["phi"].add(name)
        elif "goal_head" in name:
            buckets["goal_head"].add(name)
        elif "action_pool" in name:
            buckets["action_pool"].add(name)
        elif "alpha_raw" in name:
            buckets["distance"].add(name)
        else:
            buckets["other"].add(name)
    return buckets


def assert_phase1_trainable(module: torch.nn.Module, cfg: Config) -> dict[str, int]:
    """§14's launch-time guard: the trainable set in phase 1 is EXACTLY {LoRA, psi, phi}.

    Two config knobs legitimately add parameters and are allowed only when they are on:
    `heads.action_pool: attention` and `distance.variant: iqe` (IQE's learned alpha). A
    `value_head` or a `goal_head` in this set means someone reintroduced §7.9 or §7.7.
    """
    buckets = classify_trainable(trainable_parameter_names(module))
    problems = []
    if not buckets["lora"]:
        problems.append("no LoRA parameters are trainable")
    if not buckets["psi"]:
        problems.append("psi is not trainable (did the heads get un-frozen AFTER PEFT wrapped?)")
    if not buckets["phi"]:
        problems.append("phi is not trainable")
    if buckets["goal_head"]:
        problems.append("goal_head is trainable in phase 1 -- it must not exist until phase 2 (§7.7)")
    if buckets["action_pool"] and cfg.heads.action_pool != "attention":
        problems.append(f"action_pool has trainable params but action_pool={cfg.heads.action_pool}")
    if buckets["distance"] and cfg.distance.variant != "iqe":
        problems.append(f"distance has trainable params but variant={cfg.distance.variant}")
    if buckets["other"]:
        problems.append(f"unexpected trainable parameters: {sorted(buckets['other'])[:8]}")
    if problems:
        raise AssertionError("phase-1 trainability assert failed: " + "; ".join(problems))
    return {k: len(v) for k, v in buckets.items()}


def assert_phase2_trainable(module: torch.nn.Module) -> dict[str, int]:
    """Phase 2: the trainable set is EXACTLY {goal_head}. Everything else is frozen, which is
    what makes `.detach()` structural instead of remembered (§7.7)."""
    buckets = classify_trainable(trainable_parameter_names(module))
    if not buckets["goal_head"]:
        raise AssertionError("phase-2 trainability assert failed: goal_head is not trainable")
    leaked = {k: v for k, v in buckets.items() if k != "goal_head" and v}
    if leaked:
        raise AssertionError(
            f"phase-2 trainability assert failed: not frozen: { {k: sorted(v)[:4] for k, v in leaked.items()} }"
        )
    return {k: len(v) for k, v in buckets.items()}


def param_groups(module: torch.nn.Module, cfg: Config) -> list[dict]:
    """Two learning rates: LoRA at `lr_backbone`, fresh head params at `lr_heads` (§11)."""
    buckets = classify_trainable(trainable_parameter_names(module))
    backbone_names = buckets["lora"]
    named = dict(module.named_parameters())
    backbone = [named[n] for n in sorted(backbone_names) if named[n].requires_grad]
    heads = [
        p
        for n, p in sorted(named.items())
        if p.requires_grad and n not in backbone_names
    ]
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": cfg.train.lr_backbone, "name": "backbone_lora"})
    if heads:
        groups.append({"params": heads, "lr": cfg.train.lr_heads, "name": "heads"})
    return groups
