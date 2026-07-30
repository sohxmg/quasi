"""Why did test_memory_probe_at_the_full_batch_shape OOM?

The traceback lands on `modeling_layers.py: return super().__call__(*args, **kwargs)` --
GradientCheckpointingLayer's NON-checkpointed branch. That branch is taken when
`self.gradient_checkpointing and self.training` is False *on the decoder layer*, which is a
different object from the Qwen2Model whose flag load_backbone sets.

This prints the layer-level truth and then counts how many times
torch.utils.checkpoint.checkpoint is actually entered during one real forward. The count is
the only thing that settles it: 28 means checkpointing is live, 0 means it is not.

    python scripts/diag_gc.py
"""

from __future__ import annotations

import torch
import torch.utils.checkpoint as ckpt

from feynman_prm.config import load_config
from feynman_prm.model.backbone import load_backbone

cfg = load_config("config/default.yaml")
peft_model = load_backbone(cfg)
base = peft_model.base_model.model                      # Qwen2Model

print(f"cfg.model.gradient_checkpointing = {cfg.model.gradient_checkpointing}")
print(f"peft_model.training              = {peft_model.training}")
print(f"Qwen2Model.gradient_checkpointing= {getattr(base, 'gradient_checkpointing', 'n/a')}")
print(f"is_gradient_checkpointing        = {getattr(base, 'is_gradient_checkpointing', 'n/a')}")

layers = base.layers
flags = [getattr(l, "gradient_checkpointing", None) for l in layers]
trains = [l.training for l in layers]
funcs = [getattr(l, "_gradient_checkpointing_func", None) is not None for l in layers]
print(f"\n{len(layers)} decoder layers")
print(f"  layer.gradient_checkpointing True : {sum(bool(f) for f in flags)}")
print(f"  layer.training True               : {sum(trains)}")
print(f"  layer._gradient_checkpointing_func: {sum(funcs)}")
print(f"  layer[0] type                     : {type(layers[0]).__mro__[:3]}")

# adapter dtype: get_peft_model(autocast_adapter_dtype=True) upcasts LoRA to fp32, which makes
# every LoRA input a separate fp32 copy retained for backward.
q = layers[0].self_attn.q_proj
print(f"\n  base weight dtype  : {q.base_layer.weight.dtype}")
print(f"  lora_A weight dtype: {q.lora_A['default'].weight.dtype}")

# --- the decisive test: count real checkpoint() entries in one forward ---
calls = {"n": 0}
real = ckpt.checkpoint


def counting_checkpoint(*args, **kwargs):
    calls["n"] += 1
    return real(*args, **kwargs)


ckpt.checkpoint = counting_checkpoint
# transformers binds functools.partial(checkpoint, ...) at enable time, so patch the already
# bound partial too.
for l in layers:
    f = getattr(l, "_gradient_checkpointing_func", None)
    if f is not None:
        kw = getattr(f, "keywords", {}) or {}
        l._gradient_checkpointing_func = lambda *a, _kw=kw, **k: counting_checkpoint(*a, **{**_kw, **k})

model = peft_model.cuda()
ids = torch.randint(0, 1000, (2, 128), device="cuda")
emb = model.get_input_embeddings()(ids)
out = model(inputs_embeds=emb, attention_mask=torch.ones_like(ids))
out.last_hidden_state.sum().backward()

print(f"\ncheckpoint() entered {calls['n']} times during one forward "
      f"({len(layers)} = live, 0 = OFF)")
