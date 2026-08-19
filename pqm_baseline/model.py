"""PQM's model: Feynman-PRM's backbone, verbatim, with PQM's value head on top.

    emb = backbone.get_input_embeddings()(input_ids)          # (B, L, H)
    h   = backbone(inputs_embeds=emb, attention_mask=...)     # (B, L, H)
    r   = value_head(h[state_flat_idx])                       # (S,)  fp32

`load_backbone(cfg)` is imported UNCHANGED from `feynman_prm.model.backbone`, so the LoRA
config, the gradient-checkpointing arming and the `AutoModel`-not-`AutoModelForCausalLM`
choice are identical to a Feynman run by construction rather than by re-derivation. The
hidden-state path mirrors `FeynmanPRM.forward` exactly -- one embedding lookup, `inputs_embeds`
(which is what `enable_input_require_grads` is armed for), `use_cache=False`, and the same
`state_flat_idx` gather -- so the two runs read the same tensor at the same positions and the
only difference is what is applied to it.

**There is no psi, no phi, no quasimetric and no goal head here.** This file is the entire
reason `pqm_baseline/` lives outside `feynman_prm/`: a value head is what
`tests/test_grep_invariants.py::test_no_value_head_anywhere` exists to keep out of the
METHOD, and it is the defining feature of this BASELINE.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from feynman_prm.config import Config
from feynman_prm.data.collate import Batch
from feynman_prm.model.backbone import classify_trainable, trainable_parameter_names

from .config import PQMConfig


class ValueHead(nn.Module):
    """`Process_Q_Model/value_model.py:22-59`, at Qwen2's config.

    Dropout then `Linear(hidden_size, 1)` -- no MLP, no activation. Qwen2's config carries no
    `summary_dropout_prob`, so the `kwargs.pop("summary_dropout_prob", 0.1)` branch
    (`value_model.py:29-30`) applies and the probability is TRL's own default 0.1, which is
    also what PQM trained deepseek-math-7b-base under.

    Attribute names are the authors' (`dropout`, `summary`) so the checkpoint's keys read the
    same as theirs, and fp32 like Feynman's heads (§6.3): ~1.5k parameters, so the precision
    costs nothing and a freshly-initialised head stays numerically stable under a bf16 base.
    `value_model.py:55-56` upcasts to the head's dtype for the same reason.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1, init: str = "zero"):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.summary = nn.Linear(hidden_size, 1)
        if init == "zero":
            # Every reward is EXACTLY 0 at step 0, which makes the launch loss closed-form
            # (loss.loss_at_zero_rewards) and the §18 check an assert rather than an eyeball.
            # It also removes an overflow: the loss evaluates `exp(r + zeta)`, Qwen's
            # massive-activation channels run O(100), and fp32 `exp` overflows above r ~= 84.
            nn.init.zeros_(self.summary.weight)
            nn.init.zeros_(self.summary.bias)
        elif init != "default":
            raise ValueError(f"unknown head_init {init!r}")

    def forward(self, hidden_states: Tensor) -> Tensor:
        output = self.dropout(hidden_states)
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        return self.summary(output)


class PQMValueModel(nn.Module):
    """The shared backbone plus PQM's value head. `forward` returns (S,) fp32 rewards, one
    per state `s_0 .. s_T` of every trajectory in the batch."""

    def __init__(
        self,
        cfg: Config,
        pqm: PQMConfig,
        hidden_size: int,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.pqm = pqm
        self.hidden_size = hidden_size
        self.backbone = backbone
        # Constructed AFTER the backbone is PEFT-wrapped (§14's LoRA traps 1 and 2): PEFT
        # freezes every non-LoRA parameter at wrap time, so a head that existed first would
        # come back frozen -- and `assert_pqm_trainable` is what catches that.
        self.value_head = ValueHead(hidden_size, pqm.head_dropout, pqm.head_init)
        self.pad_id: int = 0

    @property
    def head_dtype(self) -> torch.dtype:
        return self.value_head.summary.weight.dtype

    def hidden_states(self, input_ids: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        """`FeynmanPRM.hidden_states`'s second return, and identical to it.

        `use_cache=False` explicitly: nothing generates here, so a DynamicCache is pure cost
        (~0.9 GB at the §8.1 batch shape). Gradient checkpointing suppresses it as a side
        effect, which is exactly why it must not be left to gradient checkpointing.
        """
        emb = self.backbone.get_input_embeddings()(input_ids)
        out = self.backbone(inputs_embeds=emb, attention_mask=attention_mask, use_cache=False)
        return out.last_hidden_state

    def forward(self, batch: Batch) -> Tensor:
        h = self.hidden_states(batch.input_ids, batch.attention_mask)
        H = h.shape[-1]
        h_states = h.reshape(-1, H).index_select(0, batch.state_flat_idx)      # (S, H)
        return self.value_head(h_states.to(self.head_dtype)).squeeze(-1)       # (S,)


def assert_pqm_trainable(module: nn.Module) -> dict[str, int]:
    """The launch-time guard (§14): the trainable set is EXACTLY {LoRA, value_head}.

    Same shape as `assert_phase1_trainable`, and the same two failures it exists for:

      * `value_head` absent -> the head was constructed BEFORE PEFT wrapped the backbone and
        came back frozen. It trains on nothing, the loss still falls (LoRA alone can move the
        hiddens), and the only symptom is a baseline that is quietly not PQM.
      * `psi` / `phi` / `goal_head` present -> a Feynman head leaked in and the row stops
        being "the same run with a different head and objective", which is the whole claim.
    """
    names = trainable_parameter_names(module)
    buckets = classify_trainable(names)
    value_head = {n for n in names if "value_head" in n}
    other = buckets["other"] - value_head

    problems = []
    if not buckets["lora"]:
        problems.append("no LoRA parameters are trainable")
    if not value_head:
        problems.append(
            "value_head is not trainable (was the head constructed BEFORE PEFT wrapped the "
            "backbone? PEFT freezes every non-LoRA parameter at wrap time -- §14 trap 2)"
        )
    for leaked in ("psi", "phi", "goal_head", "action_pool", "distance"):
        if buckets[leaked]:
            problems.append(
                f"{leaked} is trainable -- this baseline has NO Feynman head at all, only "
                f"{{LoRA, value_head}}"
            )
    if other:
        problems.append(f"unexpected trainable parameters: {sorted(other)[:8]}")
    if problems:
        raise AssertionError("PQM trainability assert failed: " + "; ".join(problems))
    return {"lora": len(buckets["lora"]), "value_head": len(value_head)}


VALUE_HEAD_PREFIXES = ("value_head.",)
"""Passed to `save_checkpoint(..., prefixes=...)`. The string is owned by the CALLER and
never appears anywhere in `feynman_prm/`, which is what keeps the grep guard honest."""


def load_value_head(model: nn.Module, checkpoint_dir) -> dict:
    """Restore `value_head.*` from a `heads.pt` written with `prefixes=VALUE_HEAD_PREFIXES`.

    Not `utils.checkpoint.load_heads`: that one checks against `HEAD_PREFIXES`
    (`psi.`/`phi.`/...), none of which a PQM checkpoint carries, so it would pass on an EMPTY
    payload -- which is §14's trap 3 exactly (the stock PEFT path writes the adapter and
    silently drops the head). The whole point of the check is that it fires here.
    """
    from pathlib import Path

    payload = torch.load(
        Path(checkpoint_dir) / "heads.pt", map_location="cpu", weights_only=False
    )
    heads = payload["heads"]
    if not any(k.startswith(VALUE_HEAD_PREFIXES) for k in heads):
        raise RuntimeError(
            f"{checkpoint_dir}/heads.pt carries no `value_head.*` parameters (keys: "
            f"{sorted(heads)[:8]}) -- the trained head was dropped on the way to disk (§14 "
            "trap 3). The checkpoint scores nothing; re-train or use another one."
        )
    missing, unexpected = model.load_state_dict(heads, strict=False)
    head_missing = [k for k in missing if k.startswith(VALUE_HEAD_PREFIXES)]
    if head_missing:
        raise RuntimeError(f"heads.pt is missing value-head parameters: {head_missing}")
    if unexpected:
        raise RuntimeError(f"unexpected keys in heads.pt: {unexpected[:8]}")
    return {"step": payload.get("step"), "loaded": sorted(k for k in heads)}
