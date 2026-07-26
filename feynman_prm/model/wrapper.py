"""One forward pass -> psi, phi, act_emb, in flat indexed tensors (PLAN 'Core design
decisions' 2).

    emb = backbone.get_input_embeddings()(input_ids)          # (B, L, H)
    h   = backbone(inputs_embeds=emb, attention_mask=...)     # (B, L, H)
    psi = psi_head(h[state_pos])                              # (S, D)
    act = pool(emb, step_spans)                               # (R, H)  differentiable
    phi = phi_head(cat([h[row_src], act], -1))                # (R, D)

Embedding once and passing `inputs_embeds` gives the action embedding for free and keeps it
tied to the input embedding table (§6.4) with no second lookup. A trajectory with T steps
yields T+1 states from ONE forward -- that is the structural win over the old token-level
project and it is what makes a ~347-negative pool affordable (§3, §8.1.1).

The heads are constructed AFTER the backbone is PEFT-wrapped, so they are trainable and the
adapter is nowhere near them (§14's LoRA traps 1 and 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..config import Config
from ..data.collate import Batch
from .distances import Distance
from .heads import GoalHead, StateActionRepresentation, StateRepresentation, build_action_pool


@dataclass
class Reps:
    """Everything the losses need from the model, in flat indexed tensors."""

    h_states: Tensor    # (S, H) last-layer hidden at every separator
    psi: Tensor         # (S, D)
    phi: Tensor         # (R, D)
    act_emb: Tensor     # (R, H)


class FeynmanPRM(nn.Module):
    def __init__(
        self,
        cfg: Config,
        hidden_size: int,
        backbone: Optional[nn.Module] = None,
        with_goal_head: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.backbone = backbone
        dims = tuple(cfg.heads.hidden_dims)
        latent = cfg.heads.latent_dim

        self.psi = StateRepresentation(hidden_size, dims, latent)
        self.phi = StateActionRepresentation(hidden_size, hidden_size, dims, latent)
        self.action_pool = build_action_pool(cfg.heads.action_pool, hidden_size)
        self.distance = Distance(cfg.distance.variant, cfg.distance.components)
        # Phase 2 only (§7.7). In phase 1 this attribute is None and the trainability assert
        # fails if a goal_head parameter shows up at all.
        self.goal_head = GoalHead(hidden_size, dims, latent) if with_goal_head else None

    # ---- forward ----

    def hidden_states(self, input_ids: Tensor, attention_mask: Optional[Tensor]) -> tuple[Tensor, Tensor]:
        """(input embeddings, last hidden state). One embedding lookup, one LM forward."""
        emb = self.backbone.get_input_embeddings()(input_ids)
        out = self.backbone(inputs_embeds=emb, attention_mask=attention_mask)
        return emb, out.last_hidden_state

    @property
    def head_dtype(self) -> torch.dtype:
        """The heads are fp32 while the backbone is bf16 (§6.3).

        They are ~1.6M parameters each, so fp32 costs almost nothing, it keeps a
        freshly-initialised head numerically stable, and it means the distance's mandatory
        fp32 cast (bug B10a) is not fighting the head's own precision. The seam is HERE:
        hidden states and action embeddings are cast once, on the way in.
        """
        return self.psi.net.out.weight.dtype

    def forward(self, batch: Batch) -> Reps:
        emb, h = self.hidden_states(batch.input_ids, batch.attention_mask)
        H = h.shape[-1]
        dtype = self.head_dtype
        h_states = h.reshape(-1, H).index_select(0, batch.state_flat_idx).to(dtype)  # (S, H)
        psi = self.psi(h_states)

        # Pooling runs in the embedding dtype (a mean over one step's tokens) and the (R, H)
        # result is cast, rather than casting the whole (B, L, H) embedding tensor -- that
        # would be ~200 MB of fp32 at the §8.1 batch shape for no accuracy that matters.
        act = self.action_pool(
            emb,
            batch.span_token_idx,
            batch.span_row_idx,
            batch.span_counts,
            batch.n_rows,
        ).to(dtype)
        phi = self.phi(h_states.index_select(0, batch.row_src), act)
        return Reps(h_states=h_states, psi=psi, phi=phi, act_emb=act)

    @torch.no_grad()
    def encode_states(self, input_ids: Tensor, attention_mask: Optional[Tensor], flat_idx: Tensor):
        """Forward-only hidden extraction at given flat positions. Used by phase 2's cache
        and by eval.

        Phase 2 exploits the fact that `h_{s_0}` is the hidden at the separator after the
        prompt and, under causal attention, depends only on `prompt + SEP` -- so it is
        IDENTICAL across every trajectory of a question and can come from one short
        prompt-only forward (§7.7). tests/test_gpu.py asserts short-forward == full-sequence.
        """
        _, h = self.hidden_states(input_ids, attention_mask)
        H = h.shape[-1]
        return h.reshape(-1, H).index_select(0, flat_idx)

    # ---- freezing ----

    def freeze_for_phase2(self) -> None:
        """Freeze backbone, psi, phi, action_pool, distance. Only goal_head trains (§7.7)."""
        for name, param in self.named_parameters():
            param.requires_grad_("goal_head" in name)
