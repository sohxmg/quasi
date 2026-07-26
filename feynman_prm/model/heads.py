"""psi / phi / goal_head, and the action pooling. THERE IS NO VALUE HEAD (locked #3a, §7.9).

Ported from `tmd-release/impls/utils/networks.py:35-60` (MLP) and `:520-557`
(StateRepresentation). The details that matter and are easy to get wrong:

* `Dense -> GELU(tanh approx) -> LayerNorm`, **post**-activation, and the FINAL Dense gets
  neither (`activate_final=False`). So psi/phi are 4 Linear layers,
  `1536 -> 512 -> 512 -> 512 -> 512`, with GELU+LN on the first three only. The latent output
  is UNNORMALISED -- do not LayerNorm it (§7.11).
* Init is Xavier-uniform with zero bias (flax `default_init()`).
* Flax LayerNorm eps is **1e-6**; PyTorch's default is 1e-5. We use 1e-6.

One head, one score: the distance. psi and phi are the two halves of that distance and
goal_head produces the destination at eval; none of them emits an independent score. A
`value_head` appearing anywhere means someone reintroduced §7.9 -- there is a grep test.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

LAYER_NORM_EPS = 1e-6  # flax default (networks.py uses nn.LayerNorm()); torch's is 1e-5


class MLP(nn.Module):
    """`(hidden_dims..., out_dim)` Dense stack, GELU+LayerNorm on the hidden layers only."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple[int, ...],
        out_dim: int,
        layer_norm: bool = True,
    ):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        self.hidden = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden_dims))
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(d, eps=LAYER_NORM_EPS) if layer_norm else nn.Identity()
            for d in hidden_dims
        )
        self.out = nn.Linear(dims[-1], out_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in list(self.hidden) + [self.out]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        for linear, norm in zip(self.hidden, self.norms):
            x = norm(F.gelu(linear(x), approximate="tanh"))
        return self.out(x)  # no activation, no LayerNorm on the latent


class StateRepresentation(nn.Module):
    """psi: h_{s_i} -> latent. `psi(h)` is a state's position in the metric space."""

    def __init__(self, hidden_size: int, hidden_dims: tuple[int, ...], latent_dim: int):
        super().__init__()
        self.net = MLP(hidden_size, hidden_dims, latent_dim)

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)


class StateActionRepresentation(nn.Module):
    """phi: concat(h_{s_{i-1}}, act_emb_i) -> latent.

    The input is the hidden at the PREVIOUS separator concatenated with the action embedding
    (§6.4). It must never be the hidden after the step -- that *is* s_i, and phi would
    collapse into psi-of-next, making L_T trivially satisfiable (§6.4).
    """

    def __init__(
        self, hidden_size: int, action_dim: int, hidden_dims: tuple[int, ...], latent_dim: int
    ):
        super().__init__()
        self.net = MLP(hidden_size + action_dim, hidden_dims, latent_dim)

    def forward(self, h_prev: Tensor, act_emb: Tensor) -> Tensor:
        return self.net(torch.cat([h_prev, act_emb], dim=-1))


class GoalHead(nn.Module):
    """PHASE 2 ONLY (§7.7, locked #15). `g_q = goal_head(h_{s_0})`.

    Phase 1 has no goal head at all: it does not exist while the metric is being trained, so
    `.detach()` is structural rather than remembered, and the head cannot corrupt the metric
    through h_{s_0}.
    """

    def __init__(self, hidden_size: int, hidden_dims: tuple[int, ...], latent_dim: int):
        super().__init__()
        self.net = MLP(hidden_size, hidden_dims, latent_dim)

    def forward(self, h_s0: Tensor) -> Tensor:
        return self.net(h_s0)


class MeanActionPool(nn.Module):
    """`act_emb_i` = mean of the LM input embeddings of step i's tokens (§6.4).

    Parameter-free and tied to the input embedding table: no extra LM forward, and it must
    not be a hidden state. Known weakness (§16.7): a mean of input embeddings is
    bag-of-words -- it captures which numbers and operators appear, not the structure of the
    reasoning. That matters MORE under `action_invariance: diagonal`, because the batch-wide
    grid used to push phi to ignore its action input and so masked a weak representation.
    """

    def forward(
        self, token_emb: Tensor, token_idx: Tensor, row_idx: Tensor, counts: Tensor, n_rows: int
    ) -> Tensor:
        flat = token_emb.reshape(-1, token_emb.shape[-1])
        gathered = flat.index_select(0, token_idx)
        out = torch.zeros(n_rows, flat.shape[-1], dtype=gathered.dtype, device=gathered.device)
        out = out.index_add(0, row_idx, gathered)
        return out / counts.to(out.dtype).clamp(min=1.0).unsqueeze(-1)


class AttentionActionPool(nn.Module):
    """`action_pool: attention` (§6.4, §16.7): a learned single-query attention over the
    step's tokens instead of a flat mean. Softmax is per segment, computed with scatter
    reductions so it stays one pass over the pooled tokens."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_size))
        self.scale = hidden_size**-0.5

    def forward(
        self, token_emb: Tensor, token_idx: Tensor, row_idx: Tensor, counts: Tensor, n_rows: int
    ) -> Tensor:
        flat = token_emb.reshape(-1, token_emb.shape[-1])
        gathered = flat.index_select(0, token_idx)
        scores = (gathered.float() @ self.query.float()) * self.scale

        row_max = torch.full((n_rows,), float("-inf"), device=scores.device)
        row_max = row_max.index_reduce(0, row_idx, scores, "amax", include_self=True)
        weights = torch.exp(scores - row_max.index_select(0, row_idx))
        denom = torch.zeros(n_rows, device=scores.device).index_add(0, row_idx, weights)
        weights = weights / denom.index_select(0, row_idx).clamp(min=1e-12)

        out = torch.zeros(n_rows, flat.shape[-1], dtype=gathered.dtype, device=gathered.device)
        return out.index_add(0, row_idx, gathered * weights.to(gathered.dtype).unsqueeze(-1))


def build_action_pool(action_pool: str, hidden_size: int) -> nn.Module:
    if action_pool == "mean":
        return MeanActionPool()
    if action_pool == "attention":
        return AttentionActionPool(hidden_size)
    raise ValueError(f"unknown action_pool {action_pool!r}")
