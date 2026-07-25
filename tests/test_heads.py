"""Head-shape tests (§6.3, `networks.py:35-60` and `:520-557`).

These are port details that no loss test would catch, and getting them wrong changes the
optimisation without changing any shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from feynman_prm.model.heads import (
    LAYER_NORM_EPS,
    AttentionActionPool,
    GoalHead,
    MeanActionPool,
    MLP,
    StateActionRepresentation,
    StateRepresentation,
)


def test_psi_is_four_linears_with_gelu_ln_on_the_first_three_only():
    """`Dense -> GELU -> LayerNorm`, POST-activation, and the final Dense gets neither
    (`activate_final=False`, networks.py:56-59). The latent output is UNNORMALISED."""
    psi = StateRepresentation(1536, (512, 512, 512), 512)
    linears = [m for m in psi.modules() if isinstance(m, nn.Linear)]
    norms = [m for m in psi.modules() if isinstance(m, nn.LayerNorm)]
    assert len(linears) == 4
    assert len(norms) == 3, "no LayerNorm on the latent"
    assert [tuple(m.weight.shape) for m in linears] == [
        (512, 1536), (512, 512), (512, 512), (512, 512)
    ]


def test_layer_norm_eps_is_flax_not_torch():
    """Flax LayerNorm eps is 1e-6; PyTorch's default is 1e-5."""
    assert LAYER_NORM_EPS == 1e-6
    psi = StateRepresentation(64, (32,), 16)
    assert all(m.eps == 1e-6 for m in psi.modules() if isinstance(m, nn.LayerNorm))


def test_init_is_xavier_uniform_with_zero_bias():
    torch.manual_seed(0)
    mlp = MLP(256, (128,), 64)
    for module in [*mlp.hidden, mlp.out]:
        assert torch.all(module.bias == 0)
        bound = (6.0 / (module.weight.shape[0] + module.weight.shape[1])) ** 0.5
        assert float(module.weight.abs().max()) <= bound + 1e-6


def test_phi_takes_the_previous_hidden_concatenated_with_the_action():
    """phi = phi(h_{i-1}, act_emb_i). It must NOT be the hidden after the step -- that IS
    s_i, and phi would collapse into psi-of-next, making L_T trivially satisfiable (§6.4)."""
    phi = StateActionRepresentation(1536, 1536, (512,), 512)
    first = next(m for m in phi.modules() if isinstance(m, nn.Linear))
    assert first.weight.shape[1] == 1536 * 2
    out = phi(torch.randn(4, 1536), torch.randn(4, 1536))
    assert out.shape == (4, 512)


def test_goal_head_reads_h_s0():
    head = GoalHead(1536, (512,), 512)
    assert head(torch.randn(3, 1536)).shape == (3, 512)


def test_attention_pool_is_a_weighted_mean_over_the_same_spans():
    """`action_pool: attention` (§16.7) must pool exactly the same token sets as the mean."""
    torch.manual_seed(0)
    emb = torch.randn(2, 5, 8)
    token_idx = torch.tensor([0, 1, 2, 5, 6])
    row_idx = torch.tensor([0, 0, 0, 1, 1])
    counts = torch.tensor([3.0, 2.0])

    mean = MeanActionPool()(emb, token_idx, row_idx, counts, 2)
    pool = AttentionActionPool(8)
    attn = pool(emb, token_idx, row_idx, counts, 2)
    assert attn.shape == mean.shape
    # the query starts at zero, so the softmax is uniform and the two coincide at init
    assert torch.allclose(attn, mean, atol=1e-5)
