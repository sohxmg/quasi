"""The assembly seam: one forward -> psi, phi, act_emb (PLAN 'Core design decisions' 2).

Uses a stub backbone so the wiring is tested on CPU with no download. `tests/test_gpu.py`
runs the same path against the real Qwen2Model.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from feynman_prm.data.collate import collate
from feynman_prm.data.goals import sample_goals
from feynman_prm.diagnostics.probes import batch_probes
from feynman_prm.losses.matrix import build_matrices
from feynman_prm.losses.total import phase1_loss
from feynman_prm.model.wrapper import FeynmanPRM
from conftest import synthetic_row

T, F = True, False
HIDDEN = 32


class StubBackbone(nn.Module):
    """Stands in for Qwen2Model: an embedding table plus one linear "layer"."""

    def __init__(self, vocab: int = 64, hidden: int = HIDDEN):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden)

    def get_input_embeddings(self) -> nn.Module:
        return self.emb

    def forward(self, inputs_embeds=None, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.proj(inputs_embeds))


@pytest.fixture
def tiny_cfg(cfg):
    return dataclasses.replace(
        cfg, heads=dataclasses.replace(cfg.heads, latent_dim=32, hidden_dims=(16, 16))
    )


@pytest.fixture
def model(tiny_cfg):
    torch.manual_seed(0)
    m = FeynmanPRM(tiny_cfg, HIDDEN, backbone=StubBackbone())
    m.pad_id = 0
    return m


@pytest.fixture
def batch():
    rows = [
        synthetic_row("q1", [T, T, T]),
        synthetic_row("q1", [T, F, F]),
        synthetic_row("q2", [T, T]),
        synthetic_row("q2", [F, F, F, F]),
    ]
    return collate(rows, pad_id=0)


def test_one_forward_gives_every_representation(model, batch, tiny_cfg):
    reps = model(batch)
    assert reps.h_states.shape == (batch.n_states, HIDDEN)
    assert reps.psi.shape == (batch.n_states, tiny_cfg.heads.latent_dim)
    assert reps.phi.shape == (batch.n_rows, tiny_cfg.heads.latent_dim)
    assert reps.act_emb.shape == (batch.n_rows, HIDDEN)


def test_action_embedding_is_tied_to_the_input_table_and_differentiable(model, batch):
    """§6.4: embedding once and passing `inputs_embeds` gives the action embedding for free,
    tied to the input table, with no second lookup."""
    reps = model(batch)
    reps.act_emb.sum().backward()
    assert model.backbone.emb.weight.grad is not None
    assert float(model.backbone.emb.weight.grad.abs().sum()) > 0


def test_phi_reads_the_PREVIOUS_state_hidden(model, batch):
    """phi_i = phi(h_{i-1}, act_emb_i). If it read h_i it would BE psi(s_i) and L_T would be
    trivially satisfiable (§6.4)."""
    reps = model(batch)
    expected = model.phi(
        reps.h_states.index_select(0, batch.row_src),
        model.action_pool(
            model.backbone.get_input_embeddings()(batch.input_ids),
            batch.span_token_idx,
            batch.span_row_idx,
            batch.span_counts,
            batch.n_rows,
        ),
    )
    assert torch.allclose(reps.phi, expected, atol=1e-6)


def test_end_to_end_step_is_finite_and_backward(model, batch, tiny_cfg):
    """collate -> goals -> forward -> matrices -> loss -> probes -> backward, which is every
    line of train.py's inner loop except the real backbone."""
    goals = sample_goals(batch, tiny_cfg.discount, np.random.default_rng(0))
    reps = model(batch)
    matrices = build_matrices(reps.psi, reps.phi, batch, goals, model.distance, tiny_cfg)
    out = phase1_loss(
        reps.psi, reps.phi, batch, matrices, model.distance, tiny_cfg, goal_traj=goals.goal_traj
    )
    assert torch.isfinite(out.total)
    probes = batch_probes(reps.psi, reps.phi, batch, goals, matrices, model.distance, tiny_cfg)
    assert probes["probe01/questions_in_batch"] == 2.0
    assert probes["probe01/sequences_in_batch"] == 4.0
    assert out.info["step/distinct_z"] == 2.0        # two incorrect trajectories

    out.total.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name
    assert model.psi.net.out.weight.grad is not None
    assert model.phi.net.out.weight.grad is not None


def test_bf16_backbone_with_fp32_heads(tiny_cfg, batch):
    """`train.bf16: true` loads the backbone in bf16 while the heads stay fp32 (§6.3), so the
    hidden states and the pooled action embedding must be cast at the seam. Without the cast
    the FIRST real forward dies with "expected mat1 and mat2 to have the same dtype" -- this
    test exists because the dev box has no GPU to find that on."""
    torch.manual_seed(0)
    m = FeynmanPRM(tiny_cfg, HIDDEN, backbone=StubBackbone().to(torch.bfloat16))
    m.pad_id = 0
    assert m.head_dtype == torch.float32

    reps = m(batch)
    assert reps.psi.dtype == torch.float32
    assert reps.phi.dtype == torch.float32
    assert m.distance(reps.psi[:2], reps.psi[2:4]).dtype == torch.float32

    reps.psi.sum().backward()
    assert m.backbone.emb.weight.grad is not None, "gradient still reaches the bf16 backbone"


def test_no_goal_head_in_phase_1(model):
    """§7.7: the head does not exist until phase 2, so `.detach()` is structural."""
    assert model.goal_head is None
    assert not any("goal_head" in n for n, _ in model.named_parameters())


def test_freeze_for_phase2_leaves_only_the_goal_head(tiny_cfg):
    from feynman_prm.model.backbone import assert_phase2_trainable

    m = FeynmanPRM(tiny_cfg, HIDDEN, backbone=StubBackbone(), with_goal_head=True)
    m.freeze_for_phase2()
    counts = assert_phase2_trainable(m)
    assert counts["goal_head"] > 0
    assert counts["psi"] == 0 and counts["phi"] == 0 and counts["lora"] == 0
