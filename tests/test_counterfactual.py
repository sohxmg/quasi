"""(4) L_CF tests (§7.5). The loss and the on-disk format are built now; lambda_cf = 0 and
the DATA is deferred (locked #4)."""

from __future__ import annotations

import math

import pytest
import torch

from feynman_prm.data.counterfactual import (
    CounterfactualExample,
    build_cf_batch,
    read_jsonl,
    write_jsonl,
)
from feynman_prm.losses.counterfactual import counterfactual_loss
from feynman_prm.model.distances import Distance
from conftest import SEP_ID


def _example(**kw):
    base = dict(
        question="what is x?",
        steps=("Step 1: x + 2 = 4", "Step 2: x = 2"),
        step_index=0,
        positive_rewrite="Step 1: y + 2 = 4",
        negative_rewrites=("Step 1: x - 2 = 4",),
    )
    base.update(kw)
    return CounterfactualExample(**base)


def test_jsonl_round_trip(tmp_path):
    examples = [_example(), _example(step_index=1)]
    path = tmp_path / "cf.jsonl"
    write_jsonl(examples, path)
    assert read_jsonl(path) == examples


def test_schema_is_validated():
    with pytest.raises(ValueError, match="step_index"):
        _example(step_index=5)
    with pytest.raises(ValueError, match="at least one negative"):
        _example(negative_rewrites=())


def test_positive_is_class_zero_and_the_loss_prefers_it():
    dist = Distance("full_mrn", 8)
    anchor = torch.zeros(1, 512)
    positive = torch.zeros(1, 512)              # meaning-preserving: identical
    negative = torch.full((1, 512), 5.0)        # meaning-changing: far
    phi = torch.cat([anchor, positive, negative])
    variant_example = torch.tensor([0, 0, 0])
    variant_kind = torch.tensor([0, 1, 2])

    loss, info = counterfactual_loss(phi, variant_example, variant_kind, dist)
    assert float(loss) < 0.01
    assert info["cf/positive_distance"] < info["cf/negative_distance"]

    swapped = torch.cat([anchor, negative, positive])
    worse, _ = counterfactual_loss(swapped, variant_example, variant_kind, dist)
    assert float(worse) > float(loss)


def test_uniform_at_chance():
    dist = Distance("full_mrn", 8)
    phi = torch.zeros(4, 512)                    # anchor + positive + 2 negatives, all equal
    loss, _ = counterfactual_loss(
        phi, torch.tensor([0, 0, 0, 0]), torch.tensor([0, 1, 2, 2]), dist
    )
    assert math.isclose(float(loss), math.log(3), rel_tol=1e-4)


def test_ragged_negative_counts_are_padded_with_minus_inf():
    dist = Distance("full_mrn", 8)
    torch.manual_seed(0)
    phi = torch.randn(7, 512)
    variant_example = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    variant_kind = torch.tensor([0, 1, 2, 0, 1, 2, 2])   # 1 negative vs 2 negatives
    loss, info = counterfactual_loss(phi, variant_example, variant_kind, dist)
    assert torch.isfinite(loss)
    assert info["cf/examples"] == 2.0


def test_variants_share_one_prefix_forward(tokenizer):
    """PLAN finding 1, contra §7.5: rewriting step i leaves the prefix -- hence h_{i-1} --
    untouched, so a variant costs an embedding lookup plus an MLP, not an LM forward. The
    batch therefore carries ONE prefix sequence per example and N variant token lists."""
    examples = [_example(), _example(step_index=1)]
    batch = build_cf_batch(tokenizer, examples, sep_id=SEP_ID, pad_id=0)
    assert batch.n_examples == 2
    assert batch.n_variants == 6                       # (anchor + positive + 1 negative) x 2
    flat = batch.input_ids.reshape(-1)
    assert torch.all(flat[batch.anchor_flat_idx] == SEP_ID), "anchors sit on s_{i-1}"
    assert batch.variant_kind.tolist() == [0, 1, 2, 0, 1, 2]


def test_lambda_cf_is_zero_by_default(cfg):
    """Only the DATA is deferred; the loss is wired and inert at weight 0 (§2 #4)."""
    assert cfg.losses.lambda_cf == 0.0
