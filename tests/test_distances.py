"""§15's distance tests (§6.5, `tmd.py:28-66`)."""

from __future__ import annotations

import pytest
import torch

from feynman_prm.model.distances import Distance, asym_only_distance, iqe_distance, mrn_distance


def test_d_x_x_is_the_eps_floor_not_zero():
    """`eps = 1e-6` sits INSIDE the sqrt (tmd.py:38), so each component floors at
    sqrt(1e-6) = 1e-3 and the mean over K = 8 is also 1e-3. Assert < 2e-3, not the old
    loose 5e-2 -- a looser bound would hide a real collapse."""
    x = torch.randn(32, 512)
    d = mrn_distance(x, x)
    assert torch.all(d < 2e-3)
    assert torch.all(d > 5e-4), "should sit AT the eps floor, not below it"


def test_asym_only_is_exactly_zero_at_x_equals_x():
    x = torch.randn(32, 512)
    assert torch.all(asym_only_distance(x, x) == 0.0)


def test_triangle_inequality_over_200_triples():
    torch.manual_seed(0)
    a, b, c = (torch.randn(200, 512) for _ in range(3))
    d_ac = mrn_distance(a, c)
    d_ab = mrn_distance(a, b)
    d_bc = mrn_distance(b, c)
    assert torch.all(d_ac <= d_ab + d_bc + 1e-4)


def test_asymmetry_is_possible():
    """d(s,g) != d(g,s) is what buys irreversibility detection (§1, §9.4)."""
    torch.manual_seed(0)
    x, y = torch.randn(64, 512), torch.randn(64, 512)
    assert not torch.allclose(mrn_distance(x, y), mrn_distance(y, x))


def test_broadcast_shapes_and_decomposition():
    x, y = torch.randn(7, 512), torch.randn(5, 512)
    d = mrn_distance(x[:, None, :], y[None, :, :])
    assert d.shape == (7, 5)

    d2, asym, sym = mrn_distance(x, x.roll(1, 0), return_parts=True)
    assert torch.allclose(d2, asym + sym, atol=1e-5), "the two parts sum to the total"


def test_fp32_even_when_inputs_are_bf16():
    """Bug B10a: in bf16 the small logit DIFFERENCES round to equal, the softmax goes exactly
    uniform and the gradient is zero. The cast lives inside the distance."""
    x = torch.randn(16, 512, dtype=torch.bfloat16)
    y = torch.randn(16, 512, dtype=torch.bfloat16)
    assert mrn_distance(x, y).dtype == torch.float32


def test_gradients_flow_back_to_bf16_parameters():
    x = torch.randn(8, 512, dtype=torch.bfloat16, requires_grad=True)
    y = torch.randn(8, 512, dtype=torch.bfloat16)
    mrn_distance(x, y).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad.float()).all()


@pytest.mark.parametrize("variant", ["full_mrn", "asym_only", "iqe"])
def test_all_variants_run(variant):
    dist = Distance(variant, components=8)
    x, y = torch.randn(6, 512), torch.randn(4, 512)
    out = dist(x[:, None, :], y[None, :, :])
    assert out.shape == (6, 4)
    assert torch.isfinite(out).all()


def test_iqe_alpha_is_learnable():
    dist = Distance("iqe", components=8)
    assert dist.alpha_raw.requires_grad
    x, y = torch.randn(4, 512), torch.randn(4, 512)
    dist(x, y).sum().backward()
    assert dist.alpha_raw.grad is not None


def test_symmetric_share_is_reported():
    """Diagnostic #4 / old R6: at latent 1536 the distance measured ~80% symmetric, at 512
    ~73%. If the asymmetric term is a minority, do not claim asymmetry drives the result."""
    dist = Distance("full_mrn", components=8)
    x, y = torch.randn(64, 512), torch.randn(64, 512)
    share = dist.symmetric_share(x, y)
    assert 0.0 < share < 1.0
    assert Distance("asym_only", 8).symmetric_share(x, y) == 0.0


def test_iqe_matches_the_tmd_reshape_convention():
    """tmd.py:53 reshapes to (D // k, k) -- the INNER dim is `components` and the max/mean
    runs over D//k groups. Reproduced on purpose; the repo wins (§7.11)."""
    x, y = torch.randn(3, 512), torch.randn(3, 512)
    alpha = torch.tensor(0.5)
    out = iqe_distance(x, y, components=8, alpha=alpha)
    assert out.shape == (3,)
    assert torch.all(iqe_distance(x, x, 8, alpha) <= 1e-5)
