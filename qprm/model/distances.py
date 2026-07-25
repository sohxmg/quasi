"""The quasimetric distance. Ported from `tmd-release/impls/agents/tmd.py:28-74`.

    mrn_distance (tmd.py:28-46)
        split the latent into K components
        per component, split in half:
            asymmetric half:  max( relu(x - y) )      <- can tell direction apart
            symmetric  half:  sqrt( sum (x-y)^2 + eps )   eps = 1e-6, INSIDE the sqrt
            component distance = asym + sym
        d(x, y) = mean over the K components

`d(from, to)` -- state first, goal second, everywhere. No exceptions.

Two things that are not negotiable:

* **fp32.** Old bug B10a: in bf16 the small logit *differences* round to equal, the softmax
  goes exactly uniform and the gradient is zero. Distances are cast to fp32 here, inside the
  distance, whatever dtype the heads run in. Gradients still flow back to bf16 params.
* **`eps = 1e-6` sits inside the sqrt** (`tmd.py:38`), so each component floors at
  `sqrt(1e-6) = 1e-3` and `d(x, x) ~= 1e-3` for full_mrn -- not 0. §15 asserts `< 2e-3`.
  For `asym_only` there is no sqrt and `d(x, x)` is exactly 0.

Memory: the halves are sliced BEFORE subtracting, so an (R, C, 512) difference is never
materialised -- at R=348, C=172 the largest live intermediate is (R, C, K, 32) fp32 ~= 61 MB.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

EPS = 1e-6  # tmd.py:38 -- inside the sqrt


def _split_components(x: Tensor, components: int) -> Tensor:
    """(..., D) -> (..., K, D/K), contiguous chunks, matching `jnp.split(x, K, axis=-1)`."""
    D = x.shape[-1]
    if D % components != 0:
        raise ValueError(f"latent dim {D} is not divisible by components {components}")
    return x.reshape(*x.shape[:-1], components, D // components)


def mrn_distance(
    x: Tensor, y: Tensor, components: int = 8, return_parts: bool = False
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """MRN distance, fp32. Returns d, or (d, asym_part, sym_part) if `return_parts`."""
    x = x.float()
    y = y.float()
    xs = _split_components(x, components)
    ys = _split_components(y, components)
    half = xs.shape[-1] // 2

    # max(relu(x - y)) over the asymmetric half. TMD masks the other half to zero and takes
    # the max over the whole component (tmd.py:37); relu is non-negative so slicing is the
    # same value, and it halves the intermediate.
    asym = torch.relu(xs[..., :half] - ys[..., :half]).amax(dim=-1)
    sym = torch.sqrt(torch.square(xs[..., half:] - ys[..., half:]).sum(dim=-1) + EPS)

    dist = (asym + sym).mean(dim=-1)
    if return_parts:
        return dist, asym.mean(dim=-1), sym.mean(dim=-1)
    return dist


def asym_only_distance(
    x: Tensor, y: Tensor, components: int = 8, return_parts: bool = False
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """`full_mrn` with the symmetric half dropped: mean_K max(relu(x - y)) over the WHOLE
    component (not just its first half, so the full latent is used).

    Exactly 0 at x == y, and the only variant where the reported distance is 100%
    asymmetric -- which is what §9.4's goal-free irreversibility score would want if it is
    ever promoted from a diagnostic to a result (§16.10).
    """
    x = x.float()
    y = y.float()
    xs = _split_components(x, components)
    ys = _split_components(y, components)
    asym = torch.relu(xs - ys).amax(dim=-1)
    dist = asym.mean(dim=-1)
    if return_parts:
        return dist, dist, torch.zeros_like(dist)
    return dist


def iqe_distance(x: Tensor, y: Tensor, components: int, alpha: Tensor) -> Tensor:
    """Interval Quasimetric Embedding, ported from `tmd.py:48-66`.

    Note TMD's reshape is `(D // k, k)` -- the *inner* dim is `components` and the max/mean
    at the end runs over `D // k` groups. That is a transposed reading of "components"
    relative to `mrn_distance`, and it is reproduced here on purpose: the repo wins.
    """
    x = x.float()
    y = y.float()
    D_total = x.shape[-1]
    xs = x.reshape(*x.shape[:-1], D_total // components, components)
    ys = y.reshape(*y.shape[:-1], D_total // components, components)
    xs, ys = torch.broadcast_tensors(xs, ys)

    valid = xs < ys
    D = xs.shape[-1]
    xy = torch.cat([xs, ys], dim=-1)
    ixy = xy.argsort(dim=-1)
    sxy = torch.gather(xy, -1, ixy)
    sign = torch.where(ixy < D, -1.0, 1.0)
    # gather's index may be LONGER than the source along the gathered dim, which is what
    # `ixy % D` needs: 2D sorted positions indexing back into D validity flags
    # (jnp.take_along_axis does the same at tmd.py:60).
    neg_inc_copies = torch.gather(valid, -1, ixy % D).float() * sign
    neg_inp_copies = torch.cumsum(neg_inc_copies, dim=-1)
    neg_f = (neg_inp_copies < 0).float() * -1.0
    neg_incf = torch.cat([neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], dim=-1)
    comps = (sxy * neg_incf).sum(dim=-1)
    return alpha * comps.mean(dim=-1) + (1 - alpha) * comps.amax(dim=-1)


class Distance(nn.Module):
    """Dispatch over `distance.variant` (`tmd.py:68-74`), plus the sym/asym decomposition
    that diagnostic #4 logs every run.

    §6.3/old R6: at latent_dim 1536 the MRN distance measured ~80% symmetric, at 512 ~73%.
    If the asymmetric term is a minority, do not claim asymmetry drives the result.
    """

    def __init__(self, variant: str = "full_mrn", components: int = 8):
        super().__init__()
        self.variant = variant
        self.components = components
        if variant == "iqe":
            # IQE's alpha is a learned scalar (tmd.py:51-52). It is the only distance
            # parameter in the repo, and the trainability assert accounts for it explicitly.
            self.alpha_raw = nn.Parameter(torch.zeros(()))

    @property
    def alpha(self) -> Tensor:
        return torch.sigmoid(self.alpha_raw)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if self.variant == "full_mrn":
            return mrn_distance(x, y, self.components)
        if self.variant == "asym_only":
            return asym_only_distance(x, y, self.components)
        if self.variant == "iqe":
            return iqe_distance(x, y, self.components, self.alpha)
        raise ValueError(f"unknown distance variant {self.variant!r}")

    def parts(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(d, asymmetric part, symmetric part). IQE has no such split; it reports all
        weight on the asymmetric side because IQE is asymmetric by construction."""
        if self.variant == "full_mrn":
            return mrn_distance(x, y, self.components, return_parts=True)
        if self.variant == "asym_only":
            return asym_only_distance(x, y, self.components, return_parts=True)
        d = self.forward(x, y)
        return d, d, torch.zeros_like(d)

    def symmetric_share(self, x: Tensor, y: Tensor) -> float:
        """Diagnostic #4."""
        _, asym, sym = self.parts(x, y)
        total = asym.mean() + sym.mean()
        return float(sym.mean() / total) if float(total) != 0.0 else 0.0
