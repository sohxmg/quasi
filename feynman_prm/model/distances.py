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

IQE is the expensive one and needs watching: it sorts 2k values per group, so its working set
is (R, C, D/k, 2k) and there are several of them. Its mask is built under no_grad at int16 and
the sort permutation is applied to the MASK rather than to the data, which keeps the graph at
13.1 kB/pair instead of 49.9 (full_mrn is 6.2). Measure with `saved_tensors_hooks`, not by
eye, before changing anything in `iqe_distance` -- and keep the change bit-exact.
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

    D = xs.shape[-1]
    xy = torch.cat([xs, ys], dim=-1)

    # Everything from here to `w` is the interval-counting mask: argsort / `<` / cumsum /
    # a difference of two {0, -1} tensors. NOT ONE STEP OF IT HAS A GRADIENT, so it runs
    # under no_grad and at int16 -- the transcription from tmd.py:56-63 left it in fp32 and
    # int64 inside grad mode, which pinned nine (..., D/k, 2k) tensors for the backward and
    # is what made IQE ~8x the memory of full_mrn per pair (49.9 kB/pair of graph against
    # full_mrn's 6.2 kB). Values here are bounded by 2k, so int16 cannot overflow.
    with torch.no_grad():
        ixy = xy.argsort(dim=-1)
        # tmd.py:60 indexes D validity flags from 2D sorted positions with `ixy % D`. Doing
        # that literally allocates a SECOND int64 (..., 2k) tensor -- 24.6 kB/pair, live at
        # the same moment as `ixy`, which is what set the peak. Tiling the flags to 2k
        # instead costs 3 kB/pair of bool and indexes identically:
        # cat([v, v])[i] == v[i % k] for every i in [0, 2k).
        valid = xs < ys
        inc = torch.gather(torch.cat([valid, valid], dim=-1), -1, ixy).to(torch.int16)
        del valid
        inc = torch.where(ixy < D, -inc, inc)          # tmd.py:59's `sign`, folded in
        inc = inc.cumsum_(-1)
        negf = inc.less_(0).to(torch.int16).neg_()     # {0, -1}
        del inc
        w_sorted = torch.empty_like(negf)
        w_sorted[..., :1] = negf[..., :1]
        torch.sub(negf[..., 1:], negf[..., :-1], out=w_sorted[..., 1:])
        del negf
        # tmd.py:63 gathers xy into sorted order and multiplies. `ixy` is a PERMUTATION of
        # 0..2D-1, so sum_j xy[ixy[j]] * w_sorted[j] == sum_i xy[i] * w_sorted[ixy^-1[i]],
        # and scatter_ along ixy is exactly that inverse. Same sum, but the permutation now
        # happens on the mask instead of on the data -- so no differentiable `gather` saves
        # the int64 `ixy` (24.6 kB/pair, the largest single tensor here) for the backward.
        w = torch.zeros_like(ixy, dtype=torch.int16).scatter_(-1, ixy, w_sorted)
        del ixy, w_sorted
        w = w.to(xy.dtype)

    comps = (xy * w).sum(dim=-1)   # the only differentiable op in the function
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
