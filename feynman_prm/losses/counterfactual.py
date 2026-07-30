"""(4) L_CF -- counterfactual invariance (§7.5). BUILT NOW, lambda_cf = 0 (locked #4).

              /        e^{ f( phi(s_1), phi(s_2) ) }        \\
       log    |  -----------------------------------------  |   -> 0        f -> -d
              \\   sum_i  e^{ f( phi(s_i), phi(n_i) ) }      /

`s_2` is the meaning-PRESERVING rewrite (the positive) and `n_i` the meaning-CHANGING ones.
Implemented as a cross-entropy over `{s_2} u {n_i}` with the positive at index 0.

Why it matters more than it looks (§4.4): this dataset has almost no natural branching --
~95% of states have exactly one observed action -- so L_CF is the mechanism that
MANUFACTURES the branching structure the quasimetric losses need. Nothing in TMD corresponds
to it; it is spec-only.

Only the DATA is deferred. §8.3's ~30k mined branch points are NOT a substitute (PLAN
finding 2): L_CF needs a meaning-preserving positive, and a branch point is two
meaning-changing continuations with a correctness label.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..model.distances import Distance


def counterfactual_loss(
    phi_variants: Tensor,
    variant_example: Tensor,
    variant_kind: Tensor,
    distance: Distance,
    temperature: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Cross-entropy over each example's candidate set.

    `phi_variants` is (V, D): one phi per (example, variant), all sharing their example's
    `h_{i-1}` and differing only in `act_emb` -- which is why a variant costs an embedding
    lookup plus an MLP and not an LM forward (PLAN finding 1).

    `variant_kind`: 0 = the original step (anchor), 1 = positive, 2 = negative.
    """
    n_examples = int(variant_example.max()) + 1 if variant_example.numel() else 0
    if n_examples == 0:
        zero = phi_variants.sum() * 0.0
        return zero, {"cf/loss": 0.0, "cf/examples": 0.0}

    anchors = torch.zeros(n_examples, dtype=torch.long, device=phi_variants.device)
    anchors[variant_example[variant_kind == 0]] = torch.nonzero(
        variant_kind == 0, as_tuple=False
    ).flatten()

    candidates = torch.nonzero(variant_kind != 0, as_tuple=False).flatten()
    cand_example = variant_example[candidates]
    d = distance(phi_variants[anchors[cand_example]], phi_variants[candidates])  # (n_cand,)

    # Scatter into a padded (n_examples, max_candidates) logit matrix with the positive
    # first, so the target is always class 0.
    order = torch.argsort(cand_example * 1000 + variant_kind[candidates], stable=True)
    cand_example = cand_example[order]
    d = d[order]
    slot = torch.zeros_like(cand_example)
    counts: dict[int, int] = {}
    for i, ex in enumerate(cand_example.tolist()):
        slot[i] = counts.get(ex, 0)
        counts[ex] = slot[i] + 1
    width = int(slot.max()) + 1

    logits = torch.full(
        (n_examples, width), float("-inf"), dtype=d.dtype, device=d.device
    )
    logits[cand_example, slot] = -d / temperature
    loss = F.cross_entropy(logits, torch.zeros(n_examples, dtype=torch.long, device=d.device))

    with torch.no_grad():
        info = {
            "cf/loss": float(loss),
            "cf/examples": float(n_examples),
            "cf/positive_distance": float(d[slot == 0].mean()),
            "cf/negative_distance": float(d[slot > 0].mean()) if (slot > 0).any() else 0.0,
        }
    return loss, info
