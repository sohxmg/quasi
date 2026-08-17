"""(4) L_CF -- counterfactual invariance (§7.5). BUILT NOW, lambda_cf = 0 (locked #4).

**MULTI-POSITIVE since 2026-08-08, at the human's request. This is a DEVIATION from the
spec image, which writes a single positive `s_2`** (§7.5, §7.10's discipline). One example
now carries an EQUIVALENCE CLASS `C = {anchor} u positives` -- several meaning-preserving
wordings of one step -- and the loss is SupCon's `L_out` over that class:

    for each query q in C:

                       1                        exp( -d(q,p) / tau )
        L_q  =  - ----------- .  sum      log  -----------------------------------------
                   |C \\ {q}|      p in C\\{q}     sum       exp(-d(q,p')/tau)
                                                p' in C\\{q}
                                              + sum       exp(-d(q,n)/tau)
                                                n in N

        L    =  mean over q in C,  then mean over examples

Two properties are REQUIREMENTS, not incidentals:

* **Positives stay in the DENOMINATOR.** That is what makes `C` one equivalence class rather
  than `|P|` independent one-positive problems. Take them out and each positive is free to
  sit at its own distance from the anchor.
* **NEGATIVES ARE NEVER QUERIES.** There is no negative-negative term anywhere in the sum, so
  nothing pulls or pushes the negatives among themselves -- two different WRONG actions have
  no reason to coincide, and asserting they do would be a claim this data cannot support.
  An off-the-shelf SupCon iterates its queries over the whole batch and gets this wrong;
  `tests/test_counterfactual.py` pins both readings against each other.

**There are TWO callers since 2026-08-08 and this loss is agnostic to both.** It takes a flat
`(V, D)` tensor, a group id per row and a kind per row; it does not know or care what a
"variant" is. (4) `L_CF` groups by CF example and means "reworded steps of one action";
(7) `L_term` (`losses/terminal_class.py`, §7.13/§16.26) groups by QUESTION and means "correct
endings of one question". Neither is a mode of the other -- there is no `mode=` flag here and
there must not be one.

`d` is asymmetric (MRN, §6.5), so iterating the query over `C` trains BOTH orderings of every
class pair -- which is correct and not redundant: two surface forms of one action must be at
zero distance in both directions to be the same point in a quasimetric. `d(from, to)` keeps
its §6.5 order throughout: the query is the source.

**What was here before, and why it was a trap.** The previous implementation sorted the
candidates by `cand_example * 1000 + variant_kind` and targeted class 0. Fed three positives
it kept the first and trained the other two AS NEGATIVES -- actively pushing meaning-preserving
rewrites of one step apart -- while `cf/positive_distance`, computed on slot 0 alone, still
fell. Nothing crashed and no curve showed it.

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
from torch import Tensor

from ..model.distances import Distance

_NEG_INF = float("-inf")


def counterfactual_loss(
    phi_variants: Tensor,
    variant_example: Tensor,
    variant_kind: Tensor,
    distance: Distance,
    temperature: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Supervised-contrastive loss over each example's equivalence class.

    `phi_variants` is (V, D): one phi per (example, variant), all sharing their example's
    `h_{i-1}` and differing only in `act_emb` -- which is why a variant costs an embedding
    lookup plus an MLP and not an LM forward (PLAN finding 1).

    `variant_kind`: 0 = the original step (anchor), 1 = positive, 2 = negative. The anchor and
    the positives are ONE class; there may be any number of positives (>=1 in practice, see
    `data/counterfactual.py`) and any number of negatives.
    """
    device = phi_variants.device
    n_examples = int(variant_example.max()) + 1 if variant_example.numel() else 0
    if n_examples == 0:
        zero = phi_variants.sum() * 0.0
        return zero, _empty_info()

    variant_example = variant_example.long()
    V = int(variant_example.numel())

    # --- Slot assignment, vectorised. This was a Python loop over V; V is ~10 per example
    # now (3 positives + 6 negatives + the anchor), so the loop was the wrong shape.
    order = torch.argsort(variant_example, stable=True)
    counts = torch.bincount(variant_example, minlength=n_examples)
    starts = torch.cumsum(counts, dim=0) - counts
    slot = torch.empty(V, dtype=torch.long, device=device)
    slot[order] = torch.arange(V, device=device) - starts[variant_example[order]]
    width = int(counts.max())

    # (n_examples, width) padded views. `idx_pad` is only read where `valid`.
    idx_pad = torch.zeros(n_examples, width, dtype=torch.long, device=device)
    idx_pad[variant_example, slot] = torch.arange(V, device=device)
    valid = torch.zeros(n_examples, width, dtype=torch.bool, device=device)
    valid[variant_example, slot] = True
    is_class = torch.zeros(n_examples, width, dtype=torch.bool, device=device)
    is_class[variant_example, slot] = variant_kind != 2

    # --- Queries are the CLASS members only. Negatives never appear on the left of `d`.
    q_example, q_slot = torch.nonzero(is_class, as_tuple=True)
    if q_example.numel() == 0:
        return phi_variants.sum() * 0.0, _empty_info(n_examples)

    q_idx = idx_pad[q_example, q_slot]                       # (Q,) query variant index
    cand_idx = idx_pad[q_example]                            # (Q, width) candidates
    # d(query, candidate): query is the SOURCE, per §6.5's fixed argument order.
    d = distance(phi_variants[q_idx].unsqueeze(1), phi_variants[cand_idx])   # (Q, width)

    self_mask = torch.arange(width, device=device).unsqueeze(0) == q_slot.unsqueeze(1)
    denom_mask = valid[q_example] & ~self_mask               # C\{q} u N -- everything but q
    pos_mask = is_class[q_example] & ~self_mask              # C\{q}
    neg_mask = denom_mask & ~is_class[q_example]             # N

    logits = -d / temperature
    log_denom = torch.logsumexp(logits.masked_fill(~denom_mask, _NEG_INF), dim=1)
    n_pos = pos_mask.sum(dim=1)

    # -(1/|P|) sum_p [logit_p - log_denom]  ==  log_denom - mean_p logit_p.
    mean_pos_logit = (logits * pos_mask).sum(dim=1) / n_pos.clamp(min=1)
    per_query = log_denom - mean_pos_logit

    # A class of size 1 (an anchor with no positive) has nothing to pull together; it is
    # dropped rather than crashed. `__post_init__` forbids it on disk, but a caller can build
    # a batch by hand.
    keep = n_pos > 0
    per_query = per_query[keep]
    kept_example = q_example[keep]
    if per_query.numel() == 0:
        return phi_variants.sum() * 0.0, _empty_info(n_examples, variant_kind)

    zeros = torch.zeros(n_examples, dtype=per_query.dtype, device=device)
    ex_sum = zeros.index_add(0, kept_example, per_query)
    ex_count = zeros.index_add(0, kept_example, torch.ones_like(per_query))
    has_query = ex_count > 0
    loss = (ex_sum[has_query] / ex_count[has_query]).mean()

    with torch.no_grad():
        d_det = d.detach()
        # The value `cf/loss` takes when `d` carries no preference at all -- the level to read
        # `cf/loss` AGAINST, and it is NOT a constant. A query scores |C\{q}| positives and |N|
        # negatives in ONE softmax, so a flat softmax gives L_q = log(P + N); every query of an
        # example gets the same value, and the loss means over examples exactly as below.
        # (4) was the only contrastive term in the set without this key, and that omission is
        # what let cf/loss ~= 2.157 read as "dead" for a whole run when it was correct-at-chance
        # for a 9-candidate denominator (log 9 = 2.197). §7.5.6: a statistic is only readable
        # against its own chance level.
        chance_q = torch.log(
            denom_mask.sum(dim=1).clamp(min=1).to(per_query.dtype)
        )[keep]
        chance_sum = zeros.index_add(0, kept_example, chance_q)
        info = {
            "cf/loss": float(loss),
            "cf/chance": float((chance_sum[has_query] / ex_count[has_query]).mean()),
            "cf/examples": float(int(has_query.sum())),
            # Over ALL class pairs, both orderings -- not slot 0 against the anchor.
            "cf/positive_distance": _masked_mean(d_det, pos_mask),
            # Over ALL class -> negative pairs.
            "cf/negative_distance": _masked_mean(d_det, neg_mask),
            "cf/positives_per_example": float(int((variant_kind == 1).sum())) / n_examples,
            "cf/negatives_per_example": float(int((variant_kind == 2).sum())) / n_examples,
        }
    return loss, info


def _masked_mean(values: Tensor, mask: Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(values[mask].mean())


def _empty_info(n_examples: int = 0, variant_kind: Tensor | None = None) -> dict[str, float]:
    """The empty path logs the SAME key set as the populated one -- a diagnostic that
    disappears when a batch is degenerate is a diagnostic that cannot be plotted."""
    per_example = lambda kind: (  # noqa: E731
        float(int((variant_kind == kind).sum())) / n_examples
        if variant_kind is not None and n_examples > 0
        else 0.0
    )
    return {
        "cf/loss": 0.0,
        "cf/chance": 0.0,
        "cf/examples": 0.0,
        "cf/positive_distance": 0.0,
        "cf/negative_distance": 0.0,
        "cf/positives_per_example": per_example(1),
        "cf/negatives_per_example": per_example(2),
    }
