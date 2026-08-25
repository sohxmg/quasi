"""The QRL objective: one push term and two Lagrangian constraints.

    L  =  push  +  cf_neg_push_weight * neg_push
          +  lambda_local * ( E[relu(d(psi_z, psi_{z+1}) - step_cost)^2]  - epsilon_local^2 )
          +  lambda_cf    * ( E[(relu(d(anchor, v))^2 + relu(d(v, anchor))^2)/2] - epsilon_cf^2 )

**There is no latent-dynamics term and there is no `phi`.** QRL needs both because its
environment is a simulator whose next state is not available at training time: `phi(s, a)` is
a LEARNED prediction of where action `a` lands, and `L_dyn` is the loss that drags that
prediction onto the observed `psi(s')`. Here the environment is a text prefix and an action is
a step appended to it, so

    s' = s ++ a

is deterministic and already tokenised. `psi(s')` is READ, not predicted: for an observed
transition it is `psi[row_dst]`, already in the batch from the same forward pass; for a
counterfactual rewrite it is `psi(prompt + steps[:i] + variant + SEP)`, which is what
`cf_encode.py` builds. Q is `d(s', g)` throughout -- the same quantity
`eval/processbench.py:6` scores with -- so nothing in this objective, and nothing at eval,
ever asks a head to guess a state.

That deletes three things at once: the `dyn` term, the `phi` head (frozen and untrained under
`qrl_prm/`, see `train.py`), and the whole question of whether phi-space and psi-space have
drifted apart, which is what `qrl/dyn_sq_dists` used to be read for.

**Not one Feynman-PRM phase-1 term is computed here** -- no L_NCE, no L_I, no L_T, no L_CF,
no L_step, no L_good, no L_term. That is the experiment: replace a fixed-weight soft ruler
whose known failure is that it DECAYS (IMPLEMENTATION.md §9: `backup/delta_mean` drifts off
`-log gamma`) with a constrained program whose ruler is held by a dual variable.

---

## 1. The push term (`losses/global_push.py:44-48`)

QRL:

    dists = qm(zx, torch.roll(zy, 1, dims=0))
    F.softplus(self.softplus_offset - dists, beta=self.softplus_beta).mean()

Same transform, same `.mean()`. **Only the pair SAMPLER is adapted, and it is a documented
divergence**: QRL rolls its batch to get `B` random pairs (~50); we take the full `S x C`
grid of batch states against the sampled goal columns, ~404 x 172 ~ 69k pairs per
micro-batch. Both are `E[softplus(offset - d(x, y))]` over the batch marginals -- the
estimator is the same, ours has ~1400x the pairs, and the goal side is drawn by
`sample_goals` (geometric, discount 0.5) so it matches BOTH the distribution the metric is
queried with at eval and the goal distribution every baseline run trained under.

Read `qrl/push_saturated_frac` against `softplus_offset`: it is the fraction of pairs already
past the offset, where the transform is flat and the term has no gradient left.

## 2. The local constraint (`losses/local_constraint.py:48-63`)

`E[relu(d(psi_z, psi_{z+1}) - step_cost)^2] <= epsilon_local^2` over every observed
transition -- correct AND incorrect trajectories, because a transition is an observed step
whatever the step's verdict, and the metric has to be able to measure both.

**The order is mean-then-target**, exactly as QRL writes it: square the per-transition
deviation, mean over transitions, THEN subtract `epsilon^2`. `mean(relu(d-c)^2) - eps^2` and
`mean(relu(d-c)^2 - eps^2)` agree here only because the target is constant; keeping QRL's
order means the violation is one scalar with a readable sign rather than a per-pair quantity.

`qrl/local_dist_mean` is THE ruler. It should pin to ~`step_cost`.

## 3. The CF constraint -- a star centred on the ANCHOR's own arrived state

A CF example rewrites step `i`, and every variant of it is encoded as a full sequence
(`cf_encode.py`). So the class is `{psi(prefix + anchor), psi(prefix + pos_1), ...}`, all of
them real points of `psi`-space, and the hub is the ANCHOR:

        psi(prefix + anchor)         pairs, both directions:
             /        \\                (anchor, pos_k), (pos_k, anchor)
          pos_1      pos_2

`epsilon_cf` bounds each member's distance to the anchor in both directions, so by the
triangle inequality the class DIAMETER is bounded by `2 * epsilon_cf` without ever forming
the O(|C|^2) pairwise grid.

**The hub used to be `psi[row_dst]` -- the arrived state of the HOST trajectory -- and that
was wrong on the examples the prefix join exists to catch.** `cf_attach.py` keys on
`(question, steps[:i])`, deliberately, so a CF example can ride any SIBLING trajectory
agreeing on the first `i` steps. A sibling agrees on the prefix and is free to differ at step
`i` itself -- that is the 6.3 points of attach rate the prefix key buys over a trajectory key
-- and on exactly those attachments `psi[row_dst]` is the arrived state of a DIFFERENT step
than the one the class was written about. Under `phi` the anchor was a prediction and the
mismatch was invisible; encoding the anchor makes the right hub available directly, and
`hub_of_state` is gone with the bug.

It also removes a whole drop path: a variant departing from a trajectory's terminal state
used to have no successor and no hub (`qrl/cf_hub_missing`). It carries its own prefix now,
so the terminal case is not special and there is nothing to count.

**CF NEGATIVES ARE NEVER IN THE CONSTRAINT.** Two different wrong rewrites of a step have no
reason to be the same point, and asserting it is a claim this data cannot support (the same
rule `losses/counterfactual.py` states for its own negatives). They enter the PUSH term as
sources -- `d(psi(prefix + neg), psi_g)` against same-question goal columns -- which is the
eval-aligned direction: a broken state as the source of the query. That is the ONE thing
`variant_state` is still read for: it names the host trajectory, hence the question whose
goal columns a negative is scored against.

## 4. Empty paths

A micro-batch where nothing attaches contributes an EXACT zero from the CF terms and still
logs the FULL `qrl/cf_*` and `qrl/neg_push_*` key set (`_empty_cf_info`), following
`losses/counterfactual.py`'s `_empty_info`: a diagnostic that disappears when a batch is
degenerate is a diagnostic that cannot be plotted.

fp32 comes free -- `model/distances.py` casts internally, whatever dtype the heads run in
(bug B10a).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from feynman_prm.data.collate import Batch
from feynman_prm.data.goals import GoalIndex
from feynman_prm.model.distances import Distance

from .config import QRLConfig
from .lagrange import LagrangeMultipliers


@dataclass
class QRLLoss:
    total: Tensor
    terms: dict[str, Tensor] = field(default_factory=dict)
    info: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------
# the push term
# ---------------------------------------------------------------------------------------


def _column_slices(n_cols: int, chunk: int):
    """Column ranges for the push matrix. `chunk <= 0` means one shot."""
    if chunk <= 0 or chunk >= n_cols:
        return [slice(0, n_cols)]
    return [slice(a, min(a + chunk, n_cols)) for a in range(0, n_cols, chunk)]


def push_term(
    sources: Tensor,
    psi_goal: Tensor,
    distance: Distance,
    qrl: QRLConfig,
    masks: Optional[dict[str, Tensor]] = None,
    prefix: str = "qrl/push",
) -> tuple[Tensor, dict[str, float]]:
    """`mean_{s,g} softplus_beta(offset - d(s, g))` over the full grid.

    Accumulated as a SUM over column chunks and divided by the pair count at the end, so the
    chunked and one-shot forms agree exactly (up to fp associativity) -- a chunked mean of
    means would not, because the last chunk is short. `masks` are (N, C) bool tensors whose
    mean distance is reported separately; they never affect the loss.
    """
    N, C = int(sources.shape[0]), int(psi_goal.shape[0])
    if N == 0 or C == 0:
        zero = sources.sum() * 0.0 if sources.numel() else psi_goal.sum() * 0.0
        info = {f"{prefix}_dist_mean": 0.0, f"{prefix}_saturated_frac": 0.0, f"{prefix}_pairs": 0.0}
        for name in masks or {}:
            info[f"{prefix}_dist_mean_{name}"] = 0.0
            info[f"{prefix}_pairs_{name}"] = 0.0
        return zero, info

    total_pairs = N * C
    push_sum = sources.new_zeros((), dtype=torch.float32)
    dist_sum = 0.0
    saturated = 0.0
    group_sum = {name: 0.0 for name in (masks or {})}
    group_n = {name: 0.0 for name in (masks or {})}

    for sl in _column_slices(C, qrl.push_chunk_cols):
        d = distance(sources[:, None, :], psi_goal[sl][None, :, :])      # (N, c)
        push_sum = push_sum + F.softplus(
            qrl.softplus_offset - d, beta=qrl.softplus_beta
        ).sum()
        with torch.no_grad():
            dd = d.detach()
            dist_sum += float(dd.sum())
            saturated += float((dd > qrl.softplus_offset).sum())
            for name, mask in (masks or {}).items():
                m = mask[:, sl]
                group_n[name] += float(m.sum())
                if bool(m.any()):
                    group_sum[name] += float(dd[m].sum())

    info = {
        f"{prefix}_dist_mean": dist_sum / total_pairs,
        f"{prefix}_saturated_frac": saturated / total_pairs,
        f"{prefix}_pairs": float(total_pairs),
    }
    for name in masks or {}:
        info[f"{prefix}_dist_mean_{name}"] = (
            group_sum[name] / group_n[name] if group_n[name] else 0.0
        )
        info[f"{prefix}_pairs_{name}"] = group_n[name]
    return push_sum / total_pairs, info


def push_masks(batch: Batch, goals: GoalIndex) -> dict[str, Tensor]:
    """The three (S, C) splits `qrl/push_dist_mean_*` reports.

    They are diagnostics only -- the push term itself is over the WHOLE grid, deliberately:
    QRL's random pair distribution has no notion of "same question", and restricting it would
    make the objective a different one. But a metric whose cross-question distances grow while
    its same-trajectory distances grow just as fast is not learning structure, it is inflating
    a scale, and only the split says which is happening.
    """
    state_traj = batch.state_traj                                     # (S,)
    state_q = batch.traj_qid[state_traj]                              # (S,)
    goal_q = batch.traj_qid[goals.goal_traj]                          # (C,)
    same_traj = state_traj[:, None] == goals.goal_traj[None, :]
    same_q = (state_q[:, None] == goal_q[None, :]) & ~same_traj
    cross_q = state_q[:, None] != goal_q[None, :]
    return {"same_traj": same_traj, "same_question": same_q, "cross_question": cross_q}


# ---------------------------------------------------------------------------------------
# the local constraint
# ---------------------------------------------------------------------------------------


def local_violation(
    psi: Tensor, batch: Batch, distance: Distance, qrl: QRLConfig
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """`(violation, d_step, info)`. `local_constraint.py:56-59`, transition-for-transition."""
    if batch.n_rows == 0:
        # EXACT ZERO, and the logged violation is 0.0 to match. A batch with no transitions is
        # not evidence that the constraint is satisfied, so it must not push the multiplier
        # either way; `-epsilon^2` here would make every empty batch a vote to lower
        # lambda_local. `qrl/local_transitions = 0` is what distinguishes "no data" from
        # "violation happens to be zero". Same rule as `_empty_cf_info`.
        zero = psi.sum() * 0.0
        return zero, psi.new_zeros(0), {
            "qrl/local_dist_mean": 0.0,
            "qrl/local_dist_max": 0.0,
            "qrl/local_sq_dev": 0.0,
            "qrl/local_violation": 0.0,
            "qrl/local_over_cost_frac": 0.0,
            "qrl/local_transitions": 0.0,
        }
    d_step = distance(psi.index_select(0, batch.row_src), psi.index_select(0, batch.row_dst))
    sq_deviation = (d_step - qrl.step_cost).relu().square().mean()
    violation = sq_deviation - qrl.local_target
    with torch.no_grad():
        dd = d_step.detach()
        info = {
            # THE RULER. Should pin to ~step_cost. Watch it against IMPLEMENTATION.md §9's
            # decaying `backup/delta_mean`: that is the failure this objective exists to fix.
            "qrl/local_dist_mean": float(dd.mean()),
            "qrl/local_dist_max": float(dd.max()),
            "qrl/local_sq_dev": float(sq_deviation),
            "qrl/local_violation": float(violation),
            # The constraint is one-sided, so only this fraction is being pushed on at all.
            "qrl/local_over_cost_frac": float((dd > qrl.step_cost).float().mean()),
            "qrl/local_transitions": float(dd.numel()),
        }
    return violation, d_step, info


# ---------------------------------------------------------------------------------------
# the CF constraint and the CF-negative push
# ---------------------------------------------------------------------------------------

_CF_KEYS = (
    "qrl/cf_sq_dev",
    "qrl/cf_violation",
    "qrl/cf_dist_mean",
    "qrl/cf_dist_fwd_mean",
    "qrl/cf_dist_bwd_mean",
    "qrl/cf_p95",
    "qrl/cf_pairs",
    "qrl/cf_active",
    "qrl/cf_examples",
    "qrl/cf_variants",
    "qrl/cf_positives",
    "qrl/cf_negatives",
    "qrl/cf_anchor_missing",
    "qrl/neg_push_dist_mean",
    "qrl/neg_push_pairs",
    "qrl/neg_push_gap",
)


def _empty_cf_info(qrl: QRLConfig) -> dict[str, float]:
    """The empty path logs the SAME key set as the populated one (`losses/counterfactual.py`'s
    `_empty_info` rule), all at 0.0 -- including `qrl/cf_violation`.

    **0.0, not `-cf_target`.** A micro-batch where nothing attached is not evidence that the
    CF constraint is satisfied; it is no evidence at all. Logging `-cf_target` would put a
    real negative violation into the dual gradient, so every CF-free batch would vote to lower
    `lambda_cf` and the multiplier would drift down on data it never saw. `qrl/cf_active`
    (1.0 when the constraint had pairs, 0.0 when it did not) is what separates "no data" from
    "violation happens to be zero", so neither reading is ambiguous on a plot.
    """
    return {k: 0.0 for k in _CF_KEYS}


def anchor_of_example(variant_example: Tensor, variant_kind: Tensor, n_examples: int) -> Tensor:
    """`anchor[e] = the variant index that is example e's anchor`, or -1 if it has none.

    Built by scattering the anchor rows at their example slot. `cf_encode.py` emits exactly
    one `kind == 0` row per example or drops the example whole, so the scatter is a total
    function on the slots present and cannot collide -- but "cannot" is asserted rather than
    assumed: a class whose anchor went missing has no hub, and silently centring it on
    whichever positive happened to sort first would make the constraint mean something else
    on a fraction of examples with no curve to say so. Those examples are dropped and counted
    in `qrl/cf_anchor_missing`.
    """
    anchor = torch.full(
        (n_examples,), -1, dtype=torch.long, device=variant_example.device
    )
    idx = torch.nonzero(variant_kind == 0, as_tuple=False).flatten()
    anchor[variant_example.index_select(0, idx)] = idx
    return anchor


def cf_terms(
    psi: Tensor,
    batch: Batch,
    psi_goal: Tensor,
    goal_q: Tensor,
    distance: Distance,
    qrl: QRLConfig,
    cf: tuple[Tensor, Tensor, Tensor, Tensor],
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """`(cf_violation, neg_push, info)` for one micro-batch's encoded CF variants.

    `cf` is `(psi_variants, variant_state, variant_example, variant_kind)`. `psi_variants` is
    `psi(prompt + steps[:i] + variant + SEP)` from `cf_encode.encode_cf_psi` -- a real
    arrived state, not a `phi` prediction -- so `psi` and `batch` are read here for ONE thing
    only: the question id a CF negative's push is scored against.
    """
    psi_v, variant_state, variant_example, variant_kind = cf
    if psi_v.numel() == 0:
        zero = psi_v.sum() * 0.0
        return zero, zero, _empty_cf_info(qrl)

    n_examples = int(variant_example.max()) + 1
    anchor = anchor_of_example(variant_example, variant_kind, n_examples)

    info: dict[str, float] = {k: 0.0 for k in _CF_KEYS}
    info["qrl/cf_variants"] = float(psi_v.shape[0])
    info["qrl/cf_negatives"] = float(int((variant_kind == 2).sum()))

    # ---- the constraint: anchor -> positive, both directions ------------------------
    is_positive = variant_kind == 1
    anchor_row = anchor.index_select(0, variant_example)          # (V,)
    keep = is_positive & (anchor_row >= 0)
    info["qrl/cf_positives"] = float(int(is_positive.sum()))
    info["qrl/cf_anchor_missing"] = float(int((is_positive & (anchor_row < 0)).sum()))

    if bool(keep.any()):
        idx = torch.nonzero(keep, as_tuple=False).flatten()
        psi_hub = psi_v.index_select(0, anchor_row.index_select(0, idx))   # (P, D)
        psi_pos = psi_v.index_select(0, idx)                              # (P, D)
        d_fwd = distance(psi_hub, psi_pos)                                # anchor -> positive
        d_bwd = distance(psi_pos, psi_hub)                                # positive -> anchor
        # relu is cosmetic (both distances are non-negative by construction) and is kept so
        # the expression matches the constraint as written in the plan and in the README.
        sq_deviation = ((d_fwd.relu().square() + d_bwd.relu().square()) / 2.0).mean()
        cf_violation = sq_deviation - qrl.cf_target
        with torch.no_grad():
            both = torch.cat([d_fwd.detach(), d_bwd.detach()])
            info["qrl/cf_sq_dev"] = float(sq_deviation)
            info["qrl/cf_violation"] = float(cf_violation)
            info["qrl/cf_dist_mean"] = float(both.mean())
            info["qrl/cf_dist_fwd_mean"] = float(d_fwd.detach().mean())
            info["qrl/cf_dist_bwd_mean"] = float(d_bwd.detach().mean())
            # The TAIL, not the mean (§7.12's lesson, in a different loss): a class whose
            # mean sits inside epsilon_cf can still hold a few pairs far outside it, and
            # those are the paraphrases that flip a verdict.
            info["qrl/cf_p95"] = float(torch.quantile(both.float(), 0.95))
            info["qrl/cf_pairs"] = float(both.numel())
            info["qrl/cf_examples"] = float(
                int(torch.unique(variant_example.index_select(0, idx)).numel())
            )
            info["qrl/cf_active"] = 1.0
    else:
        # Variants encoded but not one positive has an anchor to be measured against. Exact
        # zero, and `qrl/cf_anchor_missing` / `qrl/cf_positives` say which of the two it is.
        cf_violation = psi_v.sum() * 0.0

    # ---- CF negatives, as PUSH SOURCES against same-question goals ------------------
    neg = torch.nonzero(variant_kind == 2, as_tuple=False).flatten()
    if neg.numel() and psi_goal.shape[0]:
        psi_neg = psi_v.index_select(0, neg)                                     # (Nn, D)
        neg_state = variant_state.index_select(0, neg)
        neg_q = batch.traj_qid.index_select(0, batch.state_traj.index_select(0, neg_state))
        mask = neg_q[:, None] == goal_q[None, :]                                 # (Nn, C)
        if bool(mask.any()):
            # Only SAME-QUESTION pairs are kept, so building the whole (Nn, C) grid and then
            # indexing it threw away ~1 - 1/questions_per_batch of the work (~92% at the
            # logged 11.8 questions/batch) after paying IQE's full per-pair cost for it.
            # Boolean-mask indexing and `nonzero` both walk the grid row-major, so gathering
            # the kept pairs first gives an ELEMENT-FOR-ELEMENT identical `sel` -- same
            # `neg_push`, same `qrl/neg_push_*` diagnostics, same gradients.
            rows, cols = mask.nonzero(as_tuple=True)
            sel = distance(
                psi_neg.index_select(0, rows), psi_goal.index_select(0, cols)
            )                                                                    # (P,)
            neg_push = F.softplus(
                qrl.softplus_offset - sel, beta=qrl.softplus_beta
            ).mean()
            with torch.no_grad():
                info["qrl/neg_push_dist_mean"] = float(sel.detach().mean())
                info["qrl/neg_push_pairs"] = float(sel.numel())
        else:
            neg_push = psi_v.sum() * 0.0
    else:
        neg_push = psi_v.sum() * 0.0
    return cf_violation, neg_push, info


# ---------------------------------------------------------------------------------------
# the assembled loss
# ---------------------------------------------------------------------------------------


def qrl_loss(
    psi: Tensor,
    batch: Batch,
    goals: GoalIndex,
    distance: Distance,
    qrl: QRLConfig,
    lagrange: LagrangeMultipliers,
    cf: Optional[tuple[Tensor, Tensor, Tensor, Tensor]] = None,
) -> QRLLoss:
    """One micro-batch of the QRL objective.

    **`phi` is not an argument.** Under deterministic dynamics the arrived state is read, not
    predicted (see the header), so every term here is a function of `psi` alone: the observed
    transitions come from `psi[row_src] -> psi[row_dst]`, and the counterfactual variants come
    from `cf_encode.encode_cf_psi`, which is `psi` of a real sequence.

    `cf` is `(psi_variants, variant_state, variant_example, variant_kind)` or None. None and
    "encoded but nothing survives" take the same exact-zero path with the same key set.
    """
    psi_goal = psi.index_select(0, goals.goal_state)                  # (C, D)
    goal_q = batch.traj_qid.index_select(0, goals.goal_traj)          # (C,)

    # ---- the objective: maximize distances (minimize the softplus transform) --------
    # `push_masks` is built unconditionally, including at C = 0, so the `qrl/push_*_same_traj`
    # family never vanishes from `metrics.jsonl` on a degenerate batch (house rule: keys never
    # disappear, `losses/counterfactual.py::_empty_info`).
    l_push, push_info = push_term(
        psi, psi_goal, distance, qrl, masks=push_masks(batch, goals)
    )

    # ---- constraint 1: observed transitions cost about one step --------------------
    viol_local, _, local_info = local_violation(psi, batch, distance, qrl)
    l_local = lagrange.local(viol_local)

    # ---- constraint 2: an equivalence class is one point --------------------------
    if cf is not None and cf[0].numel():
        viol_cf, l_neg_push, cf_info = cf_terms(
            psi, batch, psi_goal, goal_q, distance, qrl, cf
        )
    else:
        zero = psi.sum() * 0.0
        viol_cf, l_neg_push, cf_info = zero, zero, _empty_cf_info(qrl)
    l_cf = lagrange.cf(viol_cf)

    total = (
        l_push
        + qrl.cf_neg_push_weight * l_neg_push
        + l_local
        + l_cf
    )

    info: dict[str, float] = {}
    info.update(push_info)
    info.update(local_info)
    info.update(cf_info)
    info.update(lagrange.values())
    info["qrl/neg_push_gap"] = (
        cf_info["qrl/neg_push_dist_mean"] - push_info["qrl/push_dist_mean"]
        if cf_info["qrl/neg_push_pairs"]
        else 0.0
    )
    terms = {
        "push": l_push.detach(),
        "neg_push": l_neg_push.detach(),
        "local": l_local.detach(),
        "cf": l_cf.detach(),
    }
    info.update({f"loss/{k}": float(v) for k, v in terms.items()})
    info["loss/total"] = float(total.detach())
    return QRLLoss(total=total, terms=terms, info=info)


# ---------------------------------------------------------------------------------------
# §18: the initialisation probe. COMPUTE these, never eyeball them
# ---------------------------------------------------------------------------------------


def expected_init_values(
    qrl: QRLConfig,
    lagrange: LagrangeMultipliers,
    info: dict[str, float],
) -> dict[str, float]:
    """Closed forms for each term FROM THE MEASURED BATCH MEANS in `info`.

    §18's discipline is that an ASSUMED initialisation value is how two regressions got
    through, so nothing here is a guessed constant -- every prediction is a function of a
    quantity this same micro-batch measured:

    * **`push`** -- `softplus_beta(offset - push_dist_mean)`. This is a LOWER BOUND, not an
      equality: softplus is convex, so by Jensen `mean(f(d)) >= f(mean(d))`. Asserting
      equality would fire on a correct run whose distances have any spread at all. The
      one-sided check is what is exact, and it still catches the failure that matters (a
      push term that is not the transform of its own logged distances).
    * **`local` / `cf`** -- `lambda * violation` with the MEASURED violation. `grad_mul` is
      the identity in the forward pass, so these are EXACT and are asserted as equalities.
      They are the check on the multiplier plumbing: a raw scalar stored without
      `softplus_inv` starts the multiplier 70x high and this is where that shows.
    * **`neg_push`** -- the same transform of its own logged mean, a lower bound for the same
      Jensen reason.

    There is no `dyn` row and there is no `phi` in this file (see the header): the arrived
    state is read, not predicted, so there is no latent-dynamics term to predict a value for.

    Returns the predictions; `train.py` does the comparing and the asserting.
    """
    beta, offset = qrl.softplus_beta, qrl.softplus_offset
    softplus = lambda x: math.log1p(math.exp(min(beta * x, 500.0))) / beta  # noqa: E731

    lam = lagrange.values()
    out = {
        "push": softplus(offset - info["qrl/push_dist_mean"]),
        "local": lam["qrl/lagrange_local"] * info["qrl/local_violation"],
        "cf": lam["qrl/lagrange_cf"] * info["qrl/cf_violation"],
        "neg_push": (
            softplus(offset - info["qrl/neg_push_dist_mean"])
            if info["qrl/neg_push_pairs"]
            else 0.0
        ),
    }
    out["total"] = (
        out["push"]
        + qrl.cf_neg_push_weight * out["neg_push"]
        + out["local"]
        + out["cf"]
    )
    return out
