"""(4) L_CF tests (§7.5). The loss and the on-disk format are built now; lambda_cf = 0 and
the DATA is deferred (locked #4).

The loss is MULTI-POSITIVE since 2026-08-08: one example carries an equivalence class
`C = {anchor} u positives`, pulled together, with the negatives pushed away from the class
and never used as queries. Two of the tests below exist because the single-positive
implementation this replaced failed them SILENTLY -- it trained positives 2..P as negatives
while `cf/positive_distance`, read off slot 0, still fell.
"""

from __future__ import annotations

import dataclasses
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

D = 512


def _example(**kw):
    base = dict(
        question="what is x?",
        steps=("Step 1: x + 2 = 4", "Step 2: x = 2"),
        step_index=0,
        positive_rewrites=("Step 1: 4 - x = 2", "Step 1: x equals 4 minus 2"),
        negative_rewrites=("Step 1: x - 2 = 4",),
    )
    base.update(kw)
    return CounterfactualExample(**base)


def _phi(*rows: torch.Tensor) -> torch.Tensor:
    return torch.cat([r.reshape(1, -1) for r in rows])


def _kinds(n_positives: int, n_negatives: int) -> tuple[torch.Tensor, torch.Tensor]:
    kind = [0] + [1] * n_positives + [2] * n_negatives
    return torch.zeros(len(kind), dtype=torch.long), torch.tensor(kind)


# ------------------------------------------------------------------ the on-disk format

def test_jsonl_round_trip_with_three_positives(tmp_path):
    examples = [
        _example(positive_rewrites=("a plus b", "b plus a", "the sum of a and b")),
        _example(step_index=1),
    ]
    path = tmp_path / "cf.jsonl"
    write_jsonl(examples, path)
    assert read_jsonl(path) == examples
    assert len(read_jsonl(path)[0].positive_rewrites) == 3


def test_schema_is_validated():
    with pytest.raises(ValueError, match="step_index"):
        _example(step_index=5)
    with pytest.raises(ValueError, match="at least one negative"):
        _example(negative_rewrites=())
    with pytest.raises(ValueError, match="at least one positive"):
        _example(positive_rewrites=())


# ----------------------------------------------------------------------------- the loss

def test_moving_a_non_first_positive_closer_lowers_the_loss():
    """REGRESSION GUARD, and the highest-value test in this file.

    Three positives and three negatives. Positive #2 -- not the first one -- is moved toward
    the anchor. Under the multi-positive loss that must LOWER the loss: it is a member of the
    equivalence class and the class is being pulled together.

    **The single-positive implementation this replaced FAILS here, in the opposite
    direction.** It sorted candidates by `cand_example * 1000 + variant_kind` and targeted
    class 0, so only the FIRST kind-1 row was the positive and rows 2..P were trained as
    negatives -- the loss pushed meaning-preserving rewrites of one step apart. Nothing
    crashed, and `cf/positive_distance` (slot 0 only) still fell, so no curve showed it.
    """
    dist = Distance("full_mrn", 8)
    torch.manual_seed(0)
    anchor = torch.randn(D)
    far = [torch.randn(D) * 3.0 for _ in range(3)]     # 3 negatives
    p1, p2, p3 = anchor + 0.5 * torch.randn(D), anchor + 2.0 * torch.randn(D), anchor + 0.5 * torch.randn(D)

    variant_example, variant_kind = _kinds(3, 3)
    before, _ = counterfactual_loss(
        _phi(anchor, p1, p2, p3, *far), variant_example, variant_kind, dist
    )
    # Halve p2's displacement from the anchor: strictly closer, nothing else moved.
    p2_closer = anchor + 0.5 * (p2 - anchor)
    after, _ = counterfactual_loss(
        _phi(anchor, p1, p2_closer, p3, *far), variant_example, variant_kind, dist
    )
    assert float(after) < float(before), (
        "a positive that is not in slot 0 is being trained as a negative -- "
        "the single-positive L_CF is back"
    )


def test_whichever_class_member_is_the_outlier_is_pulled_in():
    """The loss falls when the class member currently FARTHEST from the rest is brought in --
    for each of the three positives in turn, when it is the one made the outlier.

    "Farthest", not "any". `dL_q/ds_p = softmax_p - 1/|P|`, so a positive is pulled in only
    while it carries LESS than its fair share of the softmax mass. The minimum always does
    (the shares sum to at most 1), which makes the outlier case the theorem and the general
    case false -- see the next test.
    """
    dist = Distance("full_mrn", 8)
    negatives = [torch.randn(D) * 4.0 for _ in range(3)]
    variant_example, variant_kind = _kinds(3, 3)

    for outlier in range(3):
        torch.manual_seed(10 + outlier)
        anchor = torch.randn(D)
        positives = [anchor + 0.5 * torch.randn(D) for _ in range(3)]
        positives[outlier] = anchor + 2.5 * torch.randn(D)
        base, _ = counterfactual_loss(
            _phi(anchor, *positives, *negatives), variant_example, variant_kind, dist
        )
        pulled = list(positives)
        pulled[outlier] = anchor + 0.4 * (positives[outlier] - anchor)
        loss, _ = counterfactual_loss(
            _phi(anchor, *pulled, *negatives), variant_example, variant_kind, dist
        )
        assert float(loss) < float(base), f"the outlier at slot {outlier + 1} is not pulled in"


def test_a_positive_that_already_dominates_is_not_pulled_further():
    """**This is what keeping the positives in the DENOMINATOR buys**, and it is not a defect.

    Collapsing one positive onto the anchor while the others stay far does NOT lower the
    loss -- it raises it. The objective wants the class UNIFORMLY tight, which is the
    difference between one equivalence class and `|P|` independent one-positive problems.
    Recorded as a test so it is not "fixed" into a per-positive loss later.
    """
    dist = Distance("full_mrn", 8)
    torch.manual_seed(11)
    anchor = torch.randn(D)
    positives = [anchor + 1.5 * torch.randn(D) for _ in range(3)]
    negatives = [torch.randn(D) * 4.0 for _ in range(3)]
    variant_example, variant_kind = _kinds(3, 3)

    base, _ = counterfactual_loss(
        _phi(anchor, *positives, *negatives), variant_example, variant_kind, dist
    )
    collapsed = [anchor + 0.001 * torch.randn(D)] + positives[1:]
    loss, _ = counterfactual_loss(
        _phi(anchor, *collapsed, *negatives), variant_example, variant_kind, dist
    )
    assert float(loss) > float(base)


def test_every_negative_is_pushed_away():
    """Unconditional, unlike the positive case: `dL_q/ds_n = softmax_n > 0` for every
    negative and every query, so moving any negative closer always raises the loss."""
    dist = Distance("full_mrn", 8)
    torch.manual_seed(12)
    anchor = torch.randn(D)
    positives = [anchor + 1.5 * torch.randn(D) for _ in range(3)]
    negatives = [anchor + 4.0 * torch.randn(D) for _ in range(3)]
    variant_example, variant_kind = _kinds(3, 3)
    base, _ = counterfactual_loss(
        _phi(anchor, *positives, *negatives), variant_example, variant_kind, dist
    )

    for i in range(3):
        moved = list(negatives)
        moved[i] = anchor + 0.25 * (negatives[i] - anchor)
        loss, _ = counterfactual_loss(
            _phi(anchor, *positives, *moved), variant_example, variant_kind, dist
        )
        assert float(loss) > float(base), f"negative {i} is not being pushed away"


def test_permuting_the_negatives_leaves_the_loss_unchanged():
    """No negative is ever a QUERY, so the negatives enter only as a set."""
    dist = Distance("full_mrn", 8)
    torch.manual_seed(2)
    rows = [torch.randn(D) for _ in range(7)]          # anchor + 2 positives + 4 negatives
    variant_example, variant_kind = _kinds(2, 4)

    base, _ = counterfactual_loss(_phi(*rows), variant_example, variant_kind, dist)
    for permutation in ([6, 5, 4, 3], [4, 6, 3, 5], [5, 3, 6, 4]):
        shuffled = rows[:3] + [rows[i] for i in permutation]
        loss, _ = counterfactual_loss(_phi(*shuffled), variant_example, variant_kind, dist)
        assert float(loss) == float(base)


def test_moving_two_negatives_closer_together_changes_nothing():
    """There is no negative-negative term: two different WRONG actions have no reason to
    coincide, and asserting they do is a claim this data cannot support.

    Construction. Put the whole class at the origin. Then `d(0, n)` is
    `mean_K [ max(relu(-n))_asym + L2(n)_sym ]`, and BOTH halves are invariant to permuting
    coordinates WITHIN a component-half -- a max and an L2 do not care about order. So a
    negative can be moved (changing `d(n_i, n_j)`) with every class->negative distance held
    EXACTLY fixed. If the loss moves, a negative was a query.
    """
    dist = Distance("full_mrn", 8)
    components, half = 8, D // 8 // 2
    torch.manual_seed(3)
    klass = [torch.zeros(D) for _ in range(3)]          # anchor + 2 positives, all at 0
    n1, n2 = torch.randn(D).abs() + 0.5, torch.randn(D).abs() + 0.5
    variant_example, variant_kind = _kinds(2, 2)

    def shuffle_within_halves(vector: torch.Tensor, seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        blocks = vector.reshape(components, 2, half).clone()
        for k in range(components):
            for h in range(2):
                blocks[k, h] = blocks[k, h][torch.randperm(half, generator=generator)]
        return blocks.reshape(D)

    base, base_info = counterfactual_loss(
        _phi(*klass, n1, n2), variant_example, variant_kind, dist
    )
    # Of a few candidate reorderings, take the one that brings the two negatives CLOSEST.
    candidates = [shuffle_within_halves(n2, seed) for seed in range(8)]
    n2_closer = min(candidates, key=lambda v: float(dist(n1.reshape(1, -1), v.reshape(1, -1))))

    assert float(dist(n1[None], n2_closer[None])) < float(dist(n1[None], n2[None])), \
        "the fixture did not actually move the two negatives closer together"
    assert torch.allclose(dist(klass[0][None], n2_closer[None]), dist(klass[0][None], n2[None])), \
        "the fixture changed a class->negative distance, so it proves nothing"

    moved, moved_info = counterfactual_loss(
        _phi(*klass, n1, n2_closer), variant_example, variant_kind, dist
    )
    assert float(moved) == pytest.approx(float(base), abs=1e-6)
    assert moved_info["cf/negative_distance"] == pytest.approx(
        base_info["cf/negative_distance"], abs=1e-6
    )


def _single_positive_reference(phi, dist, temperature=1.0):
    """The documented single-positive form of §7.5: a cross-entropy over `{s_2} u {n_i}`
    with the positive at index 0, scored from the anchor. Row 0 is the anchor, row 1 the
    positive, rows 2.. the negatives."""
    d = dist(phi[0:1].expand(phi.shape[0] - 1, -1), phi[1:])
    return -torch.log_softmax(-d / temperature, dim=0)[0]


def test_one_positive_reproduces_the_single_positive_loss():
    """The new form must GENERALISE the old one, not replace it.

    Scored on a fixture where the class is one point, so the anchor-as-query and
    positive-as-query terms are the same number and their mean is the old value exactly.
    """
    dist = Distance("full_mrn", 8)
    torch.manual_seed(4)
    anchor = torch.randn(D)
    phi = _phi(anchor, anchor.clone(), *[torch.randn(D) * 2.0 for _ in range(4)])
    variant_example, variant_kind = _kinds(1, 4)

    loss, _ = counterfactual_loss(phi, variant_example, variant_kind, dist)
    assert float(loss) == pytest.approx(float(_single_positive_reference(phi, dist)), rel=1e-5)


def test_uniform_at_chance():
    """All variants identical: every query sees a flat softmax over its 3 candidates."""
    dist = Distance("full_mrn", 8)
    phi = torch.zeros(4, D)                      # anchor + positive + 2 negatives
    loss, _ = counterfactual_loss(
        phi, torch.tensor([0, 0, 0, 0]), torch.tensor([0, 1, 2, 2]), dist
    )
    assert math.isclose(float(loss), math.log(3), rel_tol=1e-4)


def test_ragged_variant_counts_are_handled():
    dist = Distance("full_mrn", 8)
    torch.manual_seed(0)
    phi = torch.randn(11, D)
    variant_example = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    variant_kind = torch.tensor([0, 1, 1, 2, 2, 0, 1, 1, 1, 2, 2])   # 2 vs 3 positives
    loss, info = counterfactual_loss(phi, variant_example, variant_kind, dist)
    assert torch.isfinite(loss)
    assert info["cf/examples"] == 2.0
    assert info["cf/positives_per_example"] == 2.5
    assert info["cf/negatives_per_example"] == 2.0


def test_diagnostics_cover_all_pairs_not_slot_zero():
    """`cf/positive_distance` is the mean over ALL class pairs, both orderings, not the
    anchor against the first positive -- which is what hid the old bug."""
    dist = Distance("full_mrn", 8)
    torch.manual_seed(5)
    anchor = torch.randn(D)
    near, far = anchor + 0.01 * torch.randn(D), anchor + 6.0 * torch.randn(D)
    negatives = [torch.randn(D) * 3.0 for _ in range(2)]
    variant_example, variant_kind = _kinds(2, 2)

    _, info = counterfactual_loss(
        _phi(anchor, near, far, *negatives), variant_example, variant_kind, dist
    )
    # 3 class members -> 6 ordered pairs; a slot-0-only reading would report ~d(anchor, near),
    # which is nearly zero.
    assert info["cf/positive_distance"] > float(dist(anchor[None], near[None])) * 10
    assert info["cf/positives_per_example"] == 2.0
    assert info["cf/negatives_per_example"] == 2.0


def test_gradients_reach_every_variant():
    dist = Distance("full_mrn", 8)
    torch.manual_seed(6)
    phi = torch.randn(7, D, requires_grad=True)
    variant_example, variant_kind = _kinds(2, 4)
    loss, _ = counterfactual_loss(phi, variant_example, variant_kind, dist)
    loss.backward()
    assert torch.isfinite(phi.grad).all()
    assert (phi.grad.abs().sum(dim=1) > 0).all(), "some variant receives no gradient"


def test_empty_batch_logs_the_same_keys():
    dist = Distance("full_mrn", 8)
    phi = torch.zeros(0, D)
    loss, info = counterfactual_loss(
        phi, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), dist
    )
    assert float(loss) == 0.0
    populated = counterfactual_loss(
        torch.zeros(4, D), torch.tensor([0, 0, 0, 0]), torch.tensor([0, 1, 2, 2]), dist
    )[1]
    assert set(info) == set(populated)


# --------------------------------------------------------------------------- the batch

def test_variants_share_one_prefix_forward(tokenizer):
    """PLAN finding 1, contra §7.5: rewriting step i leaves the prefix -- hence h_{i-1} --
    untouched, so a variant costs an embedding lookup plus an MLP, not an LM forward. The
    batch therefore carries ONE prefix sequence per example and N variant token lists."""
    examples = [
        _example(
            positive_rewrites=("Step 1: 4 - x = 2", "Step 1: x is 4 less 2", "Step 1: 2 = 4 - x"),
            negative_rewrites=("Step 1: x - 2 = 4", "Step 1: x + 2 = 5"),
        ),
        _example(step_index=1),
    ]
    batch = build_cf_batch(tokenizer, examples, sep_id=SEP_ID, pad_id=0)
    assert batch.n_examples == 2
    assert batch.n_variants == (1 + 3 + 2) + (1 + 2 + 1)
    flat = batch.input_ids.reshape(-1)
    assert torch.all(flat[batch.anchor_flat_idx] == SEP_ID), "anchors sit on s_{i-1}"
    # THREE kind-1 rows for the first example, not one.
    assert batch.variant_kind.tolist() == [0, 1, 1, 1, 2, 2, 0, 1, 1, 2]
    assert int((batch.variant_kind[batch.variant_example == 0] == 1).sum()) == 3


def test_lambda_cf_ships_at_one(cfg):
    """RAISED 0.0 -> 1.0 ON 2026-08-15, and this pin flipped with it.

    It asserted 0.0 from the day the loss was written, because locked #4 deferred the DATA
    and the term was wired-and-inert. Both of #4's gates are now cleared -- 27,114 examples
    on disk (§7.5.2) and the main-batch attach path wired (§7.5.13) -- so the shipped weight
    is 1.0, at parity with every other spec term in the set.

    The pin is kept rather than deleted: NOTHING ABOUT THIS WEIGHT IS MEASURED (§16.8), so
    the value is worth a deliberate act to change, in either direction.
    """
    assert cfg.losses.lambda_cf == 1.0


def _cf_forward(cfg):
    """One batch, one set of latents, and a real 3-positive/6-negative CF forward on top."""
    from test_smoke import _run
    from feynman_prm.losses.total import phase1_loss

    torch.manual_seed(7)
    batch, goals, psi, phi, distance, matrices, _ = _run(cfg)
    n_cf, n_pos, n_neg = 4, 3, 6
    per_example = 1 + n_pos + n_neg
    phi_variants = torch.randn(n_cf * per_example, cfg.heads.latent_dim, requires_grad=True)
    variant_example = torch.arange(n_cf).repeat_interleave(per_example)
    variant_kind = torch.tensor(([0] + [1] * n_pos + [2] * n_neg) * n_cf)
    return phase1_loss(
        psi, phi, batch, matrices, distance, cfg,
        goal_traj=goals.goal_traj,
        cf=(phi_variants, variant_example, variant_kind),
    )


def test_the_total_is_bit_identical_with_and_without_the_cf_term_at_lambda_zero(cfg):
    """§15: at `lambda_cf = 0` the losses are bit-identical whether or not the CF paths run.

    This is what makes `--set losses.lambda_cf=0.0` an EXACT reproduction of the 2026-08-04
    baseline rather than an approximation of it, which is the whole value of the one-line
    revert. `lambda_cf` is now overridden explicitly instead of read off the shipped config:
    the property under test is about the weight being zero, not about which weight ships, and
    tying it to the default is what made this test fail when the default moved.

    Mirrors `tests/test_smoke.py::test_cf_is_inert_at_lambda_zero`, with the multi-positive
    term ACTUALLY FED so the guard is `lambda_cf == 0`, not `cf is None`.
    """
    from test_smoke import _run

    off = dataclasses.replace(cfg, losses=dataclasses.replace(cfg.losses, lambda_cf=0.0))

    torch.manual_seed(7)
    without = _run(off)[-1]
    with_cf = _cf_forward(off)

    assert float(with_cf.total) == float(without.total)
    assert float(with_cf.terms["cf"]) == 0.0


def test_at_the_shipped_weight_the_cf_term_actually_moves_the_total(cfg):
    """The complement, and the one that would have caught "wired but inert".

    §7.5.13 shipped a path that was fed, logged and multiplied by zero; the bit-identical
    test above PASSES in that state and says nothing about whether raising the weight does
    anything. So assert the other direction at the weight that actually ships: (4) is
    nonzero, it enters the total, and gradient reaches the variants.
    """
    from test_smoke import _run

    assert cfg.losses.lambda_cf > 0.0

    torch.manual_seed(7)
    without = _run(cfg)[-1]
    with_cf = _cf_forward(cfg)

    assert float(with_cf.terms["cf"]) > 0.0
    assert float(with_cf.total) != float(without.total)
    assert float(with_cf.total) == pytest.approx(
        float(without.total) + cfg.losses.lambda_cf * float(with_cf.terms["cf"]), rel=1e-5
    ), "(4) must enter the total at exactly lambda_cf, unscaled and unwarmed"


def test_tau_cf_is_a_deliberate_pick_and_not_an_inherited_default(cfg):
    """§7.13's rule, applied to (4) on 2026-08-18 -- and this one had it WORSE than (7) did.

    `total.py` called `counterfactual_loss(*cf, distance=distance)` with no `temperature=` at
    all, so the 2026-08-15 run raised `lambda_cf` 0.0 -> 1.0 against the function's own 1.0
    default: not merely an unexamined pick, but no pick at all. The pin is the same shape as
    `test_tau_is_a_deliberate_pick_and_not_an_inherited_default` in `test_terminal_class.py`.

    0.1, not 1.0, because §7.5.14 MEASURED (4) operating at a distance spread of ~0.06 --
    every variant shares `h_{i-1}` and differs only in a frozen `act_emb` -- so at tau = 1.0
    the ~9-way softmax is flat to three decimals and `cf/loss` reads chance while the term is
    visibly separating.
    """
    import re
    from pathlib import Path

    assert cfg.losses.lambda_cf_temperature == 0.1

    total = Path(__file__).resolve().parents[1] / "feynman_prm" / "losses" / "total.py"
    code = re.sub(r"#.*", "", total.read_text())
    call = code[code.index("counterfactual_loss("):]
    assert "lambda_cf_temperature" in call[: call.index(")") + 200], (
        "total.py must pass tau EXPLICITLY from the config -- letting it default is the "
        "unexamined inheritance §7.13 forbids, and it is how (4) trained at tau = 1.0"
    )


def test_the_configured_tau_reaches_the_loss(cfg):
    """The pin above is a grep; this one is the behaviour. Two `phase1_loss` calls that
    differ ONLY in `losses.lambda_cf_temperature` must produce different `cf/loss` values on
    identical latents -- which they cannot if the argument is dropped on the way through.
    """
    import dataclasses as _dc

    hot = _dc.replace(cfg, losses=_dc.replace(cfg.losses, lambda_cf_temperature=1.0))
    cold = _dc.replace(cfg, losses=_dc.replace(cfg.losses, lambda_cf_temperature=0.1))

    torch.manual_seed(7)
    a = _cf_forward(hot).info["cf/loss"]
    torch.manual_seed(7)
    b = _cf_forward(cold).info["cf/loss"]

    assert a != pytest.approx(b, rel=1e-4), (
        "cf/loss is identical at tau 1.0 and 0.1 -- the config value is not reaching "
        "counterfactual_loss"
    )


def test_cf_chance_is_the_flat_softmax_level(cfg):
    """Diagnostic #19: a contrastive loss is only readable against its OWN chance level, and
    (4) was the last term in the set without one. `cf/loss` ~= 2.157 read as "dead" for a
    whole run when log(P+N) = 2.19 says it was correct-at-chance (§7.5.14).

    On identical variants every query sees a flat softmax over its `P + N` candidates, so the
    loss IS the chance level -- and the key must equal it exactly, at any temperature.
    """
    dist = Distance("full_mrn", 8)
    phi = torch.zeros(10, D)                       # anchor + 3 positives + 6 negatives
    variant_example, variant_kind = _kinds(3, 6)

    for temperature in (1.0, 0.1):
        loss, info = counterfactual_loss(
            phi, variant_example, variant_kind, dist, temperature=temperature
        )
        assert info["cf/chance"] == pytest.approx(math.log(9), rel=1e-6)
        assert info["cf/loss"] == pytest.approx(info["cf/chance"], rel=1e-4)


def test_cf_chance_moves_with_the_batchs_ragged_counts():
    """`log(P+N)` is NOT a constant -- it moves with the batch, exactly as `term/chance` does
    and unlike `nce/chance`'s `log(R)`. Two examples with different denominators mean the
    chance level is a mean over them, not either one."""
    dist = Distance("full_mrn", 8)
    phi = torch.zeros(11, D)
    variant_example = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    variant_kind = torch.tensor([0, 1, 1, 2, 2, 0, 1, 1, 1, 2, 2])   # 4 vs 5 candidates
    _, info = counterfactual_loss(phi, variant_example, variant_kind, dist)
    assert info["cf/chance"] == pytest.approx((math.log(4) + math.log(5)) / 2, rel=1e-6)
