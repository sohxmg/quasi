"""`qrl_prm/` -- QRL's constrained objective plus a counterfactual-invariance constraint.

CPU only, no model, no GPU, no download. Everything here runs on the same synthetic rows the
rest of the suite uses, because `qrl_prm` inherits the project's separation of index
bookkeeping from the model (PLAN 'Core design decisions' 1).

Most tests run against `DirectedDistance` rather than the real MRN/IQE heads. That is
deliberate: `d(x, y) = sum(relu(x - y))` is strongly asymmetric, exactly zero at `x == y`, and
CLOSED FORM, so "both directions enter" and "the hub is the arrived state" become equalities
against hand-computed numbers instead of inequalities against a black box. The real `Distance`
is exercised in the integration test at the bottom, at `variant=iqe` -- the head the decided
run uses.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from conftest import REPO_ROOT, synthetic_row

from feynman_prm.data.collate import collate
from feynman_prm.data.goals import sample_goals
from feynman_prm.model.distances import Distance
from qrl_prm.config import QRLConfig, load_qrl_config, split_overrides
from qrl_prm.lagrange import (
    LagrangeMultiplier,
    LagrangeMultipliers,
    grad_mul,
    softplus_inv_float,
)
from qrl_prm.cf_encode import CF_ENCODE_KEYS, CFEncodeContext, empty_encode_info
from qrl_prm.loss import (
    _CF_KEYS,
    _empty_cf_info,
    anchor_of_example,
    cf_terms,
    expected_init_values,
    local_violation,
    push_masks,
    push_term,
    qrl_loss,
)

T, F = True, False
D = 8


class DirectedDistance(nn.Module):
    """`d(x, y) = sum(relu(x - y))`. A quasimetric: non-negative, 0 at x == y, triangle
    inequality holds, and `d(x, y) != d(y, x)` in general. Closed form, so every expectation
    below is an equality."""

    def forward(self, x, y):
        return (x - y).relu().sum(-1)


@pytest.fixture
def qrl() -> QRLConfig:
    return QRLConfig()


@pytest.fixture
def batch():
    """Two questions, each one correct and one incorrect trajectory -- `conftest.small_batch`'s
    shape, rebuilt here so the state layout can be asserted against directly."""
    rows = [
        synthetic_row("q1", [T, T, T]),        # traj 0: states 0..3,  rows 0..2
        synthetic_row("q1", [T, T, F, F]),     # traj 1: states 4..8,  rows 3..6
        synthetic_row("q2", [T, T]),           # traj 2: states 9..11, rows 7..8
        synthetic_row("q2", [F, F, F]),        # traj 3: states 12..15, rows 9..11
    ]
    return collate(rows, pad_id=0)


def ladder_psi(batch, gap: float = 1.0) -> torch.Tensor:
    """`psi[s] = -gap * step(s) * e_0`, so under `DirectedDistance` every observed transition
    measures EXACTLY `gap` forward and 0 backward."""
    psi = torch.zeros(batch.n_states, D)
    psi[:, 0] = -gap * batch.state_step.float()
    return psi.requires_grad_(True)


def goals_for(batch, seed: int = 0):
    return sample_goals(batch, 0.5, np.random.default_rng(seed))


# =======================================================================================
# 1. multiplier mechanics
# =======================================================================================


def test_softplus_inv_round_trips_so_init_lagrange_means_the_multiplier():
    """`init_lagrange: 0.01` is the MULTIPLIER's value, not the raw scalar's. Storing 0.01 raw
    would start the multiplier at softplus(0.01) = 0.698 -- 70x high, and nothing downstream
    would say so."""
    for value in (0.001, 0.01, 0.5, 3.0, 25.0):
        m = LagrangeMultiplier(value)
        assert float(m.value) == pytest.approx(value, rel=1e-5)
    # above torch's documented threshold softplus is the identity, and so is its inverse
    assert softplus_inv_float(25.0) == 25.0


def test_grad_mul_is_identity_forward_and_negates_backward():
    x = torch.tensor(3.0, requires_grad=True)
    y = grad_mul(x, -1.0)
    assert float(y) == 3.0                      # forward: untouched
    y.backward()
    assert float(x.grad) == -1.0                # backward: sign flipped


def test_forward_value_is_exactly_lambda_times_violation():
    m = LagrangeMultiplier(0.01)
    viol = torch.tensor(0.7)
    assert float(m(viol)) == pytest.approx(float(m.value) * 0.7, rel=1e-6)


@pytest.mark.parametrize("violation,direction", [(0.5, "up"), (-0.5, "down")])
def test_violation_sign_drives_the_multiplier(violation, direction):
    """Positive violation => lambda RISES (the constraint needs more weight); negative
    violation => lambda FALLS. This is the whole point of the dual variable, and it is what
    `grad_mul(-1)` buys."""
    m = LagrangeMultiplier(0.5)
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    before = float(m.value)
    m(torch.tensor(violation)).backward()
    opt.step()
    after = float(m.value)
    assert (after > before) if direction == "up" else (after < before)


def test_the_primal_still_descends_the_violation():
    """The sign flip must reach the MULTIPLIER only. If it leaked to the primal, the model
    would be trained to VIOLATE its own constraints -- which no curve would reveal, because
    the multiplier would still look like it was behaving."""
    m = LagrangeMultiplier(0.5)
    d = torch.tensor(2.0, requires_grad=True)
    viol = (d - 1.0).relu().square() - 0.0625
    m(viol).backward()
    assert float(d.grad) > 0     # gradient DESCENT on d lowers the violation


# =======================================================================================
# 2. the local constraint
# =======================================================================================


def test_local_constraint_is_zero_at_exactly_step_cost(batch, qrl):
    psi = ladder_psi(batch, gap=qrl.step_cost)
    viol, d_step, info = local_violation(psi, batch, DirectedDistance(), qrl)
    assert info["qrl/local_dist_mean"] == pytest.approx(qrl.step_cost)
    assert info["qrl/local_sq_dev"] == pytest.approx(0.0, abs=1e-6)
    assert float(viol) == pytest.approx(-qrl.local_target)


def test_local_constraint_is_one_sided(batch, qrl):
    """A transition SHORTER than one step is free (`(d - step_cost).relu()`). The push term is
    what stops everything collapsing to zero; the constraint never pulls upward."""
    psi = ladder_psi(batch, gap=0.25 * qrl.step_cost)
    _, _, info = local_violation(psi, batch, DirectedDistance(), qrl)
    assert info["qrl/local_dist_mean"] == pytest.approx(0.25)
    assert info["qrl/local_sq_dev"] == 0.0
    assert info["qrl/local_over_cost_frac"] == 0.0


def test_local_constraint_squares_before_it_means(batch, qrl):
    """`mean(relu(d - c)^2)`, NOT `mean(relu(d - c))^2` -- QRL's own order
    (`local_constraint.py:59`). The two agree only when every deviation is equal, which is
    exactly the fixture a careless test would use.
    """
    # Two trajectories at gap 1 (deviation 0) and two at gap 3 (deviation 2).
    psi = torch.zeros(batch.n_states, D)
    fast = torch.isin(batch.state_traj, torch.tensor([2, 3]))
    gap = torch.where(fast, 3.0, 1.0)
    psi[:, 0] = -gap * batch.state_step.float()
    psi.requires_grad_(True)

    _, d_step, info = local_violation(psi, batch, DirectedDistance(), qrl)
    dev = (d_step.detach() - qrl.step_cost).relu()
    assert info["qrl/local_sq_dev"] == pytest.approx(float(dev.square().mean()))
    assert info["qrl/local_sq_dev"] != pytest.approx(float(dev.mean()) ** 2)


def test_local_constraint_covers_incorrect_trajectories_too(batch, qrl):
    """Every observed transition, correct AND incorrect: a step is an observed step whatever
    its verdict, and a metric that cannot measure the wrong ones cannot score them either."""
    psi = ladder_psi(batch)
    _, _, info = local_violation(psi, batch, DirectedDistance(), qrl)
    assert info["qrl/local_transitions"] == float(batch.n_rows) == 12.0


# =======================================================================================
# 3. the CF constraint -- the star topology, centred on the ANCHOR
# =======================================================================================


def make_cf(batch, states, kinds, values, dim: int = D):
    """`(psi_variants, variant_state, variant_example, variant_kind)` for one CF example per
    distinct departure state, in `EncodedCF`'s layout.

    **`psi_variants` is psi, not phi.** Under deterministic dynamics a variant IS a state --
    `psi(prompt + steps[:i] + variant + SEP)`, built by `cf_encode.py` and produced by a real
    LM forward -- so these rows live in the same space as `psi` itself and `dim` must match the
    psi they are measured against.
    """
    psi_v = torch.zeros(len(states), dim)
    for v, value in enumerate(values):
        psi_v[v, 0] = value
    psi_v.requires_grad_(True)
    uniq = {}
    example = [uniq.setdefault(int(s), len(uniq)) for s in states]
    return (
        psi_v,
        torch.as_tensor(states, dtype=torch.long),
        torch.as_tensor(example, dtype=torch.long),
        torch.as_tensor(kinds, dtype=torch.long),
    )


def test_the_hub_is_the_example_s_own_anchor_never_a_batch_state(batch, qrl):
    """THE indexing test, and it replaces `hub_of_state`.

    The hub used to be `psi[row_dst]` -- the arrived state of the HOST trajectory -- and that
    was wrong on exactly the attachments the prefix join exists to buy: a sibling trajectory
    agrees on `steps[:i]` and is free to differ at step `i`, so its `row_dst` is the arrived
    state of a DIFFERENT step than the one the class was written about. Now the class centres
    on its own encoded anchor, which is the same text the positives are rewrites of.

    Two examples departing from ONE state pin it: under the old derivation both would have
    shared `row_dst[0]` and their two classes would have been measured against one point.
    """
    anchor = anchor_of_example(
        torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]), n_examples=2
    )
    assert anchor.tolist() == [0, 2]

    psi = torch.zeros(batch.n_states, D, requires_grad=True)
    goals = goals_for(batch)
    psi_goal = psi.index_select(0, goals.goal_state)
    goal_q = batch.traj_qid[goals.goal_traj]
    # both examples depart from state 0; anchors at 0 and 10, positives one unit above each
    cf = make_cf(batch, [0, 0, 0, 0], [0, 1, 0, 1], [0.0, 1.0, 10.0, 11.0])
    cf = (cf[0], cf[1], torch.tensor([0, 0, 1, 1]), cf[3])
    _, _, info = cf_terms(psi, batch, psi_goal, goal_q, DirectedDistance(), qrl, cf)
    assert info["qrl/cf_examples"] == 2.0
    assert info["qrl/cf_pairs"] == 4.0                    # 2 positives x 2 directions
    # each positive measured against ITS OWN anchor: 1 unit backward, 0 forward, both classes
    assert info["qrl/cf_sq_dev"] == pytest.approx(0.5)


def test_a_variant_departing_from_a_terminal_is_no_longer_special(batch, qrl):
    """`qrl/cf_hub_missing` is GONE, and this is why: a variant carries its own prefix now, so
    a class whose departure state is a trajectory's terminal has an anchor like any other.
    State 3 is trajectory 0's terminal and used to make the whole example unmeasurable."""
    psi = torch.zeros(batch.n_states, D, requires_grad=True)
    goals = goals_for(batch)
    psi_goal = psi.index_select(0, goals.goal_state)
    goal_q = batch.traj_qid[goals.goal_traj]
    cf = make_cf(batch, states=[3, 3], kinds=[0, 1], values=[0.0, 1.0])
    viol, _, info = cf_terms(psi, batch, psi_goal, goal_q, DirectedDistance(), qrl, cf)
    assert info["qrl/cf_active"] == 1.0
    assert info["qrl/cf_pairs"] == 2.0
    assert float(viol) == pytest.approx(0.5 - qrl.cf_target)
    assert "qrl/cf_hub_missing" not in _CF_KEYS
    assert set(_CF_KEYS) <= set(info)


def test_a_positive_whose_example_lost_its_anchor_is_dropped_and_counted(batch, qrl):
    """`cf_encode.py` drops an example WHOLE or not at all, so this cannot happen -- and
    "cannot" is counted rather than assumed. Centring the orphan on whichever positive sorted
    first would make the constraint mean something else on a fraction of examples, with no
    curve to say so."""
    psi = torch.zeros(batch.n_states, D, requires_grad=True)
    goals = goals_for(batch)
    psi_goal = psi.index_select(0, goals.goal_state)
    goal_q = batch.traj_qid[goals.goal_traj]
    cf = make_cf(batch, states=[0, 0], kinds=[1, 1], values=[1.0, 2.0])   # no kind 0
    viol, _, info = cf_terms(psi, batch, psi_goal, goal_q, DirectedDistance(), qrl, cf)
    assert info["qrl/cf_positives"] == 2.0
    assert info["qrl/cf_anchor_missing"] == 2.0
    assert info["qrl/cf_active"] == 0.0
    assert float(viol) == 0.0            # exact zero, NOT -cf_target: no data is not evidence
    assert set(_CF_KEYS) <= set(info)


def test_cf_constraint_takes_both_directions(batch, qrl):
    """`(relu(d(anchor, v))^2 + relu(d(v, anchor))^2) / 2`. Under `DirectedDistance` a positive
    one unit ABOVE its anchor measures 0 forward and 1 backward, so a one-directional
    implementation would report 0.0 or 1.0 and this pins 0.5."""
    psi = torch.zeros(batch.n_states, D, requires_grad=True)
    goals = goals_for(batch)
    psi_goal = psi.index_select(0, goals.goal_state)
    goal_q = batch.traj_qid[goals.goal_traj]
    cf = make_cf(batch, states=[0, 0], kinds=[0, 1], values=[0.0, 1.0])
    viol, _, info = cf_terms(psi, batch, psi_goal, goal_q, DirectedDistance(), qrl, cf)
    assert info["qrl/cf_dist_fwd_mean"] == pytest.approx(0.0)     # d(anchor, v) = relu(0 - 1)
    assert info["qrl/cf_dist_bwd_mean"] == pytest.approx(1.0)     # d(v, anchor) = relu(1 - 0)
    assert info["qrl/cf_sq_dev"] == pytest.approx(0.5)
    assert float(viol) == pytest.approx(0.5 - qrl.cf_target)


def test_the_anchor_is_the_hub_and_is_never_its_own_pair(batch, qrl):
    """The anchor is measured against nothing -- `d(anchor, anchor)` is 0 by construction and
    would only dilute the mean. Under the old `phi` scheme that pair WAS the constraint's
    latent-dynamics job; here the anchor is a real encoded state, so the job is structural and
    the pair is gone. Adding a second positive must add exactly 2 pairs, not 4."""
    psi = torch.zeros(batch.n_states, D, requires_grad=True)
    goals = goals_for(batch)
    psi_goal = psi.index_select(0, goals.goal_state)
    goal_q = batch.traj_qid[goals.goal_traj]
    one = make_cf(batch, [0, 0], [0, 1], [0.0, 1.0])
    two = make_cf(batch, [0, 0, 0], [0, 1, 1], [0.0, 1.0, 1.0])
    _, _, info_one = cf_terms(psi, batch, psi_goal, goal_q, DirectedDistance(), qrl, one)
    _, _, info_two = cf_terms(psi, batch, psi_goal, goal_q, DirectedDistance(), qrl, two)
    assert info_one["qrl/cf_pairs"] == 2.0
    assert info_two["qrl/cf_pairs"] == 4.0
    assert info_two["qrl/cf_variants"] == 3.0
    assert info_two["qrl/cf_positives"] == 2.0


def test_cf_constraint_never_sees_negatives(batch, qrl):
    """NEGATIVES ARE NEVER IN THE CONSTRAINT. Two different WRONG rewrites of a step have no
    reason to be the same point, and asserting it is a claim this data cannot support -- the
    same rule `losses/counterfactual.py` states for its own negatives. Here the negative sits
    50 units from the anchor: if it entered, `cf_sq_dev` would be ~625 instead of 0.5.
    """
    psi = torch.zeros(batch.n_states, D, requires_grad=True)
    goals = goals_for(batch)
    psi_goal = psi.index_select(0, goals.goal_state)
    goal_q = batch.traj_qid[goals.goal_traj]

    without = make_cf(batch, [0, 0], [0, 1], [0.0, 1.0])
    withneg = make_cf(batch, [0, 0, 0], [0, 1, 2], [0.0, 1.0, 50.0])
    _, _, info_a = cf_terms(psi, batch, psi_goal, goal_q, DirectedDistance(), qrl, without)
    _, _, info_b = cf_terms(psi, batch, psi_goal, goal_q, DirectedDistance(), qrl, withneg)
    assert info_a["qrl/cf_sq_dev"] == pytest.approx(info_b["qrl/cf_sq_dev"])
    assert info_b["qrl/cf_pairs"] == info_a["qrl/cf_pairs"] == 2.0   # 1 positive x 2 directions
    assert info_b["qrl/cf_negatives"] == 1.0


def test_cf_star_pairs_are_anchor_to_member_only_not_the_pairwise_grid(batch, qrl):
    """|C| members give 2(|C|-1) pairs (each positive against the anchor, both ways) -- not
    |C|(|C|-1) member-to-member pairs. The class DIAMETER is bounded at 2*epsilon_cf by the
    triangle inequality instead, which is what makes the star affordable."""
    psi = torch.zeros(batch.n_states, D, requires_grad=True)
    goals = goals_for(batch)
    psi_goal = psi.index_select(0, goals.goal_state)
    goal_q = batch.traj_qid[goals.goal_traj]
    cf = make_cf(batch, [0, 0, 0, 0], [0, 1, 1, 1], [0.0, 1.0, 2.0, 3.0])
    _, _, info = cf_terms(psi, batch, psi_goal, goal_q, DirectedDistance(), qrl, cf)
    assert info["qrl/cf_pairs"] == 6.0                     # 3 positives x 2 directions
    assert info["qrl/cf_examples"] == 1.0
    # anchor at the origin, positives at 1..3: forward all 0, backward 1..3
    assert info["qrl/cf_sq_dev"] == pytest.approx((1 + 4 + 9) / 3 / 2)


def test_empty_cf_path_is_an_exact_zero_with_the_full_key_set(batch, qrl):
    psi = ladder_psi(batch)
    goals = goals_for(batch)
    lag = LagrangeMultipliers(qrl.init_lagrange)
    out = qrl_loss(psi, batch, goals, DirectedDistance(), qrl, lag, cf=None)
    assert float(out.terms["cf"]) == 0.0
    assert float(out.terms["neg_push"]) == 0.0
    assert set(_CF_KEYS) <= set(out.info)
    assert _empty_cf_info(qrl) == {k: 0.0 for k in _CF_KEYS}
    # and the graph is still connected, so `.backward()` cannot fail on a CF-free batch
    out.total.backward()
    assert torch.isfinite(psi.grad).all()


# =======================================================================================
# 4. the push term
# =======================================================================================


def test_push_is_monotone_decreasing_in_distance(batch, qrl):
    """The objective MAXIMIZES distance, so the transform must fall as distance grows."""
    goals = goals_for(batch)
    values = []
    for scale in (0.0, 1.0, 4.0):
        psi = torch.zeros(batch.n_states, D)
        psi[:, 0] = scale * torch.arange(batch.n_states, dtype=torch.float32)
        psi.requires_grad_(True)
        loss, _ = push_term(psi, psi.index_select(0, goals.goal_state), DirectedDistance(),
                            qrl, masks=push_masks(batch, goals))
        values.append(float(loss))
    assert values[0] > values[1] > values[2]


def test_push_matches_qrl_s_own_transform(batch, qrl):
    """`F.softplus(offset - d, beta).mean()` -- `global_push.py:46-47`, pair for pair."""
    goals = goals_for(batch)
    torch.manual_seed(0)
    psi = torch.randn(batch.n_states, D, requires_grad=True)
    dist = DirectedDistance()
    psi_goal = psi.index_select(0, goals.goal_state)
    loss, info = push_term(psi, psi_goal, dist, qrl, masks=push_masks(batch, goals))
    reference = torch.nn.functional.softplus(
        qrl.softplus_offset - dist(psi[:, None, :], psi_goal[None, :, :]),
        beta=qrl.softplus_beta,
    ).mean()
    assert float(loss) == pytest.approx(float(reference), rel=1e-6)
    assert info["qrl/push_pairs"] == batch.n_states * goals.n_goals


def test_push_chunking_is_exact(batch, qrl):
    """`push_chunk_cols` is the 16 GB fallback; it must not change the number. Accumulating a
    SUM and dividing once is what makes it exact -- a mean of chunk means would be wrong
    whenever the last chunk is short, which is almost always."""
    goals = goals_for(batch)
    torch.manual_seed(1)
    psi = torch.randn(batch.n_states, D, requires_grad=True)
    psi_goal = psi.index_select(0, goals.goal_state)
    masks = push_masks(batch, goals)
    full, info_full = push_term(psi, psi_goal, DirectedDistance(), qrl, masks=masks)
    for chunk in (1, 3, 7, 1000):
        chunked, info_chunked = push_term(
            psi, psi_goal, DirectedDistance(),
            QRLConfig(push_chunk_cols=chunk), masks=masks,
        )
        assert float(chunked) == pytest.approx(float(full), rel=1e-6)
        for key in info_full:
            assert info_chunked[key] == pytest.approx(info_full[key], rel=1e-6)


def test_cf_negatives_enter_the_push_as_sources_only(batch, qrl):
    """The eval-aligned direction: `d(psi(prefix + neg), psi_goal)`, broken state as the
    SOURCE. With a directed distance the two orderings differ by construction, so this is an
    equality against the right one and a failure against the wrong one."""
    torch.manual_seed(2)
    psi = torch.randn(batch.n_states, D, requires_grad=True)
    goals = goals_for(batch)
    psi_goal = psi.index_select(0, goals.goal_state)
    goal_q = batch.traj_qid[goals.goal_traj]
    dist = DirectedDistance()

    cf = make_cf(batch, [0, 0], [0, 2], [0.5, 3.0])     # one anchor, one negative
    _, neg_push, info = cf_terms(psi, batch, psi_goal, goal_q, dist, qrl, cf)

    psi_neg = cf[0][1:2]
    mask = goal_q == batch.traj_qid[batch.state_traj[0]]
    forward = dist(psi_neg[:, None, :], psi_goal[None, :, :])[:, mask]
    assert info["qrl/neg_push_dist_mean"] == pytest.approx(float(forward.mean()), rel=1e-6)
    assert float(neg_push) == pytest.approx(
        float(torch.nn.functional.softplus(
            qrl.softplus_offset - forward, beta=qrl.softplus_beta).mean()), rel=1e-6
    )
    reverse = dist(psi_goal[:, None, :], psi_neg[None, :, :])
    assert info["qrl/neg_push_dist_mean"] != pytest.approx(float(reverse.mean()), rel=1e-3)


def test_cf_negatives_are_scored_against_same_question_goals_only(batch, qrl):
    torch.manual_seed(3)
    psi = torch.randn(batch.n_states, D, requires_grad=True)
    goals = goals_for(batch)
    psi_goal = psi.index_select(0, goals.goal_state)
    goal_q = batch.traj_qid[goals.goal_traj]
    cf = make_cf(batch, [0], [2], [1.0])                # a q1 negative
    _, _, info = cf_terms(psi, batch, psi_goal, goal_q, DirectedDistance(), qrl, cf)
    assert info["qrl/neg_push_pairs"] == float(int((goal_q == 0).sum()))
    assert info["qrl/neg_push_pairs"] < goals.n_goals    # q2's columns were excluded


def test_push_splits_partition_every_pair(batch):
    goals = goals_for(batch)
    masks = push_masks(batch, goals)
    counts = sum(int(m.sum()) for m in masks.values())
    assert counts == batch.n_states * goals.n_goals
    assert not bool((masks["same_traj"] & masks["same_question"]).any())
    assert not bool((masks["same_question"] & masks["cross_question"]).any())


# =======================================================================================
# 5. the §18 initialisation helper
# =======================================================================================


def test_expected_init_values_against_the_actual_terms(random_reps, small_batch, qrl):
    """The two CONSTRAINT terms are exact (grad_mul is the identity forward), and the push
    terms satisfy their Jensen lower bound. Same checks `train.py` asserts at launch."""
    psi, _ = random_reps
    goals = goals_for(small_batch)
    lag = LagrangeMultipliers(qrl.init_lagrange)
    cf = make_cf(small_batch, [0, 0, 0], [0, 1, 2], [0.3, 0.4, 2.0], dim=psi.shape[-1])
    out = qrl_loss(psi, small_batch, goals, Distance("full_mrn", 8), qrl, lag, cf=cf)

    expected = expected_init_values(qrl, lag, out.info)
    for name in ("local", "cf"):
        assert float(out.terms[name]) == pytest.approx(expected[name], abs=1e-6)
    for name in ("push", "neg_push"):
        assert float(out.terms[name]) >= expected[name] - 1e-6      # softplus is convex
    assert float(out.total) == pytest.approx(
        float(out.terms["push"])
        + qrl.cf_neg_push_weight * float(out.terms["neg_push"])
        + float(out.terms["local"])
        + float(out.terms["cf"]),
        rel=1e-6,
    )
    assert "dyn" not in out.terms and "dyn" not in expected


def test_expected_init_push_is_the_softplus_of_the_measured_mean(qrl):
    """Nothing in the helper is a constant: `push` is `softplus_beta(offset - push_dist_mean)`
    evaluated on the mean THIS batch measured (§18 -- an ASSUMED init value is how two
    regressions got through)."""
    lag = LagrangeMultipliers(qrl.init_lagrange)
    info = {k: 0.0 for k in _CF_KEYS}
    info.update({
        "qrl/push_dist_mean": 3.5,
        "qrl/local_violation": 0.25,
        "qrl/cf_violation": -0.01,
    })
    out = expected_init_values(qrl, lag, info)
    beta, offset = qrl.softplus_beta, qrl.softplus_offset
    assert out["push"] == pytest.approx(math.log1p(math.exp(beta * (offset - 3.5))) / beta)
    assert out["local"] == pytest.approx(float(lag.local.value) * 0.25)
    assert out["cf"] == pytest.approx(float(lag.cf.value) * -0.01)
    assert out["neg_push"] == 0.0            # no negative pairs this batch


# =======================================================================================
# 6. knob inertness -- an off switch must be an EXACT zero, with its keys still logged
# =======================================================================================


def test_there_is_no_dyn_term_and_no_phi_in_the_loss(batch, qrl):
    """The deletion, pinned. `dyn` existed to drag a PREDICTED next state onto the observed
    one; there is no prediction here, so there is no term and `qrl_loss` does not take a
    `phi` argument at all. A re-added weight would show up as an extra key in `terms`."""
    import inspect

    import qrl_prm.loss as loss_mod

    psi = ladder_psi(batch)
    goals = goals_for(batch)
    lag = LagrangeMultipliers(qrl.init_lagrange)
    out = qrl_loss(psi, batch, goals, DirectedDistance(), qrl, lag)
    assert set(out.terms) == {"push", "neg_push", "local", "cf"}
    assert float(out.total) == pytest.approx(
        float(out.terms["push"]) + float(out.terms["local"]), rel=1e-6
    )
    assert not any(k.startswith("qrl/dyn") for k in out.info)
    assert "phi" not in inspect.signature(qrl_loss).parameters
    assert not hasattr(loss_mod, "dynamics_term")
    assert not hasattr(loss_mod, "hub_of_state")
    assert "dyn_weight" not in QRLConfig().to_dict()


def test_cf_neg_push_weight_zero_is_an_exact_zero_and_still_logs(batch, qrl):
    psi = ladder_psi(batch)
    goals = goals_for(batch)
    lag = LagrangeMultipliers(qrl.init_lagrange)
    cf = make_cf(batch, [0, 0], [0, 2], [0.5, 3.0])
    on = qrl_loss(psi, batch, goals, DirectedDistance(), qrl, lag, cf=cf)
    off = qrl_loss(psi, batch, goals, DirectedDistance(),
                   QRLConfig(cf_neg_push_weight=0.0), lag, cf=cf)
    assert off.info["qrl/neg_push_dist_mean"] == pytest.approx(
        on.info["qrl/neg_push_dist_mean"]
    )                                                     # still measured and logged
    assert float(off.terms["neg_push"]) > 0.0
    assert float(off.total) == pytest.approx(float(on.total) - float(on.terms["neg_push"]),
                                             rel=1e-6)


# =======================================================================================
# 7. config + integration
# =======================================================================================


def test_shipped_yaml_matches_the_dataclass_defaults():
    """The yaml is documentation AND the default; a drift between them means the annotated
    reasoning in `qrl.yaml` describes a value nothing runs at."""
    from qrl_prm.config import QRL_YAML

    assert load_qrl_config(QRL_YAML) == QRLConfig()


def test_unknown_qrl_key_is_a_hard_error(tmp_path):
    path = tmp_path / "qrl.yaml"
    path.write_text("softplus_ofset: 25\n")               # typo
    with pytest.raises(Exception) as exc:
        load_qrl_config(path)
    assert "softplus_offset" in str(exc.value)            # closest legal name is suggested


def test_epsilon_cf_above_the_margin_ruler_is_refused():
    with pytest.raises(Exception, match="margin ruler"):
        QRLConfig(epsilon_cf=1.5)


def test_split_overrides_partitions_by_prefix():
    feynman, qrl_over = split_overrides(
        ["run.name=x", "qrl.cf_encode_max_tokens=8192", "distance.variant=iqe",
         "qrl.epsilon_cf=0.1"]
    )
    assert feynman == ["run.name=x", "distance.variant=iqe"]
    assert qrl_over == ["cf_encode_max_tokens=8192", "epsilon_cf=0.1"]


@pytest.mark.parametrize("variant", ["iqe", "full_mrn"])
def test_integration_one_step_on_the_real_distance(small_batch, random_reps, qrl, variant):
    """The decided head is IQE; `full_mrn` is the one-line control. Both must run on CPU,
    stay finite, and put a gradient on psi, the CF variants and both multipliers."""
    psi, _ = random_reps
    goals = goals_for(small_batch)
    lag = LagrangeMultipliers(qrl.init_lagrange)
    dist = Distance(variant, components=8)
    cf = make_cf(
        small_batch, [0, 0, 0, 5], [0, 1, 2, 0], [0.1, 0.2, 1.5, 0.3], dim=psi.shape[-1]
    )
    psi_v = cf[0]

    out = qrl_loss(psi, small_batch, goals, dist, qrl, lag, cf=cf)
    assert torch.isfinite(out.total)
    for name, term in out.terms.items():
        assert torch.isfinite(term), name
    assert all(math.isfinite(v) for v in out.info.values())

    out.total.backward()
    for tensor, name in ((psi, "psi"), (psi_v, "cf psi")):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all(), name
    for m, name in ((lag.local, "lambda_local"), (lag.cf, "lambda_cf")):
        assert m.raw.grad is not None and float(m.raw.grad) != 0.0, name
    if variant == "iqe":
        assert dist.alpha_raw.grad is not None      # the learned alpha trains


def test_alpha_raw_is_inside_the_saved_head_prefixes():
    """`utils/checkpoint.py::HEAD_PREFIXES` includes `"distance."`, so IQE's learned alpha is
    checkpointed. If it were dropped, phase 2 and eval would read a DIFFERENT metric than
    phase 1 trained (alpha back at sigmoid(0) = 0.5) and nothing downstream would say so.
    `train.py` asserts this at launch; this is the unit-level pin."""
    from feynman_prm.utils.checkpoint import HEAD_PREFIXES

    module = nn.Module()
    module.distance = Distance("iqe", components=8)
    names = [n for n in module.state_dict() if n.startswith(HEAD_PREFIXES)]
    assert "distance.alpha_raw" in names


# =======================================================================================
# 8. training-loop mechanics -- model-free, the way the rest of the suite tests them
# =======================================================================================


def test_the_dual_optimizer_steps_once_per_grad_accum_boundary(batch, qrl):
    """The multipliers are updated at the SAME boundary as the primal, on the gradient
    accumulated over the same `grad_accum` micro-batches. A dual step per MICRO-batch would
    take twice as many steps at the same lr as the primal -- the two would be riding different
    schedules and the constraint curve would not mean what it says.
    """
    grad_accum = 2
    lag = LagrangeMultipliers(qrl.init_lagrange)
    dual = torch.optim.AdamW(lag.parameters(), lr=qrl.lagrange_lr, betas=(0.9, 0.999),
                             weight_decay=0.0)
    psi = ladder_psi(batch, gap=3.0)          # violating, so the multipliers have work to do
    goals = goals_for(batch)

    seen = []
    for micro in range(4):
        out = qrl_loss(psi, batch, goals, DirectedDistance(), qrl, lag)
        (out.total / grad_accum).backward()
        if (micro + 1) % grad_accum == 0:
            dual.step()
            dual.zero_grad(set_to_none=True)
        seen.append(float(lag.local.value))
    # `seen[i]` is read AFTER micro-batch i, so the two boundaries fall at i = 1 and i = 3:
    # the value is unchanged across (0 -> 1 is the first step) and holds between steps.
    assert seen[0] != seen[1]                 # boundary at micro 1
    assert seen[1] == seen[2]                 # micro 2 accumulates, does not step
    assert seen[2] != seen[3]                 # boundary at micro 3
    assert seen[3] > seen[1] > seen[0]        # violating => lambda_local rises, monotonically


def test_the_accumulated_dual_gradient_is_the_sum_of_the_micro_batches(batch, qrl):
    lag = LagrangeMultipliers(qrl.init_lagrange)
    psi = ladder_psi(batch, gap=3.0)
    goals = goals_for(batch)

    one = qrl_loss(psi, batch, goals, DirectedDistance(), qrl, lag)
    (one.total / 2).backward()
    single = float(lag.local.raw.grad)
    two = qrl_loss(psi, batch, goals, DirectedDistance(), qrl, lag)
    (two.total / 2).backward()
    assert float(lag.local.raw.grad) == pytest.approx(2 * single, rel=1e-6)


def test_the_multipliers_are_not_model_parameters(cfg):
    """`model/backbone.py::param_groups` sweeps every trainable non-LoRA parameter into the
    "heads" group at lr_heads on the COSINE schedule. A multiplier registered on the model
    would join the primal optimiser, decay to a standstill exactly when the constraints start
    binding, and be stepped without the sign flip. They live outside the model on purpose --
    which is also why `assert_qrl_phase1_trainable` need not know about them."""
    import qrl_prm.train as qtrain

    src = (REPO_ROOT / "qrl_prm" / "train.py").read_text()
    assert "LagrangeMultipliers(qrl.init_lagrange).to(device)" in src
    assert "dual_optimizer = torch.optim.AdamW(" in src
    assert "lagrange.parameters()" in src
    # the dual optimiser is built from the multipliers ALONE, never from model.parameters()
    dual_block = src.split("dual_optimizer = torch.optim.AdamW(")[1].split(")")[0]
    assert "model" not in dual_block
    assert qtrain.INIT_TOLERANCE > 0


def test_stray_losses_override_is_refused_not_ignored():
    """Old bug B4's exact shape: `config/default.yaml`'s `losses:` block is read by NOTHING in
    `qrl_prm/`, so `--set losses.lambda_cf=2` would run to completion and change nothing."""
    import qrl_prm.train as qtrain

    with pytest.raises(SystemExit) as exc:
        qtrain.main(["--set", "losses.lambda_cf=2"])
    assert "silently INERT" in str(exc.value)
    assert "LEARNED Lagrange multipliers" in str(exc.value)
    assert "cf_encode_max_tokens" in str(exc.value)   # the legal `qrl.*` names are offered


def test_the_checkpoint_carries_the_multipliers_and_the_knobs(tmp_path, cfg, qrl):
    """The multipliers are not model parameters, so `save_checkpoint` cannot pick them up --
    they ride in the payload `extra`, raw (pre-softplus) so the reload is exact."""
    from feynman_prm.utils.checkpoint import HEAD_PREFIXES
    from qrl_prm.config import load_qrl_config_from_checkpoint
    from qrl_prm.train import save_qrl_checkpoint

    model = nn.Module()
    model.psi = nn.Linear(4, 4)
    model.distance = Distance("iqe", components=8)
    lag = LagrangeMultipliers(qrl.init_lagrange)
    with torch.no_grad():
        lag.local.raw.add_(1.25)

    path = save_qrl_checkpoint(tmp_path / "step10", model, cfg, qrl, lag, step=10)
    payload = torch.load(path / "heads.pt", map_location="cpu", weights_only=False)
    assert payload["step"] == 10
    assert payload["qrl_lagrange"]["local"] == pytest.approx(float(lag.local.raw), rel=1e-6)
    assert payload["qrl_lagrange"]["cf"] == pytest.approx(float(lag.cf.raw), rel=1e-6)
    # phi is frozen and untrained under qrl_prm/, and the payload says so rather than leaving
    # a random head sitting byte-for-byte where a trained one lives
    assert payload["qrl_phi_untrained"] is True
    # IQE's learned alpha rides the DEFAULT head prefixes -- §14's LoRA trap 3
    assert "distance.alpha_raw" in payload["heads"]
    assert HEAD_PREFIXES[-1] == "distance."
    assert load_qrl_config_from_checkpoint(path) == qrl


def test_comparability_probes_produce_the_same_keys_every_baseline_logged(small_batch, cfg):
    """Diagnostic #14's three-way Delta histogram is "the single best predictor of
    ProcessBench F1" (§7.6.6) and is only worth having if the QRL row's numbers mean exactly
    what `abl_cf_only`'s do. `comparability_probes` calls `build_matrices` and `batch_probes`
    VERBATIM rather than re-deriving Delta -- a second definition would drift from the one
    every other run logged, which is the trap `matrix.step_deltas` was extracted to close."""
    from feynman_prm.model.wrapper import Reps
    from qrl_prm.train import comparability_probes

    torch.manual_seed(5)
    latent = cfg.heads.latent_dim
    psi = torch.randn(small_batch.n_states, latent)
    phi = torch.randn(small_batch.n_rows, latent)   # present on Reps, and never read
    model = nn.Module()
    model.distance = Distance(cfg.distance.variant, cfg.distance.components)
    reps = Reps(h_states=psi, psi=psi, phi=phi, act_emb=phi)
    goals = goals_for(small_batch)

    out = comparability_probes(reps, small_batch, goals, model, cfg)
    for key in (
        "probe14/delta_good_of_correct/mean",
        "probe14/delta_boundary/mean",
        "probe14/delta_good_of_correct/frac_above_natural",
        "probe02/delta_good_mean",
        "probe03/gap",
        "probe09_4/irreversibility_mean",
    ):
        assert key in out, key
    # and it must not touch the graph: no QRL term reads Dist, Next or D_term
    assert all(not torch.is_tensor(v) or not v.requires_grad for v in out.values())

    # `psi[row_dst]` is passed where `build_matrices` takes `phi` -- s' = s ++ a, read rather
    # than predicted -- so `reps.phi` must not reach the panel at all. Scrambling it changes
    # nothing; scrambling psi changes everything.
    scrambled = comparability_probes(
        Reps(h_states=psi, psi=psi, phi=phi * 100.0, act_emb=phi),
        small_batch, goals, model, cfg,
    )
    assert set(scrambled) == set(out)
    for key, value in out.items():
        assert scrambled[key] == value or (math.isnan(value) and math.isnan(scrambled[key])), key


# =======================================================================================
# 9. the §18 launch check, and the micro-batch path train.py actually runs
# =======================================================================================


def test_check_init_values_passes_on_a_real_micro_batch(small_batch, random_reps, qrl):
    from qrl_prm.train import check_init_values

    psi, _ = random_reps
    goals = goals_for(small_batch)
    lag = LagrangeMultipliers(qrl.init_lagrange)
    out = qrl_loss(psi, small_batch, goals, Distance("iqe", 8), qrl, lag)
    check_init_values(qrl, out.terms, out.info, expected_init_values(qrl, lag, out.info))


def test_check_init_values_catches_a_multiplier_that_skipped_softplus_inv(qrl):
    """The failure the equality check exists for: storing `init_lagrange` RAW starts the
    multiplier at softplus(0.01) = 0.698, 70x its intended value. Nothing downstream would
    say so -- the constraint would simply be weighted 70x too hard from step one."""
    from qrl_prm.train import check_init_values

    good = LagrangeMultipliers(qrl.init_lagrange)
    terms = {"push": torch.tensor(25.0), "neg_push": torch.tensor(0.0),
             "local": good.local.value.detach() * 0.5, "cf": torch.tensor(0.0)}
    info = {k: 0.0 for k in _CF_KEYS}
    info.update({"qrl/push_dist_mean": 30.0, "qrl/local_violation": 0.5,
                 "qrl/cf_violation": 0.0, "qrl/push_saturated_frac": 0.1})
    check_init_values(qrl, terms, info, expected_init_values(qrl, good, info))

    bad = LagrangeMultipliers(qrl.init_lagrange)
    with torch.no_grad():
        bad.local.raw.fill_(qrl.init_lagrange)          # raw, not softplus_inv(raw)
    with pytest.raises(AssertionError, match="softplus_inv"):
        check_init_values(qrl, terms, info, expected_init_values(qrl, bad, info))


def test_check_init_values_catches_a_saturated_push(qrl):
    """A `softplus_offset` below the untrained distance scale leaves the push term flat, so
    the objective is two constraints and nothing to maximise. Caught at launch, not at hour
    three."""
    from qrl_prm.train import check_init_values

    lag = LagrangeMultipliers(qrl.init_lagrange)
    info = {k: 0.0 for k in _CF_KEYS}
    info.update({"qrl/push_dist_mean": 400.0, "qrl/local_violation": 0.0,
                 "qrl/cf_violation": 0.0, "qrl/push_saturated_frac": 1.0})
    expected = expected_init_values(qrl, lag, info)
    terms = {k: torch.tensor(float(v)) for k, v in expected.items() if k != "total"}
    with pytest.raises(AssertionError, match="softplus_offset"):
        check_init_values(qrl, terms, info, expected)


class StubModel(nn.Module):
    """`FeynmanPRM`'s interface without the 1.5B backbone: `__call__ -> Reps`,
    `.hidden_states`, `.psi`, `.head_dtype`, `.distance`, `.pad_id`. `run_micro_batch` is a
    pure function of those, so this exercises the EXACT path `train.py` runs -- collate,
    goals, CF attach, BOTH forwards, loss -- with no download and no GPU.

    Note what it must provide that the old stub did not: `hidden_states` and `psi`, because
    `encode_cf_psi` runs a real second forward over the variant SEQUENCES. `cf_phi` is gone
    with `phi`.
    """

    def __init__(self, latent: int, variant: str = "iqe"):
        super().__init__()
        self.latent = latent
        self.distance = Distance(variant, components=8)
        self.pad_id = 0
        self.embed = nn.Embedding(64, latent)
        self.psi = nn.Identity()

    @property
    def head_dtype(self):
        return torch.float32

    def hidden_states(self, input_ids, attention_mask):
        emb = self.embed(input_ids % 64)
        return emb, emb

    def forward(self, batch):
        from feynman_prm.model.wrapper import Reps

        h = self.embed(batch.state_flat_idx % 64)
        return Reps(h_states=h, psi=h, phi=self.embed(batch.row_src % 64), act_emb=h)


class StubCFContext:
    """`CFEncodeContext.attach`'s interface: an anchor, a positive and a negative, all of one
    example departing from state 0, as SEQUENCES with their own last-separator index."""

    def attach(self, batch, rng):
        from qrl_prm.cf_encode import EncodedCF, empty_encode_info

        long_ = lambda x: torch.as_tensor(x, dtype=torch.long)  # noqa: E731
        info = empty_encode_info()
        info.update({"cf/examples_attached": 1.0, "cf/examples_eligible": 1.0,
                     "cf/attach_rate": 1.0, "cf/variants": 3.0, "cf/encode_sequences": 3.0,
                     "cf/encode_real_tokens": 9.0, "cf/encode_padded_tokens": 9.0,
                     "cf/encode_max_len": 3.0})
        return EncodedCF(
            variant_state=long_([0, 0, 0]),
            variant_example=long_([0, 0, 0]),
            variant_kind=long_([0, 1, 2]),
            input_ids=long_([[7, 7, 1], [7, 8, 1], [9, 9, 1]]),
            attention_mask=long_([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            state_flat_idx=long_([2, 5, 8]),        # v * L + last state position
            info=info,
        )


def test_run_micro_batch_end_to_end_on_a_stub_model(cfg, qrl):
    """The whole per-step path, model-free: collate -> sample_goals -> attach CF -> BOTH
    forwards -> qrl_loss -> backward, plus the comparability probe panel and the §18 check. If
    the CF tuple's shape, the anchor derivation or the probe wiring is wrong, it fails here
    rather than twenty minutes into a GPU launch."""
    from qrl_prm.train import check_init_values, comparability_probes, run_micro_batch

    rows = [
        synthetic_row("q1", [T, T, T]),
        synthetic_row("q1", [T, T, F, F]),
        synthetic_row("q2", [T, T]),
        synthetic_row("q2", [F, F, F]),
    ]
    torch.manual_seed(6)
    model = StubModel(cfg.heads.latent_dim, cfg.distance.variant)
    lag = LagrangeMultipliers(qrl.init_lagrange)
    device = torch.device("cpu")

    batch, goals, reps, out = run_micro_batch(
        model, rows, list(range(len(rows))), cfg, qrl, lag, device,
        np.random.default_rng(0), cf_ctx=StubCFContext(),
    )
    assert torch.isfinite(out.total)
    assert out.info["cf/attach_rate"] == 1.0            # the attach info is merged in
    assert out.info["cf/encode_sequences"] == 3.0       # and so is the second forward's cost
    assert out.info["qrl/cf_active"] == 1.0             # and the constraint actually engaged
    assert out.info["qrl/cf_negatives"] == 1.0
    assert out.info["qrl/neg_push_pairs"] > 0.0

    check_init_values(qrl, out.terms, out.info, expected_init_values(qrl, lag, out.info))
    (out.total / cfg.train.grad_accum).backward()
    assert any(p.grad is not None for p in model.parameters())
    assert lag.local.raw.grad is not None and lag.cf.raw.grad is not None

    metrics = comparability_probes(reps, batch, goals, model, cfg)
    assert "probe14/delta_boundary/mean" in metrics


def test_the_cf_key_set_is_logged_even_when_nothing_attaches(cfg, qrl):
    """House rule (`losses/counterfactual.py::_empty_info`): a diagnostic that disappears on a
    degenerate batch cannot be plotted. `cf_ctx.attach` returning None is a NORMAL outcome --
    no eligible prefix, or the token budget dropped everything -- and it must still log."""
    from qrl_prm.train import run_micro_batch

    class NothingAttaches:
        def attach(self, batch, rng):
            return None

    rows = [synthetic_row("q1", [T, T, T]), synthetic_row("q1", [T, T, F, F])]
    torch.manual_seed(7)
    model = StubModel(cfg.heads.latent_dim, cfg.distance.variant)
    lag = LagrangeMultipliers(qrl.init_lagrange)
    for ctx in (None, NothingAttaches()):
        _, _, _, out = run_micro_batch(
            model, rows, [0, 1], cfg, qrl, lag, torch.device("cpu"),
            np.random.default_rng(0), cf_ctx=ctx,
        )
        assert set(CF_ENCODE_KEYS) <= set(out.info)
        assert set(_CF_KEYS) <= set(out.info)
        assert out.info["cf/encode_sequences"] == 0.0
        assert float(out.terms["cf"]) == 0.0


# =======================================================================================
# 10. the REAL counterfactual corpus, through the real attach path, into the QRL loss
# =======================================================================================

CF_GLOB = "data/cf_train/cf70k.jsonl,data/cf_train/cf70k_gm.jsonl,data/cf_train/cf70k_oai.jsonl"
_cf_files = [REPO_ROOT / p for p in CF_GLOB.replace(",", " ").split()]
needs_cf_corpus = pytest.mark.skipif(
    not all(p.exists() for p in _cf_files),
    reason="the CF snapshot is not on this machine (it ships to the GPU box only)",
)


@pytest.fixture(scope="module")
def cf_examples():
    from feynman_prm.data.counterfactual import read_cf_glob

    return read_cf_glob(",".join(str(p) for p in _cf_files))


@needs_cf_corpus
def test_the_whole_corpus_loads_and_is_well_formed(cf_examples):
    """Every record parses, `step_index` is in range, and no rewrite list is empty --
    `CounterfactualExample.__post_init__` raises on each of those, so a malformed line
    anywhere in the corpus aborts a launch AFTER the model is on the GPU. Checked here
    instead, on the whole file, in three seconds.

    The corpus grows between snapshots (data/cf_train/MANIFEST.md tracks it: 27,114 ->
    36,073 -> 41,380), so this is deliberately a SHAPE test and not a count test -- pinning
    the count would fail on every refresh for no defect.
    """
    assert len(cf_examples) > 40_000
    for ex in cf_examples:
        assert 0 <= ex.step_index < len(ex.steps)
        assert ex.positive_rewrites and ex.negative_rewrites
        assert ex.steps[ex.step_index].strip()
        assert all(p.strip() for p in ex.positive_rewrites)
        assert all(n.strip() for n in ex.negative_rewrites)


@needs_cf_corpus
def test_every_prefix_is_distinct_so_nothing_double_weights(cf_examples):
    """`build_cf_index` maps prefix hash -> example list, and `attach_cf` puts EVERY example
    at a matching hash into the eligible pool. Two examples sharing a prefix is legal and
    handled, but it means one state feeds two CF classes; a 1:1 index is the property the
    corpus has actually had at every snapshot, so a break here is a generator regression
    (a re-run written under a new anchor), not a code path to fix."""
    from feynman_prm.data.cf_attach import build_cf_index

    index = build_cf_index(cf_examples)
    assert len(index) == len(cf_examples)
    assert max(len(v) for v in index.values()) == 1


@needs_cf_corpus
def test_real_cf_examples_encode_and_flow_through_the_qrl_loss(cf_examples, cfg, qrl, tokenizer):
    """END TO END on the real corpus: real examples -> `CFEncodeContext.attach` -> the star
    constraint and the negative push.

    The batch is synthetic (the parquet ships to the GPU box, this does not), but the JOIN is
    real: the batch's `state_prefix_hash` is overwritten with hashes computed by
    `prefix_hash` from the examples' own question and prefix steps, exactly as
    `prepare_data.py` writes them. So this exercises `build_cf_index`, the hash join, the cap,
    `build_sequence` on real Math-Shepherd text and every shape `cf_terms` derives from the
    result -- including the anchor derivation across several examples in one batch.

    **State 3 is trajectory 0's TERMINAL and it is no longer special**: a variant carries its
    own prefix, so an example landing there is measured like any other. That is the drop path
    the old `phi` scheme had (`qrl/cf_hub_missing`) and this scheme does not.
    """
    from feynman_prm.data.prefix_hash import prefix_hash
    from feynman_prm.data.tokenize import sep_token_id

    rows = [
        synthetic_row("q1", [T, T, T]),        # states 0..3   (3 is the terminal)
        synthetic_row("q1", [T, T, F, F]),     # states 4..8
        synthetic_row("q2", [T, T]),           # states 9..11
        synthetic_row("q2", [F, F, F]),        # states 12..15
    ]
    batch = collate(rows, pad_id=0)

    picked = cf_examples[:6]
    states = [0, 1, 4, 5, 9, 3]
    hashes = batch.state_prefix_hash.clone()
    for state, ex in zip(states, picked):
        hashes[state] = prefix_hash(ex.question, ex.steps[: ex.step_index])
    batch.state_prefix_hash = hashes

    ctx = CFEncodeContext(
        picked, tokenizer, pad_id=0, max_examples=cfg.data.cf_max_per_batch,
        sep_id=sep_token_id(tokenizer, cfg.data.sep_token),
        prompt_format=cfg.data.prompt_format, max_len=4096, max_tokens=10**9,
    )
    enc = ctx.attach(batch, np.random.default_rng(0))
    assert enc is not None
    assert enc.info["cf/examples_attached"] == float(len(picked))
    assert enc.n_variants == sum(
        1 + len(e.positive_rewrites) + len(e.negative_rewrites) for e in picked
    )
    # every variant's flat index lands on the SEPARATOR that follows its own step -- s' is the
    # prefix with the step appended, and nothing else
    flat = enc.input_ids.reshape(-1)
    sep_id = sep_token_id(tokenizer, cfg.data.sep_token)
    assert bool((flat.index_select(0, enc.state_flat_idx) == sep_id).all())

    torch.manual_seed(7)
    latent = 64
    psi = torch.randn(batch.n_states, latent, requires_grad=True)
    psi_v = torch.randn(enc.n_variants, latent, requires_grad=True)
    goals = goals_for(batch)
    lag = LagrangeMultipliers(qrl.init_lagrange)
    cf = (psi_v, enc.variant_state, enc.variant_example, enc.variant_kind)

    out = qrl_loss(psi, batch, goals, Distance("iqe", 8), qrl, lag, cf=cf)
    assert torch.isfinite(out.total)
    assert all(math.isfinite(v) for v in out.info.values())
    assert out.info["qrl/cf_active"] == 1.0
    assert out.info["qrl/cf_anchor_missing"] == 0.0        # every example kept its anchor
    assert out.info["qrl/cf_examples"] == float(len(picked))   # INCLUDING the terminal one
    assert out.info["qrl/cf_positives"] == float(
        sum(len(e.positive_rewrites) for e in picked)
    )
    assert out.info["qrl/cf_negatives"] == float(sum(len(e.negative_rewrites) for e in picked))
    assert out.info["qrl/neg_push_pairs"] > 0.0

    out.total.backward()
    assert torch.isfinite(psi_v.grad).all()
    assert lag.cf.raw.grad is not None and float(lag.cf.raw.grad) != 0.0


@needs_cf_corpus
def test_the_encode_selection_is_attach_cf_s_selection_call_for_call(cf_examples, cfg, tokenizer):
    """`CFEncodeContext` replicates `attach_cf`'s prefix join, cap and seeded draw so that this
    run sees the same CF examples at the same rate as every baseline -- the objective is meant
    to be the only difference between the rows. The replication is the drift risk, so it is
    pinned against the original rather than argued.
    """
    from feynman_prm.data.cf_attach import CFContext
    from feynman_prm.data.prefix_hash import prefix_hash
    from feynman_prm.data.tokenize import sep_token_id

    rows = [synthetic_row("q1", [T] * 5) for _ in range(4)]
    batch = collate(rows, pad_id=0)
    over_cap = cfg.data.cf_max_per_batch + 8
    picked = cf_examples[:over_cap]
    hashes = batch.state_prefix_hash.clone()
    for state, ex in enumerate(picked):
        hashes[state % batch.n_states] = prefix_hash(ex.question, ex.steps[: ex.step_index])
    batch.state_prefix_hash = hashes

    baseline = CFContext(picked, tokenizer, pad_id=0, max_examples=cfg.data.cf_max_per_batch)
    ours = CFEncodeContext(
        picked, tokenizer, pad_id=0, max_examples=cfg.data.cf_max_per_batch,
        sep_id=sep_token_id(tokenizer, cfg.data.sep_token),
        prompt_format=cfg.data.prompt_format, max_len=4096, max_tokens=10**9,
    )
    attached = baseline.attach(batch, np.random.default_rng(1))
    enc = ours.attach(batch, np.random.default_rng(1))
    assert attached is not None and enc is not None
    # same cap, same eligible pool, same draw -> same variants in the same order
    assert enc.info["cf/examples_eligible"] == attached.info["cf/examples_eligible"]
    assert enc.info["cf/examples_attached"] == attached.info["cf/examples_attached"]
    assert enc.variant_state.tolist() == attached.variant_state.tolist()
    assert enc.variant_example.tolist() == attached.variant_example.tolist()
    assert enc.variant_kind.tolist() == attached.variant_kind.tolist()
    assert enc.info["cf/examples_attached"] <= float(cfg.data.cf_max_per_batch)


@needs_cf_corpus
def test_the_token_budget_drops_whole_examples_off_the_tail_and_counts_them(
    cf_examples, cfg, tokenizer
):
    """`sequences x longest` is what the GPU pays for, so that is what the budget counts.
    WHOLE examples go, never individual variants -- half a class is a different constraint,
    not a smaller one -- and they go off the TAIL of the seeded order, which is a uniform draw
    and therefore unbiased in length. Trimming the longest instead would train the constraint
    on short steps only and no curve would say so."""
    from feynman_prm.data.prefix_hash import prefix_hash
    from feynman_prm.data.tokenize import sep_token_id

    rows = [synthetic_row("q1", [T] * 5) for _ in range(4)]
    batch = collate(rows, pad_id=0)
    picked = cf_examples[:6]
    hashes = batch.state_prefix_hash.clone()
    for state, ex in enumerate(picked):
        hashes[state % batch.n_states] = prefix_hash(ex.question, ex.steps[: ex.step_index])
    batch.state_prefix_hash = hashes

    def build(max_tokens):
        ctx = CFEncodeContext(
            picked, tokenizer, pad_id=0, max_examples=cfg.data.cf_max_per_batch,
            sep_id=sep_token_id(tokenizer, cfg.data.sep_token),
            prompt_format=cfg.data.prompt_format, max_len=4096, max_tokens=max_tokens,
        )
        return ctx.attach(batch, np.random.default_rng(2))

    full = build(10**9)
    assert full.info["cf/examples_dropped_budget"] == 0.0
    trimmed = build(int(full.info["cf/encode_padded_tokens"]) // 2)
    assert trimmed is not None
    assert trimmed.info["cf/examples_attached"] < full.info["cf/examples_attached"]
    assert trimmed.info["cf/examples_dropped_budget"] == (
        full.info["cf/examples_attached"] - trimmed.info["cf/examples_attached"]
    )
    # the survivors are a PREFIX of the full selection: the tail went, nothing was reordered
    n = trimmed.n_variants
    assert trimmed.variant_kind.tolist() == full.variant_kind[:n].tolist()
    assert trimmed.variant_example.tolist() == full.variant_example[:n].tolist()
    # and every surviving example kept ALL of its variants
    for slot in trimmed.variant_example.unique().tolist():
        assert int((trimmed.variant_example == slot).sum()) == int(
            (full.variant_example == slot).sum()
        )
    # the empty-info key set is the one a populated attach logs
    assert set(empty_encode_info()) == set(CF_ENCODE_KEYS)
    assert set(trimmed.info) == set(CF_ENCODE_KEYS)


def test_the_push_probe_picks_a_different_batch_than_the_length_probe():
    """`longest_batch_index` maximises n_sequences x max_length -- the BACKBONE cost. QRL's
    (S, C) push matrix ignores sequence length entirely, so the batch that maximises IT is a
    different one, and probing only the first under-measures the peak by exactly the tensor
    this objective adds (measured 2026-08-25: the length probe said 12.15 GB, the allocator
    then hit 43 MB free on a shorter batch a few steps in).

    Batch A is few LONG sequences; batch B is many SHORT ones. Each probe must pick its own.
    """
    from feynman_prm.data.sampler import longest_batch_index
    from qrl_prm.train import largest_push_batch_index

    long_rows = [synthetic_row("q1", [T, T], step_len=200) for _ in range(3)]
    short_rows = [synthetic_row("q2", [T] * 8, step_len=1) for _ in range(20)]
    rows = long_rows + short_rows
    batches = [list(range(len(long_rows))),
               list(range(len(long_rows), len(rows)))]

    assert longest_batch_index(batches, rows) == 0        # few long sequences
    assert largest_push_batch_index(batches, rows) == 1    # many short ones -> more states


def test_a_trainable_phi_is_refused_at_launch(cfg):
    """φ was REMOVED from this objective (2026-08-25), and the guard is what keeps it removed.

    `feynman_prm/model/backbone.py::assert_phase1_trainable` requires φ to be trainable, because
    every Feynman phase-1 term reads it. `qrl_prm/` requires the OPPOSITE: the arrived state is
    `psi(prefix + step)`, read rather than predicted, so nothing here ever calls `model.phi`. A
    trainable φ would sit in the optimiser's "heads" group at lr_heads collecting exactly zero
    gradient while `launch/model` reported 2,364,416 parameters as trained -- true of the
    baselines in experiments.md §4, and a silent lie here. Refuse it, do not merely freeze it.
    """
    from qrl_prm.train import assert_qrl_phase1_trainable

    class Fake(nn.Module):
        def __init__(self, phi_trainable: bool):
            super().__init__()
            self.base = nn.Module()
            self.base.lora_A = nn.Parameter(torch.zeros(2))
            self.psi = nn.Linear(2, 2)
            self.phi = nn.Linear(2, 2)
            self.phi.requires_grad_(phi_trainable)

    ok = assert_qrl_phase1_trainable(Fake(phi_trainable=False), cfg)
    assert ok["phi"] == 0                    # the value launch/model prints
    assert ok["psi"] and ok["lora"]

    with pytest.raises(AssertionError, match="phi IS trainable"):
        assert_qrl_phase1_trainable(Fake(phi_trainable=True), cfg)


def test_phi_is_exactly_the_gap_between_this_run_and_the_baselines_param_count():
    """`launch/model` reports `trainable_params: 20042753` where experiments.md §4's baselines
    report 22,407,168. The difference must be φ ALONE (plus IQE's one α scalar) -- if anything
    else went missing with it, this is the arithmetic that catches it.

    Measured on `config/default.yaml`: latent_dim 512, hidden_dims (512, 512, 512).
    """
    import yaml
    from feynman_prm.model.heads import StateActionRepresentation

    h = yaml.safe_load((REPO_ROOT / "config" / "default.yaml").read_text())["heads"]
    phi = StateActionRepresentation(
        hidden_size=1536,
        action_dim=h.get("action_dim", 1536),
        hidden_dims=tuple(h["hidden_dims"]),
        latent_dim=h["latent_dim"],
    )
    phi_params = sum(p.numel() for p in phi.parameters())
    assert phi_params == 2_364_416
    assert sum(1 for _ in phi.parameters()) == 14      # the `phi: 14` the baselines printed
    assert 22_407_168 - phi_params + 1 == 20_042_753   # +1 = IQE's learned alpha
