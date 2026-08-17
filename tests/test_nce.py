"""§15's (1) L_NCE tests (§7.2)."""

from __future__ import annotations

import math

import pytest
import torch

from feynman_prm.losses.nce import nce_loss


def test_softmax_is_over_sources_within_a_column_not_goals_within_a_row():
    """TMD's BACKWARD NCE: `tmd.py:96-98` applies the cross-entropy to `logits.T`, so it
    normalises over SOURCES per goal column. A deliberately asymmetric fixture distinguishes
    the two directions -- getting this backwards is silent."""
    # 3 rows, 2 columns. Column 0's positive is row 0; column 1's positive is row 0 too.
    # Row 0 is close to both goals; row 2 is close to goal 1 only.
    Dist = torch.tensor([[0.1, 0.1], [5.0, 5.0], [5.0, 0.0]])
    pos_row = torch.tensor([0, 0])

    loss, info = nce_loss(Dist, pos_row, temperature=1.0)
    # over sources per column: column 1's softmax has row 2 (d=0.0) beating row 0 (d=0.1),
    # so the loss is noticeably worse than a row-normalised reading would give.
    expected = -(
        torch.log_softmax(-Dist[:, 0], dim=0)[0] + torch.log_softmax(-Dist[:, 1], dim=0)[0]
    ) / 2
    assert torch.allclose(loss, expected)
    row_wise = -torch.log_softmax(-Dist[0, :], dim=0)[0]
    assert not torch.allclose(loss, row_wise)


def test_chance_is_log_R():
    """§18: at init L_NCE sits at log(R) ~= 5.85 at R = 348. Pinned there WITH logit_std ~= 0
    is bug B10a (the fp32 cast is not effective); pinned there with logit_std > 0 and
    pos ~= neg is geometry collapse (§14's table)."""
    R, C = 40, 12
    Dist = torch.full((R, C), 2.0)
    pos_row = torch.arange(C)
    loss, info = nce_loss(Dist, pos_row)
    assert math.isclose(float(loss), math.log(R), rel_tol=1e-6)
    assert math.isclose(info["nce/chance"], math.log(R), rel_tol=1e-9)
    assert info["nce/logit_std"] == 0.0


def test_loss_drops_and_argmax_recovers_the_positive_on_a_separable_toy():
    torch.manual_seed(0)
    R, C, D = 24, 8, 32
    goals = torch.randn(C, D)
    phi = torch.randn(R, D) * 0.1
    pos_row = torch.arange(C)
    phi = phi.clone()
    phi[:C] = goals                                   # make the positives exactly reachable
    param = phi.clone().requires_grad_(True)

    from feynman_prm.model.distances import mrn_distance

    before = nce_loss(mrn_distance(param[:, None, :], goals[None, :, :]), pos_row)[0]
    opt = torch.optim.SGD([param], lr=0.5)
    for _ in range(50):
        loss, info = nce_loss(mrn_distance(param[:, None, :], goals[None, :, :]), pos_row)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < float(before)
    assert info["nce/categorical_accuracy_backward"] == 1.0


def test_same_trajectory_masking_never_masks_the_positive():
    """locked #12 keeps this OFF, but when enabled it must not remove a column's own
    positive -- that would make the loss undefined."""
    Dist = torch.tensor([[0.0, 3.0], [1.0, 1.0], [3.0, 0.0]])
    pos_row = torch.tensor([0, 2])
    row_traj = torch.tensor([0, 0, 1])
    goal_traj = torch.tensor([0, 1])
    loss, info = nce_loss(
        Dist, pos_row, mask_same_traj=True, row_traj=row_traj, goal_traj=goal_traj
    )
    assert torch.isfinite(loss)
    # column 0 belongs to trajectory 0, so row 1 (same trajectory, not the positive) is masked
    unmasked = nce_loss(Dist, pos_row)[0]
    assert float(loss) < float(unmasked)


def _traj_fixture():
    """One 6-step trajectory (rows phi_1..phi_6) plus a decoy trajectory.

    Column 0: goal at s_6, positive phi_3  -> nearer set {phi_4, phi_5, phi_6}
    Column 1: goal at s_6, positive phi_5  -> nearer set {phi_6}   (the DUPLICATE column)
    """
    R = 8
    row_traj = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
    row_step = torch.tensor([1, 2, 3, 4, 5, 6, 1, 2])
    goal_traj = torch.tensor([0, 0])
    goal_step = torch.tensor([6, 6])
    pos_row = torch.tensor([2, 4])                  # phi_3 (index 2), phi_5 (index 4)
    Dist = torch.arange(R, dtype=torch.float)[:, None].repeat(1, 2)
    return Dist, pos_row, row_traj, row_step, goal_traj, goal_step


def test_nearer_mask_drops_only_rows_between_the_positive_and_the_goal():
    """§9.9.2 / §16.4. Goal at s_6 with positive phi_3: phi_4, phi_5, phi_6 are all NEARER the
    goal than the positive and `F.cross_entropy` currently marks them wrong. Rows EARLIER than
    the positive must survive -- they are honest hard negatives and the only within-solution
    gradient there is. That is the whole difference from `mask_same_traj`."""
    Dist, pos_row, row_traj, row_step, goal_traj, goal_step = _traj_fixture()
    kw = dict(row_traj=row_traj, row_step=row_step, goal_traj=goal_traj, goal_step=goal_step)

    _, info = nce_loss(Dist, pos_row, mask_nearer_same_traj=True, **kw)
    # column 0 masks {3,4,5}, column 1 masks {5}  ->  mean 2.0 per column
    assert info["nce/negatives_masked"] == pytest.approx(2.0)
    assert info["nce/negatives_effective"] == pytest.approx(7.0 - 2.0)

    # the blunt switch drops the early rows too, which this one must not
    _, blunt = nce_loss(Dist, pos_row, mask_same_traj=True, **kw)
    assert blunt["nce/negatives_masked"] > info["nce/negatives_masked"]


def test_nearer_mask_dissolves_the_duplicate_column_contradiction():
    """29.6% of goal columns are byte-identical to another with a DIFFERENT positive
    (`probe01/distinct_goal_ratio` = 0.704, §9.9.3). Both columns of the fixture have goal s_6.
    Unmasked, column 0 asserts phi_3 > phi_5 while column 1 asserts phi_5 > phi_3 -- the two
    gradients oppose inside one backward pass. Masked, only column 1 speaks about the pair, and
    it says what L_T says."""
    Dist, pos_row, row_traj, row_step, goal_traj, goal_step = _traj_fixture()
    kw = dict(row_traj=row_traj, row_step=row_step, goal_traj=goal_traj, goal_step=goal_step)

    excluded = nce_loss(Dist, pos_row, mask_nearer_same_traj=True, **kw)[0]
    logits = -Dist
    # column 0 no longer ranks phi_5 at all; column 1 still ranks phi_3
    assert torch.isfinite(excluded)
    d = Dist.clone().requires_grad_(True)
    loss = nce_loss(d, pos_row, mask_nearer_same_traj=True, **kw)[0]
    loss.backward()
    assert d.grad[4, 0] == 0.0, "column 0 must not push phi_5 anywhere"
    assert d.grad[2, 1] != 0.0, "column 1 must still rank phi_3 against its positive"
    del logits


def test_nearer_mask_never_masks_the_positive_and_is_a_noop_at_offset_one():
    """`goal_step == pos_step` (the sampler drew offset 1, 50% of draws at discount 0.5) has an
    empty nearer set, so the mask must be inert there and must never remove a positive."""
    Dist = torch.tensor([[0.0, 3.0], [1.0, 1.0], [3.0, 0.0]])
    pos_row = torch.tensor([0, 2])
    kw = dict(
        row_traj=torch.tensor([0, 0, 1]),
        row_step=torch.tensor([1, 2, 1]),
        goal_traj=torch.tensor([0, 1]),
        goal_step=torch.tensor([1, 1]),           # offset 1 on both columns
    )
    masked, info = nce_loss(Dist, pos_row, mask_nearer_same_traj=True, **kw)
    assert info["nce/negatives_masked"] == 0.0
    assert torch.allclose(masked, nce_loss(Dist, pos_row)[0])


def test_sibling_late_mask_is_narrower_than_the_blunt_same_question_correct_one():
    """§16.25(a): only a sibling CORRECT trajectory's LATE states, and only against TERMINAL
    goal columns. Its early states stay -- they are legitimately far and L_T prices them."""
    R = 6
    row_traj = torch.tensor([0, 0, 0, 1, 1, 1])       # traj 0 owns the goal, traj 1 is a sibling
    row_step = torch.tensor([1, 2, 3, 1, 2, 3])
    row_correct = torch.tensor([True] * 6)
    row_steps_to_end = torch.tensor([2, 1, 0, 2, 1, 0])
    SQ = torch.ones(R, 2, dtype=torch.bool)           # one question
    goal_traj = torch.tensor([0, 0])
    goal_is_terminal = torch.tensor([True, False])    # column 1 is NOT a terminal
    Dist = torch.arange(R, dtype=torch.float)[:, None].repeat(1, 2)
    pos_row = torch.tensor([0, 1])

    _, late = nce_loss(
        Dist, pos_row, mask_sibling_correct_late=True, sibling_late_margin=1,
        row_traj=row_traj, goal_traj=goal_traj, row_correct=row_correct,
        row_steps_to_end=row_steps_to_end, goal_is_terminal=goal_is_terminal, SQ=SQ,
    )
    # terminal column masks sibling rows 4 and 5 (steps_to_end <= 1); non-terminal masks none
    assert late["nce/negatives_masked"] == pytest.approx(1.0)

    _, blunt = nce_loss(
        Dist, pos_row, mask_same_question_correct=True,
        row_correct=row_correct, SQ=SQ,
    )
    assert blunt["nce/negatives_masked"] > late["nce/negatives_masked"]


def test_argmax_in_nearer_set_is_reported_with_every_mask_off():
    """The §9.9.2 pre-flight. It has to be readable on the CURRENT default config, so it is
    computed from the raw logits regardless of which masks are enabled."""
    Dist, pos_row, row_traj, row_step, goal_traj, goal_step = _traj_fixture()
    # row 0 is nearest everywhere, so both columns' argmax is row 0 -- NOT in the nearer set
    _, info = nce_loss(
        Dist, pos_row, row_traj=row_traj, row_step=row_step,
        goal_traj=goal_traj, goal_step=goal_step,
    )
    assert info["nce/negatives_masked"] == 0.0, "diagnostic must not enable any mask"
    assert info["nce/argmax_in_nearer_set"] == 0.0
    assert info["nce/columns_with_nearer"] == 1.0
    assert info["nce/nearer_set_size"] == pytest.approx(2.0)

    # now make phi_6 (a nearer-set row) the closest for both columns
    D2 = Dist.clone()
    D2[5] = -5.0
    _, hit = nce_loss(
        D2, pos_row, row_traj=row_traj, row_step=row_step,
        goal_traj=goal_traj, goal_step=goal_step,
    )
    assert hit["nce/argmax_in_nearer_set"] == 1.0


def test_temperature_is_a_float_knob():
    """tmd.py:92 uses 1/sqrt(512) = 22.6; we use 1.0 (§7.2, bug B10a). Raising tau shrinks
    the logit spread, which is exactly the near-uniform-softmax signature."""
    torch.manual_seed(0)
    Dist = torch.rand(20, 6) * 10
    pos_row = torch.arange(6)
    _, hot = nce_loss(Dist, pos_row, temperature=1.0)
    _, cold = nce_loss(Dist, pos_row, temperature=22.6)
    assert cold["nce/logit_std"] < hot["nce/logit_std"] / 10


def _split_fixture(n_questions=11, rows_per_question=32, sep=12.0, seed=0):
    """A batch shaped like the real one (§8.1.1: R ~ 348, Q ~ 11-13), where cross-question
    separation is PERFECT and within-question ordering is pure noise."""
    torch.manual_seed(seed)
    R = n_questions * rows_per_question
    row_q = torch.arange(R) % n_questions
    C = n_questions                                   # one goal column per question
    goal_q = torch.arange(C)
    SQ = row_q[:, None] == goal_q[None, :]
    Dist = torch.rand(R, C) + torch.where(SQ, 0.0, sep)   # same-question rows are all close
    pos_row = torch.tensor([int((row_q == q).nonzero()[0]) for q in range(C)])
    return Dist, pos_row, SQ, rows_per_question - 1


def test_nce_loss_floors_at_log_of_the_same_question_pool():
    """WHY `nce/loss` stalls around 3.5 and more data will not move it (§16.4).

    With cross-question negatives fully separated and within-question ordering at chance, the
    softmax mass is exactly the same-question pool, so the loss sits at log(1 + n_same) -- at
    the measured Q = 11 and R ~ 348 that is log(32) = 3.47. `nce/loss` alone cannot tell this
    apart from undertraining; the split can."""
    Dist, pos_row, SQ, n_same = _split_fixture()
    loss, info = nce_loss(Dist, pos_row, SQ=SQ)

    assert info["nce/negatives_same_question"] == pytest.approx(n_same, abs=0.01)
    assert float(loss) == pytest.approx(math.log(1 + n_same), abs=0.15)
    assert float(loss) == pytest.approx(info["nce/floor_same_question"], abs=0.15)

    # ...while cross-question discrimination is in fact solved, and the split says so.
    assert info["nce/loss_cross_question"] < 0.01
    assert info["nce/accuracy_within_question"] == pytest.approx(1 / (1 + n_same), abs=0.15)

    # and the level is set by Q, not by how well the model has learned: fewer questions in the
    # batch means a bigger same-question pool and a HIGHER floor, at identical geometry.
    wide, wide_pos, wide_SQ, wide_n = _split_fixture(n_questions=22, rows_per_question=16)
    wide_loss, wide_info = nce_loss(wide, wide_pos, SQ=wide_SQ)
    assert float(wide_loss) < float(loss)
    assert wide_info["nce/floor_same_question"] < info["nce/floor_same_question"]


def test_within_question_learning_shows_up_in_the_split_not_in_nce_loss():
    """The point of the split: real progress is within-question ranking, and it moves
    `accuracy_within_question` long before it moves `nce/loss` much."""
    Dist, pos_row, SQ, n_same = _split_fixture()
    _, chance = nce_loss(Dist, pos_row, SQ=SQ)

    # now order the same-question rows correctly: the positive is nearest within its question
    Dist = Dist.clone()
    Dist[pos_row, torch.arange(Dist.shape[1])] -= 6.0
    learned_loss, learned = nce_loss(Dist, pos_row, SQ=SQ)

    assert learned["nce/accuracy_within_question"] == 1.0
    assert chance["nce/accuracy_within_question"] < 0.2
    assert float(learned_loss) < 0.2, "only within-question ranking can get below the floor"
