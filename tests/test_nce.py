"""§15's (1) L_NCE tests (§7.2)."""

from __future__ import annotations

import math

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


def test_temperature_is_a_float_knob():
    """tmd.py:92 uses 1/sqrt(512) = 22.6; we use 1.0 (§7.2, bug B10a). Raising tau shrinks
    the logit spread, which is exactly the near-uniform-softmax signature."""
    torch.manual_seed(0)
    Dist = torch.rand(20, 6) * 10
    pos_row = torch.arange(6)
    _, hot = nce_loss(Dist, pos_row, temperature=1.0)
    _, cold = nce_loss(Dist, pos_row, temperature=22.6)
    assert cold["nce/logit_std"] < hot["nce/logit_std"] / 10
