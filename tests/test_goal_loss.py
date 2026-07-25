"""§15's L_goal tests, including the collapse test that documents why §7.7 is a separate
phase."""

from __future__ import annotations

import torch

from feynman_prm.losses.goal import goal_loss, terminal_spread_ratio
from feynman_prm.model.distances import Distance


def test_both_directions_are_summed():
    """The distance is one-way, so a single direction would let the guess drift to somewhere
    REACHABLE FROM the real ending rather than BEING it (§7.7, locked #14)."""
    dist = Distance("full_mrn", 8)
    torch.manual_seed(0)
    pred = torch.randn(2, 512)
    targets = torch.randn(3, 512)
    target_example = torch.tensor([0, 0, 1])
    loss, info = goal_loss(pred, targets, target_example, dist)
    assert torch.allclose(
        loss, torch.tensor(info["goal/d_pred_to_target"] + info["goal/d_target_to_pred"]),
        atol=1e-5,
    )


def test_mean_of_distances_never_a_distance_to_a_mean():
    """Root cause D: a latent-space centroid over 30k terminals collapses onto the population
    mean and the distance becomes an atypicality detector. There is no centroid anywhere in
    this codebase -- with targets far apart, the mean-of-distances is large where a
    distance-to-the-mean would be small."""
    dist = Distance("full_mrn", 8)
    pred = torch.zeros(1, 512)
    targets = torch.stack([torch.full((512,), 5.0), torch.full((512,), -5.0)])
    loss, _ = goal_loss(pred, targets, torch.tensor([0, 0]), dist)
    centroid_distance = dist(pred[0], targets.mean(dim=0))
    assert float(loss) > float(centroid_distance) * 10


def test_frozen_psi_cannot_collapse_the_targets():
    """With psi trainable and no .detach(), a toy optimisation drives all endings to one
    point. With psi FROZEN (phase 2) it cannot -- which is why the phase split exists."""
    dist = Distance("full_mrn", 8)
    torch.manual_seed(0)
    target_example = torch.tensor([0, 0, 1, 1])

    # joint: both the prediction and the "psi" outputs move
    pred_j = torch.randn(2, 64, requires_grad=True)
    psi_j = torch.randn(4, 64, requires_grad=True)
    opt = torch.optim.Adam([pred_j, psi_j], lr=0.1)
    for _ in range(200):
        loss, _ = goal_loss(pred_j, psi_j, target_example, dist)
        opt.zero_grad()
        loss.backward()
        opt.step()
    joint_spread = psi_j.std(dim=0).mean()

    # phase 2: psi is frozen, only the head moves
    pred_f = torch.randn(2, 64, requires_grad=True)
    psi_f = torch.randn(4, 64)
    before = psi_f.std(dim=0).mean().clone()
    opt = torch.optim.Adam([pred_f], lr=0.1)
    for _ in range(200):
        loss, _ = goal_loss(pred_f, psi_f, target_example, dist)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.equal(psi_f.std(dim=0).mean(), before), "frozen targets cannot move"
    assert float(joint_spread) < float(before), "joint training squashes them together"


def test_gate_ratio_separates_clustered_from_unclustered_terminals():
    """§10.1: ratio < 0.3 -> proceed; ratio -> 1 -> the goal head cannot work no matter how
    it is trained."""
    dist = Distance("full_mrn", 8)
    torch.manual_seed(0)
    question = torch.tensor([0, 0, 1, 1])

    centres = torch.randn(2, 512) * 10
    clustered = centres[question] + torch.randn(4, 512) * 0.01
    unclustered = torch.randn(4, 512) * 10

    assert terminal_spread_ratio(clustered, question, dist)["gate/ratio"] < 0.3
    assert terminal_spread_ratio(unclustered, question, dist)["gate/ratio"] > 0.5


def test_pred_variance_is_logged_for_probe_6():
    """Diagnostic #6: near-zero variance means the head learned a constant, i.e. a global
    anchor rather than a question-conditioned goal."""
    dist = Distance("full_mrn", 8)
    constant = torch.zeros(4, 512)
    _, info = goal_loss(constant, torch.randn(4, 512), torch.arange(4), dist)
    assert info["goal/pred_variance"] == 0.0
