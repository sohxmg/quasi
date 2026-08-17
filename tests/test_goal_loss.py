"""§15's L_goal tests, including the collapse test that documents why §7.7 is a separate
phase."""

from __future__ import annotations

import pytest
import torch

from feynman_prm.losses.goal import goal_loss, terminal_separability, terminal_spread_ratio
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


def test_gate_ratio_null_is_one_and_0p3_is_not_the_separability_boundary():
    """Why the gate is read on `gate/auc` and not on `gate/ratio` (§10.1).

    Two facts, both at the gate's real shape (200 questions, ~2.7 terminals each, D=512):

      1. the ratio's null is **1.000**, not something diffuse -- so the ratio measures
         "is there any question structure at all", which is the only thing it measures well;
      2. a ratio of **0.63** is already PERFECT same-question retrieval, so §10.1's "< 0.3"
         rejects representations the goal head can use. That threshold has no §17 provenance
         entry; this test is what it should have been checked against.
    """
    dist = Distance("full_mrn", 8)
    torch.manual_seed(0)
    question = torch.arange(200).repeat_interleave(3)[:534]

    null = torch.randn(534, 512)
    assert terminal_spread_ratio(null, question, dist)["gate/ratio"] == pytest.approx(1.0, abs=0.02)
    null_sep = terminal_separability(null, question, dist)
    assert null_sep["gate/auc"] == pytest.approx(0.5, abs=0.02), "AUC chance is 0.5, scale-free"
    assert null_sep["gate/recall_at_1"] < 0.05

    centres = torch.randn(200, 512)
    loose = centres[question] + torch.randn(534, 512) * 0.8
    ratio = terminal_spread_ratio(loose, question, dist)["gate/ratio"]
    sep = terminal_separability(loose, question, dist)
    assert 0.55 < ratio < 0.70, "a ratio §10.1 would fail"
    assert sep["gate/auc"] > 0.99 and sep["gate/recall_at_1"] == 1.0, "...yet perfectly separable"


def test_epoch_log_is_a_mean_over_the_epoch_not_the_last_minibatch(cfg, tmp_path):
    """Phase 2 is 20 epochs, so 20 points ARE the curve. Logging the last minibatch's `info`
    made every point a single `batch_size` draw, which is far too noisy to read a trend off.

    Built so the last batch is deliberately unrepresentative: the epoch mean must land
    between the batches, never on the last one.
    """
    import dataclasses

    from feynman_prm.diagnostics.logging import RunLogger, read_metrics
    from feynman_prm.model.wrapper import FeynmanPRM
    from feynman_prm.train_goal_head import fit_goal_head

    torch.manual_seed(0)
    tiny = dataclasses.replace(
        cfg,
        heads=dataclasses.replace(cfg.heads, latent_dim=32, hidden_dims=(16,)),
        goal_head=dataclasses.replace(cfg.goal_head, epochs=2, batch_size=4),
    )
    model = FeynmanPRM(tiny, 32, backbone=None, with_goal_head=True)

    n_q, n_term = 3, 10                                  # 10 / 4 -> 3 batches, last one short
    cache = (
        torch.randn(n_q, 32),
        torch.randn(n_term, 32),
        torch.randint(0, n_q, (n_term,)),
        [f"q{i}" for i in range(n_q)],
    )
    logger = RunLogger(tmp_path, "phase2")
    fit_goal_head(model, cache, tiny, torch.device("cpu"), logger)
    logger.close()

    records = read_metrics(tmp_path / "phase2" / "metrics.jsonl")
    assert [r["goal/epoch"] for r in records] == [0, 1], "one record per epoch"
    assert [r["goal/optimizer_step"] for r in records] == [3, 6], "steps accumulate ACROSS epochs"
    for r in records:
        assert r["goal/batches"] == 3
        assert r["goal/terminals_seen"] == n_term, "the weighted mean covers every terminal"
        assert r["goal/seconds"] >= 0.0
        assert r["goal/loss"] == pytest.approx(
            r["goal/d_pred_to_target"] + r["goal/d_target_to_pred"], abs=1e-4
        ), "the averaged fields stay mutually consistent"

    events = (tmp_path / "phase2" / "events.jsonl").read_text()
    assert '"optimizer_steps": 6' in events, "the step count is announced before the fit"


def test_progress_ticks_on_time_and_never_reads_broken(capsys):
    """`build_cache` is tens of minutes of backbone forwards and used to print NOTHING, which
    is indistinguishable from a hang. The line ticks on elapsed time, so it stays readable at
    any loop speed -- and it clamps, so a caller with a slightly wrong `total` gets a pinned
    bar rather than "101.1%" and a negative ETA."""
    from feynman_prm.diagnostics.logging import Progress

    p = Progress("cache/test", 100, every_seconds=1e9)     # never tick on time
    p.advance(30)
    assert "30/100" not in capsys.readouterr().out, "an interim tick inside the time budget"

    p.advance(70)                                          # reaching total always prints
    line = capsys.readouterr().out
    assert "100/100 (100.0%)" in line and "eta" in line

    p.advance(50)                                          # overshoot: clamped, never negative
    over = capsys.readouterr().out
    assert "(100.0%)" in over and "-" not in over.split("eta")[1]


def test_gate_subsample_keeps_the_pairwise_matrix_affordable():
    """`terminal_spread_ratio` materialises the FULL N x N x D matrix. On the phase-2 cache
    (every correct terminal of every selected question, N ~ 95,000) that is 8,621 GiB and it
    OOMed a 16 GiB card. §10.1 sizes the gate as an ESTIMATE, so it runs on a subsample.
    """
    from feynman_prm.train_goal_head import (
        GATE_MAX_QUESTIONS,
        GATE_MAX_TERMINALS_PER_QUESTION,
        gate_subsample,
    )

    torch.manual_seed(0)
    # 900 questions x 6 terminals, plus 40 singletons that can form no within-question pair.
    q = torch.arange(900).repeat_interleave(6)
    q = torch.cat([q, torch.arange(900, 940)])
    psi = torch.randn(len(q), 8)

    sub_psi, sub_q = gate_subsample(psi, q)
    kept = torch.unique(sub_q)
    assert len(kept) == GATE_MAX_QUESTIONS
    assert len(sub_psi) == GATE_MAX_QUESTIONS * GATE_MAX_TERMINALS_PER_QUESTION
    assert (kept < 900).all(), "singleton questions contribute no within-pair and are skipped"
    counts = torch.bincount(sub_q)[kept]
    assert (counts == GATE_MAX_TERMINALS_PER_QUESTION).all()
    # deterministic: the gate must be comparable across runs and against goal_gate.py
    again_psi, again_q = gate_subsample(psi, q)
    torch.testing.assert_close(sub_psi, again_psi)
    torch.testing.assert_close(sub_q, again_q)

    # and the rows really are the cached vectors, not copies of the wrong index
    for row, label in zip(sub_psi, sub_q):
        assert any(torch.equal(row, psi[i]) for i in (q == label).nonzero().flatten())


def test_gate_subsample_passes_small_caches_through():
    """Below the caps nothing is dropped, so a short debug run gates on everything it has."""
    from feynman_prm.train_goal_head import gate_subsample

    q = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    psi = torch.randn(7, 8)
    sub_psi, sub_q = gate_subsample(psi, q)
    assert len(sub_psi) == 7 and torch.equal(sub_q, q)


def test_empty_cache_names_the_cause(cfg, tmp_path):
    """An empty cache used to reach the epoch log with `info` unassigned and die on an
    UnboundLocalError far from the actual problem."""
    import dataclasses

    from feynman_prm.diagnostics.logging import RunLogger
    from feynman_prm.model.wrapper import FeynmanPRM
    from feynman_prm.train_goal_head import fit_goal_head

    tiny = dataclasses.replace(
        cfg, heads=dataclasses.replace(cfg.heads, latent_dim=32, hidden_dims=(16,))
    )
    model = FeynmanPRM(tiny, 32, backbone=None, with_goal_head=True)
    cache = (torch.randn(2, 32), torch.zeros(0, 32), torch.zeros(0, dtype=torch.long), [])
    with pytest.raises(RuntimeError, match="no cached terminals"):
        fit_goal_head(model, cache, tiny, torch.device("cpu"), RunLogger(tmp_path, "empty"))


def test_pred_variance_is_logged_for_probe_6():
    """Diagnostic #6: near-zero variance means the head learned a constant, i.e. a global
    anchor rather than a question-conditioned goal."""
    dist = Distance("full_mrn", 8)
    constant = torch.zeros(4, 512)
    _, info = goal_loss(constant, torch.randn(4, 512), torch.arange(4), dist)
    assert info["goal/pred_variance"] == 0.0


def test_phase2_val_split_is_by_question_not_by_terminal(cfg, tmp_path):
    """Added 2026-08-04 with `goal_head.val_questions`.

    The whole point of the held-out set is to say whether raising `epochs` is learning or
    memorising, and it can only say that if the split is the same KIND of generalisation eval
    asks for. Holding out one of a question's terminals while training on its siblings leaks
    `h_s0` -- the head sees that exact question's input during training and the val number
    becomes optimistic for a reason no eval will reproduce.

    So: no question may contribute terminals to both sides, and the two sides must partition
    the cache exactly (a dropped terminal would silently shrink training).
    """
    import dataclasses

    from feynman_prm.diagnostics.logging import RunLogger
    from feynman_prm.model.wrapper import FeynmanPRM
    from feynman_prm.train_goal_head import fit_goal_head

    tiny = dataclasses.replace(
        cfg,
        heads=dataclasses.replace(cfg.heads, latent_dim=32, hidden_dims=(16,)),
        goal_head=dataclasses.replace(cfg.goal_head, epochs=2, batch_size=16, val_questions=7),
    )
    n_q, n_t = 20, 120
    model = FeynmanPRM(tiny, 32, backbone=None, with_goal_head=True)
    terminal_question = torch.randint(0, n_q, (n_t,))
    cache = (torch.randn(n_q, 32), torch.randn(n_t, 32), terminal_question, list(range(n_q)))

    events: list[tuple[str, dict]] = []
    logger = RunLogger(tmp_path, "split")
    logger.event = lambda name, payload: events.append((name, payload))  # type: ignore[method-assign]
    fit_goal_head(model, cache, tiny, torch.device("cpu"), logger)

    sched = next(p for n, p in events if n == "phase2/schedule")
    assert sched["terminals_train"] + sched["terminals_val"] == n_t, "the split lost terminals"
    assert sched["terminals_val"] > 0, "fixture must actually hold something out"

    # Reproduce the split the same way the fitter does and assert the question sets are
    # disjoint. This is the assertion that fails if anyone ever "simplifies" it to a
    # randperm over terminals.
    g = torch.Generator(device="cpu").manual_seed(tiny.run.seed)
    val_q = set(torch.randperm(n_q, generator=g)[: tiny.goal_head.val_questions].tolist())
    val_rows = torch.tensor([int(q) in val_q for q in terminal_question.tolist()])
    assert int(val_rows.sum()) == sched["terminals_val"]
    train_qs = set(terminal_question[~val_rows].tolist())
    assert train_qs.isdisjoint(val_q), "a question contributed terminals to both sides"
