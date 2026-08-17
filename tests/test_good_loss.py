"""§15's (6) L_good tests (§7.12, added 2026-07-28).

L_good is the mirror image of L_step and almost every property is reversed, which is why it
gets its own file rather than a few cases bolted onto test_step_loss.py:

    L_step   -log sigma(Delta_{z+1} - m)    m > 0    decreasing in Delta    ONE pair per traj
    L_good   f(Delta_i - c)                 c < 0    increasing in Delta    every good i

`f` is `losses.good_loss.form`: `relu`, `relu_squared` (SHIPPED since 2026-08-04) or
`softplus`. Anything asserting a LEVEL must therefore go through `good_penalty` or
`good_bounds` rather than a literal -- a hardcoded `relu` bound is how a correct run gets
aborted by its own launch guard (B11/B12).

**The sign of `c` is the test that matters most here.** `c = +0.693` trains good steps one
full step AWAY from the goal per step, converges perfectly cleanly, and shows up on no curve
in the run -- it would only appear as a dead F1 after the GPU hours were spent.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from feynman_prm.data.collate import collate, state_index_of
from feynman_prm.data.goals import correct_terminal_columns
from feynman_prm.losses.good import good_bounds, good_loss, good_penalty
from feynman_prm.losses.matrix import step_deltas
from conftest import synthetic_row

T, F = True, False


def _fixture(*specs, qid="q"):
    """specs are (labels, qid) or just labels; all share one question by default.

    **Two correct trajectories per question is the minimum that produces any L_good term at
    all**, because a trajectory's own terminal is excluded -- a one-correct fixture measures
    the incorrect prefix only, which is a different mask.
    """
    rows = []
    for spec in specs:
        labels, this_qid = spec if isinstance(spec, tuple) else (spec, qid)
        rows.append(synthetic_row(this_qid, list(labels)))
    batch = collate(rows, pad_id=0)
    terminal_states, terminal_traj = correct_terminal_columns(batch)
    return batch, terminal_traj, terminal_states


def _dterm(batch, terminal_traj, fill=None, grad=False):
    """(S, T_c) with the column count the batch actually has. Getting this wrong is an
    IndexError against the mask, not a silent miscount, but it is still noise in a test."""
    shape = (batch.n_states, int(terminal_traj.numel()))
    out = torch.full(shape, float(fill)) if fill is not None else torch.randn(*shape)
    return out.requires_grad_(grad)


def _ramp(batch, terminal_traj, per_step, grad=False):
    """(S, T_c) with `d[s] = s * per_step`. State indices are consecutive within a
    trajectory, so every Delta_i comes out at exactly `per_step` -- which is how a fixture
    puts the whole batch AT the target, or a fixed distance below it."""
    ramp = torch.arange(batch.n_states, dtype=torch.float32)[:, None] * per_step
    out = ramp.expand(batch.n_states, int(terminal_traj.numel())).contiguous()
    return out.requires_grad_(grad)


def _selected_rows(batch, terminal_traj, cfg):
    """The source rows L_good actually reads, as a set of (trajectory, i) pairs."""
    sd = step_deltas(_dterm(batch, terminal_traj), batch, terminal_traj)
    mask = sd.good(cfg.losses.good_loss.include_incorrect_prefix)
    rows = torch.nonzero(mask.any(dim=1), as_tuple=False).flatten()
    return {(int(batch.row_traj[r]), int(batch.row_step[r])) for r in rows}


def _variant(cfg, **kwargs):
    return dataclasses.replace(
        cfg,
        losses=dataclasses.replace(
            cfg.losses, good_loss=dataclasses.replace(cfg.losses.good_loss, **kwargs)
        ),
    )


# ======================================================================================
# the margin: sign, derivation, and the one knob it reads
# ======================================================================================


def test_margin_is_negative_and_moves_with_discount(cfg):
    """c = -margin_steps * (-log gamma). NEGATIVE, and it is the good step's TARGET -- the
    same -0.693 (3) L_T prices a step at, not a slack allowance around it. A test that
    hardcodes -0.693 would not catch a revert to gamma = 0.7, where c must become -0.357."""
    assert cfg.good_margin < 0.0, "c is NEGATIVE (§7.12). This is the whole term."
    assert math.isclose(cfg.good_margin, -cfg.neg_log_gamma)
    assert math.isclose(cfg.good_margin, -0.69315, rel_tol=1e-4)

    at_07 = dataclasses.replace(cfg, discount=0.7)
    assert math.isclose(at_07.good_margin, -0.35667, rel_tol=1e-4)
    assert at_07.good_margin < 0.0

    two_steps = _variant(cfg, margin_steps=2.0)
    assert math.isclose(two_steps.good_margin, -1.38629, rel_tol=1e-4)
    # ...and it is the exact negation of L_step's m at the same margin_steps, which is what
    # makes "one step of progress" mean the same thing in both terms.
    assert math.isclose(two_steps.good_margin, -cfg.step_margin, rel_tol=1e-9)


def test_config_rejects_a_pre_negated_margin(cfg):
    """The negation lives in `Config.good_margin` and nowhere else. Writing -1.0 into the
    YAML would double-negate to c = +0.693 and train good steps away from the goal."""
    from feynman_prm.config import ConfigError, config_from_dict

    data = cfg.to_dict()
    data["losses"]["good_loss"]["margin_steps"] = -1.0
    with pytest.raises(ConfigError, match="must be > 0"):
        config_from_dict(data)

    data["losses"]["good_loss"]["margin_steps"] = 0.0
    with pytest.raises(ConfigError, match="must be > 0"):
        config_from_dict(data)


def test_the_wrong_sign_still_trains_and_that_is_why_this_file_exists(cfg):
    """Documents the failure mode: with c = +0.693 the loss is finite, differentiable and
    minimised by pushing Delta UP -- away from the goal, one step per step. Nothing in a loss
    curve distinguishes it from the correct term."""
    # One step each, so no state is both a source and a destination and the gradients do not
    # cancel the way they do on an interior state.
    batch, terminal_traj, _ = _fixture([T], [T])
    D_term = _dterm(batch, terminal_traj, fill=0.0, grad=True)

    loss, _ = good_loss(D_term, batch, terminal_traj, cfg)
    loss.backward()

    # Every Delta is 0 here, so every term is active and the loss is exactly f(0 - c) in the
    # configured form -- 0.69315 at `relu`, its square at `relu_squared`. Written through
    # `good_penalty` rather than as a literal so this stays a test of the SIGN, which is the
    # point of the file, and not a second place the shipped form has to be edited.
    expected = float(good_penalty(torch.tensor([-cfg.good_margin]), cfg.losses.good_loss.form))
    assert float(loss) == pytest.approx(expected, rel=1e-4)
    # Gradient on d(psi_i, g) is POSITIVE and on d(psi_{i-1}, g) NEGATIVE: descending this
    # term shrinks the distance across the step, i.e. makes it progress. Under c = +0.693 the
    # loss and its gradient have the IDENTICAL shape -- only the point where relu switches off
    # moves, from -0.693 to +0.693 -- which is exactly why no curve would show the flipped
    # sign, and why the assertion that matters is on `cfg.good_margin < 0`.
    assert float(D_term.grad[batch.row_dst].sum()) > 0.0
    assert float(D_term.grad[batch.row_src].sum()) < 0.0
    assert cfg.good_margin < 0.0


# ======================================================================================
# scope: which Delta_i the term is allowed to touch (§6.1)
# ======================================================================================


def test_scope_on_labels_TTFF_reads_delta_1_and_delta_2_only(cfg):
    """labels = [T, T, F, F] => z = 2. The good steps of that trajectory are i = 1, 2
    (Delta_1, Delta_2). i = 3 is Delta_{z+1}, (5) L_step's pair, and i = 4 is post-error --
    BOTH excluded (§7.12)."""
    batch, terminal_traj, _ = _fixture([T, T, T], [T, T, F, F])
    incorrect = 1
    assert int(batch.traj_z[incorrect]) == 2

    read = {i for traj, i in _selected_rows(batch, terminal_traj, cfg) if traj == incorrect}
    assert read == {1, 2}, "Delta_1 and Delta_2 only"

    # Confirmed through the gradient too, which is where a wrong index shows up. Note the
    # INTERIOR states cancel exactly here (psi_1 is Delta_1's destination and Delta_2's
    # source, +1 and -1 with both relus active), so the assertion is one-sided on purpose:
    # what must hold is that nothing at or past the boundary is touched.
    D_term = _dterm(batch, terminal_traj, fill=0.0, grad=True)
    good_loss(D_term, batch, terminal_traj, cfg)[0].backward()
    touched = {i for i in range(batch.n_states) if float(D_term.grad[i, 0].abs()) > 0}
    states = {
        i: state_index_of(batch, incorrect, i) for i in range(int(batch.traj_T[incorrect]) + 1)
    }
    assert touched <= {states[0], states[1], states[2]}
    assert states[3] not in touched, "i = z+1 is L_step's pair (§7.12)"
    assert states[4] not in touched, "i > z+1 is post-error; no loss defines Delta there"


def test_include_incorrect_prefix_false_drops_the_on_track_prefix(cfg):
    batch, terminal_traj, _ = _fixture([T, T, T], [T, T], [T, T, F, F])
    D_term = _dterm(batch, terminal_traj)

    both = good_loss(D_term, batch, terminal_traj, cfg)[1]["good/terms"]
    correct_only = good_loss(
        D_term, batch, terminal_traj, _variant(cfg, include_incorrect_prefix=False)
    )[1]["good/terms"]
    assert both > correct_only > 0

    off = _variant(cfg, include_incorrect_prefix=False)
    assert all(traj != 2 for traj, _ in _selected_rows(batch, terminal_traj, off))
    assert any(traj == 2 for traj, _ in _selected_rows(batch, terminal_traj, cfg))


def test_a_trajectorys_own_terminal_is_excluded(cfg):
    """A correct trajectory's distance to its OWN ending runs to d(x, x) = 0, so its last
    Delta is a spurious spike that belongs to no step. Same exclusion the probe uses."""
    batch, terminal_traj, _ = _fixture([T, T, T])           # one correct trajectory, alone
    D_term = _dterm(batch, terminal_traj, grad=True)
    loss, info = good_loss(D_term, batch, terminal_traj, cfg)
    assert info["good/terms"] == 0.0, "the only terminal available is the row's own"
    assert float(loss) == 0.0
    loss.backward()                                          # still differentiable

    # With a SECOND correct trajectory of the same question, the first one's rows get a
    # legitimate column and the terms appear.
    batch, terminal_traj, _ = _fixture([T, T, T], [T, T])
    D_term = _dterm(batch, terminal_traj)
    assert good_loss(D_term, batch, terminal_traj, cfg)[1]["good/terms"] > 0


def test_cross_question_columns_are_excluded(cfg):
    """Delta against another question's ending is not the eval query and L_T already owns
    the cross-question mass (§7.4.2)."""
    batch, terminal_traj, _ = _fixture(([T, T], "q1"), ([T, T], "q2"))
    D_term = _dterm(batch, terminal_traj)
    assert good_loss(D_term, batch, terminal_traj, cfg)[1]["good/terms"] == 0.0


def test_z_zero_is_finite_and_differentiable(cfg):
    """45.4% of incorrect trajectories have z = 0 (§4.2.1), and those contribute NO good
    steps at all -- every i is at or past the boundary. Must be finite, not an index error."""
    batch, terminal_traj, _ = _fixture([T, T], [T, T], [F, F])
    D_term = _dterm(batch, terminal_traj, grad=True)
    loss, info = good_loss(D_term, batch, terminal_traj, cfg)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(D_term.grad).all()
    # The z = 0 trajectory (index 2) gets no L_good gradient at all: it has no on-track
    # prefix, so every one of its Delta_i is at or past the boundary.
    z_zero = 2
    assert int(batch.traj_z[z_zero]) == 0
    for i in range(int(batch.traj_T[z_zero]) + 1):
        assert float(D_term.grad[state_index_of(batch, z_zero, i), 0]) == 0.0


def test_masked_not_crashed_when_no_goal_exists(cfg):
    """A question with no correct trajectory has no terminal column at all."""
    batch, terminal_traj, _ = _fixture([T, F], [F, F])
    D_term = _dterm(batch, terminal_traj, grad=True)
    assert D_term.shape[1] == 0, "no correct trajectory means no goal column at all"
    loss, info = good_loss(D_term, batch, terminal_traj, cfg)
    assert float(loss) == 0.0 and info["good/terms"] == 0.0
    loss.backward()                              # _empty keeps a zero in the graph
    assert set(info) == {
        "good/loss", "good/terms", "good/delta_mean", "good/delta_min", "good/delta_max",
        "good/margin", "good/above_target_fraction",
    }, "the empty path must log the SAME keys, or the panel goes ragged mid-run"


# ======================================================================================
# the form: relu / relu_squared vs softplus (§7.12's ablation)
# ======================================================================================

HINGES = ("relu", "relu_squared")        # the two that switch off at c. softplus does not.


@pytest.mark.parametrize("form", HINGES)
def test_the_hinges_are_exactly_zero_below_c_and_softplus_is_not(cfg, form):
    """`Delta >= c` is a hard floor implied by L_I + L_T, so pushing past it costs one of
    them. Both relu forms STOP at c; softplus applies half its gradient AT c and never
    reaches zero. That is the property softplus is rejected on, and it is why `relu_squared`
    is a change of TAIL PRICING and not a return to softplus' overshoot."""
    batch, terminal_traj, _ = _fixture([T, T, T], [T, T])
    c = cfg.good_margin

    # every Delta a full unit below c: both hinges are exactly 0, softplus is not
    below = _ramp(batch, terminal_traj, c - 1.0)
    hinge_loss = good_loss(below, batch, terminal_traj, _variant(cfg, form=form))[0]
    soft_loss = good_loss(below, batch, terminal_traj, _variant(cfg, form="softplus"))[0]
    assert float(hinge_loss) == 0.0
    assert float(soft_loss) > 0.0


@pytest.mark.parametrize("form", HINGES)
def test_hinge_gradient_vanishes_at_and_below_the_target(cfg, form):
    """The ablation's whole argument: softplus' gradient is 0.5 AT the target and never hits
    0, so it overshoots to Delta = -1.556 and breaks L_T's ruler. Both hinges stop --
    relu_squared's gradient is `2 * excess`, which is 0 at excess = 0 just as relu's is."""
    batch, terminal_traj, _ = _fixture([T, T, T], [T, T])
    # Delta_i = d_i - d_{i-1} = c exactly for every i
    at_target = _ramp(batch, terminal_traj, cfg.good_margin, grad=True)

    hinge_l = good_loss(at_target, batch, terminal_traj, _variant(cfg, form=form))[0]
    hinge_l.backward()
    assert float(at_target.grad.abs().max()) == 0.0, f"{form} applies NO pressure at target"

    at_target.grad = None
    soft_l = good_loss(at_target, batch, terminal_traj, _variant(cfg, form="softplus"))[0]
    soft_l.backward()
    assert float(at_target.grad.abs().max()) > 0.0, "softplus keeps pushing past the target"


def test_relu_squared_prices_the_tail_against_the_bulk(cfg):
    """**The reason the shipped form changed on 2026-08-04, as a property rather than a
    number.** `mean relu(Delta - c)` is a mean of a LINEAR hinge, so it is indifferent
    between one violator at 4x and four at 1x -- and mid-run that is what happened: the bulk
    fell to a mean of -0.412 while p99 climbed to 2.43 and delta_max hit 7.58 (§7.12).

    Under relu the two batches below cost the SAME; under relu_squared the concentrated one
    costs 4x more, because the gradient on a violator is proportional to how far out it is.
    """
    batch, terminal_traj, _ = _fixture([T, T, T, T, T], [T, T])
    c = cfg.good_margin
    n = int((step_deltas(_dterm(batch, terminal_traj), batch, terminal_traj)
             .good(True)).sum())
    assert n >= 4, "need enough good terms for the spread/concentrated split to differ"

    spread = _ramp(batch, terminal_traj, c + 1.0)            # every Delta 1.0 over target
    delta = torch.zeros(batch.n_states)
    delta[1:] = c                                            # everything at target...
    delta[batch.row_dst[0]] = c + float(n)                   # ...but one violator at n
    concentrated = torch.cumsum(delta, 0)[:, None].expand(
        batch.n_states, int(terminal_traj.numel())
    ).contiguous()

    relu_cfg, sq_cfg = _variant(cfg, form="relu"), _variant(cfg, form="relu_squared")
    r_spread = float(good_loss(spread, batch, terminal_traj, relu_cfg)[0])
    r_conc = float(good_loss(concentrated, batch, terminal_traj, relu_cfg)[0])
    s_spread = float(good_loss(spread, batch, terminal_traj, sq_cfg)[0])
    s_conc = float(good_loss(concentrated, batch, terminal_traj, sq_cfg)[0])

    # relu: same total excess -> same loss. The tail is invisible to it.
    assert r_conc == pytest.approx(r_spread, rel=1e-5)
    # relu_squared: the SAME total excess concentrated in one term costs n times more.
    assert s_conc == pytest.approx(s_spread * n, rel=1e-5)
    assert s_conc > s_spread


@pytest.mark.parametrize("form", ("relu", "relu_squared", "softplus"))
def test_matches_a_reference_implementation(cfg, form):
    batch, terminal_traj, _ = _fixture([T, T, T], [T, T], [T, T, F, F])
    torch.manual_seed(0)
    D_term = _dterm(batch, terminal_traj)
    variant = _variant(cfg, form=form)

    sd = step_deltas(D_term, batch, terminal_traj)
    selected = sd.delta[sd.good_correct | sd.good_incorrect]
    hinge = torch.clamp(selected - cfg.good_margin, min=0.0)
    reference = {
        "relu": hinge,
        "relu_squared": hinge ** 2,
        "softplus": torch.nn.functional.softplus(selected - cfg.good_margin),
    }[form].mean()

    loss, info = good_loss(D_term, batch, terminal_traj, variant)
    assert torch.allclose(loss, reference, atol=1e-7)
    # The DIAGNOSTICS are form-independent: they are statistics of Delta, not of the loss,
    # so probe14 and good/* read the same quantity whichever form is training.
    assert info["good/terms"] == float(selected.numel())
    assert info["good/delta_mean"] == pytest.approx(float(selected.mean()), abs=1e-7)
    assert info["good/above_target_fraction"] == pytest.approx(
        float((selected > cfg.good_margin).float().mean()), abs=1e-7
    )


@pytest.mark.parametrize("form", ("relu", "relu_squared", "softplus"))
def test_the_sandwich_runs_opposite_to_l_steps(cfg, form):
    """Every form is INCREASING in Delta, so f(delta_min - c) <= L_good <= f(delta_max - c).
    The lower bound is legitimately 0 whenever every good step already sits at or below
    target -- that is a converged term, not a dead one (§18).

    Asserted through `good_bounds`, which is what train.py's launch guard calls: a
    relu-shaped bound would abort a correct relu_squared run, and a guard that fires on
    healthy training is B11/B12 all over again."""
    batch, terminal_traj, _ = _fixture([T, T, T, T], [T, T], [T, T, F, F])
    variant = _variant(cfg, form=form)
    torch.manual_seed(1)
    for scale in (0.1, 1.0, 5.0):
        D_term = _dterm(batch, terminal_traj) * scale
        loss, info = good_loss(D_term, batch, terminal_traj, variant)
        lo, hi = good_bounds(info["good/delta_min"], info["good/delta_max"], variant)
        assert lo - 1e-6 <= float(loss) <= hi + 1e-6

    # ...and the degenerate end of that bound is reachable and correct for the two hinges.
    below = _ramp(batch, terminal_traj, cfg.good_margin - 1.0)
    loss, info = good_loss(below, batch, terminal_traj, variant)
    lo, _ = good_bounds(info["good/delta_min"], info["good/delta_max"], variant)
    if form in HINGES:
        assert float(loss) == 0.0
        assert lo == 0.0
    else:
        assert float(loss) > 0.0


def test_a_relu_bound_would_reject_a_correct_relu_squared_run(cfg):
    """Pins the reason `good_bounds` exists rather than a literal in train.py. With any
    violator more than one unit past c, relu_squared's loss is ABOVE the relu upper bound --
    so the pre-2026-08-04 launch assert would have killed this run at step 1."""
    batch, terminal_traj, _ = _fixture([T, T, T], [T, T])
    sq = _variant(cfg, form="relu_squared")
    far = _ramp(batch, terminal_traj, cfg.good_margin + 3.0)     # every Delta 3.0 over target

    loss, info = good_loss(far, batch, terminal_traj, sq)
    relu_hi = max(info["good/delta_max"] - cfg.good_margin, 0.0)
    assert float(loss) > relu_hi, "the old hardcoded bound is violated -- that is the point"

    lo, hi = good_bounds(info["good/delta_min"], info["good/delta_max"], sq)
    assert lo - 1e-6 <= float(loss) <= hi + 1e-6


# ======================================================================================
# the refactor: step_deltas is ONE definition, and the probe panel did not move
# ======================================================================================


def test_step_deltas_matches_the_pre_refactor_literals(cfg):
    """probes.py built these five tensors inline before 2026-07-28. Pinned against the
    literal expressions so the extraction cannot have changed the probe panel."""
    batch, terminal_traj, _ = _fixture([T, T, T], [T, T, F, F], [F, F, F])
    torch.manual_seed(0)
    D_term = torch.randn(batch.n_states, 1)

    row_q = batch.traj_qid[batch.row_traj]
    term_q = batch.traj_qid[terminal_traj]
    valid = (row_q[:, None] == term_q[None, :]) & (
        batch.row_traj[:, None] != terminal_traj[None, :]
    )
    delta = D_term[batch.row_dst] - D_term[batch.row_src]
    row_correct = batch.traj_correct[batch.row_traj]
    z = batch.traj_z[batch.row_traj]
    i = batch.row_step

    sd = step_deltas(D_term, batch, terminal_traj)
    assert torch.equal(sd.delta, delta)                       # bitwise, not allclose
    assert torch.equal(sd.valid, valid)
    assert torch.equal(sd.good_correct, valid & row_correct[:, None])
    assert torch.equal(sd.good_incorrect, valid & (~row_correct & (i <= z))[:, None])
    assert torch.equal(sd.boundary, valid & (~row_correct & (i == z + 1))[:, None])
    assert torch.equal(sd.post_error, valid & (~row_correct & (i > z + 1))[:, None])

    # The four masks partition `valid` -- no Delta counted twice, none dropped.
    union = sd.good_correct | sd.good_incorrect | sd.boundary | sd.post_error
    assert torch.equal(union, valid)
    total = sum(
        int(m.sum()) for m in (sd.good_correct, sd.good_incorrect, sd.boundary, sd.post_error)
    )
    assert total == int(valid.sum())


def test_step_deltas_detach_keeps_values_and_drops_the_graph(cfg):
    batch, terminal_traj, _ = _fixture([T, T, T], [T, T, F, F])
    D_term = torch.randn(batch.n_states, 1, requires_grad=True)
    sd = step_deltas(D_term, batch, terminal_traj)
    detached = sd.detach()
    assert sd.delta.requires_grad and not detached.delta.requires_grad
    assert torch.equal(sd.delta.detach(), detached.delta)
    assert torch.equal(sd.good_correct, detached.good_correct)


@pytest.fixture
def probe_panel(cfg, distance):
    """Two questions, TWO correct trajectories each plus an incorrect one. The `small_batch`
    fixture has only one correct per question, so `delta_good_of_correct` is empty there --
    every own-terminal column is excluded and the group that matters most has no members."""
    import numpy as np

    from feynman_prm.data.goals import sample_goals
    from feynman_prm.losses.matrix import build_matrices

    rows = [
        synthetic_row("q1", [T, T, T]),
        synthetic_row("q1", [T, T]),
        synthetic_row("q1", [T, T, F, F]),
        synthetic_row("q2", [T, T]),
        synthetic_row("q2", [T, T, T]),
        synthetic_row("q2", [T, F, F]),
    ]
    batch = collate(rows, pad_id=0)
    torch.manual_seed(0)
    psi = torch.randn(batch.n_states, 64, requires_grad=True)
    phi = torch.randn(batch.n_rows, 64, requires_grad=True)
    goals = sample_goals(batch, cfg.discount, np.random.default_rng(0))
    matrices = build_matrices(psi, phi, batch, goals, distance, cfg)
    return batch, goals, psi, phi, matrices


def test_probe_panel_is_unchanged_by_the_refactor(cfg, probe_panel, distance):
    """The probe values must be bit-identical to the inline computation they replaced --
    §7.12's refactor was allowed to add keys, never to move one."""
    from feynman_prm.diagnostics.probes import batch_probes

    batch, goals, psi, phi, matrices = probe_panel
    out = batch_probes(psi, phi, batch, goals, matrices, distance, cfg)

    D_term = matrices.D_term.detach()
    row_q = batch.traj_qid[batch.row_traj]
    term_q = batch.traj_qid[matrices.terminal_traj]
    valid = (row_q[:, None] == term_q[None, :]) & (
        batch.row_traj[:, None] != matrices.terminal_traj[None, :]
    )
    delta = D_term[batch.row_dst] - D_term[batch.row_src]
    row_correct = batch.traj_correct[batch.row_traj]
    z = batch.traj_z[batch.row_traj]
    i = batch.row_step

    for mask, name in (
        (valid & row_correct[:, None], "delta_good_of_correct"),
        (valid & (~row_correct & (i <= z))[:, None], "delta_good_of_incorrect"),
        (valid & (~row_correct & (i == z + 1))[:, None], "delta_boundary"),
        (valid & (~row_correct & (i > z + 1))[:, None], "delta_post_error"),
    ):
        expected = delta[mask]
        assert expected.numel() > 0, name
        assert out[f"probe14/{name}/n"] == float(expected.numel())
        assert out[f"probe14/{name}/mean"] == float(expected.mean())   # bitwise
        assert out[f"probe14/{name}/std"] == float(expected.std())
        assert out[f"probe14/{name}/positive_fraction"] == float((expected > 0).float().mean())

    good_all = delta[
        (valid & row_correct[:, None]) | (valid & (~row_correct & (i <= z))[:, None])
    ]
    bad_all = delta[valid & (~row_correct & (i == z + 1))[:, None]]
    assert out["probe02/delta_good_mean"] == float(good_all.mean())
    assert out["probe03/delta_bad_mean"] == float(bad_all.mean())
    assert out["probe03/gap"] == float(bad_all.mean() - good_all.mean())


def test_probe_logs_the_quantiles_that_decide_f1(cfg, probe_panel, distance):
    """"The mean hid this for a whole run" -- +0.240 looked like a bounded offset while a
    third of good steps sat above tau. frac_above_* and p90/p99 are the finer read (§7.12)."""
    from feynman_prm.diagnostics.probes import batch_probes
    from feynman_prm.eval.calibrate import natural_tau

    batch, goals, psi, phi, matrices = probe_panel
    out = batch_probes(psi, phi, batch, goals, matrices, distance, cfg)

    for group in ("correct", "incorrect"):
        for key in ("frac_above_0", "frac_above_natural", "frac_above_1", "frac_above_2",
                    "p90", "p99"):
            assert f"probe14/delta_good_of_{group}/{key}" in out
    assert out["probe14/natural_tau"] == pytest.approx(natural_tau(cfg))

    # frac_above is monotone in the threshold, and the thresholds are ordered 0 < natural < 1.
    fr = [out[f"probe14/delta_good_of_correct/frac_above_{k}"]
          for k in ("0", "natural", "1", "2")]
    assert all(fr[i] >= fr[i + 1] for i in range(3)), fr
    assert out["probe14/delta_good_of_correct/p90"] <= out["probe14/delta_good_of_correct/p99"]

    # The tail is not derivable from the mean, which is the entire lesson of §7.12: at step
    # 750 the mean read +0.240 and frac_above_natural read 0.34.
    assert 0.0 <= out["probe14/delta_good_of_correct/frac_above_natural"] <= 1.0


# ======================================================================================
# integration with the total (§7.0)
# ======================================================================================


def _phase1_at(cfg, small_batch, random_reps, distance, step=None):
    import numpy as np

    from feynman_prm.data.goals import sample_goals
    from feynman_prm.losses.matrix import build_matrices
    from feynman_prm.losses.total import phase1_loss

    psi, phi = random_reps
    goals = sample_goals(small_batch, cfg.discount, np.random.default_rng(0))
    matrices = build_matrices(psi, phi, small_batch, goals, distance, cfg)
    return phase1_loss(psi, phi, small_batch, matrices, distance, cfg,
                       goal_traj=goals.goal_traj, step=step)


def _phase1(cfg, small_batch, random_reps, distance):
    import numpy as np

    from feynman_prm.data.goals import sample_goals
    from feynman_prm.losses.matrix import build_matrices
    from feynman_prm.losses.total import phase1_loss

    psi, phi = random_reps
    goals = sample_goals(small_batch, cfg.discount, np.random.default_rng(0))
    matrices = build_matrices(psi, phi, small_batch, goals, distance, cfg)
    return phase1_loss(psi, phi, small_batch, matrices, distance, cfg,
                       goal_traj=goals.goal_traj)


def test_the_shipped_default_is_lambda_good_one(cfg):
    """§16.21, signed off 2026-07-28. Pinned so a revert to 0.0 is a test failure and not a
    silent one: at 0.0 everything still runs, every curve still looks healthy, and the only
    symptom is the F1 ceiling at the end of the run."""
    assert cfg.losses.lambda_good == 1.0
    # relu_squared since 2026-08-04 (§7.12's mid-run block: relu moved the bulk and lost the
    # tail). Pinned for the same reason lambda_good is -- the shipped form is a decision, and
    # a silent revert to `relu` would look identical in every curve except probe14's p99.
    # NEVER softplus: it applies gradient AT the target and stretches L_T's ruler.
    assert cfg.losses.good_loss.form == "relu_squared"
    assert cfg.good_margin < 0.0


def test_inert_at_lambda_zero(cfg, small_batch, random_reps, distance):
    """§7.10's discipline applied to §7.12: `--set losses.lambda_good=0.0` must reproduce the
    five-term loss set EXACTLY, so the escape hatch in §16.22 is a real one — that is the
    re-run that separates `L_good` from the `n_questions` change if the guards trip."""
    off = dataclasses.replace(cfg, losses=dataclasses.replace(cfg.losses, lambda_good=0.0))
    out = _phase1(off, small_batch, random_reps, distance)

    without = (
        off.losses.lambda_nce * out.terms["nce"]
        + off.losses.lambda_i * out.terms["invariance"]
        + off.losses.zeta * out.terms["backup"]
        + off.losses.lambda_cf * out.terms["cf"]
        + off.losses.lambda_step * out.terms["step"]
    )
    assert float(out.total) == float(without)          # bitwise: 0.0 * L_good is an exact zero
    # ...and the term is still COMPUTED and logged. Switching the training off must not switch
    # the diagnostic off, or the A/B re-run has nothing to compare.
    assert "good" in out.terms
    assert float(out.terms["good"]) > 0.0
    assert "good/above_target_fraction" in out.info


def test_lambda_good_enters_the_total(cfg, small_batch, random_reps, distance):
    off = dataclasses.replace(cfg, losses=dataclasses.replace(cfg.losses, lambda_good=0.0))
    off_out = _phase1(off, small_batch, random_reps, distance)
    on_out = _phase1(cfg, small_batch, random_reps, distance)       # the shipped 1.0

    assert float(on_out.total) == pytest.approx(
        float(off_out.total) + cfg.losses.lambda_good * float(off_out.terms["good"]), abs=1e-6
    )
    assert float(off_out.terms["good"]) > 0.0, "random latents are nowhere near the target"


def test_detach_goal_builds_a_second_matrix_and_spares_the_terminals(
    cfg, small_batch, random_reps, distance
):
    """§16.17 in reverse. With detach_goal the terminal states receive no L_good gradient,
    and (5) L_step keeps reading the ATTACHED D_term either way."""
    import numpy as np

    from feynman_prm.data.goals import sample_goals
    from feynman_prm.losses.matrix import build_matrices

    psi, phi = random_reps
    goals = sample_goals(small_batch, cfg.discount, np.random.default_rng(0))

    off = build_matrices(psi, phi, small_batch, goals, distance, cfg)
    assert off.D_term_good is off.D_term, "no second tensor when the flag is off"

    on_cfg = _variant(cfg, detach_goal=True)
    on = build_matrices(psi, phi, small_batch, goals, distance, on_cfg)
    assert on.D_term_good is not on.D_term
    assert torch.allclose(on.D_term_good, on.D_term)

    terminals = on.terminal_states
    psi.grad = None
    good_loss(on.D_term_good, small_batch, on.terminal_traj, on_cfg)[0].backward(
        retain_graph=True
    )
    detached_grad = psi.grad[terminals].abs().sum()

    psi.grad = None
    good_loss(on.D_term, small_batch, on.terminal_traj, on_cfg)[0].backward()
    attached_grad = psi.grad[terminals].abs().sum()

    assert float(attached_grad) > 0.0
    assert float(detached_grad) < float(attached_grad)


def test_good_and_step_pull_opposite_ends_of_the_same_trajectory(cfg):
    """The division of labour (§7.6.6, §7.12): L_step raises Delta at i = z+1, L_good lowers
    it everywhere else. They must never touch the same Delta or they fight."""
    from feynman_prm.losses.step import step_loss

    batch, terminal_traj, _ = _fixture([T, T, T], [T, T, F, F])
    D_term = torch.randn(batch.n_states, 1, requires_grad=True)

    good_loss(D_term, batch, terminal_traj, cfg)[0].backward(retain_graph=True)
    good_touched = {i for i in range(batch.n_states) if float(D_term.grad[i, 0].abs()) > 0}

    D_term.grad = None
    step_loss(D_term, batch, terminal_traj, cfg)[0].backward()
    step_touched = {i for i in range(batch.n_states) if float(D_term.grad[i, 0].abs()) > 0}

    incorrect, z = 1, 2
    boundary_states = {
        state_index_of(batch, incorrect, z),
        state_index_of(batch, incorrect, z + 1),
    }
    assert step_touched == boundary_states
    # psi_z is shared -- it is Delta_z's destination AND Delta_{z+1}'s source, which is
    # correct and is not a conflict; psi_{z+1} is L_step's alone.
    assert state_index_of(batch, incorrect, z + 1) not in good_touched


# ======================================================================================
# the warmup ramp (§7.12)
# ======================================================================================


def test_warmup_ramps_the_weight_and_never_the_term(cfg, small_batch, random_reps, distance):
    """relu applies FULL gradient the whole time it is violated -- it does not taper near its
    target the way every other term does. So L_good would hit at full strength from step 1,
    during the ~100 steps when L_I has not closed the psi/phi gap and L_T is still on the
    LINEX linear branch: the ruler `c` is expressed in does not exist yet.

    The ramp must scale the WEIGHT only. Every good/* diagnostic and the §18 sandwich read
    the unscaled term, at every point in the warmup."""
    from feynman_prm.losses.total import good_warmup_scale

    assert cfg.losses.good_loss.warmup_steps == 100
    assert good_warmup_scale(cfg, 0) == 0.0
    assert good_warmup_scale(cfg, 50) == 0.5
    assert good_warmup_scale(cfg, 100) == 1.0
    assert good_warmup_scale(cfg, 5000) == 1.0          # clamped, never above 1
    assert good_warmup_scale(cfg, None) == 1.0          # no schedule: tests and the init probe

    off = _variant(cfg, warmup_steps=0)
    assert good_warmup_scale(off, 0) == 1.0, "0 means full weight from step 1, not 'no L_good'"

    ramped = _phase1_at(cfg, small_batch, random_reps, distance, step=50)
    full = _phase1_at(cfg, small_batch, random_reps, distance, step=None)
    # the TERM is identical...
    assert float(ramped.terms["good"]) == float(full.terms["good"])
    for key in ("good/loss", "good/delta_mean", "good/delta_min", "good/delta_max",
                "good/margin", "good/above_target_fraction", "good/terms"):
        assert ramped.info[key] == full.info[key], key
    # ...only its contribution to the total is halved.
    assert ramped.info["good/lambda_effective"] == pytest.approx(0.5 * cfg.losses.lambda_good)
    assert float(full.total) - float(ramped.total) == pytest.approx(
        0.5 * cfg.losses.lambda_good * float(full.terms["good"]), abs=1e-6
    )


def test_at_step_zero_the_ramp_makes_l_good_exactly_inert(cfg, small_batch, random_reps,
                                                          distance):
    """Step 0 is the memory probe and the first micro-batch. L_good must contribute an exact
    zero there -- the ruler is at its least trustworthy and relu is at its most aggressive."""
    at_zero = _phase1_at(cfg, small_batch, random_reps, distance, step=0)
    inert = _phase1_at(
        dataclasses.replace(cfg, losses=dataclasses.replace(cfg.losses, lambda_good=0.0)),
        small_batch, random_reps, distance, step=None,
    )
    assert at_zero.info["good/lambda_effective"] == 0.0
    assert float(at_zero.total) == float(inert.total)
    assert float(at_zero.terms["good"]) > 0.0, "still computed and logged, just unweighted"
