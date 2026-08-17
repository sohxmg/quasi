"""(7) L_term tests (§7.13, §16.26). CPU, no model, `lambda_term = 0.0`.

`L_term` IS `counterfactual_loss` with `group = traj_qid` -- there is no second loss and no
`mode=` flag -- so §7.5.1's two REQUIREMENT properties are requirements here and are pinned
here as well as in `tests/test_counterfactual.py`. Duplicated deliberately: the properties are
about what the CALLER asks for, and a caller that grouped by trajectory instead of by question
would pass every test in that file.

The class here is "the correct endings of one question" and the negatives are "the incorrect
endings of the same question". Two solutions that are wrong in different ways end at different
wrong answers, so nothing may pull them together -- hence the bit-identical permutation test.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from feynman_prm.data.collate import collate
from feynman_prm.losses.counterfactual import counterfactual_loss
from feynman_prm.losses.terminal_class import (
    _CF_TO_TERM,
    terminal_class_chance,
    terminal_class_index,
    terminal_class_loss,
)
from feynman_prm.model.distances import Distance
from conftest import synthetic_row

T, F = True, False
D = 64


def _rows(spec):
    """`spec` is [(qid, n_correct, n_incorrect), ...] -- correct first, in that order."""
    rows = []
    for qid, k_c, k_i in spec:
        rows += [synthetic_row(qid, [T, T, T]) for _ in range(k_c)]
        rows += [synthetic_row(qid, [T, F, F]) for _ in range(k_i)]
    return rows


def _batch(spec):
    return collate(_rows(spec), pad_id=0)


def _psi(batch, terminals: dict[int, torch.Tensor]) -> torch.Tensor:
    """Random `psi` with the named trajectories' TERMINAL rows overwritten.

    Only `psi[batch.traj_terminal]` is read by the loss, so every other row is filler; giving
    it real noise rather than zeros keeps an accidental read from looking like a pass.
    """
    psi = torch.randn(batch.n_states, D)
    for traj, vector in terminals.items():
        psi[int(batch.traj_terminal[traj])] = vector
    return psi


@pytest.fixture
def dist():
    return Distance("full_mrn", 8)


# --------------------------------------------------------------- the grouping and the kinds

def test_grouping_is_by_question_not_by_trajectory(dist):
    """Two trajectories of DIFFERENT questions never share a class, however they interleave.

    Built as an interleaved batch -- q1, q2, q1, q2, ... -- because a grouping bug that keyed
    on batch position rather than on `traj_qid` would survive a block-ordered fixture.
    """
    rows = [
        synthetic_row("q1", [T, T, T]),
        synthetic_row("q2", [T, T]),
        synthetic_row("q1", [T, T, T, T]),
        synthetic_row("q2", [T, T, T]),
        synthetic_row("q1", [T, F, F]),
        synthetic_row("q2", [F, F]),
    ]
    batch = collate(rows, pad_id=0)
    group, kind, n_correct, n_incorrect = terminal_class_index(batch)

    assert group.tolist() == [0, 1, 0, 1, 0, 1], "grouping did not follow traj_qid"
    # first correct of each question is the anchor (0), its sibling is a positive (1)
    assert kind.tolist() == [0, 0, 1, 1, 2, 2]
    assert n_correct.tolist() == [2, 2]
    assert n_incorrect.tolist() == [1, 1]


def test_a_replica_question_interleaved_changes_nothing(dist):
    """The value-level version of the grouping claim, and it is exact.

    `q2` is an EXACT replica of `q1` -- same three terminal vectors -- interleaved row by row.
    Every question is scored independently and the loss means over questions, so two identical
    questions must give the single question's value unchanged. If the two questions' terminals
    shared a class, `q1`'s class would gain two members and a negative and the number would
    move a long way.

    Interleaved rather than block-ordered on purpose: a grouping bug that keyed on batch
    position rather than on `traj_qid` survives a block-ordered fixture.
    """
    torch.manual_seed(0)
    a, b, n = torch.randn(D), torch.randn(D) * 1.5, torch.randn(D) * 3.0

    solo = _batch([("q1", 2, 1)])
    solo_loss, solo_info = terminal_class_loss(_psi(solo, {0: a, 1: b, 2: n}), solo, dist)

    pair = collate(
        [
            synthetic_row("q1", [T, T, T]), synthetic_row("q2", [T, T, T]),
            synthetic_row("q1", [T, T]),    synthetic_row("q2", [T, T]),
            synthetic_row("q1", [T, F]),    synthetic_row("q2", [T, F]),
        ],
        pad_id=0,
    )
    pair_loss, pair_info = terminal_class_loss(
        _psi(pair, {0: a, 1: a, 2: b, 3: b, 4: n, 5: n}), pair, dist
    )

    assert pair_info["term/questions"] == 2.0
    assert float(pair_loss) == pytest.approx(float(solo_loss), rel=1e-6)
    assert pair_info["term/positive_distance"] == pytest.approx(
        solo_info["term/positive_distance"], rel=1e-6
    )


def test_kinds_are_first_correct_then_correct_then_incorrect():
    batch = _batch([("q1", 3, 2), ("q2", 1, 1)])
    _, kind, n_correct, n_incorrect = terminal_class_index(batch)
    assert kind.tolist() == [0, 1, 1, 2, 2, 0, 2]
    assert n_correct.tolist() == [3, 1]
    assert n_incorrect.tolist() == [2, 1]


def test_which_correct_terminal_is_the_anchor_does_not_matter(dist):
    """Slot 0 is arbitrary. The anchor is a query like every other class member (§7.5.1 --
    iterating the query over `C` trains both orderings of every pair), so relabelling which
    correct trajectory is the anchor must leave the loss unchanged.
    """
    batch = _batch([("q1", 3, 2)])
    torch.manual_seed(1)
    psi = torch.randn(batch.n_states, D)
    group, kind, _, _ = terminal_class_index(batch)
    psi_t = psi.index_select(0, batch.traj_terminal)

    base, _ = counterfactual_loss(psi_t, group, kind, dist)
    for anchor_at in (1, 2):
        relabelled = kind.clone()
        relabelled[0] = 1
        relabelled[anchor_at] = 0
        loss, _ = counterfactual_loss(psi_t, group, relabelled, dist)
        assert float(loss) == pytest.approx(float(base), rel=1e-6)


# ------------------------------------------------------------------------ the SupCon properties

def test_two_correct_terminals_of_one_question_are_pulled_together(dist):
    """The whole point of the term (§16.26): the ONLY thing in the loss set that says "pull
    these together" rather than "stop pushing these apart"."""
    batch = _batch([("q1", 2, 3)])
    torch.manual_seed(2)
    anchor = torch.randn(D)
    sibling = anchor + 2.0 * torch.randn(D)
    negatives = {2 + i: torch.randn(D) * 4.0 for i in range(3)}

    base, _ = terminal_class_loss(_psi(batch, {0: anchor, 1: sibling, **negatives}), batch, dist)
    closer = anchor + 0.4 * (sibling - anchor)
    loss, _ = terminal_class_loss(_psi(batch, {0: anchor, 1: closer, **negatives}), batch, dist)
    assert float(loss) < float(base)


def test_a_third_correct_terminal_that_already_dominates_is_not_pulled_further(dist):
    """**What keeping the positives in the DENOMINATOR buys** (§7.5.1), asserted on the
    terminal grouping and not only on the CF one.

    `dL_q/ds_p = softmax_p - 1/|P|`: a class member is pulled in only while it carries LESS
    than its fair share of the softmax mass. Collapsing the one that is ALREADY closest onto
    the anchor therefore RAISES the loss -- the objective wants the class uniformly tight, and
    that is what makes a question's correct terminals ONE class instead of `|P|` independent
    pull-together problems. Recorded so it is not "fixed" into a per-positive loss later.
    """
    batch = _batch([("q1", 3, 3)])
    torch.manual_seed(3)
    anchor = torch.randn(D)
    nearest = anchor + 0.3 * torch.randn(D)          # already the closest sibling
    far = anchor + 2.0 * torch.randn(D)
    negatives = {3 + i: torch.randn(D) * 4.0 for i in range(3)}

    base, _ = terminal_class_loss(
        _psi(batch, {0: anchor, 1: nearest, 2: far, **negatives}), batch, dist
    )
    collapsed = anchor + 0.001 * torch.randn(D)
    loss, _ = terminal_class_loss(
        _psi(batch, {0: anchor, 1: collapsed, 2: far, **negatives}), batch, dist
    )
    assert float(loss) > float(base), (
        "the positives left the denominator -- this is |P| one-positive problems, not one class"
    )


def test_every_incorrect_terminal_is_pushed_away_unconditionally(dist):
    """Unlike the positive case: `dL_q/ds_n = softmax_n > 0` for every negative and every
    query, so moving ANY incorrect terminal toward the class always raises the loss."""
    batch = _batch([("q1", 2, 3)])
    torch.manual_seed(4)
    anchor = torch.randn(D)
    sibling = anchor + 1.0 * torch.randn(D)
    negatives = {2 + i: anchor + 4.0 * torch.randn(D) for i in range(3)}

    base, _ = terminal_class_loss(_psi(batch, {0: anchor, 1: sibling, **negatives}), batch, dist)
    for traj in negatives:
        moved = dict(negatives)
        moved[traj] = anchor + 0.25 * (negatives[traj] - anchor)
        loss, _ = terminal_class_loss(
            _psi(batch, {0: anchor, 1: sibling, **moved}), batch, dist
        )
        assert float(loss) > float(base), f"incorrect terminal {traj} is not being pushed away"


def _negatives_as_queries_reference(psi_terminal, kind, distance):
    """An off-the-shelf batch-wide SupCon over the two labels `{class, negative}`: EVERY row
    is a query, so the negatives are pulled together among themselves.

    This is the implementation §7.5.1 says an off-the-shelf SupCon gives you, written out so
    that "the guard has teeth" is a measured number rather than a hope. Reference for the
    tests only -- it must never be imported by the package.
    """
    n = psi_terminal.shape[0]
    d = distance(psi_terminal[:, None, :], psi_terminal[None, :, :])
    eye = torch.eye(n, dtype=torch.bool)
    label = kind != 2                                     # the off-the-shelf label
    pos = (label[:, None] == label[None, :]) & ~eye
    logits = (-d).masked_fill(eye, float("-inf"))
    log_denom = torch.logsumexp(logits, dim=1)
    mean_pos_logit = logits.masked_fill(~pos, 0.0).sum(dim=1) / pos.sum(dim=1).clamp(min=1)
    return (log_denom - mean_pos_logit).mean()


def test_permuting_the_incorrect_terminals_leaves_the_loss_unchanged(dist):
    """NEGATIVES ARE NEVER QUERIES (§7.5.1). The negatives enter only as a SET, so permuting
    them must not move the loss.

    **The brief asked for float equality and this codebase cannot deliver it -- the property
    is exact in exact arithmetic and the implementation is fp32 by mandate.** `log_denom` is a
    `logsumexp` over the denominator, so permuting the negatives permutes the summation order,
    and fp32 rounds differently. MEASURED 2026-08-08 over all 24 permutations of 4 negatives,
    40 seeds x {D=64, D=512}: the deviation is 0 on most seeds and **1.19e-7 = 2^-23, i.e. one
    fp32 ULP, on ~36% of them**. `torch.float64` does not help: `mrn_distance` casts to fp32
    inside the distance and always will (bug B10a, `model/distances.py`).

    So `tests/test_counterfactual.py::test_permuting_the_negatives_leaves_the_loss_unchanged`
    passes on float equality **by luck of its seed**, not by a property of the loss. Do not
    copy that assertion here and do not "fix" this one into it.

    Replaced by a tolerance of 1e-6, ~10 ULP.

    **And note what this test is and is not.** A permutation of same-kind rows is a
    RELABELLING, and a negatives-as-queries loss is symmetric in the negative set too -- so it
    would pass this as well. This catches position-dependent handling of the negatives (the
    shape of the §7.5.4 slot-0 bug); the test that actually catches a negative used as a query
    is `test_moving_two_incorrect_terminals_toward_each_other_changes_nothing`, which carries
    the sensitivity floor.
    """
    batch = _batch([("q1", 2, 4)])
    torch.manual_seed(5)
    vectors = {i: torch.randn(D) for i in range(6)}
    base, _ = terminal_class_loss(_psi(batch, vectors), batch, dist)

    for permutation in ([5, 4, 3, 2], [3, 5, 2, 4], [4, 2, 5, 3]):
        shuffled = dict(vectors)
        for slot, source in zip(range(2, 6), permutation):
            shuffled[slot] = vectors[source]
        loss, _ = terminal_class_loss(_psi(batch, shuffled), batch, dist)
        assert float(loss) == pytest.approx(float(base), abs=1e-6)


def test_moving_two_incorrect_terminals_toward_each_other_changes_nothing(dist):
    """There is no negative-negative term. Two solutions that are wrong in DIFFERENT ways end
    at different wrong answers; asserting they coincide is a claim this data cannot support.

    Construction is `tests/test_counterfactual.py`'s: put the whole class at the origin, where
    `d(0, n) = mean_K [ max(relu(-n))_asym + L2(n)_sym ]` is invariant to permuting coordinates
    WITHIN a component-half. So a negative can be moved -- changing `d(n_i, n_j)` -- with every
    class -> negative distance held EXACTLY fixed. If the loss moves, a negative was a query.
    """
    batch = _batch([("q1", 3, 2)])
    components, half = 8, D // 8 // 2
    torch.manual_seed(6)
    klass = {0: torch.zeros(D), 1: torch.zeros(D), 2: torch.zeros(D)}
    n1, n2 = torch.randn(D).abs() + 0.5, torch.randn(D).abs() + 0.5

    def shuffle_within_halves(vector: torch.Tensor, seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        blocks = vector.reshape(components, 2, half).clone()
        for k in range(components):
            for h in range(2):
                blocks[k, h] = blocks[k, h][torch.randperm(half, generator=generator)]
        return blocks.reshape(D)

    base, base_info = terminal_class_loss(_psi(batch, {**klass, 3: n1, 4: n2}), batch, dist)
    candidates = [shuffle_within_halves(n2, seed) for seed in range(8)]
    n2_closer = min(candidates, key=lambda v: float(dist(n1.reshape(1, -1), v.reshape(1, -1))))

    assert float(dist(n1[None], n2_closer[None])) < float(dist(n1[None], n2[None])), \
        "the fixture did not actually move the two negatives closer together"
    assert torch.allclose(dist(klass[0][None], n2_closer[None]), dist(klass[0][None], n2[None])), \
        "the fixture changed a class -> negative distance, so it proves nothing"

    moved, moved_info = terminal_class_loss(
        _psi(batch, {**klass, 3: n1, 4: n2_closer}), batch, dist
    )
    assert float(moved) == pytest.approx(float(base), abs=1e-6)
    assert moved_info["term/negative_distance"] == pytest.approx(
        base_info["term/negative_distance"], abs=1e-6
    )

    # THE SENSITIVITY FLOOR. The assertion above is only worth something if the loss it guards
    # against actually moves on this fixture, so score the off-the-shelf batch-wide SupCon on
    # the same two batches: it pulls the negatives together, so bringing them together must
    # lower it, and by orders of magnitude more than the 1e-6 tolerance.
    _, kind, _, _ = terminal_class_index(batch)
    terminals = lambda psi: psi.index_select(0, batch.traj_terminal)  # noqa: E731
    broken_base = float(_negatives_as_queries_reference(
        terminals(_psi(batch, {**klass, 3: n1, 4: n2})), kind, dist
    ))
    broken_moved = float(_negatives_as_queries_reference(
        terminals(_psi(batch, {**klass, 3: n1, 4: n2_closer})), kind, dist
    ))
    # MEASURED 2026-08-08 on this fixture: 3.4e-3, i.e. ~3,400x the 1e-6 tolerance above.
    # Direction-free on purpose -- what matters is that a negative-negative term registers at
    # all, not which way the off-the-shelf loss happens to move.
    assert abs(broken_base - broken_moved) > 1e-3, (
        "the guard has no teeth on this fixture: even a negatives-as-queries loss barely "
        "moves when the two negatives are brought together, so the assertion above is vacuous"
    )


# -------------------------------------------------------------- degenerate and ragged batches

def test_a_question_with_one_correct_terminal_is_skipped_and_counted(dist):
    """The loss already drops a class of size 1 rather than crashing. **Do not rely on that
    silently** -- §14: a guard that fails toward "healthy" is worse than no guard.
    """
    batch = _batch([("q1", 3, 2), ("q2", 1, 3)])
    torch.manual_seed(7)
    loss, info = terminal_class_loss(torch.randn(batch.n_states, D), batch, dist)

    assert torch.isfinite(loss)
    assert info["term/questions"] == 1.0, "q2 contributed a query it has no positive for"
    assert info["term/questions_skipped_single_correct"] == 1.0


def test_a_question_with_no_correct_terminal_lands_in_the_same_count(dist):
    """§8.1's carry rule lets a partially included question arrive with 0 correct
    trajectories. It has no anchor, contributes no query, and must be counted, not crashed."""
    rows = _rows([("q1", 2, 1)]) + [synthetic_row("q2", [F, F])]
    batch = collate(rows, pad_id=0)
    torch.manual_seed(8)
    loss, info = terminal_class_loss(torch.randn(batch.n_states, D), batch, dist)

    assert torch.isfinite(loss)
    assert info["term/questions"] == 1.0
    assert info["term/questions_skipped_single_correct"] == 1.0


def test_ragged_correct_and_incorrect_counts_in_one_batch(dist):
    batch = _batch([("q1", 4, 3), ("q2", 2, 1), ("q3", 3, 0), ("q4", 1, 2)])
    torch.manual_seed(9)
    loss, info = terminal_class_loss(torch.randn(batch.n_states, D), batch, dist)

    assert torch.isfinite(loss)
    assert info["term/questions"] == 3.0                       # q4 has one correct terminal
    assert info["term/questions_skipped_single_correct"] == 1.0
    # 4 questions; kind-1 rows are (4-1) + (2-1) + (3-1) + 0 = 6, kind-2 rows are 3+1+0+2 = 6.
    assert info["term/positives_per_question"] == 6 / 4
    assert info["term/negatives_per_question"] == 6 / 4


def test_gradients_reach_every_terminal_in_a_scored_class(dist):
    batch = _batch([("q1", 3, 2)])
    torch.manual_seed(10)
    psi = torch.randn(batch.n_states, D, requires_grad=True)
    loss, _ = terminal_class_loss(psi, batch, dist)
    loss.backward()

    assert torch.isfinite(psi.grad).all()
    reached = psi.grad.abs().sum(dim=1)[batch.traj_terminal]
    assert (reached > 0).all(), "some terminal of a scored question receives no gradient"
    # ...and NOTHING else: this term reads terminals only.
    non_terminal = torch.ones(batch.n_states, dtype=torch.bool)
    non_terminal[batch.traj_terminal] = False
    assert float(psi.grad[non_terminal].abs().sum()) == 0.0


# ------------------------------------------------------------------------------ diagnostics

def test_the_empty_path_logs_the_same_keys_as_the_populated_one(dist):
    """`_empty_info` in `counterfactual.py` says why: a diagnostic that disappears when a
    batch is degenerate is a diagnostic that cannot be plotted."""
    populated = terminal_class_loss(
        torch.randn(_batch([("q1", 2, 2)]).n_states, D), _batch([("q1", 2, 2)]), dist
    )[1]
    # Every question skipped -> no queries at all, the deepest degenerate path there is.
    empty_batch = _batch([("q1", 1, 2), ("q2", 1, 1)])
    loss, empty = terminal_class_loss(torch.randn(empty_batch.n_states, D), empty_batch, dist)

    assert float(loss) == 0.0
    assert set(empty) == set(populated)
    assert empty["term/questions_skipped_single_correct"] == 2.0


def test_the_term_key_map_covers_the_cf_key_set_exactly(dist):
    """`L_term` renames `cf/*` to `term/*` and adds two keys of its own. A key added to
    `counterfactual.py` and not to `_CF_TO_TERM` would raise a KeyError on the next batch --
    catch it here instead of in hour three of a run."""
    cf_keys = set(
        counterfactual_loss(
            torch.randn(4, D), torch.tensor([0, 0, 0, 0]), torch.tensor([0, 1, 2, 2]), dist
        )[1]
    )
    assert cf_keys == set(_CF_TO_TERM)

    term_keys = set(terminal_class_loss(
        torch.randn(_batch([("q1", 2, 2)]).n_states, D), _batch([("q1", 2, 2)]), dist
    )[1])
    assert term_keys == set(_CF_TO_TERM.values()) | {
        "term/questions_skipped_single_correct",
        "term/chance",
        "term/within_question_terminal_spread",
    }


def test_positive_distance_is_over_all_class_pairs_both_orderings(dist):
    """Not one slot against the anchor -- that reading is what hid the old single-positive
    L_CF bug (§7.5.4), and it would hide the same bug here."""
    batch = _batch([("q1", 3, 2)])
    torch.manual_seed(11)
    anchor = torch.randn(D)
    near, far = anchor + 0.01 * torch.randn(D), anchor + 6.0 * torch.randn(D)
    negatives = {3: torch.randn(D) * 3.0, 4: torch.randn(D) * 3.0}

    _, info = terminal_class_loss(
        _psi(batch, {0: anchor, 1: near, 2: far, **negatives}), batch, dist
    )
    # 3 class members -> 6 ordered pairs; a slot-0-only reading would report ~d(anchor, near).
    assert info["term/positive_distance"] > float(dist(anchor[None], near[None])) * 10


def test_within_question_terminal_spread_is_the_positive_distance(dist):
    """§16.26 names `within_question_terminal_spread` as the statistic this term is supposed
    to move. It is `term/positive_distance` -- the mean over ordered same-question
    correct-terminal pairs -- by construction, and this pins the identity so it cannot drift
    into two different definitions of one statistic.
    """
    batch = _batch([("q1", 3, 2), ("q2", 2, 2)])
    torch.manual_seed(12)
    _, info = terminal_class_loss(torch.randn(batch.n_states, D), batch, dist)
    assert info["term/within_question_terminal_spread"] == info["term/positive_distance"]
    assert info["term/within_question_terminal_spread"] > 0.0


def test_chance_is_the_flat_softmax_level(dist):
    """`term/chance` is the level `term/loss` is read AGAINST, and it moves with the batch's
    ragged counts -- unlike `L_NCE`'s `log(R)` there is no constant to quote (§7.13)."""
    # q1: 3 correct, 2 incorrect -> log(3-1+2); q2: 2 correct, 1 incorrect -> log(2-1+1);
    # q3: 1 correct -> skipped, contributes nothing.
    batch = _batch([("q1", 3, 2), ("q2", 2, 1), ("q3", 1, 4)])
    _, kind, n_correct, n_incorrect = terminal_class_index(batch)
    expected = (math.log(4) + math.log(2)) / 2
    assert terminal_class_chance(n_correct, n_incorrect) == pytest.approx(expected, rel=1e-9)

    # ...and an all-identical psi actually sits there, which is what makes it "chance".
    psi = torch.zeros(batch.n_states, D)
    loss, info = terminal_class_loss(psi, batch, dist)
    assert float(loss) == pytest.approx(expected, rel=1e-5)
    assert info["term/chance"] == pytest.approx(expected, rel=1e-9)


# ------------------------------------------------------------------------------ the wiring

def _at_lambda_term(cfg, lam: float):
    return dataclasses.replace(cfg, losses=dataclasses.replace(cfg.losses, lambda_term=lam))


def test_lambda_term_ships_at_one(cfg):
    """RAISED 0.0 -> 1.0 ON 2026-08-15 BY THE HUMAN, and this pin flipped with it.

    It asserted 0.0 from the day the term was built. §16.26's order of work puts the §9.9
    masks first and calls this term a hypothesis rather than a fix; the human was shown that
    and asked for the weight anyway, which is the decision. The pin is kept rather than
    deleted because the value is unmeasured (§16.8) and worth a deliberate act to change.

    What the decision does NOT waive is §7.13.1: `answer_match_auc` is 0.927 against a chance
    of 0.500, so a run that improves `gate/recall_at_1` has two explanations. The separator is
    `scripts/goal_gate.py --mask-answer` against the same checkpoint, and it is post-hoc.
    """
    assert cfg.losses.lambda_term == 1.0


def test_tau_is_a_deliberate_pick_and_not_an_inherited_default(cfg):
    """§7.13's rule: whoever raises `lambda_term` picks `tau` in the SAME change.

    It became `losses.lambda_term_temperature` on 2026-08-15 for that reason -- a defaulted
    function argument nobody has to type is what "unexamined" means. 1.0, because (7)'s
    denominator is ~4.9 candidates and sqrt(512) would pin the softmax at log(c-1+w): an off
    switch, not a temperature. Pinned so the two cannot drift apart silently, since `tau` and
    `lambda` are the same knob (gradient ~ lambda/tau).
    """
    import re
    from pathlib import Path

    assert cfg.losses.lambda_term_temperature == 1.0

    total = (Path(__file__).resolve().parents[1] / "feynman_prm" / "losses" / "total.py")
    code = re.sub(r"#.*", "", total.read_text())
    call = code[code.index("terminal_class_loss("):]
    assert "lambda_term_temperature" in call[: call.index(")") + 200], (
        "total.py must pass tau EXPLICITLY from the config -- letting it default is the "
        "unexamined inheritance §7.13 forbids"
    )


def test_the_total_is_bit_identical_with_and_without_the_term_at_lambda_zero(cfg):
    """§15, mirroring `test_counterfactual.py`'s CF version. The term is COMPUTED on every
    batch (its diagnostics are the point), so what is tested is `lambda_term == 0.0` giving an
    exact zero, not the term being skipped.

    `lambda_term` is overridden explicitly rather than read off the shipped config: the
    property is about the weight being zero, not about which weight ships, and tying it to the
    default is what made this test fail when the default moved.
    """
    from test_smoke import _run

    off = _at_lambda_term(cfg, 0.0)
    out = _run(off)[-1]
    assert float(out.terms["term"]) > 0.0, "the term is not being computed at all"

    manual = out.total - off.losses.lambda_term * out.terms["term"]
    assert float(manual) == float(out.total)

    # ...and the same batch reconstructed WITHOUT the term in the sum lands on the same bits.
    without = (
        off.losses.lambda_nce * out.terms["nce"]
        + off.losses.lambda_i * out.terms["invariance"]
        + off.losses.zeta * out.terms["backup"]
        + off.losses.lambda_cf * out.terms["cf"]
        + off.losses.lambda_step * out.terms["step"]
        + off.losses.lambda_good * out.terms["good"]
    )
    with_term = without + off.losses.lambda_term * out.terms["term"]
    assert float(with_term) == float(without)


def test_the_term_receives_no_gradient_at_lambda_zero(cfg):
    """The stronger reading of "inert": at 0.0 the term must not move a single parameter.
    A weight of exactly 0.0 zeroes the gradient of the product, so the terminal rows get
    whatever the OTHER terms give them and nothing from this one.
    """
    from test_smoke import _batch as smoke_batch

    batch = smoke_batch(0)
    torch.manual_seed(13)
    psi = torch.randn(batch.n_states, D, requires_grad=True)
    distance = Distance(cfg.distance.variant, cfg.distance.components)

    loss, _ = terminal_class_loss(psi, batch, distance)
    (0.0 * loss).backward()
    assert psi.grad is None or float(psi.grad.abs().sum()) == 0.0


def test_at_the_shipped_weight_the_term_reaches_the_terminal_latents(cfg):
    """The complement, and the one that would have caught "computed but never trained".

    (7) was fed, logged and multiplied by zero for a week; the bit-identical test above passes
    in that state and says nothing about whether raising the weight does anything. So assert
    the other direction at the weight that ships: the term is nonzero, it enters the total at
    exactly `lambda_term`, and its gradient reaches the TERMINAL rows -- which are the only
    rows it is entitled to move (§7.13: `psi[batch.traj_terminal]`, no fresh head).
    """
    from test_smoke import _batch as smoke_batch, _run

    lam = cfg.losses.lambda_term
    assert lam > 0.0

    out = _run(cfg)[-1]
    off = _run(_at_lambda_term(cfg, 0.0))[-1]
    assert float(out.terms["term"]) > 0.0
    assert float(out.total) != float(off.total)
    assert float(out.total) == pytest.approx(
        float(off.total) + lam * float(out.terms["term"]), rel=1e-5
    ), "(7) must enter the total at exactly lambda_term"

    batch = smoke_batch(0)
    torch.manual_seed(13)
    psi = torch.randn(batch.n_states, D, requires_grad=True)
    distance = Distance(cfg.distance.variant, cfg.distance.components)
    loss, _ = terminal_class_loss(psi, batch, distance, temperature=cfg.losses.lambda_term_temperature)
    (lam * loss).backward()

    assert psi.grad is not None and float(psi.grad.abs().sum()) > 0.0
    touched = set(torch.nonzero(psi.grad.abs().sum(dim=1)).flatten().tolist())
    terminals = set(batch.traj_terminal.tolist())
    assert touched <= terminals, (
        "(7) moved a NON-terminal state -- it reads psi[batch.traj_terminal] and nothing else"
    )
