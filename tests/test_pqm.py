"""`pqm_baseline/` -- the PQM (Li & Li, ICLR 2025) baseline, matched to Feynman-PRM.

CPU only, no model, no GPU, no download. Everything here runs on the same synthetic rows the
rest of the suite uses, because `pqm_baseline` inherits the project's separation of index
bookkeeping from the model (PLAN 'Core design decisions' 1).

This file lives in `tests/`, NOT under `feynman_prm/`, and that is deliberate:
`test_grep_invariants.py::test_no_value_head_anywhere` scans `feynman_prm/**/*.py` +
`scripts/*.py`, and a value head is exactly what that guard exists to keep out of the METHOD.
`tests/` is not scanned, so the guard stays honest instead of being renamed around.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from conftest import synthetic_row

from feynman_prm.data.collate import collate
from feynman_prm.utils.indexing import predicted_label_from_deltas
from pqm_baseline.loss import (
    PAD_LABEL,
    build_padded,
    loss_at_zero_rewards,
    pqm_diagnostics,
    pqm_ranking_loss,
    rewards_at_steps,
    step_labels_from_z,
)
from pqm_baseline.model import ValueHead, assert_pqm_trainable

T, F = True, False


# ---------------------------------------------------------------------------------------
# 1. the port is bit-for-bit the authors' function
# ---------------------------------------------------------------------------------------


def reference_ranking_loss(rewards, labels, has_neg, zeta):
    """`Process_Q_Model/train_main.py:61-78`, copied here so a future "simplification" of the
    port fails loudly instead of quietly moving the baseline.

    Two mechanical edits only: `self` dropped, and the module-global `args.zeta` passed in.
    """
    pos_rewards_exp = torch.where(labels == 1, (rewards).exp(), 0)
    neg_rewards_exp = torch.where(labels == 0, (rewards+zeta).exp(), 0).flip(dims=[-1])
    neg_reward_sum = neg_rewards_exp.sum(-1)

    pos_rewards_cumsum = torch.cat([torch.zeros(rewards.shape[0], 1, device=rewards.device).exp(), pos_rewards_exp],
                                   dim=1).cumsum(-1)[:, :-1]
    pos_rewards_cumsum = torch.cat([torch.zeros(rewards.shape[0], 1, device=rewards.device), pos_rewards_cumsum],
                                   dim=-1)

    reward_exp_cur = torch.where(labels == 1, pos_rewards_exp, 1)
    reward_exp_cur = torch.cat([torch.zeros(rewards.shape[0], 1, device=rewards.device).exp(), reward_exp_cur], dim=-1)

    loss = -torch.log(reward_exp_cur / (reward_exp_cur + pos_rewards_cumsum + neg_reward_sum[..., None] + 1e-5))

    labels = torch.cat([has_neg[..., None], labels], dim=-1)
    loss = (torch.where(labels == 1, loss, 0).sum(-1) / torch.where(labels == 1, 1, 0).sum(-1)).mean()
    return loss


def _random_padded(seed: int, B: int = 6, Tmax: int = 7):
    """A ragged (rewards, labels, has_neg) triple with real `-100` padding."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.full((B, Tmax), PAD_LABEL, dtype=torch.long)
    rewards = torch.zeros(B, Tmax)
    for b in range(B):
        n = int(torch.randint(1, Tmax + 1, (1,), generator=g))
        labels[b, :n] = torch.randint(0, 2, (n,), generator=g)
        rewards[b, :n] = torch.randn(n, generator=g) * 2.0
    has_neg = (labels == 0).sum(-1).bool().long()
    return rewards, labels, has_neg


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
@pytest.mark.parametrize("zeta", [4.0, 1.0, 8.0])
def test_ranking_loss_matches_the_authors_function(seed, zeta):
    rewards, labels, has_neg = _random_padded(seed)
    ours = pqm_ranking_loss(rewards, labels, zeta, has_neg)
    theirs = reference_ranking_loss(rewards, labels, has_neg, zeta)
    assert torch.equal(ours, theirs), (ours.item(), theirs.item())


def test_has_neg_is_recomputed_pqm_s_way_when_not_passed():
    """`train_main.py:225` builds it as `1 if 0 in labels else 0`; recomputing it from the
    padded labels gives the same value and keeps the ported function self-contained."""
    rewards, labels, has_neg = _random_padded(11)
    assert torch.equal(
        pqm_ranking_loss(rewards, labels, 4.0),
        pqm_ranking_loss(rewards, labels, 4.0, has_neg),
    )


def test_the_vestigial_flip_is_still_there():
    """`.flip(dims=[-1])` on the negatives (line 63) is immediately summed over and changes
    nothing -- it stays so a diff against the authors' file is empty. Grep, because the whole
    point is that the LINE exists."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "pqm_baseline" / "loss.py").read_text()
    assert ".flip(dims=[-1])" in src
    assert "+ 1e-5" in src, "the denominator epsilon is part of the closed form, not noise"


# ---------------------------------------------------------------------------------------
# 2. the index round-trip -- the highest-value test here (§15)
# ---------------------------------------------------------------------------------------


def test_index_round_trip_from_z_through_the_eval_mapping():
    """`labels = [T, T, F, F]` => `z = 2`. A low reward at STEP INDEX 2 must come back as
    `predicted_label == 2` -- not 1 and not 3.

    This is the off-by-one §7.6 records as invisible in every loss curve: a model whose low
    reward lands one step early scores `acc_error = 0` on every errored ProcessBench sample
    and F1 collapses through the harmonic mean while the training curves stay healthy.
    """
    row = synthetic_row("q", [T, T, F, F])
    assert row.z == 2
    batch = collate([row], pad_id=0)

    # reward high everywhere except at the state step 2 LANDS IN, which is s_3.
    rewards = torch.full((batch.n_states,), 5.0)
    rewards[int(batch.traj_state_offset[0]) + 3] = -5.0

    step_rewards = rewards_at_steps(batch, rewards)
    assert step_rewards.tolist() == [5.0, 5.0, -5.0, 5.0]

    deltas = (-step_rewards).tolist()                      # eval's scale: higher = worse
    assert predicted_label_from_deltas(deltas, tau=0.0) == 2

    # and the labels the loss trains on agree with the gold index
    assert step_labels_from_z(batch).tolist() == [1, 1, 0, 0]


def test_the_reward_for_step_i_is_read_at_the_state_it_lands_in():
    """`row_dst` is `s_i`, `row_src` is `s_{i-1}`. Reading the wrong one shifts every score by
    one step, which is the same failure as above wearing a different hat."""
    batch = collate([synthetic_row("q", [T, T, T])], pad_id=0)
    assert batch.row_dst.tolist() == [1, 2, 3]
    assert batch.row_src.tolist() == [0, 1, 2]
    assert batch.row_step.tolist() == [1, 2, 3]


def test_z_equals_zero_is_handled(  ):
    """45.4% of incorrect Math-Shepherd trajectories have `z = 0` (§4.2.1). Every step is
    negative, `has_neg` is 1, and the loss must be finite."""
    batch = collate([synthetic_row("q", [F, F, F])], pad_id=0)
    assert batch.traj_z.tolist() == [0]
    assert step_labels_from_z(batch).tolist() == [0, 0, 0]
    rewards = torch.zeros(batch.n_states, requires_grad=True)
    rp, lp, hn = build_padded(batch, rewards)
    loss = pqm_ranking_loss(rp, lp, 4.0, hn)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(rewards.grad).all()


# ---------------------------------------------------------------------------------------
# 3. flat -> padded
# ---------------------------------------------------------------------------------------


@pytest.fixture
def ragged_batch():
    return collate(
        [
            synthetic_row("q1", [T, T, T]),
            synthetic_row("q1", [T, T, F, F]),
            synthetic_row("q2", [T, T]),
            synthetic_row("q2", [F, F, F]),
        ],
        pad_id=0,
    )


def test_flat_to_padded_lands_at_traj_and_step_minus_one(ragged_batch):
    batch = ragged_batch
    rewards = torch.arange(batch.n_states, dtype=torch.float32)
    rp, lp, hn = build_padded(batch, rewards)

    assert rp.shape == (batch.n_traj, int(batch.traj_T.max()))
    for r in range(batch.n_rows):
        b, t = int(batch.row_traj[r]), int(batch.row_step[r]) - 1
        assert rp[b, t] == rewards[int(batch.row_dst[r])]

    assert lp.tolist() == [
        [1, 1, 1, PAD_LABEL],
        [1, 1, 0, 0],
        [1, 1, PAD_LABEL, PAD_LABEL],
        [0, 0, 0, PAD_LABEL],
    ]
    assert hn.tolist() == [0, 1, 0, 1]


def test_padded_slots_are_exactly_zero_and_minus_100(ragged_batch):
    """The zero-fill is load-bearing (see the NaN test below), so it is asserted rather than
    assumed: `torch.where(labels == 1, rewards.exp(), 0)` evaluates `exp()` on every slot."""
    batch = ragged_batch
    rewards = torch.randn(batch.n_states) * 10.0
    rp, lp, _ = build_padded(batch, rewards)
    pad = lp == PAD_LABEL
    assert pad.any()
    assert torch.equal(rp[pad], torch.zeros(int(pad.sum())))


def test_permuting_the_batch_leaves_the_loss_unchanged(ragged_batch):
    """The loss is a mean over trajectories, so it is permutation-invariant. A failure here
    means the flat->padded mapping is reading a positional index it should not."""
    rows = [
        synthetic_row("q1", [T, T, T]),
        synthetic_row("q1", [T, T, F, F]),
        synthetic_row("q2", [T, T]),
        synthetic_row("q2", [F, F, F]),
    ]
    torch.manual_seed(0)
    base_rewards = {i: torch.randn(r.n_steps + 1) for i, r in enumerate(rows)}

    def loss_for(order):
        batch = collate([rows[i] for i in order], pad_id=0)
        rewards = torch.cat([base_rewards[i] for i in order])
        rp, lp, hn = build_padded(batch, rewards)
        return float(pqm_ranking_loss(rp, lp, 4.0, hn))

    assert loss_for([0, 1, 2, 3]) == pytest.approx(loss_for([3, 1, 0, 2]), abs=1e-6)


def test_padding_width_does_not_change_the_loss():
    """Tmax is set by the batch's longest trajectory. A short trajectory's loss must not move
    when a long one joins the batch -- pads sit at the END, so they enter no cumsum a real
    slot reads."""
    short = synthetic_row("q", [T, T])
    long_ = synthetic_row("q", [T, F, F, F, F, F])
    torch.manual_seed(1)
    r_short = torch.randn(short.n_steps + 1)
    r_long = torch.randn(long_.n_steps + 1)

    def per_traj(rows, rewards):
        batch = collate(rows, pad_id=0)
        rp, lp, hn = build_padded(batch, rewards)
        # the per-trajectory term, before the batch mean
        stacked = pqm_ranking_loss(rp[:1], lp[:1], 4.0, hn[:1])
        return float(stacked)

    alone = per_traj([short], r_short)
    padded = per_traj([short, long_], torch.cat([r_short, r_long]))
    assert alone == pytest.approx(padded, abs=1e-6)


# ---------------------------------------------------------------------------------------
# 4. the 0 * inf trap
# ---------------------------------------------------------------------------------------


def test_padding_never_produces_nan():
    """Wildly mixed `T` and large rewards must backward cleanly.

    With an UNINITIALISED padded slot (`new_empty`) this is the failure: `exp()` of garbage
    overflows to `inf`, `where`'s backward multiplies `0 * inf`, and a NaN appears in the
    gradient that no forward value would ever reveal.
    """
    rows = [synthetic_row("q", [T] * n) for n in (1, 2, 12)] + [
        synthetic_row("q", [F] * 9),
        synthetic_row("q", [T, T, F, F, F]),
    ]
    batch = collate(rows, pad_id=0)
    rewards = (torch.randn(batch.n_states) * 30.0).requires_grad_(True)
    rp, lp, hn = build_padded(batch, rewards)

    assert torch.equal(rp[lp == PAD_LABEL], torch.zeros(int((lp == PAD_LABEL).sum())))
    loss = pqm_ranking_loss(rp, lp, 4.0, hn)
    assert torch.isfinite(loss), loss
    loss.backward()
    assert torch.isfinite(rewards.grad).all(), rewards.grad


def test_an_uninitialised_pad_would_have_produced_nan():
    """The trap, demonstrated. Not a guard on our code -- a guard on the REASON the zero-fill
    is there, so a future "new_empty is faster" edit has a failing test to read."""
    labels = torch.tensor([[1, 1, PAD_LABEL], [1, 0, 0]])
    rewards = torch.tensor([[0.5, 0.5, 1e30], [0.1, 0.2, 0.3]], requires_grad=True)
    loss = pqm_ranking_loss(rewards, labels, 4.0)
    loss.backward()
    assert torch.isnan(rewards.grad).any(), (
        "a huge value in a PADDED slot should poison the backward -- if this stops being "
        "true, torch changed and the zero-fill's rationale should be re-read"
    )


# ---------------------------------------------------------------------------------------
# 5. the analytic init value (§18)
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 5, 9])
@pytest.mark.parametrize("zeta", [4.0, 0.5, 8.0])
def test_analytic_init_value_is_exact(seed, zeta):
    _, labels, has_neg = _random_padded(seed, B=8, Tmax=9)
    rewards = torch.zeros_like(labels, dtype=torch.float32)
    actual = float(pqm_ranking_loss(rewards, labels, zeta, has_neg))
    assert actual == pytest.approx(loss_at_zero_rewards(labels, zeta), abs=1e-5)


def test_analytic_init_value_on_a_hand_checked_fixture():
    """One trajectory, `[1, 0]`: n_pos = 1, n_neg = 1, has_neg.

        virtual slot: log(1 + e^zeta + 1e-5)
        the positive : log(2 + 0 + e^zeta + 1e-5)
        divided by 2

    The `2 + m`, not `1 + m`: the positive slot's denominator carries BOTH its own `cur = 1`
    and the cumsum's leading `exp(0)` prepended at train_main.py:66.
    """
    labels = torch.tensor([[1, 0]])
    zeta = 4.0
    e = math.exp(zeta)
    expected = (math.log(1 + e + 1e-5) + math.log(2 + e + 1e-5)) / 2
    assert loss_at_zero_rewards(labels, zeta) == pytest.approx(expected, abs=1e-12)
    assert float(pqm_ranking_loss(torch.zeros(1, 2), labels, zeta)) == pytest.approx(
        expected, abs=1e-5
    )


def test_analytic_init_value_matches_on_a_real_ragged_batch(ragged_batch):
    batch = ragged_batch
    rewards = torch.zeros(batch.n_states)
    rp, lp, hn = build_padded(batch, rewards)
    assert float(pqm_ranking_loss(rp, lp, 4.0, hn)) == pytest.approx(
        loss_at_zero_rewards(lp, 4.0), abs=1e-5
    )


def test_a_fully_correct_trajectory_has_no_virtual_slot():
    """`has_neg = 0` drops the prepended term entirely, so the closed form is the positives
    alone -- and at n_neg = 0 the `n_neg * e^zeta` term vanishes with it."""
    labels = torch.tensor([[1, 1]])
    expected = (math.log(2 + 0 + 1e-5) + math.log(3 + 0 + 1e-5)) / 2
    assert loss_at_zero_rewards(labels, 4.0) == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------------------
# 6. the trainability assert
# ---------------------------------------------------------------------------------------


class _FakeLora(nn.Module):
    """Enough of a PEFT-wrapped module for `classify_trainable`: parameter names carrying
    `lora_` are what it buckets on."""

    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(2, 2))
        self.lora_B = nn.Parameter(torch.zeros(2, 2))


class _FakePQM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _FakeLora()
        self.value_head = ValueHead(4, 0.1, "zero")


def test_assert_pqm_trainable_accepts_exactly_lora_and_value_head():
    counts = assert_pqm_trainable(_FakePQM())
    assert counts == {"lora": 2, "value_head": 2}


def test_assert_pqm_trainable_rejects_a_feynman_head():
    model = _FakePQM()
    model.psi = nn.Linear(2, 2)
    with pytest.raises(AssertionError, match="psi is trainable"):
        assert_pqm_trainable(model)

    model = _FakePQM()
    model.goal_head = nn.Linear(2, 2)
    with pytest.raises(AssertionError, match="goal_head is trainable"):
        assert_pqm_trainable(model)


def test_assert_pqm_trainable_catches_a_head_frozen_by_the_peft_wrap():
    """§14's LoRA trap 2: PEFT freezes every non-LoRA parameter at wrap time, so a head
    constructed BEFORE the wrap comes back frozen. It then trains on nothing while the loss
    still falls -- LoRA alone can move the hiddens -- and the only symptom is a baseline that
    is quietly not PQM."""
    model = _FakePQM()
    for p in model.value_head.parameters():
        p.requires_grad_(False)
    with pytest.raises(AssertionError, match="value_head is not trainable"):
        assert_pqm_trainable(model)


def test_assert_pqm_trainable_rejects_a_stray_trainable_parameter():
    model = _FakePQM()
    model.extra = nn.Parameter(torch.zeros(3))
    with pytest.raises(AssertionError, match="unexpected trainable parameters"):
        assert_pqm_trainable(model)


def test_value_head_is_pqm_s_shape_and_zero_init():
    """`Process_Q_Model/value_model.py:22-59` at Qwen2's config: Dropout(0.1) then
    Linear(H, 1), fp32, no MLP."""
    head = ValueHead(16, 0.1, "zero")
    assert isinstance(head.dropout, nn.Dropout) and head.dropout.p == 0.1
    assert isinstance(head.summary, nn.Linear)
    assert head.summary.weight.shape == (1, 16) and head.summary.weight.dtype is torch.float32
    assert torch.equal(head.summary.weight, torch.zeros(1, 16))
    assert torch.equal(head.summary.bias, torch.zeros(1))
    head.eval()
    assert torch.equal(head(torch.randn(5, 16)).squeeze(-1), torch.zeros(5))


def test_value_head_default_init_is_a_plain_linear():
    head = ValueHead(16, 0.1, "default")
    assert not torch.equal(head.summary.weight, torch.zeros(1, 16))


# ---------------------------------------------------------------------------------------
# 7. label derivation -- pinned, including where it DIFFERS
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "labels",
    [
        [T, T, T],
        [F, F, F],
        [T, T, F, F],
        [T],
        [F],
        [T, T, T, T, F],
    ],
)
def test_label_derivation_from_z_reproduces_monotone_labels(labels):
    """98.5% of Math-Shepherd trajectories are label-monotone (§4.2), and on those `from_z`
    reproduces the raw label vector exactly."""
    batch = collate([synthetic_row("q", labels)], pad_id=0)
    assert step_labels_from_z(batch).tolist() == [int(l) for l in labels]


def test_label_derivation_from_z_monotonises_a_recovery():
    """The documented divergence, pinned rather than hidden. `[T, F, T]` has `z = 1`, so
    `from_z` returns `[1, 0, 0]` and the raw labels are `[1, 0, 1]`.

    This affects the 1.48% of trajectories with a `False -> True` recovery (§4.2, §16.15), and
    it is exactly how Feynman's (5)/(6) treat the same rows -- the divergence from PQM's
    recipe is in the direction of MATCHING the run this is compared against.
    """
    labels = [T, F, T]
    row = synthetic_row("q", labels)
    assert row.z == 1 and row.recovery
    batch = collate([row], pad_id=0)
    derived = step_labels_from_z(batch).tolist()
    raw = [int(l) for l in labels]
    assert derived == [1, 0, 0]
    assert derived != raw, "if these ever agree, `z` or the derivation changed"


def test_label_source_raw_is_refused_with_the_reason():
    from feynman_prm.config import ConfigError
    from pqm_baseline.config import PQMConfig

    with pytest.raises(ConfigError, match="labels` column"):
        PQMConfig(label_source="raw")


# ---------------------------------------------------------------------------------------
# config, and the naming trap
# ---------------------------------------------------------------------------------------


def test_shipped_pqm_yaml_is_pqm_s_own_defaults():
    from pqm_baseline.config import PQM_YAML, load_pqm_config

    cfg = load_pqm_config(PQM_YAML)
    assert cfg.zeta == 4.0                 # train_main.py:181
    assert cfg.loss_type == "rank"         # train_main.py:182-183
    assert cfg.head_dropout == 0.1         # value_model.py:29-30
    assert cfg.head_init == "zero"
    assert cfg.label_source == "from_z"
    assert cfg.natural_tau_delta == 2.0    # +zeta/2 on the negated (eval) scale
    assert cfg.natural_tau_reward == -2.0


def test_unknown_pqm_key_is_a_hard_error():
    from feynman_prm.config import ConfigError
    from pqm_baseline.config import PQM_YAML, load_pqm_config

    with pytest.raises(ConfigError, match="unknown config key 'pqm.zetta'"):
        load_pqm_config(PQM_YAML, ["zetta=8"])


def test_set_overrides_are_partitioned_between_the_two_configs():
    """`--set pqm.zeta=8` must reach pqm.yaml and `--set run.name=x` must NOT -- that is what
    keeps `config/default.yaml` (strict-parsed) free of a `pqm:` block it would reject."""
    from pqm_baseline.config import split_overrides

    feynman, pqm = split_overrides(["run.name=x", "pqm.zeta=8", "losses.zeta=0.1"])
    assert feynman == ["run.name=x", "losses.zeta=0.1"]
    assert pqm == ["zeta=8"]


def test_pqm_zeta_and_feynman_losses_zeta_are_different_quantities(cfg):
    """The naming trap. `losses.zeta` is Feynman's (3) L_T backup weight; PQM's zeta is a
    reward offset in a different file, and the two are unrelated.

    The grep is on CODE only -- comments and docstrings are stripped the way
    `test_grep_invariants._code_lines` strips them -- because the point is that no module
    here READS `cfg.losses.zeta`. The single exception is train.py's launch guard, which
    exists to catch `--set losses.zeta=4` (a value at PQM's scale can only be that mistake,
    and it would be silently inert).
    """
    import re
    from pathlib import Path

    from pqm_baseline.config import PQM_YAML, load_pqm_config

    assert cfg.losses.zeta < 1.0                     # 0.05 / 0.1 -- a loss weight
    assert load_pqm_config(PQM_YAML).zeta == 4.0     # a reward offset

    root = Path(__file__).resolve().parents[1] / "pqm_baseline"
    readers = {}
    for path in sorted(root.rglob("*.py")):
        code = re.sub(r'"""..*?"""', "", path.read_text(), flags=re.S)
        code = re.sub(r"#.*", "", code)
        if "losses.zeta" in code:
            readers[path.name] = [l.strip() for l in code.splitlines() if "losses.zeta" in l]
    assert set(readers) == {"train.py"}, readers
    assert any("cfg.losses.zeta >= 1.0" in l for l in readers["train.py"]), readers["train.py"]


# ---------------------------------------------------------------------------------------
# diagnostics -- the signal the run is read against
# ---------------------------------------------------------------------------------------


def test_diagnostics_report_the_separation_and_the_anchors():
    rows = [synthetic_row("q", [T, T, T]), synthetic_row("q", [F, F])]
    batch = collate(rows, pad_id=0)
    zeta = 4.0
    # positives at +1 (above 0), negatives at -6 (below -zeta)
    rewards = torch.zeros(batch.n_states)
    step_labels = step_labels_from_z(batch)
    for r in range(batch.n_rows):
        rewards[int(batch.row_dst[r])] = 1.0 if step_labels[r] == 1 else -6.0

    rp, lp, hn = build_padded(batch, rewards)
    loss = pqm_ranking_loss(rp, lp, zeta, hn)
    info = pqm_diagnostics(batch, rewards, rp, lp, hn, loss, zeta)

    assert info["pqm/reward_pos_mean"] == pytest.approx(1.0)
    assert info["pqm/reward_neg_mean"] == pytest.approx(-6.0)
    assert info["pqm/reward_gap"] == pytest.approx(7.0)
    assert info["pqm/frac_pos_above_0"] == 1.0
    assert info["pqm/frac_neg_below_neg_zeta"] == 1.0
    assert info["pqm/steps"] == 5.0
    assert info["pqm/trajectories"] == 2.0
    assert info["pqm/has_neg_fraction"] == pytest.approx(0.5)
    # the correct trajectory's three steps all sit above -zeta/2 = -2, so nothing leaks
    assert info["pqm/good_steps_below_tau_natural"] == 0.0
    assert info["pqm/loss_at_zero_rewards"] == pytest.approx(loss_at_zero_rewards(lp, zeta))


def test_good_steps_below_tau_natural_counts_only_correct_trajectories():
    """The false-positive leak: a step of a CORRECT trajectory scored below `-zeta/2` is one
    eval would flag. Steps of incorrect trajectories are not in scope -- flagging those is the
    job."""
    rows = [synthetic_row("q", [T, T]), synthetic_row("q", [F, F])]
    batch = collate(rows, pad_id=0)
    rewards = torch.full((batch.n_states,), -10.0)      # everything below -zeta/2
    rp, lp, hn = build_padded(batch, rewards)
    info = pqm_diagnostics(batch, rewards, rp, lp, hn, pqm_ranking_loss(rp, lp, 4.0, hn), 4.0)
    assert info["pqm/good_steps_below_tau_natural"] == 1.0

    rewards = torch.zeros(batch.n_states)
    rp, lp, hn = build_padded(batch, rewards)
    info = pqm_diagnostics(batch, rewards, rp, lp, hn, pqm_ranking_loss(rp, lp, 4.0, hn), 4.0)
    assert info["pqm/good_steps_below_tau_natural"] == 0.0
    assert info["pqm/reward_std"] == 0.0                # a dead head reads exactly this


def test_diagnostics_keys_are_stable():
    """`metrics.jsonl` is plotted; a renamed key silently empties a panel."""
    batch = collate([synthetic_row("q", [T, F])], pad_id=0)
    rewards = torch.zeros(batch.n_states)
    rp, lp, hn = build_padded(batch, rewards)
    info = pqm_diagnostics(batch, rewards, rp, lp, hn, pqm_ranking_loss(rp, lp, 4.0, hn), 4.0)
    assert set(info) == {
        "pqm/loss",
        "pqm/loss_at_zero_rewards",
        "pqm/reward_pos_mean",
        "pqm/reward_neg_mean",
        "pqm/reward_gap",
        "pqm/frac_pos_above_0",
        "pqm/frac_neg_below_neg_zeta",
        "pqm/reward_min",
        "pqm/reward_max",
        "pqm/reward_std",
        "pqm/good_steps_below_tau_natural",
        "pqm/steps",
        "pqm/trajectories",
        "pqm/has_neg_fraction",
    }


# ---------------------------------------------------------------------------------------
# the checkpoint kwarg -- the one edit to existing code
# ---------------------------------------------------------------------------------------


def test_save_checkpoint_takes_a_prefixes_argument_and_defaults_unchanged(cfg, tmp_path):
    from feynman_prm.utils.checkpoint import head_state_dict, save_checkpoint

    model = _FakePQM()
    # the DEFAULT prefixes find no psi/phi here, which is exactly §14's empty-heads failure
    with pytest.raises(AssertionError, match="head state dict is EMPTY"):
        save_checkpoint(tmp_path / "a", model, cfg)

    path = save_checkpoint(
        tmp_path / "b", model, cfg, step=3, prefixes=("value_head.",)
    )
    payload = torch.load(path / "heads.pt", weights_only=False)
    assert sorted(payload["heads"]) == ["value_head.summary.bias", "value_head.summary.weight"]
    assert payload["step"] == 3
    assert set(head_state_dict(model, ("value_head.",))) == set(payload["heads"])


def test_load_value_head_refuses_a_checkpoint_with_no_head(cfg, tmp_path):
    """§14 trap 3 -- the stock PEFT path writes the adapter and silently drops the head.
    `utils.checkpoint.load_heads` checks against psi./phi./..., none of which a PQM checkpoint
    carries, so it would PASS on an empty payload. That is why this loader exists."""
    from pqm_baseline.model import load_value_head

    (tmp_path / "ckpt").mkdir()
    torch.save({"heads": {"psi.net.out.weight": torch.zeros(2, 2)}, "step": 1},
               tmp_path / "ckpt" / "heads.pt")
    with pytest.raises(RuntimeError, match="carries no `value_head"):
        load_value_head(_FakePQM(), tmp_path / "ckpt")


def test_load_value_head_round_trips(cfg, tmp_path):
    from feynman_prm.utils.checkpoint import save_checkpoint
    from pqm_baseline.model import load_value_head

    model = _FakePQM()
    with torch.no_grad():
        model.value_head.summary.weight.fill_(0.25)
    save_checkpoint(tmp_path / "ckpt", model, cfg, step=9, prefixes=("value_head.",))

    fresh = _FakePQM()
    info = load_value_head(fresh, tmp_path / "ckpt")
    assert info["step"] == 9
    assert torch.equal(fresh.value_head.summary.weight, model.value_head.summary.weight)


# ---------------------------------------------------------------------------------------
# the eval scale convention
# ---------------------------------------------------------------------------------------


def test_deltas_are_the_negated_rewards():
    """The one idea that makes the Feynman eval stack reusable. Feynman's Delta is
    "higher = worse", PQM's reward is "higher = better"; negating makes them the same object,
    so `predicted_label_from_deltas` and the F1 metric apply unchanged."""
    from pqm_baseline.loss import deltas_from_rewards

    assert deltas_from_rewards([1.0, -2.0, 0.5]) == [-1.0, 2.0, -0.5]
    # a reward of -3 at step index 1, threshold on the negated scale at tau = 2.0
    assert predicted_label_from_deltas(deltas_from_rewards([1.0, -3.0, 0.5]), tau=2.0) == 1
    assert predicted_label_from_deltas(deltas_from_rewards([1.0, 1.0, 0.5]), tau=2.0) == -1


def test_tau_verdict_is_multiplicative_not_additive():
    """§14's B12: an ADDITIVE slack on a quantity with no additive scale printed "the margin
    held" over a 3.4x overshoot. The PQM verdict is a ratio."""
    from pqm_baseline.eval_processbench import tau_verdict

    assert "anchors took" in tau_verdict(2.0, 2.0)
    assert "anchors took" in tau_verdict(3.0, 2.0)
    assert "**" in tau_verdict(9.0, 2.0)
    assert "**" in tau_verdict(0.2, 2.0)


def test_sweep_tau_puts_the_natural_value_in_the_grid(cfg):
    """The natural tau is reported as a CHECK, so it has to be an exact grid point rather than
    whichever linspace sample lands nearest it."""
    from pqm_baseline.eval_processbench import sweep_tau

    deltas = [[-1.0, 3.0], [-1.0, -1.0], [0.0, 5.0, -2.0]]
    labels = [1, -1, 1]
    out = sweep_tau(deltas, labels, cfg, natural_tau=2.0)
    assert out["calibration/expected_tau"] == 2.0
    assert any(row["tau"] == 2.0 for row in out["curve"])
    assert 0.0 <= out["calibration/f1"] <= 1.0
