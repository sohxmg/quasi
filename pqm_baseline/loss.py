"""PQM's Q-ranking loss, **ported verbatim**, plus the flat -> padded builder and the
diagnostics that make the run readable while it is running.

`pqm_ranking_loss` is a line-for-line port of `PRMTrainer.ranking_loss`,
`Process_Q_Model/train_main.py:61-78` (identical to the function the released README calls
`PQM_loss`). The only changes are mechanical: `self` is dropped and the module-global
`args.zeta` becomes an argument.

**Do not "clean it up".** Kept deliberately, because each one changes the number:

  * the `.flip(dims=[-1])` on the negatives (line 63) is vestigial -- the very next line
    sums over that axis -- but it stays, so a diff against the authors' file is empty;
  * the `1e-5` in the denominator (line 74) is not a stabiliser you can drop: it shifts every
    term, and the analytic init value below is derived WITH it;
  * the prepended virtual slot (lines 66-72) whose label is `has_neg` is the "no negative
    beats the reference" term, and it is what anchors positives above `exp(0) = 1` in
    ABSOLUTE terms rather than only relative to each other.

`tests/test_pqm.py::test_ranking_loss_matches_the_authors_function` pins this against a copy
of `train_main.py:61-78` that lives in the test file, so a future simplification fails loudly
instead of quietly moving the baseline.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor

from feynman_prm.data.collate import Batch

# `train_main.py:230` pads `step_labels` with -100. Real labels are 1 (good) / 0 (bad); the
# loss reads `labels == 1` and `labels == 0`, so -100 is inert in both.
PAD_LABEL = -100


# ---------------------------------------------------------------------------------------
# the port
# ---------------------------------------------------------------------------------------


def pqm_ranking_loss(
    rewards: Tensor,
    labels: Tensor,
    zeta: float,
    has_neg: Optional[Tensor] = None,
) -> Tensor:
    """`Process_Q_Model/train_main.py:61-78`, verbatim.

    `rewards` (B, T) fp32, `labels` (B, T) int64 in {1, 0, -100}. `has_neg` (B,) 0/1 is
    PQM's own collator output (`train_main.py:225`); left None it is recomputed the same way
    from the labels, which keeps the function self-contained without changing its value.
    """
    if has_neg is None:
        # `train_main.py:225`: `has_neg.append(1 if 0 in d['labels'] else 0)` -> an int64 0/1
        # tensor. The dtype matters: line 76 cats it onto `labels`.
        has_neg = (labels == 0).sum(-1).bool().to(labels.dtype)

    # ---- BEGIN verbatim (train_main.py:62-78) ------------------------------------------
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
    # ---- END verbatim ------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# the analytic init value (§18)
# ---------------------------------------------------------------------------------------

# The denominator epsilon of the ported line 74. It is part of the closed form, not noise.
_EPS = 1e-5


def loss_at_zero_rewards(labels: Tensor, zeta: float) -> float:
    """The EXACT value of `pqm_ranking_loss(zeros_like(rewards), labels, zeta)`.

    Derivation. With every reward 0 (which `head_init: zero` guarantees, and which the
    padding zero-fill guarantees for the padded slots too):

        pos_rewards_exp  = 1 at a positive slot, 0 elsewhere
        neg_reward_sum   = n_neg * e^zeta
        reward_exp_cur   = 1 everywhere, and 1 in the prepended virtual slot
        pos_rewards_cumsum[0]   = 0                      (the virtual slot)
        pos_rewards_cumsum[i+1] = 1 + #{positives strictly before i}

    so, with `loss_j = -log(cur_j / (cur_j + cumsum_j + neg_sum + eps)) = log(...)`:

        virtual slot (counted only when has_neg):  log(1 + n_neg*e^zeta + eps)
        the m-th positive, m = 0 .. n_pos-1:       log(2 + m + n_neg*e^zeta + eps)

    and the trajectory's term is their sum over `1{has_neg} + n_pos`. The batch value is the
    mean over trajectories.

    **The `2 + m`, not `1 + m`.** The positive slot's denominator carries BOTH its own
    `cur_j = 1` and the cumsum's leading `1` (the `exp(0)` prepended at line 66), which is a
    different `1` from the virtual slot's. `1 + m` is off by exactly one unit of `e^0` and
    the discrepancy is largest at low `n_neg` -- i.e. on the trajectories that carry the
    clean half of F1. It is checked against the ported function on ragged fixtures in
    `tests/test_pqm.py::test_analytic_init_value_is_exact`, which is the point: §18's rule is
    that an ASSUMED init value is how two regressions got through, so this one is derived and
    pinned rather than eyeballed.

    Pure Python on CPU counts -- it runs once per launch and it must not be a second
    implementation of the tensor code, or it would drift with it.
    """
    n_pos = (labels == 1).sum(-1).tolist()
    n_neg = (labels == 0).sum(-1).tolist()
    exp_zeta = math.exp(zeta)

    totals = []
    for pos, neg in zip(n_pos, n_neg):
        base = neg * exp_zeta + _EPS
        terms = []
        if neg > 0:                                   # has_neg -> the virtual slot counts
            terms.append(math.log(1.0 + base))
        terms.extend(math.log(2.0 + m + base) for m in range(pos))
        if not terms:
            raise ValueError(
                "a trajectory with no positive and no negative step: `build_sequence` "
                "refuses a zero-step trajectory, so this cannot come from the parquet"
            )
        totals.append(sum(terms) / len(terms))
    return sum(totals) / len(totals)


# ---------------------------------------------------------------------------------------
# flat (Feynman) -> padded (PQM)
# ---------------------------------------------------------------------------------------


def step_labels_from_z(batch: Batch) -> Tensor:
    """(R,) int64 in {1, 0}, one per source row, from the trajectory's `z`.

    §6.1: `completions[k]` takes `s_k -> s_{k+1}` and `labels[k]` describes it, so the row
    with `row_step = i` (1-based) carries `labels[i-1]`. `z` is the FIRST error index,
    0-based, -1 for a fully correct trajectory:

        label(i) = 1  iff  z == -1  or  (i - 1) < z

    This is `label_source: from_z`. It MONOTONISES the 1.48% of Math-Shepherd trajectories
    with a `False -> True` recovery -- a `[T, F, T]` trajectory has `z = 1` and comes back
    `[1, 0, 0]`, not `[1, 0, 1]`. That is exactly how Feynman's (5)/(6) treat the same rows
    (§16.15), which is the reason it is the default here: the divergence from PQM's recipe is
    in the direction of MATCHING the run this is compared against, not of favouring either
    side. `tests/test_pqm.py::test_label_derivation_from_z` pins the difference.
    """
    z = batch.traj_z.index_select(0, batch.row_traj)             # (R,)
    good = (z == -1) | ((batch.row_step - 1) < z)
    return good.long()


def build_padded(batch: Batch, rewards: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """`(rewards_pad, labels_pad, has_neg)` at (B, T_max), from the flat Feynman batch.

    `rewards` is (S,) -- one per state `s_0 .. s_T` of every trajectory. PQM's per-step
    reward for step `i` is read at the state that step LANDS IN, `s_i`, which is
    `batch.row_dst`. `s_0` carries no step and never enters the loss.

    **The zero-fill is load-bearing, not tidiness.** `torch.where(labels == 1, rewards.exp(),
    0)` evaluates `exp()` on EVERY slot including the padding, before the `where` selects.
    An uninitialised padded slot (`new_empty`) can hold anything, `exp()` of it overflows to
    `inf`, and `where`'s backward then multiplies `0 * inf` into a NaN that no forward value
    would ever reveal. `tests/test_pqm.py::test_padding_never_produces_nan` is that trap.
    """
    B = batch.n_traj
    Tmax = int(batch.traj_T.max())
    rewards_pad = rewards.new_zeros(B, Tmax)                              # ZEROS, not empty
    labels_pad = torch.full((B, Tmax), PAD_LABEL, dtype=torch.long, device=rewards.device)

    b, t = batch.row_traj, batch.row_step - 1                             # row_step is 1-based
    rewards_pad[b, t] = rewards.index_select(0, batch.row_dst)
    labels_pad[b, t] = step_labels_from_z(batch)

    has_neg = (labels_pad == 0).sum(-1).bool().to(labels_pad.dtype)
    return rewards_pad, labels_pad, has_neg


# ---------------------------------------------------------------------------------------
# diagnostics (§10)
# ---------------------------------------------------------------------------------------


@torch.no_grad()
def pqm_diagnostics(
    batch: Batch,
    rewards: Tensor,
    rewards_pad: Tensor,
    labels_pad: Tensor,
    has_neg: Tensor,
    loss: Tensor,
    zeta: float,
) -> dict[str, float]:
    """Everything the run is read against, logged every `log_every` steps.

    §10's discipline: the old project's failure was invisible for a full cycle because none
    of these existed. Read them in this order:

      pqm/reward_gap                    THE SIGNAL. Eval thresholds exactly this separation;
                                        if it does not open, nothing downstream can work.
      pqm/frac_pos_above_0              the loss's two ABSOLUTE anchors. A global tau is only
      pqm/frac_neg_below_neg_zeta       meaningful across questions if these move.
      pqm/reward_std                    ~0 is a dead head (B10a's analogue).
      pqm/good_steps_below_tau_natural  false-positive leak on CORRECT trajectories -- the
                                        §7.6.6 / diagnostic #14 analogue, and the single best
                                        predictor of the clean half of F1.
      pqm/loss vs pqm/loss_at_zero_rewards
                                        the loss against its own analytic chance anchor
                                        (§10 #19's contrastive-loss-vs-chance rule).
    """
    real = labels_pad != PAD_LABEL
    pos = labels_pad == 1
    neg = labels_pad == 0
    r = rewards_pad

    def mean_where(mask: Tensor) -> float:
        n = int(mask.sum())
        return float(r[mask].mean()) if n else float("nan")

    def frac(mask: Tensor, cond: Tensor) -> float:
        n = int(mask.sum())
        return float((cond & mask).sum()) / n if n else float("nan")

    pos_mean, neg_mean = mean_where(pos), mean_where(neg)
    real_rewards = r[real]

    # Rows of CORRECT trajectories only, on the flat side -- `traj_correct` is per
    # trajectory, so it is gathered through `row_traj`. The natural tau in REWARD units is
    # `-zeta/2` (config.py), and a good step below it is a step eval would flag.
    correct_row = batch.traj_correct.index_select(0, batch.row_traj)
    step_rewards = rewards.index_select(0, batch.row_dst)
    n_good = int(correct_row.sum())
    good_below = (
        float((step_rewards[correct_row] < -zeta / 2.0).sum()) / n_good
        if n_good
        else float("nan")
    )

    return {
        "pqm/loss": float(loss),
        "pqm/loss_at_zero_rewards": loss_at_zero_rewards(labels_pad, zeta),
        "pqm/reward_pos_mean": pos_mean,
        "pqm/reward_neg_mean": neg_mean,
        "pqm/reward_gap": pos_mean - neg_mean,
        "pqm/frac_pos_above_0": frac(pos, r > 0.0),
        "pqm/frac_neg_below_neg_zeta": frac(neg, r < -zeta),
        "pqm/reward_min": float(real_rewards.min()) if real_rewards.numel() else float("nan"),
        "pqm/reward_max": float(real_rewards.max()) if real_rewards.numel() else float("nan"),
        "pqm/reward_std": float(real_rewards.std()) if real_rewards.numel() > 1 else 0.0,
        "pqm/good_steps_below_tau_natural": good_below,
        "pqm/steps": float(int(real.sum())),
        "pqm/trajectories": float(batch.n_traj),
        "pqm/has_neg_fraction": float(has_neg.float().mean()),
    }


def summarise_labels(labels_pad: Tensor) -> dict[str, float]:
    """`(n_pos, n_neg, has_neg)` counts, for the launch log's init block."""
    return {
        "positives": float(int((labels_pad == 1).sum())),
        "negatives": float(int((labels_pad == 0).sum())),
        "trajectories_with_a_negative": float(int((labels_pad == 0).any(-1).sum())),
        "padded_slots": float(int((labels_pad == PAD_LABEL).sum())),
    }


def rewards_at_steps(batch: Batch, rewards: Tensor) -> Tensor:
    """(R,) the per-step reward `r_i`, read at `s_i`. The one index mapping eval shares with
    training, and the one §7.6 records as invisible in every loss curve when it is wrong."""
    return rewards.index_select(0, batch.row_dst)


def deltas_from_rewards(step_rewards: Sequence[float]) -> list[float]:
    """**The idea that makes the whole Feynman eval stack reusable: PQM's score enters as
    `-r_i`.**

    Feynman's `Delta_i` is "higher = worse"; PQM's reward is "higher = better". Negating
    makes the two the same object, so `predicted_label_from_deltas`, `processbench_metrics`,
    the leak split and the tau sweep all apply UNCHANGED, and the `deltas.npz` schema is
    identical -- which means `scripts/analyze_deltas.py` and `scripts/error_rank.py` run on a
    PQM checkpoint for free.
    """
    return [-float(r) for r in step_rewards]
