# experiments.md — the run log

One section per training run, appended as they finish. **This file records what was
measured and what it means; `CLAUDE.md` is the locked spec and wins on any disagreement.**

House rules, same as §17:

* Every number is a **window mean over the last 20% of logged steps**, not a last-row
  reading — each of these is batch-noisy at ±0.3.
* Tag every claim **MEASURED** / **DERIVED** / **HYPOTHESIS**. A number with no tag is a
  number nobody has to defend.
* Corrections go in `§C` and stay there. A withdrawn claim that is quietly deleted is a
  claim that gets rediscovered.

---

## Run index

| dir | date | τ_NCE | L_good form | mask | ζ | steps | val F1 † | status |
|---|---|---|---|---|---|---|---|---|
| `phase1_lambda_good_0_baseline` | 2026-07-27 | 1.0 | — (λ=0) | off | 0.05 | 970 | 0.456 @750 | done |
| `phase1` | 2026-07-29 | 1.0 | relu | off | 0.05 | 1460 | **0.5313** | done, **ProcessBench F1 0.241** (0.235 clean), the baseline |
| `phase1_nce_temp_relu2` | 2026-08-04 | √512 | relu_squared | **on** | 0.1 | 1460 | **0.5597** | done, **ProcessBench F1 0.259** (0.242 clean) — **best**, §6, §7 |
| `phase1_mask_relu2` | 2026-08-04 | 1.0 | relu_squared | **on** | 0.05 | 1460 | **0.5074** | done, ruler fixed, **val F1 lost**, §4/§6 |

† `val_f1.py` at each checkpoint's own fitted τ. **A CEILING, not a result** — it is handed a real
terminal that eval never has (§9.5). Comparable across rows: same 400 val questions, same 4,549
trajectories, τ refit per checkpoint. **Not** comparable to the ProcessBench column.

"clean" in the status column is the mean over the four subsets with locked #5's **clean 413**
`math` substituted for the full 1,000. The other three subsets have 0% overlap (§4.3(b)), so it
is a partial correction — but it halves the gap between the two runs (+0.018 → **+0.007**) and
must be quoted alongside the headline (§7.6).

`runs/phase1/` is the reference: every §9.3.1 F1 and §9.7 rank number was measured on it.

---

## 1. `phase1` — the baseline

λ_good = 1.0, relu, no mask, τ_NCE = 1.0, ζ = 0.05, `n_questions` 34,650, 1,460 steps.
Phase 2 and ProcessBench both done (§18.6, §18.7): **mean F1 0.241**.

**MEASURED** (window means, `metrics.jsonl`, read 2026-08-04):

| key | value |
|---|---|
| `backup/delta_mean` (the ruler) | 0.4432 |
| `backup/dist_mean` | 9.1741 |
| `nce/pos_dist` | 2.4072 |
| `nce/neg_dist` | 9.1969 |
| `nce/logit_std` | 4.1153 |
| `nce/accuracy_within_question` | 0.3313 = **10.8× chance** (chance 1/33) |
| `nce/categorical_accuracy_backward` | 0.2836 |
| `probe02/delta_good_mean` | −0.3213 |
| `probe03/gap` | 2.0049 |
| `probe14/delta_boundary/mean` | 1.6836 |
| `probe14/…/frac_above_natural` | 0.1776 |
| `probe14/…/p99` | 2.9965 |
| `invariance/residual_diagonal` | 0.2268 |
| `good/above_target_fraction` | 0.6075 |
| `step/loss` | 1.0096 |
| `backup/loss` | −3.4663 |
| `train/grad_norm` (pre-clip) | 6.3901 |

The 10.8× reproduces §9.8.2's recorded 10.6× on the same series — which is what says the
re-read is reading the right tensor.

---

## 2. `phase1_nce_temp_relu2` — four changes at once (wandb `m1pqt8ot`)

Launched `bash scripts/train.sh --set losses.zeta=0.1`. Changes vs `phase1`:
**τ_NCE 1.0 → √512**, **L_good relu → relu_squared**, **`nce_mask_nearer_same_traj` on**,
**ζ 0.05 → 0.1**. `n_questions` unchanged, so the selection SHA still matches §8.2.

147 logged points, last step 1460. Phase 2 and eval **not run** — see the verdict below.

**MEASURED** (window means, with `phase1` alongside):

| key | phase1 | this run | ratio |
|---|---|---|---|
| `backup/delta_mean` (ruler) | 0.4432 | 0.4093 | **0.92** |
| `backup/dist_mean` | 9.1741 | 1.5899 | **0.17** |
| `nce/pos_dist` | 2.4072 | 1.1854 | 0.49 |
| `nce/neg_dist` | 9.1969 | 1.5922 | **0.17** |
| `nce/logit_std` | 4.1153 | 0.0372 | 0.01 |
| `nce/logit_std × τ` | 4.1153 | 0.8423 | **0.20** |
| `nce/accuracy_within_question` | 0.3313 (10.8×) | 0.0956 (**3.1×**) | 0.29 |
| `nce/categorical_accuracy_backward` | 0.2836 | 0.0140 | 0.05 |
| `nce/argmax_in_nearer_set` | 0.369 † | 0.0139 | 0.04 |
| `probe02/delta_good_mean` | −0.3213 | −0.3061 | 0.95 |
| `probe03/gap` | 2.0049 | 1.0857 | 0.54 |
| `probe14/delta_boundary/mean` | 1.6836 | 0.7796 | 0.46 |
| `probe14/…/frac_above_natural` | 0.1776 | 0.0606 | 0.34 |
| `probe14/…/p99` | 2.9965 | 1.3222 | 0.44 |
| `invariance/residual_diagonal` | 0.2268 | 0.0374 | 0.16 |
| `good/above_target_fraction` | 0.6075 | 0.8327 | 1.37 |
| `step/loss` | 1.0096 | 1.2198 | 1.21 |
| `backup/loss` | −3.4663 | −0.2764 | 0.08 |
| `train/grad_norm` | 6.3901 | 3.3740 | 0.53 |

† `phase1`'s `argmax_in_nearer_set` is `scripts/nce_preflight.py` on `runs/phase1/final`
(§9.9.6), not a window mean — 0.369 against a null of 0.00189, 195×.

### 2.1 The geometry collapsed — MEASURED

`backup/dist_mean` and `nce/neg_dist` both fell **5.8×**. `neg_dist / pos_dist` went
**3.82 → 1.34**: negatives used to sit ~4× further from the goal than positives and now
sit barely further at all. `logit_std × τ` fell 5× as well, so this is a real contraction
in **distance space**, not a temperature artifact — the config's own instruction to read
`logit_std × τ` rather than `logit_std` is what makes that legible.

`nce/accuracy_within_question` 10.8× → **3.1×** chance is the same fact in ranking terms.

**Attribution: τ_NCE.** HYPOTHESIS, but a strong one — ① is the only term with an
unbounded push on negative distances (§9.9.7 has `neg_dist` climbing 1.1 → 9.1 on
`phase1`), and it is the term that was muted 22.6×. Testable by `phase1_mask_relu2`.

### 2.2 relu_squared worked — MEASURED

`frac_above_natural` **0.178 → 0.061** (target 0.05) and `p99` **3.00 → 1.32**. This
survives renormalisation by each run's own ruler — `p99` in ruler units went
**6.76 → 3.23** — so it is not an artifact of the collapsed scale. Keep it.

`good/above_target_fraction` rose 0.61 → 0.83, which is consistent rather than
contradictory: the square prices a violator by how far out it is, so it pulls the extreme
tail in hard and leaves a dense band sitting just above `c`.

### 2.3 Everything in ruler units — DERIVED

Each run's quantities divided by its own measured `backup/delta_mean`:

| | target | phase1 | this run |
|---|---|---|---|
| ruler | 1.00 | 1.00 | 1.00 |
| good Δ (`probe02`) | 1.00 | 0.73 | 0.75 |
| error Δ (`delta_boundary`) | 2.00 | **3.80** | **1.91** |
| gap (`probe03`) | 3.00 | 4.52 | 2.65 |
| good tail (`p99`) | — | 6.76 | 3.23 |

Gap target is `m − (−log γ)` = 2.079 absolute = 3.0 rulers; `run_report.py`'s 1.8 is a
looser empirical guard from §7.12.

The error step landing at **1.91 rulers against a `margin_steps` of 2.0** says this run's
margin is very close to what the config asks for, and that `phase1` at 3.80 rulers was the
one overshooting. The absolute collapse of `probe03/gap` (2.00 → 1.09) is still real and
still matters, because eval applies an **absolute** τ.

### 2.4 The mask is unreadable — MEASURED, and predicted before launch

`argmax_in_nearer_set` fell 0.369 → 0.0139, but `accuracy_within_question` fell by a
similar factor over the same window. The argmax moved off the nearer-set because the
argmax is near-random everywhere, not because the mask fixed the false negatives.
`probe02/delta_good_mean`, §9.9's designated judge, is −0.306 against `phase1`'s −0.321 —
unchanged.

This is exactly §9.10.3's self-defeating pair, flagged before launch and accepted as a
decision: the mask is a repair **to** L_NCE, shipped in the run that muted L_NCE 22.6×.
**Zero information about the mask.**

### 2.5 Verdict

Four changes: one worked (relu²), one caused the damage (τ_NCE), one is unreadable (the
mask), one is unresolved (ζ, §3.2). Not evaluated on ProcessBench — the geometry collapse
means an F1 here would measure the contraction, not the loss set.

`scripts/val_f1.py --checkpoint runs/phase1_nce_temp_relu2/final` is still worth ~15 min
of GPU whenever it is free: it is the only number that converts any of this into F1, and
it needs no phase 2.

---

## 3. Standing findings

### 3.1 The ruler cannot reach 0.693 — DERIVED

`temporal.py:66`, exponential branch, `Next` detached:

```
div        = γ·exp(Dist − Next) − Dist
∂div/∂Dist = γ·exp(δ) − 1            → zero exactly at δ = −log γ = 0.693
```

So 0.693 is L_T's stationary point **only if nothing else touches `Dist`**. In the real
loss set, at equilibrium:

```
ζ·(γ·exp(δ) − 1) + G = 0,    G = every other term's gradient on Dist
```

Every other term that touches `Dist` wants it **smaller** — ① pulls positives in, ⑥ pushes
`d(ψ_i, g)` down, and with `residual_diagonal` low, ψ ≈ φ so ⑥'s pull lands on `Dist`
directly. **`G > 0` forces `δ < 0.693` strictly, at any finite ζ.** The shortfall measures
`G`.

Backing `G` out of the two runs (ignores the `diag_backup` mix and the clip, so read the
ratio and not the absolute):

| | ζ | δ | implied G |
|---|---|---|---|
| `phase1` | 0.05 | 0.4432 | 0.0111 |
| `phase1_nce_temp_relu2` | 0.10 | 0.4093 | **0.0247** |

With `G` held at `phase1`'s value, ζ = 0.1 predicts **δ = 0.576**. Observed 0.409, so
something roughly doubled its pull on `Dist` at the same time.

~~**Suspect: relu_squared.**~~ **WITHDRAWN 2026-08-04 by §4.3 — see §C5.** `relu_squared` is
**on** in `phase1_mask_relu2`, where `G` fell **2.8×** against `phase1` at matched ζ *and*
matched ambient scale. The hinge form is not what raised `G`.

**Third point, and the only pair that is safe to compare** (`phase1_nce_temp_relu2` sits in a
5.4× smaller space, so its row is a scale artifact as much as anything else):

| | ζ | δ | implied G | `dist_mean` |
|---|---|---|---|---|
| `phase1` | 0.05 | 0.4293 | 0.01159 | 9.174 |
| `phase1_nce_temp_relu2` | 0.10 | 0.4093 | 0.02471 | **1.590** |
| `phase1_mask_relu2` | 0.05 | 0.6090 | **0.00407** | 8.589 |

**The equation's standing status is: consistent, twice, and never yet used to predict.** It has
two matched-scale points now, so `abl_zeta02` at ζ = 0.2 can be given a numeric δ before launch
— which is the only thing that would promote it from a bookkeeping identity to a model.

**HYPOTHESIS for what lowered `G`: the mask**, per §9.9.3's pre-registered mechanism. Untested.
`abl_relu` separates it from `relu_squared`.

### 3.2 ζ has never been isolated — status: UNKNOWN, but EXONERATED for the ruler decay

**Added 2026-08-04.** §16.23 named ζ the leading suspect for `backup/delta_mean` decaying to
~0.49 on both prior runs. **`phase1_mask_relu2` stopped the decay with ζ held at 0.05** (§4.3),
so ζ is not the cause and raising it is no longer the indicated fix. What ζ *is* worth remains
unknown for the reasons below.

Both earlier calls in this session were wrong; see §C1 and §C2. What is true:

* Absolute ruler went 0.4432 (ζ=0.05) → 0.4093 (ζ=0.1), i.e. slightly **worse**.
* §9.10.4 prices ζ = 0.1 as still 5th of 5 in gradient magnitude, and 0.2 as the
  principled stopping point.
* Every ζ observation to date is confounded with τ_NCE, the L_good form and the mask.

Do not quote ζ as helping or as a null until `abl_zeta02` runs against a prediction.

### 3.5 `probe03/gap` is the wrong statistic — it is `gap / spread` that tracks F1

**HYPOTHESIS, opened 2026-08-04 on three points.** §4.5 read `probe03/gap` 2.00 → 1.10 as a
regression and §6 then measured `phase1_nce_temp_relu2` — `gap` 1.09, i.e. equally "regressed" —
scoring the **best** val F1 of the three. `gap` alone does not order them. A d′-shaped quantity
does, using series that are already logged:

```
d' = ( probe14/delta_boundary/mean − probe14/delta_good_of_correct/mean )
     ────────────────────────────────────────────────────────────────────
                probe14/delta_good_of_correct/std
```

| run | error Δ | good Δ mean | good Δ std | **d′** | `probe03/gap` | **val F1** |
|---|---|---|---|---|---|---|
| `phase1_nce_temp_relu2` | 0.7796 | −0.3339 | 0.4881 | **2.281** | 1.0857 | **0.5597** |
| `phase1` | 1.6836 | −0.3557 | 0.9973 | **2.045** | 2.0049 | **0.5313** |
| `phase1_mask_relu2` | 0.6929 | −0.4258 | 0.6443 | **1.736** | 1.1044 | **0.5074** |

**d′ orders val F1 perfectly and `gap` orders it backwards.** It also explains §4.5: this run
compressed Δ-space ~2× (`std` 1.00 → 0.64), which shrinks the *numerator* and the *denominator*
together — so `gap` halving is not by itself a loss, and the run only lost because the numerator
shrank slightly more than the spread did.

**Three points, post hoc, and a derived quantity — this is a hypothesis and nothing more.** It
is cheap to falsify: `abl_relu` produces a d′ from `metrics.jsonl` alone, so **write the
predicted val F1 down before running `val_f1.py` on it.** If it holds, `run_report.py`'s 1.8
guard on `gap` should become a guard on d′, and §16.3's "the gap collapsing means the error
signal is flattening" needs the same correction — a gap that falls with the spread is not
flattening anything.

### 3.3 `run_report.py`'s NCE comparator was wrong — B12 again, FIXED 2026-08-04

`scripts/run_report.py:135-137` multiplies `nce/categorical_accuracy_backward` by
`nce/negatives_per_column` (chance `1/R` ≈ 1/323) and compares the product against
**10.6×, which §9.8.2 measured on `nce/accuracy_within_question`** (chance
`1/(1+n_same)` ≈ 1/32). Different statistic, different pool, tenfold different chance
level. The `< 4.0` abort threshold inherits the same error, so
`phase1_nce_temp_relu2` printed "4.5×" and passed a guard it was never eligible for; the
like-for-like number is **3.1× against 10.8×**.

Same shape as B12 (an additive check on a multiplicative quantity) and as §10.1.1's
underived `< 0.3`.

> **FIXED 2026-08-04, `scripts/run_report.py:132-158`, before the `phase1_mask_relu2` read.**
> Each statistic now scores against **its own** chance level: `accuracy_within_question`
> against `1/(1 + negatives_same_question)` and `categorical_accuracy_backward` against `1/R`,
> with the printed line saying in as many words that the two multiples are not comparable. The
> gauge row in `GUARDS` moved to `accuracy_within_question` and the abort threshold is `< 6×`
> on that key. Verified against `runs/phase1`: it prints **10.8× (chance 1/33)**, reproducing
> §9.8.2's 10.6× on the same series, where the old code printed the product against 1/324.
>
> **The residual, and it is not this bug:** `accuracy_within_question` is computed on the
> **masked** logits, so it is inflated whenever a mask is on (§4.4). The report cannot fix
> that; `nce_preflight.py` is the instrument that can.

### 3.4 `train/grad_norm` has been binding on every run — OPEN

6.39 on `phase1`, 3.37 on `phase1_nce_temp_relu2`, **6.62 on `phase1_mask_relu2`**, against
`train.grad_clip = 1.0`. §14: far above it all run means the clip has become an LR rescale
rather than a guard. Left at 1.0 for all three so they share the same rescale and stay
comparable — but the τ = 1.0 prediction landed (3.37 → 6.62) and the item is still open. **It
is now the oldest untouched knob in the loss set, and it rescales every term equally**, so it
cannot explain any of §4's per-term movements.

---

## 4. `phase1_mask_relu2` — launched 2026-08-04 (wandb `jr3hpurd`)

```bash
bash scripts/train.sh --set losses.nce_temperature=1.0 --set run.name=phase1_mask_relu2
```

τ_NCE **1.0**, relu_squared, mask **on**, ζ **0.05**, `n_questions` 34,650 (SHA unchanged,
`prepare_data.py` not re-run). **Exactly two changes vs `phase1`: relu² and the mask.**

ζ is held at 0.05 deliberately, and not only for change-count discipline: §9.9 names
`probe02/delta_good_mean` as the mask's judge, and ζ sets that same quantity (the good-step
Δ is set by L_T, not by `m`). Moving both makes a good result unattributable to either.

**Predictions, recorded before the run finishes:**

| | expect | reads as |
|---|---|---|
| `nce/accuracy_within_question` | ≥ 10× chance | the run is readable at all — gate everything else on this |
| `neg_dist / pos_dist` | ~3.8 | geometry restored, §2.1 confirmed |
| `nce/argmax_in_nearer_set` | **below** 0.369 at equal-or-better NCE accuracy | the mask worked. Falling *with* accuracy is §2.4 again |
| `probe02/delta_good_mean` | better than −0.3213 | §9.9's judge, first unmuted reading |
| `frac_above_natural` / `p99` | ~0.06 / ~1.3 | relu² holds its win at full geometry |
| `backup/delta_mean` | ~0.44, possibly lower | **not a failure** — §3.1, ⑥ adds to G |
| `invariance/residual_diagonal` | ≤ 0.15 by step 200 | the live guard. 0.037 was measured in a collapsed space, so ⑥ may bill L_I after all |

Live guard: if `residual_diagonal` is above 0.15 at step ~200 or still rising, kill and
relaunch at `--set losses.lambda_good=0.5`.

### 4.1 Launch checks — all four pass, MEASURED

| | |
|---|---|
| `optimizer_steps` | **1464** (§11.1; ~106 would mean the `n_questions`/`grad_accum` regression) |
| `sequences_total` | 149,351 · `sequences_per_batch_mean` **51.008**, matching the config's own measured note |
| `good_margin` | **−0.6931**, negative — the only print that catches the flipped sign (§7.12) |
| memory probe | 12.128 GB peak on the longest batch (index 1391, run first) of 16 GB |
| init expected/actual | nce 6.254/6.278 · inv 10.597/10.594 · step 1.903/1.980 · backup **+4.55** (positive, must go negative) |
| `nce_temperature` | **1.0** — the `--set` landed |
| `logit_std` | 0.335, non-zero → not B10a |
| `good_form` / `lambda_good` | `relu_squared` / 1.0 |
| sampler | `distinct_z` 25.9 (21 = old caps) · `step_pairs` 58.4 (35 = `2c+1i`, 96 = quotas) · Q 11.83 |
| trainable | 22,407,168 — lora 392, ψ 14, φ 14, **goal_head 0** (phase 2 only) |

Two mild deviations from §18, neither blocking:

* `linear_branch_fraction` **0.608** where §18 predicts ~1.0 at init. L_I is at 10.59 as
  expected so the ψ/φ gap is present; δ just averages nearer `t = 3.689` than §18's
  estimated 9.8. Resolves as L_I closes — the real check is `backup/loss` going negative
  by ~100 steps.
* `step_pairs` 58.4 / Q 11.83 against §18's 64 / 12.9. Same ballpark, and both failure
  signatures (35, 96) are absent.

Step 1: `L=19.12 nce=6.21 inv=10.42 bkp=5.30 step=2.22 good=0.63 gap=−0.29`. The negative
gap is correct at init — nothing has trained yet.

### 4.2 Results — 1,464 steps in 4h01, all four predictions read

**MEASURED** (window means over the last 20% of 147 logged points, `phase1` alongside):

| key | phase1 | this run | ratio | prediction |
|---|---|---|---|---|
| `backup/delta_mean` (the ruler) | 0.4432 | **0.6206** | **1.40** | "~0.44, possibly lower" — **beaten** |
| `backup/dist_mean` | 9.1741 | 8.5889 | 0.94 | geometry intact |
| `nce/pos_dist` | 2.4072 | 2.2339 | 0.93 | |
| `nce/neg_dist` | 9.1969 | 8.6247 | 0.94 | |
| **`neg_dist / pos_dist`** | **3.821** | **3.861** | **1.01** | "~3.8" — **hit exactly** |
| `nce/logit_std × τ` | 4.1153 | 3.7993 | 0.92 | |
| `nce/accuracy_within_question` | 0.3313 (10.8×) | 0.4032 (**12.9×**) | 1.22 | "≥10× chance" — **passes, but see §4.4** |
| `nce/categorical_accuracy_backward` | 0.2836 | 0.3342 | 1.18 | |
| `nce/argmax_in_nearer_set` | 0.369 † | 0.3346 | 0.91 | "below 0.369" — **the prediction was wrong, §4.4** |
| `probe02/delta_good_mean` | −0.3213 | **−0.4115** | 1.28 | "better than −0.3213" — **hit** |
| `probe14/…/frac_above_natural` | 0.1776 | **0.1082** | 0.61 | "~0.06" — halfway, missed |
| `probe14/…/p99` | 2.9965 | **1.4904** | 0.50 | "~1.3" — near |
| `probe14/…/p90` | 0.8097 | 0.3452 | 0.43 | |
| `probe14/…/std` | 0.9973 | 0.6443 | 0.65 | |
| `good/delta_max` | 4.7330 | 2.3127 | 0.49 | |
| `invariance/residual_diagonal` | 0.2268 | **0.2239** | 0.99 | "≤0.15" — wrong bar (§17), unchanged is the real read |
| **`probe03/gap`** | **2.0049** | **1.1044** | **0.55** | not predicted — **the regression** |
| **`probe14/delta_boundary/mean`** | **1.6836** | **0.6929** | **0.41** | not predicted — **the regression** |
| `step/loss` | 1.0096 | 1.2880 | 1.28 | |
| `good/above_target_fraction` | 0.6075 | 0.6603 | 1.09 | |
| `train/grad_norm` (pre-clip) | 6.3901 | 6.6200 | 1.04 | still binding, §3.4 |

† `phase1`'s is `nce_preflight.py` on `runs/phase1/final` (§9.9.6), not a window mean. The key
was added 2026-08-04 and `phase1`'s `metrics.jsonl` does not carry it.

**Launch checks all reproduced §4.1**: 1464 optimizer steps, 149,351 sequences at 51.008 per
batch, 12.128 GB peak, `good_margin` −0.6931, 22,407,168 trainable with `goal_head 0`.

### 4.3 The ruler recovered, and ζ is exonerated — MEASURED

`backup/delta_mean` per 300-step band, against a target of 0.693:

| steps | 0–300 | 300–600 | 600–900 | 900–1200 | 1200–1460 |
|---|---|---|---|---|---|
| `phase1` | 1.0741 | 0.7268 | 0.5636 | 0.4379 | **0.4293** |
| this run | 1.0821 | 0.8120 | 0.7158 | 0.6191 | **0.6090** |

**`phase1` was still falling at the last band; this run flattens.** 0.6191 → 0.6090 is a
plateau, at a 12% shortfall against `phase1`'s 38% and §16.23's 0.49.

**ζ was held at 0.05 in both.** §16.23 named ζ *"the leading suspect on measured evidence"* for
the decaying ruler. **It is not the cause** — the decay stopped with ζ untouched.

§3.1's equation backs the shortfall out as `G`, and **this comparison is safe where §2's was
not**: both runs sit at the same ζ *and* within 6% on `backup/dist_mean` (8.59 vs 9.17), so C2's
denominator objection does not apply.

| | ζ | δ | implied G | `dist_mean` |
|---|---|---|---|---|
| `phase1` | 0.05 | 0.4293 | 0.01159 | 9.174 |
| this run | 0.05 | 0.6090 | **0.00407** | 8.589 |

**Competing pull on `Dist` fell 2.8× at matched ζ and matched scale.** DERIVED (the equation
still ignores the `diag_backup` mix and the clip).

**This kills §3.1's named suspect.** It read *"Suspect: relu_squared… it is the one term that
got stronger"*. `relu_squared` is **on** in the run where `G` fell 2.8×. Whatever raised `G` in
`phase1_nce_temp_relu2`, it was not the hinge form. **HYPOTHESIS for what did lower it here:
the mask** — §9.9.3 predicted this exact mechanism in advance (*"`L_NCE` does not need to learn
the ruler; it needs to stop contradicting it"*), and the contradicting rows are precisely what
`nce_mask_nearer_same_traj` removes. Untested against `relu_squared`; `abl_relu` separates them.

### 4.4 The mask's own readout is unreadable, and the prediction was inverted — MEASURED

`nce/argmax_in_nearer_set` **rises monotonically** through this run: 0.0196 → 0.1052 → 0.2289 →
0.2972 → **0.3346** over the five bands.

`nce.py:206` computes it as `(-Dist / temperature).argmax(dim=0)` — **pre-mask, on raw logits**,
by design (§9.9.6: *"a property of the checkpoint, not of the config"*). So it reads the
**geometry**, not the loss. With the mask on, nothing penalises those rows any more, so the
model is **free to leave them nearest** — the statistic should hold or rise, and it rises.

**§4's prediction — "below 0.369 at equal-or-better NCE accuracy reads as the mask worked" — is
withdrawn (§C4).** It is inverted: a *fall* would have meant the geometry moved the genuinely
nearer states away from the goal, which is the thing §16.4 objects to. 0.3346 against 0.369 is
the same number by any reasonable error bar, and it says the mask changed neither.

**And the gate number is inflated.** `nce/accuracy_within_question` is computed on the **masked**
`logits` (`nce.py:243`), so the 0.67 rows/column the mask sets to `-inf` are exactly the closest
same-question competitors and cannot win the argmax. 12.9× vs `phase1`'s 10.8× is **not
like-for-like** and part of the gain is mechanical. `nce/loss` is on masked logits too — and it
barely moved (3.0612 vs 3.0765), which is itself odd if those rows carried mass.

**The like-for-like read is `nce_preflight.py` on this checkpoint**, which forces every mask off
(`nce_preflight.py:110-119`) and is directly comparable to `runs/phase1/final`'s 0.369 / 0.2715 /
47.1% same-question share. **Not yet run — it is the first command owed on this run.**

Same-question share of the softmax mass, by §9.9.5's exact method
(`exp(nce/loss) − exp(nce/loss_cross_question)`): **49.6%** here against `phase1`'s **51.0%**.
The nearer mask is within-trajectory, so it was never going to move this; §16.25's sibling mask is.

### 4.5 The cost: `L_step` halved — MEASURED, and it is the finding that decides the next run

| steps | 0–300 | 300–600 | 600–900 | 900–1200 | 1200–1460 |
|---|---|---|---|---|---|
| `probe14/delta_boundary/mean`, `phase1` | −0.0530 | 0.5638 | 1.2212 | 1.6244 | **1.6546** |
| `probe14/delta_boundary/mean`, this run | −0.1372 | 0.0687 | 0.4237 | 0.6748 | **0.6741** |
| `probe03/gap`, `phase1` | 0.1573 | 0.8544 | 1.5521 | 1.9728 | **1.9763** |
| `probe03/gap`, this run | 0.0604 | 0.4013 | 0.8168 | 1.0967 | **1.0874** |

**Both plateau by step 900 — this is a converged worse value, not a slower climb.** `Δ_{z+1}`
reaches **0.693 against `m` = 1.386**: exactly half the margin the config asks for, where
`phase1` overshot it at 1.68. `probe03/gap` at 1.10 is **below `run_report.py`'s 1.8 guard**
(§16.3, diagnostic #3 — the error signal flattening).

Everything in each run's own ruler units — DERIVED:

| | target | phase1 | temp_relu2 | this run |
|---|---|---|---|---|
| good Δ (`probe02`) | −1.00 | −0.72 | −0.75 | −0.66 |
| error Δ (`delta_boundary`) | **+2.00** | 3.80 | 1.90 | **1.12** |
| gap (`probe03`) | +3.00 | 4.52 | 2.65 | **1.78** |
| good tail (`p99`) | — | 6.76 | 3.23 | **2.40** |

**The tail win and the boundary loss are the same 2× compression.** `p99` 0.50×, `delta_max`
0.49×, `std` 0.65×, `delta_boundary` 0.41× — Δ-space narrowed roughly uniformly, and the error
step narrowed with it. `backup/dist_mean` moved only −6%, so this is **not** an ambient scale
collapse like §2.1; it is specific to Δ. **HYPOTHESIS: `relu_squared` bought its tail
suppression by compressing Δ globally rather than selectively**, and ⑤ `L_step` — whose pair is
outside ⑥'s scope and which is ~6× more concentrated per state (§7.12) — lost the tug-of-war
anyway. Not established. `abl_relu` is the one-line test.

### 4.6 Verdict

Two changes, and they pull in opposite directions:

* **the ruler is fixed and ζ is exonerated** (§4.3) — the single most valuable thing this run
  produced, and it closes §16.23's suspect;
* **`relu_squared` delivered on the tail at full geometry** — `p99` 3.00 → 1.49, `frac_above_natural`
  0.178 → 0.108 (target 0.05, so halfway), **at no cost in `L_I`** (0.2268 → 0.2239), which
  retires §16.28's unbounded-quadratic worry;
* **`L_step` converged to half its margin** (§4.5), and `probe03/gap` fails its guard;
* **the mask is still unread** (§4.4) — for the second run running, now for a different reason.

§9.6.2/§9.7.6 put the good-step tail in the **+0.038** headroom column and detection/separation
above it, so a run that trades `probe03/gap` for `p99` is not obviously a win on F1.

> **MEASURED 2026-08-04 → §6.1, and it is not a win: val F1 0.5313 → 0.5074, −0.024 against the
> baseline, all of it in `acc_error`.** The tail fix bought **+0.010** of `acc_correct` against
> §9.6.2's predicted +0.038 ceiling — priced correctly — and the halved `L_step` cost 0.031.
> **§4.5's "the finding that decides the next run" stands; "a trade" in the verdict above does
> not.** Two further corrections from §6: `probe03/gap` is the wrong statistic to have read it
> through (§3.5), and the mask is measured ~inert (§6.2), so §4.3's attribution of the ruler
> recovery to the mask is now the less likely half of its own hypothesis.

---

## 5. Queue

Reordered 2026-08-04 after §4.2. **Everything in group A is minutes and none of it needs a
training run; do all of it before spending 4h on B.**

**A — cheap. ALL DONE 2026-08-04 → §6.**

**B — next, and §6 reordered it. ~75 min beats 4 h:**

1. ~~**Phase 2 + ProcessBench on `phase1_nce_temp_relu2/final`.**~~ **DONE 2026-08-07 → §7.
   Mean F1 0.259 against the baseline's 0.241 (0.242 vs 0.235 leak-adjusted) — the first
   improvement in the project.** The path is `<phase-1 run>/phase2`, not
   `<phase-1 run>/final/phase2` (`train_goal_head.py:255` is `ckpt.parent / "phase2"`), so it
   landed beside `final/` and `runs/phase1/phase2` is untouched. `goal_head.epochs` was refit to
   **13** off `phase2/val_best` with `--from-cache` (seconds, not the ~75-min cache rebuild), and
   the eval ran on that head — verified by the two `phase2/schedule` events and the JSON's
   timestamp. `gate_before_fit` for the checkpoint is recorded at the head of §7.

   **§9.7.3's "do not run phase 2 again" did not bar this** — that ruling was against re-running
   phase 2 to *improve the goal head* on a checkpoint that already had one; a new phase-1
   checkpoint has no goal head and cannot be evaluated at all without one.

   **§7.9's order of work replaces the rest of this group.** In particular #1 there —
   `error_rank.py --stratify` on the new `deltas.npz` — is free, needs no GPU, and gates the
   reading of §7.6.

2. **The same on `phase1_mask_relu2/final`. UNBLOCKED by #1**: val F1 tracked ProcessBench in
   sign (0.5597 val → 0.259, against `phase1`'s 0.5313 → 0.241), so the §6.1 ceiling table is
   usable and this run is worth its ~75 min. **Write the §7.8 equal-error τ prediction down
   before the eval runs** — this is the pre-registration that test needs, and it is free to
   attach to a run that was happening anyway.

3. **`abl_relu`** — `--set losses.good_loss.form=relu --set run.name=abl_relu`, everything else
   as `phase1_mask_relu2`. 4 h. Separates §4.3's ruler recovery (mask vs `relu_squared`) from
   §4.5's halved `L_step`. **§6.2 has already tilted this**: the mask is ~inert on its own
   pre-flight, so `relu_squared` is now the favourite for both. **Write the §3.5 d′ prediction
   down before running `val_f1.py` on it.**

4. **`abl_nomask`** — `--set sampling.nce_mask_nearer_same_traj=false --set run.name=abl_nomask`.
   The other half of the 2×2. **Demoted by §6.2** — a change measured as ~inert is a poor use
   of 4 h unless #3 comes back saying the mask owns the ruler after all.

5. **§16.26 — sibling correct terminals as POSITIVES.** §6.3 makes this the best-evidenced open
   item in the project: three matched checkpoints say phase 1 leaves same-question terminal
   retrieval **2.1× worse than an untrained model**, and the masks were the cheap test that
   would have pre-empted it. `nce_mask_nearer_same_traj` is measured inert (§6.2) and
   `nce_mask_sibling_correct_late` is still unrun — **run that one first** (`--set
   sampling.nce_mask_sibling_correct_late=true`), judged on `gate/recall@1` against the matched
   untrained **0.6533**, before writing a new loss term.

6. **`abl_zeta02`** — `--set losses.zeta=0.2`. **Demoted.** §4.3 exonerates ζ for the decay, so
   this is no longer a fix — it is now a clean test of §3.1's equation, which after §4.3 has
   two matched-scale points and can predict δ before launch.

**Done:** ~~fix `run_report.py:135-137`~~ (§3.3, fixed 2026-08-04 — each statistic now scores
against its own chance level and `accuracy_within_question` is the gauge row).
~~A1 preflight~~, ~~A2 val_f1 ×3~~, ~~A3 goal_gate matched pair~~ — **all done, §6.**

---

## 6. Group A readouts — MEASURED 2026-08-04, and they reverse §4.6

Three cheap measurements, no training run. **They change the verdict on `phase1_mask_relu2`
from "a trade" to "a net loss", and they promote §16.26 to the live item.**

### 6.1 `val_f1.py` — the matched three-way, and this run is LAST

400 val questions, 4,549 trajectories, identical for all three; τ refit per checkpoint.
**A CEILING, not a result** (§9.5 — it is handed a real terminal eval never has).

| run | fitted τ | τ / natural | acc_error | acc_correct | **F1** |
|---|---|---|---|---|---|
| `phase1` (baseline) | 1.0206 | **2.94×** | 0.4173 | 0.7309 | **0.5313** |
| `phase1_nce_temp_relu2` | **0.3503** | **1.01×** | **0.4428** | **0.7605** | **0.5597** |
| `phase1_mask_relu2` | 0.4218 | **1.22×** | 0.3859 | 0.7404 | **0.5074** |

Sensitivity 0.005–0.006 on all three: the curve is flat within ±0.1 of the optimum, so none of
these is a threshold artifact.

**1. §7.12's τ target is HIT, by both new runs.** It asked for the fitted τ to come from 2.39
down to *"~0.35, the natural 0.347"*. `phase1/final` had already improved that to 1.02; the two
`relu_squared` runs land at **0.350 and 0.422**, i.e. 1.01× and 1.22× the natural midpoint. The
ruler and the margin now mean what §7.6.4 says they mean. **This is `relu_squared`'s win and it
is unambiguous** — it is the one change common to both and absent from both earlier runs.

**2. And §9.6.2 priced that win correctly, to the decimal.** It said perfecting the entire clean
side is worth **+0.038** and the tail lives in that column. `acc_correct` moved 0.7309 → 0.7404,
**+0.010**. The tail fix delivered exactly the small thing it was predicted to deliver.

**3. `phase1_mask_relu2` loses 0.024 of val F1 against the baseline, and all of it is
`acc_error`** — 0.4173 → 0.3859, against `acc_correct` +0.010. That is §4.5's halved `L_step`
converting into F1, and it is the second time the file has measured the false-positive lever
being small and the detection lever being large.

**4. `phase1_nce_temp_relu2` is the best of the three on BOTH halves**, and §2.5 said its F1
*"would measure the contraction, not the loss set."* **That was wrong as a reason not to measure
it** — `val_f1.py` refits τ, so a uniform Δ rescale is absorbed by construction and the 5.4×
collapse cannot be what produced 0.5597. **It is not, however, a recommendation to keep τ=√512**:
this is a *skyline* ceiling and §9.5.1 measured the skyline and the goal head disagreeing in
sign on ProcessBench. See §6.4.

### 6.2 The preflight — the mask changed nothing it was aimed at

`nce_preflight.py --checkpoint runs/phase1_mask_relu2/final`, 6 batches, every mask forced off,
directly comparable to `runs/phase1/final` (§9.9.6):

| | `phase1/final` | `phase1_mask_relu2/final` |
|---|---|---|
| `nce/argmax_in_nearer_set` | 0.36908 | **0.37812** |
| null (`nearer_set_size / R`) | 0.00189 | 0.00189 |
| ratio | 195× | **199.6×** |
| `nce/categorical_accuracy_backward` | 0.2715 | 0.2830 |
| `nce/columns_with_nearer` | 0.426 | 0.426 |
| `nce/nearer_set_size` | 0.728 | 0.728 |
| `nce/pos_dist` / `neg_dist` | 2.384 / 9.297 | 2.228 / 8.716 |

**Training with those rows masked out of the loss for 1,464 steps left the statistic where it
was.** That is the benign reading and it is the one §C4 predicted: those rows are *genuinely*
nearer the goal, the mask stops penalising them for it, and the geometry keeps them nearest.
**The mask is neither the fix nor the damage** — it is close to a no-op on everything it was
supposed to touch, which makes §4.3's ruler recovery more likely to be `relu_squared`'s after
all, contrary to §4.3's own hypothesis. `abl_relu` still decides it.

### 6.3 The goal gate — §9.8.3's collapse is CONFIRMED, matched, and it is the biggest number here

`goal_gate.py --questions 200`, trained vs `--untrained`. **Fully matched**: all three columns
report `terminals` 525, `within_pairs` 1002, `across_pairs` 274,098 — the same sample, only the
model differs. **This resolves §9.8.3's "NOT YET MATCHED — do not quote as established".**

| | untrained | `phase1` † | `phase1_mask_relu2` |
|---|---|---|---|
| `auc` | 0.9069 | 0.9065 | **0.9304** |
| **`recall@1`** | **0.6533** | **0.2762** | **0.3048** |
| `ratio` | 0.5770 | 0.3034 | 0.2600 |
| `within_question_terminal_spread` | 4.044 | 2.905 | **2.253** |
| `across_question_terminal_spread` | 7.009 | 9.577 | 8.664 |
| `questions_fully_clustered` | **0.495** | 0.060 | 0.100 |
| `questions_fully_scattered` | 0.305 | 0.620 | 0.595 |

† `runs/phase1/phase2/events.jsonl`, `phase2/gate_before_fit`.

**Phase 1 more than halves same-question terminal retrieval against an untrained model, on both
runs.** 0.653 → 0.276 / 0.305, and `questions_fully_clustered` **0.495 → 0.06 / 0.10**. The
untrained baseline is 0.6533, slightly *higher* than §10.1.1's 0.618 from its own sample, so the
collapse is marginally worse than the file has been recording.

**Read `recall@1` against `ratio`, which moved the other way** (0.577 → 0.260, "better"), and
`within_spread` likewise (4.04 → 2.25). This is §10.1.1's mean-vs-distribution failure exactly:
a tenth of questions cluster hard enough to pull the mean `within` distance down while **59.5%
scatter completely**. **A mean ratio cannot see it and this is the run that proves the point.**

**`phase1_mask_relu2` is mildly better than `phase1` on every column here** — `recall@1` +0.029,
`within_spread` −0.65, `auc` **+0.024 over untrained where `phase1` managed +0.000**. §9.9's
order of work #3 named `within_question_terminal_spread` and `probe02/delta_good_mean` as the
mask's judges and both moved in the right direction. But the effect is small against a 2.1×
deficit, and §6.2 says the mask itself is nearly inert, so **do not attribute it to the mask.**

**This is §16.26, and it is now the best-evidenced open item in the project.** Nothing in the
loss set pulls two correct terminals of one question together; ~24% of `L_NCE` pushes them
apart (§9.9.5); phase 2 then has to predict one vector per question from a cluster that phase 1
spent the run dispersing. **Three matched checkpoints now say phase 1 makes this strictly worse
than doing nothing.**

### 6.4 What group A settles, and what it does not

**Settled:**

* `relu_squared` fixed τ. 2.94× → 1.01×/1.22× of natural, and it cost nothing in `L_I` (§4.2).
  **Keep it.**
* The tail was worth +0.010 of `acc_correct`, against §9.6.2's predicted +0.038 ceiling. **The
  false-positive lever is small and this is now measured twice, not derived.**
* `phase1_mask_relu2` is a **net −0.024 val F1** against the baseline, entirely through
  `acc_error`. §4.6 called it "a trade"; on the number that matters it is a loss.
* The `recall@1` collapse is real, matched, and present on every trained checkpoint (§6.3).
* The mask is ~inert on its own pre-flight (§6.2).

**Not settled, and do not act as if it were:**

* **Whether τ=√512 is actually good.** `phase1_nce_temp_relu2` wins this table by +0.028, but
  every number in it is a *skyline* ceiling, and §9.5.1 measured the skyline and the goal head
  disagreeing in sign on ProcessBench. A val ceiling is not a ProcessBench F1 — `phase1`'s own
  pair is 0.5313 val against **0.241** ProcessBench. **The only way to price τ is phase 2 +
  `processbench.py` on that checkpoint** (~75 min), and §9.7.3's "do not run phase 2 again"
  does not forbid it — that ruling was against re-running phase 2 to *improve the goal head*,
  not against evaluating a new phase-1 checkpoint.
* **Which change fixed the ruler** (§4.3). `abl_relu`.
* **Whether d′ (§3.5) predicts F1** or is a 3-point coincidence.

---

## 7. ProcessBench on `phase1_nce_temp_relu2` — MEASURED 2026-08-07

`runs/phase1_nce_temp_relu2/phase2/final/{processbench.json, deltas.npz}`. Producers:
`eval/processbench.py`, then `scripts/report_processbench.py`, `scripts/analyze_deltas.py` and
one ad-hoc rule/τ sweep over the saved `Δ` arrays. **No GPU for anything below the first table.**

**The head is the right one.** `phase2/events.jsonl` has two `phase2/schedule` events —
`epochs 20` then `epochs 13` — and the eval JSON is timestamped after the 13-epoch refit
(`--from-cache --overwrite --set goal_head.epochs=13`, `best_epoch 12`, `best_val_loss 1.1101`
against the 20-epoch run's 1.1104). So this is the epoch-13 checkpoint, not the overfit one.
`gate_before_fit` for this checkpoint, the §5 queue item: `auc` **0.7029**, `recall@1` **0.3390**,
`ratio` 0.5936, within 0.6137, across 1.0338, `fully_scattered` 0.51, 525 terminals.

### 7.1 The result — the first improvement over the baseline

τ = **0.3403**, fitted on Math-Shepherd val, `calibration/f1` 0.5872, sensitivity 0.0095.

| subset | `phase1` (baseline) | **this run** | Δ |
|---|---|---|---|
| `gsm8k` | 0.391 | 0.376 | −0.015 |
| `math` | 0.252 | **0.304** | **+0.052** |
| `olympiadbench` | 0.144 | **0.204** | **+0.060** |
| `omnimath` | 0.177 | 0.151 | −0.026 |
| **mean over 4** | **0.241** | **0.259** | **+0.018** |

**Leak-adjusted it is +0.007, not +0.018 — read §7.6 before quoting the headline.**

### 7.2 The model improved ~8× more than the score did — MEASURED

Every threshold-free component moved far more than F1 did:

| | `phase1` | this run |
|---|---|---|
| detection AUC (`gsm8k`/`math`/`olymp`/`omni`) | 0.639 / 0.579 / 0.537 / 0.533 | **0.683 / 0.756 / 0.710 / 0.713** |
| — mean | 0.572 | **0.716** |
| exact localisation \| flagged | 0.527 / 0.263 / 0.132 / 0.172 | **0.727 / 0.377 / 0.238** / 0.158 |
| permutation-null multiple | 2.71 / 1.38 / 1.53 / 1.41 | **4.54 / 2.55 / 2.71 / 1.69** |

Detection AUC rose **+0.044 / +0.177 / +0.173 / +0.180**. §9.6.3 recorded the baseline as *"at or
near chance on three of four subsets"*; on this checkpoint no subset is near chance. **That is
the finding of this run, and F1 collected almost none of it.** The rest of §7 is where it went.

### 7.3 Leak #1 — τ transfer, +0.059, and it is NEW — MEASURED

`ORACLE tau` is fit on ProcessBench and **§9.2 forbids reporting it**. Quoted only as the
ceiling on a legal val-side refit, and only to compare the two checkpoints against each other:

| | fitted τ | mean F1 at it | one **global** oracle τ | mean F1 | gap |
|---|---|---|---|---|---|
| `phase1` | 1.1685 | 0.241 | 1.03 | 0.243 | **+0.002** |
| **this run** | **0.3403** | **0.259** | **0.120** | **0.318** | **+0.059** |

Per-subset oracle τ buys only **+0.006** more (0.324 vs 0.318), so **this is not a per-subset
problem** — one global number is fine, it is simply at the wrong value, ~2.8× too high.

**Why it appeared now.** §9.6.6 measured the baseline's τ transfer at +0.004…+0.013 and concluded
*"τ is not a lever"*. That was true **because the score was near chance** — a statistic with no
separation is equally useless at every threshold. This checkpoint has real separation, so F1 is
now strongly τ-dependent on ProcessBench, and the instrument that picks τ did not improve with it:

| Math-Shepherd val (the calibration set) | τ = 0.114 | τ = 0.205 | **τ = 0.340 (picked)** | τ = 0.499 |
|---|---|---|---|---|
| val F1 | 0.5633 | 0.5825 | **0.5872** | 0.5685 |
| val `acc_error` / `acc_correct` | 0.552 / 0.575 | 0.535 / 0.639 | **0.498 / 0.715** | 0.448 / 0.777 |

A 3× range in τ moves val F1 by 4% relative (`sensitivity 0.0095`), and the same move is worth
**+0.059 on ProcessBench**. **The calibration set cannot resolve a difference the benchmark cares
a great deal about.**

**The mechanism, and it is not "one τ, two lengths".** At τ = 0.340 val is near-balanced
(0.498 / 0.715) while ProcessBench is lopsided **4.6:1** (0.185 / 0.855). A harmonic mean peaks
near balance, so ProcessBench wants τ pushed far down to buy `acc_error`. It is lopsided at the
same τ because its `acc_error` additionally requires *exact localisation* over longer and harder
solutions (§5's distribution shift), which the val set's easier localisation hides. **The
transferable property is the balance point; the F1 argmax is not.**

> **This makes §6.1's reading of the τ target incomplete, not wrong.** §7.12 asked the fitted τ
> to come down to the natural 0.347 and this run hit it at **1.01×**. That is real and
> `relu_squared` earned it. **But the theory-correct τ is now the one costing 0.059** — the
> natural midpoint is where `L_step`'s margin and `L_T`'s ruler say a good and a bad step
> separate *on the training distribution*, and ProcessBench's optimum is at **0.35× of it**.
> Hitting the target and paying for it are the same event. See **C6**.

### 7.4 Leak #2 — `acc_correct` contains no model judgement — MEASURED

§9.6.4's independence model, re-run on this checkpoint. `r` = per-step crossing rate on clean
solutions; if crossings were coin flips knowing nothing about correctness, a clean solution
survives with probability `(1−r)^T`:

| subset | `r` | mean `T` | `(1−r)^T̄` | observed `acc_correct` |
|---|---|---|---|---|
| `gsm8k` | 0.0020 | 5.06 | 0.990 | **0.990** |
| `math` | 0.0285 | 5.98 | 0.841 | **0.855** |
| `olympiadbench` | 0.0270 | 8.67 | 0.789 | **0.811** |
| `omnimath` | 0.0433 | 7.38 | 0.721 | **0.734** |

**Reproduced to within 0.02 on every subset.** And by length, `olympiadbench`: `T≤5` → **0.971**,
`5<T≤7` → 0.928, `7<T≤9` → 0.866, `T>9` → **0.516**. Same model, same τ, same solutions —
**half the metric is decided by how many independent chances a solution has to trip the wire.**

This was already true of `phase1` (§9.6.4) and it survived a checkpoint that improved detection
AUC by 0.14, which is the point: **`acc_correct` is a one-parameter function of τ and length, and
there is no per-solution information in it to improve.** It explains, without any new hypothesis,
why §9.6.2's +0.038 ceiling on the clean side keeps being confirmed and why `relu_squared`'s tail
fix bought only +0.010 of val `acc_correct` (§6.4) — the lever is `r`, and `r` is set by τ.

### 7.5 Leak #3 — `acc_error` is a product of two coin flips — MEASURED

§9.6.1's factorisation, exact on this checkpoint to three decimals:

| subset | P(flag \| errored) | P(exact \| flagged) | product | reported `acc_error` |
|---|---|---|---|---|
| `gsm8k` | 0.319 | 0.727 | 0.232 | **0.232** |
| `math` | 0.492 | 0.377 | 0.185 | **0.185** |
| `olympiadbench` | 0.492 | 0.238 | 0.117 | **0.116** |
| `omnimath` | 0.536 | 0.158 | 0.085 | **0.084** |

**Half the errored solutions are never flagged at all**, and of those flagged, localisation
collapses with difficulty (0.727 → 0.158). Both factors improved over `phase1` on three of four
subsets — a product of two improved-but-small numbers is still small. This is the channel the
τ fix of §7.3 opens: at the global oracle τ the flag rate rises and `acc_error` with it.

### 7.6 Leak #4 — most of the `math` gain is contamination — MEASURED

Locked #5's split, both checkpoints:

| `math` | `phase1` | this run | Δ |
|---|---|---|---|
| leaked (587) | 0.274 | **0.362** | **+0.088** |
| clean (413) | 0.227 | **0.239** | **+0.012** |
| gap | +0.047 | **+0.123** | **2.6× wider** |

**The leak gap nearly tripled.** §9.3.1 read the baseline's +0.047 as *"what a metric that is not
keying on the problem at all looks like"*; this checkpoint keys on it substantially more.
Substituting clean-only `math` into the headline:

| | `phase1` | this run | Δ |
|---|---|---|---|
| mean F1, clean `math` | 0.235 | **0.242** | **+0.007** |

**So the honest uncontaminated improvement is +0.007, not +0.018 — the leak carries ~60% of the
headline gain.** `olympiadbench` and `omnimath` have no leak split (§4.3(b): 0% overlap), so this
correction is partial and applies only through the `math` column. It does not touch §7.2 —
detection AUC rose on the two zero-leakage subsets by +0.173 and +0.180.

The τ fix does not change the ratio: at the oracle τ = 0.120 the split is 0.427 / 0.303, gap
**+0.124**, essentially identical. **The τ leak and the contamination leak are independent.**

### 7.7 What was ruled out — MEASURED

Mean F1 over the four subsets, three decision rules × three thresholds:

| rule | fitted τ | one global oracle τ | per-subset oracle τ |
|---|---|---|---|
| **first-crossing (shipped)** | **0.259** | **0.318** (τ +0.12) | 0.324 |
| `argmax \| max>τ` | 0.253 | 0.312 | 0.317 |
| first-crossing, length-adjusted τ | 0.249 | 0.316 | 0.324 |

- **The decision rule is not the leak.** First-crossing is the best of the three at every
  threshold. Note this **inverts on `phase1`**, where argmax won by +0.017 (§9.9.1) — so
  §9.9.1's table is checkpoint-specific and `eval.localisation_rule` should stay
  `first_crossing`.
- **A length-adjusted τ loses.** §9.6.4 proposed `τ(T)` for constant per-solution FPR as *"the
  cheapest available fix"*; a log-in-`T` adjustment costs 0.010 at the fitted τ and 0.002 at the
  oracle. §7.4 says why it cannot help much: `acc_correct` is *already* explained by length, so
  removing the length dependence trades the two halves against each other without adding
  information.
- **Per-solution rescaling cannot help.** Between-solution `Δ` variance is 14–28% of the total,
  and `rescale[scale]` never beats plain oracle on any subset (0.440 vs 0.468 on `gsm8k`).

### 7.8 The tempting fix I do NOT trust — HYPOTHESIS, n=2, post hoc

§7.3 says the transferable property is the *balance point*, not the F1 argmax. That implies a
calibration rule that is **§9.2-legal** (fit on val only, never on ProcessBench): pick τ where
`acc_error = acc_correct` on Math-Shepherd val.

| | τ @ val-F1-argmax | mean F1 | τ @ val equal-error | mean F1 | Δ |
|---|---|---|---|---|---|
| **this run** | 0.340 | 0.259 | **0.091** | **0.315** | **+0.056** |
| `phase1` | 1.169 | 0.241 | 0.569 | 0.202 | **−0.039** |

It recovers 95% of §7.3's oracle gap on this checkpoint **and loses 0.039 on the baseline.**

> **DO NOT ADOPT THIS ON THIS EVIDENCE.** Two points, opposite signs, and the rule was chosen
> *after* looking at ProcessBench — which is the B11/B12/§9.9.1 shape exactly: a criterion
> selected because it flatters the run in front of you. The derivation in §7.3 is real and
> predicts the sign on both rows *post hoc*; that is not the same as predicting it in advance.
> **Pre-register it on the next checkpoint before its eval runs**, and judge it there.

### 7.9 What §7 settles, and what it does not

**Settled:**

* `phase1_nce_temp_relu2` beats the baseline: **0.259 vs 0.241**, or **0.242 vs 0.235** clean.
  First improvement in the project. **§6.4's "whether τ=√512 is actually good" is answered yes** —
  the val ceiling tracked ProcessBench in sign, which also unblocks §5's queue item B2.
* Detection is no longer near chance on any subset (0.572 → 0.716 mean AUC), overturning
  §9.6.3's standing reading **for this checkpoint**.
* **τ transfer is now a first-order leak (+0.059) where §9.6.6 measured it at +0.008.** The
  finding that "τ is not a lever" was a property of a near-chance score, not of the pipeline.
* `acc_correct` is a length lottery, on two checkpoints now (§7.4). Any future work aimed at it
  is buying out of §9.6.2's +0.038 column and will be throttled by `r`.
* The decision rule and per-solution rescaling are not levers (§7.7).

**Not settled:**

* **Which of the four bundled changes did this** (§16.27). The bundle is unattributed and
  §6.2 measured the mask ~inert, so `τ_NCE = √512` and `relu_squared` are the candidates.
  `abl_relu` separates them.
* **Whether the leak gap tripling is memorisation or difficulty** — §9.7.4's confound, in F1
  form. Leaked `math` is 1.1 steps shorter, and §7.4 has just measured that length alone moves
  `acc_correct` by 0.45 across the range. **Run `error_rank.py` `T`-stratified on this
  checkpoint's `deltas.npz` before calling it contamination.**
* **Whether the equal-error calibration rule generalises** (§7.8). n=2, post hoc.
* Where the +0.059 actually goes once collected legally — the oracle is a ceiling, not a value.

**Order of work, replacing §5's B group:**

1. **`error_rank.py --stratify` on this `deltas.npz`.** Free, no GPU. Settles §7.6's confound
   and gives this checkpoint's within-solution rank against `phase1`'s 0.387/0.341/0.396/0.421 —
   the one statistic that cancels τ, length and detection, and therefore the only clean read on
   whether the geometry itself improved.
2. **Write the equal-error τ prediction down** (§7.8), then run phase 2 + ProcessBench on
   `phase1_mask_relu2/final` (§5 B2) and read both τ rules on it. That is the pre-registration
   and the queued run in one.
3. **`abl_relu`**, 4 h, unchanged from §5 B3.

---

## C. Corrections

**C1 — "ζ = 0.1 is a confirmed null." WITHDRAWN 2026-08-04.** Based on the absolute ruler
alone (0.4432 → 0.4093) without checking whether the space it lives in had changed. It
had, by 5.8×.

**C2 — "ζ was doing real work; `ruler/dist` rose 5.33×." ALSO WITHDRAWN, same day.** The
ruler does not track the ambient scale (0.44 in a 9.2-unit space, 0.41 in a 1.59-unit
space), so `ruler/dist` rising is a **denominator artifact**: at ζ = 0.05 the same
collapse would have given `0.4432 / 1.5899 = 0.279` against the observed 0.257. The ratio
carries no information about ζ. Correct status is §3.2: **unknown**.

**C3 — "the boundary Δ ran out of runway."** Proposed as the explanation for
`delta_boundary` falling 2× with the collapsed space, then weakened by the same
observation as C2 — the ruler held at 0.92 through a 5.8× collapse, so available range is
not what sets these quantities. May still hold for the boundary specifically (it fell 2×
where the ruler fell 8%), but it is **untested** and must not be quoted as established.

**C4 — "`nce/argmax_in_nearer_set` falling below 0.369 reads as the mask worked."
WITHDRAWN 2026-08-04**, §4 predictions table. The statistic is computed pre-mask on raw logits
(`nce.py:206`) by explicit design, so it reads the geometry and not the loss; with the mask on,
nothing penalises those rows and the model is free to leave them nearest. A *fall* would have
meant the genuinely-nearer states got pushed away, which is the pathology §16.4 objects to.
It rose monotonically 0.020 → 0.335 through the run, which is the benign reading. **The mask
has no readout in its own run's `metrics.jsonl`; `nce_preflight.py` is the instrument.**

**C5 — "Suspect: relu_squared" (§3.1). WITHDRAWN 2026-08-04** by §4.3: `relu_squared` is on in
the run where the implied `G` fell 2.8× at matched ζ and matched ambient scale. Recorded rather
than deleted because the reasoning was sound and the counter-example is a run, not an argument.

**C6 — "the fitted τ landing at 1.01× natural is `relu_squared`'s win" (§6.1). NARROWED
2026-08-07** by §7.3. The measurement stands and so does the attribution — τ came from 2.94× to
1.01× and the two `relu_squared` runs are the ones that did it. What does not stand is the
implication that a theory-matched τ is therefore the *right* τ at eval: on ProcessBench the
optimum for this checkpoint is **0.120**, i.e. **0.35×** the natural midpoint, and the 0.340 the
val fit selected costs **0.059 mean F1**. The natural midpoint is where `L_step`'s margin and
`L_T`'s ruler separate a good step from a bad one **on the training distribution**; ProcessBench
localises over longer and harder solutions, so its two halves balance somewhere else entirely
(§7.3). **Hitting §7.12's target and paying 0.059 for it are the same event.** Recorded rather
than edited into §6.1 because the τ target was correctly reasoned and correctly measured — what
was missing was that no one had ever checked where the target transfers to.

**C7 — "τ is not a lever" (§9.6.6, CLAUDE.md, and §16.24's "PARTLY ANSWERED" block).
WITHDRAWN for this checkpoint 2026-08-07** by §7.3. Measured on `runs/phase1`, the
ProcessBench-oracle τ bought +0.004…+0.013 and the conclusion drawn was that τ transfers almost
perfectly. **That was a property of a near-chance score, not of the calibration procedure**: a
statistic with no separation is equally useless at every threshold. On
`phase1_nce_temp_relu2` — mean detection AUC 0.716 against 0.572 — the same arithmetic gives
**+0.059**, a 7× larger gap than the largest per-subset figure that produced the original claim.
The general form: **τ-transfer loss scales with how much separation there is to mis-threshold,
so it must be re-measured on every checkpoint and never inherited.** §9.6.6's numbers are not
wrong; the sentence generalising them was.
