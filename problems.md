# problems.md — the 2026-08-04 measurement round

**What this file is.** The results of the four commands run on 2026-08-04 against
`runs/phase1_mask_relu2/final` and `runs/phase1_nce_temp_relu2/final`, plus everything that can
be read out of the two new `metrics.jsonl` files without spending GPU time, and the problems
those readings expose. It is a working record, not a decision: nothing here amends §2, and every
number carries its provenance in the CLAUDE.md sense — **MEASURED** (read off an artifact),
**DERIVED** (arithmetic on measured numbers), or **NOT ESTABLISHED**.

**Producers, all read-only:**

| artifact | produced by | when |
|---|---|---|
| `runs/phase1_mask_relu2/final/val_f1.json` | `scripts/val_f1.py` | 2026-08-04 22:53 |
| `runs/phase1_nce_temp_relu2/final/val_f1.json` | `scripts/val_f1.py` | 2026-08-04 22:57 |
| the gate JSON blocks | `scripts/goal_gate.py --questions 200`, once trained once `--untrained` | 2026-08-04 |
| the pre-flight block | `scripts/nce_preflight.py --checkpoint runs/phase1_mask_relu2/final` | 2026-08-04 |
| every windowed series below | `runs/*/metrics.jsonl`, 147–150 log rows each at `log_every = 10` | in-run |
| `runs/phase1/final/val_f1.json` | `scripts/val_f1.py` | 2026-07-29 09:51 |
| `phase2/gate_before_fit` | `train_goal_head.py` | 2026-07-29 |

---

## 0. The three runs, and why they are comparable

| run | `τ_NCE` | `ζ` | `good_loss.form` | `nce_mask_nearer_same_traj` | `λ_good` | `n_questions` |
|---|---|---|---|---|---|---|
| `phase1` | 1.0 | 0.05 | `relu` | *(key absent — off)* | 1.0 | 34,650 |
| `phase1_mask_relu2` | 1.0 | 0.05 | **`relu_squared`** | **true** | 1.0 | 34,650 |
| `phase1_nce_temp_relu2` | **22.627417** | **0.1** | **`relu_squared`** | **true** | 1.0 | 34,650 |

`nce_mask_sibling_correct_late` and `nce_mask_same_question_correct` are **`false` in both new
runs.** This matters for §1 below and is the single most common way to misread this file.

**All three runs consume the identical batch sequence.** Same seed, same `n_questions`, same
selection SHA, so `prepare_data.py` was not re-run and the sampler is deterministic. Confirmed
by three sampler-only series agreeing to the printed digit from the 100–300 window onward
(MEASURED):

```
step/distinct_z             25.9259   25.9259   25.9259
nce/negatives_per_column   318.8148  318.8148  318.8148
probe01/distinct_goal_ratio   0.7039    0.7039    0.7039
nce/negatives_masked            —       0.6652    0.6652
nce/columns_with_nearer         —       0.4132    0.4132
```

Two consequences. **(a)** Differences between runs are attributable to the loss set and nothing
else — no data, batching or ordering confound. **(b)** `nce/negatives_masked ≈ 0.665` against
§9.9.6's predicted ~0.64, and `columns_with_nearer` 0.413 against the sampler simulation's
0.418, so **the mask demonstrably fired** and the §17 Monte-Carlo of `data/goals.py` is
validated a second time on the model.

**What is NOT comparable: each edge of the chain is two changes.** `phase1 → mask_relu2` moves
`relu → relu_squared` *and* the mask on. `mask_relu2 → nce_temp` moves `τ` *and* `ζ`. §16.27
flagged the four-change bundle as past the point where an escape exists; the three-point chain
narrows it to two-per-edge and no further. **Nothing below attributes a single change unless it
says so explicitly.**

---

## 1. PROBLEM 1 — the goal gate, now matched, says phase 1 destroys within-question terminal structure

### 1.1 The measurement

The trained gate run and the `--untrained` run used the **same terminal set** as the
`phase2/gate_before_fit` event from 2026-07-29 — `within_pairs 1002`, `across_pairs 274098`,
`terminals 525`, `questions 200` in all three. So this is a fully matched three-way comparison
and it is the run §9.8.3 and §9.9.7 have been waiting on (MEASURED):

| | untrained | `phase1` (relu, no mask) | `mask_relu2` (relu², mask) |
|---|---|---|---|
| **`gate/recall_at_1`** | **0.6533** | **0.2762** | **0.3048** |
| `gate/auc` | 0.9069 | 0.9065 | 0.9304 |
| `gate/ratio` | 0.5770 | 0.3034 | 0.2600 |
| `within_question_terminal_spread` | 4.0440 | 2.9054 | 2.2531 |
| `across_question_terminal_spread` | 7.0086 | 9.5772 | 8.6643 |
| `questions_fully_clustered` | 0.4950 | 0.0600 | 0.1000 |
| `questions_fully_scattered` | 0.3050 | 0.6200 | 0.5950 |
| `per_question_recall_std` | 0.4394 | 0.3252 | 0.3502 |

**Which checkpoint the trained gate ran on** — the transcript line was truncated
(`goal_gate.py --ch relu2/final`). It is `phase1_mask_relu2`: its `across_question_terminal_spread`
of 8.6643 matches that checkpoint's own `nce/neg_dist` of 8.6552, and `within` 2.2531 matches its
`nce/pos_dist` 2.2280. `nce_temp_relu2`'s geometry is compressed to `neg_dist 1.5948` and cannot
produce an across-spread of 8.66. **DERIVED, and worth re-running with the full command written
out** so the record is unambiguous.

### 1.2 What it establishes

**§9.8.3's `recall@1` collapse is now established.** CLAUDE.md §9.8.3 carries an explicit
banner — *"NOT YET MATCHED — do not quote 0.618 → 0.276 as established… Re-run **both** at the
same `--questions 200` first"* — and §9.9.7 lists it as item #2, *"still not done, still the
cheapest way to kill §9.8.3's `recall@1` claim… the oldest un-run free measurement in this
file."* It is run. It does not kill the claim; it confirms it at a **larger** untrained baseline
(0.6533 vs the 0.618 recorded on 2026-07-27's own sample).

**Training more than halves same-question terminal retrieval versus doing nothing at all.**
`questions_fully_clustered` falls 0.495 → 0.060/0.100; `fully_scattered` rises 0.305 →
0.620/0.595.

**The AUC gate is not a gate.** An untrained model with random-init ψ passes it at 0.9069, and
`goal_gate.py` prints `PROCEED` for it. §10.1.1 wrote this down in advance — *"the bar is 0.904,
not 0.9, and a trained checkpoint landing near it means phase 1 added nothing"* — and phase 1
lands at 0.9065, i.e. **+0.0004**. `mask_relu2` reaches 0.9304, **+0.0235**. The exit code and
the printed verdict are both keyed on `auc > 0.9` (`goal_gate.py:147,154`) and are therefore
uninformative on this task. **`recall@1` is the number.**

**The `ratio` improving while `recall@1` halves is the mean-vs-distribution failure
`terminal_separability` is named for.** The within-question mean distance genuinely contracts
(4.044 → 2.253) and the across-question mean genuinely expands (7.009 → 8.664), so the ratio
reads better; the *nearest-neighbour* structure is worse. A mean cannot see this, which is
§10.1.1's whole argument for reporting both.

### 1.3 The correction that must accompany it — the mask that ran is not the mask aimed at this

`nce_mask_nearer_same_traj` targets **within-trajectory** false negatives: rows on the goal's own
trajectory, later than the sampled positive (§16.4, §9.9.2). The terminal scattering has a
different mechanism — **sibling-correct** rows, §9.8.4/§9.9.5: a goal column drawn from one
correct trajectory of question `q`, with the *late states of `q`'s other correct trajectories*
sitting in the negative pool. That is what §16.25 is about, and the flag for it —
`nce_mask_sibling_correct_late` — **is `false` in both new runs.**

So the supported statement is:

- ✅ **MEASURED:** the nearer-same-traj mask improves every gate number, modestly.
  `recall@1` +0.029, `ratio` −0.043, `within_spread` −0.652 (−22%), `fully_clustered` +0.040.
- ❌ **NOT ESTABLISHED:** *"masking failed to fix the terminal scattering."* The targeted mask has
  never been enabled.

§9.9.7's stopping rule — *"if `within_question_terminal_spread` and `gate/recall_at_1` do not
move under masking, §16.26 will not move them either"* — is therefore **not yet triggerable**.
Both moved, by a little, under the wrong mask.

### 1.4 What it does and does not imply for F1

**It does not follow that this is what caps F1.** §9.8.1 measured that a *cross-question* goal
retains 103% of the localisation signal, and §9.7.3 measured that a gold reference terminal
matches the goal head to 0.2σ. If the goal channel is uninformative, terminal clustering is
downstream of nothing that the eval reads. Consistent with that: **`mask_relu2` has the better
gate numbers of the two trained runs and the worse val F1 ceiling** (§2). The gate and the
ceiling are decoupled on this data.

The gate matters for a different reason and it should be stated as that reason only: **phase 2's
regression target is ill-posed when a question's correct terminals are 525 different points**
(§16.26). That is an argument about the goal head, and §9.8.3 already exonerated the goal head
on its own terms (`goal/d_pred_to_target` 2.170 against a within-spread of 2.905).

---

## 2. PROBLEM 2 — the run that effectively deletes `L_NCE` scores best, at every threshold

### 2.1 The val F1 ceiling

All three at 400 val questions / 4,549 trajectories, real-terminal goal, **CEILING not a result**
(§9.5, `val_f1.py`'s own banner). MEASURED:

| run | fitted τ | ×natural | sensitivity | peak F1 | `acc_error` | `acc_correct` | F1 at natural τ |
|---|---|---|---|---|---|---|---|
| `phase1` (relu, no mask, τ=1, ζ=.05) | 1.0206 | **2.94×** | 0.0054 | 0.5313 | 0.4173 | 0.7309 | 0.4818 |
| `mask_relu2` (relu², mask, τ=1, ζ=.05) | 0.4218 | 1.22× | 0.0062 | 0.5074 | 0.3859 | 0.7404 | 0.5046 |
| `nce_temp` (relu², mask, τ=22.6, ζ=.1) | 0.3503 | **1.01×** | 0.0052 | **0.5597** | 0.4428 | 0.7605 | **0.5595** |

natural τ = 0.3466 throughout. Sensitivity ≈ 0.005–0.006 in all three: the curve is flat within
±0.1 of the optimum, so no nearby τ does materially better.

**`nce_temp` dominates `mask_relu2` at every τ on the 203-point curve, not just at its peak**
(MEASURED, read out of the saved `curve` arrays):

| τ | `phase1` | `mask_relu2` | `nce_temp` |
|---|---|---|---|
| 0.000 | 0.4215 | 0.4694 | **0.4923** |
| 0.197 | 0.4593 | 0.5030 | **0.5410** |
| 0.347 *(natural)* | 0.4818 | 0.5046 | **0.5595** |
| 0.500 | 0.5025 | 0.5017 | **0.5498** |
| 0.750 | **0.5207** | 0.4677 | 0.4951 |
| 1.000 | **0.5304** | 0.4168 | 0.4380 |
| 1.500 | **0.4960** | 0.3155 | 0.3220 |
| 2.000 | **0.4494** | 0.2282 | 0.2150 |

Two shapes here. **`nce_temp` ≥ `mask_relu2` everywhere** — a robust ordering, not a peak
artifact. **`phase1` is a different curve entirely**: worse below τ ≈ 0.5, better above it,
because its Δ distribution is much wider (`probe14/delta_good_of_correct/std` 1.005 against 0.648
and 0.494). Its peak therefore sits at τ = 1.02, and §9.2's check fires on it at 2.94×.

**`mask_relu2` is a small regression on peak F1 against the baseline: 0.5313 → 0.5074.** It is an
improvement at the natural τ (0.4818 → 0.5046) and a large improvement in τ placement. Which of
those two readings is the honest headline depends on whether τ is fit or fixed, and §9.2 says
fit — so **on its own terms the clean test underperformed the run it was meant to improve.**

### 2.2 `L_NCE` is not muted in `nce_temp`. It is gone.

§9.10.1 predicted `τ = √512` would demote ① from the largest per-term gradient to the second
smallest, MEASURED at 13× less departure from chance by step 50. Over the full 1,460 steps it is
far more than a demotion. Window means over steps ≥ 1200 (MEASURED):

| | `phase1` | `mask_relu2` | `nce_temp` |
|---|---|---|---|
| `nce/loss` | 3.0732 | 3.0575 | 5.6870 |
| `nce/chance` | 5.7068 | 5.7068 | 5.7068 |
| **gap below chance** | **−2.6336** | **−2.6493** | **−0.0198** |
| `nce/categorical_accuracy_backward` | 0.2842 | 0.3352 | 0.0146 |
| — as × chance (`1/(1+R)`) | 90.9× | **107×** | **4.7×** |
| `nce/accuracy_within_question` | 0.3321 | 0.4059 | 0.0992 |
| — as × chance (`1/(1+n_same)`) | 10.6× | **12.7×** | **3.1×** |
| `nce/loss_cross_question` | 2.4157 | 2.4270 | 5.5881 |
| `nce/logit_std` | 4.1586 | 3.8581 | 0.0374 |
| `nce/logit_std × τ` | 4.159 | 3.858 | 0.847 |

**134× less departure from chance** (2.6493 / 0.0198), DERIVED. Note `logit_std × τ` — the
scale-free reading §9.10.1 insists on — is 0.847 in `nce_temp`, so this is **not bug B10a**: the
distance-space spread is real and positive, fp32 is working, the geometry moves. Term ① is simply
contributing almost nothing to it.

### 2.3 §9.10.3's confound fired exactly as written

§9.10.3 flagged the self-defeating pair before launch — the mask is a repair *to* `L_NCE`, and
`τ = √512` mutes `L_NCE` by 22.6×, so the repair is muted with it. The gauge it named was
`nce/categorical_accuracy_backward`. MEASURED, and it fires:

| | `mask_relu2` | `nce_temp` |
|---|---|---|
| `nce/argmax_in_nearer_set` @1200+ | 0.3346 | **0.0147** |
| `nce/categorical_accuracy_backward` @300–600 | 0.1139 | 0.0089 |
| `nce/accuracy_within_question` @300–600, as × chance | 8.3× | **2.5×** |

§9.10.3's own threshold was *"near 2× at step 500 means the mask result cannot be read at all."*
It is 2.5× at 300–600. **The mask's effect is unreadable in the winning run.** Every mask
conclusion in this file therefore comes from `mask_relu2` and from nowhere else.

### 2.4 The mechanism — same detection gap, less noise

The detection signal is essentially identical between the two new runs; the spread is not
(MEASURED @1200+):

| | `phase1` | `mask_relu2` | `nce_temp` |
|---|---|---|---|
| `probe03/gap` (bad − good) | 1.9763 | 1.0874 | 1.0626 |
| `probe14/delta_good_of_correct/std` | 1.0054 | 0.6476 | 0.4941 |
| **gap ÷ std** | **1.966** | **1.679** | **2.151** |
| peak val F1 | 0.5313 | 0.5074 | 0.5597 |

**Peak F1 orders exactly as `probe03/gap ÷ probe14/delta_good_of_correct/std`, monotonically,
across all three runs.** DERIVED. Nothing in §10 currently names this ratio, and it is the
cheapest in-run predictor of the val ceiling available — it is logged every 10 steps and needs no
eval pass. **Proposed as a new diagnostic row; on three points it is a coincidence-sized
sample and should be checked on a fourth run before it is trusted.**

The variance it is reading has an identified source. `L_NCE`'s push-away is unbounded and never
converges (MEASURED, `phase1` and `mask_relu2`):

```
nce/pos_dist    0-100  3.59 -> 100-300 1.38 -> 300-600 2.00 -> 900-1200 2.23 -> 1200+ 2.23   FLAT
nce/neg_dist    0-100  3.61 -> 100-300 1.93 -> 300-600 4.99 -> 900-1200 7.96 -> 1200+ 8.66   STILL CLIMBING
```

That is §9.9.7's standing hypothesis in a directly readable form — *"the loss is not pulling
positives in; it is pushing identity-mismatches out, which is unbounded and cannot converge."*
In `nce_temp` the same series reads 1.19 flat and 1.59 flat: no divergence, and the whole
geometry sits at ~1/5 the scale (`probe07/within_trajectory_spread` 0.783 against 1.066 and
1.476).

**What is NOT established:** that suppressing ① is *the* cause of the F1 gain rather than ζ, or
than the compressed scale interacting with a fixed natural τ. The `nce_temp` edge moves two
knobs. §9 below has the one-line separating run.

### 2.5 `L_I` converges for the first time

`invariance/residual_diagonal`, MEASURED:

| window | `phase1` | `mask_relu2` | `nce_temp` |
|---|---|---|---|
| 100–300 | 0.2348 | 0.2049 | **0.1214** |
| 300–600 | 0.3335 | 0.2908 | 0.0899 |
| 900–1200 | 0.2509 | 0.2464 | 0.0444 |
| 1200+ | 0.2261 | 0.2231 | **0.0373** |

§7.12/§16.21's primary guard is *"`invariance/residual_diagonal ≤ 0.15` by step ~200, and not
rising after"*. **`nce_temp` is the first run in the project's history to pass it.** The other two
breach it at 100–300 and stay breached.

§17 already records that the guard was calibrated against a *simulated* λ_good=0 level of 0.098
while the model's real λ_good=0 level is ~0.26, *"so the guard fires on the baseline run itself"*
— and notes that §7.12 and §16.21 still state the simulated figure and **have not been
corrected.** That correction is still owed, and this run does not remove the need for it: 0.0373
is a real number and it shows the guard's *target* was reachable all along under a different loss
balance.

**An identity cross-check that does NOT close.** §7.12's `Δ_i = −(−log γ) − A_i − B_i` is exact
per (row, column) pair. Applied to window means (DERIVED, and see the caveat):

| | `phase1` | `mask_relu2` | `nce_temp` |
|---|---|---|---|
| `probe02/delta_good_mean` = Δ | −0.3216 | −0.4134 | −0.3074 |
| `backup/delta_mean` = δ | 0.4293 | 0.6090 | 0.4034 |
| `B = δ − 0.6931` | −0.2638 | −0.0841 | −0.2897 |
| **implied `−A = Δ + 0.6931 + B`** | **0.1077** | **0.1956** | **0.0960** |
| `invariance/residual_diagonal` | 0.2261 | 0.2231 | 0.0373 |
| ratio `−A` / residual | 0.48 | 0.88 | 2.57 |

**The implied `−A` does not track the `L_I` residual.** The run with a 6× better residual has
only a 2× smaller `−A`. This is **not evidence against the identity** — the three quantities are
means over three *different* sets (good-step Δ over same-question terminal columns, δ over the
full `R × C` backup matrix, the residual over `R` rows), and the identity is per-pair. It does
mean **the identity cannot currently be checked on the model**, because nothing logs `A` and `B`
paired. One extra key in `losses/temporal.py` would fix that and it is free.

---

## 3. PROBLEM 3 — `ζ` is exonerated. §16.23 does not close and needs a new suspect.

§16.23's stated decision rule, written 2026-08-04 at launch: *"The readout is free and
unambiguous: `backup/delta_mean` must SETTLE at 0.693… what must not happen again is passing
through 0.693 and decaying to ~0.49. **If that recurs at `ζ = 0.1`, this item does not close; it
needs a new suspect.**"*

MEASURED:

| `backup/delta_mean` (target **0.6931**) | 0–100 | 100–300 | 300–600 | 600–900 | 900–1200 | 1200+ |
|---|---|---|---|---|---|---|
| `phase1` (ζ = 0.05) | 1.8447 | 0.8359 | 0.7268 | 0.5636 | 0.4379 | **0.4293** |
| `mask_relu2` (ζ = 0.05) | 1.5689 | 0.8387 | 0.8120 | 0.7158 | 0.6191 | **0.6090** |
| `nce_temp` (ζ = **0.10**) | 1.3067 | 0.6506 | 0.6355 | 0.4986 | 0.4092 | **0.4034** |

**Doubling ζ produced the worst ruler of the three.** The best came from the mask + `relu_squared`
at *unchanged* ζ — 0.6090, within 12% of target and the closest any run has come. The decay
recurred at ζ = 0.1 exactly as the escape clause anticipated, so:

- ✅ **§16.23 does not close. Raising ζ is not the fix.**
- ✅ The item's own instruction applies: it needs a different suspect.
- ❌ **NOT ESTABLISHED:** which of `relu_squared` or the mask bought `mask_relu2`'s 0.429 → 0.609.
  Two changes on that edge.

Supporting series, MEASURED @1200+, confirming this is the exponential branch failing to hold its
own minimiser rather than a clip artifact (§16.23's original reasoning):

| | `phase1` | `mask_relu2` | `nce_temp` |
|---|---|---|---|
| `backup/linear_branch_fraction` | 0.0008 | 0.0003 | 0.0000 |
| `backup/delta_max` (clip `t` = 3.689) | 4.348 | 4.170 | 2.358 |
| `backup/gain` (cap `clip_t_gain` = 20) | 19.75 | 19.14 | 5.51 |

`backup/gain` is `max(γ·exp(δ_clipped))` (`temporal.py:97`), so 19.75 means **some pair in
essentially every batch is pinned at the clip** while the mean δ sits at 0.43. The backup has a
heavy right tail in both τ=1 runs and does not in `nce_temp`. That is consistent with, but not
proof of, ①'s unbounded `neg_dist` being what feeds it.

---

## 4. PROBLEM 4 — `nce_preflight.py`'s verdict is wrong on a masked checkpoint

### 4.1 What happened

`scripts/nce_preflight.py --checkpoint runs/phase1_mask_relu2/final` printed:

```
** The wrong picks concentrate in the nearer set.
   nce_mask_nearer_same_traj is aimed at what is taking the softmax mass. **
```

`runs/phase1_mask_relu2/final/config.yaml` has `nce_mask_nearer_same_traj: true`. **The script
recommended enabling a flag that is already on and that this very checkpoint was trained under.**

### 4.2 And the statistic did not move

| | `runs/phase1/final` (trained **without** the mask, §9.9.6) | `runs/phase1_mask_relu2/final` (trained **with** it) |
|---|---|---|
| `nce/argmax_in_nearer_set` | 0.36908 | **0.37812** |
| null (`nearer_set_size / R`) | 0.00189 | 0.00189 |
| ratio | 195× | **200×** |
| `nce/nearer_set_size` | 0.728 | 0.72775 |
| `nce/columns_with_nearer` | 0.426 | 0.42598 |
| `nce/categorical_accuracy_backward` | 0.2715 | 0.28297 |

Both at `τ_NCE = 1.0`, both 6 batches, both with every mask off inside the preflight. MEASURED.

### 4.3 Why the verdict is wrong, and what the third branch should say

The docstring's dichotomy (`high` ⇒ the mask is aimed correctly; `near zero` ⇒ the mask will be
inert) is written for a checkpoint trained **without** the mask, where a high reading means the
loss is actively fighting rows the ruler says are closest. **On a checkpoint trained with the
mask, those rows were excluded from the softmax and were never pushed away — so the model is free
to, and geometrically should, place them nearest the goal.** A high reading there is the
*intended* outcome, not a diagnosis.

That the number is flat at ~0.37 across both regimes admits two readings and this file does not
choose between them:

- **(a)** After masking the nearer rows are legitimately nearest, so 0.378 is a pass. Consistent
  with `probe02/delta_good_mean` improving in the same run (§5.2).
- **(b)** The statistic never measured what the mask changes — it measures "does the argmax land
  near the goal", which is a property of the representation having learned position, and both
  runs learned position. Consistent with §9.8.5's *"the geometry resolves POSITION along a
  trajectory, not CORRECTNESS."*

**Concrete fix:** branch on `cfg.sampling.nce_mask_nearer_same_traj` and, when it is on, print
the reading with no verdict — or with the inverted one. This is the **B12 / §10.1.1 family**
verbatim: a guard rendering a verdict against a baseline that no longer applies. §14's B12 entry
already generalises the rule — *"a guard that renders a verdict is a guard that can render the
wrong one"* — and this is its fourth occurrence.

---

## 5. What actually landed — and it is exactly the size §9.6.2 predicted

### 5.1 §7.12's two stated targets are hit

§7.12/§9.10.2 set two targets for ⑥ `L_good` at `relu_squared`. MEASURED:

| target | before (§7.12, step 750) | `phase1` (relu) @1200+ | `mask_relu2` | `nce_temp` | goal |
|---|---|---|---|---|---|
| `probe14/delta_good_of_correct/frac_above_natural` | 0.34 | 0.1787 | 0.1086 | **0.0637** | ~0.05 |
| `val_f1.py` fitted τ | 2.39 | 1.0206 | 0.4218 | **0.3503** | ~0.35 (natural 0.3466) |

The supporting tail statistics move together (MEASURED @1200+):

| | `phase1` | `mask_relu2` | `nce_temp` |
|---|---|---|---|
| `probe14/delta_good_of_correct/mean` | −0.3516 | −0.4225 | −0.3344 |
| `.../p99` | 3.0169 | 1.5144 | **1.3346** |
| `.../std` | 1.0054 | 0.6476 | **0.4941** |
| `.../positive_fraction` | 0.2782 | 0.2225 | **0.1755** |
| `good/delta_max` | 4.7948 | 2.3505 | **2.1298** |

§9.10.2 predicted `relu_squared` would reallocate the term's pull toward the tail (top decile
16.5% → 36.9%) and that the honest expectation was *"a better-behaved tail, not a better F1."*
**Both halves of that prediction hold.** The mid-run pathology §7.12 recorded — bulk moving to
−0.412 while `p99` ran to 2.43 and `frac_above_natural` *regressed* to 0.16 — is gone: the tail
now moves with the bulk.

### 5.2 §9.9.7's own criterion for the mask

§9.9.7 order-of-work #3: *"`nce_mask_nearer_same_traj` run, judged on
`within_question_terminal_spread` and `probe02/delta_good_mean` (currently −0.32 against a target
of −0.693), **not on F1**."* MEASURED @1200+:

| | `phase1` | `mask_relu2` | `nce_temp` | target |
|---|---|---|---|---|
| `probe02/delta_good_mean` | −0.3216 | **−0.4134** | −0.3074 | −0.6931 |
| `within_question_terminal_spread` | 2.9054 | **2.2531** | *(not gated)* | ↓ |

**On its own stated criterion the mask passes**, moving `delta_good_mean` 25% of the remaining
distance to target and `within_spread` −22%. It does not pass on `gate/recall_at_1` (§1.3), and
it is a small regression on peak F1 (§2.1). Confounded with `relu_squared` on the same edge.

### 5.3 The size of the win is what §9.6.2 said it would be

Peak ceiling F1 0.5313 → 0.5597 = **+0.0284**. §9.6.2's DERIVED counterfactual said perfecting
the *entire* clean side is worth **+0.038** mean F1, against localisation's +0.281 (later capped
at ≈ +0.14 by §9.7.6). **The prediction was right, including the magnitude.**

Which means the conclusion §9.6.2 drew stands, now with a measurement behind it:

- The good-step tail is closed. `frac_above_natural` 0.34 → 0.064, τ 2.39 → 0.350.
- `acc_error` is still the smaller term in the harmonic mean on **all three** runs
  (0.386–0.443 against `acc_correct` 0.740–0.761), so F1 is pinned to it.
- **The binding constraint is unchanged and it is localisation.** Nothing in this round of
  changes touched it, and none of them was ever going to.

---

## 6. Secondary observations, none acted on

MEASURED @1200+ unless stated.

| observation | `phase1` | `mask_relu2` | `nce_temp` | note |
|---|---|---|---|---|
| `train/grad_norm` (pre-clip; `grad_clip` = 1.0) | 6.44 | 6.44 | 3.21 | **3–6× and binding on every step, in all three runs.** §14/§11's own note: *"if it sits far above 1.0 all run, this stopped being a guard and became an LR rescale."* It has been an LR rescale for three runs. |
| `probe04/symmetric_share` | 0.3727 | 0.3775 | **0.2024** | §2 #11 / §10 #4. `nce_temp` is ~80% asymmetric against ~62% for the others. The quasimetric thesis rests on this share; a change this large should not go unreported. |
| `probe09_4/irreversibility_good_step` | 1.2459 | 1.0773 | **−0.1252** | §9.4's goal-free score. **`nce_temp` is the only run where good and error steps separate by SIGN** (−0.125 vs +0.896). `scripts/goal_free_score.py` exists and per §9.8.5 #2 has never been run. |
| `probe09_4/irreversibility_error_step` | 3.1848 | 2.3975 | 0.8956 | |
| `backup/div_cross_question` | −8.3499 | −7.8183 | −0.6790 | §7.4.2's `ρ` decision, diagnostic #13, *"decide from a curve and not from an argument."* At τ=1 the cross-question backup is **2.3–2.6×** same-question in magnitude; in `nce_temp` it is 1.18×. Never acted on in either regime. |
| `backup/div_same_question` | −3.5696 | −3.0355 | −0.5740 | |
| `probe08/corr_distance_psi_norm` | +0.026 | −0.169 | +0.050 | §10 #8's `r > 0.9` root-cause-D signature is **absent in all three**. |
| `probe01/distinct_goal_ratio` | 0.7039 | 0.7039 | 0.7039 | §9.9.3 claims the nearer mask *"dissolves the 29.6% duplicate-column contradiction."* This ratio is a **sampler property** and cannot move under a loss mask, so **the logged series cannot test that claim.** If the claim is to be checked, it needs a different statistic. |
| `step/delta_at_margin_fraction` | 0.4570 | 0.2513 | 0.2169 | ⑤ reaches `m = 1.386` on **half as many pairs** in the new runs. Read with `probe03/delta_bad_mean` 1.655 → 0.674 / 0.755: the whole Δ scale compressed, so this is not necessarily ⑤ losing. |
| `probe14/delta_boundary/positive_fraction` | 0.7118 | 0.6358 | 0.6884 | |
| `step/z_zero_fraction` | 0.3390 | 0.3390 | 0.3390 | §4.2.1 records 45.4% for the full dataset; the sampled batches realise **33.9%**. Minor, but the two numbers are quoted interchangeably in §7.6.7 and §7.12 and should not be. |
| `step/recovery_fraction` | 0.0061 | 0.0061 | 0.0061 | §16.15's `False → True` recoveries, 0.61% of ⑤'s pairs. §4.2 measures 1.48% at trajectory level. |

---

## 7. What has NOT been measured

1. **Neither new checkpoint has phase 2 or ProcessBench.** `val_f1.py` is a ceiling handed a real
   terminal (§9.5). The one anchor available: `phase1`'s ceiling **0.5313** → its ProcessBench
   mean F1 **0.241** (§9.3.1) — a **2.2× drop**, which is distribution shift (5-step
   Mistral-class val vs 8.8-step Qwen-class ProcessBench, §5), **not** the goal head, since
   §9.8.1 measured the goal to be uninformative. Scaling naively puts `nce_temp` at ≈ 0.25 on
   ProcessBench. **DERIVED and crude — a single-point ratio extrapolated across a
   distribution shift. It is a reason to run the eval, not a substitute for it.**
2. **No `--untrained` `val_f1` null exists for any run.** `val_f1.py` supports the flag
   (`val_f1_untrained.json`) and it has never been used. §10.1.1's lesson — *"a ratio is a
   comparison, and a checkpoint alone gives you one side of it"* — applies to the ceiling exactly
   as it applied to the gate, and the gate is the reason we know it matters.
3. **`nce_mask_sibling_correct_late` has never been run** (§1.3, §16.25). It is the mask aimed at
   the §1 failure.
4. **The `A` / `B` decomposition is not logged paired** (§2.5), so §7.12's identity cannot be
   checked on the model.
5. **Attribution.** Two changes per edge. See §9.

---

## 8. Corrections owed to CLAUDE.md

Recorded here rather than applied, so the edits are reviewable.

| § | what it currently says | what this round measured |
|---|---|---|
| **§9.8.3** | *"NOT YET MATCHED — do not quote 0.618 → 0.276 as established… Re-run both at the same `--questions 200` first."* | Run, matched on the identical 525-terminal set. **Established**, at a larger untrained baseline: 0.6533 → 0.2762 (`phase1`) / 0.3048 (`mask_relu2`). |
| **§9.9.7 / §9.8.5** order-of-work #2 | *"still not done, still the oldest un-run free measurement in this file."* | Done. §1.1 has the table. |
| **§10.1.1** | untrained baseline table (auc 0.904 / recall@1 0.618 / ratio 0.582), and a final-phase-1 column | Reproduced at `--questions 200`: **0.9069 / 0.6533 / 0.5770**. A third column is now available for `mask_relu2`. |
| **§16.23** | *"RAISED TO ζ = 0.1… the readout is free and unambiguous."* | Read. The decay **recurred and worsened** (0.4034 at ζ=0.1 against 0.6090 at ζ=0.05 + mask). **The item does not close and ζ is not the suspect.** |
| **§9.10 / §16.27** | the four-change bundle, *"what the run can say is 'this combination scored X'."* | X = **0.5597** peak ceiling, τ 0.3503. And the sibling `mask_relu2` run gives a second point: 0.5074, τ 0.4218. |
| **§9.10.3** | *"if this run is still near 2× at step 500 the mask result cannot be read."* | 2.5× at 300–600. **Fired.** The mask is unreadable in `nce_temp`. |
| **§9.10.1** | τ=√512 costs *"13× less departure from chance by step 50"* | Over the full run it is **134×**, and `categorical_accuracy_backward` lands at 4.7× chance against 107×. "Reweighting" understates it. |
| **§7.12 / §16.21** | guard `invariance/residual_diagonal ≤ 0.15 by step 200`, calibrated on a simulated λ_good=0 level of 0.098 | §17 already flags this as **uncorrected**. The model's λ_good=0 level is ~0.26; `nce_temp` is the first run to pass the guard, at 0.0373. |
| **§9.9.6 / §10 #1b** | the pre-flight's two-branch verdict | Needs a third branch for masked checkpoints (§4.3). |
| **§9.9.3** | *"the nearer mask dissolves the 29.6% duplicate-column contradiction"* | `probe01/distinct_goal_ratio` is 0.7039 in all three runs — a sampler property. **The logged series cannot test the claim.** |
| **§7.12** | *"val F1 ceiling (`val_f1.py`, skyline goal) **0.456**"* at λ_good=0, step 750 | `runs/phase1_lambda_good_0_baseline/step750/val_f1.json` (dated 2026-07-27 22:59) says **0.1882**, τ 0.1416, at the same 400 questions / 4,549 trajectories. **Discrepancy, unexplained.** Likely a `val_f1.py` change between the reading and the artifact. **Reconcile before 0.456 is quoted again.** |
| **§4.2.1 vs the sampler** | `z = 0` for 45.4% of incorrect trajectories | Realised `step/z_zero_fraction` = **0.3390** in every run. Different populations; do not quote one for the other. |

---

## 9. Order of work

**Free — no GPU, no new checkpoint:**

1. Apply §8's corrections to CLAUDE.md. §9.8.3 and §16.23 are the two that change what the next
   run should be.
2. Patch `scripts/nce_preflight.py` to branch on `cfg.sampling.nce_mask_nearer_same_traj` (§4.3).
3. Re-run the trained `goal_gate.py` with the full command recorded, to remove the §1.1 inference.
4. Log `A` and `B` paired in `losses/temporal.py` so §7.12's identity can be checked (§2.5).
5. `scripts/goal_free_score.py` on all three checkpoints — §9.8.5 #2, never run, and
   `probe09_4` now shows a **sign-separating** irreversibility in `nce_temp` (§6). If plain
   asymmetry matches the pipeline, that is a redirect, not a retune.
6. `val_f1.py --untrained` on any one checkpoint, for the missing null (§7.2).

**One eval pass, no training:**

7. **Phase 2 + ProcessBench on `phase1_nce_temp_relu2`.** Two checkpoints now differ by 0.05 on
   the ceiling and there is no measurement of what that is worth downstream. The 2.2× ceiling→PB
   ratio is a single-point extrapolation and should not stand in for the number.
   Counter-consideration: §9.8.1 exonerated the goal head and §16.24 says *"do not run phase 2
   again"* — that instruction was about **chasing** the goal head, not about obtaining a
   comparable eval number, and this is the latter.

**One run each, both one line:**

8. **Separate τ from ζ**, which is the largest open confound in §2:
   `bash scripts/train.sh --set losses.zeta=0.05 --set run.name=phase1_nce_temp_z05`.
9. **`nce_mask_sibling_correct_late` at τ = 1.0** — the mask actually aimed at §1, judged on
   `gate/recall_at_1` against the untrained 0.6533:
   `--set sampling.nce_mask_sibling_correct_late=true --set run.name=…`.
   Must be at τ = 1.0, for the §9.10.3 reason.

**Not on this evidence:**

- Another `λ_good` / `margin_steps` / good-step-tail change. The tail is closed (§5.1) and the
  measured payoff was +0.028, matching §9.6.2's +0.038 ceiling.
- Raising ζ further (§3).
- §16.26's sibling-terminals-as-positives loss change, until #9 has run — §9.9.7's stopping rule
  needs the *targeted* mask, and it has never been enabled.
