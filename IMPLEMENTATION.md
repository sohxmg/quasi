# IMPLEMENTATION.md — Feynman-PRM

Read this cold. `CLAUDE.md` is the locked design spec and `PLAN.md` is the build plan; this
document is what the code actually does, why, and how it fails.

There is no README by instruction. **`SETUP.md` is how to install and run it on the GPU box**
— commands, expected output, troubleshooting. Section 7 below is the command list in brief.

---

## 1. What the model is

A process reward model that scores each **step** of a chain-of-thought by a learned
**quasimetric distance** `d(state, goal)` — how far this partial solution is from being
solved. Trained on Math-Shepherd, evaluated on ProcessBench.

**The MDP mapping** (§3). One Math-Shepherd row is one trajectory for one question:

| RL concept | Here |
|---|---|
| state `s_i` | the prefix `(question, step_1 … step_i)`, read as the last-layer hidden at the separator token that follows step `i` |
| `s_0` | the state after the question and before any step — a separator is inserted after the prompt so it exists |
| action `a_i` | the text of step `i`; the transition `s_{i-1} --a_i--> s_i` |
| dynamics | deterministic: `s_i = s_{i-1} ⊕ a_i` |
| goal `g` | a later state of a **correct** trajectory of the same question (train) / `goal_head(h_{s_0})` (eval) |

A trajectory with `T` steps yields `T+1` states from **one** LM forward pass. That is the
structural win over the old token-level project and it is what makes a ~347-negative pool
affordable in 16 GB.

**One head, one score: the distance.** There is no value head (locked #3a). ψ and φ are the
two halves of `d`, `goal_head` supplies the destination at eval, and nothing emits an
independent scalar. Every phase-1 loss acts on `d`, so the quantity that is trained is the
quantity that is scored — the old project's root cause C was exactly the opposite.

**Why a quasimetric and not a similarity.** Two properties:

- **triangle inequality** → credit assignment and stitching across partial solutions. Caveat
  that must not be glossed: §4.4 measured that ~95% of states in this dataset have exactly
  one observed action, so the stitching story has *no natural data support here*. `L_CF`
  (§7.5) is the mechanism that would manufacture it, and its data is deferred.
- **asymmetry** (`d(s,g) ≠ d(g,s)`) → detecting irreversible dead-end states. Computed as a
  diagnostic from day one (`probe09_4/*`), never reported without a decision (§16.10). Note
  `full_mrn` at latent 512 measured ~73% *symmetric*, so the asymmetric term is a minority
  and claims about it need the logged split behind them.

**Files:** `model/distances.py` (the metric), `model/heads.py` (ψ, φ, goal_head),
`model/wrapper.py` (one forward → ψ, φ, act_emb).

---

## 2. Index conventions — read this before touching anything

Every off-by-one in this project lives here. `utils/indexing.py` is the single source, and
`tests/test_indexing.py` + `tests/test_step_loss.py` pin it.

```
[prompt tokens] SEP [step 1 tokens] SEP [step 2 tokens] SEP … [step T tokens] SEP
                 ^                   ^                   ^                     ^
                s_0                 s_1                 s_2                   s_T
```

| Object | Definition |
|---|---|
| `completions[k]`, `labels[k]` | the step that takes `s_k → s_{k+1}`, 0-based over `T` steps |
| `z` | first `k` with `labels[k] == False`, **0-based**. `-1` if fully correct |
| `ψ_z` | the **last good** state |
| `ψ_{z+1}` | the **first broken** state |
| `φ_i` | `phi(h_{i-1}, act_emb_i)`, `i = 1…T` — departs from `s_{i-1}`, lands in `s_i` |
| `Δ_i` | `d_i − d_{i-1}` for `i = 1…T`; **the cost of `steps[i-1]`** |
| prediction | `i* = first i with Δ_i > τ` ⇒ `predicted_label = i* − 1`, else `−1` |

**Never write `ψ_{z-1}`.** It is a *good* state, and it does not exist at all for the 45.4%
of incorrect trajectories with `z = 0`.

**Why this is the section that matters.** A model trained on `Δ_z` instead of `Δ_{z+1}`
converges, separates `Δ`, and predicts `z−1` on **every** errored sample: `acc_error → 0` and
F1 collapses through the harmonic mean while every loss curve still looks healthy. The round
trip is asserted in `tests/test_indexing.py::test_training_target_round_trips_to_the_right_prediction`
and the gradient-level version in `tests/test_step_loss.py::test_reads_d_z_plus_1_minus_d_z_and_never_touches_d_1`.

**Separator positions come out by arithmetic, never by scanning.** `data/tokenize.py`
tokenises each segment separately and records positions by cumulative count, so §4.7's hazard
(13.9% of solutions contain a step with an internal newline) cannot occur. CRM's post-hoc
`assert ids[pos] == sep_id` is kept as a cheap guard. There is exactly **one** sequence
builder in the repo and a grep test enforces it.

**Batch index maps** (`data/collate.py`, built on CPU from pre-tokenised rows only):

```
input_ids (B,L)   attention_mask (B,L) or None when nothing is padded
state_flat_idx (S,)  state_traj (S,)  state_step (S,)  traj_state_offset (B,)
row_src (R,)  row_dst (R,)  row_traj (R,)  row_step (R,)
traj_qid (B,)  traj_correct (B,)  traj_T (B,)  traj_z (B,)  traj_recovery (B,)  traj_terminal (B,)
span_token_idx (P,)  span_row_idx (P,)  span_counts (R,)      # segment-mean action pooling
```

`data/goals.py` then adds `goal_state (C,)`, `pos_row (C,)`, `goal_traj (C,)`,
`is_terminal (C,)`.

Every loss is a pure function of `(psi, phi, act_emb, these tensors)`. That is why the whole
CPU suite runs on random hiddens with no model and no GPU.

---

## 3. The five losses

Total (phase 1), `losses/total.py`:

```
L = λ_NCE·L_NCE + λ_I·L_I + ζ·L_T + λ_CF·L_CF + λ_step·L_step
```

**ζ weights the backup only** (`tmd.py:124`); action invariance sits at 1.0. `tmd.py:362`'s
own comment says otherwise and is wrong — the code wins. The old project put `L_I` under
ζ=0.1, making it 10× weaker than TMD's setting, and its residual never got below 0.43.

### The three matrices, built once each (`losses/matrix.py`)

| tensor | shape | grad | consumers |
|---|---|---|---|
| `Dist[r,c] = d(φ_r, ψ(g_c))` | R×C ≈ 348×172 | yes | ① and ③ — **the same tensor object**, asserted by identity |
| `Next[r,c] = d(ψ(s_r), ψ(g_c))` | R×C | no | ③ only. Unconditionally detached (`tmd.py:113`), so `no_grad` also halves its activation cost |
| `D_term[s,t] = d(ψ_s, ψ(s_T^t))` | S×T_c ≈ 404×28 | yes | ⑤, diagnostics #2/#3/#14, the §10.1 gate |

`Dist` is **rectangular and has no diagonal** — rows from incorrect trajectories are
negative-only. Every "diagonal" in `tmd.py` becomes a `pos_row` gather; a grep test forbids
`torch.diagonal`. `D_term` is our addition and it is what makes the design cheap: `L_step`
and the three-way Δ histogram read the same small matrix, and it is the *eval-shaped* query.

### ① `L_NCE` — `losses/nce.py` (§7.2, `tmd.py:91-98`)

```python
loss = F.cross_entropy(-Dist.t() / tau, pos_row)     # softmax over SOURCE ROWS per goal column
```

TMD's **backward** NCE. The spec's own annotation fixes the direction: negatives "keep the
goal, change `(s_i, a_i)`", from the same trajectory at other states and from other
trajectories, "correct or incorrect soln" — the negative pool explicitly includes incorrect
solutions, which is already a correctness signal.

`τ = 1.0` is a **documented divergence** from TMD's `1/√512 = 22.6`: at 22.6 our O(1–10)
distances become O(0.05–0.5) logits, i.e. the near-uniform softmax that is bug B10a's
signature. It is a float knob — raise it if `logit_std` blows up.

*Fails as:* pinned at `log(R)` with `logit_std ≈ 0` → bug B10a, the fp32 cast is not
effective at runtime. Pinned at `log(R)` with `logit_std > 0` and pos ≈ neg → geometry
collapse, check goal duplication (probe #1).

### ② `L_I` — `losses/invariance.py` (§7.3, `tmd.py:100-105`)

```
L_I = mean_r d( ψ(s_{row_src[r]}) , φ_r )        elementwise, diagonal
```

Argument order matters: `d(ψ(s), φ(s,a)) = 0` does **not** imply `φ = ψ(s)` in a quasimetric;
it means "φ(s,a) is reachable from s at zero cost". In MRN, `d(x,y)=0` iff `x_sym = y_sym` and
`x_asym ≤ y_asym` elementwise, so φ can sit at zero cost *from* `ψ_{i-1}` while being strictly
farther from `g`. **That slack is exactly the room `L_step` needs**, and it is why `L_I` at
weight 1.0 does not trivially flatten Δ. The old project measured the slack at +0.276 with
`L_I` stuck at 0.43; both are logged (`invariance/residual_diagonal`, probe #9).

The two `grid_*` modes exist only to reproduce §16.3's failure. They drive `φ(s,a) → ψ(s)`,
which pins `Δ` at `−log γ` good step or bad and makes `L_step` immovable — so the config
**refuses** `grid_*` while `lambda_step > 0` rather than letting you debug a silent no-op.

*Fails as:* residual plateauing (the old project's 0.43) → the action representation is not
being used, or the grid is on.

### ③ `L_T` — `losses/temporal.py` (§7.4, `tmd.py:107-122`)

**This is the ruler.** `L_NCE` only teaches ranking; `L_T` sets the scale: one good step
shrinks the distance by exactly `−log γ = 0.693`. Distances then read in units of *steps
remaining*, which is what makes one global τ mean the same thing on every question — and
`L_step`'s margin is expressed in those same units.

```python
t     = clip_t_steps * (-log gamma)                  # 19.75 at discount 0.5, scale-free
delta = Dist - Next                                  # Next already detached
mask  = delta > t
div   = torch.where(mask, delta, gamma * torch.exp(torch.where(mask, t, delta)) - Dist)
L_T   = (1-dw) * (rho*div.mean() + (1-rho)*div[SQ].mean()) + dw * div[pos_row, arange(C)].mean()
```

Three things not to "simplify":

1. The **double `torch.where`** is exponent clipping and it is load-bearing: `where` evaluates
   the discarded branch in the backward pass, so an `inf` there poisons the gradient. Fixture
   `delta = 200` must give finite loss **and** finite gradients (tested).
2. The branches are discontinuous in value *and* slope at `delta == t`. Do not smooth them.
3. Branch B subtracts **`Dist`**, not `delta`.

**All pairs, not matched pairs** (§7.4.1): ~60,000 backup terms per step, not ~348. It costs
nothing (`Dist` already exists) and it is the point — matched-only calibration lets the model
learn a different scale per question ("one step = 0.36 on algebra, 5.0 on geometry"), which
satisfies the loss perfectly and breaks a single global τ.

Minimiser: `Dist = Next − log γ`; value there is `1 − Dist`, **so L_T is expected to be
negative**. Watch plateau and NaN, not sign.

*Fails as:* `backup/linear_branch_fraction` climbing → `t` is too tight, raise
`clip_t_steps`, do not touch `discount`. `backup/div_cross_question` diverging → the
cross-question mass is eating the loss, lower `goal_scope_ratio` (§16.18, decide from the
curve).

### ④ `L_CF` — `losses/counterfactual.py` (§7.5). Built, `λ = 0`, data deferred.

Cross-entropy over `{meaning-preserving rewrite} ∪ {meaning-changing rewrites}` with `f = −d`,
positive at index 0. Format and loader in `data/counterfactual.py`.

**§7.5's cost claim is wrong and it matters** (PLAN finding 1): `φ_i = phi(h_{i-1}, act_emb_i)`
and rewriting step `i` leaves the prefix — hence `h_{i-1}` — untouched. Only `act_emb`
changes, and that is a mean of *input embeddings*, not a hidden state. A variant costs an
embedding lookup plus an MLP. So when data exists, `L_CF` can run on every step of every batch
essentially free, and it is the most direct pressure available on §16.7's bag-of-words action
representation.

### ⑤ `L_step` — `losses/step.py` (§7.6, locked #3b). The correctness loss.

```
L_step = -log σ( d(ψ_{z+1}, g) - d(ψ_z, g) - m ),    m = margin_steps * (-log γ) = 1.386
```

averaged over (incorrect trajectory × correct-terminal goal of the same question) pairs.
`g = ψ(s_T)` of a correct trajectory **in the batch** — a real terminal, never a prediction,
never a centroid, never a geometric-sampler column. That decouples it from `discount`: the
term carrying correctness queries an ending-like goal 100% of the time, which is what made
`discount = 0.5` affordable.

Equivalently it is Bradley-Terry on `Δ_{z+1}` — **the exact statistic eval thresholds**. No
other phase-1 loss touches it, and §7.6.2's gradient table shows it is the only term that
trains ψ *as a source*. It is not optional.

At init `Δ_{z+1} ≈ 0`, so `L_step = log(1 + e^m) = ln 5 = **1.6094**` exactly. If it is not
1.609, the margin or the `z` indexing is wrong.

*Fails as:* `Δ_{z+1}` refusing to go positive → φ is ignoring its action input (§7.6.5); this
is a faster diagnosis of §16.3 than probe #3. `step/distinct_z` near 21 → the sampler is on
`2c+1i`; exactly 96 pairs → the caps are being applied as quotas.

**The corrected §7.6.5 derivation** (PLAN finding 3 — §7.6.5's sign is wrong in the spec):
`d(ψ_{i-1},g) ≤ d(ψ_{i-1},φ_i) + d(φ_i,g) → 0 + d(ψ_i,g) − log γ`, so **`Δ_i ≥ log γ =
−0.693`**, not `≥ +0.693`. The bound is one-sided, a large positive `Δ_{z+1}` does not violate
it, and the φ→ψ collapse pins Δ at `log γ = −0.693`. **Both of §7.6.5's conclusions survive
intact**; only the sign in the write-up was off.

### `L_goal` (phase 2) — `losses/goal.py` (§7.7)

```
L_goal = mean over c of [ d(pred_q, ψ(s_T^c)) + d(ψ(s_T^c), pred_q) ]
```

The mean **of distances**, never a distance to a mean — a latent centroid over 30k terminals
collapses onto the population mean (root cause D). Both directions, because a one-way distance
would let the guess drift to somewhere *reachable from* the ending rather than *being* it.

---

## 4. Hyperparameters

`config/default.yaml`, strict-parsed by `config.py`: an unknown or misspelled key is a hard
error (old bug B4).

| key | value | where it came from | what it does | safe to tune? |
|---|---|---|---|---|
| `discount` | **0.5** | chosen 2026-07-25 from §4.5's measured table | goal sampler **and** backup γ — one key, two uses | yes, but **only** to 0.7 (the other sanctioned row). Never 0.99: 1.16 distinct goals = root cause B |
| `heads.latent_dim` | 512 | `tmd get_config()` | ψ/φ output dim | 1536 measured ~80% symmetric (old R6) — don't |
| `distance.variant` | `full_mrn` | locked #11 | metric family | `asym_only` if §9.4's asymmetry score is ever promoted |
| `losses.nce_temperature` | 1.0 | **divergence** from `tmd.py:92`'s 22.6 | logit scale | yes, raise if `logit_std` blows up |
| `losses.zeta` | 0.05 | `tmd get_config()` | weights **the backup only** | unvalidated (§16.8) |
| `losses.lambda_i` | 1.0 | `tmd.py:124` | invariance, **not** under ζ | unvalidated |
| `losses.lambda_step` | 1.0 | locked #3b | correctness | **not optional** (§7.6.2) |
| `step_loss.margin_steps` | 2.0 | the human's stated intent | `m = 2·(−log γ)` | drop to 1.0 if `backup/div_same_question` stops converging (§16.19) |
| `backup.clip_t_steps` | 28.5 | reproduces TMD's bare `t=3.0` at γ=0.9 | LINEX guard, **in steps** | raise if probe #15 climbs |
| `backup.diag_backup` | 0.5 | `tmd.py:364` | matched-vs-rest mix | TMD-faithful |
| `backup.goal_scope_ratio` | 1.0 | **ours**, no TMD counterpart | whole-batch vs same-question non-matched mass | decide from probe #13, not from an argument (§16.18) |
| `sampling.sequences_per_micro_batch` | 56 | §8.1 | the budget | lower it **and raise grad_accum together** if it OOMs |
| `sampling.max_*_per_question` | 4 / 3 | §8.1.1's sweep | **caps, not quotas** | flat across 3–4 × 3–4; don't spend time here |
| `train.grad_accum` | 2 | §11.1 | — | **never raise alone** — that cuts the optimizer-step count and is the knob that hid the 106-step regression |
| `data.n_questions` | 23000 | §8.2 | ~100k sequences/epoch | cut this, not `grad_accum`, if GPU time is short |

**Everything below follows from `discount` and must never be set independently:**

| follows from `discount` | at 0.7 (fallback) | **at 0.5 (chosen)** |
|---|---|---|
| per-good-step cost `−log γ` | 0.3567 | **0.6931** |
| `L_step` margin `m` | 0.7133 | **1.3863** |
| `L_step` at init | 1.1120 | **1.6094** |
| natural eval τ | 0.1783 | **0.3466** |
| backup clip `t` | 10.17 | **19.75** |
| goals landing on an ending | 55.0% | 41.1% |
| distinct goals / 6-step solution | 3.45 | **4.19** |

`Config.neg_log_gamma`, `Config.step_margin`, `Config.clip_t` are properties, so there is no
way to set them out of sync. `tests/test_goals.py::test_discount_reaches_BOTH_consumers`
asserts one key moves both consumers — the test that would have caught the old two-key split.

---

## 5. Batch composition

`data/sampler.py`. The budget is a fixed number of **sequences** (56), not of questions:

```
for each selected question, in shuffled order:
    take min(4, k_correct) correct  +  min(3, k_incorrect) incorrect
    if that allocation does not fit, EMIT THE BATCH SHORT and carry the question over
```

**4/3 are caps, not quotas.** §4.2.1 measured that only 20.0% / 29.5% of questions can fill
them and the median question has 2 of each. A hard quota discards most of the dataset. And a
*partially* included question can end up with 0 correct or 0 incorrect, which silently
produces goal-less rows and zero `L_step` pairs — hence "no question is ever split".

Realised numbers (§8.1.1, measured, not estimated):

| | value |
|---|---|
| questions per batch `Q` | ~12.9 |
| source rows `R` | ~348 |
| goal columns `C` | ~172 |
| negatives per goal column | ~347 (`R − 1`, whatever `Q` is) |
| `L_T` terms | ~60,000 |
| `L_step` pairs | **64** |
| `L_step` distinct `z` | **~28** ← the number that matters |

**Read distinct `z`, not pairs.** The `k_c` goals all compare against the same `ψ_z`; one
error step measured against four terminals is not four independent signals. (§7.6's "96 pairs"
line is stale — §8.1.1 measured 64, and 96 means the caps are being applied as quotas.
Diagnostic #17 asserts against 64.)

**One epoch = one visit per selected question.** A visit samples `min(cap, available)`, so a
question with 10 correct solutions contributes 4 and the rest are never seen that epoch.
§2 #1's "take all trajectories" governs dataset *selection*, not what the sampler consumes.

**Length-grouped batching is on by default.** Shuffle the epoch's questions, sort by longest
allocated sequence inside megabatches of ~50 batches, form batches from consecutive runs,
shuffle batch order. Padding falls from ~60% to ~10% (est. ~4 h → ~2 h per epoch), batches
with no padding pass `attention_mask=None` so SDPA picks its fastest backend, and peak memory
moves to the longest bucket — which is why **the longest batch of the epoch runs first as a
memory probe**. `group_by_length: false` reproduces the spec-literal order; `Q` and every
§8.1.1 count hold either way.

---

## 6. Diagnostics

`diagnostics/probes.py` + the `info` dicts the losses return. Logged to
`runs/<name>/metrics.jsonl` and the console; wandb is wired and optional.

| # | key | read it as |
|---|---|---|
| 1 | `probe01/distinct_goal_ratio`, `negatives_per_column` | distinct ≪ columns → root cause B is back |
| 2 | `probe02/delta_good_mean` vs `target_good_step_delta` (−0.693) | the old project's was **108× off** and nobody noticed |
| 3 | `probe03/gap` | **this gap IS the signal.** Collapsing → the error signal is being flattened |
| 4 | `probe04/symmetric_share` | if asymmetric is a minority, do not claim asymmetry drives the result |
| 5 | `gate/ratio` (phase 2 / `goal_gate.py`) | **the goal-head go/no-go gate**, §10.1 |
| 6 | `goal/pred_variance` | ≈ 0 → the head learned a constant, i.e. a global anchor |
| 7 | `probe07/within_trajectory_spread` | → 0 with masking off → states are being squashed; flip `nce_mask_same_traj` |
| 8 | `probe08/corr_distance_psi_norm` | `r > 0.9` → the goal contributes nothing; root cause D recreated |
| 9 | `invariance/residual_diagonal` | the old project's plateaued at 0.43; target 0 |
| 10 | `nce/logit_std`, `logits_pos`, `logits_neg`, `categorical_accuracy_backward` | §14's stuck-NCE table |
| 11 | every `*/loss` separately | they were never designed to be additive |
| 12 | `probe12/off_over_on` | expect ~2 within incorrect trajectories |
| 13 | `backup/div_same_question` vs `div_cross_question` | the `ρ` decision, and the `margin_steps` watch |
| 14 | `probe14/delta_{good_of_correct,good_of_incorrect,boundary}/*` | **the single best predictor of ProcessBench F1** |
| 15 | `backup/linear_branch_fraction` | the clip guard; should fire rarely |
| 16 | `probe16/goal_is_terminal_fraction` | should match §4.5 (41% at 0.5) — the train/eval mismatch |
| 17 | `step/distinct_z`, `step/pairs`, `step/recovery_fraction` | expect ~28 and 64 |

**#14 is the one to watch.** `L_step` never sees a correct trajectory, so `L_T` alone
suppresses false positives. A positive tail on `delta_good_of_correct` is F1 leaking with no
loss training against it — and *that*, not term count, is the trigger for a §7.10 pairing
expansion.

### Expected at initialisation (§18 — compute these, do not eyeball them)

| quantity | expected |
|---|---|
| `L_NCE` | `log(R) ≈ log(348) = 5.85` |
| `L_step` | **exactly `ln 5 = 1.6094`** (1.1120 at discount 0.7) |
| `L_T` | `≈ γ − mean(Dist)` — see the caveat below |
| `Q` | ~12.9 |
| `L_step` pairs / distinct `z` | 64 / ~28 |

> **Correction to §18 on `L_T`'s init value.** §18 says it "starts positive, ≈ γ". That holds
> only while `delta ≈ 0` **and** the mean distance is below γ. With Xavier-initialised heads at
> latent 512 the distances are O(1), so a *negative* `L_T` at step 1 is not by itself evidence
> the backup is broken. `expected_init_values(cfg, R, mean_dist)` reports `γ − mean(Dist)`, and
> what matters either way is that it **falls** and does not plateau or NaN.

### Failure signatures

| symptom | probe | root cause | change |
|---|---|---|---|
| `L_NCE` pinned at `log(R)`, `logit_std ≈ 0` | #10 | bug B10a — fp32 cast not effective | confirm the cast is inside the distance; clean restart |
| `L_NCE` pinned, `logit_std > 0`, pos ≈ neg | #10, #1 | geometry collapse / goal duplication | check `distinct_goal_ratio` |
| `probe03/gap` → 0 | #3, #14 | φ ignoring its action (§16.3) | confirm `action_invariance: diagonal`; consider `action_pool: attention` |
| `Δ_{z+1}` will not go positive | #14, ⑤ | same as above, diagnosed faster | ditto (§7.6.5) |
| positive Δ tail on good steps of **correct** trajectories | #14 | false positives `L_T` is not suppressing | a §7.10 pairing expansion — the only sanctioned trigger |
| `L_T` diverging | #13, #15 | cross-question mass, or `t` too tight | lower `goal_scope_ratio`; raise `clip_t_steps` |
| `backup/div_same_question` stops converging | #13 | `L_step` winning too hard | `margin_steps` → 1.0 (§16.19) |
| fitted τ ≈ 0 instead of 0.347 | §9.2 | the second step of margin never landed | `margin_steps` → 1.0, don't fight it |
| within-trajectory spread → 0 | #7 | §16.4's false negative biting | `nce_mask_same_traj: true` |
| `step/distinct_z` ≈ 21 | #17 | sampler still on `2c+1i` | check the caps |
| `step/pairs` exactly 96 | #17 | caps applied as **quotas** | check `_allocate` uses `min(cap, available)` |
| `optimizer_steps` prints ~106 | launch | the `n_questions`/`grad_accum` regression | cut `n_questions`, **never** raise `grad_accum` alone |
| terminal spread `within` grows while the gate worsens | #5 | `L_step`'s attached `g` leaking into terminals | detach `g` (§16.17) |

---

## 7. How to run it

```bash
conda create -n feynman python=3.12 -y && conda activate feynman
pip install torch --index-url https://download.pytorch.org/whl/cu128   # REQUIRED for sm_120
pip install -r requirements.txt && pip install -e .
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True                # every GPU shell
tmux new -s feynman                                                    # mandatory for any real run
```

| # | command | what it should print |
|---|---|---|
| 0 | `pytest tests/test_gpu.py -m gpu -v -s` | 26 tests: environment preflight, both trainability asserts, the bf16/fp32 seam, gradient checkpointing with `inputs_embeds`, `h_{s_0}` short-vs-full, the memory probe (**prints the measurement**), save/resume, `merge_and_unload`, the phase-2 freeze, and the eval scoring path |
| 1 | `pytest tests/ -m "not gpu"` | all green, no model, no GPU, no download |
| 2 | `python scripts/prepare_data.py` | 45,989 questions → 40,247 trainable → 36.6% correct; the **real** tokenised length distribution; the selection SHAs; ~105k branch nodes / ~63k disagreeing |
| 3 | `bash scripts/smoke_test.sh` | every loss finite and backward on random hiddens; same seed → identical losses |
| 4 | `bash scripts/train.sh --max-steps 20` | step-count assert, trainability assert, longest-batch peak VRAM, padding ~10%, init values, no NaN |
| 5 | `python scripts/goal_gate.py --checkpoint runs/phase1/final` | **ratio < 0.3 → proceed; → 1 → stop and redesign** |
| 6 | `bash scripts/train.sh` | ~889 optimizer steps, est. 2–4 h. Watch probe #14 |
| 7 | `python -m feynman_prm.train_goal_head --checkpoint runs/phase1/final` | cache, gate, then minutes of fitting |
| 8 | `bash scripts/eval_processbench.sh runs/phase1/phase2/final` | per-subset F1, the math 587-leaked vs 413-clean split, τ and its sensitivity, the labelled skyline |

Other tools: `python scripts/plot_metrics.py runs/phase1/metrics.jsonl [--csv out.csv]`
(ASCII sparklines, dependency-free), `python scripts/export_merged.py --checkpoint ... --out ...`
(merge LoRA into a standalone model directory, heads copied alongside).

**If it OOMs:** lower `sampling.sequences_per_micro_batch` *and* raise `grad_accum` to keep
the effective batch — but never raise `grad_accum` alone.

---

## 8. Deviations and open risks

### Deviations from `CLAUDE.md`, each with a reason

1. **No `losses/extras.py`** (§12's tree). The §7.10 expansions are `step_loss.pairing`
   values, so they live in `step.py` with the boundary form. §11's own config comment already
   says this.
2. **`data/tokenize.py` added** and it is the only place a sequence is built. Root cause I was
   two prompt templates diverging by whitespace. A grep test enforces the singleton.
3. **Checkpointing does not merge adapters** (§14's rule). `save_checkpoint()` writes
   `adapter/`, `heads.pt`, the resolved config and the tokenizer, and **asserts the head state
   dict is non-empty** — which is the failure §14's rule was protecting against.
   `scripts/export_merged.py` does `merge_and_unload()` when a standalone directory is wanted.
   Checkpoints stay ~100 MB instead of 3.1 GB, so mid-run saves are affordable.
4. **Two §11 config keys dropped.** `sampling.hard_negatives_post_error`: under §8.1's layout
   every state of every selected trajectory is already a source row, so there is no negative
   pool to bias and the flag has no implementable meaning — diagnostic #12 logs the realised
   on-track/off-track ratio instead. `action_invariance.grid_max_actions`: only reachable from
   the two reproduce-the-failure modes, hardcoded to 64 there.
5. **Keys §11 lists that are not config**, because they must never vary: fp32 distances,
   pad-to-batch-max, the separator-after-prompt, `strip_step_prefix`, `log_selection_sha`,
   `add_step_prefix` at eval, `report_math_leak_split`, `layer_norm`, `hidden_size` (always
   read from the downloaded `config.json`). `optimizer: adamw` and `adam_foreach: false` are
   hardcoded for the same reason (bugs B8, B9).
6. **`train.betas` / `weight_decay` / `bf16` / `schedule` are kept as config keys** — PLAN's
   abbreviated YAML omitted them but did not list them among its drops, so they carry over
   from §11 verbatim.
7. **`heads.ensemble` is a knob that refuses to be true.** The human's call was off (§16.6);
   wiring 2 members through every loss (TMD averages the critic loss over members and takes
   `min` at read time) is not built, and the config says so explicitly rather than silently
   ignoring the flag.
8. **`asym_only` uses the whole component dim**, not just the half `full_mrn` reserves for the
   max. Otherwise half the latent would be dead in that variant. `d(x,x)` is exactly 0 there.
9. **The config refuses `action_invariance: grid_*` while `lambda_step > 0`** (§7.6.5). Set
   `lambda_step: 0.0` to deliberately reproduce §16.3's failure.
10. **Over-length ProcessBench samples predict `−1` and are counted**; >1% of a subset is a
    hard failure (`assert_truncation_budget`). Truncating would drop trailing separators and
    shorten `T`.
11. **Parquet IO lives in `data/math_shepherd.py`** rather than a new module, to keep the §12
    file list intact.
12. **`scripts/plot_metrics.py` is dependency-free** (ASCII sparklines + CSV) because
    matplotlib is not in `requirements.txt`.
13. **The memory probe runs a real forward+backward and then discards the gradients.** It
    costs one micro-batch (~6 s) and turns a three-hour OOM into a 30-second one.

### Deviations from TMD, all deliberate

| | |
|---|---|
| `τ_NCE = 1.0` | vs `1/√512 = 22.6` (`tmd.py:92`) — bug B10a's signature (§7.2) |
| `discount = 0.5` | vs 0.99 — 0.99 gives 1.16 distinct goals on 6-step solutions = root cause B |
| `clip_t_steps` | vs a bare `t = 3.0`, which silently changes meaning with `discount` |
| `goal_scope_ratio` | ours, no TMD counterpart: their random goals are *reachable*, ours are not |
| rectangular `Dist`, `pos_row` | TMD's is square because every transition samples its own goal |
| no `actor_loss`, no stochastic dynamics, no policy extraction | we build a verifier, not a policy |
| `binary_accuracy`, `value_exp`, `contrastive_only` not ported | `binary_accuracy` is pinned at `1 − 1/B` because distances are non-negative — vestigial |
| `categorical_accuracy` measured in the **backward** direction | TMD's own logs the forward one (`tmd.py:127`) while its loss normalises over sources (`tmd.py:97`). We report what is optimised, and say so |

### Deviations from CRM

Its stack cannot be copied: `torch==2.6.0` has no cu128 build, so it does not support sm_120
at all. We take its **API era** (transformers ≥4.56 style, `AutoModel`, `load_dataset`, plain
`accelerate`) and none of its versions — no trl, verl, deepspeed, vllm, flash-attn. `dtype=`,
not the deprecated `torch_dtype=`. We add a separator after the prompt (CRM has no `s_0`) and
we drop over-length rows rather than truncating, which is CRM's own reasoning applied to our
failure mode.

### Open risks

1. **`L_CF`'s data is deferred, and §4.4 says that is the load-bearing gap.** ~95% of states
   have exactly one observed action, so the stitching/triangle story has no data support until
   real rewrites exist. **The ~63k mined branch points are NOT a substitute** (they are two
   meaning-*changing* continuations with a correctness label; `L_CF` needs a meaning-*preserving*
   positive). They are written to `data/processed/branch_points.jsonl` as §8.3's held-out
   diagnostic set.
2. **The action representation is bag-of-words** (§16.7) and this matters *more* under
   `diagonal` invariance, because the grid used to mask a weak action representation.
   `action_pool: attention` is implemented and off.
3. **Every loss weight is unvalidated** (§16.8). Log every curve separately.
4. **The train/eval goal-type mismatch widened at `discount = 0.5`**: eval queries an ending
   100% of the time, the sampler 41%. `L_step`'s goal is a real terminal so the correctness
   term is at 100% regardless; the cost falls on `L_NCE`/`L_T` columns. Probe #16 logs it, and
   0.7 is a one-line revert if the gate looks weak.
5. **`L_step`'s `g` is not detached** (§16.17). With `g` attached, the cheapest way to satisfy
   the loss may be to move `g` away from `ψ_z` rather than `ψ_z` away from `g` — i.e. the
   gradient leaks into exactly the terminal representations phase 2 must predict. Watch probe
   #5: if `within` grows while the gate ratio worsens, detach it.
6. **7B is arithmetically out** (closing §16.5 without a decision): `Qwen2.5-Math-7B` in bf16
   is ~15.2 GB of weights alone on a 16 GB card, before LoRA states, a 56×600-token activation
   stack and ~0.5 GB of fp32 distance matrices. 4-bit QLoRA would fit but fights the mandatory
   fp32 distance path. **1.5B-Instruct stands.**
7. **`qprm_baseline2/` does not exist in this workspace.** `OLD_PROJECT.md` §16 says to port
   its working PyTorch MRN, its `bregman_dt`/`_backup_divergence`, its 31 CPU tests and its
   per-token trace tooling. `goal-conditioned-rm/` is only the upstream Scale AI OpenRLHF fork
   (zero hits for `mrn`, `quasimetric`, `psi`, `phi`, `latent_dim`, `asym`). So every line here
   comes from TMD's JAX, and the §15 tests on the distance and the backup are **load-bearing
   rather than confirmatory**. If a copy exists on the GPU box, say so — it would de-risk the
   two hardest functions.
8. **`OLD_PROJECT.md`'s own AUROC table is milder than "at chance".** §1.2 motivates the whole
   design with "`-d_MRN` came out at chance", and the trace evidence for that (terminal ranked
   70/112, adj R² = −0.001) is genuinely damning — but the AUROC block appended at the end of
   that file reads 0.56 / 0.67 / 0.64 / 0.73 / 0.67 on five of seven benchmarks. No decision
   here depends on which framing is right, but do not describe the old distance as a flat
   failure in a write-up without checking that table.
9. **The committed HuggingFace token in `goal-conditioned-rm/` is live-format and permanently
   in git history** (`examples/experiments/train/train_rm.sh:9`, first commit). Nothing from
   that repo is copied here, but it is a real `hf_` token in a checkout on your disk —
   **rotate or report it**, don't just avoid the file. The same tree hard-codes private S3
   buckets, a private W&B host and personal EFS paths.
10. **Nothing has been run on a GPU.** Every number in this document that is not marked
    "measured" is arithmetic or an estimate: the ~9 GB / ~12 GB memory figures, the ~2 h/epoch
    wall clock, and the ~10% padding fraction. Step 4's probe replaces all three with
    measurements. The CPU suite (160 tests) is green and the full phase-1 inner loop has been
    dry-run against a stub backbone (losses finite, LR schedule moves, `L_step` exactly
    1.6094 at init); `tests/test_gpu.py` has never executed, and neither has any code path
    that touches `transformers`, `peft` or `datasets` — the dev box has none of them
    installed, by design.
