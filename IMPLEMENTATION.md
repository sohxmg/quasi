# IMPLEMENTATION.md — Feynman-PRM

Read this cold. It is what the code actually does, why each decision was made that way, and how
it fails. `CLAUDE.md` is the locked design spec (long, and the authority on any disagreement);
`PLAN.md` is the build plan; **`SETUP.md` is how to install and run it on the GPU box** —
commands, expected output, troubleshooting. There is no README by instruction.

**Current state, one paragraph.** Phase 1 (the metric) is built, tested and has been trained
twice on a real GPU. Phase 2 (the goal head) has **never been trained** and **ProcessBench has
never been evaluated** — every F1 number in this document is a *val-set ceiling* measured with a
real terminal substituted for the goal head (§8). The first run finished at an F1 ceiling of
**0.456**; the diagnosed cause was fixed by adding a sixth loss term (⑥ `L_good`), the second
run is mid-flight, and while it was running a **second, larger problem** surfaced that neither
run controls for: the temporal ruler that gives the whole metric its scale is not being
enforced — it decays (§9). That is the open item. Do not start a third run without reading §9.

**Section map**

| § | |
|---|---|
| 0 | How to read a number in this repo — measured vs simulated vs derived |
| 1 | What the model is |
| 2 | Index conventions — read before touching anything |
| 3 | The losses, each with its rationale and its failure signature |
| 4 | Hyperparameters and where every value came from |
| 5 | Batch composition |
| 6 | Diagnostics |
| 7 | How to run it |
| 8 | **Run history — what has actually happened on a GPU** |
| 9 | **The open failure: the ruler decays** |
| 10 | Failure catalogue — bugs already paid for, ours and the old project's |
| 11 | Deviations from CLAUDE.md, TMD and CRM, each with a reason |
| 12 | Open risks |
| 13 | Repo map |

---

## 0. How to read a number in this repo

Every quantitative claim in `CLAUDE.md` and here is tagged as one of three things, and the
distinction has already cost this project two wrong decisions. **Do not collapse them.**

| tag | means | trust it for |
|---|---|---|
| **measured** | read off the real parquet, the real tokenizer, or a real GPU run | levels and ratios |
| **simulated** | computed on free latents or Gaussian fixtures — no backbone involved | **orderings and ratios only, never levels** |
| **derived** | arithmetic or an identity | exactly what it says, nothing more |

Three headline numbers you will meet are **simulated on free latents**: ⑥ `L_good`'s
`relu`/`softplus` ablation, the `λ_good` sweep, and the per-term gradient norms. They were all
later checked against the model and **two of them were wrong in a way that mattered**:

- The sweep predicted ⑥ would cost `L_I` a factor of 2.7 (0.098 → 0.260). **Measured on the
  model it costs ~4%** (0.263 → 0.273). The guard built on the simulated figure fires on the
  baseline run itself — see §6's warning box.
- The sweep's headline metric was **saturated**: `frac_above_natural` read 0.000 at every
  `λ_good` including 0, because free latents close the tail unaided. It could say nothing about
  the 0.34 → 0.05 target it was run to inform.

**The general lesson, and it has now happened four times (§10.3): a threshold set by intuition
and then read as if measured.** If a number is not tagged, it is not a number yet.

---

## 1. What the model is

A process reward model that scores each **step** of a chain-of-thought by a learned
**quasimetric distance** `d(state, goal)` — how far this partial solution is from being solved.
Trained on Math-Shepherd, evaluated on ProcessBench. Adapted from **TMD** (Temporal Metric
Distillation, Myers et al., arXiv:2509.20478, `../tmd-release/`), which is goal-conditioned RL
for robots and games.

### The MDP mapping (CLAUDE.md §3)

One Math-Shepherd row is one trajectory for one question:

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

### One head, one score: the distance

There is no value head (locked #3a). ψ and φ are the two halves of `d`, `goal_head` supplies the
destination at eval, and nothing emits an independent scalar. Every phase-1 loss acts on `d`, so
the quantity that is trained is the quantity that is scored — the old project's root cause C was
exactly the opposite: it trained a scalar head that nothing at eval read, and its distance came
out at chance.

**A grep test asserts no module imports or constructs a `value_head`.** If one appears, someone
reintroduced CLAUDE.md §7.9.

### Why a quasimetric and not a similarity

Two properties, and **both have measured caveats that must not be glossed**:

- **triangle inequality** → credit assignment and stitching across partial solutions. But §4.4
  of CLAUDE.md measured that **~95% of states in this dataset have exactly one observed action**,
  so the stitching story has *no natural data support here*. `L_CF` (§3④) is the mechanism that
  would manufacture it, and its data is deferred. This is the load-bearing gap in the design.
- **asymmetry** (`d(s,g) ≠ d(g,s)`) → detecting irreversible dead-end states. Computed as a
  diagnostic from day one (`probe09_4/*`), never reported without a decision. But `full_mrn` at
  latent 512 measured **~73% symmetric**, so the asymmetric term is a minority and any claim
  about it needs the logged split behind it.

### The four things this design is reacting to

The old project (`../OLD_PROJECT.md`) trained a token-level quasimetric PRM whose score came out
at chance. Every major choice here is aimed at one of its four root causes:

| old root cause | fixed here by |
|---|---|
| **A** — the objective had no correctness signal (goals came from a trajectory's *own* future, so an incorrect solution's own incorrect ending was a legitimate positive) | ⑤ `L_step` on the distance, supervised by Math-Shepherd's per-step labels; `L_NCE` negatives explicitly include incorrect solutions |
| **B** — goal collapse (`γ=0.995` → 77% of goals clamped to the terminal → half the softmax columns byte-identical with contradictory targets) | `discount = 0.5`, measured to give **4.19 distinct goals** per solution against TMD's 1.16; step granularity; diagnostic #1 watches the realised count every batch |
| **C** — the eval score was never trained (no loss touched `d(ψ(s), ψ(g))`) | ⑤ `L_step` acts on `Δ_{z+1}`, **the exact statistic eval thresholds**, at weight 1.0, not as an option |
| **D** — `ψ(g_mean)` was a degenerate goal (a latent centroid over 30k terminals collapses onto the population mean; distance became an atypicality detector) | question-conditioned goal head; **no centroid anywhere in the codebase** |

**Files:** `model/distances.py` (the metric), `model/heads.py` (ψ, φ, goal_head),
`model/wrapper.py` (one forward → ψ, φ, act_emb).

### Backbone and heads

`Qwen/Qwen2.5-Math-1.5B-Instruct`, LoRA r16/α16/dropout 0 on the 7 Qwen2 attention + MLP
projections, **never on a head**. `attn_implementation: sdpa` (flash-attn is not installable on
sm_120). Gradient checkpointing on, `use_reentrant=False`. Token classification over hidden
states — **no vocab-sized logits anywhere**, which on Qwen's ~151k vocab is a large memory
saving and is one of the four things that buys the negative pool.

ψ and φ are MLPs `(512,512,512) → 512`, LayerNorm on the hidden layers only (the latent output is
**unnormalised** — TMD's `networks.py:35-60`; do not LayerNorm it). `latent_dim = 512`, TMD's
value, **not** 1536: the old project used 1536 = hidden size and measured that it maximises the
*symmetric* share of the distance to ~80%, undercutting the asymmetry thesis. 512 measures ~73%.

**Distances are computed in fp32** even when heads are bf16, cast inside the distance function.
In bf16 the small logit *differences* round to equal, the softmax goes exactly uniform, and the
gradient is zero — old bug B10a (§10.1).

`act_emb_i` is the **mean of the LM input embeddings of step `i`'s tokens**, tied to the input
embedding table. It needs no extra LM forward. It must **not** be the hidden after the step —
that *is* `s_i`, and φ would collapse into ψ∘next, making `L_T` trivially satisfiable. Known
weakness: a mean of input embeddings is bag-of-words, and this matters *more* now that `L_I` is
diagonal (§3②). `action_pool: attention` is implemented and off.

**Three LoRA traps, all of which fail loudly if undone.** Never adapt a head. Un-freeze the heads
*after* PEFT wraps the model (PEFT freezes every non-LoRA parameter by default, so head losses
would silently train through frozen random weights). And the launch-time guard asserts the
trainable set is **exactly** `{LoRA, psi, phi}` in phase 1 and **exactly** `{goal_head}` in
phase 2 — keep both asserts forever.

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

**Never write `ψ_{z-1}`.** It is a *good* state, and it does not exist at all for the 45.4% of
incorrect trajectories with `z = 0`.

**Why this is the section that matters.** A model trained on `Δ_z` instead of `Δ_{z+1}`
converges, separates `Δ`, and predicts `z−1` on **every** errored sample: `acc_error → 0` and F1
collapses through the harmonic mean while every loss curve still looks healthy. The draft of the
formula was written this way once (2026-07-25) and caught by the human before any code existed.
The round trip is asserted in
`tests/test_indexing.py::test_training_target_round_trips_to_the_right_prediction` and the
gradient-level version in
`tests/test_step_loss.py::test_reads_d_z_plus_1_minus_d_z_and_never_touches_d_1`.

**`z = 0` is 45.4% of incorrect trajectories, not an edge case.** Errors in this dataset are
early: mean first-error index 1.44, median 1, mean relative position 0.247. Any code path that
assumes a good prefix exists is wrong on nearly half the data.

**Separator positions come out by arithmetic, never by scanning.** `data/tokenize.py` tokenises
each segment separately and records positions by cumulative count, so the hazard that **13.9% of
solutions contain a step with an internal newline** cannot silently produce wrong state positions
(it would on one solution in seven). CRM's post-hoc `assert ids[pos] == sep_id` is kept as a
cheap guard. `SEP` is asserted to tokenise to exactly one id at load time. There is exactly
**one** sequence builder in the repo and a grep test enforces it — root cause I in the old
project was two prompt templates diverging by whitespace.

**Trajectory correctness is `all(labels)`, computed, not inferred.** CRM relies on
`all(labels) == labels[-1]`, which holds for **99.999%** of rows — 4 of 422,407 disagree, all of
the form `[T…T, F, T…T]`. We do not rely on it. (That entry read "100.0%, holds exactly" for a
while because a self-check written as a boolean `all()` over 422k rows printed `0.0` and looked
like the dataset had changed. Report it as a fraction.)

**Batch index maps** (`data/collate.py`, built on CPU from pre-tokenised rows only):

```
input_ids (B,L)   attention_mask (B,L) or None when nothing is padded
state_flat_idx (S,)  state_traj (S,)  state_step (S,)  traj_state_offset (B,)
row_src (R,)  row_dst (R,)  row_traj (R,)  row_step (R,)
traj_qid (B,)  traj_correct (B,)  traj_T (B,)  traj_z (B,)  traj_recovery (B,)  traj_terminal (B,)
span_token_idx (P,)  span_row_idx (P,)  span_counts (R,)      # segment-mean action pooling
```

`data/goals.py` then adds `goal_state (C,)`, `pos_row (C,)`, `goal_traj (C,)`, `is_terminal (C,)`.

Every loss is a pure function of `(psi, phi, act_emb, these tensors)`. **That is why the whole
CPU suite runs on random hiddens with no model and no GPU** — 206 tests in ~8 seconds.

---

## 3. The losses

Total (phase 1), `losses/total.py`:

```
L = λ_NCE·L_NCE + λ_I·L_I + ζ·L_T + λ_CF·L_CF + λ_step·L_step + λ_good·L_good
```

**⑥ `L_good` arrived 2026-07-28** (§3⑥) and ships **on at `λ_good = 1.0`** (signed off
2026-07-28), ramped over `good_loss.warmup_steps: 100`. It is the only term that bounds a *good*
step's `Δ` from above; the identity `Δ = −(−log γ) − A − B` with `−A ≥ 0` unbounded is why
nothing else can. `--set losses.lambda_good=0.0` leaves it computed and logged.

**ζ weights the backup only** (`tmd.py:124`); action invariance sits at 1.0. `tmd.py:362`'s own
comment says otherwise and is wrong — the code wins. **The old project put `L_I` under ζ=0.1,
making it 10× weaker than TMD's own setting, and its residual never got below 0.43.** Do not
repeat that.

**Every loss weight is unvalidated.** `λ_NCE`, `λ_I`, `ζ`, `λ_step`, `λ_good` were never designed
to be additive and their gradient magnitudes are only partly characterised. Log every curve
separately; do not tune silently. The old project shipped `λ = 1.0` "unvalidated" and never
revisited it.

### The three matrices, built once each (`losses/matrix.py`)

| tensor | shape | grad | consumers |
|---|---|---|---|
| `Dist[r,c] = d(φ_r, ψ(g_c))` | R×C ≈ 348×172 | yes | ① and ③ — **the same tensor object**, asserted by identity |
| `Next[r,c] = d(ψ(s_r), ψ(g_c))` | R×C | no | ③ only. Unconditionally detached (`tmd.py:113`), so `no_grad` also halves its activation cost |
| `D_term[s,t] = d(ψ_s, ψ(s_T^t))` | S×T_c ≈ 404×28 | yes | ⑤, ⑥, diagnostics #2/#3/#14, the goal-head gate |
| `D_term_good[s,t]` | S×T_c | yes | ⑥ only. `D_term` **itself** unless `good_loss.detach_goal` |

`Dist` is **rectangular and has no diagonal** — rows from incorrect trajectories are
negative-only, with no goal of their own and no positive column. TMD's is square because every
transition in its replay batch samples its own goal. Every "diagonal" in `tmd.py` becomes a
`pos_row` gather here; **a grep test forbids `torch.diagonal`.** `D_term` is our addition and it
is what makes the design cheap: `L_step`, `L_good` and the three-way Δ histogram read the same
small matrix, and it is the *eval-shaped* query.

`matrix.step_deltas()` is **the one definition of `Δ_i` in the codebase** — extracted from
`probes.py` on 2026-07-28 so ⑥ can train the exact statistic the panel reports. The probe calls
it and `.detach()`s; `losses/good.py` calls it and keeps the graph. A second hand-rolled
`D_term[row_dst] - D_term[row_src]` anywhere is a second definition and will drift. A test
asserts the extraction reproduces the old literal expressions bitwise and that its four masks
partition `valid` exactly.

### ① `L_NCE` — `losses/nce.py` (CLAUDE.md §7.2, `tmd.py:91-98`)

```python
loss = F.cross_entropy(-Dist.t() / tau, pos_row)     # softmax over SOURCE ROWS per goal column
```

TMD's **backward** NCE. The spec's own annotation fixes the direction: negatives "keep the goal,
change `(s_i, a_i)`", drawn from the same trajectory at other states and from other
trajectories, "correct or incorrect soln" — **the negative pool explicitly includes incorrect
solutions, which is already a correctness signal** and nothing extra was added for it.

We report `categorical_accuracy` in the **backward** direction. TMD's own logging reports the
forward one (`tmd.py:127`) while its loss normalises over sources (`tmd.py:97`). We report what
is optimised, and say so.

**`τ = 1.0` is a documented divergence** from TMD's `1/√512 = 22.6`: at 22.6 our O(1–10)
distances become O(0.05–0.5) logits, i.e. the near-uniform softmax that is bug B10a's signature.
It is a float knob, not a boolean — raise it if `logit_std` blows up.

*Fails as:* pinned at `log(R)` with `logit_std ≈ 0` → bug B10a, the fp32 cast is not effective at
runtime. Pinned at `log(R)` with `logit_std > 0` and pos ≈ neg → geometry collapse, check goal
duplication (probe #1).

### ② `L_I` — `losses/invariance.py` (CLAUDE.md §7.3, `tmd.py:100-105`)

```
L_I = mean_r d( ψ(s_{row_src[r]}) , φ_r )        elementwise, diagonal
```

Argument order matters. `d(ψ(s), φ(s,a)) = 0` does **not** imply `φ = ψ(s)` in a quasimetric; it
means "φ(s,a) is reachable from s at zero cost". In MRN, `d(x,y)=0` iff `x_sym = y_sym` and
`x_asym ≤ y_asym` elementwise, so φ can sit at zero cost *from* `ψ_{i-1}` while being strictly
farther from `g`. **That slack is exactly the room `L_step` needs**, and it is why `L_I` at
weight 1.0 does not trivially flatten Δ. It is also — see ⑥ — the mechanism by which good-step
`Δ` runs away upward.

**Diagonal, not the spec's grid, and this was a locked change (#16).** The spec writes
`d(ψ(s_i), φ(s_i, a_j))`, every state against every action in the batch. In OGBench that is
harmless: actions are points in a shared continuous space. **Here an action is question-specific
text**, so at `Q≈13` roughly 12/13 of grid entries assert "step 4 of question 7 is a zero-cost
action out of a state of question 3" — ~92% of `L_I` being a claim we do not believe, running at
weight 1.0, the same weight as `L_NCE`.

The two `grid_*` modes exist only to reproduce that failure. They drive `φ(s,a) → ψ(s)`, which
pins `Δ` at `−log γ` good step or bad and makes `L_step` immovable — so the config **refuses**
`grid_*` while `lambda_step > 0` rather than letting you debug a silent no-op. `grid_max_actions`
is hardcoded to 64 there.

*Fails as:* residual plateauing (the old project's 0.43) → the action representation is not being
used, or the grid is on. **Read the level against the same config's `λ_good = 0` baseline, not
against a fixed number** — see §6's warning box.

### ③ `L_T` — `losses/temporal.py` (CLAUDE.md §7.4, `tmd.py:107-122`)

**This is the ruler, and as of 2026-07-29 it is the thing that is broken (§9).** `L_NCE` only
teaches ranking; `L_T` sets the scale: one good step shrinks the distance by exactly
`−log γ = 0.693`. Distances then read in units of *steps remaining*, which is what makes one
global τ mean the same thing on every question — and `L_step`'s margin is expressed in those same
units.

```python
t     = log(clip_t_gain / gamma)                     # 3.689 at discount 0.5; gamma*exp(t)=20
delta = Dist - Next                                  # Next already detached
mask  = delta > t
div   = torch.where(mask, delta, gamma * torch.exp(torch.where(mask, t, delta)) - Dist)
L_T   = (1-dw) * (rho*div.mean() + (1-rho)*div[SQ].mean()) + dw * div[pos_row, arange(C)].mean()
```

Three things not to "simplify":

1. The **double `torch.where`** is exponent clipping and it is load-bearing: `where` evaluates the
   discarded branch in the backward pass, so an `inf` there poisons the gradient. Fixture
   `delta = 200` must give finite loss **and** finite gradients (tested).
2. The branches are discontinuous in value *and* slope at `delta == t`. Do not smooth them.
3. Branch B subtracts **`Dist`**, not `delta`.

**All pairs, not matched pairs:** ~60,000 backup terms per step, not ~348. It costs nothing
(`Dist` already exists) and it is the point — matched-only calibration lets the model learn a
different scale per question ("one step = 0.36 on algebra, 5.0 on geometry"), which satisfies the
loss perfectly and breaks a single global τ. Note what it does *not* ask: if `d(s, g') = 12.0`
for `g'` from another question, `L_T` asks for `11.895` after one step. Absolute distances are
`L_NCE`'s business; `L_T` only prices the step.

**`goal_scope_ratio` is ours, with no TMD counterpart, and it exists for the one place our domain
genuinely differs.** In OGBench a random goal is still *reachable* — it is a point in the same
maze. Here the ending of a triangle question is genuinely unreachable from an algebra state: the
state graph is **disconnected across questions**, and the backup keeps asking to shrink that
distance by `−log γ` per step forever. That may be harmless (it spreads unrelated questions
apart, which `L_NCE` wants anyway) or it may eat the loss. Default is TMD-faithful `ρ = 1.0`, and
diagnostic #13 logs `div_same_question` and `div_cross_question` separately **regardless of ρ**,
so the decision comes from a curve and not from an argument.

Minimiser: `Dist = Next − log γ`; value there is `1 − Dist`, **so L_T is expected to be
negative**. Watch plateau and NaN, not sign.

*Fails as:* `backup/linear_branch_fraction` is `≈ 1.0` at step 0 **by design** — ψ and φ are
independent networks, so `δ ≈ 9.8` before `L_I` closes the gap. It should fall to `≈ 0` within
~100 steps. Climbing *later* means the ruler is drifting: check `L_I` and #13 first, and do
**not** raise `clip_t_gain` — that uncaps `exp(δ)` rather than fixing `δ`, and is exactly the
+8760.29-at-init regression (§10.2). Never touch `discount`. `backup/div_cross_question`
diverging → the cross-question mass is eating the loss, lower `goal_scope_ratio`.

**And read §9 before you touch any of it: `backup/delta_mean` was measured decaying to 0.49 on
both full runs, which means the ruler is not being enforced at all.**

### ④ `L_CF` — `losses/counterfactual.py` (CLAUDE.md §7.5). Built, `λ = 0`, data deferred.

Cross-entropy over `{meaning-preserving rewrite} ∪ {meaning-changing rewrites}` with `f = −d`,
positive at index 0. Format and loader in `data/counterfactual.py`, loaded through a separate
interleaved dataloader so counterfactual batches need not align with main batches. Nothing in TMD
corresponds to this loss; it is spec-only, and the spec calls it *"our main contribution — long
form stitching."*

**The spec's cost claim is wrong and it matters in our favour.** `φ_i = phi(h_{i-1}, act_emb_i)`,
and rewriting step `i` leaves the prefix — hence `h_{i-1}` — untouched. Only `act_emb` changes,
and that is a mean of *input embeddings*, not a hidden state. A variant costs an embedding lookup
plus an MLP, **not an LM forward**. So when data exists, `L_CF` can run on every step of every
batch essentially free, and it is the most direct pressure available on the bag-of-words action
representation.

### ⑤ `L_step` — `losses/step.py` (CLAUDE.md §7.6, locked #3b). The correctness loss.

```
L_step = -log σ( d(ψ_{z+1}, g) - d(ψ_z, g) - m ),    m = margin_steps * (-log γ) = 1.386
```

averaged over (incorrect trajectory × correct-terminal goal of the same question) pairs. In
words: **crossing the first error must move you away from the goal.**

`g = ψ(s_T)` of a correct trajectory **in the batch** — a real terminal, never a prediction,
never a centroid, never a geometric-sampler column. That decouples it from `discount`: the term
carrying correctness queries an ending-like goal 100% of the time, which is what made
`discount = 0.5` affordable at all. Rejected alternatives, recorded so they are not re-proposed:
`goal_head(h_{s_0})` is circular in phase 1; a mid-trajectory goal column is the wrong direction
(states past the goal should be *far* from it); the trajectory's own terminal is degenerate
(`d(x,x) = 0`); `ψ(g_mean)` is root cause D.

Equivalently it is Bradley-Terry on `Δ_{z+1}` — **the exact statistic eval thresholds**. No other
phase-1 loss touches it, and it is **the only term that trains ψ as a source**:

| loss | ψ's role | gradient to ψ-as-source? |
|---|---|---|
| ① `L_NCE` | goal only — sources are φ | no |
| ③ `L_T`, `Next` | source, but stop-gradded | **no** |
| ③ `L_T`, `Dist` | goal only | no |
| ② `L_I` | source, but the target is `φ(s,a)` — a tying constraint with no correctness content | weakly |
| ⑤ `L_step` | **source, against a goal, forward direction, at the one index eval reads** | **yes** |

Meanwhile eval reads `d(ψ_i, g_q)`. **⑤ is not optional.**

**Why the boundary pair and not all (good, bad) pairs.** The obvious generalisation is broken,
and quietly: `d` measures how far you still have to go, so under the ruler an early good step is
legitimately far from the goal (`d_1 = 3.47 … d_6 = 0` on a 6-step solution at γ=0.5). Pair good
step 1 against bad step 6 and the term demands `d_6 > 3.47 + m` — the error must be worth more
than five steps of progress, and the demand grows with `(j − i)`. **That is fighting `L_T`, not
helping it**: position and correctness are confounded in `d`, and free pairing charges the
difference to correctness. `ψ_z` and `ψ_{z+1}` are adjacent, so the positional offset is exactly
one step and `m` absorbs it. Two sanctioned expansions exist (`position_corrected`, `same_index`)
and **both are off by decision** — see §11.

**`margin_steps = 2.0` is the human's stated intent**: an error you would have to undo costs more
than one step, so it must push you two full steps away. Note `margin_steps = 1.0` would put τ at
exactly 0, where the *sign* of `Δ` means progress vs regress — that symmetry was considered and
traded away for the larger separation. Note also that BT does **not** stop at `m`: `−log σ(Δ − m)`
still applies half its maximum gradient exactly at `Δ = m` and keeps pushing. `m` sets where
pressure halves, not where it ends, which is why diagnostic #13 must be watched.

**Init: `ln 5 = 1.6094` is a FIXTURE value, not the model's.** `L_step = log(1 + e^{m−Δ})`, so it
is exactly `ln 5` **where `Δ_{z+1} = 0`**. Measured on the real model 2026-07-27: `Δ_{z+1} ≈ −2.9`
and `L_step ≈ 4.26` on the GPU fixture, `Δ = −0.44` and `L_step = 2.08` on the first real training
batch. Why: `d(ψ_{z+1}, g) ≈ d(ψ_z, g)` needs ψ to map two different hidden states to nearly the
same place, and at init it does not do so *uniformly* — `h_{s_0}` is the single most atypical
hidden in the sequence, and `z = 0` for 45.4% of incorrect trajectories, so for those the pair
*is* `(ψ_0, ψ_1)`. **Check the loss against its own logged `step/delta_{min,mean,max}` via the
tolerance-free sandwich, never against a hardcoded level:**

```
softplus(m − step/delta_max)  ≤  L_step  ≤  softplus(m − step/delta_min)
```

What *is* exact is `m = margin_steps · (−log γ)` and the `z` indexing, both pinned on fixtures.

*Fails as:* `Δ_{z+1}` refusing to go positive → φ is ignoring its action input; this is a faster
diagnosis of the §16.3 collapse than probe #3. `step/distinct_z` near 21 → the sampler is on
`2c+1i`; exactly 96 pairs → the caps are being applied as quotas.

**The corrected derivation** (CLAUDE.md §7.6.5's sign is wrong in the spec):
`d(ψ_{i-1},g) ≤ d(ψ_{i-1},φ_i) + d(φ_i,g) → 0 + d(ψ_i,g) − log γ`, so **`Δ_i ≥ log γ = −0.693`**,
not `≥ +0.693`. The bound is one-sided, a large positive `Δ_{z+1}` does not violate it, and the
φ→ψ collapse pins Δ at `log γ`. **Both conclusions survive intact**; only the sign in the
write-up was off.

**That one-sided bound is the whole of ⑥ below.** `Δ_i ≥ −0.693` with no matching upper bound is
fine for the error step and fatal for the good ones.

### ⑥ `L_good` — `losses/good.py` (CLAUDE.md §7.12, added 2026-07-28). The false-positive loss.

```
L_good = mean over good transitions of  relu( Δ_i - c ),   c = -1 * (-log γ) = -0.693
```

`c` is **NEGATIVE** — the target `L_T` already prices a step at, used as a *ceiling*. It adds no
new target; it adds a ceiling to one that had only a floor. Scope is `matrix.step_deltas`:

| rows | `i` | in scope? |
|---|---|---|
| correct trajectories | every `i` | **yes** — the group that leaks F1 |
| incorrect trajectories | `i ≤ z` | **yes** (`include_incorrect_prefix`) — still on-track, and eval scores them identically |
| incorrect trajectories | `i = z+1` | **no** — ⑤'s own pair; the two terms would pull it in opposite directions |
| incorrect trajectories | `i > z+1` | **no** — already broken; no loss defines what `Δ` should be there |

Columns are correct terminals of the **same question** that are **not the row's own trajectory**
(its own runs to `d(x,x) = 0` and would spike the last `Δ`). Cross-question columns are excluded
— that mass belongs to `L_T`, and it is not the eval query.

**Why it exists. Exact identity, no assumptions:**

```
Δ_i = -(-log γ) - A_i - B_i      A_i = d(ψ_{i-1},g) - d(φ_i,g)              L_I slack
                                 B_i = d(φ_i,g) - d(ψ_i,g) - (-log γ)       L_T residual
```

At `L_I`'s optimum the triangle inequality gives `d(φ_i,g) ≥ d(ψ_{i-1},g)`, i.e. **`−A_i ≥ 0`
with no upper bound** — `L_I` *succeeding* pushes `Δ` the wrong way and nothing pushes back:

| term | what it says about a good step's `Δ` |
|---|---|
| `L_NCE` | ranking only, never a magnitude |
| `L_I` | drives `A → ≤ 0` — it *creates* the excess, it cannot cap it |
| `L_T` | prices `B`, and `B` is only one of the two terms |
| `L_step` | `Δ_{z+1} ≥ m`, a **floor on the error step**. Never sees a good step |

Measured at step 750 of run 1: good-step `Δ` mean `+0.240`, `P(Δ > 0.347) = 0.34`, fitted τ
`2.39`, F1 ceiling `0.456`. **Confirmed structural, not an optimisation failure** — a free-latent
probe driven to `L_I = 0.0024`, two orders below the run's residual, still shows the positive
tail. It is the geometry.

**`relu`, not `softplus`.** `Δ ≥ −0.693` is a hard floor implied by `L_I` + `L_T` together, so
pushing below `c` cannot be free — it has to be paid for by breaking one of them. `softplus`
applies half its gradient *at* the target and never stops, overshooting to `Δ = −1.556` and
stretching the ruler to 1.283. `relu` switches off at `c`. (SIMULATED ON FREE LATENTS — the
ordering transfers, the levels do not.)

**Warmup.** `relu` is the only term that does not taper near its target — full gradient while
violated, then nothing. `good_loss.warmup_steps: 100` ramps the *weight* linearly over the first
100 optimizer steps (~7% of the run), so ⑥ does not arrive at full strength while ② has not
closed the ψ/φ gap and ③ is still on the LINEX linear branch — i.e. **before the ruler that `c`
is expressed in exists**. `terms["good"]` and every `good/*` diagnostic stay unscaled;
`good/lambda_effective` logs the realised weight. `0` means full weight from step 1, **not off**.

**It does not drown ⑤, and the term-count framing was wrong.** Term count does not become
gradient: both terms are *means*, so 660 vs 67 makes each `L_good` term 10× *weaker*, not the sum
10× stronger. Per state touched, `L_step` is ~6× more concentrated (0.0179 vs 0.0029), and on the
one state they contest (`ψ_z`) they push the **same direction** with `L_step` 4.4× stronger.
`ψ_{z+1}` — the state that carries detection — gets exactly zero from `L_good` by the scope
exclusion. The sweep confirms it end to end: `probe03/gap` *rises* monotonically 3.97 → 4.86
across `λ ∈ {0, 0.1, 0.5, 1, 2}`.

**Init: no level is predicted, on purpose.** `expected_init_values` returns `nan`. The sandwich
runs *opposite* to ⑤'s because `relu` is increasing in `Δ`:
`relu(good/delta_min − c) ≤ L_good ≤ relu(good/delta_max − c)`, and **a lower bound of exactly 0
is legitimate** — it means every good step already sits at or below target. Do not "fix" it.
Predicting a level here and then adjusting code to hit it would be the same mistake for the
fourth time (§10.3).

**Also assert `good/margin < 0`.** The wrong sign (`c = +0.693`) trains good steps one full step
*away* from the goal per step **and converges cleanly** — no curve in the run would show it. The
negation lives in `Config.good_margin` and nowhere else, and `config.py` rejects
`margin_steps ≤ 0` so a pre-negated YAML value cannot double-negate. This is the highest-value
test in `test_good_loss.py`, for the same reason the `z` indexing is in `test_step_loss.py`.

*Fails as:* see §6's warning box for the `invariance/residual_diagonal` guard, which is
**miscalibrated** — do not act on the `≤ 0.15` figure. `backup/delta_mean` drifting off 0.693 →
the ruler is being stretched; confirm `form: relu` — **but note §9: it drifts on the λ_good=0
baseline too, so ⑥ is not the cause.** Diagnostic #5's `within` growing → `g` is being dragged
onto the trajectory; set `good_loss.detach_goal`. `probe03/gap` collapsing would mean ⑥ is
drowning ⑤ — measured small, so treat it as a surprise and re-read the scope masks before
touching a weight.

**Rejected, so they are not re-proposed:** raising `ζ` instead (helps a little — `P(Δ>τ)` 0.229 →
0.203 — and cannot be enough, because `ζ` reaches `B` and the unbounded half is `A`); moving the
readout or `L_step` to `φ` (`A` survives the move, detection *inverts*, and `φ_0` does not exist
for the 45.4% of errors at `z = 0`; `scripts/readout_bakeoff.py` was that investigation and is
deleted); `step_loss.pairing: same_index` (adds `L_step` terms; the hole is a missing constraint
on a *different* set of steps, not a shortage of `L_step` terms).

### `L_goal` (phase 2) — `losses/goal.py` (CLAUDE.md §7.7)

```
L_goal = mean over c of [ d(pred_q, ψ(s_T^c)) + d(ψ(s_T^c), pred_q) ]
```

The mean **of distances**, never a distance to a mean — a latent centroid over 30k terminals
collapses onto the population mean (root cause D). Both directions, because a one-way distance
would let the guess drift to somewhere *reachable from* the ending rather than *being* it.

**Trained in a separate phase, on frozen representations (locked #15), and that buys four
things.** `.detach()` becomes automatic, so the degenerate joint optimum (ψ squashes every ending
onto one point, the head predicts that point, loss → 0, every state identical) is unreachable —
with ψ frozen there is nothing to collapse. The head **cannot corrupt the metric**, because there
is no gradient path into the LoRA backbone. You can refit it or try variants without retraining
the 1.5B. And the go/no-go gate runs on the frozen model *before* the head is built. A collapse
test is kept in the suite anyway: it documents why the phase split exists.

**Phase 2 is cheap because everything is cacheable.** `h_{s_0}` depends only on the prompt under
causal attention, so it is **identical across all trajectories of a question** — cache one vector
per question, not per row.

---

## 4. Hyperparameters

`config/default.yaml`, strict-parsed by `config.py`: an unknown or misspelled key is a hard error
(old bug B4 was a silently-ignored config value falling back to a library default).

| key | value | where it came from | what it does | safe to tune? |
|---|---|---|---|---|
| `discount` | **0.5** | chosen 2026-07-25 from the measured goal-variety table | goal sampler **and** backup γ — one key, two uses | yes, but **only** to 0.7 (the other sanctioned row). Never 0.99: 1.16 distinct goals = root cause B |
| `heads.latent_dim` | 512 | `tmd get_config()` | ψ/φ output dim | 1536 measured ~80% symmetric (old R6) — don't |
| `heads.action_pool` | `mean` | §6.4 | bag-of-words action embedding | `attention` is implemented and unmeasured |
| `distance.variant` | `full_mrn` | locked #11 | metric family | `asym_only` if the asymmetry score is ever promoted |
| `data.prompt_format` | `raw` | CRM-faithful | raw prompt vs the Qwen chat template | `chat` builds and is untested end to end |
| `data.max_len` | 1024 | measured token distribution | rows over it are **dropped, never truncated** | see the note below |
| `losses.nce_temperature` | 1.0 | **divergence** from `tmd.py:92`'s 22.6 | logit scale | yes, raise if `logit_std` blows up |
| `losses.zeta` | 0.05 | `tmd get_config()` | weights **the backup only** | **the leading suspect for §9.** TMD-faithful, and its weighted gradient is the smallest in the loss set. Raising it is a change for the *next* run, not a mid-run edit |
| `losses.lambda_i` | 1.0 | `tmd.py:124` | invariance, **not** under ζ | unvalidated |
| `losses.lambda_step` | 1.0 | locked #3b | correctness | **not optional** |
| `losses.lambda_good` | **1.0** | signed off 2026-07-28 | the good-step ceiling; `relu(Δ − c)` | unvalidated at this weight. **The measured cost in `L_I` is ~4%, not the 2.7× the simulation predicted** — see §6's warning box before halving it. `0.0` leaves it computed and logged |
| `good_loss.margin_steps` | 1.0 | §7.12 | `c = −1·(−log γ) = −0.693`, **NEGATIVE** | it is the target `L_T` already names — changing it means disagreeing with the ruler |
| `good_loss.form` | `relu` | the ablation | where the pressure stops | **do not default to `softplus`** — it overshoots to `Δ = −1.556` and stretches the ruler |
| `good_loss.warmup_steps` | 100 | §7.12 | linear ramp on `λ_good` over the first 100 optimizer steps | it exists because `relu` does not taper. `0` = full weight from step 1, **not** "off". Scales the weight only |
| `good_loss.detach_goal` | false | §16.17 in reverse | whether ⑥ may move correct terminals | flip if diagnostic #5's `within` grows while the gate worsens |
| `step_loss.margin_steps` | 2.0 | the human's stated intent | `m = 2·(−log γ)` | drop to 1.0 if `backup/div_same_question` stops converging |
| `step_loss.exclude_recovery` | false | §16.15 | drop the 1.48% `False→True` trajectories from ⑤ | ⑤ is the one term where that label noise is not averaged away; flip it if `step/*` looks noisy |
| `backup.clip_t_gain` | 20.0 | reproduces TMD's bare `t=3.0` at **TMD's own** γ=0.99 | LINEX guard: caps `γ·exp(t)`, so max per-term gradient is `gain−1` | leave it. Probe #15 reads ~1.0 at init by design |
| `backup.diag_backup` | 0.5 | `tmd.py:364` | matched-vs-rest mix | TMD-faithful |
| `backup.goal_scope_ratio` | 1.0 | **ours**, no TMD counterpart | whole-batch vs same-question non-matched mass | decide from probe #13, not from an argument |
| `sampling.sequences_per_micro_batch` | 56 | §5 | one of two budgets | lower it **and raise grad_accum together** — but only after the token cap |
| `sampling.max_padded_tokens` | 32768 | measured 2026-07-27 | the other budget, `len(batch) × max_len` | **lower this FIRST if it OOMs.** Binds ~17% of batches; costs 5% of the NCE pool vs 52% for a sequence cut |
| `sampling.max_{correct,incorrect}_per_question` | 4 / 3 | the measured sweep | **caps, not quotas** | flat across 3–4 × 3–4; don't spend time here |
| `sampling.group_by_length` | true | §5 | padding ~60% → ~10% | `false` reproduces spec-literal order; every count holds either way |
| `train.grad_accum` | 2 | §7 arithmetic | — | **never raise alone** — that cuts the optimizer-step count and is the knob that hid the 106-step regression |
| `train.grad_clip` | 1.0 | ours; TMD clips nothing | guard, not an LR rescale | `train/grad_norm` is the **pre-clip** norm, logged every step. If it sits far above 1.0 all run, raise the clip rather than leaving it silently binding |
| `data.n_questions` | **34650** | raised 2026-07-28 | ~150k sequences/epoch, ~1,460 steps | cut this, not `grad_accum`, if GPU time is short. **Re-run `prepare_data.py` after any change** — the selection SHA moves with it. 91% of the ~38,247-question pool, so upward the only lever left is `train.epochs` |
| `eval.max_len` | 2048 | olympiadbench/omnimath run 8–9 steps on long problems | over-length samples predict `−1` and are counted | >1% of a subset is a hard failure |
| `log.wandb` | true | — | JSONL + console remain the source of truth | a missing `wandb` package only warns on stderr and trains on regardless |

**Everything below follows from `discount` and must never be set independently:**

| follows from `discount` | at 0.7 (fallback) | **at 0.5 (chosen)** |
|---|---|---|
| per-good-step cost `−log γ` | 0.3567 | **0.6931** |
| `L_step` margin `m` | 0.7133 | **1.3863** |
| `L_good` ceiling `c` | −0.3567 | **−0.6931** |
| `L_step` at `Δ = 0` (fixture, **not** the model's init) | 1.1120 | **1.6094** |
| natural eval τ | 0.1783 | **0.3466** |
| backup clip `t = log(20 / γ)` | 3.352 | **3.689** |
| goals landing on an ending | 55.0% | 41.1% |
| distinct goals / 6-step solution | 3.45 | **4.19** |

`Config.neg_log_gamma`, `Config.step_margin`, `Config.good_margin`, `Config.clip_t` are
properties, so there is no way to set them out of sync.
`tests/test_goals.py::test_discount_reaches_BOTH_consumers` asserts one key moves both consumers
— the test that would have caught the old two-key split. **A test that asserts `t` grows with
`−log γ` is asserting the +8760.29 regression back in** (§10.2).

**Why one `discount` and not two.** An earlier draft split it into `goal_discount = 0.5` and
`step_cost_gamma = 0.9`. Dropped, for a reason worth keeping: with one γ the geometric goal
distribution **is** the discounted state-occupancy the backup implies, so `d` converges to `−log`
of a real quantity. Split, it converges to `−log` of nothing the sampler produces — the two
halves of the objective describe different MDPs, and no theoretical result about temporal
distances transfers without being re-derived. It also bought nothing: the step cost works at
every value in the table.

**Why `discount = 0.5` and not 0.7.** Pick on goal variety alone. 0.5 gives 4.19 distinct goals
per solution against 0.7's 3.45, and root cause B was *goal collapse*. The objection to 0.5 —
that only 41% of sampled goals are endings while eval queries an ending 100% of the time —
assumes every goal comes from the sampler. It no longer does: ⑤ `L_step` takes `g` from a real
terminal in the batch, so the term carrying correctness is at 100% regardless, and the mismatch
costs `L_NCE`/`L_T` goal columns only. 0.7 is a one-line revert if the goal-head gate looks weak.

**On `max_len = 1024`.** Measured with the real tokenizer: median 248, p90 515, p95 631, p99 866.
**0.489% of sequences exceed 1024 and are dropped — 1,121 of 229,352.** The pre-measurement
estimate said 0.05%, so the real rate is **10× the documented one**; the estimate ran ~20% low at
every percentile. Keeping 1024 deliberately: it is a small, one-sided loss (it removes the
longest solutions, which skew hard), and `pad_to: batch_max` means `max_len` costs nothing when
it is not hit — it only truncates. Raising it to 1280 would recover ~all of the 1,121 at no
memory cost in a length-grouped batch, but it **invalidates the measured memory probe**, so it is
a change to make with a re-probe, not silently. (The config comment still says `~0.05%` and is
stale.)

---

## 5. Batch composition

`data/sampler.py`. **There are TWO budgets and a batch closes when it would break either** — a
count of sequences (56) and a count of padded tokens (32,768). Neither is a count of questions.

```
for each selected question, in shuffled order:
    take min(4, k_correct) correct  +  min(3, k_incorrect) incorrect
    if that allocation does not fit either budget, EMIT THE BATCH SHORT and carry the question over
```

**4/3 are caps, not quotas.** Only 20.0% / 29.5% of questions can fill them and **the median
question has 2 of each** — the mean of 3.62 / 5.60 is carried by a long tail out to 35 and 39, and
there is a cliff between k=2 and k=3 (68.6% of questions have ≥2 correct, only 25.0% have ≥3). A
hard quota discards most of the dataset. And a *partially* included question can end up with 0
correct or 0 incorrect, which silently produces goal-less rows and zero `L_step` pairs — hence
"no question is ever split".

**Correct trajectories are deliberately oversampled** to ~49–57% of the batch against the natural
36.6%, because they are the **only** rows that produce goal columns. Filling at the natural rate
roughly halves `C` and halves `L_T`'s term count.

Realised numbers (measured against the real distribution, not estimated):

| | value |
|---|---|
| questions per batch `Q` | ~12.9 (11.8 with the token cap binding) |
| source rows `R` | ~348 |
| goal columns `C` | ~172 |
| negatives per goal column | ~347 (`R − 1`, whatever `Q` is) |
| `L_T` terms | ~60,000 |
| `L_step` pairs | **64** |
| `L_step` distinct `z` | **~28** ← the number that matters |

**Read distinct `z`, not pairs.** The `k_c` goals all compare against the same `ψ_z`; one error
step measured against four terminals is not four independent signals. Distinct
`(ψ_z, ψ_{z+1})` gradients = the number of incorrect trajectories in the batch, because each has
exactly one first error. (CLAUDE.md §7.6's "96 pairs" line is stale — 64 is measured, and 96 means
the caps are being applied as quotas. Diagnostic #17 asserts against 64.)

**The chosen caps are a trade, not a free win.** Against the old `2c+1i` layout they buy **+37%
correctness signal** (20.8 → 28.4 distinct `z`) and cost **21% of the goal columns** (219 → 172).
Right direction because the correctness signal is the starved one — 28 examples against 60,000
`L_T` terms. "Take ALL trajectories" gives the most distinct `z` (34.0) but collapses `Q` to
**6.1** — 56 sequences from six questions, badly correlated. A sweep over `k_c ∈ [2,8] × k_i ∈
[1,4]` is **flat** in the 3–4 × 3–4 region (28–31 distinct `z`): moving off `2c+1i` was the part
that mattered, the exact caps are not worth tuning.

**Why the token cap exists, and why it is not interchangeable with the sequence budget.** A batch
costs `len(batch) × max_len` on the GPU. With length grouping on, the two decouple hard: the
median batch is 19,423 padded tokens and **the worst is 57,344** — 56 sequences all at `max_len`,
3× the median, which **OOMed a 15.46 GiB card**. Only 17% of batches exceed 32,768.

| | pool `R` mean | **`R` p10** | `Q` | peak tokens |
|---|---|---|---|---|
| 56 seqs, no cap | 364 | 177 | 12.6 | 57,344 → **OOM** |
| **56 seqs + 32,768 cap** | **344** | **176** | **11.8** | 32,760 |
| 28 seqs / accum 4 | 175 | 83 | 6.0 | 28,588 |

**Read the `R` p10 column.** The cap costs 5% of the mean pool and *nothing* at p10; the sequence
cut costs 52% of the mean and 53% at p10, and takes `Q` with it. **This is structural, not a
lucky draw:** thin-pool batches are the *short* ones (few tokens → few steps → small `R`), and
oversized batches are long sequences carrying the **largest** pools. The cap trims only where
there is pool to spare. Cost: +6% wall clock, `grad_accum` stays at 2.

**The peak-memory estimate is a lower bound.** It is a linear fit anchored on one observed OOM
that happened partway through backward, so true demand at 57,344 tokens is ≥16.7 GiB, not =16.7.
Treat anything within ~2 GiB of capacity as unproven — that is why 40,960 was not taken despite
keeping more pool. **Re-run `scripts/batch_report.py` before raising the cap.**

**A second memory lever is deliberately unspent.** `load_backbone` takes PEFT's default
`autocast_adapter_dtype=True`, so LoRA weights are fp32 under a bf16 base and each of the 7
adapted projections retains an fp32 copy of its input — ~1.1 GiB at a 32k batch. (The 1.91 GiB
allocation that actually failed was exactly `57,344 × 8,960 × 4 B`, `down_proj`'s input.) Turning
it off would make a 40,960 cap comfortable, but it is a numerics change on 22.4M trainable params
in a design that is deliberate about fp32. **Flip it only with an init-value check.**

**Length-grouped batching is on by default.** Shuffle the epoch's questions, sort by longest
allocated sequence inside megabatches of ~50 batches, form batches from consecutive runs, shuffle
batch order. Padding falls from ~60% to ~10%, batches with no padding pass `attention_mask=None`
so SDPA picks its fastest backend, and peak memory moves to the longest bucket — which is why
**the longest batch of the epoch runs first as a memory probe.**

**One epoch = one visit per selected question.** A visit samples `min(cap, available)`, so a
question with 10 correct solutions contributes 4 and the rest are never seen that epoch —
trajectory coverage is ~47% of the selected questions' rows. That is intended (breadth over
depth), but it means **locked #1's "take all trajectories" governs dataset *selection*, not what
the sampler consumes.** Do not reconcile the two by removing the caps.

---

## 6. Diagnostics

`diagnostics/probes.py` + the `info` dicts the losses return. Logged to
`runs/<name>/metrics.jsonl` and the console; wandb is wired and optional.

> ### ⚠️ Two guard thresholds in this repo are miscalibrated. Read this before acting on either.
>
> **1. `invariance/residual_diagonal ≤ 0.15 by step 200` is wrong and fires on the baseline.**
> That guard was calibrated against a **simulated** `λ_good = 0` level of 0.098. **Measured on
> the model 2026-07-29: the λ=0 baseline is 0.263 and the λ=1 run is 0.273** — so ⑥'s real cost
> in `L_I` is ~4%, not the 2.7× the simulation predicted, and the guard trips on a run with
> `L_good` switched off entirely. **Read the delta against the same config at `λ_good = 0`, never
> the absolute level.** (CLAUDE.md §17 records this; §7.12 and §16.21 still quote the simulated
> figure and have not been corrected there.)
>
> **2. The goal-head gate is `auc`, not `ratio`, and `ratio < 0.3` was never derived.** It was
> written as "~0.3", never simulated, then read as a measured threshold — and the first gate run
> returned `ratio = 0.599` and printed **"STOP AND REDESIGN"** when it should not have. At
> `D = 512` concentration of measure makes both distance distributions tight, so `ratio = 0.62` is
> already **100% same-question retrieval**. What the ratio *does* pin down is the null: iid
> terminals with no question structure give **1.000 to three decimals**. So 0.599 is a 40%
> contraction against a null of exactly 1.0 — signal, not failure. **And `scripts/goal_gate.py`'s
> own hardcoded `auc > 0.9` verdict is also not the gate**, because the *untrained* baseline is
> `auc = 0.904`. Always run `--untrained` on the same data and read the delta.

| # | key | read it as |
|---|---|---|
| 1 | `probe01/distinct_goal_ratio`, `negatives_per_column`, `questions_in_batch`, `padding_fraction` | distinct ≪ columns → root cause B is back |
| 2 | `probe02/delta_good_mean` vs `probe02/target_good_step_delta` (−0.693) | the old project's was **108× off** and nobody noticed |
| 3 | `probe03/gap` | **this gap IS the signal.** Collapsing → the error signal is being flattened. Guard: ≥ 1.8 |
| 4 | `probe04/symmetric_share` | if asymmetric is a minority (~73% symmetric at 512), do not claim asymmetry drives the result |
| 5 | `gate/{ratio, auc, recall_at_1}` (`scripts/goal_gate.py`) | **the goal-head go/no-go gate — see the warning box.** `auc` and `recall@1`, always against `--untrained` |
| 6 | `goal/pred_variance` | ≈ 0 → the head learned a constant, i.e. a global anchor |
| 7 | `probe07/within_trajectory_spread` | → 0 with masking off → states are being squashed; flip `nce_mask_same_traj` |
| 8 | `probe08/corr_distance_psi_norm` | `r > 0.9` → the goal contributes nothing; root cause D recreated |
| 9 | `invariance/residual_diagonal` | the old project's plateaued at 0.43; target 0; **the model's measured λ_good=0 level is 0.263** |
| 10 | `nce/{logit_std, logits_pos, logits_neg, categorical_accuracy_backward}` | the stuck-NCE table in §10.1. Guard: `categorical_accuracy_backward ≈ 0.25` |
| 11 | every `*/loss` separately | they were never designed to be additive |
| 12 | `probe12/off_over_on` | expect ~2 within incorrect trajectories — random state sampling over-represents post-error states 2:1 |
| 13 | `backup/{div_same_question, div_cross_question, delta_mean}` | the `ρ` decision, the `margin_steps` watch, **and the §9 readout** |
| 14 | `probe14/delta_{good_of_correct,good_of_incorrect,boundary,post_error}/{mean,std,p90,p99,frac_above_*}` | **the single best predictor of ProcessBench F1** |
| 15 | `backup/linear_branch_fraction` | the clip guard; `≈1.0` at step 0 by design, `≈0` within ~100 steps. **Rising later means the ruler is drifting — check `L_I` and #13 first, never raise `clip_t_gain`** |
| 16 | `probe16/goal_is_terminal_fraction` | should match 41% at `discount=0.5` — the train/eval mismatch |
| 17 | `step/{distinct_z, pairs, recovery_fraction}` | expect ~28 and 64; log the 1.48% `False→True` recovery fraction |
| 18 | `good/{loss, terms, delta_mean, delta_min, delta_max, margin, above_target_fraction, lambda_effective}` | ⑥, logged at **every** `λ_good` including 0, because these decide whether to raise it. **`good/margin` must print negative.** `good/terms` ~660 vs ⑤'s ~67 is **not** a drowning risk (both are means) |
| — | `probe09_4/irreversibility_*` | the asymmetry diagnostic, computed from day one, never reported without a decision |
| — | `train/grad_norm` | **pre-clip.** Far above `grad_clip = 1.0` all run → the clip became an LR rescale |

**#14 is the one to watch, and READ THE QUANTILES, NOT THE MEAN.** `L_step` never sees a correct
trajectory, so `L_T` alone suppresses false positives. At step 750 of run 1 the mean was `+0.240`
— which reads like a small bounded offset — while `frac_above_natural` was **0.34**, and *that* is
what drove τ to 2.39 and capped F1 at 0.456. The quantiles were added 2026-07-28; **the entire
first run has only the mean**, which is why the baseline row in §8's tail table is mean-only.

### Expected at initialisation (compute these, do not eyeball them)

| quantity | expected |
|---|---|
| `L_NCE` | `log(R) ≈ log(348) = 5.85` |
| `L_I` | **≈ 11** — the unrelated-latent distance. That is the geometry working, not a fault. Should fall below ~0.5 within ~100 steps; it is the term that closes the gap ③ waits on |
| `L_T` | **positive at ≈ δ ≈ 10**, on the LINEX *linear* branch, with `linear_branch_fraction ≈ 1.0`. Check all three together; any one alone is ambiguous |
| `L_step` | **≈ 4.3, measured** — `ln 5 = 1.6094` is the `Δ = 0` fixture value only; assert the sandwich against the logged deltas |
| `L_good` | **`nan` by decision.** Assert the sandwich instead, and `good/margin < 0` |
| `Q` | ~12.9 |
| `L_step` pairs / distinct `z` | 64 / ~28 |

`L_T`'s init value has been got wrong **twice** in opposite directions, and it is worth
understanding why once: at step 0, `Dist = d(φ, ψ_g) ≈ 11` (φ and ψ are unrelated maps: two
independent unit-variance 512-d latents at K=8 give a symmetric half of `√(32·2) ≈ 8.0` plus an
asymmetric half of `≈ 2.8`) while `Next = d(ψ_{s'}, ψ_g) ≈ 1.3` — **ψ against itself**, and LM
hidden states are strongly anisotropic (mean pairwise cosine ≈ 0.99 on Qwen), so ψ maps every
position to nearly the same point. `δ ≈ 9.8` is therefore a pure **ψ/φ representation offset** —
not noise, not a step count, and **independent of γ**. `TMD never sees this` because OGBench
proprioceptive observations are near-isotropic, its measured init `δ` is ≈ 0.09, and its clip
fires on ~0.1% of pairs. Only once `L_I` closes the gap (~100 steps) does the exponential branch
take over and `div ≈ γ − Dist` start to apply.

**There is no `L_BT`, no `L_CRM`, no `L_correct`.** If any appears in the logs, someone
reintroduced the value head or the trajectory-ending BT form. `good` **is** expected in the terms
dict from 2026-07-28; `λ_good = 0` makes it inert, not absent.

### Failure signatures

| symptom | probe | root cause | change |
|---|---|---|---|
| `L_NCE` pinned at `log(R)`, `logit_std ≈ 0` | #10 | bug B10a — fp32 cast not effective | confirm the cast is inside the distance; clean restart |
| `L_NCE` pinned, `logit_std > 0`, pos ≈ neg | #10, #1 | geometry collapse / goal duplication | check `distinct_goal_ratio` |
| `probe03/gap` → 0 | #3, #14 | φ ignoring its action | confirm `action_invariance: diagonal`; consider `action_pool: attention` |
| `Δ_{z+1}` will not go positive | #14, ⑤ | same as above, diagnosed faster | ditto |
| positive Δ tail on good steps of **correct** trajectories | #14 | the one-sided bound on `Δ` | ⑥ `L_good` — **not** a §7.10 pairing expansion |
| **`backup/delta_mean` falling away from 0.693** | #13 | **the ruler is not being enforced — §9, OPEN** | do not patch mid-run; read §9 |
| `L_T` diverging | #13, #15 | cross-question mass, or the ψ/φ gap not closing | lower `goal_scope_ratio`; check `L_I`. **Not** `clip_t_gain` |
| `backup/div_same_question` stops converging | #13 | `L_step` winning too hard | `margin_steps` → 1.0 |
| fitted τ ≈ 0 instead of 0.347 | §7 step 8 | the second step of margin never landed | `margin_steps` → 1.0, don't fight it |
| within-trajectory spread → 0 | #7 | the same-trajectory false negative biting | `nce_mask_same_traj: true` |
| `step/distinct_z` ≈ 21 | #17 | sampler still on `2c+1i` | check the caps |
| `step/pairs` exactly 96 | #17 | caps applied as **quotas** | check `_allocate` uses `min(cap, available)` |
| `optimizer_steps` prints ~106 | launch | the `n_questions`/`grad_accum` regression | cut `n_questions`, **never** raise `grad_accum` alone |
| launch log prints ~971 steps | launch | `prepare_data.py` was not re-run; the parquet holds the old 23,000-question selection | re-run it |
| terminal spread `within` grows while the gate worsens | #5 | ⑤'s or ⑥'s attached `g` leaking into terminals | detach `g` |

### The guards for a fresh run, in one place

Check these in this order. The first two are cheap and early; the rest need a few hundred steps.

| # | guard | why it is in this position |
|---|---|---|
| 1 | `good/margin` prints **negative** (`c = −0.693`) | the flipped sign converges cleanly and no curve would ever show it |
| 2 | `optimizer_steps` ≈ **1,460**, and the launch log's question count is **34,650** | ~106 is the step-count regression; ~971 means `prepare_data.py` was not re-run |
| 3 | `backup/delta_mean` → **0.693** | **currently FAILING (§9).** Everything below is expressed in the unit this guard establishes |
| 4 | `invariance/residual_diagonal` at or below the **same config's `λ_good = 0` level (~0.263)** | ⑥'s cost lands here. **Do not use the `≤ 0.15` figure** — see §6's warning box |
| 5 | `probe03/gap` ≥ **1.8** and `probe14/delta_boundary/mean` ≥ `m` = **1.386** | ⑤ is doing its job and ⑥ has not drowned it |
| 6 | `probe14/delta_good_of_correct/frac_above_natural` → **~0.05** | the target ⑥ exists to hit; **read this, not `good/delta_mean`** |
| 7 | `nce/categorical_accuracy_backward` ≈ **0.25** | ① is learning rather than pinned at chance |
| 8 | `gate/auc` and `gate/recall_at_1` **against `--untrained` on the same data** | the level alone is meaningless (baseline `auc = 0.904`) |

---

## 7. How to run it

```bash
conda create -n feynman python=3.12 -y && conda activate feynman
pip install torch --index-url https://download.pytorch.org/whl/cu128   # REQUIRED for sm_120
pip install -r requirements.txt && pip install -e .
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True                # every GPU shell
tmux new -s feynman                                                    # mandatory for any real run
```

Environment, in brief: **RTX 5070 Ti, 16 GB, Blackwell sm_120**, driver ≥570, Python 3.12,
**torch cu128 or newer** — cu128 is the first toolkit with sm_120 kernels, so it is a floor, not a
pin. The box measured at `torch 2.13.0+cu130` and the whole GPU suite passes on it. `flash-attn`
cannot be installed (no cp312/cu128/sm_120 wheel, source build fails without `nvcc`) — `sdpa` is
numerically equivalent and lower-memory at our lengths. **Verify the hidden size from the
downloaded `config.json` every time the model is re-downloaded**; a shape error in ψ/φ almost
always means this.

| # | command | what it should print |
|---|---|---|
| 0 | `pytest tests/test_gpu.py -m gpu -v -s` | **28 test functions** — environment preflight, both trainability asserts, LoRA-on-the-7-projections, two learning-rate groups, the bf16/fp32 seam, gradient checkpointing with `inputs_embeds`, right-padding invariance, `h_{s_0}` short-vs-full, the init-value panel, batch composition at the real budget, ⑥ reading the same `Δ` as the probe, `detach_goal`, three optimizer steps without NaN, the LR-schedule arithmetic, the memory probe (**prints the measurement**), save/resume, `merge_and_unload`, the phase-2 freeze, and the eval scoring path. **26 of these were green on 2026-07-27; the two ⑥ tests were added 2026-07-28 and have not been run on a GPU.** |
| 0b | `pytest tests/test_good_loss_ablation.py -m ablation -v -s` | **OPT-IN, not part of any normal run.** The only 2 tests that take optimizer steps on the real backbone (loads it twice, 12 steps each, ~2 min). ⑥'s A/B. **The phase-1 run measures both better and for free** — run this to diagnose a run that went wrong, not before starting one |
| 1 | `pytest tests/ -m "not gpu and not ablation"` | **206 passed, 2 skipped** in ~8 s. No model, no GPU, no download |
| 2 | `python scripts/prepare_data.py` | 45,989 questions → 40,247 trainable → 36.6% correct; the **real** tokenised length distribution; the selection SHAs; branch-point mining. **Must be re-run whenever `n_questions` changes** |
| 3 | `bash scripts/smoke_test.sh` | every loss finite and backward on random hiddens; same seed → identical losses |
| 4 | `bash scripts/train.sh --max-steps 20` | step-count assert, trainability assert, longest-batch peak VRAM, padding ~10%, init values, no NaN |
| 5 | `bash scripts/train.sh` | ~1,460 optimizer steps, est. 3–6 h. **Watch `backup/delta_mean` first (§9)**, then #14's `frac_above_natural` |
| 6 | `python scripts/val_f1.py --checkpoint runs/phase1/final [--untrained]` | the **F1 ceiling** with a real terminal substituted for the goal head — a ceiling, never a result. Read `acc_error`, `acc_correct` and their harmonic mean separately |
| 7 | `python scripts/goal_gate.py --checkpoint runs/phase1/final` **and again with `--untrained`** | `auc` / `recall@1` / `ratio`. **Read the delta against `--untrained`, not the level** (§6 warning box). The gate belongs *after* a real run — at 20 steps the terminals have not moved and it re-measures the baseline |
| 8 | `python -m feynman_prm.train_goal_head --checkpoint runs/phase1/final` | cache, gate, then minutes of fitting. **Never run yet** |
| 9 | `bash scripts/eval_processbench.sh runs/phase1/phase2/final` | per-subset F1, the math 587-leaked vs 413-clean split, τ and its sensitivity, the labelled skyline. **Never run yet** |

Other tools, all in `scripts/`: `batch_report.py` (the padded-token/pool trade — re-run before
raising the cap), `plot_metrics.py` (ASCII sparklines + `--csv`, dependency-free because
matplotlib is not in `requirements.txt`), `export_merged.py` (merge LoRA into a standalone model
directory, heads copied alongside), `diagnose_checkpoint.py`, `one_step_check.py`,
`gate_shape.py`, `diag_gc.py`, `report_processbench.py`.

**If it OOMs:** lower `sampling.max_padded_tokens` first (32768 → 24576). It bounds
`len(batch) × max_len`, which is what peak memory actually tracks, and it binds only on the ~17%
of batches that are oversized — leaving `L_NCE`'s negative pool intact on the rest. Only if that
is not enough, lower `sampling.sequences_per_micro_batch` *and* raise `grad_accum` to keep the
effective batch — but **never raise `grad_accum` alone.** `scripts/batch_report.py` prints the
exact trade for both.

**If a `CUBLAS_STATUS_NOT_SUPPORTED` appears inside SDPA `o_proj` on the first forward**, cast the
model to fp32 after load and before `.cuda()`. Do **not** try
`torch.backends.cuda.preferred_blas_library("cublaslt")` — it gets past `o_proj` and then throws
`invalid resource handle` at a layernorm.

**The step count is asserted at launch and fails below 300.** `optimizer_steps = 34,650 × 4.33 /
(56 × 2) ≈ 1,340` analytic; **expect ~1,460 measured**, because real batches close on whichever
budget binds first and the token cap makes them average ~51 sequences rather than 56 (ratio 1.092,
measured at the old scale as 1,943 batches / 971 steps against an 889 analytic). The arithmetic is
the sanity check, not the source — `train.py` derives the real number from the packed batches.

---

## 8. Run history — what has actually happened on a GPU

**Two full phase-1 runs exist. No phase-2 run and no ProcessBench evaluation exist.** Every F1
number below is `scripts/val_f1.py`: held-out Math-Shepherd val questions, ProcessBench *rule*,
with **a real terminal of another correct trajectory substituted for the goal head**. That is the
skyline substitution, and it is deliberately a **ceiling, not a result** — it is handed an ending
that eval will not have. Its job is to split one failure into two: bad here → the geometry never
learned correctness and phase 2 is wasted GPU time; good here and bad in phase 2 → the goal head
is the bottleneck, which is fixable.

### Run 1 — `fto0cx8e`, completed 2026-07-27

Loss set ①②③⑤ (`λ_good = 0`, the term did not exist yet), `n_questions = 23,000`, ~970 optimizer
steps.

| | |
|---|---|
| **F1 ceiling** | **0.456** |
| fitted τ | **2.39** against a natural 0.347 |
| good-step `Δ` mean (window 800–975) | **+0.392** |
| `P(Δ > 0.347)` at step 750 | **0.34** |
| `invariance/residual_diagonal` | 0.263 |
| `backup/delta_mean` | 0.863 → **0.522** (see §9) |

**Diagnosis, and read it as the diagnostics working.** The failure was visible in the right probe
from step 1. τ has to climb to 2.39 to dodge a third of all good steps, and TPR dies with it.
**The mean hid it**: `+0.240` at step 750 reads like a small bounded offset, and the tail is what
sets τ. That is why #14 now logs `frac_above_*`, `p90` and `p99` and not just the mean — **and why
this run has no tail series to compare against.**

**Two operational notes.** All measurements were taken on the **`step750`** checkpoint, not the
final one, because bug B11 (§10.2) raised an assertion *above* the final-checkpoint save and
destroyed it; `save_every: 250` is the only reason anything survived. And the good-step tail was
confirmed **structural rather than an optimisation failure** by a free-latent probe driven to
`L_I = 0.0024` — two orders below this run's residual — which still shows the positive tail.

### Run 2 — `jjkad2ae`, mid-run as of 2026-07-29 (~step 1030 of ~1460)

Loss set ①②③⑤⑥ (`λ_good = 1.0`, warmed up over 100 steps), `n_questions = 34,650`, ~1,460 steps.
**Fresh, never resumed from `step750`** — a loss-set change invalidates the checkpoint.

**The bulk moved exactly as the identity said it would. The tail did not — it got worse.**

| window | `delta_good_of_correct` mean | `frac_above_natural` | `p99` | `std` |
|---|---|---|---|---|
| run 1, 800–975 | **+0.392** | *(not logged)* | *(not logged)* | — |
| 100–300 | −0.258 | **0.070** | 0.86 | 0.33 |
| 300–600 | −0.269 | 0.164 | 2.25 | 0.80 |
| 600–800 | −0.302 | 0.176 | 2.94 | 1.02 |
| 800–1000 | **−0.412** | **0.159** | **2.43** | **1.05** |

Good-step `Δ` swung from `+0.392` to `−0.412` — the sign flip ⑥ exists to produce. But
`frac_above_natural` **bottomed at 0.070 around step 200 and then REGRESSED to ~0.16**, where it
has sat flat for 700 steps against a target of **0.05**. `p99` went 0.86 → 2.43,
`good/delta_max` reached **7.58**, and the spread **tripled**.

**So ⑥ is pulling the centre down while the upper tail runs away from it, and `relu` cannot stop
that** — it is a hinge on the *mean* of `relu(Δ − c)`, so a shrinking bulk buys down the loss
while a fattening tail contributes linearly and diffusely. **Do not read `good/loss` or
`good/delta_mean` for this**; #14's warning about the mean applies to ⑥'s own diagnostics exactly
as it applied to the mean that hid the `+0.240`.

**⑥ cost almost nothing in `L_I`:** 0.273 against run 1's 0.263, ~4%, against a simulated
prediction of 2.7×. See §6's warning box.

**The prime suspect is not ⑥ — it is the ruler (§9).** That decay is present in run 1 too.

**Still unknown:** `val_f1.py`'s fitted τ and F1 for this run, which is the only number that
converts any of the above into F1. A 0.8 swing in the mean may still buy a large improvement over
2.39 / 0.456 even with this tail. **Run `val_f1.py` when `jjkad2ae` finishes and record it here.**

### The attribution confound, stated rather than argued away

Run 2 changed **two things at once**: `λ_good` 0 → 1 and `n_questions` 23,000 → 34,650. Deliberate
— the loss-set change already forced a fresh run so the scale increase was free, and ⑥'s target is
a *tail* statistic (0.34 → 0.05 is a 20× smaller quantity) and tails need samples. **The cost is
that a regression cannot be attributed to one of the two.** The separating experiment is one
command, because the term stays computed and logged at any weight:

```bash
bash scripts/train.sh --set losses.lambda_good=0.0     # new n_questions, inert L_good
```

---

## 9. The open failure: the ruler decays

**MEASURED ON THE MODEL 2026-07-29, ON BOTH RUNS. This is the thing to fix next, and it is not
⑥.**

`L_T` exists to set the scale: one good step shrinks the distance by exactly `−log γ = 0.693`.
That is what makes distances read in units of *steps remaining*, and it is what makes a single
global `τ` mean the same thing on every question. `backup/delta_mean` is the readout of whether
that is being enforced.

**It never reaches 0.693 and it never plateaus short of it. It falls — identically with and
without ⑥:**

| optimizer steps | `fto0cx8e` (λ_good=0) | `jjkad2ae` (λ_good=1) |
|---|---|---|
| 100–300 | 0.863 | 0.836 |
| 300–600 | 0.617 | 0.727 |
| 600–800 | 0.657 | 0.577 |
| 800–975 | **0.522** | **0.491** |

Target is **0.693**. `jjkad2ae` logged **−0.031** at step 920 — a step that *increases* the
distance to the goal on average. `backup/linear_branch_fraction` is ~0 throughout, so this is the
**exponential branch failing to hold its own minimiser**, not a clip artifact.

**Why this is the prime suspect for run 2's tail.** An unanchored step cost lets `Δ`'s spread
grow, and it predicts exactly the `std` 0.33 → 1.05 that was observed. **Fix the ruler before
concluding anything about ⑥'s form or weight.**

**What it means for eval.** A single global τ fitted on val is only meaningful if one step costs
the same everywhere. At 0.49 and falling, it does not.

**The confound does not apply.** The decay is present in the `λ_good = 0` baseline, so neither
`L_good` nor the `n_questions` rise can be the cause.

### `ζ = 0.05` is the leading suspect

Per-term gradient L2 norm on the concatenated `(ψ, φ)` at a partially-trained state
(**SIMULATED ON FREE LATENTS**, 200 steps, a §5-shaped fixture):

| term | raw ‖∇‖ | weight | weighted |
|---|---|---|---|
| ① `L_NCE` | 0.0449 | 1.0 | **0.0449** |
| ② `L_I` | 0.0233 | 1.0 | 0.0233 |
| ⑥ `L_good` | 0.0174 | 1.0 | 0.0174 |
| ⑤ `L_step` | 0.0102 | 1.0 | 0.0102 |
| ③ `L_T` | 0.0105 | **0.05** | **0.0005** |

**The ruler's weighted gradient is ~90× below `L_NCE`'s and the weakest in the loss set — and `ζ`
is what makes it so, not the loss's own scale.** The usual defence of `ζ = 0.05` is that it
normalises a LINEX whose *value* is O(10) at init. That defence is about values. On **gradients**
the raw backup norm (0.0105) is within 3% of `L_step`'s (0.0102): the two are the same order
before weighting, and `ζ` alone drops one of them by 20×.

Two benign readings were available when this was only simulated, and the measurement has killed
both:

1. *A vanishing Bellman residual is correct behaviour* — on the exponential branch
   `∂div/∂Dist = γ·exp(δ) − 1`, exactly **0** at the minimiser. A converged `L_T` *should*
   contribute nothing. **But `δ` is not converged; it is decaying past the target.**
2. *All-pairs cancellation* — ~60,000 terms pushing each row toward many goals in opposing
   directions puts the vector norm far below the scalar gradient mass. True, and a property of
   the design rather than a fault, but it does not explain a monotone decay.

**Two caveats that survive.** `ζ = 0.05` is TMD-faithful (`tmd.py`'s own `get_config`). And ⑥'s
work separately measured that raising `ζ` does **not** fix the good-step tail (`P(Δ > τ)` 0.229 →
0.203) — because `ζ` reaches `B` and the unbounded half of that tail is `A`. **That was always a
question about `A`; this one is about `B`.** Raising `ζ` is a change for the **next** run, not a
mid-run edit.

**What is no longer reasonable is to assume the ruler converges.** Confirming the mechanism wants
per-term gradient norms logged during a real run — **nothing does this today**, and it is the
cheapest instrumentation left to add.

---

## 10. Failure catalogue

### 10.1 Inherited — every one of these was paid for once in the old project

| # | symptom | cause | fix |
|---|---|---|---|
| **B4** | a config value silently ignored, library default used | no strict parsing | `config.py` hard-errors on an unknown or misspelled key |
| **B5** | `torch.cat` of two batches crashes almost every step | the collate padded two groups to *different* per-batch maxima | pad both to the common max first |
| **B6** | constant LR, silently | no scheduler was ever built | build the cosine schedule and assert it steps — **read the LR over the whole run, not start-vs-end (see B11)** |
| **B8** | `CUDA driver version is insufficient` at optimizer construction | `FusedAdam` JIT-compiles against the *system* CUDA toolkit | `torch.optim.AdamW` (identical at `weight_decay=0`). **Hardcoded, not config** |
| **B9** | OOM at the very last step | AdamW's `foreach` path allocates one large fp32 transient | `foreach=False`. **Hardcoded, not config** |
| **B10a** | `nce_loss` frozen at exactly `ln(N)`, accuracy `1/N`, `logit_std ≈ 0` | bf16 rounds the small logit *differences* to equal → exactly uniform softmax → zero gradient | compute distances in **fp32 inside the distance forward**. Gradients still flow to bf16 params |
| **B10b** | same symptom; invariance falling while NCE flat | `1/√D` temperature made NCE ~4× weaker than the invariance terms | `nce_temperature: 1.0`, fresh restart. **But `1/√D` is TMD's own setting** and only read as a bug because the old `L_I` was a batch-wide grid at weight 1.0. `L_I` is diagonal now, so the imbalance is much smaller. **The symptom to act on is `logit_std ≈ 0`, not the config value** |

**Diagnosing a stuck NCE:**

| `logit_std` | `nce_loss` | pos vs neg | verdict |
|---|---|---|---|
| > 0 | dropping below `ln(N)` | pos drifting below neg | learning — let it run |
| > 0 | stuck at `ln(N)` | pos ≈ neg | **geometry collapse** — check goal duplication |
| ≈ 0 | stuck at `ln(N)` | — | fp32 patch not effective at runtime — confirm B10a, clean restart |

**Not a bug:** `L_T` is **expected to be negative**. Watch for plateau and NaN, not sign. (The old
"settles around −1 to −1.5" was recorded at `γ = 0.9`; the settling value scales with `−log γ`, so
derive it from diagnostic #2 rather than carrying it over.)

### 10.2 Ours — paid for on this project

| # | symptom | cause | fix |
|---|---|---|---|
| **the 106-step run** | a phase-1 run that would have completed with **106 optimizer steps and 3 warmup steps**, cosine-decaying to zero, training ψ and φ from random init plus LoRA | **two independent errors multiplied.** `n_questions` was set assuming 9.18 sequences per question when the caps give **4.33**; and `grad_accum` was raised to 8 to compensate for a `Q` drop to "~8–10" when `Q` is **12.9** measured. It would have produced a checkpoint | derive `optimizer_steps` and **assert it at launch, failing below 300**. The note that let it through was *"LR/schedule defaults are sane starting points, not requirements"* — the LR **values** were reviewed and **the number of steps they act over was never derived at all.** If GPU time is short cut `n_questions`, never raise `grad_accum` alone |
| **+8760.29 at step 0** | the backup reported `+8760.29` against an expected `−10.53` — 25× the whole rest of the loss set | `clip_t_steps = 28.5` → `t = 19.75` at γ=0.5, and `exp(19.75) = 3.8e8`. **`t` bounds an exponent, not a step count.** The step-count argument was true in every clause and wrong in its conclusion; the 28.5 was also anchored at γ=0.9, a discount **TMD never runs** | `t = log(clip_t_gain / γ)` so `γ·exp(t) = 20` at every discount, reproducing TMD's bare `t=3.0` at TMD's own γ=0.99. A test asserts `t` **does not** grow with `−log γ`. **`margin_steps` is genuinely scale-free and stays that way** — the two knobs measure different things, and that is exactly the distinction that was missed |
| **B11 — the guard that destroyed its own artifact** | `AssertionError: the LR never moved -- bug B6 is back`, raised at the **end of a healthy run**, and **no final checkpoint**. Cost a 971-step run | **the B6 guard itself.** It compared `param_groups[0]["lr"]` against a value read straight after `build_scheduler` returned. **Both are 0.0**: `LambdaLR.__init__` applies `lr_lambda(0)`, which under warmup is `0/warmup = 0`, and a completed cosine ends at `0.5·(1+cos π) = 0` exactly. So it fired on every run that **finished** and passed only on runs cut short by `--max-steps` — the ones whose checkpoints do not matter. `save_checkpoint(".../final")` sat **below** the raise | track `lr_min_seen`/`lr_max_seen` across every optimizer step and fire on `max <= min`. **And write the final checkpoint BEFORE the raise** — a diagnostic must never destroy the artifact it is diagnosing. Pinned in `tests/test_schedule.py`, including a grep for the start-vs-end comparison and one asserting the save stays above the raise. Note `test_lr_schedule_arithmetic` already asserted `seen[-1] < 0.01`: **the suite knew the schedule ends at zero and the guard was never checked against it** |
| **an environment preflight rejecting a newer build** | `test_environment_*` failed on a *working* cu130 wheel | it asserted `torch.version.cuda.startswith("12.8")` | compare `(major, minor) >= (12, 8)`. A cu-version check must be `>=`, never `startswith` |
| **two GPU tests failing with the code correct** | two forwards that are mathematically identical but not bitwise identical — right-padded batch vs unpadded, and full-sequence vs prompt-only `h_{s_0}` — asserted `max|Δh| < 0.05` and measured **0.75 and 0.50** | **Qwen2 hidden states are not O(1).** A handful of channels carry *massive activations* of O(100) and dominate a max-abs difference; the bf16 grid spacing at \|x\| = 128 is already **1.0**, so 0.75 is **one rounding** and 0.05 is unreachable at any level of correctness. The two forwards also take different kernels (SDPA's `is_causal` fast path vs an explicit 4D mask), so rounding diverges over 28 layers | assert the **relative L2 per state vector**, `‖Δh‖/‖h‖ < 0.05`, and print the bf16 ulp count alongside so the absolute number is explainable. It still fails loudly on the bugs the tests exist to catch — a state read at the wrong flat index, positions shifting with padding, attention that is not causal — because those leave the vectors *unrelated* (relative L2 of O(1)), not slightly rounded (O(1e-3)). **Do not normalise per element**: bf16 noise in a residual stream scales with the vector, so a channel near zero would read as a huge relative error |

### 10.3 The mistake this project keeps making

**Four instances, one shape: a number produced by intuition and then read as if it were
measured.**

| # | the number | what it assumed | what it cost |
|---|---|---|---|
| 1 | `clip_t_steps = 28.5` | that `t` is a step count, and that TMD's γ was 0.9 | `+8760.29` at init and a wasted GPU run |
| 2 | `L_step = ln 5` at init | that two independently initialised random maps agree, so `Δ_{z+1} = 0` | a probe reporting a failure against a healthy loss (real value ~4.26) |
| 3 | the gate's `ratio < 0.3` | that a mean-distance ratio at `D = 512` behaves like one at low `D` | **"STOP AND REDESIGN"** printed against a perfectly usable 0.599 |
| 4 | `invariance/residual_diagonal ≤ 0.15` | a **simulated** λ=0 level of 0.098 | a guard that fires on the baseline run (real level 0.263) |

**Two working rules fall out of this.**

- **A ratio is a comparison, and a checkpoint alone gives you one side of it.** That is why
  `goal_gate.py --untrained` exists, and it is why it must be run on the same data every time. It
  is also why `--untrained` on the *first* gate run revealed that **all of the gate's apparent
  signal is free**: solutions to one question share that question's entire text, so their
  terminals are similar before any training — `auc = 0.904`, `recall@1 = 0.618`, `ratio = 0.582`
  on a base model with random-init ψ. **Read the delta, never the level.** Note this is signal
  phase 1 never asks for and `L_NCE` actively works against: other correct trajectories of the
  same question are negatives for each other's goal columns.
- **Do not predict a level for a new term.** `expected_init_values` returns `nan` for ⑥ on
  purpose, and the assertion is a tolerance-free **sandwich** against the term's own logged
  extrema. Predicting a level and then adjusting the code to hit it is instance #5 waiting to
  happen.

### 10.4 A note on tests that train

A test that trains is almost always a **worse instrument than the run itself**. The phase-1 run
logs `probe03/gap` and `probe14/delta_good_of_correct/frac_above_natural` every `log_every` steps,
over real batches, for ~1,460 steps; a 12-step A/B on a toy fixture is a **wiring check wearing
the clothes of an experiment**. That is why `tests/test_good_loss_ablation.py` is behind the
`ablation` marker and deselected from every normal run. Keep such tests, mark them, and reach for
them to **diagnose a run that went wrong** — never as a precondition for starting one.

---

## 11. Deviations, each with a reason

### From `CLAUDE.md`

1. **No `losses/extras.py`.** The two §7.10 expansions are `step_loss.pairing` values, so they
   live in `step.py` with the boundary form. §11's own config comment already says this.
2. **`data/tokenize.py` added** and it is the only place a sequence is built. Root cause I was two
   prompt templates diverging by whitespace. A grep test enforces the singleton.
3. **Checkpointing does not merge adapters.** `save_checkpoint()` writes `adapter/`, `heads.pt`,
   the resolved config and the tokenizer, and **asserts the head state dict is non-empty** —
   which is the failure §14's merge rule was protecting against (stock PEFT save writes the
   adapter only and silently drops the trained heads). `scripts/export_merged.py` does
   `merge_and_unload()` when a standalone directory is wanted. Checkpoints stay ~100 MB instead of
   3.1 GB, so mid-run saves are affordable — **and that is the only reason run 1's measurements
   exist** (B11 destroyed its final checkpoint; `step750` survived).
4. **Two config keys dropped.** `sampling.hard_negatives_post_error`: under §8.1's layout every
   state of every selected trajectory is already a source row, so there is no negative pool to
   bias and the flag has no implementable meaning — diagnostic #12 logs the realised
   on-track/off-track ratio instead. `action_invariance.grid_max_actions`: only reachable from the
   two reproduce-the-failure modes, hardcoded to 64 there.
5. **Keys §11 lists that are not config**, because they must never vary: fp32 distances,
   pad-to-batch-max, the separator-after-prompt, `strip_step_prefix`, `log_selection_sha`,
   `add_step_prefix` at eval, `report_math_leak_split`, `layer_norm`, `hidden_size` (always read
   from the downloaded `config.json`). `optimizer: adamw` and `adam_foreach: false` are hardcoded
   for the same reason (bugs B8, B9).
6. **`train.betas` / `weight_decay` / `bf16` / `schedule` are kept as config keys** — PLAN's
   abbreviated YAML omitted them but did not list them among its drops, so they carry over
   verbatim.
7. **Config keys added since §11 was written:** `data.prompt_format` (`raw` | `chat`),
   `step_loss.exclude_recovery`, `eval.max_len`, `eval.batch_sequences`, `goal_head.{lr, epochs,
   batch_size, max_terminals_per_question}`, `log.{wandb, wandb_project}`, `train.{log_every,
   save_every, max_steps}`, `run.{name, out_dir, seed}`, `data.dir`.
8. **`heads.ensemble` is a knob that refuses to be true.** The human's call was off; wiring 2
   members through every loss (TMD averages the critic loss over members and takes `min` at read
   time) is not built, and the config says so explicitly rather than silently ignoring the flag.
   Cost if it were built: ~3 M params, ~0.2% of a 1.5B backbone — and the `min` is a real
   conservatism mechanism against `d` being over-optimistic on unseen pairs, which is precisely
   our risk. **Still open.**
9. **`asym_only` uses the whole component dim**, not just the half `full_mrn` reserves for the
   max. Otherwise half the latent would be dead in that variant. `d(x,x)` is exactly 0 there —
   for `full_mrn` it is `≈1e-3`, because `eps = 1e-6` sits *inside* the sqrt at `tmd.py:38`, so
   each component floors at `sqrt(1e-6) = 1e-3`. Assert `< 2e-3`, not a loose `5e-2`.
10. **The config refuses `action_invariance: grid_*` while `lambda_step > 0`.** Set
    `lambda_step: 0.0` to deliberately reproduce the φ→ψ collapse.
11. **Over-length ProcessBench samples predict `−1` and are counted**; >1% of a subset is a hard
    failure (`assert_truncation_budget`). Truncating would drop trailing separators and shorten
    `T`.
12. **Parquet IO lives in `data/math_shepherd.py`** rather than a new module, to keep the §12 file
    list intact.
13. **`scripts/plot_metrics.py` is dependency-free** (ASCII sparklines + CSV) because matplotlib
    is not in `requirements.txt`.
14. **The memory probe runs a real forward+backward and then discards the gradients.** It costs
    one micro-batch (~6 s) and turns a three-hour OOM into a 30-second one.
15. **The two §7.10 pairing expansions are OFF by decision (2026-07-25), and this was raised
    explicitly and declined.** `same_index` was *measured* to be worth 4.50 examples per incorrect
    trajectory instead of 1 (~128 per batch instead of ~28) at **no extra forward passes**, because
    the correct trajectories are already in the batch as goal providers. It is off because it is
    the only term in the design that would compare states **across two different solutions**,
    assuming step `i` of one is "as far along" as step `i` of another — an approximation with no
    measurement behind it. And note: `position_corrected` would require moving `margin_steps`
    2.0 → 3.0, or the boundary pair silently loses one step of margin.

### From TMD, all deliberate

| | |
|---|---|
| `τ_NCE = 1.0` | vs `1/√512 = 22.6` (`tmd.py:92`) — bug B10a's signature |
| `discount = 0.5` | vs 0.99 — 0.99 gives 1.16 distinct goals on 6-step solutions = root cause B |
| `clip_t_gain` | vs a bare `t = 3.0`: same value at TMD's γ, but the *form* keeps `γ·exp(t)` fixed at any `discount`. The earlier `clip_t_steps` form is a **known regression**, not an alternative |
| `train.grad_clip = 1.0` | TMD clips nothing (`tmd.py:341`); it has no backbone under its heads and we do, and `L_T` is an `exp()`, so one bad micro-batch is enough |
| `goal_scope_ratio` | ours, no TMD counterpart: their random goals are *reachable*, ours are not |
| rectangular `Dist`, `pos_row` | TMD's is square because every transition samples its own goal |
| no `actor_loss`, no stochastic dynamics, no policy extraction | we build a verifier, not a policy |
| `binary_accuracy`, `value_exp`, `contrastive_only` not ported | `binary_accuracy` is pinned at `1 − 1/B` because distances are non-negative — vestigial |
| `categorical_accuracy` measured in the **backward** direction | TMD's own logs the forward one (`tmd.py:127`) while its loss normalises over sources (`tmd.py:97`). We report what is optimised, and say so |
| no ψ/φ ensemble | TMD uses 2 + `min`. Off per the human's call; see deviation 8 |

### From CRM

Its stack cannot be copied: `torch==2.6.0` has no cu128 build, so it does not support sm_120 at
all. We take its **API era** (transformers ≥4.56 style, `AutoModel`, `load_dataset`, plain
`accelerate`) and none of its versions — no trl, verl, deepspeed, vllm, flash-attn. `dtype=`, not
the deprecated `torch_dtype=`. We add a separator after the prompt (CRM has no `s_0`), we drop
over-length rows rather than truncating (CRM's own reasoning applied to our failure mode), and we
compute `all(labels)` rather than relying on `labels[-1]`. **If you ever need to run CRM itself,
use a separate conda env — its environment conflicts with ours. Do not merge them.**

---

## 12. Open risks

1. **The ruler decays and it is unexplained. See §9.** This is the top item, it is measured on two
   runs, and it undermines the unit that `τ`, `m` and `c` are all expressed in.
2. **`L_CF`'s data is deferred, and that is the load-bearing gap in the whole design.** ~95% of
   states have exactly one observed action, so the stitching/triangle story has no data support
   until real rewrites exist. **The mined branch points are NOT a substitute** — they are two
   meaning-*changing* continuations with a correctness label; `L_CF` needs a
   meaning-*preserving* positive. They are written to `data/processed/branch_points.jsonl` as a
   held-out diagnostic set. Counted on full train: **102,988 branching nodes, 30,344
   correctness-disagreeing** (29.5%). Note the earlier extrapolation from a 4,000-question sample
   said ~63,000 disagreeing — **halved, not gone**, and a standing reminder to count on the full
   split before assuming the data changed.
3. **The action representation is bag-of-words** and this matters *more* under `diagonal`
   invariance, because the grid used to mask a weak action representation. `action_pool: attention`
   is implemented and off.
4. **Every loss weight is unvalidated.** Log every curve separately.
5. **The train/eval goal-type mismatch widened at `discount = 0.5`**: eval queries an ending 100%
   of the time, the sampler 41%. `L_step`'s goal is a real terminal so the correctness term is at
   100% regardless; the cost falls on `L_NCE`/`L_T` columns. Probe #16 logs it, and 0.7 is a
   one-line revert if the gate looks weak.
6. **⑤'s `g` is not detached, and neither is ⑥'s.** With `g` attached, the cheapest way to satisfy
   ⑤ may be to move `g` away from `ψ_z` rather than `ψ_z` away from `g` — i.e. the gradient leaks
   into exactly the terminal representations phase 2 must predict. ⑥ has the same problem in
   reverse (it pulls `d(ψ_i, g)` *down*, so the cheap route is dragging terminals onto the
   trajectory). Watch probe #5: if `within` grows while the gate ratio worsens, detach.
7. **The 1.48% of trajectories with a `False → True` label recovery** reach ⑤ directly, so that
   label noise lands on the correctness term rather than being averaged into a trajectory-level
   verdict. This is the one respect in which `L_step` is more exposed than the `L_correct` it
   replaced. `step_loss.exclude_recovery` exists; diagnostic #17 logs the fraction.
8. **ProcessBench-math is 58.7% problem-contaminated with our training data** (587 of 1000 exact
   prompt matches; 0 on the other three subsets). Locked decision: **keep them and report the
   math-subset F1 split** 587 leaked vs 413 clean. **Any PRM trained on Math-Shepherd has this
   leak**, CRM included. Separately, the dataset's own `test` split is unusable — 99.6% of its
   questions appear in `train` — hence our own question-level holdout of 2,000.
9. **The gold answer must never enter the scoring path.** Knowing only `final_answer_correct`
   solves the "no error" half of the F1 **exactly, 100%, on all four subsets**, with zero
   modelling. Published PRMs score in the 60s–70s. So any path consuming a reference solution or
   gold answer is a **skyline, labelled, never a reported result** — that includes `val_f1.py` and
   `skyline.py`. **This is a hard rule.**
10. **Expect distribution shift at eval.** Our training solutions come from Mistral-7B-class
    generators at ~5 steps; ProcessBench's hard subsets come from Qwen2.5-Math-72B-class generators
    at 8–9 steps with errors *later* in the solution. Report per-subset; do not average away the
    difference.
11. **7B is arithmetically out** (closing §16.5 without a formal decision): `Qwen2.5-Math-7B` in
    bf16 is ~15.2 GB of weights alone on a 16 GB card, before LoRA states, a 56×600-token
    activation stack and ~0.5 GB of fp32 distance matrices. 4-bit QLoRA would fit but fights the
    mandatory fp32 distance path. **1.5B-Instruct stands.**
12. **`qprm_baseline2/` does not exist in this workspace.** `OLD_PROJECT.md` §16 says to port its
    working PyTorch MRN, its `bregman_dt`/`_backup_divergence`, its 31 CPU tests and its per-token
    trace tooling. `goal-conditioned-rm/` is only the upstream Scale AI OpenRLHF fork (zero hits
    for `mrn`, `quasimetric`, `psi`, `phi`, `latent_dim`, `asym`). So every line here comes from
    TMD's JAX, and **the tests on the distance and the backup are load-bearing rather than
    confirmatory.** If a copy exists on the GPU box, say so — it would de-risk the two hardest
    functions, and it is directly relevant to §9.
13. **`OLD_PROJECT.md`'s own AUROC table is milder than "at chance".** §1.2 motivates this whole
    design with "`-d_MRN` came out at chance", and the trace evidence for that (terminal ranked
    70/112, adj R² = −0.001) is genuinely damning — but the AUROC block appended at the end of that
    file reads 0.56 / 0.67 / 0.64 / 0.73 / 0.67 on five of seven benchmarks. No decision here
    depends on which framing is right, **but do not describe the old distance as a flat failure in
    a write-up without checking that table.**
14. **The committed HuggingFace token in `goal-conditioned-rm/` is live-format and permanently in
    git history** (`examples/experiments/train/train_rm.sh:9`, first commit). Nothing from that
    repo is copied here, but it is a real `hf_` token in a checkout on your disk — **rotate or
    report it**, don't just avoid the file. The same tree hard-codes private S3 buckets, a private
    W&B host and personal EFS paths.
15. **Numbers not yet measured on a GPU.** The ~2 h/epoch wall clock and the ~10% padding fraction
    are estimates; the peak-VRAM figures are extrapolated from one OOM and are lower bounds. The
    memory probe in step 4 replaces all three with measurements — **run it after any change to
    `max_len`, `max_padded_tokens` or `autocast_adapter_dtype`.**

---

## 13. Repo map

```
feynman-prm/
  CLAUDE.md                  the locked design spec — authority on any disagreement
  IMPLEMENTATION.md          this file
  PLAN.md  SETUP.md          build plan; install and run on the GPU box
  config/default.yaml        strict-parsed; every TMD value carries its provenance
  feynman_prm/
    config.py                strict parse; neg_log_gamma / step_margin / good_margin / clip_t
                             are PROPERTIES so they cannot go out of sync with `discount`
    data/
      math_shepherd.py       load, drop 15 empty-label rows, question-level select + split,
                             parquet IO. The val split is drawn BEFORE n_questions is applied
      tokenize.py            THE ONLY sequence builder. SEP positions by cumulative count
      collate.py             the batch index maps (§2)
      goals.py               geometric goal sampling -> goal_state / pos_row / is_terminal
      sampler.py             batch composition, the two budgets, length grouping (§5)
      branch_points.py       mine the correctness-disagreeing branches
      counterfactual.py      L_CF data format + interleaved loader
    model/
      backbone.py            Qwen + LoRA, launch-time trainability asserts
      heads.py               psi / phi / goal_head + action pooling.  NO value_head
      distances.py           MRN variants, fp32
      wrapper.py             one forward -> psi, phi, act_emb
    losses/
      matrix.py              the shared Dist matrix, pos_row, SQ mask, D_term
                             + step_deltas(): THE definition of Delta_i
      nce.py invariance.py temporal.py counterfactual.py step.py good.py    (1)-(6)
      goal.py                L_goal + terminal_separability — PHASE 2 only
      total.py               the weighted sum, the warmup ramp, the terms dict
    train.py                 phase 1; derives and asserts the real step count
    train_goal_head.py       phase 2, on cached frozen vectors
    eval/
      processbench.py        scoring + the §2 index mapping
      calibrate.py           tau on held-out val questions; natural_tau
      skyline.py             reference-solution goal — labelled, never a result
      metrics.py             ProcessBench F1, reproduced exactly
    diagnostics/{probes,logging}.py
    utils/{seeding,checkpoint,indexing}.py
  scripts/                   prepare_data, train.sh, smoke_test.sh, eval_processbench.sh,
                             val_f1, goal_gate, batch_report, plot_metrics, export_merged,
                             diagnose_checkpoint, one_step_check, gate_shape, diag_gc,
                             report_processbench
  tests/                     229 test functions. 206 pass + 2 skip on CPU in ~8 s
```

**No `losses/extras.py`** (deviation 1) and **no `README.md`** (by instruction).

**Grep-enforced invariants.** `tests/test_grep_invariants.py`: no `value_head` anywhere; none of
the deleted loss names (`L_BT`, `L_CRM`, `L_correct`); no `torch.diagonal` on the shared matrix;
exactly one sequence builder; **no centroid of terminals** (root cause D); no flash-attn or CRM
stack dependency; `dtype=` not the deprecated `torch_dtype=`; `AutoModel`, not
`AutoModelForCausalLM`. And in `tests/test_schedule.py`: `train.py` does not compare the final LR
against the pre-loop one (bug B11), and **the final checkpoint is written before the B6 raise.**

---

*Last reconciled against `CLAUDE.md` on 2026-07-30, including its 2026-07-29 measurements
(§7.4.5, §16.23, §7.12's mid-run reading, and §17's `invariance/residual_diagonal` baseline). The
two things to carry into the next session: **run `val_f1.py` on `jjkad2ae`**, and **§9**.*
