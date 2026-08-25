# QRL + counterfactual invariance, matched to Feynman-PRM

**QRL (Quasimetric RL, Wang & Isola, ICLR 2023) — its constrained objective, plus a
counterfactual-invariance constraint expressed in the same form, trained under Feynman-PRM's
exact conditions.** `quasimetric-rl/` (the authors' released code) stays untouched as the
vendored reference, the same treatment `../tmd-release/`, `../CRM/` and `../Process_Q_Model/`
get; the pieces used here (`grad_mul`, `softplus_inv_float`, the global-push transform, the
Lagrangian local constraint) are ported with line citations, never imported.

**The row this produces is our adaptation under matched conditions, not a published QRL
number.** Say so wherever it appears. QRL is an offline/online goal-reaching RL method on
continuous-control benchmarks; it has never been run on chain-of-thought data and reports no
ProcessBench number.

---

## 0. Why this run exists

Feynman-PRM scores each reasoning step by a learned quasimetric `d(state, goal)`. The shipped
method builds its ruler out of fixed-weight TMD-style losses, and **its known open failure is
that the ruler decays** — `backup/delta_mean` drifts off `−log γ` over a run
(IMPLEMENTATION.md §9), and §16.8 already says plainly that the weights holding it were never
validated and that the terms were never designed to be additive.

QRL removes the choice. The objective is *maximize distances everywhere*; everything we know
must stay small becomes a **constraint** with its own Lagrange multiplier, trained by gradient
ascent on the same scalar the primal descends. The ruler is then held by a dual variable
instead of by a hand-chosen λ, and the multiplier is itself a readable diagnostic while it
holds: **λ climbing without its violation falling means the constraint cannot be satisfied.**

For the CF constraint that reading is worth stating on its own, because nothing else in the
project has it: **λ_cf is a data-quality detector.** If the counterfactual corpus contains
"meaning-preserving" rewrites that are not, the constraint is unsatisfiable, and `qrl/cf_sq_dev`
will sit above `epsilon_cf²` while `qrl/lagrange_cf` climbs to try to force it down.

## 1. What "matched" means — the table for the methods paragraph

Every row is identical to the baseline runs, and identical **by construction** (the same
config file, the same `sequences.parquet`, the same code paths called), not by re-derivation:

| | this run and the baselines |
|---|---|
| dataset | `trl-lib/math_shepherd`, `data/processed/sequences.parquet` |
| selection | `n_questions: 34650`, `n_val_questions: 2000`, seed 42, same `selection_sha_train` (34,640 questions have rows; 10 lose every trajectory at tokenisation, §4.6) |
| sequences/epoch | 149,351 (`min(4,k_c)+min(3,k_i)` caps → 4.312 seqs/question, §8.1) |
| epochs | 1 |
| batching | `sequences_per_micro_batch 56`, `max_padded_tokens 32768`, `group_by_length true`, `grad_accum 2` → **2,928 micro-batches, 1,464 optimizer steps, 44 warmup** |
| tokenisation | `build_sequence`, `"\n"` single-token separator, `s_0` after the prompt, `max_len 1024`, drop-never-truncate |
| backbone | `Qwen/Qwen2.5-Math-1.5B-Instruct`, LoRA r16 α16 dropout 0 on the 7 projections, grad checkpointing, bf16, sdpa |
| heads | ψ and φ, 3×512 MLPs, `latent_dim 512`, `action_pool` mean — **unchanged `FeynmanPRM`, unchanged forward** |
| goal sampler | `sample_goals`, geometric at `discount 0.5`, offset base `i−1` — **reused verbatim**, so the goal distribution matches every baseline |
| CF corpus + attach | the same three `cf_glob` files, the same `prefix_hash` join, `cf_max_per_batch 12`, the same seeded draw. **The corpus itself is a moving snapshot** — see §2 |
| optimizer (primal) | torch AdamW, `foreach=False`, betas (0.9, 0.95), wd 0, `grad_clip 1.0` |
| lr | LoRA 9e-6, heads 3e-4, cosine, `warmup_ratio 0.03` |
| phase 2 | `feynman_prm.train_goal_head`, unchanged, on frozen ψ |
| eval | `feynman_prm.eval.processbench`, unchanged — 4 subsets, `"Step N: "` prefixes, `eval.max_len 2048`, `localisation_rule: first_crossing`, math leak split reported |
| τ | fitted on the held-out 2,000 val questions, **never** on ProcessBench (§9.2) |

The matched-data proof is free and should be run on the probe before the real launch:

```bash
diff <(jq -c 'select(.event=="launch/data")' runs/qrl_iqe_probe/events.jsonl) \
     <(jq -c 'select(.event=="launch/data")' runs/abl_cf_only/events.jsonl)
```

Only `elapsed_s` may differ.

## 2. Deliberate divergences — state these in the paper

| | baselines | here | why |
|---|---|---|---|
| **objective** | ①L_NCE ②L_I ③L_T ④L_CF ⑤L_step ⑥L_good ⑦L_term, fixed weights | global push + 2 Lagrangian constraints | **this is the experiment.** Not one Feynman phase-1 term is computed |
| **head** | `distance.variant: full_mrn` | **`iqe`** | the user's call, 2026-08-25. IQE is TMD's other quasimetric head (`tmd.py:48-66`) and is asymmetric by construction |
| CF corpus size | `abl_cf_only` trained on 27,110 examples; `cf_lam2_tau005` on 36,065 | **41,380** (2026-08-25 snapshot) | the file *list* is unchanged and never needs to change — `data/cf_train/` is a snapshot that grows between runs (27,114 → 36,073 → 41,380). **This is the one condition that is NOT matched to any baseline**, and it cannot be: the corpus is bigger than either row saw. See the note below |
| QRL's pair sampler | — | the full `S × C` grid instead of QRL's rolled batch | see §3 |
| QRL's latent-dynamics term | — | **deleted.** `s' = s ++ a` is deterministic, so the arrived state is read — `psi[row_dst]` for an observed transition, `psi(prefix + variant)` for a rewrite — and there is nothing to predict or to police | see §4 |
| new parameters | — | **2 scalars** (the multipliers, outside the model) + **1 scalar** (IQE's learned `alpha`, inside it) | `launch/model` reports `lagrange_params`, `distance_params` and the exact `trainable_params` at launch |

> **The CF corpus is bigger than any baseline's, and that is not fixable by config.**
> `data/cf_train/` is a snapshot of a campaign that is still generating
> (`MANIFEST.md` tracks it: 27,114 on 2026-08-15 → 36,073 on 08-22 → **41,380** on 08-25,
> +5,307 new anchors, 13,012 distinct questions, all inside the 34,650-question train
> selection and none in the 2,000-question val holdout). Nothing needs changing to pick the
> new examples up — they went into the *existing* three files and `data.cf_glob` names those
> three by filename, not by wildcard.
>
> **`data.cf_max_per_batch: 12` now binds, and that is what bounds the effect.** The cap was
> sized on 27,114 examples over ~2,928 micro-batches (~9.3/batch); at 41,380 the raw figure is
> ~14.1/batch. Selection among the eligible is uniform without replacement, so the per-step
> cost, the number of CF pairs entering the constraint, and the constraint's magnitude are all
> **unchanged** — the extra examples buy coverage across the epoch, not more CF signal per
> step. Read a falling `cf/attach_rate` as the cap, not as a broken join;
> `cf/examples_attached` against `cf/examples_eligible` is the pair that says which.
>
> Practically this means the QRL row saw *more distinct* counterfactual anchors over its one
> epoch than `cf_lam2_tau005` did, at the same per-step CF budget. Say so; do not quietly
> attribute it to the objective. The control, if it matters, is to re-run a baseline against
> this same snapshot rather than to shrink this one.

> **The head change is the one to be careful about.** The QRL row differs from every baseline
> row in *both* the objective and the head, so a gap is a gap against the pair.
> `bash qrl_prm/train.sh --set distance.variant=full_mrn --set run.name=qrl_mrn` is the
> one-line control that separates them, and it should be run before the attribution is
> written down if the gap matters.

## 3. The push term, and the one thing that is not QRL's

`losses/global_push.py:44-48`:

```python
dists = qm(zx, torch.roll(zy, 1, dims=0))
F.softplus(self.softplus_offset - dists, beta=self.softplus_beta).mean()
```

Same transform, same `.mean()`. **Only the pair sampler is adapted.** QRL rolls its batch to
get `B` random pairs (~50 in its own configs); we take the full grid of this micro-batch's
states against the sampled goal columns — the same `E[softplus(offset − d(x, y))]` over the
batch marginals, with three orders of magnitude more pairs, and with the goal side drawn from
`sample_goals` so it matches both the distribution the metric is queried with at eval and the
one every baseline trained under. `qrl/push_pairs` reports the realised count each step.

Two diagnostics decide whether the term is doing anything:

* **`qrl/push_saturated_frac`** — the fraction of pairs already past `softplus_offset`, where
  the transform is flat and there is no gradient left. Rising towards 1.0 means the objective
  has stopped maximising anything and `softplus_offset` is too small. `train.py` refuses to
  launch if this is ≥ 0.99 at init.
* **`qrl/push_dist_mean_{same_traj, same_question, cross_question}`** — the push is over the
  whole grid on purpose (QRL's random pair distribution has no notion of "same question"), but
  a metric whose cross-question distances grow exactly as fast as its same-trajectory ones is
  inflating a scale, not learning structure. Only the split says which.

## 4. The CF constraint — the star topology, and what grounds φ to ψ

A CF example attaches at departure state `h_{i−1}`, and `φ(h_{i−1}, act(variant))` is the
predicted arrived state for that wording of the step. The anchor variant is the *original*
step text, whose **true** arrived state `ψ(s_i)` is already in the batch. So the constraint is
a star with `ψ(s_i)` as the hub, both directions on every spoke:

```
                    ψ(s_i)                    constrained pairs:
                   /  |  \                      (hub, anchor)  (anchor, hub)
             anchor  pos₁  pos₂                 (hub, pos_k)   (pos_k, hub)

    E[ (relu(d(hub, v))² + relu(d(v, hub))²) / 2 ]  ≤  epsilon_cf²
```

This does two jobs at once, which is why the decided loss set needs no separate
latent-dynamics term:

1. **the `(hub, anchor)` pair IS QRL's latent-dynamics loss**, evaluated exactly where CF data
   lives. Without it, φ is free to drift into a private coordinate system in which "all
   variants agree" costs nothing and means nothing;
2. **the class diameter is bounded by `2 · epsilon_cf`** by the triangle inequality, without
   ever forming the `O(|C|²)` pairwise grid.

**The hub is derived, never assumed.** `variant_state` is a flat state index; the arrived
state is `row_dst[r]` for the row `r` whose `row_src == variant_state`. `row_src` is unique
within a batch, so the scatter is a total function on states that have a successor. A variant
departing from a trajectory's *terminal* has none — those are dropped and counted in
`qrl/cf_hub_missing`. Flat-index adjacency (`variant_state + 1`) would be right inside a
trajectory and wrong at every boundary, which is the worst possible failure shape;
`tests/test_qrl.py::test_hub_is_the_arrived_state_of_the_variant_s_own_transition` pins it on
the exact case that discriminates (state 3 is trajectory 0's terminal, state 4 is trajectory
1's `s_0`).

**CF negatives are never in the constraint.** Two different *wrong* rewrites of a step have no
reason to be the same point, and asserting it is a claim this data cannot support — the same
rule `losses/counterfactual.py` states for its own negatives. They enter the **push** term as
sources instead: `d(φ(neg), ψ_g)` against same-question goal columns, which is the
eval-aligned direction (a broken state as the source of the query). `qrl/neg_push_gap` —
their mean distance minus the overall push mean — is the curve that says whether that is
working.

`epsilon_cf: 0.2` is not QRL's; it is chosen against **the ruler a verdict is read against**.
The old margin ruler is `2 log 2 ≈ 1.386` (§7.6), so a 0.2 ball keeps a paraphrase ~7× too
small to move a step across the decision boundary. `config.py` refuses any `epsilon_cf` at or
above 1.386 outright.

## 5. Layout

```
qrl_prm/
  config.py              QRLConfig: the 10 knobs, strict-parsed (unknown key = hard error)
  config/qrl.yaml        the QRL-only knobs, annotated with the reasoning behind each default
  lagrange.py            grad_mul / softplus_inv_float / LagrangeMultiplier(s), ported
  loss.py                push, the two constraints, the optional dynamics term, the §18 helper
  train.py               the training loop (mirrors feynman_prm/train.py's launch discipline)
  train.sh               the local wrapper for kratoss — tmux + PYTORCH_CUDA_ALLOC_CONF
  report.py              the side-by-side table against the three shipped rows
tests/test_qrl.py        40 CPU tests on the shared fixtures
```

There is no `scripts/*.py` entry point and no `feynman_prm/` module — entry is
`python -m qrl_prm.train`. The reason is `pqm_baseline/README.md §3`'s, one step further: the
objective here is a *replacement* for the method's loss set, not an option inside it, and
`config/default.yaml`'s `losses:` block is strict-parsed, so a `qrl:` block there would have
to be declared on every Feynman run that never reads it.

**Nothing in `feynman_prm/` was modified.** `save_checkpoint`'s default `HEAD_PREFIXES`
already carries `"distance."`, which is what checkpoints IQE's learned `alpha_raw`;
`assert_phase1_trainable` already allows a trainable `distance` parameter at
`variant == iqe`; and the multipliers are deliberately *not* model parameters, so the
trainability assert needs no exemption for them.

## 6. The two optimizers

QRL's dual variables are trained by gradient **ascent** on the same scalar the primal
descends, which one `loss.backward()` achieves via `grad_mul(λ, −1)`: identity forward,
negation backward. They get their **own** AdamW — `lr 1e-2`, betas `(0.9, 0.999)`, wd 0, **no
scheduler** (`losses/__init__.py:47`) — and are zeroed and stepped at the *same* grad-accum
boundary as the primal, so a dual step sees the gradient of the same two micro-batches.

They must not join the primal optimiser. `model/backbone.py::param_groups` sweeps every
trainable non-LoRA parameter into the "heads" group at `lr_heads` on the cosine schedule, so a
multiplier registered on the model would decay to a standstill exactly when the constraints
start binding, and would be stepped by an optimiser that never saw the sign flip. They live
outside `FeynmanPRM` for that reason, and ride the checkpoint payload (`extra`) instead of the
head state dict — raw, pre-softplus, so a reload is exact.

## 7. Diagnostics — what to watch, in order

| key | read it as |
|---|---|
| **`qrl/local_dist_mean`** | **THE RULER.** Should pin near `step_cost = 1.0`. This is the direct answer to IMPLEMENTATION.md §9's decaying `backup/delta_mean` — watch it against that history |
| `qrl/lagrange_local`, `qrl/lagrange_cf` | must **rise then stabilise**. Monotone climbing while the matching violation does not fall = the constraint cannot be satisfied. For λ_cf that means the CF corpus contradicts itself |
| `qrl/local_violation`, `qrl/cf_violation` | the **sign** is the readable quantity: negative = satisfied |
| `qrl/cf_p95` | the tail, not the mean. A class whose mean sits inside `epsilon_cf` can still hold pairs far outside it, and those are the paraphrases that flip a verdict (§7.12's lesson, in a different loss) |
| `qrl/push_saturated_frac` | rising to 1.0 = the push term has no gradient left |
| `qrl/neg_push_gap` | CF negatives' distance to goals minus the overall push mean. It should **open** |
| `qrl/cf_active`, `qrl/cf_hub_missing` | 0.0 / non-zero says "no data this batch" rather than "violation happened to be zero" — the empty path logs the full key set at 0.0 and never vanishes |
| **`probe14/*`** | the three-way Δ histogram, "the single best predictor of ProcessBench F1" (§7.6.6). Computed by `build_matrices` + `batch_probes` **verbatim**, under `no_grad`, on log steps only — so the QRL row's numbers mean exactly what `abl_cf_only`'s and `pqm_zeta4`'s do. No QRL term reads those tensors |

> **Before reading a climbing `qrl/lagrange_cf` as bad CF data, check `qrl/cf_positives` and
> `cf/examples_dropped_budget`.**
>
> This used to be a three-way read against `qrl/dyn_sq_dists`, because φ was barely supervised
> — it entered only through the CF star and the CF-negative push — so it was free to drift away
> from ψ everywhere else, and the CF constraint ended up grounding φ to ψ *and* collapsing the
> class at the same time. **Measured on the first probe: `qrl/dyn_sq_dists` went 14.2 → 95.1
> over 20 steps** while both multipliers rose.
>
> **That failure no longer exists, and neither does the curve.** Under deterministic dynamics
> `s' = s ++ a`, so a variant is `psi(prompt + steps[:i] + variant + SEP)` — a real encoded
> state, in ψ-space by construction (`qrl_prm/cf_encode.py`). There is no φ to be unanchored
> and no second space to drift into, so the CF constraint does one job only.
>
> λ_cf climbing now has exactly two readings, and both are answered by keys beside it:
>
> * `qrl/cf_sq_dev` not falling with `qrl/cf_positives` healthy → **the CF corpus contradicts
>   itself**: two "meaning-preserving" rewrites of one step really do land in different places.
>   The dual variable is the data-quality detector. Read `qrl/cf_p95`, not the mean — a class
>   whose mean sits inside `epsilon_cf` can still hold a few pairs far outside it, and those
>   are the paraphrases that flip a verdict.
> * `qrl/cf_positives` low, or `cf/examples_dropped_budget` climbing → **the constraint is
>   training on very little** and λ_cf is rising on a handful of pairs. Raise
>   `qrl.cf_encode_max_tokens` if the card allows it, before auditing anything.
>
> `qrl/cf_anchor_missing` should read exactly 0.0 for the whole run: `cf_encode.py` drops a CF
> example whole or not at all, so a nonzero value means the variant tensors were built
> somewhere other than that file.


## 8. Runbook

```bash
tmux new -s qrl

# 1. the probe -> runs/qrl_iqe_probe/ (disposable)
bash qrl_prm/train.sh --max-steps 20
#    read launch/data, launch/model, launch/cf_data, launch/init_values, launch/memory_probe
#    (train.sh's header says exactly what to look for in each)

# 2. the real run, ~1,464 optimizer steps
bash qrl_prm/train.sh

# 3. phase 2 and eval -- the UNCHANGED feynman entry points
python -m feynman_prm.train_goal_head --checkpoint runs/qrl_iqe/final
python -m feynman_prm.eval.processbench --checkpoint runs/qrl_iqe/phase2/final

# 4. the table
python -m qrl_prm.report --qrl runs/qrl_iqe/phase2/final --run-dir runs/qrl_iqe
```

The `natural_tau = 0.347` line the eval prints is `−log γ`-based and is **informational only**
for a QRL checkpoint, whose ruler is `step_cost = 1.0`. The 203-point sweep on the 2,000
held-out val questions is what decides τ, exactly as for every other row.

**If the memory probe does not fit in 16 GB:** `--set qrl.push_chunk_cols=32` splits the
`S × C` push matrix by goal columns and keeps the mean exact (pinned by a test). It lowers the
transient peak of the distance's internals — IQE materialises several `(S, C, D/k, 2k)` fp32
tensors — but it does **not** shrink the autograd graph, so it is a lever, not a cure. Cutting
`sampling.sequences_per_micro_batch` is the other one, and it breaks the matched batch stream,
so it is a last resort and must be reported.

## 9. The rows to beat

| run | mean ProcessBench F1 | goal-head val F1 |
|---|---|---|
| `abl_cf_only` | 0.2599 | 0.5900 |
| `cf_lam2_tau005` | 0.2611 | 0.5954 |
| `pqm_zeta4` (PQM baseline, matched) | 0.2682 | 0.5766 |

One run each, no seed replicate — quote the gap, not a ranking, unless it is large.
`qrl_prm/report.py` prints all of this plus the math leak split and the per-baseline deltas.
