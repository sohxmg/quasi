# Feynman-PRM — Experiment Log

**Every training run, in order, with the reasoning that produced it.**
A process reward model that scores a reasoning step by a learned quasimetric distance —
what was tried, why, what it scored, and how each attempt failed.

*Compiled 2026-08-29. Sources: `CLAUDE.md` (§9, §14, §18), `CLAUDE2.md`, `runs/*/RESULT.md`,
`runs/*/{config.resolved.yaml, events.jsonl, val_f1.json, processbench.json}`, and the
`unrealparticles-iit-roor/feynman-prm` W&B project.*

---

## Experiments at a glance

|  |  |  |  |
|---|---|---|---|
| **74** | **≈ 64 h** | **22** | **17** |
| tracked training runs | of logged GPU time | experiments written up below | failure modes catalogued |

### Where the 60 runs of this project went

```
full 1,464-step training runs, completed    ██████████                       10
short GPU probes (20-120 steps)             ███████████████████              19
phase-2 goal-head fits                      ████████████████                 16
runs killed / crashed mid-training          █████                             5
aborted launches & smoke checks             ██████████                       10
                                                                     ──────────
                                            W&B project `feynman-prm`        60
                                            predecessor project `qprm`       14
                                                                     ──────────
                                                              TRACKED RUNS   74
```

| | count | note |
|---|---:|---|
| **Full training runs that completed** | **10** | 1 epoch = 1,464 optimizer steps each, ~2–7 h apiece |
| Training runs killed, crashed or stopped part-way | 5 | 270 – 1,090 steps; two were deliberate early stops |
| Short GPU probes run before committing a card | 19 | 20 – 120 steps; 7 died at the launch gate, by design |
| Phase-2 goal-head fits and re-fits | 16 | frozen representations, minutes each |
| Aborted launches and smoke checks | 10 | |
| **Runs on the predecessor project** (`qprm`) | **14** | the token-level PRM this design was written against |
| **Total tracked runs** | **74** | |
| **Total logged GPU time** | **≈ 64 h** | 57.7 h this project + 6.6 h predecessor |

### What was produced without a GPU

| | count |
|---|---:|
| Analysis passes that produced findings from files already on disk | **4** |
| Distinct code-level failure modes catalogued so they are never rediscovered | **17** (B5–B17) |
| Independent cross-checks written for one derivation *before* it was launched | **13** |
| CPU test suite at the end of the QRL work | **598 passed, 2 skipped** |
| Methods implemented from scratch under matched conditions | **3** — TMD-derived, PQM, QRL |

**Written up below:** 13 numbered runs that each carry a decision, 5 QRL calibration probes, and
4 no-GPU analysis passes — 22 experiments. The remaining ~40 W&B rows are the relaunches,
phase-2 fits and launch failures behind them.

---

## Contents

- [The idea](#the-idea)
- [The scoreboard](#the-scoreboard)
- [Run 0 — `phase1_lambda_good_0_baseline` · the first full attempt](#run-0--phase1_lambda_good_0_baseline--the-first-full-attempt)
- [Run 1 — `phase1` · ⑥ `L_good` added, and the first real number](#run-1--phase1---l_good-added-and-the-first-real-number)
- [Interlude A — four analysis passes, no GPU (2026-08-03 → 08-05)](#interlude-a--four-analysis-passes-no-gpu-2026-08-03--08-05)
- [Run 2 — `phase1_nce_temp_relu2` · four changes at once](#run-2--phase1_nce_temp_relu2--four-changes-at-once)
- [Run 3 — `phase1_mask_relu2` · the τ = 1.0 control](#run-3--phase1_mask_relu2--the---10-control)
- [Run 4 — `phase1_cf_term` · ④ and ⑦ enter a gradient](#run-4--phase1_cf_term---and--enter-a-gradient)
- [Run 5 — `phase1_cf_term_taucf01` · the relaunch, and ⑦'s verdict](#run-5--phase1_cf_term_taucf01--the-relaunch-and-s-verdict)
- [Run 6 — `abl_cf_only` · the separating ablation](#run-6--abl_cf_only--the-separating-ablation)
- [Run 7 — `pqm_zeta4` · THE BASELINE: PQM under identical conditions](#run-7--pqm_zeta4--the-baseline-pqm-under-identical-conditions)
- [Run 8 — `nce_tau4` · giving ① a live gradient at last](#run-8--nce_tau4--giving--a-live-gradient-at-last)
- [Run 9 — `nce1_masked` · the sibling-late mask, killed early on purpose](#run-9--nce1_masked--the-sibling-late-mask-killed-early-on-purpose)
- [Run 10 — `cf_lam2_tau005` · pushing ④ four times harder](#run-10--cf_lam2_tau005--pushing--four-times-harder)
- [Part II — changing the approach: TMD → QRL (2026-08-25)](#part-ii--changing-the-approach-tmd--qrl-2026-08-25)
- [Run 11 — `qrl_iqe` · QRL's constrained objective, first full run](#run-11--qrl_iqe--qrls-constrained-objective-first-full-run)
- [The QRL probe series — five short runs, and one diagnosis that was wrong](#the-qrl-probe-series--five-short-runs-and-one-diagnosis-that-was-wrong)
  - [Probe P1 — `qrl_iqe_off3_probe` · offset 3](#probe-p1--qrl_iqe_off3_probe--offset-3)
  - [Probe P2 — `qrl_iqe_off8_probe` · the offset alone](#probe-p2--qrl_iqe_off8_probe--the-offset-alone)
  - [Probe P3 — `qrl_iqe_lam15_probe` · the multiplier instead](#probe-p3--qrl_iqe_lam15_probe--the-multiplier-instead)
  - [Probe P4 — `qrl_iqe_path_probe` · the merged path constraint](#probe-p4--qrl_iqe_path_probe--the-merged-path-constraint)
  - [Probe P5 — `qrl_iqe_split_probe` · split back apart](#probe-p5--qrl_iqe_split_probe--split-back-apart)
- [Run 12 — `qrl_iqe_split` · three constraints, three multipliers](#run-12--qrl_iqe_split--three-constraints-three-multipliers)
- [Where it stands](#where-it-stands)

---

## The idea

A **process reward model that scores a reasoning step by a learned quasimetric distance**
`d(state, goal)` — "how far is this partial solution from being solved". A step is flagged as
the error when the distance to the goal *stops falling*:

```
Δ_i = d(ψ_i, g) − d(ψ_{i−1}, g)      flag the solution iff  max_i Δ_i > τ
                                      localise it at the first  i  with  Δ_i > τ
```

One Math-Shepherd row = one trajectory; state `s_i` = the prefix `(question, step_1…step_i)`,
read off the LM hidden at the separator token. `T` steps give `T+1` states from **one** forward
pass — that is what makes a large negative pool affordable.

**Why a quasimetric rather than a similarity.** Two properties are wanted and neither exists in
a cosine score: the **triangle inequality** (credit assignment / stitching across partial
solutions) and **asymmetry**, `d(s,g) ≠ d(g,s)` (irreversible dead-end states are detectable).
The loss math is adapted from **TMD** (Temporal Metric Distillation, Myers et al.), a
goal-conditioned RL method for robot environments; trained on **Math-Shepherd**, evaluated on
**ProcessBench**.

**The four root causes it was designed against.** A predecessor project trained a token-level
quasimetric PRM whose score came out *at chance*. The design brief for this repo is a direct
answer to each failure:

| Old root cause | This design's answer |
|---|---|
| **A** objective had no correctness signal (goals drawn from a trajectory's *own* future, so an incorrect ending was a legitimate positive) | ⑤ `L_step`, a first-error-boundary Bradley–Terry term on the distance itself, supervised by Math-Shepherd's per-step labels |
| **B** goal collapse (γ=0.995 → 77% of goals clamped to the terminal) | `discount = 0.5`, measured to give **4.19 distinct goals**/solution vs TMD's 1.16 |
| **C** the eval score was never trained | ⑤ acts on `Δ_{z+1}` — *the exact statistic eval thresholds* — at weight 1.0, not as an option |
| **D** `ψ(g_mean)` was a degenerate goal (averaging 30k terminals collapses to the population mean) | a question-conditioned **goal head** `g = goal_head(h_{s_0})`; no centroid anywhere in the codebase |

**The phase-1 loss set** (fixed weights, all on one shared distance matrix): ① `L_NCE`
contrastive, ② `L_I` action invariance, ③ `L_T` temporal backup (the "ruler": every good step
must cost `−log γ = 0.693`), ④ `L_CF` counterfactual invariance, ⑤ `L_step` correctness,
⑥ `L_good` false-positive control, ⑦ `L_term` multi-positive over a question's terminals.
**Phase 2** freezes everything and fits the goal head alone on cached vectors.

**Standing constraints.** No value head — one head, one score, the distance. τ is fitted on a
held-out 2,000-question Math-Shepherd split and **never** on ProcessBench. The 587 contaminated
ProcessBench-math questions stay in training and the math F1 is always reported split
leaked/clean.

---

## The scoreboard

Every full run is 1 epoch = **1,464 optimizer steps** over ~150k sequences,
`Qwen2.5-Math-1.5B-Instruct` + LoRA r16, on one 16 GB card (later a rented A100 40 GB).
Every row below shares the same data selection, tokenisation, batch shape, optimizer and
schedule — the comparisons are matched by construction, not by re-derivation.

| # | run | date | family | headline change | val F1 † | ProcessBench mean F1 ‡ |
|---|---|---|---|---|---|---|
| 0 | `phase1_lambda_good_0_baseline` | 07-27 | TMD | first full attempt | — | *died at step 970* |
| 1 | `phase1` | 07-29 | TMD | ⑥ `L_good` on, 34,650 questions | 0.5313 | **0.2409** |
| 2 | `phase1_nce_temp_relu2` | 08-04 | TMD | τ_NCE √512, `relu²`, NCE mask, ζ 0.1 | 0.5597 | **0.2588** |
| 3 | `phase1_mask_relu2` | 08-04 | TMD | the τ_NCE = 1.0 control | 0.5074 | *not run* |
| 4 | `phase1_cf_term` | 08-16 | TMD | ④ `L_CF` + ⑦ `L_term` enter a gradient | — | *crashed at 1090* |
| 5 | `phase1_cf_term_taucf01` | 08-18 | TMD | τ_cf 0.1, ζ 0.2 | 0.5615 | **0.1362** |
| 6 | `abl_cf_only` | 08-18 | TMD | ⑦ off — the separating ablation | 0.5634 | **0.2599** |
| 7 | `pqm_zeta4` | 08-19 | **PQM — the baseline** | PQM re-implemented, matched conditions | 0.5766 § | **0.2682** |
| 8 | `nce_tau4` | 08-20 | TMD | τ_NCE 22.627 → 4.0 | 0.5533 | **0.1752** |
| 9 | `nce1_masked` | 08-22 | TMD | sibling-late NCE mask | — | *stopped at 1030* |
| 10 | `cf_lam2_tau005` | 08-22 | TMD | 4× effective weight on ④ | 0.5637 | **0.2611** |
| — | *five short probes* | 08-26/27 | **QRL** | offset 3 / 8, λ 1.5, merged path, split | — | *see P1–P5* |
| 11 | `qrl_iqe` | 08-26 | **QRL** | the approach change: TMD → QRL | 0.2508 | *not run* |
| 12 | `qrl_iqe_split` | 08-28 | **QRL** | split path/local constraints | 0.2405 | **0.1077** |

**Read the family column.** Rows 0–10 are the **TMD-derived** method — a fixed-weight seven-term
loss set — being iterated on. Row 7 is the **one external baseline**: PQM (Li & Li, ICLR 2025)
re-implemented under identical conditions, the only row that is not our method. Rows 11–12 are a
**deliberate change of approach**, not a baseline: the foundation moved from TMD to **QRL**
(Quasimetric RL), with the counterfactual loss carried over and re-expressed as a Lagrangian
constraint in QRL's own form.

† phase-1 ceiling: F1 on 400 held-out Math-Shepherd questions / 4,549 trajectories, scored
against a **real terminal**, so it bypasses the goal head and measures the geometry alone.
‡ mean over gsm8k / math / olympiadbench / omnimath, one global τ fitted on val.
§ `pqm_zeta4`'s val figure is over 1,999 questions / 18,220 trajectories — a different
denominator, not directly comparable to the rows above it.

**The one-line story.** The TMD-derived method plateaus around **0.26 mean F1**, and three
separate attempts to push past it — a stronger ④, a live ①, and finally a change of foundation
to QRL — each either did nothing or made it worse. The **PQM baseline under identical conditions
scores 0.2682**, slightly above every row of our own method.


---

## Run 0 — `phase1_lambda_good_0_baseline` · the first full attempt
`fto0cx8e` · 2026-07-27 · 16 GB card · died at step 970/971 · `n_questions: 23,000`

**What it was.** The loss set as originally specified: ①②③⑤ live, `λ_good = 0` (⑥ did not
exist yet), ④ and ⑦ inert. The first time the whole pipeline ran end to end.

**Why.** It is the baseline. Everything before it was CPU tests, a single-example gradient
check and 20-step GPU probes.

**Result.** Read off the `step750` checkpoint, because there is no `final/`:

| at step 750 | |
|---|---|
| val F1 ceiling | **0.456** |
| fitted τ | **2.39** (natural τ = 0.347 → **6.9×**) |
| good-step `Δ` of *correct* trajectories, mean | **+0.240** |
| …`frac_above_natural` | **0.34** |

**What failed — two things, and both mattered.**

1. **The run was destroyed by its own guard (bug B11).** The check that asserts "the LR
   scheduler actually moved" compared the LR at the end against a value read immediately after
   the scheduler was built. Both are exactly `0.0` — `LambdaLR.__init__` applies `lr_lambda(0)`,
   which under warmup is `0/warmup`, and a completed cosine ends at `0.5·(1+cos π)`. So the
   guard fired on **every run that finished** and passed only on runs cut short by
   `--max-steps`. `save_checkpoint(".../final")` sat *below* the raise and never ran.
   *Fix, and the rule it produced:* track `lr_min_seen`/`lr_max_seen` across every step, and
   **write the artifact before the diagnostic that might destroy it.**
2. **The modelling failure: a good step is supposed to cost `−0.693` and it cost `+0.240`.**
   34% of good steps of *correct* trajectories sat above the natural threshold, so τ had to
   climb to 2.39 to dodge that tail, and true-positive rate collapsed with it. ⑤ `L_step` only
   ever trains **~28 of ~348** source rows in a batch — the first-error transition of incorrect
   trajectories. **Nothing trained the other ~92%**, and the gap is one-sided by construction.

*(An earlier lesson from the same week: ③ `L_T` was predicted to start at ≈ −10.5 and actually
started at **+8760**. `clip_t` bounds an **exponent**, not a step count. Reparameterised as
`clip_t_gain`, which reproduces TMD's setting at TMD's own γ. The general rule — a threshold on
a quantity whose scale is set by `−log γ` must be a **ratio** to that scale, never a constant
offset — recurs three more times below.)*

---

## Run 1 — `phase1` · ⑥ `L_good` added, and the first real number
`jjkad2ae` (+ phase 2 `vwe7gigh`) · 2026-07-28→29 · 1,460 steps, ~4 h · `n_questions: 34,650`

**Change.**
- **New term ⑥ `L_good`** at weight 1.0: `mean relu(Δ_i − c)`, `c = −0.693`. Not a new target —
  `c` is exactly where ③'s ruler already says a good step belongs. It adds a **ceiling** to a
  target that only had a floor.
- `n_questions` 23,000 → **34,650** (~150k sequences, 1,464 steps).
- The B11 guard rewritten; `prepare_data.py` re-run because the selection SHA moves with
  `n_questions`.

**Why.** Directly from run 0's `frac_above_natural = 0.34`. Targets written down *before*
launch: 0.34 → ~0.05 and τ 2.39 → ~0.35. A guard was written down too — `L_good` pays for its
Δ reduction out of ② `L_I`, so `invariance/residual_diagonal ≤ 0.15` by step ~200 was the
early tell that the weight was wrong.

**Result — the first end-to-end ProcessBench number in the project.**

| subset | acc_error | acc_correct | F1 |
|---|---|---|---|
| gsm8k | 0.285 | 0.622 | **0.391** |
| math | 0.182 | 0.411 | **0.252** |
| — leaked (587) / clean (413) | | | 0.274 / 0.227 |
| olympiadbench | 0.089 | 0.372 | **0.144** |
| omnimath | 0.112 | 0.419 | **0.177** |
| **mean** | | | **0.2409** |

Phase 2 itself worked: `goal/loss` 7.469 → 4.491 over 20 epochs, `goal/pred_variance` **rising**
0.535 → 0.703 (which rules out "it learned a constant"), and the whole fit took **53 seconds** of
a 4,554 s wall clock — the phase-split paying for itself. Phase-1 val F1 0.5313.

**What failed.**

- **τ still landed at 1.1685 — 3.4× the natural 0.347.** ⑥ moved the bulk and did not move the
  tail: `mean relu(·)` is a *linear* hinge, indifferent between many small violations and one
  large one. Mid-run the bulk swung +0.392 → −0.412 while `frac_above_natural` **regressed**
  0.070 → 0.16, `p99` went 0.86 → 2.43 and `good/delta_max` reached **7.58**.
- **The contamination split came out backwards.** The 587 questions whose *problems* the model
  trained on scored 0.274 against the clean 413's 0.227 — only +0.047. That is what you see if
  the metric is not keying on the problem at all.
- **The skyline (gold-answer goal) was *below* the goal head** on both fully-joined subsets —
  which reads as "the metric never learned correctness" rather than "the goal head is the
  bottleneck".
- **And a guard hid the τ overshoot for four days (bug B12).** `report_processbench.py` tested
  `τ > expected + 1.0` — an **additive** slack on a multiplicative quantity. With
  `expected = 0.347` the warning could not fire below τ = 1.347, so a 3.4× overshoot sailed
  under it and printed *"tau is near the midpoint — the ruler and the margin both held."*
  Fixed to `τ > 2.0 · expected`, and the rule generalised: **print the ratio, never a verdict.**

---

## Interlude A — four analysis passes, no GPU (2026-08-03 → 08-05)

Before spending another 4 GPU-hours, the saved `deltas.npz` and the existing metric streams were
made to answer what they could. These are not runs and produced no new numbers to report; they
are what determined runs 2–4.

**A1 — Where the headroom actually is.** Exact arithmetic on run 1's own table, perfecting one
factor at a time:

| | mean F1 |
|---|---|
| as measured | 0.241 |
| if `acc_correct` → 1.0 (never falsely flag a correct solution) | 0.279 **(+0.038)** |
| if `P(exact \| flagged)` → 1.0 (perfect localisation at current flag rates) | 0.522 **(+0.281)** |

**Seven to one in favour of localisation.** This is the single most actionable number the
project produced — and note what it says about ⑥ `L_good`: the good-step tail is an
`acc_correct` mechanism, so *even a perfect* `L_good` buys from the +0.038 column.

**A2 — Detection is at or near chance on three of four subsets.** Threshold-free detection AUC:
gsm8k 0.639, math 0.579, olympiadbench and omnimath ~0.53. On olympiadbench the entire detection
channel is worth **eight thousandths** of F1.

**A3 — The goal carries no information.** Each sample was scored twice: once with its own
`goal_head(h_{s_0})`, once with a **seeded derangement** so every sample gets *another
question's* goal.

| gsm8k, 207 errored samples | within-solution rank | signal above the 0.5 null |
|---|---|---|
| own goal | 0.3871 | 0.1129 |
| **another question's goal** | 0.3839 | **0.1161 — 103% retained** |

`diff −0.0032 ± 0.0255` (0.13σ), per-sample `r = 0.477`. The `r` is what makes the test bite:
the goals are genuinely different and genuinely change which step is picked — they just carry no
information about *which step is wrong*. The standing diagnosis became: **the geometry resolves
POSITION along a trajectory, not CORRECTNESS of a step.**

**A4 — ①'s negatives were the mechanism, and one of them was a bug.** With the goal at `s_6` and
the positive at `s_3`, an unmasked `s_5` from the same trajectory is pushed *away* from `s_6` —
but `s_5` is one step from the end and should be the *closest* state of all. Measured: **41.8%
of columns** carry at least one such row, 0.64 per column, and they are the closest rows in the
pool. A well-trained ① was also spending ~24% of its push-away gradient shoving two *correct*
solutions of one question apart — which is what took `gate/recall_at_1` from 0.618 untrained to
**0.276**. Fix built: `nce_mask_nearer_same_traj`, the *surgical* mask that keeps rows earlier
than the positive (honest hard negatives) and drops only the nearer ones.

**A5 — free +0.017 that was declined.** The documented decision rule (`argmax`) and the shipped
one (`first crossing`) are different; swapping gives mean F1 0.241 → 0.258. **Not adopted** —
that table is fit on ProcessBench, and the rule that τ is never fitted on the test set applies to
the decision rule too.

---

## Run 2 — `phase1_nce_temp_relu2` · four changes at once
`m1pqt8ot` · 2026-08-04 · 1,460 steps, ~4 h

**Changes (a deliberate bundle).**

| # | key | from → to | evidence behind it |
|---|---|---|---|
| 1 | `losses.nce_temperature` | 1.0 → **22.627** (= √512) | TMD's own scaling |
| 2 | `losses.good_loss.form` | `relu` → **`relu_squared`** | run 1's tail failure |
| 3 | `sampling.nce_mask_nearer_same_traj` | false → **true** | A4's 41.8%. The only change with a direct measurement behind it |
| 4 | `losses.zeta` | 0.05 → **0.1** | the ruler was decaying to ~0.49 against a target of 0.693 |

**Why `relu_squared`.** Its gradient is `2·excess` instead of `1`, so a violator is priced by
*how far out it is*, while keeping both properties `relu` was chosen for (exactly zero below
`c`, exactly zero gradient **at** `c`). Simulated against the measured mid-run Δ distribution
the term's *total* pull is unchanged; **where it lands is not** — the share reaching the top
decile goes 16.5% → 36.9%.

**Result.** mean ProcessBench F1 **0.2588** (+0.018 over run 1), val F1 0.5597, τ 0.3403 against
a natural 0.3466 — **the τ overshoot was cured**.

**What failed.**

- **It cannot attribute.** Four simultaneous changes is past the point where an escape exists.
  All the run can say is "this combination scored X".
- **The bundle is self-defeating, and this was flagged in writing before launch.**
  `nce_mask_nearer_same_traj` is a repair **to** ① `L_NCE`, and `τ = √512` divides ①'s gradient
  by 22.6. Measured on free latents, that demotes ① **from the largest per-term gradient in the
  loss set to the second smallest**. So the run applied a fix and turned down the term the fix
  operates on. The recommendation at the time was to run `τ = 1.0 + mask + relu²` first; the
  temperature was kept.
- **ζ = 0.1 was predicted, in advance, to be invisible.** Doubling moves ③ `L_T` from 5th place
  in the gradient ranking to 5th place. 0.2 was the principled stopping point.

---

## Run 3 — `phase1_mask_relu2` · the τ = 1.0 control
`jr3hpurd` · 2026-08-04 (same day) · 1,460 steps

**Change.** Run 2 minus the temperature: `nce_temperature` **1.0**, `zeta` back to **0.05**.
`relu_squared` and `nce_mask_nearer_same_traj` kept.

**Why.** This is the clean test run 2 was told to do first — ① at full strength, so the NCE
mask can actually be judged.

**Result.** Phase-1 val F1 **0.5074** — *below* run 2's 0.5597. Not carried into phase 2 or
ProcessBench.

**What failed / what it taught.** The clean control **lost** on the one number available for
it, which is why the √512 temperature survived into runs 4–7 despite the argument against it —
and that is exactly the trap: the comparison is two deltas (τ *and* ζ), so it does not cleanly
separate them either. The consequence compounded: ① stayed muted for **three consecutive runs**,
and every conclusion drawn about ④ and ⑦ in that window inherits the caveat.

---

## Run 4 — `phase1_cf_term` · ④ and ⑦ enter a gradient
`wc5byua1` · 2026-08-16 · **crashed at step 1090/1464**

**Changes** (two, against run 2 — a chosen confound, not an oversight):
- `lambda_cf` 0 → **1.0**. ④ `L_CF`, the counterfactual-invariance loss, trains for the first
  time. Its two gates had just cleared: **27,114 generated examples on disk**, and the
  attach-to-the-main-batch path wired.
- `lambda_term` 0 → **1.0**. ⑦ `L_term`, the only term in the set that says *"pull these
  together"* rather than *"stop pushing these apart"* — the direct counter to A4's finding that
  ① scatters a question's correct siblings.

**Why.** A3 said the metric resolves position, not correctness. ④ says a *meaning-preserving
rewrite of a step is the same point*; ⑦ says *two correct solutions of one question end in the
same place*. Both are attempts to inject correctness structure that trajectory position cannot
supply. The human was offered the sequenced pair (one change each, ~8 h) and chose the bundle
(~4 h) for faster signal.

**Result.** The 20-step probe passed every launch check (`nce` 6.2522 vs 6.2538 expected,
`invariance` 10.5939 vs 10.5969, 27,110 of 27,114 CF examples bound). The run then crashed at
step 1090 and produced no eval.

**What failed.**

- **① is not slow at τ = √512, it is not running.** Measured against run 1 at the same step:

  | at step ~700 | τ = 1.0 (run 1) | τ = 22.627 (this run) |
  |---|---|---|
  | `nce/categorical_accuracy_backward` | ~0.15 | **~0.008** |
  | distance spread `logit_std × τ` | ~3.0 | **0.68** |
  | `nce/loss` − `nce/chance` | **−1.9** | **~0** |

  Run 2's own written trigger — *"if this run is still near 2× chance at step 500 the mask
  result cannot be read"* — had fired. The mask was wired correctly (`nce/nearer_set_size` 0.65
  vs a simulated 0.64) and had been **muted for two consecutive runs**.
- Infrastructure cost a day too: `munmap_chunk(): invalid pointer` with **no Python traceback**,
  aborting after `wandb.init` and before the first data event (bug B15). Native heap corruption
  in a process holding both pyarrow and torch — and the abort was on the *free*, not the read.
  The first fix (reordering imports) was wrong and the crash dump proved it. Real fix: convert
  the parquet **once, in a child interpreter with no torch in it**, to a flat `.npz`.

---

## Run 5 — `phase1_cf_term_taucf01` · the relaunch, and ⑦'s verdict
`zor1vhd9` (+ phase 2 `8sudec6y`) · 2026-08-18 · rented **A100 40 GB** · 1,460 steps, ~2.1 h

**Changes** (two, against run 4): `lambda_cf_temperature` 1.0 → **0.1** (τ and λ are one knob —
gradient ~ λ/τ — so this is a 10× effective weight on ④), `zeta` 0.1 → **0.2**.
`λ_term_temperature` stayed 1.0, and that is the number that mattered.

**Why.** Run 4 crashed before an eval, so this is the relaunch; the two deltas ride along
because ④ at λ=1/τ=1 had shown no separation and ζ=0.1 had been predicted to be invisible.

**Result — a 47% collapse, and the diagnosis is the interesting part.**

| subset | run 2 baseline | this run | Δ |
|---|---|---|---|
| gsm8k | 0.3757 | **0.2228** | −41% |
| math | 0.3044 | **0.1768** | −42% |
| olympiadbench | 0.2037 | **0.0968** | −52% |
| omnimath | 0.1513 | **0.0483** | −68% |
| **mean** | **0.2588** | **0.1362** | **−47%** |

**Phase 1 was intact.** Val F1 **0.5615** against the 0.560 benchmark, τ fitted 0.2453 against a
natural 0.3466 with sensitivity 0.0061 — a flat curve near its predicted value. The collapse was
first attributed to the geometry and **that attribution was wrong**: `val_f1.py` hands the scorer
a real terminal and bypasses the goal head, and this is the first time the project took the
*"phase 1 good, phase 2 bad"* branch.

**What failed — ⑦ did the pushing and never did the pulling.**

The masked goal-gate measurement (score the identical rows twice, with and without the printed
answer span visible) was run for the first time:

| | run 2 (λ_term = 0) | this run, masked | this run, unmasked |
|---|---|---|---|
| `gate/recall_at_1` | 0.3390 | **0.3695** | 0.6362 |
| `gate/within_question_terminal_spread` | **0.6137** | 0.9612 | 1.0291 |
| `gate/across_question_terminal_spread` | **1.0338** | 1.8642 | **3.9582** |

1. **42% of ⑦'s apparent gain is the printed answer.** Every correct solution of a question
   prints the same final number and incorrect siblings essentially never do
   (`answer_match_auc = 0.927`), so ⑦ can be solved by clustering on that string — which
   transfers to nothing, because a PRM scores *unfinished* solutions. The shortcut was there and
   **the encoder took it.**
2. **The gauge ⑦ exists to move went the wrong way.** `within_question_terminal_spread` is
   *looser* (0.614 → 1.029) than in a run that never had the term. The entire recall/AUC gain is
   the **across** term inflating 3.8×. At convergence the negatives sit at `exp(−4.94)` ≈ **3.4%
   of the softmax denominator** — pushed until they fell out of their own loss. Repulsion is the
   cheap half of a SupCon objective and ⑦ took it.
3. **That collided with a fixed margin.** `step_margin = 1.386` is a constant; ⑦ does not scale
   it. The manifold inflated 3.8×, the goal head's val loss went 1.110 → **4.938**, and **53% of
   the inflation is the printed answer — which `goal_head` structurally cannot see**, since it
   reads `h_{s_0}`, the prompt and only the prompt. Goal error ÷ margin: 0.80 → **3.56**. Noise
   at 3.6× the signal. There is no phase-2 fix for this.
4. **The surviving real structure is ~1.1 SE.** `per_question_recall_std` 0.397 over 200
   questions ⇒ SE ≈ 0.028, against a masked gain of +0.031.

**And the skyline fell too — which is *not* a phase-1 signal.** The skyline and the reported
path share **one** τ, calibrated with the goal head in the path, so a mis-scaled goal head moves
both arms together.


---

## Run 6 — `abl_cf_only` · the separating ablation
`lv35vhln` (+ phase 2 `azglleg1`) · 2026-08-18 · A100 40 GB · 1,460 steps, ~2.1 h

**Change.** Exactly **one** delta against run 5: `lambda_term` 1.0 → **0.0**. `ζ = 0.2` and
`τ_cf = 0.1` are held at run 5's values *deliberately* — comparing against the run-2 benchmark
instead would mean a three-delta comparison and would answer a different question. `λ_term = 0`
is an exact revert: ⑦ is skipped in the total but every `term/*` diagnostic is still logged.

**Why.** Run 4 bundled ④ and ⑦; run 5 measured a −47% collapse and attributed it to ⑦. This is
the run that tests the attribution. It was designed with its own decision rule written in
advance: *returning to ~0.259 says ⑦ was the whole story and ④ is neutral; landing above 0.259
says ④ is a real win that ⑦ was masking.* Explicitly **not a reduced `λ_term`** — a smaller
weight shrinks the real structure and the answer-keyed inflation in the same proportion, and the
real structure was already at 1.1 SE.

**Result.**

| | run 2 baseline | run 5 (④+⑦) | **this run (④ only)** |
|---|---|---|---|
| mean ProcessBench F1 | 0.2588 | 0.1362 | **0.2599** |
| gsm8k | 0.3757 | 0.2228 | 0.3887 |
| phase-1 val F1 | 0.5597 | 0.5615 | 0.5634 |

**The first branch fired, cleanly.** ⑦ `L_term` was the entire collapse; ④ `L_CF` is neutral.
⑦ has shipped at `λ_term = 0.0` ever since.

**What failed.** ④ being *neutral* is not a success — it means the counterfactual corpus, ~27k
LLM-generated meaning-preserving rewrites and the single most expensive data asset in the
project, bought **+0.001 mean F1**. And ⑦ has still never been tested against a live ①, which
is the configuration it was designed for.

---

## Run 7 — `pqm_zeta4` · THE BASELINE: PQM under identical conditions
`wqzqhk55` · 2026-08-19 · A100 40 GB · 1,460 steps, ~2.1 h · `pqm_baseline/`

**What it is.** The project's one external reference point, and the only row in this log that is not our own method. **PQM** (Process Q-value Model, Li & Li, ICLR 2025)
re-implemented and trained under Feynman-PRM's *exact* conditions. It changes **exactly two
things**: the head (ψ/φ MLPs + quasimetric distance → one `Linear(1536,1)` value head,
zero-init) and the objective (the seven-term loss set → PQM's Q-ranking loss, ported verbatim
from the authors' `train_main.py:61-78`, vestigial `.flip()` and `1e-5` epsilon included, and
pinned against a copy of their function in a test). Dataset, selection SHA, seed, backbone,
LoRA config, batch shape, optimizer, LR schedule, eval protocol and τ discipline are the
identical config object.

`pqm.zeta = 4.0` is **PQM's own negative-reward offset** (negatives scored `exp(r+ζ)`), not
Feynman's `losses.zeta` — a naming collision the launch log prints side by side on purpose.

**Why.** By this point the project had a plateau (0.24 → 0.26 over four runs) and no external
reference point. A published method under matched conditions is what turns "0.26" into a
statement. It lives in a **sibling package** rather than inside `feynman_prm/`, because a grep
guard forbids a value head anywhere in the method and a value head is the defining feature of
this baseline — scope the guard, never rename around it.

**Result.**

| | Feynman (`abl_cf_only`) | **PQM** |
|---|---|---|
| gsm8k | 0.3887 | **0.3820** |
| math | 0.2988 | **0.3438** |
| olympiadbench | 0.2008 | **0.1901** |
| omnimath | 0.1513 | **0.1569** |
| **mean** | 0.2599 | **0.2682** |

The zero-init launch check landed exactly on its closed-form prediction
(`pqm/loss` 3.9835 vs the analytic 3.9835, abs error 6.8e-8), and it used **less** memory
(11.96 GiB vs 12.08) since it carries no ψ/φ MLPs.

**What it means.** A simple pointwise-ranking value head beats the quasimetric on the mean and
by +0.045 on `math`. Caveats stated in the file and worth repeating: this is *our* row under
matched conditions, **not PQM's published number** — PQM's paper reports Best-of-N on a 7B full
finetune and never reports ProcessBench, ζ=4 was tuned by its authors on that 7B setting, and a
ζ sweep here was offered and declined.

---

## Run 8 — `nce_tau4` · giving ① a live gradient at last
`mdbsf31m` (+ phase 2 `09fhcl32`) · 2026-08-20 · A100 40 GB · 1,460 steps

**Change.** One delta against run 6: `losses.nce_temperature` 22.627 → **4.0**.

**Why — the long-running thread finally addressed.** τ_NCE has a history: it started at 1.0,
was blamed for a collapse (bug B10b: `1/√D` made ① ~4× weaker than a batch-wide ② which has a
trivial minimum), was moved to √512 = 22.627 on TMD's authority — and then measured, three
separate times, to **starve ①**. At √512, ① is the second-*smallest* gradient in the loss set;
`categorical_accuracy_backward` sat at ~2–3× chance after 700 steps where τ=1.0 reached 10.6×.
Every conclusion about ④ and ⑦ in runs 4–6 was drawn with the largest term in the set muted
22.6×. **τ = 4.0 is the deliberate midpoint**: a live ① without returning to the value that had
once been blamed for a collapse, with everything else at its safe, ablated setting
(`λ_term = 0`).

**Result — and it is the strangest table in the log.**

| subset | run 6 (τ=22.627) | **this run (τ=4.0)** |
|---|---|---|
| gsm8k | 0.3887 | **0.1268** |
| math | 0.2988 | 0.2659 |
| olympiadbench | 0.2008 | 0.1649 |
| omnimath | 0.1513 | 0.1434 |
| **mean** | **0.2599** | **0.1752** |

Phase-1 val F1 was **0.5533** — essentially unchanged (0.5634 → 0.5533). The geometry did not
break.

**What failed.** gsm8k's `acc_correct` collapsed **0.9948 → 0.0933** while `acc_error` barely
moved (0.242 → 0.198). The model went from flagging almost no correct gsm8k solution to flagging
almost all of them, at a τ (0.3444) that is *right on* the natural 0.3466 — and the other three
subsets moved only moderately. **This is not explained anywhere in the project's records.** It
is a single-subset false-positive explosion at a healthy global threshold and a healthy phase-1
val F1, which points at the goal head / τ-transfer path rather than the geometry, but no
follow-up diagnostic was run. **Open finding.**

---

## Run 9 — `nce1_masked` · the sibling-late mask, killed early on purpose
`dlujoput` · 2026-08-22 · A100 40 GB · **stopped deliberately at step 1030/1464**, ~1.7 h billed

**Change.** One delta against run 8: `sampling.nce_mask_sibling_correct_late` false → **true**.

**Why.** This is the second of the two NCE-negative repairs designed back in Interlude A, and
run 8 was the first checkpoint in the project with an ① strong enough to judge it against. Run 2
had proved that judging a repair to ① while ① is muted is uninformative.

**Result — a decisive negative, obtained for 60% of the GPU cost.**

The mask **demonstrably binds** — `nce/negatives_masked` roughly doubles to triples
(0.42 → 1.58 at step 500) — and **every downstream number is identical to the baseline to 3–4
decimals**:

| step | `nce/loss − nce/chance`, run 8 | this run |
|---|---|---|
| 250 | −0.0261 | −0.0219 |
| 500 | −0.0939 | −0.0943 |
| 750 | −0.2318 | −0.2357 |
| 1000 | −0.2586 | −0.2588 |

Same for `logit_std`, `probe03/gap`, `invariance/residual_diagonal`, `backup/loss`.

**Why it was stopped, and the finding.** `nce/negatives_per_column` is ~460, so excluding ~1.5
columns instead of ~0.8 is a change to **~0.3% of the negative pool** — too small a perturbation
to move F1 out of noise. At `nce_sibling_late_margin = 1` (keeping only the last two φ states)
the principled mask **is not a lever**; raising that margin is the knob that would make it one,
and was not tried. Every health check passed (`logit_std` 0.055 → 0.390, invariance under the
0.15 guard throughout, backup negative from step ~400) — this is a run that answered its
question early, not a broken one.

---

## Run 10 — `cf_lam2_tau005` · pushing ④ four times harder
`yksnqcp5` (+ phase 2 `aaaks91e`) · 2026-08-22 · A100 40 GB · 1,460 steps, ~1.1 h

**Changes.** Two deltas against run 6, both on ④: `lambda_cf` 1.0 → **2.0** and
`lambda_cf_temperature` 0.1 → **0.05**. Since λ and τ are one knob (gradient ~ λ/τ), the
**effective weight on ④ is 4×** the baseline's, not 2× — and any ④ difference must be read
against 4×, not 2×.

**Why.** Run 6 established ④ is *neutral* at 1×. Two possibilities: the term is worthless, or it
is underweighted. `τ_cf = 0.05` is also the value `config/default.yaml` had **explicitly
rejected as saturating** — this run tests that rejection. The tell was written down before
launch: `cf/loss − cf/chance` near −0.02 means the change did nothing, −1.09 or beyond means τ
overshot.

**Result.**

| | `abl_cf_only` (1×) | **this run (4×)** |
|---|---|---|
| mean ProcessBench F1 | 0.2599 | **0.2611** |
| gsm8k / math / olymp / omni | .389/.299/.201/.151 | .383/.296/.206/.160 |
| math clean / leaked | .233 / .357 | .238 / .348 |
| phase-1 val F1 | 0.5634 | **0.5637** |

**Two findings, and the second is the one that matters.**

1. **The saturation rejection was wrong.** `cf/loss − cf/chance` held between −0.36 and −0.49
   from step 750 on — far from the −1.09 overshoot line — and stayed responsive at ~−4.0 per
   unit. τ_cf = 0.05 does not saturate.
2. **4× the weight bought no geometry.** Over steps 750–990 this run's CF positive/negative
   separation averaged **0.0970** against `abl_cf_only`'s **0.1237** — the 1× baseline is
   *ahead*. An early advantage at steps 250–500 closed and then reversed. **④'s own loss got
   sharper while the representation got no better separated.** Neither knob is a lever at these
   magnitudes.

**Also shipped here, because it was needed.** The run died at step 990 when the Modal client
cancelled the call, and `train.py --resume` was written in response — replaying `build_scheduler`
over the *full* 1,464-step cosine (step 751 reopened on 4.539822e-06, the value logged at 750 to
full recorded precision) and seeding the data position off `run.seed` alone so the resume
consumed exactly the micro-batches step 750 never saw. AdamW's moments are **not** restored —
`betas[1] = 0.95` bounds that to a ~14-step half-life against 714 remaining steps, and the
measured transient is nil (`nce/loss − nce/chance` −0.0139 before the kill, −0.0134 after). It is
recorded in the run's own RESULT.md anyway, because the comparison run has no such discontinuity
and *a difference between two runs being compared is not the reader's to rediscover.*

---

## Part II — changing the approach: TMD → QRL (2026-08-25)

**This is not a baseline run. It is a change of foundation.** Everything up to here builds the
ruler out of TMD-style fixed-weight losses. From here the method is rebuilt on **QRL
(Quasimetric RL, Wang & Isola, ICLR 2023)** instead, with the project's own counterfactual loss
carried across and re-expressed in QRL's own language as a **constraint with its own Lagrange
multiplier**. The PQM row (run 7) is the baseline; this is the method changing direction.

**Why change direction at all.** After ten runs the picture was: the geometry sits at **0.26
mean F1**; ⑦ is actively harmful; ④ is neutral at 1× and neutral at 4×; the NCE masks bind and
change nothing; the temperature that starves ① is the one that scores best; and a plain
value-head baseline under identical conditions scores slightly higher. Two structural complaints
had also been standing in the file since the beginning and were never resolved:

1. **The ruler decays.** ③ `L_T` is supposed to hold every good step at `−log γ = 0.693`.
   `backup/delta_mean` drifted to ~0.49 on **both** early runs, and raising ζ was shown by
   direct gradient accounting to be unable to fix it (ζ 0.05 → 0.1 moves ③ from 5th place in the
   gradient ranking to 5th place).
2. **The weights were never designed to be additive.** `λ_NCE`, `λ_I`, `ζ`, `λ_CF`, `λ_step`,
   `λ_good`, `λ_term` — seven fixed scalars, none validated, whose relative gradient magnitudes
   were only ever characterised by simulation on free latents.

**QRL removes the choice.** The objective becomes
*"maximise distances everywhere"*, and everything that must stay small becomes a **constraint
with its own Lagrange multiplier**, trained by gradient ascent on the same scalar the primal
descends:

```
L = L_push  +  λ_local · V_local  +  λ_cf · V_cf          (+ λ_path · V_path from run 12)

L_push  = mean over the S×C grid of  softplus_β(offset − d(x,y))     "push everything apart"
V_local = mean relu(d(s_i, s_{i+1}) − step_cost)²  − ε²              "adjacent steps cost 1"
V_cf    = a star around ψ(s_i) binding meaning-preserving rewrites   "a reword is the same point"
```

The ruler is then held by a **dual variable** instead of a hand-chosen λ — and the multiplier is
itself a readable diagnostic while it holds: **λ climbing while its violation does not fall means
the constraint cannot be satisfied.** For ④ that reading is worth stating on its own, because
nothing else in the project has it: **λ_cf is a data-quality detector** — if the counterfactual
corpus contains "meaning-preserving" rewrites that are not, the constraint is unsatisfiable and
says so.

Everything else is held matched by construction (same config object, same parquet, same code
paths): data, selection SHA, batch shape, backbone, LoRA, heads, goal sampler, optimizer,
schedule, phase 2, eval, τ discipline. Three declared divergences: the objective itself (that is
the experiment), the distance head (`full_mrn` → **`iqe`**, TMD's other quasimetric head,
asymmetric by construction — a user decision, and the one-line `--set distance.variant=full_mrn`
control that would separate head from objective has **never been run**), and the CF corpus,
which had grown 27,114 → 41,380 between runs and *cannot* be matched to an earlier row.

---

## Run 11 — `qrl_iqe` · QRL's constrained objective, first full run
`loj243n4` (+ phase 2 `k1pzh4i8`) · 2026-08-25→26 · 1,460 steps, **6.8 h**

**Settings.** `softplus_offset 25`, `softplus_beta 0.1`, `step_cost 1.0`, `epsilon_local 0.25`,
`epsilon_cf 0.2`, **`init_lagrange 0.01`** (QRL's own default, one shared init),
`lagrange_lr 0.01`. Not one Feynman phase-1 term is computed.

**Result — a degraded checkpoint, and a very precise diagnosis.**

| at step 1460 | |
|---|---|
| `lagrange_local` / `lagrange_cf` | 3.492 / 5.348 |
| `local_dist_mean` (the ruler, target 1.0) | 1.482 |
| `local_dist_max` | **6.721** |
| `local_over_cost_frac` | **0.770** |
| `push_dist_mean_same_traj` | **11.162** |
| same_traj ÷ cross_question, **untrained** | 0.8509 |
| same_traj ÷ cross_question, **step 1460** | **0.8759** |
| phase-2 val F1 | 0.2508 |

**What failed — three findings, each of which produced a design change.**

1. **Training made the structure slightly *worse* while multiplying the scale by 6.** That last
   pair of rows is the whole experiment: a metric whose cross-question distances grow exactly as
   fast as its same-trajectory ones is **inflating a scale, not learning structure**.
2. **The dual variable is a clock, not a controller.** AdamW on the multiplier normalises by
   running gradient magnitude, so the raw variable advances at roughly `lagrange_lr` per step
   *regardless of how large the violation is*. Measured on this run: a violation of **172.6** and
   a violation of **0.16** moved the multiplier at the same rate. Starting at 0.01 therefore cost
   **~600 steps — 41% of the run — just climbing to the λ where the ruler finally turned over**,
   with the push term unopposed for every one of them. Upstream can afford that ramp; it trains
   for 2×10⁵ steps. This run gets 1,464.
3. **The constraint controlled the mean, not the tail.** `relu(d − cost)².mean()` averages a few
   large violations away against many satisfied ones, and the triangle inequality gives only an
   *upper* bound, so a heavy tail makes it loose. A same-trajectory pair at a mean gap of 2.01,
   with adjacent transitions costing 1.418, should measure ≈ 2.85. It measured **11.162** — 3.9×
   too far, and 83% of the cross-question distance. Bucketing the run by λ proved λ alone cannot
   fix it: from λ 1–2 to λ 3–4, `local_dist_max` falls only 7.60 → 5.74, `over_cost_frac`
   plateaus at 0.80, and `same_traj` **flatlines at ~11.2**.

---

## The QRL probe series — five short runs, and one diagnosis that was wrong
2026-08-26 → 08-27 · 20–120 steps each · ~1.5 GPU-hours total

Rather than spend another 6.8 GPU-hours after run 11, its failure was attacked with **five short
probes**. They are listed individually because each carries its own decision — and because
**the first diagnosis was wrong, and the probes are what overturned it.**

**The wrong diagnosis, recorded because it cost a round trip.** The first explanation of run 11
was that `softplus_offset = 25` was ~10× miscalibrated, and it was cut to 5. **That argument
assumed the offset sets the distance scale. It does not — λ does.** At `softplus_beta = 0.1` the
sloped region of the push transform is ~40 units wide, so the per-pair gradient is 0.500 at
offset 3 and 0.900 at offset 25 — an **8× knob change for 1.8× of pressure**. P2 and P3 are the
A/B that settled it.

### Probe P1 — `qrl_iqe_off3_probe` · offset 3
`kfn8luc6` · 08-26 · **died at init**

**Change / why.** `softplus_offset` 25 → 3, testing the miscalibration theory at its extreme.

**Result.** Never produced a metric row: `push_saturated_frac` = **0.9965** against the 0.99
launch-refusal threshold.

**What failed — and it failed for the wrong reason.** It was launched with a bare
`python -m qrl_prm.train` instead of `train.sh`, so it picked up `config/default.yaml`'s
`full_mrn` default (untrained scale **6.7233**) instead of `iqe` (**2.995**). It died on the init
guard for *that*, not because offset 3 was wrong — on `iqe` it would have passed.
**Rule adopted: launch through `train.sh`; it is what supplies `distance.variant`.**

### Probe P2 — `qrl_iqe_off8_probe` · the offset alone
`3lrh895a` · 08-26 · 120 steps

**Change / why.** `softplus_offset` 25 → 8 with `init_lagrange` held at QRL's 0.01, config-diffed
so that *only* the offset, the run name and `max_steps` differ. This isolates the offset.

**Result.** `local_dist_mean` fell only to **6.768** — still 6.7× its target of 1.0 — and
`push_saturated_frac` rose 0 → **0.796** by step 120.

**What failed.** The offset moved the ruler ~26% and pushed the run toward the flat region of the
transform at the same time. It is not the knob.

### Probe P3 — `qrl_iqe_lam15_probe` · the multiplier instead
`1wbpyf2g` · 08-26 · 120 steps

**Change / why.** `softplus_offset` 5, **`init_lagrange_local` 0.01 → 1.5**. The user's own
question was the design: *"if it has to climb up anyway, why not just put it up there in the
first place?"* Two independent estimates land on 1.5 — empirical (λ = 1.329 at step 600 is where
`local_sq_dev` first fell below 1.0) and analytic (at the ε boundary the push and constraint
gradients balance at λ ≈ 1.19–1.83 depending on offset).

**Result — decisive.**

| at step 120 | offset 25, λ=0.01 | offset 8, λ=0.01 | **offset 5, λ=1.5** |
|---|---|---|---|
| `local_dist_mean` | 9.15 | 6.77 | **1.397** |
| `local_sq_dev` | 103.2 | 50.8 | **0.535** |
| `local_over_cost_frac` | 0.99 | 0.98 | **0.668** |

The offset moved the ruler ~26%; **λ moved it 9.15 → 1.40.** By step 120 this probe already beat
run 11's *final* `over_cost_frac` (0.668 vs 0.770). And starting λ high is the **safe**
direction, which is the whole argument: the constraint is one-sided (`(d − k·c)⁺`), so it costs
exactly nothing below target — a large λ cannot crush distances below the step cost, only stop
overshoot. **There is no collapse failure mode on that side**, and overshoot self-corrects while
undershoot does not.

**What failed.** Offset 5 introduced a *new* failure: `push_saturated_frac` climbed
0.091 → 0.453 over 80 steps, tracking to cross 0.9 by ~step 250 — the flat region where the
objective stops maximising anything. **Reverted to offset 25.** What the offset actually sets is
the **dynamic range available for separation**, and 5 is too little of it. A headroom table
settled it: at offset 25 the equilibrium sits 11.5 *below* the offset with the gradient still
0.76; at 5 and 8 the equilibrium is *past* it.

### Probe P4 — `qrl_iqe_path_probe` · the merged path constraint
`0lcrduzl` · 08-27 · 20 steps

**Change / why.** Run 11's third finding was that the constraint controls the mean, not the tail.
The proposed fix: extend the k = 1 local constraint to **every observed sub-path** — for any
`i < j` the trace *is* a path of `j−i` steps, so `d(s_i,s_j) ≤ (j−i)·c` is a true statement about
the ground-truth quasimetric. Exact arithmetic makes k = 1 and k > 1 redundant; soft, mean-based
enforcement makes them load-bearing. Merged into **one** multiplier `λ_path`, on the argument
that pair count cancels (each state joins `2N/S` pairs and the term divides by `N`).

Before launch the `k` derivation was cross-checked by **13 independent checks** that never read
`state_step` or `state_traj` at all — they walk `row_src → row_dst` counting edges — including
recomputing the violation in plain Python (`+3.901676` vs `+3.901677`) and confirming that the
reversed direction (`+4.379549`) and a flat target (`+7.114177`) both give different numbers.
Four became permanent tests.

**Result — it went the wrong way.** All four launch checks passed: the plumbing was right, the
calibration was not.

| `local_dist_mean`, target 1.0 | step 1 | step 10 | step 20 |
|---|---|---|---|
| **`0lcrduzl`** merged | 2.263 | 2.299 | **3.009 ↑** |
| `1wbpyf2g` local-only (P3) | 2.263 | 1.798 | **1.390 ↓** |
| `loj243n4` run 11 | 2.263 | 2.633 | 5.281 |

`over_cost_frac_k1` hit **1.000** — every adjacent pair over budget. And **λ was not the
variable**: `lagrange_path` ran 1.508 → 1.647 against P3's 1.508 → 1.632. The multiplier did the
same thing; the environment got harder.

**What failed, measured on the identical seed-42 batch at step 1:**

| pair set | pairs | violating | Σ dev² | mean |
|---|---|---|---|---|
| k = 1 | 489 | 486 (99.4%) | 969.4 | **1.982** |
| k ≥ 2 | 3,119 | 418 (13.4%) | 226.3 | 0.073 |
| pooled | 3,608 | 904 (25.1%) | 1,195.7 | 0.331 |

**The mean diluted k = 1 by 7.38×** — the rows carrying 81% of the violation received 13.5% of
the weight — and the dilution factor is *batch-dependent* (7.38 / 6.53 / 3.12 at steps 1/10/20),
so the effective λ swung with trajectory-length composition. The merge would have needed
λ ≈ 18–26, and the clock (run 11, finding 2) makes that unreachable: at 0.01/step, 1.5 → 18 is
~1,650 steps against a 1,464-step budget.

**And the "pair count cancels" argument was wrong in the way that mattered.** It is right at
**equilibrium** and wrong in the **transient** — and the transient is what the run spends its
budget in. Because the constraint is one-sided, slack long-gap pairs contribute exactly 0 to the
numerator while still sitting in the denominator.

A third finding fell out of the same probe: `cf_violation` reached **7.411** by step 20 against
run 11's 5.924 on the identical batch — 25% higher and rising 1.6× faster — because the new
pos↔neg push term is the direct negation of the CF constraint. `init_lagrange_cf` was raised
0.01 → 3.0 as a result.

### Probe P5 — `qrl_iqe_split_probe` · split back apart
`8lq2e46s` · 08-27 · 20 steps

**Change / why.** P4's merge undone: k = 1 and 2 ≤ k ≤ 3 become **disjoint** pair sets under
**separate** multipliers (a test asserts they partition the forward pairs), so the wide set
cannot dilute the ruler. `lagrange_params` goes 2 → **3**. The objection to two multipliers
dissolves precisely *because* the sets are now disjoint.

**Result — the gate, written in advance, and what it read.**

| at step 20 | required | measured |
|---|---|---|
| `qrl/local_dist_mean` | falling, ≤ ~1.8 | **1.340** ✓ |
| `qrl/local_over_cost_frac` | < 1.0 and falling | ✓ |
| `qrl/path_ratio_mean` | — | 0.619 |
| all three λ | rising cleanly from their inits | ✓ |
| `lagrange_params` | 3 | ✓ |
| `qrl/push_saturated_frac` | still ~0 | ✓ |

Cleared for the full launch.

**What it did not settle.** `λ_path = 3.0` has **no empirical anchor** — k ≥ 2 has never been run
standalone, and 3.0 is an analytic value times a factor measured on k = 1. `λ_local`'s 5.0 and
`λ_cf`'s 3.0 both trace to measured settling values; `λ_path`'s does not.

---

## Run 12 — `qrl_iqe_split` · three constraints, three multipliers
`9ubkxy7x` (+ phase 2 `wwbglo8k`) · 2026-08-27→28 · 1,460 steps, **6.7 h**

**Changes.** `L = L_push + w_neg·L_neg + w_pn·L_pn + λ_local·V_local + λ_path·V_path + λ_cf·V_cf`
— `lagrange_params: 3`.

| knob | was | is | why |
|---|---|---|---|
| `init_lagrange_local` | 0.01 | **5.0** | run 11 settled at 3.479 with the ruler still at 1.482 — a lower bound that had not converged, i.e. 1.9× the analytic 1.83 |
| `init_lagrange_path` | — | **3.0** | analytic λ_eq at k=2,3 is 1.81/1.80 × the same 1.9 |
| `init_lagrange_cf` | 0.01 | **3.0** | run 11's λ_cf ended at **5.336**; climbing 0.01 → 5.336 at 0.01/step is **~994 steps, 68% of the run**. The constraint converged *barely*, and the clock ate two thirds of the budget doing it |
| `path_max_gap` | — | **3** | with an absolute deviation, `dev² ≈ 4k²`, so uncapped the k≥2 mean is steered by the longest sub-paths — run 11's "mean, not tail" failure relocated. The cap also concentrates the violation 20× |
| **new term** `L_pn` | — | pos↔neg push | the direct negation of the CF constraint: ε_cf says a meaning-*preserving* rewrite is the same point; **nothing said a meaning-*breaking* rewrite is a different one.** Without it a paraphrase-sized perturbation can move a step across the verdict boundary |

**Result — the constraints behaved, and the eval collapsed anyway.**

| | step 20 | step 750 | step 1460 |
|---|---|---|---|
| `local_dist_mean` (target 1.0) | 1.341 | 0.796 | **0.926** ✓ |
| `local_over_cost_frac` | 0.810 | 0.307 | **0.267** ✓ |
| `cf_violation` | 0.698 | 0.0218 | **0.0185** ✓ |
| `push_saturated_frac` | 0 | 0.056 | **0.078** ✓ |
| `pos_neg_push_gap` | −0.016 | 8.94 | **10.47** ✓ |
| `lagrange_local` / `path` / `cf` | 5.17 / 3.13 / 3.17 | 7.13 / 4.20 / 7.36 | 9.02 / 5.75 / 9.95 |
| `local_violation` / `path_violation` | | | 1.188 / 0.981 — **still positive** |

| eval | |
|---|---|
| phase-1 val F1 | **0.2405** (τ sensitivity **0.0**) |
| ProcessBench gsm8k / math / olymp / omni | 0.2175 / 0.0549 / 0.0738 / 0.0845 |
| **mean** | **0.1077** |

**What failed.**

- **The engineering worked and the objective did not.** Every gate the probe set was cleared:
  the ruler converged to 0.926 against a target of 1.0, the over-cost fraction fell 0.81 → 0.27,
  the CF constraint converged to 0.0185, the push never saturated, and the new pos↔neg gap opened
  from ~0 to 10.5. This is by a wide margin the best-behaved *internal* geometry the project has
  produced. **It scores 0.108.**
- **The multipliers never settled.** All three climbed monotonically for the whole run while
  `local_violation` (1.188) and `path_violation` (0.981) stayed positive. By the project's own
  reading — *λ climbing while its violation does not fall means the constraint cannot be
  satisfied* — the local and path constraints did not converge; only CF did.
- **The decision statistic went flat, and inverted.** `probe03/gap` — the good-step vs bad-step
  Δ separation, and the project's own best predictor of ProcessBench F1 — is **+0.511** on
  run 11 and **−0.566** here: sign-flipped. The τ calibration curve is nearly flat (F1 0.187 →
  0.196 across the whole τ range, **sensitivity exactly 0.0**), which is what a degenerate delta
  distribution looks like — there is no threshold that separates anything, so the fitted τ is
  arbitrary. *(This is a reading of the two runs' probe metrics side by side, not a documented
  root cause — it has not been confirmed by a follow-up diagnostic.)*
- **The honest summary: replacing the objective made a well-conditioned metric that is not a
  reward model.** A constrained objective that maximises distance everywhere and pins observed
  transitions to their step count produces exactly that geometry — and "distance along the
  observed trace" is *position*, which is the diagnosis Interlude A reached back on 2026-08-03
  by a completely different route.

---

## Where it stands

**The result.** Best mean ProcessBench F1 is **0.2611** (`cf_lam2_tau005`), against **0.2682** for
a PQM baseline under identical conditions. Phase-1 val F1 — the geometry with the goal head
bypassed — is stable at **0.55–0.56** across every Feynman run from 2026-08-04 onward, i.e. the
representation is *not* what moved between them.

**The standing diagnosis, unchanged since 2026-08-03 and re-derived twice since.**
*The geometry resolves POSITION along a trajectory, not CORRECTNESS of a step.* Three
independent measurements say so: replacing a sample's goal with **another question's goal**
retains **103%** of the signal; detection AUC is at or near chance on three of four subsets; and
the QRL run, whose objective is explicitly *about* position, produced the cleanest internal
geometry and the worst eval.

**What the numbers say the next lever is.** Perfect false-positive control is worth **+0.038**
mean F1. Perfect localisation at current flag rates is worth **+0.281**. Seven to one — and
almost all the work above bought from the +0.038 column.

**Open, and named.**
1. **`nce_tau4`'s gsm8k anomaly** — `acc_correct` 0.995 → 0.093 on one subset at a healthy global
   τ and an unchanged phase-1 val F1. Never diagnosed.
2. **⑦ `L_term` has still never met a live ①.** It was built to counteract a force that was muted
   22.6× in every run it appeared in.
3. **The QRL head control was never run.** `--set distance.variant=full_mrn` is one line, and
   without it the QRL rows differ from everything before them in *both* the objective and the
   distance head, so a gap is a gap against the pair.
4. **The oldest un-run free measurement in the file**: `goal_gate.py --untrained`. The
   0.618 → 0.276 recall collapse that motivated ⑦ has never been matched on sample size.
5. **Best-of-N against CRM** — the harness is built, vendored byte-identical down to the answer
   grader, with six aggregators and three baselines, and its expectations were written down
   before the fact. **It has never been run.**

**Two habits that this log is really a record of.** First, every run has a written-in-advance
decision rule — *"landing at ~25.9 says ⑦ was the whole story; landing above says ④ is a real
win"* — so the result could not be constructed after the fact. Second, when a run's question
could be answered from files already on disk, it was: **Interlude A is four analysis passes
that cost no GPU time**, and each one redirected the next run. The five QRL probes are the same
habit applied to hardware — ~1.5 GPU-hours spent to avoid a wrong 6.8-hour run.

**And the recurring bug, which is the most transferable finding here.** Five separate times
(B11, B12, B13, B16, B17) the thing that broke a run or hid a result was **a guard that failed
toward "healthy"** — an additive slack on a multiplicative quantity, a scheduler check that
fired on every run that finished, a collision check that fired on the documented workflow, an
attach-rate counter that was tautological by construction. The rule that came out of it:
**print the ratio, not the verdict; scope a guard rather than renaming around it; and write the
artifact before the diagnostic that might destroy it.**
