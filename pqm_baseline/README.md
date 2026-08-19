# PQM baseline, matched to Feynman-PRM

**PQM (Process Q-value Model, Li & Li, ICLR 2025) — its head and its objective, trained under
Feynman-PRM's exact conditions.** `Process_Q_Model/` (the authors' released code) stays
untouched as the vendored reference, the same treatment `../tmd-release/` and `../CRM/` get;
the loss is ported here with line citations, never imported.

**The row this produces is our re-implementation under matched conditions, not PQM's published
numbers.** Say so wherever it appears. PQM's own paper reports Best-of-N on a 7B full finetune
and never reports ProcessBench.

---

## 1. What "matched" means — the table for the methods paragraph

Every row is identical between the two runs, and identical **by construction** (the same config
file and the same `sequences.parquet`), not by re-derivation:

| | both runs |
|---|---|
| dataset | `trl-lib/math_shepherd`, `data/processed/sequences.parquet` |
| selection | `n_questions: 34650`, `n_val_questions: 2000`, seed 42, same `selection_sha_train` |
| sequences/epoch | ~150k (`min(4,k_c)+min(3,k_i)` caps, §8.1) |
| epochs | 1 |
| batching | `sequences_per_micro_batch 56`, `max_padded_tokens 32768`, `group_by_length true`, `grad_accum 2` → ~1,460 optimizer steps |
| tokenisation | `build_sequence`, `"\n"` single-token separator, `s_0` after the prompt, `max_len 1024`, drop-never-truncate |
| backbone | `Qwen/Qwen2.5-Math-1.5B-Instruct`, LoRA r16 α16 dropout 0 on the 7 projections, grad checkpointing, bf16, sdpa |
| optimizer | torch AdamW, `foreach=False`, betas (0.9, 0.95), wd 0, `grad_clip 1.0` |
| lr | LoRA 9e-6, fresh head 3e-4, cosine, `warmup_ratio 0.03` |
| eval | ProcessBench × 4 subsets, `"Step N: "` prefixes added, `eval.max_len 2048`, `localisation_rule: first_crossing`, math leak split reported |
| τ | fitted on the held-out 2,000 val questions, **never** on ProcessBench (§9.2) |

**PQM changes exactly two things**: the head (Feynman's 3×512 MLP ψ/φ + quasimetric distance →
one `Linear(1536, 1)` value head) and the objective (the seven-term loss set → PQM's Q-ranking
loss).

## 2. Deliberate divergences from PQM's published recipe — state these in the paper

None is optional; each exists to make the comparison possible, and each is in the direction of
matching Feynman-PRM rather than favouring either side.

| PQM's paper/code | here | why |
|---|---|---|
| deepseek-math-7b-base, full finetune | Qwen2.5-Math-1.5B-Instruct + LoRA r16 | the comparison is against a 1.5B LoRA PRM on one 16 GB card |
| 2 epochs, lr 1e-6/2e-6, 8×A100 + DeepSpeed ZeRO-3 | 1 epoch, Feynman's schedule, 1 GPU | matched training budget |
| whole Math-Shepherd (~422k rows) | the 34,650-question selection (~150k sequences) | matched data |
| `[PRM]` special token, `resize_token_embeddings` | `"\n"` separator, read at `state_pos[1..T]` | identical tokenisation to Feynman. Under LoRA a freshly added token's embedding row never trains, and a separate vocab means a separate parquet, which breaks the identical-data guarantee. CRM also separates steps with `"\n"` |
| length-bucketed batches (64/24/8 by length) | Feynman's question-grouped sampler | identical batch stream. PQM's loss is per-sequence, so batch *composition* does not enter its objective at all — only gradient noise |
| labels = the raw per-step Math-Shepherd labels | labels derived from `z` (`label_k = z == -1 or k < z`) | the parquet stores `z`, not the label vector, and this is exactly how Feynman's ⑤/⑥ treat the same trajectories (§16.15). It monotonises the **1.48%** of trajectories with a False→True recovery. Recorded, pinned in a test, and reversible — see §6 |
| value head init: plain `nn.Linear` | zero-init (`pqm.head_init: zero`, default) | gives an exactly predictable init loss for the §18 launch check, and removes an `exp(r+ζ)` overflow risk on Qwen hiddens whose massive-activation channels run O(100). `pqm.head_init: default` reproduces PQM's init; the launch log prints the reward distribution either way |

## 3. Layout

```
pqm_baseline/
  config.py              PQMConfig: zeta, loss_type, head_dropout, head_init, label_source
  config/pqm.yaml        the PQM-only knobs (strict-parsed, unknown key = hard error)
  model.py               PQMValueModel: the shared backbone + PQM's ValueHead
  loss.py                pqm_ranking_loss (ported verbatim), the padded builder, diagnostics
  train.py               the training loop (mirrors feynman_prm/train.py's launch discipline)
  eval_processbench.py   tau on val + the 4 subsets; writes processbench.json + deltas.npz
  report.py              the side-by-side table
  train.sh               tmux/env wrapper
tests/test_pqm.py        71 CPU tests -- in the existing tests/ tree, which is NOT grep-scanned
```

**Why a sibling package and not a module inside `feynman_prm/`.**
`tests/test_grep_invariants.py::test_no_value_head_anywhere` scans exactly
`feynman_prm/**/*.py` + `scripts/*.py`. A value head is the one thing that guard exists to keep
out of the *method*, and it is the defining feature of this *baseline*. Living outside the
scanned tree keeps the guard honest instead of renaming a head to dodge it (§14 B13: scope a
guard, never rename around it). For the same reason there is **no `scripts/*.py` entry point** —
entry is `python -m pqm_baseline.train`.

**Exactly one edit to existing code**, additive and default-preserving:
`feynman_prm/utils/checkpoint.py` gives `head_state_dict()` and `save_checkpoint()` an optional
`prefixes: tuple[str, ...] = HEAD_PREFIXES`, so this trainer saves through the same asserted
path (§14's LoRA trap 3: the stock PEFT save writes the adapter and silently drops the head)
instead of duplicating it. The string `"value_head."` is passed by the caller and never appears
in `feynman_prm/`. Nothing under `feynman_prm/eval/` is touched — the PQM eval driver imports
its *pure* parts and writes its own scoring loop.

## 4. The loss, and its analytic init value

`pqm_ranking_loss` is `Process_Q_Model/train_main.py:61-78` verbatim, including the vestigial
`.flip(dims=[-1])`, the `1e-5` denominator epsilon and the prepended virtual slot whose label is
`has_neg`. A pinned test compares it against a copy of the authors' function; do not clean it up.

With `head_init: zero` every reward is exactly 0 at step 0, so the loss is **closed-form** from
`(n_pos, n_neg, has_neg)` per trajectory:

```
L0 = mean_q [ ( 1{has_neg}·log(1 + n_neg·e^ζ + ε)
              + Σ_{m=0..n_pos-1} log(2 + m + n_neg·e^ζ + ε) ) / (1{has_neg} + n_pos) ],  ε = 1e-5
```

`train.py` computes this on the first micro-batch and asserts `|L_actual − L0| < 1e-4`.

> **`2 + m`, not `1 + m`.** The positive slot's denominator carries both its own `cur = 1` and
> the cumsum's leading `exp(0)` prepended at `train_main.py:66`. The planning note wrote
> `1 + j`; it is off by one unit of `e^0`, worst at low `n_neg` — i.e. on the trajectories that
> carry the clean half of F1. Derived, and pinned in
> `test_pqm.py::test_analytic_init_value_is_exact`, because §7.4.3/§7.6.7's lesson is that an
> *assumed* init value is how two regressions got through.

**The zero-fill of the padded reward slots is load-bearing, not tidiness.**
`torch.where(labels == 1, rewards.exp(), 0)` evaluates `exp()` on *every* slot including
padding; an uninitialised slot overflows to `inf`, and `where`'s backward multiplies `0 * inf`
into a NaN no forward value would reveal. Both directions are pinned in the tests.

## 5. Reading a run (§10)

| key | read it as |
|---|---|
| `pqm/reward_gap` | **the signal.** Eval thresholds exactly this separation; flat by step ~300 means the head or the lr is wrong and the rest of the run is wasted |
| `pqm/frac_pos_above_0`, `pqm/frac_neg_below_neg_zeta` | the loss's two absolute anchors. These are what make a *global* τ meaningful across questions |
| `pqm/loss` vs `pqm/loss_at_zero_rewards` | the loss against its own analytic chance level (§10 #19's contrastive-loss-vs-chance rule) |
| `pqm/reward_std` | `≈ 0` is a dead head (B10a's analogue) |
| `pqm/reward_min`, `pqm/reward_max` | the overflow guard: fp32 `exp(r + ζ)` overflows above `r ≈ 84` |
| `pqm/good_steps_below_tau_natural` | false-positive leak on correct trajectories — the §7.6.6 / diagnostic #14 analogue, and the single best predictor of the clean half of F1 |

**The natural τ for PQM is `+ζ/2` on the negated (eval) scale**, `−ζ/2` in reward units: the
ranking loss anchors positives above the virtual `e^0 = 1` slot and negatives below `−ζ`, so
their midpoint is `−ζ/2 = −2` at ζ=4. Reported as a **check, not a constraint**, in the same
spirit as §9.2's 0.347.

> `scripts/report_processbench.py`'s τ verdict line is calibrated against Feynman's ruler
> (`natural_tau = (m − (−log γ))/2 = 0.347`) and is **meaningless for a PQM checkpoint**.
> `pqm_baseline.eval_processbench` prints its own verdict against `ζ/2`.

> **The val-F1 comparison must not be made against `scripts/val_f1.py`'s 0.5615.** That script
> substitutes a *real terminal* for the goal (the §9.5 skyline substitution) and its own
> docstring calls it a ceiling, not a result. The comparable Feynman number is the goal-head val
> F1 recorded as `calibration/f1` in each run's `phase2/final/processbench.json` — **0.5900**
> for `abl_cf_only`, **0.5872** for `phase1_nce_temp_relu2`. `report.py` reads that field.

## 6. Not done, on purpose

`label_source: raw` (PQM-faithful per-step labels) needs a `labels` column in
`sequences.parquet` — `prepare_data.py` + `sequence_cache` + `SequenceRow`, ~20 lines, and a
re-run of `prepare_data.py`. `config.py` **refuses** the value rather than silently ignoring it.
Not done now: it changes a shared artifact — the same parquet both runs read — for a 1.48%
effect.

Best-of-N was offered and declined. If a reviewer asks: PQM's canonical BoN score `min_i r_i` is
exactly `neg_max_delta` on the negated scores, so `feynman_prm/eval/bon.py`'s aggregator
machinery already covers it with no new code beyond a scoring loop.

## 7. Risks, stated up front

- **A single run is a single draw.** Neither row has a seed replicate. Say "under matched
  conditions" and quote the gap, not a ranking, unless it is large.
- **PQM is a Best-of-N method being scored on ProcessBench.** Its paper never reports
  ProcessBench, so the localisation rule ("the first step whose reward falls below τ") is *our*
  protocol applied to its scores. It is the same protocol Feynman is scored under, which is what
  makes it fair — but it must be described as ours.
- **LoRA at 1.5B is not where PQM was tuned.** ζ=4 was chosen on a 7B full finetune. A ζ sweep
  was offered and declined; `--set pqm.zeta=…` is a one-line re-run and it is the first thing to
  try before drawing a conclusion.
