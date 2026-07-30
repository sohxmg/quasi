# SETUP.md — getting Feynman-PRM running on the GPU box

For a co-worker sitting down at the RTX 5070 Ti with a fresh checkout. Follow it top to
bottom; every step has a check with the output you should see.

If something breaks, go to §7 **before** debugging — most of what can go wrong here has gone
wrong once already and is written down.

> **Read `IMPLEMENTATION.md` before you change anything.** This file is how to run the code;
> that one is what the code does and why. `CLAUDE.md` is the locked spec.

---

## 1. What you need

| | |
|---|---|
| GPU | RTX 5070 Ti, 16 GB, **Blackwell / sm_120** |
| Driver | **≥ 570** (Blackwell needs it) |
| Python | **3.12** |
| torch | a **cu128-or-newer** build — sm_120 kernels first appear in cu128, and cu128 wheels start at torch 2.7. Newer toolkits are fine: **`torch 2.13.0+cu130` is what the box runs and the GPU suite passes on it (2026-07-27)** |
| Disk | ~15 GB free: ~3.1 GB model, ~2 GB HuggingFace dataset cache, ~200 MB processed parquet, ~100 MB per checkpoint |
| Network | HuggingFace access for `Qwen/Qwen2.5-Math-1.5B-Instruct`, `trl-lib/math_shepherd`, `Qwen/ProcessBench` (all public — no token needed) |
| tmux | **mandatory** for every GPU run |

Check the driver and card first — if this is wrong, nothing later will make sense:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

```
name, driver_version, memory.total [MiB]
NVIDIA GeForce RTX 5070 Ti, 5xx.xx, 16376 MiB
```

---

## 2. One-time setup

```bash
conda create -n feynman python=3.12 -y
conda activate feynman

# torch FIRST, from the cu128 index. Installing requirements.txt first can pull a default
# CPU/cu121 wheel that silently does not support sm_120.
pip install torch --index-url https://download.pytorch.org/whl/cu128

cd /path/to/feynman-prm
pip install -r requirements.txt
pip install -e .

# wandb ships in requirements.txt and log.wandb defaults to true, so log in ONCE here.
# Skipping this makes every training launch fail immediately -- deliberately, see §6.
wandb login
```

**Check the package resolves** (`pip install -e .` finds `feynman_prm/`; the directory name and
the import name must match or nothing below runs):

```bash
python -c "import feynman_prm; print(feynman_prm.__file__)"
```

**Check it:**

```bash
python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('device', torch.cuda.get_device_name(0), 'sm_%d%d' % torch.cuda.get_device_capability())
print('bf16 matmul ok:', bool((torch.randn(8,8,device='cuda',dtype=torch.bfloat16) @
                               torch.randn(8,8,device='cuda',dtype=torch.bfloat16)).isfinite().all()))
"
```

```
torch 2.13.0+cu130 cuda 13.0            # measured 2026-07-27; anything >= cu128 is fine
device NVIDIA GeForce RTX 5070 Ti sm_120
bf16 matmul ok: True
```

If `sm_120` does not appear, or `torch.version.cuda` is **below 12.8**, stop and fix that — see
§7.1. Compare it as a version, not a string: cu130 is newer than cu128, not different from it.

**Every shell that touches the GPU:**

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The `scripts/*.sh` wrappers set it themselves; set it by hand if you run
`python -m feynman_prm.…` directly. Consider putting it in the conda env's activate hook:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' \
  > "$CONDA_PREFIX/etc/conda/activate.d/feynman.sh"
```

**Do not install** `trl`, `verl`, `deepspeed`, `vllm`, `flash-attn`, `bitsandbytes` or
`openrlhf`. None is used, and CRM's pinned versions of them cannot run on this card at all
(`torch==2.6.0` has no cu128 build). If you need to run CRM itself as a baseline, give it its
own conda env — do not merge them.

---

## 3. Verify the install

### 3.1 CPU suite — no GPU, no model, no download

```bash
pytest tests/ -m "not gpu" -q
```

```
197 passed, 2 skipped
```

The skips are `tests/test_gpu.py` and `tests/test_good_loss_ablation.py`, both deselected by
their markers. **This suite must be green before anything else** — it pins the index
conventions, and an off-by-one there is invisible in every loss curve later.

### 3.2 GPU suite — the first thing that touches CUDA and the real model

```bash
pytest tests/test_gpu.py -m gpu -v -s
```

Downloads Qwen2.5-Math-1.5B-Instruct (~3.1 GB) on first run, then takes a few minutes. `-s`
matters: several tests **print measurements** that replace estimates in the docs —

* `test_environment_is_blackwell_with_a_cu128_or_newer_torch` → your device/sm/torch/arch line
* `test_bf16_backbone_with_fp32_heads_and_fp32_distances` → `logit_std`, which must be `> 0`
  (`≈ 0` is bug B10a, the fp32 distance cast not taking effect)
* `test_memory_probe_at_the_full_batch_shape` → **peak VRAM and a projected h/epoch**
* `test_h_s0_from_a_prompt_only_forward_equals_the_full_sequence` → the number phase 2's
  whole cost model rests on
* `test_l_good_reads_the_same_delta_the_probe_reports` → ⑥ `L_good`'s `Δ` panel beside
  diagnostic #14's, which must agree because they are the same tensor (§7.12)

**Nothing in this file trains.** Every test is forwards and single backwards on the shared
model fixture. The two ⑥ tests that *do* take optimizer steps live in
`tests/test_good_loss_ablation.py` behind the `ablation` marker and are **not run here**:

```bash
pytest tests/test_good_loss_ablation.py -m ablation -v -s     # OPT-IN. ~2 min, loads the
                                                              # backbone twice, 12 steps each
```

**Do not run that before the phase-1 run.** It answers "does `L_good` flatten the good/bad
separation", and the run itself answers that better and for free — `probe03/gap` and
`probe14/delta_good_of_correct/frac_above_natural` are logged every `log_every` steps over
real batches, from step 1. The ablation is for diagnosing a run that went wrong.

If `test_environment_*` fails, fix that first — everything after it will fail for the wrong
reason.

### 3.3 Smoke test

```bash
bash scripts/smoke_test.sh
```

Every loss finite and backward on random hiddens; same seed → identical losses.

---

## 4. Prepare the data (once, ~30–60 min, CPU-bound)

```bash
python scripts/prepare_data.py
```

It loads `trl-lib/math_shepherd`, groups by question, splits by question, tokenises every
selected trajectory with the real Qwen tokenizer, mines the branch points, and writes
`data/processed/`.

**What to check in the output:**

**All values below are MEASURED from the first real run, 2026-07-27.** Three rows of this table
used to hold pre-run estimates and all three were wrong — see the note underneath, because two of
them looked alarming and were not.

| line | expected | if it differs |
|---|---|---|
| `"questions": 45989` | exact | the dataset changed under us — every number in `CLAUDE.md` §4 is then suspect |
| `"trainable_questions": 40247` | exact | same |
| `"fraction_correct": 0.3658` | ±0.001 | same |
| `"all_labels_equals_last_label": 0.99999` | ±0.00001, with `"last_label_disagreements": 4` | more than a handful of disagreements means the label semantics moved. **A `0.0` here means you are on an old build** where this stat was a boolean |
| `"recovery_fraction": 0.0148` | ±0.0005 | §16.15's label noise changed |
| `REAL tokenised lengths` | median 248, p99 866, `over_max_len_fraction` 0.0049 | a different tokenizer or `prompt_format`; §4.6 |
| `disagreeing_branch_points` | ~24,000 over the selected 34,650 questions (30,344 on full train; the ~15,900 figure was counted at the old `n_questions: 23000`) | §4.4 / §8.3 |
| `"trajectories_per_question_on_disk": 9.17` | ~9.18 | **this is the on-disk rate, not the epoch rate.** The sampler takes 4.33/question (§8.1's caps), so one epoch is ~150k sequences and ~1,460 steps at `n_questions: 34650`. Reading 9.18 as the epoch rate is what produced the 106-step run (§11.1) |
| `selection_sha_train` | **changes with `n_questions`.** `e81beeb8d527…` was seed 42 / `n_questions: 23000`; the shipped config is now **34650** and the SHA moves with it — **write down whatever this run prints** (R10). If you see the old SHA, `prepare_data.py` did not re-run and the training set is still the 23,000-question one |

**What the first run corrected, so nobody re-panics:** this table asked for
`all_labels_equals_last_label: 1.0` "exactly", flagged as *"CRM's core assumption broke"*, and the
run printed `0.0`. **Nothing broke.** The stat was a boolean `all()` over 422,407 rows and CLAUDE
§4.2's "100.0%" was a rounded 99.999% — **4 rows** disagree, all `[T…T, F, T…T]`. Every loss
consumes `all(labels)` (`trajectory_is_correct`), so those four are incorrect with `z` at their
first False, exactly as §16.15 says. The other two: token lengths ran ~20% above the chars/3.4
estimate (`over_max_len` 0.489% against a claimed 0.05%, so 1,121 sequences dropped rather than
~200 — small, one-sided, and `max_len` stays 1024 unless the memory probe is re-run), and the
`~63,000` disagreeing branch points was extrapolated from a 4,000-question sample and is really
~30,000 on full train. None of the three blocks training, and none of them required re-running
`prepare_data.py` — nothing reads `selection.json` but a human.

Mismatches print with `!!` and do not abort; that is deliberate — you decide whether the
dataset moved or the config did.

Output files:

```
data/processed/
  sequences.parquet              pre-tokenised train+val rows (~200 MB)
  selection.json                 SHAs, counts, the real length distribution
  train_questions.txt            the ACTUAL selected ids, not a seed that implies them
  val_questions.txt
  branch_points.jsonl            §8.3's held-out diagnostic set (NOT L_CF data)
  processbench_math_leak.json    per-sample leak flags for the math subset (locked #5)
```

Offline box, or ProcessBench not cached? `--skip-leak-check` skips the last file; eval then
reports math F1 unsplit and says so.

Dev shortcut while you are wiring things up: `--limit 20000` reads only the first 20k rows.
Do not use a limited run for anything real — the counts and the split will not match.

---

## 5. Train

### 5.1 Short probe first — always

```bash
tmux new -s feynman
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/train.sh --max-steps 20
```

**Read all four launch blocks before walking away.** They print in this order:

1. `[launch/data]` — `optimizer_steps` (expect **~1,460** for the shipped config; ~971 means
   `prepare_data.py` was not re-run after `n_questions` rose to 34,650) and
   `warmup_steps` (~27), plus `questions_per_batch_mean` (~12.9),
   `distinct_z_per_batch_mean` (~28), `step_pairs_per_batch_mean` (~64) and
   `padding_fraction` (~0.10 with `group_by_length`).
   **If `optimizer_steps` prints ~106, stop** — the `n_questions`/`grad_accum` regression is
   back (§11.1). The run below 300 steps aborts by itself unless you pass `--max-steps`.
2. `[launch/model]` — the trainable tensor buckets. It must be exactly `{lora, psi, phi}`.
   A `goal_head` or anything under `other` is a hard failure.
3. `[launch/memory_probe]` — the **longest batch of the epoch, run first on purpose**, with
   its peak VRAM. An OOM shows up here in 30 seconds instead of three hours in.
4. `[launch/init_values]` — actual vs expected at step 0:
   * `nce ≈ log(R) ≈ 5.85` — pinned there with `logit_std ≈ 0` is bug B10a
   * `step ≈ 2.1`, **not** 1.6094 — `ln 5 = 1.6094` needs `Δ_{z+1} = 0`, which is a fixture
     identity, not the model at init (`ψ_0` is the prompt-only state and starts away from
     mid-solution states, so `Δ` starts negative). **Measured 2.0796 at `step_delta_mean`
     −0.44.** The level moves with the batch's z mix, so check the RELATION, not the number:
     `step == log(1 + exp(m − Δ))` with `m = 1.386`. See CLAUDE.md §7.6.7
   * `backup` prints `nan` whenever `linear_branch_fraction` is between 0.05 and 0.95 — delta is
     bimodal at init and no mean-based prediction is valid there. Read
     `linear_branch_fraction` falling 1.0 → ~0 over ~100 steps instead (measured: 0.52 at step
     0, 0.003 by step 20)
   * `backup ≈ γ − mean(Dist)` — see the note in `IMPLEMENTATION.md` §6; the sign at step 1
     is not diagnostic, the *trend* is
   * `cf = 0.0` — deferred by decision

Then per-step console lines: `step 10  L=… nce=… inv=… bkp=… step=… gap=… Q=… z=…`.

### 5.2 The go/no-go gate — run it before spending hours

```bash
python scripts/goal_gate.py --checkpoint runs/phase1/final
```

```
ratio = 0.xxx  ->  PROCEED            # < 0.3
ratio = 0.9xx  ->  STOP AND REDESIGN  # -> 1
```

`within / across` terminal spread (§10.1). If it is near 1, correct endings do not cluster by
question and **the goal head cannot work no matter how well it is trained** — the fallback is
the goal-free asymmetry score (§9.4), *not* a reference goal (that is a skyline, §5.1). The
script exits non-zero on a failing gate, so it is safe to chain.

You can run this on the 20-step probe checkpoint; that is the point of it being cheap.

### 5.3 Full phase 1

```bash
tmux new -s feynman
bash scripts/train.sh                 # ~1,460 optimizer steps, estimated 3–6 h
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t feynman`.

**What to watch, in priority order:**

| | |
|---|---|
| `probe14/delta_good_of_correct/frac_above_natural` | **the single best predictor of ProcessBench F1, and read THIS rather than the mean.** `L_step` never sees a correct trajectory, so a positive Δ tail here is F1 leaking. On the first full run the mean was `+0.240` — which looks like a small offset — while this read **0.34**, τ fitted to 2.39 and F1 capped at 0.456 (§7.12). `p90`/`p99` are logged beside it |
| `invariance/residual_diagonal` | **the `λ_good` guard.** ⑥ pays for its Δ reduction out of ② (measured 0.098 → 0.260 at λ=1, §7.12). Under **0.15 by step ~200** and not rising after; otherwise halve `losses.lambda_good` to 0.5. Visible from step 1, so a wrong λ costs ~100 steps to catch instead of the whole run |
| `good/above_target_fraction`, `good/delta_mean`, `good/lambda_effective` | ⑥ `L_good`'s own view, logged even at `λ_good = 0`. `good/margin` must print **negative**; `lambda_effective` ramps 0 → 1 over the first 100 steps (`good_loss.warmup_steps`) |
| `probe02/delta_good_mean` vs `−0.693` | the ruler. The old project's was 108× off and nobody noticed |
| `probe03/gap` | good-vs-bad Δ separation. Collapsing → φ is ignoring its action |
| `nce/logit_std` | `≈ 0` with the loss stuck at `log(R)` → bug B10a |
| `backup/loss` | **expected negative.** Watch plateau and NaN, not sign |
| `step/distinct_z` | ~28. Near 21 → the sampler is on the old caps; exactly 96 pairs → caps applied as quotas |

Live summary at any time (no matplotlib needed):

```bash
python scripts/plot_metrics.py runs/phase1/metrics.jsonl
python scripts/plot_metrics.py runs/phase1/metrics.jsonl --csv /tmp/run.csv   # for your own plots
```

Checkpoints land in `runs/phase1/step{N}/` every 250 steps and `runs/phase1/final/`, ~100 MB
each (adapter + heads + resolved config + tokenizer), not 3.1 GB — mid-run saves are cheap on
purpose.

### 5.4 Phase 2 — the goal head

```bash
python -m feynman_prm.train_goal_head --checkpoint runs/phase1/final
```

Caches `h_{s_0}` (one short forward per **question**) and `ψ(s_T)` for correct trajectories,
re-runs the gate on the trained representations, then fits the head alone on the cached
vectors — minutes of fitting after ~40 min of caching (estimate). Writes
`runs/phase1/phase2/final/`.

### 5.5 Eval

```bash
bash scripts/eval_processbench.sh runs/phase1/phase2/final
```

Calibrates τ on **held-out Math-Shepherd validation questions** (never on ProcessBench), then
scores all four subsets. Expect ~5 minutes for all 3,400 samples.

The printed τ is itself a check: it should land near **0.347** (the midpoint the ruler and the
margin imply). τ ≈ 0 means the second step of margin never landed — drop `margin_steps` to 1.0
rather than fighting it.

Results go to `runs/.../processbench.json`, including the math subset split **587 leaked vs
413 clean** and the skyline on the joinable samples. **The skyline is labelled and is never a
reported result** — knowing only the gold answer already solves half the metric exactly
(§5.1).

---

## 6. Changing the config

`config/default.yaml` is strict-parsed: **an unknown or misspelled key is a hard error**, on
purpose (a silently ignored config value is old bug B4). Override without editing the file:

```bash
bash scripts/train.sh --set discount=0.7 --set run.name=phase1_g07
```

Three rules that are not negotiable:

1. **Never set `m` or `t` by hand.** They are derived from `discount`
   (`m = margin_steps·(−log γ)`, `t = clip_t_steps·(−log γ)`), and the only sanctioned
   `discount` values are **0.5** (shipped) and **0.7** (documented fallback).
2. **If it OOMs, lower `sampling.max_padded_tokens` FIRST — not the sequence budget.**
   A batch costs `len(batch) x max_len`, so the token cap binds only on the ~17% of batches
   that are actually too big and leaves the rest (and their `L_NCE` negative pool) alone.
   Cutting `sequences_per_micro_batch` shrinks *every* batch and halves the pool and `Q`
   with it — measured in CLAUDE.md §8.1.2. `grad_accum` does not need to move for the cap.
   If you do cut the sequence budget, lower it **and raise `train.grad_accum` together**:
   raising `grad_accum` alone cuts the optimizer-step count, which is exactly the regression
   that produced a 106-step run. Re-run `python scripts/batch_report.py` either way — it
   prints the exact pool and `Q` you would be trading.
3. **If GPU time is short, cut `data.n_questions`, not `grad_accum`.** Coverage falls
   linearly; step count does not.

A config change invalidates checkpoints. Start a fresh `run.name`.

`log.wandb` now defaults to **true**, and `wandb` is in `requirements.txt`. If it is missing or
you are not logged in, the run **fails at launch** rather than warning to stderr and training on
— see §2. Turn it off with `--set log.wandb=false`. JSONL + console always work either way and
are the source of truth.

Only per-step metrics reach wandb: every loss term and every §10 probe. The launch blocks, the
memory probe, the init values and the gate results go through `logger.event()`, which writes
`events.jsonl` and the console **only**. Read those in tmux; they will not be in the dashboard.

---

## 7. Troubleshooting

### 7.1 Environment

| symptom | cause | fix |
|---|---|---|
| `CUDA error: no kernel image is available for execution` | torch built without sm_120 (a cu121/CPU wheel, or torch < 2.7) | reinstall from the cu128 index (or any newer one); `torch.version.cuda` must be **≥ 12.8**, and `torch.cuda.get_arch_list()` should mention `120` |
| `pip install flash-attn` fails | there is **no** cp312/cu128/sm_120 wheel and a source build needs `nvcc` | don't. `sdpa` is already a fused flash-family kernel and attention is only ~2% of FLOPs at our lengths |
| `CUBLAS_STATUS_NOT_SUPPORTED` inside SDPA `o_proj` on the **first** forward | Blackwell + bf16 path | cast the model to fp32 after load, before `.cuda()`. **Do not** try `torch.backends.cuda.preferred_blas_library("cublaslt")` — it gets past `o_proj` and then throws `invalid resource handle` at a layernorm |
| `ImportError: No module named feynman_prm` | `pip install -e .` not run | run it (the `scripts/*.py` also self-bootstrap `sys.path`) |
| HuggingFace download stalls / 429 | rate limit | retry; set `HF_HOME=/big/disk/hf` if `~` is small |

### 7.2 Training

| symptom | probe | cause | fix |
|---|---|---|---|
| `optimizer_steps` prints ~106 | launch | the `n_questions` / `grad_accum` regression | sequences per question is `min(4,k_c)+min(3,k_i)` = 4.33, **not** 9.18 |
| `nce` stuck at `log(R)`, `logit_std ≈ 0` | #10 | bug B10a — fp32 cast not effective | confirm the cast is inside the distance; clean restart |
| `nce` stuck at `log(R)`, `logit_std > 0`, pos ≈ neg | #10, #1 | geometry collapse / duplicate goals | check `probe01/distinct_goal_ratio` |
| `probe03/gap` → 0, or `Δ_{z+1}` will not go positive | #3, #14 | φ ignoring its action input | confirm `action_invariance: diagonal`; consider `action_pool: attention` |
| positive Δ tail on good steps of **correct** trajectories | #14 | false positives `L_T` alone is not suppressing | the **only** sanctioned trigger for a §7.10 pairing expansion |
| `backup/linear_branch_fraction` climbing | #15 | `t` too tight | raise `clip_t_steps`; **do not** change `discount` |
| `backup/div_cross_question` diverging | #13 | unreachable cross-question goals eating the loss | lower `goal_scope_ratio` |
| `backup/div_same_question` stops converging | #13 | `L_step` winning too hard | `margin_steps` → 1.0 |
| NaN a few hundred steps in | — | someone "simplified" the double `torch.where` in `temporal.py` | put it back: `where` evaluates the discarded branch in backward |
| OOM at the very last step | — | AdamW's `foreach` path allocates a large fp32 transient | `foreach=False` (already set; check nobody changed it) |
| `CUDA driver version is insufficient` at optimizer construction | — | `FusedAdam` JIT-compiles against the system CUDA | plain `torch.optim.AdamW` (already used) |
| constant LR | — | no scheduler | `train.py` asserts the LR moved at the end of the run |
| within-trajectory distance spread → 0 | #7 | states being squashed | `sampling.nce_mask_same_traj: true` |

### 7.3 Eval

| symptom | cause | fix |
|---|---|---|
| `no goal head: phase 2 must run before ProcessBench` | eval pointed at a phase-1 checkpoint | point it at `.../phase2/final`. A reference goal is a skyline, not a substitute |
| `over the 1% budget` assertion | too many over-length samples | raise `eval.max_len`. **Never truncate** — it drops trailing separators and shortens `T` |
| math F1 reported unsplit | `processbench_math_leak.json` missing | re-run `prepare_data.py` without `--skip-leak-check` |
| τ lands at ~0 | the second step of margin never landed | `margin_steps` → 1.0; do not hand-tune τ |

---

## 8. Housekeeping

```
runs/<run.name>/
  config.resolved.yaml    exactly what ran
  metrics.jsonl           every loss and probe, per logged step
  events.jsonl            launch asserts, memory probe, init values, gate results
  step{N}/ final/         adapter/ heads.pt config.yaml tokenizer/     (~100 MB each)
  phase2/final/           the same, plus goal_head in heads.pt
```

Want a plain model directory (adapter merged into the weights)?

```bash
python scripts/export_merged.py --checkpoint runs/phase1/final --out /tmp/merged
```

It copies `heads.pt` alongside — **a merged backbone without the heads is a useless artifact**,
which is the exact failure the old project shipped once.

**Ground truth about what has and has not been run** (updated 2026-07-27):

* **CPU suite** — green, 164 tests.
* **`tests/test_gpu.py`** — has now run on the box, 26/26 green on `torch 2.13.0+cu130`. Its four
  initial failures were **wrong assertions, not wrong code**: a `startswith("12.8")` cu-version
  check, two absolute bf16 tolerances on hidden states (0.05 where one bf16 rounding at Qwen's
  massive activations is already 0.75), and `L_step = ln 5` at init, which is a `Δ = 0` fixture
  identity rather than the model's value (§7.6.7).
* **`prepare_data.py`** — has now run. Every §4 count reproduced exactly; three lines in §4
  turned out to be estimates rather than counts and are corrected in place (see §4's table
  above).
* **Not yet run:** phase-1 training past the smoke probe, the §10.1 gate on trained
  representations, phase 2, and ProcessBench eval.

When a printed number contradicts a doc, **count it on the full split before assuming the data
moved** — that is how all three of the above resolved.
