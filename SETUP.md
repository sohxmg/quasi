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
| torch | a **cu128** build — required for sm_120, and cu128 wheels start at torch 2.7 |
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
torch 2.8.x+cu128 cuda 12.8
device NVIDIA GeForce RTX 5070 Ti sm_120
bf16 matmul ok: True
```

If `sm_120` does not appear, or `torch.version.cuda` is not `12.8`, stop and fix that — see
§7.1.

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
161 passed, 1 skipped
```

The skip is `tests/test_gpu.py`, which is deselected by the marker. **This suite must be green
before anything else** — it pins the index conventions, and an off-by-one there is invisible
in every loss curve later.

### 3.2 GPU suite — the first thing that touches CUDA and the real model

```bash
pytest tests/test_gpu.py -m gpu -v -s
```

Downloads Qwen2.5-Math-1.5B-Instruct (~3.1 GB) on first run, then takes a few minutes. `-s`
matters: several tests **print measurements** that replace estimates in the docs —

* `test_environment_is_blackwell_with_a_cu128_torch` → your device/sm/torch line
* `test_bf16_backbone_with_fp32_heads_and_fp32_distances` → `logit_std`, which must be `> 0`
  (`≈ 0` is bug B10a, the fp32 distance cast not taking effect)
* `test_memory_probe_at_the_full_batch_shape` → **peak VRAM and a projected h/epoch**
* `test_h_s0_from_a_prompt_only_forward_equals_the_full_sequence` → the number phase 2's
  whole cost model rests on

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

| line | expected | if it differs |
|---|---|---|
| `"questions": 45989` | exact | the dataset changed under us — every number in `CLAUDE.md` §4 is then suspect |
| `"trainable_questions": 40247` | exact | same |
| `"fraction_correct": 0.366` | ±0.001 | same |
| `"all_labels_equals_last_label": 1.0` | exactly 1.0 | CRM's core assumption broke |
| `REAL tokenised lengths` | p99 near 702, `over_max_len_fraction` ~0.0005 | this **replaces** §4.6's chars/3.4 estimate — record it |
| `disagreeing_branch_points` | ~63,000 (over the selected train questions) | §8.3 |
| `selection_sha_train` | any hex | **write it down** — it identifies the exact question set (R10) |

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

1. `[launch/data]` — `optimizer_steps` (expect **~889** for the shipped config) and
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
   * `step ≈ 1.6094` — this one is nearly exact; if it is far off, the margin or the `z`
     indexing is wrong
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
bash scripts/train.sh                 # ~889 optimizer steps, estimated 2–4 h
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t feynman`.

**What to watch, in priority order:**

| | |
|---|---|
| `probe14/delta_good_of_correct/positive_fraction` | **the single best predictor of ProcessBench F1.** `L_step` never sees a correct trajectory, so a positive Δ tail here is F1 leaking with no loss training against it |
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
2. **If it OOMs, lower `sampling.sequences_per_micro_batch` AND raise `train.grad_accum`
   together.** Raising `grad_accum` alone cuts the optimizer-step count — that is exactly the
   regression that produced a 106-step run.
3. **If GPU time is short, cut `data.n_questions`, not `grad_accum`.** Coverage falls
   linearly; step count does not.

A config change invalidates checkpoints. Start a fresh `run.name`.

`log.wandb: true` enables wandb if it is installed; JSONL + console always work and are the
source of truth.

---

## 7. Troubleshooting

### 7.1 Environment

| symptom | cause | fix |
|---|---|---|
| `CUDA error: no kernel image is available for execution` | torch built without sm_120 (a cu121/CPU wheel, or torch < 2.7) | reinstall from the cu128 index; `torch.version.cuda` must read `12.8` |
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

**Ground truth about what has and has not been run:** the CPU suite is green, and the phase-1
inner loop has been dry-run against a stub backbone. `tests/test_gpu.py`, `prepare_data.py`,
and every path that touches `transformers`/`peft`/`datasets` **have never executed** — the
machine this was written on has no GPU and none of those packages. Treat §3.2's printed
numbers as the first real measurements, and tell the author if any of them contradict the
estimates in `IMPLEMENTATION.md`.
