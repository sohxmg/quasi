# QRL + CF — run it on Kratos, start to finish

Every command runs from the **repo root** (`feynman-prm/` — the dir holding `config/`,
`scripts/`, `feynman_prm/`, `qrl_prm/`). Not from inside `feynman_prm/`: that is the
*package*, and `cd`-ing into it breaks every relative path (`config/default.yaml`,
`data/processed/…`).

> **`pip install -e .` does NOT make `qrl_prm` importable.** `pyproject.toml`'s
> `packages.find` is `include = ["feynman_prm*"]`, so an editable install ships the package
> and nothing else — exactly as for `pqm_baseline/`. Nothing needs installing: running from
> the repo root puts `qrl_prm` on `sys.path` via cwd. This is the same reason every command
> below says "from the repo root".

> **tmux is mandatory for every GPU command here** (§13). A dropped ssh session kills a
> 4-hour run otherwise.

---

## 0. Preflight — five minutes, and it saves the run

```bash
cd ~/feynman-prm            # or wherever the clone is; it is the dir holding config/
conda activate feynman      # whatever the env is called on this box
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

```bash
# --- the card ---
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
#   NVIDIA GeForce RTX 5070 Ti, 5xx.xx, 16376 MiB, ~0 MiB
#   memory.used must be ~0. Something else on the card is the #1 cause of an OOM that
#   the memory probe passed.

# --- torch sees sm_120 ---
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability())"
#   torch.version.cuda must be >= 12.8 (compare as a VERSION: cu130 is newer, not different)

# --- both packages import from the repo root ---
python -c "import feynman_prm, qrl_prm; print(feynman_prm.__file__); print(qrl_prm.__file__)"
```

```bash
# --- the data is the SAME parquet the baselines read. If this SHA differs from the one
#     abl_cf_only / cf_lam2_tau005 trained on, the comparison is not matched and nothing
#     else in this file matters ---
python -c "import json;print(json.load(open('data/processed/selection.json'))['selection_sha_train'])"
ls -la data/processed/sequences.parquet
```

```bash
# --- the CF corpus. It GROWS between runs; the three filenames never change ---
wc -l data/cf_train/*.jsonl          # 41,380 total on the 2026-08-25 snapshot
head -30 data/cf_train/MANIFEST.md
# If data/cf/ on this box is newer than data/cf_train/, refresh the snapshot FIRST:
#   cp data/cf/cf70k.jsonl data/cf/cf70k_gm.jsonl data/cf/cf70k_oai.jsonl data/cf_train/
# and do NOT move the originals out of data/cf/ (cf_exclude_generated.py globs them).
```

```bash
# --- the CPU suite, including the 54 qrl_prm tests. Green before anything touches the GPU ---
pytest tests/ -q -m "not gpu and not ablation"
#   575 passed, 2 skipped
pytest tests/test_qrl.py -q
#   54 passed
```

```bash
# --- wandb: log.wandb defaults to TRUE and a missing login FAILS the launch (deliberate) ---
wandb login          # once per box
#   or add `--set log.wandb=false` to every train command below
```

---

## 1. The probe — 20 steps, always, before the real run

```bash
tmux new -s qrl
cd ~/feynman-prm && conda activate feynman

bash qrl_prm/train.sh --max-steps 20
```

Writes to **`runs/qrl_iqe_probe/`** (the `_probe` suffix is automatic and the directory is
disposable — `rm -rf runs/qrl_iqe_probe` when done).

### Read all five launch blocks before launching for real

```bash
jq -c 'select(.event|startswith("launch/"))' runs/qrl_iqe_probe/events.jsonl
```

| block | what must be true |
|---|---|
| `launch/data` | `optimizer_steps: 1464`, `warmup_steps: 44`, `questions: 34640`, `sequences_total: 149351`. **If it prints ~106, stop** — the `n_questions`/`grad_accum` regression (§11.1) |
| `launch/config` | `qrl/*` knobs are the shipped defaults; `distance/variant: iqe`; `note_head` names the deliberate divergence |
| `launch/cf_data` | `examples: ~41380`, `rows_with_prefix_hash == rows` (a mismatch aborts by design), `cap_binds: true`, `examples_dropped_question_absent` a handful. A **leaked val question is fatal at any count** and stops the launch |
| `launch/model` | `trainable_tensors` = `{lora: 392, psi: 14, phi: 0, action_pool: 0, distance: 1, …}` — **`distance: 1` is IQE's learned alpha and must be there**, and **`phi: 0` is required, not a regression**: φ is frozen under `qrl_prm/` because the arrived state is read rather than predicted, and `assert_qrl_phase1_trainable` refuses to start if it is trainable (read `note_phi` in the same event). `lagrange_params: 2`; `trainable_params: 20042753` — the baselines' 22,407,168 MINUS φ's 2,364,416, PLUS α. Do not compare this number to §4's baselines, which trained φ |
| `launch/init_values` | asserted, so it either passes or the run stops. Eyeball `push_saturated_frac` (must be well under 1.0) and `local_dist_mean` (starts far above 1.0 — correct) |
| `launch/memory_probe` | **`peak_vram_gb_with_probes` under ~15 GB.** That is the real high-water mark; `peak_vram_gb` is the training step alone |

**The matched-data proof, and it is free:**

```bash
diff <(jq -c 'select(.event=="launch/data")|del(.elapsed_s)' runs/qrl_iqe_probe/events.jsonl) \
     <(jq -c 'select(.event=="launch/data")|del(.elapsed_s)' runs/abl_cf_only/events.jsonl)
#   no output = identical batch stream. Anything else = the comparison is not matched.
```

**Estimate the wall clock off the probe** before committing 4+ hours:

```bash
jq -r 'select(.step)|"\(.step) \(.elapsed_s)"' runs/qrl_iqe_probe/metrics.jsonl | tail -2
#   seconds-per-step x 1464 / 3600 = hours. The baselines ran 3-6 h; IQE is heavier per pair
#   than full_mrn and this adds an S x C push matrix, so expect the top of that range or above.
```

### If the memory probe does not fit

In order, and only the first two keep the run matched:

```bash
bash qrl_prm/train.sh --max-steps 20 --set qrl.push_chunk_cols=32   # exact, chunks the push matrix
bash qrl_prm/train.sh --max-steps 20 --set qrl.push_chunk_cols=16
# LAST RESORT -- breaks the matched batch stream and MUST be reported:
#   --set sampling.max_padded_tokens=24576
```

`push_chunk_cols` lowers the transient peak of IQE's internals; it does not shrink the
autograd graph, so it is a lever, not a cure.

### The go/no-go gate (optional, cheap, runs on the probe checkpoint)

```bash
python scripts/goal_gate.py --checkpoint runs/qrl_iqe_probe/final
#   read gate/auc, not gate/ratio (see the script's docstring)
```

---

## 2. The full run — ~1,464 optimizer steps

```bash
rm -rf runs/qrl_iqe_probe          # optional; it is disposable

tmux new -s qrl
cd ~/feynman-prm && conda activate feynman
bash qrl_prm/train.sh
```

Detach `Ctrl-b d`, reattach `tmux attach -t qrl`. Checkpoints land in
`runs/qrl_iqe/step{250,500,…}/` and `runs/qrl_iqe/final/`, ~100 MB each.

### Watch these, in priority order

```bash
python scripts/plot_metrics.py runs/qrl_iqe/metrics.jsonl --keys \
  qrl/local_dist_mean qrl/local_violation qrl/lagrange_local \
  qrl/cf_sq_dev qrl/cf_violation qrl/cf_p95 qrl/lagrange_cf qrl/cf_active \
  qrl/push_dist_mean qrl/push_saturated_frac qrl/neg_push_gap \
  loss/total loss/push loss/local loss/cf \
  probe14/delta_good_of_correct/frac_above_natural probe03/gap
```

| curve | reading |
|---|---|
| **`qrl/local_dist_mean` → 1.0** | **THE RULER.** The direct answer to IMPLEMENTATION.md §9's decaying `backup/delta_mean`. If it does not come down toward `step_cost` by ~step 300, the objective is not doing its job |
| `qrl/lagrange_local`, `qrl/lagrange_cf` | must **rise then stabilise**. Monotone climbing for the whole run while the matching violation does not fall = the constraint cannot be satisfied. For λ_cf that is the CF corpus contradicting itself |
| `qrl/local_violation`, `qrl/cf_violation` | the **sign** is what matters: negative = satisfied |
| `qrl/cf_p95` | the tail, not the mean. Fat tail = paraphrases that could still flip a verdict |
| `qrl/push_saturated_frac` | rising to 1.0 = the push term has no gradient left; the offset is too small |
| `qrl/neg_push_gap` | should **open** (CF negatives getting further from goals than the average pair) |
| `probe14/delta_good_of_correct/frac_above_natural` | the cross-method comparable — same function every baseline logged |

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


Quick tail without the summariser:

```bash
tail -1 runs/qrl_iqe/metrics.jsonl | jq '{step, "qrl/local_dist_mean", "qrl/lagrange_local", "qrl/lagrange_cf", "qrl/cf_sq_dev", "qrl/push_saturated_frac"}'
watch -n 300 'tail -1 runs/qrl_iqe/metrics.jsonl | jq "{step, \"qrl/local_dist_mean\", \"qrl/lagrange_cf\"}"'
```

**Flat curves by step ~300 mean the rest of the run is wasted.** Kill it, do not wait.

---

## 3. Phase 2 — the goal head (UNCHANGED feynman entry point)

The checkpoint is format-identical (same `FeynmanPRM`, same head names), so this is the same
command every baseline ran.

```bash
tmux new -s qrl2
cd ~/feynman-prm && conda activate feynman
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m feynman_prm.train_goal_head --checkpoint runs/qrl_iqe/final
```

Writes `runs/qrl_iqe/phase2/final/`. ~40–75 min of caching (`h_{s_0}` per question and
`ψ(s_T)` per correct trajectory on a frozen backbone), then seconds of fitting.

Refit at a different epoch count without re-caching:

```bash
python -m feynman_prm.train_goal_head --checkpoint runs/qrl_iqe/final \
    --from-cache --overwrite --set goal_head.epochs=13
```

> Phase 2 reads its config from the **phase-1 checkpoint's** `config.yaml`, not from
> `config/default.yaml`. Editing that file does nothing here — use `--set`.

---

## 4. Eval — ProcessBench (UNCHANGED feynman entry point)

```bash
bash scripts/eval_processbench.sh runs/qrl_iqe/phase2/final
#   equivalently: python -m feynman_prm.eval.processbench --checkpoint runs/qrl_iqe/phase2/final
```

~5 minutes for all 3,400 samples. τ is calibrated on the **2,000 held-out Math-Shepherd val
questions** (203-point sweep), never on ProcessBench. Results go to
`runs/qrl_iqe/phase2/final/processbench.json`.

> The `natural_tau = 0.347` line is `−log γ`-based and is **informational only** for a QRL
> checkpoint, whose ruler is `step_cost = 1.0`. The sweep is what decides τ.

---

## 5. The table

```bash
python -m qrl_prm.report --qrl runs/qrl_iqe/phase2/final --run-dir runs/qrl_iqe
```

Prints the four ProcessBench subsets + mean + goal-head val F1 + τ against the three shipped
rows (`abl_cf_only` 0.2599/0.5900, `cf_lam2_tau005` 0.2611/0.5954, `pqm_zeta4` 0.2682/0.5766),
the math leak split (587 leaked / 413 clean — **report both halves**), the per-baseline deltas,
and the constraint footer read off the run's last metrics line.

---

## 6. Controls and ablations — one line each, run only if the numbers ask for them

```bash
# THE HEAD CONTROL. The QRL row differs from every baseline in BOTH objective and head;
# this separates them. Run it before writing down any attribution.
bash qrl_prm/train.sh --set distance.variant=full_mrn --set run.name=qrl_mrn

# QRL-faithful: add the standard latent-dynamics term over all rows
bash qrl_prm/train.sh --set qrl.cf_encode_max_tokens=8192 --set run.name=qrl_cheap

# is the CF-negative push doing anything?
bash qrl_prm/train.sh --set qrl.cf_neg_push_weight=0 --set run.name=qrl_nonegpush

# the push offset, if push_saturated_frac ran high
bash qrl_prm/train.sh --set qrl.softplus_offset=50 --set run.name=qrl_off50
```

Each needs its own phase 2 + eval, then add it to the table:

```bash
python -m feynman_prm.train_goal_head --checkpoint runs/qrl_mrn/final
bash scripts/eval_processbench.sh runs/qrl_mrn/phase2/final
python -m qrl_prm.report --qrl runs/qrl_iqe/phase2/final --run-dir runs/qrl_iqe \
    --baseline runs/abl_cf_only/phase2/final \
    --baseline runs/cf_lam2_tau005/phase2/final \
    --baseline runs/pqm_zeta4/final \
    --baseline runs/qrl_mrn/phase2/final
```

---

## 7. When something goes wrong

| symptom | cause | do |
|---|---|---|
| `already holds checkpoint(s)` | §14 B14 guard — a real run is in that directory | use a new `--set run.name=…`, or `--overwrite` if you truly mean to discard |
| `these overrides are silently INERT` | you passed `--set losses.*`; nothing in `qrl_prm/` reads that block | use the `qrl.*` knob the message lists |
| `not one of N rows carries a prefix_hash` | pre-2026-08-15 parquet | `python scripts/prepare_data.py` |
| `CF examples sit on VAL question(s)` | corpus/selection mismatch — **fatal, and correctly so** | regenerate the CF corpus against this selection, or filter |
| `alpha_raw is not in the saved head state dict` | someone changed `HEAD_PREFIXES` | put `"distance."` back; without it phase 2 reads a different metric |
| OOM at the memory probe | IQE's push matrix | `--set qrl.push_chunk_cols=32` (see §1) |
| `push_saturated_frac ... at INIT` assert | `softplus_offset` below the untrained distance scale | raise it: `--set qrl.softplus_offset=50` |
| `the LR never moved` at the very end | bug B6 | **the final checkpoint was written first and is intact** — it is a diagnostic, not a loss |
| run died mid-way | — | there is **no `--resume`** in `qrl_prm/train.py` (unlike `feynman_prm/train.py`). Restart from step 0 under a fresh `run.name`, or add the flag by mirroring the feynman block |

---

## 8. Copy-paste: the whole happy path

```bash
cd ~/feynman-prm && conda activate feynman
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pytest tests/ -q -m "not gpu and not ablation"

tmux new -s qrl
bash qrl_prm/train.sh --max-steps 20
jq -c 'select(.event|startswith("launch/"))' runs/qrl_iqe_probe/events.jsonl
diff <(jq -c 'select(.event=="launch/data")|del(.elapsed_s)' runs/qrl_iqe_probe/events.jsonl) \
     <(jq -c 'select(.event=="launch/data")|del(.elapsed_s)' runs/abl_cf_only/events.jsonl)

rm -rf runs/qrl_iqe_probe
bash qrl_prm/train.sh                                              # ~1,464 steps

python -m feynman_prm.train_goal_head --checkpoint runs/qrl_iqe/final
bash scripts/eval_processbench.sh runs/qrl_iqe/phase2/final
python -m qrl_prm.report --qrl runs/qrl_iqe/phase2/final --run-dir runs/qrl_iqe
```
