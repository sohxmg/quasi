# PQM baseline — run it, start to finish

All commands from the **clone root** (`~/quasi` — the dir holding `config/`, `scripts/`,
`feynman_prm/`, `pqm_baseline/`), on the training box. `python -m`, not a script in `scripts/`
— see `README.md` §3 for why.

---

## 0. Setup — fresh box (~10 min, once)

Written for a **rented A100 (sm_80)**. Kratos does not need this section.

```bash
# --- system. jq is NOT optional: every check below is piped through it ---
sudo apt-get update && sudo apt-get install -y git tmux jq python3.12-venv
```

```bash
# --- repo. The clone IS the repo root -- there is no nested feynman-prm/ dir inside it ---
git clone https://github.com/xtechsouthie/quasi.git && cd quasi
```

> Every command in this file runs from **`~/quasi`** (the clone root: the dir holding `config/`,
> `scripts/`, `feynman_prm/`, `pqm_baseline/`). `feynman_prm/` is the *package*, not the root --
> `cd`-ing into it breaks every relative path (`data/processed/...`, `config/default.yaml`).

```bash
# --- env. Python >=3.12 (pyproject), and the venv lives in the repo ---
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt          # ~5 min, torch is most of it
```

> **Ignore `requirements.txt`'s "install from the cu128 index" note here.** That is a *Kratos*
> constraint — sm_120 (Blackwell) has no other wheel. An A100 is **sm_80** and the default PyPI
> torch wheel works. Do not go hunting for a cu128 index; you will only slow the install down.

> **Do NOT `pip install -e .`.** `pyproject.toml`'s `packages.find` includes `feynman_prm*` and
> nothing else, so an editable install leaves `pqm_baseline` **not importable** and every
> `python -m pqm_baseline.*` in this file fails. Nothing needs installing — run from the repo
> root (`feynman-prm/`) and cwd puts both packages on `sys.path`. This is the same reason
> §1 onward says "all commands from the repo root".

```bash
# --- credentials ---
export HF_TOKEN=hf_...                   # Math-Shepherd + Qwen2.5-Math-1.5B are both gated pulls
wandb login                              # `log.wandb` defaults to TRUE
```

> **A missing `wandb` is a hard error, not a warning** — `RunLogger` raises before the first step
> (`feynman_prm/diagnostics/logging.py:80`). `config/default.yaml:515`'s comment claiming it
> "only warns and trains on regardless" is **stale**; the behaviour was deliberately changed
> because three hours of GPU time with an empty dashboard is worse than a crash at second zero.
> If you do not want wandb, say so explicitly and it is honoured — `metrics.jsonl` is written
> either way:
> ```bash
> bash pqm_baseline/train.sh --set log.wandb=false
> ```

```bash
# --- verify the card before spending money on it ---
nvidia-smi
python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

`is_bf16_supported()` must print **True** — the whole stack is bf16, and a card without it
(anything pre-Ampere, e.g. T4/sm_75) cannot run this at all.

**Disk:** budget **~10 GB** — Qwen weights ~3.1 GB, the Math-Shepherd cache ~183 MB, the
`sequences.parquet` + `.npz` cache, and ~85 MB per run under `runs/`.

---

## 1. Preflight (2 minutes, do not skip)

```bash
pytest -m "not gpu"                                   # expect 517+71 passed
```

```bash
# THE MATCHED-DATA CHECK. If this SHA is not the one the Feynman runs used, the comparison
# is not matched and nothing else in this file matters.
python -c "import json;d=json.load(open('data/processed/selection.json'));print(d['selection_sha_train'], d['n_train_questions'], d['n_val_questions'])"
```

`n_train_questions` must read **34650** and `n_val_questions` **2000**.

**On a fresh box `data/processed/` does not exist yet** and this raises `FileNotFoundError:
data/processed/selection.json` — that is expected, not a fault. Build it (~10 min: pulls the
183 MB Math-Shepherd cache from HF, so `HF_TOKEN` must be exported), then re-run the check
above:

```bash
python scripts/prepare_data.py     # -> sequences.parquet, selection.json, the .npz cache,
                                   #    branch_points.jsonl, train/val_questions.txt
```

Same command if the SHA is present but at **another selection** — the parquet is rebuilt, not
patched.

---

## 2. Probe (~1 min on GPU)

```bash
tmux new -s pqm
bash pqm_baseline/train.sh --max-steps 20             # -> runs/pqm_zeta4_probe/  (disposable)
```

Read four things off `runs/pqm_zeta4_probe/events.jsonl` before launching for real:

```bash
jq -c 'select(.event|startswith("launch/"))' runs/pqm_zeta4_probe/events.jsonl
```

```bash
# launch/data must be IDENTICAL to a Feynman run's -- this is the matched-data proof, free:
diff <(jq -cS 'select(.event=="launch/data")|del(.elapsed_s)' runs/pqm_zeta4_probe/events.jsonl) \
     <(jq -cS 'select(.event=="launch/data")|del(.elapsed_s)' runs/phase1_nce_temp_relu2/events.jsonl)
```

> **`runs/` is gitignored, so on a fresh clone the right-hand side does not exist** and this diff
> cannot be run there. Either `scp` the one file over (`runs/phase1_nce_temp_relu2/events.jsonl`,
> 2 KB) from the box that holds it, or check the twelve values by hand against this — the
> **verified** `launch/data` of `phase1_nce_temp_relu2`:
>
> ```json
> {"questions": 34640, "sequences_per_question": 4.312, "optimizer_steps": 1464,
>  "warmup_steps": 44, "n_batches": 2928, "sequences_total": 149351,
>  "sequences_per_batch_mean": 51.0079, "questions_per_batch_mean": 11.8306,
>  "distinct_z_per_batch_mean": 25.8525, "step_pairs_per_batch_mean": 58.4358,
>  "padding_fraction": 0.2628, "max_padded_tokens": 32768}
> ```

| block | what must be true |
|---|---|
| `launch/data` | `optimizer_steps` ~1460, `questions` 34640, identical to the diff above |
| `launch/model` | `trainable_tensors == {"lora": 392, "value_head": 2}` — nothing else |
| `launch/init_values` | `pqm/loss == pqm/loss_at_zero_rewards` (asserted to 1e-4), `reward_std == 0.0` |
| `launch/memory_probe` | `peak_vram_gb` below the card, and **below Feynman's 12.128 GiB** — no ψ/φ MLPs, no R×C matrix, no CF variants. Measured PQM: **11.958 GiB**. Only −0.17 GiB, and that is correct: the probe batch is 32×1024, where the backbone activations dominate and the deleted heads are small. Comparable because both land on the *same* longest batch (`batch_index` 1391, `sequences` 32, `max_length` 1024) |

---

## 3. Train (~3–6 h, in tmux)

```bash
bash pqm_baseline/train.sh                            # -> runs/pqm_zeta4/, ~1,460 steps
```

Watch two curves — if `pqm/reward_gap` is flat by step ~300 the head or the lr is wrong and the
rest of the run is wasted:

```bash
tail -f runs/pqm_zeta4/metrics.jsonl | jq -c '{step,gap:."pqm/reward_gap",neg:."pqm/frac_neg_below_neg_zeta",loss:."pqm/loss",chance:."pqm/loss_at_zero_rewards"}'
```

`pqm/reward_gap` must **open**; `pqm/frac_neg_below_neg_zeta` must **rise**. Read `pqm/loss`
against `pqm/loss_at_zero_rewards`, never raw (§10 #19).

---

## 4. Eval (~30–45 min)

```bash
python -m pqm_baseline.eval_processbench --checkpoint runs/pqm_zeta4/final
```

Writes `processbench.json`, `deltas.npz` and `val_f1.json` into that directory. Sanity: fitted τ
near `ζ/2 = 2.0` on the negated scale, τ-sensitivity small, over-length under the 1% budget on
all four subsets (the run **asserts** the last one).

> τ is fitted on the held-out 2,000 val questions, never on ProcessBench (§9.2). Do **not** read
> `scripts/report_processbench.py`'s τ verdict for this checkpoint — it is calibrated against
> Feynman's ruler (0.347) and is meaningless here. The eval prints its own verdict against ζ/2.

---

## 5. The paper table

```bash
python -m pqm_baseline.report --pqm runs/pqm_zeta4/final \
    --feynman runs/abl_cf_only/phase2/final \
    --feynman runs/phase1_nce_temp_relu2/phase2/final
```

The Feynman side is already on disk and needs no re-run. Report the math leak split (587 leaked
/ 413 clean) and τ for both rows — `report.py` prints both.

---

## If the number looks bad

```bash
# ζ=4 was chosen on a 7B FULL finetune; this is a 1.5B LoRA. One line, and it is the FIRST
# thing to try before drawing any conclusion.
bash pqm_baseline/train.sh --set pqm.zeta=8 --set run.name=pqm_zeta8
```

```bash
# PQM's own head init instead of zero-init (loses the exact launch check; watch pqm/reward_max,
# fp32 exp(r+zeta) overflows above r ~= 84)
bash pqm_baseline/train.sh --set pqm.head_init=default --set run.name=pqm_defaultinit
```

**`pqm.zeta` is NOT `losses.zeta`.** The latter is Feynman's ③ `L_T` backup weight (0.05/0.1)
and nothing in `pqm_baseline/` reads it; `train.py` refuses to launch if it is set to something
at PQM's scale.

**One run is one draw.** Neither row has a seed replicate — quote the gap, not a ranking, unless
it is large. And the PQM row is *our re-implementation under matched conditions*, not PQM's
published numbers; say so wherever it appears.
