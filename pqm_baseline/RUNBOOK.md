# PQM baseline — run it, start to finish

All commands from the repo root (`feynman-prm/`), on the training box. `python -m`, not a
script in `scripts/` — see `README.md` §3 for why.

---

## 0. Preflight (2 minutes, do not skip)

```bash
pytest -m "not gpu"                                   # expect 517+71 passed
```

```bash
# THE MATCHED-DATA CHECK. If this SHA is not the one the Feynman runs used, the comparison
# is not matched and nothing else in this file matters.
python -c "import json;d=json.load(open('data/processed/selection.json'));print(d['selection_sha_train'], d['n_train_questions'], d['n_val_questions'])"
```

`n_train_questions` must read **34650** and `n_val_questions` **2000**. If the parquet is
missing or at another selection:

```bash
python scripts/prepare_data.py                        # writes sequences.parquet + the npz cache
```

---

## 1. Probe (~1 min on GPU)

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

| block | what must be true |
|---|---|
| `launch/data` | `optimizer_steps` ~1460, `questions` 34640, identical to the diff above |
| `launch/model` | `trainable_tensors == {"lora": 392, "value_head": 2}` — nothing else |
| `launch/init_values` | `pqm/loss == pqm/loss_at_zero_rewards` (asserted to 1e-4), `reward_std == 0.0` |
| `launch/memory_probe` | `peak_vram_gb` below the card. Expect **less** than Feynman's ~11.5 GiB — no ψ/φ MLPs, no R×C matrix, no CF variants |

---

## 2. Train (~3–6 h, in tmux)

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

## 3. Eval (~30–45 min)

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

## 4. The paper table

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
