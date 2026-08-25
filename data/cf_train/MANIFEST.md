# `data/cf_train/` — the (4) `L_CF` corpus, snapshot 2026-08-25

**This is a SNAPSHOT of the kept-example files in `data/cf/`, not the campaign directory.**
`data/cf/` is ~440 MB and only ~82 MB of it is data a training run can read; the rest is
`.raw` / `.rejected` / `.discarded` / `.responses` artifacts with different schemas
(§7.5.13's "sixteen files" warning). This directory is the ~82 MB, and it is the only data
directory that ships to the GPU box.

`config/default.yaml`'s `data.cf_glob` points here. It is a comma-separated **list**, never a
glob — see §7.5.13. The list is unchanged by this refresh: the three filenames are the same,
only their contents grew.

## Contents

| file | examples | pos/ex | neg/ex | bytes | sha256[:16] |
|---|---:|---:|---:|---:|---|
| `cf70k.jsonl`     (bharatcode) | 1,560 | 2.99 | 5.69 | 3,231,239 | `290f932a7d437d1c` |
| `cf70k_gm.jsonl`  (gemini)     | 35,983 | 2.99 | 5.94 | 71,395,707 | `5cd23432894f8f82` |
| `cf70k_oai.jsonl` (openai)     | 3,837 | 2.91 | 5.94 | 7,742,917 | `acd6b1343c4ca9bb` |
| **total** | **41,380** | | | **82,369,863** | |

- **368,875 rewrites** (123,406 positive / 245,469 negative) over 41,380 anchors — **410,255
  vectors per epoch** counting the anchors themselves, i.e. **9.91 per example**, the same
  9.91-ish figure §7.5.2 quotes at every size this corpus has been.
- **13,012 distinct questions**, all inside the 34,650-question train selection, none in the
  2,000-question val holdout.
- **0 duplicate `(question, step_index)`** across the three campaigns, and 0 anchor overlap on
  any pair of them — the slice arithmetic still holds after the two nights of generation that
  followed the 2026-08-22 snapshot.
- **Every anchor traces back to the 70,070-anchor `cf_items*.jsonl` pool** (checked
  2026-08-25: 0 off-pool anchors in any of the three files), which is what carries the train-
  selection / val-holdout property above forward from the 2026-08-15 snapshot. The launch-time
  guard in `data/cf_attach.select_cf_examples_for_train` is still the authority (§16 B16);
  this is a provenance check, not a substitute for it.
- **Schema clean on all 41,380**: every record has `question` / `steps` / `step_index` /
  `positive_rewrites` / `negative_rewrites`, `step_index` in range, and no empty or blank
  rewrite list. 0 malformed lines.
- **Parity held on all three** — token-id AUC 0.479 (bharatcode) / 0.595 (gemini) / 0.591
  (openai), every one inside the 0.15 flag. Regex fallback tokenizer, as everywhere in §7.5.
- Validator floors are **still mixed, and still only bharatcode is the odd one out**:
  `cf70k_gm` and `cf70k_oai` are both at ≥1 positive / ≥3 negatives (measured minima 1/3);
  `cf70k.jsonl` is still the 2/5 output. §7.5.2 prices a `--replay` of bharatcode's saved
  responses at **+348 examples for no API calls** — still unclaimed, and still one `--replay`
  away.

## What changed since the 2026-08-22 snapshot

| | 2026-08-15 | 2026-08-22 | **2026-08-25** | Δ vs 08-22 |
|---|---:|---:|---:|---:|
| bharatcode | 1,560 | 1,560 | 1,560 | — |
| gemini | 22,851 | 31,186 | **35,983** | **+4,797** |
| openai | 2,703 | 3,327 | **3,837** | **+510** |
| **total** | **27,114** | **36,073** | **41,380** | **+5,307 (+14.7%)** |
| distinct questions | 8,639 | 11,390 | **13,012** | +1,622 |

Both floors were already 1/3 going into this batch, so unlike 08-22's openai number, **all
+5,307 are new anchors** rather than partly a floor change. The 2026-08-22 snapshot's anchors
are a strict subset of this one on all three files (0 anchors lost).

> **`data.cf_max_per_batch: 12` WAS ALREADY BINDING AT 36,073 AND IS NOW BINDING BY MORE.**
> §7.5.13 sized it on 27,114 examples over ~2,920 micro-batches = ~9.3 per batch; 36,073 put
> it at ~12.4 raw / ~11.3 after the measured 91.6% attach rate. At 41,380 it is **~14.2 per
> batch** raw, **~13.0 after attach** — over the cap on the mean on both figures now, where in
> August it was over on only one of them. Nothing breaks: selection among the eligible is
> uniform without replacement, so the effect is that a growing fraction of eligible examples
> is simply not drawn in a given epoch, and `cf/examples_attached` vs `cf/examples_eligible`
> will show it. **Raising the cap is a cost decision (`L_CF`'s magnitude and the per-step cost
> both scale with it) and has still not been made** — it is recorded here so the next person
> reads the attach ratio as a cap effect and not as a broken join. The practical reading: the
> +14.7% of new examples buys coverage across epochs, not more CF signal per step.

## It is a snapshot, so it goes stale

A campaign that keeps generating rewrites `data/cf/*.jsonl` and **not** these copies — the
gemini file grew 17,895 → 22,851 during a single session on 2026-08-15, 22,851 → 31,186 over
the week to 2026-08-22, and 31,186 → 35,983 in the three days to 2026-08-25. Before a run that
is meant to see new examples:

```bash
cp data/cf/cf70k.jsonl data/cf/cf70k_gm.jsonl data/cf/cf70k_oai.jsonl data/cf_train/
```

then re-rsync, and re-run this manifest's counts. **Do not move the originals out of
`data/cf/`**: `scripts/cf_exclude_generated.py` globs `data/cf/*responses.jsonl` to skip
anchors already paid for, and a campaign that stops finding them regenerates them.
