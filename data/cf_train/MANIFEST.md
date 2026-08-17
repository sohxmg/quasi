# `data/cf_train/` — the (4) `L_CF` corpus, snapshot 2026-08-15

**This is a SNAPSHOT of the kept-example files in `data/cf/`, not the campaign directory.**
`data/cf/` is 426 MB and only 52 MB of it is data a training run can read; the rest is
`.raw` / `.rejected` / `.discarded` / `.responses` artifacts with different schemas
(§7.5.13's "sixteen files" warning). This directory is the 52 MB, and it is the only data
directory that ships to the GPU box.

`config/default.yaml`'s `data.cf_glob` points here. It is a comma-separated **list**, never a
glob — see §7.5.13.

## Contents

| file | examples | pos/ex | neg/ex | bytes | sha256[:16] |
|---|---:|---:|---:|---:|---|
| `cf70k.jsonl`     (bharatcode) | 1,560 | 2.99 | 5.69 | 3,231,239 | `290f932a7d437d1c` |
| `cf70k_gm.jsonl`  (gemini)     | 22,851 | 2.99 | 5.94 | 45,383,304 | `0ef4ef9a2da5b112` |
| `cf70k_oai.jsonl` (openai)     | 2,703 | 2.96 | 5.97 | 5,417,806 | `e4a05657573b1d6d` |
| **total** | **27,114** | | | **54,032,349** | |

- **241,779 rewrites** over 27,114 anchors — 268,893 vectors per epoch counting the anchors
  themselves, which is the 9.92/example figure §7.5.2 quotes.
- **8,639 distinct questions**, all inside the 34,650-question train selection, none in the
  2,000-question val holdout.
- **0 duplicate `(question, step_index)`** across the three campaigns.
- Validator floors are **mixed and that is a leftover, not a decision** (§7.5.2): `cf70k_gm`
  was generated at ≥1 positive / ≥3 negatives, the other two are still at ≥2 / ≥5. The 455
  examples that separate this from the 27,569 a uniform 1/3 replay would yield are one free
  `--replay` per campaign away.

## It is a snapshot, so it goes stale

A campaign that keeps generating rewrites `data/cf/*.jsonl` and **not** these copies — the
gemini file grew 17,895 → 22,851 during a single session on 2026-08-15. Before a run that is
meant to see new examples:

```bash
cp data/cf/cf70k.jsonl data/cf/cf70k_gm.jsonl data/cf/cf70k_oai.jsonl data/cf_train/
```

then re-rsync, and re-run this manifest's counts. **Do not move the originals out of
`data/cf/`**: `scripts/cf_exclude_generated.py` globs `data/cf/*responses.jsonl` to skip
anchors already paid for, and a campaign that stops finding them regenerates them.
