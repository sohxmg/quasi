# nce1_masked — STOPPED AT STEP 1030 / 1464, NO EVAL

```
bash scripts/train_cloud.sh --set losses.zeta=0.2 \
    --set losses.nce_temperature=4.0 \
    --set sampling.nce_mask_sibling_correct_late=true \
    --set losses.lambda_term=0.0 \
    --set run.name=nce1_masked
```

Modal A100 40 GB, PROFILE=match, ~1.7 h billed. **Stopped deliberately at step 1030 of
1,464** once the curves had shown what they were going to show, so there is NO `final/`
checkpoint, NO `val_f1.json` and NO `processbench.json`. What exists is 104 logged points
and four mid-run checkpoints (step250/500/750/1000).

wandb: https://wandb.ai/unrealparticles-iit-roor/feynman-prm/runs/dlujoput

## Why it was stopped

This is a ONE-DELTA run against `runs/nce_tau4` (same zeta 0.2, same tau_NCE 4.0, same
lambda_term 0.0); the only difference is §16.25(a)'s `nce_mask_sibling_correct_late`.
The mask demonstrably BINDS -- it roughly doubles to triples `nce/negatives_masked` -- and
every downstream number is nevertheless identical to the baseline to 3-4 decimals.

`nce/negatives_per_column` is ~460, so excluding ~1.5 columns instead of ~0.8 is a change
to **~0.3% of the negative pool**. That is too small a perturbation to move F1 out of
noise, which is the finding: at `nce_sibling_late_margin: 1` (keeping only the last two
phi states) the principled mask is not a lever. Raising that margin is the knob that would
make it one.

## The comparison, at matched logged steps

| step | metric | nce_tau4 | nce1_masked | delta |
|---|---|---|---|---|
| 250 | `nce/negatives_masked` | 0.7921 | 0.9944 | +0.2022 |
| 250 | `nce/loss` | 6.1095 | 6.1137 | +0.0042 |
| 250 | `nce/chance` | 6.1356 | 6.1356 | +0.0000 |
| 250 | `nce/logit_std` | 0.0548 | 0.0547 | -0.0001 |
| 250 | `probe03/gap` | 0.1403 | 0.1445 | +0.0042 |
| 250 | `invariance/residual_diagonal` | 0.1452 | 0.1437 | -0.0015 |
| 250 | `backup/loss` | 0.0904 | 0.1007 | +0.0103 |
| 250 | `probe14/frac_above_natural` | 0.0208 | 0.0208 | +0.0000 |
| 500 | `nce/negatives_masked` | 0.4222 | 1.5778 | +1.1556 |
| 500 | `nce/loss` | 5.2721 | 5.2717 | -0.0004 |
| 500 | `nce/chance` | 5.3660 | 5.3660 | +0.0000 |
| 500 | `nce/logit_std` | 0.1979 | 0.1975 | -0.0004 |
| 500 | `probe03/gap` | 0.7103 | 0.7230 | +0.0127 |
| 500 | `invariance/residual_diagonal` | 0.1356 | 0.1262 | -0.0094 |
| 500 | `backup/loss` | -0.2874 | -0.2808 | +0.0067 |
| 500 | `probe14/frac_above_natural` | 0.0957 | 0.0783 | -0.0174 |
| 750 | `nce/negatives_masked` | 0.4058 | 1.4348 | +1.0290 |
| 750 | `nce/loss` | 4.7099 | 4.7060 | -0.0039 |
| 750 | `nce/chance` | 4.9416 | 4.9416 | +0.0000 |
| 750 | `nce/logit_std` | 0.2932 | 0.2857 | -0.0075 |
| 750 | `probe03/gap` | 0.4534 | 0.4223 | -0.0311 |
| 750 | `invariance/residual_diagonal` | 0.0952 | 0.1045 | +0.0093 |
| 750 | `backup/loss` | -0.5348 | -0.5070 | +0.0278 |
| 750 | `probe14/frac_above_natural` | 0.0976 | 0.1220 | +0.0244 |
| 1000 | `nce/negatives_masked` | 0.8282 | 1.5521 | +0.7239 |
| 1000 | `nce/loss` | 5.6413 | 5.6411 | -0.0002 |
| 1000 | `nce/chance` | 5.8999 | 5.8999 | +0.0000 |
| 1000 | `nce/logit_std` | 0.3210 | 0.3153 | -0.0057 |
| 1000 | `probe03/gap` | 0.9945 | 0.9890 | -0.0055 |
| 1000 | `invariance/residual_diagonal` | 0.1099 | 0.1069 | -0.0030 |
| 1000 | `backup/loss` | -0.7297 | -0.7131 | +0.0166 |
| 1000 | `probe14/frac_above_natural` | 0.0983 | 0.0940 | -0.0043 |
| 1030 | `nce/negatives_masked` | 0.7358 | 1.2767 | +0.5409 |
| 1030 | `nce/loss` | 5.6076 | 5.6089 | +0.0013 |
| 1030 | `nce/chance` | 5.9454 | 5.9454 | +0.0000 |
| 1030 | `nce/logit_std` | 0.4031 | 0.3898 | -0.0133 |
| 1030 | `probe03/gap` | 0.2741 | 0.2634 | -0.0106 |
| 1030 | `invariance/residual_diagonal` | 0.0928 | 0.0933 | +0.0005 |
| 1030 | `backup/loss` | -0.8810 | -0.8414 | +0.0396 |
| 1030 | `probe14/frac_above_natural` | 0.1220 | 0.1268 | +0.0049 |

## L_NCE read against its own chance

`nce/chance = log(R)` moves with the batch, so the raw curve is not comparable across
batches; the difference is (diagnostic #19). It is the same in both runs:

| step | nce_tau4 | nce1_masked |
|---|---|---|
| 250 | -0.0261 | -0.0219 |
| 500 | -0.0939 | -0.0943 |
| 750 | -0.2318 | -0.2357 |
| 1000 | -0.2586 | -0.2588 |
| 1030 | -0.3378 | -0.3365 |

## Health checks that did pass

Not a broken run -- a run that answered its question early. `logit_std` 0.055 -> 0.390
(no bug B10a), `invariance/residual_diagonal` 0.144 -> 0.093 (under the 0.15 lambda_good
guard throughout), `backup` negative from step ~400 as expected, `probe03/gap` climbing
0.14 -> 0.99, `optimizer_steps` 1464 (not the 106-step regression), memory probe 12.1 GB.

## What is committed here

`metrics.jsonl`, `events.jsonl`, `config.resolved.yaml`, this file. The four mid-run
checkpoints (~101 MB each: adapter 74 MB + heads 16 MB + tokenizer 11 MB) were downloaded
to the run machine but are NOT in git.
