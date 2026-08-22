# cf_lam2_tau005

```
bash scripts/train_cloud.sh --set losses.zeta=0.2 \
    --set losses.nce_temperature=22.627417 \
    --set losses.lambda_term=0.0 \
    --set losses.lambda_cf=2.0 \
    --set losses.lambda_cf_temperature=0.05 \
    --set run.name=cf_lam2_tau005
```

A100 40 GB on Modal, PROFILE=match (batch shape is `config/default.yaml`'s, so every
loss statistic is comparable to the other 56-sequence runs).

TWO-DELTA against `runs/abl_cf_only`, which is identical on every other axis
(zeta 0.2, lambda_term 0.0, nce_temperature sqrt(512)). Both deltas are on (4) L_CF:
lambda_cf 1.0 -> 2.0 and lambda_cf_temperature 0.1 -> 0.05. Per the config, tau and
lambda are the same knob (gradient ~ lambda/tau), so the EFFECTIVE weight on (4) is
4x the baseline's, not 2x -- read any L_CF difference against that, not against 2x.

`lambda_cf_temperature = 0.05` is the value `config/default.yaml` explicitly rejected
as saturating. THE TELL IS `cf/loss - cf/chance`, never raw `cf/loss`: near -0.02 the
change did nothing, -1.09 or beyond means tau overshot and 0.2 is the fallback.

## This run was RESUMED, and `abl_cf_only` was not

The first attempt died at step 990 of 1,464 when the Modal client cancelled the call
(`.remote()` binds a call to the client session; the launcher now uses `.spawn()`).
Steps 751-1464 were re-run from `step750` with `train.py --resume`, added for this.

Restored exactly: the weights, the LR (`build_scheduler` replayed over the FULL 1,464-step
cosine, so step 751 opened on 4.539822e-06 -- the value `metrics.jsonl` logged at 750, to
its full recorded precision), and the data position (`epoch_batches` is seeded off
`run.seed` alone, so the run consumed `batches[1500:]`, precisely the micro-batches step750
never saw).

NOT restored: AdamW's moments, which are not in a checkpoint and restarted at zero.
`betas[1] = 0.95` bounds that -- a ~14-step second-moment half-life against 714 remaining
steps -- and the measured transient is nil: `nce/loss - nce/chance` was -0.0139 over steps
700-750 (before the kill) and -0.0134 over 760-790 (after the resume), improving
monotonically to -0.0204 by the end. No bump is visible in any curve.

It is recorded anyway because the comparison run has no such discontinuity, and a
difference between two runs being compared is not the reader's to rediscover.

## Phase-1 ceiling (held-out Math-Shepherd val)

- val F1 at best tau: **0.5636827260156378**  (tau = 0.2676096643209447)

## ProcessBench

- tau = 0.29827932119369494

| subset | F1 |
|---|---|
| gsm8k | 0.3828 |
| math | 0.2962 |
| olympiadbench | 0.2056 |
| omnimath | 0.16 |
| **mean** | **0.2611** |

Math subset, leak split: clean **0.2378**, leaked 0.3478.

Skyline rows in `processbench.json` are labelled and are never a reported result
(the gold answer alone solves half the metric).

wandb: https://wandb.ai/unrealparticles-iit-roor/feynman-prm/runs/yksnqcp5

## What is here and what is not

Committed: the resolved config, both metrics/events streams, `val_f1.json`,
`processbench.json`, `deltas.npz`. NOT committed: the LoRA adapters, `heads.pt`,
the tokenizers and `phase2/cache.pt` -- ~1 GB of weights that live on the box the
run was fetched to.

```json
{
  "run": "cf_lam2_tau005",
  "val_f1_at_best_tau": 0.5636827260156378,
  "val_f1_tau": 0.2676096643209447,
  "processbench": {
    "gsm8k": 0.3828,
    "math": 0.2962,
    "olympiadbench": 0.2056,
    "omnimath": 0.16
  },
  "processbench_mean_f1": 0.2611,
  "processbench_tau": 0.29827932119369494,
  "math_clean_f1": 0.2378,
  "math_leaked_f1": 0.3478,
  "wandb_url": "https://wandb.ai/unrealparticles-iit-roor/feynman-prm/runs/yksnqcp5"
}
```
