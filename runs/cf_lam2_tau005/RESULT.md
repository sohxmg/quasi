# cf_lam2_tau005 — CANCELLED AT STEP 990 / 1464, NO EVAL

```
bash scripts/train_cloud.sh --set losses.zeta=0.2 \
    --set losses.nce_temperature=22.627417 \
    --set losses.lambda_term=0.0 \
    --set losses.lambda_cf=2.0 \
    --set losses.lambda_cf_temperature=0.05 \
    --set run.name=cf_lam2_tau005
```

wandb: https://wandb.ai/unrealparticles-iit-roor/feynman-prm/runs/tng5sf4a

Modal A100 40 GB, PROFILE=match, ~1.5 h billed. TWO-DELTA against `runs/abl_cf_only`,
which is identical on every other axis (zeta 0.2, lambda_term 0.0, nce_temperature
sqrt(512)). Both deltas are on (4) L_CF: lambda_cf 1.0 -> 2.0 and lambda_cf_temperature
0.1 -> 0.05. Per `config/default.yaml`, tau and lambda are the same knob (gradient ~
lambda/tau), so the EFFECTIVE weight on (4) is **4x** the baseline's, not 2x.

## Why there is no eval

The run did NOT fail and was NOT stopped on its merits. The launcher called
`train_and_eval.remote()`, which ties the function call to the client session; when that
client was reaped the call was cancelled (`Successfully canceled input`) and the app
stopped at step 990. `modal run --detach` guards against the parent being KILLED, not
against the client cancelling on a graceful shutdown. **`Function.spawn()` is the
fire-and-forget call that has no client to cancel it** and is what a relaunch must use.

Checkpoints step250/500/750 and 100 logged points through step 990 survived.

## Finding 1: tau_cf = 0.05 did NOT saturate

`config/default.yaml` rejected 0.05 outright: "it saturates (p_pos -> 1, loss -> -log(1)
territory) if the separation keeps growing, and a saturated softmax has no gradient left."
That did not happen at the separations this run reached.

THE TELL IS `cf/loss - cf/chance`, never raw `cf/loss`. Near -0.02 the change did nothing;
-1.09 or beyond means tau overshot. It held between -0.36 and -0.49 from step 750 on, and
stayed RESPONSIVE to separation (~-4.0 per unit) rather than flattening -- flattening while
separation climbs is the saturation signature, and it never appeared.

| step | separation | cf-chance (tau 0.05) | abl_cf_only (tau 0.1) |
|---|---|---|---|
| 50 | 0.0018 | -0.0170 | -0.0063 |
| 100 | 0.0058 | -0.0397 | -0.0100 |
| 150 | 0.0282 | -0.1763 | -0.0853 |
| 200 | 0.0208 | -0.1346 | -0.0668 |
| 250 | 0.0483 | -0.2672 | -0.1566 |
| 300 | 0.0907 | -0.4400 | -0.1683 |
| 350 | 0.0618 | -0.3686 | -0.2304 |
| 400 | 0.0474 | -0.1833 | -0.1898 |
| 450 | 0.0618 | -0.3414 | -0.1322 |
| 500 | 0.0544 | -0.2211 | -0.0928 |
| 550 | 0.0646 | -0.4296 | -0.1555 |
| 600 | 0.0605 | -0.3667 | -0.0679 |
| 650 | 0.0900 | -0.4076 | -0.2238 |
| 700 | 0.0402 | -0.1005 | -0.1555 |
| 750 | 0.0935 | -0.3630 | -0.3458 |
| 800 | 0.1011 | -0.4172 | -0.3277 |
| 850 | 0.0851 | -0.4028 | -0.2103 |
| 900 | 0.1079 | -0.4531 | -0.3996 |
| 950 | 0.0973 | -0.4928 | -0.3853 |

## Finding 2: the 4x effective weight did NOT buy separation

Over steps 750-990 this run averages **0.0970** pos/neg separation against
`abl_cf_only`'s **0.1237** -- the 1x baseline is AHEAD. L_CF's own loss is sharper (the
tell is consistently more negative, as halving tau dictates) while the geometry it is
supposed to shape is not better separated. An early 250-500 advantage of ~33% closed and
then reversed.

Neither knob is a lever at these magnitudes. That is the result, and it did not need the
remaining 474 steps to show.

## What is committed

`metrics.jsonl`, `events.jsonl`, `config.resolved.yaml`, this file. The three mid-run
checkpoints (~101 MB each) stayed on the run machine.
