"""(6) L_good's ablation on the real model — **OPT-IN, and not part of any normal run.**

    pytest tests/test_good_loss_ablation.py -m ablation -v -s

**These are the only tests in the repo that take optimizer steps on the real backbone.** Each
one loads Qwen2.5-Math-1.5B twice and runs 12 steps on a 12-sequence toy batch — a couple of
minutes, no checkpoint, no data, nothing written. That is still more than `test_gpu.py` should
cost, which is why they live here behind their own marker and are deselected by
`-m "gpu and not ablation"`.

**Do not run these before the phase-1 run just to feel safe about `lambda_good`.** The real
run measures the same thing better and for free: `probe03/gap` and
`probe14/delta_good_of_correct/frac_above_natural` are logged every `log_every` steps from
step 1 (§10 #14, #18), over real batches, for the whole run. 12 steps on a toy fixture is a
*wiring* check, not evidence.

They exist for one situation: **the run goes wrong and you need to know whether `L_good` or
something else did it.** Then an A/B from a fixed seed, with everything else held constant, is
worth the two minutes — and it is the only place §7.12's weakest claim ("it does not drown
`L_step`") can be tested at all, because free latents make detection too easy to be
informative.

`tests/test_gpu.py`'s own `D2` block covers the cheap half — that `L_good` and the probe read
the same `Δ`, and that `detach_goal` does what it says — with no optimizer steps at all.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.ablation]

pytest.importorskip("transformers")
pytest.importorskip("peft")

from feynman_prm.config import load_config                                  # noqa: E402
from feynman_prm.model.backbone import load_backbone, param_groups, read_hidden_size  # noqa: E402
from feynman_prm.model.wrapper import FeynmanPRM                            # noqa: E402
from conftest import REPO_ROOT                                              # noqa: E402
from test_gpu import _good_specs, forward_batch, make_rows                  # noqa: E402


@pytest.fixture(scope="module")
def gpu_cfg():
    return load_config(REPO_ROOT / "config" / "default.yaml")


@pytest.fixture(scope="module")
def tokenizer(gpu_cfg):
    from feynman_prm.model.backbone import load_tokenizer

    return load_tokenizer(gpu_cfg)


def _train(gpu_cfg, tokenizer, rows, cfg, steps=12, probe=False):
    """A fresh model, `steps` optimizer steps, then teardown. Deliberately minimal: this is
    the same loop `train.py` runs, minus the logging, checkpoints and every launch assert."""
    from feynman_prm.diagnostics.probes import batch_probes
    from feynman_prm.train import build_scheduler

    torch.manual_seed(0)
    backbone = load_backbone(gpu_cfg)
    m = FeynmanPRM(gpu_cfg, read_hidden_size(gpu_cfg.model.name), backbone=backbone)
    m.pad_id = tokenizer.pad_token_id
    m = m.cuda().train()
    opt = torch.optim.AdamW(param_groups(m, cfg), betas=tuple(cfg.train.betas),
                            weight_decay=cfg.train.weight_decay, foreach=False)
    sched = build_scheduler(opt, total_steps=steps, cfg=cfg)

    first = last = None
    for k in range(steps):
        batch, goals, reps, matrices, out = forward_batch(m, cfg, rows, seed=k)
        snapshot = dict(out.info)
        if probe:
            snapshot.update(
                batch_probes(reps.psi, reps.phi, batch, goals, matrices, m.distance, cfg)
            )
        if first is None:
            first = snapshot
        last = snapshot
        out.total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in m.parameters() if p.requires_grad], cfg.train.grad_clip
        )
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

    del m, opt
    torch.cuda.empty_cache()
    return first, last


def test_l_good_pulls_the_tail_down_without_collapsing_the_boundary_gap(gpu_cfg, tokenizer):
    """**§7.12's weakest claim.** `L_good` carries an order of magnitude more terms than
    `L_step` (~660 vs ~67 pairs at the §8.1.1 layout), so "it does not drown `L_step`" is the
    sentence to distrust — and free latents make detection too easy to test it there.

    Twelve steps, twice, from the same seed: `lambda_good = 0` against `lambda_good = 1`.
    BOTH halves must hold —

        probe14/delta_good_of_correct/frac_above_natural   must FALL   (the point)
        probe03/gap  (bad-step Delta − good-step Delta)    must NOT collapse   (the risk)

    Twelve steps is not a training run and the deltas are nowhere near converged. This is a
    directional check that catches "L_good flattens everything"; the real measurement is the
    full run's own `probe03/gap` curve.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    rows = make_rows(tokenizer, gpu_cfg, _good_specs(n_questions=4))
    on_cfg = dataclasses.replace(
        gpu_cfg, losses=dataclasses.replace(gpu_cfg.losses, lambda_good=1.0)
    )
    keys = ("probe14/delta_good_of_correct/frac_above_natural",
            "probe14/delta_good_of_correct/p90", "probe03/gap",
            "probe14/delta_boundary/mean", "good/delta_mean", "backup/delta_mean", "step/loss")

    off_first, off_last = _train(gpu_cfg, tokenizer, rows, gpu_cfg, probe=True)
    on_first, on_last = _train(gpu_cfg, tokenizer, rows, on_cfg, probe=True)

    for name, first, last in (("lambda_good = 0.0", off_first, off_last),
                              ("lambda_good = 1.0", on_first, on_last)):
        print(f"\n  {name}")
        for k in keys:
            print(f"    {k:<48} {first[k]:+.4f} -> {last[k]:+.4f}")

    assert on_last["good/delta_mean"] < on_first["good/delta_mean"], (
        "L_good did not move its own statistic in 12 steps — check the sign of c (§7.12)"
    )
    assert on_last["probe14/delta_good_of_correct/frac_above_natural"] <= (
        off_last["probe14/delta_good_of_correct/frac_above_natural"] + 1e-6
    ), "the good-step tail is no lower with L_good on than off. That is the whole term."
    # The guard, and the reason this test is not a one-liner: probe03/gap collapsing toward
    # zero means the error signal has been flattened (§16.3, diagnostic #3).
    assert on_last["probe03/gap"] > 0.25 * off_last["probe03/gap"], (
        f"probe03/gap collapsed from {off_last['probe03/gap']:+.4f} to "
        f"{on_last['probe03/gap']:+.4f} with L_good on — L_good is drowning L_step. Lower "
        "lambda_good (§7.12)."
    )


def test_relu_does_not_overshoot_the_ruler_where_softplus_does(gpu_cfg, tokenizer):
    """§7.12's form ablation, on the model rather than on free latents. `softplus` applies
    half its gradient AT the target and never stops, so it drives `Δ` past `c` and takes
    `L_T`'s ruler with it; `relu` switches off at `c`.

    Asserted loosely and printed fully: 12 steps cannot reproduce the simulated −1.556 vs
    −0.834, and §7.12's table is labelled SIMULATED for that reason. What must hold is the
    ORDER — `softplus` ends at or below `relu` on `good/delta_mean`.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    rows = make_rows(tokenizer, gpu_cfg, _good_specs(n_questions=4))

    def variant(form):
        return dataclasses.replace(
            gpu_cfg,
            losses=dataclasses.replace(
                gpu_cfg.losses,
                lambda_good=1.0,
                good_loss=dataclasses.replace(gpu_cfg.losses.good_loss, form=form),
            ),
        )

    relu = _train(gpu_cfg, tokenizer, rows, variant("relu"))[1]
    soft = _train(gpu_cfg, tokenizer, rows, variant("softplus"))[1]

    print(f"\n  target c {gpu_cfg.good_margin:+.4f}")
    for name, info in (("relu", relu), ("softplus", soft)):
        print(f"    {name:<9} good/delta_mean {info['good/delta_mean']:+.4f}  "
              f"above target {info['good/above_target_fraction']:.3f}  "
              f"backup/delta_mean {info['backup/delta_mean']:+.4f}  "
              f"L_good {info['good/loss']:.4f}")

    assert soft["good/delta_mean"] <= relu["good/delta_mean"] + 1e-3, (
        "softplus did not push at least as hard as relu — it applies gradient AT the target "
        "and relu does not, so this ordering is structural (§7.12)"
    )
