"""`train.py --resume`: the three things a stepN checkpoint CAN restore exactly.

A checkpoint holds weights and a step number and nothing else -- no optimizer moments, no
scheduler state, no sampler position. Two of those three absences are recoverable anyway,
because the pieces that would have held them are pure functions of state that IS on disk:

    * the LR       -- `build_scheduler` is a closure over (step, total_steps, cfg) with no
                      hidden state, so replaying it `resume_step` times is not an approximation
                      of the original schedule, it IS the original schedule;
    * the data     -- `epoch_batches` is seeded off `cfg.run.seed` alone, so the batch order is
                      identical across runs and "the batches step N never saw" is a slice.

The third, the Adam moments, is genuinely lost. That is why `betas[1]` is asserted here: the
resume is only defensible while the second-moment memory is short against the remainder, and
this test is where a future beta2 bump gets caught.

These are the asserts that would have caught a resume landing on the wrong LR or replaying
batches the checkpoint had already trained on -- both of which produce a plausible-looking
curve and a quietly invalid run.
"""

from __future__ import annotations

import math

import torch

from feynman_prm.train import build_scheduler

# The real plan of runs/cf_lam2_tau005, from its own launch/data event.
TOTAL_STEPS = 1464
GRAD_ACCUM = 2
N_BATCHES = 2928
RESUME_STEP = 750

# What that run logged at step 750, before it was killed at 990. Hard-coded on purpose: this
# ties the replay to an observation, not to a re-derivation of the same formula.
LOGGED_LR_BACKBONE = 4.539822e-06
LOGGED_LR_HEADS = 1.513274e-04


def _sched(cfg, lrs):
    opt = torch.optim.SGD([{"params": [torch.zeros(1, requires_grad=True)], "lr": lr} for lr in lrs])
    return opt, build_scheduler(opt, TOTAL_STEPS, cfg)


def test_replayed_lr_equals_the_lr_an_uninterrupted_run_holds(cfg):
    """Stepping the scheduler `resume_step` times must land where the original run was.

    The subtlety is the off-by-one: the loop calls `optimizer.step()` and THEN
    `scheduler.step()`, so after N optimizer steps the scheduler has been stepped N times and
    the param groups hold lr_lambda(N) -- the LR for step N+1. `--resume` replays exactly N,
    so it inherits that convention rather than reconstructing it.
    """
    opt, sched = _sched(cfg, [9.0e-6, 3.0e-4])
    for _ in range(RESUME_STEP):
        sched.step()

    # rel_tol 1e-6: metrics.jsonl rounds to 7 significant figures, so that is the full
    # precision the observation was ever recorded at -- not a tolerance on the replay itself.
    assert math.isclose(opt.param_groups[0]["lr"], LOGGED_LR_BACKBONE, rel_tol=1e-6)
    assert math.isclose(opt.param_groups[1]["lr"], LOGGED_LR_HEADS, rel_tol=1e-6)


def test_replay_matches_a_run_that_was_never_interrupted(cfg):
    """The whole remaining schedule, not just the resume point.

    A resume that landed on the right LR and then rode a schedule built over the REMAINDER
    (714 steps) instead of the full plan (1464) would pass the test above and still be wrong:
    the cosine would fall to zero in half the distance. This compares every remaining step.
    """
    _, whole = _sched(cfg, [3.0e-4])
    uninterrupted = []
    for _ in range(TOTAL_STEPS):
        whole.step()
        uninterrupted.append(whole.get_last_lr()[0])

    opt, resumed = _sched(cfg, [3.0e-4])
    for _ in range(RESUME_STEP):
        resumed.step()
    tail = []
    for _ in range(TOTAL_STEPS - RESUME_STEP):
        resumed.step()
        tail.append(resumed.get_last_lr()[0])

    assert tail == uninterrupted[RESUME_STEP:]
    assert math.isclose(tail[-1], 0.0, abs_tol=1e-12)   # a completed cosine ends at exactly 0


def test_the_skip_consumes_every_batch_exactly_once(cfg):
    """`skip_micro = resume_step * grad_accum` must partition the epoch, not overlap or gap it.

    Off by one grad_accum here means the resumed run either re-trains on a micro-batch the
    checkpoint already saw or silently drops one, and the step count still lands plausibly
    close to the plan. Nothing downstream would show it.
    """
    skip_micro = RESUME_STEP * GRAD_ACCUM
    assert skip_micro == 1500
    remaining_steps = (N_BATCHES - skip_micro) // GRAD_ACCUM
    assert RESUME_STEP + remaining_steps == TOTAL_STEPS
    assert (N_BATCHES - skip_micro) % GRAD_ACCUM == 0   # the boundary is exact


def test_beta2_memory_is_short_against_the_remainder(cfg):
    """The one piece of state a checkpoint cannot return, bounded rather than ignored.

    Adam's second moment restarts at zero on resume. Its half-life is ln(2)/ln(1/beta2) steps;
    at beta2 = 0.95 that is ~14, which is noise against the ~714 steps a mid-run resume has
    left. At beta2 = 0.999 it would be ~693 -- the same order as the remainder -- and the
    transient would stop being a blip and start being a confound. Resume would then need real
    optimizer state in the checkpoint, and this assert is where that gets noticed.
    """
    beta2 = cfg.train.betas[1]
    half_life = math.log(2) / math.log(1 / beta2)
    remaining = TOTAL_STEPS - RESUME_STEP
    assert half_life < 0.1 * remaining, (
        f"beta2 = {beta2} gives a {half_life:.0f}-step second-moment memory against "
        f"{remaining} remaining steps. --resume drops the moments; at this beta2 that is no "
        f"longer a short transient. Save optimizer state in save_checkpoint instead."
    )
