"""The B6 guard, and the trap it walked into on 2026-07-27 (§14 B6, §11.1).

The guard is supposed to catch "no scheduler was ever built". As written it compared the LR at
the end of the run against the LR read right after `build_scheduler` returned, and BOTH of those
are 0.0 on a healthy cosine run:

    * LambdaLR's constructor applies `lr_lambda(0)`, which under warmup is `0/warmup` = 0.0
    * a completed cosine ends at `0.5*(1 + cos(pi))` = 0.0, exactly

so the assert fired on every run that FINISHED and passed only on runs cut short by
`--max-steps`. It killed a 971-step run whose `save_checkpoint("final")` sat below the raise.

These tests pin both halves: the schedule really does start and end at 0.0 (so nobody
"restores" the start-vs-end comparison thinking it was fine), and a range read over the whole
run separates a real scheduler from a missing one.
"""

from __future__ import annotations

import re

import pytest
import torch

from conftest import REPO_ROOT
from feynman_prm.train import build_scheduler


def _run(cfg, total_steps):
    """Returns (lr_before, lrs) exactly as train.py's loop observes them."""
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=1.0, foreach=False)
    scheduler = build_scheduler(optimizer, total_steps, cfg)
    lr_before = optimizer.param_groups[0]["lr"]
    lrs = []
    for _ in range(total_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    return lr_before, lrs


def test_a_completed_cosine_run_starts_and_ends_at_exactly_the_same_lr(cfg):
    """THE TRAP. Both endpoints are 0.0, so `lr_end == lr_before` is a true statement about a
    perfectly healthy run. Any B6 check written against these two numbers is a false positive
    machine that only ever passes on truncated runs."""
    lr_before, lrs = _run(cfg, total_steps=971)

    assert lr_before == 0.0, "LambdaLR's constructor already applied lr_lambda(0)"
    assert lrs[-1] == 0.0, "a completed cosine decays to exactly zero"
    assert lrs[-1] == lr_before

    assert max(lrs) == pytest.approx(1.0, abs=0.02), "and yet the LR plainly moved"


def test_the_lr_range_over_the_whole_run_separates_scheduler_from_no_scheduler(cfg):
    """What the guard reads instead: min/max across every optimizer step, seeded with the
    pre-loop value so even a 1-step run has two samples."""
    lr_before, lrs = _run(cfg, total_steps=971)
    lo, hi = min([lr_before] + lrs), max([lr_before] + lrs)
    assert hi > lo, "a real cosine schedule moves"

    # No scheduler: the LR is whatever the optimizer was built with, forever.
    flat = [1.0] * 971
    assert max([1.0] + flat) <= min([1.0] + flat), "bug B6 is what this must catch"


def test_the_guard_survives_a_one_step_run(cfg):
    lr_before, lrs = _run(cfg, total_steps=1)
    assert max([lr_before] + lrs) > min([lr_before] + lrs)


def test_train_py_does_not_compare_the_final_lr_against_the_pre_loop_one():
    """Grep, because the failure is a line of code existing (§15). `lr_before` may be recorded
    -- it seeds the range -- but it must never be the right-hand side of the B6 comparison."""
    code = (REPO_ROOT / "feynman_prm" / "train.py").read_text()
    code = re.sub(r"#.*", "", code)
    code = re.sub(r'"""..*?"""', "", code, flags=re.S)
    assert 'param_groups[0]["lr"] == lr_before' not in code
    assert "lr_max_seen" in code and "lr_min_seen" in code


def test_the_final_checkpoint_is_written_before_the_b6_raise():
    """A diagnostic must never destroy the artifact it is diagnosing. This is the whole reason
    the 2026-07-27 run lost 971 steps of training."""
    code = (REPO_ROOT / "feynman_prm" / "train.py").read_text()
    final_save = code.index('/ "final"')
    b6_raise = code.index("bug B6 (no scheduler) is back")
    assert final_save < b6_raise, "save_checkpoint('final') must sit ABOVE the B6 raise"
