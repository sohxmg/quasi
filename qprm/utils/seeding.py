"""Seeding. There are no DataLoader workers anywhere in this repo (PLAN 3a), so there is
no worker-seeding nondeterminism to manage: batch assembly is a deterministic generator in
the main process."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # the data-only paths do not need torch
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epoch_rng(seed: int, epoch: int) -> np.random.Generator:
    """The sampler's RNG. Distinct per epoch, reproducible from (seed, epoch) alone."""
    return np.random.default_rng([seed, epoch])


def goal_rng(seed: int, epoch: int, step: int) -> np.random.Generator:
    """The geometric goal sampler's RNG (§7.1). Keyed by step so a resumed run at the same
    step draws the same goals."""
    return np.random.default_rng([seed, epoch, step, 0xC0A1])
