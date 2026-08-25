"""QRL (Quasimetric RL, Wang & Isola, ICLR 2023) + counterfactual invariance, as a
constrained-optimization variant of Feynman-PRM.

**This is our adaptation of QRL's objective to Feynman-PRM's exact training conditions --
not QRL's published numbers.** QRL is an offline/online goal-reaching RL method on
continuous-control benchmarks; it has never been run on chain-of-thought data and reports no
ProcessBench number. Say so wherever the row appears.

`quasimetric-rl/` (the authors' released code) stays UNTOUCHED as the vendored reference, the
same treatment `../tmd-release/`, `../CRM/` and `../Process_Q_Model/` get. The pieces used
here -- `grad_mul`, `softplus_inv_float`, the global-push transform, the local-constraint
Lagrangian -- are ported with line citations, never imported.

**What replaces what.** Feynman-PRM's phase-1 set trains a soft ruler out of fixed-weight
losses, and its known open failure is that the ruler decays (`backup/delta_mean` drifts off
`-log gamma`, IMPLEMENTATION.md §9). QRL replaces the soft ruler with a CONSTRAINT: maximize
distances everywhere, subject to Lagrangian constraints that hold the known-small ones down.
So none of (1) L_NCE, (2) L_I, (3) L_T, (4) L_CF, (5) L_step, (6) L_good, (7) L_term is
computed here. The loss set is three terms and two dual variables:

    L = push  +  lambda_local * (local violation)  +  lambda_cf * (CF violation)

Why a sibling package rather than a module inside `feynman_prm/`: the same reason
`pqm_baseline/` is one (see its README §3). The objective is a REPLACEMENT for the method's
loss set, not an option within it, and `config/default.yaml`'s `losses:` block is strict-parsed
-- a `qrl:` block there would have to be declared on every Feynman run that never reads it.
Entry is `python -m qrl_prm.train`; there is no `scripts/*.py` entry point.

    python -m qrl_prm.train --set run.name=qrl_iqe --set distance.variant=iqe
    python -m feynman_prm.train_goal_head --checkpoint runs/qrl_iqe/final
    python -m feynman_prm.eval.processbench --checkpoint runs/qrl_iqe/phase2/final
    python -m qrl_prm.report --qrl runs/qrl_iqe/phase2/final \
        --baseline runs/abl_cf_only/phase2/final
"""

from __future__ import annotations

__all__ = ["QRLConfig", "load_qrl_config"]


def __getattr__(name: str):     # lazy, so importing the package costs no torch
    if name in __all__:
        from .config import QRLConfig, load_qrl_config

        return {"QRLConfig": QRLConfig, "load_qrl_config": load_qrl_config}[name]
    raise AttributeError(name)
