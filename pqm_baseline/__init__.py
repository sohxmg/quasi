"""PQM (Process Q-value Model, Li & Li, ICLR 2025) as a baseline for Feynman-PRM.

**This is our re-implementation of PQM's head and objective under Feynman-PRM's exact
training conditions -- not PQM's published numbers.** Say so wherever the row appears; PQM's
own paper reports Best-of-N on a 7B full finetune and never reports ProcessBench.

`Process_Q_Model/` (the authors' released code) stays UNTOUCHED as the vendored reference,
the same treatment `../tmd-release/` and `../CRM/` get. The loss is ported here with line
citations, never imported.

Why a sibling package rather than a module inside `feynman_prm/`:
`tests/test_grep_invariants.py::test_no_value_head_anywhere` scans exactly
`feynman_prm/**/*.py` + `scripts/*.py`. A value head is the one thing that guard exists to
keep out of the METHOD, and it is the defining feature of this BASELINE. Living outside the
scanned tree keeps the guard honest instead of renaming a head to dodge it. For the same
reason there is no `scripts/*.py` entry point -- entry is `python -m pqm_baseline.train`.

    python -m pqm_baseline.train --set run.name=pqm_zeta4
    python -m pqm_baseline.eval_processbench --checkpoint runs/pqm_zeta4/final
    python -m pqm_baseline.report --pqm runs/pqm_zeta4/final \
        --feynman runs/abl_cf_only/phase2/final
"""

from __future__ import annotations

__all__ = ["PQMConfig", "load_pqm_config"]


def __getattr__(name: str):     # lazy, so importing the package costs no torch
    if name in __all__:
        from .config import PQMConfig, load_pqm_config

        return {"PQMConfig": PQMConfig, "load_pqm_config": load_pqm_config}[name]
    raise AttributeError(name)
