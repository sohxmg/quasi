"""The PQM-only knobs, strict-parsed exactly like `feynman_prm/config.py` (old bug B4: a
config value silently ignored and a library default used instead).

**Everything else comes from `config/default.yaml`, unchanged.** That is the entire point of
this baseline: the dataset, the selection, the seed, the batch stream, the tokenisation, the
backbone, the optimiser, the schedule and the eval protocol are shared by construction, so
the only difference between the PQM row and the Feynman-PRM row of the paper's table is the
head and the objective.

> **Naming trap.** `losses.zeta` in `config/default.yaml` is Feynman's (3) `L_T` backup
> weight (0.05 TMD-faithful, 0.1 in the shipped runs) and has NOTHING to do with PQM's ζ,
> which is the negative-reward offset in the Q-ranking loss (4, `train_main.py:181`). They
> live in different files and `train.py` logs both side by side at launch.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

# Reusing the strict parser rather than re-typing it: unknown key -> hard error with the
# closest legal name, wrong type -> hard error with the path. `_resolve_annotations` is
# required because `from __future__ import annotations` makes every `f.type` a string.
from feynman_prm.config import ConfigError, _build, _resolve_annotations, apply_override


@dataclass(frozen=True)
class PQMConfig:
    """PQM's own defaults, from `Process_Q_Model/train_main.py:178-181`."""

    # `train_main.py:181`: `--zeta`, default 4. The negative offset: a negative step is
    # scored `exp(r + zeta)` against the positives' `exp(r)`, so the loss drives negatives
    # below `-zeta` and positives above 0. Those two ABSOLUTE anchors are what make a single
    # global tau meaningful across questions (see eval_processbench.py).
    zeta: float = 4.0

    # `train_main.py:182-183`: {'rank', 'orm', 'mse', 'bce'}. Decided with the human before
    # planning: ONE run, `rank`, PQM's own default and the paper's method. The other three
    # are pointwise heads with a different objective entirely and are not ported.
    loss_type: str = "rank"

    # `value_model.py:27-33`: Qwen2's config has no `summary_dropout_prob`, so TRL's own
    # default 0.1 applies -- which is what PQM trained with on deepseek-math-7b-base too.
    head_dropout: float = 0.1

    # `zero` gives an EXACTLY predictable init loss for the launch check (train.py's
    # `launch/init_values`), and removes an `exp(r + zeta)` overflow risk on Qwen hiddens
    # whose massive-activation channels run O(100). `default` reproduces PQM's plain
    # `nn.Linear` init; the launch log prints the reward distribution either way.
    head_init: str = "zero"

    # `from_z`: labels are derived from the parquet's `z` (`label_k = z == -1 or k < z`).
    # The parquet stores `z`, not the label vector, and this is exactly how Feynman's (5) and
    # (6) treat the same trajectories. It MONOTONISES the 1.48% of trajectories with a
    # False -> True recovery. Recorded and reversible -- see README.md.
    label_source: str = "from_z"

    def __post_init__(self) -> None:
        if self.loss_type != "rank":
            raise ConfigError(
                f"pqm.loss_type: {self.loss_type!r} is not ported. Only 'rank' -- PQM's own "
                "default and the paper's method -- is implemented here (loss.py). The "
                "pointwise heads ('orm', 'mse', 'bce') are a different objective and were "
                "not part of the decided scope."
            )
        if self.head_init not in ("zero", "default"):
            raise ConfigError(f"pqm.head_init: {self.head_init!r} is not one of ('zero', 'default')")
        if self.label_source not in ("from_z", "raw"):
            raise ConfigError(
                f"pqm.label_source: {self.label_source!r} is not one of ('from_z', 'raw')"
            )
        if self.label_source == "raw":
            raise ConfigError(
                "pqm.label_source: 'raw' needs a `labels` column in sequences.parquet, which "
                "it does not have (prepare_data.py + sequence_cache + SequenceRow, ~20 lines, "
                "and a re-run of prepare_data.py). Not done: it changes a SHARED artifact -- "
                "the same parquet both runs read -- for a 1.48% effect. See README.md."
            )
        if not 0.0 <= self.head_dropout < 1.0:
            raise ConfigError(f"pqm.head_dropout must be in [0, 1), got {self.head_dropout}")
        if self.zeta <= 0.0:
            raise ConfigError(
                f"pqm.zeta must be > 0, got {self.zeta}. It is the offset that pushes "
                "negatives below `-zeta`; at 0 the loss loses its lower anchor entirely."
            )

    # ---- derived ----

    @property
    def natural_tau_reward(self) -> float:
        """The midpoint of the loss's two absolute anchors, IN REWARD UNITS: `-zeta/2`.

        The ranking loss puts positives above the virtual `e^0 = 1` slot (`r > 0`) and
        negatives below `-zeta` (`r + zeta < 0`), so their midpoint is `-zeta/2` -- `-2.0` at
        `zeta = 4`. On the NEGATED scale the eval works in (`delta = -r`, so higher = worse,
        like Feynman's Delta) it is `+zeta/2`.

        A CHECK, never a constraint, in the same spirit as §9.2's 0.347: tau is still
        whatever maximises val F1. A fitted tau far from this one means the loss's absolute
        anchors did not take, and a single global threshold is then not doing what it looks
        like it is doing.
        """
        return -self.zeta / 2.0

    @property
    def natural_tau_delta(self) -> float:
        """`natural_tau_reward` on the eval's negated scale: `+zeta/2`, `+2.0` at zeta 4."""
        return self.zeta / 2.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["_derived"] = {
            "natural_tau_reward": self.natural_tau_reward,
            "natural_tau_delta": self.natural_tau_delta,
        }
        path.write_text(yaml.safe_dump(payload, sort_keys=False))


_resolve_annotations(PQMConfig)


def load_pqm_config(path: str | Path, overrides: list[str] | None = None) -> PQMConfig:
    """Parse `pqm_baseline/config/pqm.yaml` strictly. `overrides` are `key=value` strings
    with the `pqm.` prefix ALREADY STRIPPED -- `train.py` partitions `--set` so that
    `--set pqm.zeta=8` lands here and `--set run.name=x` lands on the Feynman config."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for assignment in overrides or []:
        apply_override(raw, assignment)
    return _build(PQMConfig, raw, "pqm")


def pqm_config_from_dict(data: dict[str, Any]) -> PQMConfig:
    """Same strictness, from an already-parsed mapping (checkpoints, tests)."""
    return _build(PQMConfig, {k: v for k, v in data.items() if k != "_derived"}, "pqm")


def load_pqm_config_from_checkpoint(checkpoint_dir: str | Path) -> PQMConfig:
    return pqm_config_from_dict(yaml.safe_load((Path(checkpoint_dir) / "pqm.yaml").read_text()))


def split_overrides(overrides: list[str]) -> tuple[list[str], list[str]]:
    """Partition `--set` assignments into (feynman, pqm). A `pqm.`-prefixed key has the
    prefix stripped; everything else passes through untouched.

    This is what keeps `config/default.yaml` free of a `pqm:` block it would have to declare
    (it is strict-parsed, so an undeclared key there is a hard error) while still giving the
    one-line re-run `--set pqm.zeta=8` the plan's risk section asks for.
    """
    feynman, pqm = [], []
    for assignment in overrides:
        if "=" not in assignment:
            raise ConfigError(f"override {assignment!r} is not of the form key.path=value")
        key = assignment.split("=", 1)[0].strip()
        if key == "pqm" or key.startswith("pqm."):
            pqm.append(assignment.split(".", 1)[1] if "." in key else assignment)
        else:
            feynman.append(assignment)
    return feynman, pqm


PQM_YAML = Path(__file__).resolve().parent / "config" / "pqm.yaml"


def field_names() -> tuple[str, ...]:
    return tuple(f.name for f in fields(PQMConfig))
