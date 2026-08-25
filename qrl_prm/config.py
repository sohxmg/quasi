"""The QRL-only knobs, strict-parsed exactly like `feynman_prm/config.py` (old bug B4: a
config value silently ignored and a library default used instead).

**Everything else comes from `config/default.yaml`, unchanged.** That is the entire point of
this variant: the dataset, the selection, the seed, the batch stream, the tokenisation, the
backbone, the LoRA, the optimiser, the schedule, the CF corpus and the eval protocol are
shared by construction, so the only difference between the QRL row and the Feynman-PRM row is
the objective (and, deliberately, the head -- see README.md §2).

> **Naming trap.** `losses.*` in `config/default.yaml` -- `lambda_cf`, `lambda_step`,
> `lambda_good`, `zeta`, ... -- is Feynman-PRM's fixed-weight loss set and **not one of those
> keys is read by anything in `qrl_prm/`**. QRL's weights are DUAL VARIABLES: they are
> learned, they live on `raw_lambda_local` / `raw_lambda_cf` in `lagrange.py`, and their
> initial values are `init_lagrange` here. `train.py` refuses to start if a `--set losses.*`
> override is passed, because that override would be silently inert -- exactly bug B4's shape.
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
class QRLConfig:
    """QRL's own defaults where QRL has one, and a derivation where it does not.

    Line citations are into `quasimetric-rl/quasimetric_rl/modules/quasimetric_critic/`.
    """

    # `losses/global_push.py:26`. QRL's comment: "should be greater than most GT distances
    # between sampled pairs". QRL's online default is 15 and its offline maze2d config uses
    # 500 for a horizon of hundreds of steps; our horizon is T ~ 10 reasoning steps at
    # step_cost 1, and 25 is also the `clip_t_gain = 20` precedent's order. Read
    # `qrl/push_saturated_frac` at launch: if a large fraction of pairs already sit above the
    # offset, the push term is flat there and the offset is too small.
    softplus_offset: float = 25.0

    # `losses/global_push.py:23`. Smaller => smoother. QRL's default, unchanged.
    softplus_beta: float = 0.1

    # `losses/local_constraint.py:29`. One observed reasoning step is the unit of the metric,
    # so d(psi_z, psi_{z+1}) -> 1.0 is what `qrl/local_dist_mean` should pin to. THAT is the
    # ruler IMPLEMENTATION.md §9 says decays under the fixed-weight losses, and here it is a
    # constraint rather than a target a soft loss trades away.
    step_cost: float = 1.0

    # `losses/local_constraint.py:22`. QRL's default. The constraint is one-sided
    # (`(d - step_cost).relu()`): a transition SHORTER than one step is free.
    epsilon_local: float = 0.25

    # No QRL counterpart -- CF invariance is this project's term. It is the radius of the
    # equivalence ball two meaning-preserving wordings of one step may sit in, and it must be
    # far below the ruler the verdict is read against: the old margin ruler is
    # `2 * log 2 ~= 1.386` (§7.6), so 0.2 keeps a paraphrase ~7x too small to flip a step's
    # verdict. Both directions are constrained, so by the triangle inequality the class
    # diameter is bounded by 2 * epsilon_cf.
    epsilon_cf: float = 0.2

    # `losses/local_constraint.py:31`. QRL's default, used for BOTH multipliers.
    init_lagrange: float = 0.01

    # `losses/__init__.py:47`: `lagrange_mult_optim: AdamWSpec.Conf(lr=1e-2)`. Constant --
    # the dual variable must be free to track the constraint for the whole run, so it does
    # NOT ride the primal's cosine decay.
    lagrange_lr: float = 1e-2

    # Weight on the CF-NEGATIVE push term: `softplus(offset - d(psi(prefix + neg), psi_g))` over
    # same-question goal columns. Broken states are pushed AWAY from goals, in the direction
    # eval queries (broken state as source). 0.0 makes the term an exact zero with its keys
    # still logged.
    cf_neg_push_weight: float = 1.0

    # Padded-token budget for the CF variant forward: `sequences x longest`, the shape the GPU
    # actually pays for. **This is the knob `data.cf_max_per_batch` used to be.** Under
    # `cf_phi` a variant cost an embedding lookup and an MLP; under `cf_encode.py` it is a full
    # sequence through the backbone, so 12 examples x ~8 rewrites is ~96 sequences against a
    # main batch of ~51. 16384 is half `sampling.max_padded_tokens`, which is what the main
    # batch is capped at. Whole EXAMPLES are dropped off the tail of the seeded order once the
    # budget is met -- unbiased in length, and counted in `cf/examples_dropped_budget`.
    cf_encode_max_tokens: int = 16384

    # 0 = build the (S, C) push matrix in one shot. A positive value splits it into chunks of
    # that many GOAL COLUMNS and accumulates the mean exactly (weighted by each chunk's column
    # count). It lowers the TRANSIENT peak of the distance's internals -- IQE materialises
    # several (S, C, D/k, 2k) fp32 tensors -- not the autograd graph, which is the same size
    # either way. Reach for it only if `launch/memory_probe` says the card cannot take the
    # one-shot form; tests pin chunked == unchunked.
    push_chunk_cols: int = 0

    def __post_init__(self) -> None:
        if self.softplus_offset <= 0.0:
            raise ConfigError(
                f"qrl.softplus_offset must be > 0, got {self.softplus_offset}. It is the point "
                "the push term stops rewarding distance; at 0 there is no push at all."
            )
        if self.softplus_beta <= 0.0:
            raise ConfigError(f"qrl.softplus_beta must be > 0, got {self.softplus_beta}")
        if self.step_cost <= 0.0:
            raise ConfigError(
                f"qrl.step_cost must be > 0, got {self.step_cost}. It is the unit the whole "
                "metric is expressed in -- `qrl/local_dist_mean` is read against it."
            )
        if self.epsilon_local <= 0.0:
            raise ConfigError(
                f"qrl.epsilon_local must be > 0, got {self.epsilon_local}. At 0 the constraint "
                "is unsatisfiable (the violation can never go negative) and the multiplier "
                "diverges instead of stabilising."
            )
        if self.epsilon_cf <= 0.0:
            raise ConfigError(
                f"qrl.epsilon_cf must be > 0, got {self.epsilon_cf}. Same failure as "
                "epsilon_local: a zero-radius ball is unreachable and lambda_cf would climb "
                "for the whole run with nothing to report."
            )
        if self.init_lagrange <= 0.0:
            raise ConfigError(
                f"qrl.init_lagrange must be > 0, got {self.init_lagrange}. softplus_inv is "
                "undefined at 0, and a multiplier that starts at exactly 0 has no gradient "
                "path back into the primal on the first step."
            )
        if self.lagrange_lr <= 0.0:
            raise ConfigError(f"qrl.lagrange_lr must be > 0, got {self.lagrange_lr}")
        if self.cf_neg_push_weight < 0.0:
            raise ConfigError(
                f"qrl.cf_neg_push_weight must be >= 0, got {self.cf_neg_push_weight}. A "
                "negative weight would PULL broken states towards goals."
            )
        if self.cf_encode_max_tokens <= 0:
            raise ConfigError(
                f"qrl.cf_encode_max_tokens must be > 0, got {self.cf_encode_max_tokens}. At 0 "
                "the budget would drop every CF example after the first and the CF CONSTRAINT "
                "-- one of the two constraints this objective is defined by -- would train on "
                "one class per batch."
            )
        if self.push_chunk_cols < 0:
            raise ConfigError(
                f"qrl.push_chunk_cols must be >= 0 (0 = one shot), got {self.push_chunk_cols}"
            )
        if self.epsilon_cf >= 2.0 * 0.6931471805599453:
            raise ConfigError(
                f"qrl.epsilon_cf = {self.epsilon_cf} is at or above the old margin ruler "
                "2*log 2 = 1.3863 (§7.6). An equivalence ball that wide lets a paraphrase move "
                "a step across the decision boundary, which is the exact property (4) L_CF "
                "exists to prevent. If this is deliberate, say so in the run's README row."
            )

    # ---- derived ----

    @property
    def local_target(self) -> float:
        """The constraint's right-hand side, `epsilon_local^2`. `qrl/local_violation` is
        `mean(relu(d - step_cost)^2) - this`, so its SIGN is the readable quantity: negative
        means the constraint is satisfied and lambda_local should be falling."""
        return self.epsilon_local**2

    @property
    def cf_target(self) -> float:
        """`epsilon_cf^2`. Same reading as `local_target`, for `qrl/cf_violation`."""
        return self.epsilon_cf**2

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["_derived"] = {
            "local_target": self.local_target,
            "cf_target": self.cf_target,
        }
        path.write_text(yaml.safe_dump(payload, sort_keys=False))


_resolve_annotations(QRLConfig)


def load_qrl_config(path: str | Path, overrides: list[str] | None = None) -> QRLConfig:
    """Parse `qrl_prm/config/qrl.yaml` strictly. `overrides` are `key=value` strings with the
    `qrl.` prefix ALREADY STRIPPED -- `train.py` partitions `--set` so that
    `--set qrl.cf_encode_max_tokens=8192` lands here and `--set run.name=x` lands on the
    Feynman config."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for assignment in overrides or []:
        apply_override(raw, assignment)
    return _build(QRLConfig, raw, "qrl")


def qrl_config_from_dict(data: dict[str, Any]) -> QRLConfig:
    """Same strictness, from an already-parsed mapping (checkpoints, tests)."""
    return _build(QRLConfig, {k: v for k, v in data.items() if k != "_derived"}, "qrl")


def load_qrl_config_from_checkpoint(checkpoint_dir: str | Path) -> QRLConfig:
    return qrl_config_from_dict(yaml.safe_load((Path(checkpoint_dir) / "qrl.yaml").read_text()))


def split_overrides(overrides: list[str]) -> tuple[list[str], list[str]]:
    """Partition `--set` assignments into (feynman, qrl). A `qrl.`-prefixed key has the prefix
    stripped; everything else passes through untouched.

    This is what keeps `config/default.yaml` free of a `qrl:` block it would have to declare
    (it is strict-parsed, so an undeclared key there is a hard error) while still giving the
    one-line ablations -- `--set qrl.cf_encode_max_tokens=8192`,
    `--set qrl.cf_neg_push_weight=0` -- a
    place to land.
    """
    feynman, qrl = [], []
    for assignment in overrides:
        if "=" not in assignment:
            raise ConfigError(f"override {assignment!r} is not of the form key.path=value")
        key = assignment.split("=", 1)[0].strip()
        if key == "qrl" or key.startswith("qrl."):
            qrl.append(assignment.split(".", 1)[1] if "." in key else assignment)
        else:
            feynman.append(assignment)
    return feynman, qrl


QRL_YAML = Path(__file__).resolve().parent / "config" / "qrl.yaml"


def field_names() -> tuple[str, ...]:
    return tuple(f.name for f in fields(QRLConfig))
