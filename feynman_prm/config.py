"""Strict config loader.

Old bug B4 was "config value silently ignored, library default used instead". So every key
is typed, and an unknown or misspelled key is a hard error with the offending path and the
closest legal name.

Three quantities are DERIVED from `discount` and must never be set independently (§7.8):

    neg_log_gamma = -log(discount)          per-good-step cost
    step_margin   = margin_steps * neg_log_gamma        (5) L_step's m      §7.6.4
    clip_t        = clip_t_steps * neg_log_gamma        (3) L_T's t         §7.4.3
"""

from __future__ import annotations

import dataclasses
import difflib
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Union, get_args, get_origin

import yaml


class ConfigError(ValueError):
    """Raised for an unknown key, a bad type, or a value outside its legal set."""


# --------------------------------------------------------------------------------------
# the schema
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    name: str = "phase1"
    out_dir: str = "runs"
    seed: int = 42


@dataclass(frozen=True)
class LoraConfig:
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0


@dataclass(frozen=True)
class ModelConfig:
    name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    lora: LoraConfig = field(default_factory=LoraConfig)
    gradient_checkpointing: bool = True
    attn_implementation: str = "sdpa"

    def __post_init__(self) -> None:
        _one_of("model.attn_implementation", self.attn_implementation,
                ("sdpa", "eager", "flash_attention_2"))


@dataclass(frozen=True)
class HeadsConfig:
    latent_dim: int = 512
    hidden_dims: tuple[int, ...] = (512, 512, 512)
    ensemble: bool = False
    action_pool: str = "mean"

    def __post_init__(self) -> None:
        _one_of("heads.action_pool", self.action_pool, ("mean", "attention"))
        if self.ensemble:
            raise ConfigError(
                "heads.ensemble: not implemented. The knob exists because §16.6 leaves the "
                "decision open, but the human's call was OFF and wiring 2 members through "
                "every loss (TMD averages the critic loss over members and takes min at read "
                "time, tmd.py:170-173) is not built. Keep it false."
            )


@dataclass(frozen=True)
class DistanceConfig:
    variant: str = "full_mrn"
    components: int = 8

    def __post_init__(self) -> None:
        _one_of("distance.variant", self.variant, ("full_mrn", "asym_only", "iqe"))
        if self.components < 1:
            raise ConfigError("distance.components must be >= 1")


@dataclass(frozen=True)
class DataConfig:
    dir: str = "data/processed"
    max_len: int = 1024
    sep_token: str = "\n"
    prompt_format: str = "raw"
    n_questions: int = 23000
    n_val_questions: int = 2000

    def __post_init__(self) -> None:
        _one_of("data.prompt_format", self.prompt_format, ("raw", "chat"))


@dataclass(frozen=True)
class SamplingConfig:
    sequences_per_micro_batch: int = 56
    max_correct_per_question: int = 4
    max_incorrect_per_question: int = 3
    nce_mask_same_traj: bool = False
    group_by_length: bool = True

    def __post_init__(self) -> None:
        if self.max_correct_per_question < 1 or self.max_incorrect_per_question < 1:
            raise ConfigError(
                "sampling caps must be >= 1: L_step needs one correct trajectory for its "
                "terminal goal g and one incorrect for its error index z (§7.6)"
            )
        if self.sequences_per_micro_batch < (
            self.max_correct_per_question + self.max_incorrect_per_question
        ):
            raise ConfigError(
                "sequences_per_micro_batch is smaller than one question's allocation, so no "
                "question would ever fit (PLAN 'Core design decisions' 4)"
            )


@dataclass(frozen=True)
class ActionInvarianceConfig:
    mode: str = "diagonal"

    def __post_init__(self) -> None:
        _one_of("losses.action_invariance.mode", self.mode,
                ("diagonal", "grid_within_question", "grid_batch"))


@dataclass(frozen=True)
class StepLossConfig:
    margin_steps: float = 2.0
    pairing: str = "boundary"
    exclude_recovery: bool = False

    def __post_init__(self) -> None:
        _one_of("losses.step_loss.pairing", self.pairing,
                ("boundary", "position_corrected", "same_index"))


@dataclass(frozen=True)
class BackupConfig:
    clip_t_steps: float = 28.5
    diag_backup: float = 0.5
    goal_scope_ratio: float = 1.0
    stopgrad_psi_backup: bool = False

    def __post_init__(self) -> None:
        _in_unit("losses.backup.diag_backup", self.diag_backup)
        _in_unit("losses.backup.goal_scope_ratio", self.goal_scope_ratio)


@dataclass(frozen=True)
class LossesConfig:
    lambda_nce: float = 1.0
    lambda_i: float = 1.0
    zeta: float = 0.05
    lambda_cf: float = 0.0
    lambda_step: float = 1.0
    nce_temperature: float = 1.0
    action_invariance: ActionInvarianceConfig = field(default_factory=ActionInvarianceConfig)
    step_loss: StepLossConfig = field(default_factory=StepLossConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    stopgrad_phi_invariance: bool = False

    def __post_init__(self) -> None:
        if self.nce_temperature <= 0:
            raise ConfigError("losses.nce_temperature must be > 0")
        if self.lambda_step > 0 and self.action_invariance.mode != "diagonal":
            raise ConfigError(
                "losses.action_invariance.mode != 'diagonal' with lambda_step > 0. The grid "
                "modes drive phi(s,a) -> psi(s) (§16.3), which tightens the triangle bound to "
                "an equality and pins Delta_{z+1} at -log gamma good step or bad, so L_step "
                "cannot move (§7.6.5). Set lambda_step: 0.0 if you are deliberately "
                "reproducing that failure."
            )


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1
    grad_accum: int = 2
    lr_backbone: float = 9.0e-6
    lr_heads: float = 3.0e-4
    schedule: str = "cosine"
    warmup_ratio: float = 0.03
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    bf16: bool = True
    log_every: int = 10
    save_every: int = 250
    max_steps: Optional[int] = None

    def __post_init__(self) -> None:
        _one_of("train.schedule", self.schedule, ("cosine", "constant"))
        if self.grad_accum < 1:
            raise ConfigError("train.grad_accum must be >= 1")


@dataclass(frozen=True)
class GoalHeadConfig:
    lr: float = 1.0e-3
    epochs: int = 20
    batch_size: int = 512
    max_terminals_per_question: int = 8


@dataclass(frozen=True)
class EvalConfig:
    subsets: tuple[str, ...] = ("gsm8k", "math", "olympiadbench", "omnimath")
    max_len: int = 2048
    batch_sequences: int = 16
    skyline: bool = True

    def __post_init__(self) -> None:
        for s in self.subsets:
            _one_of("eval.subsets", s, ("gsm8k", "math", "olympiadbench", "omnimath"))


@dataclass(frozen=True)
class LogConfig:
    wandb: bool = False
    wandb_project: str = "feynman-prm"


@dataclass(frozen=True)
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    discount: float = 0.5
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    losses: LossesConfig = field(default_factory=LossesConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    goal_head: GoalHeadConfig = field(default_factory=GoalHeadConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    log: LogConfig = field(default_factory=LogConfig)

    def __post_init__(self) -> None:
        if not 0.0 < self.discount < 1.0:
            raise ConfigError(f"discount must be in (0, 1), got {self.discount}")
        per = self.heads.latent_dim / self.distance.components
        if per != int(per) or int(per) % 2 != 0:
            raise ConfigError(
                f"heads.latent_dim ({self.heads.latent_dim}) must be divisible by "
                f"distance.components ({self.distance.components}) with an EVEN quotient: "
                "mrn_distance splits each component in half (tmd.py:36-39)"
            )

    # ---- derived. Never set these independently (§7.4.3, §7.6.4, §7.8) ----

    @property
    def neg_log_gamma(self) -> float:
        """Per-good-step cost, -log(discount). 0.69315 at 0.5, 0.35667 at 0.7."""
        return -math.log(self.discount)

    @property
    def step_margin(self) -> float:
        """(5) L_step's m. 1.38629 at discount 0.5 with margin_steps 2.0."""
        return self.losses.step_loss.margin_steps * self.neg_log_gamma

    @property
    def clip_t(self) -> float:
        """(3) L_T's LINEX clip t. 19.75 at discount 0.5 with clip_t_steps 28.5."""
        return self.losses.backup.clip_t_steps * self.neg_log_gamma

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["_derived"] = {
            "neg_log_gamma": self.neg_log_gamma,
            "step_margin": self.step_margin,
            "clip_t": self.clip_t,
        }
        path.write_text(yaml.safe_dump(payload, sort_keys=False))


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def _one_of(path: str, value: Any, legal: tuple[Any, ...]) -> None:
    if value not in legal:
        raise ConfigError(f"{path}: {value!r} is not one of {legal}")


def _in_unit(path: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{path}: {value!r} must be in [0, 1]")


def _coerce(path: str, value: Any, typ: Any) -> Any:
    """Coerce a YAML scalar/list to the annotated type, or raise."""
    origin = get_origin(typ)

    if origin is Union:  # only Optional[X] is used in this schema
        args = [a for a in get_args(typ) if a is not type(None)]
        if value is None:
            return None
        return _coerce(path, value, args[0])

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        args = get_args(typ)
        elem = args[0]
        return tuple(_coerce(f"{path}[{i}]", v, elem) for i, v in enumerate(value))

    if is_dataclass(typ):
        return _build(typ, value, path)

    if typ is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a bool, got {value!r}")
        return value
    if typ is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an int, got {value!r}")
        return value
    if typ is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a float, got {value!r}")
        return float(value)
    if typ is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a str, got {value!r}")
        return value

    raise ConfigError(f"{path}: unsupported annotation {typ!r} (config.py needs updating)")


def _build(cls: type, data: Any, path: str = "") -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path or '<root>'}: expected a mapping, got {type(data).__name__}")

    known = {f.name: f.type for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        full = f"{path}.{key}" if path else key
        if key not in known:
            hint = difflib.get_close_matches(str(key), list(known), n=1)
            suffix = f" (did you mean {hint[0]!r}?)" if hint else ""
            raise ConfigError(
                f"unknown config key {full!r}{suffix}. Legal keys here: "
                f"{sorted(known)}. Unknown keys are a hard error on purpose - old bug B4 was a "
                "silently ignored config value."
            )
        kwargs[key] = _coerce(full, value, known[key])
    return cls(**kwargs)


def _resolve_annotations(cls: type) -> None:
    """`from __future__ import annotations` makes f.type a string; resolve once, in place."""
    import typing

    hints = typing.get_type_hints(cls)
    for f in fields(cls):
        f.type = hints[f.name]
        if is_dataclass(f.type):
            _resolve_annotations(f.type)
        elif get_origin(f.type) is Union:
            for arg in get_args(f.type):
                if is_dataclass(arg):
                    _resolve_annotations(arg)


_resolve_annotations(Config)


def apply_override(data: dict[str, Any], assignment: str) -> None:
    """Apply one `a.b.c=value` override in place. `value` is parsed as YAML."""
    if "=" not in assignment:
        raise ConfigError(f"override {assignment!r} is not of the form key.path=value")
    key, raw = assignment.split("=", 1)
    parts = key.strip().split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override {key!r} descends into a non-mapping")
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Parse a YAML file into a Config, strictly. `overrides` are `key.path=value` strings."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for assignment in overrides or []:
        apply_override(raw, assignment)
    return _build(Config, raw)


def config_from_dict(data: dict[str, Any]) -> Config:
    """Same strictness, from an already-parsed mapping (used by tests and checkpoints)."""
    data = {k: v for k, v in data.items() if k != "_derived"}
    return _build(Config, data)
