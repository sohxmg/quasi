"""Strict config loader.

Old bug B4 was "config value silently ignored, library default used instead". So every key
is typed, and an unknown or misspelled key is a hard error with the offending path and the
closest legal name.

Four quantities are DERIVED from `discount` and must never be set independently (§7.8):

    neg_log_gamma = -log(discount)          per-good-step cost
    step_margin   = margin_steps * neg_log_gamma        (5) L_step's m      §7.6.4
    good_margin   = -margin_steps * neg_log_gamma       (6) L_good's c      §7.12  NEGATIVE
    clip_t        = log(clip_t_gain / discount)         (3) L_T's t         §7.4.3

`step_margin` and `good_margin` read DIFFERENT `margin_steps` keys (`losses.step_loss` and
`losses.good_loss`) and carry opposite signs. `m` is a positive floor on the error step's
Delta; `c` is the negative target a good step's Delta must not exceed.

`clip_t` is derived from `discount` but it does NOT scale with `neg_log_gamma` -- it moves the
other way, by `log(1/gamma)`. See §7.4.3: `t` bounds an exponent, not a step count.
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
    # (4) L_CF's data, attached to the MAIN batch (§7.5.3 option (b), 2026-08-15). A glob of
    # generator output; "" disables the CF path entirely and (4) stays an exact zero.
    cf_glob: str = ""
    # Cap on CF examples attached per micro-batch. A batch that happens to hold many
    # CF-covered questions must not swing L_CF's magnitude, and the cap is what keeps the
    # per-step cost flat. 12 covers the ~9.3/batch an epoch needs at 27,114 examples over
    # ~2,920 micro-batches, with headroom for the uneven ones.
    cf_max_per_batch: int = 12

    def __post_init__(self) -> None:
        _one_of("data.prompt_format", self.prompt_format, ("raw", "chat"))
        if self.cf_max_per_batch < 0:
            raise ConfigError("data.cf_max_per_batch must be >= 0")


@dataclass(frozen=True)
class SamplingConfig:
    sequences_per_micro_batch: int = 56
    max_padded_tokens: int = 32768
    max_correct_per_question: int = 4
    max_incorrect_per_question: int = 3
    nce_mask_same_traj: bool = False
    # **§16.4's false negative -- the surgical version of the switch above, and the one to
    # run.** Drops only the same-trajectory rows BETWEEN the sampled positive and the goal:
    # goal at s_6 with positive phi_3 masks phi_4/phi_5/phi_6, which are all nearer the goal
    # than the positive is. Rows earlier than the positive stay (honest hard negatives). Also
    # dissolves the 29.6% duplicate-column contradiction. ~0.64 rows of ~347. §9.9.2.
    nce_mask_nearer_same_traj: bool = False
    # Drops SAME-QUESTION, CORRECT-trajectory rows from the L_NCE negative pool -- they are
    # false negatives (two correct solutions to one problem end in the same place). A targeted
    # exception to locked #12, OFF by default, needs sign-off. See losses/nce.py.
    # OVER-BROAD by §16.25(a)'s own admission; prefer `nce_mask_sibling_correct_late`.
    nce_mask_same_question_correct: bool = False
    # §16.25(a)'s position-aware variant: only the LATE states of a sibling correct trajectory,
    # and only against TERMINAL goal columns. §9.9.5.
    nce_mask_sibling_correct_late: bool = False
    # Steps remaining to a row's OWN terminal for it to count as "late". 1 keeps {T, T-1}.
    nce_sibling_late_margin: int = 1
    group_by_length: bool = True

    def __post_init__(self) -> None:
        if self.max_padded_tokens < 1:
            raise ConfigError("sampling.max_padded_tokens must be >= 1")
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
class GoodLossConfig:
    """(6) L_good internals (§7.12). `margin_steps` here is the TARGET, not a slack."""

    margin_steps: float = 1.0
    # The code default stays `relu` -- §7.12's ablation is the only measurement of any form
    # and it is relu-vs-softplus. **config/default.yaml ships `relu_squared` since
    # 2026-08-04** (the same split as `lambda_good`, whose code default is 0.0 and whose
    # shipped value is 1.0): a linear hinge on a MEAN is indifferent between many small
    # violations and one large one, and mid-run that is exactly what happened -- the bulk
    # moved (Delta mean +0.392 -> -0.412) while the tail ran away (frac_above_natural
    # 0.070 -> 0.16, p99 0.86 -> 2.43, delta_max 7.58). The square prices a violator by HOW
    # FAR out it is and keeps relu's two load-bearing properties: exactly zero below `c` and
    # exactly zero gradient AT `c`. §7.12, losses/good.py.
    form: str = "relu"
    include_incorrect_prefix: bool = True
    detach_goal: bool = False
    warmup_steps: int = 100

    def __post_init__(self) -> None:
        _one_of("losses.good_loss.form", self.form, ("relu", "relu_squared", "softplus"))
        if self.warmup_steps < 0:
            raise ConfigError(
                f"losses.good_loss.warmup_steps must be >= 0, got {self.warmup_steps}. "
                "0 means full weight from step 1, not 'no L_good'."
            )
        if self.margin_steps <= 0.0:
            # `cfg.good_margin` is -margin_steps * neg_log_gamma and it MUST come out
            # negative: it is where a good step is supposed to land, -0.693 at discount 0.5.
            # A non-positive margin_steps puts c at 0 or above, which trains good steps to
            # move AWAY from the goal -- and that converges cleanly, so no curve would show
            # it. The sign is checked here because it cannot be checked later.
            raise ConfigError(
                f"losses.good_loss.margin_steps must be > 0, got {self.margin_steps}. It is a "
                "step count that is NEGATED into the target c = -margin_steps * (-log gamma) "
                "= -0.693 at discount 0.5. Do not pre-negate it here (§7.12)."
            )


@dataclass(frozen=True)
class BackupConfig:
    clip_t_gain: float = 20.0
    diag_backup: float = 0.5
    goal_scope_ratio: float = 1.0
    stopgrad_psi_backup: bool = False

    def __post_init__(self) -> None:
        _in_unit("losses.backup.diag_backup", self.diag_backup)
        _in_unit("losses.backup.goal_scope_ratio", self.goal_scope_ratio)
        if self.clip_t_gain <= 1.0:
            raise ConfigError(
                f"losses.backup.clip_t_gain must be > 1, got {self.clip_t_gain}. It is the cap "
                "on the LINEX exponential term gamma*exp(t); at <= 1 the guard clips below the "
                "loss's own minimiser (delta = -log gamma) and the backup can never converge."
            )


@dataclass(frozen=True)
class LossesConfig:
    lambda_nce: float = 1.0
    lambda_i: float = 1.0
    zeta: float = 0.05
    lambda_cf: float = 0.0
    lambda_step: float = 1.0
    lambda_good: float = 0.0
    # (7) L_term -- §7.13 / §16.26. The CODE default stays 0.0; **config/default.yaml ships
    # 1.0 since 2026-08-15**, the same split `lambda_good` and `good_loss.form` already use.
    # The term is computed and logged at every weight (its diagnostics are what decide whether
    # it is worth training) and `0.0 * L_term` is an exact zero, so the total is bit-identical
    # with and without it at the code default.
    lambda_term: float = 0.0
    # (7)'s SupCon temperature. A CONFIG KEY as of 2026-08-15, not a function argument:
    # §7.13's rule is that whoever raises `lambda_term` picks `tau` in the same change rather
    # than inheriting an unexamined default, and a default nobody has to type is exactly what
    # "unexamined" means. 1.0 -- see config/default.yaml for the arithmetic. `tau` and
    # `lambda` are the same knob here (the CE gradient on distances scales as 1/tau), so
    # moving this moves `lambda_term` by the reciprocal.
    lambda_term_temperature: float = 1.0
    # (4)'s SupCon temperature. A CONFIG KEY as of 2026-08-18 for exactly the reason
    # `lambda_term_temperature` became one: it was UNPLUMBED -- `total.py` called
    # `counterfactual_loss` with no `temperature=`, so `lambda_cf` went 0.0 -> 1.0 in the
    # 2026-08-15 run against a 1.0 nobody picked. §7.13's rule is that whoever turns a term on
    # picks its tau as an explicit key. The CODE default stays 1.0 -- the same code-default /
    # yaml-split `lambda_term` uses -- and **config/default.yaml ships 0.1**; see there for the
    # arithmetic. `tau` and `lambda` are the same knob (the CE gradient on distances scales as
    # 1/tau), so moving this moves `lambda_cf` by the reciprocal.
    lambda_cf_temperature: float = 1.0
    nce_temperature: float = 1.0
    action_invariance: ActionInvarianceConfig = field(default_factory=ActionInvarianceConfig)
    step_loss: StepLossConfig = field(default_factory=StepLossConfig)
    good_loss: GoodLossConfig = field(default_factory=GoodLossConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    stopgrad_phi_invariance: bool = False

    def __post_init__(self) -> None:
        if self.nce_temperature <= 0:
            raise ConfigError("losses.nce_temperature must be > 0")
        if self.lambda_good < 0:
            raise ConfigError("losses.lambda_good must be >= 0")
        if self.lambda_term < 0:
            raise ConfigError("losses.lambda_term must be >= 0")
        if self.lambda_term_temperature <= 0:
            raise ConfigError("losses.lambda_term_temperature must be > 0")
        if self.lambda_cf_temperature <= 0:
            raise ConfigError("losses.lambda_cf_temperature must be > 0")
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
    grad_clip: float = 1.0
    log_every: int = 10
    save_every: int = 250
    max_steps: Optional[int] = None

    def __post_init__(self) -> None:
        _one_of("train.schedule", self.schedule, ("cosine", "constant"))
        if self.grad_accum < 1:
            raise ConfigError("train.grad_accum must be >= 1")
        if self.grad_clip <= 0:
            raise ConfigError("train.grad_clip must be > 0 (0 does not mean 'off')")


@dataclass(frozen=True)
class GoalHeadConfig:
    lr: float = 1.0e-3
    epochs: int = 100
    batch_size: int = 512
    max_terminals_per_question: int = 8
    val_questions: int = 2000


@dataclass(frozen=True)
class EvalConfig:
    subsets: tuple[str, ...] = ("gsm8k", "math", "olympiadbench", "omnimath")
    max_len: int = 2048
    batch_sequences: int = 16
    skyline: bool = True
    # §9.1 step 6. `first_crossing` is the shipped rule and every reported number is that
    # rule -- §9.6.1's prose says `argmax` and is WRONG about what runs (§9.9.1). argmax is
    # worth +0.017 mean F1 on ProcessBench, but **that table is fit on ProcessBench and §9.2
    # forbids choosing on it.** Move this only on a Math-Shepherd-val comparison.
    localisation_rule: str = "first_crossing"

    def __post_init__(self) -> None:
        for s in self.subsets:
            _one_of("eval.subsets", s, ("gsm8k", "math", "olympiadbench", "omnimath"))
        _one_of("eval.localisation_rule", self.localisation_rule,
                ("first_crossing", "argmax"))


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
        one_question = (
            self.sampling.max_correct_per_question + self.sampling.max_incorrect_per_question
        ) * self.data.max_len
        if self.sampling.max_padded_tokens < one_question:
            raise ConfigError(
                f"sampling.max_padded_tokens ({self.sampling.max_padded_tokens}) is below one "
                f"question's worst-case padded cost ({one_question} = "
                f"{self.sampling.max_correct_per_question + self.sampling.max_incorrect_per_question}"
                f" seqs x data.max_len {self.data.max_len}). No question is ever split (PLAN "
                "'Core design decisions' 4), so the cap would be exceeded by single-question "
                "batches and would not bound peak memory at all."
            )
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
    def good_margin(self) -> float:
        """(6) L_good's `c`. **NEGATIVE**: -0.69315 at discount 0.5, margin_steps 1.0 (§7.12).

        It is the target `Delta` of a good step -- the same `-log gamma` (3) `L_T` prices a
        step at -- expressed as the point where `relu` switches on. The sign is the whole
        content of this property: `+0.693` trains good steps to move one step AWAY from the
        goal per step, and it converges just as cleanly, so nothing downstream would catch it.
        Opposite convention to `step_margin`, which is a positive threshold on `Delta_{z+1}`.
        """
        return -self.losses.good_loss.margin_steps * self.neg_log_gamma

    @property
    def clip_t(self) -> float:
        """(3) L_T's LINEX clip t, 3.689 at discount 0.5 with clip_t_gain 20.0 (§7.4.3).

        `t` caps the exponential branch at `gamma*exp(t) = clip_t_gain`, so the steepest
        per-term gradient the backup can put on `Dist` is `clip_t_gain - 1`. Solving for `t`:

            t = log(clip_t_gain / gamma)

        `clip_t_gain = 20` reproduces TMD's bare `t = 3.0` at TMD's OWN `discount = 0.99`
        (0.99*e^3 = 19.9). At our 0.5 it gives 3.689, at the 0.7 fallback 3.352.
        """
        return math.log(self.losses.backup.clip_t_gain / self.discount)

    @property
    def backup_gain(self) -> float:
        """The realised cap on the LINEX exponential, `gamma*exp(clip_t)`. Equals
        `clip_t_gain` by construction -- kept as a property so the launch log can print the
        quantity that is actually bounded rather than the knob that sets it."""
        return self.discount * math.exp(self.clip_t)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["_derived"] = {
            "neg_log_gamma": self.neg_log_gamma,
            "step_margin": self.step_margin,
            "good_margin": self.good_margin,
            "clip_t": self.clip_t,
            "backup_gain": self.backup_gain,
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
