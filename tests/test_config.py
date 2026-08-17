"""Config tests: strict parsing (bug B4) and the derived quantities (§7.4.3, §7.6.4, §7.8)."""

from __future__ import annotations

import math

import pytest
import yaml

from feynman_prm.config import ConfigError, Config, config_from_dict, load_config
from conftest import REPO_ROOT

DEFAULT = REPO_ROOT / "config" / "default.yaml"


def test_defaults_match_the_locked_decisions(cfg):
    assert cfg.discount == 0.5                                   # locked #9
    assert cfg.heads.latent_dim == 512                           # §6.3, NOT 1536
    assert cfg.distance.variant == "full_mrn"                    # locked #11
    assert cfg.losses.action_invariance.mode == "diagonal"       # locked #16
    assert cfg.losses.step_loss.pairing == "boundary"            # locked #3b
    assert cfg.losses.lambda_step == 1.0                         # not optional (§7.6.2)
    assert cfg.sampling.nce_mask_same_traj is False              # locked #12, the BLUNT one
    assert cfg.sampling.sequences_per_micro_batch == 56          # §8.1
    assert cfg.sampling.max_padded_tokens == 32768               # measured 2026-07-27
    assert cfg.model.name == "Qwen/Qwen2.5-Math-1.5B-Instruct"


def test_the_2026_08_04_run_config(cfg):
    """The three changes that ship together, pinned so none of them reverts silently. Each
    would leave a healthy-looking set of curves behind it -- which is the whole reason these
    are tests and not comments.

    They are also three changes in one run, so the attribution re-runs are named here:
    `--set losses.nce_temperature=1.0`, `--set losses.good_loss.form=relu`,
    `--set sampling.nce_mask_nearer_same_traj=false`, each with its own `run.name`."""
    # tau_NCE = sqrt(latent_dim), i.e. TMD's own `-dist / sqrt(phi.shape[-1])` (tmd.py:92).
    # Reverses §7.2's documented divergence, which shipped 1.0 for every run to date.
    assert math.isclose(cfg.losses.nce_temperature, math.sqrt(cfg.heads.latent_dim), rel_tol=1e-6)
    # §9.9.2 / §16.4: the SURGICAL mask (rows between the positive and the goal), not the
    # blunt one above -- rows EARLIER than the positive stay, they are honest hard negatives.
    assert cfg.sampling.nce_mask_nearer_same_traj is True
    # §9.9.7's order of work runs the nearer mask alone first; these two stay off.
    assert cfg.sampling.nce_mask_sibling_correct_late is False
    assert cfg.sampling.nce_mask_same_question_correct is False
    # §7.12: relu moved the bulk and lost the tail. Never softplus -- it applies gradient AT
    # the target and stretches L_T's ruler (measured overshoot to Delta = -1.556).
    assert cfg.losses.good_loss.form == "relu_squared"
    # One directory per loss-set change: `runs/phase1/` is the previous run and the baseline
    # every §9.3.1 / §9.7 number was measured on.
    assert cfg.run.name != "phase1"


def test_derived_values_match_section_7_8s_table(cfg):
    import dataclasses

    assert math.isclose(cfg.neg_log_gamma, 0.69315, rel_tol=1e-4)
    assert math.isclose(cfg.step_margin, 1.38629, rel_tol=1e-4)
    # clip_t is log(clip_t_gain / discount) -- it moves AGAINST neg_log_gamma, not with it
    # (§7.4.3). 3.6889 at 0.5, 3.3524 at 0.7.
    assert math.isclose(cfg.clip_t, 3.6889, rel_tol=1e-3)

    fallback = dataclasses.replace(cfg, discount=0.7)
    assert math.isclose(fallback.neg_log_gamma, 0.35667, rel_tol=1e-4)
    assert math.isclose(fallback.step_margin, 0.71335, rel_tol=1e-4)
    assert math.isclose(fallback.clip_t, 3.3524, rel_tol=1e-3)


def test_unknown_key_is_a_hard_error():
    """Old bug B4: "config value silently ignored, library default used"."""
    with pytest.raises(ConfigError, match="unknown config key 'discountt'"):
        load_config(DEFAULT, ["discountt=0.5"])
    with pytest.raises(ConfigError, match="did you mean 'margin_steps'"):
        load_config(DEFAULT, ["losses.step_loss.margin_step=2.0"])


def test_types_are_checked():
    with pytest.raises(ConfigError, match="expected a float"):
        load_config(DEFAULT, ["discount=half"])
    with pytest.raises(ConfigError, match="expected a bool"):
        load_config(DEFAULT, ["sampling.group_by_length=yes_please"])


def test_illegal_values_are_rejected():
    with pytest.raises(ConfigError, match="not one of"):
        load_config(DEFAULT, ["distance.variant=cosine"])
    with pytest.raises(ConfigError, match="must be in"):
        load_config(DEFAULT, ["discount=1.0"])
    with pytest.raises(ConfigError, match="not implemented"):
        load_config(DEFAULT, ["heads.ensemble=true"])


def test_grid_invariance_with_step_loss_is_refused():
    """§7.6.5: the grid modes drive phi(s,a) -> psi(s), which pins Delta_{z+1} at -log gamma
    good step or bad, so L_step cannot move. Refusing the combination is cheaper than
    debugging a run where L_step silently does nothing."""
    with pytest.raises(ConfigError, match="grid"):
        load_config(DEFAULT, ["losses.action_invariance.mode=grid_batch"])
    cfg = load_config(DEFAULT, ["losses.action_invariance.mode=grid_batch",
                                "losses.lambda_step=0.0"])
    assert cfg.losses.action_invariance.mode == "grid_batch"


def test_latent_dim_must_split_into_even_halves():
    with pytest.raises(ConfigError, match="EVEN quotient"):
        load_config(DEFAULT, ["distance.components=512"])


def test_save_round_trips_through_strict_parsing(cfg, tmp_path):
    path = tmp_path / "resolved.yaml"
    cfg.save(path)
    payload = yaml.safe_load(path.read_text())
    assert payload["_derived"]["step_margin"] == pytest.approx(cfg.step_margin)
    assert config_from_dict(payload) == cfg


def test_deleted_keys_stay_deleted():
    """§7.9: there is no value head, so no lambda_bt / lambda_crm / lambda_correct; and the
    two §11 keys PLAN drops are gone (`hard_negatives_post_error`, `grid_max_actions`)."""
    fields = {f for f in Config.__dataclass_fields__}
    assert "value_head" not in fields
    for dead in (
        "losses.lambda_bt=1.0",
        "losses.lambda_crm=1.0",
        "losses.lambda_correct=1.0",
        "sampling.hard_negatives_post_error=true",
        "losses.action_invariance.grid_max_actions=64",
        "goal_head.substitution_p=0.1",
    ):
        with pytest.raises(ConfigError, match="unknown config key"):
            load_config(DEFAULT, [dead])
