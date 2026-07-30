"""Checkpointing (§14's LoRA trap 3, PLAN 'Core design decisions' 7).

The stock PEFT save path writes the adapter only and SILENTLY DROPS the trained heads. We own
the save path, so the assert lives here and runs on CPU -- `tests/test_gpu.py` covers the real
adapter and `merge_and_unload()`.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from feynman_prm.model.wrapper import FeynmanPRM
from feynman_prm.utils.checkpoint import (
    head_state_dict,
    load_config_from_checkpoint,
    load_heads,
    save_checkpoint,
)


@pytest.fixture
def model(cfg):
    import dataclasses

    tiny = dataclasses.replace(
        cfg, heads=dataclasses.replace(cfg.heads, latent_dim=32, hidden_dims=(16,))
    )
    return tiny, FeynmanPRM(tiny, 32, backbone=None)


def test_heads_are_saved_and_restored(model, tmp_path):
    cfg, m = model
    save_checkpoint(tmp_path / "ckpt", m, cfg, step=7)
    before = head_state_dict(m)
    assert before, "psi/phi/distance must be in the head state dict"

    with torch.no_grad():                       # corrupt, then restore
        m.psi.net.out.weight.zero_()
    load_heads(m, tmp_path / "ckpt")
    for key, tensor in before.items():
        assert torch.equal(tensor, head_state_dict(m)[key])


def test_empty_head_state_dict_is_refused(cfg, tmp_path):
    """The exact failure §14 records: an artifact with no heads in it."""
    with pytest.raises(AssertionError, match="head state dict is EMPTY"):
        save_checkpoint(tmp_path / "ckpt", nn.Linear(2, 2), cfg)


def test_config_round_trips_through_strict_parsing(model, tmp_path):
    cfg, m = model
    save_checkpoint(tmp_path / "ckpt", m, cfg)
    assert load_config_from_checkpoint(tmp_path / "ckpt") == cfg


def test_missing_head_parameters_are_a_hard_error(model, tmp_path):
    cfg, m = model
    save_checkpoint(tmp_path / "ckpt", m, cfg)
    payload = torch.load(tmp_path / "ckpt" / "heads.pt", weights_only=False)
    payload["heads"] = {k: v for k, v in payload["heads"].items() if not k.startswith("phi.")}
    torch.save(payload, tmp_path / "ckpt" / "heads.pt")
    with pytest.raises(RuntimeError, match="missing head parameters"):
        load_heads(m, tmp_path / "ckpt")


def test_phase1_checkpoint_loads_into_a_phase2_model(model, tmp_path):
    """§7.7 / locked #15: phase 1 has NO goal head, so its `heads.pt` legitimately carries no
    `goal_head.*`. Phase 2 builds one and fits it from scratch, so the load must succeed with
    the head left at its fresh init.

    Without `allow_missing` this raised before `build_cache` ran, so `train_goal_head.py` died
    ~5s in against every phase-1 checkpoint -- fast enough to read as a completed run, since
    the work it skipped is ~34M tokens through the backbone (11-38 min).
    """
    cfg, phase1 = model
    assert phase1.goal_head is None, "phase 1 must not construct a goal head (train.py:147)"
    save_checkpoint(tmp_path / "ckpt", phase1, cfg, step=1460)

    phase2 = FeynmanPRM(cfg, 32, backbone=None, with_goal_head=True)
    before = phase2.goal_head.net.out.weight.clone()

    with pytest.raises(RuntimeError, match="missing head parameters"):
        load_heads(phase2, tmp_path / "ckpt")               # the guard, unexempted

    loaded = load_heads(phase2, tmp_path / "ckpt", allow_missing=("goal_head.",))
    assert loaded["step"] == 1460
    assert loaded["freshly_initialised"] == ["goal_head"]
    # psi came from the checkpoint; the goal head did NOT move.
    torch.testing.assert_close(phase2.goal_head.net.out.weight, before)
    torch.testing.assert_close(phase2.psi.net.out.weight, phase1.psi.net.out.weight)


def test_allow_missing_does_not_weaken_the_guard_for_trained_heads(model, tmp_path):
    """The exemption is per-prefix and per-call-site. Dropping psi -- §14's actual trap -- must
    still abort even when the caller is exempting the goal head."""
    cfg, m = model
    save_checkpoint(tmp_path / "ckpt", m, cfg, step=1)
    payload = torch.load(tmp_path / "ckpt" / "heads.pt", weights_only=False)
    payload["heads"] = {k: v for k, v in payload["heads"].items() if not k.startswith("psi.")}
    torch.save(payload, tmp_path / "ckpt" / "heads.pt")

    phase2 = FeynmanPRM(cfg, 32, backbone=None, with_goal_head=True)
    with pytest.raises(RuntimeError, match=r"missing head parameters: \['psi\."):
        load_heads(phase2, tmp_path / "ckpt", allow_missing=("goal_head.",))


def test_run_logger_writes_jsonl(tmp_path):
    from feynman_prm.diagnostics.logging import RunLogger, read_metrics

    logger = RunLogger(tmp_path, "run")
    logger.log(1, {"loss/total": 1.5, "nce/loss": 5.85})
    logger.event("launch/data", {"optimizer_steps": 889})
    logger.close()

    records = read_metrics(tmp_path / "run" / "metrics.jsonl")
    assert records[0]["step"] == 1 and records[0]["nce/loss"] == 5.85
    assert "optimizer_steps" in (tmp_path / "run" / "events.jsonl").read_text()
    assert "L=+1.5000" in RunLogger.format_console(1, {"loss/total": 1.5})
