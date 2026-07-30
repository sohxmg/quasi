"""Checkpointing -- a deliberate deviation from §14 (PLAN 'Core design decisions' 7).

§14 says "merge adapters before saving so the artifact is a plain model directory". That rule
exists because OpenRLHF's `save_model` PEFT branch silently wrote the adapter only and
dropped the trained heads. We own the save path, so instead:

    save_checkpoint()  writes adapter/, heads.pt, the resolved config and the tokenizer,
                       and ASSERTS the head state dict is non-empty
    scripts/export_merged.py  does merge_and_unload() when a standalone model directory is
                              wanted

Checkpoints stay ~100 MB instead of 3.1 GB, which is what makes mid-run saves affordable,
and the merge path is still covered by the §15 GPU test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

HEAD_PREFIXES = ("psi.", "phi.", "goal_head.", "action_pool.", "distance.")


def head_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.startswith(HEAD_PREFIXES)
    }


def save_checkpoint(
    out_dir: str | Path,
    model: torch.nn.Module,
    cfg,
    tokenizer=None,
    step: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)

    heads = head_state_dict(model)
    if not heads:
        raise AssertionError(
            "head state dict is EMPTY -- this is the exact failure §14 records (the stock "
            "PEFT save path writes the adapter only and silently drops the trained heads)"
        )
    torch.save({"heads": heads, "step": step, **(extra or {})}, path / "heads.pt")

    if getattr(model, "backbone", None) is not None and hasattr(model.backbone, "save_pretrained"):
        model.backbone.save_pretrained(path / "adapter")
    if tokenizer is not None:
        tokenizer.save_pretrained(path / "tokenizer")
    cfg.save(path / "config.yaml")
    return path


def load_heads(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    strict: bool = False,
    allow_missing: tuple[str, ...] = (),
) -> dict:
    """`allow_missing` names prefixes the checkpoint is EXPECTED not to carry, so they stay
    at their fresh initialisation instead of aborting the load.

    Exactly one caller needs it, and it is not an edge case: **phase 1 has no goal head at
    all** (§7.7, locked #15), so `runs/phase1/*/heads.pt` legitimately holds only `psi.*` and
    `phi.*`, while phase 2 builds the model `with_goal_head=True` and fits that head from
    scratch. Without the exemption `train_goal_head.py` cannot load ANY phase-1 checkpoint --
    it raises before `build_cache` runs, which reads as the script finishing in seconds.

    Keep it explicit at the call site rather than exempting `goal_head.` globally. The guard's
    real job is §14's LoRA trap -- the stock PEFT save path writes the adapter only and
    silently drops the trained heads -- and an eval loading a PHASE-2 checkpoint must still
    fail loudly if the goal head is absent, because there the head is trained weight.
    """
    payload = torch.load(Path(checkpoint_dir) / "heads.pt", map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(payload["heads"], strict=False)
    if strict and unexpected:
        raise RuntimeError(f"unexpected keys in heads.pt: {unexpected}")
    head_missing = [
        k
        for k in missing
        if k.startswith(HEAD_PREFIXES) and not (allow_missing and k.startswith(allow_missing))
    ]
    if head_missing:
        raise RuntimeError(f"heads.pt is missing head parameters: {head_missing[:8]}")
    return {
        "step": payload.get("step"),
        "missing": missing,
        "unexpected": unexpected,
        "freshly_initialised": sorted(
            {k.split(".")[0] for k in missing if allow_missing and k.startswith(allow_missing)}
        ),
    }


def load_config_from_checkpoint(checkpoint_dir: str | Path):
    import yaml

    from ..config import config_from_dict

    return config_from_dict(yaml.safe_load((Path(checkpoint_dir) / "config.yaml").read_text()))
