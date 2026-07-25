#!/usr/bin/env python
"""Merge the LoRA adapter into the base weights and write a standalone model directory.

This exists because §14 says "merge adapters before saving so the artifact is a plain model
directory" -- a rule that came from OpenRLHF's `save_model` PEFT branch silently writing the
adapter only and dropping the trained heads. `utils/checkpoint.py` handles that failure
directly (it asserts the head state dict is non-empty), so merging is a separate, explicit
step rather than part of every mid-run save (PLAN 'Core design decisions' 7).

The heads are copied alongside: a merged backbone WITHOUT `heads.pt` is exactly the artifact
§14 warns about.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run without `pip install -e .`

import argparse
import shutil

from feynman_prm.model.backbone import load_backbone_with_adapter, load_tokenizer
from feynman_prm.utils.checkpoint import load_config_from_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="merge LoRA into a standalone model dir")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    ckpt, out = Path(args.checkpoint), Path(args.out)
    cfg = load_config_from_checkpoint(ckpt)
    model = load_backbone_with_adapter(cfg, ckpt / "adapter")
    merged = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    load_tokenizer(cfg).save_pretrained(out)

    shutil.copy(ckpt / "heads.pt", out / "heads.pt")     # never ship the backbone alone
    cfg.save(out / "config.yaml")
    print(f"merged model + heads written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
