"""(4) L_CF's on-disk format and loader (§7.5). DATA IS DEFERRED; lambda_cf = 0.

    {"question": "...", "steps": ["...", "..."], "step_index": 2,
     "positive_rewrites": ["...", "..."], "negative_rewrites": ["...", "..."]}

`positive_rewrites` preserve the mathematical meaning of `steps[step_index]`;
`negative_rewrites` change it. The spec's procedure: "give Claude all steps, and ask it to
change a particular step so that the underlying mathematical meaning is same."

**`positive_rewrites` is PLURAL as of 2026-08-08** (was a single `positive_rewrite`). The
anchor and its positives are one equivalence class, pulled together by the multi-positive
L_CF; the generator requests 3 and keeps >=2. There is no migration path for the old
single-positive records because none were ever generated -- the data has always been
deferred.

**The cost claim §7.5 gets wrong** (PLAN finding 1). §7.5 says "each perturbed step needs
its own LM forward, so cap how many appear per batch". But `phi_i = phi(h_{i-1}, act_emb_i)`
and rewriting step i leaves the prefix -- hence `h_{i-1}` -- untouched. Only `act_emb`
changes, and that is a mean of *input embeddings* (§6.4), not a hidden state. So each variant
costs an embedding lookup plus an MLP.

What is built here is the self-contained form: one prefix forward per example (prompt +
steps[:i] + SEP, which ends exactly at s_{i-1}), then all variants share it. When real data
exists and its examples are attached to trajectories already in the training batch, even
that prefix forward disappears -- `h_{i-1}` is already computed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .tokenize import build_sequence


@dataclass(frozen=True)
class CounterfactualExample:
    question: str
    steps: tuple[str, ...]
    step_index: int              # 0-based index of the rewritten step
    positive_rewrites: tuple[str, ...]
    negative_rewrites: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.step_index < len(self.steps):
            raise ValueError(f"step_index {self.step_index} out of range for {len(self.steps)} steps")
        if not self.positive_rewrites:
            raise ValueError(
                "L_CF pulls an equivalence class together: it needs at least one positive rewrite"
            )
        if not self.negative_rewrites:
            raise ValueError("L_CF is a contrastive loss: it needs at least one negative rewrite")


def read_jsonl(path: str | Path) -> list[CounterfactualExample]:
    examples = []
    with Path(path).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            examples.append(
                CounterfactualExample(
                    question=record["question"],
                    steps=tuple(record["steps"]),
                    step_index=int(record["step_index"]),
                    positive_rewrites=tuple(record["positive_rewrites"]),
                    negative_rewrites=tuple(record["negative_rewrites"]),
                )
            )
    return examples


def read_cf_glob(pattern: str) -> list[CounterfactualExample]:
    """Every CF example matching a COMMA-SEPARATED list of globs, deduplicated on
    `(question, step_index)`.

    **Comma-separated and not one bare glob, because a bare glob is a foot-gun here.** The
    generator writes five files per campaign and only ONE of them is kept examples:
    `cf70k.jsonl` beside `cf70k.raw.jsonl`, `.rejected.jsonl`, `.discarded.jsonl` and
    `.responses.jsonl`. `data/cf/cf70k*.jsonl` matches all sixteen files in that directory
    and only three are the ones wanted -- the rest have different schemas, so the failure is
    a `KeyError` deep in a launch if you are lucky and silently wrong data if you are not.
    Listing the three explicitly is the only form that cannot pick up a sibling.

    The campaigns were assigned DISJOINT contiguous slices of the selection order
    (bharatcode ranks 1-1,064, gemini 13,209-32,863, openai 32,873-34,321 as of 2026-08-15),
    so the dedup should be a no-op -- it is here because a re-run written to a new filename
    is the obvious way that stops being true, and two copies of one example would silently
    double its weight in L_CF. Files are read in sorted order so the survivor of a duplicate
    is deterministic.
    """
    from glob import glob

    paths: list[str] = []
    for part in pattern.split(","):
        part = part.strip()
        if not part:
            continue
        matched = sorted(glob(part))
        if not matched:
            raise FileNotFoundError(f"data.cf_glob pattern matched no files: {part!r}")
        paths.extend(matched)

    seen: set[tuple[str, int]] = set()
    out: list[CounterfactualExample] = []
    for path in sorted(set(paths)):
        try:
            examples = read_jsonl(path)
        except KeyError as exc:
            raise ValueError(
                f"{path} is not a CF example file: no {exc} field. The generator's kept "
                "output is `<campaign>.jsonl`; `.raw`, `.rejected`, `.discarded` and "
                "`.responses` are different schemas and must not be in data.cf_glob."
            ) from exc
        for ex in examples:
            key = (ex.question, ex.step_index)
            if key in seen:
                continue
            seen.add(key)
            out.append(ex)
    return out


def write_jsonl(examples: Sequence[CounterfactualExample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for ex in examples:
            fh.write(
                json.dumps(
                    {
                        "question": ex.question,
                        "steps": list(ex.steps),
                        "step_index": ex.step_index,
                        "positive_rewrites": list(ex.positive_rewrites),
                        "negative_rewrites": list(ex.negative_rewrites),
                    }
                )
                + "\n"
            )


@dataclass
class CounterfactualBatch:
    """A CF micro-batch. Within each example the variants run anchor, then every
    meaning-preserving positive, then every meaning-changing negative -- `variant_kind`
    0, then 1 repeated, then 2 repeated. **Nothing may assume exactly one kind-1 row.**"""

    input_ids: Tensor            # (N, L) prefix sequences, one per example
    attention_mask: Tensor | None
    anchor_flat_idx: Tensor      # (N,) flat index of the s_{i-1} separator of each prefix
    variant_example: Tensor      # (V,) which example each variant belongs to
    variant_kind: Tensor         # (V,) 0 anchor, 1 positive, 2 negative
    variant_token_idx: Tensor    # (P,) flat index into V_pad * L_var
    variant_row_idx: Tensor      # (P,) which variant each pooled token belongs to
    variant_tokens: Tensor       # (V, L_var) token ids of the rewritten step
    variant_counts: Tensor       # (V,) float, tokens per variant

    @property
    def n_examples(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def n_variants(self) -> int:
        return int(self.variant_example.numel())

    def to(self, device) -> "CounterfactualBatch":
        return CounterfactualBatch(
            **{
                k: (v.to(device, non_blocking=True) if isinstance(v, Tensor) else v)
                for k, v in self.__dict__.items()
            }
        )


def build_cf_batch(
    tokenizer,
    examples: Sequence[CounterfactualExample],
    sep_id: int,
    pad_id: int,
    prompt_format: str = "raw",
) -> CounterfactualBatch:
    """Tokenise prefixes (through the ONE sequence builder) and the step variants."""
    prefixes, variants, variant_example, variant_kind = [], [], [], []
    for n, ex in enumerate(examples):
        # prompt + steps[:i] + SEP: the last state of this sequence IS s_{i-1}.
        prefix_steps = list(ex.steps[: ex.step_index])
        seq = build_sequence(
            tokenizer,
            ex.question,
            prefix_steps if prefix_steps else [""],
            sep_id,
            prompt_format=prompt_format,
        ) if prefix_steps else None
        if seq is None:
            # step_index == 0: the anchor state is s_0, so the prefix is prompt + SEP only.
            ids = list(tokenizer(ex.question, add_special_tokens=False)["input_ids"]) + [sep_id]
            prefixes.append((ids, len(ids) - 1))
        else:
            prefixes.append((list(seq.input_ids), seq.state_pos[-1]))

        for kind, text in (
            (0, ex.steps[ex.step_index]),
            *[(1, pos) for pos in ex.positive_rewrites],
            *[(2, neg) for neg in ex.negative_rewrites],
        ):
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) == 0:
                raise ValueError(f"CF variant tokenises to zero tokens: {text!r}")
            variants.append(list(ids))
            variant_example.append(n)
            variant_kind.append(kind)

    L = max(len(ids) for ids, _ in prefixes)
    N = len(prefixes)
    input_ids = np.full((N, L), pad_id, dtype=np.int64)
    mask = np.zeros((N, L), dtype=np.int64)
    anchor = np.zeros(N, dtype=np.int64)
    for n, (ids, pos) in enumerate(prefixes):
        input_ids[n, : len(ids)] = ids
        mask[n, : len(ids)] = 1
        anchor[n] = n * L + pos

    L_var = max(len(v) for v in variants)
    V = len(variants)
    variant_tokens = np.full((V, L_var), pad_id, dtype=np.int64)
    token_idx, row_idx, counts = [], [], []
    for v, ids in enumerate(variants):
        variant_tokens[v, : len(ids)] = ids
        counts.append(float(len(ids)))
        for j in range(len(ids)):
            token_idx.append(v * L_var + j)
            row_idx.append(v)

    long_ = lambda x: torch.as_tensor(x, dtype=torch.long)  # noqa: E731
    return CounterfactualBatch(
        input_ids=torch.from_numpy(input_ids),
        attention_mask=None if mask.all() else torch.from_numpy(mask),
        anchor_flat_idx=torch.from_numpy(anchor),
        variant_example=long_(variant_example),
        variant_kind=long_(variant_kind),
        variant_token_idx=long_(token_idx),
        variant_row_idx=long_(row_idx),
        variant_tokens=torch.from_numpy(variant_tokens),
        variant_counts=torch.as_tensor(counts, dtype=torch.float32),
    )
