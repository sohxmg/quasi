"""CF variants as REAL next states: `psi(prompt + steps[:i] + variant + SEP)`.

    s' = s ++ a.   The environment is deterministic, so the next state is not predicted --
    it is the current prefix with the step appended, and it is READ OFF the encoder.

This file is what replaces `phi` inside `qrl_prm/`. `feynman_prm/data/cf_attach.py` +
`FeynmanPRM.cf_phi` represent a rewritten step as `phi(h_{i-1}, act_emb(variant))` -- a
LEARNED state-action head standing in for the arrived state, so that a variant costs an
embedding lookup and an MLP instead of an LM forward (§7.5.3 option (b)). That trade is
what QRL's latent-dynamics term exists to police: `phi` has to be pulled onto `psi` by a
loss, because nothing structural says the two live in the same space.

**Under deterministic dynamics there is nothing to predict, so there is nothing to police.**
Appending the variant text to the prefix and running the encoder gives the arrived state
directly, in `psi`-space, by construction. That removes the dynamics term, the `phi` head and
the whole question of whether the two spaces have drifted -- see `loss.py`'s header.

---

## What it costs, and where the cost is capped

One extra LM forward per micro-batch over `V` variant sequences, each `prompt + steps[:i] +
variant + SEP` long. At `data.cf_max_per_batch = 12` and ~8 rewrites per example that is
~96 sequences against a main batch of ~51, so **`data.cf_max_per_batch` went from nearly free
to the second-largest cost in the step** and `qrl.cf_encode_max_tokens` is the ceiling that
keeps it bounded. Whole EXAMPLES are dropped at the budget, never individual variants: half a
class is not a smaller class, it is a different constraint.

The drop is off the TAIL of the seeded selection order, which is a uniform draw without
replacement (`attach_cf`), so it is unbiased in length -- trimming the longest instead would
quietly train the constraint on short steps only. It is counted in
`cf/examples_dropped_budget`, never silent (§14).

## What is kept from `cf_attach.py`, and why

The prefix-hash join, the `data.cf_max_per_batch` cap and the seeded draw are replicated here
**call for call**, including the order `eligible` is built in, so this run sees the same CF
examples at the same rate as every baseline row. Under the new scheme the join is no longer
REQUIRED -- a variant carries its own prefix now, so any example could be drawn uniformly from
the corpus and the attach rate would be 1.0 instead of ~0.92 -- but taking that coverage would
change the DATA as well as the objective, and the whole comparison rests on the objective
being the only difference. `tests/test_qrl.py` pins the selection against `attach_cf`'s.

`variant_state` survives for one job only: the CF NEGATIVES' push term needs the question id
of the trajectory a negative departs from, so it can be scored against same-question goal
columns. It no longer derives any hub (`loss.py` §3).

## No memo, deliberately

`cf_attach._variant_tokens` memoises its ~30-token rewrites because an example attaches on
most epochs and the text never changes. The same memo HERE would hold whole prefixes --
~41k examples x ~8 variants x ~500 ids, ~80 MB of int32 to save ~50 ms of tokeniser time on a
step that spends seconds on two LM forwards. Tokenised fresh instead.

**Sequences are built THROUGH `build_sequence`**, never assembled here: the old project's
root cause I was two prompt templates diverging by whitespace, and
`tests/test_grep_invariants.py::test_exactly_one_sequence_builder` is the standing guard.
The arguments are `prepare_data.py`'s, so a variant sequence is tokenised exactly the way the
trajectory rows in `sequences.parquet` were.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from feynman_prm.data.cf_attach import build_cf_index
from feynman_prm.data.collate import Batch
from feynman_prm.data.counterfactual import CounterfactualExample
from feynman_prm.data.tokenize import EmptyStep, SequenceTooLong, build_sequence

# Logged on EVERY micro-batch, including the ones where nothing attached
# (`losses/counterfactual.py::_empty_info`'s rule): a diagnostic that disappears on a
# degenerate batch is a diagnostic that cannot be plotted.
CF_ENCODE_KEYS = (
    "cf/examples_attached",
    "cf/examples_eligible",
    "cf/attach_rate",
    "cf/variants",
    "cf/examples_dropped_empty",
    "cf/examples_dropped_too_long",
    "cf/examples_dropped_budget",
    "cf/encode_sequences",
    "cf/encode_real_tokens",
    "cf/encode_padded_tokens",
    "cf/encode_max_len",
)


def empty_encode_info() -> dict[str, float]:
    return {k: 0.0 for k in CF_ENCODE_KEYS}


@dataclass
class EncodedCF:
    """One micro-batch's CF variants, as SEQUENCES ready for an LM forward.

    `state_flat_idx` points at each sequence's LAST separator -- the one that follows the
    variant step -- flattened into `V * L`, which is `Batch.state_flat_idx`'s convention so
    the gather below reads the same way as the main forward's.
    """

    variant_state: Tensor        # (V,) host state s_{i-1}; used ONLY for a negative's qid
    variant_example: Tensor      # (V,) which CF example each variant belongs to
    variant_kind: Tensor         # (V,) 0 anchor, 1 positive, 2 negative
    input_ids: Tensor            # (V, L) prompt + steps[:i] + variant + SEP, right-padded
    attention_mask: Tensor       # (V, L)
    state_flat_idx: Tensor       # (V,) index into V * L of the separator after the variant
    info: dict[str, float]

    @property
    def n_variants(self) -> int:
        return int(self.variant_example.numel())

    def to(self, device) -> "EncodedCF":
        return EncodedCF(
            **{
                k: (v.to(device, non_blocking=True) if isinstance(v, Tensor) else v)
                for k, v in self.__dict__.items()
            }
        )


class CFEncodeContext:
    """`CFContext`'s interface -- `.attach(batch, rng) -> EncodedCF | None` -- over full
    sequences instead of bare step tokens.

    Built once at launch. `index` is the prefix-hash index, identical to `CFContext`'s, and it
    is the reason this is a class: ~41k entries rebuilt per step would sit on the critical
    path for nothing.
    """

    def __init__(
        self,
        examples: Sequence[CounterfactualExample],
        tokenizer,
        pad_id: int,
        max_examples: int,
        sep_id: int,
        prompt_format: str,
        max_len: int,
        max_tokens: int,
    ):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.max_examples = max_examples
        self.sep_id = sep_id
        self.prompt_format = prompt_format
        self.max_len = max_len
        self.max_tokens = max_tokens
        self.index = build_cf_index(self.examples)

    # ---- selection: `attach_cf`'s, call for call ------------------------------------

    def eligible_pairs(self, batch: Batch) -> list[tuple[int, int]]:
        """`[(example index, host state index)]`, in `attach_cf`'s exact build order.

        The order matters twice over: `rng.choice` indexes into this list, and the budget
        trim below drops off its tail. `state_of` takes the FIRST state carrying each hash --
        two trajectories of one question that share a prefix give the same `h_{i-1}`, so
        either is correct and the first is taken deterministically rather than by whichever
        the dict happened to see last.
        """
        if not self.examples or batch.state_prefix_hash is None:
            return []
        state_of: dict[int, int] = {}
        for s, key in enumerate(batch.state_prefix_hash.tolist()):
            if key and key not in state_of:
                state_of[key] = s
        eligible: list[tuple[int, int]] = []
        for key, state in state_of.items():
            for n in self.index.get(key, ()):  # noqa: B905
                eligible.append((n, state))
        return eligible

    def choose(
        self, batch: Batch, rng: np.random.Generator
    ) -> tuple[list[tuple[int, int]], int]:
        """`(chosen, n_eligible)`. The seeded cap draw: same rng call, same argument, same
        `sorted(pick)` as `attach_cf`, and the rng is consumed ONLY when the cap binds."""
        eligible = self.eligible_pairs(batch)
        if not eligible:
            return [], 0
        if len(eligible) > self.max_examples:
            pick = rng.choice(len(eligible), size=self.max_examples, replace=False)
            return [eligible[int(i)] for i in sorted(pick)], len(eligible)
        return eligible, len(eligible)

    # ---- sequence building ----------------------------------------------------------

    def _example_sequences(self, ex: CounterfactualExample):
        """`[(kind, ids, last state position)]` for one example, or a drop reason.

        Every variant of an example goes through or the WHOLE example is dropped, in
        `_variant_tokens`' spirit: an equivalence class missing half its members is not a
        smaller class, it is a different constraint, and a class whose anchor is gone has no
        hub at all.
        """
        rows = []
        prefix = list(ex.steps[: ex.step_index])
        for kind, text in (
            (0, ex.steps[ex.step_index]),
            *[(1, p) for p in ex.positive_rewrites],
            *[(2, g) for g in ex.negative_rewrites],
        ):
            try:
                seq = build_sequence(
                    self.tokenizer,
                    ex.question,
                    prefix + [text],
                    self.sep_id,
                    prompt_format=self.prompt_format,
                    max_len=self.max_len,
                )
            except SequenceTooLong:
                return None, "too_long"
            except EmptyStep:
                return None, "empty"
            # state_pos[-1] is the separator after the variant: that IS s' = s ++ a.
            rows.append((kind, seq.input_ids, seq.state_pos[-1]))
        return rows, None

    def attach(self, batch: Batch, rng: np.random.Generator) -> EncodedCF | None:
        chosen, n_eligible = self.choose(batch, rng)
        info = empty_encode_info()
        info["cf/examples_eligible"] = float(n_eligible)
        if not chosen:
            return None

        built: list[tuple[int, list]] = []          # (host state, rows)
        for n, state in chosen:
            rows, reason = self._example_sequences(self.examples[n])
            if rows is None:
                info[f"cf/examples_dropped_{reason}"] += 1.0
                continue
            built.append((state, rows))

        # ---- the padded-token budget ------------------------------------------------
        # Cost is `sequences x longest`, not the token sum, because that is what the GPU
        # pays. Examples are added in the seeded order and the tail is dropped, so the trim
        # is unbiased in length; taking the longest out instead would train the constraint on
        # short steps only and no curve would say so.
        keep: list[tuple[int, list]] = []
        n_seq = 0
        longest = 0
        for state, rows in built:
            cand_seq = n_seq + len(rows)
            cand_long = max(longest, max(len(ids) for _, ids, _ in rows))
            if keep and cand_seq * cand_long > self.max_tokens:
                info["cf/examples_dropped_budget"] += 1.0
                continue
            keep.append((state, rows))
            n_seq, longest = cand_seq, cand_long

        if not keep:
            return None

        variant_state, variant_example, variant_kind = [], [], []
        ids_list, last_pos = [], []
        for slot, (state, rows) in enumerate(keep):
            for kind, ids, pos in rows:
                variant_state.append(state)
                variant_example.append(slot)
                variant_kind.append(kind)
                ids_list.append(ids)
                last_pos.append(pos)

        V = len(ids_list)
        L = max(len(ids) for ids in ids_list)
        tokens = np.full((V, L), self.pad_id, dtype=np.int64)
        mask = np.zeros((V, L), dtype=np.int64)
        real = 0
        for v, ids in enumerate(ids_list):
            tokens[v, : len(ids)] = ids
            mask[v, : len(ids)] = 1
            real += len(ids)

        long_ = lambda x: torch.as_tensor(x, dtype=torch.long)  # noqa: E731
        info["cf/examples_attached"] = float(len(keep))
        info["cf/attach_rate"] = (
            float(len(keep)) / info["cf/examples_eligible"] if info["cf/examples_eligible"] else 0.0
        )
        info["cf/variants"] = float(V)
        info["cf/encode_sequences"] = float(V)
        info["cf/encode_real_tokens"] = float(real)
        info["cf/encode_padded_tokens"] = float(V * L)
        info["cf/encode_max_len"] = float(L)
        return EncodedCF(
            variant_state=long_(variant_state),
            variant_example=long_(variant_example),
            variant_kind=long_(variant_kind),
            input_ids=torch.from_numpy(tokens),
            attention_mask=torch.from_numpy(mask),
            # Right-padding under causal attention cannot touch a real position, so the flat
            # index is `v * L + pos` with the BATCH's L -- absolute, exactly as `collate`
            # builds `state_flat_idx`.
            state_flat_idx=long_([v * L + p for v, p in enumerate(last_pos)]),
            info=info,
        )


def encode_cf_psi(model, enc: EncodedCF) -> Tensor:
    """`(V, D)` -- `psi` at the separator that follows each variant step.

    One LM forward, then `psi`. **`model.phi` and `model.action_pool` are not touched**: a
    variant is a state now, not a state-action pair, so there is no action to pool and no
    state-action head to run. Gradients reach the LoRA adapter through the whole variant
    sequence (prefix included) and the psi head, which is a strictly wider path than
    `cf_phi`'s -- there the prefix's `h_{i-1}` was borrowed from the main forward.

    The cast to `head_dtype` is at the same seam the main forward uses (`wrapper.py`): the
    backbone runs bf16, the heads fp32 (§6.3).
    """
    _, h = model.hidden_states(enc.input_ids, enc.attention_mask)
    H = int(h.shape[-1])
    h_last = h.reshape(-1, H).index_select(0, enc.state_flat_idx).to(model.head_dtype)
    return model.psi(h_last)
