"""Attach CF examples to the trajectories already in a main micro-batch (§7.5.3 option (b),
chosen by the human 2026-08-15 over the separate interleaved loader).

**What this buys and what it costs.** `phi_i = phi(h_{i-1}, act_emb_i)` and rewriting step
`i` leaves `h_{i-1}` untouched (PLAN finding 1), so a CF example whose prefix is already
forwarded in this batch costs an embedding lookup plus an MLP -- no LM forward at all. The
alternative (a separate interleaved loader) pays one prefix forward per CF example.

**MEASURED 2026-08-15, offline, no GPU, on the 34,650-question selection and the 27,114
examples on disk: 24,911 of 27,114 CF prefixes (91.9%) are reachable.** That is the level
`cf/attach_rate` should sit near over an epoch; per batch it is noisy.

**It came out ABOVE the 85.6% predicted from the sampler's caps, and the gap is the whole
argument for hashing the PREFIX instead of the trajectory.** The cap argument was:
`cmd_sample` draws one random correct trajectory per question, the sampler takes
`min(4, k_c)`, so 6,253 of 8,639 CF questions (72.4%) have every correct trajectory in the
batch and the rest get `4/k_c` -- ~85.6% of examples. That reasoning silently assumes a CF
example can only ride the trajectory it was generated from. It cannot: `h_{i-1}` depends on
`steps[:i]` alone, so ANY trajectory agreeing on the first `i` steps is an equally valid
host, and sibling solutions of one question agree on their opening steps often enough to
buy 6.3 points. **A trajectory-level key would have measured 85.6% and been correct-but-worse.**

If the realised rate runs far below 91.9%, the join is broken, not the data.

**The miss is DROPPED, not backfilled**, and it is counted (`cf/examples_unmatched`). A
backfill would mean a prefix forward inside the main step, which is the cost option (b)
exists to avoid; and a CF example seen on 85.6% of epochs is a sampling property, not a
defect. §14's rule applies: the drop is relied on, so it is logged rather than silent.

**The hash is TRUSTED, and here is the budget that justifies it.** `prefix_hash` is 64-bit
over ~660k keys (~150k rows x ~4.4 states), so P(any collision anywhere in the dataset) is
~2.4e-8 by the birthday bound. A collision would attach a CF example to a state with a
different prefix and **nothing downstream could detect it** -- so this is a risk accepted
on arithmetic, not a check that was implemented.

There is deliberately NO `cf/hash_collisions` counter. Verifying would mean re-tokenising
each CF example's prefix and comparing it against the batch's input ids, which costs more
than the join it guards; a counter that cannot fire is worse than no counter (§14), so the
honest form is this paragraph rather than a metric pinned at 0.0. If the 64-bit budget ever
stops being comfortable -- a much larger dataset, or a second CF corpus -- widen
`prefix_hash` to 128 bits rather than adding a verification pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .collate import Batch
from .counterfactual import CounterfactualExample
from .math_shepherd import question_id
from .prefix_hash import prefix_hash


@dataclass
class AttachedCF:
    """CF variants bound to states of the CURRENT batch.

    `variant_state` is the piece option (a) does not have: the flat index, into this
    batch's `(S, ...)` state tensors, of the `s_{i-1}` each variant departs from. The model
    reads `h_states[variant_state]` instead of forwarding a prefix of its own.
    """

    variant_state: Tensor        # (V,) flat state index of s_{i-1}
    variant_example: Tensor      # (V,) which CF example each variant belongs to
    variant_kind: Tensor         # (V,) 0 anchor, 1 positive, 2 negative
    variant_tokens: Tensor       # (V, L_var) token ids of the rewritten step
    variant_token_idx: Tensor    # (P,) flat index into V * L_var
    variant_row_idx: Tensor      # (P,) which variant each pooled token belongs to
    variant_counts: Tensor       # (V,) float, tokens per variant
    info: dict[str, float]

    @property
    def n_variants(self) -> int:
        return int(self.variant_example.numel())

    def to(self, device) -> "AttachedCF":
        return AttachedCF(
            **{
                k: (v.to(device, non_blocking=True) if isinstance(v, Tensor) else v)
                for k, v in self.__dict__.items()
            }
        )


class CFContext:
    """Everything the CF attach needs, built ONCE at launch and reused every micro-batch.

    Holds the examples, the prefix index, the tokeniser and the variant-token memo. The
    memo is the reason this is a class and not a free function: a CF example attaches on
    most epochs and its ~10 rewrites tokenise to the same ids every time, so tokenising
    them per attach would put ~90 short tokeniser calls on the critical path of every step
    for no new information.
    """

    def __init__(self, examples, tokenizer, pad_id: int, max_examples: int):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.max_examples = max_examples
        self.index = build_cf_index(self.examples)
        self._variant_cache: dict[int, list[tuple[int, list[int]]]] = {}

    def attach(self, batch: Batch, rng: np.random.Generator) -> "AttachedCF | None":
        return attach_cf(
            batch,
            self.examples,
            self.index,
            self.tokenizer,
            self.pad_id,
            self.max_examples,
            rng,
            self._variant_cache,
        )


# A CF example can be off the train rows for two reasons that are NOT the same event, and
# the launch guard used to charge both to the worse one. See `select_cf_examples_for_train`.
# 0.01 is ~65x the rate a whole question can vanish at tokenisation (§4.6 drops 0.489% of
# SEQUENCES for length; a question loses ALL of them far more rarely -- 10 of 34,650 on the
# 2026-08-16 parquet) and orders of magnitude below a genuine selection mismatch, which
# misses most of the corpus rather than a handful of it. It is a tripwire, not a tuned knob.
ABSENT_QUESTION_TOLERANCE = 0.01


def select_cf_examples_for_train(
    examples: Sequence[CounterfactualExample],
    train_qids: set[str],
    val_qids: set[str],
    tolerance: float = ABSENT_QUESTION_TOLERANCE,
) -> tuple[list[CounterfactualExample], dict]:
    """Keep the CF examples that sit on a question the train rows actually contain.

    **Two different failures used to share one message, and only one of them is a failure.**
    The original guard tested `question_id(ex.question) in {r.qid for r in rows}` and raised
    on any miss as *"not on a train-split question (§8.2) ... or the CF data is from a
    different selection."* But that set is the qids **on disk in the train split**, which is
    not the train SELECTION: a question whose every trajectory was dropped at tokenisation
    (§4.6 -- 0.489% of sequences exceed `max_len`) is in the selection and absent from the
    rows. Its CF examples are then unattachable, which `attach_cf` already handles by
    counting a miss, and they are charged to leakage instead.

    So the two cases are separated here, by looking at the VAL qids rather than inferring:

      * **on a val question -> ALWAYS fatal, at any count.** This is the §8.2 hazard the
        guard exists for: phase 1 training on a held-out question is silent leakage that no
        curve would show, and one example is enough to invalidate `val_f1.py`.
      * **on neither split -> dropped and COUNTED**, unless it is more than `tolerance` of
        the corpus, which is the "different selection" case the message also names and which
        cannot look like a handful of examples.

    Returns `(kept, info)`; `info` goes into the `launch/cf_data` event, because a drop that
    is relied on has to be logged rather than silent (§14).
    """
    examples = list(examples)
    qids = [question_id(ex.question) for ex in examples]

    kept = [ex for ex, qid in zip(examples, qids) if qid in train_qids]
    missing = [qid for qid in qids if qid not in train_qids]
    leaked = sorted({qid for qid in missing if qid in val_qids})
    absent = sorted({qid for qid in missing if qid not in val_qids})
    n_leaked = sum(qid in val_qids for qid in missing)
    n_absent = len(missing) - n_leaked

    if leaked:
        raise AssertionError(
            f"{n_leaked} of {len(examples)} CF examples sit on {len(leaked)} VAL question(s) "
            f"-- phase 1 would train on the held-out split and nothing downstream could "
            f"detect it (§8.2). First few: {', '.join(leaked[:5])}. Regenerate the CF corpus "
            f"against this selection (same seed and n_questions), or filter them out."
        )

    if n_absent > tolerance * max(len(examples), 1):
        raise AssertionError(
            f"{n_absent} of {len(examples)} CF examples ({n_absent / len(examples):.1%}) are "
            f"on questions in NEITHER split of sequences.parquet, which is past the "
            f"{tolerance:.0%} a tokenisation drop can explain -- the CF corpus is from a "
            f"different selection (§8.2). First few: {', '.join(absent[:5])}."
        )

    info = {
        "examples_total": len(examples),
        "examples_kept": len(kept),
        # Their question is in no split on disk: every one of its trajectories was dropped at
        # tokenisation, so there is no prefix for them to attach to and nothing to fix.
        "examples_dropped_question_absent": n_absent,
        "questions_absent": len(absent),
        "questions_absent_sample": absent[:5],
    }
    return kept, info


def build_cf_index(examples: Sequence[CounterfactualExample]) -> dict[int, list[int]]:
    """`prefix_hash of the anchor's s_{i-1}` -> indices into `examples`.

    Built ONCE per run, not per batch: it is ~27k entries and rebuilding it inside the
    training loop would put a hash of every CF prefix on the critical path of every step.
    """
    index: dict[int, list[int]] = {}
    for n, ex in enumerate(examples):
        key = prefix_hash(ex.question, ex.steps[: ex.step_index])
        index.setdefault(key, []).append(n)
    return index


def attach_cf(
    batch: Batch,
    examples: Sequence[CounterfactualExample],
    index: dict[int, list[int]],
    tokenizer,
    pad_id: int,
    max_examples: int,
    rng: np.random.Generator,
    variant_cache: dict[int, list[tuple[int, list[int]]]] | None = None,
) -> AttachedCF | None:
    """Find the CF examples whose prefix is in `batch` and build their variant tensors.

    `max_examples` caps how many attach per micro-batch -- the §7.5.2 budget argument in
    reverse: a batch that happens to hold many CF-covered questions must not swing L_CF's
    magnitude, and the cap is what keeps the per-step cost flat. Selection among the
    eligible is uniform without replacement, drawn on `rng` so it is seeded with the epoch.

    Returns None when nothing attached, which is a normal outcome and not an error: the
    caller passes `cf=None` and `total_loss` contributes an exact zero.
    """
    if not examples or batch.state_prefix_hash is None:
        return None

    hashes = batch.state_prefix_hash.tolist()
    # state index -> the FIRST state carrying each hash. Two trajectories of one question
    # that share a prefix give identical h_{i-1}, so either is correct and the first is
    # taken deterministically rather than by whichever the dict happened to see last.
    state_of: dict[int, int] = {}
    for s, key in enumerate(hashes):
        if key and key not in state_of:
            state_of[key] = s

    eligible: list[tuple[int, int]] = []          # (example index, state index)
    for key, state in state_of.items():
        for n in index.get(key, ()):  # noqa: B905
            eligible.append((n, state))
    if not eligible:
        return None

    n_eligible = len(eligible)
    if n_eligible > max_examples:
        pick = rng.choice(n_eligible, size=max_examples, replace=False)
        chosen = [eligible[int(i)] for i in sorted(pick)]
    else:
        chosen = eligible

    variant_state, variant_example, variant_kind, variants = [], [], [], []
    kept = 0
    for n, state in chosen:
        ex = examples[n]
        rows = _variant_tokens(ex, n, tokenizer, variant_cache)
        if rows is None:
            continue
        slot = kept
        for kind, ids in rows:
            variants.append(ids)
            variant_state.append(state)
            variant_example.append(slot)
            variant_kind.append(kind)
        kept += 1

    if kept == 0:
        return None

    V = len(variants)
    L_var = max(len(v) for v in variants)
    tokens = np.full((V, L_var), pad_id, dtype=np.int64)
    token_idx, row_idx, counts = [], [], []
    for v, ids in enumerate(variants):
        tokens[v, : len(ids)] = ids
        counts.append(float(len(ids)))
        for j in range(len(ids)):
            token_idx.append(v * L_var + j)
            row_idx.append(v)

    long_ = lambda x: torch.as_tensor(x, dtype=torch.long)  # noqa: E731
    info = {
        "cf/examples_attached": float(kept),
        "cf/examples_eligible": float(n_eligible),
        # The §7.5.3-(b) prediction is ~0.856 over an epoch. Per batch it is noisy; read it
        # as a running mean, and read it against that number rather than against 1.0.
        "cf/attach_rate": float(kept) / float(n_eligible) if n_eligible else 0.0,
        "cf/variants": float(V),
    }
    return AttachedCF(
        variant_state=long_(variant_state),
        variant_example=long_(variant_example),
        variant_kind=long_(variant_kind),
        variant_tokens=torch.from_numpy(tokens),
        variant_token_idx=long_(token_idx),
        variant_row_idx=long_(row_idx),
        variant_counts=torch.as_tensor(counts, dtype=torch.float32),
        info=info,
    )


def _variant_tokens(
    ex: CounterfactualExample,
    n: int,
    tokenizer,
    cache: dict[int, list[tuple[int, list[int]]]] | None,
) -> list[tuple[int, list[int]]] | None:
    """`[(kind, token ids)]` in `counterfactual_loss`'s order: anchor, positives, negatives.

    Tokenising ~10 short strings per example per attach would be the dominant CPU cost of
    this path -- an example attaches on most epochs and the text never changes -- so the
    result is memoised on the example index. A variant that tokenises to zero tokens would
    make `act_emb` a mean over an empty set (§6.4); the whole example is dropped rather
    than silently pooling a zero vector into its class.
    """
    if cache is not None and n in cache:
        return cache[n]
    rows: list[tuple[int, list[int]]] = []
    for kind, text in (
        (0, ex.steps[ex.step_index]),
        *[(1, p) for p in ex.positive_rewrites],
        *[(2, g) for g in ex.negative_rewrites],
    ):
        ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        if not ids:
            if cache is not None:
                cache[n] = None  # type: ignore[assignment]
            return None
        rows.append((kind, ids))
    if cache is not None:
        cache[n] = rows
    return rows
