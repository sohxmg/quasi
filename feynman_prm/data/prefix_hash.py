"""THE prefix identity used to attach a CF example to a trajectory already in the batch
(§7.5.3, option (b) -- decided by the human 2026-08-15).

L_CF needs `phi(h_{i-1}, act_emb(variant))`. `h_{i-1}` is the hidden at the separator that
ends `prompt + steps[:i]`, so ANY trajectory in the batch that shares that prefix carries
the state the CF example needs -- it does not have to be the trajectory the generator
sampled from. This module defines the key that says "same prefix", once, for both sides of
the join:

    prepare_data.py  ->  SequenceRow.prefix_hash[i]   for i = 0..T   (aligned with state_pos)
    cf_attach.py     ->  prefix_hash(question, steps[:step_index])

**Why a hash and not the text.** The join runs per micro-batch over ~350 states and ~10
CF examples; carrying prefix strings into the parquet would add ~36 MB of duplicated
solution text to `sequences.parquet` and cost a string compare per lookup. int64 keys are
7 MB and hash into a dict.

**Collision budget.** ~150k rows x ~4.4 states = ~660k keys in a 2^63 space. Birthday says
P(any collision) ~ 2.4e-8. A collision would attach a CF example to a state with a
different prefix -- wrong, but not detectably wrong -- so `cf_attach` VERIFIES the match by
comparing the prefix token ids before it uses a state, and counts the misses. The hash
narrows the search; it is not trusted as proof.

The separator is NOT part of the hashed text: `state_pos[i]` is the separator position, and
what identifies the state is the content before it. `\x1f` (ASCII unit separator) joins the
steps because it cannot occur in Math-Shepherd text, so `["a\nb"]` and `["a", "b"]` hash
differently -- §4.7's internal-newline hazard, which is 13.9% of solutions.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

_UNIT_SEP = "\x1f"


def prefix_hash(question: str, steps: Sequence[str]) -> int:
    """Signed int64 identity of the state reached after `question` + `steps`.

    `steps` is a PREFIX: `prefix_hash(q, [])` is `s_0`, `prefix_hash(q, steps[:i])` is
    `s_i`. Signed because parquet/torch int64 is signed and an unsigned round-trip through
    either is a silent overflow.
    """
    payload = question + _UNIT_SEP + _UNIT_SEP.join(steps)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int(np.frombuffer(digest, dtype=np.int64)[0])


def prefix_hashes(question: str, steps: Sequence[str]) -> np.ndarray:
    """`(T+1,)` int64, one per state `s_0..s_T`, aligned with `TokenizedSequence.state_pos`.

    Incremental by construction so a T-step trajectory costs T+1 hashes of growing strings
    rather than a quadratic re-join; the payload is rebuilt each time because blake2b has no
    resumable state we can snapshot cheaply, and T is ~4.
    """
    return np.asarray(
        [prefix_hash(question, steps[:i]) for i in range(len(steps) + 1)], dtype=np.int64
    )
