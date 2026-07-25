"""Shared fixtures. Everything here is CPU-only, model-free and download-free.

That is not an accident: index bookkeeping is separated from the model (PLAN 'Core design
decisions' 1), so every loss is a pure function of `(psi, phi, act_emb, index tensors)` and
the whole §15 suite runs on random hiddens.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch

from feynman_prm.config import Config, load_config
from feynman_prm.data.collate import SequenceRow, collate

REPO_ROOT = Path(__file__).resolve().parents[1]
SEP_ID = 1


class DummyTokenizer:
    """Word-level tokenizer where "\\n" is its own single id.

    Deliberately emits the separator id for an in-step newline, which is §4.7's hazard:
    13.9% of solutions contain a step with an internal newline, so scanning the ids for the
    separator gives wrong state positions on one solution in seven. build_sequence computes
    positions by arithmetic and is immune; tests/test_tokenize.py proves it.
    """

    pad_token_id = 0

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {"\n": SEP_ID}
        self._next = 2

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        ids = []
        for token in re.findall(r"\n|[^\s]+", text):
            if token not in self._vocab:
                self._vocab[token] = self._next
                self._next += 1
            ids.append(self._vocab[token])
        return {"input_ids": ids}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True) -> str:
        return "<|im_start|> " + messages[0]["content"] + " <|im_end|>"


@pytest.fixture
def tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


@pytest.fixture
def cfg() -> Config:
    return load_config(REPO_ROOT / "config" / "default.yaml")


def synthetic_row(
    qid: str,
    labels: list[bool],
    prompt_len: int = 3,
    step_len: int = 2,
    correct: bool | None = None,
) -> SequenceRow:
    """A pre-tokenised row built by the same arithmetic build_sequence uses (§6.1)."""
    from feynman_prm.utils.indexing import first_error_index, has_recovery

    ids: list[int] = [9] * prompt_len
    state_pos = [len(ids)]
    ids.append(SEP_ID)
    starts, ends = [], []
    for _ in labels:
        starts.append(len(ids))
        ids.extend([7] * step_len)
        ends.append(len(ids))
        state_pos.append(len(ids))
        ids.append(SEP_ID)

    z = first_error_index(labels)
    return SequenceRow(
        qid=qid,
        input_ids=np.asarray(ids, dtype=np.int64),
        state_pos=np.asarray(state_pos, dtype=np.int64),
        span_start=np.asarray(starts, dtype=np.int64),
        span_end=np.asarray(ends, dtype=np.int64),
        correct=all(labels) if correct is None else correct,
        z=-1 if z is None else z,
        recovery=has_recovery(labels),
    )


@pytest.fixture
def make_row():
    return synthetic_row


@pytest.fixture
def small_batch():
    """Two questions, each with one correct and one incorrect trajectory."""
    T, F = True, False
    rows = [
        synthetic_row("q1", [T, T, T]),
        synthetic_row("q1", [T, T, F, F]),
        synthetic_row("q2", [T, T]),
        synthetic_row("q2", [F, F, F]),
    ]
    return collate(rows, pad_id=0)


@pytest.fixture
def random_reps(small_batch):
    """(psi, phi) random latents with gradients, shaped for `small_batch`."""
    torch.manual_seed(0)
    D = 64
    psi = torch.randn(small_batch.n_states, D, requires_grad=True)
    phi = torch.randn(small_batch.n_rows, D, requires_grad=True)
    return psi, phi


@pytest.fixture
def distance():
    from feynman_prm.model.distances import Distance

    return Distance("full_mrn", components=8)
