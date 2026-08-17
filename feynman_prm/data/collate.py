"""Batch assembly: padding plus every index map the losses need (PLAN 'Core design
decisions' 1).

Index bookkeeping is deliberately separated from the model. Everything here is built on
CPU from pre-tokenised rows only, and every loss is a pure function of
`(psi, phi, act_emb, these index tensors)` -- which is what makes the whole §15 suite
runnable on random hiddens with no model and no GPU.

Padding is to the BATCH's longest sequence, never to max_len (§4.6): that is the single
biggest lever on how many negatives fit in a batch. Padding is on the right, so every
position index is absolute and unaffected by it; under causal attention trailing pad tokens
cannot influence any real position. (Old bug B5 was two groups padded to different maxima
and then concatenated; there is one padding site here and it is this function.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class SequenceRow:
    """One pre-tokenised trajectory, as written to parquet by scripts/prepare_data.py."""

    qid: str
    input_ids: np.ndarray      # (L,) int64
    state_pos: np.ndarray      # (T+1,) int64  -- s_0 .. s_T
    span_start: np.ndarray     # (T,) int64
    span_end: np.ndarray       # (T,) int64
    correct: bool
    z: int                     # first error index, 0-based; -1 if fully correct (§6.1)
    recovery: bool = False
    # (T+1,) int64, aligned with state_pos: the §7.5.3-(b) join key that lets a CF example
    # reuse a prefix already forwarded in this batch. None on rows read from a parquet
    # written before 2026-08-15; `cf_attach` treats a None row as "carries no prefix" and
    # counts it, rather than crashing a run on a stale file.
    prefix_hash: np.ndarray | None = None

    @property
    def n_steps(self) -> int:
        return len(self.span_start)

    @property
    def length(self) -> int:
        return len(self.input_ids)


@dataclass
class Batch:
    input_ids: Tensor              # (B, L) int64
    attention_mask: Optional[Tensor]  # (B, L) int64, or None when nothing is padded

    # states s_0..s_T of every trajectory, flattened
    state_flat_idx: Tensor         # (S,) index into B*L -- gathers h at the separators
    state_traj: Tensor             # (S,)
    state_step: Tensor             # (S,) i in 0..T
    traj_state_offset: Tensor      # (B,) flat state index of s_0

    # source rows: phi_i = phi(h_{i-1}, act_emb_i) for i = 1..T
    row_src: Tensor                # (R,) state index of s_{i-1}
    row_dst: Tensor                # (R,) state index of s_i
    row_traj: Tensor               # (R,)
    row_step: Tensor               # (R,) i, 1-based

    # trajectory-level
    traj_qid: Tensor               # (B,) question index within the batch, 0..Q-1
    traj_correct: Tensor           # (B,) bool
    traj_T: Tensor                 # (B,)
    traj_z: Tensor                 # (B,) -1 if fully correct
    traj_recovery: Tensor          # (B,) bool
    traj_terminal: Tensor          # (B,) state index of s_T

    # segment-mean action pooling (§6.4): scatter token embeddings into rows
    span_token_idx: Tensor         # (P,) flat index into B*L
    span_row_idx: Tensor           # (P,)
    span_counts: Tensor            # (R,) float32

    qids: tuple[str, ...]          # per-trajectory question ids, for logging
    n_real_tokens: int
    n_pad_tokens: int

    # (S,) int64, aligned with state_flat_idx: `prefix_hash` of each state (§7.5.3-(b)).
    # 0 marks a state whose row carried no hash (a pre-2026-08-15 parquet); `cf_attach`
    # skips those rather than matching on a sentinel.
    state_prefix_hash: Tensor = None  # type: ignore[assignment]

    # ---- shapes ----
    @property
    def n_traj(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def n_states(self) -> int:
        return int(self.state_flat_idx.numel())

    @property
    def n_rows(self) -> int:
        return int(self.row_src.numel())

    @property
    def n_questions(self) -> int:
        return int(self.traj_qid.max().item()) + 1 if self.n_traj else 0

    @property
    def padding_fraction(self) -> float:
        total = self.n_real_tokens + self.n_pad_tokens
        return self.n_pad_tokens / total if total else 0.0

    def to(self, device) -> "Batch":
        moved = {}
        for name, value in self.__dict__.items():
            moved[name] = value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        return Batch(**moved)


def collate(rows: Sequence[SequenceRow], pad_id: int) -> Batch:
    """Pad to the batch max and build every index map. Pure CPU, pure numpy/torch."""
    if not rows:
        raise ValueError("empty batch")

    lengths = [r.length for r in rows]
    L = max(lengths)
    B = len(rows)

    input_ids = np.full((B, L), pad_id, dtype=np.int64)
    mask = np.zeros((B, L), dtype=np.int64)
    for b, row in enumerate(rows):
        input_ids[b, : row.length] = row.input_ids
        mask[b, : row.length] = 1

    qid_to_index: dict[str, int] = {}
    state_flat, state_traj, state_step, state_prefix_hash = [], [], [], []
    traj_offset, traj_terminal = [], []
    row_src, row_dst, row_traj, row_step = [], [], [], []
    span_token_idx, span_row_idx, span_counts = [], [], []
    traj_qid, traj_correct, traj_T, traj_z, traj_recovery = [], [], [], [], []

    for b, row in enumerate(rows):
        offset = len(state_flat)
        traj_offset.append(offset)
        T = row.n_steps

        # 0 for a row with no prefix_hash: `cf_attach` never matches on 0, so a stale
        # parquet degrades to "no CF attaches here" instead of mis-joining.
        row_prefix = row.prefix_hash if row.prefix_hash is not None else np.zeros(T + 1, dtype=np.int64)
        # A short `prefix_hash` would otherwise IndexError deep in the loop, or -- if it were
        # LONG -- silently bind CF examples to the wrong states. It must be one entry per
        # state, exactly like `state_pos`.
        if len(row_prefix) != T + 1:
            raise ValueError(
                f"prefix_hash has {len(row_prefix)} entries for a {T}-step trajectory "
                f"(expected T+1 = {T + 1}), qid {row.qid!r}. It is indexed by state, so it "
                "must be built from the SAME step list the sequence was tokenised from."
            )
        for i, pos in enumerate(row.state_pos):
            state_flat.append(b * L + int(pos))
            state_traj.append(b)
            state_step.append(i)
            state_prefix_hash.append(int(row_prefix[i]))
        traj_terminal.append(offset + T)

        for i in range(1, T + 1):
            r = len(row_src)
            row_src.append(offset + i - 1)     # s_{i-1}: the state phi_i departs from
            row_dst.append(offset + i)         # s_i:     the state phi_i lands in
            row_traj.append(b)
            row_step.append(i)
            start, end = int(row.span_start[i - 1]), int(row.span_end[i - 1])
            span_counts.append(float(end - start))
            for pos in range(start, end):
                span_token_idx.append(b * L + pos)
                span_row_idx.append(r)

        traj_qid.append(qid_to_index.setdefault(row.qid, len(qid_to_index)))
        traj_correct.append(bool(row.correct))
        traj_T.append(T)
        traj_z.append(int(row.z))
        traj_recovery.append(bool(row.recovery))

    n_real = int(sum(lengths))
    long_ = lambda x: torch.as_tensor(x, dtype=torch.long)  # noqa: E731

    return Batch(
        input_ids=torch.from_numpy(input_ids),
        attention_mask=None if min(lengths) == L else torch.from_numpy(mask),
        state_flat_idx=long_(state_flat),
        state_traj=long_(state_traj),
        state_step=long_(state_step),
        traj_state_offset=long_(traj_offset),
        row_src=long_(row_src),
        row_dst=long_(row_dst),
        row_traj=long_(row_traj),
        row_step=long_(row_step),
        traj_qid=long_(traj_qid),
        traj_correct=torch.as_tensor(traj_correct, dtype=torch.bool),
        traj_T=long_(traj_T),
        traj_z=long_(traj_z),
        traj_recovery=torch.as_tensor(traj_recovery, dtype=torch.bool),
        traj_terminal=long_(traj_terminal),
        span_token_idx=long_(span_token_idx),
        span_row_idx=long_(span_row_idx),
        span_counts=torch.as_tensor(span_counts, dtype=torch.float32),
        qids=tuple(r.qid for r in rows),
        n_real_tokens=n_real,
        n_pad_tokens=B * L - n_real,
        state_prefix_hash=long_(state_prefix_hash),
    )


def state_index_of(batch: Batch, traj: int, i: int) -> int:
    """Flat state index of s_i of trajectory `traj`. Used by tests and diagnostics."""
    T = int(batch.traj_T[traj])
    if not 0 <= i <= T:
        raise IndexError(f"state s_{i} does not exist: trajectory {traj} has T = {T}")
    return int(batch.traj_state_offset[traj]) + i
