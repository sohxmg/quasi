"""Batch assembly: padding to the batch max plus every index map (PLAN 1, §4.6)."""

from __future__ import annotations

import torch

from feynman_prm.data.collate import collate, state_index_of
from conftest import synthetic_row


def test_pads_to_batch_max_not_max_len():
    rows = [synthetic_row("q", [True], step_len=2), synthetic_row("q", [True] * 5, step_len=2)]
    batch = collate(rows, pad_id=0)
    assert batch.seq_len == max(r.length for r in rows)
    assert batch.attention_mask is not None
    assert int(batch.attention_mask[0].sum()) == rows[0].length


def test_no_attention_mask_when_nothing_is_padded():
    """A batch with no padding needs no mask at all, which lets SDPA pick its fastest
    backend -- the flash path cannot take an arbitrary mask (PLAN 4a)."""
    rows = [synthetic_row("q", [True, True]), synthetic_row("q", [False, False])]
    batch = collate(rows, pad_id=0)
    assert batch.attention_mask is None
    assert batch.padding_fraction == 0.0


def test_state_and_row_counts():
    rows = [synthetic_row("q1", [True, True, True]), synthetic_row("q1", [True, False])]
    batch = collate(rows, pad_id=0)
    assert batch.n_states == (3 + 1) + (2 + 1), "T+1 states per trajectory (§3)"
    assert batch.n_rows == 3 + 2, "one source row per step"
    assert int(batch.traj_terminal[0]) == state_index_of(batch, 0, 3)


def test_question_index_groups_trajectories():
    rows = [
        synthetic_row("qA", [True]),
        synthetic_row("qB", [True]),
        synthetic_row("qA", [False]),
    ]
    batch = collate(rows, pad_id=0)
    assert batch.traj_qid.tolist() == [0, 1, 0]
    assert batch.n_questions == 2


def test_state_flat_index_points_at_the_separator():
    rows = [synthetic_row("q", [True, True]), synthetic_row("q", [True] * 4)]
    batch = collate(rows, pad_id=0)
    flat = batch.input_ids.reshape(-1)
    assert torch.all(flat[batch.state_flat_idx] == 1), "every state sits on a separator id"


def test_action_spans_pool_the_step_tokens_only():
    """Segment-mean action pooling (§6.4): the spans cover step tokens, never separators,
    and index_add over them reproduces a plain mean."""
    from feynman_prm.model.heads import MeanActionPool

    rows = [synthetic_row("q", [True, True], step_len=3)]
    batch = collate(rows, pad_id=0)
    H = 4
    emb = torch.arange(batch.seq_len * H, dtype=torch.float32).reshape(1, batch.seq_len, H)
    act = MeanActionPool()(
        emb, batch.span_token_idx, batch.span_row_idx, batch.span_counts, batch.n_rows
    )
    assert act.shape == (2, H)
    start, end = int(rows[0].span_start[0]), int(rows[0].span_end[0])
    assert torch.allclose(act[0], emb[0, start:end].mean(dim=0))
    assert batch.span_counts.tolist() == [3.0, 3.0]


def test_z_and_correctness_travel_with_the_row():
    rows = [synthetic_row("q", [True, True, False, False]), synthetic_row("q", [True, True])]
    batch = collate(rows, pad_id=0)
    assert batch.traj_z.tolist() == [2, -1]
    assert batch.traj_correct.tolist() == [False, True]
    assert batch.traj_T.tolist() == [4, 2]
