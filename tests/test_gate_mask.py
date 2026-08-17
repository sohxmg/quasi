"""`scripts/goal_gate.py --mask-answer` -- §7.13.1's separator. CPU, no model, no GPU.

**Why this exists.** `lambda_term` was raised to 1.0 on 2026-08-15. §7.13.1 measured that the
printed final answer is a near-perfect shortcut into (7) `L_term` -- `answer_match_auc` 0.927
against a chance of 0.500, with an incorrect sibling essentially never printing a correct
sibling's answer (n = 0.001) -- so the term can be solved by clustering on the final NUMBER,
which transfers to nothing because a PRM scores *unfinished* solutions. §16.26 therefore
requires that a `lambda_term > 0` run be **designed to separate** two explanations of any
`gate/recall_at_1` improvement. This is that instrument, and these are its properties.

What is pinned here is the JOIN and the POPULATION, not the gate arithmetic
(`tests/test_goal_loss.py` owns that): the text a row is masked from must be the row's own,
and both passes must score the identical rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pytest

from feynman_prm.data.collate import SequenceRow
from feynman_prm.data.tokenize import build_sequence, sep_token_id
from feynman_prm.diagnostics.terminal_shortcut import strip_answer_span
from goal_gate import decode_row, mask_answer_row, matched_masked_pairs

PROMPT = "Janet has 3 apples and buys 5 more. How many?"
STEPS = ["Step 1: she starts with 3", "Step 2: 3 + 5 = 8", "Step 3: The answer is: 8"]


def _row(tokenizer, cfg, prompt=PROMPT, steps=STEPS, qid="q1") -> SequenceRow:
    seq = build_sequence(
        tokenizer, prompt, list(steps), sep_token_id(tokenizer, cfg.data.sep_token),
        prompt_format=cfg.data.prompt_format, max_len=cfg.data.max_len,
    )
    return SequenceRow(
        qid=qid,
        input_ids=np.asarray(seq.input_ids, dtype=np.int64),
        state_pos=np.asarray(seq.state_pos, dtype=np.int64),
        span_start=np.asarray([s for s, _ in seq.step_spans], dtype=np.int64),
        span_end=np.asarray([e for _, e in seq.step_spans], dtype=np.int64),
        correct=True, z=len(steps), recovery=False,
    )


# ------------------------------------------------------------------------------- the join

def test_decode_row_recovers_the_rows_own_prompt_and_steps(tokenizer, cfg):
    """The parquet carries no text, and the alternative -- re-reading math_shepherd and
    joining on `qid` -- is a join that can silently mismatch. The row already knows where
    everything is (`state_pos[0]`, `span_start/span_end`), so this is exact by construction.
    """
    prompt, steps = decode_row(_row(tokenizer, cfg), tokenizer)
    assert prompt == PROMPT
    assert steps == STEPS


def test_the_answer_span_means_the_same_thing_here_as_in_the_availability_measurement(
    tokenizer, cfg
):
    """`strip_answer_span` is IMPORTED from `diagnostics/terminal_shortcut.py`, not re-written.

    §7.13.1's 0.927 and this instrument must agree on what "the answer span" is, or the two
    numbers are not comparable while looking like they are.
    """
    masked = mask_answer_row(_row(tokenizer, cfg), tokenizer, sep_token_id(tokenizer, "\n"), cfg)
    _, steps = decode_row(masked, tokenizer)
    assert steps[-1] == strip_answer_span(STEPS[-1])
    assert "8" not in steps[-1], "the printed answer survived masking"
    assert steps[:-1] == STEPS[:-1], "masking touched a step other than the last"
    assert len(steps) == len(STEPS), "masking changed T, which moves every state index"


def test_masking_leaves_the_state_count_and_the_separator_invariant(tokenizer, cfg):
    """T+1 states for T steps (§3), and every `state_pos` still holds the separator -- the
    §6.1 conventions the re-tokenised row has to keep for the gate to read the same thing."""
    row = _row(tokenizer, cfg)
    masked = mask_answer_row(row, tokenizer, sep_token_id(tokenizer, "\n"), cfg)
    sep = sep_token_id(tokenizer, "\n")
    assert len(masked.state_pos) == len(masked.span_start) + 1
    assert len(masked.state_pos) == len(row.state_pos)
    assert all(masked.input_ids[p] == sep for p in masked.state_pos)


def test_a_step_that_is_nothing_but_the_answer_cannot_be_masked(tokenizer, cfg):
    """Stripping empties it, and `build_sequence` rejects a zero-token step. Returning None
    is the honest outcome; substituting a placeholder would silently feed the encoder a token
    that appears nowhere in training."""
    row = _row(tokenizer, cfg, steps=["Step 1: 3 + 5 = 8", "The answer is: 8"])
    assert mask_answer_row(row, tokenizer, sep_token_id(tokenizer, "\n"), cfg) is None


def test_a_terminal_with_no_printed_answer_is_masked_to_itself(tokenizer, cfg):
    """4.7% of correct terminals print no answer at all (§7.13.1). Those are not dropped --
    there is nothing to remove, so the row is unchanged and still comparable."""
    steps = ["Step 1: 3 + 5 = 8", "Step 2: so she has eight apples"]
    row = _row(tokenizer, cfg, steps=steps)
    masked = mask_answer_row(row, tokenizer, sep_token_id(tokenizer, "\n"), cfg)
    assert masked is not None
    assert decode_row(masked, tokenizer)[1] == steps


# -------------------------------------------------------------------------- the population

def test_both_passes_score_the_identical_rows(tokenizer, cfg):
    """**The property the comparison rests on.** Drop an unmaskable row from the masked pass
    alone and `recall@1` moves because the population moved, not because the representation
    did -- and that delta would be read as shortcut evidence. Same length, same order, same
    question indices."""
    pairs = [
        (0, _row(tokenizer, cfg, qid="q1")),
        (0, _row(tokenizer, cfg, steps=["Step 1: 3+5", "Step 2: The answer is: 8"], qid="q1")),
        (1, _row(tokenizer, cfg, qid="q2")),
        (1, _row(tokenizer, cfg, qid="q2")),
    ]
    plain, masked, dropped = matched_masked_pairs(pairs, tokenizer, cfg)

    assert len(plain) == len(masked) == dropped["rows_kept"]
    assert [qi for qi, _ in plain] == [qi for qi, _ in masked]
    assert all(p.qid == m.qid for (_, p), (_, m) in zip(plain, masked))


def test_a_question_that_falls_below_two_terminals_is_dropped_from_BOTH(tokenizer, cfg):
    """A question needs >= 2 correct terminals to contribute a within-question pair at all.
    If masking takes one of a 2-terminal question's rows, the survivor measures nothing and
    must not sit in either pass inflating `gate/questions`."""
    unmaskable = _row(tokenizer, cfg, steps=["Step 1: 3+5", "The answer is: 8"], qid="q1")
    pairs = [
        (0, _row(tokenizer, cfg, qid="q1")),
        (0, unmaskable),
        (1, _row(tokenizer, cfg, qid="q2")),
        (1, _row(tokenizer, cfg, qid="q2")),
    ]
    plain, masked, dropped = matched_masked_pairs(pairs, tokenizer, cfg)

    assert dropped["rows_unmaskable"] == 1
    assert dropped["rows_lost_with_their_question"] == 1, "q1's survivor must go too"
    assert {qi for qi, _ in plain} == {1}
    assert dropped["questions_kept"] == 1 and dropped["questions_total"] == 2


def test_the_drop_is_counted_and_not_absorbed(tokenizer, cfg):
    """§14: a drop that is relied on is counted. Every row is accounted for in exactly one
    bucket, so a silently vanishing population cannot look like a healthy one."""
    pairs = [(0, _row(tokenizer, cfg)), (0, _row(tokenizer, cfg)), (1, _row(tokenizer, cfg))]
    _, _, dropped = matched_masked_pairs(pairs, tokenizer, cfg)
    assert (
        dropped["rows_kept"]
        + dropped["rows_unmaskable"]
        + dropped["rows_lost_with_their_question"]
        == dropped["rows_total"]
    )


def test_mask_answer_defaults_off_so_the_recorded_gate_numbers_stay_comparable(cfg):
    """§9.8.3's baseline (recall@1 0.618 untrained / 0.276 trained) was measured without any
    of this. The masking path must be opt-in, and the plain path must not acquire the
    maskability filter -- otherwise the new numbers silently stop being comparable."""
    import argparse

    from goal_gate import add_gate_args

    args = add_gate_args(argparse.ArgumentParser()).parse_args(["--checkpoint", "x"])
    assert args.mask_answer is False
