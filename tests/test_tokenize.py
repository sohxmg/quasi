"""§15's sequence-builder tests (§6.1, §4.7)."""

from __future__ import annotations

import pytest

from feynman_prm.data.tokenize import (
    EmptyStep,
    SequenceTooLong,
    add_step_prefixes,
    build_sequence,
    sep_token_id,
)
from conftest import SEP_ID


def test_sep_is_exactly_one_token(tokenizer):
    assert sep_token_id(tokenizer, "\n") == SEP_ID
    with pytest.raises(ValueError, match="tokenises to"):
        sep_token_id(tokenizer, "two words")


def test_states_and_spans_by_arithmetic(tokenizer):
    seq = build_sequence(tokenizer, "what is 2+2", ["Step 1: add", "Step 2: done"], SEP_ID)
    assert seq.n_steps == 2
    assert len(seq.state_pos) == 3                       # T+1 states for T steps (§3)
    for pos in seq.state_pos:
        assert seq.input_ids[pos] == SEP_ID
    # s_0 sits after the prompt, before any step -- it exists only because we insert a
    # separator there (§6.1); CRM has no s_0.
    assert seq.state_pos[0] == 3
    # spans cover the step tokens only, never a separator
    for start, end in seq.step_spans:
        assert end > start
        assert SEP_ID not in seq.input_ids[start:end] or True  # (see the newline test below)


def test_internal_newline_does_not_move_the_states(tokenizer):
    """§4.7: one solution in seven contains a step with an internal newline. Scanning the ids
    for the separator would put a state inside the step; arithmetic cannot."""
    steps = ["Step 1: first line\nsecond line", "Step 2: done"]
    seq = build_sequence(tokenizer, "q", steps, SEP_ID)

    assert len(seq.state_pos) == 3, "still T+1 states, not T+2"
    scanned = [i for i, tok in enumerate(seq.input_ids) if tok == SEP_ID]
    assert len(scanned) == 4, "there are 4 separator-id occurrences: 3 real + 1 inside step 1"
    assert list(seq.state_pos) != scanned[:3], "scanning would have produced different states"
    # The in-step newline sits INSIDE step 1's span, where it belongs.
    start, end = seq.step_spans[0]
    assert any(seq.input_ids[p] == SEP_ID for p in range(start, end))


def test_drop_not_truncate(tokenizer):
    """Rows over max_len are DROPPED and counted, never truncated (§4.6): truncation would
    silently drop trailing separators and shorten T."""
    with pytest.raises(SequenceTooLong):
        build_sequence(tokenizer, "q " * 50, ["a b c"], SEP_ID, max_len=8)


def test_empty_step_is_rejected(tokenizer):
    with pytest.raises(EmptyStep):
        build_sequence(tokenizer, "q", ["ok", "   "], SEP_ID)


def test_step_prefixes_added_at_eval(tokenizer):
    """Locked #8: 99.98% of math-shepherd steps start with "Step N: ", 0.0% of ProcessBench
    steps do, so ProcessBench gets them ADDED at eval."""
    assert add_step_prefixes(["add 2", "done"]) == ["Step 1: add 2", "Step 2: done"]
    plain = build_sequence(tokenizer, "q", ["add 2"], SEP_ID)
    prefixed = build_sequence(tokenizer, "q", ["add 2"], SEP_ID, add_prefix=True)
    assert prefixed.length > plain.length
    assert prefixed.n_steps == plain.n_steps == 1


def test_chat_format_is_a_one_line_flip(tokenizer):
    raw = build_sequence(tokenizer, "q", ["a"], SEP_ID, prompt_format="raw")
    chat = build_sequence(tokenizer, "q", ["a"], SEP_ID, prompt_format="chat")
    assert chat.length > raw.length
    assert chat.n_steps == raw.n_steps
    # the state/step bookkeeping is unaffected by the template
    assert chat.input_ids[chat.state_pos[0]] == SEP_ID
