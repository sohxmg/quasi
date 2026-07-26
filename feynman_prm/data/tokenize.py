"""THE sequence builder. Train, val, ProcessBench eval and the skyline all call this.

There is exactly one definition of a sequence in this repo, enforced by a grep test
(tests/test_grep_invariants.py). The old project's root cause I was two prompt templates
diverging by whitespace.

    ids = tok(prompt) + [SEP] + tok(step_1) + [SEP] + ... + tok(step_T) + [SEP]
                         ^                     ^                           ^
                        s_0                   s_1                         s_T

Separator positions come out **by arithmetic**, because each segment is tokenised
separately (§6.1). §4.7's hazard -- 13.9% of solutions contain a step with an internal
newline, so scanning the ids for "\\n" gives wrong state positions on one solution in seven
-- cannot occur at all here. CRM's post-hoc `assert ids[pos] == sep_id`
(`BoN/eval/eval_BoN.py:118-134`) is kept anyway as a cheap guard.

A separator is inserted after the prompt as well, so `s_0` exists. This differs from CRM,
which has no s_0; we need it because the goal head reads `h_{s_0}` (§7.7).
"""

from __future__ import annotations

from dataclasses import dataclass

STEP_PREFIX = "Step {n}: "


class SequenceTooLong(ValueError):
    """Raised instead of truncating (§4.6). Callers DROP the row and count it.

    CRM never truncates on purpose (`train_RM.py:47-49`) because truncation promotes a
    mid-trajectory label to "outcome". Our analogue: truncation silently drops trailing
    separators and shortens T.
    """


class EmptyStep(ValueError):
    """A step that tokenises to zero tokens would make its action embedding a mean over an
    empty set (§6.4). Callers drop the row and count it."""


@dataclass(frozen=True)
class TokenizedSequence:
    input_ids: tuple[int, ...]
    state_pos: tuple[int, ...]          # T+1 positions: s_0 .. s_T
    step_spans: tuple[tuple[int, int], ...]  # T half-open [start, end) token ranges, steps only

    @property
    def n_steps(self) -> int:
        return len(self.step_spans)

    @property
    def length(self) -> int:
        return len(self.input_ids)


def sep_token_id(tokenizer, sep_token: str = "\n") -> int:
    """Assert the separator is exactly ONE token and return its id (§6.1, §13).

    CRM does the same at `eval_BoN.py:118-120`.
    """
    ids = tokenizer(sep_token, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(
            f"sep_token {sep_token!r} tokenises to {len(ids)} ids ({ids}), not 1. The "
            "separator has to be a single token: it is the position the state is read at."
        )
    return int(ids[0])


def format_prompt(tokenizer, prompt: str, prompt_format: str) -> str:
    """`raw` is CRM-faithful (`eval_BoN.py`: the raw question text, no template). `chat` is
    the Qwen template, a one-line flip. §6.1 is silent on this; see IMPLEMENTATION.md.
    """
    if prompt_format == "raw":
        return prompt
    if prompt_format == "chat":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError(f"unknown prompt_format {prompt_format!r}")


def add_step_prefixes(steps: list[str]) -> list[str]:
    """Locked #8: 99.98% of math-shepherd steps start with "Step N: " and 0.0% of
    ProcessBench steps do, so ProcessBench steps get the prefix ADDED at eval (§4.7)."""
    return [STEP_PREFIX.format(n=i + 1) + s for i, s in enumerate(steps)]


def build_sequence(
    tokenizer,
    prompt: str,
    steps: list[str],
    sep_id: int,
    prompt_format: str = "raw",
    max_len: int | None = None,
    add_prefix: bool = False,
) -> TokenizedSequence:
    """Build one trajectory's token sequence and its exact state/step index maps.

    Raises SequenceTooLong if the result exceeds `max_len` -- we drop, never truncate.
    """
    if not steps:
        raise EmptyStep("a trajectory with zero steps has no s_1..s_T")
    if add_prefix:
        steps = add_step_prefixes(steps)

    prompt_text = format_prompt(tokenizer, prompt, prompt_format)
    ids = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

    state_pos = [len(ids)]          # s_0 sits at the separator that follows the prompt
    ids.append(sep_id)
    spans: list[tuple[int, int]] = []

    for step in steps:
        step_ids = tokenizer(step, add_special_tokens=False)["input_ids"]
        if len(step_ids) == 0:
            raise EmptyStep(f"step tokenises to zero tokens: {step!r}")
        start = len(ids)
        ids.extend(step_ids)
        spans.append((start, len(ids)))
        state_pos.append(len(ids))  # s_i sits at the separator that follows step i
        ids.append(sep_id)

    if max_len is not None and len(ids) > max_len:
        raise SequenceTooLong(f"{len(ids)} tokens > max_len {max_len}")

    # Belt and braces, in CRM's spirit: the arithmetic above is exact, so this can only
    # fire if the tokenizer emitted the separator id from inside a segment boundary.
    for pos in state_pos:
        if ids[pos] != sep_id:
            raise AssertionError(
                f"state position {pos} holds id {ids[pos]}, not the separator {sep_id}"
            )
    assert len(state_pos) == len(spans) + 1, "T+1 states for T steps (§3)"

    return TokenizedSequence(
        input_ids=tuple(ids), state_pos=tuple(state_pos), step_spans=tuple(spans)
    )
