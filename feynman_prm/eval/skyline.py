"""The skyline: reference-solution goals, LABELLED AS A SKYLINE, never reported as a result
(§9.5, and §5.1's hard rule).

§5.1 measured that knowing only `final_answer_correct` -- which the gold answer gives you
free -- solves the "no error" half of the F1 exactly (100.0%) on all four subsets, while
published PRMs score in the 60s-70s. So any scoring path that consumes a reference solution
or gold answer is a skyline.

Its only job is to split one failure mode into two:

    skyline good, goal-head bad  -> the geometry works, the goal head is the bottleneck
    both bad                     -> the metric never learned correctness. That is the old
                                    project again, and you know it in one run.

Expect the reference terminals to be OUT OF DISTRIBUTION for psi (terse human-written GSM8K
arithmetic, competition-style Omni-MATH prose, against Mistral-generated Math-Shepherd
steps). The skyline may legitimately underperform the goal head. That is informative, not a
bug.

Joins (§5.2, verified by problem-text join): gsm8k 400/400 against `openai/gsm8k` test
`answer`; omnimath 1000/1000 against `KbsdJames/Omni-MATH` `solution`; math 112/1000 against
`HuggingFaceH4/MATH-500`. olympiadbench was not attempted. **Do not spend effort chasing the
rest** -- 1,400 samples across one easy and one hard subset is enough for the diagnostic.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ..config import Config
from ..data.collate import SequenceRow, collate
from ..data.tokenize import SequenceTooLong, build_sequence, sep_token_id
from .processbench import Sample

REFERENCE_SOURCES = {
    "gsm8k": ("openai/gsm8k", "main", "test", "question", "answer"),
    "math": ("HuggingFaceH4/MATH-500", None, "test", "problem", "solution"),
    "omnimath": ("KbsdJames/Omni-MATH", None, "test", "problem", "solution"),
}

_WS = re.compile(r"\s+")


def normalise_problem(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def load_reference_solutions(subset: str) -> dict[str, str]:
    """{normalised problem -> reference solution text} for one subset, or {} if unjoinable."""
    if subset not in REFERENCE_SOURCES:
        return {}
    from datasets import load_dataset

    name, config, split, problem_key, solution_key = REFERENCE_SOURCES[subset]
    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    return {normalise_problem(row[problem_key]): row[solution_key] for row in ds}


def split_reference_into_steps(solution: str) -> list[str]:
    """Reference solutions are prose, not "Step N:" lists. Split on blank/newline boundaries
    and keep non-empty lines; the terminal state is all we read, so the split only has to be
    reasonable, not faithful."""
    steps = [line.strip() for line in solution.splitlines() if line.strip()]
    return steps or [solution.strip()]


@torch.no_grad()
def reference_goal_fn(
    model,
    tokenizer,
    cfg: Config,
    device,
    references: dict[str, str],
):
    """Build a `goal_fn` for `processbench.score_samples`: g = psi(terminal of the reference
    solution). Falls back to the goal head for samples with no join."""
    sep_id = sep_token_id(tokenizer, cfg.data.sep_token)
    pad_id = tokenizer.pad_token_id
    cache: dict[str, Optional[Tensor]] = {}

    def encode(sample: Sample) -> Optional[Tensor]:
        key = normalise_problem(sample.problem)
        if key in cache:
            return cache[key]
        solution = references.get(key)
        result: Optional[Tensor] = None
        if solution is not None:
            try:
                seq = build_sequence(
                    tokenizer,
                    sample.problem,
                    split_reference_into_steps(solution),
                    sep_id,
                    prompt_format=cfg.data.prompt_format,
                    max_len=cfg.eval.max_len,
                    add_prefix=True,
                )
                row = SequenceRow(
                    qid=sample.id,
                    input_ids=np.asarray(seq.input_ids, dtype=np.int64),
                    state_pos=np.asarray(seq.state_pos, dtype=np.int64),
                    span_start=np.asarray([s for s, _ in seq.step_spans], dtype=np.int64),
                    span_end=np.asarray([e for _, e in seq.step_spans], dtype=np.int64),
                    correct=True,
                    z=-1,
                )
                batch = collate([row], pad_id=pad_id).to(device)
                reps = model(batch)
                result = reps.psi[int(batch.traj_terminal[0])].detach()
            except (SequenceTooLong, ValueError):
                result = None
        cache[key] = result
        return result

    def goal_fn(h_s0: Tensor, samples: Sequence[Sample]) -> Tensor:
        goals = []
        for b, sample in enumerate(samples):
            ref = encode(sample)
            if ref is None:
                if model.goal_head is None:
                    raise RuntimeError("skyline fallback needs the phase-2 goal head")
                ref = model.goal_head(h_s0[b : b + 1])[0]
            goals.append(ref)
        return torch.stack(goals)

    return goal_fn


def joinable_count(samples: Sequence[Sample], references: dict[str, str]) -> int:
    return sum(normalise_problem(s.problem) in references for s in samples)
