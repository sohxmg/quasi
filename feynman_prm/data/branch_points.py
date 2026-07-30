"""Mine the branch points: same prefix, one correct continuation and one incorrect (§8.3).

§4.4 measured that ~95% of states have exactly one observed action -- essentially no natural
branching, which is a fundamental difference from OGBench where trajectories cross
constantly. The 4.5% that do branch are high-value. **COUNTED on full train 2026-07-27:
102,988 branching nodes, 30,344 of them correctness-disagreeing (29.5%)** -- normalised; 97,103
and 28,551 on exact match. §4.4's "~63,000 disagreeing" was extrapolated from a 4,000-question
sample and is 2.1x too high; the branching total it predicted (~105,000) was right. That is
"same state, right move vs wrong move", i.e. exactly the credit-assignment data.

Mining the SELECTED 23,000 training questions gives ~15,900 disagreeing points, not 30,344 --
scale by the question count before comparing the two.

**These are NOT L_CF data** (PLAN finding 2, a flagged deviation from §8.3). L_CF is a
cross-entropy that needs a meaning-PRESERVING positive; a branch point is two
meaning-CHANGING continuations with a correctness label. They are mined and written to disk
as §8.3's permanently held-out diagnostic set, and L_CF stays at lambda 0 until real
rewrites exist, rather than wiring in data that does not fit the loss.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .math_shepherd import Question

_CALCULATOR = re.compile(r"<<[^>]*>>")
_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")


def normalise_step(step: str) -> str:
    """Case, punctuation and calculator annotations removed -- the normalisation §4.4 used
    for its second column (5.5% -> 7.1% of pairs sharing a leading step)."""
    text = _CALCULATOR.sub(" ", step.lower())
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


@dataclass(frozen=True)
class BranchPoint:
    qid: str
    prompt: str
    depth: int                    # 0-based index of the branching step
    prefix: tuple[str, ...]       # steps[0:depth], shared by both continuations
    correct_step: str             # continuation labelled True at `depth`
    incorrect_step: str           # continuation labelled False at `depth`
    n_children: int               # distinct continuations observed at this node


def mine_question(question: Question, normalise: bool = True) -> tuple[list[BranchPoint], int]:
    """Branch points of one question, plus the number of branching nodes seen.

    A node is (question, prefix). Its children are the distinct next steps. It is a branch
    point when it has >1 distinct child, and correctness-disagreeing when two children carry
    different labels at that depth.
    """
    key = normalise_step if normalise else (lambda s: s)
    nodes: dict[tuple[str, ...], dict[str, tuple[str, bool]]] = {}

    for traj in question.trajectories:
        for depth, step in enumerate(traj.steps):
            prefix = tuple(key(s) for s in traj.steps[:depth])
            child_key = key(step)
            node = nodes.setdefault(prefix, {})
            # First occurrence wins; a later duplicate with a different label is label noise
            # and is not what we are mining for.
            node.setdefault(child_key, (step, bool(traj.labels[depth])))

    found: list[BranchPoint] = []
    n_branching = 0
    for prefix, children in nodes.items():
        if len(children) < 2:
            continue
        n_branching += 1
        good = next((v for v in children.values() if v[1]), None)
        bad = next((v for v in children.values() if not v[1]), None)
        if good is None or bad is None:
            continue
        found.append(
            BranchPoint(
                qid=question.qid,
                prompt=question.prompt,
                depth=len(prefix),
                prefix=prefix,
                correct_step=good[0],
                incorrect_step=bad[0],
                n_children=len(children),
            )
        )
    return found, n_branching


def mine(questions: Iterable[Question], normalise: bool = True) -> tuple[list[BranchPoint], dict]:
    """Mine every question. Returns (branch points, counters)."""
    points: list[BranchPoint] = []
    counters = {"branching_nodes": 0, "disagreeing_branch_points": 0, "questions": 0}
    for question in questions:
        found, n_branching = mine_question(question, normalise=normalise)
        points.extend(found)
        counters["questions"] += 1
        counters["branching_nodes"] += n_branching
    counters["disagreeing_branch_points"] = len(points)
    return points, counters


def write_jsonl(points: Sequence[BranchPoint], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for point in points:
            record = asdict(point)
            record["prefix"] = list(record["prefix"])
            fh.write(json.dumps(record) + "\n")


def read_jsonl(path: str | Path) -> list[BranchPoint]:
    points = []
    with Path(path).open() as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                record["prefix"] = tuple(record["prefix"])
                points.append(BranchPoint(**record))
    return points
