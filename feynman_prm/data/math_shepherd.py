"""Math-Shepherd loading, filtering, question-level selection and splitting (§4, §8.2).

Dataset: `trl-lib/math_shepherd`, schema `prompt: str`, `completions: list[str]`,
`labels: list[bool]`. Same dataset CRM uses (`CRM/scripts/train_our.sh:11`).

Nothing here imports `datasets` at module level, so the whole selection/splitting path is
testable on hand-built rows with no network and no HuggingFace install.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from ..utils.indexing import first_error_index, has_recovery, trajectory_is_correct

DATASET_NAME = "trl-lib/math_shepherd"

# §4.1/§4.2, measured from the train parquet. prepare_data.py checks itself against these.
EXPECTED_TRAIN = {
    "rows_raw": 422_422,
    "rows_after_empty_label_filter": 422_407,
    "questions": 45_989,
    "trainable_questions": 40_247,      # >=1 correct AND >=1 incorrect (87.5%)
    "fraction_correct": 0.366,          # of trajectories
}


@dataclass(frozen=True)
class Trajectory:
    steps: tuple[str, ...]
    labels: tuple[bool, ...]

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def correct(self) -> bool:
        return trajectory_is_correct(self.labels)

    @property
    def z(self) -> int:
        """First error index, 0-based; -1 for a fully correct trajectory (§6.1)."""
        z = first_error_index(self.labels)
        return -1 if z is None else z

    @property
    def recovery(self) -> bool:
        """A False -> True label flip. 1.48% of trajectories (§4.2, §16.15)."""
        return has_recovery(self.labels)


@dataclass
class Question:
    qid: str
    prompt: str
    trajectories: list[Trajectory] = field(default_factory=list)

    @property
    def n_correct(self) -> int:
        return sum(t.correct for t in self.trajectories)

    @property
    def n_incorrect(self) -> int:
        return sum(not t.correct for t in self.trajectories)

    @property
    def trainable(self) -> bool:
        """§8.2: >=1 correct (to supply L_step's terminal goal g) and >=1 incorrect (to
        supply an error index z). 87.5% of questions qualify."""
        return self.n_correct >= 1 and self.n_incorrect >= 1


def question_id(prompt: str) -> str:
    """Stable question id: sha1 of the exact prompt text. Questions are grouped by exact
    prompt, which is how §4.1's 45,989 was measured."""
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def iter_hf_rows(dataset_name: str = DATASET_NAME, split: str = "train") -> Iterator[dict]:
    """Stream rows from HuggingFace. `datasets` is imported here, not at module level.

    §4.3(a): the `test` split is unusable as validation -- 99.6% of its questions also
    appear in `train`. Carve our own split by question instead (§2 #2).
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    for row in ds:
        yield row


def build_questions(rows: Iterable[dict]) -> tuple[list[Question], dict[str, int]]:
    """Group rows by prompt, dropping the 15 empty-label rows (§4.7, CRM does the same).

    Returns (questions in first-seen order, counters).
    """
    counters = {"rows_raw": 0, "dropped_empty_labels": 0, "dropped_length_mismatch": 0}
    by_qid: dict[str, Question] = {}
    for row in rows:
        counters["rows_raw"] += 1
        labels = row["labels"]
        completions = row["completions"]
        if not labels:
            counters["dropped_empty_labels"] += 1
            continue
        if len(labels) != len(completions):
            counters["dropped_length_mismatch"] += 1
            continue
        prompt = row["prompt"]
        qid = question_id(prompt)
        question = by_qid.get(qid)
        if question is None:
            question = Question(qid=qid, prompt=prompt)
            by_qid[qid] = question
        question.trajectories.append(
            Trajectory(steps=tuple(completions), labels=tuple(bool(x) for x in labels))
        )
    return list(by_qid.values()), counters


def dataset_stats(questions: Sequence[Question]) -> dict[str, float]:
    """Reproduce §4.1/§4.2's measured numbers as a self-check (milestone §18.1)."""
    n_traj = sum(len(q.trajectories) for q in questions)
    n_correct = sum(q.n_correct for q in questions)
    steps = [t.n_steps for q in questions for t in q.trajectories]
    z_values = [t.z for q in questions for t in q.trajectories if not t.correct]
    rel_pos = [
        t.z / t.n_steps for q in questions for t in q.trajectories if not t.correct and t.n_steps
    ]
    recoveries = sum(t.recovery for q in questions for t in q.trajectories)
    trainable = [q for q in questions if q.trainable]
    ge1_correct = sum(q.n_correct >= 1 for q in questions)
    ge2_correct = sum(q.n_correct >= 2 for q in questions)
    pairs = sum(q.n_correct * q.n_incorrect for q in questions)
    # §4.2 quotes this as a FRACTION ("100.0%"), so report a fraction. It used to be
    # `all(...)`, which is a boolean over 422,407 rows and therefore reads 0.0 the moment a
    # single row disagrees -- which four of them do, so the real prepare_data run printed
    # `all_labels_equals_last_label: 0.0` and looked like the dataset had changed under us.
    # **Measured 2026-07-27: 4 rows of 422,407 (0.0009%)**, all of the shape
    # `[T ... T, F, T ... T]` -- a recovery whose last label is True. 6,263 rows (1.48%) have a
    # recovery at all; almost all of them fall back to False before the end, which is why the
    # two §4.2 facts looked contradictory but are not. We compute `all(labels)` everywhere
    # (`trajectory_is_correct`) and never rely on the last label, so those four rows are simply
    # treated as incorrect with z at their first False, per §16.15.
    n_last_label_matches = sum(
        t.correct == t.labels[-1] for q in questions for t in q.trajectories
    )
    return {
        "questions": len(questions),
        "trajectories": n_traj,
        "trajectories_per_question_mean": n_traj / max(len(questions), 1),
        "fraction_correct": n_correct / max(n_traj, 1),
        "steps_per_solution_mean": sum(steps) / max(len(steps), 1),
        "states_total": sum(s + 1 for s in steps),
        "first_error_index_mean": sum(z_values) / max(len(z_values), 1),
        "first_error_relative_mean": sum(rel_pos) / max(len(rel_pos), 1),
        "fraction_z_zero": sum(z == 0 for z in z_values) / max(len(z_values), 1),
        "recovery_fraction": recoveries / max(n_traj, 1),
        "questions_ge1_correct": ge1_correct / max(len(questions), 1),
        # §16.16 / §4.2.1: the one number CLAUDE.md had neither measured nor estimated.
        "questions_ge2_correct": ge2_correct / max(len(questions), 1),
        "trainable_questions": len(trainable),
        "trainable_fraction": len(trainable) / max(len(questions), 1),
        "within_question_pairs": pairs,
        "correct_per_question_mean": n_correct / max(len(questions), 1),
        "all_labels_equals_last_label": n_last_label_matches / max(n_traj, 1),
        "last_label_disagreements": n_traj - n_last_label_matches,     # 4 on full train
    }


def selection_sha(qids: Sequence[str]) -> str:
    """SHA256 of the SORTED question id list (§8.2, old R10).

    The old project documented a "seed 42 shuffle" for two years that turned out not to
    exist. Log the actual selection, not the seed that implies it.
    """
    payload = "\n".join(sorted(qids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_sequences_parquet(rows: Sequence[dict], path: str | Path) -> None:
    """Pre-tokenised sequences, written ONCE by scripts/prepare_data.py (PLAN 3a).

    Training then does no tokenisation in the hot path, collate.py is pure padding plus
    index building, and the real tokenised length distribution comes out as a byproduct --
    which is the measurement §4.6/§17 flags as the only estimated number in CLAUDE.md.
    """
    import pandas as pd

    from .sequence_cache import cache_path_for, columns_from_rows, save_columns

    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    # The numpy cache every reader actually loads (§14 B15). Written here from the rows we
    # already hold, so the common path never pays a read-back and a fresh parquet is never
    # one launch away from a conversion. It is keyed on the parquet's size and mtime, so it
    # must be written AFTER the parquet, not before.
    save_columns(columns_from_rows(rows), cache_path_for(path), path)


def read_sequences_parquet(path: str | Path, split: str | None = None) -> list:
    """Returns a list of `data.collate.SequenceRow`.

    **The parquet is NOT read in this process** -- `sequence_cache` converts it once in a
    child interpreter that has no torch in it, and this reads flat numpy arrays (§14 B15).
    `FEYNMAN_SEQUENCE_CACHE=0` restores the direct pandas read.
    """
    import numpy as np

    from .collate import SequenceRow
    from .sequence_cache import load_sequence_columns

    columns = load_sequence_columns(path)
    n_rows = len(columns["qid"])

    keep = np.arange(n_rows)
    if split is not None:
        if "split" not in columns:
            raise KeyError(f"{path} carries no `split` column; cannot select split={split!r}")
        keep = keep[columns["split"] == split]

    # Absent on a parquet written before 2026-08-15. `cf_attach` counts rows with no hash and
    # attaches nothing to them, so an old file trains exactly as it did before instead of
    # failing at the first micro-batch.
    has_prefix_hash = "prefix_hash__offsets" in columns

    def span(name: str, i: int, dtype) -> np.ndarray:
        # `np.array`, not `np.asarray`: every row owns its own buffer, as it did when each one
        # came out of `np.asarray(record.input_ids, ...)`. A view would leave rows aliasing one
        # flat array, so an in-place edit anywhere would silently reach into its neighbours.
        offsets = columns[f"{name}__offsets"]
        return np.array(columns[f"{name}__values"][offsets[i]:offsets[i + 1]], dtype=dtype)

    out = []
    for i in keep:
        i = int(i)
        out.append(
            SequenceRow(
                qid=str(columns["qid"][i]),
                input_ids=span("input_ids", i, np.int64),
                state_pos=span("state_pos", i, np.int64),
                span_start=span("span_start", i, np.int64),
                span_end=span("span_end", i, np.int64),
                correct=bool(columns["correct"][i]),
                z=int(columns["z"][i]),
                recovery=bool(columns["recovery"][i]),
                prefix_hash=span("prefix_hash", i, np.int64) if has_prefix_hash else None,
            )
        )
    return out


def split_questions(
    questions: Sequence[Question],
    n_val_questions: int,
    n_questions: int,
    seed: int,
) -> tuple[list[Question], list[Question]]:
    """Question-level split (§2 #2, §8.2). Validation is held out BEFORE training questions
    are sampled, and both come from the trainable pool so every val question can supply
    both a goal and an error index for tau calibration (§9.2).

    Returns (train_questions, val_questions). No question is in both, by construction.
    """
    import numpy as np

    trainable = sorted((q for q in questions if q.trainable), key=lambda q: q.qid)
    if len(trainable) < n_val_questions + 1:
        raise ValueError(
            f"only {len(trainable)} trainable questions, cannot hold out {n_val_questions}"
        )
    order = np.random.default_rng(seed).permutation(len(trainable))
    val = [trainable[i] for i in order[:n_val_questions]]
    rest = [trainable[i] for i in order[n_val_questions:]]
    train = rest[:n_questions]
    return train, val
