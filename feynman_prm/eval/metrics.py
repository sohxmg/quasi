"""ProcessBench's metric, reproduced exactly (§5).

Per subset:
    acc_error   = fraction of `label != -1` samples whose predicted index EQUALS the gold
    acc_correct = fraction of `label == -1` samples predicted -1
    F1          = harmonic mean of the two

The harmonic mean is why an off-by-one is fatal: a model trained on Delta_z instead of
Delta_{z+1} scores acc_error = 0, which drags F1 to 0 no matter how good acc_correct is.
"""

from __future__ import annotations

from typing import Sequence


def harmonic_mean(a: float, b: float) -> float:
    return 0.0 if (a + b) == 0 else 2 * a * b / (a + b)


def processbench_metrics(predictions: Sequence[int], labels: Sequence[int]) -> dict[str, float]:
    if len(predictions) != len(labels):
        raise ValueError(f"{len(predictions)} predictions vs {len(labels)} labels")
    errored = [(p, l) for p, l in zip(predictions, labels) if l != -1]
    clean = [(p, l) for p, l in zip(predictions, labels) if l == -1]
    acc_error = sum(p == l for p, l in errored) / len(errored) if errored else 0.0
    acc_correct = sum(p == -1 for p, _ in clean) / len(clean) if clean else 0.0
    return {
        "acc_error": acc_error,
        "acc_correct": acc_correct,
        "f1": harmonic_mean(acc_error, acc_correct),
        "n_error": float(len(errored)),
        "n_correct": float(len(clean)),
    }


def split_metrics(
    predictions: Sequence[int], labels: Sequence[int], group: Sequence[bool]
) -> dict[str, dict[str, float]]:
    """Locked #5: report math-subset F1 SPLIT -- 587 leaked questions vs 413 clean ones.

    The gap between the two is the measurement of what the contamination is worth. Any PRM
    trained on Math-Shepherd has this same leak, CRM included (§4.3b).
    """
    def take(flag: bool) -> dict[str, float]:
        idx = [i for i, g in enumerate(group) if bool(g) == flag]
        return processbench_metrics([predictions[i] for i in idx], [labels[i] for i in idx])

    return {"leaked": take(True), "clean": take(False)}


def gold_answer_upper_bound(final_answer_correct: Sequence[bool], labels: Sequence[int]) -> dict:
    """§5.1: knowing only `final_answer_correct` solves the "no error" half exactly (100.0%
    on all four subsets) and bounds the other half at 65.9-96.6%.

    Published PRMs score in the 60s-70s. **Therefore any scoring path that consumes a
    reference solution or gold answer is a SKYLINE, not a result.** This function exists so
    that fact stays visible in the eval output, not to be reported as a number of ours.
    """
    clean = [c for c, l in zip(final_answer_correct, labels) if l == -1]
    errored = [c for c, l in zip(final_answer_correct, labels) if l != -1]
    return {
        "acc_correct_from_gold_answer": sum(bool(c) for c in clean) / len(clean) if clean else 0.0,
        "acc_error_upper_bound": sum(not bool(c) for c in errored) / len(errored)
        if errored
        else 0.0,
    }
