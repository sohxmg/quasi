"""§15's eval tests: ProcessBench's metric reproduced exactly, and tau calibration (§5, §9)."""

from __future__ import annotations

import math

from feynman_prm.eval.calibrate import calibrate_tau, natural_tau
from feynman_prm.eval.metrics import (
    gold_answer_upper_bound,
    harmonic_mean,
    processbench_metrics,
    split_metrics,
)
from feynman_prm.eval.processbench import predict


def test_f1_is_the_harmonic_mean_of_the_two_halves():
    predictions = [2, -1, 0, -1]
    labels = [2, -1, 1, 3]
    m = processbench_metrics(predictions, labels)
    assert m["acc_error"] == 1 / 3            # samples 0, 2, 3 have errors; only 0 is right
    assert m["acc_correct"] == 1.0            # the single label == -1 sample is predicted -1
    assert math.isclose(m["f1"], harmonic_mean(1 / 3, 1.0))


def test_an_off_by_one_zeroes_acc_error_and_collapses_f1():
    """This is why the z indexing is the highest-value test in the repo: predicting z-1 on
    every errored sample still gives a perfect acc_correct, and F1 still goes to 0."""
    labels = [2, 3, 4, -1, -1]
    off_by_one = [1, 2, 3, -1, -1]
    m = processbench_metrics(off_by_one, labels)
    assert m["acc_error"] == 0.0
    assert m["acc_correct"] == 1.0
    assert m["f1"] == 0.0


def test_math_leak_split_is_reported_separately():
    """Locked #5: keep the 587 leaked questions in training, report math F1 SPLIT. The gap is
    the measurement of what the contamination is worth."""
    predictions = [1, -1, 1, -1]
    labels = [1, -1, 2, -1]
    leaked = [True, True, False, False]
    split = split_metrics(predictions, labels, leaked)
    assert split["leaked"]["f1"] == 1.0
    assert split["clean"]["acc_error"] == 0.0


def test_gold_answer_upper_bound_stays_visible():
    """§5.1: `final_answer_correct` alone solves the "no error" half exactly (100.0% on all
    four subsets) while published PRMs score in the 60s-70s. Any scoring path that consumes a
    reference solution or gold answer is a SKYLINE, not a result."""
    labels = [-1, -1, 2, 3]
    final_answer_correct = [True, True, False, False]
    bound = gold_answer_upper_bound(final_answer_correct, labels)
    assert bound["acc_correct_from_gold_answer"] == 1.0
    assert bound["acc_error_upper_bound"] == 1.0


def test_predict_applies_the_delta_rule_per_sample():
    deltas = [[-0.7, 2.0], [-0.7, -0.7], []]
    assert predict(deltas, 0.347) == [1, -1, -1]


def test_natural_tau_is_the_midpoint_implied_by_the_ruler(cfg):
    """Training puts good steps at Delta ~ -0.693 and pushes Delta_{z+1} >= m = 1.386, so the
    fitted tau should land near 0.347. tau ~ 0 means the second step of margin never landed
    -- drop margin_steps to 1.0 rather than fighting it (§9.2, §16.19)."""
    assert math.isclose(natural_tau(cfg), 0.34657, rel_tol=1e-4)


def test_calibration_finds_the_separating_threshold(cfg):
    """tau is fit on held-out Math-Shepherd VALIDATION questions, never on ProcessBench."""
    deltas = [[-0.7, -0.7, 2.0], [-0.7, -0.7, -0.7], [1.5, -0.7], [-0.7, -0.7]]
    labels = [2, -1, 0, -1]
    calibration = calibrate_tau(deltas, labels, cfg)
    assert calibration.f1 == 1.0
    # any tau in [-0.7, 1.5) separates these deltas: the rule is `> tau`, strictly
    assert -0.7 <= calibration.tau < 1.5
    assert calibration.expected_tau == natural_tau(cfg)
    assert calibration.sensitivity >= 0.0
