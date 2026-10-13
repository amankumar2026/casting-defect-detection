"""Scores and the cost-based threshold, checked on tiny hand-made examples."""

import numpy as np
import pytest

from inspector import config
from inspector.metrics import cost, pick_threshold, summarize, wilson_interval

LABELS = np.array([1, 1, 1, 0, 0, 0])
PROBS = np.array([0.9, 0.6, 0.2, 0.7, 0.3, 0.1])


def test_summarize_counts_each_kind_of_mistake():
    result = summarize(LABELS, PROBS, threshold=0.5)
    assert result["missed_defects"] == 1      # the defect scored 0.2
    assert result["false_rejects"] == 1       # the good part scored 0.7
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["precision"] == pytest.approx(2 / 3)


def test_a_score_equal_to_the_threshold_counts_as_defective():
    result = summarize(np.array([1, 0]), np.array([0.5, 0.5]), threshold=0.5)
    assert result["missed_defects"] == 0 and result["false_rejects"] == 1


def test_cost_weighs_a_missed_defect_more_than_a_false_reject():
    assert cost(missed_defects=1, false_rejects=0) == config.COST_MISSED_DEFECT
    assert cost(missed_defects=0, false_rejects=1) == config.COST_FALSE_REJECT
    assert config.COST_MISSED_DEFECT > config.COST_FALSE_REJECT


def test_threshold_drops_to_catch_a_defect_even_if_it_rejects_good_parts():
    # Catching the defect at 0.2 means also rejecting the good part at 0.3
    # and the one at 0.7. Two false rejects (cost 2) beat one miss (cost 10).
    threshold = pick_threshold(LABELS, PROBS)
    assert 0.1 < threshold <= 0.2
    assert summarize(LABELS, PROBS, threshold)["missed_defects"] == 0


def test_tied_thresholds_return_the_middle_of_the_range():
    # Every threshold from 0.21 to 0.80 separates these perfectly.
    labels = np.array([1, 1, 0, 0])
    probs = np.array([0.95, 0.8, 0.2, 0.05])
    assert pick_threshold(labels, probs) == pytest.approx(0.51, abs=0.011)


def test_wilson_interval_contains_the_observed_rate_and_stays_in_range():
    low, high = wilson_interval(258, 260)
    assert 0 <= low < 258 / 260 < high <= 1


def test_wilson_interval_is_not_a_single_point_for_a_perfect_score():
    low, high = wilson_interval(260, 260)
    assert high == pytest.approx(1.0) and 0.98 < low < 0.99


def test_wilson_interval_narrows_with_more_data():
    small = wilson_interval(9, 10)
    large = wilson_interval(900, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])
