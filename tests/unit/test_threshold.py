import numpy as np
import pytest

from conversion_router.modeling.threshold import select_threshold


def test_perfectly_separable_scores_pick_a_separating_threshold():
    y_true = [0, 0, 0, 1, 1, 1]
    y_proba = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

    selection = select_threshold(y_true, y_proba)

    assert 0.3 < selection.threshold <= 0.7
    assert selection.n_false_negatives == 0
    assert selection.n_false_positives == 0
    assert selection.expected_cost == 0.0


def test_higher_false_negative_cost_lowers_the_threshold():
    rng = np.random.default_rng(0)
    y_true = (rng.random(500) < 0.2).astype(int)
    y_proba = np.clip(y_true * 0.5 + rng.normal(0, 0.25, size=500) + 0.3, 0, 1)

    low_fn_cost = select_threshold(y_true, y_proba, fn_cost=1.0, fp_cost=1.0)
    high_fn_cost = select_threshold(y_true, y_proba, fn_cost=10.0, fp_cost=1.0)

    assert high_fn_cost.threshold <= low_fn_cost.threshold


def test_selection_is_deterministic():
    rng = np.random.default_rng(1)
    y_true = (rng.random(300) < 0.15).astype(int)
    y_proba = rng.random(300)

    a = select_threshold(y_true, y_proba)
    b = select_threshold(y_true, y_proba)

    assert a == b


def test_threshold_is_within_grid_bounds():
    rng = np.random.default_rng(2)
    y_true = (rng.random(200) < 0.15).astype(int)
    y_proba = rng.random(200)

    selection = select_threshold(y_true, y_proba)

    assert 0.0 < selection.threshold < 1.0


def test_confusion_counts_are_consistent_with_labels():
    rng = np.random.default_rng(3)
    n = 200
    y_true = (rng.random(n) < 0.15).astype(int)
    y_proba = rng.random(n)

    selection = select_threshold(y_true, y_proba)

    n_positive = int(np.sum(y_true))
    n_negative = n - n_positive
    assert selection.n_true_positives + selection.n_false_negatives == n_positive
    assert selection.n_true_negatives + selection.n_false_positives == n_negative


def test_expected_cost_matches_cost_formula():
    y_true = [0, 0, 1, 1]
    y_proba = [0.9, 0.1, 0.9, 0.1]  # one FP-shaped case, one FN-shaped case at any threshold

    selection = select_threshold(y_true, y_proba, fn_cost=3.0, fp_cost=2.0)

    expected = (
        selection.false_negative_cost * selection.n_false_negatives
        + selection.false_positive_cost * selection.n_false_positives
    )
    assert selection.expected_cost == pytest.approx(expected)
