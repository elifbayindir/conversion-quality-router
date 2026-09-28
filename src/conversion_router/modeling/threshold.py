"""Business-cost decision threshold selection.

Selected on validation data only, never the test set (PROJECT_BLUEPRINT.md
Section 5.4). The cost rule: a false negative (missing a session that would
have converted) is assumed FALSE_NEGATIVE_COST times more expensive than a
false positive (an unnecessary operational review) — a lost-revenue
opportunity is judged materially costlier than one extra manual review. This
ratio is a stated modeling assumption, not a measured value; it is recorded
alongside the selected threshold so it can be revisited.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FALSE_NEGATIVE_COST = 5.0
FALSE_POSITIVE_COST = 1.0
THRESHOLD_GRID = tuple(round(t, 2) for t in np.arange(0.01, 1.00, 0.01))


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    expected_cost: float
    false_negative_cost: float
    false_positive_cost: float
    n_false_negatives: int
    n_false_positives: int
    n_true_positives: int
    n_true_negatives: int


def select_threshold(
    y_true,
    y_proba,
    fn_cost: float = FALSE_NEGATIVE_COST,
    fp_cost: float = FALSE_POSITIVE_COST,
    grid: tuple[float, ...] = THRESHOLD_GRID,
) -> ThresholdSelection:
    """Pick the threshold in `grid` minimizing fn_cost*FN + fp_cost*FP.

    Ties are broken deterministically in favor of the lowest threshold
    encountered first while scanning the grid in ascending order.
    """
    y_true_bool = np.asarray(y_true).astype(bool)
    y_proba_arr = np.asarray(y_proba, dtype=float)

    best: ThresholdSelection | None = None
    for t in grid:
        y_pred = y_proba_arr >= t
        fn = int(np.sum(y_true_bool & ~y_pred))
        fp = int(np.sum(~y_true_bool & y_pred))
        tp = int(np.sum(y_true_bool & y_pred))
        tn = int(np.sum(~y_true_bool & ~y_pred))
        cost = fn_cost * fn + fp_cost * fp
        if best is None or cost < best.expected_cost:
            best = ThresholdSelection(
                threshold=float(t),
                expected_cost=float(cost),
                false_negative_cost=fn_cost,
                false_positive_cost=fp_cost,
                n_false_negatives=fn,
                n_false_positives=fp,
                n_true_positives=tp,
                n_true_negatives=tn,
            )
    assert best is not None
    return best
