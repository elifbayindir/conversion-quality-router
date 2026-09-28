#!/usr/bin/env python3
"""Select the baseline's decision threshold from validation predictions only.

Reads data/processed/baseline_validation_predictions.csv (produced by
scripts/train_baseline.py), applies the explicit business-cost rule in
src/conversion_router/modeling/threshold.py, and records the result in
artifacts/metadata/baseline_metadata.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from conversion_router.modeling.threshold import select_threshold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAL_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_validation_predictions.csv"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "baseline_metadata.json"


def main() -> None:
    if not VAL_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{VAL_PREDICTIONS_PATH} not found. Run scripts/train_baseline.py first."
        )
    predictions = pd.read_csv(VAL_PREDICTIONS_PATH)

    selection = select_threshold(predictions["y_true"], predictions["y_pred_proba"])

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metadata["threshold_selection"] = {
        "method": "business_cost_grid_search_on_validation_only",
        "threshold": selection.threshold,
        "false_negative_cost": selection.false_negative_cost,
        "false_positive_cost": selection.false_positive_cost,
        "cost_rationale": (
            "A missed converting session (false negative) is assumed "
            f"{selection.false_negative_cost / selection.false_positive_cost:.0f}x costlier "
            "than an unnecessary manual review (false positive): a lost potential sale "
            "outweighs one extra review. This is a stated modeling assumption, not a "
            "measured business figure."
        ),
        "expected_cost_at_threshold": selection.expected_cost,
        "n_false_negatives": selection.n_false_negatives,
        "n_false_positives": selection.n_false_positives,
        "n_true_positives": selection.n_true_positives,
        "n_true_negatives": selection.n_true_negatives,
        "n_validation_rows": len(predictions),
        "selected_on": "validation_only",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(
        f"Selected threshold={selection.threshold} "
        f"(FN={selection.n_false_negatives}, FP={selection.n_false_positives}, "
        f"expected_cost={selection.expected_cost})"
    )
    print(f"Updated {METADATA_PATH}")


if __name__ == "__main__":
    main()
