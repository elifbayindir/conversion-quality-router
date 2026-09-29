import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from evaluate_test import (  # noqa: E402
    calibration_curve_bins,
    compute_metrics,
    curve_points,
    segment_table,
)

TEST_EVAL_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "test_evaluation.json"
REPORT_PATH = PROJECT_ROOT / "docs" / "test-evaluation-report.md"
MLP_METADATA_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "mlp_metadata.json"


def test_compute_metrics_matches_manual_confusion_counts():
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.2, 0.7, 0.9])
    threshold = 0.5
    # predictions: [0, 1, 0, 1, 1] -> TN=1(idx0), FP=1(idx1), FN=1(idx2), TP=2(idx3,4)

    metrics = compute_metrics(y_true, y_proba, threshold)

    cm = metrics["confusion_matrix"]
    assert cm == {"true_negative": 1, "false_positive": 1, "false_negative": 1, "true_positive": 2}
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_curve_points_are_bounded_and_subsampled():
    rng = np.random.default_rng(0)
    y_true = (rng.random(500) < 0.2).astype(int)
    y_proba = rng.random(500)

    curves = curve_points(y_true, y_proba, max_points=20)

    assert len(curves["precision_recall"]["precision"]) <= 20
    assert len(curves["roc"]["fpr"]) <= 20
    assert all(0.0 <= v <= 1.0 for v in curves["roc"]["fpr"])
    assert all(0.0 <= v <= 1.0 for v in curves["roc"]["tpr"])


def test_calibration_curve_bins_cover_all_rows():
    rng = np.random.default_rng(1)
    y_true = (rng.random(300) < 0.3).astype(int)
    proba = rng.random(300)

    bins = calibration_curve_bins(y_true, proba, n_bins=5)

    assert sum(b["n"] for b in bins) == 300
    for b in bins:
        assert 0.0 <= b["mean_predicted"] <= 1.0
        assert 0.0 <= b["empirical_rate"] <= 1.0


def test_segment_table_groups_and_sorts_by_size():
    df = pd.DataFrame(
        {
            "Revenue": [True, False, False, True, False],
            "mlp_calibrated_proba": [0.6, 0.2, 0.3, 0.7, 0.1],
            "mlp_pred": [True, False, False, True, False],
            "mlp_uncertain": [False, True, False, False, True],
            "VisitorType": ["A", "A", "A", "B", "B"],
        }
    )

    rows = segment_table(df, "VisitorType")

    assert rows[0]["segment"] == "A"
    assert rows[0]["n_sessions"] == 3
    assert rows[1]["segment"] == "B"
    assert rows[1]["n_sessions"] == 2
    assert rows[0]["actual_positive_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_evaluate_test_script_runs_and_freezes_expected_outputs():
    if not MLP_METADATA_PATH.exists():
        pytest.skip("MLP metadata not present; run the full P3 pipeline first.")

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_test.py")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    evaluation = json.loads(TEST_EVAL_PATH.read_text(encoding="utf-8"))
    assert evaluation["evaluated_on"] == "test_only_single_pass"
    for key in ("baseline", "mlp", "comparison", "segment_analysis"):
        assert key in evaluation

    for segment_col in ("VisitorType", "Weekend", "Month", "TrafficType"):
        assert segment_col in evaluation["segment_analysis"]

    cm = evaluation["mlp"]["metrics"]["confusion_matrix"]
    n_test = evaluation["n_test_rows"]
    assert (
        cm["true_negative"] + cm["false_positive"] + cm["false_negative"] + cm["true_positive"]
        == n_test
    )

    report_text = REPORT_PATH.read_text(encoding="utf-8")
    for heading in (
        "# Final Test Evaluation",
        "## Metric comparison",
        "## Confusion matrices",
        "## Representative MLP false negatives",
        "## Representative MLP false positives",
        "## Segment analysis",
    ):
        assert heading in report_text
