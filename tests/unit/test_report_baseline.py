import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "baseline_metadata.json"
REPORT_PATH = PROJECT_ROOT / "docs" / "baseline-report.md"
VAL_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_validation_predictions.csv"


def _require_real_artifacts():
    if not VAL_PREDICTIONS_PATH.exists():
        import pytest

        pytest.skip(
            "data/processed/baseline_validation_predictions.csv not present; "
            "run scripts/train_baseline.py first (not required for the base test suite)."
        )


def test_report_baseline_script_runs_and_writes_expected_outputs():
    _require_real_artifacts()

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "report_baseline.py")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    assert REPORT_PATH.exists()
    report_text = REPORT_PATH.read_text(encoding="utf-8")
    for heading in (
        "# Baseline (Logistic Regression) Validation Report",
        "## Metrics",
        "## Confusion matrix",
        "## Representative false negatives",
        "## Representative false positives",
    ):
        assert heading in report_text

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metrics = metadata["validation_metrics"]
    for key in ("pr_auc", "roc_auc", "precision", "recall", "f1", "confusion_matrix"):
        assert key in metrics
    assert metrics["evaluated_on"] == "validation_only"
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0

    cm = metrics["confusion_matrix"]
    n_val = metadata["n_validation_rows"]
    assert (
        cm["true_negative"] + cm["false_positive"] + cm["false_negative"] + cm["true_positive"]
        == n_val
    )
