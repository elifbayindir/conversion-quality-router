#!/usr/bin/env python3
"""Final, one-time evaluation on the frozen test split.

This is the ONLY script that reads data/processed/split_indices.json's
test_index. It must run after scripts/finalize_mlp.py (threshold, uncertainty
band, and calibration are already frozen before this point) and must never
be used to re-select a model, threshold, or uncertainty band -- it only
measures and reports what the already-frozen pipeline does on held-out data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from conversion_router.data.load import load_raw_dataset
from conversion_router.data.preprocess import load_preprocessor
from conversion_router.data.split import load_split_indices
from conversion_router.data.validate import validate_raw_schema
from conversion_router.modeling.baseline import load_baseline_model, predict_positive_proba
from conversion_router.modeling.calibrate import apply_calibration, is_uncertain, load_calibrator
from conversion_router.modeling.network import predict_proba as mlp_predict_proba
from conversion_router.modeling.train import load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_METADATA_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "baseline_metadata.json"
MLP_METADATA_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "mlp_metadata.json"
TEST_PRED_PATH = PROJECT_ROOT / "data" / "processed" / "test_predictions.csv"
EVAL_JSON_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "test_evaluation.json"
REPORT_PATH = PROJECT_ROOT / "docs" / "test-evaluation-report.md"

TARGET_COLUMN = "Revenue"
SEGMENT_COLUMNS = ("VisitorType", "Weekend", "Month", "TrafficType")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


EXAMPLE_FEATURE_COLUMNS = (
    "VisitorType",
    "Month",
    "Weekend",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = y_proba >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": threshold,
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def curve_points(y_true: np.ndarray, y_proba: np.ndarray, max_points: int = 60) -> dict:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    def _subsample(arr: np.ndarray, n: int) -> list[float]:
        if len(arr) <= n:
            return [round(float(v), 4) for v in arr]
        idx = np.linspace(0, len(arr) - 1, n).astype(int)
        return [round(float(v), 4) for v in arr[idx]]

    return {
        "precision_recall": {
            "precision": _subsample(precision, max_points),
            "recall": _subsample(recall, max_points),
        },
        "roc": {
            "fpr": _subsample(fpr, max_points),
            "tpr": _subsample(tpr, max_points),
        },
    }


def calibration_curve_bins(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> list[dict]:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(proba, bin_edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lower": round(float(bin_edges[b]), 2),
                "bin_upper": round(float(bin_edges[b + 1]), 2),
                "n": int(mask.sum()),
                "mean_predicted": round(float(proba[mask].mean()), 4),
                "empirical_rate": round(float(np.asarray(y_true)[mask].mean()), 4),
            }
        )
    return rows


def segment_table(test_df: pd.DataFrame, column: str) -> list[dict]:
    rows = []
    for value, group in test_df.groupby(column, observed=True):
        rows.append(
            {
                "segment": str(value),
                "n_sessions": int(len(group)),
                "actual_positive_rate": round(float(group[TARGET_COLUMN].mean()), 4),
                "mean_mlp_calibrated_proba": round(float(group["mlp_calibrated_proba"].mean()), 4),
                "mlp_predicted_positive_rate": round(float(group["mlp_pred"].mean()), 4),
                "n_mlp_uncertain": int(group["mlp_uncertain"].sum()),
            }
        )
    return sorted(rows, key=lambda r: -r["n_sessions"])


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "_(no rows)_"
    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = [
        "| " + " | ".join(str(row[c]) for c in columns) + " |" for row in rows
    ]
    return "\n".join([header, sep] + body)


def main() -> None:
    df = load_raw_dataset()
    validate_raw_schema(df)
    split = load_split_indices()
    test_df = df.loc[split.test_index].copy()  # test data touched here, once
    y_true = test_df[TARGET_COLUMN].to_numpy().astype(int)

    preprocessor = load_preprocessor()
    X_test = preprocessor.transform(test_df)

    baseline_model = load_baseline_model()
    baseline_metadata = json.loads(BASELINE_METADATA_PATH.read_text(encoding="utf-8"))
    baseline_threshold = baseline_metadata["threshold_selection"]["threshold"]
    baseline_proba = predict_positive_proba(baseline_model, X_test)

    mlp_model = load_checkpoint()
    calibrator = load_calibrator()
    mlp_metadata = json.loads(MLP_METADATA_PATH.read_text(encoding="utf-8"))
    mlp_threshold = mlp_metadata["threshold_selection"]["threshold"]
    mlp_raw_proba = mlp_predict_proba(mlp_model, X_test)
    mlp_calibrated_proba = apply_calibration(calibrator, mlp_raw_proba)
    mlp_uncertain = is_uncertain(mlp_calibrated_proba, mlp_threshold)

    test_df["baseline_proba"] = baseline_proba
    test_df["baseline_pred"] = baseline_proba >= baseline_threshold
    test_df["mlp_raw_proba"] = mlp_raw_proba
    test_df["mlp_calibrated_proba"] = mlp_calibrated_proba
    test_df["mlp_pred"] = mlp_calibrated_proba >= mlp_threshold
    test_df["mlp_uncertain"] = mlp_uncertain

    baseline_metrics = compute_metrics(y_true, baseline_proba, baseline_threshold)
    mlp_metrics = compute_metrics(y_true, mlp_calibrated_proba, mlp_threshold)
    mlp_brier = round(float(brier_score_loss(y_true, mlp_calibrated_proba)), 4)
    mlp_raw_brier = round(float(brier_score_loss(y_true, mlp_raw_proba)), 4)

    baseline_curves = curve_points(y_true, baseline_proba)
    mlp_curves = curve_points(y_true, mlp_calibrated_proba)
    mlp_calibration_bins = calibration_curve_bins(y_true, mlp_calibrated_proba)

    segments = {col: segment_table(test_df, col) for col in SEGMENT_COLUMNS}

    error_cols = [
        "row_index",
        "y_true",
        "mlp_calibrated_proba",
        "mlp_uncertain",
        *EXAMPLE_FEATURE_COLUMNS,
    ]
    test_df["row_index"] = test_df.index.astype(int)
    test_df["y_true"] = y_true.astype(int)
    test_df["mlp_calibrated_proba"] = test_df["mlp_calibrated_proba"].round(4).astype(float)
    test_df["mlp_uncertain"] = test_df["mlp_uncertain"].astype(bool)
    fn_mask = (test_df["y_true"] == 1) & (~test_df["mlp_pred"])
    fp_mask = (test_df["y_true"] == 0) & (test_df["mlp_pred"])
    fn_examples = (
        test_df[fn_mask].sort_values("mlp_calibrated_proba", ascending=False).head(3)[error_cols]
    )
    fp_examples = (
        test_df[fp_mask].sort_values("mlp_calibrated_proba", ascending=False).head(3)[error_cols]
    )

    predictions_out = test_df[
        [
            "row_index",
            "y_true",
            "baseline_proba",
            "baseline_pred",
            "mlp_raw_proba",
            "mlp_calibrated_proba",
            "mlp_pred",
            "mlp_uncertain",
        ]
    ]
    TEST_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions_out.to_csv(TEST_PRED_PATH, index=False)

    evaluation = {
        "n_test_rows": len(test_df),
        "test_positive_rate": round(float(y_true.mean()), 4),
        "baseline": {"metrics": baseline_metrics, "curves": baseline_curves},
        "mlp": {
            "metrics": mlp_metrics,
            "curves": mlp_curves,
            "brier_score_calibrated": mlp_brier,
            "brier_score_raw": mlp_raw_brier,
            "calibration_bins": mlp_calibration_bins,
            "n_uncertain": int(mlp_uncertain.sum()),
        },
        "comparison": {
            "pr_auc_delta_mlp_minus_baseline": round(
                mlp_metrics["pr_auc"] - baseline_metrics["pr_auc"], 4
            ),
            "roc_auc_delta_mlp_minus_baseline": round(
                mlp_metrics["roc_auc"] - baseline_metrics["roc_auc"], 4
            ),
        },
        "segment_analysis": segments,
        "error_examples": {
            "false_negatives": fn_examples.to_dict("records"),
            "false_positives": fp_examples.to_dict("records"),
        },
        "evaluated_on": "test_only_single_pass",
    }
    EVAL_JSON_PATH.write_text(
        json.dumps(evaluation, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )

    lines = []
    lines.append("# Final Test Evaluation: Baseline vs. MLP")
    lines.append("")
    lines.append(
        f"Generated by `scripts/evaluate_test.py`. The test set ({len(test_df)} rows) is "
        "touched exactly once here; the model, threshold, uncertainty band, and calibration "
        "used were all frozen beforehand (T203/T204 for the baseline, T303/T304 for the MLP)."
    )
    lines.append("")
    lines.append("## Metric comparison (test set)")
    lines.append("")
    lines.append("| Metric | Baseline (LogReg) | MLP (calibrated) |")
    lines.append("|---|---|---|")
    lines.append(f"| Threshold | {baseline_metrics['threshold']} | {mlp_metrics['threshold']} |")
    lines.append(f"| **PR-AUC** | **{baseline_metrics['pr_auc']}** | **{mlp_metrics['pr_auc']}** |")
    lines.append(f"| ROC-AUC | {baseline_metrics['roc_auc']} | {mlp_metrics['roc_auc']} |")
    lines.append(f"| Precision | {baseline_metrics['precision']} | {mlp_metrics['precision']} |")
    lines.append(f"| Recall | {baseline_metrics['recall']} | {mlp_metrics['recall']} |")
    lines.append(f"| F1 | {baseline_metrics['f1']} | {mlp_metrics['f1']} |")
    lines.append(
        f"| Brier score | n/a (uncalibrated baseline) | {mlp_brier} (raw: {mlp_raw_brier}) |"
    )
    lines.append("")
    lines.append("## Confusion matrices (test set, at each model's frozen threshold)")
    lines.append("")
    for name, m in (("Baseline", baseline_metrics), ("MLP", mlp_metrics)):
        cm = m["confusion_matrix"]
        lines.append(f"**{name}**")
        lines.append("")
        lines.append("| | Predicted negative | Predicted positive |")
        lines.append("|---|---|---|")
        lines.append(
            f"| Actual negative | {cm['true_negative']} (TN) | {cm['false_positive']} (FP) |"
        )
        lines.append(
            f"| Actual positive | {cm['false_negative']} (FN) | {cm['true_positive']} (TP) |"
        )
        lines.append("")
    lines.append(
        f"MLP flagged {int(mlp_uncertain.sum())} of {len(test_df)} test sessions as uncertain."
    )
    lines.append("")
    lines.append("## Representative MLP false negatives (missed conversions)")
    lines.append("")
    lines.append(markdown_table(fn_examples.to_dict("records")))
    lines.append("")
    lines.append("## Representative MLP false positives (unnecessary reviews)")
    lines.append("")
    lines.append(markdown_table(fp_examples.to_dict("records")))
    lines.append("")
    lines.append("## Segment analysis (MLP, test set)")
    lines.append("")
    for col in SEGMENT_COLUMNS:
        lines.append(f"### {col}")
        lines.append("")
        lines.append(markdown_table(segments[col]))
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Baseline: PR-AUC={baseline_metrics['pr_auc']} ROC-AUC={baseline_metrics['roc_auc']}")
    print(f"MLP:      PR-AUC={mlp_metrics['pr_auc']} ROC-AUC={mlp_metrics['roc_auc']}")
    print(f"Wrote {TEST_PRED_PATH}")
    print(f"Wrote {EVAL_JSON_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
