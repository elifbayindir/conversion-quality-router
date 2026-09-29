#!/usr/bin/env python3
"""Fit isotonic calibration for the MLP on validation data only.

Reads the trained MLP checkpoint and the shared preprocessor, computes raw
sigmoid probabilities on the validation split, fits calibration, and saves
the calibrator plus Brier-score evidence. The uncertainty rule itself
(decision_margin / is_uncertain) is deterministic code, versioned in
src/conversion_router/modeling/calibrate.py, not fit from data.
"""

from __future__ import annotations

import json
from pathlib import Path

from conversion_router.data.load import load_raw_dataset
from conversion_router.data.preprocess import load_preprocessor
from conversion_router.data.split import load_split_indices
from conversion_router.data.validate import validate_raw_schema
from conversion_router.modeling.calibrate import (
    CALIBRATION_VERSION,
    DECISION_POLICY_VERSION,
    UNCERTAINTY_BAND,
    apply_calibration,
    calibration_evidence,
    fit_calibrator,
    save_calibrator,
)
from conversion_router.modeling.network import predict_proba
from conversion_router.modeling.train import load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_METADATA_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "mlp_calibration.json"
TARGET_COLUMN = "Revenue"


def main() -> None:
    df = load_raw_dataset()
    validate_raw_schema(df)
    split = load_split_indices()
    val_df = df.loc[split.validation_index]

    preprocessor = load_preprocessor()
    model = load_checkpoint()

    X_val = preprocessor.transform(val_df)
    y_val = val_df[TARGET_COLUMN].to_numpy().astype(int)

    raw_proba = predict_proba(model, X_val)
    calibrator = fit_calibrator(y_val, raw_proba)
    save_calibrator(calibrator)

    calibrated_proba = apply_calibration(calibrator, raw_proba)
    evidence = calibration_evidence(y_val, raw_proba, calibrated_proba)

    metadata = {
        "calibration_version": CALIBRATION_VERSION,
        "decision_policy_version": DECISION_POLICY_VERSION,
        "method": "isotonic_regression",
        "fit_on": "validation_only",
        "uncertainty_band": UNCERTAINTY_BAND,
        "uncertainty_rule": "uncertain = |calibrated_probability - threshold| < uncertainty_band",
        "brier_score_raw": round(evidence.brier_score_raw, 4),
        "brier_score_calibrated": round(evidence.brier_score_calibrated, 4),
        "n_validation_rows": len(val_df),
    }
    CALIBRATION_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(
        f"Brier (raw)={evidence.brier_score_raw:.4f} "
        f"Brier (calibrated)={evidence.brier_score_calibrated:.4f}"
    )
    print(f"Wrote {CALIBRATION_METADATA_PATH}")


if __name__ == "__main__":
    main()
