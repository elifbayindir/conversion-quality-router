#!/usr/bin/env python3
"""Train the logistic regression baseline on the frozen train split.

Fits the preprocessing pipeline on train only, trains a class-weighted
logistic regression, and persists the model, the preprocessor, validation
predictions, and reproducibility metadata.
"""

from __future__ import annotations

import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import sklearn

from conversion_router.data.load import load_raw_dataset
from conversion_router.data.preprocess import (
    fit_preprocessor,
    get_output_feature_names,
    save_preprocessor,
)
from conversion_router.data.split import SEED, load_split_indices, stratified_split
from conversion_router.data.validate import validate_raw_schema
from conversion_router.modeling.baseline import (
    BASELINE_VERSION,
    predict_positive_proba,
    save_baseline_model,
    train_baseline,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_INDICES_PATH = PROJECT_ROOT / "data" / "processed" / "split_indices.json"
VAL_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_validation_predictions.csv"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "baseline_metadata.json"
TARGET_COLUMN = "Revenue"


def main() -> None:
    df = load_raw_dataset()
    validate_raw_schema(df)

    if SPLIT_INDICES_PATH.exists():
        split = load_split_indices()
    else:
        split, _ = stratified_split(df)

    train_df = df.loc[split.train_index]
    val_df = df.loc[split.validation_index]

    preprocessor = fit_preprocessor(train_df)
    save_preprocessor(preprocessor)

    X_train = preprocessor.transform(train_df)
    X_val = preprocessor.transform(val_df)
    y_train = train_df[TARGET_COLUMN].to_numpy()
    y_val = val_df[TARGET_COLUMN].to_numpy()

    model = train_baseline(X_train, y_train, seed=SEED)
    save_baseline_model(model)

    val_proba = predict_positive_proba(model, X_val)
    predictions_df = pd.DataFrame(
        {
            "row_index": val_df.index,
            "y_true": y_val,
            "y_pred_proba": val_proba,
        }
    )
    VAL_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(VAL_PREDICTIONS_PATH, index=False)

    metadata = {
        "model_name": "baseline_logistic_regression",
        "model_version": BASELINE_VERSION,
        "seed": SEED,
        "sklearn_params": {
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": SEED,
        },
        "feature_contract_version": "1.0.0",
        "preprocessor_output_dim": len(get_output_feature_names(preprocessor)),
        "n_train_rows": len(train_df),
        "n_validation_rows": len(val_df),
        "train_positive_rate": round(float(y_train.mean()), 4),
        "validation_positive_rate": round(float(y_val.mean()), 4),
        "package_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": version("pandas"),
            "numpy": version("numpy"),
        },
        "artifact_paths": {
            "model": "artifacts/models/baseline_logreg.joblib",
            "preprocessor": "artifacts/preprocessors/preprocessor.joblib",
            "validation_predictions": "data/processed/baseline_validation_predictions.csv",
        },
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Trained on {len(train_df)} rows, validated on {len(val_df)} rows.")
    print(f"Wrote {VAL_PREDICTIONS_PATH}")
    print(f"Wrote {METADATA_PATH}")


if __name__ == "__main__":
    main()
