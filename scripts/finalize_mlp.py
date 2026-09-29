#!/usr/bin/env python3
"""Select the MLP's decision threshold and freeze all model metadata.

Must run after scripts/train_mlp.py and scripts/calibrate_mlp.py, and must
run BEFORE scripts/evaluate_test.py touches the test set. Assembles a single
frozen artifacts/metadata/mlp_metadata.json: threshold, uncertainty band,
seed, feature contract version, package versions, artifact hashes, and
training configuration -- everything needed to reproduce or audit the model
without looking at test data.
"""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import sklearn
import torch

from conversion_router.data.load import load_raw_dataset
from conversion_router.data.preprocess import (
    DEFAULT_PREPROCESSOR_PATH,
    get_output_feature_names,
    load_preprocessor,
)
from conversion_router.data.split import load_split_indices
from conversion_router.data.validate import validate_raw_schema
from conversion_router.modeling.calibrate import (
    DEFAULT_CALIBRATOR_PATH,
    apply_calibration,
    load_calibrator,
)
from conversion_router.modeling.network import MLP_VERSION, predict_proba
from conversion_router.modeling.threshold import select_threshold
from conversion_router.modeling.train import DEFAULT_CHECKPOINT_PATH, load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_HISTORY_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "mlp_training_history.json"
CALIBRATION_METADATA_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "mlp_calibration.json"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "mlp_metadata.json"
FEATURE_CONTRACT_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "feature_contract.json"
VAL_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "mlp_validation_predictions.csv"
TARGET_COLUMN = "Revenue"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    df = load_raw_dataset()
    validate_raw_schema(df)
    split = load_split_indices()
    val_df = df.loc[split.validation_index]
    n_train_rows = len(split.train_index)

    preprocessor = load_preprocessor()
    model = load_checkpoint()
    calibrator = load_calibrator()

    X_val = preprocessor.transform(val_df)
    y_val = val_df[TARGET_COLUMN].to_numpy().astype(int)

    raw_proba = predict_proba(model, X_val)
    calibrated_proba = apply_calibration(calibrator, raw_proba)

    selection = select_threshold(y_val, calibrated_proba)

    VAL_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_index": val_df.index,
            "y_true": y_val,
            "raw_proba": raw_proba,
            "calibrated_proba": calibrated_proba,
        }
    ).to_csv(VAL_PREDICTIONS_PATH, index=False)

    training_history = json.loads(TRAINING_HISTORY_PATH.read_text(encoding="utf-8"))
    calibration_metadata = json.loads(CALIBRATION_METADATA_PATH.read_text(encoding="utf-8"))
    feature_contract = json.loads(FEATURE_CONTRACT_PATH.read_text(encoding="utf-8"))

    metadata = {
        "model_name": "conversion_mlp",
        "model_version": MLP_VERSION,
        "seed": training_history["seed"],
        "feature_contract_version": feature_contract["feature_contract_version"],
        "preprocessor_output_dim": len(get_output_feature_names(preprocessor)),
        "n_train_rows": n_train_rows,
        "n_validation_rows": len(val_df),
        "training_configuration": {
            "hidden_sizes": [128, 64, 32],
            "dropout": 0.3,
            "learning_rate": training_history["learning_rate"],
            "batch_size": training_history["batch_size"],
            "max_epochs": training_history["max_epochs"],
            "patience": training_history["patience"],
            "best_epoch": training_history["best_epoch"],
            "n_epochs_run": training_history["n_epochs_run"],
            "best_val_pr_auc_raw": training_history["best_val_pr_auc"],
        },
        "calibration": calibration_metadata,
        "threshold_selection": {
            "method": "business_cost_grid_search_on_validation_only",
            "threshold": selection.threshold,
            "false_negative_cost": selection.false_negative_cost,
            "false_positive_cost": selection.false_positive_cost,
            "cost_rationale": (
                "Same business-cost rationale as the baseline (T203): a missed "
                "converting session is assumed "
                f"{selection.false_negative_cost / selection.false_positive_cost:.0f}x "
                "costlier than one unnecessary manual review. Applied to calibrated "
                "MLP probabilities on validation data only."
            ),
            "expected_cost_at_threshold": selection.expected_cost,
            "n_false_negatives": selection.n_false_negatives,
            "n_false_positives": selection.n_false_positives,
            "n_true_positives": selection.n_true_positives,
            "n_true_negatives": selection.n_true_negatives,
            "n_validation_rows": len(val_df),
            "selected_on": "validation_only",
        },
        "package_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": version("pandas"),
            "numpy": version("numpy"),
        },
        "artifact_hashes": {
            "mlp_checkpoint_sha256": sha256_of(DEFAULT_CHECKPOINT_PATH),
            "preprocessor_sha256": sha256_of(DEFAULT_PREPROCESSOR_PATH),
            "calibrator_sha256": sha256_of(DEFAULT_CALIBRATOR_PATH),
        },
        "artifact_paths": {
            "model": "artifacts/models/mlp_checkpoint.pt",
            "preprocessor": "artifacts/preprocessors/preprocessor.joblib",
            "calibrator": "artifacts/preprocessors/calibrator.joblib",
        },
        "frozen_before_test_evaluation": True,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(
        f"Selected MLP threshold={selection.threshold} "
        f"(FN={selection.n_false_negatives}, FP={selection.n_false_positives})"
    )
    print(f"Wrote {METADATA_PATH}")


if __name__ == "__main__":
    main()
