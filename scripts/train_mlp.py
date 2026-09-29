#!/usr/bin/env python3
"""Train the PyTorch MLP on the frozen train split, monitored on validation.

Reuses the same fitted preprocessor as the baseline (artifacts/preprocessors/
preprocessor.joblib) so both models are compared on identical features. Run
scripts/train_baseline.py first if that artifact does not exist yet.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from conversion_router.data.load import load_raw_dataset
from conversion_router.data.preprocess import load_preprocessor
from conversion_router.data.split import load_split_indices
from conversion_router.data.validate import validate_raw_schema
from conversion_router.modeling.network import MLP_VERSION
from conversion_router.modeling.train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    SEED,
    save_checkpoint,
    train_mlp,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "models" / "mlp_checkpoint.pt"
HISTORY_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "mlp_training_history.json"
TARGET_COLUMN = "Revenue"


def main() -> None:
    df = load_raw_dataset()
    validate_raw_schema(df)
    split = load_split_indices()

    train_df = df.loc[split.train_index]
    val_df = df.loc[split.validation_index]

    preprocessor = load_preprocessor()
    X_train = preprocessor.transform(train_df)
    X_val = preprocessor.transform(val_df)
    y_train = train_df[TARGET_COLUMN].to_numpy().astype(int)
    y_val = val_df[TARGET_COLUMN].to_numpy().astype(int)

    result = train_mlp(X_train, y_train, X_val, y_val, seed=SEED)
    save_checkpoint(result.model, CHECKPOINT_PATH)

    history_payload = {
        "mlp_version": MLP_VERSION,
        "train_version": "1.0.0",
        "seed": SEED,
        "max_epochs": DEFAULT_MAX_EPOCHS,
        "patience": DEFAULT_PATIENCE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "batch_size": DEFAULT_BATCH_SIZE,
        "best_epoch": result.best_epoch,
        "best_val_pr_auc": round(result.best_val_pr_auc, 4),
        "n_epochs_run": len(result.history),
        "epochs": [asdict(record) for record in result.history],
        "package_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history_payload, indent=2) + "\n", encoding="utf-8")

    print(
        f"Best epoch {result.best_epoch}/{len(result.history)}: "
        f"val_pr_auc={result.best_val_pr_auc:.4f}"
    )
    print(f"Wrote {CHECKPOINT_PATH}")
    print(f"Wrote {HISTORY_PATH}")


if __name__ == "__main__":
    main()
