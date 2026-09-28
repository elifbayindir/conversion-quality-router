"""Class-weighted logistic regression baseline.

Purpose: give the PyTorch MLP (P3) a real comparison point, not just a
number. Trained on the same frozen, group-aware split and the same
deployment-safe feature contract as every other model in this project.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

SEED = 42
BASELINE_VERSION = "1.0.0"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "baseline_logreg.joblib"


def build_baseline_model(seed: int = SEED) -> LogisticRegression:
    return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)


def train_baseline(
    X_train: np.ndarray, y_train: np.ndarray, seed: int = SEED
) -> LogisticRegression:
    model = build_baseline_model(seed)
    model.fit(X_train, y_train)
    return model


def predict_positive_proba(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def save_baseline_model(model: LogisticRegression, path: Path = DEFAULT_MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_baseline_model(path: Path = DEFAULT_MODEL_PATH) -> LogisticRegression:
    if not path.exists():
        raise FileNotFoundError(f"Baseline model artifact not found at {path}.")
    return joblib.load(path)
