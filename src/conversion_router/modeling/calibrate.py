"""Probability calibration and the deterministic uncertainty rule.

Raw MLP sigmoid output is not treated as "confidence" directly
(PROJECT_BLUEPRINT.md Section 5.5). It is calibrated on validation data only
(isotonic regression, monotonic and distribution-free) before being used for
any decision. `uncertain` is a deterministic function of the calibrated
probability's distance from the decision threshold -- computed in code here,
never decided by the LLM agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

CALIBRATION_VERSION = "1.0.0"
DECISION_POLICY_VERSION = "1.0.0"
UNCERTAINTY_BAND = 0.05  # |calibrated_proba - threshold| < this => uncertain

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CALIBRATOR_PATH = PROJECT_ROOT / "artifacts" / "preprocessors" / "calibrator.joblib"


def fit_calibrator(y_true, raw_proba) -> IsotonicRegression:
    """Fit isotonic calibration. Callers must pass validation data only."""
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(np.asarray(raw_proba), np.asarray(y_true))
    return calibrator


def apply_calibration(calibrator: IsotonicRegression, raw_proba) -> np.ndarray:
    return calibrator.predict(np.asarray(raw_proba))


def save_calibrator(calibrator: IsotonicRegression, path: Path = DEFAULT_CALIBRATOR_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, path)


def load_calibrator(path: Path = DEFAULT_CALIBRATOR_PATH) -> IsotonicRegression:
    if not path.exists():
        raise FileNotFoundError(f"Calibrator artifact not found at {path}.")
    return joblib.load(path)


@dataclass(frozen=True)
class CalibrationEvidence:
    brier_score_raw: float
    brier_score_calibrated: float


def calibration_evidence(y_true, raw_proba, calibrated_proba) -> CalibrationEvidence:
    return CalibrationEvidence(
        brier_score_raw=float(brier_score_loss(y_true, raw_proba)),
        brier_score_calibrated=float(brier_score_loss(y_true, calibrated_proba)),
    )


def decision_margin(calibrated_proba, threshold: float) -> np.ndarray:
    """Deterministic distance from the decision threshold. Used by the agent's input."""
    return np.abs(np.asarray(calibrated_proba, dtype=float) - threshold)


def is_uncertain(calibrated_proba, threshold: float, band: float = UNCERTAINTY_BAND) -> np.ndarray:
    """Deterministic uncertainty flag: within `band` of the threshold => True."""
    return decision_margin(calibrated_proba, threshold) < band
