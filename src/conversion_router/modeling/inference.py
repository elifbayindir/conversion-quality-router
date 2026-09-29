"""Frozen-artifact loading and prediction service.

Model layer only: loads the preprocessor, calibrator, and trained MLP
exactly once, verifies their SHA-256 hashes against the values frozen in
`mlp_metadata.json` (T304), and serves deterministic single/batch
predictions plus deterministic, rule-based `signals`. This module never
calls an LLM or n8n -- see PROJECT_BLUEPRINT.md Section 2.2 for the
model/agent/automation layer boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from conversion_router.data.preprocess import DEFAULT_PREPROCESSOR_PATH, load_preprocessor
from conversion_router.modeling.calibrate import (
    DEFAULT_CALIBRATOR_PATH,
    apply_calibration,
    decision_margin,
    is_uncertain,
    load_calibrator,
)
from conversion_router.modeling.network import predict_proba as mlp_predict_proba
from conversion_router.modeling.train import DEFAULT_CHECKPOINT_PATH, load_checkpoint
from conversion_router.schemas import (
    ModelInfo,
    PredictedClass,
    PredictionDetail,
    PredictionResponse,
    SessionFeaturesRequest,
    Signal,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "mlp_metadata.json"

# Deterministic, fixed thresholds for the human-readable `signals` field only.
# They are simple explainability heuristics derived from the dataset-wide
# means/median reported in docs/eda-report.md; they never influence the
# model's own probability, threshold, or uncertainty computation.
BOUNCE_RATE_CUTOFF = 0.02
EXIT_RATE_CUTOFF = 0.04
PRODUCT_DURATION_CUTOFF = 600.0
ADMINISTRATIVE_ENGAGEMENT_CUTOFF = 5
INFORMATIONAL_ENGAGEMENT_CUTOFF = 2


class ArtifactUnavailableError(RuntimeError):
    """Raised when a required artifact or metadata file is missing."""


class ArtifactIntegrityError(RuntimeError):
    """Raised when a loaded artifact's SHA-256 hash does not match mlp_metadata.json."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadedArtifacts:
    preprocessor: object
    model: object
    calibrator: object
    metadata: dict


def load_artifacts(
    metadata_path: Path = DEFAULT_METADATA_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    preprocessor_path: Path = DEFAULT_PREPROCESSOR_PATH,
    calibrator_path: Path = DEFAULT_CALIBRATOR_PATH,
) -> LoadedArtifacts:
    """Load and integrity-check all frozen artifacts. Call once at startup.

    Raises ArtifactUnavailableError if metadata or any artifact file is
    missing, and ArtifactIntegrityError if a file's SHA-256 does not match
    the hash frozen in mlp_metadata.json.
    """
    if not metadata_path.exists():
        raise ArtifactUnavailableError(f"{metadata_path} not found.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    for path, hash_key in (
        (checkpoint_path, "mlp_checkpoint_sha256"),
        (preprocessor_path, "preprocessor_sha256"),
        (calibrator_path, "calibrator_sha256"),
    ):
        if not path.exists():
            raise ArtifactUnavailableError(f"Required artifact missing: {path}")
        expected = metadata.get("artifact_hashes", {}).get(hash_key)
        if not expected:
            raise ArtifactUnavailableError(f"No recorded hash for {hash_key} in {metadata_path}")
        actual = sha256_of(path)
        if actual != expected:
            raise ArtifactIntegrityError(
                f"{path.name} hash mismatch: expected {expected}, got {actual}"
            )

    preprocessor = load_preprocessor(preprocessor_path)
    model = load_checkpoint(checkpoint_path)
    calibrator = load_calibrator(calibrator_path)
    return LoadedArtifacts(
        preprocessor=preprocessor, model=model, calibrator=calibrator, metadata=metadata
    )


def _compute_signals(row: dict) -> list[Signal]:
    signals: list[Signal] = []

    if row["VisitorType"] == "Returning_Visitor":
        signals.append(Signal.RETURNING_VISITOR)
    elif row["VisitorType"] == "New_Visitor":
        signals.append(Signal.NEW_VISITOR)

    signals.append(Signal.WEEKEND_SESSION if row["Weekend"] else Signal.WEEKDAY_SESSION)

    signals.append(
        Signal.HIGH_BOUNCE_RATE if row["BounceRates"] >= BOUNCE_RATE_CUTOFF
        else Signal.LOW_BOUNCE_RATE
    )
    signals.append(
        Signal.HIGH_EXIT_RATE if row["ExitRates"] >= EXIT_RATE_CUTOFF else Signal.LOW_EXIT_RATE
    )
    signals.append(
        Signal.LONG_PRODUCT_RELATED_DURATION
        if row["ProductRelated_Duration"] >= PRODUCT_DURATION_CUTOFF
        else Signal.SHORT_PRODUCT_RELATED_DURATION
    )

    if row["SpecialDay"] > 0:
        signals.append(Signal.NEAR_SPECIAL_DAY)
    if row["Administrative"] >= ADMINISTRATIVE_ENGAGEMENT_CUTOFF:
        signals.append(Signal.HIGH_ADMINISTRATIVE_ENGAGEMENT)
    if row["Informational"] >= INFORMATIONAL_ENGAGEMENT_CUTOFF:
        signals.append(Signal.HIGH_INFORMATIONAL_ENGAGEMENT)

    return signals


def predict_batch(
    artifacts: LoadedArtifacts,
    requests: list[SessionFeaturesRequest],
    request_ids: list[str],
) -> list[PredictionResponse]:
    """Deterministic batch prediction. request_ids must be supplied by the caller."""
    if len(requests) != len(request_ids):
        raise ValueError("requests and request_ids must have the same length")
    if not requests:
        return []

    frame = pd.DataFrame([r.model_dump() for r in requests])
    features = artifacts.preprocessor.transform(frame)
    raw_proba = mlp_predict_proba(artifacts.model, features)
    calibrated_proba = apply_calibration(artifacts.calibrator, raw_proba)

    threshold = artifacts.metadata["threshold_selection"]["threshold"]
    band = artifacts.metadata["calibration"]["uncertainty_band"]
    margins = decision_margin(calibrated_proba, threshold)
    uncertain_flags = is_uncertain(calibrated_proba, threshold, band=band)

    model_info = ModelInfo(
        name=artifacts.metadata["model_name"],
        version=artifacts.metadata["model_version"],
        feature_contract_version=artifacts.metadata["feature_contract_version"],
    )

    responses = []
    for i, (request, request_id) in enumerate(zip(requests, request_ids, strict=True)):
        proba = float(calibrated_proba[i])
        predicted_class = (
            PredictedClass.LIKELY_TO_CONVERT
            if proba >= threshold
            else PredictedClass.UNLIKELY_TO_CONVERT
        )
        responses.append(
            PredictionResponse(
                request_id=request_id,
                prediction=PredictionDetail(
                    purchase_probability=round(proba, 4),
                    decision_threshold=threshold,
                    predicted_class=predicted_class,
                    decision_margin=round(float(margins[i]), 4),
                    uncertain=bool(uncertain_flags[i]),
                ),
                signals=_compute_signals(request.model_dump()),
                model=model_info,
            )
        )
    return responses


def predict_single(
    artifacts: LoadedArtifacts, request: SessionFeaturesRequest, request_id: str
) -> PredictionResponse:
    return predict_batch(artifacts, [request], [request_id])[0]
