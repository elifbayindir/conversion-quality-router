"""Health, readiness, and prediction routes.

Only these three endpoints exist in this phase. `/decide` and `/route`
depend on the LLM decision agent and n8n automation, which are separate,
later phases (PROJECT_BLUEPRINT.md Sections 7 and 9) -- not implemented here.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from conversion_router.modeling.inference import ArtifactIntegrityError, predict_single
from conversion_router.schemas import (
    ErrorCode,
    ErrorDetail,
    ErrorResponse,
    PredictionResponse,
    SessionFeaturesRequest,
)

router = APIRouter()


def generate_request_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return f"req_{timestamp}_{uuid.uuid4().hex[:8]}"


def _model_unavailable_response(request_id: str, load_error: Exception | None) -> JSONResponse:
    error = ErrorResponse(
        request_id=request_id,
        error=ErrorDetail(
            code=ErrorCode.MODEL_UNAVAILABLE,
            message=(
                str(load_error) if load_error else "Prediction service is unavailable."
            )[:500],
            retryable=not isinstance(load_error, ArtifactIntegrityError),
        ),
    )
    return JSONResponse(status_code=503, content=error.model_dump(mode="json"))


@router.get("/health")
def health() -> dict:
    """Process liveness only. Never loads or checks model artifacts."""
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request):
    state = request.app.state.artifact_state
    if state.artifacts is None:
        return _model_unavailable_response(generate_request_id(), state.load_error)
    metadata = state.artifacts.metadata
    return {
        "status": "ready",
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "feature_contract_version": metadata["feature_contract_version"],
    }


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: SessionFeaturesRequest, request: Request):
    state = request.app.state.artifact_state
    request_id = generate_request_id()
    if state.artifacts is None:
        return _model_unavailable_response(request_id, state.load_error)

    response: PredictionResponse = predict_single(state.artifacts, payload, request_id)
    return JSONResponse(status_code=200, content=response.model_dump(mode="json"))
