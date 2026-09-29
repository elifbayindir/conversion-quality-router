"""Health, readiness, prediction, decision, and routing endpoints.

`/decide` and `/route` call the LLM decision agent implemented in
`conversion_router.agent` (PROJECT_BLUEPRINT.md Section 7). n8n automation
(Section 9) is a separate external workflow that calls this API over HTTP
(see `automation/` and `docs/n8n-workflow.md`); it is not part of this
Python codebase.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from conversion_router.agent.client import get_decision
from conversion_router.agent.fallback import build_fallback_decision
from conversion_router.modeling.inference import ArtifactIntegrityError, predict_single
from conversion_router.schemas import (
    DecisionResponse,
    ErrorCode,
    ErrorDetail,
    ErrorResponse,
    PredictionResponse,
    ReasonCode,
    SessionFeaturesRequest,
)

logger = logging.getLogger("conversion_router.api")

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


def _safe_get_decision(
    prediction: PredictionResponse, request_id: str, request: Request
) -> DecisionResponse:
    """get_decision() already handles every anticipated failure internally and
    never raises; this is one more defensive layer so a genuinely unexpected
    bug in the agent path still degrades to a safe fallback instead of a
    500 error.

    `request.app.state.llm_provider`, if set, is passed through to
    `get_decision` -- production leaves it unset (a real AnthropicProvider is
    built lazily); tests set it to a fake/mock provider so no test run ever
    makes a real network call.
    """
    provider = getattr(request.app.state, "llm_provider", None)
    try:
        return get_decision(prediction, request_id, provider=provider)
    except Exception as exc:  # noqa: BLE001 -- last-resort safety net, see docstring
        logger.exception("Unexpected error in agent decision path: %s", exc)
        return build_fallback_decision(request_id, reason=ReasonCode.AGENT_UNAVAILABLE)


@router.post("/decide", response_model=DecisionResponse)
def decide(payload: PredictionResponse, request: Request) -> DecisionResponse:
    """Turn an already-validated prediction into a routing decision.

    Always returns 200: a deterministic SYSTEM_FALLBACK is a valid,
    successful response from this endpoint's point of view, not an error.
    """
    return _safe_get_decision(payload, payload.request_id, request)


@router.post("/route", response_model=DecisionResponse)
def route(payload: SessionFeaturesRequest, request: Request):
    """Run prediction (T402) then the decision agent (P5) in one call."""
    state = request.app.state.artifact_state
    request_id = generate_request_id()
    if state.artifacts is None:
        return _model_unavailable_response(request_id, state.load_error)

    prediction = predict_single(state.artifacts, payload, request_id)
    decision = _safe_get_decision(prediction, request_id, request)
    return JSONResponse(status_code=200, content=decision.model_dump(mode="json"))
