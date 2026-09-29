"""Strict Pydantic contracts for the API, model, and (future) agent layers.

Every model here forbids unknown fields. Numeric ranges and categorical
whitelists are derived from the same single source of truth used by data
validation (`conversion_router.data.validate`), so the API contract and the
training-time schema can never silently drift apart.

This module defines data shapes only. It does not implement the LLM agent
or n8n behavior (see PROJECT_BLUEPRINT.md Sections 7 and 9) -- the
`DecisionResponse` schema exists so later phases have a stable contract to
target, not because the decision logic is implemented yet.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from conversion_router.data.validate import (
    CATEGORICAL_WHITELISTS,
    NONNEGATIVE_NUMERIC_COLUMNS,
    UNIT_INTERVAL_COLUMNS,
)

SCHEMA_VERSION = "1.0.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Prediction request (the 16 deployment-safe features only -- no PageValues)
# ---------------------------------------------------------------------------


class SessionFeaturesRequest(StrictModel):
    Administrative: int = Field(ge=0)
    Administrative_Duration: float = Field(ge=0)
    Informational: int = Field(ge=0)
    Informational_Duration: float = Field(ge=0)
    ProductRelated: int = Field(ge=0)
    ProductRelated_Duration: float = Field(ge=0)
    BounceRates: float = Field(ge=0, le=1)
    ExitRates: float = Field(ge=0, le=1)
    SpecialDay: float = Field(ge=0, le=1)
    Month: str
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str
    Weekend: bool

    @field_validator(
        "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"
    )
    @classmethod
    def _must_be_whitelisted(cls, value, info):
        whitelist = CATEGORICAL_WHITELISTS[info.field_name]
        if value not in whitelist:
            raise ValueError(
                f"{info.field_name}={value!r} is not in the allowed set "
                f"{sorted(whitelist, key=str)}"
            )
        return value


_REQUEST_NONNEGATIVE_FIELDS = {
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
}
_REQUEST_UNIT_INTERVAL_FIELDS = {"BounceRates", "ExitRates", "SpecialDay"}

# Every numeric range enforced on the request must be grounded in the same
# constants used by data validation, never invented independently.
assert _REQUEST_NONNEGATIVE_FIELDS <= set(NONNEGATIVE_NUMERIC_COLUMNS)
assert _REQUEST_UNIT_INTERVAL_FIELDS <= set(UNIT_INTERVAL_COLUMNS)


# ---------------------------------------------------------------------------
# Prediction response (PROJECT_BLUEPRINT.md Section 6)
# ---------------------------------------------------------------------------


class PredictedClass(StrEnum):
    LIKELY_TO_CONVERT = "likely_to_convert"
    UNLIKELY_TO_CONVERT = "unlikely_to_convert"


class Signal(StrEnum):
    NEW_VISITOR = "new_visitor"
    RETURNING_VISITOR = "returning_visitor"
    WEEKEND_SESSION = "weekend_session"
    WEEKDAY_SESSION = "weekday_session"
    HIGH_BOUNCE_RATE = "high_bounce_rate"
    LOW_BOUNCE_RATE = "low_bounce_rate"
    HIGH_EXIT_RATE = "high_exit_rate"
    LOW_EXIT_RATE = "low_exit_rate"
    LONG_PRODUCT_RELATED_DURATION = "long_product_related_duration"
    SHORT_PRODUCT_RELATED_DURATION = "short_product_related_duration"
    NEAR_SPECIAL_DAY = "near_special_day"
    HIGH_ADMINISTRATIVE_ENGAGEMENT = "high_administrative_engagement"
    HIGH_INFORMATIONAL_ENGAGEMENT = "high_informational_engagement"


class PredictionDetail(StrictModel):
    purchase_probability: float = Field(ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)
    predicted_class: PredictedClass
    decision_margin: float = Field(ge=0, le=1)
    uncertain: bool


class ModelInfo(StrictModel):
    name: str
    version: str
    feature_contract_version: str


class PredictionResponse(StrictModel):
    request_id: str
    prediction: PredictionDetail
    signals: list[Signal] = Field(default_factory=list, max_length=len(Signal))
    model: ModelInfo


# ---------------------------------------------------------------------------
# Decision response (PROJECT_BLUEPRINT.md Section 7.3) -- schema only.
# The agent that produces this payload is implemented in a later phase.
# ---------------------------------------------------------------------------


class Decision(StrEnum):
    LOG_ONLY = "LOG_ONLY"
    PRIORITY_REVIEW = "PRIORITY_REVIEW"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    SYSTEM_FALLBACK = "SYSTEM_FALLBACK"


class Priority(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class AllowedAction(StrEnum):
    ADD_TO_LOG = "ADD_TO_LOG"
    ADD_TO_REVIEW_QUEUE_PRIORITY = "ADD_TO_REVIEW_QUEUE_PRIORITY"
    ADD_TO_REVIEW_QUEUE = "ADD_TO_REVIEW_QUEUE"


class ReasonCode(StrEnum):
    MODEL_UNCERTAIN = "MODEL_UNCERTAIN"
    NEAR_DECISION_THRESHOLD = "NEAR_DECISION_THRESHOLD"
    HIGH_CONFIDENCE_POSITIVE = "HIGH_CONFIDENCE_POSITIVE"
    HIGH_CONFIDENCE_NEGATIVE = "HIGH_CONFIDENCE_NEGATIVE"
    AGENT_OUTPUT_INVALID = "AGENT_OUTPUT_INVALID"
    AGENT_UNAVAILABLE = "AGENT_UNAVAILABLE"


class DecisionResponse(StrictModel):
    request_id: str
    decision: Decision
    priority: Priority
    allowed_action: AllowedAction
    requires_human_approval: bool
    reason_codes: list[ReasonCode] = Field(min_length=1, max_length=len(ReasonCode))
    explanation: str = Field(max_length=500)
    agent_version: str
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Error response (PROJECT_BLUEPRINT.md Section 8.3)
# ---------------------------------------------------------------------------


class ErrorCode(StrEnum):
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    ARTIFACT_INTEGRITY_ERROR = "ARTIFACT_INTEGRITY_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorDetail(StrictModel):
    code: ErrorCode
    message: str = Field(max_length=500)
    retryable: bool


class ErrorResponse(StrictModel):
    request_id: str
    error: ErrorDetail
