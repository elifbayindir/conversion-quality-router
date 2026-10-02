"""Feedback and audit-record contracts.

Every model-side and agent-side field is copied from the existing
`PredictionResponse` / `DecisionResponse` contracts; only the reviewer fields
are new. Human decisions reuse the `Decision` enum (minus `SYSTEM_FALLBACK`,
which is a system state, not a choice a reviewer can make).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from conversion_router.schemas import (
    AllowedAction,
    Decision,
    DecisionResponse,
    PredictedClass,
    PredictionResponse,
    Priority,
    ReasonCode,
    SessionFeaturesRequest,
    Signal,
    StrictModel,
)

FEEDBACK_SCHEMA_VERSION = "1.0.0"
NOTE_MAX_LENGTH = 500

HUMAN_DECISIONS = (Decision.LOG_ONLY, Decision.PRIORITY_REVIEW, Decision.HUMAN_REVIEW)


class HumanResponse(StrEnum):
    AGREE = "AGREE"
    OVERRIDE = "OVERRIDE"


class OverrideReason(StrEnum):
    CONTEXT_NOT_IN_FEATURES = "CONTEXT_NOT_IN_FEATURES"
    SIGNAL_STRONGER_THAN_SCORED = "SIGNAL_STRONGER_THAN_SCORED"
    SIGNAL_WEAKER_THAN_SCORED = "SIGNAL_WEAKER_THAN_SCORED"
    UNCERTAINTY_RESOLVED_BY_REVIEW = "UNCERTAINTY_RESOLVED_BY_REVIEW"
    FALLBACK_RESOLVED_BY_REVIEW = "FALLBACK_RESOLVED_BY_REVIEW"
    OTHER = "OTHER"


class FeedbackError(ValueError):
    """Raised when reviewer feedback is inconsistent with the recommendation."""


class FeedbackSubmission(StrictModel):
    """What the reviewer entered in the UI."""

    response: HumanResponse
    override_decision: Decision | None = None
    override_reason: OverrideReason | None = None
    note: str | None = Field(default=None, max_length=NOTE_MAX_LENGTH)

    @field_validator("note")
    @classmethod
    def _blank_note_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def _consistent(self) -> FeedbackSubmission:
        if self.response == HumanResponse.AGREE:
            if self.override_decision is not None or self.override_reason is not None:
                raise ValueError("An agreement cannot carry an override decision or reason.")
        else:
            if self.override_decision is None or self.override_reason is None:
                raise ValueError("An override needs a decision and a structured reason.")
            if self.override_decision not in HUMAN_DECISIONS:
                raise ValueError(f"{self.override_decision} is not a reviewer decision.")
            if self.override_reason == OverrideReason.OTHER and not self.note:
                raise ValueError("Override reason OTHER requires a note.")
        return self


class AuditRecord(StrictModel):
    """One reviewed case: model output, agent decision, and human response."""

    event_id: str
    created_at: str
    request_id: str
    input_fingerprint: str = Field(min_length=64, max_length=64)
    # Model evidence (from PredictionResponse)
    purchase_probability: float = Field(ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)
    decision_margin: float = Field(ge=0, le=1)
    uncertain: bool
    predicted_class: PredictedClass
    signals: list[Signal]
    model_name: str
    model_version: str
    feature_contract_version: str
    # Agent recommendation (from DecisionResponse)
    agent_decision: Decision
    priority: Priority
    recommended_action: AllowedAction
    requires_human_approval: bool
    agent_reason_codes: list[ReasonCode]
    agent_version: str
    decision_schema_version: str
    # Human response
    human_response: HumanResponse
    override_reason: OverrideReason | None
    reviewer_note: str | None = Field(default=None, max_length=NOTE_MAX_LENGTH)
    final_decision: Decision
    feedback_schema_version: str = FEEDBACK_SCHEMA_VERSION


def input_fingerprint(session: SessionFeaturesRequest) -> str:
    """SHA-256 of the canonical request JSON: links a record to its input
    without storing the raw feature values. (Not the split fingerprint of
    `conversion_router.data.split`, which serves a different purpose.)"""
    canonical = json.dumps(session.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_audit_record(
    session: SessionFeaturesRequest,
    prediction: PredictionResponse,
    decision: DecisionResponse,
    submission: FeedbackSubmission,
    *,
    now: datetime | None = None,
    event_id: str | None = None,
) -> AuditRecord:
    if decision.request_id != prediction.request_id:
        raise FeedbackError("Decision and prediction belong to different requests.")
    if submission.response == HumanResponse.AGREE:
        if decision.decision == Decision.SYSTEM_FALLBACK:
            raise FeedbackError(
                "A safety fallback carries no recommendation to agree with; "
                "select the decision you want instead."
            )
        final_decision = decision.decision
    else:
        if submission.override_decision == decision.decision:
            raise FeedbackError("An override must choose a different decision.")
        final_decision = submission.override_decision

    detail = prediction.prediction
    return AuditRecord(
        event_id=event_id or f"fb_{uuid.uuid4().hex}",
        created_at=(now or datetime.now(UTC)).isoformat(timespec="seconds"),
        request_id=prediction.request_id,
        input_fingerprint=input_fingerprint(session),
        purchase_probability=detail.purchase_probability,
        decision_threshold=detail.decision_threshold,
        decision_margin=detail.decision_margin,
        uncertain=detail.uncertain,
        predicted_class=detail.predicted_class,
        signals=list(prediction.signals),
        model_name=prediction.model.name,
        model_version=prediction.model.version,
        feature_contract_version=prediction.model.feature_contract_version,
        agent_decision=decision.decision,
        priority=decision.priority,
        recommended_action=decision.allowed_action,
        requires_human_approval=decision.requires_human_approval,
        agent_reason_codes=list(decision.reason_codes),
        agent_version=decision.agent_version,
        decision_schema_version=decision.schema_version,
        human_response=submission.response,
        override_reason=submission.override_reason,
        reviewer_note=submission.note,
        final_decision=final_decision,
    )
