"""Pure display helpers: turn API contract values into analyst-facing text.

No thresholds, bands, or routing rules live here. Every function formats a
value that the API already returned (`PredictionResponse`/`DecisionResponse`).
"""

from __future__ import annotations

from dataclasses import dataclass

from conversion_router.feedback.schemas import HumanResponse, OverrideReason
from conversion_router.schemas import (
    AllowedAction,
    Decision,
    DecisionResponse,
    Priority,
    ReasonCode,
    Signal,
)


@dataclass(frozen=True)
class RouteDisplay:
    title: str
    summary: str
    tone: str  # one of: "neutral", "positive", "attention", "warning"


ROUTE_DISPLAY: dict[Decision, RouteDisplay] = {
    Decision.LOG_ONLY: RouteDisplay(
        "Log automatically",
        "No reviewer time needed. The session is recorded for audit only.",
        "neutral",
    ),
    Decision.PRIORITY_REVIEW: RouteDisplay(
        "Priority review",
        "Strong purchase signal. Put this session at the front of the review queue.",
        "positive",
    ),
    Decision.HUMAN_REVIEW: RouteDisplay(
        "Human review",
        "The model is not confident either way. A reviewer should decide.",
        "attention",
    ),
    Decision.SYSTEM_FALLBACK: RouteDisplay(
        "Safety fallback: human review",
        "The decision agent's output could not be validated, so the system routed "
        "the session to human review instead of guessing.",
        "warning",
    ),
}

ACTION_LABELS: dict[AllowedAction, str] = {
    AllowedAction.ADD_TO_LOG: "Add to audit log",
    AllowedAction.ADD_TO_REVIEW_QUEUE_PRIORITY: "Add to priority review queue",
    AllowedAction.ADD_TO_REVIEW_QUEUE: "Add to review queue",
}

PRIORITY_LABELS: dict[Priority, str] = {
    Priority.LOW: "Low",
    Priority.MEDIUM: "Medium",
    Priority.HIGH: "High",
}

REASON_LABELS: dict[ReasonCode, str] = {
    ReasonCode.MODEL_UNCERTAIN: "Model is uncertain",
    ReasonCode.NEAR_DECISION_THRESHOLD: "Probability is near the decision threshold",
    ReasonCode.HIGH_CONFIDENCE_POSITIVE: "Clearly above the decision threshold",
    ReasonCode.HIGH_CONFIDENCE_NEGATIVE: "Clearly below the decision threshold",
    ReasonCode.AGENT_OUTPUT_INVALID: "Decision agent output failed validation",
    ReasonCode.AGENT_UNAVAILABLE: "Decision agent unavailable",
}

SIGNAL_LABELS: dict[Signal, str] = {
    Signal.NEW_VISITOR: "New visitor",
    Signal.RETURNING_VISITOR: "Returning visitor",
    Signal.WEEKEND_SESSION: "Weekend session",
    Signal.WEEKDAY_SESSION: "Weekday session",
    Signal.HIGH_BOUNCE_RATE: "High bounce rate",
    Signal.LOW_BOUNCE_RATE: "Low bounce rate",
    Signal.HIGH_EXIT_RATE: "High exit rate",
    Signal.LOW_EXIT_RATE: "Low exit rate",
    Signal.LONG_PRODUCT_RELATED_DURATION: "Long time on product pages",
    Signal.SHORT_PRODUCT_RELATED_DURATION: "Short time on product pages",
    Signal.NEAR_SPECIAL_DAY: "Close to a special shopping day",
    Signal.HIGH_ADMINISTRATIVE_ENGAGEMENT: "Many account pages viewed",
    Signal.HIGH_INFORMATIONAL_ENGAGEMENT: "Many information pages viewed",
}


def format_probability(value: float) -> str:
    return f"{value * 100:.1f}%"


def uncertainty_text(uncertain: bool, decision_margin: float) -> str:
    distance = f"{decision_margin * 100:.1f} percentage points from the threshold"
    if uncertain:
        return f"Uncertain: inside the uncertainty band ({distance})."
    return f"Clear signal: outside the uncertainty band ({distance})."


def route_display(decision: DecisionResponse) -> RouteDisplay:
    return ROUTE_DISPLAY[decision.decision]


def is_fallback(decision: DecisionResponse) -> bool:
    return decision.decision == Decision.SYSTEM_FALLBACK


def policy_version_text(decision: DecisionResponse) -> str:
    return f"agent {decision.agent_version} / schema {decision.schema_version}"


# -- reviewer feedback -----------------------------------------------------

FEEDBACK_NOTICE = (
    "Feedback is stored locally as evidence for a future, human-led review of the "
    "decision policy. Recording feedback does not retrain the model, change the "
    "decision threshold, or modify the policy. It does not resume an n8n approval."
)

HUMAN_RESPONSE_LABELS: dict[HumanResponse, str] = {
    HumanResponse.AGREE: "Agree with recommendation",
    HumanResponse.OVERRIDE: "Override recommendation",
}

OVERRIDE_REASON_LABELS: dict[OverrideReason, str] = {
    OverrideReason.CONTEXT_NOT_IN_FEATURES: "Reviewer has context the model does not see",
    OverrideReason.SIGNAL_STRONGER_THAN_SCORED: "Session looks stronger than the score suggests",
    OverrideReason.SIGNAL_WEAKER_THAN_SCORED: "Session looks weaker than the score suggests",
    OverrideReason.UNCERTAINTY_RESOLVED_BY_REVIEW: "Review resolved the model's uncertainty",
    OverrideReason.FALLBACK_RESOLVED_BY_REVIEW: "Reviewer decided a safety-fallback case",
    OverrideReason.OTHER: "Other (explain in the note)",
}


NOT_APPLICABLE_AGREED = "Not applicable — reviewer agreed"


def override_reason_text(reason: OverrideReason | None) -> str:
    """Display text only; agree records keep no reason in the store."""
    return OVERRIDE_REASON_LABELS[reason] if reason is not None else NOT_APPLICABLE_AGREED


def short_request_id(request_id: str) -> str:
    """Compact form for tables; the full ID stays in details and the audit record."""
    return request_id if len(request_id) <= 12 else f"…{request_id[-8:]}"


FEEDBACK_DISCLAIMER = (
    "These records are entered manually in this local prototype and stored only on this "
    "machine. A handful of demo reviews is not a user study. The override rate shows how "
    "often a reviewer chose a different route; it is not a measure of model performance. "
    "Feedback never changes the model, the threshold, or the policy automatically."
)
