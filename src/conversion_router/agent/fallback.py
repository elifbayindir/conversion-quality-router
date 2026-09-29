"""Deterministic, non-LLM fallback decision (PROJECT_BLUEPRINT.md Section 7.5).

This is never an LLM call. It always returns a schema-valid DecisionResponse
so /decide and /route can never crash or return an unvalidated payload, even
when the agent is completely unavailable.
"""

from __future__ import annotations

from conversion_router.schemas import (
    AllowedAction,
    Decision,
    DecisionResponse,
    Priority,
    ReasonCode,
)

AGENT_VERSION = "1.0.0"


def build_fallback_decision(
    request_id: str,
    reason: ReasonCode = ReasonCode.AGENT_UNAVAILABLE,
) -> DecisionResponse:
    return DecisionResponse(
        request_id=request_id,
        decision=Decision.SYSTEM_FALLBACK,
        priority=Priority.HIGH,
        allowed_action=AllowedAction.ADD_TO_REVIEW_QUEUE,
        requires_human_approval=True,
        reason_codes=[reason],
        explanation="Agent output could not be validated; routed to human review.",
        agent_version=AGENT_VERSION,
    )
