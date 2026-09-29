"""Deterministic cross-field validation for LLM decision output.

Runs AFTER Pydantic schema validation (which only checks structure/enums in
isolation). This enforces the policy stated in
`prompts/decision_agent_v1.md` that a JSON Schema alone cannot express --
e.g. that each decision pairs with exactly one allowed_action, and that an
uncertain prediction forces HUMAN_REVIEW regardless of what the model chose.
"""

from __future__ import annotations

from conversion_router.schemas import (
    AllowedAction,
    LLMDecision,
    LLMDecisionCore,
    LLMReasonCode,
    PredictionResponse,
)

ALLOWED_ACTION_FOR_DECISION = {
    LLMDecision.LOG_ONLY: AllowedAction.ADD_TO_LOG,
    LLMDecision.PRIORITY_REVIEW: AllowedAction.ADD_TO_REVIEW_QUEUE_PRIORITY,
    LLMDecision.HUMAN_REVIEW: AllowedAction.ADD_TO_REVIEW_QUEUE,
}


class DecisionValidationError(ValueError):
    """Raised when an LLM decision core violates the deterministic cross-field policy."""


def validate_decision_core(core: LLMDecisionCore, prediction: PredictionResponse) -> None:
    """Raises DecisionValidationError listing every violated rule, if any."""
    errors: list[str] = []

    expected_action = ALLOWED_ACTION_FOR_DECISION[core.decision]
    if core.allowed_action != expected_action:
        errors.append(
            f"decision={core.decision} must pair with allowed_action={expected_action}, "
            f"got allowed_action={core.allowed_action}"
        )

    if prediction.prediction.uncertain:
        if core.decision != LLMDecision.HUMAN_REVIEW:
            errors.append("uncertain=true requires decision=HUMAN_REVIEW")
        if not core.requires_human_approval:
            errors.append("uncertain=true requires requires_human_approval=true")
        if core.allowed_action != AllowedAction.ADD_TO_REVIEW_QUEUE:
            errors.append("uncertain=true requires allowed_action=ADD_TO_REVIEW_QUEUE")
        if LLMReasonCode.MODEL_UNCERTAIN not in core.reason_codes:
            errors.append("uncertain=true requires MODEL_UNCERTAIN in reason_codes")

    if errors:
        raise DecisionValidationError("; ".join(errors))
