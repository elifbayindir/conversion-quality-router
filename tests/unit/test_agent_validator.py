import pytest

from conversion_router.agent.validator import DecisionValidationError, validate_decision_core
from conversion_router.schemas import (
    AllowedAction,
    LLMDecision,
    LLMDecisionCore,
    LLMReasonCode,
    ModelInfo,
    PredictedClass,
    PredictionDetail,
    PredictionResponse,
    Priority,
)


def _prediction(uncertain: bool, proba: float = 0.7, threshold: float = 0.5) -> PredictionResponse:
    return PredictionResponse(
        request_id="req_test",
        prediction=PredictionDetail(
            purchase_probability=proba,
            decision_threshold=threshold,
            predicted_class=(
                PredictedClass.LIKELY_TO_CONVERT
                if proba >= threshold
                else PredictedClass.UNLIKELY_TO_CONVERT
            ),
            decision_margin=abs(proba - threshold),
            uncertain=uncertain,
        ),
        signals=[],
        model=ModelInfo(name="conversion_mlp", version="1.0.0", feature_contract_version="1.0.0"),
    )


def _core(**overrides) -> LLMDecisionCore:
    kwargs = {
        "decision": LLMDecision.PRIORITY_REVIEW,
        "priority": Priority.HIGH,
        "allowed_action": AllowedAction.ADD_TO_REVIEW_QUEUE_PRIORITY,
        "requires_human_approval": False,
        "reason_codes": [LLMReasonCode.HIGH_CONFIDENCE_POSITIVE],
        "explanation": "Strong signal.",
    }
    kwargs.update(overrides)
    return LLMDecisionCore(**kwargs)


def test_valid_certain_decision_passes():
    core = _core()
    validate_decision_core(core, _prediction(uncertain=False))  # must not raise


def test_valid_uncertain_human_review_passes():
    core = _core(
        decision=LLMDecision.HUMAN_REVIEW,
        priority=Priority.MEDIUM,
        allowed_action=AllowedAction.ADD_TO_REVIEW_QUEUE,
        requires_human_approval=True,
        reason_codes=[LLMReasonCode.MODEL_UNCERTAIN],
    )
    validate_decision_core(core, _prediction(uncertain=True))  # must not raise


def test_uncertain_true_with_log_only_is_rejected():
    core = _core(
        decision=LLMDecision.LOG_ONLY,
        allowed_action=AllowedAction.ADD_TO_LOG,
        reason_codes=[LLMReasonCode.MODEL_UNCERTAIN],
    )
    with pytest.raises(DecisionValidationError, match="HUMAN_REVIEW"):
        validate_decision_core(core, _prediction(uncertain=True))


def test_uncertain_true_without_requires_human_approval_is_rejected():
    core = _core(
        decision=LLMDecision.HUMAN_REVIEW,
        allowed_action=AllowedAction.ADD_TO_REVIEW_QUEUE,
        requires_human_approval=False,
        reason_codes=[LLMReasonCode.MODEL_UNCERTAIN],
    )
    with pytest.raises(DecisionValidationError, match="requires_human_approval"):
        validate_decision_core(core, _prediction(uncertain=True))


def test_uncertain_true_without_model_uncertain_reason_code_is_rejected():
    core = _core(
        decision=LLMDecision.HUMAN_REVIEW,
        allowed_action=AllowedAction.ADD_TO_REVIEW_QUEUE,
        requires_human_approval=True,
        reason_codes=[LLMReasonCode.NEAR_DECISION_THRESHOLD],
    )
    with pytest.raises(DecisionValidationError, match="MODEL_UNCERTAIN"):
        validate_decision_core(core, _prediction(uncertain=True))


def test_log_only_with_wrong_action_is_rejected():
    core = _core(decision=LLMDecision.LOG_ONLY, allowed_action=AllowedAction.ADD_TO_REVIEW_QUEUE)
    with pytest.raises(DecisionValidationError, match="ADD_TO_LOG"):
        validate_decision_core(core, _prediction(uncertain=False))


def test_priority_review_with_wrong_action_is_rejected():
    core = _core(
        decision=LLMDecision.PRIORITY_REVIEW, allowed_action=AllowedAction.ADD_TO_REVIEW_QUEUE
    )
    with pytest.raises(DecisionValidationError):
        validate_decision_core(core, _prediction(uncertain=False))


def test_human_review_with_wrong_action_is_rejected():
    core = _core(
        decision=LLMDecision.HUMAN_REVIEW, allowed_action=AllowedAction.ADD_TO_REVIEW_QUEUE_PRIORITY
    )
    with pytest.raises(DecisionValidationError):
        validate_decision_core(core, _prediction(uncertain=False))
