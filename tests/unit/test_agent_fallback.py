from conversion_router.agent.fallback import build_fallback_decision
from conversion_router.schemas import AllowedAction, Decision, DecisionResponse, ReasonCode


def test_fallback_decision_is_a_valid_decision_response():
    response = build_fallback_decision("req_fallback_1")
    assert isinstance(response, DecisionResponse)


def test_fallback_decision_uses_system_fallback_and_safe_defaults():
    response = build_fallback_decision("req_fallback_2")
    assert response.decision == Decision.SYSTEM_FALLBACK
    assert response.requires_human_approval is True
    assert response.allowed_action == AllowedAction.ADD_TO_REVIEW_QUEUE
    assert response.reason_codes == [ReasonCode.AGENT_UNAVAILABLE]


def test_fallback_decision_accepts_a_different_reason_code():
    response = build_fallback_decision("req_fallback_3", reason=ReasonCode.AGENT_OUTPUT_INVALID)
    assert response.reason_codes == [ReasonCode.AGENT_OUTPUT_INVALID]


def test_fallback_decision_carries_the_given_request_id():
    response = build_fallback_decision("req_fallback_4")
    assert response.request_id == "req_fallback_4"
