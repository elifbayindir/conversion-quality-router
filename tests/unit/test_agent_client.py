import json

import httpx
import pytest

from conversion_router.agent.client import (
    ANTHROPIC_MODEL,
    AgentCredentialError,
    AgentOutputError,
    AgentTransportError,
    AnthropicProvider,
    get_decision,
)
from conversion_router.schemas import (
    Decision,
    ModelInfo,
    PredictedClass,
    PredictionDetail,
    PredictionResponse,
    ReasonCode,
)


def _prediction(uncertain: bool = False, proba: float = 0.7, threshold: float = 0.5):
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


VALID_CERTAIN_RESPONSE = json.dumps(
    {
        "decision": "PRIORITY_REVIEW",
        "priority": "HIGH",
        "allowed_action": "ADD_TO_REVIEW_QUEUE_PRIORITY",
        "requires_human_approval": False,
        "reason_codes": ["HIGH_CONFIDENCE_POSITIVE"],
        "explanation": "Strong, comfortably-above-threshold signal.",
    }
)

VALID_UNCERTAIN_RESPONSE = json.dumps(
    {
        "decision": "HUMAN_REVIEW",
        "priority": "MEDIUM",
        "allowed_action": "ADD_TO_REVIEW_QUEUE",
        "requires_human_approval": True,
        "reason_codes": ["MODEL_UNCERTAIN"],
        "explanation": "Prediction is near the decision threshold.",
    }
)


class FakeProvider:
    """Implements the LLMProvider protocol; queues canned responses or exceptions."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls = 0

    def complete_decision(self, system_prompt: str, user_content: str) -> str:
        self.calls += 1
        item = self._responses[min(self.calls - 1, len(self._responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_get_decision_succeeds_on_first_attempt():
    provider = FakeProvider([VALID_CERTAIN_RESPONSE])
    result = get_decision(_prediction(), "req_1", provider=provider)
    assert provider.calls == 1
    assert result.decision == Decision.PRIORITY_REVIEW
    assert result.request_id == "req_1"


def test_get_decision_retries_once_then_succeeds():
    provider = FakeProvider(["not json", VALID_CERTAIN_RESPONSE])
    result = get_decision(_prediction(), "req_2", provider=provider)
    assert provider.calls == 2
    assert result.decision == Decision.PRIORITY_REVIEW


def test_get_decision_uncertain_prediction_with_valid_human_review():
    provider = FakeProvider([VALID_UNCERTAIN_RESPONSE])
    result = get_decision(_prediction(uncertain=True), "req_3", provider=provider)
    assert result.decision == Decision.HUMAN_REVIEW
    assert result.requires_human_approval is True


# ---------------------------------------------------------------------------
# Adversarial: malformed / invalid content -> exactly one retry, then fallback
# ---------------------------------------------------------------------------


def test_invalid_json_falls_back_after_retry():
    provider = FakeProvider(["not json at all", "still not json"])
    result = get_decision(_prediction(), "req_4", provider=provider)
    assert provider.calls == 2
    assert result.decision == Decision.SYSTEM_FALLBACK
    assert result.reason_codes == [ReasonCode.AGENT_OUTPUT_INVALID]


def test_extra_field_falls_back_after_retry():
    bad = json.dumps({**json.loads(VALID_CERTAIN_RESPONSE), "extra_field": "not allowed"})
    provider = FakeProvider([bad, bad])
    result = get_decision(_prediction(), "req_5", provider=provider)
    assert result.decision == Decision.SYSTEM_FALLBACK


def test_unknown_decision_enum_falls_back_after_retry():
    bad = json.dumps({**json.loads(VALID_CERTAIN_RESPONSE), "decision": "MAYBE_REVIEW"})
    provider = FakeProvider([bad, bad])
    result = get_decision(_prediction(), "req_6", provider=provider)
    assert result.decision == Decision.SYSTEM_FALLBACK


def test_unknown_action_enum_falls_back_after_retry():
    bad = json.dumps({**json.loads(VALID_CERTAIN_RESPONSE), "allowed_action": "SEND_EMAIL"})
    provider = FakeProvider([bad, bad])
    result = get_decision(_prediction(), "req_7", provider=provider)
    assert result.decision == Decision.SYSTEM_FALLBACK


def test_unknown_reason_code_falls_back_after_retry():
    bad = json.dumps(
        {**json.loads(VALID_CERTAIN_RESPONSE), "reason_codes": ["MODEL_IS_ALWAYS_RIGHT"]}
    )
    provider = FakeProvider([bad, bad])
    result = get_decision(_prediction(), "req_8", provider=provider)
    assert result.decision == Decision.SYSTEM_FALLBACK


def test_inconsistent_decision_action_combo_falls_back_after_retry():
    bad = json.dumps(
        {**json.loads(VALID_CERTAIN_RESPONSE), "decision": "LOG_ONLY"}
    )  # allowed_action still ADD_TO_REVIEW_QUEUE_PRIORITY -> mismatch
    provider = FakeProvider([bad, bad])
    result = get_decision(_prediction(), "req_9", provider=provider)
    assert result.decision == Decision.SYSTEM_FALLBACK


def test_uncertain_true_but_llm_returns_log_only_falls_back_after_retry():
    bad = json.dumps(
        {
            "decision": "LOG_ONLY",
            "priority": "LOW",
            "allowed_action": "ADD_TO_LOG",
            "requires_human_approval": False,
            "reason_codes": ["HIGH_CONFIDENCE_NEGATIVE"],
            "explanation": "Looks safe to skip.",
        }
    )
    provider = FakeProvider([bad, bad])
    result = get_decision(_prediction(uncertain=True), "req_10", provider=provider)
    assert result.decision == Decision.SYSTEM_FALLBACK
    assert result.requires_human_approval is True


def test_llm_attempting_system_fallback_is_rejected_and_falls_back():
    bad = json.dumps({**json.loads(VALID_CERTAIN_RESPONSE), "decision": "SYSTEM_FALLBACK"})
    provider = FakeProvider([bad, bad])
    result = get_decision(_prediction(), "req_11", provider=provider)
    assert result.decision == Decision.SYSTEM_FALLBACK
    assert result.reason_codes == [ReasonCode.AGENT_OUTPUT_INVALID]


def test_llm_attempting_to_set_immutable_fields_is_rejected():
    tampered = json.dumps(
        {
            **json.loads(VALID_CERTAIN_RESPONSE),
            "request_id": "req_hijacked",
            "purchase_probability": 0.99,
            "agent_version": "9.9.9",
        }
    )
    provider = FakeProvider([tampered, tampered])
    result = get_decision(_prediction(), "req_12", provider=provider)
    assert result.decision == Decision.SYSTEM_FALLBACK
    assert result.request_id == "req_12"  # app-assigned, never the tampered value
    assert result.agent_version == "1.0.0"


def test_oversized_explanation_falls_back_after_retry():
    bad = json.dumps({**json.loads(VALID_CERTAIN_RESPONSE), "explanation": "x" * 500})
    provider = FakeProvider([bad, bad])
    result = get_decision(_prediction(), "req_13", provider=provider)
    assert result.decision == Decision.SYSTEM_FALLBACK


def test_empty_content_falls_back_after_retry():
    provider = FakeProvider([AgentOutputError("empty content"), AgentOutputError("empty content")])
    result = get_decision(_prediction(), "req_14", provider=provider)
    assert result.decision == Decision.SYSTEM_FALLBACK
    assert result.reason_codes == [ReasonCode.AGENT_OUTPUT_INVALID]


def test_refusal_falls_back_after_retry():
    provider = FakeProvider([AgentOutputError("refused"), AgentOutputError("refused")])
    result = get_decision(_prediction(), "req_15", provider=provider)
    assert result.decision == Decision.SYSTEM_FALLBACK


def test_timeout_falls_back_after_retry():
    provider = FakeProvider(
        [AgentTransportError("timeout"), AgentTransportError("timeout")]
    )
    result = get_decision(_prediction(), "req_16", provider=provider)
    assert result.decision == Decision.SYSTEM_FALLBACK
    assert result.reason_codes == [ReasonCode.AGENT_UNAVAILABLE]


def test_http_429_falls_back_after_retry():
    provider = FakeProvider(
        [AgentTransportError("429 rate limited"), AgentTransportError("429 rate limited")]
    )
    result = get_decision(_prediction(), "req_17", provider=provider)
    assert result.decision == Decision.SYSTEM_FALLBACK


def test_http_5xx_falls_back_after_retry():
    provider = FakeProvider(
        [AgentTransportError("server error (500)"), AgentTransportError("server error (500)")]
    )
    result = get_decision(_prediction(), "req_18", provider=provider)
    assert result.decision == Decision.SYSTEM_FALLBACK


def test_success_on_retry_after_transport_failure_does_not_fall_back():
    provider = FakeProvider([AgentTransportError("timeout"), VALID_CERTAIN_RESPONSE])
    result = get_decision(_prediction(), "req_19", provider=provider)
    assert provider.calls == 2
    assert result.decision == Decision.PRIORITY_REVIEW


def test_at_most_two_calls_are_ever_made():
    provider = FakeProvider(
        [AgentTransportError("x")] * 10
    )  # would loop forever if retry were unbounded
    get_decision(_prediction(), "req_20", provider=provider)
    assert provider.calls == 2


def test_missing_api_key_falls_back_without_a_provider(monkeypatch):
    # No provider passed (get_decision would build a real AnthropicProvider);
    # force the credential to be absent so this can never make a real network
    # call, and confirm it falls back cleanly instead of raising.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("conversion_router.agent.client._ensure_dotenv_loaded", lambda: None)

    result = get_decision(_prediction(), "req_21", provider=None)

    assert result.decision == Decision.SYSTEM_FALLBACK
    assert result.reason_codes == [ReasonCode.AGENT_UNAVAILABLE]
    assert result.request_id == "req_21"


# ---------------------------------------------------------------------------
# Fallback payload shape
# ---------------------------------------------------------------------------


def test_fallback_response_passes_pydantic_schema_validation():
    provider = FakeProvider(["broken", "still broken"])
    result = get_decision(_prediction(), "req_22", provider=provider)
    # Round-tripping through the schema must succeed without raising.
    from conversion_router.schemas import DecisionResponse

    DecisionResponse.model_validate(json.loads(result.model_dump_json()))


# ---------------------------------------------------------------------------
# AnthropicProvider: request body shape (no real network call)
# ---------------------------------------------------------------------------


def test_anthropic_provider_missing_api_key_raises_credential_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "conversion_router.agent.client._ensure_dotenv_loaded", lambda: None
    )
    with pytest.raises(AgentCredentialError):
        AnthropicProvider(api_key=None, client=httpx.Client())


def test_anthropic_provider_sends_pinned_model_temperature_max_tokens_and_json_schema():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": VALID_CERTAIN_RESPONSE}],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-key", client=client)
    text = provider.complete_decision("system prompt text", "user content text")

    assert text == VALID_CERTAIN_RESPONSE
    body = captured["body"]
    assert body["model"] == ANTHROPIC_MODEL
    assert body["temperature"] == 0
    assert isinstance(body["max_tokens"], int) and body["max_tokens"] > 0
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "schema" in body["output_config"]["format"]
    assert captured["headers"]["x-api-key"] == "test-key"


def test_anthropic_provider_raises_transport_error_on_429():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-key", client=client)
    with pytest.raises(AgentTransportError):
        provider.complete_decision("sys", "user")


def test_anthropic_provider_raises_transport_error_on_500():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server error"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-key", client=client)
    with pytest.raises(AgentTransportError):
        provider.complete_decision("sys", "user")


def test_anthropic_provider_raises_transport_error_on_connect_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("simulated timeout", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-key", client=client)
    with pytest.raises(AgentTransportError):
        provider.complete_decision("sys", "user")


def test_anthropic_provider_raises_output_error_on_refusal():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"stop_reason": "refusal", "content": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-key", client=client)
    with pytest.raises(AgentOutputError):
        provider.complete_decision("sys", "user")


def test_anthropic_provider_raises_output_error_on_empty_content():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"stop_reason": "end_turn", "content": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-key", client=client)
    with pytest.raises(AgentOutputError):
        provider.complete_decision("sys", "user")


def test_get_decision_with_real_provider_and_mock_transport_falls_back_on_5xx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-key", client=client)
    result = get_decision(_prediction(), "req_23", provider=provider)
    assert result.decision == Decision.SYSTEM_FALLBACK
    assert result.reason_codes == [ReasonCode.AGENT_UNAVAILABLE]
