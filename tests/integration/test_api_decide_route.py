import json

import pytest
from fastapi.testclient import TestClient

from conversion_router.api.app import create_app
from conversion_router.modeling.inference import DEFAULT_METADATA_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_METADATA_PATH.exists(),
    reason="Frozen MLP artifacts not present; run the full P2-P4 pipeline scripts first.",
)

VALID_CERTAIN_RESPONSE = json.dumps(
    {
        "decision": "PRIORITY_REVIEW",
        "priority": "HIGH",
        "allowed_action": "ADD_TO_REVIEW_QUEUE_PRIORITY",
        "requires_human_approval": False,
        "reason_codes": ["HIGH_CONFIDENCE_POSITIVE"],
        "explanation": "Strong signal.",
    }
)


class FakeProvider:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def complete_decision(self, system_prompt: str, user_content: str) -> str:
        self.calls += 1
        item = self._responses[min(self.calls - 1, len(self._responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def _valid_prediction_payload(**overrides) -> dict:
    payload = {
        "request_id": "req_decide_test",
        "prediction": {
            "purchase_probability": 0.7,
            "decision_threshold": 0.5,
            "predicted_class": "likely_to_convert",
            "decision_margin": 0.2,
            "uncertain": False,
        },
        "signals": ["returning_visitor"],
        "model": {
            "name": "conversion_mlp",
            "version": "1.0.0",
            "feature_contract_version": "1.0.0",
        },
    }
    payload.update(overrides)
    return payload


def _valid_session_payload(**overrides) -> dict:
    payload = {
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 1,
        "ProductRelated_Duration": 10.0,
        "BounceRates": 0.1,
        "ExitRates": 0.1,
        "SpecialDay": 0.0,
        "Month": "Feb",
        "OperatingSystems": 1,
        "Browser": 1,
        "Region": 1,
        "TrafficType": 1,
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }
    payload.update(overrides)
    return payload


@pytest.fixture()
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def _set_provider(client: TestClient, provider) -> None:
    client.app.state.llm_provider = provider


# ---------------------------------------------------------------------------
# /decide
# ---------------------------------------------------------------------------


def test_decide_returns_200_with_valid_decision_on_success(client):
    _set_provider(client, FakeProvider([VALID_CERTAIN_RESPONSE]))
    response = client.post("/decide", json=_valid_prediction_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "PRIORITY_REVIEW"
    assert body["request_id"] == "req_decide_test"


def test_decide_rejects_unknown_field_in_request(client):
    response = client.post("/decide", json={**_valid_prediction_payload(), "extra": 1})
    assert response.status_code == 422


def test_decide_malformed_llm_json_falls_back_to_200_system_fallback(client):
    _set_provider(client, FakeProvider(["not json", "still not json"]))
    response = client.post("/decide", json=_valid_prediction_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "SYSTEM_FALLBACK"
    assert body["requires_human_approval"] is True


def test_decide_contradictory_llm_decision_action_falls_back(client):
    bad = json.dumps({**json.loads(VALID_CERTAIN_RESPONSE), "decision": "LOG_ONLY"})
    _set_provider(client, FakeProvider([bad, bad]))
    response = client.post("/decide", json=_valid_prediction_payload())
    assert response.status_code == 200
    assert response.json()["decision"] == "SYSTEM_FALLBACK"


def test_decide_uncertain_prediction_with_llm_log_only_falls_back(client):
    bad = json.dumps(
        {
            "decision": "LOG_ONLY",
            "priority": "LOW",
            "allowed_action": "ADD_TO_LOG",
            "requires_human_approval": False,
            "reason_codes": ["HIGH_CONFIDENCE_NEGATIVE"],
            "explanation": "Looks fine.",
        }
    )
    _set_provider(client, FakeProvider([bad, bad]))
    payload = _valid_prediction_payload()
    payload["prediction"]["uncertain"] = True
    response = client.post("/decide", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "SYSTEM_FALLBACK"
    assert body["requires_human_approval"] is True


def test_decide_llm_attempting_system_fallback_is_rejected(client):
    bad = json.dumps({**json.loads(VALID_CERTAIN_RESPONSE), "decision": "SYSTEM_FALLBACK"})
    _set_provider(client, FakeProvider([bad, bad]))
    response = client.post("/decide", json=_valid_prediction_payload())
    assert response.status_code == 200
    assert response.json()["decision"] == "SYSTEM_FALLBACK"


def test_decide_llm_attempting_to_tamper_immutable_fields_is_ignored(client):
    tampered = json.dumps(
        {
            **json.loads(VALID_CERTAIN_RESPONSE),
            "request_id": "req_hijacked",
            "agent_version": "9.9.9",
        }
    )
    _set_provider(client, FakeProvider([tampered, tampered]))
    response = client.post("/decide", json=_valid_prediction_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "SYSTEM_FALLBACK"
    assert body["request_id"] == "req_decide_test"


def test_decide_provider_unavailable_returns_safe_fallback_not_500(client):
    class BrokenProvider:
        def complete_decision(self, system_prompt, user_content):
            raise RuntimeError("boom -- totally unexpected internal failure")

    _set_provider(client, BrokenProvider())
    response = client.post("/decide", json=_valid_prediction_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "SYSTEM_FALLBACK"


def test_decide_fallback_response_matches_decision_response_schema(client):
    _set_provider(client, FakeProvider(["broken", "broken"]))
    response = client.post("/decide", json=_valid_prediction_payload())
    body = response.json()
    for key in (
        "request_id", "decision", "priority", "allowed_action",
        "requires_human_approval", "reason_codes", "explanation",
        "agent_version", "schema_version",
    ):
        assert key in body


# ---------------------------------------------------------------------------
# /route
# ---------------------------------------------------------------------------


def test_route_returns_200_with_decision_for_valid_session(client):
    _set_provider(client, FakeProvider([VALID_CERTAIN_RESPONSE]))
    response = client.post("/route", json=_valid_session_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] in ("LOG_ONLY", "PRIORITY_REVIEW", "HUMAN_REVIEW", "SYSTEM_FALLBACK")
    assert body["request_id"].startswith("req_")


def test_route_rejects_invalid_session_payload(client):
    response = client.post("/route", json={**_valid_session_payload(), "Month": "Notamonth"})
    assert response.status_code == 422


def test_route_agent_failure_still_returns_200_fallback(client):
    _set_provider(client, FakeProvider(["not json", "not json"]))
    response = client.post("/route", json=_valid_session_payload())
    assert response.status_code == 200
    assert response.json()["decision"] == "SYSTEM_FALLBACK"


def test_route_model_unavailable_returns_503_and_never_calls_agent(client):
    calls = {"n": 0}

    class ShouldNotBeCalledProvider:
        def complete_decision(self, system_prompt, user_content):
            calls["n"] += 1
            return VALID_CERTAIN_RESPONSE

    _set_provider(client, ShouldNotBeCalledProvider())
    client.app.state.artifact_state.artifacts = None
    client.app.state.artifact_state.load_error = RuntimeError("simulated model failure")

    response = client.post("/route", json=_valid_session_payload())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"
    assert calls["n"] == 0


def test_predict_still_works_unaffected_by_agent_wiring(client):
    response = client.post("/predict", json=_valid_session_payload())
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_health_and_ready_still_work_unaffected_by_agent_wiring(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
