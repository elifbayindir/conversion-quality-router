import json

import httpx
import pytest

from conversion_router.schemas import DecisionResponse, PredictionResponse, SessionFeaturesRequest
from conversion_router.ui.api_client import ApiClient, ApiError, ApiErrorKind

SESSION = SessionFeaturesRequest.model_validate(
    {
        "Administrative": 0, "Administrative_Duration": 0.0,
        "Informational": 0, "Informational_Duration": 0.0,
        "ProductRelated": 3, "ProductRelated_Duration": 60.0,
        "BounceRates": 0.01, "ExitRates": 0.02, "SpecialDay": 0.0,
        "Month": "Nov", "OperatingSystems": 2, "Browser": 2, "Region": 1,
        "TrafficType": 2, "VisitorType": "Returning_Visitor", "Weekend": False,
    }
)

PREDICTION = {
    "request_id": "req_ui_client",
    "prediction": {
        "purchase_probability": 0.4,
        "decision_threshold": 0.12,
        "predicted_class": "likely_to_convert",
        "decision_margin": 0.28,
        "uncertain": False,
    },
    "signals": ["returning_visitor"],
    "model": {"name": "conversion_mlp", "version": "1.0.0", "feature_contract_version": "1.0.0"},
}


def _decision(request_id: str) -> dict:
    return {
        "request_id": request_id,
        "decision": "PRIORITY_REVIEW",
        "priority": "HIGH",
        "allowed_action": "ADD_TO_REVIEW_QUEUE_PRIORITY",
        "requires_human_approval": False,
        "reason_codes": ["HIGH_CONFIDENCE_POSITIVE"],
        "explanation": "Strong signal.",
        "agent_version": "1.0.0",
        "schema_version": "1.0.0",
    }


def _client(handler) -> ApiClient:
    return ApiClient(
        http_client=httpx.Client(
            base_url="http://127.0.0.1:8000", transport=httpx.MockTransport(handler)
        )
    )


def test_route_chains_predict_then_decide_with_the_same_prediction():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/predict":
            return httpx.Response(200, json=PREDICTION)
        return httpx.Response(200, json=_decision(PREDICTION["request_id"]))

    result = _client(handler).route(SESSION)

    assert [path for path, _ in seen] == ["/predict", "/decide"]
    assert seen[0][1] == SESSION.model_dump(mode="json")
    # /decide receives exactly the prediction /predict returned -- no second model run.
    assert seen[1][1] == PREDICTION
    assert isinstance(result.prediction, PredictionResponse)
    assert isinstance(result.decision, DecisionResponse)
    assert result.request_id == result.decision.request_id == "req_ui_client"


def test_route_rejects_a_decision_for_a_different_request():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/predict":
            return httpx.Response(200, json=PREDICTION)
        return httpx.Response(200, json=_decision("req_someone_else"))

    with pytest.raises(ApiError) as excinfo:
        _client(handler).route(SESSION)
    assert excinfo.value.kind == ApiErrorKind.CONTRACT_VIOLATION


def test_model_only_predict_never_calls_decide():
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json=PREDICTION)

    prediction = _client(handler).predict(SESSION)
    assert paths == ["/predict"]
    assert prediction.prediction.purchase_probability == 0.4


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (httpx.Response(422, json={"detail": []}), ApiErrorKind.INVALID_INPUT),
        (
            httpx.Response(
                503,
                json={
                    "request_id": "req_x",
                    "error": {"code": "MODEL_UNAVAILABLE", "message": "missing",
                              "retryable": True},
                },
            ),
            ApiErrorKind.MODEL_UNAVAILABLE,
        ),
        (httpx.Response(500, text="boom"), ApiErrorKind.HTTP_ERROR),
        (httpx.Response(200, json={"unexpected": True}), ApiErrorKind.CONTRACT_VIOLATION),
    ],
)
def test_error_responses_map_to_typed_errors(response, kind):
    with pytest.raises(ApiError) as excinfo:
        _client(lambda request: response).predict(SESSION)
    assert excinfo.value.kind == kind


def test_model_unavailable_keeps_the_service_request_id():
    response = httpx.Response(
        503,
        json={"request_id": "req_503", "error": {"code": "MODEL_UNAVAILABLE",
                                                 "message": "missing", "retryable": True}},
    )
    with pytest.raises(ApiError) as excinfo:
        _client(lambda request: response).predict(SESSION)
    assert excinfo.value.request_id == "req_503"


def test_timeout_maps_to_timeout_error():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ApiError) as excinfo:
        _client(handler).predict(SESSION)
    assert excinfo.value.kind == ApiErrorKind.TIMEOUT


def test_connection_refused_maps_to_unreachable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ApiError) as excinfo:
        _client(handler).ready()
    assert excinfo.value.kind == ApiErrorKind.UNREACHABLE


def test_ready_parses_model_versions():
    body = {"status": "ready", "model_name": "conversion_mlp", "model_version": "1.0.0",
            "feature_contract_version": "1.0.0"}
    info = _client(lambda request: httpx.Response(200, json=body)).ready()
    assert (info.model_name, info.model_version) == ("conversion_mlp", "1.0.0")


def test_default_base_url_is_loopback(monkeypatch):
    monkeypatch.delenv("CQR_API_BASE_URL", raising=False)
    assert ApiClient.from_env().base_url == "http://127.0.0.1:8000"
