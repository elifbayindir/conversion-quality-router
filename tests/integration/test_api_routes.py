import pytest
from fastapi.testclient import TestClient

from conversion_router.api.app import create_app
from conversion_router.modeling.inference import DEFAULT_METADATA_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_METADATA_PATH.exists(),
    reason="Frozen MLP artifacts not present; run the full P2-P4 pipeline scripts first.",
)


def _valid_payload(**overrides) -> dict:
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


def test_health_returns_ok_and_never_touches_artifacts(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_200_when_artifacts_loaded(client):
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model_name"] == "conversion_mlp"


def test_ready_returns_503_when_artifacts_missing(monkeypatch):
    app = create_app()
    with TestClient(app) as test_client:
        # Simulate a load failure discovered at startup without touching real artifacts.
        test_client.app.state.artifact_state.artifacts = None
        test_client.app.state.artifact_state.load_error = RuntimeError("simulated failure")

        response = test_client.get("/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["error"]["code"] == "MODEL_UNAVAILABLE"


def test_predict_returns_valid_schema_for_valid_payload(client):
    response = client.post("/predict", json=_valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["prediction"]["purchase_probability"] <= 1.0
    assert body["prediction"]["predicted_class"] in ("likely_to_convert", "unlikely_to_convert")
    assert body["model"]["name"] == "conversion_mlp"
    assert body["request_id"].startswith("req_")


def test_predict_rejects_unknown_field(client):
    response = client.post("/predict", json=_valid_payload(UnexpectedField=1))
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    payload = _valid_payload()
    del payload["Weekend"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_out_of_range_value(client):
    response = client.post("/predict", json=_valid_payload(BounceRates=5.0))
    assert response.status_code == 422


def test_predict_rejects_invalid_category(client):
    response = client.post("/predict", json=_valid_payload(Month="Notamonth"))
    assert response.status_code == 422


def test_predict_returns_503_when_model_unavailable(client):
    client.app.state.artifact_state.artifacts = None
    client.app.state.artifact_state.load_error = RuntimeError("simulated failure")

    response = client.post("/predict", json=_valid_payload())
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "MODEL_UNAVAILABLE"
    assert body["error"]["retryable"] is True
