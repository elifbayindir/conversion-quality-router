import json
import shutil

import pytest

from conversion_router.modeling.inference import (
    DEFAULT_CALIBRATOR_PATH,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_PREPROCESSOR_PATH,
    ArtifactIntegrityError,
    ArtifactUnavailableError,
    load_artifacts,
    predict_batch,
    predict_single,
)
from conversion_router.schemas import PredictedClass, SessionFeaturesRequest, Signal

pytestmark = pytest.mark.skipif(
    not DEFAULT_METADATA_PATH.exists(),
    reason="Frozen MLP artifacts not present; run the full P2-P4 pipeline scripts first.",
)


def make_request(**overrides) -> SessionFeaturesRequest:
    kwargs = {
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
    kwargs.update(overrides)
    return SessionFeaturesRequest(**kwargs)


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


def test_load_artifacts_succeeds_and_is_internally_consistent(artifacts):
    assert artifacts.metadata["model_name"] == "conversion_mlp"
    assert "threshold_selection" in artifacts.metadata
    assert "calibration" in artifacts.metadata


def test_load_artifacts_raises_on_missing_metadata(tmp_path):
    with pytest.raises(ArtifactUnavailableError):
        load_artifacts(metadata_path=tmp_path / "does_not_exist.json")


def test_load_artifacts_raises_on_hash_mismatch(tmp_path):
    tmp_checkpoint = tmp_path / "mlp_checkpoint.pt"
    tmp_preprocessor = tmp_path / "preprocessor.joblib"
    tmp_calibrator = tmp_path / "calibrator.joblib"
    shutil.copy(DEFAULT_CHECKPOINT_PATH, tmp_checkpoint)
    shutil.copy(DEFAULT_PREPROCESSOR_PATH, tmp_preprocessor)
    shutil.copy(DEFAULT_CALIBRATOR_PATH, tmp_calibrator)

    metadata = json.loads(DEFAULT_METADATA_PATH.read_text(encoding="utf-8"))
    metadata["artifact_hashes"]["mlp_checkpoint_sha256"] = "0" * 64  # deliberately wrong
    tmp_metadata = tmp_path / "mlp_metadata.json"
    tmp_metadata.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError):
        load_artifacts(
            metadata_path=tmp_metadata,
            checkpoint_path=tmp_checkpoint,
            preprocessor_path=tmp_preprocessor,
            calibrator_path=tmp_calibrator,
        )


def test_predict_single_is_reproducible(artifacts):
    request = make_request()
    first = predict_single(artifacts, request, "req_test_a")
    second = predict_single(artifacts, request, "req_test_b")
    assert first.prediction.purchase_probability == second.prediction.purchase_probability
    assert first.prediction.decision_margin == second.prediction.decision_margin
    assert first.prediction.uncertain == second.prediction.uncertain


def test_predict_batch_matches_predict_single(artifacts):
    request_a = make_request(ProductRelated_Duration=5000.0, BounceRates=0.0)
    request_b = make_request(ProductRelated_Duration=0.0, BounceRates=0.2)

    batch_results = predict_batch(artifacts, [request_a, request_b], ["req_a", "req_b"])
    single_a = predict_single(artifacts, request_a, "req_a")
    single_b = predict_single(artifacts, request_b, "req_b")

    assert (
        batch_results[0].prediction.purchase_probability
        == single_a.prediction.purchase_probability
    )
    assert (
        batch_results[1].prediction.purchase_probability
        == single_b.prediction.purchase_probability
    )


def test_predict_batch_empty_returns_empty(artifacts):
    assert predict_batch(artifacts, [], []) == []


def test_predict_batch_mismatched_lengths_raises(artifacts):
    request = make_request()
    with pytest.raises(ValueError, match="same length"):
        predict_batch(artifacts, [request], ["req_1", "req_2"])


def test_prediction_fields_are_internally_consistent(artifacts):
    request = make_request(ProductRelated_Duration=3000.0)
    response = predict_single(artifacts, request, "req_consistency")

    p = response.prediction
    expected_class = (
        PredictedClass.LIKELY_TO_CONVERT
        if p.purchase_probability >= p.decision_threshold
        else PredictedClass.UNLIKELY_TO_CONVERT
    )
    assert p.predicted_class == expected_class
    assert p.decision_margin == pytest.approx(
        abs(p.purchase_probability - p.decision_threshold), abs=1e-6
    )

    band = artifacts.metadata["calibration"]["uncertainty_band"]
    assert p.uncertain == (p.decision_margin < band)


def test_signals_are_always_within_the_allowed_enum(artifacts):
    request = make_request(
        VisitorType="New_Visitor", Weekend=True, SpecialDay=0.6,
        Administrative=10, Informational=5,
    )
    response = predict_single(artifacts, request, "req_signals")
    assert all(isinstance(s, Signal) for s in response.signals)
    assert Signal.NEW_VISITOR in response.signals
    assert Signal.WEEKEND_SESSION in response.signals
    assert Signal.NEAR_SPECIAL_DAY in response.signals
    assert Signal.HIGH_ADMINISTRATIVE_ENGAGEMENT in response.signals
    assert Signal.HIGH_INFORMATIONAL_ENGAGEMENT in response.signals


def test_model_info_matches_frozen_metadata(artifacts):
    request = make_request()
    response = predict_single(artifacts, request, "req_model_info")
    assert response.model.name == artifacts.metadata["model_name"]
    assert response.model.version == artifacts.metadata["model_version"]
    assert (
        response.model.feature_contract_version == artifacts.metadata["feature_contract_version"]
    )
