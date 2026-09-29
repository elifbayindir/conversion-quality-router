import json

import pytest
from pydantic import ValidationError

from conversion_router.data.validate import EXPECTED_COLUMNS
from conversion_router.schemas import (
    AllowedAction,
    Decision,
    DecisionResponse,
    ErrorDetail,
    ErrorResponse,
    ModelInfo,
    PredictedClass,
    PredictionDetail,
    PredictionResponse,
    Priority,
    ReasonCode,
    SessionFeaturesRequest,
    Signal,
)


def _valid_request_kwargs(**overrides) -> dict:
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
    return kwargs


def test_valid_request_parses():
    req = SessionFeaturesRequest(**_valid_request_kwargs())
    assert req.Month == "Feb"


def test_request_fields_cover_all_16_deployment_safe_features():
    fields = set(SessionFeaturesRequest.model_fields)
    raw_predictors = set(EXPECTED_COLUMNS) - {"Revenue"}
    assert fields == raw_predictors - {"PageValues"}


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        SessionFeaturesRequest(**_valid_request_kwargs(), UnexpectedField=1)


def test_negative_numeric_value_is_rejected():
    with pytest.raises(ValidationError):
        SessionFeaturesRequest(**_valid_request_kwargs(ProductRelated_Duration=-1.0))


def test_out_of_range_unit_interval_value_is_rejected():
    with pytest.raises(ValidationError):
        SessionFeaturesRequest(**_valid_request_kwargs(BounceRates=1.5))


def test_invalid_month_category_is_rejected():
    with pytest.raises(ValidationError):
        SessionFeaturesRequest(**_valid_request_kwargs(Month="Xxx"))


def test_invalid_visitor_type_is_rejected():
    with pytest.raises(ValidationError):
        SessionFeaturesRequest(**_valid_request_kwargs(VisitorType="Robot"))


def test_invalid_operating_systems_code_is_rejected():
    with pytest.raises(ValidationError):
        SessionFeaturesRequest(**_valid_request_kwargs(OperatingSystems=999))


def test_missing_required_field_is_rejected():
    kwargs = _valid_request_kwargs()
    del kwargs["Weekend"]
    with pytest.raises(ValidationError):
        SessionFeaturesRequest(**kwargs)


def test_prediction_response_round_trips_through_json():
    response = PredictionResponse(
        request_id="req_test_001",
        prediction=PredictionDetail(
            purchase_probability=0.73,
            decision_threshold=0.58,
            predicted_class=PredictedClass.LIKELY_TO_CONVERT,
            decision_margin=0.15,
            uncertain=False,
        ),
        signals=[Signal.RETURNING_VISITOR, Signal.LOW_EXIT_RATE],
        model=ModelInfo(name="conversion_mlp", version="1.0.0", feature_contract_version="1.0.0"),
    )
    payload = json.loads(response.model_dump_json())
    assert payload["prediction"]["predicted_class"] == "likely_to_convert"
    assert payload["signals"] == ["returning_visitor", "low_exit_rate"]


def test_prediction_response_rejects_unknown_top_level_field():
    with pytest.raises(ValidationError):
        PredictionResponse.model_validate(
            {
                "request_id": "req_test_002",
                "prediction": {
                    "purchase_probability": 0.5,
                    "decision_threshold": 0.5,
                    "predicted_class": "likely_to_convert",
                    "decision_margin": 0.0,
                    "uncertain": True,
                },
                "signals": [],
                "model": {
                    "name": "conversion_mlp",
                    "version": "1.0.0",
                    "feature_contract_version": "1.0.0",
                },
                "extra_field": "not allowed",
            }
        )


def test_prediction_response_rejects_invalid_predicted_class():
    with pytest.raises(ValidationError):
        PredictionResponse.model_validate(
            {
                "request_id": "req_test_003",
                "prediction": {
                    "purchase_probability": 0.5,
                    "decision_threshold": 0.5,
                    "predicted_class": "maybe",
                    "decision_margin": 0.0,
                    "uncertain": True,
                },
                "signals": [],
                "model": {
                    "name": "conversion_mlp",
                    "version": "1.0.0",
                    "feature_contract_version": "1.0.0",
                },
            }
        )


def test_decision_response_valid_payload():
    response = DecisionResponse(
        request_id="req_test_004",
        decision=Decision.HUMAN_REVIEW,
        priority=Priority.MEDIUM,
        allowed_action=AllowedAction.ADD_TO_REVIEW_QUEUE,
        requires_human_approval=True,
        reason_codes=[ReasonCode.MODEL_UNCERTAIN, ReasonCode.NEAR_DECISION_THRESHOLD],
        explanation="Prediction is near the decision threshold.",
        agent_version="1.0.0",
    )
    assert response.schema_version == "1.0.0"


def test_decision_response_rejects_invalid_decision_enum():
    with pytest.raises(ValidationError):
        DecisionResponse.model_validate(
            {
                "request_id": "req_test_005",
                "decision": "MAYBE_REVIEW",
                "priority": "MEDIUM",
                "allowed_action": "ADD_TO_REVIEW_QUEUE",
                "requires_human_approval": True,
                "reason_codes": ["MODEL_UNCERTAIN"],
                "explanation": "x",
                "agent_version": "1.0.0",
            }
        )


def test_decision_response_rejects_invented_reason_code():
    with pytest.raises(ValidationError):
        DecisionResponse.model_validate(
            {
                "request_id": "req_test_006",
                "decision": "HUMAN_REVIEW",
                "priority": "MEDIUM",
                "allowed_action": "ADD_TO_REVIEW_QUEUE",
                "requires_human_approval": True,
                "reason_codes": ["MODEL_IS_DEFINITELY_RIGHT"],
                "explanation": "x",
                "agent_version": "1.0.0",
            }
        )


def test_decision_response_rejects_oversized_explanation():
    with pytest.raises(ValidationError):
        DecisionResponse(
            request_id="req_test_007",
            decision=Decision.LOG_ONLY,
            priority=Priority.LOW,
            allowed_action=AllowedAction.ADD_TO_LOG,
            requires_human_approval=False,
            reason_codes=[ReasonCode.HIGH_CONFIDENCE_NEGATIVE],
            explanation="x" * 501,
            agent_version="1.0.0",
        )


def test_decision_response_requires_at_least_one_reason_code():
    with pytest.raises(ValidationError):
        DecisionResponse(
            request_id="req_test_008",
            decision=Decision.LOG_ONLY,
            priority=Priority.LOW,
            allowed_action=AllowedAction.ADD_TO_LOG,
            requires_human_approval=False,
            reason_codes=[],
            explanation="x",
            agent_version="1.0.0",
        )


def test_error_response_valid_payload():
    response = ErrorResponse(
        request_id="req_test_009",
        error=ErrorDetail(
            code="MODEL_UNAVAILABLE",
            message="Prediction service is temporarily unavailable.",
            retryable=True,
        ),
    )
    assert response.error.retryable is True


def test_error_response_rejects_invalid_code():
    with pytest.raises(ValidationError):
        ErrorResponse.model_validate(
            {
                "request_id": "req_test_010",
                "error": {
                    "code": "SOMETHING_WEIRD",
                    "message": "x",
                    "retryable": False,
                },
            }
        )


def test_error_response_rejects_unknown_field():
    with pytest.raises(ValidationError):
        ErrorResponse.model_validate(
            {
                "request_id": "req_test_011",
                "error": {"code": "MODEL_UNAVAILABLE", "message": "x", "retryable": False},
                "debug_info": "should not be here",
            }
        )
