"""Shared builders for feedback tests (real contract objects, no network)."""

from conversion_router.schemas import DecisionResponse, PredictionResponse, SessionFeaturesRequest

SESSION = SessionFeaturesRequest.model_validate(
    {
        "Administrative": 2, "Administrative_Duration": 2.0,
        "Informational": 3, "Informational_Duration": 261.0,
        "ProductRelated": 10, "ProductRelated_Duration": 281.0,
        "BounceRates": 0.0, "ExitRates": 0.042857143, "SpecialDay": 0.0,
        "Month": "Mar", "OperatingSystems": 2, "Browser": 2, "Region": 1,
        "TrafficType": 2, "VisitorType": "Returning_Visitor", "Weekend": False,
    }
)


def prediction(request_id="req_fb", uncertain=True, proba=0.1111) -> PredictionResponse:
    return PredictionResponse.model_validate(
        {
            "request_id": request_id,
            "prediction": {
                "purchase_probability": proba,
                "decision_threshold": 0.12,
                "predicted_class": "unlikely_to_convert",
                "decision_margin": 0.0089,
                "uncertain": uncertain,
            },
            "signals": ["returning_visitor", "low_bounce_rate"],
            "model": {"name": "conversion_mlp", "version": "1.0.0",
                      "feature_contract_version": "1.0.0"},
        }
    )


def decision(request_id="req_fb", value="HUMAN_REVIEW") -> DecisionResponse:
    actions = {
        "HUMAN_REVIEW": ("MEDIUM", "ADD_TO_REVIEW_QUEUE", True, "MODEL_UNCERTAIN"),
        "PRIORITY_REVIEW": ("HIGH", "ADD_TO_REVIEW_QUEUE_PRIORITY", False,
                            "HIGH_CONFIDENCE_POSITIVE"),
        "SYSTEM_FALLBACK": ("HIGH", "ADD_TO_REVIEW_QUEUE", True, "AGENT_OUTPUT_INVALID"),
    }
    priority, action, approval, reason = actions[value]
    return DecisionResponse(
        request_id=request_id, decision=value, priority=priority, allowed_action=action,
        requires_human_approval=approval, reason_codes=[reason],
        explanation="Test explanation.", agent_version="1.0.0",
    )
