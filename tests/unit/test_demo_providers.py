import ast
import json
from pathlib import Path

from conversion_router.agent.client import get_decision
from conversion_router.agent.demo_providers import (
    DEMO_PROVIDERS,
    AlwaysInvalidProvider,
    RuleBasedFakeProvider,
)
from conversion_router.schemas import Decision, PredictionResponse, ReasonCode

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "conversion_router"


def _prediction(proba: float, uncertain: bool) -> PredictionResponse:
    return PredictionResponse.model_validate(
        {
            "request_id": "req_demo_provider",
            "prediction": {
                "purchase_probability": proba,
                "decision_threshold": 0.12,
                "predicted_class": "likely_to_convert" if proba >= 0.12 else "unlikely_to_convert",
                "decision_margin": abs(proba - 0.12),
                "uncertain": uncertain,
            },
            "signals": [],
            "model": {"name": "conversion_mlp", "version": "1.0.0",
                      "feature_contract_version": "1.0.0"},
        }
    )


def test_rule_based_provider_passes_the_real_agent_chain_for_each_route():
    provider = RuleBasedFakeProvider()
    cases = [
        (_prediction(0.9, False), Decision.PRIORITY_REVIEW),
        (_prediction(0.11, True), Decision.HUMAN_REVIEW),
        (_prediction(0.01, False), Decision.LOG_ONLY),
    ]
    for prediction, expected in cases:
        decision = get_decision(prediction, prediction.request_id, provider=provider)
        assert decision.decision == expected


def test_rule_based_provider_output_is_deterministic():
    content = json.dumps(_prediction(0.11, True).model_dump(mode="json"))
    provider = RuleBasedFakeProvider()
    assert provider.complete_decision("", content) == provider.complete_decision("", content)


def test_invalid_provider_triggers_the_real_fallback():
    prediction = _prediction(0.9, False)
    decision = get_decision(prediction, prediction.request_id, provider=AlwaysInvalidProvider())
    assert decision.decision == Decision.SYSTEM_FALLBACK
    assert decision.reason_codes == [ReasonCode.AGENT_OUTPUT_INVALID]


def test_demo_provider_registry_matches_qa_server_modes():
    assert set(DEMO_PROVIDERS) == {"rule_based", "invalid"}


def test_no_application_module_imports_demo_providers():
    """Demo providers are injection-only: never a production default."""
    offenders = []
    for path in SRC_ROOT.rglob("*.py"):
        if path.name == "demo_providers.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "demo_providers" in node.module:
                offenders.append(str(path))
            if isinstance(node, ast.Import) and any(
                "demo_providers" in alias.name for alias in node.names
            ):
                offenders.append(str(path))
    assert offenders == []
