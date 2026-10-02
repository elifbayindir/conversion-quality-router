"""Deterministic, non-LLM providers for demos and tests only.

These implement the `LLMProvider` protocol from `conversion_router.agent.client`
without any network call, so the REAL agent chain (schema validation,
cross-field validation, bounded retry, deterministic fallback) runs end to end
while no external provider is contacted.

They are never a production default: `get_decision()` builds an
`AnthropicProvider` when no provider is passed, and nothing in the application
imports this module. A provider from here is used only when explicitly injected
(for example `app.state.llm_provider = RuleBasedFakeProvider()` in
`scripts/qa_fake_agent_api_server.py`, or `get_decision(..., provider=...)`).
"""

from __future__ import annotations

import json


class RuleBasedFakeProvider:
    """Deterministically reproduces the decision policy from
    prompts/decision_agent_v1.md given the REAL PredictionResponse it
    receives. No network call, no randomness, no LLM involved."""

    def complete_decision(self, system_prompt: str, user_content: str) -> str:
        prediction = json.loads(user_content)["prediction"]
        if prediction["uncertain"]:
            body = {
                "decision": "HUMAN_REVIEW",
                "priority": "MEDIUM",
                "allowed_action": "ADD_TO_REVIEW_QUEUE",
                "requires_human_approval": True,
                "reason_codes": ["MODEL_UNCERTAIN"],
                "explanation": "Prediction is within the uncertainty band of the threshold.",
            }
        elif prediction["predicted_class"] == "likely_to_convert":
            body = {
                "decision": "PRIORITY_REVIEW",
                "priority": "HIGH",
                "allowed_action": "ADD_TO_REVIEW_QUEUE_PRIORITY",
                "requires_human_approval": False,
                "reason_codes": ["HIGH_CONFIDENCE_POSITIVE"],
                "explanation": "Purchase probability is comfortably above the decision threshold.",
            }
        else:
            body = {
                "decision": "LOG_ONLY",
                "priority": "LOW",
                "allowed_action": "ADD_TO_LOG",
                "requires_human_approval": False,
                "reason_codes": ["HIGH_CONFIDENCE_NEGATIVE"],
                "explanation": "Purchase probability is comfortably below the decision threshold.",
            }
        return json.dumps(body)


class AlwaysInvalidProvider:
    """Deterministically returns unusable content on every attempt, so the
    real agent client's bounded-retry-then-fallback chain runs for real."""

    def complete_decision(self, system_prompt: str, user_content: str) -> str:
        return "this is not valid JSON output"


DEMO_PROVIDERS = {"rule_based": RuleBasedFakeProvider, "invalid": AlwaysInvalidProvider}
