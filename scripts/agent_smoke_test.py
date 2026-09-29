#!/usr/bin/env python3
"""Controlled live test of the real Anthropic decision-agent call.

Disabled by default -- the normal test suite and this script both make zero
real network calls unless RUN_LIVE_AGENT_TEST=1 is set. When enabled, this
makes EXACTLY ONE real Anthropic API call (the provider is called directly,
bypassing the agent's own bounded-retry loop, so a single run can never cost
more than one call) and prints only a sanitized decision summary -- never
the API key, never the raw provider response.

Usage:
    RUN_LIVE_AGENT_TEST=1 venv/bin/python scripts/agent_smoke_test.py
"""

from __future__ import annotations

import json
import os
import sys

from pydantic import ValidationError

from conversion_router.agent.client import AnthropicProvider, build_user_content, load_system_prompt
from conversion_router.agent.validator import DecisionValidationError, validate_decision_core
from conversion_router.schemas import (
    LLMDecisionCore,
    ModelInfo,
    PredictedClass,
    PredictionDetail,
    PredictionResponse,
)


def _sample_prediction() -> PredictionResponse:
    return PredictionResponse(
        request_id="req_agent_smoke_test",
        prediction=PredictionDetail(
            purchase_probability=0.71,
            decision_threshold=0.5,
            predicted_class=PredictedClass.LIKELY_TO_CONVERT,
            decision_margin=0.21,
            uncertain=False,
        ),
        signals=["returning_visitor", "long_product_related_duration"],
        model=ModelInfo(name="conversion_mlp", version="1.0.0", feature_contract_version="1.0.0"),
    )


def main() -> int:
    if os.environ.get("RUN_LIVE_AGENT_TEST") != "1":
        print("RUN_LIVE_AGENT_TEST is not '1'; skipping live agent call (this is the default).")
        return 0

    print("RUN_LIVE_AGENT_TEST=1: making exactly one real Anthropic API call...")
    prediction = _sample_prediction()
    system_prompt = load_system_prompt()
    user_content = build_user_content(prediction)

    try:
        provider = AnthropicProvider()
    except Exception as exc:
        print(f"FAILED to construct provider ({type(exc).__name__}); check ANTHROPIC_API_KEY.")
        return 1

    try:
        raw_text = provider.complete_decision(system_prompt, user_content)
    except Exception as exc:
        print(f"FAILED: transport/output error ({type(exc).__name__}).")
        return 1

    try:
        parsed = json.loads(raw_text)
        core = LLMDecisionCore.model_validate(parsed)
        validate_decision_core(core, prediction)
    except (json.JSONDecodeError, ValidationError, DecisionValidationError) as exc:
        print(f"Call succeeded but output failed validation ({type(exc).__name__}).")
        return 1

    print("SUCCESS: one real Anthropic call returned a valid, policy-compliant decision.")
    print("Sanitized decision summary (no raw response, no credentials):")
    print(
        json.dumps(
            {
                "decision": core.decision.value,
                "priority": core.priority.value,
                "allowed_action": core.allowed_action.value,
                "requires_human_approval": core.requires_human_approval,
                "reason_codes": [rc.value for rc in core.reason_codes],
                "explanation": core.explanation,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
