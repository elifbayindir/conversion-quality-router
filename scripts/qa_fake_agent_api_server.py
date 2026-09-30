#!/usr/bin/env python3
"""QA-only server for the T702 acceptance matrix.

Runs the REAL FastAPI app -- real model artifacts, real /route prediction,
real agent orchestration/retry/validation/fallback chain in
conversion_router.agent.client.get_decision -- with a deterministic,
non-LLM fake provider standing in for Anthropic, so the real n8n workflow
can be exercised end to end over a real HTTP webhook call without ever
making a real Anthropic network call.

Never used in production and never started by application code; only by
scripts/run_acceptance_matrix.py.

Usage:
    venv/bin/python scripts/qa_fake_agent_api_server.py --mode rule_based --port 8000
    venv/bin/python scripts/qa_fake_agent_api_server.py --mode invalid --port 8000
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request

import uvicorn

from conversion_router.api.app import create_app


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


PROVIDERS = {"rule_based": RuleBasedFakeProvider, "invalid": AlwaysInvalidProvider}


def _install_once_ready(app, provider, port: int) -> None:
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(100):
        try:
            with urllib.request.urlopen(url, timeout=0.5) as resp:  # noqa: S310 -- loopback only
                if resp.status == 200:
                    break
        except Exception:
            pass
        time.sleep(0.1)
    app.state.llm_provider = provider
    print("QA_FAKE_API_READY", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app()
    provider = PROVIDERS[args.mode]()
    threading.Thread(
        target=_install_once_ready, args=(app, provider, args.port), daemon=True
    ).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
