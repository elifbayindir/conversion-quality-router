#!/usr/bin/env python3
"""Local mock of the FastAPI /route endpoint, for n8n workflow testing only.

This is never used by the real API or the real agent -- it exists so the
n8n workflow's four Switch branches (and its error paths) can be exercised
repeatedly without calling the real Anthropic-backed /route endpoint. Which
canned DecisionResponse (or failure) is returned is selected by the
request's "TrafficType" field, by convention documented in
docs/n8n-workflow.md and automation/fixtures/.

Usage:
    venv/bin/python scripts/mock_route_server.py [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIXTURES = {
    1: {  # LOG_ONLY
        "request_id": "req_mock_log_only",
        "decision": "LOG_ONLY",
        "priority": "LOW",
        "allowed_action": "ADD_TO_LOG",
        "requires_human_approval": False,
        "reason_codes": ["HIGH_CONFIDENCE_NEGATIVE"],
        "explanation": "Low predicted conversion probability with a comfortable margin.",
        "agent_version": "1.0.0",
        "schema_version": "1.0.0",
    },
    2: {  # PRIORITY_REVIEW, no approval required
        "request_id": "req_mock_priority_no_approval",
        "decision": "PRIORITY_REVIEW",
        "priority": "HIGH",
        "allowed_action": "ADD_TO_REVIEW_QUEUE_PRIORITY",
        "requires_human_approval": False,
        "reason_codes": ["HIGH_CONFIDENCE_POSITIVE"],
        "explanation": "Strong, comfortably-above-threshold conversion signal.",
        "agent_version": "1.0.0",
        "schema_version": "1.0.0",
    },
    3: {  # HUMAN_REVIEW
        "request_id": "req_mock_human_review",
        "decision": "HUMAN_REVIEW",
        "priority": "MEDIUM",
        "allowed_action": "ADD_TO_REVIEW_QUEUE",
        "requires_human_approval": True,
        "reason_codes": ["MODEL_UNCERTAIN"],
        "explanation": "Prediction is near the decision threshold.",
        "agent_version": "1.0.0",
        "schema_version": "1.0.0",
    },
    4: {  # SYSTEM_FALLBACK (as returned by the real API's own agent fallback)
        "request_id": "req_mock_system_fallback",
        "decision": "SYSTEM_FALLBACK",
        "priority": "HIGH",
        "allowed_action": "ADD_TO_REVIEW_QUEUE",
        "requires_human_approval": True,
        "reason_codes": ["AGENT_UNAVAILABLE"],
        "explanation": "Agent output could not be validated; routed to human review.",
        "agent_version": "1.0.0",
        "schema_version": "1.0.0",
    },
    5: {  # Invalid payload (missing required fields) -- for n8n's own response validation
        "request_id": "req_mock_invalid_payload",
        "decision": "LOG_ONLY",
        # priority intentionally missing
        "allowed_action": "ADD_TO_LOG",
    },
    6: "HTTP_422",  # simulate a FastAPI validation error
    7: "HTTP_503",  # simulate model-unavailable
    8: "TIMEOUT",  # simulate a slow/hanging response beyond the workflow's timeout
    10: {  # PRIORITY_REVIEW with human approval required
        "request_id": "req_mock_priority_with_approval",
        "decision": "PRIORITY_REVIEW",
        "priority": "HIGH",
        "allowed_action": "ADD_TO_REVIEW_QUEUE_PRIORITY",
        "requires_human_approval": True,
        "reason_codes": ["HIGH_CONFIDENCE_POSITIVE", "NEAR_DECISION_THRESHOLD"],
        "explanation": "Strong signal but close enough to warrant a quick human check.",
        "agent_version": "1.0.0",
        "schema_version": "1.0.0",
    },
    9: {  # inconsistent decision/action combo -- must fail n8n's cross-field check
        "request_id": "req_mock_inconsistent",
        "decision": "LOG_ONLY",
        "priority": "LOW",
        "allowed_action": "ADD_TO_REVIEW_QUEUE",
        "requires_human_approval": False,
        "reason_codes": ["HIGH_CONFIDENCE_NEGATIVE"],
        "explanation": "Inconsistent on purpose for adversarial testing.",
        "agent_version": "1.0.0",
        "schema_version": "1.0.0",
    },
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        print(f"[mock_route_server] {self.address_string()} - {fmt % args}")

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler naming convention)
        if self.path != "/route":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            body = {}

        selector = body.get("TrafficType", 1)
        fixture = FIXTURES.get(selector, FIXTURES[1])

        if fixture == "TIMEOUT":
            time.sleep(20)
            fixture = FIXTURES[1]
        if fixture == "HTTP_422":
            self._send_json(422, {"detail": "simulated validation error"})
            return
        if fixture == "HTTP_503":
            self._send_json(
                503,
                {
                    "request_id": "req_mock_unavailable",
                    "error": {
                        "code": "MODEL_UNAVAILABLE",
                        "message": "simulated model unavailable",
                        "retryable": True,
                    },
                },
            )
            return

        self._send_json(200, fixture)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Mock /route server listening on http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
