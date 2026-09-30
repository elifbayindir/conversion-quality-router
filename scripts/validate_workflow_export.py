#!/usr/bin/env python3
"""Structural validation of automation/conversion_quality_router.json.

Checks that the exported workflow is credential-free, contains no secrets
or hardcoded webhook/API URLs, has the expected node set and Switch
branches, and is otherwise safe to publish. Does not require a running n8n
instance -- this is pure static JSON inspection, run as part of the P6
phase gate and importable by pytest.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = PROJECT_ROOT / "automation" / "conversion_quality_router.json"

REQUIRED_NODE_NAMES = {
    "POST Webhook",
    "Validate Envelope",
    "IF Envelope Valid",
    "Respond Invalid Envelope",
    "Duplicate Check",
    "IF Duplicate",
    "Respond Duplicate",
    "Call FastAPI Route",
    "IF Route Call Failed",
    "Build Automation Fallback (API Error)",
    "Validate Decision Response",
    "IF Decision Response Valid",
    "Build Automation Fallback (Invalid Response)",
    "Normalize Routing Record",
    "Routing Record Ready",
    "Switch on Decision",
    "Set Log Only Complete",
    "Respond Log Only",
    "Send Priority Notification",
    "Send Review Notification",
    "Send Fallback Notification",
    "Respond Pending Human Review",
    "Wait For Human Approval",
    "Evaluate Approval Result",
    "Send Final Outcome Notification",
}

SECRET_PATTERNS = [
    re.compile(r"hooks\.slack\.com/services/"),
    re.compile(r"discord\.com/api/webhooks/"),
    re.compile(r"sk-ant-[A-Za-z0-9\-_]+"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


def load_workflow() -> dict:
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


def check_no_secrets(raw_text: str) -> list[str]:
    errors = []
    for pattern in SECRET_PATTERNS:
        if pattern.search(raw_text):
            errors.append(f"possible secret/credential matched pattern: {pattern.pattern}")
    return errors


def check_no_credentials_block(workflow: dict) -> list[str]:
    errors = []
    for node in workflow.get("nodes", []):
        if node.get("credentials"):
            errors.append(f"node {node.get('name')!r} has an attached credential (must be none)")
    return errors


def check_required_nodes(workflow: dict) -> list[str]:
    names = {n.get("name") for n in workflow.get("nodes", [])}
    missing = REQUIRED_NODE_NAMES - names
    return [f"missing required node: {n}" for n in sorted(missing)]


def check_switch_branches(workflow: dict) -> list[str]:
    errors = []
    switch_nodes = [n for n in workflow["nodes"] if n["name"] == "Switch on Decision"]
    if not switch_nodes:
        return ["Switch on Decision node not found"]
    switch = switch_nodes[0]
    rules = switch.get("parameters", {}).get("rules", {}).get("values", [])
    expected = ["LOG_ONLY", "PRIORITY_REVIEW", "HUMAN_REVIEW", "SYSTEM_FALLBACK"]
    actual = [r.get("outputKey") for r in rules]
    if actual != expected:
        errors.append(f"Switch branches expected {expected}, got {actual}")
    fallback = switch.get("parameters", {}).get("options", {}).get("fallbackOutput")
    if fallback != "extra":
        errors.append(
            f"Switch fallbackOutput expected 'extra' (defensive 5th branch), got {fallback!r}"
        )
    return errors


def check_env_var_usage_not_hardcoded(workflow: dict) -> list[str]:
    errors = []
    text = json.dumps(workflow)
    if "$env.N8N_NOTIFY_WEBHOOK_URL" not in text:
        errors.append("expected notification nodes to reference $env.N8N_NOTIFY_WEBHOOK_URL")
    if "$env.FASTAPI_BASE_URL" not in text:
        errors.append("expected the API call node to reference $env.FASTAPI_BASE_URL")
    return errors


def check_no_execution_or_user_data(workflow: dict) -> list[str]:
    errors = []
    forbidden_keys = {"pinData"}
    for key in forbidden_keys:
        value = workflow.get(key)
        if value not in (None, {}, []):
            errors.append(f"unexpected non-empty {key!r} in export (should be empty)")
    return errors


def main() -> int:
    if not WORKFLOW_PATH.exists():
        print(f"FAIL: {WORKFLOW_PATH} does not exist")
        return 1

    raw_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = json.loads(raw_text)

    all_errors: list[str] = []
    all_errors += check_no_secrets(raw_text)
    all_errors += check_no_credentials_block(workflow)
    all_errors += check_required_nodes(workflow)
    all_errors += check_switch_branches(workflow)
    all_errors += check_env_var_usage_not_hardcoded(workflow)
    all_errors += check_no_execution_or_user_data(workflow)

    if all_errors:
        print(f"FAIL: {len(all_errors)} issue(s) found in {WORKFLOW_PATH.name}:")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print(
        f"PASS: {WORKFLOW_PATH.name} -- {len(workflow['nodes'])} nodes, "
        "no secrets, no credentials, all required nodes present, "
        "4 Switch branches + defensive fallback, env-var references confirmed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
