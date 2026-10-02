#!/usr/bin/env python3
"""Safe fixture-posting helper for demo recording (T803).

Posts a fixture's `input` envelope (never the fixture's own
description/expected_* metadata -- the webhook only accepts the envelope
itself) to the real, running n8n workflow's production webhook, and prints
only sanitized fields. If the response is PENDING_HUMAN_REVIEW, it
automatically resolves the approval by reading the resume URL from the
local mock notification log -- internally only, never printed -- and POSTs
an "approve" action, then prints just the final sanitized outcome.

Never prints: the approval_resume_url, any secret, any raw provider
response, or the webhook URL's full value beyond the fixed loopback address
this script itself hardcodes. Never modifies the fixture file on disk.

Usage:
    venv/bin/python scripts/demo_post_fixture.py automation/fixtures/real_model_log_only.json
    venv/bin/python scripts/demo_post_fixture.py \
        automation/fixtures/real_model_human_review.json --no-auto-approve
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALLOWED_WEBHOOK_URL = "http://127.0.0.1:5678/webhook/conversion-quality-router"
DEFAULT_NOTIFICATIONS_LOG = PROJECT_ROOT / "logs" / "demo-runtime" / "demo_notifications.jsonl"

SANITIZED_FIELDS = (
    "status",
    "decision",
    "priority",
    "notification_status",
    "approval_status",
    "workflow_request_id",
    "model_request_id",
)


def _post_json(url: str, body: dict, timeout: float = 30) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- loopback only
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"_raw": raw}
        return exc.code, payload


def _sanitized(body: dict) -> dict:
    return {key: body.get(key) for key in SANITIZED_FIELDS}


def _print_sanitized(label: str, body: dict) -> None:
    print(f"{label}:")
    for key, value in _sanitized(body).items():
        print(f"  {key}: {value}")


def _last_notification_for(workflow_request_id: str, log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("workflow_request_id") == workflow_request_id:
            return payload
    return None


def _auto_approve(workflow_request_id: str, log_path: Path) -> str:
    """Resolve a pending approval internally; return only a final status word.

    The resume URL is read from the local mock notification log and used
    immediately -- it is never printed or returned to the caller.
    """
    deadline = time.monotonic() + 15
    notif = None
    while time.monotonic() < deadline:
        notif = _last_notification_for(workflow_request_id, log_path)
        if notif and notif.get("approval_resume_url"):
            break
        time.sleep(0.3)
    if not notif or not notif.get("approval_resume_url"):
        return "UNRESOLVED (no resume URL observed in the local mock notification log)"

    resume_url = notif["approval_resume_url"]
    status = None
    for _ in range(10):
        status, _body = _post_json(resume_url, {"action": "approve", "note": "demo auto-approve"})
        if status != 409:
            break
        time.sleep(0.5)
    return "APPROVED" if status == 200 else f"UNRESOLVED (resume call returned HTTP {status})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture_path", type=Path)
    parser.add_argument(
        "--notifications-log",
        type=Path,
        default=DEFAULT_NOTIFICATIONS_LOG,
        help="Local mock notification receiver's log file (never a real Slack/Discord log).",
    )
    parser.add_argument(
        "--no-auto-approve",
        action="store_true",
        help="Do not automatically resolve a PENDING_HUMAN_REVIEW approval.",
    )
    args = parser.parse_args()

    fixture = json.loads(args.fixture_path.read_text(encoding="utf-8"))
    envelope = json.loads(json.dumps(fixture["input"]))  # deep copy; never mutates the fixture file
    suffix = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    envelope["workflow_request_id"] = f"{envelope['workflow_request_id']}-{suffix}"

    status, body = _post_json(ALLOWED_WEBHOOK_URL, envelope)
    print(f"HTTP status: {status}")
    _print_sanitized("Response", body)

    if (
        not args.no_auto_approve
        and body.get("status") == "PENDING_HUMAN_REVIEW"
        and body.get("approval_status") == "PENDING"
    ):
        outcome = _auto_approve(envelope["workflow_request_id"], args.notifications_log)
        print(f"Approval resolution: {outcome}")

    return 0 if status in (200, 202) else 1


if __name__ == "__main__":
    sys.exit(main())
