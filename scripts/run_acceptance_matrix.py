#!/usr/bin/env python3
"""T702: five-scenario end-to-end acceptance matrix.

Runs, automatically and reproducibly:
  1. LOG_ONLY       (low probability / high confidence)
  2. PRIORITY_REVIEW (high probability / high confidence)
  3. HUMAN_REVIEW    (near-threshold / uncertain)
  4. SYSTEM_FALLBACK (invalid agent output -> real API validation/fallback chain)
  5. API timeout     (bounded retry + automation-origin fallback)

Scenarios 1-3 use real validation-split session rows (never the test set)
run through the REAL frozen model/inference layer, with a deterministic,
non-LLM fake provider standing in for Anthropic (scripts/
qa_fake_agent_api_server.py --mode rule_based). Scenario 4 uses the same
real API/model layer with a fake provider that always returns unusable
content, exercising the real agent retry-then-fallback chain. Scenario 5
reuses the existing mock-route-server-based timeout fixture to exercise
n8n's own real HTTP timeout/retry behavior (unrelated to model correctness).
All five scenarios are posted to the real, already-imported, active n8n
workflow's production webhook. Notification target is the local mock
receiver only; zero real Anthropic or Slack/Discord calls are made.

Usage:
    venv/bin/python scripts/run_acceptance_matrix.py

Writes machine-readable results to docs/qa_matrix_results.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
FIXTURES_DIR = PROJECT_ROOT / "automation" / "fixtures"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTIFICATIONS_LOG = LOGS_DIR / "qa_matrix_notifications.jsonl"
RESULTS_PATH = PROJECT_ROOT / "docs" / "qa_matrix_results.json"

WEBHOOK_URL = "http://127.0.0.1:5678/webhook/conversion-quality-router"
MOCK_NOTIFY_URL = "http://127.0.0.1:8090/"
FASTAPI_PORT = 8000


def log(msg: str) -> None:
    print(f"[matrix] {msg}", flush=True)


def _post_json(url: str, body: dict, timeout: float = 30) -> tuple[int, dict, float]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- loopback only
            elapsed = time.monotonic() - t0
            return resp.status, json.loads(resp.read().decode("utf-8")), elapsed
    except urllib.error.HTTPError as exc:
        elapsed = time.monotonic() - t0
        raw = exc.read().decode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"_raw": raw}
        return exc.code, payload, elapsed


def _wait_for_http_ok(url: str, timeout_s: float = 15) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:  # noqa: S310 -- loopback only
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def _port_open(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _start_bg(cmd: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )


def _stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _wait_for_marker(proc: subprocess.Popen, marker: str, timeout_s: float = 30) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                return False
            time.sleep(0.05)
            continue
        if marker in line:
            return True
    return False


RUN_SUFFIX = ""  # set once in main(); keeps workflow_request_id unique across repeated runs


def _load_fixture(name: str) -> dict:
    """Load a fixture and make its workflow_request_id unique to this run.

    n8n's duplicate-request dedup map (workflow static data, keyed on
    workflow_request_id) persists across n8n process restarts in practice
    (see docs/n8n-workflow.md "Idempotency"), so re-running this matrix
    with the fixture files' literal IDs would spuriously 409 on the second
    and later runs. The suffix keeps this script idempotent-safe to re-run
    without mutating the checked-in fixture files themselves.
    """
    fixture = json.loads((FIXTURES_DIR / name).read_text())
    if RUN_SUFFIX:
        fixture["input"]["workflow_request_id"] = (
            f"{fixture['input']['workflow_request_id']}-{RUN_SUFFIX}"
        )
    return fixture


def _last_notification_for(workflow_request_id: str) -> dict | None:
    if not NOTIFICATIONS_LOG.exists():
        return None
    for line in reversed(NOTIFICATIONS_LOG.read_text().splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("workflow_request_id") == workflow_request_id:
            return payload
    return None


def _resolve_human_review(
    workflow_request_id: str, action: str = "approve", note: str = "qa-matrix automated resume"
) -> dict:
    deadline = time.monotonic() + 15
    notif = None
    while time.monotonic() < deadline:
        notif = _last_notification_for(workflow_request_id)
        if notif and notif.get("approval_resume_url"):
            break
        time.sleep(0.3)
    if not notif or not notif.get("approval_resume_url"):
        return {
            "resolved": False,
            "reason": "no approval_resume_url observed in mock notification log",
        }
    resume_url = notif["approval_resume_url"]

    # The notification (and its resume URL) is sent slightly before the
    # execution actually finishes registering its Wait-node "waiting" state
    # with n8n's WaitingWebhooks handler. A resume call that arrives in that
    # narrow window gets a transient 409 ("execution is running already"),
    # not a real failure -- retry briefly instead of treating it as fatal.
    status, body, elapsed = None, None, None
    last_attempt = 0
    for attempt in range(1, 11):
        last_attempt = attempt
        status, body, elapsed = _post_json(resume_url, {"action": action, "note": note}, timeout=10)
        if status != 409:
            break
        time.sleep(0.5)
    return {
        "resolved": True,
        "action": action,
        "resume_http_status": status,
        "resume_response": body,
        "resume_elapsed_seconds": round(elapsed, 3),
        "resume_attempts": last_attempt,
    }


# ---------------------------------------------------------------------------
# Scenarios (run against the real n8n webhook)
# ---------------------------------------------------------------------------


def scenario_1_log_only() -> dict:
    fixture = _load_fixture("real_model_log_only.json")
    status, body, elapsed = _post_json(WEBHOOK_URL, fixture["input"])
    passed = (
        status == fixture["expected_http_status"]
        and body.get("decision") == "LOG_ONLY"
        and body.get("status") == "COMPLETED"
        and body.get("notification_status") == "NOT_ATTEMPTED"
        and body.get("approval_status") == "NOT_REQUIRED"
    )
    return {
        "scenario": "1_low_probability_high_confidence",
        "fixture": "real_model_log_only.json",
        "expected_decision": "LOG_ONLY",
        "http_status": status,
        "elapsed_seconds": round(elapsed, 3),
        "response": body,
        "pass": passed,
    }


def scenario_2_priority_review() -> dict:
    fixture = _load_fixture("real_model_priority_review.json")
    status, body, elapsed = _post_json(WEBHOOK_URL, fixture["input"])
    passed = (
        status == fixture["expected_http_status"]
        and body.get("decision") == "PRIORITY_REVIEW"
        and body.get("status") == "COMPLETED"
        and body.get("notification_status") == "SENT"
        and body.get("approval_status") == "NOT_REQUIRED"
    )
    return {
        "scenario": "2_high_probability_high_confidence",
        "fixture": "real_model_priority_review.json",
        "expected_decision": "PRIORITY_REVIEW",
        "http_status": status,
        "elapsed_seconds": round(elapsed, 3),
        "response": body,
        "pass": passed,
    }


def scenario_3_human_review() -> dict:
    fixture = _load_fixture("real_model_human_review.json")
    status, body, elapsed = _post_json(WEBHOOK_URL, fixture["input"])
    passed = (
        status == fixture["expected_http_status"]
        and body.get("decision") == "HUMAN_REVIEW"
        and body.get("status") == "PENDING_HUMAN_REVIEW"
        and body.get("notification_status") == "SENT"
        and body.get("approval_status") == "PENDING"
    )
    approval = _resolve_human_review(fixture["input"]["workflow_request_id"], action="approve")
    return {
        "scenario": "3_near_threshold_uncertain",
        "fixture": "real_model_human_review.json",
        "expected_decision": "HUMAN_REVIEW",
        "http_status": status,
        "elapsed_seconds": round(elapsed, 3),
        "response": body,
        "approval": approval,
        "pass": passed
        and approval.get("resolved") is True
        and approval.get("resume_http_status") == 200,
    }


def scenario_4_invalid_agent_output() -> dict:
    fixture = _load_fixture("real_model_invalid_agent_fallback.json")
    status, body, elapsed = _post_json(WEBHOOK_URL, fixture["input"])
    passed = (
        status == fixture["expected_http_status"]
        and body.get("decision") == "SYSTEM_FALLBACK"
        and body.get("status") == "PENDING_HUMAN_REVIEW"
    )
    approval = _resolve_human_review(fixture["input"]["workflow_request_id"], action="approve")
    return {
        "scenario": "4_invalid_agent_output",
        "fixture": "real_model_invalid_agent_fallback.json",
        "expected_decision": "SYSTEM_FALLBACK",
        "http_status": status,
        "elapsed_seconds": round(elapsed, 3),
        "response": body,
        "approval": approval,
        "pass": passed and approval.get("resolved") is True,
    }


def scenario_5_api_timeout() -> dict:
    fixture = _load_fixture("api_timeout.json")
    status, body, elapsed = _post_json(WEBHOOK_URL, fixture["input"], timeout=40)
    # Node timeout is 10s with maxTries=2 and 1s wait-between -> ~21s expected.
    passed = (
        status == fixture["expected_http_status"]
        and body.get("decision") == "SYSTEM_FALLBACK"
        and body.get("error", {}).get("origin", body.get("origin")) in (None, "automation_error")
        and 15 <= elapsed <= 35
    )
    approval = _resolve_human_review(fixture["input"]["workflow_request_id"], action="approve")
    return {
        "scenario": "5_api_timeout_bounded_retry",
        "fixture": "api_timeout.json",
        "expected_decision": "SYSTEM_FALLBACK",
        "http_status": status,
        "elapsed_seconds": round(elapsed, 3),
        "response": body,
        "approval": approval,
        "pass": passed and approval.get("resolved") is True,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> int:
    global RUN_SUFFIX
    RUN_SUFFIX = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    started_n8n = False
    n8n_proc: subprocess.Popen | None = None
    notif_proc: subprocess.Popen | None = None
    fastapi_proc: subprocess.Popen | None = None

    try:
        # --- notification receiver (shared across all rounds) ---
        log("Starting mock notification receiver on :8090 ...")
        notif_proc = _start_bg(
            [
                PYTHON,
                "scripts/mock_notification_receiver.py",
                "--port",
                "8090",
                "--log",
                str(NOTIFICATIONS_LOG),
            ]
        )
        if not _wait_for_http_ok("http://127.0.0.1:8090/health", timeout_s=5) and not _port_open(
            8090
        ):
            time.sleep(1.0)
        if not _port_open(8090):
            raise RuntimeError("mock notification receiver did not come up on :8090")

        # --- n8n itself: only start if not already running; never touch owner setup ---
        if _port_open(5678):
            log("n8n already listening on :5678 -- reusing the running instance as-is.")
        else:
            log("Starting n8n (automation/start_n8n.sh) with the mock notification URL forced ...")
            fastapi_url = f"http://127.0.0.1:{FASTAPI_PORT}"
            n8n_env_cmd = [
                "bash",
                "-c",
                f'N8N_NOTIFY_WEBHOOK_URL="{MOCK_NOTIFY_URL}" '
                f'FASTAPI_BASE_URL="{fastapi_url}" '
                f"exec ./automation/start_n8n.sh",
            ]
            n8n_proc = _start_bg(n8n_env_cmd)
            started_n8n = True
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline and not _port_open(5678):
                if n8n_proc.poll() is not None:
                    raise RuntimeError("n8n process exited before binding :5678")
                time.sleep(0.5)
            if not _port_open(5678):
                raise RuntimeError("n8n did not bind :5678 within 90s")
            time.sleep(2.0)  # let the production webhook trigger finish registering

        # =====================================================================
        # Round 1: real model + rule-based deterministic fake provider (1-3)
        # =====================================================================
        log("Round 1: starting real API + rule-based fake provider on :8000 ...")
        fastapi_proc = _start_bg(
            [
                PYTHON,
                "scripts/qa_fake_agent_api_server.py",
                "--mode",
                "rule_based",
                "--port",
                str(FASTAPI_PORT),
            ]
        )
        if not _wait_for_marker(fastapi_proc, "QA_FAKE_API_READY", timeout_s=30):
            raise RuntimeError("qa_fake_agent_api_server (rule_based) did not become ready")

        log("Scenario 1/5: LOG_ONLY (real model, low probability) ...")
        results.append(scenario_1_log_only())
        log("Scenario 2/5: PRIORITY_REVIEW (real model, high probability) ...")
        results.append(scenario_2_priority_review())
        log("Scenario 3/5: HUMAN_REVIEW (real model, uncertain) ...")
        results.append(scenario_3_human_review())

        _stop(fastapi_proc)
        fastapi_proc = None

        # =====================================================================
        # Round 1b: real model + always-invalid fake provider (4)
        # =====================================================================
        log("Round 1b: starting real API + always-invalid fake provider on :8000 ...")
        fastapi_proc = _start_bg(
            [
                PYTHON,
                "scripts/qa_fake_agent_api_server.py",
                "--mode",
                "invalid",
                "--port",
                str(FASTAPI_PORT),
            ]
        )
        if not _wait_for_marker(fastapi_proc, "QA_FAKE_API_READY", timeout_s=30):
            raise RuntimeError("qa_fake_agent_api_server (invalid) did not become ready")

        log("Scenario 4/5: SYSTEM_FALLBACK (real API retry+fallback, invalid agent output) ...")
        results.append(scenario_4_invalid_agent_output())

        _stop(fastapi_proc)
        fastapi_proc = None

        # =====================================================================
        # Round 2: mock route server, real n8n HTTP timeout/retry (5)
        # =====================================================================
        log("Round 2: starting mock /route server on :8000 for the timeout scenario ...")
        fastapi_proc = _start_bg(
            [PYTHON, "scripts/mock_route_server.py", "--port", str(FASTAPI_PORT)]
        )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not _port_open(FASTAPI_PORT):
            time.sleep(0.2)
        if not _port_open(FASTAPI_PORT):
            raise RuntimeError("mock_route_server did not come up on :8000")

        log("Scenario 5/5: API timeout (real n8n HTTP node timeout + bounded retry) ...")
        results.append(scenario_5_api_timeout())

        _stop(fastapi_proc)
        fastapi_proc = None

    finally:
        log("Tearing down everything this script started ...")
        _stop(fastapi_proc)
        _stop(notif_proc)
        if started_n8n:
            _stop(n8n_proc)
        else:
            log("n8n was already running before this script started -- left as-is.")

    passed = sum(1 for r in results if r.get("pass"))
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_scenarios": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    log(f"Wrote machine-readable results to {RESULTS_PATH.relative_to(PROJECT_ROOT)}")

    print("")
    print("=== T702 ACCEPTANCE MATRIX SUMMARY ===")
    for r in results:
        status = "PASS" if r.get("pass") else "FAIL"
        decision = r["response"].get("decision")
        print(f"  [{status}] {r['scenario']} -> decision={decision} http={r['http_status']}")
    print(f"  TOTAL: {passed}/{len(results)} passed")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
