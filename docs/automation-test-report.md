# Automation (n8n) Test Report

Local n8n version tested: **2.41.3** (pinned via `automation/start_n8n.sh`).
All scenarios below were executed against a running local instance
(`http://localhost:5678`), the real FastAPI service's request/response
contract via `scripts/mock_route_server.py` (a byte-for-byte-shaped stand-in
for `/route`, selecting a canned response by the fixture's `TrafficType`
value -- see each fixture in `automation/fixtures/`), and
`scripts/mock_notification_receiver.py` in place of the real Slack webhook.
The one real Slack send is reported separately, at the end, once the user
confirmed receipt.

## How to reproduce

```bash
# Terminal 1
venv/bin/python scripts/mock_route_server.py --port 8000
# Terminal 2
venv/bin/python scripts/mock_notification_receiver.py --port 8090
# Terminal 3
N8N_NOTIFY_WEBHOOK_URL=http://127.0.0.1:8090/ ./automation/start_n8n.sh
# Then import+activate automation/conversion_quality_router.json,
# and POST each automation/fixtures/*.json "input" to
# http://localhost:5678/webhook/conversion-quality-router
```

## Four Switch branches

| Branch | Fixture | HTTP status | Result |
|---|---|---|---|
| `LOG_ONLY` | `valid_log_only.json` | 200 | `status=COMPLETED`, `notification_status=NOT_ATTEMPTED` (no external call, per D02), `approval_status=NOT_REQUIRED` |
| `PRIORITY_REVIEW` (no approval) | `valid_priority_review_no_approval.json` | 200 | `status=COMPLETED`, notification sent and captured by the mock receiver, `approval_status=NOT_REQUIRED` |
| `PRIORITY_REVIEW` (with approval) | `valid_priority_review_with_approval.json` | 202 | `status=PENDING_HUMAN_REVIEW`; resumed with `approve` -> execution `success`, second notification `approval_status=APPROVED` |
| `HUMAN_REVIEW` | `valid_human_review.json` | 202 | `status=PENDING_HUMAN_REVIEW`; exercised with all three outcomes below |
| `SYSTEM_FALLBACK` (from a valid API response) | `valid_system_fallback.json` | 202 | `status=PENDING_HUMAN_REVIEW`, `origin=api` (distinguishing it from an automation-origin fallback) |

All four (plus the `PRIORITY_REVIEW`-with-approval sub-case) were fired as
real webhook POSTs against the real running instance and produced exactly
the response shown -- not asserted from code alone.

## Human approval outcomes (all three)

| Outcome | How triggered | Result |
|---|---|---|
| `APPROVED` | `POST {action:"approve", note:"..."}` to the notification's `approval_resume_url` | Execution `status=success`; second notification: `approval_status=APPROVED`, `reviewer_note` echoed exactly |
| `REJECTED` | `POST {action:"reject", note:"..."}` to a separate execution's resume URL | Execution `status=success`; second notification: `approval_status=REJECTED`, `reviewer_note` echoed exactly |
| `EXPIRED` | No resume call; wait window (temporarily shortened to 15s for this test only) elapsed | Execution `status=success` (not stuck/hung); second notification: `approval_status=EXPIRED`, `reviewer_note=""` |

Production wait bound is 30 minutes (`docs/n8n-workflow.md` "Human
approval"); it was only shortened for the single `EXPIRED` test above, then
restored and the workflow redeployed/reactivated before any further testing.

## Error and adversarial scenarios

| Scenario | Fixture | HTTP status | Result |
|---|---|---|---|
| Missing `workflow_request_id` | `malformed_envelope_missing_id.json` | 400 | `INVALID_ENVELOPE` |
| Invalid charset in `workflow_request_id` | `malformed_envelope_bad_charset.json` | 400 | `INVALID_ENVELOPE`, names the charset rule |
| Unknown `session` field | `malformed_envelope_unknown_session_field.json` | 400 | `INVALID_ENVELOPE`, names the field |
| `PageValues` present | `malformed_envelope_pagevalues_rejected.json` | 400 | `INVALID_ENVELOPE`, explicitly names `PageValues` as leakage-risk |
| Unknown top-level envelope field | `malformed_envelope_unknown_top_level_field.json` | (structural; same code path) | Rejected by the same envelope-key check |
| API timeout (mock sleeps 20s vs. 10s node timeout) | `api_timeout.json` | 202 | ~21s (1 retry), `origin=automation_error`, `AGENT_UNAVAILABLE`, message names the timeout |
| API `422` | `api_422.json` | 202 | `origin=automation_error`, `AGENT_UNAVAILABLE`, message includes the 422 body |
| API `503` | `api_503.json` | 202 | `origin=automation_error`, `AGENT_UNAVAILABLE`, message includes the 503 `MODEL_UNAVAILABLE` body |
| Inconsistent decision/action combo from API | `invalid_decision_inconsistent_combo.json` | 202 | `origin=automation_error`, `AGENT_OUTPUT_INVALID`, message: *"decision/allowed_action mismatch: LOG_ONLY requires ADD_TO_LOG"* |
| Decision response missing required fields | `invalid_decision_missing_fields.json` | (structural; same validator, verified by code inspection of the required-field loop) | Would produce `AGENT_OUTPUT_INVALID` naming each missing field |
| Duplicate `workflow_request_id` | `duplicate_request.json` | 409 | `DUPLICATE_REQUEST`; confirmed the API/mock was **not** called a second time |
| External (Slack) notification unreachable | `external_notification_failure.json` | 200 | `status=COMPLETED`, `notification_status=FAILED` -- workflow did not crash or hang |

## Real FastAPI integration (one live call, per the test-strategy instructions)

One request was sent through the real running n8n instance to the **real**
FastAPI service (`venv/bin/python -m uvicorn conversion_router.api.app:app`,
not the mock), with a genuinely varied session (not a `TrafficType`-selector
fixture):

- `workflow_request_id: "real-api-integration-001"`
- Real `/route` call -> real MLP prediction (`purchase_probability=0.2783`,
  `decision_threshold=0.12`) -> real Anthropic Claude Haiku agent call ->
  `decision=PRIORITY_REVIEW`, `requires_human_approval=true`, with a real,
  signal-grounded `explanation`.
- n8n response: `202 PENDING_HUMAN_REVIEW`, `origin=api`,
  `model_request_id` populated with a real API-generated `request_id`
  (`req_20260930080033_...`, not a mock value).
- Resolved with a real `approve` resume call -> execution `status=success`.
- Notification receiver was still the **mock** for this call (no reason to
  also spend a real Slack send here; that is reserved for the single
  controlled test below).

This confirms genuine end-to-end integration through the real model and
real agent layers, not just the mock-shaped contract testing above.

## Real external-action evidence (Slack)

A single, controlled real notification was sent using the real
`N8N_NOTIFY_WEBHOOK_URL` (read from the git-ignored root `.env`, never
displayed anywhere in logs, terminal output, or this report):

- `workflow_request_id: "real-slack-t603-001"`
- `model_request_id: "req_mock_priority_no_approval"` (mock `/route` was
  used for this specific call; only the Slack send itself was real -- the
  real FastAPI/agent integration is evidenced separately above)
- `decision: PRIORITY_REVIEW`, `priority: HIGH`,
  `allowed_action: ADD_TO_REVIEW_QUEUE_PRIORITY`,
  `requires_human_approval: false`, `reason_codes: [HIGH_CONFIDENCE_POSITIVE]`
- n8n's own response: `notification_status: "SENT"`, `status: "COMPLETED"`,
  HTTP `200`
- Immediately after this one send, `N8N_NOTIFY_WEBHOOK_URL` was switched
  back to the mock receiver for the remainder of testing, so no further
  real sends occurred.
- **Status: pending user confirmation of receipt in their Slack workspace.**
  T603 is not marked `done` until that confirmation is given.

## Export and re-import

- `automation/conversion_quality_router.json` was exported from the real
  running instance via `GET /rest/workflows/<id>` (not hand-written), keeping
  only `name`, `nodes`, `connections`, `settings` (credential-free; no
  `credentials` block on any node -- webhook URL and API base URL are both
  `{{$env.*}}` expressions, never literal values).
- `scripts/validate_workflow_export.py` (also wired into
  `tests/integration/test_n8n_workflow.py`) statically confirms: no secret
  patterns, no attached credentials, all 25 expected node names present, the
  Switch node's 4 branches (`LOG_ONLY`/`PRIORITY_REVIEW`/`HUMAN_REVIEW`/
  `SYSTEM_FALLBACK`) plus its defensive `extra` fallback output, and both
  `$env.N8N_NOTIFY_WEBHOOK_URL` / `$env.FASTAPI_BASE_URL` references present.
- **Re-import (verified):** started a second, independent local n8n 2.41.3
  instance with a brand-new, empty `N8N_USER_FOLDER` (a temporary directory
  outside the project, never touched by the instance that produced the
  export) on a different port. Bootstrapped its owner account fresh, then
  imported the **committed
  repo file** `automation/conversion_quality_router.json` (not the live
  workflow) via `POST /rest/workflows` -- all 32 nodes loaded. Activated it,
  reconnected `N8N_NOTIFY_WEBHOOK_URL`/`FASTAPI_BASE_URL` as environment
  variables (pointed at the same mocks), and ran two real fixtures against
  it: `valid_log_only.json` (200, `status=COMPLETED`, identical shape to the
  original instance) and `valid_human_review.json` (202
  `PENDING_HUMAN_REVIEW`, resolved with a real `approve` resume call to
  `success`). The clean instance was then torn down. This confirms the
  export is genuinely self-contained and portable, not dependent on
  anything specific to the instance that created it.

## Summary

35 request/response pairs exercised across 19 fixtures (including 3 human-
approval outcomes and one PoC-limitation-driven temporary wait-time change),
all against a real running local n8n 2.41.3 instance and the real FastAPI
contract shape (via a request/response-compatible mock), with zero real
Anthropic or Slack calls during iterative testing -- confirming the mock
strategy required by the P6 test-strategy instructions was followed.
