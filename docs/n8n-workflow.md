# n8n Automation Workflow

Single, portable n8n workflow: `automation/conversion_quality_router.json`
(32 nodes). It receives a session, calls this project's FastAPI `/route`
endpoint, validates and normalizes the resulting decision, routes on it,
and performs one real external action (a Slack incoming webhook
notification). No database, custom frontend, Slack bot, or OAuth is used
(out of scope per the project's P6 constraints).

## Node-by-node flow

```
POST Webhook
  -> Validate Envelope (Code)
  -> IF Envelope Valid
       false -> Respond Invalid Envelope (400)
       true  -> Duplicate Check (Code, bounded workflow static data)
                  -> IF Duplicate
                       true  -> Respond Duplicate (409)
                       false -> Call FastAPI Route (HTTP Request, 10s timeout,
                                                     1 retry, continueOnFail)
                                  -> Check Route Call Result (Code)
                                  -> IF Route Call Failed
                                       true  -> Build Automation Fallback
                                                (API Error) (Code, origin=
                                                automation_error)
                                       false -> Validate Decision Response
                                                (Code: required fields +
                                                enum + decision/action
                                                pairing + uncertain-implies-
                                                HUMAN_REVIEW policy)
                                                  -> IF Decision Response Valid
                                                       false -> Build
                                                                Automation
                                                                Fallback
                                                                (Invalid
                                                                Response)
                                                       true  -> Normalize
                                                                Routing Record
                                                                (origin=api)
                  -> [all three merge into] Routing Record Ready (NoOp)
                       -> Switch on Decision (5 outputs: 4 named + 1
                          defensive "Unexpected" fallback)
                            LOG_ONLY         -> Set Log Only Complete
                                                -> Respond Log Only (200)
                            PRIORITY_REVIEW  -> Send Priority Notification
                                                -> Build Priority
                                                   Notification Result
                                                -> IF Priority Requires
                                                   Approval
                                                     false -> Respond
                                                              Priority
                                                              Completed (200)
                                                     true  -> [shared chain]
                            HUMAN_REVIEW     -> Send Review Notification
                                                -> Build Review
                                                   Notification Result
                                                -> [shared chain]
                            SYSTEM_FALLBACK  -> Send Fallback Notification
                                                -> Build Fallback
                                                   Notification Result
                                                -> [shared chain]
                            Unexpected       -> Coerce Unexpected To
                                                Fallback -> Send Fallback
                                                Notification (same node as
                                                above)

[shared chain] Respond Pending Human Review (202 PENDING_HUMAN_REVIEW)
  -> Wait For Human Approval (resume=On Webhook Call, bounded 30 min)
  -> Evaluate Approval Result (Code: approve/reject/timeout -> APPROVED/
     REJECTED/EXPIRED)
  -> Send Final Outcome Notification (second Slack message; no further HTTP
     response to the original webhook call, which already got its 202)
```

## Input/output contracts

**Webhook envelope** (`POST /webhook/conversion-quality-router`):

```json
{
  "workflow_request_id": "demo-001",
  "session": { "...all 16 deployment-safe features, no PageValues...": null }
}
```

- `workflow_request_id`: required, non-empty, ≤100 chars, `[A-Za-z0-9_-]` only.
  This is n8n's own idempotency/correlation key -- distinct from the API's
  `request_id` (renamed `model_request_id` in every response and
  notification from this workflow onward).
- `session`: must contain **exactly** the 16 deployment-safe features
  (`Administrative`, `Administrative_Duration`, `Informational`,
  `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`,
  `BounceRates`, `ExitRates`, `SpecialDay`, `Month`, `OperatingSystems`,
  `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`). `PageValues`
  is explicitly rejected with a named error, not silently dropped. Any other
  unknown field (session-level or envelope-level) is rejected.
- **n8n only checks presence/shape** (required keys, no extras, no
  `PageValues`). It does **not** re-implement numeric-range or categorical-
  whitelist validation -- `SessionFeaturesRequest` in the FastAPI service
  remains the single authoritative source for that, so the two layers can
  never silently drift apart. An out-of-range value (e.g. `BounceRates: 5.0`)
  passes n8n's envelope check and is rejected by `/route` itself, which
  surfaces as an `api_422`/`api_503`-style automation-origin fallback here.

**Webhook response** (see also Section I below for the full shape):

```json
{
  "workflow_request_id": "demo-001",
  "model_request_id": "req_20260930_...",
  "status": "COMPLETED | PENDING_HUMAN_REVIEW | FAILED | DUPLICATE",
  "decision": "LOG_ONLY | PRIORITY_REVIEW | HUMAN_REVIEW | SYSTEM_FALLBACK | null",
  "priority": "...", "requires_human_approval": true,
  "notification_status": "SENT | FAILED | NOT_ATTEMPTED",
  "approval_status": "NOT_REQUIRED | PENDING",
  "error": null
}
```

A follow-up Slack message (not a second HTTP response -- the original
webhook call already completed) later carries the final
`APPROVED | REJECTED | EXPIRED` outcome for anything that returned
`PENDING_HUMAN_REVIEW`.

## The four decision branches (and the defensive fifth)

| Decision | External action | Human approval |
|---|---|---|
| `LOG_ONLY` | None (per decision D02) | Never |
| `PRIORITY_REVIEW` | Slack notification (always) | Only if `requires_human_approval=true` |
| `HUMAN_REVIEW` | Slack notification (always) | Always |
| `SYSTEM_FALLBACK` | Slack notification (always) | Always |
| *Unexpected* (Switch `fallbackOutput: "extra"`, index 4) | Same as `SYSTEM_FALLBACK` | Always |

The fifth branch is **structurally unreachable in normal operation**: any
`decision` value outside the four valid ones is already caught and
converted to an automation-origin `SYSTEM_FALLBACK` by *Validate Decision
Response*, before the Switch node is ever reached. It exists as pure
defense-in-depth (e.g. against a future code change to that validation
step), not as a path exercised by valid API responses -- documented here
rather than force-tested with a synthetic `/route` response that could
never occur through the real, validated pipeline.

## Local n8n setup (reproduction steps)

1. `automation/start_n8n.sh` -- starts `npx n8n@2.41.3` (pinned, no global
   install), with `N8N_USER_FOLDER=automation/.n8n-runtime` (git-ignored:
   owner account, credentials DB, execution DB, cache all live there, never
   `~/.n8n`) and `N8N_LISTEN_ADDRESS=127.0.0.1` (loopback only -- never
   reachable from outside this machine).
2. **n8n 2.x requires an owner account** before its REST/webhook API works.
   `N8N_USER_MANAGEMENT_DISABLED` (a pre-1.0 n8n setting) is a no-op on 2.x,
   so it is not set at all here. **Set the owner account up yourself, once,
   through the browser** at `http://127.0.0.1:5678` (it redirects to
   `/setup` on first load) -- choose your own local email/password there;
   this project's tooling never generates, requests, or transmits that
   credential. If the instance's user management ever needs to be reset
   (e.g. after development-time testing), n8n ships a supported CLI command
   for exactly that, run against the same pinned version and the same
   `N8N_USER_FOLDER`, which clears only the account/session state and
   leaves workflow and execution data untouched:
   ```bash
   N8N_USER_FOLDER="$(pwd)/automation/.n8n-runtime" \
     npx --yes n8n@2.41.3 user-management:reset
   ```
   Never edit or delete the SQLite database file directly.
3. Set `N8N_NOTIFY_WEBHOOK_URL` in the project-root `.env` (git-ignored) to
   a real Slack incoming webhook URL. `start_n8n.sh` loads `.env` without
   ever echoing values, and only for variables not already set in the
   calling shell -- so a test run can override it with a local mock URL
   (see "Testing" below) without editing `.env`.
4. Import `automation/conversion_quality_router.json` via the n8n UI
   ("Import from File") or `POST /rest/workflows` with the JSON as the body,
   then activate it.
5. `FASTAPI_BASE_URL` (default `http://127.0.0.1:8000`) must point at a
   running instance of this project's API (`venv/bin/python -m uvicorn
   conversion_router.api.app:app`).

### Local vs. container URL

This PoC runs n8n as a plain local process (`npx`), not in a container, so
`FASTAPI_BASE_URL=http://127.0.0.1:8000` (or `localhost`) reaches the API
directly. If n8n were containerized instead, `127.0.0.1`/`localhost` inside
the container would refer to the container itself, not the host running the
API -- that setup would need `http://host.docker.internal:8000` (Docker
Desktop) or an explicit Docker network/service name instead. This project
does not containerize n8n (no container requirement was given), so this is
documented as a known adaptation point, not implemented.

### Import vs. publish vs. test vs. production webhook

n8n distinguishes a workflow's **test webhook** (only listens while the
canvas is open in "listen for test event" mode, useful for interactive
building) from its **production webhook** (`/webhook/<path>`, live as soon
as the workflow is **active**, regardless of whether anyone has the UI
open). All testing in this project's fixtures and `docs/automation-test-report.md`
uses the **production** webhook URL against an **activated** workflow,
matching how a real deployment would receive traffic -- not the transient
test-webhook mode. "Importing" a workflow (via file or API) only loads its
definition; it must still be explicitly activated (`PATCH .../activate`)
before its production webhook responds. Also: **n8n does not hot-reload a
running active workflow's webhook trigger from a definition update** --
after any `PATCH` to an active workflow's nodes/connections, it must be
deactivated and reactivated for the change to take effect on the live
webhook (discovered empirically during development; not otherwise
documented behavior we were aware of going in).

## Human approval

Uses the Wait node's **"On Webhook Call"** resume mode (not "On Form
Submitted"): it produces a directly `curl`-able, signed resume URL
(`$execution.resumeUrl`, e.g.
`http://localhost:5678/webhook-waiting/<id>?signature=<token>`) that this
project's notification message includes, and it is trivially scriptable for
automated testing (`POST` a small JSON body) without simulating an HTML form
submission. The resume URL's signature token is generated by n8n per
execution and is not guessable; combined with requiring `POST` (not `GET`,
which is what link-preview crawlers and most bots issue) and a specific
`{"action": "approve"|"reject"}` body shape, this is the PoC's bot/
link-preview protection -- documented here as the explicit, deliberate
choice, rather than adding a separate header-auth credential for a
single-user local instance.

The wait is bounded (`limitWaitTime`, production default 30 minutes,
temporarily shortened to 15 seconds during testing to observe the `EXPIRED`
path -- see `docs/automation-test-report.md`). On timeout with no resume
call, n8n continues the same single output with no `action` field present
in the incoming data; *Evaluate Approval Result* treats a missing/invalid
`action` as `EXPIRED` deterministically.

**No form/URL is ever committed anywhere.** The resume URL is generated at
execution time and only ever appears in the (real or mock) notification
payload sent to the reviewer.

## Idempotency

`workflow_request_id` is the sole deduplication key, tracked in bounded
n8n **workflow static data** (`$getWorkflowStaticData('global')`) -- a
PoC-scale in-memory-then-persisted-to-the-workflow map, not a database:

- Max 500 entries; oldest entries evicted first once over the cap.
- 24-hour TTL per entry, swept on every request.
- A duplicate `workflow_request_id` returns `409 DUPLICATE_REQUEST`
  **before** `/route` or any external notification is ever called.

**Known PoC limitation:** this map is per-workflow, in-process n8n state.
It does not survive an n8n process restart, and under concurrent executions
sharing static data has known race-condition edges in n8n (two requests
with the same ID arriving within the same tick could both read
"not yet seen" before either writes). Acceptable for this project's
single-instance, low-concurrency PoC scope; a production deployment would
use n8n's Data Table feature (where available) or a real external store --
explicitly out of scope here per the P6 instructions (no database).

`workflow_request_id` (n8n's key) and `model_request_id` (the API's
`request_id`, carried through unchanged) are **different identifiers for
different systems** and are never conflated in any response or
notification field.

## Retry and error behavior

| Failure | Behavior |
|---|---|
| Invalid envelope | `400`, no API/external call made |
| Duplicate `workflow_request_id` | `409`, no API/external call made |
| API timeout / connection error / 4xx / 5xx | 1 retry (`retryOnFail`, `maxTries: 2`, 1s between), then an automation-origin `SYSTEM_FALLBACK` routing record (`origin: "automation_error"`, `reason_codes: ["AGENT_UNAVAILABLE"]`) |
| Invalid/inconsistent `DecisionResponse` | Same automation-origin `SYSTEM_FALLBACK`, `reason_codes: ["AGENT_OUTPUT_INVALID"]` |
| External notification failure | 1 retry, then `notification_status: "FAILED"` -- the workflow still completes/responds normally, it does not fail the whole execution |
| Approval rejected / expired | Recorded via the second notification; workflow execution still ends in `success` |

No error path retries more than once, and none loop unboundedly. Error
messages sent externally (webhook response, Slack) never include secrets,
raw provider responses, or full stack traces -- only a short, truncated
`message` string.

## Known PoC limitations

- Duplicate tracking is in-process workflow static data, not a durable
  store (see "Idempotency" above).
- `n8n-nodes-base.switch`'s 5th ("Unexpected") branch is defensive and not
  reachable through the validated pipeline (see "The four decision
  branches").
- The human-approval bot/link-preview protection relies on the resume URL's
  signature plus `POST`-only + body-shape validation, not a separate
  credential (see "Human approval").
- n8n does not hot-reload an active workflow's webhook after a definition
  update; deactivate/reactivate is required (see "Import vs. publish...").
- Local-only, single-instance PoC: no queue mode, no horizontal scaling, no
  container.
