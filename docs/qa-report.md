# QA Report: Five-Scenario Acceptance Matrix (T702)

Automated, reproducible run: `scripts/run_acceptance_matrix.py`. Full
machine-readable output: `docs/qa_matrix_results.json` (regenerated on
every run; timestamps/IDs change, decisions and routing do not).

Scenarios 1-3 use real validation-split session rows (`data/raw/
online_shoppers_intention.csv` row indices 16, 6797, 214 --
`data/processed/split_indices.json`'s `validation_index`, never the test
set) sent through the real, running n8n workflow's production webhook,
which calls the **real** FastAPI `/route` endpoint (real frozen model
artifacts, real preprocessing/calibration). A deterministic, non-LLM fake
provider (`scripts/qa_fake_agent_api_server.py --mode rule_based`) stands
in for Anthropic so no real network call is made; it reproduces the exact
policy in `prompts/decision_agent_v1.md` from the real prediction it
receives. Scenario 4 uses the same real model/API layer with a fake
provider that always returns unusable content (`--mode invalid`),
exercising the real agent client's bounded retry (2 attempts) and
deterministic fallback. Scenario 5 reuses the existing mock-route-server
timeout fixture to exercise n8n's own real HTTP node timeout/retry
behavior, unrelated to model correctness. All five scenarios go through
the real, already-imported, active n8n workflow's Switch node and error
paths; the notification target is the local mock receiver only. Zero real
Anthropic or Slack/Discord calls were made during this run.

## Results (2026-09-30 run, 5/5 passed)

| # | Scenario | Fixture | Expected decision | Actual decision | HTTP status | Retry | Notification | Approval |
|---|---|---|---|---|---|---|---|---|
| 1 | Low probability / high confidence | `real_model_log_only.json` | `LOG_ONLY` | `LOG_ONLY` | 200 | n/a | `NOT_ATTEMPTED` | `NOT_REQUIRED` |
| 2 | High probability / high confidence | `real_model_priority_review.json` | `PRIORITY_REVIEW` | `PRIORITY_REVIEW` | 200 | n/a | `SENT` | `NOT_REQUIRED` |
| 3 | Near-threshold / uncertain | `real_model_human_review.json` | `HUMAN_REVIEW` | `HUMAN_REVIEW` | 202 | n/a | `SENT` | `PENDING` -> resumed `approve` -> `200` |
| 4 | Invalid agent output | `real_model_invalid_agent_fallback.json` | `SYSTEM_FALLBACK` (`origin=api`, `AGENT_OUTPUT_INVALID`) | `SYSTEM_FALLBACK` | 202 | 2 attempts (agent client) | `SENT` | `PENDING` -> resumed `approve` -> `200` |
| 5 | API timeout | `api_timeout.json` | `SYSTEM_FALLBACK` (`origin=automation_error`, `AGENT_UNAVAILABLE`) | `SYSTEM_FALLBACK` | 202 | 2 attempts (n8n HTTP node, ~21.1s observed) | `SENT` | `PENDING` -> resumed `approve` -> `200` |

Scenario 1's `purchase_probability`/`calibrated_proba` reproduced from the
real model is `0.0`; scenario 2's is `0.6667`; scenario 3's is `0.1111`,
inside the calibrated uncertainty band (`threshold=0.12 +/- 0.05`), so
`uncertain=true` and the agent's cross-field validator deterministically
forces `HUMAN_REVIEW` regardless of what the provider proposes -- see
`artifacts/metadata/mlp_metadata.json`.

Scenario 4's `origin=api` (a clean 200 from `/route` itself, not an n8n
automation-side error) correctly distinguishes it from scenario 5's
`origin=automation_error` (an n8n-side HTTP timeout), matching the two
different fallback code paths documented in `docs/n8n-workflow.md`
("Retry and error behavior").

## Reproduction

```bash
venv/bin/python scripts/run_acceptance_matrix.py
```

The script starts (and, if it started them, tears down) the mock
notification receiver, n8n (only if not already running -- it never
touches n8n's owner account), and the real FastAPI app under a
deterministic fake provider, in that order, and writes
`docs/qa_matrix_results.json`. No `.env`, real `ANTHROPIC_API_KEY`, or real
`N8N_NOTIFY_WEBHOOK_URL` is read or required.

See also `docs/automation-test-report.md` for the broader P6 evidence set
(all four Switch branches, all three human-approval outcomes, the full
error/adversarial matrix, and the one real Slack send), which this report
complements rather than duplicates.
