# Demo Script (target: 2:30–2:55 total video, including command output/transitions)

Timed storyboard for a screen-recorded walkthrough. Spoken lines are word-
counted at a natural pace (~150 words/minute) so the spoken content alone
lands well under three minutes; the extra time budget is for watching real
command output and transitioning between screens — not filler.

Total spoken word count: **verified programmatically against this exact
file** — see the word-count check at the end of this document.

Before recording, follow [`docs/demo-checklist.md`](demo-checklist.md) in
full and start the environment with
[`scripts/start_demo_environment.sh`](../scripts/start_demo_environment.sh).
**Every live command in this script uses
[`scripts/demo_post_fixture.py`](../scripts/demo_post_fixture.py)** — never
a raw `curl -d @automation/fixtures/<file>.json`, because the fixture files
contain `description`/`expected_*` metadata the webhook does not accept;
the helper extracts only the `input` envelope, and never prints a secret,
a resume URL, or the webhook URL itself.

**What's live vs. what's fixed evidence, stated honestly on screen:**
- LOG_ONLY, PRIORITY_REVIEW, and HUMAN_REVIEW below are run **live**
  against the **real model** (real MLP inference) with a **deterministic,
  non-LLM fake provider** standing in for Anthropic
  (`scripts/qa_fake_agent_api_server.py --mode rule_based`) — say this
  plainly, do not imply it's a live Anthropic call.
- The Slack notification shown is **not** a new send. It is the one real,
  previously-sent notification from P6 — the **11:48 (Türkiye time)
  PRIORITY_REVIEW message in `#conversion-router-demo`**
  (`workflow_request_id: real-slack-t603-001`,
  [`docs/automation-test-report.md`](automation-test-report.md#real-external-action-evidence-slack)),
  shown briefly as prior real integration evidence. The message body does
  not display `workflow_request_id` as visible text, so identify it on
  screen by **channel and timestamp**, not by searching Slack for the ID —
  the ID-to-message correlation was verified against the local n8n
  execution record (`execution_id: 27`, `startedAt: 2026-09-30 08:48:19
  UTC`, `status: success`), not against the Slack message body itself.
- The invalid-agent `SYSTEM_FALLBACK` branch is shown as **previously
  captured, committed evidence**
  ([`docs/qa_matrix_results.json`](qa_matrix_results.json), scenario 4),
  not re-run live — re-running it live would require restarting the API in
  a different mode mid-recording, which this script deliberately avoids
  for a clean, unambiguous recording.

---

## 0:00–0:20 — Problem and architecture

**Screen:** [`README.md`](../README.md) top, scrolled to the Mermaid
architecture diagram.

**Say:** "Not every session deserves the same attention. This system
predicts purchase probability for an e-commerce session, then routes it
through a constrained decision agent and a real automation workflow —
webhook, model, agent, and a human in the loop wherever the model itself
isn't confident."

## 0:20–0:40 — Model and data evidence

**Screen:** [`docs/model-card.md`](model-card.md) dataset/split section, or
[`notebooks/01_end_to_end_analysis.ipynb`](../notebooks/01_end_to_end_analysis.ipynb)
EDA cells.

**Say:** "It's trained on 12,330 real e-commerce sessions from the UCI
Online Shoppers dataset — imbalanced, about fifteen percent convert. The
split is fingerprint-aware, so duplicate sessions can never leak across
train, validation, and test."

## 0:40–1:00 — LOG_ONLY (live, real model)

**Screen:** terminal, demo environment already running (per checklist).

**Command:**
```bash
venv/bin/python scripts/demo_post_fixture.py automation/fixtures/real_model_log_only.json
```

**Expected sanitized output:** `status: COMPLETED`, `decision: LOG_ONLY`,
`notification_status: NOT_ATTEMPTED`, `approval_status: NOT_REQUIRED`.

**Say:** "A confidently low-probability session, scored live by the real
model, gets logged only — no notification, no review queue."

## 1:00–1:25 — PRIORITY_REVIEW (live) plus the real Slack evidence

**Screen:** terminal running the priority-review command, then a brief cut
to the Slack workspace, `#conversion-router-demo`, the **11:48
PRIORITY_REVIEW message** (identify by channel + timestamp — the
`workflow_request_id` is not shown in the message body itself; see
[`docs/demo-checklist.md`](demo-checklist.md) for how this was correlated).

**Command:**
```bash
venv/bin/python scripts/demo_post_fixture.py automation/fixtures/real_model_priority_review.json
```

**Expected sanitized output:** `status: COMPLETED`,
`decision: PRIORITY_REVIEW`, `notification_status: SENT`,
`approval_status: NOT_REQUIRED`.

**Say:** "A confidently high-probability session gets flagged
`PRIORITY_REVIEW` — this run's notification went to the local mock
receiver, but here's the real one: a genuine Slack message this workflow
sent and I confirmed receiving, back when this automation layer was first
built."

## 1:25–1:50 — HUMAN_REVIEW and approval (live)

**Screen:** terminal running the human-review command.

**Command:**
```bash
venv/bin/python scripts/demo_post_fixture.py automation/fixtures/real_model_human_review.json
```

**Expected sanitized output:** `status: PENDING_HUMAN_REVIEW`,
`decision: HUMAN_REVIEW`, `approval_status: PENDING`, then
`Approval resolution: APPROVED` — the helper resolves the approval
automatically and internally; the signed resume URL itself is never shown.

**Say:** "When the model itself is uncertain — near the decision threshold
— the session always goes to `HUMAN_REVIEW`, no matter what the agent
proposes. The workflow pauses for a real approval step; here it's resolved
automatically for the recording, but the mechanism is the same signed
approval link a reviewer would use."

## 1:50–2:05 — Invalid-agent SYSTEM_FALLBACK (prior evidence, not live)

**Screen:** [`docs/qa_matrix_results.json`](qa_matrix_results.json),
scenario 4 (`invalid_agent_output`).

**Say:** "If the agent's own output can't be validated, it never crashes —
it deterministically falls back to `SYSTEM_FALLBACK`. This is the
previously captured evidence for that exact path, from the project's own
acceptance test suite."

## 2:05–2:30 — Metrics, confusion matrix, calibration

**Screen:** [`docs/test-evaluation-report.md`](test-evaluation-report.md)
metric table and confusion matrices, or the notebook's calibration plot.

**Say:** "On a held-out test set touched exactly once, the MLP reaches a
PR-AUC of zero point three one six against a zero point three oh two
baseline — a small, honestly-reported improvement, not an inflated one. The
model is also calibrated: Brier score improves from zero point two two
three two raw to zero point one one seven seven after calibration."

## 2:30–2:45 — QA, tests, workflow export

**Screen:** [`docs/qa-report.md`](qa-report.md).

**Say:** "Everything here is covered by an automated test suite, a
five-scenario acceptance matrix, and a clean-room reinstall from scratch —
all with zero real external calls by default."

## 2:45–2:55 — Conclusion

**Say:** "A calibrated model, a constrained agent, and a real workflow —
fully reproducible."

---

## Word count check

Per-segment word counts, counted programmatically from each `**Say:**`
line in this exact file: 45 (problem/architecture) + 34 (model/data) + 19
(LOG_ONLY) + 41 (PRIORITY_REVIEW/Slack) + 52 (HUMAN_REVIEW) + 35
(SYSTEM_FALLBACK evidence) + 61 (metrics) + 28 (QA/tests) + 13 (conclusion)
= **328 words**, ≈ **2:11–2:19** at 140-150 wpm — leaving roughly 10-45
seconds of the 2:30–2:55 on-screen target for real command output, the
Slack cut, and screen transitions.
