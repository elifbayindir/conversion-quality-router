# Demo Recording Checklist

Work through this list, in order, before pressing record. Nothing in this
file is a substitute for actually looking at the screen before you start —
it is a checklist, not an automated guarantee.

## Before recording

- [ ] **Clean browser window.** A fresh window/profile, no leftover tabs
  from this session's work (n8n admin UI is fine to have open; unrelated
  tabs are not).
- [ ] **Hide bookmarks bar, personal tabs, and notification popups** (OS
  notification center, browser extensions, email/chat clients closed or
  muted).
- [ ] **No `.env` or secret file visible** in any terminal, editor tab, or
  file browser window that will be on screen.
- [ ] **No API key, webhook URL, or resume token visible** anywhere on
  screen — check terminal scrollback, not just the current prompt.
  [`scripts/demo_post_fixture.py`](../scripts/demo_post_fixture.py) never
  prints the resume URL itself, only the final `APPROVED`/`REJECTED`/
  `UNRESOLVED` outcome — still, do not manually `cat` the mock notification
  log on screen.
- [x] **Real Slack evidence verified.** The Slack message body does not
  display `workflow_request_id` as visible text, so searching Slack for
  the literal string `real-slack-t603-001` returns nothing — that is
  expected, not a sign the message is missing. Identification is by
  **channel + timestamp**, cross-checked against the local n8n execution
  record: `#conversion-router-demo`, the **11:48 (Türkiye time) PRIORITY_REVIEW
  message**, corresponding to n8n `execution_id: 27`
  (`startedAt: 2026-09-30 08:48:19 UTC` = 11:48 local, `status: success`),
  which carries `workflow_request_id: real-slack-t603-001` in the execution
  record itself (not in the Slack message body). This correlation was
  manually verified by the user before recording. During the recording,
  identify the message on screen as *"the 11:48 PRIORITY_REVIEW
  notification"* by channel and timestamp — do not rely on a
  `workflow_request_id` text search inside Slack, since the field is not
  rendered in the message body.
- [ ] **Local services health-checked** before recording starts — this is
  the last step of `scripts/start_demo_environment.sh` itself (below); no
  separate manual check is needed if that script printed
  `DEMO ENVIRONMENT READY`.

## Start the environment

```bash
./scripts/start_demo_environment.sh
```

Fails closed: refuses to start if ports 8000/5678/8090 aren't empty,
forces `N8N_NOTIFY_WEBHOOK_URL` to the local mock receiver before n8n's own
`.env` loader ever runs (a real value in `.env` can never win), unsets
`ANTHROPIC_API_KEY`, and health-checks all three services before printing
`DEMO ENVIRONMENT READY`. Makes zero real Slack/Discord or Anthropic calls.

- [ ] **One controlled real Slack notification maximum** — only the prior,
  already-verified 11:48 `#conversion-router-demo` PRIORITY_REVIEW message
  (n8n `execution_id: 27`, `workflow_request_id: real-slack-t603-001`)
  shown on screen per the storyboard; no new Slack send is made by default.
- [ ] **One controlled real Anthropic call maximum** — not used by default;
  the storyboard's live calls use the real model with a deterministic
  fake-agent provider (`scripts/qa_fake_agent_api_server.py --mode
  rule_based`), never a live Anthropic call.

## During recording

Run only the exact commands in [`docs/demo-script.md`](demo-script.md),
using [`scripts/demo_post_fixture.py`](../scripts/demo_post_fixture.py) —
never a raw `curl -d @automation/fixtures/<file>.json` (the fixture files
carry `description`/`expected_*` metadata the webhook does not accept).

## Stop the environment

```bash
./scripts/stop_demo_environment.sh
```

Stops only the PIDs `start_demo_environment.sh` recorded
(`logs/demo-runtime/*.pid`) — never a broad `pkill` — and verifies ports
8000/5678/8090 are clean afterward.

- [ ] **Stop script run and its "all ports clean" confirmation seen** before
  ending the recording session.

## After recording

- [ ] **Recording duration verified** against the target (2:30–2:55 total
  video, ~328 spoken words / ~2:11–2:19 at natural pace — see
  [`docs/demo-script.md`](demo-script.md#word-count-check)).
- [ ] **Audio reviewed** — audible, no clipping, no background noise
  covering any spoken line.
- [ ] **Video reviewed** — every on-screen element from the "before
  recording" checks above re-confirmed absent in the actual recording (not
  just the live screen), including terminal scrollback that may have
  scrolled into frame, and including the brief Slack cut (no personal
  tabs/email/bookmarks/notifications visible there either).
- [ ] **Video destination/link decision still pending the user** — do not
  publish, upload, or link the recording anywhere until that decision is
  made explicitly.

## Explicitly not done during preparation

- No real Slack or Anthropic call was made while preparing this checklist,
  the storyboard, or the start/stop scripts.
- No recording was started.
- No fake or placeholder video was created.
- T803 is not marked done by preparing this checklist — only by actually
  recording the video per this checklist and the storyboard.
- No placeholder video URL appears anywhere in this repository.
