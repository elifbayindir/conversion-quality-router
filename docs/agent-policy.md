# Decision Agent Policy and Validation Flow

This document explains how the LLM decision agent is constrained,
validated, and made to fail safely. The versioned prompt text itself lives
in `prompts/decision_agent_v1.md`; this document explains the code around
it. Implementation: `src/conversion_router/agent/{client,validator,fallback}.py`.

## What the agent receives and produces

The agent receives exactly one JSON object: an already-validated
`PredictionResponse` (`purchase_probability`, `decision_threshold`,
`predicted_class`, `decision_margin`, `uncertain`, `signals`, `model` info).
No raw session data, no free text, no user-authored content of any kind is
ever included. See `build_user_content()` in `agent/client.py`.

The agent is only allowed to produce six fields — `decision`, `priority`,
`allowed_action`, `requires_human_approval`, `reason_codes`, `explanation`
— defined by `LLMDecisionCore` in `schemas.py`. Everything else in a final
`DecisionResponse` (`request_id`, `agent_version`, `schema_version`) is
filled in by the application after validation succeeds; the model never
sees or sets these fields.

`LLMDecisionCore`'s `decision` enum (`LLMDecision`) deliberately excludes
`SYSTEM_FALLBACK`. The model cannot select it — not because of a runtime
check, but because it is not a legal value in the JSON Schema sent to the
provider. `SYSTEM_FALLBACK` only ever comes from
`agent/fallback.py::build_fallback_decision`, which is not an LLM call.

## Structured output, not free-text parsing

The request to the Anthropic Messages API sets
`output_config.format.type = "json_schema"` with a schema generated
directly from `LLMDecisionCore.model_json_schema()`. There is no regex-based
JSON extraction and no "please respond with only JSON" prompt instruction
doing the enforcement — the provider's own structured-output mechanism does.

## Validation pipeline (in order)

1. **Transport.** The HTTP call itself can fail (timeout, connection error,
   HTTP 429, HTTP 5xx, non-200 status). Raised as `AgentTransportError`.
2. **Output usability.** An empty response or a refusal (`stop_reason ==
   "refusal"`) is raised as `AgentOutputError`.
3. **JSON parsing.** The response text must parse as JSON
   (`json.JSONDecodeError` otherwise).
4. **Pydantic schema validation.** `LLMDecisionCore.model_validate(...)`
   rejects unknown fields (`extra="forbid"`), invalid enum values (including
   an attempted `SYSTEM_FALLBACK`, any invented reason code, or any
   attempt to set an immutable field, which is simply an "unknown field"
   from this schema's point of view), and an oversized `explanation`
   (>280 characters).
5. **Cross-field validation** (`agent/validator.py::validate_decision_core`).
   Structural validity is not enough; this step enforces the policy stated
   in the prompt:
   - Each `decision` value pairs with exactly one `allowed_action`
     (`LOG_ONLY` → `ADD_TO_LOG`, `PRIORITY_REVIEW` →
     `ADD_TO_REVIEW_QUEUE_PRIORITY`, `HUMAN_REVIEW` →
     `ADD_TO_REVIEW_QUEUE`).
   - If the prediction's own `uncertain` flag is `true`, the agent's
     `decision` MUST be `HUMAN_REVIEW`, `requires_human_approval` MUST be
     `true`, `allowed_action` MUST be `ADD_TO_REVIEW_QUEUE`, and
     `reason_codes` MUST include `MODEL_UNCERTAIN` — regardless of what the
     model chose. This is the one place the deterministic policy overrides
     the model's own judgement, by design (PROJECT_BLUEPRINT.md Section
     7.1: the agent does not get to present an uncertain prediction as
     confident).

Any failure at steps 1-5 counts as one failed attempt.

## Retry and fallback

`get_decision()` allows **at most one retry** (two attempts total,
`MAX_ATTEMPTS = 2`). The retry is identical to the first attempt — the same
system prompt and the same prediction payload are sent again; no
error-correction feedback is added to keep the policy simple and bounded.
If both attempts fail for any reason above, `build_fallback_decision()`
produces a deterministic `SYSTEM_FALLBACK` decision:

```json
{
  "decision": "SYSTEM_FALLBACK",
  "priority": "HIGH",
  "allowed_action": "ADD_TO_REVIEW_QUEUE",
  "requires_human_approval": true,
  "reason_codes": ["AGENT_UNAVAILABLE" | "AGENT_OUTPUT_INVALID"],
  "explanation": "Agent output could not be validated; routed to human review."
}
```

`AGENT_UNAVAILABLE` is used for transport/credential failures;
`AGENT_OUTPUT_INVALID` is used when the provider responded but the content
failed parsing or validation. This fallback is not an LLM call — it is a
plain Python function — and it always returns a schema-valid
`DecisionResponse`. `get_decision()` never raises to its caller; `/decide`
and `/route` (see below) additionally wrap the whole call in a defensive
`try/except` so a genuinely unexpected exception still degrades to the same
fallback rather than a 500 error.

## Secrets and logging

`ANTHROPIC_API_KEY` is read only from the environment (optionally loaded
from a project-root `.env` via `python-dotenv`; `.env` is git-ignored and
never committed). Log statements in `agent/client.py` record the failure
*reason* (exception message, attempt number) but never the API key and
never the raw provider response body.

## API surface

- `POST /decide` — takes an already-produced, validated `PredictionResponse`
  and returns a `DecisionResponse` (200), including the fallback case (a
  fallback is a valid, successful response from the API's point of view,
  not an error).
- `POST /route` — takes a `SessionFeaturesRequest`, runs the model
  prediction (T402) and then the agent (this document) in one call, and
  returns a `DecisionResponse` (200). If the model itself is unavailable
  (T403's `/predict` failure mode), `/route` returns the same 503
  `MODEL_UNAVAILABLE` `ErrorResponse` `/predict` would — the agent is never
  reached in that case, since there is nothing valid to hand it.
