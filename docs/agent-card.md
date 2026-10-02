# Agent Card: Conversion Quality Decision Agent

## Agent purpose

Converts one already-computed, already-validated model prediction into one
routing decision, inside a closed set of options. It does not predict
conversion itself and does not see raw session data. Implementation:
[`src/conversion_router/agent/{client,validator,fallback}.py`](../src/conversion_router/agent/).

## Exact allowed inputs

Exactly one JSON object: an already-validated `PredictionResponse`
(`purchase_probability`, `decision_threshold`, `predicted_class`,
`decision_margin`, `uncertain`, `signals`, `model` info) —
`build_user_content()` sends `prediction.model_dump_json()` and nothing
else. No raw session data, no free text, no user-authored content of any
kind is ever included in the request to the provider.

## Exact allowed outputs

The agent may produce **exactly six fields**, defined by `LLMDecisionCore`
in [`src/conversion_router/schemas.py`](../src/conversion_router/schemas.py):
`decision`, `priority`, `allowed_action`, `requires_human_approval`,
`reason_codes`, `explanation` (max 280 characters). Nothing else is a legal
field in the schema sent to the provider (`extra="forbid"`).

## Immutable, application-controlled fields

`request_id`, `agent_version`, and `schema_version` are filled in by the
application after validation succeeds. The model never sees or sets these —
they are not present in the schema it is given, so an attempted value for
any of them is rejected as an unknown field, not silently accepted and
overwritten.

## Allowed decisions / actions

| `decision` | Paired `allowed_action` |
|---|---|
| `LOG_ONLY` | `ADD_TO_LOG` |
| `PRIORITY_REVIEW` | `ADD_TO_REVIEW_QUEUE_PRIORITY` |
| `HUMAN_REVIEW` | `ADD_TO_REVIEW_QUEUE` |

`SYSTEM_FALLBACK` is **not** a legal value in `LLMDecision` — the model
cannot select it. It only ever comes from
[`agent/fallback.py::build_fallback_decision`](../src/conversion_router/agent/fallback.py),
a plain Python function, never an LLM call.

## Structured JSON schema

The request to the Anthropic Messages API sets
`output_config.format = {"type": "json_schema", "schema": LLMDecisionCore.model_json_schema()}`.
No regex-based JSON extraction and no "please respond with only JSON"
prompt instruction does the enforcement — the provider's own structured
output mechanism does. Model: `claude-haiku-4-5-20251001`, `temperature=0`,
`max_tokens=300`.

## Validation rules

In order — any failure at any step counts as one failed attempt:

1. **Transport** — HTTP failure (timeout, connection error, 429, 5xx,
   non-200) → `AgentTransportError`.
2. **Output usability** — empty response or `stop_reason == "refusal"` →
   `AgentOutputError`.
3. **JSON parsing** — response text must parse as JSON.
4. **Pydantic schema validation** — `LLMDecisionCore.model_validate(...)`,
   `extra="forbid"`, rejects unknown fields, invalid enum values (including
   an attempted `SYSTEM_FALLBACK`), and an oversized `explanation`.
5. **Cross-field validation**
   ([`agent/validator.py`](../src/conversion_router/agent/validator.py)):
   each `decision` must pair with exactly one `allowed_action`; see the
   uncertain-session policy below for the one place this overrides the
   model's own output.

## Uncertain-session policy

If the prediction's own `uncertain` flag is `true`, the agent's `decision`
**must** be `HUMAN_REVIEW`, `requires_human_approval` **must** be `true`,
`allowed_action` **must** be `ADD_TO_REVIEW_QUEUE`, and `reason_codes`
**must** include `MODEL_UNCERTAIN` — regardless of what the model actually
proposed. This is the one place the deterministic policy overrides the
model's own judgement, by design: the agent does not get to present an
uncertain prediction as confident.

## Retry limit

At most **one** retry (`MAX_ATTEMPTS = 2`, two attempts total). The retry
is identical to the first attempt — same system prompt, same prediction
payload — no error-correction feedback loop, to keep the policy simple and
bounded.

## Deterministic `SYSTEM_FALLBACK`

If both attempts fail for any reason above,
`build_fallback_decision()` — a plain Python function, not an LLM call —
produces:

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

`AGENT_UNAVAILABLE` covers transport/credential failures;
`AGENT_OUTPUT_INVALID` covers a response that parsed but failed validation.
`get_decision()` never raises to its caller; `/decide` and `/route`
additionally wrap the call in a defensive `try/except`, so a genuinely
unexpected exception still degrades to this same fallback instead of a 500.

## Provider / runtime dependency

Anthropic Messages API (`https://api.anthropic.com/v1/messages`), accessed
via a provider-neutral `httpx` client
([`AnthropicProvider`](../src/conversion_router/agent/client.py)) so the
provider could be swapped without touching the validation/retry/fallback
logic. `ANTHROPIC_API_KEY` is read from the environment only; its absence
produces `AgentCredentialError`, itself caught and converted into the same
deterministic `SYSTEM_FALLBACK` — a missing credential is a fail-safe
condition, not a crash.

## Prompt version

`agent_version: 1.0.0`, `schema_version: 1.0.0`. Versioned, public prompt
text: [`prompts/decision_agent_v1.md`](../prompts/decision_agent_v1.md),
loaded from that file at runtime, not duplicated in code.

## Human oversight

- Every branch except `LOG_ONLY` sends a notification; `HUMAN_REVIEW` and
  `SYSTEM_FALLBACK` (and `PRIORITY_REVIEW` when `requires_human_approval`)
  additionally pause on a real human approval step in the n8n workflow
  before the routing episode is considered resolved (see
  [`docs/n8n-workflow.md`](n8n-workflow.md#human-approval)).
- The uncertain-session policy above guarantees a human reviews every
  session the model itself is not confident about.
- A human reviewer approves, rejects, or lets the approval window expire;
  all three outcomes are recorded and reported, none crash the workflow.

## Prohibited actions

- The agent cannot take any external action itself — it only ever returns
  a JSON decision object; sending a notification, waiting for approval, and
  executing the routed action are all performed by the n8n automation
  layer, never by the agent or the API process.
- The agent cannot alter the model's probability, invent features, set its
  own `request_id`/`agent_version`/`schema_version`, or select
  `SYSTEM_FALLBACK` — none of these are legal outputs of its schema.

## Failure modes

| Failure | Handling |
|---|---|
| Missing/invalid API key | `AgentCredentialError` → deterministic `SYSTEM_FALLBACK` (`AGENT_UNAVAILABLE`), no retry (nothing to retry) |
| Network timeout / connection error | `AgentTransportError` → 1 retry → fallback (`AGENT_UNAVAILABLE`) |
| HTTP 429 / 5xx | `AgentTransportError` → 1 retry → fallback (`AGENT_UNAVAILABLE`) |
| Empty response / refusal | `AgentOutputError` → 1 retry → fallback (`AGENT_OUTPUT_INVALID`) |
| Invalid JSON | `json.JSONDecodeError` → 1 retry → fallback (`AGENT_OUTPUT_INVALID`) |
| Schema-invalid output (unknown field, bad enum, oversized text) | `ValidationError` → 1 retry → fallback (`AGENT_OUTPUT_INVALID`) |
| Cross-field policy violation (e.g. uncertain session not routed to `HUMAN_REVIEW`) | `DecisionValidationError` → 1 retry → fallback (`AGENT_OUTPUT_INVALID`) |
| Any other unexpected exception | Caught by `/decide`/`/route`'s defensive wrapper → fallback (`AGENT_UNAVAILABLE`), never a 500 |

## Non-use cases

- Not for free-text or open-ended reasoning tasks — the agent only ever
  receives one fixed-shape numeric/categorical payload and only ever
  returns one fixed-shape JSON object.
- Not a substitute for the human approval step — its output is a
  recommendation routed through a workflow that still requires human
  sign-off for anything beyond the most confidently negative case.
- Not designed or tested for multi-turn conversation, tool use beyond this
  one structured call, or any agentic action outside the single
  request/response boundary documented here.
