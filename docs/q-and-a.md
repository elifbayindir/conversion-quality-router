# Anticipated Questions and Answers

Concise, technically honest answers, each pointing at the file where the
underlying decision or evidence actually lives.

### Why PR-AUC?

The target is imbalanced (~15.5% positive). ROC-AUC can look strong on an
imbalanced target because the large negative class dominates the
false-positive-rate axis; PR-AUC is more sensitive to how well the model
ranks the minority (converting) class, which is what this system needs to
surface. See [`docs/evaluation-card.md`](evaluation-card.md#pr-auc-rationale).

### Why logistic regression as the baseline?

It's a strong, fast, well-understood reference point that isolates the
effect of model architecture: both models share the identical fitted
preprocessing pipeline, so any metric difference is attributable to the
MLP's capacity, not to different feature engineering. See
[`docs/baseline-report.md`](baseline-report.md).

### Why use an MLP if the improvement is small?

Reported honestly as small (+0.014 PR-AUC, +0.011 ROC-AUC on test), not
inflated. It's used because it's the deep-learning component the project
requires, and because it enables the calibration + uncertainty-band design
the whole downstream routing policy depends on — the comparison is reported
transparently either way. See
[`docs/test-evaluation-report.md`](test-evaluation-report.md).

### Why exclude `PageValues`?

The source documents it as computed from the average value of a page
relative to transactions completed *after* viewing it — it's constructed
from the same conversion outcome the model predicts, and its availability
at true decision time isn't proven. Using it would risk leaking the
target into the input. See
[`docs/leakage-audit.md`](leakage-audit.md#pagevalues-decision).

### How was duplicate leakage prevented?

The raw dataset has 125 duplicate rows across 76 groups (identical values
on all 16 deployment-safe features). The split is fingerprint-aware: every
row gets a SHA-256 fingerprint, and the 70/15/15 split is performed over
fingerprint *groups*, not raw rows, so a duplicate can never land in two
different splits.
`n_cross_split_fingerprint_groups: 0`, checked on every split run. See
[`artifacts/metadata/split_summary.json`](../artifacts/metadata/split_summary.json).

### Why calibrate?

A raw sigmoid output is not automatically a trustworthy probability.
Isotonic calibration (fit on validation only) improved the Brier score on
**both** validation (0.2126 → 0.1126) and test (0.2232 → 0.1177) — the test
improvement is what shows calibration generalizes, not just overfits
validation noise. See
[`docs/evaluation-card.md`](evaluation-card.md#calibration-brier-results).

### How is uncertainty defined?

Deterministically, not learned: `uncertain = |calibrated_probability -
threshold| < 0.05`. It's a fixed rule versioned alongside the model
metadata, not a separate model. See
[`artifacts/metadata/mlp_metadata.json`](../artifacts/metadata/mlp_metadata.json)
(`calibration.uncertainty_rule`).

### Why use an LLM agent here?

To turn a single numeric prediction into a structured, explainable routing
decision with a short natural-language rationale for the human reviewer,
without letting the model touch the prediction itself. The agent's job is
narrow by design — see
[`docs/agent-card.md`](agent-card.md#agent-purpose).

### What prevents hallucinated actions?

The agent can only ever produce six fixed fields, validated against a
strict JSON Schema sent directly to the provider's structured-output
mechanism (`extra="forbid"`); `SYSTEM_FALLBACK` isn't even a legal value in
that schema. A cross-field validator additionally enforces the
uncertain-session policy regardless of what the model proposes. The agent
never executes any action itself — only the n8n workflow does, after
independently validating the agent's output again. See
[`docs/agent-card.md`](agent-card.md#validation-rules).

### What happens if Anthropic is unavailable?

One bounded retry, then a deterministic, non-LLM `SYSTEM_FALLBACK` —
`reason_codes: ["AGENT_UNAVAILABLE"]` — routed straight to a human. The
agent client never raises to its caller, and the API wraps the call in a
second defensive layer, so this never surfaces as a 500. See
[`docs/agent-card.md`](agent-card.md#deterministic-system_fallback).

### What happens if FastAPI or Slack times out?

The n8n HTTP call to the API has a 10-second timeout with one retry; any
failure produces an automation-origin `SYSTEM_FALLBACK`
(`reason_codes: ["AGENT_UNAVAILABLE"]`, `origin: "automation_error"`) —
verified with a real ~21-second timeout-then-fallback run (10s × 2 attempts
+ 1s wait). A notification failure is retried once and marked
`notification_status: "FAILED"` without failing the whole workflow
execution. See
[`docs/n8n-workflow.md`](n8n-workflow.md#retry-and-error-behavior).

### Why human approval?

Because the model's own uncertainty band, and the agent's fixed decision
set, are both deliberately conservative: anything the model isn't
confident about, and anything the agent's output can't be validated for,
is routed to a human rather than resolved automatically. This is a design
choice, not a limitation being apologized for. See
[`docs/agent-card.md`](agent-card.md#human-oversight).

### What are the dataset's real-world limitations?

No real timestamp field (no temporal holdout was possible), a single
undisclosed retailer/time period, and no validation yet against a second,
independent population. Stated explicitly, not implied away. See
[`docs/model-card.md`](model-card.md#representativeness-limitations).

### Is `TrafficType` platform dependence?

No. It's an anonymized integer channel/source code with no documented
mapping to real channel names. This project makes no claim that it
measures or causally tests platform dependence, channel effectiveness, or
any comparison between traffic sources — it's used strictly as a
categorical proxy feature. See
[`docs/leakage-audit.md`](leakage-audit.md#per-feature-review).

### What would change for production deployment?

Re-validate calibration and threshold on the target population's own
held-out data before trusting them; replace n8n's in-process duplicate
tracking with a durable external store; add horizontal scaling / queue mode
if traffic requires it; and re-run the clean-room verification against the
actual deployment environment. See
[`docs/evaluation-card.md`](evaluation-card.md#limitations-and-next-evaluation-steps)
and [`docs/n8n-workflow.md`](n8n-workflow.md#known-poc-limitations).

### What does n8n add beyond ordinary Python code?

A visual, inspectable, exportable definition of the routing workflow
(webhook → validate → call API → validate decision → switch → notify →
wait for approval → respond) that's independently testable and re-importable
without touching the model or agent code, plus built-in retry/timeout
handling and a signed human-approval mechanism, all without writing a
custom scheduler or webhook server by hand. See
[`docs/n8n-workflow.md`](n8n-workflow.md).
