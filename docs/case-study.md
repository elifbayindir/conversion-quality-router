# Case Study: Uncertainty-Aware Decision Routing for E-Commerce Session Triage

## Executive summary

An e-commerce team reviewing potential conversions faces a resource allocation problem: prediction scores flag many sessions, but not all flagged sessions deserve the same attention. This prototype demonstrates how calibrated probability, an explicit uncertainty band, constrained routing, and structured human feedback turn a binary classification model into a decision-support workflow — one where the model's own uncertainty determines whether a human or an automatic log should handle each session.

The system was built on a public dataset (UCI Online Shoppers Purchasing Intention, 12,330 sessions) and operates as a local proof of concept. It is not connected to a live event stream and makes no causal, revenue, or ROI claims.

## The user and the problem

A reviewer responsible for e-commerce session follow-up has limited capacity. They cannot look at every session the model flags. They need to know:

1. Which flagged sessions deserve immediate attention?
2. Which ones are uncertain enough that a human judgment call is needed?
3. Which ones can be safely logged without review?

A model score alone — "this session has a 14% purchase probability" — does not answer these questions. Whether 14% is high, low, or ambiguous depends on the decision threshold, the model's reliability near that threshold, and the team's capacity for review.

## The design question

> Given a calibrated purchase probability and an explicit measure of how close that probability is to the decision boundary, can a constrained routing policy allocate reviewer attention to where it matters most — while sending uncertain cases to humans and safely logging the rest?

## The solution

The prototype chains three independently testable layers:

1. **Model layer** — a PyTorch MLP produces a purchase probability, calibrated via isotonic regression on validation data. The API returns the probability, the frozen decision threshold (0.12), an uncertainty flag (±0.05 around the threshold), and rule-based supporting signals.

2. **Agent layer** — a constrained decision provider receives the validated prediction payload and proposes one of three routes (`PRIORITY_REVIEW`, `HUMAN_REVIEW`, `LOG_ONLY`) with a priority, a matching action, and reason codes from a closed list. Schema validation and cross-field policy rules enforce the proposal. If anything fails, a deterministic fallback routes to human review.

3. **Automation layer** — an n8n workflow receives the route and sends a Slack notification for non-`LOG_ONLY` routes. `HUMAN_REVIEW` and `SYSTEM_FALLBACK` routes pause on a signed wait node until a reviewer approves or the 30-minute window expires. `PRIORITY_REVIEW` completes immediately unless `requires_human_approval` is true. `LOG_ONLY` sessions are logged without notification.

A local Streamlit UI provides a second front end to the same API, showing the probability gauge, the routing decision, and a review queue. Reviewer feedback (agree or override with a structured reason) is stored in a local append-only SQLite file.

## User journey

1. **A session arrives.** The 16 deployment-safe features are validated against the feature contract. Unknown fields, out-of-range values, and unseen categories are rejected before any model code runs.

2. **The model scores it.** The calibrated MLP returns a purchase probability. A session at 28% is above the 12% threshold and outside the uncertainty band — it is a clear positive. A session at 11% is below the threshold but within the band — it is uncertain.

3. **The routing decision is proposed.** The decision agent maps the scored session to a route, constrained by the schema and policy rules.

4. **The route is executed.** In the n8n path, a Slack notification goes out for non-`LOG_ONLY` routes. `HUMAN_REVIEW` and `SYSTEM_FALLBACK` routes wait for approval; `PRIORITY_REVIEW` completes immediately unless approval is required. In the UI path, the session appears in the review queue with its evidence.

5. **A reviewer acts.** The reviewer sees the probability, the threshold, the uncertainty flag, the supporting signals, and the recommended route. They either agree or override with a structured reason (e.g., "signal weaker than scored").

6. **The decision is recorded.** The feedback row links the model evidence, the agent's decision, and the human response in an append-only audit store.

7. **Policy review stays human.** A summary of agreement rates, override rates, and override reasons is evidence for a person deciding whether to adjust the threshold or policy. Nothing acts on this data automatically.

## Model and calibration evidence

The model is trained on the UCI Online Shoppers Purchasing Intention Dataset (12,330 sessions, ~15.5% positive rate, group-aware 70/15/15 stratified split with seed 42).

**Calibration:** Isotonic calibration on validation data reduces the Brier score from 0.2232 (raw sigmoid) to 0.1177 on the test set. The calibration curve shows improved agreement between predicted probabilities and observed conversion rates at the aggregate level, making the decision threshold and uncertainty band more meaningful.

**Test set (n = 1,845, touched once):**

| Metric | Baseline (LogReg) | MLP (calibrated) |
|---|---|---|
| PR-AUC | 0.302 | 0.316 |
| ROC-AUC | 0.7433 | 0.7543 |
| Brier score | — | 0.1177 |

The improvement is modest (+0.014 PR-AUC). The value of the MLP is not prediction accuracy — it is the calibration that makes the decision threshold and uncertainty band meaningful.

## Operational routing: an illustrative workload allocation

Applied to the frozen test split (1,845 sessions, 286 conversions), the routing policy distributes reviewer attention as follows:

| Route | Sessions | Share | Conversions | Coverage |
|---|---:|---:|---:|---:|
| Priority review | 885 | 48.0% | 223 | 78.0% |
| Human review | 362 | 19.6% | 50 | 17.5% |
| Log automatically | 598 | 32.4% | 13 | 4.5% |

The review queue captures **95.5%** of conversions while covering 67.6% of sessions (a random selection of the same size would capture 67.6%). The automatic log contains 13 conversions — the cost of not reviewing confidently negative sessions.

**Policy consistency:** all 362 uncertain sessions were routed to human review. Zero safety fallbacks occurred. The threshold was chosen with a 5:1 cost ratio (a missed conversion costs five times more than an unnecessary review), so the queue is large by design.

These numbers describe the frozen test split and represent an illustrative workload allocation. They are not a causal impact estimate.

## Two worked cases

### Case 1: Strong signal → priority review → reviewer agrees

A returning visitor in November views 256 product pages over 3.7 hours with near-zero bounce and exit rates.

**Model evidence:** calibrated purchase probability 66.7%, 54.7 pp above the threshold. Clear signal. **Route:** `PRIORITY_REVIEW`, high priority, reason `HIGH_CONFIDENCE_POSITIVE`. **Reviewer:** agrees. **Audit:** recorded with SHA-256 input fingerprint, model and agent versions.

Known outcome (validation label, unavailable at decision time): the session did not convert. The reviewer action is scripted — this illustrates the agree path, not a claim of accuracy.

### Case 2: Uncertain signal → human review → reviewer overrides

A returning visitor in March views 10 product pages over 4.7 minutes with a zero bounce rate but a 4.3% exit rate.

**Model evidence:** calibrated purchase probability 11.1%, 0.9 pp below the threshold. **Uncertain** — inside the ±5 pp band. **Route:** `HUMAN_REVIEW`, medium priority, reason `MODEL_UNCERTAIN`. Human approval required. **Reviewer:** overrides to "log automatically" with reason *signal weaker than scored* ("Only 10 product pages and a high exit rate"). **Audit:** override recorded with structured reason and reviewer note.

Known outcome: the session did not convert. The reviewer action is scripted — this illustrates the override path and shows that the uncertainty flag works as designed.

## The feedback and audit loop

Each reviewer response — agree or override — becomes one row in a local SQLite file. The row links:
- Model evidence (probability, threshold, margin, uncertainty, signals)
- Agent decision (route, priority, action, reason codes)
- Human response (agree/override, structured reason, note)
- Version identifiers (model, agent, feature contract, schema)
- A SHA-256 fingerprint of the input (not the raw features)

The database enforces:
- Append-only (triggers abort UPDATE and DELETE)
- One review per request (unique constraint)
- Schema-version compatibility (the store refuses a file with a different version)

The feedback summary shows agreement and override rates, overrides by route, and the most frequent reasons. This is evidence for a person deciding whether the threshold or policy should change. The system does not act on it.

## Architecture and n8n automation

The system has two independent front ends to the same FastAPI service:

**The local Streamlit UI** calls `/predict` and `/decide`, displays the evidence and route, and records reviewer feedback in local SQLite. It does not invoke n8n.

**The n8n workflow** calls `/route` (prediction + decision in one call), validates and normalizes the response, and routes on the decision field. Non-`LOG_ONLY` routes send a Slack notification. `HUMAN_REVIEW` and `SYSTEM_FALLBACK` routes then wait on a signed approval webhook until a reviewer approves or the 30-minute window expires. `PRIORITY_REVIEW` completes immediately unless `requires_human_approval` is true. It is a 32-node, credential-free portable workflow (webhook URL and API base URL are environment-variable expressions).

These paths do not interfere with each other. UI feedback does not resume an n8n execution. Slack is notification transport only.

![A real Slack notification shows the priority-review recommendation produced for the strong-signal case](figures/slack_priority_review.png)
*Observed end-to-end notification for the strong-signal case. The message exposes the recommended action and approval requirement in operational language while keeping Slack outside the prediction and decision boundary.*

For annotated screenshots of the UI result view, the feedback summary, the n8n workflow canvas, and this notification in context, see the [main README](../README.md#system-architecture).

## Safety controls

| Control | Mechanism |
|---|---|
| Fail to human | Any failure (model, agent, network, validation) → `SYSTEM_FALLBACK` → human review |
| Schema enforcement | Agent output validated against JSON schema; `SYSTEM_FALLBACK` is not in the schema |
| Uncertainty override | Cross-field validator forces uncertain predictions to `HUMAN_REVIEW` |
| Bounded retry | At most one retry (two total); no error-correction loop |
| Append-only audit | Database triggers abort UPDATE/DELETE; one review per request |
| No retraining | Feedback is stored for human review; nothing changes the model or policy |
| Credential isolation | Secrets from environment only; `.env` git-ignored; no credentials in exports |
| Duplicate protection | n8n deduplicates by request ID before any API call |

## Limitations

- **Single dataset, single split.** Calibration and the uncertainty band are validated on one validation split of one public dataset.
- **No temporal holdout.** The dataset has no real timestamp field.
- **Modest model improvement.** The MLP gains +0.014 PR-AUC over logistic regression. The calibration, not the architecture, is the primary contribution.
- **Scripted reviewer actions.** Worked cases use scripted feedback to illustrate the agree and override paths. No real-world reviewer data exists in this prototype.
- **Local, single-instance.** No authentication, no multi-user feedback, no horizontal scaling, no container deployment.
- **In-process duplicate store.** n8n's deduplication uses bounded workflow static data, not a durable external store.
- **No live data.** The system operates on a public dataset and is not connected to a live event stream.

## What a real deployment would require

This prototype demonstrates the routing pattern. Deploying it against live sessions would require:

- **Live event integration.** A real-time session-feature pipeline replacing the static dataset.
- **Authentication and authorization.** Reviewer identity, role-based access, and session management.
- **Governed identity and retention.** Data governance policies for reviewer feedback and session-level data.
- **Production observability.** Structured logging, metrics, alerting, and health monitoring beyond the current `/health` and `/ready` endpoints.
- **Domain-specific cost and capacity validation.** The 5:1 cost ratio and the resulting threshold are modeling assumptions. A real deployment would need cost estimates grounded in the specific business context and validated against actual reviewer capacity.
- **Representative human evaluation.** Reviewer agreement and override rates from real reviewers on real sessions, not scripted actions.
- **Monitoring and controlled policy/model updates.** Drift detection, calibration monitoring, and a governed process for threshold and model updates — not automatic retraining.
- **Durable state management.** A persistent, replicated duplicate store and feedback database.
- **Horizontal scaling and deployment infrastructure.** Container orchestration, load balancing, and blue-green or canary deployment.

None of these are implemented. The prototype validates the routing logic and the human-in-the-loop pattern at the local, single-instance level.

## Conclusion

The Conversion Quality Router demonstrates that the gap between a prediction score and a useful operational decision can be bridged with calibration, explicit uncertainty handling, constrained routing, and structured human feedback. The model's own uncertainty determines the workflow: confident cases are routed efficiently, ambiguous cases go to humans, and every decision is recorded for later review. The pattern is general — applicable wherever a prediction score must be turned into a bounded, auditable action under capacity constraints.
