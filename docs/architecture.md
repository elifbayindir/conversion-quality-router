# Architecture

This document describes the planned system architecture, component boundaries,
environment variables, secret handling, and artifact paths for the Conversion
Quality Router. It reflects the design locked in `PROJECT_BLUEPRINT.md` and the
runtime decisions recorded in the local decision log. It does not claim that any
component beyond what is explicitly marked "implemented" below is built yet.

## 1. System layers

The system is composed of three independently testable layers, chained together
only at the API and automation boundary:

1. **Model layer** — a locally trained PyTorch MLP (with a logistic regression
   baseline for comparison) predicts a purchase probability from session
   features. This layer never sees agent or automation state.
2. **Agent layer** — a structured LLM decision agent consumes the model's
   prediction payload only. It cannot alter the probability, invent features,
   or produce free-form output; it emits one of a fixed set of decisions
   validated against a strict schema, with a deterministic non-LLM fallback on
   any failure.
3. **Automation layer** — an n8n workflow calls the API, validates the agent's
   JSON, routes on the `decision` field, executes a real external action for
   non-`LOG_ONLY` branches, and returns a response.

Data flows one direction only: session features → prediction → decision →
routed action. No layer writes back into an earlier layer.

## 2. Component boundaries

| Path | Responsibility | Consumes | Produces |
|---|---|---|---|
| `src/conversion_router/data/` | Load, validate, and preprocess the raw dataset; fit train-only transforms | Raw CSV | Validated frames, fitted preprocessor artifact |
| `src/conversion_router/modeling/` | Baseline and MLP training, calibration, threshold selection, inference | Preprocessed features | Trained artifacts, prediction schema payloads |
| `src/conversion_router/agent/` | Prompt, provider-neutral LLM client, schema/cross-field validation, deterministic fallback | Prediction schema payload | Decision schema payload |
| `src/conversion_router/api/` | FastAPI request/response contracts and routes (`/health`, `/ready`, `/predict`, `/decide`, `/route`) | HTTP requests | HTTP responses per the API contract |
| `automation/` | n8n workflow definition and export | API responses | External record/notification, webhook response |
| `scripts/` | Reproducible entrypoints (download data, train, evaluate, smoke test) | — | Side effects in `data/`, `artifacts/` |
| `notebooks/` | Evidence and narrative only; no logic lives here that isn't also in `src/` | — | EDA and evaluation reports |
| `tests/` | Unit, integration, and fixture-driven acceptance tests | `src/` modules, live API | Pass/fail evidence |

No component imports "up" the chain: `data/` and `modeling/` never import
`agent/` or `api/`; `agent/` never imports `api/`.

## 3. Environment variables

All secrets are read from the process environment only. `.env` is never
committed; `.env.example` documents every variable name with an empty or
placeholder value. See `.gitignore` (`.env`, `.env.*`, with `.env.example`
explicitly re-included).

| Variable | Required by | Purpose | Secret |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | `agent/client.py` (P5) | Authenticates the provider-neutral LLM client against the Anthropic Messages API (decision D03) | Yes |
| `N8N_NOTIFY_WEBHOOK_URL` | n8n workflow (P6) | Slack/Discord incoming webhook used for the required real external action (decision D02) | Yes |
| `API_HOST` | `api/app.py` (P4) | Bind host for the local FastAPI service; defaults to `127.0.0.1` if unset | No |
| `API_PORT` | `api/app.py` (P4) | Bind port for the local FastAPI service; defaults to `8000` if unset | No |

This table is updated if a later task introduces a genuinely required new
variable; no variable is added speculatively.

## 4. Secret handling

- Secrets exist only as environment variables, sourced locally from `.env`
  (git-ignored) or the shell environment; never hardcoded.
- `.env.example` lists variable names only, with empty values.
- API request/response payloads never include secret values.
- Logs record `request_id` and normalized, non-secret decision fields only;
  raw LLM provider responses and API keys are never logged (see
  `PROJECT_BLUEPRINT.md` Sections 7.4 and 8.2).
- No personally identifiable information is collected; the dataset is
  anonymized session-level behavioral data (Section 4 of the blueprint).

## 5. Artifact paths

| Path | Contents | Tracked in git |
|---|---|---|
| `artifacts/models/` | Trained baseline and MLP weights/checkpoints | No — generated, git-ignored |
| `artifacts/preprocessors/` | Fitted preprocessing pipeline (train-only fit) | No — generated, git-ignored |
| `artifacts/reports/` | Generated evaluation reports, plots | No — generated, git-ignored |
| `artifacts/metadata/` | Model/agent metadata JSON: threshold, uncertainty band, seed, feature contract version, package versions, artifact hashes | Yes — small, human-reviewable, required for reproducibility evidence |
| `data/raw/` | Downloaded source dataset | No — generated, git-ignored |
| `data/processed/` | Split and preprocessed data | No — generated, git-ignored |
| `data/sample/` | Small committed fixture sample for tests/demos | Yes |

Generated binary/data directories are excluded from version control by
`.gitignore`; only the small, inspectable metadata and sample fixtures are
committed. This lets `artifacts/metadata/*.json` and `data/sample/` serve as
concrete, git-tracked evidence for phase-gate acceptance without committing
large or license-restricted files.

## 6. Current implementation status

As of this document's creation (P0 — Scope and scaffold):

- Implemented: repository scaffold (directories and empty `__init__.py`
  markers), `pyproject.toml` with core/dev dependency groups, `.gitignore`,
  this architecture document, `.env.example`.
- Not yet implemented: data loading/validation/preprocessing, baseline and MLP
  training, calibration, the FastAPI service, the LLM agent client and
  validator, and the n8n workflow. These are tracked as individual tasks in
  `.project-control/tasks.json` (phases P1–P6) and are not claimed as done
  until each task's acceptance criterion has recorded evidence.
