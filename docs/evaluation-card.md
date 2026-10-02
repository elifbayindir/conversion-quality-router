# Evaluation Card: Conversion Quality Router

## Evaluation design

Two layers are evaluated separately and then together: (1) the model layer
(baseline vs. MLP, on identical preprocessing and identical splits), and (2)
the full system (API, agent, and n8n automation) via structural tests, a
five-scenario acceptance matrix, and a clean-room reproduction. Every number
in this card is read directly from a committed artifact, not transcribed
from memory — see the source path next to each figure.

## Train/validation/test separation

Seed `42`. Stratified 70/15/15 split **over SHA-256 fingerprint groups**
(16 deployment-safe features), not raw row indices, so duplicate-feature
rows cannot cross a split boundary:

| Split | Rows | Positive rate |
|---|---|---|
| Train | 8,635 | 15.47% |
| Validation | 1,850 | 15.46% |
| Test | 1,845 | 15.50% |

`n_cross_split_fingerprint_groups: 0`, `n_conflicting_label_groups: 0` —
checked programmatically on every split run. Source:
[`artifacts/metadata/split_summary.json`](../artifacts/metadata/split_summary.json).

## Final-test discipline

The test set is read exactly once, by `scripts/evaluate_test.py`, and only
after the model, threshold (`0.12`), calibration, and uncertainty band
(`0.05`) are already frozen
(`frozen_before_test_evaluation: true` in
[`artifacts/metadata/mlp_metadata.json`](../artifacts/metadata/mlp_metadata.json)).
No preprocessing fit, threshold selection, calibration fit, or model
selection decision is ever made on the test set.

## Baseline comparison

Class-weighted logistic regression (`threshold=0.47`, selected the same
way as the MLP's threshold — business-cost grid search on validation only)
is trained on the identical preprocessed features as the MLP, so the
comparison isolates the effect of model architecture, not feature
engineering differences. Full report:
[`docs/baseline-report.md`](baseline-report.md).

## PR-AUC rationale

PR-AUC is the primary metric because the target is imbalanced (~15.5%
positive). ROC-AUC can look deceptively strong on an imbalanced target
because the large negative class dominates the false-positive-rate axis;
PR-AUC is more sensitive to how well the model actually ranks the minority
(positive/converting) class, which is the class this system is built to
surface.

## ROC-AUC, precision, recall, F1, and confusion matrix (test set)

| Metric | Baseline (threshold 0.47) | MLP (threshold 0.12) |
|---|---|---|
| PR-AUC | 0.302 | 0.316 |
| ROC-AUC | 0.7433 | 0.7543 |
| Precision | 0.2506 | 0.2514 |
| Recall | 0.7832 | 0.7797 |
| F1 | 0.3797 | 0.3802 |

**Baseline confusion matrix** — TN 889, FP 670, FN 62, TP 224
**MLP confusion matrix** — TN 895, FP 664, FN 63, TP 223

Source:
[`artifacts/metadata/test_evaluation.json`](../artifacts/metadata/test_evaluation.json),
[`docs/test-evaluation-report.md`](test-evaluation-report.md).

## Calibration / Brier results

Isotonic regression fit on validation-split raw sigmoid outputs only.

| | Raw | Calibrated |
|---|---|---|
| Brier (validation) | 0.2126 | 0.1126 |
| Brier (test) | 0.2232 | 0.1177 |

Calibration improves Brier score on **both** validation and the held-out
test set, which is evidence it generalizes rather than just fitting
validation-split noise. Source:
[`artifacts/metadata/mlp_metadata.json`](../artifacts/metadata/mlp_metadata.json)
(`calibration` block),
[`artifacts/metadata/test_evaluation.json`](../artifacts/metadata/test_evaluation.json).

## Threshold business-cost assumption

Both models' thresholds were selected by the same grid search: minimize
`5 × false_negatives + 1 × false_positives` on validation-only predictions
— a stated modeling assumption (a missed converting session is 5x costlier
than one unnecessary manual review), not a measured business figure.
Baseline: threshold `0.47`, expected cost `914.0` at selection time
(validation). MLP: threshold `0.12`, `FN=50, FP=628` at selection time
(validation). Source:
[`artifacts/metadata/baseline_metadata.json`](../artifacts/metadata/baseline_metadata.json),
[`artifacts/metadata/mlp_metadata.json`](../artifacts/metadata/mlp_metadata.json).

## Five-scenario acceptance matrix

Automated, reproducible (`scripts/run_acceptance_matrix.py`), run against
the real, active n8n workflow's production webhook. Scenarios 1-3 use real
validation-split session rows through the real model/inference layer, with
a deterministic non-LLM fake provider standing in for Anthropic; scenario 4
exercises the real agent retry+fallback chain; scenario 5 exercises real
n8n HTTP timeout/retry behavior.

| # | Scenario | Decision | HTTP | Notification | Approval |
|---|---|---|---|---|---|
| 1 | Low probability / high confidence | `LOG_ONLY` | 200 | `NOT_ATTEMPTED` | `NOT_REQUIRED` |
| 2 | High probability / high confidence | `PRIORITY_REVIEW` | 200 | `SENT` | `NOT_REQUIRED` |
| 3 | Near-threshold / uncertain | `HUMAN_REVIEW` | 202 | `SENT` | `PENDING` → approved |
| 4 | Invalid agent output | `SYSTEM_FALLBACK` (`origin=api`) | 202 | `SENT` | `PENDING` → approved |
| 5 | API timeout | `SYSTEM_FALLBACK` (`origin=automation_error`, ~21.1s, 1 retry) | 202 | `SENT` | `PENDING` → approved |

5/5 passed, reproducible (re-run twice). Full machine-readable results:
[`docs/qa_matrix_results.json`](qa_matrix_results.json), summary:
[`docs/qa-report.md`](qa-report.md).

## API / agent / n8n tests

`./scripts/run_quality_gate.sh` (single documented command, zero real
external calls): `tracker.py validate`, `ruff check .`, the full pytest
suite (197 passed, 1 known warning — see below), an explicit re-run of the
agent/route test subset (57 passed), n8n workflow export structural
validation (32 nodes, no secrets/credentials, all 4 Switch branches +
defensive fallback present), a clean notebook re-execution, and a live API
smoke test (9/9 checks: health, ready, valid predict, and three rejection
cases). Adversarial agent coverage includes malformed LLM JSON, contradictory
decision/action pairs, an LLM attempting to set `SYSTEM_FALLBACK` or
tamper with immutable fields, and a completely broken provider — all
degrade to the deterministic fallback, never a 500.

## Clean-room reproduction

`scripts/verify_clean_room.sh`: from a clean source snapshot (tracked +
non-ignored-untracked files only) in a temporary directory outside the
project, a fresh Python 3.11 venv, `pip install -e ".[dev]"`, a real
checksum-verified dataset download, and the full training pipeline
regenerate **every** artifact from scratch — reproducing metrics identical
to the ones reported in this card (baseline threshold `0.47`/PR-AUC
`0.3473` validation, MLP validation PR-AUC `0.3560`, calibrated Brier
`0.1126`). The real API smoke test then passes 9/9 against the freshly
regenerated model, and a second, independently-owned n8n 2.41.3 instance
imports the committed workflow export and runs a fixture successfully.
Full teardown verified. Details:
[`docs/qa-report.md`](qa-report.md).

## Known warning

`StarletteDeprecationWarning: Using httpx with starlette.testclient is
deprecated; install httpx2 instead.` Root cause: `starlette/testclient.py`
itself (third-party code), migrating its `TestClient` to a separate
`httpx2` package. This project's own `httpx` usage
([`src/conversion_router/agent/client.py`](../src/conversion_router/agent/client.py))
is unaffected and current. Not added as a dependency solely to silence a
test-runner-only warning, and not globally suppressed — documented narrowly
in `pyproject.toml`'s `[tool.pytest.ini_options]` so it stays visible if
Starlette's plans change.

## Limitations and next evaluation steps

- No temporal holdout (the source data has no timestamp field) — a real
  deployment should validate against genuinely future sessions, not just a
  random held-out split.
- Calibration and the uncertainty band are fit and evaluated on a single
  dataset from one (undisclosed) source; they have not been validated
  against a second, independent population.
- The MLP's improvement over the baseline is small (+0.014 PR-AUC on test)
  — worth re-checking with a larger or more diverse dataset before treating
  the architecture choice as settled.
- Next evaluation step, if continued: a documented sensitivity analysis
  including `PageValues` (clearly labeled as leakage-risk, never used in
  the deployment-safe path) to quantify how much predictive signal is being
  deliberately left on the table for safety.
- Segment-level metrics (by `TrafficType`, `Month`) are noisy for small
  segments (some under 10 test sessions) — not a basis for per-segment
  threshold tuning without more data.
