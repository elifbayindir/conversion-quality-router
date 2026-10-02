# Model Card: Conversion Quality Router — MLP

## Model purpose

Predicts the calibrated probability that a single e-commerce session (the
UCI Online Shoppers Purchasing Intention dataset) ends in a purchase
(`Revenue`), given 16 deployment-safe behavioral and contextual features
available at decision time. Nothing beyond this single, session-level
probability is predicted or claimed.

## Intended use

- Ranking and routing sessions by predicted conversion probability and
  model confidence, as one input to a downstream decision process that
  always keeps a human in the loop for anything beyond the most confidently
  negative case (see [`docs/agent-policy.md`](agent-policy.md) and
  [`docs/n8n-workflow.md`](n8n-workflow.md)).
- Research and demonstration of an uncertainty-aware, calibrated decision
  pipeline for session-level conversion prediction.

## Non-intended use

- **Not** for predicting customer retention, lifetime value (LTV), startup
  survival, or any "sustainable scaling" outcome.
- **Not** for individualized decisions that deny, restrict, or take an
  irreversible action against a real person without human review.
- **Not** validated for any population materially different from the
  training distribution (different market, catalog, or traffic mix) without
  re-evaluating calibration and threshold on that population's own
  held-out data.
- **Not** a causal or platform-dependence measurement tool — see
  [TrafficType limitation](#traffictype-limitation) below.

## Dataset and split

UCI Online Shoppers Purchasing Intention Dataset, 12,330 sessions, DOI
[10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q), CC BY 4.0. Seed `42`
throughout. Stratified 70/15/15 split **over SHA-256 fingerprint groups**
(not raw row indices), so duplicate-feature rows can never cross a split
boundary:

| Split | Rows | Positive rate |
|---|---|---|
| Train | 8,635 | 15.47% |
| Validation | 1,850 | 15.46% |
| Test | 1,845 | 15.50% |

Source: [`artifacts/metadata/split_summary.json`](../artifacts/metadata/split_summary.json).

## Feature contract

16 deployment-safe features (`feature_contract_version: "1.0.0"`):
`Administrative`, `Administrative_Duration`, `Informational`,
`Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`,
`BounceRates`, `ExitRates`, `SpecialDay`, `Month`, `OperatingSystems`,
`Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`. Source of
truth: [`artifacts/metadata/feature_contract.json`](../artifacts/metadata/feature_contract.json).

## Leakage exclusion

`PageValues` is excluded from the feature set. The dataset's own
documentation describes it as computed from the average value of a page
relative to transactions completed after viewing it — i.e., derived from
the same conversion outcome this model predicts. Its availability at true
decision time is not proven, so it is excluded from the deployment-safe
model entirely (never used in training, `/predict`, `/route`, or anything
the agent/automation layer consumes). Full per-feature review:
[`docs/leakage-audit.md`](leakage-audit.md).

## Duplicate-group handling

The raw dataset contains 125 duplicate rows across 76 fingerprint groups
(identical values across all 16 deployment-safe features). The split is
group-aware: every row is assigned a deterministic SHA-256 fingerprint, and
the stratified split is performed over fingerprint groups, guaranteeing
`n_cross_split_fingerprint_groups: 0` (checked on every split run, not
assumed) and `n_conflicting_label_groups: 0` (no duplicate group disagrees
on `Revenue`, but the split code would refuse to proceed if a future data
refresh introduced one). See
[`artifacts/metadata/split_summary.json`](../artifacts/metadata/split_summary.json).

## Preprocessing

`StandardScaler` for numeric features + one-hot encoding
(`handle_unknown="ignore"`) for categorical features, fit on the training
split only (71 output dimensions). The same fitted preprocessor is reused
for both the baseline and the MLP, so the two models are compared on
identical features.

## MLP architecture

`ConversionMLP`: fully connected 128 → 64 → 32 → 1, dropout 0.3 between
hidden layers. `BCEWithLogitsLoss` with `pos_weight` for class imbalance,
Adam optimizer, early stopping on validation PR-AUC (best epoch 5 of 15
trained, `n_epochs_run: 15`). Source:
[`src/conversion_router/modeling/network.py`](../src/conversion_router/modeling/network.py),
[`train.py`](../src/conversion_router/modeling/train.py).

## Class imbalance handling

The target is imbalanced (~15.5% positive). Handled via `pos_weight` in
`BCEWithLogitsLoss` for the MLP and `class_weight="balanced"` for the
logistic regression baseline — not via resampling, to keep validation and
test distributions representative of the real class balance.

## Calibration

Isotonic regression, fit on validation-split raw sigmoid outputs only
(never the test set). Brier score: 0.2126 (raw) → 0.1126 (calibrated) on
validation; 0.2232 → 0.1177 on test (confirms the calibration generalizes,
not just fits validation noise). Source:
[`artifacts/metadata/mlp_metadata.json`](../artifacts/metadata/mlp_metadata.json)
(`calibration` block) and
[`artifacts/metadata/test_evaluation.json`](../artifacts/metadata/test_evaluation.json).

## Frozen threshold

`0.12`, selected by a business-cost grid search on calibrated validation
probabilities only (false negative assumed 5x costlier than false positive
— a stated modeling assumption, not a measured business figure). Frozen in
[`artifacts/metadata/mlp_metadata.json`](../artifacts/metadata/mlp_metadata.json)
**before** any test-set access (`frozen_before_test_evaluation: true`).

## Uncertainty definition

Deterministic, not learned: `uncertain = |calibrated_probability -
threshold| < 0.05` (`uncertainty_band: 0.05`). 362 of 1,845 test sessions
(19.6%) were flagged uncertain. An uncertain prediction is never presented
as confident anywhere downstream — it deterministically forces
`HUMAN_REVIEW` at the agent layer, overriding whatever the LLM proposes
(see [`docs/agent-card.md`](agent-card.md)).

## Baseline and final metrics (test set, touched once)

| Metric | Baseline (LogReg, threshold 0.47) | MLP (calibrated, threshold 0.12) |
|---|---|---|
| PR-AUC | 0.302 | 0.316 |
| ROC-AUC | 0.7433 | 0.7543 |
| Precision | 0.2506 | 0.2514 |
| Recall | 0.7832 | 0.7797 |
| F1 | 0.3797 | 0.3802 |

The MLP's gain over the baseline is small (+0.014 PR-AUC, +0.011 ROC-AUC),
not transformative — reported as-is. Source:
[`artifacts/metadata/test_evaluation.json`](../artifacts/metadata/test_evaluation.json),
[`docs/test-evaluation-report.md`](test-evaluation-report.md).

## Error analysis

Representative false negatives (missed conversions) cluster at the
uncertainty boundary (`mlp_calibrated_proba ≈ 0.111`, flagged uncertain);
representative false positives are high-engagement sessions (long
`ProductRelated_Duration`, low bounce/exit rates) that did not convert
despite strong behavioral signals. Full examples with exact feature values:
[`docs/test-evaluation-report.md`](test-evaluation-report.md#representative-mlp-false-negatives-missed-conversions).

## Representativeness limitations

- No real timestamp field in the source data — no temporal holdout was
  possible; a real deployment would need to validate against genuinely
  future sessions, not just a held-out random split.
- Single dataset, single point in time, single (undisclosed) retailer
  context — calibration and threshold are not validated against a second,
  independent data source or population.
- Segment analysis (`docs/test-evaluation-report.md`) shows uneven
  `TrafficType` and `Month` group sizes (some segments have fewer than 10
  test sessions) — metrics for small segments are noisy and should not be
  over-interpreted.

## TrafficType limitation

`TrafficType` is an anonymized integer channel/source code with no
documented mapping to real channel names. This project treats it strictly
as a categorical proxy feature and makes **no claim** that it measures or
causally tests platform dependence, channel effectiveness, or any
comparison between traffic sources. See
[`docs/leakage-audit.md`](leakage-audit.md) for the full rationale.

## Artifact integrity / versioning

Every prediction is traceable to a frozen, hash-verified artifact set:

| Field | Value |
|---|---|
| `model_name` | `conversion_mlp` |
| `model_version` | `1.0.0` |
| `feature_contract_version` | `1.0.0` |
| `calibration_version` | `1.0.0` |
| `seed` | `42` |

`artifacts/metadata/mlp_metadata.json` records the SHA-256 hash of the
model checkpoint, preprocessor, and calibrator, plus exact package versions
(Python, PyTorch, scikit-learn, pandas, numpy) used to produce them. The
API's `/ready` endpoint verifies these hashes at startup and returns `503`
rather than serving predictions from a mismatched or missing artifact set
(`ArtifactIntegrityError`, `ArtifactUnavailableError` — see
[`src/conversion_router/modeling/inference.py`](../src/conversion_router/modeling/inference.py)).

## Ethical and operational limitations

- The model's output is a probability, not a certainty; calibration
  evidence (Brier score) supports treating it as a genuine probability
  estimate, but it is still a statistical estimate from a single imbalanced
  dataset, not a guarantee.
- No demographic or personally identifiable attributes are present in the
  dataset or used by the model; the dataset is anonymized session-level
  behavioral data.
- This model is one input to a human-reviewed decision process by design
  (see [Responsible-use / non-use cases](../README.md#responsible-use-non-use-cases)
  in the README) — it is not intended to operate as a fully autonomous
  gate on any consequential action.
