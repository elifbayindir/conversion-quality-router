# Leakage Audit and Feature Contract

This document reviews, feature by feature, whether each predictor in the raw
Online Shoppers Purchasing Intention dataset is genuinely available at the
moment an operational decision would be made (i.e., before or independent of
knowing whether the session converts), and records the resulting
deployment-safe feature list. The machine-readable version of this contract
is `artifacts/metadata/feature_contract.json`
(`feature_contract_version: "1.0.0"`).

## Review method

For each feature we ask: could this value be known and computed from
information available strictly *before or independent of* the session's
purchase outcome? A feature fails this test if its value is itself derived
from conversion/transaction outcomes (across the dataset or historically),
even if it is technically present in the row before the current session
resolves.

## Per-feature review

| Feature | Type | Available at decision time? | Rationale |
|---|---|---|---|
| `Administrative` | numeric | Yes | Running count of admin pages visited so far in the session; a session-tracking counter, not outcome-derived. |
| `Administrative_Duration` | numeric | Yes | Running time on admin pages so far; same reasoning. |
| `Informational` | numeric | Yes | Running count of informational pages visited; session-tracking counter. |
| `Informational_Duration` | numeric | Yes | Running time on informational pages; session-tracking counter. |
| `ProductRelated` | numeric | Yes | Running count of product pages visited; session-tracking counter. |
| `ProductRelated_Duration` | numeric | Yes | Running time on product pages; session-tracking counter. |
| `BounceRates` | numeric | Yes | Per-page historical aggregate metric (fraction of visitors who leave immediately), computed from the page's general traffic history, not from this session's outcome. |
| `ExitRates` | numeric | Yes | Per-page historical aggregate metric (fraction of pageviews that were the last in the session), same reasoning as `BounceRates`. |
| `PageValues` | numeric | **No — excluded** | Computed by the analytics source as the average value of a page relative to transactions completed after viewing it. It is derived from downstream conversion/transaction outcomes, so using it treats the model's own target signal as an input. See decision below. |
| `SpecialDay` | numeric | Yes | Calendar-derived closeness to a known special shopping day; known in advance, independent of this session's outcome. |
| `Month` | categorical | Yes | Calendar fact, known at session start. |
| `OperatingSystems` | categorical | Yes | Captured from the client at session start (user agent). |
| `Browser` | categorical | Yes | Captured from the client at session start (user agent). |
| `Region` | categorical | Yes | Captured from the client/network at session start. |
| `TrafficType` | categorical | Yes | Captured from referrer/campaign metadata at session start. Anonymized integer code only — per decision D06, treated strictly as a channel/source proxy; this project does not claim to measure or causally test platform dependence from it. |
| `VisitorType` | categorical | Yes | Derived from cookie/session history (new vs. returning vs. other), known at session start. |
| `Weekend` | boolean | Yes | Calendar fact, known at session start. |

## `PageValues` decision

**Decision: excluded from the deployment-safe feature set by default.**

- The dataset's own documentation describes `PageValues` as computed from the
  average value of a page *before completing an e-commerce transaction* —
  i.e., it is constructed using transaction/conversion information, which is
  exactly what the model is trying to predict.
- Its availability at true decision time (before the session's outcome is
  known) is not proven from the documentation available to this project.
- Per the project's constraints (`PROJECT_BLUEPRINT.md` Section 4.4, and this
  repository's operating constraints), `PageValues` is therefore excluded
  from the primary, deployment-safe model. If time permits later, a clearly
  labeled *sensitivity analysis* may report the metric difference when
  `PageValues` is included, with the leakage caveat stated explicitly next to
  any such number. It will never be included in the deployment-safe model,
  the API's deployment prediction path, or anything the agent/automation
  layer consumes.

## Deployment-safe feature list (v1.0.0)

16 features: `Administrative`, `Administrative_Duration`, `Informational`,
`Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`,
`BounceRates`, `ExitRates`, `SpecialDay`, `Month`, `OperatingSystems`,
`Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`.

Excluded: `PageValues` (leakage risk, see above).

Target (not a feature): `Revenue`.

This list and its version are the single source of truth consumed by the
preprocessing pipeline (T201) and the model's `feature_contract_version`
field in its output schema (`PROJECT_BLUEPRINT.md` Section 6). Any future
change to this list requires a new `feature_contract_version`.

## Duplicate-row split leakage (found and fixed)

An independent check of the initial row-index-based split found that the raw
dataset contains 125 duplicate rows (76 groups of rows with identical values
across the 16 deployment-safe features), and that a naive random/stratified
split assigns rows from the same duplicate group to different partitions —
letting a training-set row's exact duplicate leak into validation or test.
None of the 76 groups have conflicting `Revenue` labels in this dataset (i.e.
duplicated feature rows always agree on the outcome), but relying on that
being true forever, without checking it, would be a silent risk.

**Decision (see `DECISIONS.md` D07):** the split is now group-aware. Every
row is assigned a deterministic SHA-256 fingerprint over the 16
deployment-safe features (`src/conversion_router/data/split.py`,
`compute_fingerprints`), and the 70/15/15 stratified split is performed over
*fingerprint groups*, not raw row indices, so every row sharing a fingerprint
is guaranteed to land in exactly one partition. Before splitting, the code
checks every fingerprint group for label agreement and raises
`ConflictingDuplicateLabelsError` (refusing to produce a split) if any future
data refresh introduces a group with disagreeing `Revenue` values. The
resulting `artifacts/metadata/split_summary.json` records
`n_duplicate_fingerprint_groups`, `n_duplicate_extra_rows`,
`n_conflicting_label_groups`, and `n_cross_split_fingerprint_groups` (which
must be 0) as ongoing, re-checkable evidence — not a one-time claim.
