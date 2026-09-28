# Dataset

## Source

**Online Shoppers Purchasing Intention Dataset**
UCI Machine Learning Repository, dataset ID 468
DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)
License: CC BY 4.0
Source page: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

This is a real e-commerce clickstream dataset: 12,330 anonymized online
shopping sessions, each described by page-visit counts and durations,
behavioral signals (bounce/exit rates), commercial context (month, special
day proximity, visitor type, weekend flag), and technical context (operating
system, browser, region, traffic-source type). The target, `Revenue`, records
whether the session ended in a purchase. The dataset does not name or imply
any company, product, health, or wellness context — it is a general
e-commerce session dataset. See `.project-control/DECISIONS.md` decision D06
for the explicit domain-framing lock this project follows.

## Reproducing the download

```bash
python scripts/download_data.py
```

This downloads the dataset zip from the URL above, extracts
`online_shoppers_intention.csv` into `data/raw/` (git-ignored — not
committed), computes its SHA-256 checksum, and on first run records that
checksum plus source/license/DOI metadata into
`artifacts/metadata/dataset_manifest.json` (git-tracked). On every later run,
the script re-verifies the freshly downloaded file's checksum against that
recorded value and raises an error on mismatch, so silent upstream changes to
the file are caught rather than silently trained on.

See `artifacts/metadata/dataset_manifest.json` for the exact recorded
checksum, row count, and column count produced by the first successful run in
this repository.

## Committed sample fixture

`data/sample/online_shoppers_sample.csv` (git-tracked, 50 rows: 15 positive /
35 negative, matching the ~30/70 stratification of the full dataset's ~15.5%
positive rate) was drawn from the full downloaded dataset with
`pandas.DataFrame.sample(random_state=42)` per class, then shuffled with the
same seed. It exists so unit tests and demo fixtures do not require the full
dataset download, and is redistributed under the same CC BY 4.0 license as
the source.

## Known characteristics (per source documentation and blueprint Section 4)

- 12,330 sessions, 17 predictor variables, 1 target (`Revenue`).
- Class distribution: 10,422 negative (no purchase) / 1,908 positive
  (purchase) — imbalanced (~15.5% positive).
- No missing values reported by the source.
- `PageValues` is treated as a leakage-risk feature: its availability at
  decision time is not proven, so it is excluded from the deployment-safe
  feature set by default (see `.project-control/DECISIONS.md` and the
  leakage audit produced under `docs/`).
- `TrafficType` is an anonymized channel/source proxy only; this project does
  not claim to measure or causally test platform dependence from it (decision
  D06).
- There is no real timestamp field, so a temporal holdout is not the primary
  split design; this limitation is stated explicitly rather than implied
  away.

## What this dataset is used for in this project

A binary classifier (logistic regression baseline, then a PyTorch MLP)
predicts `Revenue` — the probability that a given session ends in a purchase.
Nothing beyond session-level purchase conversion probability is predicted or
claimed: not retention, not customer lifetime value, not startup survival,
and not a "sustainable scaling" outcome (decision D06).
