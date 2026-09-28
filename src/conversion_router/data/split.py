"""Reproducible, group-aware stratified train/validation/test split.

Rows are grouped by a deterministic SHA-256 fingerprint over the 16
deployment-safe features (excludes ``PageValues`` and the ``Revenue``
target, per ``artifacts/metadata/feature_contract.json`` v1.0.0) before
splitting, so that rows sharing identical deployment-safe features can never
end up in different partitions. An independent check found 125 duplicate
rows across 76 fingerprint groups in the raw dataset (0 of them with
conflicting ``Revenue`` labels). This module refuses to produce a split if a
future data refresh ever introduces a conflicting-label duplicate group
(``ConflictingDuplicateLabelsError``), rather than silently splitting through
it.

Fixed seed and fractions are shared by every task that needs the split
(baseline, MLP training, evaluation). The test set is only ever produced
here and consumed once, at final evaluation time.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
TARGET_COLUMN = "Revenue"
FINGERPRINT_SEPARATOR = "␟"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLIT_INDICES_PATH = PROJECT_ROOT / "data" / "processed" / "split_indices.json"
FEATURE_CONTRACT_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "feature_contract.json"


class ConflictingDuplicateLabelsError(ValueError):
    """Raised when rows sharing a deployment-safe feature fingerprint disagree on Revenue."""


@dataclass(frozen=True)
class DatasetSplit:
    train_index: pd.Index
    validation_index: pd.Index
    test_index: pd.Index


def load_deployment_safe_feature_names(path: Path = FEATURE_CONTRACT_PATH) -> list[str]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    return [f["name"] for f in contract["deployment_safe_features"]]


def compute_fingerprints(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.Series:
    """Deterministic per-row SHA-256 fingerprint over feature_columns, in the given order."""
    canonical = df[list(feature_columns)].astype(str).agg(FINGERPRINT_SEPARATOR.join, axis=1)
    return canonical.apply(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())


def _conflicting_label_groups(df: pd.DataFrame, fingerprints: pd.Series) -> pd.Index:
    nunique_labels = df.groupby(fingerprints)[TARGET_COLUMN].nunique()
    return nunique_labels[nunique_labels > 1].index


def stratified_split(
    df: pd.DataFrame,
    seed: int = SEED,
    feature_columns: Sequence[str] | None = None,
) -> tuple[DatasetSplit, pd.Series]:
    """Group-aware stratified 70/15/15 split.

    Returns (split, fingerprints) where fingerprints is the per-row
    fingerprint series, kept around for duplicate/leakage reporting.
    Raises ConflictingDuplicateLabelsError if any fingerprint group has
    rows with different Revenue values.
    """
    if feature_columns is None:
        feature_columns = load_deployment_safe_feature_names()

    fingerprints = compute_fingerprints(df, feature_columns)

    conflicting = _conflicting_label_groups(df, fingerprints)
    if len(conflicting):
        raise ConflictingDuplicateLabelsError(
            f"{len(conflicting)} fingerprint group(s) have rows with different "
            f"'{TARGET_COLUMN}' labels despite identical deployment-safe features. "
            "Refusing to split until this is resolved."
        )

    group_label = df.groupby(fingerprints)[TARGET_COLUMN].first()
    unique_fp = group_label.index.to_numpy()

    train_fp, remainder_fp = train_test_split(
        unique_fp,
        test_size=(VALIDATION_FRACTION + TEST_FRACTION),
        stratify=group_label.loc[unique_fp].to_numpy(),
        random_state=seed,
    )
    val_fp, test_fp = train_test_split(
        remainder_fp,
        test_size=(TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION)),
        stratify=group_label.loc[remainder_fp].to_numpy(),
        random_state=seed,
    )

    train_fp_set, val_fp_set, test_fp_set = set(train_fp), set(val_fp), set(test_fp)
    if train_fp_set & val_fp_set or train_fp_set & test_fp_set or val_fp_set & test_fp_set:
        raise RuntimeError(
            "Internal split construction bug: a fingerprint group was assigned to "
            "more than one partition."
        )

    train_idx = df.index[fingerprints.isin(train_fp_set)]
    val_idx = df.index[fingerprints.isin(val_fp_set)]
    test_idx = df.index[fingerprints.isin(test_fp_set)]

    split = DatasetSplit(
        train_index=pd.Index(sorted(train_idx)),
        validation_index=pd.Index(sorted(val_idx)),
        test_index=pd.Index(sorted(test_idx)),
    )
    return split, fingerprints


def save_split_indices(split: DatasetSplit, path: Path = DEFAULT_SPLIT_INDICES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "train_index": [int(i) for i in split.train_index],
        "validation_index": [int(i) for i in split.validation_index],
        "test_index": [int(i) for i in split.test_index],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_split_indices(path: Path = DEFAULT_SPLIT_INDICES_PATH) -> DatasetSplit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DatasetSplit(
        train_index=pd.Index(payload["train_index"]),
        validation_index=pd.Index(payload["validation_index"]),
        test_index=pd.Index(payload["test_index"]),
    )


def _cross_split_fingerprint_count(split: DatasetSplit, fingerprints: pd.Series) -> int:
    membership: dict[str, set[str]] = {}
    for name, index in (
        ("train", split.train_index),
        ("validation", split.validation_index),
        ("test", split.test_index),
    ):
        for fp in fingerprints.loc[index].unique():
            membership.setdefault(fp, set()).add(name)
    return sum(1 for parts in membership.values() if len(parts) > 1)


def split_summary(df: pd.DataFrame, split: DatasetSplit, fingerprints: pd.Series) -> dict:
    def _part(name: str, index: pd.Index) -> dict:
        subset = df.loc[index, TARGET_COLUMN]
        n = len(subset)
        n_positive = int(subset.sum())
        return {
            "name": name,
            "n_rows": n,
            "fraction_of_total": round(n / len(df), 4),
            "n_positive": n_positive,
            "n_negative": n - n_positive,
            "positive_rate": round(n_positive / n, 4) if n else 0.0,
        }

    fp_counts = fingerprints.value_counts()
    duplicate_groups = fp_counts[fp_counts > 1]
    n_conflicting = len(_conflicting_label_groups(df, fingerprints))
    n_cross_split = _cross_split_fingerprint_count(split, fingerprints)

    return {
        "seed": SEED,
        "n_total_rows": len(df),
        "overall_positive_rate": round(float(df[TARGET_COLUMN].mean()), 4),
        "splits": [
            _part("train", split.train_index),
            _part("validation", split.validation_index),
            _part("test", split.test_index),
        ],
        "duplicate_leakage_check": {
            "fingerprint_basis": (
                "16 deployment-safe features (feature_contract.json v1.0.0); "
                "excludes PageValues and Revenue"
            ),
            "n_unique_fingerprints": int(fingerprints.nunique()),
            "n_duplicate_fingerprint_groups": int(len(duplicate_groups)),
            "n_duplicate_extra_rows": (
                int((duplicate_groups - 1).sum()) if len(duplicate_groups) else 0
            ),
            "n_conflicting_label_groups": n_conflicting,
            "n_cross_split_fingerprint_groups": n_cross_split,
        },
    }
