from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conversion_router.data.split import (
    SEED,
    ConflictingDuplicateLabelsError,
    load_deployment_safe_feature_names,
    split_summary,
    stratified_split,
)

SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "sample" / "online_shoppers_sample.csv"


def _synthetic_frame(n: int = 2000, positive_rate: float = 0.15, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    revenue = rng.random(n) < positive_rate
    return pd.DataFrame({"feature": rng.normal(size=n), "Revenue": revenue})


def _split(df: pd.DataFrame, seed: int = SEED):
    return stratified_split(df, seed=seed, feature_columns=["feature"])


def test_split_is_disjoint_and_covers_all_rows():
    df = _synthetic_frame()
    split, _ = _split(df)

    train, val, test = set(split.train_index), set(split.validation_index), set(split.test_index)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(df.index)


def test_split_fractions_are_approximately_70_15_15():
    df = _synthetic_frame(n=5000)
    split, _ = _split(df)
    n = len(df)

    assert len(split.train_index) / n == pytest.approx(0.70, abs=0.01)
    assert len(split.validation_index) / n == pytest.approx(0.15, abs=0.01)
    assert len(split.test_index) / n == pytest.approx(0.15, abs=0.01)


def test_split_preserves_target_balance_per_partition():
    df = _synthetic_frame(n=5000, positive_rate=0.15)
    split, _ = _split(df)
    overall_rate = df["Revenue"].mean()

    for index in (split.train_index, split.validation_index, split.test_index):
        rate = df.loc[index, "Revenue"].mean()
        assert rate == pytest.approx(overall_rate, abs=0.03)


def test_split_is_reproducible_with_fixed_seed():
    df = _synthetic_frame()
    split_a, _ = _split(df, seed=SEED)
    split_b, _ = _split(df, seed=SEED)

    assert list(split_a.train_index) == list(split_b.train_index)
    assert list(split_a.validation_index) == list(split_b.validation_index)
    assert list(split_a.test_index) == list(split_b.test_index)


def test_different_seed_produces_a_different_split():
    df = _synthetic_frame()
    split_a, _ = _split(df, seed=SEED)
    split_b, _ = _split(df, seed=SEED + 1)

    assert list(split_a.train_index) != list(split_b.train_index)


def test_duplicate_feature_rows_stay_in_a_single_partition():
    df = _synthetic_frame(n=3000)
    # Inject a duplicate-feature group: 6 rows with identical feature value and label.
    dup_rows = pd.DataFrame({"feature": [123.456] * 6, "Revenue": [True] * 6})
    df = pd.concat([df, dup_rows], ignore_index=True)

    split, fingerprints = _split(df)

    dup_fingerprint = fingerprints.iloc[-1]
    assert (fingerprints == dup_fingerprint).sum() == 6

    partitions = set()
    if set(df.index[fingerprints == dup_fingerprint]) & set(split.train_index):
        partitions.add("train")
    if set(df.index[fingerprints == dup_fingerprint]) & set(split.validation_index):
        partitions.add("validation")
    if set(df.index[fingerprints == dup_fingerprint]) & set(split.test_index):
        partitions.add("test")
    assert len(partitions) == 1


def test_no_fingerprint_crosses_partition_boundaries():
    df = _synthetic_frame(n=3000)
    dup_rows = pd.DataFrame({"feature": [7.0] * 4, "Revenue": [False] * 4})
    df = pd.concat([df, dup_rows], ignore_index=True)

    split, fingerprints = _split(df)
    summary = split_summary(df, split, fingerprints)

    assert summary["duplicate_leakage_check"]["n_cross_split_fingerprint_groups"] == 0
    assert summary["duplicate_leakage_check"]["n_duplicate_fingerprint_groups"] >= 1


def test_conflicting_label_duplicate_group_raises():
    df = _synthetic_frame(n=500)
    conflicting_rows = pd.DataFrame({"feature": [42.0, 42.0], "Revenue": [True, False]})
    df = pd.concat([df, conflicting_rows], ignore_index=True)

    with pytest.raises(ConflictingDuplicateLabelsError):
        _split(df)


def test_split_summary_reports_zero_conflicts_on_clean_data():
    df = _synthetic_frame(n=2000)
    split, fingerprints = _split(df)
    summary = split_summary(df, split, fingerprints)

    assert summary["duplicate_leakage_check"]["n_conflicting_label_groups"] == 0


def test_load_deployment_safe_feature_names_matches_contract():
    names = load_deployment_safe_feature_names()
    assert len(names) == 16
    assert "PageValues" not in names
    assert "Revenue" not in names


def test_group_aware_split_works_on_real_sample_fixture():
    df = pd.read_csv(SAMPLE_PATH)
    split, fingerprints = stratified_split(df)
    summary = split_summary(df, split, fingerprints)

    assert summary["duplicate_leakage_check"]["n_cross_split_fingerprint_groups"] == 0
    assert summary["duplicate_leakage_check"]["n_conflicting_label_groups"] == 0
    train, val, test = set(split.train_index), set(split.validation_index), set(split.test_index)
    assert train | val | test == set(df.index)
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
