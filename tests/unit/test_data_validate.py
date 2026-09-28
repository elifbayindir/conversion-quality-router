from pathlib import Path

import pandas as pd
import pytest

from conversion_router.data.load import load_raw_dataset
from conversion_router.data.validate import (
    EXPECTED_COLUMNS,
    DataQualityReport,
    SchemaValidationError,
    validate_raw_schema,
)

SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "sample" / "online_shoppers_sample.csv"


def _valid_row(**overrides) -> dict:
    row = {
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 1,
        "ProductRelated_Duration": 0.0,
        "BounceRates": 0.2,
        "ExitRates": 0.2,
        "PageValues": 0.0,
        "SpecialDay": 0.0,
        "Month": "Feb",
        "OperatingSystems": 1,
        "Browser": 1,
        "Region": 1,
        "TrafficType": 1,
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
        "Revenue": False,
    }
    row.update(overrides)
    return row


def _valid_frame(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame([_valid_row() for _ in range(n)])[list(EXPECTED_COLUMNS)]


def test_valid_synthetic_frame_passes_and_reports_correctly():
    df = _valid_frame(n=4)
    df.loc[0, "Revenue"] = True

    report = validate_raw_schema(df)

    assert isinstance(report, DataQualityReport)
    assert report.n_rows == 4
    assert report.n_columns == 18
    assert report.target_positive == 1
    assert report.target_negative == 3
    assert report.n_missing_values == 0


def test_real_sample_fixture_passes_validation():
    df = pd.read_csv(SAMPLE_PATH)

    report = validate_raw_schema(df)

    assert report.n_rows == 50
    assert report.target_positive == 15
    assert report.target_negative == 35


def test_missing_column_is_rejected():
    df = _valid_frame().drop(columns=["PageValues"])
    with pytest.raises(SchemaValidationError, match="Column mismatch"):
        validate_raw_schema(df)


def test_extra_column_is_rejected():
    df = _valid_frame()
    df["UnexpectedColumn"] = 1
    with pytest.raises(SchemaValidationError, match="Column mismatch"):
        validate_raw_schema(df)


def test_missing_values_are_rejected():
    df = _valid_frame(n=3)
    df.loc[1, "BounceRates"] = None
    with pytest.raises(SchemaValidationError, match="missing values"):
        validate_raw_schema(df)


def test_negative_duration_is_rejected():
    df = _valid_frame()
    df.loc[0, "ProductRelated_Duration"] = -5.0
    with pytest.raises(SchemaValidationError, match="negative values"):
        validate_raw_schema(df)


def test_out_of_range_bounce_rate_is_rejected():
    df = _valid_frame()
    df.loc[0, "BounceRates"] = 1.5
    with pytest.raises(SchemaValidationError, match=r"outside \[0, 1\]"):
        validate_raw_schema(df)


def test_invalid_month_category_is_rejected():
    df = _valid_frame()
    df.loc[0, "Month"] = "Xxx"
    with pytest.raises(SchemaValidationError, match="invalid categories"):
        validate_raw_schema(df)


def test_invalid_operating_system_code_is_rejected():
    df = _valid_frame()
    df.loc[0, "OperatingSystems"] = 999
    with pytest.raises(SchemaValidationError, match="invalid categories"):
        validate_raw_schema(df)


def test_non_boolean_revenue_is_rejected():
    df = _valid_frame()
    df["Revenue"] = df["Revenue"].astype(object)
    df.loc[0, "Revenue"] = "yes"
    with pytest.raises(SchemaValidationError, match="non-boolean values"):
        validate_raw_schema(df)


def test_duplicates_are_reported_not_rejected():
    df = pd.concat([_valid_frame(n=1), _valid_frame(n=1)], ignore_index=True)
    report = validate_raw_schema(df)
    assert report.n_duplicate_rows == 1


def test_load_raw_dataset_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_raw_dataset(missing)


def test_load_raw_dataset_reads_sample_fixture():
    df = load_raw_dataset(SAMPLE_PATH)
    assert list(df.columns) == list(EXPECTED_COLUMNS)
    assert df["Revenue"].dtype == bool
