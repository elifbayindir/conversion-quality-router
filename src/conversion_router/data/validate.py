"""Schema and data-quality validation for the raw Online Shoppers dataset."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class SchemaValidationError(ValueError):
    """Raised when a dataframe violates a hard schema or data-quality constraint."""


EXPECTED_COLUMNS = (
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    "Revenue",
)

NONNEGATIVE_NUMERIC_COLUMNS = (
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "PageValues",
)

UNIT_INTERVAL_COLUMNS = ("BounceRates", "ExitRates", "SpecialDay")

CATEGORICAL_WHITELISTS: dict[str, set] = {
    "Month": {
        "Jan", "Feb", "Mar", "Apr", "May", "June",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    },
    "VisitorType": {"New_Visitor", "Returning_Visitor", "Other"},
    "OperatingSystems": set(range(1, 9)),
    "Browser": set(range(1, 14)),
    "Region": set(range(1, 10)),
    "TrafficType": set(range(1, 21)),
}

BOOLEAN_COLUMNS = ("Weekend", "Revenue")

TARGET_COLUMN = "Revenue"


@dataclass(frozen=True)
class DataQualityReport:
    n_rows: int
    n_columns: int
    n_duplicate_rows: int
    n_missing_values: int
    target_positive: int
    target_negative: int


def validate_raw_schema(df: pd.DataFrame) -> DataQualityReport:
    """Validate column names, dtypes, ranges, categories, and target values.

    Raises SchemaValidationError on any hard violation. Duplicate rows are
    reported, not rejected, since 125 exact duplicates are a documented
    characteristic of the raw source dataset rather than malformed input.
    """
    _validate_columns(df)
    _validate_missingness(df)
    _validate_numeric_ranges(df)
    _validate_categoricals(df)
    _validate_booleans(df)
    return _build_report(df)


def _validate_columns(df: pd.DataFrame) -> None:
    actual = tuple(df.columns)
    if actual != EXPECTED_COLUMNS:
        missing = set(EXPECTED_COLUMNS) - set(actual)
        extra = set(actual) - set(EXPECTED_COLUMNS)
        raise SchemaValidationError(
            f"Column mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _validate_missingness(df: pd.DataFrame) -> None:
    n_missing = int(df.isnull().sum().sum())
    if n_missing:
        by_column = df.isnull().sum()
        offending = by_column[by_column > 0].to_dict()
        raise SchemaValidationError(
            f"Unexpected missing values (source is documented as complete): {offending}"
        )


def _validate_numeric_ranges(df: pd.DataFrame) -> None:
    for column in NONNEGATIVE_NUMERIC_COLUMNS:
        series = pd.to_numeric(df[column], errors="coerce")
        if series.isnull().any():
            raise SchemaValidationError(f"Column {column!r} contains non-numeric values")
        if (series < 0).any():
            raise SchemaValidationError(f"Column {column!r} contains negative values")

    for column in UNIT_INTERVAL_COLUMNS:
        series = pd.to_numeric(df[column], errors="coerce")
        if series.isnull().any():
            raise SchemaValidationError(f"Column {column!r} contains non-numeric values")
        if (series < 0).any() or (series > 1).any():
            raise SchemaValidationError(f"Column {column!r} has values outside [0, 1]")


def _validate_categoricals(df: pd.DataFrame) -> None:
    for column, whitelist in CATEGORICAL_WHITELISTS.items():
        invalid = set(df[column].unique()) - whitelist
        if invalid:
            raise SchemaValidationError(f"Column {column!r} has invalid categories: {invalid}")


def _validate_booleans(df: pd.DataFrame) -> None:
    for column in BOOLEAN_COLUMNS:
        invalid = set(df[column].unique()) - {True, False}
        if invalid:
            raise SchemaValidationError(f"Column {column!r} has non-boolean values: {invalid}")


def _build_report(df: pd.DataFrame) -> DataQualityReport:
    target_counts = df[TARGET_COLUMN].value_counts()
    return DataQualityReport(
        n_rows=len(df),
        n_columns=len(df.columns),
        n_duplicate_rows=int(df.duplicated().sum()),
        n_missing_values=int(df.isnull().sum().sum()),
        target_positive=int(target_counts.get(True, 0)),
        target_negative=int(target_counts.get(False, 0)),
    )
