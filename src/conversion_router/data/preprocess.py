"""Train-only preprocessing: numeric scaling, boolean cast, categorical one-hot.

The pipeline must only ever be fit on the training partition. Validation and
test data are transformed with the already-fitted pipeline; unseen category
values at transform time are mapped to an all-zero one-hot vector rather than
raising, so the pipeline degrades safely on unexpected input.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

PREPROCESSOR_VERSION = "1.0.0"

NUMERIC_COLUMNS = (
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay",
)
BOOLEAN_COLUMNS = ("Weekend",)
CATEGORICAL_COLUMNS = (
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREPROCESSOR_PATH = PROJECT_ROOT / "artifacts" / "preprocessors" / "preprocessor.joblib"


def _cast_to_int(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(int)


def build_preprocessor() -> ColumnTransformer:
    """Construct an unfit ColumnTransformer. Call fit_preprocessor to fit it."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_COLUMNS)),
            (
                "boolean",
                FunctionTransformer(_cast_to_int, feature_names_out="one-to-one"),
                list(BOOLEAN_COLUMNS),
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_COLUMNS),
            ),
        ],
        verbose_feature_names_out=False,
    )


def fit_preprocessor(train_df: pd.DataFrame) -> ColumnTransformer:
    """Fit a new preprocessor on the training partition only.

    Callers must never pass validation or test data here.
    """
    preprocessor = build_preprocessor()
    preprocessor.fit(train_df)
    return preprocessor


def get_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def save_preprocessor(
    preprocessor: ColumnTransformer, path: Path = DEFAULT_PREPROCESSOR_PATH
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: Path = DEFAULT_PREPROCESSOR_PATH) -> ColumnTransformer:
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor artifact not found at {path}. Fit and save one first."
        )
    return joblib.load(path)
