import numpy as np
import pandas as pd
import pytest

from conversion_router.data.preprocess import (
    BOOLEAN_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    fit_preprocessor,
    get_output_feature_names,
    load_preprocessor,
    save_preprocessor,
)
from conversion_router.data.split import load_deployment_safe_feature_names


def _row(**overrides) -> dict:
    row = {
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 1,
        "ProductRelated_Duration": 10.0,
        "BounceRates": 0.1,
        "ExitRates": 0.1,
        "SpecialDay": 0.0,
        "Weekend": False,
        "Month": "Feb",
        "OperatingSystems": 1,
        "Browser": 1,
        "Region": 1,
        "TrafficType": 1,
        "VisitorType": "Returning_Visitor",
    }
    row.update(overrides)
    return row


def _train_frame() -> pd.DataFrame:
    rows = [
        _row(Month="Feb", ProductRelated_Duration=10.0),
        _row(Month="Mar", ProductRelated_Duration=50.0, Weekend=True),
        _row(Month="May", ProductRelated_Duration=200.0, VisitorType="New_Visitor"),
        _row(Month="Nov", ProductRelated_Duration=5.0, TrafficType=3),
    ]
    return pd.DataFrame(rows)


def test_feature_columns_match_deployment_safe_feature_contract():
    module_columns = set(NUMERIC_COLUMNS) | set(BOOLEAN_COLUMNS) | set(CATEGORICAL_COLUMNS)
    contract_columns = set(load_deployment_safe_feature_names())
    assert module_columns == contract_columns


def test_fit_on_train_produces_stable_output_columns():
    train_df = _train_frame()
    preprocessor = fit_preprocessor(train_df)

    out_a = preprocessor.transform(train_df)
    out_b = preprocessor.transform(train_df)

    assert out_a.shape == out_b.shape
    assert np.allclose(out_a, out_b)
    names_a = get_output_feature_names(preprocessor)
    names_b = get_output_feature_names(preprocessor)
    assert names_a == names_b
    assert out_a.shape[1] == len(names_a)


def test_output_has_no_nans_and_is_numeric():
    train_df = _train_frame()
    preprocessor = fit_preprocessor(train_df)
    output = preprocessor.transform(train_df)

    assert np.issubdtype(output.dtype, np.floating)
    assert not np.isnan(output).any()


def test_unknown_category_at_transform_time_does_not_raise():
    train_df = _train_frame()  # Months seen: Feb, Mar, May, Nov
    preprocessor = fit_preprocessor(train_df)

    unseen_row = pd.DataFrame([_row(Month="Dec", TrafficType=99, OperatingSystems=77)])
    output = preprocessor.transform(unseen_row)  # must not raise

    assert output.shape[0] == 1
    assert not np.isnan(output).any()

    feature_names = get_output_feature_names(preprocessor)
    month_columns = [i for i, name in enumerate(feature_names) if name.startswith("Month_")]
    assert output[0, month_columns].sum() == 0  # unseen category -> all-zero one-hot


def test_transform_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    from conversion_router.data.preprocess import build_preprocessor

    preprocessor = build_preprocessor()
    with pytest.raises(NotFittedError):
        preprocessor.transform(_train_frame())


def test_save_and_load_preprocessor_roundtrip(tmp_path):
    train_df = _train_frame()
    preprocessor = fit_preprocessor(train_df)
    expected = preprocessor.transform(train_df)

    artifact_path = tmp_path / "preprocessor.joblib"
    save_preprocessor(preprocessor, artifact_path)
    reloaded = load_preprocessor(artifact_path)
    actual = reloaded.transform(train_df)

    assert np.allclose(expected, actual)
    assert get_output_feature_names(reloaded) == get_output_feature_names(preprocessor)


def test_load_missing_preprocessor_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_preprocessor(tmp_path / "does_not_exist.joblib")
