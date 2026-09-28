import json
from pathlib import Path

from conversion_router.data.validate import EXPECTED_COLUMNS

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts" / "metadata" / "feature_contract.json"
)


def _load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_feature_contract_is_versioned():
    contract = _load_contract()
    assert contract["feature_contract_version"] == "1.0.0"


def test_page_values_is_excluded_with_a_reason():
    contract = _load_contract()
    excluded_names = {f["name"] for f in contract["excluded_features"]}
    assert "PageValues" in excluded_names
    page_values = next(f for f in contract["excluded_features"] if f["name"] == "PageValues")
    assert page_values["reason"] == "LEAKAGE_RISK"


def test_deployment_safe_features_cover_all_raw_columns_except_target_and_excluded():
    contract = _load_contract()
    safe_names = {f["name"] for f in contract["deployment_safe_features"]}
    excluded_names = {f["name"] for f in contract["excluded_features"]}
    target = contract["target_column"]

    assert target == "Revenue"
    assert "PageValues" not in safe_names
    assert len(safe_names) == 16

    raw_predictors = set(EXPECTED_COLUMNS) - {target}
    assert safe_names | excluded_names == raw_predictors
    assert safe_names.isdisjoint(excluded_names)
