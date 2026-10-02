import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import operational_impact as oi  # noqa: E402

SCRIPT = PROJECT_ROOT / "scripts" / "operational_impact.py"

needs_frozen_predictions = pytest.mark.skipif(
    not oi.TEST_PREDICTIONS.exists(),
    reason="Frozen test predictions not present (git-ignored); run the P3 pipeline first.",
)


@pytest.fixture(scope="module")
def results():
    if not oi.TEST_PREDICTIONS.exists() and not oi.VALIDATION_PREDICTIONS.exists():
        pytest.skip("No prediction files present.")
    return oi.build_results()


def _tree_hash(*roots: Path) -> str:
    digest = hashlib.sha256()
    for root in roots:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            digest.update(str(path).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@needs_frozen_predictions
def test_committed_outputs_are_current(results):
    assert oi.OUT_JSON.read_text(encoding="utf-8") == oi.results_json(results)
    assert oi.OUT_MD.read_text(encoding="utf-8") == oi.render_markdown(results)
    assert oi.OUT_FIGURE.exists()


@needs_frozen_predictions
def test_frozen_source_passes_every_integrity_check(results):
    assert results["analysis_type"] == oi.ANALYSIS_FROZEN_TEST
    checks = results["source"]["integrity_checks"]
    assert checks and all(checks.values())
    assert results["source"]["predictions_sha256"] == oi.sha256_of(oi.TEST_PREDICTIONS)


@needs_frozen_predictions
def test_metrics_reconcile_with_the_frozen_evaluation(results):
    evaluation = json.loads(oi.TEST_EVALUATION.read_text(encoding="utf-8"))
    cm = evaluation["mlp"]["metrics"]["confusion_matrix"]
    m = results["metrics"]
    assert m["n_sessions"] == evaluation["n_test_rows"]
    assert m["n_conversions"] == cm["true_positive"] + cm["false_negative"]
    assert sum(r["sessions"] for r in m["routes"]) == m["n_sessions"]
    assert sum(r["conversions"] for r in m["routes"]) == m["n_conversions"]
    consistency = m["policy_consistency"]
    assert consistency["uncertain_sessions"] == evaluation["mlp"]["n_uncertain"]
    assert (
        consistency["uncertain_sessions_routed_to_human_review"]
        == consistency["uncertain_sessions"]
    )


def test_analysis_is_deterministic(results):
    assert oi.build_results() == results


def test_analysis_writes_nothing_to_frozen_inputs():
    roots = (PROJECT_ROOT / "artifacts", PROJECT_ROOT / "data")
    before = _tree_hash(*roots)
    oi.build_results()
    assert _tree_hash(*roots) == before


def test_policy_parameters_are_frozen_inputs_not_choices(results):
    metadata = json.loads(oi.MODEL_METADATA.read_text(encoding="utf-8"))
    policy = results["policy"]
    assert policy["decision_threshold"] == metadata["threshold_selection"]["threshold"]
    assert policy["uncertainty_band"] == metadata["calibration"]["uncertainty_band"]
    assert "none" in policy["parameters_chosen_here"]


def test_no_financial_fields_in_results(results):
    def keys(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                yield key
                yield from keys(value)
        elif isinstance(obj, list):
            for item in obj:
                yield from keys(item)

    forbidden = ("revenue", "profit", "roi", "uplift", "value_usd", "gmv")
    assert not [k for k in keys(results) if any(f in k.lower() for f in forbidden)]


def test_script_never_touches_raw_data_split_or_model_execution():
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    for module in ("conversion_router.modeling.inference", "conversion_router.modeling.train",
                   "conversion_router.modeling.network", "conversion_router.data.split",
                   "conversion_router.data.load", "torch"):
        assert module not in imported
    for path_fragment in ("split_indices", "online_shoppers_intention", "data/raw"):
        assert path_fragment not in source
    for out in (oi.OUT_JSON, oi.OUT_MD, oi.OUT_FIGURE):
        assert out.relative_to(PROJECT_ROOT).parts[0] == "docs"


def test_missing_frozen_predictions_fall_back_to_labelled_validation(monkeypatch, tmp_path):
    if not oi.VALIDATION_PREDICTIONS.exists():
        pytest.skip("Validation predictions not present.")
    monkeypatch.setattr(oi, "TEST_PREDICTIONS", tmp_path / "absent.csv")
    results = oi.build_results()
    assert results["analysis_type"] == oi.ANALYSIS_VALIDATION
    markdown = oi.render_markdown(results)
    assert "Validation Operational Analysis (not final test evidence)" in markdown
    assert "optimistic" in markdown


@needs_frozen_predictions
def test_inconsistent_frozen_source_is_refused(monkeypatch, tmp_path):
    evaluation = json.loads(oi.TEST_EVALUATION.read_text(encoding="utf-8"))
    evaluation["mlp"]["n_uncertain"] += 1
    tampered = tmp_path / "test_evaluation.json"
    tampered.write_text(json.dumps(evaluation), encoding="utf-8")
    monkeypatch.setattr(oi, "TEST_EVALUATION", tampered)
    with pytest.raises(oi.SourceIntegrityError):
        oi.build_results()
