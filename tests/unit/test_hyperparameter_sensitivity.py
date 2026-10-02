"""Safeguard tests for the hyperparameter sensitivity study script.

These tests verify structural properties of the sensitivity script without
running the full 15-job study. They use inspection, mocking, and synthetic data.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "hyperparameter_sensitivity.py"


@pytest.fixture()
def sensitivity_module():
    spec = importlib.util.spec_from_file_location("hyperparameter_sensitivity", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_preregistered_seeds(sensitivity_module):
    assert sensitivity_module.PREREGISTERED_SEEDS == (17, 42, 73)


def test_preregistered_config_count(sensitivity_module):
    assert len(sensitivity_module.PREREGISTERED_CONFIGS) == 5


def test_preregistered_config_names(sensitivity_module):
    expected = {
        "v1_reference", "lower_dropout", "lower_learning_rate",
        "larger_batch", "compact_network",
    }
    assert set(sensitivity_module.PREREGISTERED_CONFIGS.keys()) == expected


def test_v1_reference_matches_frozen_defaults(sensitivity_module):
    ref = sensitivity_module.PREREGISTERED_CONFIGS["v1_reference"]
    assert ref["hidden_sizes"] == (128, 64, 32)
    assert ref["dropout"] == 0.30
    assert ref["learning_rate"] == 0.001
    assert ref["batch_size"] == 64
    assert ref["max_epochs"] == 100
    assert ref["patience"] == 10


def test_total_expected_runs(sensitivity_module):
    n_configs = len(sensitivity_module.PREREGISTERED_CONFIGS)
    n_seeds = len(sensitivity_module.PREREGISTERED_SEEDS)
    assert n_configs * n_seeds == 15


def test_script_never_imports_test_index():
    """The sensitivity script must not access split.test_index at module level."""
    source_path = SCRIPT_PATH
    with open(source_path) as f:
        source = f.read()
    assert "test_index" not in source
    assert "test_predictions" not in source
    assert "test_evaluation" not in source


def test_script_never_calls_save_checkpoint():
    source_path = SCRIPT_PATH
    with open(source_path) as f:
        source = f.read()
    assert "save_checkpoint" not in source


def test_script_never_writes_to_artifacts_models():
    source_path = SCRIPT_PATH
    with open(source_path) as f:
        source = f.read()
    assert "artifacts/models/" not in source
    assert "artifacts/preprocessors/" not in source


def test_run_single_produces_record(sensitivity_module):
    """run_single returns a RunRecord with correct fields."""
    rng = np.random.default_rng(99)
    X = rng.standard_normal((50, 10)).astype(np.float32)
    y = rng.integers(0, 2, size=50).astype(np.float32)

    config = {
        "hidden_sizes": (64, 32),
        "dropout": 0.3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "max_epochs": 3,
        "patience": 2,
    }
    record = sensitivity_module.run_single("test_config", config, 42, X, y, X, y)
    assert record.config_name == "test_config"
    assert record.seed == 42
    assert 0.0 <= record.best_val_pr_auc <= 1.0
    assert 0.0 <= record.val_roc_auc <= 1.0
    assert record.val_brier_raw >= 0.0
    assert record.val_log_loss >= 0.0
    assert record.best_epoch >= 1
    assert record.total_epochs >= 1


def test_aggregate_config(sensitivity_module):
    rng = np.random.default_rng(99)
    X = rng.standard_normal((50, 10)).astype(np.float32)
    y = rng.integers(0, 2, size=50).astype(np.float32)

    config = {
        "hidden_sizes": (64, 32),
        "dropout": 0.3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "max_epochs": 3,
        "patience": 2,
    }
    records = [
        sensitivity_module.run_single("test", config, seed, X, y, X, y)
        for seed in (17, 42, 73)
    ]
    agg = sensitivity_module.aggregate_config(records)
    assert "mean_val_pr_auc" in agg
    assert "std_val_pr_auc" in agg
    assert "median_val_pr_auc" in agg
    assert "mean_val_roc_auc" in agg
    assert "mean_val_brier_raw" in agg
    assert "mean_val_log_loss" in agg


def test_notable_candidate_rules_are_preregistered(sensitivity_module):
    rules = sensitivity_module.NOTABLE_CANDIDATE_RULES
    assert rules["min_pr_auc_improvement"] == 0.005
    assert rules["min_seed_wins"] == 2
    assert rules["max_std_increase"] == 0.005
    assert rules["max_brier_degradation"] == 0.002


def test_results_json_schema():
    """Verify the generated results JSON has the expected top-level keys."""
    import json
    with open("docs/hyperparameter_sensitivity_results.json") as f:
        results = json.load(f)

    required_keys = {
        "study_metadata", "reference_reproducibility",
        "per_run_results", "aggregate_results",
        "paired_comparisons", "candidate_outcome", "package_versions",
    }
    assert required_keys <= set(results.keys())
    assert len(results["per_run_results"]) == 15
    assert results["study_metadata"]["test_set_accessed"] is False
    assert results["study_metadata"]["candidate_checkpoints_saved"] is False
    assert results["study_metadata"]["v1_remains_authoritative"] is True
    assert results["candidate_outcome"]["promoted"] is False


def test_results_json_configs_match_preregistered():
    import json
    with open("docs/hyperparameter_sensitivity_results.json") as f:
        results = json.load(f)

    configs = set(results["study_metadata"]["configurations"].keys())
    expected = {
        "v1_reference", "lower_dropout", "lower_learning_rate",
        "larger_batch", "compact_network",
    }
    assert configs == expected
    assert results["study_metadata"]["seeds"] == [17, 42, 73]
