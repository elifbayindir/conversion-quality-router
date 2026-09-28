import numpy as np
import pytest

from conversion_router.modeling.baseline import (
    SEED,
    load_baseline_model,
    predict_positive_proba,
    save_baseline_model,
    train_baseline,
)


def _synthetic_training_data(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    # Make the label weakly dependent on the first feature so training is meaningful.
    logits = 1.5 * X[:, 0] - 2.0
    prob = 1 / (1 + np.exp(-logits))
    y = (rng.random(n) < prob).astype(int)
    return X, y


def test_train_baseline_is_reproducible_with_fixed_seed():
    X, y = _synthetic_training_data()
    model_a = train_baseline(X, y, seed=SEED)
    model_b = train_baseline(X, y, seed=SEED)

    proba_a = predict_positive_proba(model_a, X)
    proba_b = predict_positive_proba(model_b, X)
    assert np.allclose(proba_a, proba_b)


def test_predicted_probabilities_are_in_unit_interval():
    X, y = _synthetic_training_data()
    model = train_baseline(X, y)
    proba = predict_positive_proba(model, X)

    assert (proba >= 0).all()
    assert (proba <= 1).all()


def test_class_weight_balanced_is_applied():
    X, y = _synthetic_training_data()
    model = train_baseline(X, y)
    assert model.get_params()["class_weight"] == "balanced"


def test_save_and_load_baseline_model_roundtrip(tmp_path):
    X, y = _synthetic_training_data()
    model = train_baseline(X, y)
    expected = predict_positive_proba(model, X)

    path = tmp_path / "baseline.joblib"
    save_baseline_model(model, path)
    reloaded = load_baseline_model(path)
    actual = predict_positive_proba(reloaded, X)

    assert np.allclose(expected, actual)


def test_load_missing_baseline_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_baseline_model(tmp_path / "does_not_exist.joblib")
