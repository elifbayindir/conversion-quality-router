import numpy as np
import torch

from conversion_router.modeling.network import predict_proba
from conversion_router.modeling.train import (
    compute_pos_weight,
    load_checkpoint,
    save_checkpoint,
    train_mlp,
)


def _separable_data(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    logits = 2.5 * X[:, 0] - 1.0
    prob = 1 / (1 + np.exp(-logits))
    y = (rng.random(n) < prob).astype(int)
    return X, y


def test_compute_pos_weight_matches_class_ratio():
    y = np.array([0, 0, 0, 0, 1, 1])  # 4 negative, 2 positive
    weight = compute_pos_weight(y)
    assert weight.item() == 2.0


def test_training_learns_a_meaningfully_separable_signal():
    X, y = _separable_data(n=400, seed=1)
    X_train, X_val = X[:300], X[300:]
    y_train, y_val = y[:300], y[300:]

    result = train_mlp(X_train, y_train, X_val, y_val, max_epochs=25, patience=5)

    assert result.best_val_pr_auc > 0.6
    assert len(result.history) <= 25
    assert result.best_epoch >= 1


def test_training_is_reproducible_with_fixed_seed():
    X, y = _separable_data(n=200, seed=2)
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]

    result_a = train_mlp(X_train, y_train, X_val, y_val, seed=7, max_epochs=10, patience=3)
    result_b = train_mlp(X_train, y_train, X_val, y_val, seed=7, max_epochs=10, patience=3)

    assert result_a.best_val_pr_auc == result_b.best_val_pr_auc
    for p_a, p_b in zip(result_a.model.parameters(), result_b.model.parameters(), strict=True):
        assert torch.allclose(p_a, p_b)


def test_early_stopping_halts_before_max_epochs_on_flat_signal():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 4)).astype(np.float32)
    y = (rng.random(200) < 0.5).astype(int)  # no real signal in X
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]

    result = train_mlp(X_train, y_train, X_val, y_val, max_epochs=100, patience=3)

    assert len(result.history) < 100


def test_checkpoint_save_and_load_preserves_predictions(tmp_path):
    X, y = _separable_data(n=200, seed=4)
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]

    result = train_mlp(X_train, y_train, X_val, y_val, max_epochs=10, patience=3)
    expected = predict_proba(result.model, X_val)

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(result.model, path)
    reloaded = load_checkpoint(path)
    actual = predict_proba(reloaded, X_val)

    assert np.allclose(expected, actual)
    assert reloaded.input_dim == result.model.input_dim


def test_load_missing_checkpoint_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does_not_exist.pt")
