"""Regression tests proving train_mlp default behavior is unchanged."""

from __future__ import annotations

import inspect

import numpy as np

from conversion_router.modeling.network import DROPOUT, HIDDEN_SIZES, ConversionMLP
from conversion_router.modeling.train import train_mlp


def test_train_mlp_default_hidden_sizes_and_dropout():
    """Default call produces a model with the v1 architecture."""
    sig = inspect.signature(train_mlp)
    assert sig.parameters["hidden_sizes"].default == HIDDEN_SIZES
    assert sig.parameters["dropout"].default == DROPOUT


def test_train_mlp_default_produces_v1_architecture():
    """Calling train_mlp without hidden_sizes/dropout builds (128, 64, 32) dropout=0.3."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 10)).astype(np.float32)
    y = rng.integers(0, 2, size=40).astype(np.float32)
    result = train_mlp(X, y, X, y, seed=0, max_epochs=2, patience=2)
    assert isinstance(result.model, ConversionMLP)
    assert result.model.hidden_sizes == (128, 64, 32)
    assert result.model.dropout == 0.3


def test_train_mlp_custom_architecture():
    """Passing custom hidden_sizes/dropout changes the model architecture."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 10)).astype(np.float32)
    y = rng.integers(0, 2, size=40).astype(np.float32)
    result = train_mlp(
        X, y, X, y, seed=0, max_epochs=2, patience=2,
        hidden_sizes=(64, 32), dropout=0.2,
    )
    assert result.model.hidden_sizes == (64, 32)
    assert result.model.dropout == 0.2
