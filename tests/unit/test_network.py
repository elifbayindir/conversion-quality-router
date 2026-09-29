import numpy as np
import torch

from conversion_router.modeling.network import (
    DROPOUT,
    HIDDEN_SIZES,
    ConversionMLP,
    TabularDataset,
    predict_proba,
    set_seed,
)


def test_forward_pass_output_shape_and_dtype():
    set_seed(42)
    model = ConversionMLP(input_dim=10)
    x = torch.randn(16, 10)

    out = model(x)

    assert out.shape == (16,)
    assert out.dtype == torch.float32


def test_forward_pass_single_sample():
    set_seed(42)
    model = ConversionMLP(input_dim=5)
    x = torch.randn(1, 5)

    out = model(x)

    assert out.shape == (1,)


def test_architecture_matches_configured_hidden_sizes_and_dropout():
    model = ConversionMLP(input_dim=8)

    linear_out_features = [
        layer.out_features for layer in model.network if isinstance(layer, torch.nn.Linear)
    ]
    dropout_rates = [
        layer.p for layer in model.network if isinstance(layer, torch.nn.Dropout)
    ]

    assert linear_out_features == [*HIDDEN_SIZES, 1]
    assert all(rate == DROPOUT for rate in dropout_rates)


def test_set_seed_gives_reproducible_initial_weights():
    set_seed(123)
    model_a = ConversionMLP(input_dim=6)
    set_seed(123)
    model_b = ConversionMLP(input_dim=6)

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.allclose(p_a, p_b)


def test_predict_proba_is_in_unit_interval():
    set_seed(42)
    model = ConversionMLP(input_dim=4)
    features = np.random.default_rng(0).normal(size=(20, 4)).astype(np.float32)

    proba = predict_proba(model, features)

    assert proba.shape == (20,)
    assert (proba >= 0).all()
    assert (proba <= 1).all()


def test_predict_proba_does_not_require_grad_and_leaves_model_in_eval_mode():
    set_seed(42)
    model = ConversionMLP(input_dim=4)
    model.train()
    features = np.zeros((3, 4), dtype=np.float32)

    predict_proba(model, features)

    assert model.training is False


def test_tabular_dataset_length_and_item_dtypes():
    features = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float64)
    labels = np.array([0, 1] * 5, dtype=np.int64)

    dataset = TabularDataset(features, labels)

    assert len(dataset) == 10
    x, y = dataset[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert x.shape == (3,)


def test_tabular_dataset_values_match_source_arrays():
    features = np.arange(12, dtype=np.float32).reshape(4, 3)
    labels = np.array([0, 1, 0, 1], dtype=np.float32)

    dataset = TabularDataset(features, labels)
    x, y = dataset[2]

    assert torch.allclose(x, torch.tensor(features[2]))
    assert y.item() == labels[2]
