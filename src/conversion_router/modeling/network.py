"""PyTorch tabular MLP and dataset wrapper.

Architecture, seeding, and dataset construction only. Training lives in
train.py; this module must stay importable and testable without a GPU or
any downloaded data.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

MLP_VERSION = "1.0.0"
HIDDEN_SIZES = (128, 64, 32)
DROPOUT = 0.3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ConversionMLP(nn.Module):
    """Tabular MLP: input -> 128 -> 64 -> 32 -> 1 logit. No sigmoid inside."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...] = HIDDEN_SIZES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class TabularDataset(Dataset):
    """Wraps a preprocessed feature matrix and label vector as float32 tensors."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def predict_proba(model: ConversionMLP, features: np.ndarray) -> np.ndarray:
    """Sigmoid of the model's logits for a full feature matrix. No grad."""
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32)
        logits = model(x)
        return torch.sigmoid(logits).numpy()
