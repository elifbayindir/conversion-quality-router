"""Reproducible MLP training: class-aware loss, early stopping, fixed seed.

Training and validation-set monitoring only. The test set must never appear
here -- it is evaluated exactly once, later, in scripts/evaluate_test.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader

from conversion_router.modeling.network import (
    MLP_VERSION,
    ConversionMLP,
    TabularDataset,
    set_seed,
)

TRAIN_VERSION = "1.0.0"
SEED = 42
DEFAULT_MAX_EPOCHS = 100
DEFAULT_PATIENCE = 10
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 64

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "models" / "mlp_checkpoint.pt"


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_pr_auc: float


@dataclass
class TrainingResult:
    model: ConversionMLP
    history: list[EpochRecord]
    best_epoch: int
    best_val_pr_auc: float


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """BCEWithLogitsLoss pos_weight = n_negative / n_positive on the train split."""
    y_train = np.asarray(y_train)
    n_pos = float(np.sum(y_train == 1))
    n_neg = float(np.sum(y_train == 0))
    ratio = n_neg / n_pos if n_pos > 0 else 1.0
    return torch.tensor(ratio, dtype=torch.float32)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = SEED,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> TrainingResult:
    """Train with early stopping on validation PR-AUC. Returns the best checkpoint."""
    set_seed(seed)
    input_dim = X_train.shape[1]
    model = ConversionMLP(input_dim)

    train_dataset = TabularDataset(X_train, y_train)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )

    pos_weight = compute_pos_weight(y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_val_t = torch.as_tensor(X_val, dtype=torch.float32)
    y_val_t = torch.as_tensor(np.asarray(y_val), dtype=torch.float32)
    y_val_np = np.asarray(y_val)

    history: list[EpochRecord] = []
    best_val_pr_auc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = float(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = float(criterion(val_logits, y_val_t).item())
            val_proba = torch.sigmoid(val_logits).numpy()
        val_pr_auc = float(average_precision_score(y_val_np, val_proba))

        history.append(
            EpochRecord(
                epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_pr_auc=val_pr_auc
            )
        )

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(
        model=model, history=history, best_epoch=best_epoch, best_val_pr_auc=best_val_pr_auc
    )


def save_checkpoint(model: ConversionMLP, path: Path = DEFAULT_CHECKPOINT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": model.input_dim,
            "hidden_sizes": list(model.hidden_sizes),
            "dropout": model.dropout,
            "mlp_version": MLP_VERSION,
        },
        path,
    )


def load_checkpoint(path: Path = DEFAULT_CHECKPOINT_PATH) -> ConversionMLP:
    if not path.exists():
        raise FileNotFoundError(f"MLP checkpoint not found at {path}.")
    # weights_only=False: this artifact is produced locally by save_checkpoint above,
    # never loaded from an untrusted source, and includes non-tensor architecture metadata.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ConversionMLP(
        input_dim=payload["input_dim"],
        hidden_sizes=tuple(payload["hidden_sizes"]),
        dropout=payload["dropout"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
