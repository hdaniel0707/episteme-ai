"""A small feed-forward classifier trained from scratch in PyTorch."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .base import ClfAlgorithm


class _MLP(nn.Module):
    """Plain MLP: (Linear → ReLU → Dropout) per hidden layer, then a logit head."""

    def __init__(
        self, in_dim: int, hidden_dims: list[int], n_classes: int, dropout: float
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchMLPAlg(ClfAlgorithm):
    """A custom PyTorch MLP: standardise, then train an MLP under cross-entropy.

    Runs on CUDA when available, otherwise CPU. Params (all optional):
      * ``hidden_dims`` -- layer widths (default ``[256, 64]``);
      * ``dropout`` -- dropout probability (default 0.1);
      * ``epochs`` -- max epochs (default 100);
      * ``batch_size`` -- minibatch size (default 64);
      * ``lr`` -- AdamW learning rate (default 1e-3);
      * ``weight_decay`` -- AdamW weight decay (default 1e-4);
      * ``patience`` -- early-stop after this many epochs without a training-loss
        improvement (default 10; set 0 to disable and always run ``epochs``).
    """

    name = "torch_mlp"
    N_CLASSES = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPAlg":
        p = self.params
        hidden_dims = [int(h) for h in p.get("hidden_dims", [256, 64])]
        dropout = float(p.get("dropout", 0.1))
        epochs = int(p.get("epochs", 100))
        batch_size = int(p.get("batch_size", 64))
        lr = float(p.get("lr", 1e-3))
        weight_decay = float(p.get("weight_decay", 1e-4))
        patience = int(p.get("patience", 10))

        # Seed torch and numpy so a run is reproducible for a given seed.
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X).astype(np.float32)

        X_t = torch.from_numpy(Xs).to(self.device)
        y_t = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(self.device)

        self.model = _MLP(X_t.shape[1], hidden_dims, self.N_CLASSES, dropout).to(
            self.device
        )
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True
        )

        best = float("inf")
        no_improve = 0
        self.model.train()
        for _ in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            epoch_loss /= max(len(loader), 1)

            if patience:  # early stop on training loss (no val split per source)
                if best - epoch_loss > 1e-4:
                    best = epoch_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X).astype(np.float32)
        X_t = torch.from_numpy(Xs).to(self.device)
        self.model.eval()
        logits = self.model(X_t)
        return logits.argmax(dim=1).cpu().numpy()

    def save(self, path: Path) -> None:
        # Move to CPU first so the pickle can be loaded on a machine without a GPU.
        self.model.to("cpu")
        self.device = torch.device("cpu")
        joblib.dump(self, path)
