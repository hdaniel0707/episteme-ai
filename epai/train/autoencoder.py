from pathlib import Path
from typing import Optional
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).parent.parent.parent


class AE(nn.Module):
    def __init__(self, input_dim: int = 768, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class AERegressor:
    """Sklearn-compatible wrapper: trains AE for reconstruction, then fits a
    Ridge regression head on the latent space to predict y."""

    def __init__(
        self,
        latent_dim: int = 128,
        max_iter: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        tol: float = 1e-4,
        n_iter_no_change: int = 10,
        random_state: Optional[int] = None,
    ):
        self.latent_dim = latent_dim
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.ae_: Optional[AE] = None
        self.head_: Optional[Ridge] = None
        self.n_iter_: int = 0
        self.train_losses_: list[float] = []
        self.val_losses_: list[float] = []

    def _reconstruction_loss(
        self, x: torch.Tensor, x_hat: torch.Tensor
    ) -> torch.Tensor:
        mse = F.mse_loss(x_hat, x)
        cos = (1 - F.cosine_similarity(x_hat, x)).mean()
        return 0.7 * mse + 0.3 * cos

    def fit(
        self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None
    ) -> "AERegressor":
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        X_val_t = (
            torch.tensor(X_val, dtype=torch.float32).to(device)
            if X_val is not None
            else None
        )

        self.ae_ = AE(input_dim=X_t.shape[1], latent_dim=self.latent_dim).to(device)
        optimizer = torch.optim.AdamW(
            self.ae_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loader = DataLoader(
            TensorDataset(X_t), batch_size=self.batch_size, shuffle=True
        )

        best_loss = float("inf")
        no_improve = 0
        self.train_losses_ = []
        self.val_losses_ = []

        self.ae_.train()
        pbar = tqdm(range(self.max_iter), desc="Epochs", leave=False)
        for epoch in pbar:
            epoch_loss = 0.0
            for (batch,) in loader:
                x_hat, _ = self.ae_(batch)
                loss = self._reconstruction_loss(batch, x_hat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)
            self.train_losses_.append(epoch_loss)

            val_loss = None
            if X_val_t is not None:
                self.ae_.eval()
                with torch.no_grad():
                    x_hat_val, _ = self.ae_(X_val_t)
                    val_loss = self._reconstruction_loss(X_val_t, x_hat_val).item()
                self.val_losses_.append(val_loss)
                self.ae_.train()

            postfix = {"train": f"{epoch_loss:.4f}"}
            if val_loss is not None:
                postfix["val"] = f"{val_loss:.4f}"
            pbar.set_postfix(postfix)

            if best_loss - epoch_loss > self.tol:
                best_loss = epoch_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.n_iter_no_change:
                    self.n_iter_ = epoch + 1
                    break
        else:
            self.n_iter_ = self.max_iter

        self.ae_.eval()
        with torch.no_grad():
            _, Z = self.ae_(X_t)

        self.head_ = Ridge()
        self.head_.fit(Z.cpu().numpy(), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.ae_ is not None and self.head_ is not None, "Call fit() first."
        device = next(self.ae_.parameters()).device
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        self.ae_.eval()
        with torch.no_grad():
            _, Z = self.ae_(X_t)
        return self.head_.predict(Z.cpu().numpy())
