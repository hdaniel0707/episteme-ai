"""AutoencoderEncoder - unsupervised reduction with a selectable activation (plan section 4.3).

Fit on train+val (no labels); the frozen encoder then maps any rows (incl. test / generated)
to the latent vector z. Reconstruction loss is 0.7*MSE + 0.3*cosine (same as the base repo).

The AE architecture mirrors epai.autoencoder.AE (768->512->256->k->256->512->768) but is
defined here so the hidden-layer activation is configurable from .env.workflow (ACTIVATION);
the read-only epai package is left untouched. Activation modules have no parameters, so the
checkpoint stores the activation name and reloads with the same one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Route the AE(k=…) bar through the workflow progress footer when a live multi-workflow
# display is running (so it stays a single separate bottom line instead of colliding with
# the per-workflow block); falls back to the real tqdm for standalone runs.
from workflow.src.iwf.progress import tqdm

from workflow.src.iwf.activations import make_activation
from workflow.src.iwf.encoder.base import Encoder
from workflow.src.iwf.training.trainer import configure_runtime


class _AE(nn.Module):
    """Symmetric autoencoder 768->512->256->k->256->512->768 with a selectable activation."""

    def __init__(self, input_dim: int, latent_dim: int, activation: str = "gelu"):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), make_activation(activation),
            nn.Linear(512, 256), nn.LayerNorm(256), make_activation(activation),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LayerNorm(256), make_activation(activation),
            nn.Linear(256, 512), nn.LayerNorm(512), make_activation(activation),
            nn.Linear(512, input_dim),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z), z


def _recon_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return 0.7 * F.mse_loss(x_hat, x) + 0.3 * (1 - F.cosine_similarity(x_hat, x)).mean()


class AutoencoderEncoder(Encoder):
    kind = "autoencoder"

    def __init__(self, latent_dim: int, input_dim: int | None = None,
                 max_epochs: int = 200, batch_size: int = 64, lr: float = 1e-3,
                 weight_decay: float = 1e-4, patience: int = 10, seed: int = 42,
                 activation: str = "gelu"):
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed
        self.activation = activation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ae_: _AE | None = None
        self.losses_: list[float] = []

    def _build_ae(self, input_dim: int, latent_dim: int) -> nn.Module:
        """The network to train. Subclasses can swap this (e.g. the epai AE)."""
        return _AE(input_dim, latent_dim, self.activation)

    def fit(self, X: np.ndarray) -> "AutoencoderEncoder":
        configure_runtime()
        torch.manual_seed(self.seed)
        self.input_dim = X.shape[1]
        self.ae_ = self._build_ae(self.input_dim, self.latent_dim).to(self.device)
        opt = torch.optim.AdamW(self.ae_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        loader = DataLoader(TensorDataset(Xt), batch_size=self.batch_size, shuffle=True)

        best, no_improve = float("inf"), 0
        self.ae_.train()
        for _ in tqdm(range(self.max_epochs), desc=f"AE(k={self.latent_dim})", leave=False):
            epoch = 0.0
            for (batch,) in loader:
                x_hat, _ = self.ae_(batch)
                loss = _recon_loss(batch, x_hat)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch += loss.item()
            epoch /= max(len(loader), 1)
            self.losses_.append(epoch)
            if best - epoch > 1e-4:
                best, no_improve = epoch, 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break
        self.ae_.eval()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.ae_ is not None, "Call fit() first."
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, z = self.ae_(Xt)
        return z.cpu().numpy().astype(np.float32)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Encode then decode X back to embedding space (the autoencoder's x_hat)."""
        assert self.ae_ is not None, "Call fit() first."
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            x_hat, _ = self.ae_(Xt)
        return x_hat.cpu().numpy().astype(np.float32)

    def reconstruction_stats(self, X: np.ndarray) -> dict:
        """How well encode->decode recovers the original embedding: MSE and R^2.

        R^2 = 1 - MSE / Var(X); 1.0 = perfect reconstruction, 0.0 = no better than the mean.
        """
        X = np.asarray(X, dtype=np.float32)
        x_hat = self.reconstruct(X)
        mse = float(np.mean((x_hat - X) ** 2))
        var = float(np.mean((X - X.mean(axis=0)) ** 2))
        r2 = float(1.0 - mse / var) if var > 0 else float("nan")
        return {"reconstruction_mse": mse, "reconstruction_r2": r2}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.ae_.state_dict(), "input_dim": self.input_dim,
             "latent_dim": self.latent_dim, "activation": self.activation},
            path,
        )

    def load(self, path: Path) -> "AutoencoderEncoder":
        ckpt = torch.load(path, map_location=self.device)
        self.input_dim = ckpt["input_dim"]
        self.latent_dim = ckpt["latent_dim"]
        self.activation = ckpt.get("activation", "gelu")   # backward compatible default
        self.ae_ = self._build_ae(self.input_dim, self.latent_dim).to(self.device)
        self.ae_.load_state_dict(ckpt["state_dict"])
        self.ae_.eval()
        return self


class EpaiAutoencoderEncoder(AutoencoderEncoder):
    """Same training/dim-selection machinery, but uses the original epai autoencoder
    (``epai.autoencoder.AE``, fixed GELU). Selected via ENCODER_KIND=epai_autoencoder."""

    kind = "epai_autoencoder"

    def _build_ae(self, input_dim: int, latent_dim: int) -> nn.Module:
        from epai.autoencoder import AE

        return AE(input_dim=input_dim, latent_dim=latent_dim)
