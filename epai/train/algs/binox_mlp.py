"""A late-fusion MLP: the embedding is compressed, then the Binoculars score joins.

Sibling of :mod:`simple_mlp`. The difference is *where* the Binoculars score
enters the network. Concatenated at the input it has to survive the trunk's
compression of the embedding, and whatever is not linearly recoverable from the
bottleneck is gone; concatenated *after* the trunk it reaches the classifier
head on a path the compression never touches.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .base import ClfAlgorithm


class _BinoxMLP(nn.Module):
    """Trunk over the embedding → concat the score → head → logits.

    ``pre_dims`` sizes the trunk, ``post_dims`` the head, both as
    (Linear → ReLU → Dropout) blocks. An empty ``pre_dims`` leaves the trunk as
    a pass-through, which is exactly early fusion (the score concatenated at the
    input) -- the control this architecture is meant to be measured against.

    Dropout is applied to the trunk's output but never to the score: zeroing the
    score on a fraction of the steps would teach the model not to depend on it,
    which is the opposite of the point.
    """

    def __init__(
        self,
        in_dim: int,
        aux_dim: int,
        pre_dims: list[int],
        post_dims: list[int],
        score_hidden: int,
        n_classes: int,
        dropout: float,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        prev = in_dim
        for h in pre_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        trunk_dim = prev

        # A bare Linear on the score would be pointless -- an affine function of
        # one scalar that the next layer's weights simply absorb. With the ReLU
        # it becomes a set of hinges, i.e. the head can learn a non-monotonic
        # response to the score. Off by default; the zero-shot baseline's single
        # threshold already assumes monotonicity.
        if score_hidden:
            self.score = nn.Sequential(nn.Linear(aux_dim, score_hidden), nn.ReLU())
            score_dim = score_hidden
        else:
            self.score = nn.Identity()
            score_dim = aux_dim

        layers = []
        prev = trunk_dim + score_dim
        for h in post_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.trunk(x), self.score(z)], dim=1))


class BinoxMLPAlg(ClfAlgorithm):
    """A hybrid MLP over an embedding plus the zero-shot Binoculars score.

    Takes the auxiliary column the config's ``binoculars`` key names, as a
    second matrix ``Z`` alongside the design matrix ``X`` (see ``uses_aux``).
    Runs on CUDA when available, otherwise CPU. Params (all optional):
      * ``pre_dims`` -- trunk widths, applied to the embedding before the score
        is concatenated (default ``[256, 64, 8]``; ``[]`` gives early fusion);
      * ``post_dims`` -- head widths, applied after the concatenation
        (default ``[16]``);
      * ``score_hidden`` -- project the score through ``Linear → ReLU`` to this
        width before concatenating (default 0, i.e. use the raw scalar);
      * ``dropout`` -- dropout probability (default 0.1);
      * ``epochs`` -- max epochs (default 100);
      * ``batch_size`` -- minibatch size (default 64);
      * ``lr`` -- AdamW learning rate (default 1e-3);
      * ``weight_decay`` -- AdamW weight decay (default 1e-4);
      * ``patience`` -- early-stop after this many epochs without a training-loss
        improvement (default 10; set 0 to disable and always run ``epochs``).

    ``X`` and ``Z`` are standardised by two separate scalers, both fitted on the
    *source* corpus's training split. The score's distribution shifts between
    corpora, so under transfer it arrives standardised by statistics that are
    not the target's -- deliberately, since the zero-shot baseline calibrates its
    threshold on the source split too, which keeps the two comparable.
    """

    name = "binox_mlp"
    uses_aux = True
    N_CLASSES = 2

    def _clean_aux(self, Z: np.ndarray) -> np.ndarray:
        """Replace non-finite scores with the training median of their column.

        A failed Binoculars computation leaves a NaN or an inf, and a single one
        of those would take the scaler's mean and std with it.
        """
        Z = np.asarray(Z, dtype=np.float64).reshape(len(Z), -1)
        bad = ~np.isfinite(Z)
        if bad.any():
            Z = np.where(bad, self.aux_fill_, Z)
        return Z

    def fit(
        self, X: np.ndarray, y: np.ndarray, Z: np.ndarray | None = None
    ) -> "BinoxMLPAlg":
        if Z is None:
            raise ValueError(
                f"{type(self).__name__} needs the auxiliary Binoculars column; "
                f"set 'binoculars' on this algorithm's config entry"
            )

        p = self.params
        pre_dims = [int(h) for h in p.get("pre_dims", [256, 64, 8])]
        post_dims = [int(h) for h in p.get("post_dims", [16])]
        score_hidden = int(p.get("score_hidden", 0))
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

        # Median over the finite training values only, so _clean_aux has a fill
        # value that predict() can reuse unchanged on every target corpus.
        Z_raw = np.asarray(Z, dtype=np.float64).reshape(len(Z), -1)
        finite = np.where(np.isfinite(Z_raw), Z_raw, np.nan)
        self.aux_fill_ = np.nan_to_num(np.nanmedian(finite, axis=0), nan=0.0)

        self.scaler = StandardScaler()
        self.aux_scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X).astype(np.float32)
        Zs = self.aux_scaler.fit_transform(self._clean_aux(Z)).astype(np.float32)

        X_t = torch.from_numpy(Xs).to(self.device)
        Z_t = torch.from_numpy(Zs).to(self.device)
        y_t = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(self.device)

        self.model = _BinoxMLP(
            X_t.shape[1],
            Z_t.shape[1],
            pre_dims,
            post_dims,
            score_hidden,
            self.N_CLASSES,
            dropout,
        ).to(self.device)
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(
            TensorDataset(X_t, Z_t, y_t), batch_size=batch_size, shuffle=True
        )

        best = float("inf")
        no_improve = 0
        self.model.train()
        for _ in range(epochs):
            epoch_loss = 0.0
            for xb, zb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model(xb, zb), yb)
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
    def predict(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        if Z is None:
            raise ValueError(
                f"{type(self).__name__} needs the auxiliary Binoculars column at "
                f"predict time as well"
            )
        Xs = self.scaler.transform(X).astype(np.float32)
        Zs = self.aux_scaler.transform(self._clean_aux(Z)).astype(np.float32)
        X_t = torch.from_numpy(Xs).to(self.device)
        Z_t = torch.from_numpy(Zs).to(self.device)
        self.model.eval()
        logits = self.model(X_t, Z_t)
        return logits.argmax(dim=1).cpu().numpy()

    def save(self, path: Path) -> None:
        # Move to CPU first so the pickle can be loaded on a machine without a GPU.
        self.model.to("cpu")
        self.device = torch.device("cpu")
        joblib.dump(self, path)
