"""Shared torch training loop for the impact heads (classifier + regressor).

One implementation of seeding, mini-batching, early stopping and history so both heads
report the identical ``history_`` contract used by the learning-curve plots:
    history_ = {"train_loss": [...], "val_metric": [...], "best_val_metric": float,
                "best_epoch": int}
Early stopping restores the best-val weights, so ``predict`` always uses the best epoch.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from workflow.src.iwf.models.architectures import build_block


class TorchHeadBase:
    """Not an nn.Module on purpose: instances are joblib-pickled with net_ on CPU."""

    task = "reg"   # overridden by the classifier

    def __init__(self, *, hidden=None, dropout: float = 0.5, lr: float = 5e-4,
                 weight_decay: float = 0.02, max_epochs: int = 2000, patience: int = 60,
                 batch_size: int = 128, seed: int = 42, activation: str = "gelu",
                 arch: str = "mlp"):
        self.hidden = [int(h) for h in (hidden or [256, 128])]
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.activation = activation
        self.arch = arch
        self.net_: torch.nn.Module | None = None
        self.history_: dict = {}

    # ------------------------------------------------------------- task hooks
    def _out_dim(self) -> int:
        raise NotImplementedError

    def _loss_fn(self, y_train: np.ndarray):
        raise NotImplementedError

    def _target_tensor(self, y: np.ndarray, device) -> torch.Tensor:
        raise NotImplementedError

    def _val_score(self, y_true: np.ndarray, out: torch.Tensor) -> float:
        """Higher = better; drives early stopping."""
        raise NotImplementedError

    # ------------------------------------------------------------- training
    def _device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = self._device()
        if device == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        X_train = np.asarray(X_train, dtype=np.float32)
        if X_val is None or len(X_val) == 0:   # degenerate case: early-stop on train
            X_val, y_val = X_train, y_train
        X_val = np.asarray(X_val, dtype=np.float32)

        self.net_ = build_block(self.arch, X_train.shape[1], self._out_dim(),
                                task=self.task, hidden=self.hidden,
                                dropout=self.dropout, activation=self.activation
                                ).to(device)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        loss_fn = self._loss_fn(np.asarray(y_train))

        Xt = torch.tensor(X_train, device=device)
        yt = self._target_tensor(np.asarray(y_train), device)
        Xv = torch.tensor(X_val, device=device)

        history = {"train_loss": [], "val_metric": [], "best_val_metric": None,
                   "best_epoch": 0}
        best_score, best_state, no_improve = -np.inf, None, 0
        rng = np.random.default_rng(self.seed)
        n = len(Xt)

        for epoch in range(self.max_epochs):
            self.net_.train()
            order = rng.permutation(n)
            losses = []
            for start in range(0, n, self.batch_size):
                sel = order[start:start + self.batch_size]
                opt.zero_grad(set_to_none=True)
                out = self.net_(Xt[sel])
                loss = loss_fn(out, yt[sel])
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            history["train_loss"].append(float(np.mean(losses)) if losses else 0.0)

            self.net_.eval()
            with torch.no_grad():
                val_out = self.net_(Xv)
            score = float(self._val_score(np.asarray(y_val), val_out))
            history["val_metric"].append(score)

            if score > best_score + 1e-7:
                best_score, no_improve = score, 0
                history["best_epoch"] = epoch + 1
                best_state = copy.deepcopy(self.net_.state_dict())
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)
        history["best_val_metric"] = float(best_score) if np.isfinite(best_score) else None
        self.history_ = history
        return self

    @torch.no_grad()
    def _forward_numpy(self, X: np.ndarray) -> torch.Tensor:
        if self.net_ is None:
            raise RuntimeError("Model is not fitted - call fit() first.")
        device = next(self.net_.parameters()).device
        self.net_.eval()
        Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
        outs = []
        for start in range(0, len(Xt), 4096):
            outs.append(self.net_(Xt[start:start + 4096]))
        return torch.cat(outs) if outs else self.net_(Xt)
