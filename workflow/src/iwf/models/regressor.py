"""MLP regression head: features -> normalized impact in [0, 1].

The block outputs one raw value which is squashed by a sigmoid, so predictions always
land inside the target scale. Early stopping maximises val Spearman (falls back to
negative MSE when Spearman is undefined, e.g. constant predictions on tiny data).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr

from workflow.src.iwf.models.head_base import TorchHeadBase


class TorchMLPRegressor(TorchHeadBase):
    task = "reg"

    def _out_dim(self) -> int:
        return 1

    def _loss_fn(self, y_train: np.ndarray):
        mse = torch.nn.MSELoss()

        def loss(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return mse(torch.sigmoid(out).view(-1), target.view(-1))

        return loss

    def _target_tensor(self, y: np.ndarray, device) -> torch.Tensor:
        return torch.tensor(np.asarray(y, dtype=np.float32), device=device)

    def _val_score(self, y_true: np.ndarray, out: torch.Tensor) -> float:
        pred = torch.sigmoid(out).view(-1).float().cpu().numpy()
        y = np.asarray(y_true, dtype=float)
        if len(y) >= 2 and np.std(pred) > 0 and np.std(y) > 0:
            rho = spearmanr(y, pred).correlation
            if rho == rho:  # not NaN
                return float(rho)
        return -float(np.mean((pred - y) ** 2)) if len(y) else -np.inf

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self._forward_numpy(X)
        return torch.sigmoid(out).view(-1).float().cpu().numpy()
