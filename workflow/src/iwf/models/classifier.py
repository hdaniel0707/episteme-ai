"""MLP classification head: features -> ordinal impact class logits.

Class-imbalance compensation lives here (``class_weights``): 'balanced' = inverse
frequency, 'effective' = Cui et al. 2019 effective-number weighting (both normalized to
mean 1 so they are a no-op on balanced classes). Loss = cross-entropy with label
smoothing, or focal loss. Early stopping maximises val macro-F1.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from workflow.src.iwf.models.head_base import TorchHeadBase


def class_weights(y: np.ndarray, n_classes: int, scheme: str = "effective",
                  beta: float = 0.999) -> np.ndarray | None:
    """Per-class loss weights (mean-1 normalized) or None for scheme='none'."""
    scheme = (scheme or "none").strip().lower()
    if scheme in ("none", ""):
        return None
    counts = np.bincount(np.asarray(y, dtype=int), minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # an absent class must not produce inf weights
    if scheme == "balanced":
        w = counts.sum() / (n_classes * counts)
    elif scheme == "effective":
        beta = float(beta)
        effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        w = 1.0 / effective_num
    else:
        raise ValueError(f"Unknown class_weight scheme {scheme!r} "
                         "(choose none | balanced | effective).")
    return (w / w.mean()).astype(np.float64)


class _FocalLoss(torch.nn.Module):
    """Multi-class focal loss (Lin et al. 2017) with optional per-class alpha weights."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target.view(-1, 1)).view(-1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t) ** self.gamma) * logp_t
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device)[target]
        return loss.mean()


class TorchMLPClassifier(TorchHeadBase):
    task = "clf"

    def __init__(self, n_classes: int, *, class_weight: str = "effective",
                 effective_beta: float = 0.999, loss: str = "ce",
                 focal_gamma: float = 2.0, label_smoothing: float = 0.05, **common):
        super().__init__(**common)
        self.n_classes = int(n_classes)
        self.class_weight = class_weight
        self.effective_beta = float(effective_beta)
        self.loss = (loss or "ce").strip().lower()
        self.focal_gamma = float(focal_gamma)
        self.label_smoothing = float(label_smoothing)

    def _out_dim(self) -> int:
        return self.n_classes

    def _loss_fn(self, y_train: np.ndarray):
        w = class_weights(y_train, self.n_classes, self.class_weight, self.effective_beta)
        weight_t = torch.tensor(w, dtype=torch.float32) if w is not None else None
        if self.loss == "focal":
            return _FocalLoss(self.focal_gamma, weight_t)
        if self.loss != "ce":
            raise ValueError(f"Unknown loss {self.loss!r} (choose ce | focal).")

        def ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            wt = weight_t.to(logits.device) if weight_t is not None else None
            return F.cross_entropy(logits, target, weight=wt,
                                   label_smoothing=self.label_smoothing)

        return ce

    def _target_tensor(self, y: np.ndarray, device) -> torch.Tensor:
        return torch.tensor(np.asarray(y, dtype=np.int64), device=device)

    def _val_score(self, y_true: np.ndarray, out: torch.Tensor) -> float:
        pred = out.argmax(dim=1).cpu().numpy()
        return float(f1_score(np.asarray(y_true, dtype=int), pred, average="macro",
                              labels=list(range(self.n_classes)), zero_division=0))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = self._forward_numpy(X)
        return torch.softmax(out, dim=1).float().cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self._forward_numpy(X)
        return out.argmax(dim=1).cpu().numpy()
