"""Generic torch training loop with early stopping (plan section 5).

Shared by both heads. Trains on TRAIN, early-stops on a VAL metric (higher = better:
macro-F1 for clf, R^2 for reg), restores the best weights. Test is never seen here.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, r2_score
from torch.utils.data import DataLoader, TensorDataset

_RUNTIME_CONFIGURED = False


class _FocalLoss(torch.nn.Module):
    """Multiclass focal loss (Lin et al., 2017): down-weights easy examples by (1-p_t)^gamma so
    training focuses on the hard/rare cases - the classic fix for critical-event detection where
    a majority class would otherwise dominate. Accepts the same per-class ``weight`` vector."""

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp().clamp_(1e-6, 1.0)
        loss = -((1.0 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight.gather(0, target)
        return loss.mean()


def configure_runtime() -> None:
    """Make training use all cores and the fastest math paths (idempotent).

    * CPU: use every logical core for the BLAS/OpenMP-backed torch ops.
    * GPU: enable cuDNN autotuning and TF32 matmuls (big speedup on Ampere+).
    Call once at the start of training.
    """
    global _RUNTIME_CONFIGURED
    if _RUNTIME_CONFIGURED:
        return
    n_cpu = os.cpu_count() or 1
    # WF_TORCH_THREADS lets a parallel worker cap its own thread budget so N concurrent
    # workers don't each grab all cores (set by workflow.src.iwf.parallel); else use every core.
    try:
        n_threads = int(os.getenv("WF_TORCH_THREADS", "") or n_cpu)
    except ValueError:
        n_threads = n_cpu
    torch.set_num_threads(max(1, n_threads))
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"[runtime] GPU offload ON - training on CUDA ({torch.cuda.get_device_name(0)}); "
              f"TF32 + cuDNN autotune enabled")
    else:
        print(f"[runtime] no CUDA device found - training on CPU ({n_cpu} threads)")
    _RUNTIME_CONFIGURED = True


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _val_metric(task: str, y_true: np.ndarray, logits_or_pred: np.ndarray) -> float:
    if task == "clf":
        pred = logits_or_pred.argmax(axis=1)
        return float(f1_score(y_true, pred, average="macro"))
    return float(r2_score(y_true, logits_or_pred))


def train_torch_model(
    net: torch.nn.Module,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    *, task: str, lr: float, weight_decay: float,
    max_epochs: int, patience: int, batch_size: int, seed: int,
    class_weight: np.ndarray | None = None, label_smoothing: float = 0.0,
    loss: str = "ce", focal_gamma: float = 2.0,
) -> dict:
    """Train ``net`` in place; return a history dict. Best (val) weights are restored.

    ``class_weight`` / ``label_smoothing`` / ``loss`` / ``focal_gamma`` only affect the
    classifier loss; they are ignored for regression (MSE).
    """
    configure_runtime()
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)

    Xtr = torch.tensor(X_tr, dtype=torch.float32).to(device)
    Xva = torch.tensor(X_va, dtype=torch.float32).to(device)
    if task == "clf":
        ytr = torch.tensor(y_tr, dtype=torch.long).to(device)
        weight_t = (torch.as_tensor(np.asarray(class_weight, dtype=np.float32), device=device)
                    if class_weight is not None else None)
        if (loss or "ce").lower() == "focal":
            loss_fn = _FocalLoss(weight=weight_t, gamma=focal_gamma)
        else:
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight_t,
                                                label_smoothing=float(label_smoothing))
    else:
        ytr = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1).to(device)
        loss_fn = torch.nn.MSELoss()

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_metric": []}
    best_metric, best_state, no_improve = -np.inf, None, 0

    for _ in range(max_epochs):
        net.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            out = net(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= max(len(loader), 1)

        net.eval()
        with torch.no_grad():
            out_va = net(Xva).cpu().numpy()
        metric = _val_metric(task, y_va, out_va)

        history["train_loss"].append(epoch_loss)
        history["val_metric"].append(metric)

        if metric > best_metric + 1e-5:
            best_metric, no_improve = metric, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    history["best_val_metric"] = float(best_metric)
    return history
