"""Activation factory - the single place that maps an ACTIVATION name to a torch module.

Used by both the encoder (autoencoder) and the impact-predictor heads, so the hidden-layer
activation is selectable from .env.workflow:  ACTIVATION = relu | gelu | tanh | sigmoid | ...
(Output activations are fixed by the task: classifier -> softmax/logits, regressor -> sigmoid.)
"""

from __future__ import annotations

import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "htanh": nn.Hardtanh,
    "hardtanh": nn.Hardtanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "mish": nn.Mish,
}


def valid_activations() -> list[str]:
    return sorted(_ACTIVATIONS)


def make_activation(name: str) -> nn.Module:
    """Return a fresh activation module for ``name`` (case-insensitive)."""
    key = (name or "gelu").strip().lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unknown ACTIVATION {name!r}. Choose from {valid_activations()}.")
    return _ACTIVATIONS[key]()
