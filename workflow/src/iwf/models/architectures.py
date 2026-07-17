"""Model-block factory for the impact-predictor heads (registry slot ``impact_predictor``).

Every block maps a flat feature vector [B, in_dim] -> [B, out_dim] raw outputs (logits for
clf; the regressor applies its own sigmoid). Three architectures are registered:

  * mlp            plain Linear stack (hidden sizes = ``hidden``)
  * resmlp         pre-norm residual MLP: one block per ``hidden`` entry at that width,
                   LayerNorm+Linear transitions between differing widths
  * ft_transformer scalar features -> learned feature tokens + CLS -> TransformerEncoder
"""

from __future__ import annotations

import torch
import torch.nn as nn

from workflow.src.iwf.activations import make_activation

_ARCHS = ("mlp", "resmlp", "ft_transformer")


def available_archs() -> list[str]:
    return list(_ARCHS)


def _mlp(in_dim: int, out_dim: int, hidden: list[int], dropout: float,
         activation: str) -> nn.Module:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, int(h)), make_activation(activation),
                   nn.Dropout(float(dropout))]
        prev = int(h)
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class _ResBlock(nn.Module):
    def __init__(self, width: int, dropout: float, activation: str):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.act = make_activation(activation)
        self.drop = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))


class _Transition(nn.Module):
    """Width change between residual stages: LayerNorm -> Linear projection."""

    def __init__(self, in_width: int, out_width: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_width)
        self.proj = nn.Linear(in_width, out_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class _ResMLP(nn.Module):
    """Pre-norm residual MLP honoring the FULL ``hidden`` width list: one residual
    block per entry, with a LayerNorm+Linear transition wherever consecutive widths
    differ. ``[1024, 512, 128]`` = a 1024-block, a 512-block and a 128-block joined
    by projections; a constant list like ``[256, 256, 256]`` behaves like the
    classic fixed-width ResMLP."""

    def __init__(self, in_dim: int, out_dim: int, hidden: list[int], dropout: float,
                 activation: str):
        super().__init__()
        widths = [int(h) for h in hidden] or [128]
        self.inp = nn.Linear(in_dim, widths[0])
        stages: list[nn.Module] = []
        prev = widths[0]
        for w in widths:
            if w != prev:
                stages.append(_Transition(prev, w))
            stages.append(_ResBlock(w, dropout, activation))
            prev = w
        self.blocks = nn.ModuleList(stages)
        self.norm = nn.LayerNorm(prev)
        self.out = nn.Linear(prev, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        return self.out(self.norm(x))


class _FTTransformer(nn.Module):
    """FT-Transformer-style block: each scalar input becomes a learned token
    (per-feature weight * value + bias), a CLS token is prepended, a small
    TransformerEncoder mixes them, and the head reads the CLS position."""

    def __init__(self, in_dim: int, out_dim: int, hidden: list[int], dropout: float,
                 activation: str, d_token: int = 64, n_heads: int = 4):
        super().__init__()
        depth = max(1, min(4, len(hidden))) if hidden else 2
        self.weight = nn.Parameter(torch.empty(in_dim, d_token))
        self.bias = nn.Parameter(torch.zeros(in_dim, d_token))
        nn.init.normal_(self.weight, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
            dropout=float(dropout), activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(make_activation(activation),
                                  nn.Linear(d_token, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight + self.bias          # [B, F, d]
        cls = self.cls.expand(x.shape[0], -1, -1)                    # [B, 1, d]
        h = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(self.norm(h[:, 0]))


def build_block(arch: str, in_dim: int, out_dim: int, *, task: str,
                hidden: list[int], dropout: float, activation: str) -> nn.Module:
    """Build the raw [B, in_dim] -> [B, out_dim] block for ``arch`` (task-agnostic)."""
    key = (arch or "mlp").strip().lower()
    hidden = [int(h) for h in (hidden or [128])]
    if key == "mlp":
        return _mlp(in_dim, out_dim, hidden, dropout, activation)
    if key == "resmlp":
        return _ResMLP(in_dim, out_dim, hidden, dropout, activation)
    if key == "ft_transformer":
        return _FTTransformer(in_dim, out_dim, hidden, dropout, activation)
    raise ValueError(f"Unknown model_arch {arch!r}. Choose from {available_archs()}.")
