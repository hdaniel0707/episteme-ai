"""Text-transformer backbones for the tfm_* workflows (registry slot ``text_transformer``).

Four architectures register themselves on import (the registry lazy-imports this module):

  * specter2_ft   full fine-tune of allenai/specter2_base (optionally freeze first N layers)
  * scibert_ft    full fine-tune of allenai/scibert_scivocab_uncased
  * specter2_lora frozen backbone + LoRA adapters on the attention query/value projections
  * scibert_lora  same for SciBERT

Every model is an nn.Module with the trainer contract: ``tokenize(titles, abstracts)``,
``forward(batch) -> [B, out_dim]`` (clf logits; reg sigmoid-squashed to [0,1]),
``trainable_parameters()`` and ``save(out_dir)`` (HF checkpoint/ + head weights).
LoRA is implemented locally (no peft dependency): W x + (alpha/r) * B(A(dropout(x))).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from workflow.src.iwf.registry import register_module


def load_hf(fn, name: str, what: str):
    """Call a HF ``from_pretrained`` loader with a readable failure message."""
    try:
        return fn(name)
    except Exception as e:
        raise RuntimeError(
            f"Could not load the HuggingFace {what} {name!r} ({type(e).__name__}: {e}). "
            "Check the network / HF_TOKEN in the repo-root .env, or pre-download the model."
        ) from e


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a trainable low-rank update (LoRA, Hu et al. 2021)."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_a = nn.Linear(base.in_features, int(r), bias=False)
        self.lora_b = nn.Linear(int(r), base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.scaling = float(alpha) / float(r)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.drop(x))) * self.scaling


def _encoder_layers(backbone) -> list | None:
    enc = getattr(backbone, "encoder", None)
    layers = getattr(enc, "layer", None)
    return list(layers) if layers is not None else None


class TextTransformerBase(nn.Module):
    """Backbone + CLS-pooled MLP head. ``backbone``/``tokenizer`` kwargs let the offline
    smoke test inject a tiny mock BERT so nothing is ever downloaded there."""

    BACKBONE_NAME: str | None = None
    USE_LORA = False

    def __init__(self, cfg, out_dim: int, task: str, *, backbone=None, tokenizer=None):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.out_dim = int(out_dim)

        if backbone is None or tokenizer is None:
            if not self.BACKBONE_NAME:
                raise ValueError("TextTransformerBase needs an injected backbone+tokenizer "
                                 "or a subclass with BACKBONE_NAME set.")
            from transformers import AutoModel, AutoTokenizer
            tokenizer = tokenizer or load_hf(AutoTokenizer.from_pretrained,
                                             self.BACKBONE_NAME, "tokenizer")
            backbone = backbone or load_hf(AutoModel.from_pretrained,
                                           self.BACKBONE_NAME, "model")
        self.backbone = backbone
        self.tokenizer = tokenizer

        hidden_size = int(getattr(self.backbone.config, "hidden_size", 768))
        head_hidden = [int(h) for h in (getattr(cfg, "hidden", None) or [256])]
        drop = float(getattr(cfg, "dropout", 0.1) or 0.0)
        layers: list[nn.Module] = []
        prev = hidden_size
        for h in head_hidden[:1]:  # one hidden layer is plenty on top of a transformer
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(drop)]
            prev = h
        layers.append(nn.Linear(prev, self.out_dim))
        self.head = nn.Sequential(*layers)

        if self.USE_LORA:
            self._apply_lora()
        else:
            self._freeze_layers(int(getattr(cfg, "text_transformer_freeze_layers", 0) or 0))

    # ------------------------------------------------------------- adaptation
    def _freeze_layers(self, n: int) -> None:
        if n <= 0:
            return
        emb = getattr(self.backbone, "embeddings", None)
        if emb is not None:
            for p in emb.parameters():
                p.requires_grad = False
        layers = _encoder_layers(self.backbone)
        if layers:
            for layer in layers[:n]:
                for p in layer.parameters():
                    p.requires_grad = False

    def _apply_lora(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        r = int(getattr(self.cfg, "text_transformer_lora_r", 8))
        alpha = int(getattr(self.cfg, "text_transformer_lora_alpha", 16))
        drop = float(getattr(self.cfg, "text_transformer_lora_dropout", 0.1))
        layers = _encoder_layers(self.backbone)
        if not layers:
            raise ValueError("LoRA arch needs a BERT-style backbone with encoder.layer.")
        for layer in layers:
            att = layer.attention.self
            att.query = LoRALinear(att.query, r, alpha, drop)
            att.value = LoRALinear(att.value, r, alpha, drop)

    # ------------------------------------------------------------- trainer contract
    def tokenize(self, titles, abstracts) -> dict:
        max_len = int(getattr(self.cfg, "text_transformer_max_len", 256) or 256)
        return self.tokenizer(list(titles), list(abstracts), truncation=True,
                              max_length=max_len, padding=True, return_tensors="pt")

    def forward(self, batch: dict) -> torch.Tensor:
        out = self.backbone(**batch)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.head(cls)
        if self.task == "reg":
            return torch.sigmoid(logits)   # keep predictions on the [0,1] target scale
        return logits

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def save(self, out_dir: str | Path) -> None:
        ckpt = Path(out_dir) / "checkpoint"
        ckpt.mkdir(parents=True, exist_ok=True)
        try:
            self.backbone.save_pretrained(ckpt)
        except Exception as e:  # a mock backbone may not fully support HF save
            print(f"[text_transformers] backbone save skipped ({e})")
        try:
            self.tokenizer.save_pretrained(ckpt)
        except Exception:
            pass
        torch.save(self.head.state_dict(), ckpt / "head.pt")


@register_module("text_transformer", "specter2_ft")
class Specter2FT(TextTransformerBase):
    BACKBONE_NAME = "allenai/specter2_base"


@register_module("text_transformer", "scibert_ft")
class SciBertFT(TextTransformerBase):
    BACKBONE_NAME = "allenai/scibert_scivocab_uncased"


@register_module("text_transformer", "specter2_lora")
class Specter2LoRA(TextTransformerBase):
    BACKBONE_NAME = "allenai/specter2_base"
    USE_LORA = True


@register_module("text_transformer", "scibert_lora")
class SciBertLoRA(TextTransformerBase):
    BACKBONE_NAME = "allenai/scibert_scivocab_uncased"
    USE_LORA = True
