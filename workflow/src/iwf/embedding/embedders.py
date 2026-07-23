"""Compute >=4 frozen embeddings of title+abstract (plan section 4.2).

specter2 / scibert / openai are delegated to epai.utils.embedding_utils (already in the
base repo). SciNCL and E5 are added here via the existing `transformers` dependency.
Embedders are frozen, so they run on the FULL corpus (no leakage).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from epai.utils.embedding_utils import add_embedding as _epai_add_embedding

EPAI_TYPES = {"specter2", "scibert", "openai"}
LOCAL_TYPES = {"scincl", "e5"}

# SPECTER2 flavours selectable by name. Each maps to (base_model, adapter_or_None); the base
# weights are always allenai/specter2_base and the *adapter* is what specialises them:
#   specter2         base + the proximity/citation adapter (allenai/specter2) -> the "full result"
#   specter2_base    base weights only, NO adapter (for base-vs-adapter comparison)
#   specter2_adhoc   base + the ad-hoc query adapter (retrieval-flavoured)
# Because the output column is f"emb_{name}" (see add_embedding), each flavour gets its own
# cached column / encoder input automatically, so two runs can contrast them.
SPECTER2_VARIANTS = {
    "specter2": ("allenai/specter2_base", "allenai/specter2"),
    "specter2_base": ("allenai/specter2_base", None),
    "specter2_adhoc": ("allenai/specter2_base", "allenai/specter2_adhoc_query"),
}

# transformer (CLS-pooled) embedders that have a batched GPU path here (EMBED_BATCH_SIZE):
BATCHABLE_TYPES = {"specter2", "scibert", "specter2_base", "specter2_adhoc"}
DIMS = {"specter2": 768, "specter2_base": 768, "specter2_adhoc": 768,
        "scibert": 768, "scincl": 768, "openai": 1536, "e5": 1024}
# Max input tokens per embedder (title+abstract is truncated to this). Used by the analysis
# abstract-length box plot to draw a "~max chars this embedder ingests" marker (tokens * ~3).
MAX_TOKENS = {"specter2": 512, "specter2_base": 512, "specter2_adhoc": 512,
              "scibert": 512, "scincl": 512, "e5": 512, "openai": 8191}


def _embed_batch_size() -> int:
    """EMBED_BATCH_SIZE from the env (0/invalid = use the per-row epai path unchanged)."""
    try:
        return max(0, int(os.getenv("EMBED_BATCH_SIZE", "0")))
    except (TypeError, ValueError):
        return 0


def _hf_embed_batched(df: pd.DataFrame, emb_type: str, batch_size: int) -> pd.DataFrame:
    """Batched, GPU-offloaded CLS embeddings for specter2 / scibert.

    Numerically equivalent to the one-at-a-time path (padding is masked out by the
    attention_mask, so the [CLS] vector is unaffected) but processes ``batch_size`` papers
    per forward pass - a large speed-up on GPU. For specter2 the proximity adapter is
    explicitly activated, fixing the "adapters available but none are activated" warning.
    """
    from transformers import AutoTokenizer

    from workflow.src.iwf.models.text_transformers import load_hf  # readable HF errors

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if emb_type in SPECTER2_VARIANTS:
        from adapters import AutoAdapterModel

        base_model, adapter_name = SPECTER2_VARIANTS[emb_type]
        tok = load_hf(AutoTokenizer.from_pretrained, base_model, "tokenizer")
        model = load_hf(AutoAdapterModel.from_pretrained, base_model, "model")
        if adapter_name:                      # None -> plain base weights (specter2_base)
            adapter = model.load_adapter(adapter_name, source="hf",
                                         load_as="adapter", set_active=True)
            try:                              # belt-and-braces: ensure it is the active adapter
                model.set_active_adapters(adapter or "adapter")
            except Exception:
                pass
    elif emb_type == "scibert":
        from transformers import AutoModel

        tok = load_hf(AutoTokenizer.from_pretrained, "allenai/scibert_scivocab_uncased", "tokenizer")
        model = load_hf(AutoModel.from_pretrained, "allenai/scibert_scivocab_uncased", "model")
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"{emb_type!r} has no batched path.")

    model.to(device).eval()
    sep = tok.sep_token or " "
    titles = df["title"].fillna("").astype(str).tolist()
    abstracts = df["abstract"].fillna("").astype(str).tolist()
    texts = [f"{t}{sep}{a}" for t, a in zip(titles, abstracts)]

    vecs: list[list[float]] = []
    desc = f"Embedding ({emb_type}, bs={batch_size}, {device})"
    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", truncation=True, max_length=512,
                  padding=True).to(device)
        with torch.no_grad():
            out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()
        vecs.extend(row.tolist() for row in cls)

    df[f"emb_{emb_type}"] = vecs
    return df


def _hf_embed(df: pd.DataFrame, emb_type: str) -> pd.DataFrame:
    """CLS-pooled SciNCL or mean-pooled E5 embeddings (transformers)."""
    from transformers import AutoModel, AutoTokenizer

    from workflow.src.iwf.models.text_transformers import load_hf  # readable HF errors

    name = {"scincl": "malteos/scincl", "e5": "intfloat/e5-large-v2"}[emb_type]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_hf(AutoTokenizer.from_pretrained, name, "tokenizer")
    model = load_hf(AutoModel.from_pretrained, name, "model").to(device).eval()

    def embed(title: str, abstract: str) -> list[float]:
        if emb_type == "e5":
            text = "passage: " + (title or "") + " " + (abstract or "")
        else:
            text = (title or "") + tok.sep_token + (abstract or "")
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**inputs)
        hidden = out.last_hidden_state
        if emb_type == "e5":
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vec = torch.nn.functional.normalize(vec, dim=-1)
            return vec.squeeze().cpu().tolist()
        return hidden[:, 0, :].squeeze().cpu().tolist()

    col = f"emb_{emb_type}"
    df[col] = [
        embed(r["title"] or "", r["abstract"] or "")
        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Embedding ({emb_type})")
    ]
    return df


def add_embedding(df: pd.DataFrame, emb_type: str) -> pd.DataFrame:
    """Add an ``emb_{emb_type}`` column; routes to epai or the local HF path.

    For specter2/scibert, EMBED_BATCH_SIZE>0 selects a batched GPU path (faster, and it
    activates the specter2 adapter); EMBED_BATCH_SIZE=0 keeps the original per-row epai path.
    """
    col = f"emb_{emb_type}"
    if col in df.columns and df[col].notna().all():
        return df
    batch_size = _embed_batch_size()
    if emb_type in BATCHABLE_TYPES and batch_size > 0:
        return _hf_embed_batched(df, emb_type, batch_size)
    if emb_type in EPAI_TYPES:
        return _epai_add_embedding(df, emb_type)
    if emb_type in LOCAL_TYPES:
        return _hf_embed(df, emb_type)
    if emb_type in SPECTER2_VARIANTS:
        # specter2 adapter flavours (specter2_base / specter2_adhoc) only have the batched
        # HF path — that's where the adapter is selected — so use a default batch size when
        # EMBED_BATCH_SIZE is unset rather than falling through to the error below.
        return _hf_embed_batched(df, emb_type, batch_size or 32)
    raise ValueError(
        f"Unknown emb_type {emb_type!r}. "
        f"Choose from {sorted(EPAI_TYPES | LOCAL_TYPES | set(SPECTER2_VARIANTS))}."
    )


def embed_dataframe(df: pd.DataFrame, embedders: list[str]) -> pd.DataFrame:
    """Add every requested embedding column to df."""
    for emb_type in embedders:
        df = add_embedding(df, emb_type)
    return df


def emb_matrix(df: pd.DataFrame, emb_type: str) -> np.ndarray:
    """Stack an embedding column into a float32 (n, d) matrix."""
    col = f"emb_{emb_type}"
    if col not in df.columns:
        raise KeyError(f"Missing embedding column {col!r}; run main_embed first.")
    return np.array(df[col].tolist(), dtype=np.float32)
