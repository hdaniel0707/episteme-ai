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
# weights are always allenai/specter2_base and the *adapter* is what specialises them.
# AllenAI trains ONE adapter per SciRepEval task format, and the task format is supposed to
# match how the embedding is consumed downstream (see SPECTER2_ADAPTER_PURPOSE below):
#   specter2_base    base weights only, NO adapter (the base-vs-adapter control)
#   specter2         + proximity/retrieval adapter   (nearest-neighbour, link prediction)
#   specter2_clf     + classification adapter        (embeddings feeding a classifier head)
#   specter2_reg     + regression adapter            (embeddings feeding a regressor head)
#   specter2_adhoc   + ad-hoc query adapter          (SHORT text queries, not whole papers)
# Because the output column is f"emb_{name}" (see add_embedding), each flavour gets its own
# cached column / encoder input automatically, so runs can contrast them without collisions.
SPECTER2_VARIANTS = {
    "specter2": ("allenai/specter2_base", "allenai/specter2"),
    "specter2_base": ("allenai/specter2_base", None),
    "specter2_clf": ("allenai/specter2_base", "allenai/specter2_classification"),
    "specter2_reg": ("allenai/specter2_base", "allenai/specter2_regression"),
    "specter2_adhoc": ("allenai/specter2_base", "allenai/specter2_adhoc_query"),
}

# What each flavour's adapter is FOR — surfaced in the per-run description_<wf>.md so a
# results folder states which adapter produced its embeddings and whether that adapter
# matches the task it was used for (emb_clf wants specter2_clf, emb_reg wants specter2_reg).
SPECTER2_ADAPTER_PURPOSE = {
    "specter2_base": "no adapter — plain SPECTER2 base weights (base-vs-adapter control)",
    "specter2": "proximity / retrieval: nearest-neighbour search and link prediction",
    "specter2_clf": "classification: embeddings intended as features for a classifier head",
    "specter2_reg": "regression: embeddings intended as features for a regressor head",
    "specter2_adhoc": "ad-hoc search: encodes SHORT query strings, not full paper records",
}
# The adapter whose task format matches each workflow task, used only to flag a mismatch in
# the generated documentation (never to override an explicit config choice).
TASK_MATCHED_EMBEDDER = {"clf": "specter2_clf", "reg": "specter2_reg"}

# transformer (CLS-pooled) embedders that have a batched GPU path here (EMBED_BATCH_SIZE):
BATCHABLE_TYPES = {"scibert", *SPECTER2_VARIANTS}
DIMS = {**{name: 768 for name in SPECTER2_VARIANTS},
        "scibert": 768, "scincl": 768, "openai": 1536, "e5": 1024}
# Max input tokens per embedder (title+abstract is truncated to this). Used by the analysis
# abstract-length box plot to draw a "~max chars this embedder ingests" marker (tokens * ~3).
MAX_TOKENS = {**{name: 512 for name in SPECTER2_VARIANTS},
              "scibert": 512, "scincl": 512, "e5": 512, "openai": 8191}

# Every embedder name config.yaml may legally name (validated early, in Settings.validate).
KNOWN_EMBEDDERS = frozenset({*SPECTER2_VARIANTS, *EPAI_TYPES, *LOCAL_TYPES})


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
    per forward pass - a large speed-up on GPU.

    For every adapter flavour the adapter is loaded AND verified active before any paper is
    embedded: a silent activation failure would hand back plain base-weight vectors stored
    under the adapter's column name, i.e. an invalid base-vs-adapter comparison that no
    metric could reveal. It is therefore a hard error, never a warning.

    Note: the `adapters` library logs "There are adapters available but none are activated
    for the forward pass" ONCE per model load, before the active setup is read back. It is
    benign - the assertion below is the authority on whether the adapter is really active.
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
            # load_as=emb_type so the active setup reads back as e.g. Stack[specter2_clf]
            loaded = model.load_adapter(adapter_name, source="hf",
                                        load_as=emb_type, set_active=True)
            model.set_active_adapters(loaded)
            active = str(getattr(model, "active_adapters", None) or "")
            if loaded not in active:
                raise RuntimeError(
                    f"{emb_type}: adapter {adapter_name!r} loaded as {loaded!r} but is NOT "
                    f"active (active_adapters={active or 'None'}). Embedding now would "
                    f"silently produce plain {base_model} vectors under the "
                    f"'emb_{emb_type}' column. Check the installed `adapters` version."
                )
            print(f"[embed] {emb_type}: adapter {adapter_name} active ({active}) — "
                  f"{SPECTER2_ADAPTER_PURPOSE.get(emb_type, '')}")
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

    EVERY specter2 flavour goes through the local batched path, whatever EMBED_BATCH_SIZE
    says: that path is the only one that loads and activates the task adapter. Routing
    "specter2" to the epai helper instead (it is also an EPAI_TYPE) would quietly return
    un-adapted base vectors under the adapter's column name, so the flavour check comes
    first and EMBED_BATCH_SIZE only chooses the batch size.
    """
    col = f"emb_{emb_type}"
    if col in df.columns and df[col].notna().all():
        return df
    batch_size = _embed_batch_size()
    if emb_type in SPECTER2_VARIANTS:
        return _hf_embed_batched(df, emb_type, batch_size or 32)
    if emb_type in BATCHABLE_TYPES and batch_size > 0:
        return _hf_embed_batched(df, emb_type, batch_size)
    if emb_type in EPAI_TYPES:
        return _epai_add_embedding(df, emb_type)
    if emb_type in LOCAL_TYPES:
        return _hf_embed(df, emb_type)
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
