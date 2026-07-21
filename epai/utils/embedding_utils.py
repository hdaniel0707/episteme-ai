"""Text embedding helpers.

``add_embedding`` attaches an ``emb_{emb_type}`` column to a DataFrame, using
``title + sep + abstract`` as the input text (callers that only have a single
text field pass it as *title* with an empty *abstract*).

The available models are grouped by what they were trained to encode, which is
what makes them useful as a spectrum for AI-text detection:

content-heavy (strong in-domain, expected to be weak out-of-domain)
    ``specter2``   allenai/specter2_base + the ``allenai/specter2`` proximity
                   adapter.  The base checkpoint on its own is only a shell --
                   without the adapter you get raw SciBERT-ish MLM features, not
                   SPECTER2, so it is always loaded through ``AutoAdapterModel``.
    ``scincl``     malteos/scincl, same citation-graph philosophy as SPECTER2;
                   a "same family, different model" control.
    ``scibert``    allenai/scibert_scivocab_uncased.

general semantic (the mainstream middle)
    ``mpnet``      sentence-transformers/all-mpnet-base-v2, the SBERT workhorse.
    ``minilm``     sentence-transformers/all-MiniLM-L6-v2, tiny/fast.
    ``e5``         intfloat/e5-base-v2.  Prefix-sensitive: every text gets the
                   ``"query: "`` prefix (this is a classification use, so all
                   texts are treated as queries).  Dropping it quietly degrades
                   the model.
    ``bge``        BAAI/bge-base-en-v1.5.  Also prefix-sensitive for retrieval;
                   for embed-then-classify it is used raw, but consistently so.
    ``qwen3``      Qwen/Qwen3-Embedding-0.6B, top-tier on MTEB but heavy and
                   content-optimised -- needs GPU headroom.

raw LM baseline
    ``roberta``    roberta-base with mean pooling over the last hidden state.
                   The ablation that isolates what an un-specialised transformer
                   encodes.

style / authorship (the end expected to survive domain shift)
    ``styledistance``   StyleDistance/styledistance, content-independent style
                        embeddings trained on synthetic parallel examples where
                        only the style varies.
    ``style_wegmann``   AnnaWegmann/Style-Embedding, the earlier-generation
                        content-independent style embedding.
    ``luar`` / ``luar_crud``  rrivera1849/LUAR-MUD and -CRUD universal
                        authorship representations.  Non-standard input shape
                        (episodes of texts per author) and ``trust_remote_code``.

api (billed per token)
    ``openai_small``  text-embedding-3-small, 1536 dimensions.
    ``openai_large``  text-embedding-3-large, 3072 dimensions -- the stronger of
                      the two on MTEB.  Kept as a separate embedder rather than
                      replacing the small one so the pair also serves as a
                      within-family capacity comparison.

Optional dependencies (imported lazily, only for the model actually requested):
``adapters`` for ``specter2``, ``sentence-transformers`` for the ``mpnet``,
``minilm``, ``e5``, ``bge``, ``qwen3``, ``styledistance`` and ``style_wegmann``
models, ``openai`` for the ``openai_*`` models; everything else needs only
``transformers``.

Each backend is built as an ``encode(pairs) -> list[list[float]]`` callable that
takes a batch of ``(title, abstract)`` tuples, joins them the way its own model
expects and returns one vector per pair. ``add_embedding`` frees the model before
it returns, so a caller that runs several embedders in sequence does not
accumulate them all on the GPU.
"""

import gc

import torch
import pandas as pd
from tqdm import tqdm

DEFAULT_BATCH_SIZE = 32


# emb_type -> (hf model id, pooling) for plain ``transformers`` checkpoints.
_HF_MODELS = {
    "scibert": ("allenai/scibert_scivocab_uncased", "cls"),
    "scincl": ("malteos/scincl", "cls"),
    "roberta": ("roberta-base", "mean"),
}

# emb_type -> (hf model id, prefix) for sentence-transformers checkpoints; the
# pooling of each model is read from its own config by SentenceTransformer.
_ST_MODELS = {
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", ""),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "e5": ("intfloat/e5-base-v2", "query: "),
    "bge": ("BAAI/bge-base-en-v1.5", ""),
    "qwen3": ("Qwen/Qwen3-Embedding-0.6B", ""),
    "styledistance": ("StyleDistance/styledistance", ""),
    "style_wegmann": ("AnnaWegmann/Style-Embedding", ""),
}

# emb_type -> (hf model id) for the LUAR authorship models.
_LUAR_MODELS = {
    "luar": "rrivera1849/LUAR-MUD",
    "luar_crud": "rrivera1849/LUAR-CRUD",
}

# emb_type -> OpenAI API model. Both current models are here as separate
# embedders so the two sit in their own columns and can be compared directly;
# each is used at its native width (1536 and 3072), no truncation.
_OPENAI_MODELS = {
    "openai_small": "text-embedding-3-small",
    "openai_large": "text-embedding-3-large",
}

# The API caps a single embeddings request at 300k tokens across all its inputs,
# and any one input at 8192 tokens. A batch size counted in rows cannot respect a
# limit counted in tokens, so the OpenAI backend re-splits whatever batch it is
# given; batch_size is only an upper bound for it. The character-per-token ratio
# is a deliberate under-estimate (English prose runs nearer 4) so the estimate
# errs towards smaller, safer requests.
OPENAI_MAX_TOKENS_PER_REQUEST = 250_000
OPENAI_MAX_TOKENS_PER_INPUT = 8_000
OPENAI_CHARS_PER_TOKEN = 3

EMB_TYPES = (
    "specter2",
    *_HF_MODELS,
    *_ST_MODELS,
    *_LUAR_MODELS,
    *_OPENAI_MODELS,
)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor):
    """Mean of the token vectors, ignoring padding."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def _build_specter2(device: str):
    """SPECTER2 base + proximity adapter (the adapter is what makes it SPECTER2)."""
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter(
        "allenai/specter2", source="hf", load_as="specter2", set_active=True
    )
    model.to(device)
    model.eval()

    def encode(pairs):
        texts = [t + tokenizer.sep_token + a for t, a in pairs]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
        return out.last_hidden_state[:, 0, :].cpu().tolist()

    return encode


def _build_hf(model_id: str, pooling: str, device: str):
    """Plain AutoModel checkpoint pooled with ``cls`` or ``mean``."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    sep = tokenizer.sep_token or " "

    def encode(pairs):
        texts = [t + sep + a for t, a in pairs]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
        if pooling == "cls":
            vecs = out.last_hidden_state[:, 0, :]
        else:
            vecs = _mean_pool(out.last_hidden_state, inputs["attention_mask"])
        return vecs.cpu().tolist()

    return encode


def _build_st(model_id: str, prefix: str, device: str):
    """SentenceTransformer checkpoint; pooling and normalisation come from its config."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device=device)
    sep = getattr(model.tokenizer, "sep_token", None) or " "

    def encode(pairs):
        texts = [prefix + t + sep + a for t, a in pairs]
        vecs = model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    return encode


def _build_luar(model_id: str, device: str):
    """LUAR authorship model: expects (batch, episode_length, seq_len) inputs.

    One text per row means an episode of length 1, so the batch of B texts is
    reshaped to (B authors, 1 episode entry, seq_len).
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()
    sep = tokenizer.sep_token or " "

    def encode(pairs):
        texts = [t + sep + a for t, a in pairs]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        seq_len = inputs["input_ids"].shape[-1]
        inputs = {
            "input_ids": inputs["input_ids"].reshape(len(texts), 1, seq_len).to(device),
            "attention_mask": inputs["attention_mask"]
            .reshape(len(texts), 1, seq_len)
            .to(device),
        }
        with torch.no_grad():
            out = model(**inputs)
        return out.cpu().tolist()

    return encode


def _openai_requests(texts: list[str], max_tokens: int):
    """Split texts into sub-requests that stay under the per-request token cap.

    Greedy packing on an estimated token count -- a real tokenizer would be more
    accurate but would add a dependency for what only needs to be a safe
    under-estimate of how much fits in one call.
    """
    chunk: list[str] = []
    chunk_tokens = 0
    for text in texts:
        tokens = max(1, len(text) // OPENAI_CHARS_PER_TOKEN)
        if chunk and chunk_tokens + tokens > max_tokens:
            yield chunk
            chunk, chunk_tokens = [], 0
        chunk.append(text)
        chunk_tokens += tokens
    if chunk:
        yield chunk


def _build_openai(model_id: str):
    """One OpenAI embedding model, at its native dimensionality."""
    from openai import OpenAI

    client = OpenAI()

    def encode(pairs):
        texts = [t + " [SEP] " + a for t, a in pairs]
        # One over-long text would fail the whole request on its own.
        limit = OPENAI_MAX_TOKENS_PER_INPUT * OPENAI_CHARS_PER_TOKEN
        texts = [text[:limit] for text in texts]

        vectors = []
        for chunk in _openai_requests(texts, OPENAI_MAX_TOKENS_PER_REQUEST):
            response = client.embeddings.create(input=chunk, model=model_id)
            # The API does not promise input order, so sort by the returned index.
            vectors.extend(
                item.embedding for item in sorted(response.data, key=lambda d: d.index)
            )
        return vectors

    return encode


def add_embedding(
    df: pd.DataFrame, emb_type: str, batch_size: int = DEFAULT_BATCH_SIZE
) -> pd.DataFrame:
    """Add an emb_{emb_type} column to df using title + sep + abstract as input.

    See the module docstring for the supported ``emb_type`` values (also exposed
    as ``EMB_TYPES``) and the optional dependency each of them pulls in.

    ``batch_size`` is the number of texts per forward pass (per request for
    ``openai``, which accepts far larger batches than the local models). Lower it
    for the heavier checkpoints such as ``qwen3`` if you run out of GPU memory.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    col = f"emb_{emb_type}"

    if emb_type == "specter2":
        encode = _build_specter2(device)
    elif emb_type in _HF_MODELS:
        model_id, pooling = _HF_MODELS[emb_type]
        encode = _build_hf(model_id, pooling, device)
    elif emb_type in _ST_MODELS:
        model_id, prefix = _ST_MODELS[emb_type]
        encode = _build_st(model_id, prefix, device)
    elif emb_type in _LUAR_MODELS:
        encode = _build_luar(_LUAR_MODELS[emb_type], device)
    elif emb_type in _OPENAI_MODELS:
        encode = _build_openai(_OPENAI_MODELS[emb_type])
    else:
        raise ValueError(
            f"Unknown emb_type: {emb_type!r}. Choose from: {', '.join(EMB_TYPES)}"
        )

    pairs = [
        (row["title"] or "", row["abstract"] or "") for _, row in df.iterrows()
    ]

    try:
        embeddings = []
        with tqdm(total=len(pairs), desc=f"Embedding ({emb_type})") as bar:
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                embeddings.extend(encode(batch))
                bar.update(len(batch))

        df[col] = embeddings
        return df
    finally:
        # Drop the model before the caller loads the next one. The closure is the
        # only reference to it, so deleting it and collecting frees the weights;
        # empty_cache then returns the allocator's blocks to the driver. Without
        # this every model of a multi-embedder run stays resident on the GPU and
        # the later, larger ones run out of memory.
        del encode
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
