# Truncation cut-offs of the embedders in `epai/utils/embedding_utils.py`

How much of a document each `emb_*` column actually saw. Compiled 2026-07-22 from
the code in `epai/utils/embedding_utils.py` and verified against each model's
published config on Hugging Face / the OpenAI docs (sources at the bottom).

**The cut-offs are not the same across embedders.** Three of them differ from the
512-token default: `minilm` sees 256 tokens, `mpnet` 384, and `qwen3` up to 32768
(i.e. the whole document). Any comparison across embedding columns — e.g.
`epai/train/run_clf_cross_domain_embeddings.py` — therefore compares *both* the
model and how much text it was given.

Note that `add_embeddings.py` passes the full text as the *title* with an empty
*abstract*, so the limit below applies to the document itself.

## Table

| emb_type | model | truncation set in code | model's own limit | effective cut-off | source of the limit |
|---|---|---|---|---|---|
| `specter2` | allenai/specter2_base + specter2 adapter | `max_length=512` (`_build_specter2`) | 512 (`max_position_embeddings`) | **512 tokens** | explicit in code |
| `scincl` | malteos/scincl | `max_length=512` (`_build_hf`) | 512 | **512 tokens** | explicit in code |
| `scibert` | allenai/scibert_scivocab_uncased | `max_length=512` (`_build_hf`) | 512 | **512 tokens** | explicit in code |
| `roberta` | roberta-base | `max_length=512` (`_build_hf`) | 514 positions → 512 usable | **512 tokens** | explicit in code |
| `mpnet` | sentence-transformers/all-mpnet-base-v2 | none (`_build_st`) | `max_seq_length` **384** (backbone would allow 512) | **384 tokens** | checkpoint's `sentence_bert_config.json` |
| `minilm` | sentence-transformers/all-MiniLM-L6-v2 | none (`_build_st`) | `max_seq_length` **256** (backbone would allow 512) | **256 tokens** | checkpoint's `sentence_bert_config.json` |
| `e5` | intfloat/e5-base-v2 | none (`_build_st`) | `max_seq_length` 512 | **512 tokens** | checkpoint's `sentence_bert_config.json` |
| `bge` | BAAI/bge-base-en-v1.5 | none (`_build_st`) | `max_seq_length` 512 | **512 tokens** | checkpoint's `sentence_bert_config.json` |
| `qwen3` | Qwen/Qwen3-Embedding-0.6B | none (`_build_st`) | no `sentence_bert_config.json`; ST falls back to `min(max_position_embeddings=32768, model_max_length=131072)` | **32768 tokens** (whole document) | derived by sentence-transformers |
| `styledistance` | StyleDistance/styledistance (roberta-base) | none (`_build_st`) | `max_seq_length` 512 | **512 tokens** | checkpoint's `sentence_bert_config.json` |
| `style_wegmann` | AnnaWegmann/Style-Embedding | none (`_build_st`) | `max_seq_length` 512 | **512 tokens** | checkpoint's `sentence_bert_config.json` |
| `luar` | rrivera1849/LUAR-MUD (paraphrase-distilroberta-base-v1) | `max_length=512`, `padding="max_length"` (`_build_luar`) | backbone 514 positions → 512 usable | **512 tokens**, every input padded to exactly 512 | explicit in code |
| `luar_crud` | rrivera1849/LUAR-CRUD | same as `luar` | same | **512 tokens**, padded to 512 | explicit in code |
| `openai_small` | text-embedding-3-small | `text[:OPENAI_MAX_TOKENS_PER_INPUT * OPENAI_CHARS_PER_TOKEN]` = **24 000 characters** | 8192 tokens per input (≈32 000 chars of English) | **24 000 characters ≈ 6 000 tokens** | character slice in code, stricter than the API |
| `openai_large` | text-embedding-3-large | same | 8192 tokens per input | **24 000 characters ≈ 6 000 tokens** | character slice in code, stricter than the API |

512 tokens is roughly 350–400 English words, or ~2 000–2 500 characters.

## What this means

1. **`minilm` and `mpnet` are handicapped, not just smaller.** At 256 and 384
   tokens they see 50% and 75% of what the 512-token models see. Both
   checkpoints' backbones support 512; the shorter limits are the values
   sentence-transformers ships in the checkpoint, and `_build_st` does not
   override them. Setting `model.max_seq_length = 512` after loading would put
   all the BERT-sized encoders on equal footing — worth doing before reading the
   embedding comparison as a statement about the models.

2. **`qwen3` is in a different regime.** It is the only embedder that sees a
   whole long document. If it wins the cross-domain comparison, "it read the
   entire text" is a live explanation alongside "it is the better model".

3. **The OpenAI cut is characters, not tokens, and it binds first.** The code
   slices at 24 000 characters (8 000 × 3), while the API allows 8 192 tokens
   ≈ 32 000 characters of English prose. The 3-chars-per-token constant is a
   deliberate under-estimate, so the effect is a safety margin: documents
   between ~24 000 and ~32 000 characters are cut shorter than they need to be.

4. **LUAR pads every input to the full 512 tokens** (`padding="max_length"`),
   which is why `add_embeddings.py` gives it a smaller batch size. Note also
   that the LUAR model card's basic example uses `max_length=32` — LUAR was
   built for short authorship episodes, so 512 is a defensible choice but not
   the model's native regime.

5. **How much is actually lost depends on the corpora.** The cut-offs above are
   ceilings; a document shorter than the ceiling is unaffected. Cross-check
   against `results/text_lengths/` (from `epai/analyse/analyse_text_lengths.py`)
   to see what share of each corpus exceeds ~380 words, and therefore which
   columns are really comparing truncated text.

## Sources

- [sentence-transformers/all-mpnet-base-v2 `sentence_bert_config.json`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/raw/main/sentence_bert_config.json) — `max_seq_length: 384`
- [sentence-transformers/all-MiniLM-L6-v2 `sentence_bert_config.json`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/raw/main/sentence_bert_config.json) — `max_seq_length: 256`
- [intfloat/e5-base-v2 `sentence_bert_config.json`](https://huggingface.co/intfloat/e5-base-v2/raw/main/sentence_bert_config.json) — `max_seq_length: 512`
- [BAAI/bge-base-en-v1.5 `sentence_bert_config.json`](https://huggingface.co/BAAI/bge-base-en-v1.5/raw/main/sentence_bert_config.json) — `max_seq_length: 512`
- [StyleDistance/styledistance `sentence_bert_config.json`](https://huggingface.co/StyleDistance/styledistance/raw/main/sentence_bert_config.json) — `max_seq_length: 512`
- [AnnaWegmann/Style-Embedding `sentence_bert_config.json`](https://huggingface.co/AnnaWegmann/Style-Embedding/raw/main/sentence_bert_config.json) — `max_seq_length: 512`
- [Qwen/Qwen3-Embedding-0.6B `config.json`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/raw/main/config.json) — `max_position_embeddings: 32768`; [`tokenizer_config.json`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/raw/main/tokenizer_config.json) — `model_max_length: 131072`; no `sentence_bert_config.json` in the repo (404)
- [Qwen3 Embedding model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) — 32K context
- [allenai/specter2_base `config.json`](https://huggingface.co/allenai/specter2_base/raw/main/config.json) — `max_position_embeddings: 512`
- [allenai/scibert_scivocab_uncased `config.json`](https://huggingface.co/allenai/scibert_scivocab_uncased/raw/main/config.json) — 512
- [malteos/scincl `config.json`](https://huggingface.co/malteos/scincl/raw/main/config.json) — 512
- [FacebookAI/roberta-base `config.json`](https://huggingface.co/FacebookAI/roberta-base/raw/main/config.json) — `max_position_embeddings: 514` (2 reserved → 512 usable)
- [rrivera1849/LUAR-MUD `config.json`](https://huggingface.co/rrivera1849/LUAR-MUD/raw/main/config.json) — backbone `paraphrase-distilroberta-base-v1`, `max_position_embeddings: 514`; [model card](https://huggingface.co/rrivera1849/LUAR-MUD) — `(batch, episode_length, max_token_length)` inputs, examples at 32 and 512 tokens
- [OpenAI embeddings guide](https://developers.openai.com/api/docs/guides/embeddings) — 8192 max input tokens, 300 000 tokens summed per request
