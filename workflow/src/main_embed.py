"""Phase 2 - embed: compute the frozen title+abstract embeddings (plan section 4.2).

Embedders are frozen, so they run on the full corpus (no leakage). Adds emb_<name> columns
for every embedder in EMBEDDERS, computed on title + [SEP] + abstract.

Embedding vectors are cached by ``paper_id``: a re-run reuses the cached vectors and only
recomputes when a paper is new (or FORCE_EMBED=true). This keeps re-runs cheap AND keeps the
output consistent with the latest ``prepare`` columns (target / split / metadata), so changing
TARGET_KIND never leaves a stale embedding snapshot downstream.

    python -m workflow.src.main_embed
"""

from __future__ import annotations

from workflow.src.iwf import paths
from workflow.src.iwf.config import Settings, get_bool, load_settings
from workflow.src.iwf.data.dataset import load_parquet, save_parquet
from workflow.src.iwf.data.identity import PAPER_ID, keyed_merge
from workflow.src.iwf.embedding.embedders import embed_dataframe

# ===== CONFIG =====
# Embedders come from .env.workflow (EMBEDDERS). Override here to force a set:
EMBEDDERS_OVERRIDE: list[str] | None = None
# ==================


def main(settings: Settings | None = None) -> None:
    s = settings or load_settings()
    embedders = EMBEDDERS_OVERRIDE or s.embedders
    df = load_parquet(paths.prepared_parquet(s.dataset_id))

    # Reuse cached embedding vectors (keyed by paper_id) unless FORCE_EMBED is set. Vectors are
    # attached onto the FRESH prepared df, so prepare's columns are always up to date.
    out_path = paths.embedded_parquet(s.dataset_id)
    if out_path.is_file() and not get_bool("FORCE_EMBED", False):
        cached = load_parquet(out_path)
        emb_cols = [c for c in cached.columns if c.startswith("emb_")]
        if emb_cols and PAPER_ID in cached.columns and PAPER_ID in df.columns:
            df = keyed_merge(df, cached, emb_cols)
            print(f"[embed] reusing cached vectors {emb_cols} by {PAPER_ID}")

    print(f"[embed] computing: {embedders}")
    df = embed_dataframe(df, embedders)   # recomputes only embedders whose column is missing
    save_parquet(df, out_path)


if __name__ == "__main__":
    main()
