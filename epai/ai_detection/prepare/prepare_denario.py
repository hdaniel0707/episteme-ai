"""
Transform the Denario example-papers SQLite DB into a Parquet file that follows
the same schema as the main datasets (e.g. data/parquet/0515_A_500_AI_260.parquet).

The Denario DB only carries a handful of fields (category, title, abstract, ...),
so every column that has no Denario counterpart is written as Null.

Mappings:
    title     -> title
    abstract  -> abstract
    category  -> domains      (stored as a single-element list, like the other datasets)
    pdf_url   -> paper_id      (sha1 hash, matching the 40-char hex ids used elsewhere)
All remaining columns (embedding, citation_count, year, venue, queries,
author_count, best_author_h_index, emb_specter2) are Null.

Run:
    uv run python epai/ai_detection/prepare/prepare_denario.py
    uv run python epai/ai_detection/prepare/prepare_denario.py --db data/sql/denario_papers.db -o data/parquet/denario_papers.parquet
"""

import hashlib
import sqlite3
import argparse
from pathlib import Path

import pandas as pd

from epai.utils.embedding_utils import add_embedding
from epai.utils.parquet_utils import save_parquet

BASE_DIR = Path(__file__).parent.parent.parent.parent
DB_DIR = BASE_DIR / "data"

DEFAULT_DB_ID = "denario_papers"
DEFAULT_DB = DB_DIR / "sql" / (DEFAULT_DB_ID + ".db")
DEFAULT_OUT = DB_DIR / "parquet" / (DEFAULT_DB_ID + ".parquet")

GENERATED_BY = "denario"  # Denario papers are AI-generated end-to-end

# Canonical column order, matching the main datasets.
REFERENCE_COLUMNS = [
    "paper_id",
    "title",
    "abstract",
    "embedding",
    "citation_count",
    "influential_citation_count",
    "year",
    "venue",
    "queries",
    "domains",
    "generated_by",
    "author_count",
    "best_author_h_index",
    "source_db",
    "emb_specter2",
]

# Columns with no Denario counterpart, grouped by the Null representation they need.
NULL_OBJECT_COLUMNS = ["embedding", "queries", "venue", "emb_specter2"]
NULL_INT_COLUMNS = ["citation_count", "influential_citation_count", "year"]
NULL_FLOAT_COLUMNS = ["author_count", "best_author_h_index"]


def make_paper_id(pdf_url: str | None, fallback: object) -> str:
    """Deterministic id, namespaced to avoid collisions with Semantic Scholar paper ids."""
    key = pdf_url if pdf_url else f"fallback:{fallback}"
    return "den:" + hashlib.sha1(key.encode("utf-8")).hexdigest()


def load_denario(path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(path)
    df = pd.read_sql("SELECT * FROM papers", conn)
    conn.close()
    return df


def to_dataset_schema(raw: pd.DataFrame, source_db: str) -> pd.DataFrame:
    n = len(raw)
    out = pd.DataFrame(index=range(n))

    out["paper_id"] = [
        make_paper_id(raw["pdf_url"].iloc[i] if "pdf_url" in raw else None, i)
        for i in range(n)
    ]
    out["title"] = raw.get("title")
    out["abstract"] = raw.get("abstract")
    # Keep the Denario category as the single research domain (list, like the others).
    out["domains"] = [
        [c] if isinstance(c, str) and c else None
        for c in raw.get("category", pd.Series([None] * n))
    ]
    out["generated_by"] = GENERATED_BY
    out["source_db"] = source_db

    for col in NULL_OBJECT_COLUMNS:
        out[col] = pd.Series([None] * n, dtype="object")
    for col in NULL_INT_COLUMNS:
        out[col] = pd.array([pd.NA] * n, dtype="Int64")
    for col in NULL_FLOAT_COLUMNS:
        out[col] = pd.Series([pd.NA] * n, dtype="Float64")

    return out[REFERENCE_COLUMNS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", type=Path, default=DEFAULT_DB,
                   help="SQLite database to read the papers from.")
    p.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT,
                   help="Output parquet path.")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="Only keep the first N rows (a quick smoke test).")
    p.add_argument("--no-embedding", action="store_true",
                   help="Leave emb_specter2 empty instead of computing it.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_denario(args.db)
    if args.limit:
        raw = raw.head(args.limit)
    print(f"Loaded {len(raw)} rows from {args.db}")

    df = to_dataset_schema(raw, source_db=args.db.stem)

    if not args.no_embedding:
        df = add_embedding(df, emb_type="specter2")

    save_parquet(df, args.out)


if __name__ == "__main__":
    main()
