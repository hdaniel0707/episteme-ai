"""
Transform the Ghostbuster Reuters corpus (copied to data/reuter) into a Parquet
file, following the same conventions as the other prepare_* scripts
(e.g. epai/prepare_denario.py).

Directory layout produced by ghostbuster-data:

    data/reuter/<generator>/<author>/<n>.txt      # one article per file (n = 1..20)
    data/reuter/<generator>/<author>/logprobs/    # ignored (token log-probs)
    data/reuter/gpt/<author>/headlines/           # ignored (headlines, not articles)

Each top-level <generator> directory is a different producer of the text:
human, gpt, claude, gpt_prompt1, gpt_prompt2, gpt_semantic, gpt_writing.

Output schema (data/parquet/reuter.parquet):
    id            -> "reut:<author>:<n>"  (per-source-story id)
    author        -> author directory name
    text          -> raw article text
    generated_by  -> generator directory name (human / gpt / claude / ...)
    emb_specter2  -> SPECTER2 embedding of the article text (Null unless ADD_EMB_SPECTER2)

The id is the (author, file-number) of the source story, so the human original
and every generated rewrite of the same story share one id (differing only by
generated_by) — mirroring how paper_id links human/LLM rows in the other
datasets, which downstream analyses (e.g. human-vs-LLM distance) rely on.

Run:
    python -m epai.prepare_reuter
"""

from pathlib import Path

import pandas as pd

from epai.utils.embedding_utils import add_embedding
from epai.utils.parquet_utils import save_parquet

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

######################################
REUTER_DIR = DATA_DIR / "reuter"
PARQUET_PATH = DATA_DIR / "parquet" / "reuter.parquet"

# Subdirectories inside each author folder that are not articles.
IGNORE_SUBDIRS = {"logprobs", "headlines"}

DEBUG = False  # if True, only process the first 20 files
ADD_EMB_SPECTER2 = True
######################################

# Canonical column order for this dataset.
REFERENCE_COLUMNS = ["id", "author", "text", "generated_by", "emb_specter2"]


def load_reuter(root: Path) -> pd.DataFrame:
    """Collect every article .txt under data/reuter into a tidy DataFrame.

    Files are gathered in a deterministic order (generator, author, numeric
    filename) so the running "reut:<i>" ids are stable across runs.
    """
    records = []
    for generator_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        generated_by = generator_dir.name
        for author_dir in sorted(p for p in generator_dir.iterdir() if p.is_dir()):
            if author_dir.name in IGNORE_SUBDIRS:
                continue
            author = author_dir.name
            # Only the article files live at the author root; logprobs/headlines
            # are nested subdirectories and are skipped by globbing "*.txt".
            for txt_path in sorted(
                author_dir.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
            ):
                records.append(
                    {
                        "id": f"reut:{author}:{txt_path.stem}",
                        "author": author,
                        "text": txt_path.read_text(encoding="utf-8").strip(),
                        "generated_by": generated_by,
                    }
                )

    return pd.DataFrame.from_records(
        records, columns=["id", "author", "text", "generated_by"]
    )


def add_text_embedding(df: pd.DataFrame) -> pd.DataFrame:
    """Attach emb_specter2 for the article text.

    add_embedding embeds "title + sep + abstract"; the Reuters articles have no
    title/abstract split, so the full text is passed as the title with an empty
    abstract. The embedding is computed on a scratch frame to keep df clean.
    """
    tmp = pd.DataFrame({"title": df["text"], "abstract": ""})
    tmp = add_embedding(tmp, emb_type="specter2")
    df["emb_specter2"] = tmp["emb_specter2"].values
    return df


def main() -> None:
    df = load_reuter(REUTER_DIR)
    if DEBUG:
        df = df.head(20)
    print(f"Loaded {len(df)} articles from {REUTER_DIR}")
    print(df["generated_by"].value_counts())

    if ADD_EMB_SPECTER2:
        df = add_text_embedding(df)
    else:
        df["emb_specter2"] = pd.Series([None] * len(df), dtype="object")

    save_parquet(df[REFERENCE_COLUMNS], PARQUET_PATH)


if __name__ == "__main__":
    main()
