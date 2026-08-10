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
    generated_by  -> generator directory name, with a name of the form
                     <model_slug>_<prompt_id> rewritten to the corpus label
                     <model_slug>:<prompt_id> (gpt56luna_0701A ->
                     gpt56luna:0701A); every other name is kept verbatim
                     (human / gpt / claude / gpt_prompt1 / ...). See
                     epai/utils/label_utils.py.
    emb_specter2  -> SPECTER2 embedding of the article text (Null unless ADD_EMB_SPECTER2)

The id is the (author, file-number) of the source story, so the human original
and every generated rewrite of the same story share one id (differing only by
generated_by) — mirroring how paper_id links human/LLM rows in the other
datasets, which downstream analyses (e.g. human-vs-LLM distance) rely on.

Run:
    uv run python epai/ai_detection/prepare/prepare_reuter.py
    uv run python epai/ai_detection/prepare/prepare_reuter.py --source-dir data/ghostbuster-data/reuter -o data/parquet/reuter.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

from epai.utils.embedding_utils import add_embedding
from epai.utils.label_utils import generator_dirs, label_from_dir_name
from epai.utils.parquet_utils import save_parquet

BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"

DEFAULT_SOURCE_DIR = DATA_DIR / "ghostbuster-data-gpt54" / "reuter"
DEFAULT_OUT = DATA_DIR / "parquet" / "reuter_gpt54mini.parquet"

# Subdirectories inside each author folder that are not articles.
IGNORE_SUBDIRS = {"logprobs", "headlines"}


# Canonical column order for this dataset.
REFERENCE_COLUMNS = ["id", "author", "text", "generated_by", "emb_specter2"]


def load_reuter(root: Path, generators: list[str] | None = None) -> pd.DataFrame:
    """Collect every article .txt under data/reuter into a tidy DataFrame.

    Files are gathered in a deterministic order (generator, author, numeric
    filename) so the running "reut:<i>" ids are stable across runs.
    """
    records = []
    for generator_dir in generator_dirs(root, generators):
        generated_by = label_from_dir_name(generator_dir.name)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
                   help="Directory of source .txt files to collect.")
    p.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT,
                   help="Output parquet path.")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="Only keep the first N rows (a quick smoke test).")
    p.add_argument("--generators", nargs="+", default=None, metavar="DIR",
                   help="Only read these generator directories, by name "
                        "(e.g. --generators human gpt56luna_0701A). Default: "
                        "every directory found, which also picks up generators "
                        "left behind by an earlier run — name them when the "
                        "corpus has to hold exactly one run's output.")
    p.add_argument("--no-embedding", action="store_true",
                   help="Leave emb_specter2 empty instead of computing it "
                        "(the slow step; add_embeddings.py can fill it later).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_reuter(args.source_dir, args.generators)
    if args.limit:
        df = df.head(args.limit)
    print(f"Loaded {len(df)} articles from {args.source_dir}")
    print(df["generated_by"].value_counts())

    if args.no_embedding:
        df["emb_specter2"] = pd.Series([None] * len(df), dtype="object")
    else:
        df = add_text_embedding(df)

    save_parquet(df[REFERENCE_COLUMNS], args.out)


if __name__ == "__main__":
    main()
