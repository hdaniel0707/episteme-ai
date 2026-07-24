"""
Transform the Ghostbuster essay corpus, in data/ghostbuster-data/essay, into a
Parquet file, following the same conventions as the other prepare_* scripts
(e.g. epai/prepare_reuter.py).

Directory layout produced by ghostbuster-data:

    data/ghostbuster-data/essay/<generator>/<n>.txt          # one essay per file
    data/ghostbuster-data/essay/<generator>/logprobs/        # ignored (token log-probs)
    data/ghostbuster-data/essay/gpt/logprobs_babbage/        # ignored (token log-probs)
    data/ghostbuster-data/essay/prompts/                     # ignored (the prompts themselves)

Like the WritingPrompts corpus, essays live directly under each <generator>
directory (there is no per-author level), so <n> alone identifies the source
essay. Each top-level <generator> directory is a different producer of the
text: human, gpt, claude, gpt_prompt1, gpt_prompt2, gpt_semantic, gpt_writing.
"prompts" is a sibling directory holding the prompts, not a generator.

Output schema (data/parquet/essay.parquet):
    id            -> "essay:<n>"  (per-source-essay id)
    text          -> raw essay text
    generated_by  -> generator directory name (human / gpt / claude / ...)
    emb_specter2  -> SPECTER2 embedding of the essay text (Null unless ADD_EMB_SPECTER2)

The id is the file-number of the source essay, so the human original and every
generated rewrite of the same essay share one id (differing only by
generated_by) — mirroring how paper_id links human/LLM rows in the other
datasets, which downstream analyses (e.g. human-vs-LLM distance) rely on.

Run:
    python -m epai.ai_detection.prepare.prepare_essay
"""

from pathlib import Path

import pandas as pd

from epai.utils.embedding_utils import add_embedding
from epai.utils.parquet_utils import save_parquet

BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"

######################################
ESSAY_DIR = DATA_DIR / "ghostbuster-data-gpt54" / "essay"
PARQUET_PATH = DATA_DIR / "parquet" / "essay_gpt54mini.parquet"

# Top-level directories that are not text generators.
IGNORE_GENERATORS = {"prompts"}

DEBUG = False  # if True, only process the first 20 files
ADD_EMB_SPECTER2 = True
######################################

# Canonical column order for this dataset.
REFERENCE_COLUMNS = ["id", "text", "generated_by", "emb_specter2"]


def load_essay(root: Path) -> pd.DataFrame:
    """Collect every essay .txt under data/ghostbuster-data/essay into a tidy DataFrame.

    Files are gathered in a deterministic order (generator, numeric filename) so
    the "essay:<n>" ids are stable across runs. The logprobs*/ subdirectories
    are skipped implicitly by globbing "*.txt" only at each generator root.
    """
    records = []
    for generator_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if generator_dir.name in IGNORE_GENERATORS:
            continue
        generated_by = generator_dir.name
        # Only the essay files live at the generator root; logprobs and
        # logprobs_babbage (and any other nested subdirectories) are skipped by
        # globbing "*.txt".
        for txt_path in sorted(
            generator_dir.glob("*.txt"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
        ):
            records.append(
                {
                    "id": f"essay:{txt_path.stem}",
                    "text": txt_path.read_text(encoding="utf-8").strip(),
                    "generated_by": generated_by,
                }
            )

    return pd.DataFrame.from_records(
        records, columns=["id", "text", "generated_by"]
    )


def add_text_embedding(df: pd.DataFrame) -> pd.DataFrame:
    """Attach emb_specter2 for the essay text.

    add_embedding embeds "title + sep + abstract"; the essays have no
    title/abstract split, so the full text is passed as the title with an empty
    abstract. The embedding is computed on a scratch frame to keep df clean.
    """
    tmp = pd.DataFrame({"title": df["text"], "abstract": ""})
    tmp = add_embedding(tmp, emb_type="specter2")
    df["emb_specter2"] = tmp["emb_specter2"].values
    return df


def main() -> None:
    df = load_essay(ESSAY_DIR)
    if DEBUG:
        df = df.head(20)
    print(f"Loaded {len(df)} essays from {ESSAY_DIR}")
    print(df["generated_by"].value_counts())

    if ADD_EMB_SPECTER2:
        df = add_text_embedding(df)
    else:
        df["emb_specter2"] = pd.Series([None] * len(df), dtype="object")

    save_parquet(df[REFERENCE_COLUMNS], PARQUET_PATH)


if __name__ == "__main__":
    main()
