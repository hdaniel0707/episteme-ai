"""Recompute an embedding column in a Parquet from its text column, in place.

Reads a Parquet, re-embeds the ``--text-col`` with the same function used by the
prepare_* scripts (``epai.utils.embedding_utils.add_embedding``) and overwrites
the ``--emb-col`` with the fresh vectors.

Both columns must already exist in the Parquet — if either is missing the script
aborts without computing anything. Rows whose text is empty (None/NaN/blank) are
skipped and their embedding is set to Null.

Like the prepare_* scripts for text-only corpora (essay/wp/reuter), the full text
is passed to add_embedding as the *title* with an empty *abstract*, so the values
produced here are directly comparable to the ones already in the file.

At the end a summary reports how many rows were skipped, how many are newly
filled, and how many embeddings actually changed. "Changed" is measured by cosine
similarity between the old and the new vector, not exact equality: re-running the
same model on the same text rarely reproduces bit-identical floats (GPU/CPU
kernels, batching, library versions), so tiny drift is separated from real change:

    cos >= 0.999999   identical (numerically the same vector)
    cos >= 0.99       drifted   (float noise / minor nondeterminism)
    cos <  0.99       changed   (a genuinely different embedding)

Rows where the old value was Null, or had a different dimension, are counted
separately since there is nothing meaningful to compare against.

Run:
    uv run python epai/ai_detection/prepare/recompute_embedding.py data/parquet/essay_binox0.parquet
    uv run python epai/ai_detection/prepare/recompute_embedding.py data/parquet/reuter_binox0.parquet
    uv run python epai/ai_detection/prepare/recompute_embedding.py data/parquet/wp_binox0.parquet

    uv run python epai/ai_detection/prepare/recompute_embedding.py data/parquet/essay_gpt54mini_binox0.parquet
    uv run python epai/ai_detection/prepare/recompute_embedding.py data/parquet/reuter_gpt54mini_binox0.parquet
    uv run python epai/ai_detection/prepare/recompute_embedding.py data/parquet/wp_gpt54mini_binox0.parquet
        
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.embedding_utils import add_embedding
from epai.utils.parquet_utils import load_parquet, save_parquet

BASE_DIR = Path(__file__).parent.parent.parent.parent

DEFAULT_TEXT_COL = "text"
DEFAULT_EMB_COL = "emb_specter2"
DEFAULT_EMB_TYPE = "specter2"

# Cosine-similarity thresholds used to classify an old/new embedding pair.
IDENTICAL_COS = 0.999999
DRIFT_COS = 0.99


def _is_empty_text(value) -> bool:
    """True for None/NaN or a string that is empty or only whitespace."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def _as_vector(value) -> np.ndarray | None:
    """Coerce a stored embedding cell to a float vector, or None if it has none."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(value, dtype=float)
        return arr if arr.size else None
    return None


def _check_columns(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{path} is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def recompute(
    df: pd.DataFrame,
    text_col: str,
    emb_col: str,
    emb_type: str,
) -> tuple[pd.DataFrame, list, list]:
    """Overwrite emb_col with fresh embeddings of text_col.

    Empty texts get a Null embedding and are never sent to the model. Returns
    (df, old_values, new_values) with the two value lists aligned to df's rows so
    the caller can diff them.
    """
    old_values = list(df[emb_col])
    empty_mask = df[text_col].apply(_is_empty_text)
    to_embed = df.loc[~empty_mask, text_col]

    new_values: list = [None] * len(df)
    if len(to_embed):
        # Same call path as the prepare_* scripts: full text as title, no abstract.
        tmp = pd.DataFrame({"title": to_embed.astype(str), "abstract": ""})
        tmp = add_embedding(tmp, emb_type=emb_type)
        computed = list(tmp[f"emb_{emb_type}"])
        for pos, vec in zip(np.flatnonzero(~empty_mask.to_numpy()), computed):
            new_values[pos] = vec

    df[emb_col] = pd.Series(new_values, index=df.index, dtype="object")
    return df, old_values, new_values


def summarise(old_values: list, new_values: list) -> dict[str, object]:
    """Classify every row by what happened to its embedding."""
    stats: dict[str, object] = {
        "rows": len(new_values),
        "skipped_empty": 0,  # empty text -> embedding set to Null
        "cleared": 0,  # empty text, but there *was* an old embedding
        "embedded": 0,  # embedding computed
        "new": 0,  #   ... where the old value was Null (nothing to compare)
        "dim_mismatch": 0,  #   ... where old/new dimensions differ
        "identical": 0,  #   ... cos >= IDENTICAL_COS
        "drifted": 0,  #   ... DRIFT_COS <= cos < IDENTICAL_COS
        "changed": 0,  #   ... cos < DRIFT_COS
    }
    cosines: list[float] = []

    for old_raw, new_raw in zip(old_values, new_values):
        old = _as_vector(old_raw)
        new = _as_vector(new_raw)

        if new is None:
            stats["skipped_empty"] += 1
            if old is not None:
                stats["cleared"] += 1
            continue

        stats["embedded"] += 1
        if old is None:
            stats["new"] += 1
            continue
        if old.shape != new.shape:
            stats["dim_mismatch"] += 1
            continue

        denom = float(np.linalg.norm(old) * np.linalg.norm(new))
        cos = float(np.dot(old, new) / denom) if denom else 0.0
        cosines.append(cos)

        if cos >= IDENTICAL_COS:
            stats["identical"] += 1
        elif cos >= DRIFT_COS:
            stats["drifted"] += 1
        else:
            stats["changed"] += 1

    stats["cosines"] = cosines
    return stats


def print_summary(
    in_path: Path,
    out_path: Path,
    text_col: str,
    emb_col: str,
    emb_type: str,
    stats: dict[str, object],
) -> None:
    cosines = stats["cosines"]
    comparable = len(cosines)

    print("=" * 74)
    print("Recomputed embeddings")
    print(f"  input       : {in_path}")
    print(f"  output      : {out_path}")
    print(f"  text column : {text_col!r}")
    print(f"  emb column  : {emb_col!r}  (model: {emb_type})")
    print("=" * 74)
    print(f"Rows                                 : {stats['rows']}")
    print(f"Skipped, empty text -> Null          : {stats['skipped_empty']}")
    print(f"  of which had an embedding before   : {stats['cleared']}")
    print(f"Embedded                             : {stats['embedded']}")
    print(f"  new (old value was Null)           : {stats['new']}")
    print(f"  dimension mismatch (old vs new)    : {stats['dim_mismatch']}")
    print(f"  comparable with the old embedding  : {comparable}")

    if comparable:
        arr = np.asarray(cosines)
        print(f"    identical (cos >= {IDENTICAL_COS})   : {stats['identical']}")
        print(f"    drifted   (cos >= {DRIFT_COS})       : {stats['drifted']}")
        print(f"    changed   (cos <  {DRIFT_COS})       : {stats['changed']}")
        print(
            f"  cosine(old, new): min {arr.min():.6f}  "
            f"median {np.median(arr):.6f}  mean {arr.mean():.6f}"
        )
    print()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parquet", type=Path, help="Parquet to recompute embeddings for.")
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output path (default: overwrite the input in place).",
    )
    ap.add_argument(
        "--text-col",
        default=DEFAULT_TEXT_COL,
        help=f"Column holding the text to embed (default: {DEFAULT_TEXT_COL!r}).",
    )
    ap.add_argument(
        "--emb-col",
        default=DEFAULT_EMB_COL,
        help=f"Embedding column to overwrite (default: {DEFAULT_EMB_COL!r}).",
    )
    ap.add_argument(
        "--emb-type",
        default=DEFAULT_EMB_TYPE,
        help=f"Embedding model passed to add_embedding (default: {DEFAULT_EMB_TYPE!r}).",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt and write immediately.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out or args.parquet

    df = load_parquet(args.parquet)

    # Abort before touching the model if either column is absent.
    _check_columns(df, [args.text_col, args.emb_col], args.parquet)

    n_empty = int(df[args.text_col].apply(_is_empty_text).sum())
    print(
        f"Will re-embed {len(df) - n_empty} row(s) from {args.text_col!r} into "
        f"{args.emb_col!r} ({args.emb_type}); {n_empty} empty row(s) -> Null."
    )
    if not args.yes:
        answer = (
            input(f"Overwrite {args.emb_col!r} and write to {out_path}? Type 'yes': ")
            .strip()
            .lower()
        )
        if answer != "yes":
            print("Aborted — no changes made.")
            return

    df, old_values, new_values = recompute(
        df, args.text_col, args.emb_col, args.emb_type
    )
    stats = summarise(old_values, new_values)
    print_summary(
        args.parquet, out_path, args.text_col, args.emb_col, args.emb_type, stats
    )

    save_parquet(df, out_path)


if __name__ == "__main__":
    main()
