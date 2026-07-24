"""Print the statistics of a Parquet written by ``epai/ai_detection/embed/add_embeddings.py``.

Three sections, all to stdout:

  1. File     -- path, size on disk, rows, columns, which column holds the text.
  2. Columns  -- every column. The ``emb_*`` ones get their embedding dimension,
     element dtype, mean L2 norm and a count of vectors holding NaN/Inf; the
     rest get dtype, unique count and either numeric or text-length statistics.
  3. Missing  -- per-column missing counts with example row positions, plus the
     rows that no embedder covers at all.

"Missing" is decided per row on the value, not on the schema: a column that
exists but is half Null counts as half missing, and an embedding cell counts as
missing when it is None/NaN or a zero-length vector. Where a text column is
present, missing embeddings are split into rows that *have* a text (a real gap,
add_embeddings would fill them) and rows whose text is empty (expected to be
Null -- add_embeddings never sends those to a model).

Dimensions are read from the values rather than from the Parquet schema, so a
column whose rows disagree is reported as MIXED with every width it contains
instead of silently showing the first one.

Run:
    uv run python epai/embed/summary_parquet.py store/wp_gpt54mini_binox0.parquet
    uv run python epai/embed/summary_parquet.py store/essay_binox0.parquet --max-rows 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.parquet_utils import load_parquet

BASE_DIR = Path(__file__).parent.parent.parent.parent

# Prefix add_embeddings gives every embedding column it writes.
EMB_PREFIX = "emb_"

DEFAULT_TEXT_COL = "text"

# How many example row positions to list per column before truncating.
MAX_ROWS_SHOWN = 20

# Width of the rules between sections.
RULE = 92


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parquet", type=Path, help="Parquet to summarise.")
    ap.add_argument(
        "--text-col",
        default=DEFAULT_TEXT_COL,
        help=f"Column holding the embedded text, used to tell a real gap from an "
        f"expected Null (default: {DEFAULT_TEXT_COL!r}; ignored if absent).",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=MAX_ROWS_SHOWN,
        help=f"Example row positions to list per column (default: {MAX_ROWS_SHOWN}).",
    )
    return ap.parse_args()


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
    """Coerce a stored embedding cell to a float vector, or None if it has none.

    Same rule as add_embeddings: anything that is not a non-empty sequence counts
    as "no embedding here".
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(value, dtype=float)
        return arr if arr.size else None
    return None


def _is_missing(value) -> bool:
    """True for a missing value in any column: Null, blank string, empty container."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict, np.ndarray, pd.Series)):
        return len(value) == 0
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _element_dtype(series: pd.Series) -> str:
    """dtype of the numbers inside an embedding column, from the first vector.

    The column itself is ``object`` (each cell is its own array), so the dtype
    that matters -- float32 vs float64, i.e. the file size -- is one level down.
    """
    for value in series:
        if isinstance(value, np.ndarray) and value.size:
            return str(value.dtype)
        if isinstance(value, (list, tuple)) and len(value):
            return f"{type(value[0]).__name__} (python)"
    return "-"


def _fmt_count(n: int, total: int) -> str:
    """"123 (4.6%)", or just "0" when there is nothing to report."""
    if n == 0:
        return "0"
    pct = 100.0 * n / total if total else 0.0
    return f"{n} ({pct:.1f}%)"


def _fmt_positions(positions: list[int], max_rows: int) -> str:
    """Example row positions, truncated with a count of what was left out."""
    shown = positions[:max_rows]
    body = ", ".join(str(p) for p in shown)
    if len(positions) > max_rows:
        body += f", ... (+{len(positions) - max_rows} more)"
    return f"[{body}]"


def _file_size(path: Path) -> str:
    mib = path.stat().st_size / 1024**2
    return f"{mib:.1f} MiB" if mib < 1024 else f"{mib / 1024:.2f} GiB"


def embedding_columns(df: pd.DataFrame) -> list[str]:
    """The ``emb_*`` columns, in the order they appear in the file."""
    return [c for c in df.columns if c.startswith(EMB_PREFIX)]


def describe_embedding(series: pd.Series, has_text: np.ndarray | None) -> dict:
    """Per-row statistics for one embedding column.

    ``has_text`` is the mask of rows with a non-empty text, or None when the file
    has no text column -- without it a missing embedding cannot be told apart
    from an embedding that is *supposed* to be Null.
    """
    vectors = [_as_vector(v) for v in series]
    filled = np.array([v is not None for v in vectors], dtype=bool)
    present = [v for v in vectors if v is not None]

    dims = sorted({v.size for v in present})
    norms = np.array([float(np.linalg.norm(v)) for v in present]) if present else None
    n_nonfinite = sum(1 for v in present if not np.isfinite(v).all())
    # All-zero vectors are not Null but carry no signal, and a cosine against
    # them is undefined -- worth separating from the healthy rows.
    n_zero = sum(1 for v in present if not np.any(v))

    missing = ~filled
    if has_text is None:
        missing_with_text, missing_no_text = missing, np.zeros(len(series), dtype=bool)
    else:
        missing_with_text = missing & has_text
        missing_no_text = missing & ~has_text

    return {
        "n_filled": int(filled.sum()),
        "dims": dims,
        "element_dtype": _element_dtype(series),
        "norms": norms,
        "n_nonfinite": n_nonfinite,
        "n_zero": n_zero,
        "missing": missing,
        "missing_with_text": missing_with_text,
        "missing_no_text": missing_no_text,
    }


def dim_label(dims: list[int]) -> str:
    if not dims:
        return "-"
    if len(dims) == 1:
        return str(dims[0])
    return "MIXED " + "/".join(str(d) for d in dims)


def print_file(path: Path, df: pd.DataFrame, text_col: str | None) -> None:
    emb_cols = embedding_columns(df)
    print("=" * RULE)
    print("File")
    print(f"  path          : {path}")
    print(f"  size on disk  : {_file_size(path)}")
    print(f"  rows          : {len(df)}")
    print(f"  columns       : {len(df.columns)}  ({len(emb_cols)} embedding)")
    if text_col is None:
        print("  text column   : - (none found, missing embeddings cannot be judged)")
    else:
        n_empty = int(df[text_col].apply(_is_empty_text).sum())
        print(f"  text column   : {text_col!r}  ({n_empty} empty)")


def print_embeddings(df: pd.DataFrame, stats: dict[str, dict]) -> None:
    """The embedding columns: dimension first, then how complete they are."""
    n_rows = len(df)
    print()
    print("=" * RULE)
    print(f"Embedding columns ({len(stats)})")
    print("=" * RULE)
    if not stats:
        print(f"  (none — no {EMB_PREFIX}* column in this file)")
        return

    header = (
        f"{'column':<24}{'dim':>12}{'dtype':>10}{'filled':>9}"
        f"{'missing':>9}{'|v| mean':>10}  note"
    )
    print(header)
    print("-" * RULE)
    for col, st in stats.items():
        n_missing = int(st["missing"].sum())
        norms = st["norms"]
        norm_label = f"{norms.mean():.3f}" if norms is not None and norms.size else "-"

        notes = []
        if len(st["dims"]) > 1:
            notes.append("inconsistent dimensions")
        if st["n_nonfinite"]:
            notes.append(f"{st['n_nonfinite']} with NaN/Inf")
        if st["n_zero"]:
            notes.append(f"{st['n_zero']} all-zero")
        if st["n_filled"] == 0:
            notes.append("empty column")
        elif n_missing:
            n_gap = int(st["missing_with_text"].sum())
            notes.append(f"{n_gap} gap(s) with a text" if n_gap else "only empty texts")

        print(
            f"{col:<24}{dim_label(st['dims']):>12}{st['element_dtype']:>10}"
            f"{st['n_filled']:>9}{n_missing:>9}{norm_label:>10}  {', '.join(notes)}"
        )
    print("-" * RULE)

    # A near-1.0 mean norm means the model already returns unit vectors, which
    # decides whether a downstream cosine needs its own normalisation step.
    print(f"{'column':<24}{'|v| min':>12}{'|v| median':>14}{'|v| max':>12}")
    for col, st in stats.items():
        norms = st["norms"]
        if norms is None or not norms.size:
            continue
        print(
            f"{col:<24}{norms.min():>12.4f}{np.median(norms):>14.4f}{norms.max():>12.4f}"
        )
    print(f"  ({n_rows} rows in the file)")


def print_other_columns(df: pd.DataFrame, emb_cols: set[str]) -> None:
    """Everything that is not an embedding: dtype, uniques, and a value summary."""
    n_rows = len(df)
    others = [c for c in df.columns if c not in emb_cols]
    print()
    print("=" * RULE)
    print(f"Other columns ({len(others)})")
    print("=" * RULE)
    if not others:
        print("  (none)")
        return

    print(f"{'column':<24}{'dtype':<12}{'filled':>9}{'unique':>9}  summary")
    print("-" * RULE)
    for col in others:
        series = df[col]
        missing = series.apply(_is_missing).to_numpy()
        n_filled = int((~missing).sum())
        values = series[~missing]

        try:
            n_unique = values.nunique()
        except TypeError:  # unhashable cells (lists, dicts)
            n_unique = -1
        unique_label = "-" if n_unique < 0 else str(n_unique)

        summary = ""
        if n_filled:
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
                series
            ):
                summary = (
                    f"min {values.min():.4g}  median {values.median():.4g}  "
                    f"mean {values.mean():.4g}  max {values.max():.4g}"
                )
            elif values.map(lambda v: isinstance(v, str)).all():
                lengths = values.str.len()
                if n_unique >= 0 and n_unique <= 8:
                    counts = values.value_counts()
                    summary = ", ".join(f"{k!r}: {v}" for k, v in counts.items())
                else:
                    summary = (
                        f"chars: min {int(lengths.min())}  "
                        f"median {int(lengths.median())}  max {int(lengths.max())}"
                    )
            else:
                summary = f"e.g. {values.iloc[0]!r}"[:60]

        print(
            f"{col:<24}{str(series.dtype):<12}{n_filled:>9}{unique_label:>9}  {summary}"
        )
    print("-" * RULE)
    print(f"  ({n_rows} rows in the file)")


def print_missing(
    df: pd.DataFrame,
    emb_stats: dict[str, dict],
    text_col: str | None,
    max_rows: int,
) -> None:
    """Every column that has a hole, with example row positions."""
    n_rows = len(df)
    emb_cols = set(emb_stats)
    print()
    print("=" * RULE)
    print("Missing data")
    print("=" * RULE)

    any_missing = False
    for col in df.columns:
        if col in emb_cols:
            st = emb_stats[col]
            missing = st["missing"]
        else:
            missing = df[col].apply(_is_missing).to_numpy()

        n_missing = int(missing.sum())
        if n_missing == 0:
            continue
        any_missing = True

        positions = np.flatnonzero(missing).tolist()
        print(f"  {col}: {_fmt_count(n_missing, n_rows)} missing")
        if col in emb_cols and text_col is not None:
            n_gap = int(emb_stats[col]["missing_with_text"].sum())
            n_no_text = int(emb_stats[col]["missing_no_text"].sum())
            print(
                f"      with a non-empty text (real gap) : {n_gap}\n"
                f"      text empty (expected Null)       : {n_no_text}"
            )
        print(f"      rows: {_fmt_positions(positions, max_rows)}")

    if not any_missing:
        print("  (none — every column is fully populated)")

    if emb_stats:
        # Rows no embedder covers are the ones that drop out of every downstream
        # experiment at once, so they are worth calling out separately.
        all_missing = np.ones(n_rows, dtype=bool)
        any_missing_row = np.zeros(n_rows, dtype=bool)
        for st in emb_stats.values():
            all_missing &= st["missing"]
            any_missing_row |= st["missing"]
        print()
        print(f"  rows missing at least one embedding : {_fmt_count(int(any_missing_row.sum()), n_rows)}")
        print(f"  rows missing every embedding        : {_fmt_count(int(all_missing.sum()), n_rows)}")
        if all_missing.any():
            print(f"      rows: {_fmt_positions(np.flatnonzero(all_missing).tolist(), max_rows)}")
    print("=" * RULE)


def main() -> None:
    args = parse_args()

    df = load_parquet(args.parquet)
    text_col = args.text_col if args.text_col in df.columns else None
    has_text = (
        ~df[text_col].apply(_is_empty_text).to_numpy() if text_col is not None else None
    )

    emb_stats = {
        col: describe_embedding(df[col], has_text) for col in embedding_columns(df)
    }

    print_file(args.parquet, df, text_col)
    print_embeddings(df, emb_stats)
    print_other_columns(df, set(emb_stats))
    print_missing(df, emb_stats, text_col, args.max_rows)


if __name__ == "__main__":
    main()
