"""Print every scorer column's value for one ``id``, across all its rows.

An ``id`` names one source document; every generator's row for it (the human
original plus each machine ``generated_by`` arm) shares that ``id`` but has its
own ``uid``. This prints each such row side by side, so the human and machine
scores for the same document line up for comparison.

"Scorer" means a plain numeric detector/judge output — every column matching
one of SCORE_PREFIXES below. Not embeddings (``emb_*``), not the LLM judge's
raw cues/reason/logprob-window columns — those are judge internals, not a
score.

Run:
    .venv/bin/python epai/scripts/print_scorer_results.py \
        data/parquet/manugram_gb.parquet essay:1

    # only one scorer/judge family, by substring or glob pattern
    .venv/bin/python epai/scripts/print_scorer_results.py \
        data/parquet/manugram_gb.parquet essay:1 -s fusionjudge

    .venv/bin/python epai/scripts/print_scorer_results.py \
        data/parquet/manugram_gb.parquet essay:1 -s 'llm_judge_score_*'
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

# Every prefix a numeric scorer column is written under, across the detectors
# in this repo (binoculars, fastdetect's crit/prob, the misc `detector_score_*`
# family, the LLM judge, and the two aggregates built on top of it). Excludes
# `fastdetect_ntok_*` / `detector_ntok_*` (token counts, not scores) and
# `emb_*` (embeddings).
SCORE_PREFIXES = (
    "binoculars_score",
    "fastdetect_crit_",
    "fastdetect_prob_",
    "detector_score_",
    "llm_judge_score_",
    "logjudge_ord2_",
    "fusionjudge_",
)

ID_COL = "id"
LABEL_COLS = ("uid", "generated_by")


def score_columns(schema_names: list[str], pattern: str | None) -> list[str]:
    cols = [c for c in schema_names if c.startswith(SCORE_PREFIXES)]
    if pattern is None:
        return cols
    # A literal substring (the common case: -s fusionjudge) as well as an
    # explicit glob (-s 'llm_judge_score_*'), so a bare scorer/judge name
    # works without the caller having to know it needs a trailing '*'.
    matched = [c for c in cols if pattern in c or fnmatch.fnmatchcase(c, pattern)]
    return matched


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parquet", type=Path)
    ap.add_argument("id", help="value of the `id` column to look up")
    ap.add_argument(
        "-s", "--scorer", default=None, metavar="PATTERN",
        help="only scorer columns matching this substring or glob pattern "
        "(e.g. fusionjudge, 'llm_judge_score_*')",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.parquet.is_file():
        raise SystemExit(f"not found: {args.parquet}")

    schema_names = pq.ParquetFile(args.parquet).schema_arrow.names
    if ID_COL not in schema_names:
        raise SystemExit(f"{args.parquet}: no '{ID_COL}' column")

    scores = score_columns(schema_names, args.scorer)
    if not scores:
        msg = f"{args.parquet}: no scorer columns found"
        if args.scorer:
            msg += f" matching {args.scorer!r}"
        raise SystemExit(msg)

    label_cols = [c for c in LABEL_COLS if c in schema_names]
    table = pq.read_table(args.parquet, columns=[ID_COL] + label_cols + scores)
    table = table.filter(pc.equal(table.column(ID_COL).cast("string"), args.id))

    if table.num_rows == 0:
        raise SystemExit(f"{args.parquet}: no row with {ID_COL}={args.id!r}")

    rows = table.to_pylist()
    for row in rows:
        label = "  ".join(f"{c}={row[c]}" for c in label_cols)
        print(f"\n{label}" if label else "\nrow")
        for col in scores:
            print(f"  {col}: {row[col]}")


if __name__ == "__main__":
    main()
