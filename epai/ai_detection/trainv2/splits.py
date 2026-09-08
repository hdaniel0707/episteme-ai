"""The SCRIBE benchmark: its four splits, their parquet files, their paper names.

One registry, imported by every trainv2 script that works across the benchmark
(``calibrate_*_baseline.py``, ``run_embedding_effectiveness.py``). The short
keys (``sci``/``gb``/``gb_orig``/``denario``) are what a ``--splits`` /
``--corpora`` flag takes and what lands in a CSV's ``split`` column; the
``DISPLAY`` names are for figure labels only.
"""

from __future__ import annotations

from pathlib import Path

POOLED = "all"  #: the pooled-over-everything pseudo-split

#: short key -> parquet basename under ``data/parquet/``. This dict's order is
#: the fixed left-to-right order every table and figure uses: the two Ghostbuster
#: splits (original GPT-3.5 generations, then the modern suite), then the
#: scholarly split, then the Denario OOD holdout. The pooled column ('all' /
#: 'mean') always comes last, after these.
SPLITS: dict[str, str] = {
    "gb_orig": "scribe_gb_orig.parquet",
    "gb": "scribe_gb.parquet",
    "sci": "scribe_sci.parquet",
    "denario": "scribe_denario.parquet",
}

#: short key -> paper-facing name, for plot titles / legends only.
DISPLAY: dict[str, str] = {
    "gb_orig": "SCRIBE-GB-orig",
    "gb": "SCRIBE-GB",
    "sci": "SCRIBE-Sci",
    "denario": "SCRIBE-Denario",
    "all": "SCRIBE-All",
    "mean": "mean",
}

#: short key -> matplotlib marker, for any figure that encodes the split as a
#: point shape. The two Ghostbuster splits are a triangle pair (up = original
#: GPT-3.5, down = modern suite); ``sci`` is a square, ``denario`` a diamond;
#: the pooled column ('all' / 'mean') is a circle -- reserved, so a per-split
#: point never collides with the mean marker.
MARKER: dict[str, str] = {
    "gb_orig": "^",
    "gb": "v",
    "sci": "s",
    "denario": "D",
    "all": "o",
    "mean": "o",
}


def display(key: str) -> str:
    return DISPLAY.get(key, key)


def marker(key: str) -> str:
    return MARKER.get(key, "P")


def resolve(names: list[str], parquet_dir: Path) -> list[tuple[str, Path]]:
    """``["all"]`` / ``[]`` -> every split; else the named subset. ->
    ``[(key, parquet_path), ...]`` in registry order."""
    if not names or names == ["all"]:
        chosen = list(SPLITS)
    else:
        unknown = [n for n in names if n not in SPLITS]
        if unknown:
            raise SystemExit(
                f"unknown split(s) {unknown}. choose from {list(SPLITS)} or 'all'"
            )
        chosen = [k for k in SPLITS if k in names]
    return [(k, parquet_dir / SPLITS[k]) for k in chosen]
