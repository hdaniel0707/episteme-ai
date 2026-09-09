"""The MANUGRAM benchmark: its five splits, their parquet files, their paper names.

One registry, imported by every trainv2 script that works across the benchmark
(``calibrate_*_baseline.py``, ``run_embedding_effectiveness.py``). The short
keys (``gb_orig``/``mg_gb``/``mg_sci``/``mg_reph``/``mg_den``) are what a
``--splits`` / ``--corpora`` flag takes and what lands in a CSV's ``split``
column; the ``DISPLAY`` names are for figure labels only.
"""

from __future__ import annotations

from pathlib import Path

POOLED = "all"  #: the pooled-over-everything pseudo-split

#: short key -> parquet basename under ``data/parquet/``. This dict's order is
#: the fixed left-to-right order every table and figure uses: the original
#: Ghostbuster split (GPT-3.5), then the modern Ghostbuster suite, then the
#: scholarly split, then the rephrase split, then the Denario OOD holdout. The
#: pooled column ('all' / 'mean') always comes last, after these.
SPLITS: dict[str, str] = {
    "gb_orig": "ghostbuster_orig.parquet",
    "mg_gb": "manugram_gb.parquet",
    "mg_sci": "manugram_sci.parquet",
    "mg_reph": "manugram_reph.parquet",
    "mg_den": "manugram_den.parquet",
}

#: short key -> paper-facing name, for plot titles / legends only.
DISPLAY: dict[str, str] = {
    "gb_orig": "GB-orig",
    "mg_gb": "MG-GB",
    "mg_sci": "MG-Sci",
    "mg_reph": "MG-Reph",
    "mg_den": "MG-Den",
    "all": "MG-All",
    "mean": "mean",
}

#: short key -> matplotlib marker, for any figure that encodes the split as a
#: point shape. The two Ghostbuster splits are a triangle pair (up = original
#: GPT-3.5, down = modern suite); ``mg_sci`` is a square, ``mg_reph`` an x,
#: ``mg_den`` a diamond; the pooled column ('all' / 'mean') is a circle --
#: reserved, so a per-split point never collides with the mean marker.
MARKER: dict[str, str] = {
    "gb_orig": "^",
    "mg_gb": "v",
    "mg_sci": "s",
    "mg_reph": "X",
    "mg_den": "D",
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
