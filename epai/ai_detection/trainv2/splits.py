"""The MANUGRAM benchmark: its splits, their parquet files, their paper names.

One registry, imported by every trainv2 script that works across the benchmark
(``calibrate_*_baseline.py``, ``run_embedding_effectiveness.py``,
``run_zeroshot_performance.py`` …). The short keys are what a ``--splits`` /
``--corpora`` flag takes and what lands in a CSV's ``split`` column; the
``DISPLAY`` names are for figure labels only.

:data:`DEFAULT` is what a bare flag (or ``all``) expands to: the four core
splits. **MG-Reph and its two halves are registered but NOT in the default** --
MG-Reph is human vs. one generator (gpt-5.6-luna) with only the gpt56luna judge
scored on both classes, so it does not belong in the pooled calibrations or the
cross-generator / multi-judge runs. Pass it explicitly for a targeted
experiment: ``--splits mg_reph`` / ``mg_reph_sci`` / ``mg_reph_gb``
(``split_manugram_reph.py`` writes the two halves).
"""

from __future__ import annotations

from pathlib import Path

POOLED = "all"  #: the pooled-over-everything pseudo-split

#: short key -> parquet basename under ``data/parquet/``. This dict's order is
#: the fixed left-to-right order every table and figure uses: the original
#: Ghostbuster split (GPT-3.5), then the modern Ghostbuster suite, then the
#: scholarly split, then the rephrase split (+ its ghostbuster / science
#: halves), then the Denario OOD holdout. The pooled column ('all' / 'mean')
#: always comes last, after these.
SPLITS: dict[str, str] = {
    "gb_orig": "ghostbuster_orig.parquet",
    "mg_gb": "manugram_gb.parquet",
    "mg_sci": "manugram_sci.parquet",
    "mg_reph": "manugram_reph.parquet",
    "mg_reph_sci": "manugram_reph_sci.parquet",
    "mg_reph_gb": "manugram_reph_gb.parquet",
    "mg_den": "manugram_den.parquet",
}

#: what an empty / ``["all"]`` ``--splits`` / ``--corpora`` expands to, and what
#: ``rebuild_all.py`` runs. The MG-Reph family is deliberately absent -- see the
#: module docstring.
DEFAULT: tuple[str, ...] = ("gb_orig", "mg_gb", "mg_sci", "mg_den")

#: short key -> paper-facing name, for plot titles / legends only.
DISPLAY: dict[str, str] = {
    "gb_orig": "GB-orig",
    "mg_gb": "MG-GB",
    "mg_sci": "MG-Sci",
    "mg_reph": "MG-Reph",
    "mg_reph_sci": "MG-Reph-Sci",
    "mg_reph_gb": "MG-Reph-GB",
    "mg_den": "MG-Den",
    "all": "MG-All",
    "mean": "mean",
}

#: short key -> matplotlib marker, for any figure that encodes the split as a
#: point shape. The two Ghostbuster splits are a triangle pair (up = original
#: GPT-3.5, down = modern suite); ``mg_sci`` is a square, the ``mg_reph`` family
#: an x / plus / star, ``mg_den`` a diamond; the pooled column ('all' / 'mean')
#: is a circle -- reserved, so a per-split point never collides with the mean
#: marker.
MARKER: dict[str, str] = {
    "gb_orig": "^",
    "mg_gb": "v",
    "mg_sci": "s",
    "mg_reph": "X",
    "mg_reph_sci": "P",
    "mg_reph_gb": "*",
    "mg_den": "D",
    "all": "o",
    "mean": "o",
}


def display(key: str) -> str:
    return DISPLAY.get(key, key)


def marker(key: str) -> str:
    return MARKER.get(key, "P")


def resolve(names: list[str], parquet_dir: Path) -> list[tuple[str, Path]]:
    """``["all"]`` / ``[]`` -> :data:`DEFAULT` (the four core splits); else the
    named subset, validated against :data:`SPLITS` (so the opt-in MG-Reph keys
    resolve when asked for by name). -> ``[(key, parquet_path), ...]`` in
    registry order."""
    if not names or names == ["all"]:
        chosen = list(DEFAULT)
    else:
        unknown = [n for n in names if n not in SPLITS]
        if unknown:
            raise SystemExit(
                f"unknown split(s) {unknown}. choose from {list(SPLITS)} or 'all'"
            )
        chosen = [k for k in SPLITS if k in names]
    return [(k, parquet_dir / SPLITS[k]) for k in chosen]
