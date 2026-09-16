"""The MANUGRAM benchmark: its splits, their parquet files, their paper names.

One registry, imported by every trainv2 script that works across the benchmark
(``calibrate_*_baseline.py``, ``run_embedding_effectiveness.py``,
``run_zeroshot_performance.py`` …). The short keys are what a ``--splits`` /
``--corpora`` flag takes and what lands in a CSV's ``split`` column; the
``DISPLAY`` names are for figure labels only.

:data:`DEFAULT` is what a bare flag (or ``all``) expands to: the four core
splits. **MG-Reph (+ its two halves) and GB-orig-stand are registered but NOT
in the default** -- MG-Reph is human vs. one generator (gpt-5.6-luna) with
only the gpt56luna judge scored on both classes, so it does not belong in the
pooled calibrations or the cross-generator / multi-judge runs; GB-orig-stand
is GB-orig with standardized features, kept only for the ad-hoc
``compare_gb_orig_stand.py`` check (does standardizing change anything? no).
Pass either explicitly for a targeted run: ``--splits mg_reph`` /
``mg_reph_sci`` / ``mg_reph_gb`` / ``gb_orig_stand`` (``split_manugram_reph.py``
writes the MG-Reph halves).

:data:`POOL_SPLITS` is what feeds the pooled ``all`` pseudo-split (MG-All) --
currently all four core splits (a plain arithmetic mean / full pool over
GB-orig, MG-GB, MG-Sci, MG-AI-Sci), kept as its own constant so a split can be
carved out of the pool again later without touching any script.
"""

from __future__ import annotations

from pathlib import Path

POOLED = "all"  #: the pooled-over-everything pseudo-split

#: short key -> parquet basename under ``data/parquet/``. This dict's order is
#: the fixed left-to-right order every table and figure uses: the original
#: Ghostbuster split (GPT-3.5), its standardized-feature twin, then the modern
#: Ghostbuster suite, then the scholarly split, then the rephrase split (+ its
#: ghostbuster / science halves), then the Manugram-AI-Scientist OOD holdout.
#: The pooled column ('all' / 'mean') always comes last, after these.
SPLITS: dict[str, str] = {
    "gb_orig": "ghostbuster_orig.parquet",
    "gb_orig_stand": "ghostbuster_orig_stand.parquet",
    "mg_gb": "manugram_gb.parquet",
    "mg_sci": "manugram_sci.parquet",
    "mg_reph": "manugram_reph.parquet",
    "mg_reph_sci": "manugram_reph_sci.parquet",
    "mg_reph_gb": "manugram_reph_gb.parquet",
    # Was "mg_den" -> manugram_den.parquet (Denario only, 80+80). Switched
    # 2026-09-11 to the merged Manugram-AI-Scientist corpus (200 rows, 100+100):
    # Denario's original 80+80 plus 20 more human-matched pairs against Sakana
    # AI-Scientist v1/v2, CycleResearcher and Zochi (see trainv2's
    # MODEL_BIAS.md and epai/ai_detection/fetch/fetch_ai_scientist_papers.py).
    # The key is renamed too, not just the file, so a `--splits mg_den` typo
    # from an old command fails loudly instead of silently resolving to
    # different data than it used to.
    "mg_aisci": "manugram_aiscientist.parquet",
}

#: what an empty / ``["all"]`` ``--splits`` / ``--corpora`` expands to, and what
#: ``rebuild_all.py`` runs. The MG-Reph family and GB-orig-stand are
#: deliberately absent -- see the module docstring.
DEFAULT: tuple[str, ...] = ("gb_orig", "mg_gb", "mg_sci", "mg_aisci")

#: the splits pooled into the ``all`` pseudo-split (MG-All). Currently all of
#: :data:`DEFAULT` -- see the module docstring.
POOL_SPLITS: tuple[str, ...] = DEFAULT

#: per-figure-kind exclusions, read by :func:`plots.zeroshot_scatter` and
#: :func:`plots.goodness_bars` (the shared machinery behind every AUROC-vs-TPR
#: scatter and goodness-bar figure across trainv2 -- run_zeroshot_performance,
#: run_best_zeroshot_performance, run_in_domain_performance, run_lodo_performance,
#: run_cascade_performance, run_dr_transfer, run_robust_performance, and the
#: MG-Reph rephrase scripts). A split listed under a kind here is still scored
#: and still lands in the CSV -- it just isn't drawn on that kind of figure.
#: Splits not listed are drawn as normal. This is the one place to add/drop a
#: split from every figure of that kind at once; no script or plots.py edit
#: needed. Currently empty -- GB-orig-stand is opt-in only (see DEFAULT), so it
#: never reaches these figures in the first place.
PLOT_EXCLUDE: dict[str, frozenset[str]] = {
    "scatter": frozenset(),
    "goodness": frozenset(),
}

#: short key -> paper-facing name, for plot titles / legends only.
DISPLAY: dict[str, str] = {
    "gb_orig": "GB-orig",
    "gb_orig_stand": "GB-orig-stand",
    "mg_gb": "MG-GB",
    "mg_sci": "MG-Sci",
    "mg_reph": "MG-Reph",
    "mg_reph_sci": "MG-Reph-Sci",
    "mg_reph_gb": "MG-Reph-GB",
    "mg_aisci": "MG-AI-Sci",
    "all": "MG-All",
    "mean": "Mean",
}

#: short key -> matplotlib marker, for any figure that encodes the split as a
#: point shape. The Ghostbuster splits are a triangle family (up = original
#: GPT-3.5, left = its standardized twin, down = modern suite); ``mg_sci`` is a
#: square, the ``mg_reph`` family an x / plus / star, ``mg_aisci`` a diamond;
#: the pooled column ('all' / 'mean') is a circle -- reserved, so a per-split
#: point never collides with the mean marker.
MARKER: dict[str, str] = {
    "gb_orig": "^",
    "gb_orig_stand": "<",
    "mg_gb": "v",
    "mg_sci": "s",
    "mg_reph": "X",
    "mg_reph_sci": "P",
    "mg_reph_gb": "*",
    "mg_aisci": "D",
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
