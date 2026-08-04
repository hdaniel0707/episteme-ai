r"""Regenerate the step-1 surface-artifact figures from saved CSV, without re-auditing.

``analyse_surface_artifacts.py`` writes every number it plots to CSV alongside
the figures, so redrawing needs no parquet, no text processing and no n-gram
pass — use this after touching anything in the plotting code.

Dispatch is on filename, the same way ``replot_cross_domain.py`` works. A
directory is treated as an audit result when it contains ``sign_table.csv``;
each figure is then redrawn from whichever CSV is present:

  sign_table.csv            -> fig_sign_heatmap.png
  artifact_alignment.csv    -> fig_alignment_heatmap.png
  alignment_vs_transfer.csv -> fig_alignment_vs_accuracy.png
                               fig_alignment_vs_macro_f1.png
                               (only written by runs given --compare-matrix;
                               the Spearman rhos in their titles are recomputed
                               here from the same columns)

The input can be a single run directory or any folder above it — matching
directories are found recursively, so the scope replotted is exactly the folder
you point at.

Run:
    uv run python epai/ai_detection/analyse/replot_surface_artifacts.py \
        results/surface_artifacts/artifacts_20260804_120000
    uv run python epai/ai_detection/analyse/replot_surface_artifacts.py \
        results/surface_artifacts --heatmap-features 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import pandas as pd  # noqa: E402  (after backend set)

from epai.ai_detection.analyse.analyse_surface_artifacts import (  # noqa: E402
    plot_alignment,
    plot_alignment_vs_transfer,
    plot_sign_heatmap,
    spearman,
)

SIGN_TABLE = "sign_table.csv"
ALIGNMENT = "artifact_alignment.csv"
ALIGNMENT_VS_TRANSFER = "alignment_vs_transfer.csv"


def replot_run(run_dir: Path, heatmap_features: int) -> tuple[int, int]:
    """Redraw every figure derivable from the CSVs in one audit directory.

    Returns ``(n_figures, n_errors)``. A missing CSV is not an error — a run
    without ``--compare-matrix`` legitimately has no transfer comparison.
    """
    drawn = errors = 0

    table = pd.read_csv(run_dir / SIGN_TABLE, index_col=0)
    corpora = list(table.columns)
    try:
        plot_sign_heatmap(table, corpora, run_dir, heatmap_features)
        drawn += 1
    except Exception as exc:
        print(f"  [ERROR] {SIGN_TABLE}: {type(exc).__name__}: {exc}")
        errors += 1

    alignment_path = run_dir / ALIGNMENT
    if alignment_path.is_file():
        try:
            plot_alignment(pd.read_csv(alignment_path, index_col=0), run_dir)
            drawn += 1
        except Exception as exc:
            print(f"  [ERROR] {ALIGNMENT}: {type(exc).__name__}: {exc}")
            errors += 1

    merged_path = run_dir / ALIGNMENT_VS_TRANSFER
    if merged_path.is_file():
        try:
            merged = pd.read_csv(merged_path)
            rho_acc = spearman(
                merged["alignment"].to_numpy(), merged["accuracy"].to_numpy()
            )
            rho_f1 = spearman(
                merged["alignment"].to_numpy(), merged["macro_f1"].to_numpy()
            )
            plot_alignment_vs_transfer(merged, rho_acc, rho_f1, run_dir)
            drawn += 1
            print(f"  rho(alignment, accuracy) = {rho_acc:+.3f}   "
                  f"rho(alignment, macro-F1) = {rho_f1:+.3f}")
        except Exception as exc:
            print(f"  [ERROR] {ALIGNMENT_VS_TRANSFER}: {type(exc).__name__}: {exc}")
            errors += 1

    return drawn, errors


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "folder",
        type=Path,
        help="Audit result directory, or any folder containing some. "
        "Searched recursively.",
    )
    ap.add_argument(
        "--heatmap-features",
        type=int,
        default=35,
        help="Features shown in the sign heatmap, strongest first (default: 35). "
        "The CSV holds them all, so this can be raised without re-auditing.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.folder
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    # rglob matches root itself only via its children, so check root separately.
    run_dirs = sorted({p.parent for p in root.rglob(SIGN_TABLE)})
    if not run_dirs:
        raise SystemExit(
            f"No {SIGN_TABLE} found under {root}. Point this at a "
            f"results/surface_artifacts/<run>/ folder or the tree above it."
        )

    print(f"Replotting from: {root}\n")
    total_drawn = total_errors = 0
    for run_dir in run_dirs:
        print(f"[audit]  {run_dir}")
        drawn, errors = replot_run(run_dir, args.heatmap_features)
        total_drawn += drawn
        total_errors += errors

    print(f"\nDone. Redrew {total_drawn} figure(s) across {len(run_dirs)} run(s).")
    if total_errors:
        print(f"{total_errors} figure(s) failed — see the [ERROR] lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
