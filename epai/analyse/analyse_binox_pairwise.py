"""Compare paired Binoculars parquets (``<dataset>_binox0`` vs ``<dataset>_binox1``).

For a dataset the two files hold the *same* rows (identical ``id`` order) scored
by Binoculars twice. This script pairs the rows, computes the pairwise distance
between the two scorings, draws a violin plot of those distances (one violin per
dataset for side-by-side comparison) and annotates each with its MSE.

Pairing & distance
-------------------
Rows are paired by merging on the composite key ``(id, generated_by)`` — ``id``
alone is not unique, but ``id`` + ``generated_by`` is. For each matched pair the
score vector across the shared ``binoculars_*`` columns is taken from file 0 and
file 1; the pairwise distance is the Euclidean distance between them (``|a - b|``
for the usual single-column case) and the squared errors feed the
MSE = mean over all valid element pairs of ``(a - b)**2``. Keys present in only
one file cannot be paired and are reported in the summary.

Invalid pairs
-------------
If *either* element of a pair is null or not a real number, the pair is skipped,
recorded, and reported in the closing summary statistics.

Figures + tables saved to results/binox_pairwise/<datasets>/:
  - pairwise_distance_violin.png    — |binox0 − binox1| per dataset, MSE annotated
  - pairwise_signed_diff_violin.png — signed binox0 − binox1 (which scoring is larger)
  - pairwise_summary.csv            — per-dataset n, skipped, dist stats, signed mean, MSE
  - pairwise_summary.log            — full text summary incl. skipped-pair details

Usage:
    uv run python epai/analyse/analyse_binox_pairwise.py            # wp, essay, reuter
    uv run python epai/analyse/analyse_binox_pairwise.py wp essay   # subset
    uv run python epai/analyse/analyse_binox_pairwise.py wp2 essay2 reuter2
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent
PARQUET_DIR = BASE_DIR / "data" / "parquet"
RESULTS_DIR = BASE_DIR / "results" / "binox_pairwise"

DEFAULT_DATASETS = ["wp", "essay", "reuter"]
KEY_COLS = ["id", "generated_by"]  # composite key: unique per row within a file


class Tee:
    """Collect printed lines so they can be echoed to stdout and a log file."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def print(self, line: str = "") -> None:
        print(line)
        self._lines.append(line)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._lines) + "\n")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _binox_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    """Binoculars score columns shared by both frames, sorted."""
    shared = set(df_a.columns) & set(df_b.columns)
    return sorted(c for c in shared if c.startswith("binoculars_"))


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerce the given columns to floats; non-numeric / null become NaN."""
    return df[cols].apply(pd.to_numeric, errors="coerce")


class PairResult:
    """Everything computed for one dataset's binox0-vs-binox1 comparison."""

    def __init__(self, dataset: str, cols: list[str]) -> None:
        self.dataset = dataset
        self.cols = cols
        self.n_pairs = 0  # matched pairs (key present in both files)
        self.n_only0 = 0  # keys present only in binox0
        self.n_only1 = 0  # keys present only in binox1
        self.distances = np.empty(0)  # Euclidean / |a-b| — magnitude only
        self.signed_diffs = np.empty(0)  # binox0 - binox1 (mean over cols) — keeps sign
        self.mse = float("nan")
        # skipped pairs: list of dicts with the key and the raw values
        self.skipped: list[dict] = []

    @property
    def n_valid(self) -> int:
        return int(self.distances.size)

    @property
    def n_skipped(self) -> int:
        return len(self.skipped)


def compute_pairwise(dataset: str, parquet_dir: Path, out: Tee) -> PairResult:
    """Load the paired parquets for ``dataset`` and compute pairwise distances."""
    path0 = parquet_dir / f"{dataset}_binox0.parquet"
    path1 = parquet_dir / f"{dataset}_binox1.parquet"
    for p in (path0, path1):
        if not p.exists():
            raise SystemExit(f"Missing parquet: {p}")

    df0 = pd.read_parquet(path0)
    df1 = pd.read_parquet(path1)

    cols = _binox_columns(df0, df1)
    if not cols:
        raise SystemExit(f"[{dataset}] no shared 'binoculars_*' columns to compare.")

    keys = [k for k in KEY_COLS if k in df0.columns and k in df1.columns]
    if len(keys) != len(KEY_COLS):
        raise SystemExit(
            f"[{dataset}] key columns {KEY_COLS} missing from one of the files "
            f"(found shared: {keys})."
        )

    # The composite key must be unique within each file to pair unambiguously.
    for name, df in ((path0.name, df0), (path1.name, df1)):
        dup = df.duplicated(subset=keys).sum()
        if dup:
            raise SystemExit(
                f"[{dataset}] key {keys} is not unique in {name} ({dup} duplicates) "
                "— cannot pair unambiguously."
            )

    # Coerce scores up front so null / non-numeric become NaN, then pair by merge.
    left = pd.concat([df0[keys], _to_numeric(df0, cols).add_suffix("_0")], axis=1)
    right = pd.concat([df1[keys], _to_numeric(df1, cols).add_suffix("_1")], axis=1)
    merged = left.merge(right, on=keys, how="outer", indicator=True)

    res = PairResult(dataset, cols)
    res.n_only0 = int((merged["_merge"] == "left_only").sum())
    res.n_only1 = int((merged["_merge"] == "right_only").sum())
    paired = merged[merged["_merge"] == "both"].reset_index(drop=True)
    res.n_pairs = len(paired)

    a = paired[[f"{c}_0" for c in cols]].to_numpy(float)  # (n, k)
    b = paired[[f"{c}_1" for c in cols]].to_numpy(float)

    # A pair is valid only if every element (both files, all cols) is a real number.
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    for i in np.nonzero(~valid)[0]:
        res.skipped.append(
            {
                "key": {k: paired.loc[i, k] for k in keys},
                "values_binox0": a[i].tolist(),
                "values_binox1": b[i].tolist(),
            }
        )

    diff = a[valid] - b[valid]  # (n_valid, k)
    res.distances = np.sqrt((diff**2).sum(axis=1))  # Euclidean; |a-b| for k=1
    res.signed_diffs = diff.mean(axis=1)  # signed; == binox0 - binox1 for k=1
    res.mse = float((diff**2).mean()) if diff.size else float("nan")
    return res


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _violin_figure(
    usable: list[PairResult],
    data: list[np.ndarray],
    *,
    ylabel: str,
    title: str,
    annotate,
    zero_line: bool,
    fname: str,
    fig_dir: Path,
) -> None:
    """Draw one violin per dataset and save it.

    ``annotate(r)`` returns the short text placed above each violin.
    """
    positions = list(range(1, len(usable) + 1))
    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(usable) + 2), 6))
    parts = ax.violinplot(
        data, positions=positions, showmeans=True, showextrema=True, widths=0.8
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_edgecolor("black")
        pc.set_alpha(0.6)
    if "cmeans" in parts:
        parts["cmeans"].set_color("#d1495b")

    if zero_line:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    labels = []
    for x, d, r in zip(positions, data, usable):
        top = float(d.max())
        pad = 0.01 * (abs(top) or 1) + 0.005
        ax.text(
            x, top + pad, annotate(r),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
        labels.append(
            f"{r.dataset}\nn={r.n_valid}"
            + (f"  (skipped {r.n_skipped})" if r.n_skipped else "")
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.margins(y=0.12)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  [OK] {fname}")


def plot_violins(results: list[PairResult], fig_dir: Path) -> None:
    usable = [r for r in results if r.n_valid > 0]
    if not usable:
        print("  [skip] no valid pairs in any dataset — nothing to plot")
        return

    # Figure 1 — magnitude of disagreement, annotated with MSE.
    _violin_figure(
        usable,
        [r.distances for r in usable],
        ylabel="Pairwise distance  |binox0 − binox1|",
        title="Binoculars re-scoring stability: pairwise distance (magnitude)",
        annotate=lambda r: f"MSE={r.mse:.4g}",
        zero_line=False,
        fname="pairwise_distance_violin.png",
        fig_dir=fig_dir,
    )

    # Figure 2 — signed difference, to see which scoring is larger.
    _violin_figure(
        usable,
        [r.signed_diffs for r in usable],
        ylabel="Signed difference  (binox0 − binox1)",
        title="Binoculars re-scoring: signed difference  (>0 ⇒ binox0 larger)",
        annotate=lambda r: f"mean={r.signed_diffs.mean():+.4g}",
        zero_line=True,
        fname="pairwise_signed_diff_violin.png",
        fig_dir=fig_dir,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _dist_stats(d: np.ndarray) -> dict:
    if d.size == 0:
        return {k: float("nan") for k in ("mean", "std", "min", "median", "max")}
    return {
        "mean": float(d.mean()),
        "std": float(d.std()),
        "min": float(d.min()),
        "median": float(np.median(d)),
        "max": float(d.max()),
    }


def report(results: list[PairResult], fig_dir: Path, out: Tee) -> None:
    out.print("=" * 70)
    out.print("Binoculars pairwise comparison (binox0 vs binox1)")
    out.print(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    out.print("=" * 70)

    rows = []
    for r in results:
        s = _dist_stats(r.distances)
        out.print(f"\n[{r.dataset}]  columns: {r.cols}")
        out.print(
            f"  matched pairs={r.n_pairs}  valid={r.n_valid}  skipped={r.n_skipped}"
            f"  (unmatched: only in binox0={r.n_only0}, only in binox1={r.n_only1})"
        )
        out.print(
            f"  distance  mean={s['mean']:.4g}  std={s['std']:.4g}  "
            f"min={s['min']:.4g}  median={s['median']:.4g}  max={s['max']:.4g}"
        )
        mean_signed = float(r.signed_diffs.mean()) if r.n_valid else float("nan")
        larger = (
            "binox0 larger" if mean_signed > 0
            else "binox1 larger" if mean_signed < 0
            else "equal"
        )
        out.print(f"  mean signed diff (binox0 - binox1)={mean_signed:+.4g}  → {larger}")
        out.print(f"  MSE={r.mse:.6g}")
        rows.append(
            {
                "dataset": r.dataset,
                "columns": ",".join(r.cols),
                "n_pairs": r.n_pairs,
                "n_valid": r.n_valid,
                "n_skipped": r.n_skipped,
                "n_only0": r.n_only0,
                "n_only1": r.n_only1,
                "dist_mean": s["mean"],
                "dist_std": s["std"],
                "dist_min": s["min"],
                "dist_median": s["median"],
                "dist_max": s["max"],
                "mean_signed_diff": mean_signed,
                "mse": r.mse,
            }
        )

    # --- skipped-pair detail ---
    out.print("\n" + "-" * 70)
    out.print("Skipped pairs (element null / non-numeric in binox0 or binox1):")
    total_skipped = sum(r.n_skipped for r in results)
    if total_skipped == 0:
        out.print("  (none — every pair had numeric scores in both files)")
    else:
        for r in results:
            if not r.skipped:
                continue
            out.print(f"\n  [{r.dataset}] {r.n_skipped} skipped:")
            for sk in r.skipped:
                out.print(
                    f"    key={sk['key']}  "
                    f"binox0={sk['values_binox0']}  binox1={sk['values_binox1']}"
                )
        out.print(f"\n  TOTAL skipped across datasets: {total_skipped}")

    out.print("\n" + "=" * 70)

    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_path = fig_dir / "pairwise_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  [OK] {csv_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help=f"Dataset stems to compare (default: {DEFAULT_DATASETS}). "
        "Each needs <name>_binox0.parquet and <name>_binox1.parquet.",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=PARQUET_DIR,
        help=f"Directory holding the parquet files (default: {PARQUET_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = args.datasets or DEFAULT_DATASETS

    out = Tee()
    results: list[PairResult] = []
    for ds in datasets:
        print(f"Computing pairwise distances for '{ds}' ...")
        results.append(compute_pairwise(ds, args.parquet_dir, out))

    fig_dir = RESULTS_DIR / "__".join(datasets)
    print()
    plot_violins(results, fig_dir)
    report(results, fig_dir, out)

    out.save(fig_dir / "pairwise_summary.log")
    print(f"\nAll outputs saved to {fig_dir}")


if __name__ == "__main__":
    main()
