r"""Calibrate Fast-DetectGPT pairs: per-domain and pooled Youden's J thresholds.

    uv run python epai/ai_detection/trainv2/run_fastdetect_youden_calibration.py \
        data/parquet/manugram_sci.parquet

For every ``fastdetect_crit_*`` column (a HIGH score means machine -- the
opposite direction from Binoculars and the LLM judge; see
``epai.utils.cd_core.DEFAULT_FASTDETECT_COL`` / the ``fastdetect``
``BaselineKind``), per domain and pooled
(:data:`~epai.ai_detection.trainv2.core.POOLED`), this:

  * fits the Youden's J threshold (max TPR - FPR) on that group's own rows --
    :func:`epai.utils.cd_core.youden_threshold`, the same function every
    trainv2 / ``run_clf_*`` accuracy and macro-F1 number is calibrated
    against, so a threshold calibrated here is the same kind of number a
    training run would have fit on a calibration slice;
  * reports AUROC, the threshold in the pair's own raw (criterion) units, and
    the TPR / FPR that threshold actually realizes there;
  * plots the human vs. machine score distribution for that group, with the
    threshold marked;
  * writes one row per (pair, group) to a CSV, meant to be READ by a later
    Fast-DetectGPT-performance run so the operating point is fit once, here,
    rather than re-fit -- and drifting -- inside every experiment that uses it.

Only ``_crit_`` columns are read, never ``_prob_``/``_ntok_``: the prob
columns are Null for every pair upstream did not calibrate and non-monotone
where they exist (see ``cfg/detector_suite.yaml``), so the criterion is the
only column that behaves like a score everywhere.

CALIBRATION, NOT EVALUATION. The threshold for a group is fit and reported on
the SAME rows plotted -- there is no held-out split. This is the
Fast-DetectGPT sibling of ``run_judge_youden_calibration.py`` /
``run_binox_youden_calibration.py``; the same calibration-vs-transport
distinction applies.

Output:
    <csv>                    one row per (pair column, group)
    <fig-dir>/<pair>.png     one figure per pair: a panel per domain, plus
                             'all' pooled, each with its own cut marked

Run:
    uv run python epai/ai_detection/trainv2/run_fastdetect_youden_calibration.py data/parquet/manugram_sci.parquet
    uv run python epai/ai_detection/trainv2/run_fastdetect_youden_calibration.py data/parquet/manugram_den.parquet --no-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.cd_core import auroc_score, oriented_scores, rates_at_threshold, youden_threshold

from epai.ai_detection.trainv2.core import POOLED

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "fastdetect_calibration"

FASTDETECT_PREFIX = "fastdetect_crit_"
GEN_COL = "generated_by"

# Fast-DetectGPT: a HIGH criterion means machine -- the opposite of Binoculars
# and the LLM judge over the same model pairs. Fixed here, not a flag -- this
# script is Fast-DetectGPT-only, and matches cd_core's own fastdetect
# BaselineKind.
MACHINE_IS_HIGH = True

_HUMAN_COLOR = "#888888"
_MACHINE_COLOR = "#076b52"
_THRESHOLD_COLOR = "#e05353"


def fastdetect_columns(df: pd.DataFrame, cols: list[str] | None) -> list[str]:
    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"not in this parquet: {missing}")
        return cols
    found = sorted(c for c in df.columns if c.startswith(FASTDETECT_PREFIX))
    if not found:
        raise SystemExit(f"no {FASTDETECT_PREFIX}* column in this parquet")
    return found


def groups_of(df: pd.DataFrame, by: str | None) -> list[tuple[str, pd.DataFrame]]:
    """``[(POOLED, df), (domain, slice), ...]`` -- pooled always first."""
    groups = [(POOLED, df)]
    if by and by in df.columns:
        groups += [(str(g), gdf) for g, gdf in df.groupby(by)]
    return groups


def calibrate(raw: np.ndarray, y: np.ndarray) -> dict | None:
    """Youden's J threshold and what it realizes, on these rows as given.

    Returns ``None`` on a single-class or too-small group -- there is no ROC
    to fit a threshold from, and reporting one as -inf/nan would read as a
    number rather than as "not enough data here".
    """
    if np.unique(y).size < 2 or len(y) < 10:
        return None
    oriented = oriented_scores(raw, MACHINE_IS_HIGH)
    thr_or = youden_threshold(y, oriented)
    tpr, fpr = rates_at_threshold(y, oriented, thr_or)
    # oriented = raw here (MACHINE_IS_HIGH is True), so the oriented threshold
    # IS the raw threshold a caller applying the raw criterion column wants.
    thr_raw = thr_or if MACHINE_IS_HIGH else -thr_or
    return {
        "n": int(len(y)),
        "n_human": int((y == 0).sum()),
        "n_machine": int((y == 1).sum()),
        "auroc": auroc_score(y, oriented),
        "threshold": float(thr_raw),
        "tpr": tpr,
        "fpr": fpr,
        "youden_j": tpr - fpr,
        "_raw": raw,
        "_y": y,
    }


def plot_pair(column: str, rows: list[dict], path: Path, bins: int) -> None:
    """One figure per pair: a panel per group, human vs. machine, cut marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pooled first, then domains alphabetically -- matches groups_of's order,
    # so the leftmost panel is always the one every other panel is a slice of.
    ordered = [r for r in rows if r["group"] == POOLED] + sorted(
        (r for r in rows if r["group"] != POOLED), key=lambda r: r["group"]
    )
    n = len(ordered)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
    axes = axes[0]
    all_raw = np.concatenate([r["_raw"] for r in ordered])
    lo, hi = float(np.min(all_raw)), float(np.max(all_raw))
    edges = np.linspace(lo, hi, bins + 1)

    for ax, r in zip(axes, ordered):
        raw, y = r["_raw"], r["_y"]
        h = np.histogram(raw[y == 0], bins=edges, density=True)[0]
        m = np.histogram(raw[y == 1], bins=edges, density=True)[0]
        mids = (edges[:-1] + edges[1:]) / 2
        ax.step(mids, h, where="mid", color=_HUMAN_COLOR, lw=1.6, label="human")
        ax.fill_between(mids, h, step="mid", color=_HUMAN_COLOR, alpha=0.15)
        ax.step(mids, m, where="mid", color=_MACHINE_COLOR, lw=1.6, label="machine")
        ax.fill_between(mids, m, step="mid", color=_MACHINE_COLOR, alpha=0.15)
        ax.axvline(r["threshold"], color=_THRESHOLD_COLOR, lw=1.6, ls="--")
        ax.set_title(
            f"{r['group']}  (n={r['n']})\n"
            f"AUROC {r['auroc']:.3f}  J {r['youden_j']:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("Fast-DetectGPT criterion")
        ax.set_xlim(lo, hi)
        ax.grid(True, lw=0.4, alpha=0.4)
        ax.text(
            r["threshold"], ax.get_ylim()[1] * 0.98,
            f" thr={r['threshold']:.3g}", color=_THRESHOLD_COLOR,
            fontsize=8, va="top",
            ha="left" if r["threshold"] < lo + 0.85 * (hi - lo) else "right",
        )
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{column} -- Youden's J calibration by domain", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dataset", type=Path, help="a parquet with human + Fast-DetectGPT rows")
    ap.add_argument("--cols", nargs="+", default=None,
                    help="Fast-DetectGPT columns (default: every fastdetect_crit_* column)")
    ap.add_argument("--by", default=None,
                    help="grouping column (default: 'domain' if present). "
                         "'all' (pooled) is always reported too.")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--csv", type=Path, default=None,
                    help="where to write the thresholds (default: "
                         f"{DEFAULT_OUT_DIR}/<dataset stem>_thresholds.csv)")
    ap.add_argument("--fig-dir", type=Path, default=None,
                    help="where to write one PNG per pair (default: "
                         f"{DEFAULT_OUT_DIR}/fig/<dataset stem>/)")
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    stem = args.dataset.stem
    csv_path = args.csv or DEFAULT_OUT_DIR / f"{stem}_thresholds.csv"
    fig_dir = args.fig_dir or DEFAULT_OUT_DIR / "fig" / stem

    df = pd.read_parquet(args.dataset)
    if GEN_COL not in df.columns:
        raise SystemExit(f"{args.dataset}: no {GEN_COL!r} column")
    cols = fastdetect_columns(df, args.cols)
    by = args.by or ("domain" if "domain" in df.columns else None)
    groups = groups_of(df, by)

    print(f"dataset  : {args.dataset}  ({len(df)} rows)")
    print(f"pairs    : {cols}")
    print(f"grouped by: {by or '-'}  ({len(groups)} group(s) incl. {POOLED!r})\n")

    rows_by_col: dict[str, list[dict]] = {}
    all_rows: list[dict] = []
    print(f"{'group':<16} {'pair':<28} {'n':>6} {'auroc':>7} "
          f"{'threshold':>10} {'tpr':>7} {'fpr':>7} {'J':>7}")
    for col in cols:
        for gname, gdf in groups:
            sub = gdf[[col, GEN_COL]].dropna(subset=[col])
            raw = sub[col].to_numpy(dtype=float)
            y = (sub[GEN_COL].astype(str) != args.human_label).to_numpy().astype(int)
            r = calibrate(raw, y)
            if r is None:
                print(f"{gname:<16} {col:<28} {'-- skipped: single class or n<10 --':>44}")
                continue
            r["column"] = col
            r["group"] = gname
            print(f"{gname:<16} {col:<28} {r['n']:>6} {r['auroc']:>7.3f} "
                  f"{r['threshold']:>10.3g} {r['tpr']:>7.3f} {r['fpr']:>7.3f} "
                  f"{r['youden_j']:>7.3f}")
            rows_by_col.setdefault(col, []).append(r)
            all_rows.append(r)

    if not all_rows:
        raise SystemExit("\nnothing calibrated -- every (pair, group) was skipped")

    if not args.no_plots:
        for col, rows in rows_by_col.items():
            plot_pair(col, rows, fig_dir / f"{col}.png", args.bins)

    csv_rows = [
        {k: v for k, v in r.items() if not k.startswith("_")} for r in all_rows
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
