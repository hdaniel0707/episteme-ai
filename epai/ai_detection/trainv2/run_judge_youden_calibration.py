r"""Calibrate LLM judges: per-domain and pooled Youden's J thresholds.

    uv run python epai/ai_detection/trainv2/run_judge_youden_calibration.py \
        data/parquet/manugram_sci.parquet

For every ``llm_judge_score_*`` column (P(human) -- a LOW score means machine,
the judge's own convention; see ``epai.utils.cd_core.DEFAULT_JUDGE_COL``), per
domain and pooled (:data:`~epai.ai_detection.trainv2.core.POOLED`), this:

  * fits the Youden's J threshold (max TPR - FPR) on that group's own rows --
    :func:`epai.utils.cd_core.youden_threshold`, the same function every
    trainv2 / ``run_clf_*`` accuracy and macro-F1 number is calibrated
    against, so a threshold calibrated here is the same kind of number a
    training run would have fit on a calibration slice;
  * reports AUROC, the threshold in the judge's own P(human) units, and the
    TPR / FPR that threshold actually realizes there;
  * plots the human vs. machine score distribution for that group, with the
    threshold marked;
  * writes one row per (judge, group) to a CSV, meant to be READ by a later
    judge-performance run so the operating point is fit once, here, rather
    than re-fit -- and drifting -- inside every experiment that uses it.

CALIBRATION, NOT EVALUATION. The threshold for a group is fit and reported on
the SAME rows plotted -- there is no held-out split. That is the point: this
produces one fixed operating point per (judge, domain) meant to be applied
unchanged to other data, not a claim about how well it would generalise. A run
that needs the transport-honest reading instead -- threshold fit on a
calibration slice, scored on disjoint rows -- is
``epai/ai_detection/analyse/analyse_judge_operating_point.py``.

Output:
    <csv>                     one row per (judge column, group)
    <fig-dir>/<judge>.png     one figure per judge: a panel per domain, plus
                              'all' pooled, each with its own cut marked

Run:
    uv run python epai/ai_detection/trainv2/run_judge_youden_calibration.py data/parquet/manugram_sci.parquet
    uv run python epai/ai_detection/trainv2/run_judge_youden_calibration.py data/parquet/ghostbuster_multi_sci3.parquet \
        --cols llm_judge_score_gpt56luna_v3 llm_judge_score_qwen3_14b_v3
    uv run python epai/ai_detection/trainv2/run_judge_youden_calibration.py data/parquet/manugram_den.parquet --no-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.cd_core import auroc_score, oriented_scores, rates_at_threshold, youden_threshold

from epai.ai_detection.trainv2.core import POOLED

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "judge_calibration"

JUDGE_PREFIX = "llm_judge_score_"
GEN_COL = "generated_by"

# The judge writes P(human): a LOW score means machine, same convention as
# Binoculars and every other score cd_core.oriented_scores knows about. Fixed
# here rather than a flag -- this script is judge-only by name, and a judge
# reading the other way would be a different kind of column, not an option.
MACHINE_IS_HIGH = False

_HUMAN_COLOR = "#888888"
_MACHINE_COLOR = "#076b52"
_THRESHOLD_COLOR = "#e05353"


def judge_columns(df: pd.DataFrame, cols: list[str] | None) -> list[str]:
    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"not in this parquet: {missing}")
        return cols
    found = sorted(c for c in df.columns if c.startswith(JUDGE_PREFIX))
    if not found:
        raise SystemExit(f"no {JUDGE_PREFIX}* column in this parquet")
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
    # oriented = -raw here (MACHINE_IS_HIGH is False), so the oriented rule
    # "oriented >= thr_or => machine" is "raw <= -thr_or => machine" -- the
    # threshold a caller applying the RAW P(human) column actually wants.
    thr_raw = -thr_or if not MACHINE_IS_HIGH else thr_or
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


def plot_judge(column: str, rows: list[dict], path: Path, bins: int) -> None:
    """One figure per judge: a panel per group, human vs. machine, cut marked."""
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
    edges = np.linspace(0.0, 1.0, bins + 1)

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
        ax.set_xlabel("P(human)")
        ax.set_xlim(0, 1)
        ax.grid(True, lw=0.4, alpha=0.4)
        ax.text(
            r["threshold"], ax.get_ylim()[1] * 0.98,
            f" thr={r['threshold']:.2f}", color=_THRESHOLD_COLOR,
            fontsize=8, va="top", ha="left" if r["threshold"] < 0.85 else "right",
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
    ap.add_argument("dataset", type=Path, help="a parquet with human + judge rows")
    ap.add_argument("--cols", nargs="+", default=None,
                    help="judge score columns (default: every llm_judge_score_* column)")
    ap.add_argument("--by", default=None,
                    help="grouping column (default: 'domain' if present). "
                         "'all' (pooled) is always reported too.")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--csv", type=Path, default=None,
                    help="where to write the thresholds (default: "
                         f"{DEFAULT_OUT_DIR}/<dataset stem>_thresholds.csv)")
    ap.add_argument("--fig-dir", type=Path, default=None,
                    help="where to write one PNG per judge (default: "
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
    cols = judge_columns(df, args.cols)
    by = args.by or ("domain" if "domain" in df.columns else None)
    groups = groups_of(df, by)

    print(f"dataset  : {args.dataset}  ({len(df)} rows)")
    print(f"judges   : {cols}")
    print(f"grouped by: {by or '-'}  ({len(groups)} group(s) incl. {POOLED!r})\n")

    rows_by_col: dict[str, list[dict]] = {}
    all_rows: list[dict] = []
    print(f"{'group':<16} {'judge':<28} {'n':>6} {'auroc':>7} "
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
                  f"{r['threshold']:>10.3f} {r['tpr']:>7.3f} {r['fpr']:>7.3f} "
                  f"{r['youden_j']:>7.3f}")
            rows_by_col.setdefault(col, []).append(r)
            all_rows.append(r)

    if not all_rows:
        raise SystemExit("\nnothing calibrated -- every (judge, group) was skipped")

    if not args.no_plots:
        for col, rows in rows_by_col.items():
            plot_judge(col, rows, fig_dir / f"{col}.png", args.bins)

    csv_rows = [
        {k: v for k, v in r.items() if not k.startswith("_")} for r in all_rows
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
