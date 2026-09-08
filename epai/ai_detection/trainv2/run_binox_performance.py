r"""AUROC vs. TPR@1%FPR across datasets, one fixed threshold per Binoculars pair.

    uv run python epai/ai_detection/trainv2/run_binox_performance.py \
        data/parquet/scribe_sci.parquet \
        data/parquet/scribe_denario.parquet \
        --cfg cfg/ai_detection/trainv2/binox_baseline_calibrated.yaml

Reads each pair's threshold from a ``binoculars_baseline`` config block -- the
SAME shape ``cd_core``'s zero-shot baselines read, produced by
``calibrate_binox_baseline.py`` -- and applies it UNCHANGED to every dataset
given. AUROC needs no threshold and is read straight off each dataset's own
ROC; TPR and the realized FPR are read at the config's fixed cutoff. This is
the deployable / transported reading, not one refit per dataset: because the
threshold does not move, a dataset where TPR is low despite a high AUROC is
telling you the pair's calibrated cutoff did not survive the move to that
corpus.

One point per (pair, dataset), pooled over every domain in that dataset.
Colour = pair (fixed hue, one per pair); shade = dataset (lighter earlier ->
darker later, in the order the datasets are given).

This is the Binoculars sibling of ``run_judge_performance.py``.

Output:
    <csv>   one row per (pair, dataset): auroc, threshold, tpr, fpr, n
    <fig>   the AUROC vs TPR@<fpr-target>%FPR scatter

Run:
    uv run python epai/ai_detection/trainv2/run_binox_performance.py \
        data/parquet/scribe_sci.parquet data/parquet/scribe_denario.parquet \
        --cfg cfg/ai_detection/trainv2/binox_baseline_calibrated.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from epai.utils.cd_core import (
    BASELINE_KINDS,
    auroc_score,
    baseline_entries,
    oriented_scores,
    rates_at_threshold,
)

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CFG = BASE_DIR / "cfg" / "ai_detection" / "trainv2" / "binox_baseline_calibrated.yaml"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "binox_performance"
GEN_COL = "generated_by"
BINOX_PREFIX = "binoculars_score"

_BINOX_KIND = next(k for k in BASELINE_KINDS if k.key == "binoculars")

# One base hue per pair -- extend as new pairs are added; anything unlisted
# falls back to grey. Keyed by the pair label calibrate_binox_baseline.py
# writes (dashes, e.g. 'qwen25-7b'), not the column name.
_BASE_HUES: dict[str, tuple[int, int, int]] = {
    "falcon-7b": (7, 107, 82),
    "falcon3-7b": (217, 119, 87),
    "qwen25-7b": (106, 81, 163),
    "mistral-v03": (43, 130, 191),
    "llama31-8b": (191, 90, 43),
}
_FALLBACK_HUE = (110, 110, 110)

# Dataset identity is carried by MARKER SHAPE now, not colour, so any number
# of datasets stays legible without relying on a shade gradient. 'o' (circle)
# is reserved for a pair's mean across datasets, so it never appears here.
_DATASET_MARKERS = ["s", "^", "D", "v", "P", "*", "h", "X"]


def _hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _light(rgb: tuple[int, int, int]) -> str:
    """A pale tint of ``rgb``, for the per-dataset markers -- shape already
    carries which dataset a point is, so the colour only needs to whisper
    which pair it belongs to, staying out of the way of that pair's own
    solid, full-colour mean marker."""
    t = 0.35
    return _hex(tuple(int(255 - (255 - c) * t) for c in rgb))


def _pair_key(column: str) -> str:
    """``binoculars_score`` -> ``falcon-7b``; ``binoculars_score_qwen25_7b`` ->
    ``qwen25-7b`` -- matches ``calibrate_binox_baseline.py``'s ``_pair_label``."""
    if column == BINOX_PREFIX:
        return "falcon-7b"
    return column.removeprefix(f"{BINOX_PREFIX}_").replace("_", "-")


def load_thresholds(cfg_path: Path) -> list[dict]:
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    entries = baseline_entries(cfg, _BINOX_KIND)
    if not entries:
        raise SystemExit(f"{cfg_path}: no usable binoculars_baseline entries")
    return entries


def evaluate(df: pd.DataFrame, column: str, threshold: float, human_label: str) -> dict | None:
    if column not in df.columns:
        return None
    sub = df[[column, GEN_COL]].dropna(subset=[column])
    if not len(sub):
        return None
    raw = sub[column].to_numpy(dtype=float)
    y = (sub[GEN_COL].astype(str) != human_label).to_numpy().astype(int)
    if np.unique(y).size < 2 or len(y) < 10:
        return None

    oriented = oriented_scores(raw, _BINOX_KIND.machine_is_high)
    # Invert the config's raw threshold into the oriented space rates_at_threshold
    # expects -- the same relationship calibrate_binox_baseline.py used going
    # the other way (oriented = -raw when machine_is_high is False).
    thr_or = -threshold if not _BINOX_KIND.machine_is_high else threshold
    tpr, fpr = rates_at_threshold(y, oriented, thr_or)
    return {"n": int(len(y)), "auroc": auroc_score(y, oriented), "tpr": tpr, "fpr": fpr}


def plot(rows: list[dict], fpr_target: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    datasets: list[str] = []
    for r in rows:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])
    pairs = sorted({r["pair"] for r in rows})
    marker_of = {d: _DATASET_MARKERS[i % len(_DATASET_MARKERS)] for i, d in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    # Chance reference: an uninformative score (AUROC 0.5) has TPR = FPR at
    # any threshold, so a threshold calibrated to fpr_target gives TPR =
    # fpr_target too -- a single POINT, not a diagonal, since the x-axis here
    # is AUROC (one number per model), not FPR.
    ax.axvline(0.5, color="#b5495b", lw=1.0, ls=":", zorder=0, alpha=0.6)
    ax.axhline(fpr_target, color="#b5495b", lw=1.0, ls=":", zorder=0, alpha=0.6)
    ax.text(0.17, fpr_target, f"{fpr_target:g}", color="#b5495b", fontsize=8,
            va="bottom", ha="left", zorder=0)
    ax.scatter([0.5], [fpr_target], s=130, color="#b5495b",
               lw=1.8, marker="x", zorder=0, alpha=0.6)
    # Per-dataset points: shape = dataset, light tint of the pair's hue.
    for r in rows:
        hue = _BASE_HUES.get(r["pair"], _FALLBACK_HUE)
        ax.scatter([r["auroc"]], [r["tpr"]], s=70, color=_light(hue),
                    edgecolors="#666666", lw=0.6, marker=marker_of[r["dataset"]],
                    zorder=3)

    # Pair mean: solid circle, full hue, on top of everything -- the answer
    # the light, shape-coded points are supporting evidence for.
    for pair in pairs:
        p_rows = [r for r in rows if r["pair"] == pair]
        mean_auroc = sum(r["auroc"] for r in p_rows) / len(p_rows)
        mean_tpr = sum(r["tpr"] for r in p_rows) / len(p_rows)
        ax.scatter([mean_auroc], [mean_tpr], s=150,
                    color=_hex(_BASE_HUES.get(pair, _FALLBACK_HUE)),
                    edgecolors="#222222", lw=0.9, marker="o", zorder=5)

    ax.set_xlabel("AUROC (machine = positive)")
    ax.set_ylabel(f"TPR at the calibrated {fpr_target * 100:g}%-FPR threshold")
    ax.set_xlim(0.15, 1.02)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.set_title(f"Binoculars pairs: AUROC vs. deployable TPR@{fpr_target * 100:g}%FPR "
                 f"(fixed, calibrated cutoff)")

    handles = [Line2D([], [], marker="o", ls="", color=_hex(_BASE_HUES.get(p, _FALLBACK_HUE)),
                       label=f"{p} mean") for p in pairs]
    handles += [Line2D([], [], marker=marker_of[d], ls="", color="#888888",
                        label=d) for d in datasets]
    handles += [Line2D([], [], marker="x", ls="", color="#b5495b",
                        label=f"chance (0.5, {fpr_target * 100:g}%)")]
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.9,
              title="colour = pair, shape = dataset, o = mean")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("datasets", nargs="+", type=Path)
    ap.add_argument("--cfg", type=Path, default=DEFAULT_CFG,
                     help="binoculars_baseline YAML (see calibrate_binox_baseline.py)")
    ap.add_argument("--fpr-target", type=float, default=0.01,
                     help="for the axis/title label only -- the threshold itself "
                          "comes from --cfg, not recomputed here")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--fig", type=Path, default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    entries = load_thresholds(args.cfg)
    csv_path = args.csv or DEFAULT_OUT_DIR / "binox_performance.csv"
    fig_path = args.fig or DEFAULT_OUT_DIR / "binox_performance.png"

    print(f"cfg      : {args.cfg}  ({len(entries)} pair(s))")
    print(f"datasets : {[d.name for d in args.datasets]}\n")
    print(f"{'dataset':<28}{'pair':<14}{'n':>7}{'auroc':>8}{'thr':>10}{'tpr':>8}{'fpr':>8}")

    rows: list[dict] = []
    for ds_path in args.datasets:
        df = pd.read_parquet(ds_path)
        for e in entries:
            r = evaluate(df, e["column"], e["threshold"], args.human_label)
            pair = _pair_key(e["column"])
            if r is None:
                print(f"{ds_path.stem:<28}{pair:<14}  -- skipped: column missing, "
                      f"single class, or n<10 --")
                continue
            row = {"dataset": ds_path.stem, "pair": pair, "column": e["column"],
                   "threshold": e["threshold"], **r}
            rows.append(row)
            print(f"{ds_path.stem:<28}{pair:<14}{row['n']:>7}{row['auroc']:>8.3f}"
                  f"{row['threshold']:>10.3g}{row['tpr']:>8.3f}{row['fpr']:>8.3f}")

    if not rows:
        raise SystemExit("\nnothing evaluated")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")

    if not args.no_plot:
        plot(rows, args.fpr_target, fig_path)


if __name__ == "__main__":
    main()
