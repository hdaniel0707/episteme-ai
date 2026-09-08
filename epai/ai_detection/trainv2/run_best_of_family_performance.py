r"""Best-of-family performance: the single strongest judge, Binoculars pair,
Fast-DetectGPT pair, and the pretrained detector, on one AUROC-vs-TPR@1%FPR
figure.

    uv run python epai/ai_detection/trainv2/run_best_of_family_performance.py \
        data/parquet/scribe_sci.parquet data/parquet/scribe_denario.parquet

Four families, one point per family per dataset:
    judge         best of the llm_judge_baseline columns
    binoculars    best of the binoculars_baseline columns
    fastdetect    best of the fastdetect_baseline columns
    detector      the pretrained detector -- currently just radar

"BEST" IS DECIDED ONCE, NOT PER DATASET. Whichever column has the highest
AUROC on ``--calibration-dataset`` (default: ``scribe_sci.parquet``,
the same corpus every ``*_baseline_calibrated.yaml`` was fit on) wins that
family -- for every dataset in this run, not re-picked per dataset. Picking
the winner separately on each evaluation corpus would be exactly the
on-the-test-set selection the calibrate-once approach exists to avoid: the
"best" judge on denario might not be the "best" judge on science, and
reporting whichever one wins locally on each corpus would make every family
look artificially strong everywhere.

The winning column is then evaluated on every dataset using its
ALREADY-CALIBRATED threshold from that family's config (the transported
reading; ``run_zeroshot_performance.py`` is the oracle counterpart, threshold
refit per split) -- so a dataset where TPR drops despite a high reference
AUROC is telling you the winning column's calibration did not transport there.

Needs all four calibrated configs to already exist:
    cfg/ai_detection/trainv2/judge_baseline_calibrated.yaml
    cfg/ai_detection/trainv2/binox_baseline_calibrated.yaml
    cfg/ai_detection/trainv2/fastdetect_baseline_calibrated.yaml
    cfg/ai_detection/trainv2/detector_baseline_calibrated.yaml
(see calibrate_judge_baseline.py / calibrate_binox_baseline.py /
calibrate_fastdetect_baseline.py / calibrate_detector_baseline.py.)

Colour = family (fixed hue, one per family); shade = dataset (lighter earlier
-> darker later, in the order the datasets are given).

Output:
    <csv>   one row per (family, dataset): winning column, auroc, threshold, tpr, fpr, n
    <fig>   the AUROC vs TPR@<fpr-target>%FPR scatter

Run:
    uv run python epai/ai_detection/trainv2/run_best_of_family_performance.py \
        data/parquet/scribe_sci.parquet data/parquet/scribe_denario.parquet
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
CFG_DIR = BASE_DIR / "cfg" / "ai_detection" / "trainv2"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "best_of_family_performance"
DEFAULT_CALIBRATION_DATASET = BASE_DIR / "data" / "parquet" / "scribe_sci.parquet"
GEN_COL = "generated_by"

_KIND_BY_KEY = {k.key: k for k in BASELINE_KINDS}

# family label -> (BaselineKind key, default config path)
FAMILIES: dict[str, tuple[str, Path]] = {
    "judge": ("llm_judge", CFG_DIR / "judge_baseline_calibrated.yaml"),
    "binoculars": ("binoculars", CFG_DIR / "binox_baseline_calibrated.yaml"),
    "fastdetect": ("fastdetect", CFG_DIR / "fastdetect_baseline_calibrated.yaml"),
    "detector": ("detector", CFG_DIR / "detector_baseline_calibrated.yaml"),
}

_FAMILY_HUES: dict[str, tuple[int, int, int]] = {
    "judge": (7, 107, 82),
    "binoculars": (43, 130, 191),
    "fastdetect": (106, 81, 163),
    "detector": (191, 90, 43),
}
_FALLBACK_HUE = (110, 110, 110)

# Dataset identity is carried by MARKER SHAPE now, not colour, so any number
# of datasets stays legible without relying on a shade gradient. 'o' (circle)
# is reserved for a family's mean, so it never appears here.
_DATASET_MARKERS = ["s", "^", "D", "v", "P", "*", "h", "X"]


def _hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _light(rgb: tuple[int, int, int]) -> str:
    """A pale tint of ``rgb``, for the per-dataset markers -- shape already
    carries which dataset a point is, so the colour only needs to whisper
    which family it belongs to, staying out of the way of that family's own
    solid, full-colour mean marker."""
    t = 0.35
    return _hex(tuple(int(255 - (255 - c) * t) for c in rgb))


def _auroc_of(df: pd.DataFrame, column: str, machine_is_high: bool,
              human_label: str) -> float | None:
    if column not in df.columns:
        return None
    sub = df[[column, GEN_COL]].dropna(subset=[column])
    if len(sub) < 10:
        return None
    raw = sub[column].to_numpy(dtype=float)
    y = (sub[GEN_COL].astype(str) != human_label).to_numpy().astype(int)
    if np.unique(y).size < 2:
        return None
    return auroc_score(y, oriented_scores(raw, machine_is_high))


def pick_best(cfg_path: Path, kind_key: str, calib_df: pd.DataFrame,
              human_label: str) -> tuple[dict, float]:
    """The ``(entry, reference AUROC)`` with the highest AUROC on ``calib_df``."""
    kind = _KIND_BY_KEY[kind_key]
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    entries = baseline_entries(cfg, kind)
    if not entries:
        raise SystemExit(f"{cfg_path}: no usable {kind.cfg_key} entries")
    scored = [
        (e, a) for e in entries
        if (a := _auroc_of(calib_df, e["column"], kind.machine_is_high, human_label)) is not None
    ]
    if not scored:
        raise SystemExit(f"{cfg_path}: no column could be scored on the calibration dataset")
    return max(scored, key=lambda t: t[1])


def evaluate(df: pd.DataFrame, column: str, threshold: float, machine_is_high: bool,
             human_label: str) -> dict | None:
    if column not in df.columns:
        return None
    sub = df[[column, GEN_COL]].dropna(subset=[column])
    if not len(sub):
        return None
    raw = sub[column].to_numpy(dtype=float)
    y = (sub[GEN_COL].astype(str) != human_label).to_numpy().astype(int)
    if np.unique(y).size < 2 or len(y) < 10:
        return None
    oriented = oriented_scores(raw, machine_is_high)
    thr_or = threshold if machine_is_high else -threshold
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
    families = sorted({r["family"] for r in rows})
    label_of = {r["family"]: r["label"] for r in rows}
    marker_of = {d: _DATASET_MARKERS[i % len(_DATASET_MARKERS)] for i, d in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    # Chance reference: an uninformative score (AUROC 0.5) has TPR = FPR at
    # any threshold, so a threshold calibrated to fpr_target gives TPR =
    # fpr_target too -- a single POINT, not a diagonal, since the x-axis here
    # is AUROC (one number per model), not FPR. Kept in the background (low
    # zorder, muted alpha) so it reads as a reference, not a result.
    ax.axvline(0.5, color="#b5495b", lw=1.0, ls=":", zorder=0, alpha=0.6)
    ax.axhline(fpr_target, color="#b5495b", lw=1.0, ls=":", zorder=0, alpha=0.6)
    ax.text(0.17, fpr_target, f"{fpr_target:g}", color="#b5495b", fontsize=8,
            va="bottom", ha="left", zorder=0)
    ax.scatter([0.5], [fpr_target], s=130, color="#b5495b", lw=1.8, marker="x",
               zorder=0, alpha=0.6)
    # Per-dataset points: shape = dataset, light tint of the family's hue.
    for r in rows:
        hue = _FAMILY_HUES.get(r["family"], _FALLBACK_HUE)
        ax.scatter([r["auroc"]], [r["tpr"]], s=70, color=_light(hue),
                    edgecolors="#666666", lw=0.6, marker=marker_of[r["dataset"]],
                    zorder=3)

    # Family mean: solid circle, full family colour, on top of everything --
    # the answer the light, shape-coded points are supporting evidence for.
    for family in families:
        fam_rows = [r for r in rows if r["family"] == family]
        mean_auroc = sum(r["auroc"] for r in fam_rows) / len(fam_rows)
        mean_tpr = sum(r["tpr"] for r in fam_rows) / len(fam_rows)
        ax.scatter([mean_auroc], [mean_tpr], s=150,
                    color=_hex(_FAMILY_HUES.get(family, _FALLBACK_HUE)),
                    edgecolors="#222222", lw=0.9, marker="o", zorder=5)

    ax.set_xlabel("AUROC (machine = positive)")
    ax.set_ylabel(f"TPR at the calibrated {fpr_target * 100:g}%-FPR threshold")
    ax.set_xlim(0.15, 1.02)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.set_title("Best of each family: AUROC vs. deployable TPR@"
                 f"{fpr_target * 100:g}%FPR (fixed, calibrated cutoff)")

    handles = [Line2D([], [], marker="o", ls="", color=_hex(_FAMILY_HUES.get(f, _FALLBACK_HUE)),
                       label=f"{f} ({label_of[f]}) mean") for f in families]
    handles += [Line2D([], [], marker=marker_of[d], ls="", color="#888888",
                        label=d) for d in datasets]
    handles += [Line2D([], [], marker="x", ls="", color="#b5495b",
                        label=f"chance (0.5, {fpr_target * 100:g}%)")]
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.9,
              title="colour = family, shape = dataset, o = mean")

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
    ap.add_argument("--calibration-dataset", type=Path, default=DEFAULT_CALIBRATION_DATASET,
                     help="decides 'best' per family, by AUROC, once (default: "
                          f"{DEFAULT_CALIBRATION_DATASET.relative_to(BASE_DIR)} -- the "
                          "same corpus every *_baseline_calibrated.yaml was fit on)")
    ap.add_argument("--judge-cfg", type=Path, default=FAMILIES["judge"][1])
    ap.add_argument("--binox-cfg", type=Path, default=FAMILIES["binoculars"][1])
    ap.add_argument("--fastdetect-cfg", type=Path, default=FAMILIES["fastdetect"][1])
    ap.add_argument("--detector-cfg", type=Path, default=FAMILIES["detector"][1])
    ap.add_argument("--fpr-target", type=float, default=0.01,
                     help="for the axis/title label only -- the thresholds come from "
                          "each family's config, not recomputed here")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--fig", type=Path, default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    cfg_paths = {
        "judge": args.judge_cfg, "binoculars": args.binox_cfg,
        "fastdetect": args.fastdetect_cfg, "detector": args.detector_cfg,
    }

    calib_df = pd.read_parquet(args.calibration_dataset)
    if GEN_COL not in calib_df.columns:
        raise SystemExit(f"{args.calibration_dataset}: no {GEN_COL!r} column")

    print(f"calibration dataset: {args.calibration_dataset}  ({len(calib_df)} rows)\n")
    print(f"{'family':<12}{'winner':<20}{'ref auroc':>10}")
    winners: dict[str, tuple[dict, float]] = {}
    for family, (kind_key, _) in FAMILIES.items():
        entry, ref_auroc = pick_best(cfg_paths[family], kind_key, calib_df, args.human_label)
        winners[family] = (entry, ref_auroc)
        print(f"{family:<12}{entry['label']:<20}{ref_auroc:>10.3f}")

    csv_path = args.csv or DEFAULT_OUT_DIR / "best_of_family_performance.csv"
    fig_path = args.fig or DEFAULT_OUT_DIR / "best_of_family_performance.png"

    print(f"\ndatasets : {[d.name for d in args.datasets]}\n")
    print(f"{'dataset':<28}{'family':<12}{'winner':<16}{'n':>7}{'auroc':>8}"
          f"{'thr':>10}{'tpr':>8}{'fpr':>8}")

    rows: list[dict] = []
    for ds_path in args.datasets:
        df = pd.read_parquet(ds_path)
        for family, (kind_key, _) in FAMILIES.items():
            entry, ref_auroc = winners[family]
            kind = _KIND_BY_KEY[kind_key]
            r = evaluate(df, entry["column"], entry["threshold"], kind.machine_is_high,
                         args.human_label)
            if r is None:
                print(f"{ds_path.stem:<28}{family:<12}{entry['label']:<16}  -- skipped: "
                      f"column missing, single class, or n<10 --")
                continue
            row = {
                "dataset": ds_path.stem, "family": family, "column": entry["column"],
                "label": entry["label"], "ref_auroc": ref_auroc,
                "threshold": entry["threshold"], **r,
            }
            rows.append(row)
            print(f"{ds_path.stem:<28}{family:<12}{entry['label']:<16}{row['n']:>7}"
                  f"{row['auroc']:>8.3f}{row['threshold']:>10.3g}{row['tpr']:>8.3f}"
                  f"{row['fpr']:>8.3f}")

    if not rows:
        raise SystemExit("\nnothing evaluated")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")

    if not args.no_plot:
        plot(rows, args.fpr_target, fig_path)


if __name__ == "__main__":
    main()
