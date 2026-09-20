r"""The hand-picked best zero-shot detector of each family, on ONE figure.

Reads a flat list of score columns from ``--config`` (default
``cfg/ai_detection/trainv2/best_zeroshot.yaml``), scores each on every MANUGRAM
split exactly as ``run_zeroshot_performance.py`` does -- oriented AUROC +
ORACLE TPR@1%FPR, threshold refit on that split's own ROC -- and plots them
all on one AUROC-vs-TPR@1%FPR scatter:

    colour = detector (``<family>/<version>``)
    marker shape = MANUGRAM split (splits.MARKER)
    bold filled circle = that detector's mean across the splits

MG-AI-Sci now has 100 human rows (was 80, pre-merge with the ai_scientist
corpus). At exactly 100 rows, ``reachable = n_human * fpr >= 1.0`` flips TRUE
for the 1%-FPR level -- unlike the old 80-row file, its TPR@1%FPR now DOES
count toward the pooled mean's TPR, not just its AUROC (the exclusion rule
below, inherited from ``run_zeroshot_performance.py``, still applies verbatim
-- it is the input that changed, not the rule). Read a lone MG-AI-Sci
TPR@1%FPR reading as a floor regardless: one threshold's worth of headroom is
still not a stable operating point.

The curated-picks companion to ``run_zeroshot_performance.py`` -- that sweeps
every column of every family; this scores just the hand-picked list and puts
them on one figure.

``--fpr`` takes one or more oracle FPR levels (default 1% and 5%); each gets its
own figure set, suffixed ``_1pct`` / ``_5pct``.

compute -> CSV -> plot from the CSV:

    results/trainv2/best_zeroshot_performance/
        performance.csv       one row per (detector, split, fpr) AND (…, 'mean')
        best_zeroshot_1pct.png / best_zeroshot_5pct.png
        best_zeroshot_goodness_1pct.png / …_5pct.png

Run:
    uv run python epai/ai_detection/trainv2/run_best_zeroshot_performance.py
    uv run python epai/ai_detection/trainv2/run_best_zeroshot_performance.py \
        --replot results/trainv2/best_zeroshot_performance/performance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from epai.ai_detection.trainv2.core import fpr_tag, rows_at_fpr
from epai.ai_detection.trainv2.run_zeroshot_performance import (
    CSV_FIELDS,
    FAMILIES,
    compute,
)
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_CONFIG = BASE_DIR / "cfg" / "ai_detection" / "trainv2" / "best_zeroshot.yaml"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "best_zeroshot_performance"

_FAMILY_ABBR = {"binoculars": "binox", "fastdetect": "fastdetect", "judge": "judge",
                 "detector": "detector"}
OUT_FIELDS = ["detector", *CSV_FIELDS]

# Style for this one figure only (zeroshot_scatter's color_override/
# light_border/emphasize/chance_color default to a no-op everywhere else):
# baselines keep normal, fully-saturated colours from the paper's existing
# validated 8-hue palette -- they're real published methods, not props, so
# they don't get greyed out, and every marker (baseline or ours) is the same
# size with a black border. "Ours" (LLM-judge family = blue, FusionJudge =
# purple) pops instead through a bolder/thicker border plus drawing on top
# (emphasize), against baselines' thinner black one (light_border) -- weight
# and layering, not size, colour, or desaturation. Blue is vivified from the
# palette's default slot (#2a78d6 -> #1d6fe0, OKLCH chroma 0.16 -> 0.19)
# since it still clears the CVD pair check against purple (ΔE 12.3 deutan /
# 15.0 normal-vision) at that chroma; purple is left at its existing slot
# (#4a3aa7) because pushing it more vivid too is what collapses the
# blue/purple separation (tested: drops below the ΔE 15 normal-vision
# floor). Baselines pull three more slots from the same documented palette
# (green/gold/red); the full 5-hue set clears CVD/lightness, with a
# WCAG-contrast and a CVD WARN on the baseline pair that the legend text +
# per-split marker shapes + emphasize treatment already cover as secondary
# encoding (see the dataviz skill's validator -- run via conversation, not
# committed here). The chance cross/dotted lines are recoloured to a neutral
# grey so they read as a reference mark, not a seventh series.
_ACCENT_BLUE = "#1d6fe0"
_ACCENT_PURPLE = "#4a3aa7"
_CHANCE_GRAY = "#8a8a86"
_BASELINE_STYLE = {
    "binox/llama31-8b": "#1baf7a",
    "fastdetect/llama31-8b": "#eda100",
    "detector/radar": "#e34948",
}
_COLOR_OVERRIDE = {
    **_BASELINE_STYLE,
    "judge/gpt56luna": _ACCENT_BLUE,
    "judge/gpt56luna_ord2": _ACCENT_BLUE,
    "fusionjudge/gpt56luna_v3": _ACCENT_PURPLE,
}
_LIGHT_BORDER = frozenset(_BASELINE_STYLE)
_EMPHASIZE = frozenset({"judge/gpt56luna", "judge/gpt56luna_ord2", "fusionjudge/gpt56luna_v3"})

# Legend text for this figure only: short official names, no backbone/version
# suffix and no AT-HM score (show_scores=False handles the rest) -- the
# per-point detail lives in the table, not the legend. The two contributions
# also get an explicit "(ours)" tag alongside their existing bold weight.
_LEGEND_LABEL = {
    "fusionjudge/gpt56luna_v3": "FusionJudge (ours)",
    "judge/gpt56luna_ord2": "LogJudge (ours)",
    "judge/gpt56luna": "BaseJudge",
    "fastdetect/llama31-8b": "Fast-DetectGPT",
    "binox/llama31-8b": "Binoculars",
    "detector/radar": "RADAR",
}

# Style for the AT-HM bars figure (athm_bars) only -- it has no hollow-pair
# scheme, so BaseJudge needs its own colour instead of inheriting LogJudge's
# via the scatter's variant-suffix pairing. #4f77a5 is a steel blue, chosen
# over a pale tint of LogJudge's own hue (too easily misread as the same
# detector) and over a neutral grey (reads as "greyed out"/excluded) --
# saturated enough to read as its own colour while staying in the same "cool
# blue" family as LogJudge/FusionJudge. Only the two contributions get the
# bold x-tick label + full-strength AT-HM outline -- BaseJudge is a baseline
# here.
_ATHM_BARS_COLOR = {**_COLOR_OVERRIDE, "judge/gpt56luna": "#4f77a5"}
_ATHM_BARS_EMPHASIZE = frozenset({"judge/gpt56luna_ord2", "fusionjudge/gpt56luna_v3"})


def _detector(family: str, version: str) -> str:
    return f"{_FAMILY_ABBR.get(family, family)}/{version}"


def _load_columns(path: Path) -> list[str]:
    cfg = yaml.safe_load(path.read_text()) or {}
    cols = cfg.get("columns")
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        raise SystemExit(f"{path}: expected a 'columns:' list of score column names")
    unknown = [c for c in cols
               if not any(c.startswith(f["prefix"]) for f in FAMILIES.values())]
    if unknown:
        raise SystemExit(f"{path}: columns match no known family prefix: {unknown}")
    return cols


def plots_from_csv(csv_path: Path, meta: dict, out_dir: Path) -> None:
    rows = pd.read_csv(csv_path).to_dict("records")
    from epai.ai_detection.trainv2.plots import athm_bars, goodness_bars, zeroshot_scatter

    out_dir.mkdir(parents=True, exist_ok=True)
    for fpr in meta["fprs"]:
        fmeta = {**meta, "fpr": fpr}
        tag = fpr_tag(fpr)
        frows = rows_at_fpr(rows, fpr)
        zeroshot_scatter(
            frows, fmeta,
            [out_dir / f"best_zeroshot_{tag}.png", out_dir / f"best_zeroshot_{tag}.pdf"],
            key="detector", variant_suffix="_ord2",
            color_override=_COLOR_OVERRIDE, light_border=_LIGHT_BORDER,
            emphasize=_EMPHASIZE, chance_color=_CHANCE_GRAY,
            label_override=_LEGEND_LABEL, show_scores=False,
        )
        goodness_bars(
            frows, fmeta, out_dir / f"best_zeroshot_goodness_{tag}.png",
            key="detector", combine=meta.get("combine", "hmean"),
            variant_suffix="_ord2",
            title=f"Best zero-shot detector per family — goodness "
                  f"(TPR@{fpr * 100:g}%FPR) across the MANUGRAM splits",
        )
        athm_bars(
            frows, fmeta,
            [out_dir / f"best_zeroshot_bars_{tag}.png", out_dir / f"best_zeroshot_bars_{tag}.pdf"],
            key="detector", combine=meta.get("combine", "hmean"),
            color_override=_ATHM_BARS_COLOR, emphasize=_ATHM_BARS_EMPHASIZE,
            label_override=_LEGEND_LABEL,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--splits", nargs="+", default=None, metavar="KEY")
    ap.add_argument("--fpr", nargs="+", type=float, default=[0.01],
                    metavar="LEVEL",
                    help="oracle FPR level(s); one figure set per level "
                         "(default: [0.01] -- this runner's headline figure "
                         "is 1%%-FPR only; pass --fpr 0.01 0.05 for both)")
    ap.add_argument("--combine", choices=("hmean", "gmean", "amean"), default="hmean",
                    help="goodness-bar score = this mean of Gini and oracle TPR")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="PERFORMANCE_CSV")
    args = ap.parse_args()

    splits = resolve_splits(args.splits or [], PARQUET_DIR)
    meta = {"fprs": args.fpr, "splits": [s for s, _ in splits],
            "combine": args.combine}

    if args.replot is not None:
        plots_from_csv(args.replot, meta, args.replot.parent)
        print(f"replotted from {args.replot}")
        return

    cols = _load_columns(args.config)
    missing = [str(p) for _, p in splits if not p.exists()]
    if missing:
        raise SystemExit(f"parquet(s) not found: {missing}")

    print(f"config : {args.config}")
    print(f"columns: {cols}")
    print(f"splits : {[s for s, _ in splits]}")
    print(f"metric : AUROC (oriented), oracle TPR@"
          f"{[f'{f * 100:g}%' for f in args.fpr]}FPR\n")

    rows = compute(splits, FAMILIES, cols, args.fpr, args.human_label)
    for r in rows:
        r["detector"] = _detector(r["family"], r["version"])
    order = {c: i for i, c in enumerate(cols)}
    split_ix = {s: i for i, s in enumerate([*meta["splits"], "mean"])}
    rows.sort(key=lambda r: (order.get(r["column"], 9), float(r["fpr"]),
                             split_ix.get(r["split"], 9)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "performance.csv"
    pd.DataFrame(rows, columns=OUT_FIELDS).to_csv(csv_path, index=False)
    print(pd.DataFrame(rows, columns=["detector", "split", "fpr", "auroc",
                                      "tpr_at_fpr", "reachable"]).to_string(index=False))
    print(f"\nwrote {csv_path}")

    if not args.no_plots:
        plots_from_csv(csv_path, meta, args.out_dir)


if __name__ == "__main__":
    main()
