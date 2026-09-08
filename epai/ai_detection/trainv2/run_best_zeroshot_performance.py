r"""The hand-picked best zero-shot detector of each family, on ONE figure.

Reads a flat list of score columns from ``--config`` (default
``cfg/ai_detection/trainv2/best_zeroshot.yaml``), scores each on every SCRIBE
split exactly as ``run_zeroshot_performance.py`` does -- oriented AUROC +
ORACLE TPR@1%FPR, threshold refit on that split's own ROC -- and plots them
all on one AUROC-vs-TPR@1%FPR scatter:

    colour = detector (``<family>/<version>``)
    marker shape = SCRIBE split (splits.MARKER)
    bold filled circle = that detector's mean across the splits

SCRIBE-Denario (80 human) can't reach 1% FPR, so its cells are kept in the
mean's AUROC but left out of the mean's TPR -- same rule as
``run_zeroshot_performance.py``.

The curated-picks counterpart of ``run_best_of_family_performance.py`` (which
auto-picks the strongest column per family and applies a transported cutoff).

compute -> CSV -> plot from the CSV:

    results/trainv2/best_zeroshot_performance/
        performance.csv     one row per (detector, split) AND (detector, 'mean')
        best_zeroshot.png

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

_FAMILY_ABBR = {"binoculars": "binox", "fastdetect": "fastdetect", "judge": "judge"}
OUT_FIELDS = ["detector", *CSV_FIELDS]


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
    from epai.ai_detection.trainv2.plots import zeroshot_scatter

    out_dir.mkdir(parents=True, exist_ok=True)
    zeroshot_scatter(
        rows, meta, out_dir / "best_zeroshot.png", key="detector",
        title=f"Best zero-shot detector per family — AUROC vs. oracle "
              f"TPR@{meta['fpr'] * 100:g}%FPR across the SCRIBE splits",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--splits", nargs="+", default=None, metavar="KEY")
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="PERFORMANCE_CSV")
    args = ap.parse_args()

    splits = resolve_splits(args.splits or [], PARQUET_DIR)
    meta = {"fpr": args.fpr, "splits": [s for s, _ in splits]}

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
    print(f"metric : AUROC (oriented), oracle TPR@{args.fpr * 100:g}%FPR\n")

    rows = compute(splits, FAMILIES, cols, args.fpr, args.human_label)
    for r in rows:
        r["detector"] = _detector(r["family"], r["version"])
    order = {c: i for i, c in enumerate(cols)}
    split_ix = {s: i for i, s in enumerate([*meta["splits"], "mean"])}
    rows.sort(key=lambda r: (order.get(r["column"], 9), split_ix.get(r["split"], 9)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "performance.csv"
    pd.DataFrame(rows, columns=OUT_FIELDS).to_csv(csv_path, index=False)
    print(pd.DataFrame(rows, columns=["detector", "split", "auroc", "tpr_at_fpr",
                                      "reachable"]).to_string(index=False))
    print(f"\nwrote {csv_path}")

    if not args.no_plots:
        plots_from_csv(csv_path, meta, args.out_dir)


if __name__ == "__main__":
    main()
