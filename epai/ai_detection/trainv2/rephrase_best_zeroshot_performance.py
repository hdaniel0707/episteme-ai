r"""The hand-picked best zero-shot detector per family on ONE figure -- rephrase
task, rephrased text counted as HUMAN ground truth.

The rephrase-GT sibling of ``run_best_zeroshot_performance.py``: reads the same
flat column list from ``--config`` (``cfg/ai_detection/trainv2/best_zeroshot.yaml``)
**minus any gemma judge** (null on the rephrase rows), and scores each on
MG-GB / MG-Sci -- plus MG-All pooled, kept in the CSV only -- with

    y = 1   the full gpt56luna generation ('...:0701A' GB, '...:0811A' Sci)
    y = 0   human AND rephrase (gpt56luna:0908R)

reporting oriented AUROC and the ORACLE TPR@1%FPR, all on one
AUROC-vs-TPR@1%FPR scatter:

    colour = detector (``<family>/<version>``)
    marker shape = corpus (MG-GB / MG-Sci)
    bold filled circle = that detector's mean across MG-GB and MG-Sci

compute -> CSV -> plot from the CSV:

    results/trainv2/rephrase_best_zeroshot_performance/
        performance.csv     one row per (detector, split) AND (detector, 'mean')
        best_zeroshot.png

Run:
    uv run python epai/ai_detection/trainv2/rephrase_best_zeroshot_performance.py
    uv run python epai/ai_detection/trainv2/rephrase_best_zeroshot_performance.py \
        --replot results/trainv2/rephrase_best_zeroshot_performance/performance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epai.ai_detection.trainv2.plot_score_histograms import (
    CORPUS_SOURCES,
    PARQUET_DIR,
    build,
)
from epai.ai_detection.trainv2.core import fpr_tag, rows_at_fpr
from epai.ai_detection.trainv2.rephrase_zeroshot_performance import CORPORA, SPLITS, _cell
from epai.ai_detection.trainv2.run_best_zeroshot_performance import (
    DEFAULT_CONFIG,
    OUT_FIELDS,
    _detector,
    _load_columns,
)
from epai.ai_detection.trainv2.run_zeroshot_performance import DEFAULT_FPRS
from epai.ai_detection.trainv2.run_zeroshot_performance import FAMILIES as ZS_FAMILIES

BASE = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = BASE / "results" / "trainv2" / "rephrase_best_zeroshot_performance"

_GEMMA_JUDGE = "llm_judge_score_gemma"  # dropped -- null on the rephrase rows


def _family_of(column: str) -> tuple[str, dict]:
    for fam, spec in ZS_FAMILIES.items():
        if column.startswith(spec["prefix"]):
            return fam, spec
    raise SystemExit(f"{column!r} matches no known family prefix")


def _reph_schema() -> set[str]:
    """Columns present in EVERY rephrase parquet (full + reph half of each
    corpus). The reph parquets were scored before the logprob-window / ensemble
    judges, so ``best_zeroshot.yaml`` names columns they don't carry."""
    import pyarrow.parquet as pq

    common: set[str] | None = None
    for c in CORPUS_SOURCES.values():
        for pf in (c["full"], c["reph"]):
            names = set(pq.read_schema(PARQUET_DIR / pf).names)
            common = names if common is None else (common & names)
    return common or set()


def _pick_columns(config: Path) -> list[str]:
    cols = _load_columns(config)
    kept = [c for c in cols if not c.startswith(_GEMMA_JUDGE)]
    if dropped := [c for c in cols if c not in kept]:
        print(f"dropped (gemma judge -- null on the rephrase rows): {dropped}")
    schema = _reph_schema()
    present = [c for c in kept if c in schema]
    if absent := [c for c in kept if c not in schema]:
        print(f"dropped (not scored on the rephrase parquets): {absent}")
    return present


def compute(df: pd.DataFrame, columns: list[str], fprs: list[float]) -> list[dict]:
    rows: list[dict] = []
    for column in columns:
        fam, spec = _family_of(column)
        ver = spec["label"](column)
        det = _detector(fam, ver)
        cells: dict[tuple, dict] = {}
        for corpus in CORPORA:
            for r in _cell(df[df.corpus == corpus], column,
                           spec["machine_is_high"], fprs):
                rows.append({"detector": det, "family": fam, "version": ver,
                             "column": column, "split": corpus, **r})
                cells[(corpus, r["fpr"])] = r
        for f in fprs:
            real = [cells[(s, f)] for s in SPLITS if (s, f) in cells]
            if not real:
                continue
            tprs = [c["tpr_at_fpr"] for c in real if c["reachable"]]
            rows.append({
                "detector": det, "family": fam, "version": ver, "column": column,
                "split": "mean", "fpr": f, "n": "", "n_human": "", "n_machine": "",
                "auroc": float(np.mean([c["auroc"] for c in real])),
                "tpr_at_fpr": float(np.mean(tprs)) if tprs else float("nan"),
                "fpr_at_fpr": "", "reachable": "",
            })
    return rows


def plots_from_csv(csv_path: Path, meta: dict, out_dir: Path) -> None:
    from epai.ai_detection.trainv2.plots import zeroshot_scatter

    rows = [r for r in pd.read_csv(csv_path).to_dict("records")
            if r["split"] in (*meta["splits"], "mean")]
    out_dir.mkdir(parents=True, exist_ok=True)
    for fpr in meta["fprs"]:
        frows = rows_at_fpr(rows, fpr)
        zeroshot_scatter(
            frows, {**meta, "fpr": fpr},
            out_dir / f"best_zeroshot_{fpr_tag(fpr)}.png", key="detector",
            title=f"Best zero-shot detector per family — rephrase counted as human "
                  f"— AUROC vs. oracle TPR@{fpr * 100:g}%FPR",
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--fpr", nargs="+", type=float, default=DEFAULT_FPRS,
                    metavar="LEVEL",
                    help=f"oracle FPR level(s); one figure per level "
                         f"(default: {DEFAULT_FPRS})")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="PERFORMANCE_CSV")
    args = ap.parse_args()

    meta = {"fprs": args.fpr, "splits": SPLITS}

    if args.replot is not None:
        plots_from_csv(args.replot, meta, args.replot.parent)
        print(f"replotted from {args.replot}")
        return 0

    columns = _pick_columns(args.config)
    if not columns:
        raise SystemExit("no columns left after filtering")

    for pf in {p for c in CORPUS_SOURCES.values() for p in (c["full"], c["reph"])}:
        if not (PARQUET_DIR / pf).exists():
            raise SystemExit(f"not found: {PARQUET_DIR / pf}  (run split_manugram_reph.py first?)")

    print(f"config  : {args.config}")
    print(f"columns : {columns}")
    print("target  : y=1 the full gpt56luna generation; human AND rephrase -> y=0")
    print(f"metric  : AUROC (oriented), oracle TPR@"
          f"{[f'{f * 100:g}%' for f in args.fpr]}FPR\n")

    df = build(columns)
    rows = compute(df, columns, args.fpr)
    if not rows:
        raise SystemExit("nothing evaluated")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "performance.csv"
    pd.DataFrame(rows, columns=OUT_FIELDS).to_csv(csv_path, index=False)
    print(pd.DataFrame(rows, columns=["detector", "split", "fpr", "auroc",
                                      "tpr_at_fpr", "reachable"]).to_string(index=False))
    print(f"\nwrote {csv_path}")

    if not args.no_plots:
        plots_from_csv(csv_path, meta, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
