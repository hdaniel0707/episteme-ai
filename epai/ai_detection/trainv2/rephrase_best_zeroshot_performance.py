r"""The hand-picked best zero-shot detector per family -- rephrase task, AI
(original + rephrased) vs. human (original + rephrased) -- on ONE figure that
shows the SHIFT from each detector's pre-rephrase reading.

``performance.csv`` (all 5 hand-picked detectors, incl. RADAR) and its
results table are computed exactly as before: reads the same flat column
list from ``--config`` (``cfg/ai_detection/trainv2/best_zeroshot.yaml``)
**minus any gemma judge** (null on the rephrase rows), and scores each on
MG-GB / MG-Sci -- plus MG-All pooled, kept in the CSV only -- with

    y = 1   ai + ai_rephrased (gpt56luna full generation, and that generation
            rephrased again by gpt56luna: '...:0701A'/'...:0811A' + '...:0916R')
    y = 0   human + human_rephrased (gpt56luna:0908R)

The FIGURE is different: only the four detectors with a real zero-shot
signal here (Binoculars, Fast-DetectGPT, the parsed-answer judge, LogJudge --
RADAR dropped) are plotted, each paired with its own PRE-rephrase reading
(``compute_baseline()``: the same detectors on MG-GB / MG-Sci, human vs. the
full unrephrased generation only -- i.e. ``run_best_zeroshot_performance.py``'s
own scoring, restricted to those two splits). Both readings share the
detector's colour; the rephrase-task point is filled, the pre-rephrase point
is hollow, and the two means are joined by a line -- reusing
``plots.zeroshot_scatter``'s existing hollow-base/filled-variant machinery
(normally LogJudge vs. its parsed-answer base) with ``variant_suffix="_reph"``
instead, so here each detector pairs with ITSELF on the other task rather
than with a different detector. The result reads as a before/after shift per
detector rather than five independent points. No title on the figure -- the
caption carries it, matching the article's other figures.

compute -> CSV -> plot from the CSV:

    results/trainv2/rephrase_best_zeroshot_performance/
        performance.csv                    all 5 detectors; feeds the results table only
        shift_performance.csv              4 detectors x {baseline, rephrase}; feeds the figure
        rephrase_best_zeroshot_1pct.png     also written as .pdf
        rephrase_best_zeroshot_1pct.pdf

Run:
    uv run python epai/ai_detection/trainv2/rephrase_best_zeroshot_performance.py
    uv run python epai/ai_detection/trainv2/rephrase_best_zeroshot_performance.py \
        --replot results/trainv2/rephrase_best_zeroshot_performance/shift_performance.csv
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
from epai.ai_detection.trainv2.run_zeroshot_performance import compute as zs_compute
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

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


#: the figure drops RADAR -- the request was the four detectors with a
#: meaningful zero-shot AUROC on this task (binox, fastdetect, judge, LogJudge).
#: performance.csv / the results table (rephrase_general_results_table.py)
#: are untouched and still carry all five.
FIG_OUT_FIELDS = ["kind", "detector", "family", "version", "column", "split", "fpr",
                   "n", "n_human", "n_machine", "auroc", "tpr_at_fpr", "fpr_at_fpr",
                   "reachable"]


def fig_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if not c.startswith("detector_score_")]


def compute_baseline(columns: list[str], fprs: list[float]) -> list[dict]:
    """The same detectors' PRE-rephrase reading: MG-GB / MG-Sci, human vs. the
    full (unrephrased) generation only -- what each rephrase-task point in
    Figure~\\ref{fig:rephrasing-scatter} is measured against."""
    splits = resolve_splits(["mg_gb", "mg_sci"], PARQUET_DIR)
    rows = zs_compute(splits, ZS_FAMILIES, columns, fprs, "human")
    for r in rows:
        r["detector"] = _detector(r["family"], r["version"])
    return rows


def plots_from_csv(csv_path: Path, meta: dict, out_dir: Path) -> None:
    """Reads the ``shift_performance.csv`` (``kind`` in {baseline, rephrase}):
    a rephrase-task row plots filled, keyed ``<detector>_reph``; its baseline
    counterpart plots hollow, keyed ``<detector>`` -- ``zeroshot_scatter``'s
    existing ``_reph``-suffix variant machinery then draws same-colour
    hollow/filled pairs connected at their mean, exactly like the LogJudge/
    parsed-judge pairing on the main figure, but pairing each detector with
    its own pre-rephrase reading instead of a different detector."""
    from epai.ai_detection.trainv2.plots import zeroshot_scatter

    df = pd.read_csv(csv_path)
    df = df[df["split"].isin((*meta["splits"], "mean"))].copy()
    df["plot_key"] = df["detector"] + np.where(df["kind"] == "rephrase", "_reph", "")
    rows = df.to_dict("records")
    out_dir.mkdir(parents=True, exist_ok=True)
    for fpr in meta["fprs"]:
        frows = rows_at_fpr(rows, fpr)
        name = f"rephrase_best_zeroshot_{fpr_tag(fpr)}"
        paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
        zeroshot_scatter(frows, {**meta, "fpr": fpr}, paths, key="plot_key",
                          variant_suffix="_reph")


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
    ap.add_argument("--replot", type=Path, default=None, metavar="SHIFT_PERFORMANCE_CSV")
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
    print("target  : y=1 ai + ai_rephrased; y=0 human + human_rephrased")
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

    fcols = fig_columns(columns)
    baseline_rows = compute_baseline(fcols, args.fpr)
    shift_rows = ([{**r, "kind": "baseline"} for r in baseline_rows]
                  + [{**r, "kind": "rephrase"} for r in rows if r["column"] in fcols])
    shift_csv = args.out_dir / "shift_performance.csv"
    pd.DataFrame(shift_rows, columns=FIG_OUT_FIELDS).to_csv(shift_csv, index=False)
    print(f"wrote {shift_csv}")

    if not args.no_plots:
        plots_from_csv(shift_csv, meta, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
