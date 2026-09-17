r"""Zero-shot detector performance on the rephrase task: AI (original +
rephrased) vs. human (original + rephrased).

    y = 1   ai             the full gpt56luna generation ('...:0701A' GB, '...:0811A' Sci)
    y = 1   ai_rephrased   that generation, rephrased again by gpt56luna ('...:0916R')
    y = 0   human          generated_by == 'human'
    y = 0   human_rephrased generated_by == 'gpt56luna:0908R'

For every Binoculars column and, separately, every Fast-DetectGPT column, on
MG-GB / MG-Sci (and MG-All pooled, kept in the CSV): oriented AUROC
(machine = positive) and the ORACLE TPR@1%FPR -- the threshold read off that
corpus's own ROC -- exactly like ``run_zeroshot_performance.py``. One scatter
per family:

    colour = detector version, marker shape = corpus,
    bold filled circle = mean across MG-GB and MG-Sci

``--fpr`` takes one or more oracle FPR levels (default 1% and 5%); the CSV
carries an ``fpr`` column and each level gets its own ``_1pct`` / ``_5pct``
figure set.

compute -> CSV -> plots from the CSV:

    results/trainv2/rephrase_zeroshot_performance/
        performance.csv   one row per (family, version, split, fpr) AND (…, 'mean');
                          split in {mg_gb, mg_sci, all}
        binoculars_1pct.png / binoculars_5pct.png
        fastdetect_1pct.png / fastdetect_5pct.png

Run:
    uv run python epai/ai_detection/trainv2/rephrase_zeroshot_performance.py
    uv run python epai/ai_detection/trainv2/rephrase_zeroshot_performance.py \
        --replot results/trainv2/rephrase_zeroshot_performance/performance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.cd_core import (
    auroc_score,
    oriented_scores,
    rates_at_threshold,
    roc_point_at_fpr,
)

from epai.ai_detection.trainv2.core import fpr_tag, rows_at_fpr
from epai.ai_detection.trainv2.plot_score_histograms import (
    CORPUS_SOURCES,
    PARQUET_DIR,
    build,
)
from epai.ai_detection.trainv2.run_zeroshot_performance import DEFAULT_FPRS
from epai.ai_detection.trainv2.run_zeroshot_performance import FAMILIES as _ZS_FAMILIES

BASE = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = BASE / "results" / "trainv2" / "rephrase_zeroshot_performance"

#: only the two score families the user asked for -- reuse the exact specs
#: (prefix / machine_is_high / label) from run_zeroshot_performance.
FAMILIES = {k: _ZS_FAMILIES[k] for k in ("binoculars", "fastdetect")}

SPLITS = ["mg_gb", "mg_sci"]          # the real corpora; the 'mean' is over these
CORPORA = [*SPLITS, "all"]            # + pooled (CSV only, not plotted)

CSV_FIELDS = [
    "family", "version", "column", "split", "fpr", "n", "n_human", "n_machine",
    "auroc", "tpr_at_fpr", "fpr_at_fpr", "reachable",
]

_TITLE = {"binoculars": "Binoculars pairs", "fastdetect": "Fast-DetectGPT pairs"}


def _cols(spec: dict, names: set[str]) -> list[str]:
    return sorted(c for c in names if c.startswith(spec["prefix"])
                  and c not in spec.get("exclude", frozenset()))


def _cell(sub: pd.DataFrame, column: str, machine_is_high: bool,
          fprs: list[float]) -> list[dict]:
    """One (corpus, column) reading at each FPR level: {ai, ai_rephrased} vs.
    {human, human_rephrased}. ``[]`` on a degenerate slice."""
    v = sub[column].to_numpy(dtype=float)
    cls = sub["cls"].to_numpy()
    ok = np.isfinite(v)
    v, cls = v[ok], cls[ok]
    y = np.isin(cls, ["ai", "ai_rephrased"]).astype(int)  # human(_rephrased) -> 0
    if np.unique(y).size < 2 or len(y) < 10:
        return []
    oriented = oriented_scores(v, machine_is_high)
    auroc_v = float(auroc_score(y, oriented))
    n_h = int((y == 0).sum())
    out: list[dict] = []
    for fpr in fprs:
        tpr, thr = roc_point_at_fpr(y, oriented, fpr)
        _, fpr_act = rates_at_threshold(y, oriented, thr)
        out.append({
            "fpr": fpr, "n": int(len(y)), "n_human": n_h,
            "n_machine": int((y == 1).sum()), "auroc": auroc_v,
            "tpr_at_fpr": float(tpr), "fpr_at_fpr": float(fpr_act),
            "reachable": bool(n_h * fpr >= 1.0 and np.isfinite(thr)),
        })
    return out


def compute(df: pd.DataFrame, fprs: list[float]) -> list[dict]:
    names = set(df.columns)
    rows: list[dict] = []
    for fam, spec in FAMILIES.items():
        for col in _cols(spec, names):
            ver = spec["label"](col)
            cells: dict[tuple, dict] = {}
            for corpus in CORPORA:
                for r in _cell(df[df.corpus == corpus], col,
                               spec["machine_is_high"], fprs):
                    rows.append({"family": fam, "version": ver, "column": col,
                                 "split": corpus, **r})
                    cells[(corpus, r["fpr"])] = r
            for f in fprs:
                real = [cells[(s, f)] for s in SPLITS if (s, f) in cells]
                if not real:
                    continue
                tprs = [c["tpr_at_fpr"] for c in real if c["reachable"]]
                rows.append({
                    "family": fam, "version": ver, "column": col, "split": "mean",
                    "fpr": f, "n": "", "n_human": "", "n_machine": "",
                    "auroc": float(np.mean([c["auroc"] for c in real])),
                    "tpr_at_fpr": float(np.mean(tprs)) if tprs else float("nan"),
                    "fpr_at_fpr": "", "reachable": "",
                })
    return rows


def _print_table(rows: list[dict], fprs: list[float]) -> None:
    print(f"\n{'family':<12}{'version':<14}{'split':<9}{'FPR':>6}{'n':>7}"
          f"{'auroc':>8}{'  TPR':>9}{'  fpr':>7}")
    for r in rows:
        tpr = "--" if not np.isfinite(r["tpr_at_fpr"]) else f"{r['tpr_at_fpr']:.3f}"
        fp = "" if r["split"] == "mean" else f"{r['fpr_at_fpr']:.3f}"
        mark = "  <<" if r["split"] == "mean" else (
            "" if r["reachable"] or r["split"] == "mean" else " (floor)")
        n = "" if r["split"] == "mean" else r["n"]
        print(f"{r['family']:<12}{r['version']:<14}{r['split']:<9}"
              f"{float(r['fpr']) * 100:>5g}%{str(n):>7}"
              f"{r['auroc']:>8.3f}{tpr:>9}{fp:>7}{mark}")


def plots_from_csv(csv_path: Path, meta: dict, out_dir: Path) -> None:
    from epai.ai_detection.trainv2.plots import zeroshot_scatter

    all_rows = pd.read_csv(csv_path).to_dict("records")
    rows = [r for r in all_rows if r["split"] in (*meta["splits"], "mean")]
    out_dir.mkdir(parents=True, exist_ok=True)
    for fpr in meta["fprs"]:
        fmeta = {**meta, "fpr": fpr}
        tag = fpr_tag(fpr)
        frows = rows_at_fpr(rows, fpr)
        for family in sorted({r["family"] for r in frows}):
            fam_rows = [r for r in frows if r["family"] == family]
            title = (f"{_TITLE.get(family, family)} — AI+rephrase vs. "
                     f"human+rephrase — zero-shot AUROC vs. oracle TPR@{fpr * 100:g}%FPR")
            zeroshot_scatter(fam_rows, fmeta, out_dir / f"{family}_{tag}.png",
                             key="version", title=title)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fpr", nargs="+", type=float, default=DEFAULT_FPRS,
                    metavar="LEVEL",
                    help=f"oracle FPR level(s); one figure set per level "
                         f"(default: {DEFAULT_FPRS})")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="PERFORMANCE_CSV",
                    help="skip scoring; redraw the PNGs from an existing CSV")
    args = ap.parse_args()

    meta = {"fprs": args.fpr, "splits": SPLITS}

    if args.replot is not None:
        plots_from_csv(args.replot, meta, args.replot.parent)
        print(f"replotted from {args.replot}")
        return 0

    import pyarrow.parquet as pq

    names: set[str] | None = None
    for pf in {p for c in CORPUS_SOURCES.values() for p in (c["full"], c["reph"])}:
        path = PARQUET_DIR / pf
        if not path.exists():
            raise SystemExit(f"not found: {path}  (run split_manugram_reph.py first?)")
        s = set(pq.read_schema(path).names)
        names = s if names is None else (names & s)

    columns = sorted({c for spec in FAMILIES.values() for c in _cols(spec, names)})
    if not columns:
        raise SystemExit("no binoculars_score* / fastdetect_crit_* columns present")
    print(f"columns : {columns}")
    print("target  : y=1 ai + ai_rephrased; y=0 human + human_rephrased")
    print(f"metric  : AUROC (oriented), oracle TPR@"
          f"{[f'{f * 100:g}%' for f in args.fpr]}FPR\n")

    df = build(columns)
    rows = compute(df, args.fpr)
    if not rows:
        raise SystemExit("nothing evaluated")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "performance.csv"
    pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
    _print_table(rows, args.fpr)
    print(f"\nwrote {csv_path}")

    if not args.no_plots:
        plots_from_csv(csv_path, meta, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
