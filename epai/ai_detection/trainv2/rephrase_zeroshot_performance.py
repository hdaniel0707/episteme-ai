r"""Zero-shot detector performance on the rephrase task, with rephrased text
counted as HUMAN ground truth.

The detection target is the FULL gpt-5.6-luna generation only:

    y = 1   rewritten    the full gpt56luna generation ('...:0701A' GB, '...:0811A' Sci)
    y = 0   human        generated_by == 'human'
    y = 0   rephrase     generated_by == 'gpt56luna:0908R'   <-- counted as human

For every Binoculars column and, separately, every Fast-DetectGPT column, on
MG-GB / MG-Sci (and MG-All pooled, kept in the CSV): oriented AUROC
(machine = positive) and the ORACLE TPR@1%FPR -- the threshold read off that
corpus's own ROC -- exactly like ``run_zeroshot_performance.py``. One scatter
per family:

    colour = detector version, marker shape = corpus,
    bold filled circle = mean across MG-GB and MG-Sci

compute -> CSV -> plots from the CSV:

    results/trainv2/rephrase_zeroshot_performance/
        performance.csv   one row per (family, version, split) AND (..., 'mean');
                          split in {mg_gb, mg_sci, all}
        binoculars.png
        fastdetect.png

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

from epai.ai_detection.trainv2.plot_score_histograms import (
    CORPUS_SOURCES,
    PARQUET_DIR,
    build,
)
from epai.ai_detection.trainv2.run_zeroshot_performance import FAMILIES as _ZS_FAMILIES

BASE = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = BASE / "results" / "trainv2" / "rephrase_zeroshot_performance"

#: only the two score families the user asked for -- reuse the exact specs
#: (prefix / machine_is_high / label) from run_zeroshot_performance.
FAMILIES = {k: _ZS_FAMILIES[k] for k in ("binoculars", "fastdetect")}

SPLITS = ["mg_gb", "mg_sci"]          # the real corpora; the 'mean' is over these
CORPORA = [*SPLITS, "all"]            # + pooled (CSV only, not plotted)

CSV_FIELDS = [
    "family", "version", "column", "split", "n", "n_human", "n_machine",
    "auroc", "tpr_at_fpr", "fpr_at_fpr", "reachable",
]

_TITLE = {"binoculars": "Binoculars pairs", "fastdetect": "Fast-DetectGPT pairs"}


def _cols(spec: dict, names: set[str]) -> list[str]:
    return sorted(c for c in names if c.startswith(spec["prefix"])
                  and c not in spec.get("exclude", frozenset()))


def _cell(sub: pd.DataFrame, column: str, machine_is_high: bool, fpr: float) -> dict | None:
    """One (corpus, column) reading: rephrase + human vs. the full rewrite."""
    v = sub[column].to_numpy(dtype=float)
    cls = sub["cls"].to_numpy()
    ok = np.isfinite(v)
    v, cls = v[ok], cls[ok]
    y = (cls == "rewritten").astype(int)  # human and rephrase -> 0
    if np.unique(y).size < 2 or len(y) < 10:
        return None
    oriented = oriented_scores(v, machine_is_high)
    tpr, thr = roc_point_at_fpr(y, oriented, fpr)
    _, fpr_act = rates_at_threshold(y, oriented, thr)
    n_h = int((y == 0).sum())
    return {
        "n": int(len(y)), "n_human": n_h, "n_machine": int((y == 1).sum()),
        "auroc": float(auroc_score(y, oriented)),
        "tpr_at_fpr": float(tpr), "fpr_at_fpr": float(fpr_act),
        "reachable": bool(n_h * fpr >= 1.0 and np.isfinite(thr)),
    }


def compute(df: pd.DataFrame, fpr: float) -> list[dict]:
    names = set(df.columns)
    rows: list[dict] = []
    for fam, spec in FAMILIES.items():
        for col in _cols(spec, names):
            ver = spec["label"](col)
            cells: dict[str, dict] = {}
            for corpus in CORPORA:
                r = _cell(df[df.corpus == corpus], col, spec["machine_is_high"], fpr)
                if r is None:
                    continue
                rows.append({"family": fam, "version": ver, "column": col,
                             "split": corpus, **r})
                cells[corpus] = r
            real = [cells[s] for s in SPLITS if s in cells]
            if not real:
                continue
            tprs = [c["tpr_at_fpr"] for c in real if c["reachable"]]
            rows.append({
                "family": fam, "version": ver, "column": col, "split": "mean",
                "n": "", "n_human": "", "n_machine": "",
                "auroc": float(np.mean([c["auroc"] for c in real])),
                "tpr_at_fpr": float(np.mean(tprs)) if tprs else float("nan"),
                "fpr_at_fpr": "", "reachable": "",
            })
    return rows


def _print_table(rows: list[dict], fpr: float) -> None:
    pct = f"{fpr * 100:g}"
    print(f"\n{'family':<12}{'version':<14}{'split':<9}{'n':>7}{'auroc':>8}"
          f"{f'  TPR@{pct}%':>11}{'  fpr':>7}")
    for r in rows:
        tpr = "--" if not np.isfinite(r["tpr_at_fpr"]) else f"{r['tpr_at_fpr']:.3f}"
        fp = "" if r["split"] == "mean" else f"{r['fpr_at_fpr']:.3f}"
        mark = "  <<" if r["split"] == "mean" else (
            "" if r["reachable"] or r["split"] == "mean" else " (floor)")
        n = "" if r["split"] == "mean" else r["n"]
        print(f"{r['family']:<12}{r['version']:<14}{r['split']:<9}{str(n):>7}"
              f"{r['auroc']:>8.3f}{tpr:>11}{fp:>7}{mark}")


def plots_from_csv(csv_path: Path, meta: dict, out_dir: Path) -> None:
    from epai.ai_detection.trainv2.plots import zeroshot_scatter

    all_rows = pd.read_csv(csv_path).to_dict("records")
    rows = [r for r in all_rows if r["split"] in (*meta["splits"], "mean")]
    out_dir.mkdir(parents=True, exist_ok=True)
    for family in sorted({r["family"] for r in rows}):
        fam_rows = [r for r in rows if r["family"] == family]
        title = (f"{_TITLE.get(family, family)} — rephrase counted as human — "
                 f"zero-shot AUROC vs. oracle TPR@{meta['fpr'] * 100:g}%FPR")
        zeroshot_scatter(fam_rows, meta, out_dir / f"{family}.png",
                         key="version", title=title)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="PERFORMANCE_CSV",
                    help="skip scoring; redraw the PNGs from an existing CSV")
    args = ap.parse_args()

    meta = {"fpr": args.fpr, "splits": SPLITS}

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
    print(f"target  : y=1 the full gpt56luna generation; human AND rephrase -> y=0")
    print(f"metric  : AUROC (oriented), oracle TPR@{args.fpr * 100:g}%FPR\n")

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
