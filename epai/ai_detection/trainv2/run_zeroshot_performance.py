r"""Zero-shot detector performance across the SCRIBE splits: AUROC vs. the
ORACLE TPR@1%FPR, one scatter per detector family.

No training, no calibrated cutoff, no seeds. For every score column of a
family, on each SCRIBE split (pooled over its domains and generators), this
reads AUROC (threshold-free, oriented so machine = positive) and the ORACLE
TPR@1%FPR -- the threshold read off THAT split's own ROC
(``cd_core.roc_point_at_fpr``), an upper bound on the operating point and the
one reading that needs no calibration slice.

compute -> CSV -> 3 plots drawn from the CSV:

    results/trainv2/zeroshot_performance/
        performance.csv    one row per (family, version, split) AND (…, 'mean')
        binoculars.png     AUROC vs oracle TPR@1%FPR, Binoculars pairs
        fastdetect.png     … Fast-DetectGPT pairs
        judge.png          … LLM judges

In each plot: colour = detector version, marker shape = SCRIBE split, and a
bold filled circle = that version's mean across the splits (the answer; the
small shape-coded points are the per-split evidence).

Two notes on the mean:
  * SCRIBE-Denario has 80 human rows, so a 1%-FPR threshold sits on < 1 of
    them -- its TPR@1%FPR is a floor, not a measurement. Those cells are kept
    in the CSV (``reachable = False``) but left OUT of the mean's TPR (still
    in the mean's AUROC).
  * ``llm_judge_score_sonnet5_v3`` is absent on SCRIBE-Sci and dropped by
    default -- pass it via ``--cols`` to include it.

Replaces run_{binox,fastdetect,judge}_performance.py (which applied a fixed
transported cutoff); the oracle sibling of run_embedding_effectiveness.py.

Run:
    uv run python epai/ai_detection/trainv2/run_zeroshot_performance.py
    uv run python epai/ai_detection/trainv2/run_zeroshot_performance.py \
        --families judge --splits gb gb_orig
    uv run python epai/ai_detection/trainv2/run_zeroshot_performance.py \
        --replot results/trainv2/zeroshot_performance/performance.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.cd_core import (
    auroc_score,
    oriented_scores,
    rates_at_threshold,
    roc_point_at_fpr,
)

from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "zeroshot_performance"
GEN_COL = "generated_by"


def _binox_label(c: str) -> str:
    if c == "binoculars_score":
        return "falcon-7b"
    return c.removeprefix("binoculars_score_").replace("_", "-")


def _fastdetect_label(c: str) -> str:
    return c.removeprefix("fastdetect_crit_").replace("_", "-")


def _judge_label(c: str) -> str:
    return c.removeprefix("llm_judge_score_").replace("_v3", "")


#: prefix / orientation / label / default-excluded columns per family.
FAMILIES: dict[str, dict] = {
    "binoculars": dict(prefix="binoculars_score", machine_is_high=False,
                       label=_binox_label, exclude=frozenset()),
    "fastdetect": dict(prefix="fastdetect_crit_", machine_is_high=True,
                       label=_fastdetect_label, exclude=frozenset()),
    "judge": dict(prefix="llm_judge_score_", machine_is_high=False,
                  label=_judge_label,
                  exclude=frozenset({"llm_judge_score_sonnet5_v3"})),
}

CSV_FIELDS = [
    "family", "version", "column", "split", "n", "n_human", "n_machine",
    "auroc", "tpr_at_fpr", "fpr_at_fpr", "reachable",
]


def _cols(names, spec: dict, cols_arg: list[str] | None) -> list[str]:
    if cols_arg:
        return [c for c in cols_arg if c in names and c.startswith(spec["prefix"])]
    return sorted(c for c in names if c.startswith(spec["prefix"])
                  and c not in spec["exclude"])


def compute(
    splits: list[tuple[str, Path]],
    families: dict[str, dict],
    cols_arg: list[str] | None,
    fpr: float,
    human_label: str,
) -> list[dict]:
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for split_key, path in splits:
        names = pq.read_schema(path).names
        want = {GEN_COL}
        for spec in families.values():
            want |= set(_cols(names, spec, cols_arg))
        df = pd.read_parquet(path, columns=sorted(want & set(names)))

        for fam_key, spec in families.items():
            for col in _cols(df.columns, spec, cols_arg):
                sub = df[[col, GEN_COL]].dropna(subset=[col])
                raw = sub[col].to_numpy(dtype=float)
                y = (sub[GEN_COL].astype(str) != human_label).to_numpy().astype(int)
                if np.unique(y).size < 2 or len(y) < 10:
                    continue
                oriented = oriented_scores(raw, spec["machine_is_high"])
                tpr, thr = roc_point_at_fpr(y, oriented, fpr)
                _, fpr_act = rates_at_threshold(y, oriented, thr)
                n_h = int((y == 0).sum())
                rows.append({
                    "family": fam_key, "version": spec["label"](col), "column": col,
                    "split": split_key, "n": int(len(y)), "n_human": n_h,
                    "n_machine": int((y == 1).sum()),
                    "auroc": float(auroc_score(y, oriented)),
                    "tpr_at_fpr": float(tpr), "fpr_at_fpr": float(fpr_act),
                    "reachable": bool(n_h * fpr >= 1.0 and np.isfinite(thr)),
                })

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["family"], r["version"], r["column"])].append(r)
    for (fam, ver, col), cells in grouped.items():
        tprs = [c["tpr_at_fpr"] for c in cells if c["reachable"]]
        rows.append({
            "family": fam, "version": ver, "column": col, "split": "mean",
            "n": "", "n_human": "", "n_machine": "",
            "auroc": float(np.mean([c["auroc"] for c in cells])),
            "tpr_at_fpr": float(np.mean(tprs)) if tprs else float("nan"),
            "fpr_at_fpr": "", "reachable": "",
        })
    return rows


def _sorted_rows(rows: list[dict], split_order: list[str]) -> list[dict]:
    fam_ix = {f: i for i, f in enumerate(FAMILIES)}
    split_ix = {s: i for i, s in enumerate([*split_order, "mean"])}
    return sorted(rows, key=lambda r: (fam_ix.get(r["family"], 9), r["version"],
                                       split_ix.get(r["split"], 9)))


def _print_table(rows: list[dict], fpr: float) -> None:
    pct = f"{fpr * 100:g}"
    print(f"\n{'family':<12}{'version':<14}{'split':<12}{'n':>7}{'auroc':>8}"
          f"{f'  TPR@{pct}%':>11}{'  fpr':>7}")
    for r in rows:
        star = "" if r["reachable"] or r["split"] == "mean" else " (floor)"
        n = r["n"] if r["split"] != "mean" else ""
        tpr = "--" if not np.isfinite(r["tpr_at_fpr"]) else f"{r['tpr_at_fpr']:.3f}"
        fp = "" if r["split"] == "mean" else f"{r['fpr_at_fpr']:.3f}"
        mark = "  <<" if r["split"] == "mean" else ""
        print(f"{r['family']:<12}{r['version']:<14}{r['split']:<12}{str(n):>7}"
              f"{r['auroc']:>8.3f}{tpr:>11}{fp:>7}{mark}{star}")


_FAMILY_TITLE = {"binoculars": "Binoculars pairs", "fastdetect": "Fast-DetectGPT pairs",
                 "judge": "LLM judges"}


def plots_from_csv(csv_path: Path, meta: dict, out_dir: Path) -> None:
    rows = pd.read_csv(csv_path).to_dict("records")
    from epai.ai_detection.trainv2.plots import zeroshot_scatter

    out_dir.mkdir(parents=True, exist_ok=True)
    fpr = meta["fpr"] * 100
    for family in sorted({r["family"] for r in rows}):
        fam_rows = [r for r in rows if r["family"] == family]
        title = (f"{_FAMILY_TITLE.get(family, family)} — zero-shot AUROC vs. "
                 f"oracle TPR@{fpr:g}%FPR across the SCRIBE splits")
        zeroshot_scatter(fam_rows, meta, out_dir / f"{family}.png",
                         key="version", title=title)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--families", nargs="+", default=None, choices=list(FAMILIES),
                    help="detector families (default: all three)")
    ap.add_argument("--splits", nargs="+", default=None, metavar="KEY",
                    help="SCRIBE splits (default: all)")
    ap.add_argument("--cols", nargs="+", default=None,
                    help="score columns (default: every one of each family's "
                         "prefix, minus the family's exclude list)")
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="PERFORMANCE_CSV",
                    help="skip scoring; redraw the PNGs from an existing CSV")
    args = ap.parse_args()

    splits = resolve_splits(args.splits or [], PARQUET_DIR)
    meta = {"fpr": args.fpr, "splits": [s for s, _ in splits]}

    if args.replot is not None:
        plots_from_csv(args.replot, meta, args.replot.parent)
        print(f"replotted from {args.replot}")
        return

    families = {k: FAMILIES[k] for k in (args.families or FAMILIES)}
    missing = [str(p) for _, p in splits if not p.exists()]
    if missing:
        raise SystemExit(f"parquet(s) not found: {missing}")

    print(f"families : {list(families)}")
    print(f"splits   : {[s for s, _ in splits]}")
    print(f"metric   : AUROC (oriented), oracle TPR@{args.fpr * 100:g}%FPR")

    rows = _sorted_rows(
        compute(splits, families, args.cols, args.fpr, args.human_label),
        meta["splits"],
    )
    if not rows:
        raise SystemExit("nothing evaluated")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "performance.csv"
    pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
    _print_table(rows, args.fpr)
    print(f"\nwrote {csv_path}")

    if not args.no_plots:
        plots_from_csv(csv_path, meta, args.out_dir)


if __name__ == "__main__":
    main()
