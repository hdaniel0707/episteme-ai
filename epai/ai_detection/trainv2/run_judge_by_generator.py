r"""Judge LLM x generator LLM: does each judge catch each generator's text?

For every (judge, generator) pair, on each of the three multi-generator SCRIBE
splits (SCRIBE-GB-orig, SCRIBE-GB, SCRIBE-Sci) -- human rows vs. that ONE
generator's rows, every other generator excluded from the cell -- this reports
two ORACLE numbers, each its own matrix:

    AUROC            threshold-free, oriented so machine = positive
    TPR@1%FPR        threshold read off THAT CELL's own ROC
                     (cd_core.roc_point_at_fpr), nothing carried in

Cells where the judge and the generator are the same model family are marked
``*`` -- whether a judge is soft on its own family is the reason this matrix
exists. (SCRIBE-GB-orig is all GPT-3.5, so it has no self-family cells.)

compute -> CSV -> plots from the CSV:

    results/trainv2/judge_by_generator/
        matrix.csv           long: split, judge, generator, n, auroc,
                              tpr_at_fpr, self_family
        auroc.png            one heatmap panel per split
        tpr_at_1pct_fpr.png  one heatmap panel per split

``llm_judge_score_sonnet5_v3`` is dropped by default (absent on SCRIBE-Sci);
pass it via ``--cols`` to include it.

Replaces the old single-dataset, gemma-less version.

Run:
    uv run python epai/ai_detection/trainv2/run_judge_by_generator.py
    uv run python epai/ai_detection/trainv2/run_judge_by_generator.py --splits gb sci
    uv run python epai/ai_detection/trainv2/run_judge_by_generator.py \
        --replot results/trainv2/judge_by_generator/matrix.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.cd_core import auroc_score, oriented_scores, roc_point_at_fpr

from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "judge_by_generator"

JUDGE_PREFIX = "llm_judge_score_"
JUDGE_EXCLUDE = frozenset({"llm_judge_score_sonnet5_v3"})
GEN_COL = "generated_by"
# The judge writes P(human): a LOW score means machine. Matches cd_core's
# llm_judge BaselineKind.
MACHINE_IS_HIGH = False
#: the multi-generator splits -- SCRIBE-Denario has one generator, no matrix.
DEFAULT_SPLITS = ["gb_orig", "gb", "sci"]

CSV_FIELDS = ["split", "judge", "generator", "gen_model", "n", "auroc",
              "tpr_at_fpr", "self_family"]


def _judge_key(column: str) -> str:
    return column.removeprefix(JUDGE_PREFIX).replace("_v3", "")


def _gen_model(generator: str) -> str:
    """``gpt56luna:0701A`` -> ``gpt56luna``."""
    return generator.split(":", 1)[0]


def _norm(s: str) -> str:
    """Strip separators and case so 'gptoss_120b' and 'gptoss120b' compare equal."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _cell(df: pd.DataFrame, column: str, generator: str, human_label: str,
          fpr: float) -> dict | None:
    sub = df[df[GEN_COL].isin([human_label, generator])][[column, GEN_COL]]
    sub = sub.dropna(subset=[column])
    raw = sub[column].to_numpy(dtype=float)
    y = (sub[GEN_COL].astype(str) != human_label).to_numpy().astype(int)
    if np.unique(y).size < 2 or len(y) < 10:
        return None
    oriented = oriented_scores(raw, MACHINE_IS_HIGH)
    tpr, _ = roc_point_at_fpr(y, oriented, fpr)
    return {"n": int(len(y)), "auroc": float(auroc_score(y, oriented)),
            "tpr_at_fpr": float(tpr)}


def compute(splits, cols_arg, fpr, human_label) -> list[dict]:
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for split_key, path in splits:
        names = pq.read_schema(path).names
        cols = (cols_arg if cols_arg
                else sorted(c for c in names if c.startswith(JUDGE_PREFIX)
                            and c not in JUDGE_EXCLUDE))
        cols = [c for c in cols if c in names]
        df = pd.read_parquet(path, columns=[GEN_COL, *cols])
        generators = sorted(g for g in df[GEN_COL].astype(str).unique()
                            if g != human_label)
        for col in cols:
            judge = _judge_key(col)
            for gen in generators:
                r = _cell(df, col, gen, human_label, fpr)
                if r is None:
                    continue
                rows.append({
                    "split": split_key, "judge": judge, "generator": gen,
                    "gen_model": _gen_model(gen), "n": r["n"],
                    "auroc": r["auroc"], "tpr_at_fpr": r["tpr_at_fpr"],
                    "self_family": _norm(judge) == _norm(_gen_model(gen)),
                })
    return rows


def _overall(rows: list[dict]) -> list[dict]:
    """One row per judge: its grand mean of each metric over every cell
    (all splits, all generators), sorted best AUROC first."""
    import numpy as np

    out = []
    for j in sorted({r["judge"] for r in rows}):
        cells = [r for r in rows if r["judge"] == j]
        out.append({
            "judge": j, "n_cells": len(cells),
            "mean_auroc": float(np.mean([c["auroc"] for c in cells])),
            "mean_tpr_at_fpr": float(np.mean([c["tpr_at_fpr"] for c in cells])),
        })
    return sorted(out, key=lambda r: r["mean_auroc"], reverse=True)


def _print_matrices(rows: list[dict], split_order: list[str], fpr: float) -> None:
    ovr = {o["judge"]: o for o in _overall(rows)}
    for metric, key, lbl in (("auroc", "mean_auroc", "AUROC"),
                             ("tpr_at_fpr", "mean_tpr_at_fpr",
                              f"oracle TPR@{fpr * 100:g}%FPR")):
        judges = sorted(ovr, key=lambda j: ovr[j][key], reverse=True)
        for sk in split_order:
            sub = {(r["judge"], r["gen_model"]): r
                   for r in rows if r["split"] == sk}
            if not sub:
                continue
            gens = sorted({g for _, g in sub})
            print(f"\n{sk} — {lbl}  (* = same model family; rows ranked by overall)")
            print(f"{'judge':<14}" + "".join(f"{g:>13}" for g in gens)
                  + f"{'overall':>13}")
            for j in judges:
                line = f"{j:<14}"
                for g in gens:
                    r = sub.get((j, g))
                    line += (f"{'--':>13}" if r is None else
                             f"{r[metric]:>12.3f}{'*' if r['self_family'] else ' '}")
                print(line + f"{ovr[j][key]:>13.3f}")


def plots_from_csv(csv_path: Path, meta: dict, out_dir: Path) -> None:
    rows = pd.read_csv(csv_path).to_dict("records")
    from epai.ai_detection.trainv2.plots import judge_generator_heatmaps

    out_dir.mkdir(parents=True, exist_ok=True)
    judge_generator_heatmaps(rows, "auroc", meta, out_dir / "auroc.png")
    judge_generator_heatmaps(rows, "tpr_at_fpr", meta,
                             out_dir / f"tpr_at_{meta['fpr'] * 100:g}pct_fpr.png")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--splits", nargs="+", default=None, metavar="KEY",
                    help=f"SCRIBE splits (default: {DEFAULT_SPLITS})")
    ap.add_argument("--cols", nargs="+", default=None,
                    help=f"judge columns (default: every {JUDGE_PREFIX}* column "
                         f"minus {sorted(JUDGE_EXCLUDE)})")
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="MATRIX_CSV")
    args = ap.parse_args()

    splits = resolve_splits(args.splits or DEFAULT_SPLITS, PARQUET_DIR)
    meta = {"fpr": args.fpr, "splits": [s for s, _ in splits]}

    if args.replot is not None:
        plots_from_csv(args.replot, meta, args.replot.parent)
        print(f"replotted from {args.replot}")
        return

    missing = [str(p) for _, p in splits if not p.exists()]
    if missing:
        raise SystemExit(f"parquet(s) not found: {missing}")

    print(f"splits : {[s for s, _ in splits]}")
    print(f"metric : AUROC (oriented), oracle TPR@{args.fpr * 100:g}%FPR, per (judge, generator)")

    rows = compute(splits, args.cols, args.fpr, args.human_label)
    if not rows:
        raise SystemExit("nothing evaluated")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "matrix.csv"
    overall_path = args.out_dir / "overall.csv"
    pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
    pd.DataFrame(_overall(rows),
                 columns=["judge", "n_cells", "mean_auroc", "mean_tpr_at_fpr"]
                 ).to_csv(overall_path, index=False)
    _print_matrices(rows, meta["splits"], args.fpr)
    print(f"\nwrote {csv_path}\nwrote {overall_path}")

    if not args.no_plots:
        plots_from_csv(csv_path, meta, args.out_dir)


if __name__ == "__main__":
    main()
