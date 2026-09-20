r"""Zero-shot detector performance across the MANUGRAM splits: AUROC vs. the
ORACLE TPR@1%FPR, one scatter per detector family.

No training, no calibrated cutoff, no seeds. For every score column of a
family, on each MANUGRAM split (pooled over its domains and generators), this
reads AUROC (threshold-free, oriented so machine = positive) and the ORACLE
TPR@1%FPR -- the threshold read off THAT split's own ROC
(``cd_core.roc_point_at_fpr``), an upper bound on the operating point and the
one reading that needs no calibration slice.

``--fpr`` takes one or more oracle FPR levels (default 1% and 5%); the CSV
carries an ``fpr`` column and each level gets its own figure set, suffixed
``_1pct`` / ``_5pct``.

compute -> CSV -> plots drawn from the CSV (one scatter + one goodness bar per
family per FPR level):

    results/trainv2/zeroshot_performance/
        performance.csv      one row per (family, version, split, fpr) AND (…, 'mean')
        binoculars_1pct.png  AUROC vs oracle TPR@1%FPR, Binoculars pairs
        binoculars_5pct.png  … at 5% FPR
        fastdetect_1pct.png  … Fast-DetectGPT pairs
        judge_1pct.png       … LLM judges
        detector_1pct.png    … pretrained detectors (RADAR)

In each plot: colour = detector version, marker shape = MANUGRAM split, and a
bold filled circle = that version's mean across the splits (the answer; the
small shape-coded points are the per-split evidence).

Two notes on the mean:
  * MG-AI-Sci has 100 human rows (post-merge with the ai_scientist corpus --
    was 80), which puts a 1%-FPR threshold right AT the reachability boundary
    (``reachable = n_human * fpr >= 1.0``) instead of clearly below it, so its
    TPR@1%FPR cells now count in the CSV (``reachable = True``) and toward the
    mean's TPR too, not just its AUROC. Still read it as a floor, not a
    measurement -- one row's worth of margin either way flips the whole
    number.
  * ``llm_judge_score_sonnet5_v3`` is absent on MG-Sci and dropped by
    default -- pass it via ``--cols`` to include it.

Replaces run_{binox,fastdetect,judge}_performance.py (which applied a fixed
transported cutoff); the oracle sibling of run_embedding_effectiveness.py.

Run:
    uv run python epai/ai_detection/trainv2/run_zeroshot_performance.py
    uv run python epai/ai_detection/trainv2/run_zeroshot_performance.py \
        --families judge --splits mg_gb gb_orig
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

from epai.ai_detection.trainv2.core import eff_fpr_level, fpr_tag, rows_at_fpr
from epai.ai_detection.trainv2.splits import POOL_SPLITS
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

DEFAULT_FPRS = [0.01, 0.05]

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
    """``llm_judge_score_gpt56luna_v3`` -> ``gpt56luna`` (the parsed-answer
    baseline, unchanged); ``logjudge_ord2_gpt56luna_v3`` -> ``gpt56luna_ord2``
    (the paper's ord2 readout) -- an explicit ``_ord2`` suffix so
    :func:`plots.zeroshot_scatter`'s hollow-base pairing (``variant_suffix``)
    still connects it to its parsed base, the same way ``_wlp`` used to."""
    if c.startswith("logjudge_ord2_"):
        return c.removeprefix("logjudge_ord2_").replace("_v3", "") + "_ord2"
    return c.removeprefix("llm_judge_score_").replace("_v3", "")


def _detector_label(c: str) -> str:
    return c.removeprefix("detector_score_").replace("_", "-")


def _fusionjudge_label(c: str) -> str:
    """``fusionjudge_score`` -> ``gpt56luna_v3`` (the paper's plain mean); a
    variant column ``fusionjudge_<variant>_score``
    (score_fusionjudge_variants_gpt56luna_v3.py, ``_logodds`` or ``_bayes``)
    -> ``gpt56luna_v3_<variant>`` -- same one member pair, a different
    (self-dual) combination rule."""
    body = c.removeprefix("fusionjudge_")
    return "gpt56luna_v3" if body == "score" else f"gpt56luna_v3_{body.removesuffix('_score')}"


#: prefix / orientation / label / default-excluded columns per family.
FAMILIES: dict[str, dict] = {
    "binoculars": dict(prefix="binoculars_score", machine_is_high=False,
                       label=_binox_label, exclude=frozenset()),
    "fastdetect": dict(prefix="fastdetect_crit_", machine_is_high=True,
                       label=_fastdetect_label, exclude=frozenset()),
    "judge": dict(prefix=("llm_judge_score_", "logjudge_ord2_"), machine_is_high=False,
                  label=_judge_label,
                  exclude=frozenset({
                      "llm_judge_score_sonnet5_v3",
                      # Superseded logprob-window scoring experiments -- the
                      # comparison kept the winners (`_wlp` per model + the
                      # soft-min ensemble at T=0.05). `_slp` / `_sbc` / `_lp`
                      # and the other `_smin_*` taus lost; see
                      # epai/ai_detection/score/score_logprob_windows.md.
                      # Pass any of them via --cols to plot it anyway.
                      "llm_judge_score_gemma4_31b_v3_lp",
                      "llm_judge_score_gemma4_31b_v3_slp",
                      "llm_judge_score_gemma4_31b_v3_sbc",
                      "llm_judge_score_gpt56luna_v3_slp",
                      "llm_judge_score_gpt56luna_v3_sbc",
                      "llm_judge_score_ens_v3_smin_0p1",
                      "llm_judge_score_ens_v3_smin_0p15",
                      "llm_judge_score_ens_v3_smin_0p2",
                  })),
    "detector": dict(prefix="detector_score_", machine_is_high=True,
                     label=_detector_label, exclude=frozenset()),
    # score_fusionjudge_gpt56luna_v3.py: mean(llm_judge_score_gpt56luna_v3,
    # logjudge_ord2_gpt56luna_v3) -- both already P(human) on the same
    # scale/orientation, so a plain mean is meaningful (unlike a
    # binoculars/fastdetect/judge fusion, whose raw ranges and directions
    # don't agree -- see the conversation that led here).
    # score_fusionjudge_variants_gpt56luna_v3.py adds two more self-dual
    # combination rules of the same two members (log-odds average, Bayes/
    # product rule) as fusionjudge_<variant>_score -- the shared
    # "fusionjudge_" prefix picks up all three. A wider sweep (gamma family,
    # agreement gate, disagreement penalties) was tried and dropped; see
    # that script's docstring.
    "fusionjudge": dict(prefix="fusionjudge_", machine_is_high=False,
                     label=_fusionjudge_label, exclude=frozenset()),
}

CSV_FIELDS = [
    "family", "version", "column", "split", "fpr", "n", "n_human", "n_machine",
    "auroc", "tpr_at_fpr", "fpr_at_fpr", "reachable", "tpr_eff", "eff_fpr",
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
    fprs: list[float],
    human_label: str,
) -> list[dict]:
    """One row per (family, version, column, split, **fpr**) plus the per-fpr
    ``split='mean'`` row. AUROC is threshold-free so it repeats across a
    column's fpr rows; ``tpr_at_fpr`` / ``reachable`` / ``tpr_eff`` are per-fpr."""
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
                auroc_v = float(auroc_score(y, oriented))
                n_h = int((y == 0).sum())
                for fpr in fprs:
                    tpr, thr = roc_point_at_fpr(y, oriented, fpr)
                    _, fpr_act = rates_at_threshold(y, oriented, thr)
                    eff = eff_fpr_level(n_h, fpr)
                    tpr_eff, _ = roc_point_at_fpr(y, oriented, eff)
                    rows.append({
                        "family": fam_key, "version": spec["label"](col), "column": col,
                        "split": split_key, "fpr": fpr, "n": int(len(y)), "n_human": n_h,
                        "n_machine": int((y == 1).sum()),
                        "auroc": auroc_v,
                        "tpr_at_fpr": float(tpr), "fpr_at_fpr": float(fpr_act),
                        "reachable": bool(n_h * fpr >= 1.0 and np.isfinite(thr)),
                        "tpr_eff": float(tpr_eff), "eff_fpr": float(eff),
                    })

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["family"], r["version"], r["column"], r["fpr"])].append(r)
    for (fam, ver, col, fpr), cells in grouped.items():
        # "mean" is the MG-All aggregate (goodness_bars relabels it as such) --
        # pool over the MANUGRAM splits only, same exclusion as splits.POOL_SPLITS.
        pool_cells = [c for c in cells if c["split"] in POOL_SPLITS]
        tprs = [c["tpr_at_fpr"] for c in pool_cells if c["reachable"]]
        effs = [c["tpr_eff"] for c in pool_cells]
        rows.append({
            "family": fam, "version": ver, "column": col, "split": "mean", "fpr": fpr,
            "n": "", "n_human": "", "n_machine": "",
            "auroc": float(np.mean([c["auroc"] for c in pool_cells])) if pool_cells else float("nan"),
            "tpr_at_fpr": float(np.mean(tprs)) if tprs else float("nan"),
            "fpr_at_fpr": "", "reachable": "",
            "tpr_eff": float(np.mean(effs)) if effs else float("nan"), "eff_fpr": fpr,
        })
    return rows


def _sorted_rows(rows: list[dict], split_order: list[str]) -> list[dict]:
    fam_ix = {f: i for i, f in enumerate(FAMILIES)}
    split_ix = {s: i for i, s in enumerate([*split_order, "mean"])}
    return sorted(rows, key=lambda r: (fam_ix.get(r["family"], 9), r["version"],
                                       float(r["fpr"]), split_ix.get(r["split"], 9)))


def _print_table(rows: list[dict], fprs: list[float]) -> None:
    print(f"\n{'family':<12}{'version':<18}{'split':<11}{'FPR':>6}{'n':>7}"
          f"{'auroc':>8}{'  TPR':>9}{'  fpr':>7}")
    for r in rows:
        star = "" if r["reachable"] or r["split"] == "mean" else " (floor)"
        n = r["n"] if r["split"] != "mean" else ""
        tpr = "--" if not np.isfinite(r["tpr_at_fpr"]) else f"{r['tpr_at_fpr']:.3f}"
        fp = "" if r["split"] == "mean" else f"{r['fpr_at_fpr']:.3f}"
        mark = "  <<" if r["split"] == "mean" else ""
        print(f"{r['family']:<12}{r['version']:<18}{r['split']:<11}"
              f"{float(r['fpr']) * 100:>5g}%{str(n):>7}"
              f"{r['auroc']:>8.3f}{tpr:>9}{fp:>7}{mark}{star}")


_FAMILY_TITLE = {"binoculars": "Binoculars pairs", "fastdetect": "Fast-DetectGPT pairs",
                 "judge": "LLM judges", "detector": "Pretrained detectors"}


def plots_from_csv(csv_path: Path, meta: dict, out_dir: Path) -> None:
    rows = pd.read_csv(csv_path).to_dict("records")
    from epai.ai_detection.trainv2.plots import goodness_bars, zeroshot_scatter

    out_dir.mkdir(parents=True, exist_ok=True)
    combine = meta.get("combine", "hmean")
    for fpr in meta["fprs"]:
        fmeta = {**meta, "fpr": fpr}
        tag = fpr_tag(fpr)
        frows = rows_at_fpr(rows, fpr)
        for family in sorted({r["family"] for r in frows}):
            fam_rows = [r for r in frows if r["family"] == family]
            title = (f"{_FAMILY_TITLE.get(family, family)} — zero-shot AUROC vs. "
                     f"oracle TPR@{fpr * 100:g}%FPR across the MANUGRAM splits")
            zeroshot_scatter(fam_rows, fmeta, out_dir / f"{family}_{tag}.png",
                             key="version", title=title)
            goodness_bars(fam_rows, fmeta, out_dir / f"{family}_goodness_{tag}.png",
                          key="version", combine=combine,
                          title=f"{_FAMILY_TITLE.get(family, family)} — goodness "
                                f"(TPR@{fpr * 100:g}%FPR) across the MANUGRAM splits")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--families", nargs="+", default=None, choices=list(FAMILIES),
                    help="detector families (default: all four)")
    ap.add_argument("--splits", nargs="+", default=None, metavar="KEY",
                    help="MANUGRAM splits (default: all)")
    ap.add_argument("--cols", nargs="+", default=None,
                    help="score columns (default: every one of each family's "
                         "prefix, minus the family's exclude list)")
    ap.add_argument("--fpr", nargs="+", type=float, default=DEFAULT_FPRS,
                    metavar="LEVEL",
                    help=f"oracle FPR level(s); one figure set per level "
                         f"(default: {DEFAULT_FPRS})")
    ap.add_argument("--combine", choices=("hmean", "gmean", "amean"), default="hmean",
                    help="goodness-bar score = this mean of Gini and oracle TPR "
                         "(default: harmonic)")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="PERFORMANCE_CSV",
                    help="skip scoring; redraw the PNGs from an existing CSV")
    args = ap.parse_args()

    splits = resolve_splits(args.splits or [], PARQUET_DIR)
    meta = {"fprs": args.fpr, "splits": [s for s, _ in splits],
            "combine": args.combine}

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
    print(f"metric   : AUROC (oriented), oracle TPR@"
          f"{[f'{f * 100:g}%' for f in args.fpr]}FPR")

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
