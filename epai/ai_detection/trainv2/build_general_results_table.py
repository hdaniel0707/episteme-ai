"""Fill the General Results LaTeX table from
``results/trainv2/best_zeroshot_performance/performance.csv``.

Rows = every hand-picked zero-shot/LogJudge detector (run_best_zeroshot_
performance.py's config), ranked by pooled-mean AT-HM, same order as that
run's scatter legend. Columns = each MANUGRAM core split + (unless
``--skip-mean``) the pooled mean, each split into AUROC / TPR@1%FPR / AT-HM
(goodness() at alpha=0.01, the paper's fixed definition -- recomputed here
since the CSV doesn't store a goodness column for zero-shot rows, same as
plots.py's goodness_bars()). Every metric header carries an ``\\uparrow``
(higher is better everywhere here); the best value in each column is
bolded; our own contributions (LogJudge, FusionJudge) get their row label
bolded too.

Every AT-HM cell -- including the Mean column's -- is ``goodness(auroc, tpr)``
computed from that same column's own (pooled, for Mean) AUROC/TPR pair, never
the mean of the four per-split AT-HM scores; this is the same convention
:func:`plots.zeroshot_scatter`'s legend and :func:`plots.goodness_bars` use,
so the Mean AT-HM printed here always matches the figure. The caption says
so explicitly.

Writes a standalone ``\\begin{table}...\\end{table}`` block (booktabs +
multirow style, already used elsewhere in the paper) to a .txt file for
manual copy-paste into the .tex -- this script does not touch the .tex
itself. ``--appendix`` transposes the full 5-group table instead of
squeezing it into \\linewidth as 16 tiny columns: rows become
(split, metric) pairs, columns become the 7 detectors, so it's a portrait
table only 9 columns wide (no rotation, no extra package -- multirow and
graphicx, both already in the preamble, are all it needs).

``--domains`` goes one level deeper: every core split that actually has a
``domain`` column with >1 value (GB-orig, MG-GB: essay/reuter/wp; MG-Sci:
computer_science/mathematics/sociology -- from
run_best_zeroshot_domains.py's ``performance_by_domain.csv``, run that
first) gets a row per domain PLUS its own pooled-mean row, so "the mean of
them" sits right under its domains. MG-AI-Sci is left as a single row --
its ten AI-Scientist-source "domains" run 8-40 rows each (some under 10
human examples), too small for a per-domain reading to mean much on top of
the split-level floor it already is. This table is long (domains x splits x
3 metrics), so it uses ``longtable`` instead of a single-page float --
needs ``\\usepackage{longtable}`` in the preamble.

Run:
    .venv/bin/python epai/ai_detection/trainv2/build_general_results_table.py --only-mean
    .venv/bin/python epai/ai_detection/trainv2/build_general_results_table.py --appendix
    .venv/bin/python epai/ai_detection/trainv2/run_best_zeroshot_domains.py  # once, for --domains
    .venv/bin/python epai/ai_detection/trainv2/build_general_results_table.py --domains
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from epai.ai_detection.trainv2.core import goodness
from epai.ai_detection.trainv2.detector_names import display_detector as _display_detector
from epai.ai_detection.trainv2.detector_names import is_judge_family
from epai.ai_detection.trainv2.splits import DEFAULT as CORE_SPLITS
from epai.ai_detection.trainv2.splits import display as _sd

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CSV = (BASE_DIR / "results" / "trainv2" / "best_zeroshot_performance"
               / "performance.csv")
DEFAULT_DOMAIN_CSV = (BASE_DIR / "results" / "trainv2" / "best_zeroshot_performance"
                       / "performance_by_domain.csv")
DEFAULT_OUT = (BASE_DIR / "results" / "trainv2" / "best_zeroshot_performance"
               / "general_results_table.txt")

METRICS = ("hm", "auroc", "tpr")
METRIC_LABEL = {"auroc": "AUROC", "tpr": "TPR@1\\%FPR", "hm": "AT-HM"}
#: a heavier separator between dataset groups (vs. a single \midrule between
#: rows within the same dataset, e.g. a split's domains).
DOUBLE_RULE = "\\midrule\n\\midrule"

#: Row/column labels for this table family (General Results + its --appendix
#: and --domains variants, and rephrase_general_results_table.py's table,
#: which calls into _plain_tabular directly) -- matches the main figure's
#: legend (run_best_zeroshot_performance.py's _LEGEND_LABEL): short official
#: names, no backbone/version suffix, "(ours)" alongside the existing bold
#: weight on our two contributions, and a \citep for each baseline (same
#: citation keys as the paper's own "As baselines we used ..." sentence).
#: Plain prose, not \texttt -- these are proper names now, not raw column
#: identifiers. A detector id absent from this map (e.g. a future addition)
#: falls back to the old \texttt{display_detector(...)} rendering.
TABLE_LABEL: dict[str, str] = {
    "fusionjudge/gpt56luna_v3": "FusionJudge (ours)",
    "judge/gpt56luna_ord2": "LogJudge (ours)",
    "judge/gpt56luna": "Judge",
    "fastdetect/llama31-8b": r"Fast-DetectGPT~\citep{bao2024fastdetectgpt}",
    "binox/llama31-8b": r"Binoculars~\citep{hans_spotting_2024}",
    "detector/radar": r"RADAR~\citep{hu_radar_2023}",
}


def _tex_escape(name: str) -> str:
    return name.replace("_", r"\_")


def _table_label(det: str) -> str:
    if det in TABLE_LABEL:
        return TABLE_LABEL[det]
    return f"\\texttt{{{_tex_escape(_display_detector(det))}}}"


def _group_label(key: str) -> str:
    """Column-group header -- same as :func:`splits.display` except the
    pooled-mean group reads "All Benchmark Mean" and GB-orig reads
    "Ghostbuster (Original)" with its citation, in this table only
    (``splits.DISPLAY``'s "Mean"/"GB-orig" stay the shared default
    everywhere else, e.g. the scatter legend and the --appendix/--domains
    table variants)."""
    if key == "mean":
        return "All Benchmark Mean"
    if key == "gb_orig":
        return r"Ghostbuster (Original)~\citep{verma_ghostbuster_2024}"
    return _sd(key)




def build(csv_path: Path, fpr: float, skip_mean: bool, only_mean: bool,
          appendix: bool, columns_override: list[str] | None = None) -> str:
    df = pd.read_csv(csv_path)
    df = df[df["fpr"] == fpr]
    if columns_override:
        columns = columns_override
    elif only_mean:
        columns = ["mean"]
    elif skip_mean:
        columns = list(CORE_SPLITS)
    else:
        columns = [*CORE_SPLITS, "mean"]

    cell = {(r["detector"], r["split"]): r for _, r in df.iterrows()}

    def metrics(det: str, split: str) -> dict[str, float] | None:
        r = cell.get((det, split))
        if r is None or pd.isna(r["auroc"]) or pd.isna(r["tpr_at_fpr"]):
            return None
        return {"auroc": r["auroc"], "tpr": r["tpr_at_fpr"],
                "hm": goodness(r["auroc"], r["tpr_at_fpr"], "hmean")}

    # ranking always by pooled-mean AT-HM, whether or not the mean column is shown
    detectors = sorted(df["detector"].unique(),
                        key=lambda d: -((metrics(d, "mean") or {"hm": -1.0})["hm"]))

    values = {(det, split): metrics(det, split)
              for det in detectors for split in columns}
    best = {(split, m): max((values[(det, split)][m] for det in detectors
                             if values[(det, split)] is not None), default=None)
            for split in columns for m in METRICS}

    if appendix:
        tabular, resize, bold_note = _transposed_tabular(detectors, columns, values, best)
    else:
        tabular, resize, bold_note = _plain_tabular(detectors, columns, values, best)

    groups = " and ".join(_group_label(c) for c in columns)
    caption = (
        f"General Results: zero-shot detectors on {groups}, oriented AUROC, "
        f"oracle TPR@{fpr * 100:g}\\%FPR, and AT-HM "
        f"(Section~\\ref{{sec:method_metric}}); higher is better. {bold_note}."
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        *([r"\resizebox{\linewidth}{!}{%"] if resize else []),
        *tabular,
        *(["}"] if resize else []),
        f"\\caption{{{caption}}}",
        r"\label{tab:general-results}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _plain_tabular(detectors, columns, values, best) -> tuple[list[str], bool, str]:
    """Rows = detectors, columns = split x metric -- fine up to ~2 split
    groups (``--only-mean``/``--skip-mean``); the full 5-group version is 16
    columns wide and reads better transposed (see :func:`_transposed_tabular`,
    ``--appendix``)."""
    header1 = " & " + " & ".join(
        f"\\multicolumn{{3}}{{c}}{{{_group_label(c)}}}" for c in columns) + r" \\"
    n_cols = len(columns)
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(n_cols))
    metric_labels = " & ".join(f"{METRIC_LABEL[m]} $\\uparrow$" for m in METRICS)
    header2 = "Detector & " + " & ".join([metric_labels] * n_cols) + r" \\"

    rows = []
    for det in detectors:
        cells = []
        for split in columns:
            v = values[(det, split)]
            for m in METRICS:
                if v is None:
                    cells.append("--")
                    continue
                s = f"{v[m]:.3f}"
                if best[(split, m)] is not None and s == f"{best[(split, m)]:.3f}":
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
        label = _table_label(det)
        if is_judge_family(det):
            label = f"\\textbf{{{label}}}"
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")

    col_spec = "l" + " ccc" * n_cols
    resize = n_cols > 1  # a 1-group table is narrow enough at its natural size
    tabular = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header1,
        cmidrules,
        header2,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}%" if resize else r"\end{tabular}",
    ]
    bold_note = (r"\textbf{Bold} row labels are our contributions "
                 r"(LogJudge, FusionJudge); \textbf{bold} values are the "
                 r"best per column")
    return tabular, resize, bold_note


def _rotated_header(detectors) -> str:
    """Two blank label cells + one rotated header per detector, bolded for
    the LLM-judge family -- shared by every transposed (rows = split x
    metric) table."""
    return " & & " + " & ".join(
        (r"\textbf{" if is_judge_family(det) else "")
        + f"\\rotatebox{{90}}{{{_table_label(det)}}}"
        + ("}" if is_judge_family(det) else "")
        for det in detectors) + r" \\"


def _metric_rows(detectors, key, values, best, group_label: str) -> list[str]:
    """The 3 body rows (AUROC/TPR/AT-HM) for one row-group (a split, or a
    split's domain), multirow-labelled with ``group_label`` on the first."""
    rows = []
    for i, m in enumerate(METRICS):
        cells = []
        for det in detectors:
            v = values.get((det, key))
            if v is None:
                cells.append("--")
                continue
            s = f"{v[m]:.3f}"
            if best.get((key, m)) is not None and s == f"{best[(key, m)]:.3f}":
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        group = f"\\multirow{{3}}{{*}}{{{group_label}}}" if i == 0 else ""
        rows.append(f"{group} & {METRIC_LABEL[m]} $\\uparrow$ & "
                    + " & ".join(cells) + r" \\")
    return rows


def _transposed_tabular(detectors, columns, values, best) -> tuple[list[str], bool, str]:
    """Rows = (split, metric) triples (multirow-grouped by split), columns =
    detectors -- a portrait table only ``2 + len(detectors)`` columns wide
    instead of ``1 + 3 * len(columns)``, so the full 5-group table stays
    readable without rotating the page. Needs multirow + graphicx
    (\\rotatebox), both already in the paper's preamble."""
    header = _rotated_header(detectors)
    rows = []
    for split in columns:
        rows += _metric_rows(detectors, split, values, best, _sd(split))
        rows.append(DOUBLE_RULE)  # every group here is its own dataset
    rows.pop()  # no rule after the last group

    col_spec = "ll" + " c" * len(detectors)
    tabular = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}%",
    ]
    bold_note = (r"\textbf{Bold} column headers are our contributions "
                 r"(LogJudge, FusionJudge); \textbf{bold} values are the "
                 r"best detector per row")
    return tabular, True, bold_note


#: splits whose parquet has a real (>1-value) ``domain`` column, from
#: run_best_zeroshot_domains.py -- MG-AI-Sci's ten AI-Scientist-source
#: "domains" are excluded, see build_domains_table()'s docstring.
DOMAIN_SPLITS = ("gb_orig", "mg_gb", "mg_sci")
DOMAIN_DISPLAY = {
    "essay": "Essay", "reuter": "Reuter", "wp": "WP",
    "computer_science": "CS", "mathematics": "Math", "sociology": "Soc",
}


def build_domains_table(csv_path: Path, domain_csv_path: Path, fpr: float) -> str:
    """The domain-augmented table: every :data:`DOMAIN_SPLITS` split gets a
    row per domain plus its own pooled-mean row (from ``csv_path``, the same
    split-level file every other table variant reads); MG-AI-Sci and the
    overall pooled Mean are single rows, same as :func:`_transposed_tabular`.
    Uses ``longtable`` since this easily runs to 40+ rows."""
    df = pd.read_csv(csv_path)
    df = df[df["fpr"] == fpr]
    scell = {(r["detector"], r["split"]): r for _, r in df.iterrows()}

    ddf = pd.read_csv(domain_csv_path)
    ddf = ddf[(ddf["fpr"] == fpr) & (ddf["split"].isin(DOMAIN_SPLITS))]
    dcell = {(r["detector"], r["split"], r["domain"]): r for _, r in ddf.iterrows()}
    domains_of = {s: sorted(ddf[ddf["split"] == s]["domain"].unique())
                  for s in DOMAIN_SPLITS}

    def _m(r) -> dict[str, float] | None:
        if r is None or pd.isna(r["auroc"]) or pd.isna(r["tpr_at_fpr"]):
            return None
        return {"auroc": r["auroc"], "tpr": r["tpr_at_fpr"],
                "hm": goodness(r["auroc"], r["tpr_at_fpr"], "hmean")}

    detectors = sorted(df["detector"].unique(),
                        key=lambda d: -(_m(scell.get((d, "mean"))) or {"hm": -1.0})["hm"])

    # (key, label, starts_a_new_dataset) -- a dataset's own domains + its
    # trailing "/ Mean" row share a single \midrule; the boundary INTO the
    # next dataset (or into MG-AI-Sci / the overall Mean) gets DOUBLE_RULE.
    groups: list[tuple[object, str, bool]] = []
    for split in DOMAIN_SPLITS:
        for j, dom in enumerate(domains_of[split]):
            groups.append(((split, dom),
                            f"{_sd(split)} / {DOMAIN_DISPLAY.get(dom, dom)}", j == 0))
        groups.append((split, f"{_sd(split)} / Mean", False))
    groups.append(("mg_aisci", _sd("mg_aisci"), True))
    groups.append(("mean", _sd("mean"), True))

    values = {
        (det, key): (_m(dcell.get((det, *key))) if isinstance(key, tuple)
                     else _m(scell.get((det, key))))
        for det in detectors for key, _, _ in groups
    }
    best = {(key, m): (max((values[(det, key)][m] for det in detectors
                           if values[(det, key)] is not None), default=None))
            for key, _, _ in groups for m in METRICS}

    header = _rotated_header(detectors)
    rows = []
    for i, (key, label, _) in enumerate(groups):
        rows += _metric_rows(detectors, key, values, best, label)
        if i < len(groups) - 1:
            rows.append(DOUBLE_RULE if groups[i + 1][2] else r"\midrule")

    n_dcols = 2 + len(detectors)
    col_spec = "ll" + " c" * len(detectors)
    caption = (
        f"General Results by domain: the three Manugram splits with a domain "
        f"breakdown (GB-orig, MG-GB: essay/reuter/wp; MG-Sci: computer "
        f"science/mathematics/sociology), each domain plus that split's own "
        f"pooled mean, then MG-AI-Sci (no breakdown -- its ten AI-Scientist-"
        f"source domains run 8--40 rows, several under 10 human examples, too "
        f"small for a per-domain reading) and the overall pooled Mean. Same "
        f"detectors, metrics, and AT-HM convention as "
        f"Table~\\ref{{tab:general-results}} and "
        f"Figure~\\ref{{fig:general-results-scatter}}; higher is better "
        f"throughout. Double rules separate datasets. \\textbf{{Bold}} column "
        f"headers are the LLM-judge family (LogJudge readout and "
        f"parsed-answer baseline), \\textbf{{bold}} values are the best "
        f"detector for that row."
    )
    lines = [
        f"\\begin{{longtable}}{{{col_spec}}}",
        f"\\caption{{{caption}}}\\label{{tab:general-results-domains}} \\\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        f"\\multicolumn{{{n_dcols}}}{{l}}{{\\textit{{Table~\\ref{{tab:general-results-domains}} "
        f"continued}}}} \\\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        f"\\multicolumn{{{n_dcols}}}{{r}}{{\\textit{{continued on next page}}}} \\\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
        *rows,
        r"\end{longtable}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--skip-mean", action="store_true",
                        help="drop the pooled-mean column group -- the full "
                             "5-group table gets too cramped once resized to "
                             "\\linewidth")
    group.add_argument("--only-mean", action="store_true",
                        help="keep ONLY the pooled-mean column group -- a "
                             "compact 1-group table (still ranked/computed "
                             "the same way)")
    group.add_argument("--columns", nargs="+", metavar="SPLIT",
                        choices=[*CORE_SPLITS, "mean"],
                        help="exact column-group list and order (e.g. "
                             "--columns gb_orig mean), overriding --skip-"
                             "mean/--only-mean; the pooled-mean group prints "
                             "as \"All Benchmark Mean\" here")
    ap.add_argument("--appendix", action="store_true",
                     help="transpose the table (rows = split x metric, "
                          "columns = detector) instead of squeezing 16 "
                          "columns into \\linewidth -- for the full 5-group "
                          "table, no extra package needed")
    ap.add_argument("--domains", action="store_true",
                     help="per-domain breakdown for GB-orig/MG-GB/MG-Sci plus "
                          "their pooled means (needs "
                          "run_best_zeroshot_domains.py's CSV first); a long "
                          "table, uses longtable (needs "
                          "\\usepackage{longtable})")
    ap.add_argument("--domain-csv", type=Path, default=DEFAULT_DOMAIN_CSV)
    args = ap.parse_args()

    if args.domains:
        tex = build_domains_table(args.csv, args.domain_csv, args.fpr)
    else:
        tex = build(args.csv, args.fpr, args.skip_mean, args.only_mean, args.appendix,
                    columns_override=args.columns)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
