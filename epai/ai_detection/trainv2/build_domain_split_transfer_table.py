r"""Fill the MG-Sci domain-split transfer LaTeX table from ``run_domain_
split_transfer.py``'s ``performance.csv`` -- the companion table to
``plot_domain_split_transfer.py``'s slope-chart figure. Minimal by design
(AT-HM only, no AUROC/TPR breakdown -- those are in the results CSV):
this is the exact table the article uses, not a fuller reference table the
article picks a subset of, so there is only one design to keep in sync.

Rows = RoBERTa-Sentinel (the one trained detector the figure plots -- the
logistic-regression ablation is dropped from both, see that module's
docstring), THEN a ``\midrule``, THEN the four zero-shot picks (Binoculars,
Fast-DetectGPT, LogJudge, FusionJudge) sorted by decreasing AT-HM -- RADAR
is excluded here too, not just from the figure (see ``plot_domain_split_
transfer.ZS_ALGOS``'s comment: its raw AUROC on this eval set is below
chance, so its AT-HM floors to exactly 0.000 and carries no information) --
matching the row convention this paper's other results tables use
(``build_general_results_table.py``'s ``TABLE_LABEL`` / ``_table_label``,
reused here rather than redefined: plain names with a ``\citep`` for each
baseline, bold "(ours)" for LogJudge/FusionJudge). ``\faIcon{fire}``/
``\faIcon{snowflake}`` (fontawesome5, added to the article's preamble for
this table) mark trained vs. zero-shot at a glance -- matching the figure's
own flame-marker/snowflake markers (``plot_domain_split_transfer.
FLAME_MARKER``/``COLD_MARKER``).

Columns = the four scenarios (:data:`plot_domain_split_transfer.
SCENARIO_XLABEL`, the same short In-Domain/Near-OOD/Mid-OOD/Far-OOD names
the figure's x-axis uses -- NOT the verbose ``scenario_label`` form
``scenario_layout`` also returns, which is for a fuller table this one
isn't). A zero-shot detector's value is identical in every column -- it has
no training corpus and the fixed 30% eval slice is the one eval target
throughout, see ``run_domain_split_transfer.py``'s module docstring --
typed out in each of the four columns rather than merged into one spanning
cell (tried a ``\multicolumn`` version; reverted, the per-column repeat
read better here). "Best per column" bolding is computed independently per
column across both row shapes (each of RoBERTa-Sentinel's four own values,
and each zero-shot pick's one value re-compared against every column).

AT-HM is ``goodness(auroc, tpr, alpha=fpr)`` computed from that cell's own
AUROC/TPR pair, the same convention every other trainv2 results table uses.

Writes a standalone ``\begin{table}...\end{table}`` block (booktabs style,
same as Table~\ref{tab:general-results}) to a .txt file for manual
copy-paste into the .tex -- this script does not touch the .tex itself.

Run:
    uv run python epai/ai_detection/trainv2/build_domain_split_transfer_table.py \
        results/trainv2/domain_split_transfer_<code>_<timestamp>/performance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from epai.ai_detection.trainv2.build_general_results_table import _table_label
from epai.ai_detection.trainv2.core import goodness
from epai.ai_detection.trainv2.detector_names import is_judge_family
from epai.ai_detection.trainv2.plot_domain_split_transfer import (
    SCENARIO_XLABEL,
    TRAINED_ALGO,
    ZS_ALGOS,
    scenario_layout,
)

#: not in build_general_results_table.TABLE_LABEL (a zero-shot-only map) --
#: same citation key used everywhere else in the paper for this baseline.
TRAINED_ROW_LABEL = r"\faIcon{fire}~RoBERTa-Sentinel~\citep{chen_gpt-sentinel_2023}"
_TOL = 1e-9


def _hm(df: pd.DataFrame, fpr: float, detector: str, train_split: str) -> float | None:
    rows = df[(df["fpr"] == fpr) & (df["detector"] == detector)
             & (df["train_split"] == train_split)]
    if rows.empty:
        return None
    r = rows.iloc[0]
    return goodness(float(r["auroc"]), float(r["tpr_at_fpr"]), alpha=fpr)


def build(csv_path: Path, fpr: float) -> str:
    df = pd.read_csv(csv_path)
    scenarios, _ = scenario_layout(df)
    code = scenarios[0].removeprefix("mg_sci_in_")

    trained_vals = [_hm(df, fpr, TRAINED_ALGO, s) for s in scenarios]
    zs_vals = {a: _hm(df, fpr, a, scenarios[0]) for a in ZS_ALGOS}
    zs_sorted = sorted(ZS_ALGOS, key=lambda a: -(zs_vals[a] if zs_vals[a] is not None else -1.0))

    # Best-per-column across both row shapes: RoBERTa-Sentinel's own 4
    # values, plus each zero-shot pick's single value compared in all 4.
    best_per_col = [
        max([v for v in (trained_vals[i], *zs_vals.values()) if v is not None], default=None)
        for i in range(len(scenarios))
    ]

    def fmt(v: float | None, col_i: int) -> str:
        if v is None:
            return "--"
        s = f"{v:.3f}"
        b = best_per_col[col_i]
        return f"\\textbf{{{s}}}" if b is not None and abs(v - b) < _TOL else s

    trained_cells = " & ".join(fmt(v, i) for i, v in enumerate(trained_vals))
    rows = [f"{TRAINED_ROW_LABEL} & {trained_cells} \\\\", r"\midrule"]

    for algo in zs_sorted:
        v = zs_vals[algo]
        cells = " & ".join(fmt(v, i) for i in range(len(scenarios)))
        label = _table_label(algo)
        if is_judge_family(algo):
            label = f"\\textbf{{{label}}}"
        label = f"\\faIcon{{snowflake}}~{label}"
        rows.append(f"{label} & {cells} \\\\")

    header = "Detector & " + " & ".join(SCENARIO_XLABEL) + r" \\"
    label_ref = f"domain-split-transfer-{code}"
    subtitle = ("averaged over all 3 MG-Sci domains as the 70/30-split eval target"
               if code == "avg" else f"{code} 70/30-split for eval")
    caption = (
        f"AT-HM (Section~\\ref{{sec:method_metric}}) for Figure~\\ref{{fig:{label_ref}}}'s "
        f"four scenarios, {subtitle}; higher is better. \\faIcon{{fire}}~marks the trained "
        f"detector, \\faIcon{{snowflake}}~the zero-shot ones -- a zero-shot detector's four "
        f"columns are identical by construction, since it has no training set to shrink; "
        f"rows below the rule are ranked by decreasing AT-HM. \\textbf{{Bold}} row labels "
        f"are our contributions (LogJudge, FusionJudge); \\textbf{{bold}} values are the "
        f"best per column."
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\begin{{tabular}}{{l{'c' * len(scenarios)}}}",
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label_ref}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="run_domain_split_transfer.py's performance.csv")
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: domain_split_transfer_table.txt next to the CSV")
    args = ap.parse_args()
    out = args.out or (args.csv.parent / "domain_split_transfer_table.txt")

    tex = build(args.csv, args.fpr)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
