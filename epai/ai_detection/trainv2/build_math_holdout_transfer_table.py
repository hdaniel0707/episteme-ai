r"""Fill the MG-Sci domain-holdout transfer LaTeX table from ``run_math_
holdout_transfer.py``'s ``performance.csv`` -- the companion table to
``plot_math_holdout_transfer.py``'s figure. A direct structural copy of
``build_aisci_transfer_table.py`` with the scenario labels swapped.

Rows = the same five :data:`plot_math_holdout_transfer.ALGORITHMS` the
figure plots (two trained detectors, three zero-shot picks), in that fixed
order. Columns = the three scenarios :func:`plot_math_holdout_transfer.
scenario_layout` derives from the CSV's own data (which MG-Sci domain was
held out for this run, plus MG-GB and GB-orig), each split into AUROC /
TPR@<fpr>%FPR / AT-HM. A zero-shot detector's three column values are
identical -- it has no training corpus and the held-out domain's rows are
the one fixed eval target throughout, see ``run_math_holdout_transfer.py``'s
module docstring. Reuses that module's own ``ALGORITHMS`` / ``scenario_
layout`` / ``_points`` so every cell is the exact same (algorithm, scenario)
reading the figure plots that point at.

AT-HM is ``goodness(auroc, tpr, alpha=fpr)`` computed from that cell's own
AUROC/TPR pair, the same convention every other trainv2 results table uses.

Writes a standalone ``\begin{table}...\end{table}`` block (booktabs style,
same as Table~\ref{tab:general-results}) to a .txt file for manual
copy-paste into the .tex -- this script does not touch the .tex itself.

Run:
    uv run python epai/ai_detection/trainv2/build_math_holdout_transfer_table.py \
        results/trainv2/math_holdout_transfer_<code>_<timestamp>/performance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from epai.ai_detection.trainv2.core import goodness
from epai.ai_detection.trainv2.detector_names import display_detector, is_judge_family
from epai.ai_detection.trainv2.plot_math_holdout_transfer import (
    _CODE_READABLE,
    ALGORITHMS,
    _points,
    scenario_layout,
)

METRICS = ("auroc", "tpr", "hm")


def _tex_escape(name: str) -> str:
    return name.replace("_", r"\_").replace("·", r"$\cdot$")


def build(csv_path: Path, fpr: float) -> str:
    df = pd.read_csv(csv_path)
    scenarios, scenario_label, _ = scenario_layout(df)
    ex_key = scenarios[0]
    code = ex_key.removeprefix("mg_sci_ex_")
    points = _points(df, fpr, scenarios)

    values: dict[tuple[str, str], dict[str, float] | None] = {}
    for algo in ALGORITHMS:
        for scen in scenarios:
            p = points.get((algo, scen))
            if p is None:
                values[(algo, scen)] = None
                continue
            auroc, tpr = p
            values[(algo, scen)] = {
                "auroc": auroc, "tpr": tpr, "hm": goodness(auroc, tpr, alpha=fpr)}

    best = {
        (scen, m): max((values[(a, scen)][m] for a in ALGORITHMS
                        if values[(a, scen)] is not None), default=None)
        for scen in scenarios for m in METRICS
    }

    metric_label = {"auroc": "AUROC", "tpr": f"TPR@{fpr * 100:g}\\%FPR", "hm": "AT-HM"}
    n_cols = len(scenarios)
    header1 = " & " + " & ".join(
        f"\\multicolumn{{3}}{{c}}{{{scenario_label[s]}}}" for s in scenarios) + r" \\"
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(n_cols))
    metric_labels = " & ".join(f"{metric_label[m]} $\\uparrow$" for m in METRICS)
    header2 = "Detector & " + " & ".join([metric_labels] * n_cols) + r" \\"

    rows = []
    for algo in ALGORITHMS:
        cells = []
        for scen in scenarios:
            v = values[(algo, scen)]
            for m in METRICS:
                if v is None:
                    cells.append("--")
                    continue
                s = f"{v[m]:.3f}"
                if best[(scen, m)] is not None and s == f"{best[(scen, m)]:.3f}":
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
        label = f"\\texttt{{{_tex_escape(display_detector(algo))}}}"
        if is_judge_family(algo):
            label = f"\\textbf{{{label}}}"
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")

    col_spec = "l" + " ccc" * n_cols
    label_ref = f"math-holdout-transfer-{code}"
    fig_ref = f"math-holdout-transfer-{code}"
    readable = _CODE_READABLE.get(code, code)
    caption = (
        f"MG-Sci domain-holdout transfer ({readable} held out): the five "
        f"detectors and three training-corpus scenarios of Figure~\\ref{{fig:{fig_ref}}}, "
        f"oriented AUROC and oracle TPR@{fpr * 100:g}\\%FPR, and their AT-HM (Section~"
        f"\\ref{{sec:method_metric}}); higher is better throughout. Every cell is the "
        f"same (detector, scenario) reading Figure~\\ref{{fig:{fig_ref}}} plots for that "
        f"point. A zero-shot detector's three columns are identical by construction (one "
        f"fixed eval target, no training corpus). \\textbf{{Bold}} row labels are our "
        f"contributions (LogJudge, FusionJudge); \\textbf{{bold}} values are the best "
        f"per column."
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\resizebox{\linewidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header1,
        cmidrules,
        header2,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}%",
        "}",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label_ref}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="run_math_holdout_transfer.py's performance.csv")
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: math_holdout_transfer_table.txt next to the CSV")
    args = ap.parse_args()
    out = args.out or (args.csv.parent / "math_holdout_transfer_table.txt")

    tex = build(args.csv, args.fpr)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
