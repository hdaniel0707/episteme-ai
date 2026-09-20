r"""Fill the Manugram-AI-Scientist transfer LaTeX table from ``run_aisci_
transfer.py``'s ``performance.csv`` -- the companion table to ``plot_aisci_
transfer.py``'s figure.

Rows = the same five :data:`plot_aisci_transfer.ALGORITHMS` the figure plots
(two trained detectors, three zero-shot picks), in that fixed order. Columns
= the same three :data:`plot_aisci_transfer.SCENARIOS` (which corpus a
trained detector was fit on), each split into AUROC / TPR@<fpr>%FPR / AT-HM.
A zero-shot detector's three column values are identical -- it has no
training corpus and MG-AI-Sci is the one fixed eval target throughout, see
``run_aisci_transfer.py``'s module docstring. Reuses that module's own
``ALGORITHMS`` / ``SCENARIOS`` / ``SCENARIO_LABEL`` / ``_points`` so every
cell is the exact same (algorithm, scenario) reading the figure plots that
point at -- the table and the figure can never disagree.

AT-HM is ``goodness(auroc, tpr, alpha=fpr)`` computed from that cell's own
AUROC/TPR pair, the same convention ``build_general_results_table.py``'s
Mean column and ``build_sentinel_scenarios_table.py`` use.

Writes a standalone ``\begin{table}...\end{table}`` block (booktabs style,
same as Table~\ref{tab:general-results}) to a .txt file for manual
copy-paste into the .tex -- this script does not touch the .tex itself.

Run:
    uv run python epai/ai_detection/trainv2/build_aisci_transfer_table.py \
        results/trainv2/aisci_transfer_<timestamp>/performance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from epai.ai_detection.trainv2.core import goodness
from epai.ai_detection.trainv2.detector_names import display_detector, is_judge_family
from epai.ai_detection.trainv2.plot_aisci_transfer import ALGORITHMS, SCENARIO_LABEL, SCENARIOS, _points

METRICS = ("auroc", "tpr", "hm")


def _tex_escape(name: str) -> str:
    return name.replace("_", r"\_").replace("·", r"$\cdot$")


def build(csv_path: Path, fpr: float) -> str:
    df = pd.read_csv(csv_path)
    points = _points(df, fpr)

    values: dict[tuple[str, str], dict[str, float] | None] = {}
    for algo in ALGORITHMS:
        for scen in SCENARIOS:
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
        for scen in SCENARIOS for m in METRICS
    }

    metric_label = {"auroc": "AUROC", "tpr": f"TPR@{fpr * 100:g}\\%FPR", "hm": "AT-HM"}
    n_cols = len(SCENARIOS)
    header1 = " & " + " & ".join(
        f"\\multicolumn{{3}}{{c}}{{{SCENARIO_LABEL[s]}}}" for s in SCENARIOS) + r" \\"
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(n_cols))
    metric_labels = " & ".join(f"{metric_label[m]} $\\uparrow$" for m in METRICS)
    header2 = "Detector & " + " & ".join([metric_labels] * n_cols) + r" \\"

    rows = []
    for algo in ALGORITHMS:
        cells = []
        for scen in SCENARIOS:
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
    caption = (
        f"Manugram-AI-Scientist transfer: the five detectors and three training-"
        f"corpus scenarios of Figure~\\ref{{fig:aisci-transfer}}, oriented AUROC and "
        f"oracle TPR@{fpr * 100:g}\\%FPR, and their AT-HM (Section~\\ref{{sec:method_metric}}); "
        f"higher is better throughout. Every cell is the same (detector, scenario) "
        f"reading Figure~\\ref{{fig:aisci-transfer}} plots for that point. A zero-shot "
        f"detector's three columns are identical by construction (one fixed eval "
        f"target, no training corpus). \\textbf{{Bold}} row labels are our "
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
        r"\label{tab:aisci-transfer}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="run_aisci_transfer.py's performance.csv")
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: aisci_transfer_table.txt next to the CSV")
    args = ap.parse_args()
    out = args.out or (args.csv.parent / "aisci_transfer_table.txt")

    tex = build(args.csv, args.fpr)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
