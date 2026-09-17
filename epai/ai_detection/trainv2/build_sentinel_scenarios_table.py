r"""Fill the Training-based comparison LaTeX table from ``run_sentinel_
comparison.py``'s ``performance.csv`` -- the companion table to
``plot_sentinel_scenarios.py``'s figure (Section~\ref{sec:results_training_
comparison} "Training-based comparison", ``latex/ICLR_AIDetect/
iclr2027_conference.tex``, Fig.~\ref{fig:sentinel_scenarios}).

Rows = the same four :data:`plot_sentinel_scenarios.ALGORITHMS` the figure
plots (two trained detectors, two zero-shot picks), in that fixed order.
Columns = the same four :data:`plot_sentinel_scenarios.SCENARIOS`, in the
figure's degradation-path order (In-Domain -> Near-OOD -> Far-OOD Sci ->
Far-OOD GB), each split into AUROC / TPR@<fpr>%FPR / AT-HM. Reuses that
module's own ``ALGORITHMS`` / ``SCENARIOS`` / ``SCENARIO_LABEL`` /
``_points`` so every cell is the exact same (algorithm, scenario) mean the
figure plots that point at -- the table and the figure can never disagree.

AT-HM is ``goodness(auroc_mean, tpr_mean, alpha=fpr)`` computed from that
cell's own (already-averaged) AUROC/TPR pair -- not the mean of the
per-row AT-HM values in the CSV -- the same convention
``build_general_results_table.py``'s Mean column uses, so the number here
matches the position the figure actually draws that point at.

Writes a standalone ``\begin{table}...\end{table}`` block (booktabs style,
same as Table~\ref{tab:general-results}) to a .txt file for manual
copy-paste into the .tex -- this script does not touch the .tex itself.

Run:
    uv run python epai/ai_detection/trainv2/build_sentinel_scenarios_table.py \
        results/trainv2/sentinel_comparison_<timestamp>/performance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from epai.ai_detection.trainv2.core import goodness
from epai.ai_detection.trainv2.detector_names import display_detector
from epai.ai_detection.trainv2.plot_sentinel_scenarios import (
    ALGORITHMS,
    SCENARIO_LABEL,
    SCENARIOS,
    _points,
)

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
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")

    col_spec = "l" + " ccc" * n_cols
    caption = (
        f"Training-based comparison: the four detectors and four scenarios of "
        f"Figure~\\ref{{fig:sentinel_scenarios}}, oriented AUROC and oracle "
        f"TPR@{fpr * 100:g}\\%FPR, and their AT-HM (Section~\\ref{{sec:method_metric}}); "
        f"higher is better throughout. Every AT-HM value is computed from that "
        f"cell's own (already-averaged) AUROC and TPR, matching the point "
        f"Figure~\\ref{{fig:sentinel_scenarios}} plots for that (detector, scenario) "
        f"pair. \\textbf{{Bold}} marks the best value in each column."
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
        r"\label{tab:sentinel-scenarios}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="run_sentinel_comparison.py's performance.csv")
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: sentinel_scenarios_table.txt next to the CSV")
    args = ap.parse_args()
    out = args.out or (args.csv.parent / "sentinel_scenarios_table.txt")

    tex = build(args.csv, args.fpr)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
