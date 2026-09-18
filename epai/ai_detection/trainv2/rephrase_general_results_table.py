"""Fill the Rephrasing-section LaTeX table from
``results/trainv2/rephrase_best_zeroshot_performance/performance.csv``.

The rephrase-task sibling of ``build_general_results_table.py``: same
detectors (``run_best_zeroshot_performance.py``'s hand-picked list, minus the
gemma judge -- null on the rephrase rows, see
``rephrase_best_zeroshot_performance.py``), same ranking rule (pooled-mean
AT-HM) and cell format, same bold rules (best value per column, our own
contributions' -- LogJudge, FusionJudge -- row labels bolded) -- just over
MG-GB / MG-Sci (rephrase counted as
human) instead of the four Manugram core splits, reusing
``build_general_results_table.py``'s ``_plain_tabular`` renderer directly
rather than re-implementing it.

Writes a standalone ``\\begin{table}...\\end{table}`` block to a .txt file
for manual copy-paste into the .tex -- this script does not touch the .tex
itself.

Run:
    uv run python epai/ai_detection/trainv2/rephrase_general_results_table.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from epai.ai_detection.trainv2.build_general_results_table import _plain_tabular
from epai.ai_detection.trainv2.core import goodness
from epai.ai_detection.trainv2.detector_names import display_detector as _display_detector
from epai.ai_detection.trainv2.detector_names import is_judge_family
from epai.ai_detection.trainv2.rephrase_zeroshot_performance import SPLITS as REPH_SPLITS

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CSV = (BASE_DIR / "results" / "trainv2" / "rephrase_best_zeroshot_performance"
               / "performance.csv")
DEFAULT_OUT = (BASE_DIR / "results" / "trainv2" / "rephrase_best_zeroshot_performance"
               / "rephrase_results_table.txt")

METRICS = ("auroc", "tpr", "hm")


def build(csv_path: Path, fpr: float, skip_mean: bool) -> str:
    df = pd.read_csv(csv_path)
    df = df[df["fpr"] == fpr]
    columns = list(REPH_SPLITS) if skip_mean else [*REPH_SPLITS, "mean"]

    cell = {(r["detector"], r["split"]): r for _, r in df.iterrows()}

    def metrics(det: str, split: str) -> dict[str, float] | None:
        r = cell.get((det, split))
        if r is None or pd.isna(r["auroc"]) or pd.isna(r["tpr_at_fpr"]):
            return None
        return {"auroc": r["auroc"], "tpr": r["tpr_at_fpr"],
                "hm": goodness(r["auroc"], r["tpr_at_fpr"], "hmean")}

    detectors = sorted(df["detector"].unique(),
                        key=lambda d: -((metrics(d, "mean") or {"hm": -1.0})["hm"]))

    values = {(det, split): metrics(det, split)
              for det in detectors for split in columns}
    best = {(split, m): max((values[(det, split)][m] for det in detectors
                             if values[(det, split)] is not None), default=None)
            for split in columns for m in METRICS}

    tabular, resize, bold_note = _plain_tabular(detectors, columns, values, best)

    caption = (
        f"Rephrasing results: zero-shot and LogJudge detectors on the "
        f"Manugram-Rephrase corpora (MG-GB, MG-Sci; AI and AI-rephrased vs. "
        f"human and human-rephrased, Section~\\ref{{sec:results_rephrasing}}), "
        f"oriented AUROC and oracle TPR@{fpr * 100:g}\\%FPR, and their AT-HM "
        f"(Section~\\ref{{sec:method_metric}}); higher is better throughout. "
        f"Same detectors, metrics, and AT-HM convention as "
        f"Table~\\ref{{tab:general-results}} and "
        f"Figure~\\ref{{fig:general-results-scatter}} -- the gemma-4-31b "
        f"judge is dropped here (null on the rephrase rows). Every AT-HM "
        f"value, including the Mean column's, is computed from that "
        f"column's own (pooled, for Mean) AUROC and TPR, matching "
        f"Figure~\\ref{{fig:rephrasing-scatter}}. Ranked by pooled-mean "
        f"AT-HM; \\textbf{{bold}} {bold_note}."
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        *([r"\resizebox{\linewidth}{!}{%"] if resize else []),
        *tabular,
        *(["}"] if resize else []),
        f"\\caption{{{caption}}}",
        r"\label{tab:rephrase-results}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-mean", action="store_true",
                     help="drop the pooled-mean column group")
    args = ap.parse_args()

    tex = build(args.csv, args.fpr, args.skip_mean)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
