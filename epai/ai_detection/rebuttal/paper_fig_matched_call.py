"""Paper figure for the matched-call appendix (rebuttal A1/A2): ``matched_call``.

Reads the evaluation CSVs (``evaluate.py``) and the gpt-5.6-luna sidecars; writes
``overleaf/figs/matched_call.{pdf,png}``.

    (a-c) gpt-5.6-luna, matched call: the 6 most frequent written values with
          2 decimals, 4 decimals and no limit (share of all answers, 36,148 texts).
    (d)   gpt-5.6-luna, ROC up to 5% FPR (four-split mean): BaseJudge with
          default settings, the matched call at three precisions, A2 (expected
          value of the matched call's digit tokens) and LogJudge.
    (e)   the same for gemma-4-31b (humans_plus_third subset).

Run from the repo root:
    .venv/bin/python epai/ai_detection/rebuttal/paper_fig_matched_call.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from epai.ai_detection.rebuttal.readouts import parse_prob

ROOT = Path(__file__).resolve().parents[3]
RB = ROOT / "data/parquet/tmp/rebuttal"
EVAL = RB / "eval"
OUT = ROOT / "overleaf/figs"

FONT = 9
INK, MUTED = "#0b0b0b", "#8a8a86"
SPLITS = ["GB-orig", "MG-GB", "MG-Sci", "MG-AI-Sci"]
GRID = np.linspace(0, 0.05, 1001)

#: readout -> (legend label, colour, line style, width)
CURVES = {
    "paper_base": ("BaseJudge, default settings", "#4f77a5", "-", 1.6),
    "base_v3": ("BaseJudge, matched, 2 decimals", "#8a8a86", "-", 1.6),
    "base_hp4": ("BaseJudge, matched, 4 decimals", "#8a8a86", (0, (4, 2)), 1.2),
    "base_free": ("BaseJudge, matched, no limit", "#8a8a86", (0, (1, 1.5)), 1.2),
    "base_v3_ev": ("same call, digit tokens (2 dec.)", "#1baf7a", "-", 1.8),
    "paper_lj": ("LogJudge", "#c8571b", "-", 2.4),
}
PANELS = [("base_v3", "2 decimals"), ("base_hp4", "4 decimals"), ("base_free", "no limit")]


def _style(ax) -> None:
    ax.tick_params(labelsize=FONT - 0.5, colors=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)


def mean_roc(roc: pd.DataFrame, model: str, readout: str) -> np.ndarray:
    """Four-split mean TPR on GRID (each split's ROC linearly interpolated, so a
    tie group stays a straight segment)."""
    curves = []
    for split in SPLITS:
        g = roc[(roc.model == model) & (roc.readout == readout) & (roc.split == split)]
        g = g.groupby("fpr", as_index=False)["tpr"].max().sort_values("fpr")
        curves.append(np.interp(GRID, g.fpr.to_numpy(), g.tpr.to_numpy()))
    return np.mean(curves, axis=0)


def roc_panel(ax, roc: pd.DataFrame, model: str, title: str, legend: bool) -> None:
    for readout, (label, color, ls, lw) in CURVES.items():
        ax.plot(GRID * 100, mean_roc(roc, model, readout), color=color, ls=ls, lw=lw, label=label)
    ax.axvline(1, color=MUTED, ls=":", lw=1)
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.02, 1)
    ax.set_xlabel("FPR (%)", fontsize=FONT)
    ax.set_ylabel("TPR (four-split mean)", fontsize=FONT)
    ax.set_title(title, fontsize=FONT + 0.5, loc="left")
    if legend:
        ax.legend(fontsize=FONT - 1.5, frameon=False, loc="lower right")
    _style(ax)


def written_panel(ax, cond: str, title: str) -> None:
    df = pd.read_parquet(RB / f"gpt56luna/{cond}__full.parquet", columns=["answer"])
    vals = df.answer.map(lambda a: (a or "").strip())
    vals = vals[vals.map(parse_prob).notna()]
    share = vals.value_counts(normalize=True).head(6)[::-1] * 100
    ax.barh(range(len(share)), share.to_numpy(), color="#4f77a5", height=0.7)
    ax.set_yticks(range(len(share)))
    ax.set_yticklabels(share.index, fontsize=FONT - 0.5, family="monospace")
    ax.set_xlim(0, 36)
    ax.set_xlabel("share of answers (%)", fontsize=FONT)
    ax.set_title(title, fontsize=FONT + 0.5, loc="left")
    _style(ax)
    print(f"{cond}: {vals.nunique()} distinct written values, top 6 = "
          + ", ".join(f"{v} {s:.1f}%" for v, s in share[::-1].items()))


def main() -> None:
    roc = pd.read_csv(EVAL / "a1_roc.csv")
    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.45])
    axes = top.subplots(1, 3)
    for ax, (cond, name), letter in zip(axes, PANELS, "abc"):
        written_panel(ax, cond, f"({letter}) gpt-5.6-luna, {name}")
    a1, a2 = bottom.subplots(1, 2)
    roc_panel(a1, roc, "gpt56luna", "(d) gpt-5.6-luna", legend=False)
    roc_panel(a2, roc, "gemma4-31b", "(e) gemma-4-31b", legend=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"matched_call.{ext}", dpi=220)
    print(f"wrote {OUT}/matched_call.pdf/.png")
    for model in ("gpt56luna", "gemma4-31b"):
        at1 = {r: round(float(np.interp(0.01, GRID, mean_roc(roc, model, r))), 3) for r in CURVES}
        print(model, "mean ROC at 1% FPR (interpolated):", at1)


if __name__ == "__main__":
    main()
