"""Camera-ready figure and LaTeX table from the CSVs of ``compute_conferences.py``.

Reads only ``results/conferences/<tag>/{calibration,year_flag_rates,
year_score_hist}.csv`` (never the parquet) and writes to the same folder:

  - ``conference_flag_rate.{pdf,png}``, two panels:
      (a) share of abstracts flagged as machine per year at the MG-Sci 1%-FPR
          threshold, one line per detector, Wilson 95% CI bands; dashed line at
          the nominal 1% false-positive rate; dotted lines at model releases.
      (b) FusionJudge P(human) density, first vs. last year.
  - ``conference_flag_rate_table.tex``: per year n and flagged % for each
    detector plus FusionJudge's mean P(human), then the MG-Sci calibration rows
    (FPR on human, TPR on machine abstracts). Caption above the table, as in
    the paper.

    .venv/bin/python epai/scripts/plot_conferences.py --conference ICLR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.ticker
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from epai.ai_detection.trainv2.calibrate_common import _FPR_COLOR
from epai.ai_detection.trainv2.plots import _INK, _MUTED

RESULTS_DIR = Path("results/conferences")

# ---------------------------------------------------------------- design knobs
FIGSIZE = (8.0, 2.8)
FONT_TITLE = 10
FONT_LABEL = 9.5
FONT_TICK = 8.5
FONT_NOTE = 8.5
FONT_RELEASE = 7
GRID_COLOR = "#e5e4e0"
DETECTORS = ["BaseJudge", "LogJudge", "FusionJudge"]
# the paper's readout colors (plot_iclr_readout_zoom.READOUT_COLOR); markers are
# the secondary encoding, since BaseJudge's steel blue is low-chroma.
COLOR = {"BaseJudge": "#4f77a5", "LogJudge": "#c8571b", "FusionJudge": "#4a3aa7"}
MARKER = {"BaseJudge": "s", "LogJudge": "^", "FusionJudge": "o"}
# vertical offset (points) of the direct label at the last year, so the two
# nearly coinciding judge lines get separate labels.
LABEL_DY = {"BaseJudge": 0, "LogJudge": -7, "FusionJudge": 7}
DIST_DETECTOR = "FusionJudge"
DIST_XMAX = 1.0
# Model releases at their calendar date, reading each year tick as 1 January
# (ICLR <year> abstracts are finalized in early <year>).
# (release date as a decimal year, label, label side of the line):
# InstructGPT became the OpenAI API default on 27 Jan 2022; ChatGPT was released
# on 30 Nov 2022, GPT-4o on 13 May 2024, Claude 3.5 Sonnet on 20 Jun 2024.
RELEASES = [
    (2022 + 26 / 365, "InstructGPT", "right"),
    (2022 + 333 / 365, "ChatGPT", "right"),
    (2024 + 133 / 366, "GPT-4o", "left"),
    (2024 + 171 / 366, "Claude 3.5\nSonnet", "right"),
]
# (b) first vs. last year: a light and a dark step of the dataviz blue ramp.
FIRST_COLOR, LAST_COLOR = "#6da7ec", "#104281"


def _style(ax) -> None:
    ax.tick_params(labelsize=FONT_TICK, colors=_INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(_MUTED)
    ax.spines["bottom"].set_color(_MUTED)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def _rate_panel(ax, rates: pd.DataFrame, fpr_target: float) -> None:
    years = sorted(rates["conference_year"].unique())
    # all detectors score the same papers, so they share one x per year; the
    # CIs are bands (not bars) and the markers hollow so coinciding points stay readable.
    for det in DETECTORS:
        r = rates[rates["detector"] == det].sort_values("conference_year")
        x = r["conference_year"].to_numpy()
        y = r["flag_rate"].to_numpy()
        ax.fill_between(x, r["flag_rate_ci_lo"], r["flag_rate_ci_hi"], color=COLOR[det],
                        alpha=0.15, linewidth=0)
        ax.plot(x, y, color=COLOR[det], marker=MARKER[det], ms=5.5, mfc="white", mew=1.4,
                lw=1.6, label=det)
        ax.annotate(det, (x[-1], y[-1]), xytext=(7, LABEL_DY[det]), textcoords="offset points",
                    va="center", fontsize=FONT_NOTE, color=_INK)
    ax.axhline(fpr_target, color=_FPR_COLOR, ls="--", lw=1.1)
    ax.text(years[-1] + 0.3, fpr_target, "1% FPR", color=_INK, fontsize=FONT_NOTE,
            va="bottom", ha="left")
    for x_release, label, side in RELEASES:
        ax.axvline(x_release, color=_MUTED, ls=":", lw=1.0, zorder=0)
        x_text = x_release + (-0.03 if side == "left" else 0.03)
        ax.text(x_text, 0.99, label, transform=ax.get_xaxis_transform(), fontsize=FONT_RELEASE,
                color=_INK, rotation=90, va="top", ha="right" if side == "left" else "left",
                linespacing=1.0)
    ax.set_xticks(years)
    ax.set_xlim(years[0] - 0.4, years[-1] + 1.3)
    ax.set_ylim(0, 1.2 * rates["flag_rate_ci_hi"].max())  # headroom for the release labels
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("conference year", fontsize=FONT_LABEL)
    ax.set_ylabel("abstracts flagged as machine", fontsize=FONT_LABEL)
    ax.set_title("(a) Flag rate at the MG-Sci 1%-FPR threshold", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_NOTE, frameon=False, loc="upper left")


def _dist_panel(ax, hist: pd.DataFrame) -> None:
    h = hist[hist["detector"] == DIST_DETECTOR]
    years = sorted(h["conference_year"].unique())
    for y, color in ((years[0], FIRST_COLOR), (years[-1], LAST_COLOR)):
        b = h[h["conference_year"] == y].sort_values("bin_lo")
        mids = ((b["bin_lo"] + b["bin_hi"]) / 2).to_numpy()
        d = b["density"].to_numpy()
        ax.fill_between(mids, d, step="mid", color=color, alpha=0.2, linewidth=0)
        ax.step(mids, d, where="mid", color=color, lw=1.5, label=str(y))
    ax.set_xlim(0, DIST_XMAX)
    ax.set_ylim(0, None)
    ax.set_xlabel(f"{DIST_DETECTOR}  P(human)", fontsize=FONT_LABEL)
    ax.set_ylabel("density", fontsize=FONT_LABEL)
    ax.set_title(f"(b) {DIST_DETECTOR} score, {years[0]} vs. {years[-1]}", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_NOTE, frameon=False, loc="upper center")


def _pct(v: float) -> str:
    return f"{100 * v:.1f}"


def _n(v) -> str:
    return f"{int(v):,}".replace(",", "{,}")


def _table(rates: pd.DataFrame, calib: pd.DataFrame, tag: str) -> str:
    """LaTeX table in the paper's style (caption above, booktabs)."""
    wide = rates.pivot(index="conference_year", columns="detector", values="flag_rate")
    n = rates.groupby("conference_year")["n"].first()
    fmean = rates[rates["detector"] == "FusionJudge"].set_index("conference_year")["mean"]
    c = calib.set_index("detector")
    # a readout whose MG-Sci FPR (as printed) falls short of the target: its
    # threshold stops before a score value tied by many human abstracts.
    tied = [d for d in DETECTORS if _pct(c.loc[d, "fpr_human"]) != _pct(c.loc[d, "fpr_target"])]
    note = "".join(
        rf" $^\dagger${d}'s scores take only {c.loc[d, 'n_distinct']} values on MG-Sci, so its FPR "
        rf"cannot reach exactly 1\%: its threshold ({c.loc[d, 'threshold']:.2f}) flags "
        rf"{_pct(c.loc[d, 'fpr_human'])}\% of the human abstracts, and the next value "
        rf"({c.loc[d, 'next_value']:.2f}) is shared by {c.loc[d, 'n_human_at_next']} human abstracts and "
        rf"would flag {_pct(c.loc[d, 'fpr_human_at_next'])}\%, over budget "
        r"(Section~\ref{sec:results_readout}, Appendix~\ref{apx:readout_zoom})."
        for d in tied
    )
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + tag + r" abstracts flagged as machine per conference year, at each detector's "
        r"1\%-FPR threshold set on the MG-Sci human abstracts (all published by 2020); "
        r"all detectors use the \texttt{gpt-5.6-luna} judge. The last column is FusionJudge's mean "
        r"$P(\text{human})$. The bottom rows apply the same thresholds to MG-Sci itself: the share "
        r"of human abstracts flagged (FPR) and of machine abstracts flagged (TPR)." + note + "}",
        r"\label{tab:iclr-flag-rates}",
        r"\footnotesize",
        r"\begin{tabular}{lrcccc}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Flagged as machine (\%)} & \\",
        r"\cmidrule(lr){3-5}",
        r"Year & $n$ & BaseJudge & LogJudge & FusionJudge & Mean $P(\text{human})$ \\",
        r"\midrule",
    ]
    for y in wide.index:
        cells = [_pct(wide.loc[y, d]) for d in DETECTORS]
        lines.append(f"{y} & {_n(n[y])} & " + " & ".join(cells) + f" & {fmean[y]:.3f} \\\\")
    lines += [
        r"\midrule",
        r"MG-Sci human (FPR) & " + _n(c["n_human"].iloc[0]) + " & "
        + " & ".join(_pct(c.loc[d, "fpr_human"]) + (r"$^\dagger$" if d in tied else "") for d in DETECTORS)
        + r" & -- \\",
        r"MG-Sci machine (TPR) & " + _n(c["n_machine"].iloc[0]) + " & "
        + " & ".join(_pct(c.loc[d, "tpr_machine"]) for d in DETECTORS) + r" & -- \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conference", nargs="+", default=None, metavar="CONFERENCE_ID",
                    help="same selection as compute_conferences.py (default: all)")
    args = ap.parse_args()
    tag = "_".join(sorted(c.upper() for c in args.conference)).replace(" ", "-") if args.conference else "all"
    out_dir = RESULTS_DIR / tag

    rates = pd.read_csv(out_dir / "year_flag_rates.csv")
    hist = pd.read_csv(out_dir / "year_score_hist.csv")
    calib = pd.read_csv(out_dir / "calibration.csv")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1.15, 1]})
    _rate_panel(ax_a, rates, float(calib["fpr_target"].iloc[0]))
    _dist_panel(ax_b, hist)
    for ax in (ax_a, ax_b):
        _style(ax)
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        path = out_dir / f"conference_flag_rate{suffix}"
        fig.savefig(path, dpi=300)
        print(f"wrote {path}")
    plt.close(fig)

    path = out_dir / "conference_flag_rate_table.tex"
    path.write_text(_table(rates, calib, tag))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
