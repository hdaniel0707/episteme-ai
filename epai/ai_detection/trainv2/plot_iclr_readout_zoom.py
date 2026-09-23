r"""ICLR 2027 paper, zoomed companion to Fig. fig:mechanism ("Why the readout
matters") -- shows WHERE the 1%-FPR threshold falls for the same judge under
its two readouts, which the linear 0-1 histograms of dist_judge / dist_logjudge
hide: there, LogJudge looks like two spikes at 0 and 1 because every score
below 0.025 lands in one bin.

Three panels, same judge, all four core splits pooled (the Fig. mechanism set):

    (a) BaseJudge, lower tail on a linear axis with fine bins: the stated
        probability sits on a few exact round values, and the 1%-FPR threshold
        lands ON one of them -- the human and machine texts tied there cannot
        be split by any threshold.
    (b) LogJudge, the same tail on a log10 P(human) axis: the scores keep
        spreading over many decades, and the threshold (~1e-4) falls inside a
        continuous region.
    (c) ROC zoomed to FPR <= 5%, one marker per ACHIEVABLE operating point
        (one per distinct score): a tie block is a long straight segment with
        no marker inside, so the parsed readout has to stop at the marker
        before 1% FPR; the log-probability readout has markers all the way.

Threshold rule: cd_core.roc_point_at_fpr, the one every calibrate_* script
and the paper use (largest achievable FPR <= 1%), so the printed TPRs should
match Fig. mechanism (gpt-5.6-luna pooled: 0.36 parsed, 0.62 LogJudge).

Also prints, and writes to stats.csv, the tie structure behind the picture:
distinct score values, and how many human / machine texts share the exact
threshold score.

Outputs (results/trainv2/iclr_readout_zoom/):
    readout_zoom_<judge>_<split>.{pdf,png}
    stats_<judge>_<split>.csv

Run:
    uv run python epai/ai_detection/trainv2/plot_iclr_readout_zoom.py
    uv run python epai/ai_detection/trainv2/plot_iclr_readout_zoom.py --judge gemma4_31b
    uv run python epai/ai_detection/trainv2/plot_iclr_readout_zoom.py --split mg_sci
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from epai.ai_detection.trainv2.calibrate_common import _HUMAN_COLOR, _MACHINE_COLOR, _FPR_COLOR, GEN_COL
from epai.ai_detection.trainv2.plots import _INK, _MUTED
from epai.ai_detection.trainv2.splits import DEFAULT as CORE_SPLITS
from epai.ai_detection.trainv2.splits import SPLITS
from epai.ai_detection.trainv2.splits import display as split_display
from epai.ai_detection.trainv2.splits import resolve as resolve_splits
from epai.utils.cd_core import auroc_score, roc_point_at_fpr

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
OUT_DIR = BASE_DIR / "results" / "trainv2" / "iclr_readout_zoom"
FPR = 0.01
HUMAN_LABEL = "human"

# readout -> column template; both are P(human), so LOW = machine.
# logjudge_ord2_* is the LogJudge column of Table main-results (Eq.
# logjudge_simple exactly); dist_logjudge was drawn from the identical-formula
# llm_judge_score_<judge>_v3_wlp.
READOUTS = {
    "BaseJudge": "llm_judge_score_{judge}_v3",
    "LogJudge": "logjudge_ord2_{judge}_v3",
}
# FusionJudge = mean of the two above; scored for gpt-5.6-luna only (Table
# main-results), so it is added only for that judge.
FUSION = {"gpt56luna": "fusionjudge_score"}
JUDGE_DISPLAY = {"gpt56luna": "gpt-5.6-luna", "gemma4_31b": "gemma-4-31b"}

# ---------------------------------------------------------------- design knobs
FIGSIZE = (9.0, 2.7)       # 1 x 3, without FusionJudge
FIGSIZE_2X2 = (8.0, 5.6)   # 2 x 2, with FusionJudge
FONT_TITLE = 10
FONT_LABEL = 9.5
FONT_TICK = 8.5
FONT_NOTE = 8.5
# linear-axis readouts: (x max, bin width) -- bins narrow enough that round
# values stay spikes. FusionJudge ~ BaseJudge / 2 in the machine tail.
LINEAR_ZOOM = {"BaseJudge": (0.30, 0.005), "FusionJudge": (0.15, 0.0025)}
LOG_XMIN = -8             # (b) log10 P(human) floor; scores below are clipped onto it
LOG_BINS = 64             # (b) bins over [LOG_XMIN, 0]
HIST_YMAX = 3e5           # (a)/(b) headroom above the bars for the legend
ROC_XMAX = 0.05           # (c) FPR zoom
READOUT_COLOR = {"BaseJudge": "#4f77a5", "LogJudge": "#c8571b", "FusionJudge": "#4a3aa7"}


def _columns(judge: str) -> dict[str, str]:
    cols = {name: t.format(judge=judge) for name, t in READOUTS.items()}
    if judge in FUSION:
        cols["FusionJudge"] = FUSION[judge]
    return cols


def _load(judge: str, split: str) -> pd.DataFrame:
    cols = list(_columns(judge).values())
    frames = [pd.read_parquet(path, columns=[GEN_COL, *cols])
              for _, path in resolve_splits([split], PARQUET_DIR)]
    return pd.concat(frames, ignore_index=True)


def _readout(df: pd.DataFrame, col: str) -> dict:
    """Scores, labels, the 1%-FPR operating point and its tie structure."""
    sub = df[[col, GEN_COL]].dropna(subset=[col])
    s = sub[col].to_numpy(dtype=float)                       # P(human)
    y = (sub[GEN_COL].astype(str) != HUMAN_LABEL).to_numpy().astype(int)
    oriented = -s                                            # higher = machine
    tpr, thr_or = roc_point_at_fpr(y, oriented, FPR)
    thr = -thr_or                                            # machine iff s <= thr
    fpr_c, tpr_c, _ = roc_curve(y, oriented, drop_intermediate=False)
    # the first operating point past thr: the tie block the threshold cannot enter
    nxt = int(np.flatnonzero(fpr_c > FPR)[0])
    at = s == thr if np.isfinite(thr) else np.zeros_like(s, dtype=bool)
    above = s > thr
    first_blocked = s[above].min() if above.any() else np.nan
    blocked = s == first_blocked
    return {
        "s": s, "y": y, "thr": thr, "tpr": tpr,
        "fpr_curve": fpr_c, "tpr_curve": tpr_c,
        "auroc": auroc_score(y, oriented),
        "n": len(s), "n_human": int((y == 0).sum()), "n_machine": int((y == 1).sum()),
        "n_distinct": int(np.unique(s).size),
        "n_distinct_below_h5": int(np.unique(s[s <= np.percentile(s[y == 0], 5)]).size),
        "n_human_below_floor": int(((s < 10.0 ** LOG_XMIN) & (y == 0)).sum()),
        "n_machine_below_floor": int(((s < 10.0 ** LOG_XMIN) & (y == 1)).sum()),
        "n_human_at_thr": int((at & (y == 0)).sum()),
        "n_machine_at_thr": int((at & (y == 1)).sum()),
        "next_score": float(first_blocked),
        "n_human_next": int((blocked & (y == 0)).sum()),
        "n_machine_next": int((blocked & (y == 1)).sum()),
        "next_fpr": float(fpr_c[nxt]), "next_tpr": float(tpr_c[nxt]),
    }


def _hist(ax, x, y, bins) -> None:
    for cls, color, label in ((0, _HUMAN_COLOR, "human"), (1, _MACHINE_COLOR, "machine")):
        ax.hist(x[y == cls], bins=bins, histtype="step", lw=1.2, color=color, label=label)
        ax.hist(x[y == cls], bins=bins, histtype="stepfilled", alpha=0.18, color=color)
    ax.set_yscale("log")
    ax.set_ylim(top=HIST_YMAX)
    ax.set_ylabel("texts", fontsize=FONT_LABEL)


def _thr_line(ax, x, text) -> None:
    ax.axvline(x, color=_FPR_COLOR, ls="--", lw=1.1, label=text)


def _style(ax) -> None:
    ax.tick_params(labelsize=FONT_TICK, colors=_INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(_MUTED)
    ax.spines["bottom"].set_color(_MUTED)


def _hist_panel(ax, name: str, r: dict, letter: str) -> None:
    """Lower tail of one readout: linear fine bins, or log10 for LogJudge."""
    thr_label = f"1% FPR: {r['thr']:.2g} (TPR {r['tpr']:.0%})"
    if name == "LogJudge":
        lx = np.log10(np.clip(r["s"], 10.0 ** LOG_XMIN, 1.0))
        _hist(ax, lx, r["y"], np.linspace(LOG_XMIN, 0, LOG_BINS + 1))
        _thr_line(ax, np.log10(max(r["thr"], 10.0 ** LOG_XMIN)), thr_label)
        ax.set_xlim(LOG_XMIN, 0)
        ax.set_xticks(range(LOG_XMIN, 1, 2))
        ax.set_xticklabels([rf"$\leq\!10^{{{LOG_XMIN}}}$"]
                           + [rf"$10^{{{k}}}$" for k in range(LOG_XMIN + 2, 1, 2)])
        ax.set_xlabel(f"{name}  P(human), log scale", fontsize=FONT_LABEL)
        loc = "upper center"
    else:
        xmax, width = LINEAR_ZOOM[name]
        keep = r["s"] <= xmax
        _hist(ax, r["s"][keep], r["y"][keep], np.arange(0, xmax + width, width))
        _thr_line(ax, r["thr"], thr_label)
        ax.set_xlim(0, xmax)
        ax.set_xlabel(f"{name}  P(human)", fontsize=FONT_LABEL)
        loc = "upper right"
    ax.set_title(f"({letter}) {name}: {r['n_distinct']:,} distinct scores", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_NOTE, loc=loc, framealpha=1, edgecolor="none")


def _roc_panel(ax, res: dict[str, dict], letter: str) -> None:
    for name, r in res.items():
        m = r["fpr_curve"] <= ROC_XMAX * 1.02
        ax.plot(r["fpr_curve"][m], r["tpr_curve"][m], "-o", ms=2.2, lw=1.2,
                color=READOUT_COLOR[name], label=f"{name} ({r['tpr']:.0%})")
        op = np.flatnonzero(r["fpr_curve"] <= FPR)[-1]   # the point roc_point_at_fpr picks
        ax.plot(r["fpr_curve"][op], r["tpr_curve"][op], "o", ms=6, mfc="none", mew=1.4,
                color=READOUT_COLOR[name])
    ax.axvline(FPR, color=_FPR_COLOR, ls="--", lw=1.1)
    ax.set_xlim(0, ROC_XMAX)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("false positive rate", fontsize=FONT_LABEL)
    ax.set_ylabel("true positive rate", fontsize=FONT_LABEL)
    ax.set_title(f"({letter}) Achievable operating points", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_NOTE, frameon=False, loc="lower right")


def _plot(res: dict[str, dict], judge: str, split: str, out: Path) -> None:
    """One histogram panel per readout, then the ROC zoom; 1 x 3 without
    FusionJudge, 2 x 2 with it."""
    n = len(res) + 1
    shape, size = ((1, 3), FIGSIZE) if n == 3 else ((2, 2), FIGSIZE_2X2)
    fig, axes = plt.subplots(*shape, figsize=size)
    axes = axes.ravel()
    letters = "abcd"
    for i, (name, r) in enumerate(res.items()):
        _hist_panel(axes[i], name, r, letters[i])
    _roc_panel(axes[len(res)], res, letters[len(res)])
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    stem = f"readout_zoom_{judge}_{split}"
    for suffix in (".pdf", ".png"):
        fig.savefig(out / f"{stem}{suffix}", dpi=200)
        print(f"wrote {out / (stem + suffix)}")
    plt.close(fig)


STAT_FIELDS = ["n", "n_human", "n_machine", "auroc", "thr", "tpr", "n_distinct",
               "n_distinct_below_h5", "n_human_below_floor", "n_machine_below_floor", "n_human_at_thr", "n_machine_at_thr",
               "next_score", "n_human_next", "n_machine_next", "next_fpr", "next_tpr"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", default="gpt56luna", choices=sorted(JUDGE_DISPLAY))
    ap.add_argument("--split", default="all", choices=["all", *SPLITS],
                    help=f"'all' pools the core splits {list(CORE_SPLITS)} (default)")
    args = ap.parse_args()

    df = _load(args.judge, args.split)
    res = {name: _readout(df, col) for name, col in _columns(args.judge).items()}

    where = "all core splits pooled" if args.split == "all" else split_display(args.split)
    print(f"{JUDGE_DISPLAY[args.judge]}, {where}\n")
    rows = []
    for name, r in res.items():
        rows.append({"readout": name, **{k: r[k] for k in STAT_FIELDS}})
        print(f"{name:<10} n={r['n']} (human {r['n_human']}, machine {r['n_machine']})  "
              f"AUROC {r['auroc']:.3f}  distinct scores {r['n_distinct']} "
              f"({r['n_distinct_below_h5']} below the human 5th pct)")
        print(f"{'':<10} 1%-FPR threshold P(human) <= {r['thr']:.3g}: TPR {r['tpr']:.3f}; "
              f"at exactly that score: {r['n_human_at_thr']} human, {r['n_machine_at_thr']} machine")
        print(f"{'':<10} next score {r['next_score']:.3g} holds {r['n_human_next']} human + "
              f"{r['n_machine_next']} machine -> next operating point "
              f"FPR {r['next_fpr']:.3%}, TPR {r['next_tpr']:.3f}\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / f"stats_{args.judge}_{args.split}.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"wrote {csv}")

    _plot(res, args.judge, args.split, OUT_DIR)


if __name__ == "__main__":
    main()
