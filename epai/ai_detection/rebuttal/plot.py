r"""Figures for the rebuttal API experiments, from ``evaluate.py``'s CSVs only.

    .venv/bin/python -m epai.ai_detection.rebuttal.plot [--out <eval dir>]

Writes PNG + PDF to ``<eval dir>/figs/``. A figure whose CSV is missing is
skipped. Palette and plain style follow the release repo's ``plotting/_plots.py``.

  overview                  every BaseJudge / LogJudge / FusionJudge version in one
                            figure: luna A1 (full), luna A4 (small), gemma A1
                            (humans_plus_third); one block per subset
  a1_readouts_<model>       AUROC / TPR@1%FPR / AT-HM per readout (dots = splits,
                            bar = four-split mean): A1 + A2 at a glance
  a1_resolution_<model>     distinct values among human texts, and the size of the
                            human tie group that blocks the step past 1% FPR
  a1_written_values_<model> the most frequent written numbers per precision prompt
  a1_roc_<model>            ROC at FPR <= 5% per split: tie groups are the long
                            straight segments
  a3_stability              AT-HM per repeat, and decision flip rates at the real
                            1%-FPR threshold near it vs. away from it
  a4_averaging              BaseJudge / LogJudge / FusionJudge averaged over repeated
                            calls, against the API calls per text
  a5_labels                 LogJudge under other label words / order
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epai.ai_detection.rebuttal.conditions import WORKSPACE

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7",
           "#008300", "#e34948"]
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#8a8a86"
SPLITS = ["GB-orig", "MG-GB", "MG-Sci", "MG-AI-Sci"]
SPLIT_STYLE = {s: (PALETTE[i], m) for i, (s, m) in enumerate(zip(SPLITS, "^vsD"))}
MODEL_NAME = {"gpt56luna": "gpt-5.6-luna", "gemma4-31b": "gemma-4-31b"}
METRICS = [("auroc", "AUROC"), ("tpr", "TPR @ 1% FPR"), ("athm", "AT-HM")]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 10,
                     "legend.frameon": False, "savefig.facecolor": "white"})


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def _read(d: Path, name: str) -> pd.DataFrame | None:
    p = d / f"{name}.csv"
    # written values stay text: "0.10" and "0.1" are different answers
    return pd.read_csv(p, dtype={"value": str, "rep": str}) if p.is_file() else None


def _groups(labels: list[str]) -> list[int]:
    """Row positions after which a thin separator is drawn: between the BaseJudge
    rows, the A2 rows and the LogJudge / FusionJudge rows."""
    fam = ["A2" if l.startswith("A2") else l.split(",")[0].split(" (")[0].split(" T=")[0]
           for l in labels]
    return [i for i in range(len(fam) - 1) if fam[i] != fam[i + 1]]


def dotplot(df: pd.DataFrame, order: list[str], panels: list[tuple[str, str]], path: Path,
            name: str, title: str, logx: bool = False) -> None:
    """Rows = readouts (in ``order``, top to bottom), one panel per metric. A dot
    per split (colour + shape), a black tick for the four-split mean."""
    labels = [l for l in order if l in set(df["label"])]
    if not labels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels) + 2.2,
                                                      0.32 * len(labels) + 1.4),
                             sharey=True, squeeze=False)
    _dotrow(axes[0], df, labels, panels, logx)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.6, 1.0),
               fontsize=8, handletextpad=0.2, columnspacing=1.0)
    fig.suptitle(title, x=0.01, ha="left", y=1.06, fontsize=10)
    _save(fig, path, name)


def _dotrow(axes, df: pd.DataFrame, labels: list[str], panels: list[tuple[str, str]],
            logx: bool = False) -> None:
    """One row of dot panels (shared y = ``labels``)."""
    ypos = {l: len(labels) - 1 - i for i, l in enumerate(labels)}
    for ax, (col, xlabel) in zip(axes, panels):
        _style(ax)
        for i in _groups(labels):
            ax.axhline(ypos[labels[i]] - 0.5, color=MUTED, lw=0.5, alpha=0.5)
        for s in SPLITS:
            g = df[(df["split"] == s) & df["label"].isin(labels)]
            if g.empty or col not in g:
                continue
            c, m = SPLIT_STYLE[s]
            ax.scatter(g[col], [ypos[l] for l in g["label"]], s=26, color=c, marker=m,
                       edgecolor="white", linewidth=0.6, zorder=3, label=s)
        mean = df[(df["split"] == "mean") & df["label"].isin(labels)]
        if col in mean and mean[col].notna().any():
            ax.scatter(mean[col], [ypos[l] for l in mean["label"]], marker="|", s=220,
                       color=INK, linewidth=2, zorder=4, label="four-split mean")
        if logx:
            ax.set_xscale("symlog", linthresh=1) if (df[col] == 0).any() else ax.set_xscale("log")
        ax.set_xlabel(xlabel)
    axes[0].set_yticks(list(ypos.values()), list(ypos.keys()))
    axes[0].set_ylim(-0.6, len(labels) - 0.4)


# ---------------------------------------------------------------------------


def fig_a1(d: Path, figs: Path) -> None:
    m = _read(d, "a1_metrics")
    if m is None:
        return
    order = A1_ORDER
    for model, g in m.groupby("model"):
        name = MODEL_NAME.get(model, model)
        dotplot(g, order, METRICS, figs, f"a1_readouts_{model}",
                f"A1/A2 ({name}): BaseJudge at higher precision and its same-call expected "
                f"value vs. LogJudge, identical rows")
        dotplot(g[g["split"] != "mean"], order,
                [("n_distinct_human", "distinct values among human texts"),
                 ("tie_humans", "humans in the tie group at 1% FPR")],
                figs, f"a1_resolution_{model}",
                f"A1 ({name}): resolution, not digits. Distinct values and the tie "
                f"group that blocks the 1%-FPR step", logx=True)

    v = _read(d, "a1_written_values")
    if v is not None:
        for model, g in v.groupby("model"):
            conds = [c for c in ("base_v3", "base_hp4", "base_free") if c in set(g["readout"])]
            fig, axes = plt.subplots(1, len(conds), figsize=(3.2 * len(conds), 3.2),
                                     squeeze=False)
            for ax, c in zip(axes[0], conds):
                _style(ax)
                gc = g[g["readout"] == c].copy()
                gc["n"] = gc["count"] / gc["share"]
                total = gc.groupby("split")["n"].first().sum()
                top = gc.groupby("value")["count"].sum().sort_values(ascending=False).head(10)
                ax.barh(range(len(top))[::-1], 100 * top.to_numpy() / total, color=PALETTE[0],
                        height=0.7)
                ax.set_yticks(range(len(top))[::-1], list(top.index))
                ax.set_xlabel("share of all answers (%)")
                ax.set_title({"base_v3": "2 decimals", "base_hp4": "4 decimals",
                              "base_free": "free precision"}[c])
            fig.suptitle(f"A1 ({MODEL_NAME.get(model, model)}): the 10 most frequent "
                         f"written values", x=0.01, ha="left")
            _save(fig, figs, f"a1_written_values_{model}")

    r = _read(d, "a1_roc")
    if r is not None:
        show = ["BaseJudge (paper call)", "BaseJudge, paper call, 4 dec.",
                "BaseJudge, paper call, free precision", "BaseJudge, matched call, 4 dec.",
                "LogJudge (paper)", "FusionJudge (paper)", "A2: E[number], 4 dec."]
        for model, g in r.groupby("model"):
            splits = [s for s in SPLITS if s in set(g["split"])]
            fig, axes = plt.subplots(1, len(splits), figsize=(3.0 * len(splits), 3.0),
                                     squeeze=False)
            for ax, s in zip(axes[0], splits):
                _style(ax)
                for i, lab in enumerate(show):
                    c = g[(g["split"] == s) & (g["label"] == lab)]
                    if c.empty:
                        continue
                    ax.plot(100 * c["fpr"], c["tpr"], color=PALETTE[i], lw=1.4, label=lab,
                            ls=":" if "matched" in lab or lab.startswith("A2") else "-")
                ax.axvline(1, color=MUTED, ls="--", lw=0.8)
                ax.set_xlim(0, 5)
                ax.set_xlabel("FPR (%)")
                ax.set_title(s)
            axes[0][0].set_ylabel("TPR")
            h, l = axes[0][0].get_legend_handles_labels()
            fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12), fontsize=8)
            fig.suptitle(f"A1/A2 ({MODEL_NAME.get(model, model)}): low-FPR ROC. A straight "
                         f"segment is one tie group, flagged all at once", x=0.01, ha="left",
                         y=1.2)
            _save(fig, figs, f"a1_roc_{model}")


A1_ORDER = ["BaseJudge (paper call)", "BaseJudge, paper call, 4 dec.",
            "BaseJudge, paper call, 4 dec. rounded to 2", "BaseJudge, paper call, free precision",
            "BaseJudge, matched call, 2 dec.", "BaseJudge, matched call, 4 dec.",
            "BaseJudge, matched call, free precision", "A2: E[number], 2 dec.",
            "A2: E[number], 4 dec.", "A2: E[number], free", "LogJudge (paper)",
            "FusionJudge (paper)", "FusionJudge, paper call 4-dec. BaseJudge",
            "FusionJudge, paper call free-precision BaseJudge",
            "FusionJudge, matched BaseJudge"]
A4_ORDER = ["BaseJudge (paper call)", "BaseJudge paper call, 1 new sample",
            "BaseJudge paper call, mean of 2 new", "BaseJudge paper call, mean of 3 new",
            "BaseJudge paper call, mean of 3 new + paper", "BaseJudge, matched call, 2 dec.",
            "BaseJudge T=1, 1 sample", "BaseJudge T=1, mean of 2", "BaseJudge T=1, mean of 3",
            "LogJudge (paper)", "LogJudge, 1 new call", "LogJudge, mean of 2 new",
            "LogJudge, mean of 3 new", "LogJudge, mean of 3 new + paper",
            "FusionJudge (paper)", "FusionJudge, 1+1 new calls", "FusionJudge, means of 2+2 new",
            "FusionJudge, means of 3+3 new", "FusionJudge, means of 3+3 new + paper"]


def fig_overview(d: Path, figs: Path) -> None:
    """All versions, one block per (model, subset): metrics are only comparable
    within a block (same rows), never across blocks."""
    a1, a4 = _read(d, "a1_metrics"), _read(d, "a4_metrics")
    blocks = []
    if a1 is not None:
        for model, subset in (("gpt56luna", "full"), ("gemma4-31b", "humans_plus_third")):
            g = a1[a1["model"] == model]
            if not g.empty:
                n = int(g[g["split"] != "mean"].groupby("label")["n"].sum().max())
                blocks.append((g, A1_ORDER, f"{MODEL_NAME[model]} · A1/A2 precision and "
                                            f"readout · subset {subset} ({n:,} texts)"))
    if a4 is not None:
        n = int(a4[a4["split"] != "mean"].groupby("label")["n"].sum().max())
        blocks.append((a4, A4_ORDER, f"gpt-5.6-luna · A4 repeated calls · subset small "
                                     f"({n:,} texts)"))
    blocks = [(g, [l for l in o if l in set(g["label"])], t) for g, o, t in blocks]
    blocks.sort(key=lambda b: b[2].startswith("gemma"))      # luna blocks first
    if not blocks:
        return
    heights = [len(l) + 1.2 for _, l, _ in blocks]
    fig, axes = plt.subplots(len(blocks), len(METRICS), squeeze=False,
                             figsize=(3.0 * len(METRICS) + 3.4, 0.3 * sum(heights) + 1.2),
                             gridspec_kw={"height_ratios": heights})
    for row, (g, labels, title) in zip(axes, blocks):
        _dotrow(row, g, labels, METRICS)
        row[0].set_title(title, loc="left", fontsize=9.5, fontweight="bold", x=-0.9)
        for ax in row[1:]:
            ax.set_yticks(row[0].get_yticks(), [""] * len(row[0].get_yticks()))
            ax.set_ylim(row[0].get_ylim())
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.6, 1.0),
               fontsize=8, handletextpad=0.2, columnspacing=1.0)
    fig.suptitle("All BaseJudge / LogJudge / FusionJudge versions. Dots = splits, black "
                 "tick = four-split mean; compare only within a block (same rows)",
                 x=0.01, ha="left", y=1.02, fontsize=10)
    _save(fig, figs, "overview")


def fig_a3(d: Path, figs: Path) -> None:
    m, f = _read(d, "a3_metrics"), _read(d, "a3_flips")
    if m is None or f is None:
        return
    readouts = ["BaseJudge (matched call)", "LogJudge", "FusionJudge (matched)"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.8))
    ax = axes[0]
    _style(ax)
    mean = m[m["split"] == "mean"]
    ypos = {l: len(readouts) - 1 - i for i, l in enumerate(readouts)}
    for rep, g in mean.groupby("rep"):
        ref = rep == "ref"
        ax.scatter(g["athm"], [ypos[l] for l in g["label"]], s=60 if ref else 30,
                   facecolor="none" if ref else PALETTE[0],
                   edgecolor=INK if ref else "white", linewidth=1.2 if ref else 0.6,
                   zorder=3 if ref else 2,
                   label="reference run" if ref else ("repeat" if rep == "0" else None))
    ax.set_yticks(list(ypos.values()), list(ypos.keys()))
    ax.set_xlabel("four-split mean AT-HM")
    ax.set_title("(a) AT-HM per repeated call", loc="left")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    _style(ax)
    groups = [("near_human", "humans near threshold"), ("near_machine", "machine near threshold"),
              ("control", "random control")]
    for i, (gk, gl) in enumerate(groups):
        g = f[f["group"] == gk].copy()
        g["w"] = g["flip_rate"] * g["n"]
        agg = g.groupby("label").agg(w=("w", "sum"), n=("n", "sum"))
        agg = agg.reindex(readouts).dropna()
        ax.scatter(100 * agg["w"] / agg["n"], [ypos[l] + (1 - i) * 0.18 for l in agg.index],
                   s=30, color=PALETTE[i], marker="os^"[i], edgecolor="white", linewidth=0.6,
                   label=gl, zorder=3)
    ax.set_yticks(list(ypos.values()), [""] * len(ypos))
    ax.set_xlabel("decisions that flip across repeats (%)")
    ax.set_title("(b) flips at the real 1%-FPR threshold", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("A3 (gpt-5.6-luna): stability under repeated calls", x=0.01, ha="left")
    _save(fig, figs, "a3_stability")


def fig_a4(d: Path, figs: Path) -> None:
    """Score averaged over k repeated calls, against the API calls per text it took.
    FusionJudge from k BaseJudge + k LogJudge calls sits at 2k. The last point of the
    paper-call curves adds the paper's stored answer to the 3 new ones."""
    m = _read(d, "a4_metrics")
    if m is None:
        return
    mean = m[m["split"] == "mean"].set_index("label")
    if "calls" not in mean:
        return
    curves = [("t1_x", "BaseJudge matched call T=1, mean of k", PALETTE[0], "o"),
              ("def_x", "BaseJudge paper call, mean of k", PALETTE[5], "s"),
              ("lj_x", "LogJudge, mean of k", PALETTE[2], "D"),
              ("fus_x", "FusionJudge, means of k + k", PALETTE[3], "^")]
    singles = [("BaseJudge, matched call, 2 dec.", PALETTE[1]),
               ("BaseJudge (paper call)", PALETTE[5]),
               ("LogJudge (paper)", PALETTE[2]), ("FusionJudge (paper)", PALETTE[3])]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1))
    xmax = 1
    for ax, (col, lab) in zip(axes, [("tpr", "four-split mean TPR @ 1% FPR"),
                                     ("athm", "four-split mean AT-HM")]):
        _style(ax)
        for prefix, name, c, mk in curves:
            g = mean[mean["readout"].str.startswith(prefix)].sort_values("calls")
            if g.empty:
                continue
            xmax = max(xmax, int(g["calls"].max()))
            ax.plot(g["calls"], g[col], marker=mk, color=c, lw=1.6, ms=5, label=name)
        for l, c in singles:
            if l in mean.index:
                ax.scatter(mean.loc[l, "calls"], mean.loc[l, col], s=70, facecolor="none",
                           edgecolor=c, linewidth=1.6, zorder=4,
                           label=f"{l} (stored, {int(mean.loc[l, 'calls'])} call"
                                 f"{'s' if mean.loc[l, 'calls'] > 1 else ''})")
        ax.set_xticks(range(1, xmax + 1))
        ax.set_xlabel("API calls per text")
        ax.set_ylabel(lab)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.22), fontsize=8)
    fig.suptitle("A4 (gpt-5.6-luna, small subset): averaging repeated calls, by the calls it "
                 "takes", x=0.01, ha="left", y=1.3)
    _save(fig, figs, "a4_averaging")


def fig_a5(d: Path, figs: Path) -> None:
    m = _read(d, "a5_metrics")
    if m is None:
        return
    order = ["BaseJudge (paper call)", "BaseJudge, matched call, 2 dec.", "LogJudge (paper)",
             "LogJudge, machine listed first", "LogJudge, person / AI",
             "LogJudge, A=machine / B=human"]
    dotplot(m, order, METRICS, figs, "a5_labels",
            "A5 (gpt-5.6-luna, small subset): LogJudge under other label words and order")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=WORKSPACE / "eval",
                    help="the folder evaluate.py wrote its CSVs to")
    args = ap.parse_args()
    figs = args.out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    print(f"figures -> {figs}")
    for fn in (fig_overview, fig_a1, fig_a3, fig_a4, fig_a5):
        fn(args.out, figs)


if __name__ == "__main__":
    main()
