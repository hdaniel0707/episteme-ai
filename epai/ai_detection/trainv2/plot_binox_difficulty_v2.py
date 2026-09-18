r"""Publication figure v2: MANUGRAM gets harder for Binoculars, left to right.

One row, four columns, the original Falcon-7B backbone only
(``binoculars_score`` -- the paper's own pair): the four core MANUGRAM splits
in registry order, ``splits.DEFAULT`` -- GB-orig, MG-GB, MG-Sci, MG-AI-Sci.

v1 (``plot_binox_difficulty.py``) packed two backbones x four corpora into an
eight-panel grid with a cramped top-left legend and duplicated per-panel
stats, so the panels read as one undifferentiated, over-labeled block. v2
keeps the same human/AI histogram and base colors as
``calibrate_common._plot_column``, but marks each panel's fresh-fit-per-corpus
Youden's-J threshold (the AUROC-paired operating point) instead of a fixed
1%-FPR cut -- and:

    * one row only (drops the Llama-3.1-8B row -- a separate figure if ever
      needed, not a second row fighting this one for attention)
    * GB-orig (the original Ghostbuster/GPT-3.5 benchmark) is visually
      separated from the three MANUGRAM splits: a physical gap, a divider
      line, and a distinct card-background tint per group, so the reader
      sees "old benchmark" vs. "new benchmark" at a glance, not four
      interchangeable panels
    * far less text: one title line (corpus + AUROC) per panel, no repeated
      arrows/bands/subtitles
    * one legend, centered at the bottom, sized to read at a glance

Run:
    uv run python -m epai.ai_detection.trainv2.plot_binox_difficulty_v2

    uv run python -m epai.ai_detection.trainv2.plot_binox_difficulty_v2 \
        --corpora gb_orig mg_gb mg_sci mg_aisci \
        --col binoculars_score --backbone-label "Falcon-7B" \
        -o results/trainv2/binox_difficulty/binox_vs_manugram_v2.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.cd_core import auroc_score, oriented_scores, rates_at_threshold, youden_threshold

from .calibrate_common import PARQUET_DIR
from .splits import DEFAULT, SPLITS, display

GEN_COL = "generated_by"

_HUMAN_COLOR = "#1c6fb3"     # human = blue   (matches calibrate_common)
_MACHINE_COLOR = "#d92d20"   # AI = red
_YOUDEN_COLOR = "#0ca678"    # Youden's J threshold = teal (matches calibrate_common)
_CARD_EDGE = "#cfcfcf"       # panel card border

# Card + label colors per benchmark group -- GB-orig (the original,
# Ghostbuster/GPT-3.5 benchmark) reads neutral/gray; the MANUGRAM splits read
# in a pale violet, distinct from human/AI (blue/red) and from the Youden
# threshold line (teal).
_GROUP_STYLE = {
    "gb": {"card": "#f0f0ef", "label": "Ghostbuster (original)", "text": "#595959"},
    "mg": {"card": "#f4f0fb", "label": "Manugram", "text": "#5b3fa8"},
}

# Binoculars reads LOW = machine (see cd_core.oriented_scores docstring).
MACHINE_IS_HIGH = False


def _group(name: str) -> str:
    return "gb" if name.startswith("gb_orig") else "mg"


def _resolve(names: list[str]) -> list[tuple[str, Path]]:
    chosen = list(names) if names else list(DEFAULT)
    unknown = [n for n in chosen if n not in SPLITS]
    if unknown:
        raise SystemExit(f"unknown split(s) {unknown}; choose from {list(SPLITS)}")
    return [(k, PARQUET_DIR / SPLITS[k]) for k in chosen]


def _fit(col: str, corpus: str, df: pd.DataFrame, human_label: str) -> dict | None:
    if col not in df.columns:
        return None
    sub = df[[col, GEN_COL]].dropna(subset=[col])
    raw = sub[col].to_numpy(dtype=float)
    y = (sub[GEN_COL].astype(str) != human_label).to_numpy().astype(int)
    if np.unique(y).size < 2 or len(y) < 10:
        return None

    oriented = oriented_scores(raw, MACHINE_IS_HIGH)
    thr_or = youden_threshold(y, oriented)
    tpr, fpr = rates_at_threshold(y, oriented, thr_or)
    thr_raw = float(thr_or) if MACHINE_IS_HIGH else -float(thr_or)

    return {
        "column": col, "corpus": corpus, "raw": raw, "y": y,
        "n": int(len(y)), "n_human": int((y == 0).sum()), "n_machine": int((y == 1).sum()),
        "auroc": float(auroc_score(y, oriented)),
        "threshold": thr_raw, "tpr": float(tpr), "fpr": float(fpr),
    }


def _panel(ax, r: dict, edges: np.ndarray, mids: np.ndarray, card_color: str) -> None:
    raw, y = r["raw"], r["y"]
    h = np.histogram(raw[y == 0], bins=edges, density=True)[0]
    m = np.histogram(raw[y == 1], bins=edges, density=True)[0]

    ax.set_facecolor(card_color)
    for spine in ax.spines.values():
        spine.set_color(_CARD_EDGE)
        spine.set_linewidth(1.1)

    ax.step(mids, h, where="mid", color=_HUMAN_COLOR, lw=2.0, label="human", zorder=3)
    ax.fill_between(mids, h, step="mid", color=_HUMAN_COLOR, alpha=0.18, zorder=2)
    ax.step(mids, m, where="mid", color=_MACHINE_COLOR, lw=2.0, label="AI", zorder=3)
    ax.fill_between(mids, m, step="mid", color=_MACHINE_COLOR, alpha=0.18, zorder=2)

    if np.isfinite(r["threshold"]):
        ax.axvline(r["threshold"], color=_YOUDEN_COLOR, lw=1.8, ls="--",
                   label="Youden J cut", zorder=4)

    ax.set_title(f"{display(r['corpus'])}  ·  AUROC {r['auroc']:.2f}",
                 fontsize=14.5, fontweight="bold", pad=8)

    ax.set_xlim(edges[0], edges[-1])
    ax.set_yticks([])
    ax.grid(True, lw=0.4, alpha=0.3, color="#999999")
    ax.tick_params(axis="x", labelsize=13.5)
    ax.set_xlabel("Binoculars score", fontsize=14.5)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--corpora", nargs="+", default=None, metavar="NAME",
                    help="which splits to plot as columns, left to right "
                         "(default: the four core splits, gb_orig mg_gb "
                         "mg_sci mg_aisci)")
    ap.add_argument("--col", default="binoculars_score",
                    help="score column to plot (default: binoculars_score = "
                         "Falcon-7B, the original paper pair)")
    ap.add_argument("--backbone-label", default="Falcon-7B",
                    help="label used in the figure suptitle")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("-o", "--out", type=Path,
                    default=Path("results/trainv2/binox_difficulty/binox_vs_manugram_v2.pdf"),
                    help="figure path; also written as a sibling .png at 300 dpi")
    args = ap.parse_args()

    corpora = _resolve(args.corpora or [])
    frames = {name: pd.read_parquet(path) for name, path in corpora}

    rows: dict[str, dict] = {}
    print(f"{'corpus':<12}{'n_hum':>7}{'n_mac':>7}{'auroc':>8}{'thr@youden':>12}{'tpr':>6}{'fpr':>7}")
    for name, df in frames.items():
        r = _fit(args.col, name, df, args.human_label)
        if r is None:
            print(f"{name:<12}  -- skipped: column absent, single class, or n<10 --")
            continue
        print(f"{name:<12}{r['n_human']:>7}{r['n_machine']:>7}{r['auroc']:>8.3f}"
              f"{r['threshold']:>12.4g}{r['tpr']:>6.2f}{r['fpr']:>7.3f}")
        rows[name] = r
    print()

    names = [n for n, _ in corpora]
    missing = [n for n in names if n not in rows]
    if missing:
        raise SystemExit(f"no usable data for {missing} -- cannot draw a complete row")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12.5,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 13,
        "pdf.fonttype": 42,   # embed as real text, not curves
        "ps.fonttype": 42,
    })

    # Build the column plan: one gridspec column per panel, plus a narrow
    # spacer column wherever the benchmark group changes -- that spacer is
    # what makes GB-orig read as physically separate from the MANUGRAM trio.
    col_specs: list[tuple[str, str | None]] = []
    prev_group = None
    for name in names:
        g = _group(name)
        if prev_group is not None and g != prev_group:
            col_specs.append(("spacer", None))
        col_specs.append(("panel", name))
        prev_group = g
    width_ratios = [0.45 if kind == "spacer" else 1.0 for kind, _ in col_specs]

    fig = plt.figure(figsize=(3.9 * len(names) + 0.8, 5.6))
    fig.subplots_adjust(left=0.035, right=0.99, top=0.85, bottom=0.32, wspace=0.18)
    gs = fig.add_gridspec(1, len(col_specs), width_ratios=width_ratios)

    panel_axes: dict[str, plt.Axes] = {}
    for col_idx, (kind, name) in enumerate(col_specs):
        if kind == "panel":
            panel_axes[name] = fig.add_subplot(gs[0, col_idx])

    all_raw = np.concatenate([rows[n]["raw"] for n in names])
    lo, hi = np.percentile(all_raw, [0.5, 99.5])
    thrs = [rows[n]["threshold"] for n in names if np.isfinite(rows[n]["threshold"])]
    if thrs:
        lo, hi = min(lo, *thrs), max(hi, *thrs)
    pad = 0.03 * (hi - lo) or 0.01
    edges = np.linspace(lo - pad, hi + pad, args.bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2

    for name in names:
        g = _group(name)
        _panel(panel_axes[name], rows[name], edges, mids, card_color=_GROUP_STYLE[g]["card"])

    # Group labels + a divider line at the group boundary, positioned from
    # the panels' own (already laid-out) figure-fraction coordinates.
    groups_order: list[tuple[str, list[str]]] = []
    for name in names:
        g = _group(name)
        if not groups_order or groups_order[-1][0] != g:
            groups_order.append((g, [name]))
        else:
            groups_order[-1][1].append(name)

    top_y = panel_axes[names[0]].get_position().y1
    for g, members in groups_order:
        x0 = panel_axes[members[0]].get_position().x0
        x1 = panel_axes[members[-1]].get_position().x1
        style = _GROUP_STYLE[g]
        fig.text((x0 + x1) / 2, 0.98, style["label"], ha="center", va="top",
                  fontsize=19.5, fontweight="bold", color=style["text"])

    if len(groups_order) > 1:
        gap_x = (panel_axes[groups_order[0][1][-1]].get_position().x1
                  + panel_axes[groups_order[1][1][0]].get_position().x0) / 2
        bot_y = panel_axes[names[0]].get_position().y0
        fig.add_artist(Line2D([gap_x, gap_x], [bot_y - 0.02, top_y + 0.06],
                               transform=fig.transFigure, color="#b5b5b5",
                               lw=1.3, ls=(0, (4, 3))))

    # Footer stack, top to bottom: panels -> legend -> title, both sitting
    # below the plots (only the group labels stay up top).
    handles, labels = panel_axes[names[0]].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.194),
               ncol=len(labels), frameon=False, fontsize=15, handlelength=2.2,
               columnspacing=1.8)

    fig.suptitle(f"Binoculars vs. Manugram  ·  {args.backbone_label} backbone",
                 fontsize=15.5, fontweight="bold", y=0.098)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
