r"""One figure: 4 training scenarios x 5 algorithms, for the MG-Sci
domain-split transfer comparison (``run_domain_split_transfer.py``).

A single-axis (AT-HM) slope chart, not the AUROC-vs-TPR scatter+contour
design used elsewhere in the article: with the logistic-regression trained
detector dropped (kept: RoBERTa-Sentinel, the citation-backed baseline from
Chen et al. 2023 -- the near-duplicate logistic ablation added nothing a
reviewer needed and cost page space) the data here is one line that
degrades across 4 scenarios, plus FOUR flat zero-shot reference values --
a shape the scatter design is the wrong tool for. One coloured line for
RoBERTa-Sentinel across the 4 scenarios (x-axis, in degradation order);
four flat dashed horizontal reference lines for the zero-shot picks,
labelled at their right end, since a zero-shot detector has no training
corpus and the fixed 30% eval slice is the one eval target throughout (its
reading truly does not move, not merely "flat within noise"). RADAR is
excluded entirely, not drawn and not in the table either -- see
:data:`ZS_ALGOS`'s comment for why (its raw AUROC here is below chance, so
its AT-HM floors to exactly 0.000 and carries no information).

Colour and legend-text convention matches ``run_best_zeroshot_performance.
py``'s main General-Results figure exactly (imported from there, not
redefined, so the two can never drift apart): Binoculars and Fast-DetectGPT
keep their baseline green/gold; LogJudge and FusionJudge keep their accent
blue/purple and their bold "(ours)" legend text, this paper's two
contributions. RoBERTa-Sentinel gets its own terracotta -- deliberately
lighter/warmer than the four zero-shot colours -- since it is not one of
this paper's contributions or the headline comparison here, just the
trained reference point the zero-shot flatness is being contrasted
against.

X-axis categories name what actually changed at each step, using this
paper's own Near-OOD/Far-OOD vocabulary (Section~\ref{sec:results_training_
comparison}) plus one intermediate tier: In-Domain (70% of the held-out
domain included in training) -> Near-OOD (held-out domain fully excluded,
but still MG-Sci) -> Mid-OOD (MG-GB: different genre, contemporary
generators) -> Far-OOD (GB-orig: different genre AND a legacy, single-
generator source). Which MG-Sci domain was held out for a given run is read
off the CSV's own ``train_split`` values (see :func:`scenario_layout`), not
passed as a flag here.

Writes a PNG next to the CSV (quick preview) AND a PDF into the article's
``figs/`` directory by default (``--figs-dir ''`` to skip the PDF), named
``domain_split_transfer_<code>_<fpr-tag>`` either way.

Run:
    uv run python epai/ai_detection/trainv2/plot_domain_split_transfer.py \
        results/trainv2/domain_split_transfer_<code>_<timestamp>/performance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the Agg backend is chosen)
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402  (avoid clashing with pathlib.Path)

from epai.ai_detection.trainv2.core import goodness  # noqa: E402
from epai.ai_detection.trainv2.plots import _INK, _MUTED, _SURFACE  # noqa: E402
from epai.ai_detection.trainv2.run_best_zeroshot_performance import (  # noqa: E402
    _ACCENT_BLUE,
    _ACCENT_PURPLE,
    _BASELINE_STYLE,
    _LEGEND_LABEL,
)

#: fire/snowflake marker convention, matching Table~\ref{tab:domain-split-
#: transfer-avg}'s \faIcon{fire}/\faIcon{snowflake}. Neither the actual
#: fontawesome5 glyph nor a colour emoji font is usable here: matplotlib's
#: default font (DejaVu Sans) has no glyph for U+1F525 FIRE (renders as a
#: missing-glyph box), and matplotlib's text renderer can't load colour-
#: bitmap emoji fonts at all (tried system NotoColorEmoji.ttf -- "unknown
#: file format" from FT2Font). A hand-built flame silhouette, used as the
#: trained line's own marker (so it shows up automatically, matched to the
#: line's colour, everywhere that line appears including the legend) is
#: closer to fontawesome5's flat single-colour icon style than a coloured
#: cartoon emoji would have been anyway. U+2744 SNOWFLAKE IS in DejaVu Sans,
#: so the zero-shot/"frozen" marker (plain text, not a path) matches the
#: table exactly with no workaround needed.
def _flame_path() -> MplPath:
    verts = np.array([
        (0.50, 0.00),
        (0.12, 0.22), (0.02, 0.55), (0.22, 0.78),
        (0.34, 0.90), (0.28, 0.68), (0.44, 0.58),
        (0.58, 0.70), (0.52, 0.88), (0.42, 1.00),
        (0.62, 0.92), (0.80, 0.72), (0.86, 0.46),
        (0.92, 0.20), (0.72, 0.06), (0.50, 0.00),
    ]) - 0.5  # centre roughly at origin, as a marker path expects
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 15
    return MplPath(verts, codes)


FLAME_MARKER = _flame_path()
COLD_MARKER = "❄"

TRAINED_ALGO = "trained/roberta·sentinel_mlp"
#: written out, not the raw detector id -- matches how the paper's prose
#: refers to it (Chen et al. 2023's architecture name).
TRAINED_LABEL = "RoBERTa-Sentinel"
#: deliberately lighter/more muted than any zero-shot colour below -- this
#: is the trained reference point, not a contribution or headline result.
#: Terracotta red -- warm, distinct from the cool green/blue/purple
#: zero-shot hues.
TRAINED_COLOR = "#c1654a"

#: fixed order -> also the line-drawing order, and (via build_domain_split_
#: transfer_table.py's import of this list) the table's row set too.
#: Colours and legend text imported from run_best_zeroshot_performance.py's
#: main-figure styling, not redefined -- see the module docstring. RADAR is
#: excluded entirely (not just from the figure): its raw AUROC on this eval
#: set is 0.362, BELOW chance -- the AT-HM formula floors that to exactly
#: 0.000 (the normalized-AUROC term clamps to 0 whenever raw AUROC <= 0.5,
#: and a harmonic mean is 0 whenever either term is), so it carries no
#: information here, not a computation artifact.
ZS_ALGOS = ["binox/llama31-8b", "fastdetect/llama31-8b",
           "judge/gpt56luna_ord2", "fusionjudge/gpt56luna_v3"]
ZS_COLOR = {
    "binox/llama31-8b": _BASELINE_STYLE["binox/llama31-8b"],
    "fastdetect/llama31-8b": _BASELINE_STYLE["fastdetect/llama31-8b"],
    "judge/gpt56luna_ord2": _ACCENT_BLUE,
    "fusionjudge/gpt56luna_v3": _ACCENT_PURPLE,
}
ZS_LABEL = {a: _LEGEND_LABEL[a] for a in ZS_ALGOS}
#: "ours" (LogJudge, FusionJudge) draws bold, matching the main figure's
#: emphasis convention; the two baselines (Binoculars, Fast-DetectGPT) do
#: not.
ZS_BOLD = {a: a in ("judge/gpt56luna_ord2", "fusionjudge/gpt56luna_v3") for a in ZS_ALGOS}

#: short code -> the readable domain name, for the table's column headers
#: (this figure's own x-axis categories are fixed -- see the module
#: docstring -- and don't need the domain name at all).
_CODE_READABLE = {"math": "mathematics", "cs": "computer science", "soc": "sociology"}

#: the x-axis category for each of the 4 scenario roles, in fixed order --
#: positional, not looked up per scenario key, since scenario_layout always
#: returns [in_key, ex_key, "mg_gb", "gb_orig"] in that order.
SCENARIO_XLABEL = ["In-Domain", "Near-OOD", "Mid-OOD", "Far-OOD"]


def scenario_layout(df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """``(scenarios, scenario_label)`` derived from the CSV's own
    ``train_split`` values -- the in-/ex-domain scenarios' exact keys
    (``mg_sci_in_<code>`` / ``mg_sci_ex_<code>``) depend on which MG-Sci
    domain ``run_domain_split_transfer.py --holdout-domain`` split for this
    particular run, so they're read off the data rather than hardcoded.
    ``scenario_label`` is the verbose form used by the LaTeX table's column
    headers (``build_domain_split_transfer_table.py``); this figure's own
    x-axis uses the fixed, shorter :data:`SCENARIO_XLABEL` instead."""
    others = sorted(s for s in df["train_split"].unique() if s not in ("mg_gb", "gb_orig"))
    in_candidates = [s for s in others if s.startswith("mg_sci_in_")]
    ex_candidates = [s for s in others if s.startswith("mg_sci_ex_")]
    if len(in_candidates) != 1 or len(ex_candidates) != 1:
        raise SystemExit(
            f"expected exactly one 'mg_sci_in_<code>' and one 'mg_sci_ex_<code>' scenario "
            f"in train_split, found {others} -- was this CSV produced by "
            f"run_domain_split_transfer.py?")
    in_key, ex_key = in_candidates[0], ex_candidates[0]
    code = in_key.removeprefix("mg_sci_in_")
    scenarios = [in_key, ex_key, "mg_gb", "gb_orig"]
    if code == "avg":
        # run_domain_split_transfer_avg.py's averaged CSV -- "70% {domain}"
        # doesn't name a single domain here, so this gets its own wording
        # rather than falling through _CODE_READABLE's generic template.
        labels = {
            in_key: "Trained on MG-Sci (incl. 70% held-out domain, avg. over 3)",
            ex_key: "Trained on MG-Sci (excl. held-out domain, avg. over 3)",
            "mg_gb": "Trained on MG-GB",
            "gb_orig": "Trained on GB-orig",
        }
    else:
        readable = _CODE_READABLE.get(code, code)
        labels = {
            in_key: f"Trained on MG-Sci (incl. 70% {readable})",
            ex_key: f"Trained on MG-Sci (excl. {readable})",
            "mg_gb": "Trained on MG-GB",
            "gb_orig": "Trained on GB-orig",
        }
    return scenarios, labels


def _athm(df: pd.DataFrame, fpr: float, detector: str, train_split: str) -> float:
    r = df[(df["fpr"] == fpr) & (df["detector"] == detector)
          & (df["train_split"] == train_split)].iloc[0]
    return goodness(r["auroc"], r["tpr_at_fpr"], alpha=fpr)


def plot(df: pd.DataFrame, fpr: float, paths: list[Path]) -> None:
    scenarios, _ = scenario_layout(df)
    trained_athm = [_athm(df, fpr, TRAINED_ALGO, s) for s in scenarios]
    zs_athm = {a: _athm(df, fpr, a, scenarios[0]) for a in ZS_ALGOS}

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    xs = list(range(len(scenarios)))
    ax.plot(xs, trained_athm, "-o", color=TRAINED_COLOR, ms=8, lw=2.2, zorder=5)

    for algo in ZS_ALGOS:
        v, color = zs_athm[algo], ZS_COLOR[algo]
        weight = "bold" if ZS_BOLD[algo] else "normal"
        ax.axhline(v, color=color, lw=1.4, ls="--", alpha=0.85, zorder=2)
        ax.text(xs[-1] + 0.12, v, f"{COLD_MARKER} {ZS_LABEL[algo]}", color=color, fontsize=11,
                fontweight=weight, va="center", ha="left", zorder=6,
                bbox=dict(facecolor=_SURFACE, edgecolor="none", pad=1.0))

    ax.set_xticks(xs)
    ax.set_xticklabels(SCENARIO_XLABEL, fontsize=12)
    ax.set_ylabel("AT-HM", fontsize=13, color=_INK)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.3, len(scenarios) - 1 + 1.7)
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)
    ax.tick_params(labelsize=11, colors=_MUTED)

    # A plain (line, marker) legend tuple draws both at FULL size on top of
    # each other (matplotlib's HandlerTuple default, ndivide=1) -- the flame
    # then sits stamped on the line sample instead of following it. ndivide
    # =None splits the legend's marker box into one section per handle
    # instead, drawn left to right, so the line sample comes first and the
    # flame trails it, matching how the snowflake trails each zero-shot
    # line's own text label.
    # HandlerTuple's own `pad` shrinks each section to make room instead of
    # inserting a gap between them (tried -- the line sample shrank to a
    # stub instead of moving apart from the flame); a third, invisible
    # spacer handle reserves its own blank section between them instead.
    line_handle = Line2D([], [], color=TRAINED_COLOR, lw=2.2)
    spacer_handle = Line2D([], [], linestyle="none")
    flame_handle = Line2D([], [], color=TRAINED_COLOR, marker=FLAME_MARKER, ms=13,
                         linestyle="none", markeredgewidth=0)
    ax.legend(handles=[(line_handle, spacer_handle, flame_handle)], labels=[TRAINED_LABEL],
             handler_map={tuple: HandlerTuple(ndivide=None)},
             fontsize=11, loc="lower left", frameon=False)

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_FIGS_DIR = BASE_DIR / "latex" / "_ICLR_2027__AI_Detect" / "figs"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="run_domain_split_transfer.py's performance.csv")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: the CSV's own directory (PNG, for quick preview)")
    ap.add_argument("--figs-dir", type=Path, default=DEFAULT_FIGS_DIR,
                    help=f"where the article-ready PDF is also written (default: "
                         f"{DEFAULT_FIGS_DIR}); pass '' to skip the PDF entirely")
    args = ap.parse_args()
    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = Path(args.figs_dir) if str(args.figs_dir) else None

    df = pd.read_csv(args.csv)
    needed = [TRAINED_ALGO, *ZS_ALGOS]
    missing = [a for a in needed if a not in set(df["detector"])]
    if missing:
        raise SystemExit(
            f"performance.csv is missing detector(s) {missing} -- for the zero-shot "
            f"picks, re-run run_domain_split_transfer.py with those columns available "
            f"(see --zs-columns / --no-zeroshot).")

    in_key = scenario_layout(df)[0][0]
    code = in_key.removeprefix("mg_sci_in_")
    for fpr in sorted(df["fpr"].unique()):
        tag = f"{fpr * 100:g}pct"
        name = f"domain_split_transfer_{code}_{tag}"
        paths = [out_dir / f"{name}.png"]
        if figs_dir is not None:
            paths.append(figs_dir / f"{name}.pdf")
        plot(df, fpr, paths)


if __name__ == "__main__":
    main()
