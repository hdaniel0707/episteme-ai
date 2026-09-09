r"""Score-distribution histograms for the rephrase task, laid out like the
baseline-calibration figures: one figure per score column, a panel per corpus.

For every score column of every detector family --

    binoculars   binoculars_score + every binoculars_score_*   (low  = machine)
    fastdetect   every fastdetect_crit_*                        (high = machine)
    detector     detector_score_radar                           (high = machine)
    judge        llm_judge_score_gpt56luna_v3                    (low  = machine)

-- one figure ``<column>.png`` with three panels: **MG-GB**, **MG-Sci**,
**MG-All** (pooled). Each panel is an overlaid density histogram of three
classes:

    human       generated_by == 'human'             blue
    rephrase    generated_by == 'gpt56luna:0908R'    orange   (from the *_reph_* parquet)
    rewritten   the full gpt56luna generation        red      (':0701A' in GB, ':0811A' in Sci)

The ``*_reph_*`` human rows are the same documents (same ``id``, suite scores
copied whole), so humans come from the full corpus. Each panel marks two
Youden's-J cuts fit on just the two classes involved: human|rephrase (dotted)
and human|rewritten (dashed). ``--no-thresholds`` drops them. Not a calibration
-- writes no config.

Output (under ``--out-dir``, default ``results/trainv2/rephrase_score_histograms/``):
    <column>.png   per score column, 3 panels (MG-GB / MG-Sci / MG-All)
    summary.csv    one row per (corpus, column, class): n, mean, p1/p50/p99,
                   the two cuts, the two human-vs-other AUROCs

Run:
    uv run python epai/ai_detection/trainv2/plot_score_histograms.py
    uv run python epai/ai_detection/trainv2/plot_score_histograms.py --families binoculars fastdetect
    uv run python epai/ai_detection/trainv2/plot_score_histograms.py --no-thresholds
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.cd_core import auroc_score, oriented_scores, youden_threshold

BASE = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE / "data" / "parquet"
DEFAULT_OUT_DIR = BASE / "results" / "trainv2" / "rephrase_score_histograms"

GEN_COL = "generated_by"
HUMAN = "human"
REPHRASE_GEN = "gpt56luna:0908R"
POOLED = "all"

#: per-corpus source: the full-suite parquet (+ its gpt56luna full-generation
#: label) and the rephrase-half parquet.
CORPUS_SOURCES: dict[str, dict] = {
    "mg_gb": {"display": "MG-GB", "full": "manugram_gb.parquet",
              "rewrite_gen": "gpt56luna:0701A", "reph": "manugram_reph_gb.parquet"},
    "mg_sci": {"display": "MG-Sci", "full": "manugram_sci.parquet",
               "rewrite_gen": "gpt56luna:0811A", "reph": "manugram_reph_sci.parquet"},
}
PANEL_ORDER = ["mg_gb", "mg_sci", POOLED]  # left-to-right, pooled last
DISPLAY = {**{k: v["display"] for k, v in CORPUS_SOURCES.items()}, POOLED: "MG-All"}

#: class -> colour. Order is legend / draw order.
CLASSES: dict[str, str] = {
    "human": "#1c6fb3", "rephrase": "#e8890c", "rewritten": "#d92d20",
}

#: family -> spec. ``prefix`` = every matching column; ``cols`` = exactly these.
FAMILIES: dict[str, dict] = {
    "binoculars": {"prefix": "binoculars_score", "machine_is_high": False,
                   "xlabel": "Binoculars score"},
    "fastdetect": {"prefix": "fastdetect_crit_", "machine_is_high": True,
                   "xlabel": "Fast-DetectGPT criterion"},
    "detector": {"cols": ["detector_score_radar"], "machine_is_high": True,
                 "xlabel": "RADAR  P(AI)"},
    "judge": {"cols": ["llm_judge_score_gpt56luna_v3"], "machine_is_high": False,
              "xlabel": "judge score  P(human)"},
}

CSV_FIELDS = [
    "corpus", "family", "column", "class", "n", "mean", "p1", "p50", "p99",
    "machine_is_high", "cut_human_rephrase", "cut_human_rewritten",
    "auroc_human_rephrase", "auroc_human_rewritten",
]


def family_columns(spec: dict, names: set[str]) -> list[str]:
    if "cols" in spec:
        return [c for c in spec["cols"] if c in names]
    return sorted(c for c in names if c.startswith(spec["prefix"]))


def build(columns: list[str]) -> pd.DataFrame:
    """Rows for every panel: columns ``[corpus, cls, *columns]``.

    Per corpus, ``human`` / ``rewritten`` come from the full parquet and
    ``rephrase`` from the rephrase half; ``all`` is the two corpora pooled.
    """
    parts = []
    for key, c in CORPUS_SOURCES.items():
        full = pd.read_parquet(PARQUET_DIR / c["full"], columns=[GEN_COL, *columns])
        reph = pd.read_parquet(PARQUET_DIR / c["reph"], columns=[GEN_COL, *columns])
        picks = {
            "human": full[full[GEN_COL] == HUMAN],
            "rephrase": reph[reph[GEN_COL] == REPHRASE_GEN],
            "rewritten": full[full[GEN_COL] == c["rewrite_gen"]],
        }
        for cls, rows in picks.items():
            if rows.empty:
                raise SystemExit(f"{key}: class {cls!r} has no rows")
            parts.append(rows[columns].assign(corpus=key, cls=cls))
    per_corpus = pd.concat(parts, ignore_index=True)
    pooled = per_corpus.assign(corpus=POOLED)
    return pd.concat([per_corpus, pooled], ignore_index=True)[["corpus", "cls", *columns]]


def panel_stats(sub: pd.DataFrame, column: str, machine_is_high: bool) -> dict:
    """Per-class percentiles + the two human-vs-other Youden cuts and AUROCs,
    for one (corpus, column) slice."""
    vals = {cls: sub.loc[sub.cls == cls, column].to_numpy(dtype=float) for cls in CLASSES}
    vals = {cls: v[np.isfinite(v)] for cls, v in vals.items()}
    per_class = {
        cls: {
            "n": int(v.size),
            "mean": float(v.mean()) if v.size else float("nan"),
            "p1": float(np.percentile(v, 1)) if v.size else float("nan"),
            "p50": float(np.median(v)) if v.size else float("nan"),
            "p99": float(np.percentile(v, 99)) if v.size else float("nan"),
        }
        for cls, v in vals.items()
    }
    cuts, aurocs = {}, {}
    h = vals["human"]
    for other in ("rephrase", "rewritten"):
        o = vals[other]
        if h.size < 10 or o.size < 10:
            cuts[other] = aurocs[other] = float("nan")
            continue
        raw = np.concatenate([h, o])
        y = np.concatenate([np.zeros(h.size), np.ones(o.size)])
        oriented = oriented_scores(raw, machine_is_high)
        thr_or = youden_threshold(y, oriented)
        cuts[other] = float(thr_or if machine_is_high else -thr_or)
        aurocs[other] = float(auroc_score(y, oriented))
    return {"per_class": per_class, "cuts": cuts, "aurocs": aurocs}


def plot_column(
    df: pd.DataFrame, column: str, family: str, spec: dict,
    stats: dict[str, dict], draw_thresholds: bool, path: Path, bins: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(PANEL_ORDER), figsize=(4.7 * len(PANEL_ORDER), 4.3),
                             squeeze=False)
    axes = axes[0]

    raw_all = df[column].to_numpy(dtype=float)
    raw_all = raw_all[np.isfinite(raw_all)]
    lo, hi = np.percentile(raw_all, [0.5, 99.5])
    if draw_thresholds:
        ts = [t for s in stats.values() for t in s["cuts"].values() if np.isfinite(t)]
        if ts:
            lo, hi = min(lo, *ts), max(hi, *ts)
    pad = 0.03 * (hi - lo) or 0.01
    lo, hi = lo - pad, hi + pad
    edges = np.linspace(lo, hi, bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2

    for ax, corpus in zip(axes, PANEL_ORDER):
        s = stats[corpus]
        sub = df[df.corpus == corpus]
        for cls, colour in CLASSES.items():
            v = sub.loc[sub.cls == cls, column].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            dens = np.histogram(v, bins=edges, density=True)[0]
            ax.step(mids, dens, where="mid", color=colour, lw=1.6, label=cls)
            ax.fill_between(mids, dens, step="mid", color=colour, alpha=0.13)

        if draw_thresholds:
            for other, ls in (("rephrase", ":"), ("rewritten", "--")):
                t = s["cuts"][other]
                if not np.isfinite(t):
                    continue
                ax.axvline(t, color=CLASSES[other], lw=1.5, ls=ls)
                ax.text(t, ax.get_ylim()[1] * 0.97, f" {t:.3g}",
                        color=CLASSES[other], fontsize=8, va="top",
                        ha="left" if t < lo + 0.8 * (hi - lo) else "right")

        a_r, a_w = s["aurocs"]["rephrase"], s["aurocs"]["rewritten"]
        title = f"{DISPLAY[corpus]}  (n={len(sub)})"
        if np.isfinite(a_r) and np.isfinite(a_w):
            title += f"\nAUROC  h|reph {a_r:.3f}   h|rew {a_w:.3f}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(spec["xlabel"])
        ax.set_xlim(lo, hi)
        ax.grid(True, lw=0.4, alpha=0.4)

    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8, loc="upper left")
    suptitle = f"{column}  ({family}) — human vs. rephrase vs. rewritten"
    if draw_thresholds:
        suptitle += "   (dotted = h|rephrase Youden cut, dashed = h|rewritten)"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--families", nargs="+", choices=list(FAMILIES),
                    default=list(FAMILIES))
    ap.add_argument("--no-thresholds", action="store_true",
                    help="just the histograms, no Youden cut lines")
    ap.add_argument("--bins", type=int, default=45)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    parquets = {p for c in CORPUS_SOURCES.values() for p in (c["full"], c["reph"])}
    names: set[str] | None = None
    for pf in parquets:
        path = PARQUET_DIR / pf
        if not path.exists():
            raise SystemExit(f"not found: {path}  (run split_manugram_reph.py first?)")
        s = set(pq.read_schema(path).names)
        names = s if names is None else (names & s)

    families = {f: FAMILIES[f] for f in args.families}
    fam_cols = {f: family_columns(spec, names) for f, spec in families.items()}
    fam_cols = {f: cols for f, cols in fam_cols.items() if cols}
    if not fam_cols:
        raise SystemExit(f"none of {list(families)} have columns present")

    all_cols = sorted({c for cols in fam_cols.values() for c in cols})
    df = build(all_cols)

    rows: list[dict] = []
    for family, columns in fam_cols.items():
        spec = families[family]
        print(f"{family}: {columns}")
        for column in columns:
            stats = {
                corpus: panel_stats(df[df.corpus == corpus], column,
                                    spec["machine_is_high"])
                for corpus in PANEL_ORDER
            }
            plot_column(df, column, family, spec, stats, not args.no_thresholds,
                        args.out_dir / f"{column}.png", args.bins)
            for corpus, s in stats.items():
                for cls, pc in s["per_class"].items():
                    rows.append({
                        "corpus": corpus, "family": family, "column": column,
                        "class": cls, **pc,
                        "machine_is_high": spec["machine_is_high"],
                        "cut_human_rephrase": s["cuts"]["rephrase"],
                        "cut_human_rewritten": s["cuts"]["rewritten"],
                        "auroc_human_rephrase": s["aurocs"]["rephrase"],
                        "auroc_human_rewritten": s["aurocs"]["rewritten"],
                    })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "summary.csv"
    pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
