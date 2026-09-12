r"""Decompose MG-AI-Sci's distribution shift into its two axes -- genre and
generator identity -- by using one MANUGRAM split as an eval-only OOD corpus
for another, instead of only the real multi-agent-science holdout.

Motivation (see ``MODEL_BIAS.md`` §6 and ``ICLR_story.md``): every detector's
MG-AI-Sci collapse conflates two things at once -- new genre (real agentic
research-paper abstracts vs. essay/wp/reuter or CS/math/sociology prompted
text) AND new/undisclosed generator identity (Denario + three agentic
systems never seen in training). MG-AI-Sci alone can't say which axis is
doing the damage, and it's a single n=200 corpus. This script isolates each
axis with a much larger, already-labelled corpus on both sides of the split:

    regime               shift axis   train on   eval-only OOD
    mg_gb_to_mg_sci       genre        mg_gb      mg_sci
    mg_sci_to_mg_gb       genre        mg_sci     mg_gb
    gb_orig_to_mg_gb      generator    gb_orig    mg_gb
    mg_gb_to_gb_orig      generator    mg_gb      gb_orig

``mg_gb`` <-> ``mg_sci`` share the same four generators (gemma431b /
gpt56luna / gptoss120b / llama3370b) over disjoint genres (essay/wp/reuter
vs. CS/math/sociology) -- a genre-only shift, and share ZERO document ids
(checked below). ``gb_orig`` <-> ``mg_gb`` share the same genre family
(essay/wp/reuter on both sides) but ``gb_orig``'s machine class is one
older, different generator (gpt35t) -- meant as a generator-only shift, BUT
the two corpora also share essentially their entire document pool by group
id (~2992/2992 human rows AND the AI side's group id -- ``mg_gb`` is the
same underlying documents as ``gb_orig`` with a modern 4-generator AI class
swapped in for the rebuild, not an independently-sourced corpus). Training
on one and scoring "OOD" on the other in full would score the trained
classifier on rows whose document it already fit on. ``mg_aisci`` shares
zero ids with anything (checked) and needs no such guard.

LEAKAGE GUARD: for a train/OOD pair with shared group ids, the trained
classifier's OOD read (:func:`compute_trained_leak_safe`) excludes, per
seed, every OOD row whose group id fell in THAT seed's own training
partition -- so only documents genuinely never touched during fitting are
scored. Zero-shot judges need no such guard (:func:`compute_zeroshot`,
imported unchanged) -- a judge's score isn't fit on anything, so there is
nothing for it to have memorized. The CSV's ``leak_guarded`` column records
which (regime, split) trained-detector cells this applied to.

Reuses ``run_in_domain_performance.compute_zeroshot`` UNCHANGED, and
``compute_trained_leak_safe`` below is a sibling of that module's
``compute_trained`` (same per-seed grouped-split/fit/eval loop from
``_trained.py`` / ``core.py``), needed only for the leakage guard --
``_trained.py`` / ``core.py`` / ``splits.py`` / the main runners /
``rebuild_all.py`` are only ever imported from, never edited. Deliberately
absent from ``rebuild_all.TASKS``, same treatment as ``run_dr_transfer.py`` /
``run_lodo_performance.py``'s other opt-in siblings.

Each regime trains on exactly ONE split (unlike ``run_in_domain_performance``'s
default of pooling GB-orig/MG-GB/MG-Sci) -- the point here is a clean
single-source-domain transfer reading, not the pooled in-domain baseline.

compute -> CSV -> plot from the CSV:

    results/trainv2/shift_axis_transfer/
        performance.csv              regime, shift_axis, train_split, cross_ood,
                                      + every run_in_domain_performance.CSV_FIELDS column
        <regime>_<fpr-tag>.png       per-regime bar: in-domain vs. cross-split
                                      OOD vs. mg_aisci, one group of bars per detector
        shift_axis_summary_<fpr-tag>.png
                                      one scatter, all regimes pooled: in-domain
                                      TPR@FPR (x) vs. cross-split-OOD TPR@FPR (y),
                                      coloured by shift axis, y=x reference line --
                                      the "no free lunch" figure

Run (sandbox has no ``uv`` -- use ``.venv/bin/python`` directly):

    .venv/bin/python epai/ai_detection/trainv2/run_shift_axis_transfer.py
    .venv/bin/python epai/ai_detection/trainv2/run_shift_axis_transfer.py \
        --regimes mg_gb_to_mg_sci mg_sci_to_mg_gb --seeds 0 1 2 3 4
    .venv/bin/python epai/ai_detection/trainv2/run_shift_axis_transfer.py \
        --replot results/trainv2/shift_axis_transfer/performance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the Agg backend is chosen)
import matplotlib.lines as mlines  # noqa: E402

from epai.ai_detection.trainv2._trained import TRAINED_SPECS, eval_on_fprs, fit, load_features
from epai.ai_detection.trainv2.core import eff_fpr_level, fpr_tag, goodness, grouped_split
from epai.ai_detection.trainv2.plots import _MUTED, _PALETTE, _SURFACE
from epai.ai_detection.trainv2.run_best_zeroshot_performance import (
    DEFAULT_CONFIG as ZEROSHOT_CONFIG,
)
from epai.ai_detection.trainv2.run_in_domain_performance import (
    CSV_FIELDS as BASE_CSV_FIELDS,
    _overall_cell,
    _row,
    _std,
    compute_zeroshot,
)
from epai.ai_detection.trainv2.splits import display as split_display
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "shift_axis_transfer"

#: always scored in every regime alongside its cross-split target, as the
#: real-world reference point where both axes shift at once.
DEFAULT_REAL_WORLD_OOD = "mg_aisci"

#: (regime name, shift axis, train split, cross-split eval-only OOD)
REGIMES: list[tuple[str, str, str, str]] = [
    ("mg_gb_to_mg_sci", "genre", "mg_gb", "mg_sci"),
    ("mg_sci_to_mg_gb", "genre", "mg_sci", "mg_gb"),
    ("gb_orig_to_mg_gb", "generator", "gb_orig", "mg_gb"),
    ("mg_gb_to_gb_orig", "generator", "mg_gb", "gb_orig"),
]
REGIME_BY_NAME = {r[0]: r for r in REGIMES}

CSV_FIELDS = (["regime", "shift_axis", "train_split", "cross_ood", "leak_guarded"]
             + BASE_CSV_FIELDS)

_AXIS_COLOR = {"genre": _PALETTE[0], "generator": _PALETTE[1]}
_HIGHLIGHT_DETECTOR = "judge/gpt56luna_wlp"


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def _group_ids(path: Path, group_col: str) -> set:
    """The full set of ``group_col`` values in a parquet -- cheap (one column,
    no row filtering) and used only to detect id overlap between two splits
    before deciding whether the leakage guard is needed."""
    import pyarrow.parquet as pq
    return set(pq.read_table(path, columns=[group_col])[group_col].to_pylist())


def compute_trained_leak_safe(train_key: str, train_path: Path,
                              ood_specs: list[tuple[str, Path, bool]],
                              seeds: list[int], test_size: float, fprs: list[float],
                              group_col: str, human_label: str, combine: str,
                              ) -> tuple[list[dict], dict[str, bool]]:
    """Sibling of ``run_in_domain_performance.compute_trained``, needed only
    because that function scores every OOD corpus in FULL, with no regard for
    whether a row's group id was already used to fit the model. Fine for a
    pair with zero id overlap (checked, not assumed, via ``ood_specs``'
    ``leak_risk`` flag) -- but ``gb_orig``/``mg_gb`` share essentially their
    whole document pool by group id (same underlying corpus, AI class
    swapped in for the ``mg_gb`` rebuild -- see module docstring). For an
    ``ood_spec`` flagged ``leak_risk=True``, a row is scored only if its
    group id was NOT in that fitting seed's own train partition -- recomputed
    per seed, since the partition is reseeded per seed. Zero-shot judges need
    no such guard; ``compute_zeroshot`` (imported unchanged) handles them.

    ``ood_specs``: ``[(name, path, leak_risk), ...]``. Returns
    ``(rows, leak_guarded)`` where ``leak_guarded[name]`` records whether that
    OOD split's cell actually had the guard applied, for the CSV column."""
    X, y, Z, groups = load_features(train_path, group_col, human_label)

    ood_loaded = {name: (*load_features(path, group_col, human_label), leak_risk)
                 for name, path, leak_risk in ood_specs}
    for name, (_, oy, _, _, leak_risk) in ood_loaded.items():
        tag = "leak-guarded (shared document ids with train)" if leak_risk else "clean"
        print(f"  OOD eval corpus {name!r}: {len(oy)} rows, "
              f"{int((oy == 0).sum())} human -- {tag}")

    cells: dict[str, dict[float, list[dict]]] = {k: {f: [] for f in fprs} for k in TRAINED_SPECS}
    ood_reads: dict[str, dict[str, dict[float, list]]] = {
        k: {n: {f: [] for f in fprs} for n in ood_loaded} for k in TRAINED_SPECS}
    ood_n_human: dict[str, list[int]] = {n: [] for n in ood_loaded}

    for spec_key, spec in TRAINED_SPECS.items():
        a: list[float] = []
        t = {f: [] for f in fprs}
        te_ = {f: [] for f in fprs}
        n_tr = n_te = n_te_h = 0
        for seed in seeds:
            tr, te = grouped_split(y, groups, test_size, seed)
            clf = fit(spec, X, y, Z, groups, tr, seed)
            au, per = eval_on_fprs(clf, spec, X[te], y[te], Z[te], fprs)
            a.append(au)
            for f in fprs:
                t[f].append(per[f][0])
                te_[f].append(per[f][1])
            n_tr, n_te, n_te_h = len(tr), len(te), int((y[te] == 0).sum())

            train_ids = set(groups[tr])
            for name, (oX, oy, oZ, ogroups, leak_risk) in ood_loaded.items():
                keep = ~np.isin(ogroups, list(train_ids)) if leak_risk \
                    else np.ones(len(oy), dtype=bool)
                oau, oper = eval_on_fprs(clf, spec, oX[keep], oy[keep], oZ[keep], fprs)
                ood_n_human[name].append(int((oy[keep] == 0).sum()))
                for f in fprs:
                    ood_reads[spec_key][name][f].append((oau, oper[f][0], oper[f][1]))
        for f in fprs:
            g = [goodness(ai, ei, combine) for ai, ei in zip(a, te_[f])]
            cells[spec_key][f].append({
                "split": train_key, "fpr": f, "n_train": n_tr, "n_test": n_te,
                "n_test_human": n_te_h, "n_seeds": len(seeds),
                "auroc": float(np.mean(a)), "auroc_std": _std(a),
                "tpr_at_fpr": float(np.mean(t[f])), "tpr_at_fpr_std": _std(t[f]),
                "tpr_eff": float(np.mean(te_[f])), "tpr_eff_std": _std(te_[f]),
                "goodness": float(np.mean(g)), "goodness_std": _std(g),
                "eff_fpr": eff_fpr_level(n_te_h, f)})

    rows: list[dict] = []
    leak_guarded: dict[str, bool] = {name: leak_risk for name, (*_, leak_risk) in ood_loaded.items()}
    for spec_key, spec in TRAINED_SPECS.items():
        detector = f"trained/{spec['label']}"
        for f in fprs:
            spec_cells = list(cells[spec_key][f])
            for c in spec_cells:
                rows.append(_row(detector, c["split"], "in_domain", c))
            for name in ood_loaded:
                reads = ood_reads[spec_key][name][f]
                oa = [r[0] for r in reads]
                ot = [r[1] for r in reads]
                ote = [r[2] for r in reads]
                og = [goodness(ai, ei, combine) for ai, ei in zip(oa, ote)]
                n_h_mean = int(round(float(np.mean(ood_n_human[name]))))
                c = {"split": name, "fpr": f, "n_test": "", "n_test_human": n_h_mean,
                     "n_seeds": len(seeds),
                     "auroc": float(np.mean(oa)), "auroc_std": _std(oa),
                     "tpr_at_fpr": float(np.mean(ot)), "tpr_at_fpr_std": _std(ot),
                     "tpr_eff": float(np.mean(ote)), "tpr_eff_std": _std(ote),
                     "goodness": float(np.mean(og)), "goodness_std": _std(og),
                     "eff_fpr": eff_fpr_level(n_h_mean, f)}
                spec_cells.append(c)
                rows.append(_row(detector, name, "ood", c))
            rows.append(_row(detector, "mean", "overall",
                             _overall_cell(spec_cells, len(seeds), f)))
    return rows, leak_guarded


def _tag_rows(rows: list[dict], regime: str, axis: str, train_key: str,
             cross_ood_key: str, leak_guarded: dict[str, bool]) -> list[dict]:
    for r in rows:
        r["regime"] = regime
        r["shift_axis"] = axis
        r["train_split"] = train_key
        r["cross_ood"] = cross_ood_key
        r["leak_guarded"] = (leak_guarded.get(r["split"], False)
                             if r["kind"] == "trained" and r["role"] == "ood" else "")
    return rows


def compute_regime(regime: str, axis: str, train_key: str, cross_ood_key: str,
                   seeds: list[int], test_size: float, fprs: list[float],
                   group_col: str, human_label: str, combine: str,
                   zeroshot_config: Path, no_zeroshot: bool,
                   real_world_ood: str | None) -> list[dict]:
    ood_keys = [cross_ood_key]
    if real_world_ood and real_world_ood != cross_ood_key:
        ood_keys.append(real_world_ood)

    train_splits = resolve_splits([train_key], PARQUET_DIR)
    ood_splits = resolve_splits(ood_keys, PARQUET_DIR)
    missing = [str(p) for _, p in [*train_splits, *ood_splits] if not p.exists()]
    if missing:
        raise SystemExit(f"parquet(s) not found: {missing}")

    train_path = train_splits[0][1]
    train_ids_all = _group_ids(train_path, group_col)
    ood_specs = [(name, path, bool(train_ids_all & _group_ids(path, group_col)))
                for name, path in ood_splits]

    print(f"\n=== {regime} ({axis} shift): train on {train_key!r}, "
          f"OOD eval on {ood_keys} ===")

    rows, leak_guarded = compute_trained_leak_safe(
        train_key, train_path, ood_specs, seeds, test_size, fprs,
        group_col, human_label, combine)
    if not no_zeroshot:
        rows += compute_zeroshot(train_splits, ood_splits, zeroshot_config,
                                 seeds, test_size, fprs, group_col, human_label,
                                 combine)
    return _tag_rows(rows, regime, axis, train_key, cross_ood_key, leak_guarded)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _regime_conditions(train_key: str, cross_ood_key: str,
                       real_world_ood: str | None) -> list[tuple[str, str]]:
    """``(split, legend label)`` pairs, in bar-group order. Matched on
    ``split`` alone, not ``role`` -- ``compute_trained`` labels a training
    split's own held-out reading ``role="in_domain"`` while
    ``compute_zeroshot`` calls the identical thing ``role="eval"``; ``split``
    is the one column both kinds agree on, and it's unambiguous here since
    each of the three split values only ever appears in one slot per regime."""
    conditions = [
        (train_key, f"in-domain ({split_display(train_key)})"),
        (cross_ood_key, f"cross-split OOD ({split_display(cross_ood_key)})"),
    ]
    if real_world_ood and real_world_ood != cross_ood_key:
        conditions.append((real_world_ood,
                          f"real-world OOD ({split_display(real_world_ood)})"))
    return conditions


def plot_regime_bar(df: pd.DataFrame, regime: str, axis: str, train_key: str,
                    cross_ood_key: str, real_world_ood: str | None,
                    fpr: float, out_path: Path) -> None:
    sub = df[(df["regime"] == regime) & (df["fpr"] == fpr) & (df["split"] != "mean")]
    if sub.empty:
        return
    conditions = _regime_conditions(train_key, cross_ood_key, real_world_ood)

    per_cond: dict[str, pd.Series] = {}
    for split_key, label in conditions:
        s = sub[sub["split"] == split_key]
        per_cond[label] = s.set_index("detector")["tpr_at_fpr"]

    in_dom_label = conditions[0][1]
    order = [d for d in per_cond[in_dom_label].sort_values(ascending=False).index
             if all(d in per_cond[label] for _, label in conditions)]
    if not order:
        return

    n = len(order)
    n_cond = len(conditions)
    width = 0.8 / n_cond
    x = np.arange(n)
    colors = [_PALETTE[0], _PALETTE[3], _MUTED]

    fig, ax = plt.subplots(figsize=(max(7.5, n * 0.7), 5), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    for i, (_, label) in enumerate(conditions):
        vals = [per_cond[label].get(d, np.nan) for d in order]
        offset = (i - (n_cond - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=label, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(f"oracle TPR@{fpr * 100:g}%FPR")
    ax.set_title(f"{regime}  ({axis} shift): {split_display(train_key)} → "
                f"{split_display(cross_ood_key)}", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=_SURFACE)
    plt.close(fig)


def plot_shift_summary(df: pd.DataFrame, fpr: float, out_path: Path) -> None:
    """One scatter, every regime pooled: in-domain TPR@FPR (x) vs.
    cross-split-OOD TPR@FPR (y), coloured by shift axis. Points below the
    diagonal are detectors that degrade under that regime's shift -- the
    "no free lunch" figure."""
    sub = df[(df["fpr"] == fpr) & (df["split"] != "mean")]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 7), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.plot([0, 1.02], [0, 1.02], linestyle="--", color=_MUTED, linewidth=1, zorder=1)

    for regime, g in sub.groupby("regime"):
        _, axis, train_key, cross_ood_key = REGIME_BY_NAME[regime]
        # matched on split alone -- see _regime_conditions's docstring for why
        in_dom = g[g["split"] == train_key].set_index("detector")["tpr_at_fpr"]
        ood = g[g["split"] == cross_ood_key].set_index("detector")["tpr_at_fpr"]
        common = in_dom.index.intersection(ood.index)
        if common.empty:
            continue
        ax.scatter(in_dom[common], ood[common], color=_AXIS_COLOR.get(axis, _MUTED),
                  alpha=0.75, s=60, edgecolors="white", linewidths=0.6, zorder=2)
        if _HIGHLIGHT_DETECTOR in common:
            xy = (in_dom[_HIGHLIGHT_DETECTOR], ood[_HIGHLIGHT_DETECTOR])
            ax.annotate(regime, xy, xytext=(5, 4), textcoords="offset points",
                       fontsize=7, color=_AXIS_COLOR.get(axis, _MUTED))

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"in-domain oracle TPR@{fpr * 100:g}%FPR (own held-out test)")
    ax.set_ylabel(f"cross-split OOD oracle TPR@{fpr * 100:g}%FPR")
    ax.set_title("Every detector, every regime: in-domain vs. cross-split OOD\n"
                f"(labels mark {_HIGHLIGHT_DETECTOR}; points below the diagonal "
                "degrade under that shift)", fontsize=10)
    handles = [mlines.Line2D([0], [0], marker="o", linestyle="", color=c, label=a)
              for a, c in _AXIS_COLOR.items()]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=_SURFACE)
    plt.close(fig)


def make_plots(df: pd.DataFrame, out_dir: Path, fprs: list[float],
               real_world_ood: str | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fpr in fprs:
        tag = fpr_tag(fpr)
        for regime in df["regime"].unique():
            _, axis, train_key, cross_ood_key = REGIME_BY_NAME[regime]
            plot_regime_bar(df, regime, axis, train_key, cross_ood_key,
                           real_world_ood, fpr, out_dir / f"{regime}_{tag}.png")
        plot_shift_summary(df, fpr, out_dir / f"shift_axis_summary_{tag}.png")
    print(f"wrote plots to {out_dir}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_table(df: pd.DataFrame) -> None:
    cols = ["kind", "detector", "split", "role", "fpr", "auroc", "tpr_at_fpr"]
    for regime in df["regime"].unique():
        _, axis, train_key, cross_ood_key = REGIME_BY_NAME[regime]
        print(f"\n--- {regime} ({axis} shift: {train_key} -> {cross_ood_key}) ---")
        sub = df[(df["regime"] == regime) & (df["split"] != "mean")][cols].copy()
        sub["auroc"] = sub["auroc"].round(3)
        sub["tpr_at_fpr"] = sub["tpr_at_fpr"].round(3)
        print(sub.to_string(index=False))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--regimes", nargs="+", choices=list(REGIME_BY_NAME),
                    default=list(REGIME_BY_NAME))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--fpr", nargs="+", type=float, default=[0.01], metavar="LEVEL",
                    help="oracle FPR level(s); one figure set per level (default: 0.01)")
    ap.add_argument("--combine", choices=("hmean", "gmean", "amean"), default="hmean")
    ap.add_argument("--group-col", default="id")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--zeroshot-config", type=Path, default=ZEROSHOT_CONFIG)
    ap.add_argument("--no-zeroshot", action="store_true")
    ap.add_argument("--real-world-ood", default=DEFAULT_REAL_WORLD_OOD, metavar="KEY",
                    help="always-included reference OOD split scored in every "
                         f"regime alongside its cross-split target (default: "
                         f"{DEFAULT_REAL_WORLD_OOD!r}; pass '' to disable)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="PERFORMANCE_CSV")
    args = ap.parse_args()

    real_world_ood = args.real_world_ood or None

    if args.replot is not None:
        df = pd.read_csv(args.replot)
        make_plots(df, args.replot.parent, args.fpr, real_world_ood)
        print(f"replotted from {args.replot}")
        return

    print(f"regimes : {args.regimes}")
    print(f"seeds   : {args.seeds}   real-world reference OOD: {real_world_ood}\n")

    all_rows: list[dict] = []
    for name in args.regimes:
        _, axis, train_key, cross_ood_key = REGIME_BY_NAME[name]
        all_rows += compute_regime(
            name, axis, train_key, cross_ood_key, args.seeds, args.test_size,
            args.fpr, args.group_col, args.human_label, args.combine,
            args.zeroshot_config, args.no_zeroshot, real_world_ood,
        )

    df = pd.DataFrame(all_rows, columns=CSV_FIELDS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "performance.csv"
    df.to_csv(csv_path, index=False)
    _print_table(df)
    print(f"\nwrote {csv_path}")

    if not args.no_plots:
        make_plots(df, args.out_dir, args.fpr, real_world_ood)


if __name__ == "__main__":
    main()
