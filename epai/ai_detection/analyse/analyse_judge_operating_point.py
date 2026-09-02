r"""Why an LLM judge loses TPR@1%FPR even when its AUROC is high.

    uv run python epai/ai_detection/analyse/analyse_judge_operating_point.py \
        data/parquet/science_v3_gpt56luna_0811A.parquet \
        data/parquet/ghostbuster_gpt56luna.parquet

Read-only. Touches no score, changes no prompt, re-runs nothing. For every
``llm_judge_score_*`` column in each parquet (``--cols`` to name others), per
domain and pooled, it reports:

  n_distinct       how many distinct values the judge actually emitted. A
                   written probability clumps -- 0.35, 0.75, 0.85 -- and every
                   clump is a group of rows one threshold cannot split.
  tie_floor        sum over values holding BOTH classes of min(#human, #machine),
                   as a fraction of n. The error rate no threshold can beat on
                   this score, purely from ties. If it is 0.08, TPR@1% starts
                   8 points in the hole before overlap or transport enter.
  biggest_tie      the value carrying the most both-class mass, in raw P(human).
  auroc            machine = positive, the number the field reports.
  oracle@L         TPR at a threshold read off the TEST split's own ROC at
                   FPR<=L. The best TPR@L the score allows on those rows,
                   transport aside. Mean over seeds.
  transp@L / fpr   TPR under a threshold calibrated to FPR L on a held-out
                   grouped calibration slice and carried to the same test rows,
                   mean over seeds -- exactly run_clf_in_domain's baseline path
                   -- and the FPR it actually delivered.

Reading, one word for the binding constraint:
  INVERTED    auroc < 0.5 -- the score points the wrong way, nothing else matters
  TIES        most of the gap from a perfect oracle is forced by both-class
              values a threshold cannot enter
  OVERLAP     oracle itself is low and it is not mostly ties -- the judge
              genuinely cannot tell a chunk of machines from humans
  TRANSPORT   oracle is fine but the calibrated threshold does not survive the
              move from calibration slice to test rows
  usable      oracle high, transp close to it

``--machine-is-high`` for a score where a HIGH value means machine
(fastdetect_crit_*, detector_score_*); the judge and Binoculars default (low =
machine) needs nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.cd_core import (
    CAL_FRAC_DEFAULT,
    auroc_score,
    calibrate_thresholds,
    carve_calibration,
    grouped_split,
    level_tag,
    oriented_scores,
    rates_at_threshold,
    roc_point_at_fpr,
)

BASE_DIR = Path(__file__).resolve().parents[3]
JUDGE_PREFIX = "llm_judge_score_"


def tie_stats(raw: np.ndarray, y: np.ndarray) -> dict:
    """Granularity and the irreducible tie error, on the raw score values."""
    vals, inv = np.unique(raw, return_inverse=True)
    floor = 0
    shared = 0
    biggest = None
    biggest_mass = -1
    for k, v in enumerate(vals):
        at = inv == k
        h = int(((y == 0) & at).sum())
        m = int(((y == 1) & at).sum())
        if h and m:
            shared += 1
            floor += min(h, m)
            if h + m > biggest_mass:
                biggest_mass, biggest = h + m, (float(v), h, m)
    return {
        "n_distinct": int(vals.size),
        "n_shared": shared,
        "tie_floor": floor / len(y),
        "biggest_tie": biggest,  # (raw value, n_human, n_machine) or None
    }


def operating_point(oriented: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    levels: tuple[float, ...], seed: int, n_runs: int,
                    test_size: float, cal_frac: float) -> dict:
    """Oracle and transported TPR@L, plus realized FPR, on the SAME test rows.

    The baseline path of run_clf_in_domain: grouped split; carve the calibration
    slice out of the training half; the oracle reads its threshold off the test
    ROC, the transported one off the calibration slice. Both are scored on the
    test rows so the gap between them is only the transport.
    """
    acc: dict[str, list[float]] = {}
    for i in range(n_runs):
        s = seed + i
        tr, te = grouped_split(y, groups, test_size, s)
        _, cal_rel = carve_calibration(y[tr], groups[tr], cal_frac, s)
        cal = tr[cal_rel]
        cal_thr = calibrate_thresholds(y[cal], oriented[cal], levels)
        for lv in levels:
            tag = level_tag(lv)
            acc.setdefault(f"oracle_{tag}", []).append(
                roc_point_at_fpr(y[te], oriented[te], lv)[0])
            tpr, fpr = rates_at_threshold(y[te], oriented[te], cal_thr[tag])
            acc.setdefault(f"transp_{tag}", []).append(tpr)
            acc.setdefault(f"fpr_{tag}", []).append(fpr)
            # The oriented threshold, so the explain plot can draw it on the
            # raw P(human) axis (raw cut = -oriented cut when low means machine).
            acc.setdefault(f"thr_{tag}", []).append(float(cal_thr[tag]))
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def hist_by_class(raw: np.ndarray, y: np.ndarray, bins: int = 10) -> str:
    edges = np.linspace(0.0, 1.0, bins + 1)
    hh, _ = np.histogram(raw[y == 0], bins=edges)
    mm, _ = np.histogram(raw[y == 1], bins=edges)
    labels = [f"{edges[i]:.1f}" for i in range(bins)]
    return (
        "      P(human) " + " ".join(f"{x:>5}" for x in labels) + "\n"
        "      human    " + " ".join(f"{v:>5d}" for v in hh) + "\n"
        "      machine  " + " ".join(f"{v:>5d}" for v in mm)
    )


def reading(auroc: float, tie_floor: float, oracle: float, transp: float) -> str:
    if auroc < 0.5:
        return "INVERTED"
    ceiling_loss = max(1.0 - oracle, 1e-9)
    if tie_floor / ceiling_loss >= 0.5 and tie_floor >= 0.03:
        return "TIES"
    if oracle < 0.6:
        return "OVERLAP"
    if oracle - transp >= 0.10:
        return "TRANSPORT"
    return "usable"


_JUDGE_COLOR = {
    "gpt56luna": "#076b52", "sonnet5": "#d97757", "qwen3_14b": "#6a51a3",
}
_GEN_MARKER = ["o", "s", "^", "D", "v", "P"]


def _short_gen(parquet_name: str) -> str:
    stem = parquet_name.replace(".parquet", "")
    for a, b in (("ghostbuster_gpt35t", "gpt-3.5t"),
                 ("ghostbuster_gpt54mini", "gpt-5.4-mini"),
                 ("ghostbuster_gpt56luna", "gpt-5.6-luna"),
                 ("science_v3_gpt56luna_0811A", "science / luna")):
        if stem.startswith(a):
            return b
    return stem


def plot_gap(rows: list[dict], level_tag_str: str, path: Path) -> None:
    """AUROC (x) vs TPR@L (y): open marker = oracle, filled = transported.

    One (generator, judge) pair per column of the diagnostic, at group 'all'.
    The vertical stem between the two markers is the transport cost; the
    distance of the filled marker below the top of the plot is the whole
    operating-point gap Leg 2 is about.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [r for r in rows if r["group"] == "all"]
    gens = sorted({r["parquet"] for r in pts})
    marker = {g: _GEN_MARKER[i % len(_GEN_MARKER)] for i, g in enumerate(gens)}

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot([0, 1], [0, 1], "--", lw=1, color="#bbbbbb", zorder=1)
    for r in pts:
        judge = r["column"].replace(JUDGE_PREFIX, "").replace("_v3", "")
        c = _JUDGE_COLOR.get(judge, "#555555")
        x = r["auroc"]
        lo, hi = r[f"transp_{level_tag_str}"], r[f"oracle_{level_tag_str}"]
        ax.plot([x, x], [min(lo, hi), max(lo, hi)], "-", lw=1, color=c, alpha=0.5, zorder=2)
        ax.scatter([x], [hi], s=55, facecolors="none", edgecolors=c,
                   marker=marker[r["parquet"]], zorder=3)
        ax.scatter([x], [lo], s=55, color=c, marker=marker[r["parquet"]], zorder=4)

    ax.set_xlabel("AUROC (machine = positive)")
    ax.set_ylabel(f"TPR at {level_tag_str} FPR")
    ax.set_xlim(0.15, 1.02)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.set_title("LLM judge: high AUROC, low deployable TPR@1%FPR")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker="o", ls="", label=j)
               for j, c in _JUDGE_COLOR.items() if any(
                   r["column"].replace(JUDGE_PREFIX, "").replace("_v3", "") == j for r in pts)]
    handles += [Line2D([], [], color="#555", marker=marker[g], ls="", label=_short_gen(g))
                for g in gens]
    handles += [
        Line2D([], [], color="#555", marker="o", mfc="none", ls="", label="oracle (best here)"),
        Line2D([], [], color="#555", marker="o", ls="", label="transported (deployable)"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def plot_explain(rows: list[dict], lv0: str, path: Path) -> None:
    """Two panels that walk through why a high-AUROC judge fails at 1% FPR.

    Left  -- one bar group per (generator, judge): AUROC (what the ROC promises)
             next to the transported TPR@1% (what you can actually deploy). The
             height a judge loses between the two bars is the whole story.
    Right -- the mechanism, for the single widest-gap judge: the human and
             machine P(human) distributions, and the vertical line a 1%-FPR
             threshold has to sit at. Everything left of the line is called
             machine; the machine mass to the RIGHT of it is what TPR@1% misses.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [r for r in rows if r["group"] == "all"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- left: AUROC -> deployable TPR, one dumbbell per (generator, judge)
    live = sorted((r for r in pts if r["auroc"] >= 0.5),
                  key=lambda r: r["auroc"] - r[f"transp_{lv0}"])
    ys = np.arange(len(live))
    for yy, r in zip(ys, live):
        judge = r["column"].replace(JUDGE_PREFIX, "").replace("_v3", "")
        c = _JUDGE_COLOR.get(judge, "#555")
        a, t = r["auroc"], r[f"transp_{lv0}"]
        axL.plot([t, a], [yy, yy], "-", color="#e05353", lw=2, zorder=1)
        axL.scatter([a], [yy], s=70, color="#c9c9c9", edgecolors="#888", zorder=2)
        axL.scatter([t], [yy], s=70, color=c, zorder=3)
        axL.annotate(f"-{a - t:.2f}", (min(t, a) - 0.02, yy), ha="right", va="center",
                     fontsize=7.5, color="#e05353")
    axL.set_yticks(ys)
    axL.set_yticklabels([f"{_short_gen(r['parquet'])} · "
                         f"{r['column'].replace(JUDGE_PREFIX, '').replace('_v3', '')}"
                         for r in live], fontsize=8)
    axL.set_ylim(-0.6, len(live) - 0.4)
    axL.set_xlim(0, 1.05)
    axL.set_xlabel("score")
    axL.set_title(f"AUROC (grey) → deployable TPR@{lv0} FPR (colour)")
    axL.grid(True, axis="x", lw=0.4, alpha=0.4)
    from matplotlib.lines import Line2D
    axL.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#c9c9c9", mec="#888", label="AUROC"),
        Line2D([], [], marker="o", ls="", color="#076b52", label=f"TPR@{lv0} (deployable)"),
    ], fontsize=8, loc="lower left")

    # ---- right: the mechanism, widest non-inverted gap ------------------
    cand = [r for r in pts if r["auroc"] >= 0.5 and r["_h_human"]]
    ex = max(cand, key=lambda r: r["auroc"] - r[f"transp_{lv0}"]) if cand else pts[0]
    edges = np.array(ex["_h_edges"])
    mids = (edges[:-1] + edges[1:]) / 2
    h = np.array(ex["_h_human"], float)
    m = np.array(ex["_h_machine"], float)
    h = h / h.sum() if h.sum() else h
    m = m / m.sum() if m.sum() else m
    axR.step(mids, h, where="mid", color="#888", lw=1.6, label="human")
    axR.fill_between(mids, h, step="mid", color="#888", alpha=0.15)
    axR.step(mids, m, where="mid", color="#076b52", lw=1.6, label="machine")
    axR.fill_between(mids, m, step="mid", color="#076b52", alpha=0.15)
    # oriented cut -> raw P(human) cut (low P(human) means machine here)
    thr_raw = -ex[f"thr_{lv0}"] if not ex["_machine_is_high"] else ex[f"thr_{lv0}"]
    thr_raw = float(np.clip(thr_raw, 0, 1))
    ymax = max(h.max(), m.max()) * 1.15
    axR.set_ylim(0, ymax)
    axR.axvline(thr_raw, color="#e05353", lw=1.6)
    axR.axvspan(0, thr_raw, color="#e05353", alpha=0.06)
    axR.text(thr_raw + 0.015, ymax * 0.94,
             f"1%-FPR cut, P(human) = {thr_raw:.2f}\n"
             f"left of it -> call machine", color="#e05353", fontsize=8, va="top")
    missed = 1.0 - ex[f"transp_{lv0}"]
    axR.text(0.42, ymax * 0.5,
             f"{missed:.0%} of machine documents\nland to the RIGHT of the cut\n"
             f"-> TPR@{lv0} = {ex[f'transp_{lv0}']:.2f}\n"
             f"(AUROC is {ex['auroc']:.2f})",
             fontsize=8.5, color="#076b52")
    exj = ex["column"].replace(JUDGE_PREFIX, "").replace("_v3", "")
    axR.set_title(f"why: {_short_gen(ex['parquet'])} / {exj}  "
                  f"({ex['n_distinct']} distinct score values)")
    axR.set_xlabel("P(human) the judge wrote")
    axR.set_ylabel("share of documents")
    axR.set_xlim(0, 1)
    axR.legend(fontsize=8, loc="upper right", title="the judge's score")
    axR.grid(True, lw=0.4, alpha=0.4)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def analyse_column(df: pd.DataFrame, col: str, group_col: str, human_label: str,
                   machine_is_high: bool, levels: tuple[float, ...],
                   seed: int, n_runs: int, test_size: float,
                   cal_frac: float) -> dict | None:
    sub = df[[col, group_col, "generated_by"]].dropna(subset=[col, group_col])
    raw = sub[col].to_numpy(dtype=float)
    y = (sub["generated_by"].astype(str) != human_label).to_numpy().astype(int)
    groups = sub[group_col].to_numpy()
    if np.unique(y).size < 2 or len(y) < 50:
        return None

    oriented = oriented_scores(raw, machine_is_high)
    row = {
        "column": col,
        "n": int(len(y)),
        "n_human": int((y == 0).sum()),
        "n_machine": int((y == 1).sum()),
        "auroc": auroc_score(y, oriented),
        **tie_stats(raw, y),
    }
    row.update(operating_point(oriented, y, groups, levels, seed, n_runs,
                               test_size, cal_frac))
    row["_hist"] = hist_by_class(raw, y)
    edges = np.linspace(0.0, 1.0, 41)
    row["_h_edges"] = edges.tolist()
    row["_h_human"] = np.histogram(raw[y == 0], bins=edges)[0].tolist()
    row["_h_machine"] = np.histogram(raw[y == 1], bins=edges)[0].tolist()
    row["_machine_is_high"] = bool(machine_is_high)
    lv0 = level_tag(levels[0])
    row["reading"] = reading(row["auroc"], row["tie_floor"],
                             row[f"oracle_{lv0}"], row[f"transp_{lv0}"])
    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquets", nargs="*", type=Path,
                    help="corpora to analyse (omit when using --from-clf-json)")
    ap.add_argument("--cols", nargs="+", default=None,
                    help="score columns (default: every llm_judge_score_* column)")
    ap.add_argument("--machine-is-high", action="store_true",
                    help="set for a score where HIGH means machine")
    ap.add_argument("--by", default=None,
                    help="grouping column (default: 'domain' if present). "
                         "'all' is always reported too.")
    ap.add_argument("--group-col", default="id")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.01, 0.05])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--cal-frac", type=float, default=CAL_FRAC_DEFAULT)
    ap.add_argument("--json", type=Path, default=None, help="also write results here")
    ap.add_argument("--plot", type=Path, default=None,
                    help="write the judges-only AUROC-vs-TPR@L scatter here (.png)")
    ap.add_argument("--explain", type=Path, default=None,
                    help="write the two-panel walk-through figure here (.png)")
    ap.add_argument("--from-clf-json", type=Path, default=None, metavar="JSON",
                    help="a clf_id corpus_comparison.json or a clf_cd "
                         "suite_comparison.json / variant_comparison.json -- draw "
                         "the full Figure 3 (every detector family) to --fig3")
    ap.add_argument("--fig3", type=Path, default=None,
                    help="output for --from-clf-json (default: <json dir>/operating_point.png)")
    ap.add_argument("--hist", action="store_true", help="print the score histogram")
    args = ap.parse_args()

    if args.from_clf_json:
        from epai.utils.operating_point_viz import (
            cd_panels, id_panels, plot_operating_point,
        )
        comp = json.loads(args.from_clf_json.read_text())
        panels = (cd_panels(comp)
                  if any("transfer" in k for v in comp.get("variants", [])[:1] for k in v)
                  else id_panels(comp))
        out = args.fig3 or args.from_clf_json.parent / "operating_point.png"
        plot_operating_point(
            panels, out,
            title=f"{comp.get('alg_label', 'detectors')} — operating-point gap",
            subtitle=comp.get("subtitle", ""),
            level=level_tag(args.levels[0]),
        )
        print(f"wrote {out}")
        return

    if not args.parquets:
        ap.error("give one or more parquets, or use --from-clf-json")

    levels = tuple(args.levels)
    lv0 = level_tag(levels[0])
    all_rows: list[dict] = []

    for pq in args.parquets:
        df = pd.read_parquet(pq)
        cols = args.cols or sorted(c for c in df.columns if c.startswith(JUDGE_PREFIX))
        cols = [c for c in cols if c in df.columns]
        if not cols:
            print(f"\n{pq}: no judge columns")
            continue

        by = args.by or ("domain" if "domain" in df.columns else None)
        groups = [("all", df)]
        if by and by in df.columns:
            groups += [(str(g), d) for g, d in df.groupby(by)]

        print(f"\n{'=' * 104}\n{pq.name}   (grouped by {by or '-'})\n{'=' * 104}")
        print(f"{'group':<16} {'judge':<24} {'n':>6} {'dist':>5} {'tie_fl':>7} "
              f"{'auroc':>7} {'orac@'+lv0:>8} {'trn@'+lv0:>8}{'fpr':>7}  reading   biggest_tie")
        any_off = False
        for gname, gdf in groups:
            for col in cols:
                r = analyse_column(gdf, col, args.group_col, args.human_label,
                                   args.machine_is_high, levels, args.seed,
                                   args.n_runs, args.test_size, args.cal_frac)
                if r is None:
                    continue
                r["parquet"] = pq.name
                r["group"] = gname
                all_rows.append(r)
                bt = r["biggest_tie"]
                bt_s = f"{bt[0]:.2f}: {bt[1]}h/{bt[2]}m" if bt else "-"
                short = col.replace(JUDGE_PREFIX, "").replace("_v3", "")
                # A transported TPR whose realized FPR is far off the target is
                # not really "@L" -- mark it so the number is not read as one.
                off = "*" if r[f"fpr_{lv0}"] > 1.8 * levels[0] else " "
                any_off |= off == "*"
                print(f"{gname:<16} {short:<24} {r['n']:>6} {r['n_distinct']:>5} "
                      f"{r['tie_floor']:>7.3f} {r['auroc']:>7.3f} "
                      f"{r['oracle_'+lv0]:>8.3f} {r['transp_'+lv0]:>7.3f}{off}"
                      f"{r['fpr_'+lv0]:>7.3f}  {r['reading']:<9} {bt_s}")
                if args.hist:
                    print(r["_hist"])
        if any_off:
            print(f"  * realized FPR > {1.8 * levels[0]:.1%}: the transported "
                  f"threshold missed the target, so trn@{lv0} is not at {lv0} FPR "
                  f"-- read it against the fpr column.")

    if args.plot and all_rows:
        plot_gap(all_rows, lv0, args.plot)
    if args.explain and all_rows:
        plot_explain(all_rows, lv0, args.explain)
    if (args.plot or args.explain) and not all_rows:
        print("nothing to plot")

    for r in all_rows:
        for k in ("_hist", "_h_edges", "_h_human", "_h_machine", "_machine_is_high"):
            r.pop(k, None)
    if args.json:
        args.json.write_text(json.dumps(all_rows, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
