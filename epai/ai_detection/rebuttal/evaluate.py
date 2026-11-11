r"""Evaluate the rebuttal API runs (TODO block A in ``reviews/review_summary.html``)
against the reviewers' questions. CPU only, no calls. Run ``check.py`` first.

    .venv/bin/python -m epai.ai_detection.rebuttal.evaluate
    .venv/bin/python -m epai.ai_detection.rebuttal.plot        # figures from the CSVs

Every score is recomputed from the sidecars' answers and token windows
(``readouts.py``). Metrics are the paper's: oracle TPR at the largest
achievable FPR <= 1%, AUROC, AT-HM, the four-split mean (AUROC and TPR averaged,
AT-HM of the means), and the tie statistics of ``tie_table``. Every comparison
within one table uses identical rows (rows where all its readouts exist).

Sections, the question each answers, and the CSVs it writes to ``<out>/``:

A1/A2  WpPs-2, xghj-2, xghj Q(i): is the LogJudge gain the readout, or BaseJudge's
       2-decimal output, prompt and call? BaseJudge as in the paper, the same
       prompt called like LogJudge (base_v3), 4 decimals (base_hp4), free
       precision (base_free), and A2 (expected written number, same call) vs
       LogJudge and FusionJudge. luna on ``full``, gemma on ``humans_plus_third``.
         a1_metrics.csv  a1_written_values.csv  a1_decimals.csv
         a1_ev_mass.csv  a1_roc.csv
A3     Lma8 Q3, xghj-5, xghj Q(iii): are decisions at the REAL 1%-FPR thresholds
       stable under repeated calls, and do Fusion > LogJudge > BaseJudge keep
       their sign? The near-threshold rows' scores are replaced by each repeat
       (the rows away from the threshold keep their score), and the full split is
       re-evaluated per repeat. Flip rates use the reference run's 1%-FPR threshold.
         a3_metrics.csv  a3_signs.csv  a3_flips.csv  a3_drift.csv
A4     xghj Q(i): does averaging repeated BaseJudge samples at T=1 close the gap?
         a4_metrics.csv
A5     xghj-2, xghj Q(iii): does LogJudge depend on the label words or their order?
         a5_metrics.csv  a5_agreement.csv

A section whose sidecars are missing is skipped with a note. ``summary.txt``
repeats the printed digest.
"""

from __future__ import annotations

import argparse
import io
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_curve

from epai.ai_detection.rebuttal.conditions import (
    CORE_STEMS,
    HUMAN,
    PARQUET_DIR,
    WORKSPACE,
    _fpr_position,
)
from epai.ai_detection.rebuttal.readouts import (
    FPR,
    SPLIT_NAMES,
    add_readouts,
    first_token_is_label,
    mean_row,
    n_decimals,
    split_metrics,
)
from epai.utils.cd_core import roc_point_at_fpr

# the paper's columns per judge (gemma's FusionJudge is the same mean, built here)
PAPER = {
    "gpt56luna": {"base": "llm_judge_score_gpt56luna_v3", "lj": "logjudge_ord2_gpt56luna_v3",
                  "fusion": "fusionjudge_score"},
    "gemma4-31b": {"base": "llm_judge_score_gemma4_31b_v3",
                   "lj": "logjudge_ord2_gemma4_31b_v3", "fusion": None},
}
A1_SUBSET = {"gpt56luna": "full", "gemma4-31b": "humans_plus_third"}
KEY = ["stem", "uid"]

# readout key -> display label (plot.py uses the labels as written here)
LABELS = {
    "paper_base": "BaseJudge (paper call)",
    "base_v3": "BaseJudge, matched call, 2 dec.",
    "base_hp4": "BaseJudge, 4 dec.",
    "base_free": "BaseJudge, free precision",
    "base_v3_ev": "A2: E[number], 2 dec.",
    "base_hp4_ev": "A2: E[number], 4 dec.",
    "base_free_ev": "A2: E[number], free",
    "paper_lj": "LogJudge (paper)",
    "paper_fusion": "FusionJudge (paper)",
    "fusion_v3": "FusionJudge, matched BaseJudge",
    "t1_x1": "BaseJudge T=1, 1 sample",
    "t1_x2": "BaseJudge T=1, mean of 2",
    "t1_x3": "BaseJudge T=1, mean of 3",
    "lj_swap": "LogJudge, machine listed first",
    "lj_alt": "LogJudge, person / AI",
    "lj_letters": "LogJudge, A=machine / B=human",
    "lj": "LogJudge",
    "base": "BaseJudge (matched call)",
    "fusion": "FusionJudge (matched)",
}


def _mean(*cols: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return pd.Series(np.clip(np.nanmean(np.vstack([c.to_numpy(float) for c in cols]), 0),
                                 0, 1), index=cols[0].index)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_master() -> pd.DataFrame:
    """The four core splits: stem, uid, y (1 = machine) and the paper's judge columns."""
    cols = sorted({c for p in PAPER.values() for c in p.values() if c})
    parts = []
    for stem in CORE_STEMS:
        df = pd.read_parquet(PARQUET_DIR / f"{stem}.parquet",
                             columns=["uid", "generated_by", *cols])
        df.insert(0, "stem", stem)
        parts.append(df)
    m = pd.concat(parts, ignore_index=True)
    m["y"] = (m["generated_by"].astype(str) != HUMAN).astype(int)
    g = PAPER["gemma4-31b"]
    m["gemma_fusion"] = _mean(m[g["base"]], m[g["lj"]])
    return m.set_index(KEY)


def load_run(ws: Path, model: str, cond: str, subset: str) -> pd.DataFrame | None:
    """Answered rows of one sidecar with recomputed readouts, or None if absent."""
    path = ws / model / f"{cond}__{subset}.parquet"
    if not path.is_file():
        print(f"  (no {model}/{cond}__{subset} -- skipped)")
        return None
    df = pd.read_parquet(path)
    df = df[df["answer"].notna()].drop_duplicates(["stem", "uid", "rep"], keep="last")
    df["rep"] = df["rep"].astype(int)
    return add_readouts(df, cond)


def wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """(stem, uid) x rep table of one readout."""
    return df.pivot_table(index=KEY, columns="rep", values=col, aggfunc="first")


def paper_cols(master: pd.DataFrame, model: str) -> dict[str, pd.Series]:
    p = PAPER[model]
    fus = master[p["fusion"]] if p["fusion"] else master["gemma_fusion"]
    return {"paper_base": master[p["base"]], "paper_lj": master[p["lj"]], "paper_fusion": fus}


# ---------------------------------------------------------------------------
# metric tables
# ---------------------------------------------------------------------------


def metric_rows(frame: pd.DataFrame, readouts: list[str], **tags) -> list[dict]:
    """Per split + mean, on the rows where every readout exists."""
    frame = frame.dropna(subset=readouts)
    out = []
    for key in readouts:
        per = []
        for stem in CORE_STEMS:
            g = frame.loc[frame.index.get_level_values("stem") == stem]
            if g.empty or g["y"].nunique() < 2:
                continue
            r = split_metrics(g["y"].to_numpy(), g[key].to_numpy())
            per.append(r)
            out.append({**tags, "readout": key, "label": LABELS.get(key, key),
                        "split": SPLIT_NAMES[stem], **r})
        if per:
            out.append({**tags, "readout": key, "label": LABELS.get(key, key),
                        "split": "mean", **mean_row(per)})
    return out


def print_table(rows: list[dict], cols=("auroc", "tpr", "athm", "n_distinct_human",
                                        "tie_humans")) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    mean = df[df["split"] == "mean"].set_index("label")[["auroc", "tpr", "athm"]]
    per = df[df["split"] != "mean"].groupby("label", sort=False)
    extra = per[[c for c in cols if c in df and c not in ("auroc", "tpr", "athm")]].median()
    t = mean.join(extra.add_suffix(" (median split)"))
    n = df[df["split"] != "mean"].groupby("label", sort=False)["n"].sum()
    t["rows"] = n
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.4f}".format):
        print(t.to_string())


def roc_rows(frame: pd.DataFrame, readouts: list[str], model: str, max_fpr=0.05) -> list[dict]:
    """ROC points up to (and one past) ``max_fpr``. A tie group shows as one
    straight diagonal segment, so the figure shows the ties directly."""
    out = []
    frame = frame.dropna(subset=readouts)
    for stem in CORE_STEMS:
        g = frame.loc[frame.index.get_level_values("stem") == stem]
        if g.empty or g["y"].nunique() < 2:
            continue
        for key in readouts:
            fpr, tpr, _ = roc_curve(g["y"], -g[key].to_numpy(float), drop_intermediate=False)
            stop = int(np.searchsorted(fpr, max_fpr, side="right")) + 1
            out += [{"model": model, "readout": key, "label": LABELS[key],
                     "split": SPLIT_NAMES[stem], "fpr": f, "tpr": t}
                    for f, t in zip(fpr[:stop], tpr[:stop])]
    return out


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def section_a1(master, ws, out) -> None:
    print("\n" + "=" * 100 + "\nA1/A2  Is the gain the readout, or BaseJudge's precision / prompt / "
          "call?  (WpPs-2, xghj-2, xghj Q(i))\n" + "=" * 100)
    metrics, values, decs, masses, rocs = [], [], [], [], []
    for model, subset in A1_SUBSET.items():
        print(f"\n--- {model}, subset {subset} ---")
        frame = pd.DataFrame(paper_cols(master, model)).join(master["y"])
        runs = {}
        for cond in ("base_v3", "base_hp4", "base_free"):
            r = load_run(ws, model, cond, subset)
            if r is None:
                continue
            runs[cond] = r
            r = r.set_index(KEY)
            frame = frame.join(r["parsed"].rename(cond)).join(r["ev"].rename(cond + "_ev"))
            for split, g in r.groupby("stem"):
                vc = g["answer"].str.strip().value_counts()   # as written: "0.10" != "0.1"
                values += [{"model": model, "readout": cond, "split": SPLIT_NAMES[split],
                            "value": v, "count": int(c), "share": c / len(g), "rank": i + 1}
                           for i, (v, c) in enumerate(vc.head(20).items())]
            dc = r["answer"].map(n_decimals).value_counts(normalize=True)
            last = r["answer"].str.strip().str[-1].value_counts(normalize=True)
            decs += [{"model": model, "readout": cond, "what": "n_decimals", "key": k,
                      "share": s} for k, s in dc.items()]
            decs += [{"model": model, "readout": cond, "what": "last_digit", "key": k,
                      "share": s} for k, s in last.items()]
            m = r["mass"].dropna()
            masses.append({"model": model, "readout": cond, "n": len(m),
                           **{f"q{int(q * 100):02d}": m.quantile(q)
                              for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                           "share_below_0.5": (m < 0.5).mean(), "mean": m.mean()})
        if not runs:
            continue
        frame = frame.loc[frame.index.isin(next(iter(runs.values())).set_index(KEY).index)]
        if "base_v3" in runs:
            frame["fusion_v3"] = _mean(frame["base_v3"], frame["paper_lj"])
        readouts = [k for k in ("paper_base", "base_v3", "base_hp4", "base_free", "base_v3_ev",
                                "base_hp4_ev", "base_free_ev", "paper_lj", "paper_fusion",
                                "fusion_v3") if k in frame]
        rows = metric_rows(frame, readouts, experiment="A1A2", model=model, subset=subset)
        metrics += rows
        print_table(rows)
        rocs += roc_rows(frame, readouts, model)
        mm = pd.DataFrame(masses)
        if not mm.empty:
            mm = mm[mm["model"] == model]
            print("\n  A2 digit mass captured at the first fractional position (median, "
                  "share < 0.5): " + ", ".join(f"{r.readout} {r.q50:.3f} / {r['share_below_0.5']:.1%}"
                                              for _, r in mm.iterrows()))
    _write(out, "a1_metrics", metrics)
    _write(out, "a1_written_values", values)
    _write(out, "a1_decimals", decs)
    _write(out, "a1_ev_mass", masses)
    _write(out, "a1_roc", rocs)


def _near_groups(master: pd.DataFrame) -> pd.Series:
    """near_human / near_machine / control, by the rule that built the ``near``
    subset (conditions.build_subset): rows near the paper's luna LogJudge or
    FusionJudge 1%-FPR threshold."""
    grp = pd.Series("control", index=master.index)
    for stem in CORE_STEMS:
        sel = master.index.get_level_values("stem") == stem
        g = master.loc[sel]
        gh = (g["y"] == 0).to_numpy()
        nh = np.zeros(len(g), bool)
        nm = np.zeros(len(g), bool)
        for col in ("logjudge_ord2_gpt56luna_v3", "fusionjudge_score"):
            pos = _fpr_position(g[col].to_numpy(), gh)
            nh |= gh & (pos <= 0.05)
            nm |= ~gh & (pos >= 0.003) & (pos <= 0.03)
        lab = np.where(nh, "near_human", np.where(nm, "near_machine", "control"))
        grp.loc[sel] = lab
    return grp


def section_a3(master, ws, out) -> None:
    print("\n" + "=" * 100 + "\nA3  Repeated calls at the real 1%-FPR threshold "
          "(Lma8 Q3, xghj-5, xghj Q(iii))\n" + "=" * 100)
    lj = load_run(ws, "gpt56luna", "lj_v3", "near")
    bn = load_run(ws, "gpt56luna", "base_v3", "near")
    bf = load_run(ws, "gpt56luna", "base_v3", "full")
    if lj is None or bn is None or bf is None:
        return
    ljw, bnw = wide(lj, "lj"), wide(bn, "parsed")
    reps = sorted(set(ljw.columns) & set(bnw.columns))
    near_idx = ljw.index.intersection(bnw.index)
    full = master[["y", "logjudge_ord2_gpt56luna_v3"]].rename(
        columns={"logjudge_ord2_gpt56luna_v3": "lj"})
    full = full.join(bf.set_index(KEY)["parsed"].rename("base")).dropna()
    near_idx = near_idx.intersection(full.index)
    group = _near_groups(master).loc[near_idx]

    # one full-split score vector per (readout, rep); "ref" = the stored / full run
    vecs = {"ref": full.copy()}
    for r in reps:
        v = full.copy()
        v.loc[near_idx, "lj"] = ljw.loc[near_idx, r]
        v.loc[near_idx, "base"] = bnw.loc[near_idx, r]
        vecs[r] = v
    for v in vecs.values():
        v["fusion"] = _mean(v["base"], v["lj"])

    metrics = []
    for rep, v in vecs.items():
        metrics += metric_rows(v, ["base", "lj", "fusion"], experiment="A3",
                               rep=str(rep))
    _write(out, "a3_metrics", metrics)
    m = pd.DataFrame(metrics)
    mean = m[m["split"] == "mean"].pivot_table(index="rep", columns="readout", values="athm")
    signs = pd.DataFrame({"rep": mean.index, "fusion_minus_lj": mean["fusion"] - mean["lj"],
                          "lj_minus_base": mean["lj"] - mean["base"]})
    _write(out, "a3_signs", signs.to_dict("records"))
    print("\n  four-split mean AT-HM per repeat (ref = paper LogJudge + matched base_v3 full run):")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(mean.join(signs.set_index("rep")).to_string())

    # flips at the reference run's 1%-FPR threshold, and at 0.5
    flips = []
    for key in ("base", "lj", "fusion"):
        for stem in CORE_STEMS:
            sel = full.index.get_level_values("stem") == stem
            ref = vecs["ref"].loc[sel]
            _, thr = roc_point_at_fpr(ref["y"].to_numpy(), -ref[key].to_numpy(), FPR)
            idx = near_idx[near_idx.get_level_values("stem") == stem]
            dec = np.column_stack([-vecs[r].loc[idx, key].to_numpy() >= thr for r in reps])
            dec05 = np.column_stack([vecs[r].loc[idx, key].to_numpy() < 0.5 for r in reps])
            ref_dec = -vecs["ref"].loc[idx, key].to_numpy() >= thr
            g = group.loc[idx].to_numpy()
            for gname in ("near_human", "near_machine", "control", "all"):
                k = np.ones(len(idx), bool) if gname == "all" else g == gname
                if not k.any():
                    continue
                flips.append({
                    "readout": key, "label": LABELS[key], "split": SPLIT_NAMES[stem],
                    "group": gname, "n": int(k.sum()), "threshold": -thr,
                    "flip_rate": float((dec[k].min(1) != dec[k].max(1)).mean()),
                    "vs_ref_rate": float((dec[k] != ref_dec[k, None]).mean()),
                    "flip_rate_at_0.5": float((dec05[k].min(1) != dec05[k].max(1)).mean()),
                })
    _write(out, "a3_flips", flips)
    f = pd.DataFrame(flips)
    if not f.empty:
        print("\n  decision flip rate across repeats at the 1%-FPR threshold "
              "(rows pooled over splits; weighted by n):")
        f["w"] = f["flip_rate"] * f["n"]
        f["w05"] = f["flip_rate_at_0.5"] * f["n"]
        agg = f.groupby(["label", "group"]).agg(n=("n", "sum"), w=("w", "sum"), w05=("w05", "sum"))
        agg["flip @1%FPR"] = agg["w"] / agg["n"]
        agg["flip @0.5"] = agg["w05"] / agg["n"]
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(agg[["n", "flip @1%FPR", "flip @0.5"]].to_string())

    # drift: the repeats against the paper's stored LogJudge on the same rows
    drift = []
    paper = master.loc[near_idx, "logjudge_ord2_gpt56luna_v3"]
    for r in reps:
        new = ljw.loc[near_idx, r]
        d = (new - paper).abs()
        for stem in CORE_STEMS + ["all"]:
            k = np.ones(len(d), bool) if stem == "all" else \
                (d.index.get_level_values("stem") == stem)
            if k.sum() < 3:
                continue
            drift.append({"rep": r, "split": SPLIT_NAMES.get(stem, "all"), "n": int(k.sum()),
                          "pearson": pearsonr(new[k], paper[k])[0],
                          "spearman": spearmanr(new[k], paper[k])[0],
                          "mean_abs_diff": d[k].mean(), "median_abs_diff": d[k].median(),
                          "share_abs_diff_gt_0.01": (d[k] > 0.01).mean()})
    _write(out, "a3_drift", drift)
    dd = pd.DataFrame(drift)
    if not dd.empty:
        a = dd[dd["split"] == "all"]
        print("\n  model drift, lj_v3 repeats vs the paper's stored LogJudge (near rows): "
              + "; ".join(f"rep {r.rep}: Pearson {r.pearson:.4f}, median |diff| "
                          f"{r.median_abs_diff:.4f}" for _, r in a.iterrows()))


def section_a4(master, ws, out) -> None:
    print("\n" + "=" * 100 + "\nA4  Averaged repeated BaseJudge at T=1 (xghj Q(i))\n" + "=" * 100)
    t1 = load_run(ws, "gpt56luna", "base_v3_t1", "small")
    if t1 is None:
        return
    w = wide(t1, "parsed")
    reps = sorted(w.columns)
    frame = pd.DataFrame(paper_cols(master, "gpt56luna")).join(master["y"]).loc[w.index]
    for k in range(1, len(reps) + 1):
        frame[f"t1_x{k}"] = w[reps[:k]].mean(axis=1)
    bf = load_run(ws, "gpt56luna", "base_v3", "full")
    if bf is not None:
        frame = frame.join(bf.set_index(KEY)["parsed"].rename("base_v3"))
    readouts = [c for c in ("paper_base", "base_v3", *[f"t1_x{k}" for k in
                range(1, len(reps) + 1)], "paper_lj", "paper_fusion") if c in frame]
    rows = metric_rows(frame, readouts, experiment="A4", model="gpt56luna", subset="small")
    for r in rows:
        r["calls"] = int(r["readout"][-1]) if r["readout"].startswith("t1_x") else 1
    print_table(rows)
    _write(out, "a4_metrics", rows)


def section_a5(master, ws, out) -> None:
    print("\n" + "=" * 100 + "\nA5  Label words and order (xghj-2, xghj Q(iii))\n" + "=" * 100)
    frame = pd.DataFrame(paper_cols(master, "gpt56luna")).join(master["y"])
    have, extra = [], []
    for cond in ("lj_swap", "lj_alt", "lj_letters"):
        r = load_run(ws, "gpt56luna", cond, "small")
        if r is None:
            continue
        have.append(cond)
        ri = r.set_index(KEY)
        frame = frame.join(ri["lj"].rename(cond))
        extra.append({"readout": cond,
                      "first_token_is_label": float(np.mean(
                          [first_token_is_label(cond, t) for t in r["tokens"]])),
                      "median_label_mass": float(ri["label_mass"].median())})
    if not have:
        return
    frame = frame.dropna(subset=have)
    bf = load_run(ws, "gpt56luna", "base_v3", "full")
    if bf is not None:
        frame = frame.join(bf.set_index(KEY)["parsed"].rename("base_v3"))
    readouts = [c for c in ("paper_base", "base_v3", "paper_lj", *have) if c in frame]
    rows = metric_rows(frame, readouts, experiment="A5", model="gpt56luna", subset="small")
    print_table(rows)
    _write(out, "a5_metrics", rows)

    agree = []
    for cond in have:
        for stem in CORE_STEMS:
            g = frame.loc[frame.index.get_level_values("stem") == stem].dropna(
                subset=["paper_lj", cond])
            if g.empty:
                continue
            y = g["y"].to_numpy()
            _, ta = roc_point_at_fpr(y, -g["paper_lj"].to_numpy(), FPR)
            _, tb = roc_point_at_fpr(y, -g[cond].to_numpy(), FPR)
            fa, fb = -g["paper_lj"].to_numpy() >= ta, -g[cond].to_numpy() >= tb
            union = (fa | fb).sum()
            agree.append({"readout": cond, "label": LABELS[cond], "split": SPLIT_NAMES[stem],
                          "n": len(g), "spearman_vs_paper_lj":
                              spearmanr(g["paper_lj"], g[cond])[0],
                          "decision_agreement": float((fa == fb).mean()),
                          "flagged_jaccard": float((fa & fb).sum() / union) if union else np.nan})
    for e in extra:
        agree.append({"readout": e["readout"], "label": LABELS[e["readout"]], "split": "all",
                      **{k: v for k, v in e.items() if k != "readout"}})
    _write(out, "a5_agreement", agree)
    a = pd.DataFrame(agree)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print("\n  agreement with the paper's LogJudge (each at its own 1%-FPR threshold "
              "on these rows):")
        print(a[a["split"] != "all"].groupby("label")[["spearman_vs_paper_lj",
              "decision_agreement", "flagged_jaccard"]].mean().to_string())
        print(a[a["split"] == "all"][["label", "first_token_is_label",
                                      "median_label_mass"]].to_string(index=False))


def _write(out: Path, name: str, rows) -> None:
    if len(rows):
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", type=Path, default=WORKSPACE,
                    help="folder holding <model>/<condition>__<subset>.parquet")
    ap.add_argument("--out", type=Path, default=None, help="default: <workspace>/eval")
    ap.add_argument("--only", nargs="+", choices=["a1", "a3", "a4", "a5"], default=None)
    args = ap.parse_args()
    out = args.out or args.workspace / "eval"
    out.mkdir(parents=True, exist_ok=True)

    master = load_master()
    buf = io.StringIO()
    with redirect_stdout(buf):
        for name, fn in (("a1", section_a1), ("a3", section_a3), ("a4", section_a4),
                         ("a5", section_a5)):
            if args.only is None or name in args.only:
                fn(master, args.workspace, out)
    print(buf.getvalue())
    (out / "summary.txt").write_text(buf.getvalue())
    print(f"wrote CSVs and summary.txt to {out}")


if __name__ == "__main__":
    main()
