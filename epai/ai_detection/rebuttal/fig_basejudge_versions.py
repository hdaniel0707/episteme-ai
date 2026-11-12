r"""Figure-3-style AUROC vs. TPR@1%FPR scatters of the gpt-5.6-luna BaseJudge
versions, three figures, each on one set of rows:

    auroc_tpr_bj_matched    BaseJudge with LogJudge's call settings (no reasoning) at
                            2 / 4 decimals / no limit, + the paper's BaseJudge, LogJudge
                            and FusionJudge for reference. All 36,148 texts.
    auroc_tpr_bj_decimals   the paper's BaseJudge call at 2 / 4 decimals / no limit, +
                            LogJudge. All 36,148 texts.
    auroc_tpr_bj_ensemble   averaging repeated calls (A4): BaseJudge (paper call) and
                            LogJudge each averaged over 3 calls, FusionJudge from the two
                            means (3+3 calls), vs. the paper's single-call BaseJudge,
                            LogJudge and FusionJudge; all on the ``small`` subset (6,200
                            texts), the only rows with repeats.

Each BaseJudge version is hollow; the FusionJudge built from it (the paper's rule: mean
of BaseJudge and LogJudge) is filled in the same colour and joined to it by a line.
Drawn with the release repo's own ``best_zeroshot.scatter_inset_legend`` (same warp,
iso-lines, markers as Figure 3); the split metrics and four-split mean are the paper's.

    .venv/bin/python -m epai.ai_detection.rebuttal.fig_basejudge_versions

Writes (new files only, nothing in overleaf/):
    <eval>/basejudge_versions.csv
    <eval>/figs/auroc_tpr_bj_{matched,decimals,ensemble}.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from epai.ai_detection.rebuttal import evaluate as ev
from epai.ai_detection.rebuttal.conditions import WORKSPACE

RELEASE_SRC = Path(__file__).resolve().parents[3].parent / "logjudge-manugram" / "src"
sys.path.insert(0, str(RELEASE_SRC))
from logjudge_manugram.plotting import best_zeroshot as bz  # noqa: E402

SPLIT_KEY = {"ghostbuster_orig": "gb_orig", "manugram_gb": "mg_gb",
             "manugram_sci": "mg_sci", "manugram_aiscientist": "mg_aisci"}
FUS = "_fus"   # FusionJudge id = BaseJudge id + FUS (scatter_inset_legend's pairing rule)
LJ = "judge/gpt56luna_ord2"   # "_ord2" = bold in the release style, as in Figure 3
LJ_COLOR = "#1d6fe0"

# Figure spec: detector id -> (frame column, legend label, colour). A BaseJudge id with a
# "<id>_fus" partner is drawn hollow and joined to it (scatter_inset_legend's pairing).
_P = "#4a3aa7"   # the paper's BaseJudge / FusionJudge (Figure 3's FusionJudge purple)
PAPER_PAIR = {
    "base/paper": ("paper_base", "BaseJudge, paper call, 2 dec. (paper)", _P),
    "base/paper_fus": ("paper_fusion", "FusionJudge (paper)", _P),
    LJ: ("paper_lj", "LogJudge (paper)", LJ_COLOR),
}


def _pair(vid: str, col: str, name: str, hex_: str) -> dict:
    return {vid: (col, f"BaseJudge, {name}", hex_),
            vid + FUS: (vid + FUS, f"FusionJudge w/ {name}", hex_)}


FIGURES = {   # name -> (subset, detectors)
    "matched": ("full", {
        **PAPER_PAIR,
        **_pair("base/matched_2dec", "base_v3", "matched call, 2 dec.", "#1baf7a"),
        **_pair("base/matched_4dec", "base_hp4", "matched call, 4 dec.", "#eda100"),
        **_pair("base/matched_free", "base_free", "matched call, no limit", "#e34948"),
    }),
    "decimals": ("full", {
        **PAPER_PAIR,
        **_pair("base/paper_4dec", "base_default_hp4", "paper call, 4 dec.", "#e87ba4"),
        **_pair("base/paper_free", "base_default_free", "paper call, no limit", "#eb6834"),
    }),
    # all on the small subset; the label says how many API calls per text each takes
    "ensemble": ("small", {
        "base/paper": ("paper_base", "BaseJudge (paper), 1 call", _P),
        "base/paper_fus": ("paper_fusion", "FusionJudge (paper), 1+1 calls", _P),
        LJ: ("paper_lj", "LogJudge (paper), 1 call", LJ_COLOR),
        "base/x3": ("base_x3", "BaseJudge, mean of 3 calls", "#eb6834"),
        "base/x3" + FUS: ("fus_x3", "FusionJudge, means of 3+3 calls", "#eb6834"),
        "lj/x3": ("lj_x3", "LogJudge, mean of 3 calls", "#1baf7a"),
    }),
}


def _reps_mean(run: pd.DataFrame, col: str, index) -> pd.Series:
    w = ev.wide(run, col)
    return w[sorted(w.columns)[:3]].mean(axis=1, skipna=False).reindex(index)


def load_frames() -> dict[str, pd.DataFrame]:
    """One frame per subset: y, the paper's columns and every readout on it."""
    master = ev.load_master()
    base = pd.DataFrame(ev.paper_cols(master, "gpt56luna")).join(master["y"])
    full = base.copy()
    for cond in ("base_default_hp4", "base_default_free", "base_v3", "base_hp4", "base_free"):
        r = ev.load_run(WORKSPACE, "gpt56luna", cond, "full")
        if r is None:
            raise SystemExit(f"missing gpt56luna/{cond}__full -- rsync first")
        full = full.join(r.set_index(ev.KEY)["parsed"].rename(cond))
    for vid, (col, _, _) in {**FIGURES["matched"][1], **FIGURES["decimals"][1]}.items():
        if vid.endswith(FUS) and vid != "base/paper_fus":
            base_col = {**FIGURES["matched"][1], **FIGURES["decimals"][1]}[vid[:-len(FUS)]][0]
            full[col] = ev._mean(full[base_col], full["paper_lj"])

    bd = ev.load_run(WORKSPACE, "gpt56luna", "base_default", "small")
    lv = ev.load_run(WORKSPACE, "gpt56luna", "lj_v3", "small")
    if bd is None or lv is None:
        raise SystemExit("missing gpt56luna/base_default__small or lj_v3__small -- rsync first")
    idx = ev.wide(bd, "parsed").index
    small = base.loc[idx].copy()
    small["base_x3"] = _reps_mean(bd, "parsed", idx)
    small["lj_x3"] = _reps_mean(lv, "lj", idx)
    small["fus_x3"] = ev._mean(small["base_x3"], small["lj_x3"])
    for f in (full, small):   # the paper's FusionJudge must come out of the rule exactly
        assert (ev._mean(f["paper_base"], f["paper_lj"]) - f["paper_fusion"]).abs().max() < 1e-12
    return {"full": full.dropna(), "small": small.dropna()}


def metric_rows(f: pd.DataFrame, dets: dict, figure: str, subset: str) -> list[dict]:
    out = []
    for det, (col, _, _) in dets.items():
        per = []
        for stem, key in SPLIT_KEY.items():
            g = f.loc[f.index.get_level_values("stem") == stem]
            m = ev.split_metrics(g["y"].to_numpy(), g[col].to_numpy())
            per.append(m)
            out.append({"figure": figure, "subset": subset, "detector": det, "split": key,
                        "n": m["n"], "auroc": m["auroc"], "tpr_at_fpr": m["tpr"],
                        "athm": m["athm"]})
        mm = ev.mean_row(per)
        out.append({"figure": figure, "subset": subset, "detector": det, "split": "mean",
                    "n": sum(p["n"] for p in per), "auroc": mm["auroc"],
                    "tpr_at_fpr": mm["tpr"], "athm": mm["athm"]})
    return out


def main() -> None:
    out = WORKSPACE / "eval"
    frames = load_frames()
    all_rows = []
    for name, (subset, dets) in FIGURES.items():
        rows = metric_rows(frames[subset], dets, name, subset)
        all_rows += rows
        bz.scatter_inset_legend(rows, out / "figs" / f"auroc_tpr_bj_{name}", key="detector",
                                color={d: c for d, (_, _, c) in dets.items()},
                                label={d: l for d, (_, l, _) in dets.items()},
                                variant_suffix=FUS,
                                emphasize=frozenset({LJ, "base/paper_fus"}),
                                figsize=(6.0, 5.2))
        df = pd.DataFrame(rows)
        print(f"\n{name}: {len(frames[subset]):,} texts ({subset})\n" + df[
            df["split"] == "mean"].set_index("detector")[["auroc", "tpr_at_fpr", "athm"]
                                                         ].round(4).to_string())
    pd.DataFrame(all_rows).to_csv(out / "basejudge_versions.csv", index=False)


if __name__ == "__main__":
    main()
