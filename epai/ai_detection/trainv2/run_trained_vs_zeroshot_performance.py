r"""Trained classifiers alongside the best zero-shot detector of each family.

    uv run python epai/ai_detection/trainv2/run_trained_vs_zeroshot_performance.py

Extends ``run_best_of_family_performance.py``'s comparison (best judge, best
Binoculars pair, best Fast-DetectGPT pair, radar) with three TRAINED
classifiers, all on ``emb_roberta``:

    roberta_logistic       emb_roberta -> LogisticAlg (plain linear classifier)
    roberta_fusion_cuevec  emb_roberta (LATE fusion) + llm_judge_cuevec_gpt56luna_v3
                            -> FusionMLPAlg
    roberta_gated_cuevec   emb_roberta (GATED fusion) + llm_judge_cuevec_gpt56luna_v3
                            -> GatedMLPAlg

``fusion_mlp`` and ``gated_mlp`` are the two hybrid algorithms this repo's
training configs already use together (see
``cfg/ai_detection/train/clf_id/clf_id_science_v3f.yaml``), there run on
``emb_luar_crud``; this script is the same pairing (embedding + the judge's
cue vector) on ``emb_roberta`` instead, with the same architecture params.

TRAIN/TEST SPLIT. One grouped 70/30 split of ``--science`` by ``id`` (a
document and its human/machine rows never straddle the cut) -- all three
algorithms share the SAME split and the SAME row-missing mask, so a
difference between them is the algorithm, not which rows they happened to
train or test on. The 30% held-out science slice does double duty as the
threshold-calibration source (the only science rows the models never trained
on) AND as the reported "science" test point -- so, like every zero-shot
family's own science reading in this comparison, it is fit and reported on
the same rows, not a further held-out slice. ``--denario`` is evaluated in
full, with no split: it was never trained on, so there is nothing to hold out
-- "from denario there is only test."

ONE seed, not averaged: this is a single illustrative comparison, not a
robustness study. Pass ``--seed`` to try another split.

The zero-shot side is unchanged from ``run_best_of_family_performance.py``:
best-of-family picked once by AUROC on the FULL ``--science`` file, each
winner's ALREADY-CALIBRATED (``calibrate_*_baseline.py``) 1%-FPR threshold
applied to the full science file and to denario. That is why the science
point for the zero-shot families and the science point for the trained
algorithms use different marker shapes here: one reads the full corpus (nothing
was trained on it), the other reads a held-out 30% slice (something was).
Both share one shape for denario, since that reading means the same thing for
either kind of method.

Output:
    <csv>   one row per (series, dataset): auroc, threshold, tpr, fpr, n
    <fig>   the AUROC vs TPR@<fpr-target>%FPR scatter, all seven series

Run:
    uv run python epai/ai_detection/trainv2/run_trained_vs_zeroshot_performance.py
    uv run python epai/ai_detection/trainv2/run_trained_vs_zeroshot_performance.py --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epai.algs import build_algorithm
from epai.utils.cd_core import auroc_score, oriented_scores, rates_at_threshold, roc_point_at_fpr

from .data import _is_missing
from .run_best_of_family_performance import (
    FAMILIES,
    GEN_COL,
    _FAMILY_HUES as _ZEROSHOT_HUES,
    _KIND_BY_KEY,
    evaluate as evaluate_zeroshot,
    pick_best,
)

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SCIENCE = BASE_DIR / "data" / "parquet" / "science_v3_multi_0811A.parquet"
DEFAULT_DENARIO = BASE_DIR / "data" / "parquet" / "denario_hc.parquet"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "trained_vs_zeroshot_performance"

GROUP_COL = "id"
EMB_COL = "emb_roberta"
AUX_COL = "llm_judge_cuevec_gpt56luna_v3"
HUMAN_LABEL = "human"

# The two hybrid architectures cfg/ai_detection/train/clf_id/clf_id_science_v3f.yaml
# runs together on emb_luar_crud + the same cue-vector column -- reused here
# unchanged, only the embedding swapped for emb_roberta. ``cuevec_logistic`` is
# that same config's third feature_sets row (the cue vector ALONE, no
# embedding) -- 'feature_col' is what makes that possible: every spec names
# which column is its X, so a spec need not be built on EMB_COL at all.
TRAINED_SPECS: dict[str, dict] = {
    "roberta_logistic": {
        "label": "RoBERTa + Logistic",
        "alg": "logistic",
        "feature_col": EMB_COL,
        "aux": False,
        "params": {"C": 1.0, "max_iter": 1000},
    },
    "roberta_fusion_cuevec": {
        "label": "RoBERTa ⊕ judge cues (fusion)",
        "alg": "fusion_mlp",
        "feature_col": EMB_COL,
        "aux": True,
        "params": {"pre_dims": [256, 16], "post_dims": [32], "aux_dims": [32]},
    },
    "roberta_gated_cuevec": {
        "label": "RoBERTa + gated judge cues",
        "alg": "gated_mlp",
        "feature_col": EMB_COL,
        "aux": True,
        "params": {"emb_dims": [256, 64], "aux_dims": [32], "gate": "sigmoid"},
    },
    "cuevec_logistic": {
        "label": "Judge cues (11-dim) + Logistic",
        "alg": "logistic",
        "feature_col": AUX_COL,
        "aux": False,
        "params": {"C": 1.0, "max_iter": 1000},
    },
}

_TRAINED_HUES: dict[str, tuple[int, int, int]] = {
    "roberta_logistic": (204, 153, 0),
    "roberta_fusion_cuevec": (219, 39, 119),
    "roberta_gated_cuevec": (20, 150, 130),
    "cuevec_logistic": (153, 101, 21),
}
_ALL_HUES: dict[str, tuple[int, int, int]] = {**_ZEROSHOT_HUES, **_TRAINED_HUES}
_FALLBACK_HUE = (110, 110, 110)

_DATASET_MARKERS = ["s", "^", "D", "v", "P", "*", "h", "X"]


def _hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _light(rgb: tuple[int, int, int]) -> str:
    t = 0.35
    return _hex(tuple(int(255 - (255 - c) * t) for c in rgb))


def _stack(series: pd.Series) -> np.ndarray:
    return np.asarray(np.stack(series.to_numpy()), dtype=np.float32)


def load_features(
    path: Path, need_aux: bool, extra_cols: tuple[str, ...] = ()
) -> pd.DataFrame:
    """``extra_cols`` are read through unchanged (e.g. ``domain``, for a LODO
    split elsewhere) -- they play no part in the missing-row mask below, which
    only ever depends on the feature columns actually fit on."""
    cols = [GROUP_COL, GEN_COL, EMB_COL] + ([AUX_COL] if need_aux else []) + list(extra_cols)
    df = pd.read_parquet(path, columns=list(dict.fromkeys(cols)))
    bad = df[EMB_COL].map(_is_missing)
    if need_aux:
        bad = bad | df[AUX_COL].map(_is_missing)
    if bad.any():
        bad_groups = df.loc[bad, GROUP_COL].unique()
        df = df[~df[GROUP_COL].isin(bad_groups)]
    return df.reset_index(drop=True)


def fit_and_calibrate(
    spec: dict, df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
    seed: int, fpr_target: float,
) -> dict:
    """Fit on ``train_idx``, calibrate a 1%-FPR threshold on ``test_idx``.

    Returns the fitted ``(alg,)`` plus the science-test row (dataset label
    filled in by the caller) and the threshold, so the caller can apply the
    SAME threshold, unchanged, to another corpus (denario).
    """
    y = (df[GEN_COL].astype(str) != HUMAN_LABEL).to_numpy().astype(int)
    groups = df[GROUP_COL].to_numpy()
    X = _stack(df[spec["feature_col"]])
    Z = _stack(df[AUX_COL]) if spec["aux"] else None

    alg = build_algorithm(spec["alg"], seed=seed, **spec["params"])
    alg.set_fit_groups(groups[train_idx])
    if spec["aux"]:
        alg.fit(X[train_idx], y[train_idx], Z[train_idx])
        test_scores = alg.decision_score(X[test_idx], Z[test_idx])
    else:
        alg.fit(X[train_idx], y[train_idx])
        test_scores = alg.decision_score(X[test_idx])

    y_test = y[test_idx]
    # decision_score's own contract is already "higher = machine" -- no
    # further orientation needed (oriented_scores(..., True) is a no-op).
    oriented_test = oriented_scores(test_scores, True)
    tpr, threshold = roc_point_at_fpr(y_test, oriented_test, fpr_target)
    _, fpr_actual = rates_at_threshold(y_test, oriented_test, threshold)
    row = {
        "n": int(len(y_test)), "auroc": auroc_score(y_test, oriented_test),
        "tpr": tpr, "fpr": fpr_actual,
    }
    return {"alg": alg, "threshold": float(threshold), "science_test_row": row}


def evaluate_trained(spec: dict, alg, threshold: float, df: pd.DataFrame) -> dict | None:
    y = (df[GEN_COL].astype(str) != HUMAN_LABEL).to_numpy().astype(int)
    if np.unique(y).size < 2 or len(y) < 10:
        return None
    X = _stack(df[spec["feature_col"]])
    if spec["aux"]:
        Z = _stack(df[AUX_COL])
        scores = alg.decision_score(X, Z)
    else:
        scores = alg.decision_score(X)
    oriented = oriented_scores(scores, True)
    tpr, fpr = rates_at_threshold(y, oriented, threshold)
    return {"n": int(len(y)), "auroc": auroc_score(y, oriented), "tpr": tpr, "fpr": fpr}


def plot(rows: list[dict], fpr_target: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    datasets: list[str] = []
    for r in rows:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])
    series = sorted({r["series"] for r in rows})
    label_of = {r["series"]: r["label"] for r in rows}
    marker_of = {d: _DATASET_MARKERS[i % len(_DATASET_MARKERS)] for i, d in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.axvline(0.5, color="#b5495b", lw=1.0, ls=":", zorder=0, alpha=0.6)
    ax.axhline(fpr_target, color="#b5495b", lw=1.0, ls=":", zorder=0, alpha=0.6)
    ax.text(0.17, fpr_target, f"{fpr_target:g}", color="#b5495b", fontsize=8,
            va="bottom", ha="left", zorder=0)
    ax.scatter([0.5], [fpr_target], s=130, color="#b5495b", lw=1.8, marker="x",
               zorder=0, alpha=0.6)

    for r in rows:
        hue = _ALL_HUES.get(r["series"], _FALLBACK_HUE)
        ax.scatter([r["auroc"]], [r["tpr"]], s=70, color=_light(hue),
                    edgecolors="#666666", lw=0.6, marker=marker_of[r["dataset"]],
                    zorder=3)

    for s in series:
        s_rows = [r for r in rows if r["series"] == s]
        mean_auroc = sum(r["auroc"] for r in s_rows) / len(s_rows)
        mean_tpr = sum(r["tpr"] for r in s_rows) / len(s_rows)
        ax.scatter([mean_auroc], [mean_tpr], s=150, color=_hex(_ALL_HUES.get(s, _FALLBACK_HUE)),
                    edgecolors="#222222", lw=0.9, marker="o", zorder=5)

    ax.set_xlabel("AUROC (machine = positive)")
    ax.set_ylabel(f"TPR at the calibrated {fpr_target * 100:g}%-FPR threshold")
    ax.set_xlim(0.15, 1.02)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.set_title("Zero-shot best-of-family vs. trained RoBERTa classifiers")

    handles = [Line2D([], [], marker="o", ls="", color=_hex(_ALL_HUES.get(s, _FALLBACK_HUE)),
                       label=f"{s} ({label_of[s]}) mean") for s in series]
    handles += [Line2D([], [], marker=marker_of[d], ls="", color="#888888",
                        label=d) for d in datasets]
    handles += [Line2D([], [], marker="x", ls="", color="#b5495b",
                        label=f"chance (0.5, {fpr_target * 100:g}%)")]
    ax.legend(handles=handles, fontsize=7.5, loc="upper left", framealpha=0.9,
              title="colour = series, shape = dataset, o = mean")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--science", type=Path, default=DEFAULT_SCIENCE)
    ap.add_argument("--denario", type=Path, default=DEFAULT_DENARIO)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--fpr-target", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--human-label", default=HUMAN_LABEL)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--fig", type=Path, default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    csv_path = args.csv or DEFAULT_OUT_DIR / "trained_vs_zeroshot_performance.csv"
    fig_path = args.fig or DEFAULT_OUT_DIR / "trained_vs_zeroshot_performance.png"
    rows: list[dict] = []

    # ---- zero-shot side: unchanged from run_best_of_family_performance.py ----
    calib_df = pd.read_parquet(args.science)
    if GEN_COL not in calib_df.columns:
        raise SystemExit(f"{args.science}: no {GEN_COL!r} column")
    denario_df = pd.read_parquet(args.denario)

    print(f"zero-shot: best of each family, by AUROC on {args.science.name} "
          f"({len(calib_df)} rows)\n")
    print(f"{'family':<12}{'winner':<20}{'ref auroc':>10}")
    for family, (kind_key, cfg_path) in FAMILIES.items():
        entry, ref_auroc = pick_best(cfg_path, kind_key, calib_df, args.human_label)
        print(f"{family:<12}{entry['label']:<20}{ref_auroc:>10.3f}")
        kind = _KIND_BY_KEY[kind_key]
        for ds_name, ds_df in ((args.science.stem, calib_df), (args.denario.stem, denario_df)):
            r = evaluate_zeroshot(ds_df, entry["column"], entry["threshold"],
                                   kind.machine_is_high, args.human_label)
            if r is None:
                continue
            rows.append({"series": family, "label": entry["label"], "dataset": ds_name, **r})

    # ---- trained side: emb_roberta, 70/30 split on science, full denario ----
    need_aux = any(s["aux"] for s in TRAINED_SPECS.values())
    sci_df = load_features(args.science, need_aux=True)  # shared rows/mask for all three
    den_df = load_features(args.denario, need_aux=True)

    y_all = (sci_df[GEN_COL].astype(str) != args.human_label).to_numpy().astype(int)
    groups_all = sci_df[GROUP_COL].to_numpy()
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, train_size=args.train_frac, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(np.zeros((len(y_all), 1)), y_all, groups_all))

    science_test_name = f"{args.science.stem} (test {int(round((1 - args.train_frac) * 100))}%)"
    print(f"\ntrained: {len(train_idx)} train / {len(test_idx)} test rows "
          f"({science_test_name}), seed {args.seed}\n")
    print(f"{'series':<24}{'dataset':<32}{'n':>7}{'auroc':>8}{'thr':>10}{'tpr':>8}{'fpr':>8}")

    for name, spec in TRAINED_SPECS.items():
        fit = fit_and_calibrate(spec, sci_df, train_idx, test_idx, args.seed, args.fpr_target)
        rows.append({"series": name, "label": spec["label"], "dataset": science_test_name,
                      **fit["science_test_row"]})
        print(f"{name:<24}{science_test_name:<32}{fit['science_test_row']['n']:>7}"
              f"{fit['science_test_row']['auroc']:>8.3f}{fit['threshold']:>10.3g}"
              f"{fit['science_test_row']['tpr']:>8.3f}{fit['science_test_row']['fpr']:>8.3f}")

        r = evaluate_trained(spec, fit["alg"], fit["threshold"], den_df)
        if r is not None:
            rows.append({"series": name, "label": spec["label"], "dataset": args.denario.stem,
                         "threshold": fit["threshold"], **r})
            print(f"{name:<24}{args.denario.stem:<32}{r['n']:>7}{r['auroc']:>8.3f}"
                  f"{fit['threshold']:>10.3g}{r['tpr']:>8.3f}{r['fpr']:>8.3f}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")

    if not args.no_plot:
        plot(rows, args.fpr_target, fig_path)


if __name__ == "__main__":
    main()
