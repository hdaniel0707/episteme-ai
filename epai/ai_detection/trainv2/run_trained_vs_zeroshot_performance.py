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
on) AND as the reported "science" test point. ``--denario`` is evaluated in
full, with no split: it was never trained on, so there is nothing to hold out
-- "from denario there is only test."

MEAN +/- STD OVER SEEDS (``--seeds``, default 3: 42/43/44). Each seed draws its
own 70/30 split AND its own model fit, so the trained side's std mixes split
variance and training-run variance rather than isolating either.

THE ZERO-SHOT SIDE GETS A STD TOO NOW, AND ONLY ON SCIENCE. A zero-shot
detector's score never changes -- there is no model fit to vary by seed -- so
its "science" reading is instead evaluated on the SAME per-seed 30% test_idx
the trained models use, one reading per seed. That is real variance (a
different random subset of science each time), not training noise, but it is
the same *kind* of quantity the trained side's std is: "how much does this
number move on a different draw", not the earlier (wrong) version of this
script, which took the std of exactly two numbers -- one science reading, one
denario reading -- and called that a std. Denario still gets no std for
either kind of method: nothing ever splits it, so every reading of it is the
same deterministic number.

Because of this, the "science" point now means the same thing for every
series here -- the held-out 30% slice -- so zero-shot and trained series share
one marker shape for it, same as they already shared one for denario.

Output, three levels in one CSV (a ``level`` column tells them apart):
    run           one row per (series, dataset, seed) -- the individual reads
                  the two means below are built from. A zero-shot detector's
                  denario reading has none to average (seed is ``None``,
                  single deterministic row); its science reading has one per
                  seed, same as a trained series.
    dataset_mean  one row per (series, dataset): mean +/- std over seeds.
                  Present for EVERY series' science reading now; absent for
                  denario, which has only the one deterministic run to show.
    overall_mean  one row per series: mean +/- std OF the (up to two)
                  per-dataset numbers feeding it (a dataset_mean where one
                  exists, the lone run otherwise) -- "the mean of the
                  datasets", what the figure draws as the solid circle and its
                  error bar.
    <fig>   the AUROC vs TPR@<fpr-target>%FPR scatter -- SAME numbers as the
            CSV, nothing computed only for one or the other: light shape-coded
            points from dataset_mean (with its own std, where it has one) or
            the lone run (no std to draw), the solid circle + its own error
            bar from overall_mean

Run:
    uv run python epai/ai_detection/trainv2/run_trained_vs_zeroshot_performance.py
    uv run python epai/ai_detection/trainv2/run_trained_vs_zeroshot_performance.py --seeds 7 8 9
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
DEFAULT_SCIENCE = BASE_DIR / "data" / "parquet" / "science_v3f_multi_0811A.parquet"
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
    path: Path, need_aux: bool, extra_cols: tuple[str, ...] = (),
    all_columns: bool = False,
) -> pd.DataFrame:
    """``extra_cols`` are read through unchanged (e.g. ``domain``, for a LODO
    split elsewhere) -- they play no part in the missing-row mask below, which
    only ever depends on the feature columns actually fit on.

    ``all_columns=True`` reads the whole parquet instead of a curated subset --
    every zero-shot detector column comes along too, so the SAME frame (and
    its row order, and any ``iloc`` split into it) serves both the trained
    fits and a zero-shot reading evaluated on the identical rows. Without it,
    a separately-loaded zero-shot frame is not guaranteed to agree row-for-row
    with this one wherever missingness differs.
    """
    if all_columns:
        df = pd.read_parquet(path)
    else:
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


def mean_row(reads: list[dict]) -> dict:
    """Mean +/- std over repeated reads of one (algorithm, corpus) cell --
    different seeds, different LODO folds, or both stacked together. ``n`` is
    the mean read size, not a sum: these are repeated reads of the same rows
    (or, for LODO, disjoint per-domain slices), never a count to add up."""
    return {
        "n": int(round(float(np.mean([r["n"] for r in reads])))),
        "auroc": float(np.mean([r["auroc"] for r in reads])),
        "auroc_std": float(np.std([r["auroc"] for r in reads])),
        "tpr": float(np.mean([r["tpr"] for r in reads])),
        "tpr_std": float(np.std([r["tpr"] for r in reads])),
        "fpr": float(np.mean([r["fpr"] for r in reads])),
        "fpr_std": float(np.std([r["fpr"] for r in reads])),
        "n_reads": len(reads),
    }


def plot(rows: list[dict], fpr_target: float, path: Path) -> None:
    """``rows`` carries three levels (see ``main``): ``run`` (one per seed /
    fold), ``dataset_mean`` (mean +/- std over those, one per series x
    dataset), ``overall_mean`` (mean +/- std OF the dataset means, one per
    series -- "the mean of the datasets"). Only the last two are drawn: the
    light, shape-coded point per dataset is ``dataset_mean`` where it exists
    (trained series) or the lone ``run`` where it does not (zero-shot has no
    seeds to average, so its single reading already IS that point); the std
    bar sits on the solid circle, from ``overall_mean``, not on the per-dataset
    points -- that circle, not any one dataset, is "the mean" this figure's
    error bars answer to.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # One display point per (series, dataset): dataset_mean if present, else
    # the lone run.
    display: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r["level"] not in ("run", "dataset_mean"):
            continue
        key = (r["series"], r["dataset"])
        if key not in display or r["level"] == "dataset_mean":
            display[key] = r
    display_rows = list(display.values())
    overall_rows = [r for r in rows if r["level"] == "overall_mean"]

    datasets: list[str] = []
    for r in display_rows:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])
    series = sorted({r["series"] for r in display_rows})
    label_of = {r["series"]: r["label"] for r in display_rows}
    marker_of = {d: _DATASET_MARKERS[i % len(_DATASET_MARKERS)] for i, d in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.axvline(0.5, color="#b5495b", lw=1.0, ls=":", zorder=0, alpha=0.6)
    ax.axhline(fpr_target, color="#b5495b", lw=1.0, ls=":", zorder=0, alpha=0.6)
    ax.text(0.17, fpr_target, f"{fpr_target:g}", color="#b5495b", fontsize=8,
            va="bottom", ha="left", zorder=0)
    ax.scatter([0.5], [fpr_target], s=130, color="#b5495b", lw=1.8, marker="x",
               zorder=0, alpha=0.6)

    for r in display_rows:
        hue = _ALL_HUES.get(r["series"], _FALLBACK_HUE)
        color = _light(hue)
        # Std over seeds/domains, when this point is a dataset_mean -- the
        # CSV's dataset_mean rows carry it, so the figure must draw it too.
        # Absent for a lone 'run' (a single deterministic zero-shot denario
        # reading has nothing to have a std of).
        if r.get("auroc_std") is not None:
            ax.errorbar([r["auroc"]], [r["tpr"]], xerr=r["auroc_std"], yerr=r["tpr_std"],
                        fmt="none", ecolor=color, elinewidth=1.0, capsize=2.5,
                        zorder=2, alpha=0.9)
        ax.scatter([r["auroc"]], [r["tpr"]], s=70, color=color,
                    edgecolors="#666666", lw=0.6, marker=marker_of[r["dataset"]],
                    zorder=3)

    for r in overall_rows:
        hue = _hex(_ALL_HUES.get(r["series"], _FALLBACK_HUE))
        ax.errorbar([r["auroc"]], [r["tpr"]], xerr=r.get("auroc_std"), yerr=r.get("tpr_std"),
                    fmt="none", ecolor=hue, elinewidth=1.3, capsize=4, zorder=4, alpha=0.9)
        ax.scatter([r["auroc"]], [r["tpr"]], s=150, color=hue,
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
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                     help="one grouped split + fit per seed; reported numbers are "
                          "mean +/- std over these (default: 3 seeds)")
    ap.add_argument("--human-label", default=HUMAN_LABEL)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--fig", type=Path, default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    csv_path = args.csv or DEFAULT_OUT_DIR / "trained_vs_zeroshot_performance.csv"
    fig_path = args.fig or DEFAULT_OUT_DIR / "trained_vs_zeroshot_performance.png"
    rows: list[dict] = []

    # ---- ONE science frame for everything: zero-shot picks/reads AND trained
    # fits/reads, so an iloc split into it means the same rows for either side.
    sci_df = load_features(args.science, need_aux=True, all_columns=True)
    if GEN_COL not in sci_df.columns:
        raise SystemExit(f"{args.science}: no {GEN_COL!r} column")
    denario_full = pd.read_parquet(args.denario)  # zero-shot denario: full file, no filter
    den_df = load_features(args.denario, need_aux=True)  # trained denario: emb/aux-filtered

    y_all = (sci_df[GEN_COL].astype(str) != args.human_label).to_numpy().astype(int)
    groups_all = sci_df[GROUP_COL].to_numpy()
    from sklearn.model_selection import GroupShuffleSplit
    # One split per seed -- a different seed means a different train/test cut
    # AND a different model init/training run, so the std below reflects both
    # sources of noise at once, not training stochasticity alone. The
    # zero-shot side reuses the same test_idx (it has nothing to train, so
    # only the "which rows" half of this applies to it).
    splits = {}
    for seed in args.seeds:
        splitter = GroupShuffleSplit(n_splits=1, train_size=args.train_frac, random_state=seed)
        splits[seed] = next(splitter.split(np.zeros((len(y_all), 1)), y_all, groups_all))

    science_test_name = f"{args.science.stem} (test {int(round((1 - args.train_frac) * 100))}%)"

    # ---- zero-shot side. Science: same per-seed test_idx as the trained
    # models below, so its std means the same thing (evaluation-sample
    # variance) instead of "distance between two arbitrary corpora". Denario:
    # a single deterministic reading, since nothing here ever splits it.
    print(f"zero-shot: best of each family, by AUROC on {args.science.name} "
          f"({len(sci_df)} rows)\n")
    print(f"{'family':<12}{'winner':<20}{'ref auroc':>10}")
    for family, (kind_key, cfg_path) in FAMILIES.items():
        entry, ref_auroc = pick_best(cfg_path, kind_key, sci_df, args.human_label)
        print(f"{family:<12}{entry['label']:<20}{ref_auroc:>10.3f}")
        kind = _KIND_BY_KEY[kind_key]

        sci_reads: list[dict] = []
        for seed in args.seeds:
            _, test_idx = splits[seed]
            r = evaluate_zeroshot(sci_df.iloc[test_idx], entry["column"], entry["threshold"],
                                   kind.machine_is_high, args.human_label)
            if r is None:
                continue
            sci_reads.append(r)
            rows.append({"series": family, "label": entry["label"], "dataset": science_test_name,
                          "level": "run", "seed": seed, **r})
        sci_mean = mean_row(sci_reads) if sci_reads else None
        if sci_mean:
            rows.append({"series": family, "label": entry["label"], "dataset": science_test_name,
                          "level": "dataset_mean", "seed": None,
                          **{k: v for k, v in sci_mean.items() if k != "n_reads"},
                          "n_reads": sci_mean["n_reads"]})

        den_r = evaluate_zeroshot(denario_full, entry["column"], entry["threshold"],
                                   kind.machine_is_high, args.human_label)
        if den_r is not None:
            rows.append({"series": family, "label": entry["label"], "dataset": args.denario.stem,
                          "level": "run", "seed": None, **den_r})

        # "The mean of the datasets" -- mean +/- std OF the science dataset_mean
        # (real std, over seeds) and the single denario reading (no std of its
        # own: nothing about it ever varies).
        dataset_means = ([sci_mean] if sci_mean else []) + ([den_r] if den_r is not None else [])
        if dataset_means:
            overall = mean_row(dataset_means)
            rows.append({"series": family, "label": entry["label"], "dataset": "overall",
                          "level": "overall_mean",
                          **{k: v for k, v in overall.items() if k != "n_reads"}})

    print(f"\ntrained: seeds {args.seeds}, {science_test_name}\n")
    print(f"{'series':<24}{'dataset':<32}{'n':>7}{'auroc':>8}{'+/-':>7}"
          f"{'tpr':>8}{'+/-':>7}{'fpr':>8}")

    for name, spec in TRAINED_SPECS.items():
        sci_reads: list[dict] = []
        den_reads: list[dict] = []
        for seed in args.seeds:
            train_idx, test_idx = splits[seed]
            fit = fit_and_calibrate(spec, sci_df, train_idx, test_idx, seed, args.fpr_target)
            sci_row = fit["science_test_row"]
            sci_reads.append(sci_row)
            rows.append({"series": name, "label": spec["label"], "dataset": science_test_name,
                          "level": "run", "seed": seed, "threshold": fit["threshold"], **sci_row})

            r = evaluate_trained(spec, fit["alg"], fit["threshold"], den_df)
            if r is not None:
                den_reads.append(r)
                rows.append({"series": name, "label": spec["label"], "dataset": args.denario.stem,
                              "level": "run", "seed": seed, "threshold": fit["threshold"], **r})

        sci_mean = mean_row(sci_reads)
        rows.append({"series": name, "label": spec["label"], "dataset": science_test_name,
                      "level": "dataset_mean", "seed": None,
                      **{k: v for k, v in sci_mean.items() if k != "n_reads"},
                      "n_reads": sci_mean["n_reads"]})
        print(f"{name:<24}{science_test_name:<32}{sci_mean['n']:>7}{sci_mean['auroc']:>8.3f}"
              f"{sci_mean['auroc_std']:>7.3f}{sci_mean['tpr']:>8.3f}{sci_mean['tpr_std']:>7.3f}"
              f"{sci_mean['fpr']:>8.3f}")

        den_mean = None
        if den_reads:
            den_mean = mean_row(den_reads)
            rows.append({"series": name, "label": spec["label"], "dataset": args.denario.stem,
                          "level": "dataset_mean", "seed": None,
                          **{k: v for k, v in den_mean.items() if k != "n_reads"},
                          "n_reads": den_mean["n_reads"]})
            print(f"{name:<24}{args.denario.stem:<32}{den_mean['n']:>7}{den_mean['auroc']:>8.3f}"
                  f"{den_mean['auroc_std']:>7.3f}{den_mean['tpr']:>8.3f}{den_mean['tpr_std']:>7.3f}"
                  f"{den_mean['fpr']:>8.3f}")

        # "The mean of the datasets" -- mean +/- std OF sci_mean and den_mean,
        # the pair the solid circle actually draws. Not a further seed
        # average: each input here is already a seed-mean.
        dataset_means = [m for m in (sci_mean, den_mean) if m]
        overall = mean_row(dataset_means)
        rows.append({"series": name, "label": spec["label"], "dataset": "overall",
                      "level": "overall_mean",
                      **{k: v for k, v in overall.items() if k != "n_reads"}})
        print(f"{name:<24}{'overall (mean of datasets)':<32}{overall['n']:>7}"
              f"{overall['auroc']:>8.3f}{overall['auroc_std']:>7.3f}{overall['tpr']:>8.3f}"
              f"{overall['tpr_std']:>7.3f}{overall['fpr']:>8.3f}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")

    if not args.no_plot:
        plot(rows, args.fpr_target, fig_path)


if __name__ == "__main__":
    main()
