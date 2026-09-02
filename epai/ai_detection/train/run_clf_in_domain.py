"""In-domain control: train on a corpus, score the held-out split of THAT corpus.

The sibling of ``run_clf_cross_domain.py``, and the question it deliberately
does *not* ask. That script trains on each corpus in turn and evaluates on all
of them, because the interesting number is the off-diagonal -- what survives
leaving the corpus. This one has no off-diagonal to report: it trains on a
single corpus, unfiltered by default, and reports the one number the matrix puts
on its diagonal.

    "how well does this classifier separate human from machine text HERE"

Not "does it transfer". A score from this script says nothing about an unseen
corpus or an unseen generator, and the ~97-99% it will report is the number that
motivated the cross-domain work in the first place -- it survives a grouped
split, a permuted-label control and a length-only baseline, and still collapses
under transfer. Read it as an upper bound on a deployed detector, never as an
estimate of one.

Why a separate script rather than a one-corpus config for the matrix: with a
single corpus every summary the cross-domain run produces is either undefined
(transfer mean, the gap, the ranking it sorts by) or a 1x1 heatmap. Rather than
branch all of that on "if there is no off-diagonal", the engine both runs share
lives in ``epai/utils/cd_core.py`` and the config axes in
``epai/utils/variant_utils.py``, and each script owns only its own evaluation.

CONFIG
------
The same shape as a cross-domain config -- the same ``feature_sets`` x
``algorithms`` cross product, the same ``model`` block, the same zero-shot score
baselines -- differing only in ``data``, which names the corpora to run rather
than the corpora to put in a matrix.

Every ``feature_sets`` entry crossed with every ``algorithms`` entry is one
**variant**, exactly as in the matrix run: each gets its own directory, its own
figure and a row in a comparison ranked by accuracy.

SEVERAL CORPORA IN ONE RUN
--------------------------
``data`` may name ONE corpus directly, as it always could::

    data:
      parquet_path: "data/parquet/ghostbuster_gpt54mini.parquet"
      human_label: "human"
      group_col: "id"
      # where: {domain: essay}      # optional; omit for ALL the data

...or carry a ``datasets`` list, which runs the whole experiment once per
corpus, plus once on the whole file if ``include_pooled`` is set::

    data:
      parquet_path: "data/parquet/ghostbuster_gpt56luna.parquet"
      human_label: "human"
      group_col: "id"
      datasets:
        - {name: essay,  where: {domain: essay}}
        - {name: reuter, where: {domain: reuter}}
        - {name: wp,     where: {domain: wp}}
      include_pooled: true          # + one more on the parquet, unfiltered

That produces the N+1 numbers this control is usually wanted for: what the
detector scores on each domain on its own, and what it scores trained on
everything available.

This list is NOT the cross-domain one, and the difference is the whole point:
nothing is ever trained on one entry and tested on another. Every corpus is a
separate, self-contained experiment, so a one-entry list is legitimate here
where the matrix runner needs two, and none of the output is a transfer claim.
The pooled row in particular is NOT independent of the per-domain rows -- it is
the same documents again -- so its bar sits in the same table but adds no
evidence to them.

Output follows the shape: one corpus writes straight into ``<alg_id>/``, exactly
as before; several write ``<alg_id>/<corpus>/`` each holding what a single-corpus
run holds, plus ``corpus_comparison.json`` and one figure across corpora.

TRAIN AND TEST
--------------
Every variant is scored twice per seed: on the held-out rows, and on the rows it
was fitted on. The unprefixed numbers are the held-out ones; ``train_*`` is the
fit, and ``overfit_gap`` is the difference. Read the gap before the headline --
99.5% test at a 0.5-point gap and 99.5% at a 15-point gap are different claims,
and only the first is about the signal.

The spread over seeds is NOT a confidence interval. The seeds resample the same
corpus, so their test halves overlap heavily; the reported +- is smaller than the
binomial error of a single test set and understates the real uncertainty.

METRICS
-------
Accuracy and macro F1 from the labels -- at a Youden's J threshold fit on the
calibration slice for a trained variant, at the declared cutoff for a ❄
baseline, exactly as in ``run_clf_cross_domain.py``; AUROC and TPR at 1% and 5% FPR from the continuous
score behind them, on both row sets. The gap is not the same size for all of
them: a model can lose almost nothing in accuracy across the split and a large
share of its TPR at 1% FPR, because that operating point lives in the tail of
the human distribution -- exactly the part a fitted model has pushed furthest
away on the rows it was fitted on.

``tprT@X`` / ``fprT@X`` are the transported reading: the threshold is calibrated
to X% FPR on a calibration slice -- a grouped ~15% cut of the training half,
never fit on and never evaluated on, see ``cd_core.carve_calibration`` -- and
applied to the held-out rows, and ``fprT@X`` is the FPR it actually produced
there. This is the weakest possible test of a calibration -- one corpus, one
generator, two random halves -- so a realized FPR that already misses its
nominal level here says the operating point is not stable even before transfer
is asked about. See ``run_clf_cross_domain.py``, where the same threshold is
carried to a corpus the model has never seen; the two runners carve their
calibration slice the same way, so a shared cell agrees exactly.

``report_by`` names columns to describe the split by (``domain``,
``generated_by``). They are read and kept aligned with the design matrix but
never stacked into it -- the model must not see ``domain``. The split is grouped
by ``group_col`` but NOT stratified, so a column's proportions hold only in
expectation; this is what says whether they actually held.

EVAL-ONLY CORPORA (optional)
----------------------------
``data.eval_only`` names corpora that are EVALUATED on but never trained on.
Nothing there takes a row from a split, contributes a cell to a table above or a
bar to a figure above; every threshold applied to it was fitted on the training
corpora and carried over unchanged. It exists for a corpus that cannot be a
``datasets`` entry at all -- the Denario papers are LLM-written end to end, so
there is no human row to split on, no ROC to read and no threshold to fit -- and
it answers the one question that needs none of those:

    "a detector calibrated at 1% FPR on essays -- how many of these does it catch?"

Output is its own: ``holdout_<name>.json`` and ``holdout_<name>_<metric>.png``,
beside the main comparison and never inside it. The columns of that table are the
corpora that supplied the threshold, so the spread across a row is what the
choice of calibration corpus costs. On an all-machine holdout accuracy IS recall,
the ceiling is 100%, and AUROC / macro F1 / the realized FPR are undefined and
omitted rather than zeroed. See ``epai/utils/holdout_core.py`` for what that
does and does not license you to claim.


WHAT IS AND IS NOT SHARED WITH THE MATRIX RUN
---------------------------------------------
Shared, so the diagonal here is the diagonal there: the grouped split (a
document and its rewrites never straddle the cut), the calibration carve inside
the training half (``cd_core.carve_calibration``, same seed), the binarised
human-vs-machine target, the group-aware missing-value drop, the per-seed
repetition and the metric definitions. Train a variant here and read the same
variant's diagonal in a cross-domain run and you should see the same number.

Not shared, because it has no meaning here: the exclusion of target rows whose
group was seen in training. There is no target corpus.

Run:
    uv run python epai/ai_detection/train/run_clf_in_domain.py
    uv run python epai/ai_detection/train/run_clf_in_domain.py --dry-run
    uv run python epai/ai_detection/train/run_clf_in_domain.py \
        --cfg cfg/ai_detection/train/clf_id/clf_id_domains_all.yaml --no-plots

    # every domain of the suite, plus the whole parquet:
    uv run python epai/ai_detection/train/run_clf_in_domain.py \
        --cfg cfg/ai_detection/train/clf_id/clf_id_domains_all.yaml

"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import json
import shutil
import sys

import numpy as np
from tqdm import tqdm

from epai.algs import build_algorithm
from epai.utils.cd_core import (
    BINARY_CLASSES,
    CAL_FRAC_DEFAULT,
    baseline_specs,
    calibrate_thresholds,
    carve_calibration,
    HEADLINE_METRICS,
    cell_metric_specs,
    check_dataset,
    check_mode,
    headline_metric_specs,
    describe_variant,
    evaluate_score_rule,
    format_threshold,
    grouped_split,
    label_predictions,
    load_prepared,
    mean_std_over_seeds,
    metrics_dict,
    oriented_scores,
    print_headline_ranking,
    print_row_mark_legend,
    print_split_composition,
    row_mark_pad,
    score_metric_keys,
    split_composition,
    youden_threshold,
)
from epai.utils.config_utils import (
    RESET,
    YELLOW,
    add_config_args,
    rel,
    resolve_configs,
)
from epai.utils.corpus_utils import (
    CorpusSpec,
    check_eval_only_corpora,
    check_in_domain_corpora,
    eval_only_specs,
    in_domain_specs,
)
from epai.utils.holdout_core import (
    HOLDOUT_KEY,
    aggregate_holdout,
    check_holdout_corpora,
    evaluate_baseline_on_holdout,
    evaluate_variant_on_holdout,
    holdout_comparison,
    load_holdout,
    merge_holdout_rows,
    print_holdout_table,
)
from epai.utils.holdout_viz import plot_holdout_comparison
from epai.utils.preflight_utils import RULE, Report, load_yaml, print_report
from epai.utils.suite_utils import corpus_comparison_rows
from epai.utils.train_utils import CLF_TARGET_COL, stack_features
from epai.utils.variant_utils import (
    Variant,
    algorithm_specs,
    report_columns,
    aux_columns,
    build_feature_sets,
    build_variants,
    check_config,
    check_files,
    comparison_noun,
    fit_alg,
    predict_alg,
    resolved_params,
    score_alg,
    variant_columns,
)
from epai.utils.visualization_utils import (
    plot_clf_id_comparison,
    plot_clf_id_corpus_comparison,
    plot_clf_id_result,
)

BASE_DIR = Path(__file__).parent.parent.parent.parent
DEFAULT_CFG_DIR = BASE_DIR / "cfg" / "ai_detection" / "train" / "clf_id"
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "clf_id"

#: Cleared by ``--no-plots``; see run_clf_cross_domain.py for why this is a flag
#: read in the loop rather than a final step to skip.
PLOTS_ENABLED = True


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def preflight_config(cfg: dict, report: Report) -> None:
    """Everything checkable about one config before a model is fitted."""
    print("\n[1/3] config")
    n_errors = len(report.errors)
    # The axes, identity, model block and baselines are validated exactly as the
    # matrix run validates them; only the `data` shape differs.
    check_mode(cfg, report, ("in_domain",))
    check_config(cfg, report, data_check=check_in_domain_corpora)
    if len(report.errors) > n_errors:
        print("  invalid — skipping the data checks for this config")
        return

    # The eval-only block is checked here rather than inside check_config: it is
    # the one part of `data` that means the same thing to both runners, and it
    # needs the resolved corpus names to refuse a holdout that collides with one.
    check_eval_only_corpora(
        cfg, report, [s.name for s in in_domain_specs(cfg, BASE_DIR)]
    )

    fsets = build_feature_sets(cfg)
    variants = build_variants(cfg)
    baselines = baseline_specs(cfg)
    n_fits = len(variants) * cfg["model"]["n_runs"]
    baseline_note = (
        " + score baselines "
        + ", ".join(f"{v['label']} ({v['column']})" for v in baselines)
        if baselines
        else ""
    )
    specs = in_domain_specs(cfg, BASE_DIR)
    print(
        f"  {len(specs)} corpus/corpora, {len(fsets)} feature set(s), "
        f"{len(algorithm_specs(cfg))} algorithm(s){baseline_note}, "
        f"{cfg['model']['n_runs']} seeds  → {len(variants)} variant(s), "
        f"{n_fits * len(specs)} fits"
    )
    for spec in specs:
        print(f"    {spec.name:<26} {spec.source}")
    restricted = [s for s in algorithm_specs(cfg) if s["feature_sets"] is not None]
    if restricted:
        # The cross product no longer explains the variant count, so name the
        # cells that were dropped before a reader wonders where they went.
        print(
            "  restricted to some feature sets: "
            + ", ".join(f"{s['id']} → {list(s['feature_sets'])}" for s in restricted)
        )
    hybrids = [v for v in variants if v.hybrid]
    if hybrids:
        print(
            f"  {len(hybrids)} hybrid variant(s) also take an auxiliary column: "
            + ", ".join(f"{v.id} ({v.aux_col})" for v in hybrids)
        )

    print("\n[2/3] parquet file")
    n_errors = len(report.errors)
    check_files(cfg, specs, report)
    if len(report.errors) > n_errors:
        print("  unusable — skipping the data checks for this config")
        return

    print("\n[3/3] data, dimensions and split")
    # Every corpus is checked even after one of them fails: the point of a
    # preflight this thorough is that one pass names everything to fix.
    outer = report.context
    for spec in specs:
        report.context = f"{outer} · {spec.name}" if outer else spec.name
        preflight_corpus(cfg, spec, report)
    report.context = outer

    # Once for the whole config, not once per corpus: a holdout is the same file
    # whichever corpus supplies the threshold, and it has to carry every column
    # every variant and baseline in the run reads.
    check_holdout_corpora(cfg, report, BASE_DIR)


def preflight_corpus(cfg: dict, spec: CorpusSpec, report: Report) -> None:
    """Stage 3 for ONE corpus: its rows, its dimensions, its splits."""
    fsets = build_feature_sets(cfg)
    group_col: str = cfg["data"]["group_col"]
    human_label: str = cfg["data"]["human_label"]
    test_size: float = cfg["model"].get("test_size", 0.30)
    cal_frac: float = cfg["model"].get("cal_frac", CAL_FRAC_DEFAULT)

    print(f"\n  corpus: {spec.name}  ({spec.source})")
    print(f"{'feature set':<26}{'rows':>8}{'dropped':>9}{'dim':>7}{'groups':>8}  classes")
    print("-" * RULE)

    for fs in fsets:
        try:
            prep = load_prepared(spec, variant_columns(fs, cfg), group_col, human_label)
        except Exception as exc:
            report.error(f"{fs.id}: failed to load: {type(exc).__name__}: {exc}")
            continue

        ok = check_dataset(prep, cfg, fs.id, report)
        dropped = prep.n_class_dropped + prep.n_missing_dropped
        print(
            f"{fs.id:<26}{len(prep.df):>8}{dropped:>9}"
            f"{(prep.dim_of(list(fs.columns)) if prep.dims_per_col else 0):>7}"
            f"{(prep.n_groups if len(prep.df) else 0):>8}  "
            f"{prep.class_counts if len(prep.df) else '-'}"
        )
        if not ok:
            continue

        # Replay every split the run will take. The transfer half of the
        # cross-domain check has nothing to test here, so this is the part that
        # is left: a training half holding one class cannot be fitted at all.
        y = prep.df["y"].to_numpy()
        groups = prep.df["group"].to_numpy()
        for i in range(cfg["model"]["n_runs"]):
            seed = cfg["seed"] + i
            tag = f"{fs.id} (seed {seed})"
            try:
                train_idx, test_idx = grouped_split(y, groups, test_size, seed)
            except Exception as exc:
                report.error(f"{tag}: grouped split failed: {type(exc).__name__}: {exc}")
                continue
            if len(train_idx) == 0 or len(test_idx) == 0:
                report.error(
                    f"{tag}: split leaves {len(train_idx)} train / {len(test_idx)} "
                    f"test rows; adjust test_size={test_size}"
                )
            elif np.unique(y[train_idx]).size < 2:
                report.error(
                    f"{tag}: the training half holds a single class — with "
                    f"{prep.n_groups} groups a grouped split cannot keep both; "
                    f"lower test_size or check {group_col!r}"
                )
            elif np.unique(y[test_idx]).size < 2:
                report.warn(
                    f"{tag}: the test half holds a single class; its accuracy is a "
                    f"base rate, not a result"
                )
            else:
                # The calibration carve out of the training half (see
                # cd_core.carve_calibration): replay it too, or a corpus with
                # too few groups to survive a second grouped cut only fails at
                # fit time.
                try:
                    fit_idx, cal_idx = carve_calibration(
                        y[train_idx], groups[train_idx], cal_frac, seed
                    )
                except Exception as exc:
                    report.error(
                        f"{tag}: calibration carve failed: {type(exc).__name__}: {exc}"
                    )
                    continue
                if len(fit_idx) == 0 or len(cal_idx) == 0:
                    report.error(
                        f"{tag}: calibration carve leaves {len(fit_idx)} fit / "
                        f"{len(cal_idx)} calibration rows; adjust cal_frac={cal_frac}"
                    )
                elif np.unique(y[train_idx][fit_idx]).size < 2:
                    report.error(
                        f"{tag}: the fit rows (after carving out the calibration "
                        f"slice) hold a single class; lower cal_frac={cal_frac} or "
                        f"check {group_col!r}"
                    )
                elif np.unique(y[train_idx][cal_idx]).size < 2:
                    report.warn(
                        f"{tag}: the calibration slice holds a single class; its "
                        f"Youden's J threshold and calibrate_thresholds will be NaN"
                    )

    # A score baseline reads its own column, so it stands on its own row set --
    # the judge only scores the generators it was asked for.
    for bl in baseline_specs(cfg):
        try:
            prep = load_prepared(spec, [bl["column"]], group_col, human_label)
        except Exception as exc:
            report.error(
                f"{bl['label']}: cannot read {bl['column']}: {type(exc).__name__}: {exc}"
            )
            continue
        note = f"  ({prep.n_missing_dropped} unscored)" if prep.n_missing_dropped else ""
        print(
            f"{bl['label']:<26}{len(prep.df):>8}{prep.n_missing_dropped:>9}{1:>7}"
            f"{(prep.n_groups if len(prep.df) else 0):>8}  "
            f"{prep.class_counts if len(prep.df) else '-'}{note}"
        )
        check_dataset(prep, cfg, bl["column"], report)


def preflight(cfg_paths: list[Path]) -> tuple[Report, list[tuple[Path, dict]]]:
    """Check every config before any of them trains."""
    report = Report()
    loaded: list[tuple[Path, dict]] = []

    print("=" * RULE)
    print(f"PREFLIGHT — {len(cfg_paths)} config(s)")
    print("=" * RULE)

    for cfg_path in cfg_paths:
        report.context = cfg_path.name
        print(f"\n{'-' * RULE}\n{cfg_path}\n{'-' * RULE}")
        cfg = load_yaml(cfg_path, report)
        if cfg is None:
            continue
        preflight_config(cfg, report)
        loaded.append((cfg_path, cfg))

    report.context = ""
    alg_ids = [cfg.get("alg_id") for _, cfg in loaded if isinstance(cfg, dict)]
    dupes = sorted({a for a in alg_ids if a is not None and alg_ids.count(a) > 1})
    if dupes:
        report.error(f"alg_id must be unique across the configs run, repeated: {dupes}")

    return report, loaded


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def aggregate_runs(per_seed: list[dict]) -> dict:
    """Mean and spread over seeds. The in-domain analogue of ``aggregate_seeds``.

    One cell rather than a source x target matrix, so it is its own (short)
    function instead of a degenerate call into the matrix one -- the unprefixed
    keys are the TEST split, matching a cross-domain cell so the two can be
    compared directly, and ``train_*`` is the same model scored on the rows it
    was fitted on.

    ``overfit_gap`` is train minus test accuracy. It is the number to read before
    the headline: a model at 99.9% train and 99.5% test has learned the corpus, a
    model at 99.9% and 85% has memorised it.

    Every score metric present (AUROC, both TPR@FPR readings, the thresholds
    behind them) is averaged the same way, on the held-out rows and on the
    training rows, so the overfit question can be asked of them too -- an AUROC
    of 1.000 on train and 0.94 on test is the same story the accuracy gap tells,
    read where the low-FPR operating point actually lives.
    """
    accs = [r["accuracy"] for r in per_seed]
    f1s = [r["macro_f1"] for r in per_seed]
    tr_accs = [r["train_accuracy"] for r in per_seed]
    tr_f1s = [r["train_macro_f1"] for r in per_seed]
    out = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "train_accuracy_mean": float(np.mean(tr_accs)),
        "train_accuracy_std": float(np.std(tr_accs)),
        "train_macro_f1_mean": float(np.mean(tr_f1s)),
        "train_macro_f1_std": float(np.std(tr_f1s)),
        "overfit_gap": float(np.mean(tr_accs) - np.mean(accs)),
        "confusion_mean": np.mean([r["confusion"] for r in per_seed], axis=0).tolist(),
        "train_n": per_seed[0]["train_n"],
        "cal_n": per_seed[0]["cal_n"],
        "n": per_seed[0]["n"],
        "class_balance": per_seed[0]["class_balance"],
    }
    for key in score_metric_keys():
        out.update(mean_std_over_seeds(per_seed, key))
        out.update(mean_std_over_seeds(per_seed, f"train_{key}"))
    return out


def summarise(payload: dict) -> dict:
    """One variant's comparison row: what the ranking and the figures read.

    Every score metric the summary holds is carried through under its own name,
    train and test alike, so the comparison figures can draw one panel per
    metric without reopening each variant's ``in_domain.json``.
    """
    s = payload["summary"]
    row = {
        "variant": payload["variant"],
        "label": payload["label"],
        "feature_dim": payload["feature_dim"],
        "accuracy_mean": s["accuracy_mean"],
        "accuracy_std": s["accuracy_std"],
        "macro_f1_mean": s["macro_f1_mean"],
        "macro_f1_std": s["macro_f1_std"],
        "train_accuracy_mean": s["train_accuracy_mean"],
        "train_accuracy_std": s["train_accuracy_std"],
        "overfit_gap": s["overfit_gap"],
        "confusion_mean": s["confusion_mean"],
        "n": s["n"],
        "train_n": s["train_n"],
        "baseline": bool(payload.get("baseline")),
        "baseline_kind": payload.get("baseline_kind"),
    }
    if payload.get("column"):
        # Only baselines carry one. A family contributes one row per score
        # column, and the column is what distinguishes two rows of the same
        # family when the figures merge rows that draw identically.
        row["column"] = payload["column"]
    for key in score_metric_keys():
        for name in (f"{key}_mean", f"{key}_std", f"train_{key}_mean",
                     f"train_{key}_std"):
            if name in s:
                row[name] = s[name]
    # Carried, not merged: the holdout numbers ride along on the row so the
    # cross-corpus layer can stitch them into their own table, and every
    # consumer of this row reads named keys, so they change nothing here.
    if payload.get(HOLDOUT_KEY):
        row[HOLDOUT_KEY] = payload[HOLDOUT_KEY]
    if payload.get("hybrid"):
        row["hybrid"] = True
        row["aux_column"] = payload.get("aux_column")
    return row


def run_variant(
    variant: Variant,
    cfg: dict,
    X: np.ndarray,
    df,
    out_dir: Path,
    save_models: bool = True,
    holdouts: dict | None = None,
    source: str = "",
) -> dict:
    """Fit and score one variant on the corpus's held-out split, over every seed.

    ``X`` / ``df`` are loaded once per feature set by ``run_config`` and shared
    across every algorithm trained on it, exactly as the matrix run does.

    ``holdouts`` are the eval-only corpora (see ``epai/utils/holdout_core.py``),
    already loaded for this feature set. They are scored inside the seed loop --
    the only place the fitted model and the thresholds fitted on ITS calibration
    slice exist together -- and nothing about them touches the held-out split
    above: they take no rows from it, contribute to no metric in it, and their
    results are written under their own key. ``source`` names the corpus that
    supplied the thresholds, which is this corpus, and becomes the column a
    holdout figure reads.
    """
    base_seed: int = cfg["seed"]
    n_runs: int = cfg["model"]["n_runs"]
    test_size: float = cfg["model"].get("test_size", 0.30)
    cal_frac: float = cfg["model"].get("cal_frac", CAL_FRAC_DEFAULT)

    y = df["y"].to_numpy()
    groups = df["group"].to_numpy()
    # The auxiliary column of a hybrid, kept out of X so it can enter the network
    # late. None for every other algorithm, and never reaches them.
    Z = stack_features(df, [variant.aux_col])[0] if variant.aux_col else None

    per_seed: list[dict] = []
    for i in tqdm(range(n_runs), desc="Seeds", leave=False):
        seed = base_seed + i
        # The same grouped split the cross-domain run takes for its diagonal, so
        # the two numbers are directly comparable.
        train_idx, test_idx = grouped_split(y, groups, test_size, seed)

        # The calibration slice: carved out of the training half exactly as
        # the cross-domain run carves it (cd_core.carve_calibration), so the
        # two runners agree on where a threshold comes from. Never the fit
        # rows -- a model's scores there are optimistic, which is what used to
        # make this runner's transported TPR too strict -- and never the test
        # rows, which would be the oracle reading.
        fit_idx, cal_idx = carve_calibration(y[train_idx], groups[train_idx], cal_frac, seed)
        fit_idx, cal_idx = train_idx[fit_idx], train_idx[cal_idx]

        alg = build_algorithm(variant.alg_name, seed, **variant.params)
        alg.set_fit_groups(groups[fit_idx])
        fit_alg(alg, X[fit_idx], y[fit_idx], Z[fit_idx] if Z is not None else None)

        Z_fit = Z[fit_idx] if Z is not None else None
        Z_cal = Z[cal_idx] if Z is not None else None
        Z_test = Z[test_idx] if Z is not None else None

        # The transported operating point of variant (b): the threshold is
        # calibrated to each nominal FPR on the calibration slice and the
        # realized FPR is read on the held-out test rows. Youden's J on the
        # same slice drives accuracy/macro F1 below.
        s_cal = score_alg(alg, X[cal_idx], Z_cal)
        cal_thresholds = (
            calibrate_thresholds(y[cal_idx], s_cal) if s_cal is not None else None
        )
        label_threshold = (
            youden_threshold(y[cal_idx], s_cal) if s_cal is not None else None
        )

        s_test = score_alg(alg, X[test_idx], Z_test)
        res = metrics_dict(
            y[test_idx],
            label_predictions(
                predict_alg(alg, X[test_idx], Z_test), s_test, label_threshold
            ),
            scores=s_test,
            cal_thresholds=cal_thresholds,
            label_threshold=label_threshold,
        )
        # The same model on the rows it was actually fitted on (fit_idx, not
        # the whole training half -- the calibration slice was never fit on).
        # Not a result -- it is the denominator for one: fit minus test is
        # what the model failed to generalise, and a headline test score means
        # something different at a 0.4-point gap than at a 15-point one.
        s_fit = score_alg(alg, X[fit_idx], Z_fit)
        train_res = metrics_dict(
            y[fit_idx],
            label_predictions(
                predict_alg(alg, X[fit_idx], Z_fit), s_fit, label_threshold
            ),
            scores=s_fit,
        )
        res["train_accuracy"] = train_res["accuracy"]
        res["train_macro_f1"] = train_res["macro_f1"]
        res["train_confusion"] = train_res["confusion"]
        for key in score_metric_keys():
            if key in train_res:
                res[f"train_{key}"] = train_res[key]
        res["train_n"] = int(len(fit_idx))
        res["cal_n"] = int(len(cal_idx))

        # The eval-only corpora, under THIS seed's thresholds. Fitted on the
        # calibration slice above and applied unchanged -- nothing is refitted
        # here and no threshold is chosen on these rows, which is what makes the
        # number a deployment reading rather than an oracle one. The training
        # groups are passed so the "a target row whose group was trained on is a
        # free hit" rule applies here too; on a holdout from a different file it
        # removes nothing, and the count says so.
        if holdouts:
            res[HOLDOUT_KEY] = {
                name: evaluate_variant_on_holdout(
                    alg, hold, cal_thresholds, label_threshold,
                    train_groups=set(groups[train_idx]),
                    aux=(
                        stack_features(hold.df, [variant.aux_col])[0]
                        if variant.aux_col and len(hold.df)
                        else None
                    ),
                )
                for name, hold in holdouts.items()
            }
        per_seed.append(res)

        if save_models:
            # Persist last: the torch algorithms' save() moves the model to CPU,
            # so saving before the predictions above would drop them off the GPU.
            seed_dir = out_dir / str(i) / "model"
            seed_dir.mkdir(parents=True, exist_ok=True)
            try:
                alg.save(seed_dir / "model.joblib")
            except Exception as exc:  # persistence must never abort a long run
                tqdm.write(
                    f"{YELLOW}  ⚠️  could not save {variant.alg_name} (seed {seed}): "
                    f"{type(exc).__name__}: {exc}{RESET}"
                )

    var_cfg = dict(cfg)
    var_cfg["features"] = list(variant.features)
    var_cfg["binoculars"] = variant.aux_col
    var_cfg["model"] = {**cfg["model"], "ml_alg": variant.alg_name}
    var_cfg["alg_id"] = f"{cfg['alg_id']}__{variant.id}"
    var_cfg["alg_label"] = f"{cfg['alg_label']} — {variant.label}"

    payload = {
        "alg_id": var_cfg["alg_id"],
        "alg_label": var_cfg["alg_label"],
        "variant": variant.id,
        "label": variant.label,
        "algorithm": variant.alg_name,
        "params": variant.params,
        "resolved_params": resolved_params(variant, base_seed),
        "binoculars": variant.aux_col,
        "feature_dim": int(X.shape[1]),
        "classes": BINARY_CLASSES,
        "config": var_cfg,
        "summary": aggregate_runs(per_seed),
        "per_seed": per_seed,
    }
    if holdouts:
        payload[HOLDOUT_KEY] = _holdout_summary(per_seed, holdouts, source)
    if variant.aux_col:
        payload["hybrid"] = True
        payload["aux_column"] = variant.aux_col
    with open(out_dir / "in_domain.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def _holdout_summary(per_seed: list[dict], holdouts: dict, source: str) -> dict:
    """``{holdout: {source: aggregate}}`` from the per-seed holdout cells.

    The extra ``source`` level looks redundant for an in-domain run, where one
    pass has exactly one source -- and it is exactly what lets ``run_config``
    stitch the passes back into one table whose columns are the corpora. See
    ``holdout_core.merge_holdout_rows``.
    """
    return {
        name: {
            source: aggregate_holdout(
                [r[HOLDOUT_KEY][name] for r in per_seed if name in r.get(HOLDOUT_KEY, {})]
            )
        }
        for name in holdouts
    }


def run_baseline(
    cfg: dict,
    spec: CorpusSpec,
    bl: dict,
    out_dir: Path,
    holdouts: dict | None = None,
) -> dict:
    """One zero-shot score baseline on the held-out split. Nothing is fitted.

    Same split and seeds as the trained variants, so its row lands in the same
    table. The declared cutoff never looks at the data at all and is still
    recorded, but no longer drives accuracy/macro F1 -- those come from a
    Youden's J threshold fit on the same calibration slice the trained
    variant of this corpus uses, for the reasons ``cd_core.BaselineKind``
    documents.
    """
    group_col: str = cfg["data"]["group_col"]
    human_label: str = cfg["data"]["human_label"]
    n_runs: int = cfg["model"]["n_runs"]
    test_size: float = cfg["model"].get("test_size", 0.30)
    cal_frac: float = cfg["model"].get("cal_frac", CAL_FRAC_DEFAULT)
    col, machine_is_high = bl["column"], bool(bl.get("machine_is_high", False))
    threshold = float(bl["threshold"])

    prep = load_prepared(spec, [col], group_col, human_label)
    note = f"  ({prep.n_missing_dropped} row(s) without a {col})" if prep.n_missing_dropped else ""
    tqdm.write(f"  {prep.name:<20} {len(prep.df):>6} rows  {prep.class_counts}{note}")

    scores = prep.df[col].to_numpy(dtype=float)
    y = prep.df["y"].to_numpy()
    groups = prep.df["group"].to_numpy()

    per_seed: list[dict] = []
    for i in range(n_runs):
        seed = cfg["seed"] + i
        train_idx, test_idx = grouped_split(y, groups, test_size, seed)
        # The calibration slice: carved from the training half exactly as the
        # trained variant of this corpus carves it, same seed, so the two
        # transport a threshold fit on identical rows. Nothing is fitted here,
        # so this is unbiased in the same sense the old training-half
        # calibration was -- a zero-shot score never saw the fit rows either --
        # but it now sits on the same rows as every other row of the table.
        _, cal_idx = carve_calibration(y[train_idx], groups[train_idx], cal_frac, seed)
        cal_idx = train_idx[cal_idx]
        oriented_cal = oriented_scores(scores[cal_idx], machine_is_high)
        cal_thresholds = calibrate_thresholds(y[cal_idx], oriented_cal)
        label_threshold = youden_threshold(y[cal_idx], oriented_cal)
        res = evaluate_score_rule(
            scores[test_idx], y[test_idx], threshold, machine_is_high,
            cal_thresholds=cal_thresholds, label_threshold=label_threshold,
        )
        # The same cutoff on the training half. The gap against the held-out
        # half should be ~0: the cutoff saw no data at all, so there is no
        # calibration optimism for it to carry.
        train_res = evaluate_score_rule(
            scores[train_idx], y[train_idx], threshold, machine_is_high
        )
        res["train_accuracy"] = train_res["accuracy"]
        res["train_macro_f1"] = train_res["macro_f1"]
        res["train_confusion"] = train_res["confusion"]
        for key in score_metric_keys():
            if key in train_res:
                res[f"train_{key}"] = train_res[key]
        res["train_n"] = int(len(train_idx))
        res["cal_n"] = int(len(cal_idx))
        res["threshold"] = float(threshold)

        # The eval-only corpora, read off this baseline's own column. Same two
        # thresholds the cells above answer to, plus the DECLARED cutoff, which
        # is the only genuinely zero-shot reading anywhere in the run and the one
        # an off-the-shelf deployment would actually get.
        if holdouts:
            res[HOLDOUT_KEY] = {
                name: evaluate_baseline_on_holdout(
                    hold, threshold, machine_is_high,
                    cal_thresholds, label_threshold,
                    train_groups=set(groups[train_idx]),
                )
                for name, hold in holdouts.items()
            }
        per_seed.append(res)

    summary = aggregate_runs(per_seed)
    summary["threshold_mean"] = float(np.mean([r["threshold"] for r in per_seed]))

    bl_cfg = dict(cfg)
    bl_cfg["features"] = [col]
    bl_cfg["model"] = {
        **cfg["model"],
        "ml_alg": f"{bl['kind']} (fixed threshold {format_threshold(threshold)})",
    }
    bl_cfg["alg_id"] = f"{cfg['alg_id']}__{bl['name']}"
    bl_cfg["alg_label"] = f"{cfg['alg_label']} — {bl['label']}"

    payload = {
        "alg_id": bl_cfg["alg_id"],
        "alg_label": bl_cfg["alg_label"],
        "variant": bl["name"],
        "label": bl["label"],
        "baseline": True,
        "baseline_kind": bl["kind"],
        "machine_is_high": machine_is_high,
        "column": col,
        "threshold": threshold,
        "feature_dim": 1,
        "classes": BINARY_CLASSES,
        "config": bl_cfg,
        "summary": summary,
        "per_seed": per_seed,
    }
    if holdouts:
        payload[HOLDOUT_KEY] = _holdout_summary(per_seed, holdouts, spec.name)
    with open(out_dir / "in_domain.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def print_result(label: str, summary: dict) -> None:
    """One variant's train and held-out numbers, and the gap between them."""
    print(f"\n{'='*72}")
    print(f"  {label} — TRAIN vs HELD-OUT SPLIT OF THE SAME CORPUS")
    print(f"{'='*72}")
    print(f"{'':14}{'accuracy':>18}{'macro F1':>18}")
    print(
        f"  {'train':<12}"
        f"{summary['train_accuracy_mean']*100:>11.1f}% ± {summary['train_accuracy_std']*100:<4.1f}"
        f"{summary['train_macro_f1_mean']*100:>11.1f}% ± {summary['train_macro_f1_std']*100:<4.1f}"
    )
    print(
        f"  {'test':<12}"
        f"{summary['accuracy_mean']*100:>11.1f}% ± {summary['accuracy_std']*100:<4.1f}"
        f"{summary['macro_f1_mean']*100:>11.1f}% ± {summary['macro_f1_std']*100:<4.1f}"
    )
    print(f"  {'gap':<12}{summary['overfit_gap']*100:>11.1f} points  (train − test)")

    # The score metrics, on the same two row sets. Skipped entirely for a variant
    # whose algorithm has no ranking score.
    specs = [
        m for m in cell_metric_specs()
        if f"{m.key}_mean" in summary and m.key not in ("accuracy", "macro_f1")
    ]
    if specs:
        print(f"\n{'':14}" + "".join(f"{m.short:>18}" for m in specs))
        for side, prefix in (("train", "train_"), ("test", "")):
            row = f"  {side:<12}"
            for m in specs:
                mean = summary.get(f"{prefix}{m.key}_mean")
                std = summary.get(f"{prefix}{m.key}_std")
                row += (
                    f"{'—':>18}"
                    if mean is None or not np.isfinite(mean)
                    else f"{mean*100:>11.1f}% ± {(std or 0.0)*100:<4.1f}"
                )
            print(row)
        print(
            "\n  tpr@X is (a): the threshold is fit on the very rows it is scored\n"
            "  on. tprT@X / fprT@X are (b): the threshold is fit to X% FPR on the\n"
            "  calibration slice and carried to the held-out ones — fprT@X is what\n"
            "  it actually cost there. Within one corpus this is the easiest possible\n"
            "  test of a calibration; a miss here will only widen under transfer."
        )

    print(f"\n  train / test rows : {summary['train_n']} / {summary['n']}   "
          f"{summary['class_balance']}")
    print("\n  In-domain only. Chance is 50%; a score here says nothing about an")
    print("  unseen corpus or generator — run_clf_cross_domain.py answers that.")
    print("  The ± is the spread over seeds on OVERLAPPING splits of one corpus,")
    print("  not a confidence interval: it understates the sampling error.")


def print_ranking(alg_label: str, rows: list[dict], noun: str = "variant") -> None:
    """Rank the variants by held-out accuracy, then repeat under every metric.

    Two tables, the same shape the matrix run prints: the accuracy ranking with
    the train / test / gap columns that are this script's whole reason to exist,
    then the SAME rows under ``cd_core.HEADLINE_METRICS`` -- so an in-domain
    result can be laid beside a cross-domain or leave-one-domain-out one without
    opening the JSON to find its AUROC.
    """
    order = sorted(rows, key=lambda r: r["accuracy_mean"], reverse=True)
    width = max(26, max((len(r["label"]) for r in order), default=0) + 4)
    print()
    print("=" * RULE)
    print(f"  {alg_label} — {noun.upper()}S RANKED BY IN-DOMAIN ACCURACY")
    print("=" * RULE)
    print(
        f"{noun:<{width}}{'dim':>6}{'train':>10}{'test':>10}{'gap':>8}{'test n':>9}"
    )
    print("-" * RULE)
    for r in order:
        mark = row_mark_pad(r)
        print(
            f"{mark + r['label']:<{width}}{r['feature_dim']:>6}"
            f"{r['train_accuracy_mean']*100:>9.1f}%{r['accuracy_mean']*100:>9.1f}%"
            f"{r['overfit_gap']*100:>8.1f}{r['n']:>9}"
        )
    print("-" * RULE)
    print("  Held-out grouped split of the SAME corpus; chance is 50%.")
    print("  gap = train − test accuracy, in points: what did not generalise even")
    print("        within this corpus. A large gap under a high test score means")
    print("        the score is carried by capacity, not by signal.")
    # Held-out rows only: the train columns belong to the table above, where the
    # pair is the point, and repeating them here would double the width to say
    # something already said.
    print_headline_ranking(
        order, noun=noun, width=width,
        title=f"{alg_label} — HELD-OUT ROWS UNDER EVERY HEADLINE METRIC",
        note=(
            "  The threshold behind tprT@X (and the Youden's J behind accuracy /\n"
            "  macro F1) is fitted on a calibration slice carved out of the\n"
            "  training half — never the fit rows, never the held-out ones — the\n"
            "  same procedure and, for a shared corpus and seed, the same rows a\n"
            "  matrix run uses. See cd_core.carve_calibration."
        ),
    )
    print_row_mark_legend(order)


def run_config(cfg_path: Path, cfg: dict, alg_dir: Path) -> None:
    """One config: every corpus it names, and the comparison across them.

    A config naming ONE corpus is exactly what it always was -- the variants,
    their figures and the ranking written straight into ``alg_dir``.

    A config with ``data.datasets`` runs that whole thing once per corpus, one
    directory deeper, and adds the level the list exists for: one row per
    variant holding its per-corpus held-out accuracy, ranked by the unweighted
    mean over corpora, as ``corpus_comparison.json`` beside the corpus
    directories. No model ever crosses two corpora here -- that is the matrix
    run's job -- so the comparison is a table of independent results, not a
    transfer claim.
    """
    alg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, alg_dir / "config.yaml")
    with open(alg_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    specs = in_domain_specs(cfg, BASE_DIR)
    # The layout follows the config's SHAPE, not how many corpora it happened to
    # resolve to: a one-entry `datasets` list still writes a corpus directory, so
    # adding a second entry later does not silently move every path.
    if not cfg["data"].get("datasets"):
        comparison = run_corpus(cfg, specs[0], alg_dir)
        # One corpus, so the holdout table has one column: what this detector
        # catches under the one threshold this run produced.
        write_holdout_comparisons(
            cfg, alg_dir,
            {specs[0].name: comparison["variants"]},
            [specs[0].name], {specs[0].name: specs[0].label},
        )
        return

    per_corpus: dict[str, list[dict]] = {}
    for spec in tqdm(specs, desc="Corpora", unit="corpus"):
        corpus_dir = alg_dir / spec.name
        corpus_dir.mkdir(parents=True, exist_ok=True)
        per_corpus[spec.name] = run_corpus(cfg, spec, corpus_dir)["variants"]

    names = [s.name for s in specs]
    labels = {s.name: s.label for s in specs}
    rows = corpus_comparison_rows(per_corpus, names)
    comparison = {
        "alg_id": cfg["alg_id"],
        "alg_label": cfg["alg_label"],
        "noun": comparison_noun(cfg),
        "subtitle": (
            f"{len(names)} corpora · {cfg['model']['n_runs']} seeds · "
            f"grouped by {cfg['data']['group_col']!r} · held-out "
            f"{cfg['model'].get('test_size', 0.30):.0%} · chance = 50%"
        ),
        "corpus_names": names,
        "corpus_labels": labels,
        "corpus_sources": {s.name: s.source for s in specs},
        "classes": BINARY_CLASSES,
        "config": cfg,
        "variants": rows,
    }
    with open(alg_dir / "corpus_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print_corpus_ranking(cfg["alg_label"], rows, names, labels,
                         noun=comparison["noun"])
    if PLOTS_ENABLED:
        plot_clf_id_corpus_comparison(alg_dir, comparison)

    # The eval-only table, at this level and only at this level: its columns are
    # the corpora that supplied the thresholds, and no single corpus's pass
    # knows about the others. Written beside the main comparison, never into it.
    write_holdout_comparisons(cfg, alg_dir, per_corpus, names, labels)


def write_holdout_comparisons(
    cfg: dict,
    alg_dir: Path,
    per_corpus: dict[str, list[dict]],
    names: list[str],
    labels: dict[str, str],
) -> None:
    """One ``holdout_<name>.json`` and its figures per eval-only corpus.

    The Denario question in one table: rows are the detectors this run produced,
    columns are the corpora whose calibration slice supplied the threshold, and
    every cell is the same documents scored again. Nothing here is a transfer
    cell -- no model was trained on the holdout and none was tested on a corpus
    it did not train on -- and nothing here enters ``corpus_comparison.json`` or
    any figure that reads it.
    """
    hold_specs = eval_only_specs(cfg, BASE_DIR)
    if not hold_specs:
        return

    group_col: str = cfg["data"]["group_col"]
    human_label: str = cfg["data"]["human_label"]

    for spec in hold_specs:
        rows = merge_holdout_rows(per_corpus, spec.name, names)
        if not rows:
            print(f"\nNo holdout results for {spec.name} — nothing written.")
            continue
        # Re-read the corpus once, for the row counts and class balance the
        # payload describes itself with. Cheap (one scalar column) and it means
        # the JSON says what it stood on without trusting a cell to carry it.
        hold = load_holdout(spec, [CLF_TARGET_COL], group_col, human_label)
        payload = holdout_comparison(
            cfg, hold, rows, labels, names,
            alg_label=f"{cfg['alg_label']} — eval-only",
        )
        with open(alg_dir / f"holdout_{spec.name}.json", "w") as f:
            json.dump(payload, f, indent=2)
        print_holdout_table(payload, rows, names)
        if PLOTS_ENABLED:
            plot_holdout_comparison(alg_dir, payload)


def _num(value, default: float = float("nan")) -> float:
    """One aggregate as a float, with a missing or unparseable cell as NaN.

    Only the printing below needs it -- the row building moved to
    ``suite_utils.corpus_comparison_rows``, which has its own copy for the same
    reason: a table has to render a gap as a dash, not raise on it.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def print_corpus_ranking(
    alg_label: str,
    rows: list[dict],
    names: list[str],
    labels: dict[str, str],
    noun: str = "variant",
) -> None:
    """Rank the variants by mean in-domain accuracy, corpora still visible.

    Two tables: the per-corpus accuracy columns -- the N+1 view this level exists
    for -- and then the same rows under every ``cd_core.HEADLINE_METRICS``,
    meaned over those same corpora. The second is what makes this run
    comparable to a matrix run, which is ranked on transfer AUROC and the
    transported TPR long before anyone looks at its accuracy.
    """
    order = sorted(
        rows, key=lambda r: (-_num(r.get("accuracy_mean"), -1.0), r["label"])
    )
    width = max(26, max((len(r["label"]) for r in order), default=0) + 4)
    col = max(10, max((len(labels.get(n, n)) for n in names), default=0) + 2)
    header = (
        f"{noun:<{width}}"
        + "".join(f"{labels.get(n, n):>{col}}" for n in names)
        + f"{'mean':>{col}}{'±corpora':>10}{'worst':>9}"
    )
    print()
    print("=" * max(RULE, len(header)))
    print(f"  {alg_label} — {noun.upper()}S RANKED BY MEAN IN-DOMAIN ACCURACY")
    print("=" * max(RULE, len(header)))
    print(header)
    print("-" * max(RULE, len(header)))
    for r in order:
        mark = row_mark_pad(r)
        line = f"{mark + r['label']:<{width}}"
        for n in names:
            value = _num(r["per_corpus"].get(n, {}).get("accuracy_mean"))
            line += (
                f"{'—':>{col}}" if not np.isfinite(value)
                else f"{value*100:>{col-1}.1f}%"
            )
        line += (
            f"{_num(r.get('accuracy_mean'))*100:>{col-1}.1f}%"
            f"{_num(r.get('accuracy_across_corpora_std'))*100:>9.1f} "
            f"{_num(r.get('accuracy_worst'))*100:>8.1f}%"
        )
        print(line)
    print("-" * max(RULE, len(header)))
    print("  Each column is a SEPARATE experiment: trained and tested on that")
    print("  corpus alone. Nothing here crosses corpora — for that, run")
    print("  epai/ai_detection/train/run_clf_cross_domain.py.")
    print("  mean     = unweighted over the columns; ±corpora is their spread,")
    print("             which is 'do the corpora agree', not a seed error bar.")
    print("  Chance is 50%. These are upper bounds on a deployed detector.")
    print_row_mark_legend(order)

    # The same rows under every headline metric, each one meaned over the same
    # corpora. Ranking on accuracy alone is how a variant that separates well but
    # cannot hold an operating point ends up at the top of the only table anyone
    # reads.
    print_headline_ranking(
        order, noun=noun, width=width,
        title=f"{alg_label} — MEAN OVER CORPORA, EVERY HEADLINE METRIC",
        note=(
            "  Every column is the unweighted mean over the same corpora as the\n"
            "  table above, so a row is only comparable to another row of this\n"
            "  table — never to a single corpus's number."
        ),
    )


def run_corpus(cfg: dict, spec: CorpusSpec, alg_dir: Path) -> dict:
    """Every variant on ONE corpus, plus its comparison figure.

    Returns the ``in_domain_comparison`` payload it wrote, which is what the
    cross-corpus layer above stacks.
    """
    alg_dir.mkdir(parents=True, exist_ok=True)

    group_col: str = cfg["data"]["group_col"]
    human_label: str = cfg["data"]["human_label"]
    fsets = build_feature_sets(cfg)
    variants = build_variants(cfg)
    aux = aux_columns(cfg)
    report_by = report_columns(cfg)
    composition: dict[str, dict] = {}

    tqdm.write("\n" + "=" * RULE)
    tqdm.write(f"  ALG ID     : {cfg['alg_id']}")
    tqdm.write(f"  CORPUS     : {spec.name}  ({spec.source})")
    tqdm.write(f"  VARIANTS   : {[v.id for v in variants]}")
    if aux:
        tqdm.write(f"  AUXILIARY  : {aux}  (hybrid entries only)")
    tqdm.write(
        f"  SPLIT      : grouped by {group_col!r}, "
        f"test_size={cfg['model'].get('test_size', 0.30)}, "
        f"{cfg['model']['n_runs']} seeds (base {cfg['seed']})"
    )
    if report_by:
        tqdm.write(f"  REPORT BY  : {report_by}  (split composition only, never a feature)")
    hold_specs = eval_only_specs(cfg, BASE_DIR)
    if hold_specs:
        tqdm.write(
            f"  EVAL-ONLY  : {[h.name for h in hold_specs]}  "
            f"(scored under this corpus's thresholds; never trained on)"
        )
    tqdm.write(f"  OUTPUT     : {alg_dir}")
    tqdm.write("=" * RULE)

    rows: list[dict] = []
    for fs in tqdm(fsets, desc="Feature sets", unit="set", leave=False):
        # An entry's `feature_sets` restriction can leave a set with nothing to
        # train; loading the corpus to fit nothing on it is wasted I/O. The
        # preflight has already warned about it.
        fs_variants = [v for v in variants if v.fs_id == fs.id]
        if not fs_variants:
            tqdm.write(f"\n{'─' * 60}\n  feature set: {fs.id} — no algorithm, skipped")
            continue

        # Load and stack one feature set once; every algorithm on it reuses the
        # result. The missing-value filter runs over the features and every
        # auxiliary column together, so a hybrid and a plain algorithm are scored
        # on identical rows.
        # `report_by` columns are carried, not featurised: read and kept aligned
        # with the design matrix so the split can be described by them, but never
        # stacked into it -- the model must not see `domain`.
        prep = load_prepared(
            spec, variant_columns(fs, cfg), group_col, human_label,
            carry=report_by,
        )
        tqdm.write(f"\n{'─' * 60}\n  feature set: {fs.id}")
        tqdm.write(f"  {prep.name:<20} {len(prep.df):>6} rows  {prep.class_counts}")
        X = stack_features(prep.df, list(fs.columns))[0]

        # The eval-only corpora, read with the same columns and stacked the same
        # way. Loaded per feature set because the design matrix is per feature
        # set; they take no part in the split below and no part in any fit.
        holdouts = {
            h.name: load_holdout(
                h, variant_columns(fs, cfg), group_col, human_label,
                stack=list(fs.columns),
            )
            for h in hold_specs
        }
        for hold in holdouts.values():
            tqdm.write(
                f"  {('· ' + hold.name):<20} {len(hold.df):>6} rows  "
                f"{hold.class_counts}   (eval only)"
            )

        # Every variant on this feature set takes the same splits, so the
        # composition is a property of the split, measured once, not of a model.
        if report_by:
            splits = [
                grouped_split(
                    prep.df["y"].to_numpy(), prep.df["group"].to_numpy(),
                    cfg["model"].get("test_size", 0.30), cfg["seed"] + i,
                )
                for i in range(cfg["model"]["n_runs"])
            ]
            composition[fs.id] = split_composition(prep.df, splits, report_by)
            print_split_composition(composition[fs.id])

        for variant in tqdm(
            fs_variants, desc="Variants", unit="var", leave=False,
        ):
            hybrid_note = f" + {variant.aux_col}" if variant.aux_col else ""
            tqdm.write(f"\n→ {variant.id}  ({variant.alg_name}{hybrid_note})")
            eff = resolved_params(variant, cfg["seed"])
            tqdm.write(
                "   params: " + ", ".join(f"{k}={v}" for k, v in sorted(eff.items()))
            )
            out_dir = alg_dir / variant.id
            out_dir.mkdir(parents=True, exist_ok=True)

            payload = run_variant(
                variant, cfg, X, prep.df, out_dir,
                holdouts=holdouts, source=spec.name,
            )
            print_result(payload["alg_label"], payload["summary"])
            if PLOTS_ENABLED:
                plot_clf_id_result(out_dir)
            row = summarise(payload)
            if variant.aux_col:
                row["hybrid"] = True
                row["aux_column"] = variant.aux_col
            rows.append(row)

        del prep, X, holdouts

    for bl in baseline_specs(cfg):
        tqdm.write(f"\n→ {bl['label']}  ({describe_variant(bl)})")
        bl_dir = alg_dir / bl["name"]
        bl_dir.mkdir(parents=True, exist_ok=True)
        # One scalar column, so the holdout is loaded per baseline rather than
        # per feature set -- a baseline stands on the rows ITS column is filled
        # for, on a holdout exactly as on the corpus.
        bl_holdouts = {
            h.name: load_holdout(
                h, [bl["column"]], group_col, human_label,
                score_col=bl["column"],
            )
            for h in hold_specs
        }
        payload = run_baseline(cfg, spec, bl, bl_dir, holdouts=bl_holdouts)
        print_result(payload["alg_label"], payload["summary"])
        if PLOTS_ENABLED:
            plot_clf_id_result(bl_dir)
        rows.append(summarise(payload))

    noun = comparison_noun(cfg)
    # Named after the corpus as well as the config: with several corpora in one
    # run these headings and figure titles would otherwise be identical, and the
    # only thing that differs between them is which corpus produced them.
    corpus_label = f"{cfg['alg_label']} — {spec.label}"
    comparison = {
        "alg_id": cfg["alg_id"],
        "alg_label": corpus_label,
        "noun": noun,
        "subtitle": (
            f"{spec.source} · {cfg['model']['n_runs']} seeds · "
            f"grouped by {group_col!r} · held-out "
            f"{cfg['model'].get('test_size', 0.30):.0%} · chance = 50%"
        ),
        "corpus": spec.name,
        "corpus_label": spec.label,
        "corpus_source": spec.source,
        "split_composition": composition,
        "classes": BINARY_CLASSES,
        "config": cfg,
        "variants": rows,
    }
    with open(alg_dir / "in_domain_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print_ranking(corpus_label, rows, noun=noun)
    if PLOTS_ENABLED:
        plot_clf_id_comparison(alg_dir, comparison)
    return comparison


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_args(p, DEFAULT_CFG_DIR, DEFAULT_RESULTS_DIR)
    p.add_argument("--dry-run", action="store_true",
                   help="Run the preflight and stop, without training anything.")
    p.add_argument("--no-plots", action="store_true",
                   help="Write the JSON but skip the figures.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_paths = resolve_configs(args)

    report, loaded = preflight(cfg_paths)
    print_report(report, "every config, corpus, variant and split")
    if not report.ok:
        sys.exit(1)
    if args.dry_run:
        print("\nDry run — nothing was trained.")
        return

    if args.no_plots:
        global PLOTS_ENABLED
        PLOTS_ENABLED = False

    experiment_id = "clf_id_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = args.results_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    for cfg_path, cfg in tqdm(loaded, desc="Configs", unit="cfg"):
        run_config(cfg_path, cfg, exp_dir / cfg["alg_id"])

    print(f"\nDone. Results saved to: {rel(exp_dir)}")
    if args.no_plots:
        print(
            "Figures were skipped; draw them with "
            "epai/ai_detection/train/replot_in_domain.py"
        )


if __name__ == "__main__":
    main()
