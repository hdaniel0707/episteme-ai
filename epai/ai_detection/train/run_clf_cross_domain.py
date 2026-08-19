"""Cross-domain transfer control for the human-vs-machine classifier.

The in-domain numbers (~97-99%) survive a grouped split, a permuted-label
control and a length-only baseline, so the pipeline is sound and the signal is
real. The open question is what the signal *is*: has the classifier learned
"machine-generated-ness", or has it learned "this corpus, written by this
particular generator"? Those two are indistinguishable in-domain and come apart
under transfer.

Each config lists several datasets. Every dataset is trained on in turn and
evaluated on all of them, producing a source x target accuracy matrix:

    diagonal      -> in-domain, held-out grouped test split (the usual number)
    off-diagonal  -> transfer: never saw this corpus / this generator

A classifier that has learned a general notion of machine text keeps a healthy
off-diagonal. One that has learned corpus artifacts collapses towards 50%. The
mechanics of that matrix -- the grouped split, the binarised target, the
exclusion of target rows whose group was seen in training -- live in
``epai/utils/cd_core.py`` and are identical for everything below.

METRICS
-------
Every cell is reported under several metrics, each with its own matrix and its
own figure: accuracy and macro F1 from the predicted labels, and AUROC plus TPR
at 1% and 5% FPR from the continuous score behind them. Accusing a human is the
expensive error, so the low-FPR operating point is the one a deployed detector
is actually graded on.

TPR@FPR appears twice, and the pair is the point:

  * ``tpr_at_fpr_<level>``    -- (a) the threshold is fit on the cell being
    tested. This is what the literature reports, and it measures separability:
    it assumes human text from the target corpus is available to calibrate on.
  * ``tpr_at_srcfpr_<level>`` -- (b) the threshold is fit to that FPR on the
    SOURCE's held-out split and carried to the target unchanged, with
    ``fpr_at_srcfpr_<level>`` recording what it actually cost there.

A model can hold its AUROC and its (a) score under transfer while its (b) score
collapses, because the score distribution shifted even though the ordering
survived -- which is invisible to accuracy, to AUROC and to (a) alike. On the
diagonal (b) degenerates to (a) by construction; only the off-diagonal is new.

None of this is recoverable from a matrix already on disk: a confusion matrix
cannot be turned back into a ranking, so a run predating these metrics has to be
re-run rather than replotted.

VARIANTS
--------
What this script varies is *what* goes through the matrix. A run is the cross
product of two lists, and one element of it -- one feature set with one
algorithm -- is a **variant**:

    feature_sets: [emb_luar, emb_bge]      one design matrix per entry
    algorithms:   [logistic, simple_mlp]   the registry in epai/algs

Every ``feature_sets`` entry is one matrix, whether it names one column or five,
so all three questions the codebase used to ask with three separate scripts are
the same config shape here -- hold an axis at one entry to ask about the other:

  * one feature set, several algorithms -> which model transfers best;
  * several feature sets, one algorithm -> which representation transfers best;
  * one of each -> a single matrix, the plain control;
  * both -> every pair, ranked together.

An entry may be a column, a list of columns stacked into one matrix, or a mapping
with an explicit ``id`` / ``label``, which is also how a run compares a feature
set against a superset of itself::

    feature_sets:
      - emb_luar_crud                       # the embedding alone
      - [emb_luar_crud, binoculars_score]   # ...and with the score beside it

Each variant gets its own directory, its own matrix and figures, and a row in a
comparison that ranks them all by how well they transfer. Naming follows
whichever axis actually varies, so a pure feature sweep writes
``<alg_id>/emb_luar/`` and a pure algorithm sweep ``<alg_id>/logistic/``; only a
genuine 2-D run needs ``<features>__<alg>``.

Algorithms are pluggable ``ClfAlgorithm`` subclasses under ``epai/algs``
(logistic, svm, gradient_boosting, simple_mlp, binox_mlp), selected by ``name``
and configured with per-entry ``params``. Each owns its own preprocessing, and
each declares which ``params`` it reads: an entry naming anything else is a
preflight error rather than a silently ignored key, so the hyper-parameters
``run_sweep_cross_domain.py`` finds can be transcribed into a config and either
take effect or fail loudly. The resolved set (the entry's params plus the
algorithm's defaults) is printed before each matrix and stored in its
``cross_domain.json``.

An entry may set ``binoculars`` to hand that algorithm the Binoculars score as a
second input beside the design matrix -- a *hybrid*, marked with a ◆ in the
ranking and on the figures because it is not ranked on the same information as
the rest. The column stays out of the feature set (so the plain algorithms are
unaffected) but joins the missing-value filter (so every algorithm still trains
and is scored on identical rows).

BASELINES
---------
Optional zero-shot score baselines, each one more matrix marked with a ★ in the
comparison. Two families, configured by ``binoculars_baseline`` and
``llm_judge_baseline``, each contributing exactly ONE variant: the ``threshold``
its block names, fit to nothing at all and applied unchanged to every source and
target. Nothing about a baseline is calibrated -- the headline metrics (AUROC,
TPR at a fixed FPR) never read the declared cutoff, which enters only accuracy,
macro F1 and the confusion matrix. Both families read their score the same way
round -- *low* means machine for Binoculars by construction and for the LLM
judge because it reports P(human).

PREFLIGHT
---------
The interesting failures here are slow ones: a run can be fifteen embedders x
five algorithms x nine matrix cells x three seeds, and a column missing from the
third corpus would otherwise surface twenty minutes in. So everything checkable
without fitting a model is checked first, over *every* config, before *any*
training: each config's shape and values, that every parquet exists and holds
every column, that the feature dimensions agree across corpora, that enough rows
survive the missing-value drop, and that every grouped split the run will
actually take leaves both classes in the training half and leaves something to
evaluate on. Errors from all configs are collected and reported together.

A corpus is a ``CorpusSpec`` -- a parquet plus an optional ``domain`` -- so the
six datasets of a full matrix are six slices of two combined files rather than
six separate parquets. See ``epai/utils/corpus_utils.py``.

``cfg/ai_detection/train/clf_cd/clf_cd_example.yaml`` is the annotated sample and
the folder's only config, so the zero-argument run below is that experiment.
Copy it to start a new one; each run also writes its config beside its results.

Run:
    uv run python epai/ai_detection/train/run_clf_cross_domain.py
    uv run python epai/ai_detection/train/run_clf_cross_domain.py --dry-run
    uv run python epai/ai_detection/train/run_clf_cross_domain.py \
        --cfg cfg/ai_detection/train/clf_cd/clf_cd_example.yaml --no-plots
    
    uv run python epai/ai_detection/train/run_clf_cross_domain.py \
            --cfg cfg/ai_detection/train/clf_cd/clf_cd_science_v3.yaml
    
    uv run python epai/ai_detection/train/run_clf_cross_domain.py \
        --cfg cfg/ai_detection/train/clf_cd/clf_cd_ghostbuster_gpt35t.yaml
    
    uv run python epai/ai_detection/train/run_clf_cross_domain.py \
        --cfg cfg/ai_detection/train/clf_cd/clf_cd_ghostbuster_gpt54mini.yaml

    uv run python epai/ai_detection/train/run_clf_cross_domain.py \
        --cfg cfg/ai_detection/train/clf_cd/clf_cd_ghostbuster_gpt56luna.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import json
import shutil
import sys

from tqdm import tqdm

from epai.algs import build_algorithm
from epai.utils.cd_core import (
    BINARY_CLASSES,
    Prepared,
    aggregate_seeds,
    baseline_specs,
    calibrate_thresholds,
    check_baseline_datasets,
    check_dataset,
    check_splits,
    describe_variant,
    grouped_split,
    load_prepared,
    metrics_dict,
    print_matrix,
    print_ranking,
    run_score_baseline,
    summarise_baseline,
    summarise_variant,
)
from epai.utils.config_utils import (
    RESET,
    YELLOW,
    add_config_args,
    rel,
    resolve_configs,
)
from epai.utils.corpus_utils import CorpusSpec, dataset_specs
from epai.utils.preflight_utils import RULE, Report, load_yaml, print_report
from epai.utils.train_utils import stack_features
from epai.utils.variant_utils import (
    Variant,
    algorithm_specs,
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
    plot_clf_cd_comparison,
    plot_clf_cd_result,
)

BASE_DIR = Path(__file__).parent.parent.parent.parent
DEFAULT_CFG_DIR = BASE_DIR / "cfg" / "ai_detection" / "train" / "clf_cd"
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "clf_cd"

#: Cleared by ``--no-plots``. The figures are drawn as each variant finishes, so
#: that a long run interrupted half-way still leaves usable output -- which means
#: suppressing them is a flag consulted in the loop, not a final step to skip.
#: ``replot_cross_domain.py`` draws every one of them from the JSON afterwards.
PLOTS_ENABLED = True


# ---------------------------------------------------------------------------
# Config: the two axes
# ---------------------------------------------------------------------------



def preflight_config(cfg: dict, report: Report) -> None:
    """Everything that can be checked about one config before a model is fitted.

    Runs to the end and records all problems: a run this long should not be
    restarted once per typo.
    """
    print("\n[1/3] config")
    n_errors = len(report.errors)
    check_config(cfg, report)
    if len(report.errors) > n_errors:
        # The checks below index into the config; stop while it is still unsafe.
        print("  invalid — skipping the data checks for this config")
        return

    fsets = build_feature_sets(cfg)
    variants = build_variants(cfg)
    baseline_variants = baseline_specs(cfg)
    n_datasets = len(cfg["data"]["datasets"])
    n_fits = (len(variants) + len(baseline_variants)) * cfg["model"]["n_runs"] * n_datasets
    baseline_note = (
        " + score baselines "
        + ", ".join(f"{v['label']} ({v['column']})" for v in baseline_variants)
        if baseline_variants
        else ""
    )
    axis = (
        f"{len(fsets)} feature sets"
        if len(fsets) > 1
        else f"features={list(fsets[0].columns) if fsets else []}"
    )
    print(
        f"  {n_datasets} datasets, {axis}, {len(algorithm_specs(cfg))} algorithms"
        f"{baseline_note}, {cfg['model']['n_runs']} seeds  → "
        f"{len(variants)} variant(s), {n_fits} fits"
    )
    hybrids = [v for v in variants if v.hybrid]
    if hybrids:
        # These see more than the feature set, so say so before anything runs.
        print(
            f"  {len(hybrids)} hybrid variant(s) also take an auxiliary column: "
            + ", ".join(f"{v.id} ({v.aux_col})" for v in hybrids)
        )
        print(
            f"  every variant is filtered on its features plus {aux_columns(cfg)} "
            f"so they all train and score on the same rows"
        )

    print("\n[2/3] parquet files")
    n_errors = len(report.errors)
    check_files(cfg, dataset_specs(cfg, BASE_DIR), report)
    if len(report.errors) > n_errors:
        # Without the columns there is nothing to load in stage 3.
        print("  unusable — skipping the data checks for this config")
        return

    print("\n[3/3] data, dimensions and splits")
    group_col: str = cfg["data"]["group_col"]
    human_label: str = cfg["data"]["human_label"]
    specs = dataset_specs(cfg, BASE_DIR)

    print(
        f"{'feature set':<22}{'dataset':<20}{'rows':>7}{'dropped':>9}"
        f"{'dim':>7}{'groups':>8}  classes"
    )
    print("-" * RULE)

    # One pass per feature set: the algorithms sharing it see identical rows, so
    # checking it once per algorithm would only repeat the reads.
    for fs in fsets:
        columns = variant_columns(fs, cfg)
        preps: dict[str, Prepared] = {}
        for spec in specs:
            try:
                prep = load_prepared(spec, columns, group_col, human_label)
            except Exception as exc:
                report.error(
                    f"{fs.id} | {spec.name}: failed to load: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            ok = check_dataset(prep, cfg, fs.id, report)
            dropped = prep.n_class_dropped + prep.n_missing_dropped
            # 'dim' is the shared design matrix only; auxiliary columns are handed
            # to their algorithm separately and never widen it.
            print(
                f"{fs.id:<22}{prep.name:<20}{len(prep.df):>7}{dropped:>9}"
                f"{(prep.dim_of(list(fs.columns)) if prep.dims_per_col else 0):>7}"
                f"{(prep.n_groups if len(prep.df) else 0):>8}  "
                f"{prep.class_counts if len(prep.df) else '-'}"
            )
            if ok:
                # Keep only what the split check needs; the vectors are large and
                # the training loop re-reads them one feature set at a time.
                preps[prep.name] = Prepared(
                    name=prep.name,
                    df=prep.df[["y", "group"]].copy(),
                    n_file=prep.n_file,
                    n_class_dropped=prep.n_class_dropped,
                    n_missing_dropped=prep.n_missing_dropped,
                    missing_per_col=prep.missing_per_col,
                    dims_per_col=prep.dims_per_col,
                    n_filtered=prep.n_filtered,
                )
            del prep

        if len(preps) != len(specs):
            continue  # a dataset already failed; its splits are not meaningful

        # Features must live in one shared space to be transferable: the same
        # columns, in the same order, and the same width for every corpus.
        dims = {name: p.dim_of(list(fs.columns)) for name, p in preps.items()}
        if len(set(dims.values())) != 1:
            report.error(
                f"{fs.id}: datasets disagree on feature dimensionality: {dims} "
                f"— a model trained on one corpus cannot be applied to another"
            )
            continue

        check_splits(preps, cfg, fs.id, report)

    # The baselines are filtered on their own score column instead of a feature
    # set, so their row set is checked separately -- a partly filled column (the
    # judge only scores the generators it was asked for) means they stand on
    # fewer rows than the trained variants, and that has to be visible.
    check_baseline_datasets(cfg, report)


def preflight(cfg_paths: list[Path]) -> tuple[Report, list[tuple[Path, dict]]]:
    """Check every config in the folder. Returns the report and the loaded configs."""
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

    # Each alg_id names an output directory, so a repeat would have two configs
    # writing into the same place.
    report.context = ""
    alg_ids = [cfg.get("alg_id") for _, cfg in loaded if isinstance(cfg, dict)]
    dupes = sorted({a for a in alg_ids if a is not None and alg_ids.count(a) > 1})
    if dupes:
        report.error(f"alg_id must be unique across the configs run, repeated: {dupes}")

    return report, loaded


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_variant(
    variant: Variant,
    cfg: dict,
    X_all: dict,
    data: dict,
    names: list[str],
    labels: dict,
    out_dir: Path,
    save_models: bool = True,
) -> dict:
    """Full source x target matrix for one variant. Returns the saved payload.

    ``X_all`` / ``data`` are loaded once per feature set by ``run_config`` and
    shared across every algorithm trained on it.

    ``save_models`` writes one pickled model per (seed, source). The
    hyper-parameter sweep turns it off: it fits thousands of throwaway candidates
    and only the winner is ever reloaded, so keeping them all would cost
    gigabytes for nothing.
    """
    base_seed: int = cfg["seed"]
    n_runs: int = cfg["model"]["n_runs"]
    test_size: float = cfg["model"].get("test_size", 0.30)

    feature_dim = int(next(iter(X_all.values())).shape[1])

    # The auxiliary column, kept out of X so it can enter the network late. Per
    # variant, since two entries may name different columns; one scalar column
    # per corpus, so stacking it here rather than in run_config costs nothing.
    Z_all = (
        {n: stack_features(df, [variant.aux_col])[0] for n, df in data.items()}
        if variant.aux_col
        else {n: None for n in data}
    )

    # per_seed[i][source][target] -> metrics
    per_seed: list[dict] = []

    for i in tqdm(range(n_runs), desc="Seeds", leave=False):
        seed = base_seed + i
        seed_model_dir = out_dir / str(i) / "model"
        if save_models:
            seed_model_dir.mkdir(parents=True, exist_ok=True)
        seed_res: dict[str, dict] = {}

        for source in names:
            df_s = data[source]
            X_s, y_s = X_all[source], df_s["y"].to_numpy()
            Z_s = Z_all[source]
            groups_s = df_s["group"].to_numpy()

            train_idx, test_idx = grouped_split(y_s, groups_s, test_size, seed)
            train_groups = set(groups_s[train_idx])

            # A fresh algorithm per (seed, source); each owns its preprocessing.
            alg = build_algorithm(variant.alg_name, seed, **variant.params)
            # Groups for the training rows only. An algorithm that carves its own
            # validation split out of them (the torch ones, under `val_frac`) needs
            # these to keep a document and its rewrites on one side of that split,
            # the same guarantee the outer split above gives. Ignored by the rest.
            alg.set_fit_groups(groups_s[train_idx])
            fit_alg(
                alg,
                X_s[train_idx],
                y_s[train_idx],
                Z_s[train_idx] if Z_s is not None else None,
            )

            # The source's held-out split, scored once: it is the in-domain
            # evaluation set AND the calibration set for the transported
            # thresholds of variant (b). Held-out rather than training rows on
            # purpose -- a model's scores on the rows it was fitted on push the
            # human distribution away from the boundary, which would make the
            # transported threshold reflect the overfit rather than the corpus.
            #
            # The consequence, stated plainly because no key can carry it: on the
            # diagonal the calibration set *is* the evaluation set, so the
            # transported reading degenerates to the target-calibrated one there
            # and only the off-diagonal cells of the (b) matrices are new
            # information.
            s_test = score_alg(
                alg, X_s[test_idx], Z_s[test_idx] if Z_s is not None else None
            )
            cal_thresholds = (
                calibrate_thresholds(y_s[test_idx], s_test)
                if s_test is not None
                else None
            )

            seed_res[source] = {}
            for target in names:
                if target == source:
                    # In-domain: the corpus's own held-out grouped test split.
                    res = metrics_dict(
                        y_s[test_idx],
                        predict_alg(
                            alg,
                            X_s[test_idx],
                            Z_s[test_idx] if Z_s is not None else None,
                        ),
                        scores=s_test,
                        cal_thresholds=cal_thresholds,
                    )
                    res["excluded"] = 0
                else:
                    df_t = data[target]
                    # `essay` and `essay_gpt54mini` share their human documents
                    # (same source files, same ids). Any target row whose group
                    # was trained on would be a free hit, so drop it.
                    keep = ~df_t["group"].isin(train_groups).to_numpy()
                    Z_t = Z_all[target]
                    X_t = X_all[target][keep]
                    Z_keep = Z_t[keep] if Z_t is not None else None
                    res = metrics_dict(
                        df_t.loc[keep, "y"].to_numpy(),
                        predict_alg(alg, X_t, Z_keep),
                        scores=(
                            score_alg(alg, X_t, Z_keep)
                            if cal_thresholds is not None
                            else None
                        ),
                        cal_thresholds=cal_thresholds,
                    )
                    res["excluded"] = int((~keep).sum())
                res["train_n"] = int(len(train_idx))
                seed_res[source][target] = res

            # Persist last: the torch algorithms' save() moves the model to CPU,
            # so saving before the predictions above would drop them off the GPU.
            # Best-effort.
            if save_models:
                try:
                    alg.save(seed_model_dir / f"{source}.joblib")
                except Exception as exc:  # persistence must never abort a long run
                    tqdm.write(
                        f"{YELLOW}  ⚠️  could not save {variant.alg_name} for "
                        f"{source} (seed {seed}): {type(exc).__name__}: {exc}{RESET}"
                    )

        per_seed.append(seed_res)

    summary = aggregate_seeds(per_seed, names)

    # The per-variant config copy the figures read: `features` and `model.ml_alg`
    # are what the matrix subtitles are built from, and `alg_id` / `alg_label`
    # name this variant rather than the whole config.
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
        # What the config asked for is `params`; this is what the algorithm made
        # of it, defaults filled in. Keeping both means a matrix can be traced
        # back to the exact model without re-deriving any default.
        "resolved_params": resolved_params(variant, base_seed),
        "binoculars": variant.aux_col,
        "feature_dim": feature_dim,
        "dataset_names": names,
        "dataset_labels": labels,
        "classes": BINARY_CLASSES,
        "config": var_cfg,
        "summary": summary,
        "per_seed": per_seed,
    }
    with open(out_dir / "cross_domain.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def comparison_subtitle(cfg: dict, n_corpora: int) -> str:
    """One line describing what was held fixed, for the comparison figures."""
    fsets = build_feature_sets(cfg)
    algs = algorithm_specs(cfg)

    if len(fsets) > 1:
        feats = f"{len(fsets)} feature sets"
    else:
        feats = "features: " + ", ".join(fsets[0].columns) if fsets else "features: -"
        if len(feats) > 70:
            feats = feats[:67] + "…"
    # Name the extra columns: the ◆ rows did not run on the feature set alone,
    # and the figure should say so without needing the JSON.
    aux = aux_columns(cfg)
    if aux:
        feats += f" (◆ also + {', '.join(aux)})"

    algo = algs[0]["label"] if len(algs) == 1 else f"{len(algs)} algorithms"
    # No chance level here: this line is shared by every per-metric comparison
    # figure, and chance is 50% for accuracy / macro F1 / AUROC but the nominal
    # FPR for the TPR@FPR ones. Each figure prints its own reference instead.
    return (
        f"{feats} · {algo} · {cfg['model']['n_runs']} seeds · "
        f"grouped by {cfg['data']['group_col']!r} · {n_corpora} corpora"
    )


def run_config(cfg_path: Path, cfg: dict, alg_dir: Path) -> None:
    """Every variant of one config, plus its comparison figures."""
    alg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, alg_dir / "config.yaml")
    with open(alg_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    group_col: str = cfg["data"]["group_col"]
    human_label: str = cfg["data"]["human_label"]
    specs_ds: list[CorpusSpec] = dataset_specs(cfg, BASE_DIR)
    names = [spec.name for spec in specs_ds]
    labels = {spec.name: spec.label for spec in specs_ds}
    fsets = build_feature_sets(cfg)
    variants = build_variants(cfg)
    aux = aux_columns(cfg)

    tqdm.write("\n" + "=" * RULE)
    tqdm.write(f"  ALG ID     : {cfg['alg_id']}")
    tqdm.write(f"  VARIANTS   : {[v.id for v in variants]}")
    if len(fsets) > 1:
        tqdm.write(f"  FEATURES   : {len(fsets)} sets — {[fs.id for fs in fsets]}")
    elif fsets:
        tqdm.write(f"  FEATURES   : {list(fsets[0].columns)}")
    tqdm.write(f"  ALGORITHMS : {[s['id'] for s in algorithm_specs(cfg)]}")
    if aux:
        tqdm.write(f"  AUXILIARY  : {aux}  (hybrid entries only)")
    tqdm.write(
        f"  SPLIT      : grouped by {group_col!r}, "
        f"test_size={cfg['model'].get('test_size', 0.30)}, "
        f"{cfg['model']['n_runs']} seeds (base {cfg['seed']})"
    )
    tqdm.write(f"  OUTPUT     : {alg_dir}")
    tqdm.write("=" * RULE)

    rows: list[dict] = []
    for fs in tqdm(fsets, desc="Feature sets", unit="set", leave=False):
        # Load and stack one feature set once; every algorithm on it reuses the
        # result. The missing-value filter runs over the features and the
        # auxiliary columns together, so a row a hybrid algorithm cannot use is
        # absent for the plain ones too and the comparison stays like-for-like.
        columns = variant_columns(fs, cfg)
        tqdm.write(f"\n{'─' * 60}\n  feature set: {fs.id}")
        data = {}
        for spec in specs_ds:
            prep = load_prepared(spec, columns, group_col, human_label)
            tqdm.write(
                f"  {spec.name:<20} {len(prep.df):>6} rows  {prep.class_counts}"
            )
            data[spec.name] = prep.df
        X_all = {
            name: stack_features(df, list(fs.columns))[0] for name, df in data.items()
        }

        for variant in tqdm(
            [v for v in variants if v.fs_id == fs.id],
            desc="Variants",
            unit="var",
            leave=False,
        ):
            hybrid_note = f" + {variant.aux_col}" if variant.aux_col else ""
            tqdm.write(f"\n→ {variant.id}  ({variant.alg_name}{hybrid_note})")
            # The resolved set, not the config's diff against the defaults: when
            # the params came out of a sweep leaderboard this is the line that
            # confirms the transcription took effect.
            eff = resolved_params(variant, cfg["seed"])
            tqdm.write(
                "   params: " + ", ".join(f"{k}={v}" for k, v in sorted(eff.items()))
            )
            out_dir = alg_dir / variant.id
            out_dir.mkdir(parents=True, exist_ok=True)

            payload = run_variant(
                variant, cfg, X_all, data, names, labels, out_dir
            )
            print_matrix(
                payload["alg_label"], payload["summary"], payload["dataset_names"]
            )

            # Plot as we go: an interrupted run still leaves finished variants
            # with their figures.
            if PLOTS_ENABLED:
                plot_clf_cd_result(out_dir)
            row = summarise_variant(payload)
            if variant.aux_col:
                # Flags the row for the ranking and the figures: it saw the shared
                # feature set *and* the score, so it is not on equal footing with
                # the plain algorithms and must not read as though it were.
                row["hybrid"] = True
                row["aux_column"] = variant.aux_col
            rows.append(row)

        del data, X_all

    # Optional zero-shot score baselines: one row per enabled family, at the
    # fixed cutoff its block names. Binoculars and the LLM judge both read low ⇒
    # machine; the variant carries its own direction, so both are run by the same
    # loop. Run after the variants so they show up as extra rows in the
    # comparison.
    for bl in baseline_specs(cfg):
        tqdm.write(f"\n→ {bl['label']}  ({describe_variant(bl)})")
        bl_dir = alg_dir / bl["name"]
        bl_dir.mkdir(parents=True, exist_ok=True)
        payload = run_score_baseline(cfg, bl_dir, bl)
        print_matrix(
            payload["alg_label"], payload["summary"], payload["dataset_names"]
        )
        if PLOTS_ENABLED:
            plot_clf_cd_result(bl_dir)
        rows.append(summarise_baseline(payload))

    noun = comparison_noun(cfg)
    comparison = {
        "alg_id": cfg["alg_id"],
        "alg_label": cfg["alg_label"],
        "noun": noun,
        "subtitle": comparison_subtitle(cfg, len(names)),
        "dataset_names": names,
        "dataset_labels": labels,
        "classes": BINARY_CLASSES,
        "config": cfg,
        "variants": rows,
    }
    with open(alg_dir / "variant_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print_ranking(cfg["alg_label"], rows, noun=noun)
    if PLOTS_ENABLED:
        plot_clf_cd_comparison(alg_dir, comparison)


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
                   help="Write the matrices and JSON but skip the figures.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_paths = resolve_configs(args)

    # Every config is checked before any of them trains: a folder-wide run
    # should not get half-way through and then stop on the last file.
    report, loaded = preflight(cfg_paths)
    print_report(report, "every config, dataset, variant and split")
    if not report.ok:
        sys.exit(1)
    if args.dry_run:
        print("\nDry run — nothing was trained.")
        return

    if args.no_plots:
        global PLOTS_ENABLED
        PLOTS_ENABLED = False

    experiment_id = "clf_cd_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = args.results_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    for cfg_path, cfg in tqdm(loaded, desc="Configs", unit="cfg"):
        run_config(cfg_path, cfg, exp_dir / cfg["alg_id"])

    print(f"\nDone. Results saved to: {rel(exp_dir)}")
    if args.no_plots:
        print(
            "Figures were skipped; draw them with "
            "epai/ai_detection/train/replot_cross_domain.py"
        )


if __name__ == "__main__":
    main()
