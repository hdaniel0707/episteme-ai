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
own figure: accuracy and macro F1 from the predicted labels -- at a Youden's J
threshold fit on the calibration slice, not an algorithm's default 0.5 -- and
AUROC plus TPR at 1% and 5% FPR from the continuous score behind them.
Accusing a human is the expensive error, so the low-FPR operating point is the
one a deployed detector is actually graded on.

TPR@FPR appears twice, and the pair is the point:

  * ``tpr_at_fpr_<level>``    -- (a) the threshold is fit on the cell being
    tested. This is what the literature reports, and it measures separability:
    it assumes human text from the target corpus is available to calibrate on.
  * ``tpr_at_srcfpr_<level>`` -- (b) the threshold is fit to that FPR on the
    SOURCE's calibration slice -- a grouped cut of the training pool, never
    fit on and never evaluated on (see ``cd_core.carve_calibration``) -- and
    carried to the target unchanged, with ``fpr_at_srcfpr_<level>`` recording
    what it actually cost there.

A model can hold its AUROC and its (a) score under transfer while its (b) score
collapses, because the score distribution shifted even though the ordering
survived -- which is invisible to accuracy, to AUROC and to (a) alike. (b) does
NOT degenerate to (a) on the diagonal: the calibration slice is disjoint from
the evaluation set in every cell, so every cell of the (b) matrix is new
information, not just the off-diagonal.

The mechanism behind (b) is drawn rather than only reported. Each variant gets
``cross_domain_score_hist_0.01.png``: a source x target grid of the human (blue)
and machine (red) score distributions, with the transported cut placed on each
row's diagonal panel and then drawn unmoved across the row. A detector whose
distributions merely shift shows up there as red mass sliding across a
stationary line -- which is what (b) measures and what accuracy, AUROC and (a)
all miss. It is fed by the raw first-seed scores stored under ``score_dist``
(see ``cd_core.SCORE_DIST_KEY``).

None of this is recoverable from a matrix already on disk: a confusion matrix
cannot be turned back into a ranking, and per-document scores were not kept
before ``score_dist`` existed, so a run predating either has to be re-run rather
than replotted.

MODE: HOW MANY CORPORA EACH MODEL TRAINS ON
-------------------------------------------
The matrix above is one of two training designs this script runs, chosen by the
top-level ``mode`` key. Both produce a matrix of exactly the same shape, over
exactly the same corpora, with exactly the same fit count -- and their cells
mean opposite things, which is why the mode is printed by the preflight, drawn
into every figure title and stored on every payload.

``mode: cross`` (the default, and every config written before this key). The
numbers in both sketches below are INVENTED and are there to show the shape and
where the in-domain cells land::

              tested on
              essay   reuter   wp
    trained   [98.1]   51.2   49.7      [ ] = in-domain, the diagonal
    on        49.9   [97.6]   50.4      everything else = transfer
              50.8    52.0  [98.4]

Each model sees ONE corpus. That is deliberately harsh -- a single-source model
has every incentive to memorise corpus artifacts -- and it is the design that
produced the finding this whole folder exists around: ~98% in-domain, at or
below chance off it.

``mode: lodo`` -- leave-one-domain-out::

                        tested on
                        essay   reuter   wp
    trained on          [51.0]  [97.2]  [97.9]   row `essay` = the model that
      reuter + wp                                never saw an essay
    trained on           97.4   [50.6]   97.6
      essay + wp
    trained on           97.8    97.1   [51.4]
      essay + reuter

Each model trains on the union of every corpus BUT one, so it does see domain
diversity -- the complementary and arguably fairer question. Row ``essay`` means
"essay was held out": on that row the DIAGONAL cell is the transfer number and
every other cell is a held-out grouped split of the training pool. The in-domain
mask is the exact inverse of ``cross``, so nothing downstream reads the diagonal
to find out which is which; every cell carries an ``in_domain`` flag written
from the run's ``cd_core.TrainingPlan``.

What does NOT change with the mode: the corpora, the variants, the seeds, the
grouped split, the group exclusion, the metrics, the baselines, the output
layout, the number of fits. Only which rows go into each model.

The third design -- train and test on one corpus, no matrix at all -- is
``run_clf_in_domain.py``, which also runs every corpus of a list plus the whole
file pooled. It is a separate script because with no transfer cells there is no
transfer mean, no gap and no ranking to sort them by.

SUITES: SEVERAL DATASETS IN ONE RUN
-----------------------------------
The matrix above answers "does this survive leaving its domain" for ONE dataset.
The question after it -- does the ranking it produced hold on a different corpus,
or against a different generator -- needs several such matrices, and that is what
``data.suites`` is::

    data:
      human_label: "human"
      group_col: "id"
      suites:
        - name: "ghostbuster_gpt35t"
          parquet_path: "data/parquet/ghostbuster_gpt35t.parquet"
          datasets: [essay, reuter, wp]      # this suite's matrix axes
        - name: "science_v3"
          parquet_path: "data/parquet/science_v3_gpt56luna_0811A.parquet"
          datasets: [computer_science, mathematics, sociology]

Each suite is a dataset: a parquet, and the corpora cut out of it that become the
axes of its own source x target matrix. Everything else -- ``feature_sets``,
``algorithms``, the baselines, ``model``, the seeds -- is declared once and
shared, which is what makes the resulting numbers comparable: the same variant,
trained the same way, on each dataset in turn.

Nothing about a suite changes what happens inside one. The run writes
``<alg_id>/<suite>/`` with exactly the contents a flat config writes into
``<alg_id>/``, and then adds one level above it: ``suite_comparison.json`` and
the ``suite_comparison_*`` figures, ranking every variant by its transfer score
meaned over the datasets, with each dataset still visible beside it. The mean is
UNWEIGHTED -- each dataset counts once -- and it is never printed without the
spread across datasets next to it, because a variant whose transfer swings by
dataset has not been measured, it has been sampled.

No cell ever crosses two datasets. ``essay`` and ``sociology`` are different
corpora with different generators; a matrix pooling all of them would invent
cells nobody trained for. What crosses datasets is the pair of summary numbers
each matrix already produces.

A config WITHOUT ``suites`` is unchanged, down to the directory layout: the two
shapes are told apart by the key, and ``data.datasets`` takes the path it always
did. Setting both is a preflight error.

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
Optional zero-shot score baselines, each one more matrix, whose row on every
comparison figure carries a ★ on its label and draws in the baseline colour. Two
families, configured by ``binoculars_baseline`` and ``llm_judge_baseline``, each
contributing one variant per score column it is given -- ``column:`` for one,
``columns:`` for several -- at the ``threshold`` that column names, fit to
nothing at all and applied unchanged to every source and target. Several columns
of one family is the same detector read off different machinery: Binoculars
scored with a second model pair, the judge with a second judge model. Nothing about a
baseline is calibrated -- the score metrics (AUROC, TPR at a fixed FPR) never
read the declared cutoff, which enters only accuracy, macro F1 and the confusion
matrix. Both families read their score the same way
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

``cfg/ai_detection/train/clf_cd/`` holds the ``mode: cross`` configs and is the
default ``--cfg-dir``, so the zero-argument run is that folder.
``cfg/ai_detection/train/clf_lodo/`` holds the ``mode: lodo`` ones and is reached
with ``--cfg`` (or ``--cfg-dir``), which keeps the two designs from training in
one invocation by accident. Each is an annotated sample as well as a working
experiment: copy one to start a new run, and each finished run also writes its
config beside its results.


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

Run:
    # the default folder: cfg/ai_detection/train/clf_cd/
    uv run python epai/ai_detection/train/run_clf_cross_domain.py
    uv run python epai/ai_detection/train/run_clf_cross_domain.py --dry-run

    # every dataset of the suite, with the comparison across them:
    uv run python epai/ai_detection/train/run_clf_cross_domain.py \
        --cfg cfg/ai_detection/train/clf_cd/clf_cd_suites_all.yaml
    uv run python epai/ai_detection/train/run_clf_cross_domain.py \
        --cfg cfg/ai_detection/train/clf_cd/clf_cd_suites_all.yaml --no-plots

    # the same variants, trained on two domains and tested on the third:
    uv run python epai/ai_detection/train/run_clf_cross_domain.py \
        --cfg cfg/ai_detection/train/clf_lodo/clf_lodo_suites_all.yaml

    # the in-domain control, on the same corpora plus the whole parquet:
    uv run python epai/ai_detection/train/run_clf_in_domain.py \
        --cfg cfg/ai_detection/train/clf_id/clf_id_domains_all.yaml
    
    uv run python epai/ai_detection/train/run_clf_cross_domain.py \
      --cfg cfg/ai_detection/train/clf_cd/clf_cd_suites_all.yaml \
            cfg/ai_detection/train/clf_lodo/clf_lodo_suites_all.yaml
    
    with in domain:
    uv run python epai/ai_detection/train/run_clf_in_domain.py \
        --cfg cfg/ai_detection/train/clf_id/clf_id_domains_all.yaml && \
    uv run python epai/ai_detection/train/run_clf_cross_domain.py \
        --cfg cfg/ai_detection/train/clf_cd/clf_cd_suites_all.yaml \
                cfg/ai_detection/train/clf_lodo/clf_lodo_suites_all.yaml                                                                   

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
    CAL_FRAC_DEFAULT,
    SCORE_DIST_KEY,
    SCORE_DIST_LEVEL,
    Prepared,
    aggregate_seeds,
    baseline_specs,
    calibrate_thresholds,
    carve_calibration,
    check_baseline_datasets,
    check_dataset,
    check_mode,
    check_plan,
    check_splits,
    describe_variant,
    grouped_split,
    label_predictions,
    level_tag,
    load_prepared,
    metrics_dict,
    pool_array,
    pool_frame,
    print_matrix,
    print_ranking,
    run_mode,
    run_score_baseline,
    score_dist_cell,
    score_dist_payload,
    source_labels,
    summarise_baseline,
    summarise_variant,
    training_plan,
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
    check_dataset_specs,
    check_eval_only_corpora,
    dataset_specs,
    eval_only_specs,
)
from epai.utils.holdout_core import (
    HOLDOUT_KEY,
    aggregate_holdout,
    baseline_holdout_evaluator,
    check_holdout_corpora,
    evaluate_variant_on_holdout,
    holdout_comparison,
    holdout_from_per_seed,
    holdout_row,
    load_holdout,
    print_holdout_table,
)
from epai.utils.holdout_viz import plot_holdout_comparison
from epai.utils.preflight_utils import RULE, Report, load_yaml, print_report
from epai.utils.suite_utils import (
    Suite,
    check_suites,
    has_suites,
    print_suite_ranking,
    suite_comparison_rows,
    suite_configs,
)
from epai.utils.train_utils import CLF_TARGET_COL, stack_features
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
    plot_clf_cd_suite_comparison,
)

BASE_DIR = Path(__file__).parent.parent.parent.parent
DEFAULT_CFG_DIR = BASE_DIR / "cfg" / "ai_detection" / "train" / "clf_cd"
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "clf_cd"

#: Cleared by ``--no-plots``. The figures are drawn as each variant finishes, so
#: that a long run interrupted half-way still leaves usable output -- which means
#: suppressing them is a flag consulted in the loop, not a final step to skip.
#: ``replot_cross_domain.py`` draws every one of them from the JSON afterwards.
PLOTS_ENABLED = True

#: One line per training design, printed by the preflight. The two produce
#: matrices of identical shape whose cells mean opposite things, so which one is
#: running is said out loud before any of it is read. See ``cd_core.RUN_MODES``.
MODE_BLURB = {
    "cross": "train on ONE corpus; the diagonal is in-domain, the rest transfer",
    "lodo": (
        "leave-one-domain-out: train on every corpus BUT one; the diagonal is "
        "the transfer cell, the rest is the training pool held out"
    ),
}


# ---------------------------------------------------------------------------
# Config: the two axes
# ---------------------------------------------------------------------------



def run_suites(cfg: dict) -> list[Suite]:
    """The datasets this config runs, as one ordinary config each.

    A flat config is one unnamed suite -- itself -- so everything downstream can
    loop over datasets without asking which shape the config was written in. The
    empty ``name`` is what tells the caller not to add a directory level or draw
    a cross-dataset comparison for a run that has nothing to compare.
    """
    if has_suites(cfg):
        return suite_configs(cfg)
    return [Suite(name="", label="", cfg=cfg)]


def preflight_config(cfg: dict, report: Report) -> None:
    """Everything that can be checked about one config before a model is fitted.

    Runs to the end and records all problems: a run this long should not be
    restarted once per typo.

    The shared axes -- the feature sets, the algorithms, the model block, the
    baselines -- are checked ONCE even when the config declares several datasets,
    because they are declared once; the data is then checked once per dataset,
    with every message tagged by the dataset it belongs to. A baseline column
    missing from one dataset's parquet is an error here rather than a dropped
    row later: a comparison whose ★ row silently covers two of four datasets is
    worse than one that refuses to start.
    """
    print("\n[1/3] config")
    n_errors = len(report.errors)
    check_mode(cfg, report, ("cross", "lodo"))
    check_config(
        cfg, report,
        data_check=check_suites if has_suites(cfg) else check_dataset_specs,
    )
    if len(report.errors) > n_errors:
        # The checks below index into the config; stop while it is still unsafe.
        print("  invalid — skipping the data checks for this config")
        return

    fsets = build_feature_sets(cfg)
    variants = build_variants(cfg)
    baseline_variants = baseline_specs(cfg)
    suites = run_suites(cfg)
    n_corpora = sum(len(s.cfg["data"]["datasets"]) for s in suites)
    n_fits = (len(variants) + len(baseline_variants)) * cfg["model"]["n_runs"] * n_corpora
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
    scope = (
        f"{len(suites)} datasets / {n_corpora} corpora"
        if has_suites(cfg)
        else f"{n_corpora} datasets"
    )
    print(
        f"  {scope}, {axis}, {len(algorithm_specs(cfg))} algorithms"
        f"{baseline_note}, {cfg['model']['n_runs']} seeds  → "
        f"{len(variants)} variant(s), {n_fits} fits"
    )
    # The mode is the experiment, so it is said before anything else about the
    # data: the fit count above is the same either way, and the numbers the run
    # produces mean opposite things.
    mode = run_mode(cfg)
    print(f"  mode: {mode} — {MODE_BLURB[mode]}" if mode in MODE_BLURB else
          f"  mode: {mode}")
    if has_suites(cfg):
        for suite in suites:
            names = [d["name"] for d in suite.cfg["data"]["datasets"]]
            print(f"    {suite.name:<26} {len(names)} corpora — {', '.join(names)}")
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
        # These see more than the feature set, so say so before anything runs.
        print(
            f"  {len(hybrids)} hybrid variant(s) also take an auxiliary column: "
            + ", ".join(f"{v.id} ({v.aux_col})" for v in hybrids)
        )
        print(
            f"  every variant is filtered on its features plus {aux_columns(cfg)} "
            f"so they all train and score on the same rows"
        )

    # Every dataset is checked, even after one of them fails: the point of a
    # preflight this thorough is that one pass names every parquet to fix.
    outer = report.context
    for suite in suites:
        if suite.name:
            print(f"\n{'·' * RULE}\ndataset: {suite.name}\n{'·' * RULE}")
            report.context = f"{outer} · {suite.name}" if outer else suite.name
        check_plan(
            suite.cfg,
            [d["name"] for d in suite.cfg["data"]["datasets"]],
            report,
        )
        preflight_data(suite.cfg, report)
    report.context = outer


def preflight_data(cfg: dict, report: Report) -> None:
    """Stages 2 and 3 for ONE dataset: its parquets, its rows, its splits."""
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
    fsets = build_feature_sets(cfg)
    # The eval-only corpora are not matrix axes and are checked apart from them:
    # they need every column every variant reads, and one class is legitimate.
    # Here rather than in preflight_config because a suite config carries one
    # `data` block per suite, and each is a separate set of columns to satisfy.
    check_eval_only_corpora(cfg, report, [s.name for s in specs])
    check_holdout_corpora(cfg, report, BASE_DIR)

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
    holdouts: dict | None = None,
) -> dict:
    """Full source x target matrix for one variant. Returns the saved payload.

    ``X_all`` / ``data`` are loaded once per feature set by ``run_config`` and
    shared across every algorithm trained on it.

    ``save_models`` writes one pickled model per (seed, source). The
    hyper-parameter sweep turns it off: it fits thousands of throwaway candidates
    and only the winner is ever reloaded, so keeping them all would cost
    gigabytes for nothing.

    A "source" is a row of the matrix, not necessarily a corpus: under
    ``mode: lodo`` it is every corpus but one, pooled. The plan is the only
    thing that knows the difference, and it is also what decides whether a cell
    is in-domain or transfer -- so nothing here compares ``source`` to
    ``target`` to find out.

    ``holdouts`` are the eval-only corpora (``epai/utils/holdout_core.py``).
    They are NOT targets: they are scored once per (seed, source), after the
    target loop, under the thresholds that source fitted, and filed under a key
    of their own. No cell of the matrix, no aggregate and no figure below reads
    them, so a config without ``data.eval_only`` produces the payload it always
    did.
    """
    base_seed: int = cfg["seed"]
    n_runs: int = cfg["model"]["n_runs"]
    test_size: float = cfg["model"].get("test_size", 0.30)
    cal_frac: float = cfg["model"].get("cal_frac", CAL_FRAC_DEFAULT)
    plan = training_plan(cfg, names)
    row_labels = source_labels(plan, labels)

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
    # seed_holdouts[source] -> one entry per seed, each {holdout: metrics}.
    # Stays empty unless the config names an eval-only corpus.
    seed_holdouts: dict[str, list[dict]] = {}
    # The first seed's raw scores, kept for the score-distribution figure. One
    # seed only; see cd_core.SCORE_DIST_KEY.
    dist_cells: dict[str, dict[str, dict]] = {}
    dist_thresholds: dict[str, float] = {}

    for i in tqdm(range(n_runs), desc="Seeds", leave=False):
        keep_scores = i == 0
        seed = base_seed + i
        seed_model_dir = out_dir / str(i) / "model"
        if save_models:
            seed_model_dir.mkdir(parents=True, exist_ok=True)
        seed_res: dict[str, dict] = {}

        for source in names:
            # The training pool: one corpus under `cross`, all but one under
            # `lodo`. Pooled here rather than at load time so every corpus is
            # still read from the parquet exactly once per feature set.
            #
            # Inside the seed loop on purpose, though the pool does not depend on
            # the seed: hoisting it would hold every pool's design matrix at once
            # (roughly N-1 extra copies of the corpus under `lodo`) to save a few
            # concatenations per variant, which is nothing beside fitting the
            # model. Under `cross` the pool is one corpus and `pool_*` returns the
            # loaded array itself, so neither cost exists there at all.
            members = plan.trained_on(source)
            df_s = pool_frame(members, data)
            X_s = pool_array(members, X_all)
            Z_s = pool_array(members, Z_all)
            y_s = df_s["y"].to_numpy()
            groups_s = df_s["group"].to_numpy()

            train_idx, test_idx = grouped_split(y_s, groups_s, test_size, seed)
            # The exclusion is keyed on the WHOLE training pool, fit rows and
            # calibration rows alike -- the calibration slice's labels were
            # still looked at (to choose a threshold), so a target group that
            # only appears there would still be a free hit if it were not
            # excluded.
            train_groups = set(groups_s[train_idx])

            # The calibration slice: carved out of the training pool, never
            # the fit rows and never the eval rows, in every mode. See
            # cd_core.carve_calibration. Offset from `seed` so the two draws
            # are independent rather than correlated permutations of the same
            # RNG stream.
            fit_idx, cal_idx = carve_calibration(
                y_s[train_idx], groups_s[train_idx], cal_frac, seed
            )
            fit_idx, cal_idx = train_idx[fit_idx], train_idx[cal_idx]

            # A fresh algorithm per (seed, source); each owns its preprocessing.
            alg = build_algorithm(variant.alg_name, seed, **variant.params)
            # Groups for the fit rows only. An algorithm that carves its own
            # validation split out of them (the torch ones, under `val_frac`) needs
            # these to keep a document and its rewrites on one side of that split,
            # the same guarantee the outer split above gives. Ignored by the rest.
            alg.set_fit_groups(groups_s[fit_idx])
            fit_alg(
                alg,
                X_s[fit_idx],
                y_s[fit_idx],
                Z_s[fit_idx] if Z_s is not None else None,
            )

            # The calibration slice, scored once: it is what the transported
            # thresholds of variant (b) and the Youden's J threshold behind
            # accuracy/macro F1 are fit on. Never the fit rows (a model's
            # scores on rows it was fitted on push the human distribution away
            # from the boundary, which would make the threshold reflect the
            # overfit rather than the corpus) and never the eval rows (which
            # would be the oracle reading in disguise) -- true in every cell,
            # including the diagonal under `cross` and the held-out row under
            # `lodo`, which is the whole point of the carve.
            s_cal = score_alg(
                alg, X_s[cal_idx], Z_s[cal_idx] if Z_s is not None else None
            )
            cal_thresholds = (
                calibrate_thresholds(y_s[cal_idx], s_cal)
                if s_cal is not None
                else None
            )
            label_threshold = (
                youden_threshold(y_s[cal_idx], s_cal) if s_cal is not None else None
            )

            if keep_scores and cal_thresholds is not None:
                dist_cells[source] = {}
                dist_thresholds[source] = float(
                    cal_thresholds[level_tag(SCORE_DIST_LEVEL)]
                )

            seed_res[source] = {}
            for target in names:
                if len(members) == 1 and target == source:
                    # In-domain, and the pool IS this corpus: its own held-out
                    # grouped test split.
                    Z_test = Z_s[test_idx] if Z_s is not None else None
                    s_test = score_alg(alg, X_s[test_idx], Z_test)
                    res = metrics_dict(
                        y_s[test_idx],
                        label_predictions(
                            predict_alg(alg, X_s[test_idx], Z_test),
                            s_test,
                            label_threshold,
                        ),
                        scores=s_test,
                        cal_thresholds=cal_thresholds,
                        label_threshold=label_threshold,
                    )
                    res["excluded"] = 0
                    if keep_scores and s_test is not None and source in dist_cells:
                        dist_cells[source][target] = score_dist_cell(
                            s_test, y_s[test_idx]
                        )
                else:
                    df_t = data[target]
                    # `essay` and `essay_gpt54mini` share their human documents
                    # (same source files, same ids). Any target row whose group
                    # was trained on would be a free hit, so drop it.
                    #
                    # This is also what makes a pooled row's OWN corpora into
                    # held-out splits under `lodo`: excluding the training
                    # groups from `essay` leaves exactly the part of `essay`
                    # that the pool's split held out. Same rule, same code path;
                    # only the plan says which of the two readings it is.
                    keep = ~df_t["group"].isin(train_groups).to_numpy()
                    Z_t = Z_all[target]
                    X_t = X_all[target][keep]
                    Z_keep = Z_t[keep] if Z_t is not None else None
                    s_t = (
                        score_alg(alg, X_t, Z_keep)
                        if cal_thresholds is not None
                        else None
                    )
                    res = metrics_dict(
                        df_t.loc[keep, "y"].to_numpy(),
                        label_predictions(
                            predict_alg(alg, X_t, Z_keep), s_t, label_threshold
                        ),
                        scores=s_t,
                        cal_thresholds=cal_thresholds,
                        label_threshold=label_threshold,
                    )
                    res["excluded"] = int((~keep).sum())
                    if keep_scores and s_t is not None and source in dist_cells:
                        dist_cells[source][target] = score_dist_cell(
                            s_t, df_t.loc[keep, "y"].to_numpy()
                        )
                res["train_n"] = int(len(fit_idx))
                res["cal_n"] = int(len(cal_idx))
                seed_res[source][target] = res

            # The eval-only corpora, under this source's thresholds. Outside the
            # target loop and keyed apart from it: a holdout is not a target and
            # must never become a column of the matrix. Scored before the model
            # is persisted, for the same reason the targets are -- save() moves a
            # torch model to CPU.
            if holdouts:
                seed_holdouts.setdefault(source, []).append(
                    {
                        name: evaluate_variant_on_holdout(
                            alg, hold, cal_thresholds, label_threshold,
                            train_groups=train_groups,
                            aux=(
                                stack_features(hold.df, [variant.aux_col])[0]
                                if variant.aux_col and len(hold.df)
                                else None
                            ),
                        )
                        for name, hold in holdouts.items()
                    }
                )

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

    summary = aggregate_seeds(per_seed, names, plan=plan)

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
        "mode": plan.mode,
        "dataset_names": names,
        "dataset_labels": labels,
        # The row axis, which is the corpus axis only under `cross`. Kept
        # separately so a figure never labels the one model that never saw an
        # essay "essay".
        "source_labels": row_labels,
        "source_members": {n: plan.trained_on(n) for n in names},
        "classes": BINARY_CLASSES,
        "config": var_cfg,
        "summary": summary,
        "per_seed": per_seed,
        SCORE_DIST_KEY: score_dist_payload(
            dist_cells, dist_thresholds,
            seed=base_seed, level=SCORE_DIST_LEVEL, machine_is_high=True,
        ),
    }
    if seed_holdouts:
        payload[HOLDOUT_KEY] = holdout_from_per_seed(seed_holdouts)
    if variant.aux_col:
        payload["hybrid"] = True
        payload["aux_column"] = variant.aux_col
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
    design = "leave-one-domain-out · " if run_mode(cfg) == "lodo" else ""
    # No chance level here: this line is shared by every per-metric comparison
    # figure, and chance is 50% for accuracy / macro F1 / AUROC but the nominal
    # FPR for the TPR@FPR ones. Each figure prints its own reference instead.
    return (
        f"{design}{feats} · {algo} · {cfg['model']['n_runs']} seeds · "
        f"grouped by {cfg['data']['group_col']!r} · {n_corpora} corpora"
    )


def suite_subtitle(cfg: dict, suites: list[Suite]) -> str:
    """The same line, for the figures that span every dataset of a run.

    It says how many datasets and how many corpora in total, because the two are
    different guarantees: the variants are identical across datasets (one
    ``algorithms`` list), the corpora are not (each dataset cuts its own).
    """
    n_corpora = sum(len(s.cfg["data"]["datasets"]) for s in suites)
    base = comparison_subtitle(cfg, n_corpora)
    return base.replace(
        f"{n_corpora} corpora",
        f"{len(suites)} datasets / {n_corpora} corpora",
    )


def run_config(cfg_path: Path, cfg: dict, alg_dir: Path) -> None:
    """One config: every dataset it names, and the comparison across them.

    A flat config is one dataset and this is exactly what it always was -- the
    variants, their matrices, their figures, the within-dataset ranking, all
    written straight into ``alg_dir``.

    A config with ``data.suites`` runs that whole thing once per dataset, one
    directory deeper, and then adds the level the suites exist for: one row per
    variant holding its per-dataset numbers, ranked by the unweighted mean over
    datasets, written as ``suite_comparison.json`` beside the dataset
    directories and drawn by ``plot_clf_cd_suite_comparison``.
    """
    alg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, alg_dir / "config.yaml")
    with open(alg_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    if not has_suites(cfg):
        run_suite(cfg, alg_dir)
        return

    suites = suite_configs(cfg)
    per_suite: dict[str, list[dict]] = {}
    for suite in tqdm(suites, desc="Datasets", unit="ds"):
        suite_dir = alg_dir / suite.name
        suite_dir.mkdir(parents=True, exist_ok=True)
        # The derived config, so a dataset directory says what produced it
        # without anyone having to re-apply the suite defaults by hand.
        with open(suite_dir / "config.json", "w") as f:
            json.dump(suite.cfg, f, indent=2)
        per_suite[suite.name] = run_suite(suite.cfg, suite_dir)["variants"]

    names = [s.name for s in suites]
    labels = {s.name: s.label for s in suites}
    rows = suite_comparison_rows(per_suite, names)
    comparison = {
        "alg_id": cfg["alg_id"],
        "alg_label": cfg["alg_label"],
        "noun": comparison_noun(cfg),
        "subtitle": suite_subtitle(cfg, suites),
        "suite_names": names,
        "suite_labels": labels,
        # Which corpora each dataset's own matrix ran between. Nothing reads it
        # to plot; it is what makes the JSON answer "what is behind this cell"
        # without opening four other files.
        "suite_datasets": {
            s.name: [spec.name for spec in dataset_specs(s.cfg, BASE_DIR)]
            for s in suites
        },
        "classes": BINARY_CLASSES,
        "config": cfg,
        "variants": rows,
    }
    with open(alg_dir / "suite_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print_suite_ranking(
        cfg["alg_label"], rows, names, labels, noun=comparison["noun"]
    )
    if PLOTS_ENABLED:
        plot_clf_cd_suite_comparison(alg_dir, comparison)


def run_suite(cfg: dict, alg_dir: Path) -> dict:
    """Every variant of ONE dataset, plus its comparison figures.

    Returns the ``variant_comparison`` payload it wrote, which is what the
    cross-dataset layer above stacks. ``cfg`` is a flat cross-domain config
    whether it was written as one or derived from a suite entry, so nothing
    below this line knows the difference.
    """
    alg_dir.mkdir(parents=True, exist_ok=True)

    group_col: str = cfg["data"]["group_col"]
    human_label: str = cfg["data"]["human_label"]
    specs_ds: list[CorpusSpec] = dataset_specs(cfg, BASE_DIR)
    names = [spec.name for spec in specs_ds]
    labels = {spec.name: spec.label for spec in specs_ds}
    fsets = build_feature_sets(cfg)
    variants = build_variants(cfg)
    aux = aux_columns(cfg)

    plan = training_plan(cfg, names)

    tqdm.write("\n" + "=" * RULE)
    tqdm.write(f"  ALG ID     : {cfg['alg_id']}")
    tqdm.write(f"  MODE       : {plan.mode} — {MODE_BLURB.get(plan.mode, '')}")
    if plan.mode != "cross":
        for name in names:
            tqdm.write(
                f"               row {name!r}: trained on "
                f"{', '.join(plan.trained_on(name))}"
            )
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
    hold_specs = eval_only_specs(cfg, BASE_DIR)
    if hold_specs:
        tqdm.write(
            f"  EVAL-ONLY  : {[h.name for h in hold_specs]}  "
            f"(scored under each row's thresholds; never trained on, never a column)"
        )
    tqdm.write(f"  OUTPUT     : {alg_dir}")
    tqdm.write("=" * RULE)

    holdout_payloads: list[dict] = []
    rows: list[dict] = []
    for fs in tqdm(fsets, desc="Feature sets", unit="set", leave=False):
        # An entry's `feature_sets` restriction can leave a set with nothing to
        # train, and loading four corpora to fit nothing on them is minutes of
        # I/O for no output. The preflight has already warned about it.
        fs_variants = [v for v in variants if v.fs_id == fs.id]
        if not fs_variants:
            tqdm.write(f"\n{'─' * 60}\n  feature set: {fs.id} — no algorithm, skipped")
            continue

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

        # The eval-only corpora, read with the same columns and stacked the same
        # way. They join no matrix axis and take no part in any split below.
        holdouts = {
            h.name: load_holdout(
                h, columns, group_col, human_label, stack=list(fs.columns)
            )
            for h in hold_specs
        }
        for hold in holdouts.values():
            tqdm.write(
                f"  {('· ' + hold.name):<20} {len(hold.df):>6} rows  "
                f"{hold.class_counts}   (eval only)"
            )

        for variant in tqdm(
            fs_variants, desc="Variants", unit="var", leave=False
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
                variant, cfg, X_all, data, names, labels, out_dir,
                holdouts=holdouts,
            )
            if payload.get(HOLDOUT_KEY):
                holdout_payloads.append(payload)
            print_matrix(
                payload["alg_label"], payload["summary"], payload["dataset_names"],
                row_labels=payload.get("source_labels"),
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

        del data, X_all, holdouts

    # Optional zero-shot score baselines: one row per score column of every
    # enabled family, at the fixed cutoff that column names. Binoculars and the LLM judge both read low ⇒
    # machine; the variant carries its own direction, so both are run by the same
    # loop. Run after the variants so they show up as extra rows in the
    # comparison.
    for bl in baseline_specs(cfg):
        tqdm.write(f"\n→ {bl['label']}  ({describe_variant(bl)})")
        bl_dir = alg_dir / bl["name"]
        bl_dir.mkdir(parents=True, exist_ok=True)
        # One scalar column, so the holdout is loaded per baseline: a baseline
        # stands on the rows ITS column is filled for, on a holdout as elsewhere.
        bl_holdouts = {
            h.name: load_holdout(
                h, [bl["column"]], group_col, human_label,
                score_col=bl["column"],
            )
            for h in hold_specs
        }
        payload = run_score_baseline(
            cfg, bl_dir, bl,
            holdout_eval=(
                baseline_holdout_evaluator(bl_holdouts) if bl_holdouts else None
            ),
        )
        # cd_core files the raw per-seed cells and stops there; the transpose
        # into {holdout: {source: aggregate}} is this side's job, so nothing in
        # cd_core has to know the shape a holdout figure reads. The file it just
        # wrote is re-dumped with the transposed key so a baseline's
        # cross_domain.json carries the same `holdout` block a trained variant's
        # does -- one shape for a reader to learn, not two.
        seed_cells = payload.pop("holdout_per_seed", None)
        if seed_cells:
            payload[HOLDOUT_KEY] = holdout_from_per_seed(seed_cells)
            holdout_payloads.append(payload)
            with open(bl_dir / "cross_domain.json", "w") as f:
                json.dump(payload, f, indent=2)
        print_matrix(
            payload["alg_label"], payload["summary"], payload["dataset_names"],
            row_labels=payload.get("source_labels"),
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

    write_holdout_comparisons(cfg, alg_dir, holdout_payloads, names, labels)
    return comparison


def write_holdout_comparisons(
    cfg: dict,
    alg_dir: Path,
    payloads: list[dict],
    names: list[str],
    labels: dict[str, str],
) -> None:
    """One ``holdout_<name>.json`` and its figures per eval-only corpus.

    Rows are the detectors this matrix produced; columns are the matrix's own
    ROWS -- the corpora (or, under ``lodo``, the pools) whose calibration slice
    supplied the threshold. Every cell is the same holdout documents scored
    again under a different calibration.

    None of it is a transfer cell and none of it enters
    ``variant_comparison.json`` or any figure that reads it: no model was
    trained on the holdout, and no model was tested here on a corpus it did not
    train on, because the holdout is not a corpus any model trained on.
    """
    hold_specs = eval_only_specs(cfg, BASE_DIR)
    if not hold_specs or not payloads:
        return

    group_col: str = cfg["data"]["group_col"]
    human_label: str = cfg["data"]["human_label"]
    row_labels = source_labels(training_plan(cfg, names), labels)

    for spec in hold_specs:
        rows = [r for r in (holdout_row(p, spec.name) for p in payloads) if r]
        if not rows:
            print(f"\nNo holdout results for {spec.name} — nothing written.")
            continue
        # Re-read the corpus once, for the row counts and class balance the
        # payload describes itself with: cheap (one column), and it means the
        # JSON says what it stood on rather than trusting a cell to carry it.
        hold = load_holdout(spec, [CLF_TARGET_COL], group_col, human_label)
        payload = holdout_comparison(
            cfg, hold, rows, row_labels, names,
            alg_label=f"{cfg['alg_label']} — eval-only",
        )
        with open(alg_dir / f"holdout_{spec.name}.json", "w") as f:
            json.dump(payload, f, indent=2)
        print_holdout_table(payload, rows, names)
        if PLOTS_ENABLED:
            plot_holdout_comparison(alg_dir, payload)


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
