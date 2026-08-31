r"""Evaluation-only corpora: what a trained detector catches where it cannot be scored.

The three training designs in ``cd_core`` all end at a source x target matrix (or
its 1x1 in-domain degenerate) over corpora that hold BOTH classes. That is the
only shape in which accuracy, AUROC, macro F1 and a target-calibrated TPR@FPR
mean anything, and ``check_dataset`` rejects a single-class corpus for exactly
that reason.

This module is the other case, and it is a real one. The Denario papers
(``data/parquet/denario_papers.parquet``) are LLM-written research papers, end to
end: 82 rows, every one machine, no human counterpart anywhere in the file. There
is no split to take, no ROC to read and no threshold to fit. What there IS:

    "a detector calibrated at 1% FPR on essays — how many of these does it catch?"

which is well-posed, needs no human row, and is the number a deployed detector
actually delivers. The threshold comes from the training corpus, exactly as
``cd_core`` already transports one; the holdout only supplies the rows it lands
on.

WHY THIS IS A SEPARATE MODULE
-----------------------------
Because nothing here may touch a number the main run reports. A holdout corpus
takes no share of a train/test split, contributes no row to a fit, no cell to a
matrix, no entry to a ranking and no bar to an existing figure. It is evaluated
AFTER a model is fitted, with thresholds that were already chosen, and it is
written to its own JSON and its own figures. Keeping it out of ``cd_core`` is
what makes that reviewable: if this file did not exist, the main results would be
byte-identical.

WHAT IS DEFINED ON AN ALL-MACHINE CORPUS, AND WHAT IS NOT
---------------------------------------------------------
Every metric is computed by ``cd_core`` itself -- ``metrics_dict`` and
``score_metrics``, the same functions and the same definitions the matrix runs
use -- and then the ones that cannot mean anything here are STRIPPED. Stripped,
never NaN-filled: a payload that carries a key is claiming a number, and
``score_metrics`` already sets that convention by omitting its transported keys
rather than writing NaN for them.

Present when the holdout holds only machine rows:

  accuracy              the share flagged machine at the source's Youden's J
                        threshold. On a single-class set this is identically
                        RECALL, and a detector that answers "machine" to
                        everything scores 1.000. That is not a defect of the
                        measurement, it is the ceiling of it, and every figure
                        this module draws marks that ceiling. Read it as a
                        detection rate, never as a quality score.
  tpr_at_srcfpr_<lvl>   the headline. The threshold was fitted to <lvl> FPR on
                        the SOURCE's calibration slice (``carve_calibration``:
                        never the fit rows, never the eval rows) and carried
                        here unchanged. The human-side cost is pinned on a
                        corpus that has humans; what is measured here is only
                        what that costs in detections.
  declared_accuracy     baselines only: the same share at the config's declared
                        cutoff (the paper's constant), which saw no data at all.
                        The off-the-shelf reading.
  n, n_machine, n_human, confusion, class_balance

Absent, because the corpus has no human row to define them on:

  auroc                 a ranking needs both classes.
  macro_f1              the human class's F1 is 0/0.
  tpr_at_fpr_<lvl>      variant (a) reads a threshold off THIS cell's own ROC.
  fpr_at_srcfpr_<lvl>   the realized FPR -- the price of the transported
                        threshold. Its absence is the honest limit of the whole
                        exercise and is stated on every figure: a TPR quoted
                        without the FPR it cost cannot be ranked against
                        another detector's.

A holdout that DOES carry both classes (point ``where`` at a corpus with human
rows, or build a mixed one) gets all of them back automatically -- the stripping
is keyed on which classes are present, not on the config.

THE CAVEAT THAT NO METRIC CAN CARRY
------------------------------------
With no human control from the holdout's own distribution, a low detection rate
is confounded between "missed this generator" and "reads this genre as human".
Denario is LLM-written astrophysics, biology and neuroscience; the training
corpora are essays, Reuters copy and writing prompts. Nothing in this module can
separate those two readings, and every figure it draws says so.

THE ERROR BAR IS NOT THE SEED SPREAD
-------------------------------------
The seeds resample the SOURCE -- its split, its calibration slice, hence its
threshold. They do not resample the holdout: the same 82 rows are scored every
time. So the spread over seeds is "how much does the threshold move", which is
worth having and is reported, but it is NOT the sampling error of the detection
rate. That is binomial on n, and :func:`wilson_ci` is what the figures draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.cd_core import (
    BINARY_CLASSES,
    MetricSpec,
    baseline_specs,
    cell_metric_specs,
    evaluate_score_rule,
    label_predictions,
    level_tag,
    load_prepared,
    mean_std_over_seeds,
    metrics_dict,
    oracle_tpr_key,
    transported_fpr_key,
    transported_tpr_key,
    TPR_AT_FPR_LEVELS,
)
from epai.utils.corpus_utils import CorpusSpec, eval_only_specs, parquet_schema
from epai.utils.preflight_utils import RULE, Report
from epai.utils.train_utils import CLF_TARGET_COL, stack_features
from epai.utils.variant_utils import (
    build_feature_sets,
    predict_alg,
    score_alg,
    variant_columns,
)

#: Key under which a holdout evaluation is stored inside a variant's payload,
#: and the stem of every file and figure this module writes. Distinct from
#: anything ``cd_core`` writes, so a reader never has to work out whether a
#: number came from the experiment or from the extra evaluation beside it.
HOLDOUT_KEY = "holdout"


# ---------------------------------------------------------------------------
# Which metrics survive a single-class evaluation set
# ---------------------------------------------------------------------------


def undefined_keys(y: np.ndarray) -> set[str]:
    """The metric keys that cannot be computed on an evaluation set with these labels.

    Empty when both classes are present -- a mixed holdout is scored exactly
    like a matrix cell. With no human row, everything that reads the human side
    of the distribution goes: the ranking metrics, the balance metric, and both
    readings that need an FPR.
    """
    classes = set(np.unique(np.asarray(y)).tolist())
    if {0, 1} <= classes:
        return set()

    keys = {"auroc", "macro_f1"}
    for level in TPR_AT_FPR_LEVELS:
        tag = level_tag(level)
        # (a) reads a threshold off this cell's own ROC curve; without humans
        # there is no curve. The threshold it would have reported goes with it.
        keys |= {oracle_tpr_key(level), f"threshold_at_fpr_{tag}"}
        # The realized FPR of the transported threshold -- the price of (b).
        # The TPR it bought survives; the price does not.
        keys.add(transported_fpr_key(level))
    if 1 not in classes:
        # A human-only holdout is the mirror image: the FPR is all there is.
        for level in TPR_AT_FPR_LEVELS:
            keys.add(transported_tpr_key(level))
            keys.discard(transported_fpr_key(level))
        keys.add("accuracy")
    return keys


def strip_undefined(res: dict, y: np.ndarray) -> dict:
    """A metrics dict with the keys this evaluation set cannot support removed.

    The metrics themselves are computed by ``cd_core`` unchanged, so a holdout
    number and a matrix number of the same name are the same quantity, fitted
    the same way. This only decides which of them are allowed to be written --
    and it removes them rather than setting them to NaN, because a NaN that
    reaches a mean is a number nobody asked for and a key that is absent cannot.
    """
    drop = undefined_keys(y)
    return {k: v for k, v in res.items() if k not in drop}


def holdout_metric_specs(row: dict | None = None) -> list[MetricSpec]:
    """The reportable metrics of a holdout, in order, as ``cd_core`` defines them.

    The specs are the matrix run's own -- same label, same reference level, same
    caption -- so a Denario bar and a transfer cell of the same metric are read
    on the same scale. ``row``, when given, narrows the list to what that
    comparison row actually carries.
    """
    order = [transported_tpr_key(l) for l in TPR_AT_FPR_LEVELS] + ["accuracy"]
    by_key = {m.key: m for m in cell_metric_specs()}
    specs = [by_key[k] for k in order if k in by_key]
    if row is None:
        return specs
    return [s for s in specs if f"{s.key}_mean" in row]


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials, at ~95%.

    THE error bar for a holdout detection rate, and the reason the seed spread
    is not it: the seeds vary the source's threshold, not these rows, so their
    agreement says nothing about how far a rate measured on 82 documents can be
    from the rate on the next 82. Wilson rather than the normal approximation
    because these rates live near 1.0, where ``p ± z·sqrt(p(1-p)/n)`` gives an
    interval that runs past the ceiling and collapses to width zero at exactly
    the value it is least trustworthy for.

    Returns ``(nan, nan)`` for ``n <= 0``.
    """
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def rate_ci(rate: float, n: int) -> tuple[float, float]:
    """:func:`wilson_ci` for a rate that was reported as a proportion of ``n``."""
    if not np.isfinite(rate) or n <= 0:
        return float("nan"), float("nan")
    return wilson_ci(int(round(rate * n)), n)


# ---------------------------------------------------------------------------
# The corpora
# ---------------------------------------------------------------------------


@dataclass
class Holdout:
    """One eval-only corpus, loaded for one feature set or one score column."""

    spec: CorpusSpec
    df: pd.DataFrame
    X: np.ndarray | None = None  # design matrix; None for a score baseline
    scores: np.ndarray | None = None  # the raw score column, for a baseline
    n_missing_dropped: int = 0

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def label(self) -> str:
        return self.spec.label

    @property
    def y(self) -> np.ndarray:
        return self.df["y"].to_numpy()

    @property
    def groups(self) -> np.ndarray:
        return self.df["group"].to_numpy()

    @property
    def class_counts(self) -> dict[str, int]:
        counts = self.df["y"].value_counts().to_dict()
        return {BINARY_CLASSES[c]: int(n) for c, n in sorted(counts.items())}

    @property
    def single_class(self) -> str | None:
        """``'machine'`` / ``'human'`` when only one class is present, else None."""
        present = sorted(set(self.df["y"].tolist())) if len(self.df) else []
        return BINARY_CLASSES[present[0]] if len(present) == 1 else None


def load_holdout(
    spec: CorpusSpec,
    columns: list[str],
    group_col: str,
    human_label: str,
    stack: list[str] | None = None,
    score_col: str | None = None,
) -> Holdout:
    """Read one eval-only corpus for one feature set (or one score column).

    Goes through ``cd_core.load_prepared``, so the holdout gets the same
    group-aware missing-value drop, the same binarised target and the same
    ``drop_classes`` handling as any corpus in the run. Nothing about the
    reading is special; only what is done with the rows afterwards is.

    Exactly one of the two is normally given:

    ``stack``      the columns to build a design matrix from -- the feature
                   set's own, exactly as the runners stack them, for a trained
                   variant to score.
    ``score_col``  the single scalar column a zero-shot baseline reads. A
                   baseline stands on the rows ITS column is filled for, on a
                   holdout as on any other corpus, which is why it is loaded
                   per baseline rather than per feature set.
    """
    prep = load_prepared(spec, columns, group_col, human_label)
    X = stack_features(prep.df, list(stack))[0] if stack and len(prep.df) else None
    scores = (
        prep.df[score_col].to_numpy(dtype=float)
        if score_col and score_col in prep.df.columns
        else None
    )
    return Holdout(
        spec=spec,
        df=prep.df,
        X=X,
        scores=scores,
        n_missing_dropped=prep.n_missing_dropped,
    )


# ---------------------------------------------------------------------------
# Evaluating one fitted model, or one score column, on a holdout
# ---------------------------------------------------------------------------


def _exclude_seen(hold: Holdout, train_groups: set) -> np.ndarray:
    """Rows of the holdout whose group was NOT in the source's training pool.

    A holdout is a different file and its group ids do not normally collide with
    the training corpus's, so this drops nothing in practice -- which is exactly
    why it runs. The rule that a target row whose group was trained on is a free
    hit does not stop applying because the corpus is new, and the cheapest way
    to be sure it held is to apply it and report that it removed zero rows.
    """
    if not train_groups:
        return np.ones(len(hold.df), dtype=bool)
    return ~hold.df["group"].isin(train_groups).to_numpy()


def evaluate_variant_on_holdout(
    alg,
    hold: Holdout,
    cal_thresholds: dict[str, float] | None,
    label_threshold: float | None,
    train_groups: set | None = None,
    aux: np.ndarray | None = None,
) -> dict:
    """One fitted model on one holdout corpus, under the SOURCE's thresholds.

    Nothing is fitted and nothing is calibrated here. ``cal_thresholds`` and
    ``label_threshold`` were fitted on the source's calibration slice by the
    caller -- the same two objects that produced that source's own cells -- and
    are applied unchanged, which is the whole claim: this is what the detector
    the run just trained would do to these documents on the day it was deployed.
    """
    keep = _exclude_seen(hold, train_groups or set())
    if hold.X is None or not keep.any():
        return {"n": 0, "excluded": int((~keep).sum())}

    X = hold.X[keep]
    Z = aux[keep] if aux is not None else None
    y = hold.df.loc[keep, "y"].to_numpy()

    scores = score_alg(alg, X, Z)
    res = metrics_dict(
        y,
        label_predictions(predict_alg(alg, X, Z), scores, label_threshold),
        scores=scores,
        cal_thresholds=cal_thresholds,
        label_threshold=label_threshold,
    )
    res = strip_undefined(res, y)
    res["excluded"] = int((~keep).sum())
    return res


def evaluate_baseline_on_holdout(
    hold: Holdout,
    threshold: float,
    machine_is_high: bool,
    cal_thresholds: dict[str, float] | None,
    label_threshold: float | None,
    train_groups: set | None = None,
) -> dict:
    """One zero-shot score column on one holdout corpus. Same rule, same source.

    ``declared_accuracy`` is the extra number a baseline can report and a trained
    variant cannot: the share flagged machine by the config's DECLARED cutoff --
    the paper's constant, which saw no data at all. It is the only genuinely
    zero-shot reading in the whole run, and on an all-machine corpus it is the
    one an off-the-shelf deployment would actually get.
    """
    keep = _exclude_seen(hold, train_groups or set())
    if hold.scores is None or not keep.any():
        return {"n": 0, "excluded": int((~keep).sum())}

    scores = hold.scores[keep]
    y = hold.df.loc[keep, "y"].to_numpy()

    res = evaluate_score_rule(
        scores, y, threshold, machine_is_high,
        cal_thresholds=cal_thresholds, label_threshold=label_threshold,
    )
    res = strip_undefined(res, y)

    # The declared cutoff on its own, beside the fitted one. `evaluate_score_rule`
    # only uses it as a fallback once `label_threshold` is present, so it would
    # otherwise leave no trace in the output at all.
    declared = (
        (scores >= threshold) if machine_is_high else (scores <= threshold)
    ).astype(int)
    res["declared_accuracy"] = float((declared == y).mean())
    res["declared_threshold"] = float(threshold)
    res["excluded"] = int((~keep).sum())
    return res


def baseline_holdout_evaluator(holdouts: dict[str, Holdout]):
    """A callback for ``cd_core.run_score_baseline``'s ``holdout_eval`` hook.

    That function owns the split, the calibration carve and the thresholds; it
    does not own -- and must not learn about -- the eval-only corpora. So the
    hook hands over exactly what one (seed, source) produced and this closure
    does the rest, which keeps every holdout-shaped line out of ``cd_core``
    except the six that call it.

    Returns ``{holdout_name: metrics}`` for that seed and source.
    """

    def evaluate(
        source: str,
        threshold: float,
        machine_is_high: bool,
        cal_thresholds: dict[str, float] | None,
        label_threshold: float | None,
        train_groups: set | None = None,
    ) -> dict:
        return {
            name: evaluate_baseline_on_holdout(
                hold, threshold, machine_is_high,
                cal_thresholds, label_threshold, train_groups=train_groups,
            )
            for name, hold in holdouts.items()
        }

    return evaluate


def holdout_from_per_seed(seed_holdouts: dict[str, list[dict]]) -> dict:
    """``{source: [per-seed cells]}`` -> ``{holdout: {source: aggregate}}``.

    The transpose the payload wants: a run produces holdout cells seed-major
    and source-major because that is the order the loops run in, and every
    reader wants them holdout-major.
    """
    names = {n for cells in seed_holdouts.values() for c in cells for n in c}
    return {
        name: {
            source: aggregate_holdout([c[name] for c in cells if name in c])
            for source, cells in seed_holdouts.items()
            if any(name in c for c in cells)
        }
        for name in sorted(names)
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

#: Every rate a holdout cell can report, in reporting order. Each gets a mean and
#: a spread over seeds, and -- because the seeds do not resample these rows -- a
#: Wilson interval on ``n``.
HOLDOUT_RATE_KEYS: tuple[str, ...] = (
    tuple(transported_tpr_key(level) for level in TPR_AT_FPR_LEVELS)
    + ("accuracy", "declared_accuracy")
)


def aggregate_holdout(per_seed: list[dict]) -> dict:
    """Mean, seed spread and Wilson interval for one (variant, source, holdout).

    ``per_seed`` is one cell per seed. A key absent from every seed stays absent
    -- ``mean_std_over_seeds`` returns ``{}`` for it -- so an all-machine holdout
    simply has no AUROC rather than an AUROC of NaN.

    Two uncertainties, and they answer different questions:

      ``<key>_std``       how far the number moved as the SOURCE's threshold
                          was refitted on a different seed. Small means the
                          operating point is stable; it says nothing about
                          these rows.
      ``<key>_ci_lo/hi``  the Wilson interval of the rate on ``n`` documents.
                          This is the sampling error, and on a corpus of 82 it
                          is the wider of the two by a long way.
    """
    if not per_seed:
        return {}
    n = int(per_seed[0].get("n", 0) or 0)
    out: dict = {
        "n": n,
        "excluded": int(per_seed[0].get("excluded", 0) or 0),
        "class_balance": per_seed[0].get("class_balance", {}),
        "confusion_mean": (
            np.mean([r["confusion"] for r in per_seed], axis=0).tolist()
            if all("confusion" in r for r in per_seed)
            else None
        ),
    }
    for key in HOLDOUT_RATE_KEYS:
        stats = mean_std_over_seeds(per_seed, key)
        out.update(stats)
        mean = stats.get(f"{key}_mean")
        if mean is not None:
            lo, hi = rate_ci(float(mean), n)
            out[f"{key}_ci_lo"], out[f"{key}_ci_hi"] = lo, hi
    if out["confusion_mean"] is None:
        del out["confusion_mean"]
    return out


def _finish_holdout_row(row: dict, per_source: dict[str, dict]) -> dict:
    """Fill one holdout comparison row's aggregates from its per-source cells.

    The mean is unweighted over the threshold SOURCES, which is the only
    weighting that means anything here: every source scored the same documents,
    so a weighted mean would be weighting the same rows by how big some other
    corpus was. ``_worst`` is the number to read beside it -- a detector that
    holds up under one calibration corpus and collapses under another has a
    respectable mean and no deployable operating point.
    """
    row["per_source"] = per_source
    for key in HOLDOUT_RATE_KEYS:
        values = [
            s[f"{key}_mean"] for s in per_source.values() if f"{key}_mean" in s
        ]
        if values:
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_across_sources_std"] = float(np.std(values))
            row[f"{key}_worst"] = float(np.min(values))
    # One n for the corpus: every source scores the same rows (the group
    # exclusion removes nothing in practice), so a differing n is worth seeing
    # rather than averaging away.
    ns = sorted({int(s.get("n", 0) or 0) for s in per_source.values()})
    row["n"] = ns[0] if len(ns) == 1 else ns
    return row


def _row_identity(payload: dict) -> dict:
    """The columns that identify a detector, shared by every comparison row."""
    row = {
        "variant": payload["variant"],
        "label": payload["label"],
        "feature_dim": payload.get("feature_dim"),
        "baseline": bool(payload.get("baseline")),
        "baseline_kind": payload.get("baseline_kind"),
    }
    if payload.get("column"):
        row["column"] = payload["column"]
    if payload.get("hybrid"):
        row["hybrid"] = True
        row["aux_column"] = payload.get("aux_column")
    return row


def holdout_row(payload: dict, name: str) -> dict | None:
    """One variant's comparison row for one holdout corpus, or None if absent.

    Shaped like the rows ``suite_utils.corpus_comparison_rows`` builds -- a
    per-source mapping plus the unweighted mean over sources -- so the figure
    code can read a holdout row and an in-domain corpus row through the same
    accessors.

    ``payload`` is one variant's saved result, whose ``holdout`` key maps a
    holdout name to ``{source: aggregate}``. A matrix run fills several sources
    from one payload; an in-domain run fills exactly one (its corpus) and the
    sources are stitched together across payloads by :func:`merge_holdout_rows`.
    """
    per_source = (payload.get(HOLDOUT_KEY) or {}).get(name)
    if not per_source:
        return None
    return _finish_holdout_row(_row_identity(payload), per_source)


def merge_holdout_rows(
    per_source_payloads: dict[str, list[dict]], name: str, order: list[str]
) -> list[dict]:
    """Stitch one holdout's rows together across separately-run sources.

    The in-domain runner trains each corpus in its own pass, so a variant's
    Denario result arrives one source at a time: ``essay`` knows only the
    threshold it fitted itself. This joins those passes back into the one table
    the figure draws -- rows keyed on the variant id, columns on the corpus that
    supplied the threshold.

    ``per_source_payloads`` maps a source name to the comparison rows that pass
    produced. A variant missing from one source simply has no column there,
    which is what a run whose feature set could not be built on that corpus
    genuinely measured.
    """
    merged: dict[str, dict] = {}
    per_source: dict[str, dict[str, dict]] = {}
    for source in order:
        for payload in per_source_payloads.get(source, []):
            cells = (payload.get(HOLDOUT_KEY) or {}).get(name)
            if not cells:
                continue
            key = payload["variant"]
            merged.setdefault(key, _row_identity(payload))
            # The payload's own source key is authoritative -- an in-domain pass
            # writes its corpus name, a matrix pass writes each matrix row -- so
            # a merge never invents a column the run did not produce.
            per_source.setdefault(key, {}).update(cells)
    return [
        _finish_holdout_row(merged[key], per_source[key])
        for key in merged
        if per_source.get(key)
    ]


def holdout_comparison(
    cfg: dict,
    hold: Holdout,
    rows: list[dict],
    source_labels: dict[str, str],
    source_order: list[str],
    alg_label: str,
) -> dict:
    """The payload written as ``holdout_<name>.json`` and drawn by the figures."""
    return {
        "alg_id": cfg["alg_id"],
        "alg_label": alg_label,
        "holdout": hold.name,
        "holdout_label": hold.label,
        "holdout_source": hold.spec.source,
        "single_class": hold.single_class,
        "class_balance": hold.class_counts,
        "n": int(len(hold.df)),
        "n_unscored": int(hold.n_missing_dropped),
        "source_names": list(source_order),
        "source_labels": dict(source_labels),
        "subtitle": (
            f"{hold.spec.source} · {int(len(hold.df))} rows "
            f"({', '.join(f'{k} {v}' for k, v in hold.class_counts.items())}) · "
            f"thresholds transported from {len(source_order)} training corpus/corpora · "
            f"{cfg['model']['n_runs']} seeds"
        ),
        "classes": BINARY_CLASSES,
        "config": cfg,
        "variants": rows,
    }


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def holdout_columns(cfg: dict) -> list[str]:
    """Every column an enabled holdout has to carry, features and scores alike.

    A holdout is evaluated by EVERY variant and EVERY baseline the run produces,
    so it needs every column any of them reads. The alternative -- quietly
    skipping the variants whose columns are missing -- would put a Denario
    figure on the page covering three detectors out of twelve, with nothing on
    it saying which nine are absent.
    """
    cols: list[str] = []
    for fs in build_feature_sets(cfg):
        cols += variant_columns(fs, cfg)
    cols += [s["column"] for s in baseline_specs(cfg)]
    return list(dict.fromkeys(cols))


def check_holdout_corpora(cfg: dict, report: Report, base_dir: Path | None = None) -> None:
    """Every eval-only corpus opens, carries every needed column, and has rows.

    Deliberately stricter than ``check_dataset`` about columns and looser about
    classes, and both for the same reason: a holdout is an extra evaluation of
    the WHOLE run, so it has to cover the whole run -- but it is never trained
    on, so a single class is the expected case rather than a fatal one.

    Two passes, in that order, because they fail differently. The columns are
    checked against the file's SCHEMA -- no row is read, and a missing column is
    named with the script that writes it. The rows are then counted the way the
    run will actually load them: once per feature set and once per baseline
    column, never all of them at once. That distinction is not cosmetic. The
    missing-value drop is group-aware and runs over whatever columns it is given,
    so loading eleven feature columns and eight score columns together would drop
    a row that is missing any ONE of them and report a corpus far smaller than
    any variant will really see -- an error where there is no problem.
    """
    specs = (
        eval_only_specs(cfg, base_dir) if base_dir is not None else eval_only_specs(cfg)
    )
    if not specs:
        return

    group_col: str = cfg["data"]["group_col"]
    human_label: str = cfg["data"]["human_label"]

    print()
    print("eval-only corpora — evaluated by every variant, never trained on")

    for spec in specs:
        print(f"\n  {spec.name}  ({spec.source})")
        if not spec.path.is_file():
            report.error(f"{spec.name}: eval-only parquet not found: {spec.path}")
            print("    MISSING")
            continue
        try:
            columns, n_file = parquet_schema(spec.path)
        except Exception as exc:
            report.error(
                f"{spec.name}: cannot read {spec.path}: {type(exc).__name__}: {exc}"
            )
            print("    UNREADABLE")
            continue

        needed = [group_col, CLF_TARGET_COL] + holdout_columns(cfg)
        missing = [c for c in dict.fromkeys(needed) if c not in columns]
        if missing:
            # Named with what writes them, because the fix is a scoring run and
            # not a config edit. A holdout that covers only some of the run's
            # detectors is worse than one that stops it.
            report.error(
                f"{spec.name}: the eval-only corpus is missing {missing}. A "
                f"holdout is evaluated by EVERY variant and baseline in this "
                f"config, so it has to carry every column they read — score it "
                f"with the same scripts the training corpora went through "
                f"(epai/ai_detection/prepare/, embed/, score/), or set "
                f"data.eval_only.enabled: false."
            )
            print(f"    {n_file} rows in file — MISSING {len(missing)} COLUMN(S)")
            continue

        # Rows, per load the run will actually do. Same shape as the per-corpus
        # table above it so the two can be compared line for line.
        print(f"    {'read as':<26}{'rows':>8}{'dropped':>9}{'dim':>7}  classes")
        print("    " + "-" * (RULE - 4))
        loads: list[tuple[str, list[str], list[str] | None]] = [
            (fs.id, variant_columns(fs, cfg), list(fs.columns))
            for fs in build_feature_sets(cfg)
        ] + [
            (bl["label"], [bl["column"]], None) for bl in baseline_specs(cfg)
        ]

        seen_classes: set[str] = set()
        for tag, cols, stack in loads:
            try:
                prep = load_prepared(spec, cols, group_col, human_label)
            except Exception as exc:
                report.error(
                    f"{spec.name} | {tag}: failed to load: {type(exc).__name__}: {exc}"
                )
                continue
            dim = prep.dim_of(stack) if (stack and prep.dims_per_col) else 1
            print(
                f"    {tag:<26}{len(prep.df):>8}{prep.n_missing_dropped:>9}{dim:>7}  "
                f"{prep.class_counts if len(prep.df) else '-'}"
            )
            seen_classes |= set(prep.class_counts)
            _check_one_holdout_load(spec.name, tag, prep, report)

        # Once per corpus, after every load: it is a property of the file, not
        # of a feature set, and it decides which metrics the output can carry.
        # NOT an error, unlike check_dataset -- one class is the case this whole
        # module exists for.
        if len(seen_classes) == 1:
            only = next(iter(seen_classes))
            print(
                f"    note: {only}-only corpus. Reported as a detection rate "
                f"under thresholds\n          transported from the training "
                f"corpora. AUROC, macro F1, TPR at this\n          corpus's own "
                f"FPR and the realized FPR are undefined here and are\n"
                f"          omitted, not zeroed."
            )


def _check_one_holdout_load(name: str, tag: str, prep, report: Report) -> None:
    """Value-level checks for one (holdout, feature set or score column) pair."""
    if len(prep.df) == 0:
        report.error(
            f"{name} | {tag}: no usable rows "
            f"({prep.n_corpus} in the corpus, {prep.n_missing_dropped} dropped "
            f"for missing values: {prep.missing_per_col})"
        )
        return
    if prep.ragged_cols:
        for col in prep.ragged_cols:
            report.error(
                f"{name} | {tag}: {col} has inconsistent widths "
                f"{sorted(prep.dims_per_col[col])} — it cannot be stacked"
            )
        return

    if prep.n_missing_dropped:
        report.warn(
            f"{name} | {tag}: {prep.n_missing_dropped}/{prep.n_corpus} row(s) "
            f"dropped for missing values {prep.missing_per_col} — this detector "
            f"stands on {len(prep.df)} rows here"
        )
    if len(prep.df) < 30:
        lo, hi = wilson_ci(int(round(0.8 * len(prep.df))), len(prep.df))
        report.warn(
            f"{name} | {tag}: only {len(prep.df)} rows; a detection rate near 80% "
            f"carries a 95% interval of [{lo:.0%}, {hi:.0%}] here. Read the "
            f"interval, not the point."
        )


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_holdout_table(
    payload: dict, rows: list[dict], source_order: list[str]
) -> None:
    """One holdout corpus: every detector, every threshold source, one table."""
    if not rows:
        return
    labels = payload.get("source_labels") or {}
    specs = holdout_metric_specs()
    present = [s for s in specs if any(f"{s.key}_mean" in r for r in rows)]
    if not present:
        return

    single = payload.get("single_class")
    width = max(26, max((len(r["label"]) for r in rows), default=0) + 4)
    col = max(10, max((len(labels.get(n, n)) for n in source_order), default=0) + 2)

    for spec in present:
        order = sorted(
            rows,
            key=lambda r: (-float(r.get(f"{spec.key}_mean", -1.0)), str(r["label"])),
        )
        header = (
            f"{'detector':<{width}}"
            + "".join(f"{labels.get(n, n):>{col}}" for n in source_order)
            + f"{'mean':>{col}}{'worst':>9}{'95% CI':>16}"
        )
        print()
        print("=" * max(RULE, len(header)))
        print(
            f"  {payload.get('alg_label', '')} — {payload['holdout_label'].upper()}"
            f" — {spec.label.upper()}"
        )
        print("=" * max(RULE, len(header)))
        print(header)
        print("-" * max(RULE, len(header)))
        for r in order:
            mark = "★ " if r.get("baseline") else "  "
            line = f"{mark + r['label']:<{width}}"
            for n in source_order:
                value = r.get("per_source", {}).get(n, {}).get(f"{spec.key}_mean")
                line += (
                    f"{'—':>{col}}"
                    if value is None or not np.isfinite(value)
                    else f"{float(value)*100:>{col-1}.1f}%"
                )
            mean = float(r.get(f"{spec.key}_mean", float("nan")))
            worst = float(r.get(f"{spec.key}_worst", float("nan")))
            # `n` is a LIST when the sources did not all score the same rows --
            # a partly-filled score column. There is no single binomial n then,
            # and inventing one would be the only dishonest cell in the table.
            n = r.get("n")
            lo, hi = rate_ci(mean, n) if isinstance(n, int) else (
                float("nan"), float("nan")
            )
            ci = (
                f"[{lo*100:.0f}, {hi*100:.0f}]"
                if np.isfinite(lo) and np.isfinite(hi)
                else "—"
            )
            line += f"{mean*100:>{col-1}.1f}%{worst*100:>8.1f}%{ci:>16}"
            print(line)
        print("-" * max(RULE, len(header)))

    n = payload.get("n")
    print(f"  {payload['holdout_label']}: {n} rows, {payload.get('class_balance')}")
    print("  Every column is the SAME rows under a threshold transported from a")
    print("  different training corpus. Nothing here was trained on; nothing here")
    print("  changes a number in the tables above.")
    if single == "machine":
        print(
            "  All-machine corpus, so accuracy IS the detection rate and a detector\n"
            "  that answers 'machine' to everything scores 100%. AUROC, macro F1 and\n"
            "  the realized FPR are undefined here and are omitted, not zeroed —\n"
            "  which means these rates cannot be ranked against each other on their\n"
            "  own: a rate without the false-accusation cost it carries is half a\n"
            "  result. Read them beside the same detector's transfer cells."
        )
    print(
        "  The 95% CI is Wilson on n — the sampling error of the rate. The per-source\n"
        "  spread is something else: how much the number moved as the SOURCE's\n"
        "  threshold changed."
    )
    if any(r.get("baseline") for r in rows):
        print("  ★ zero-shot baseline: no classifier trained")
