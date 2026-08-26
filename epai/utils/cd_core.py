r"""The cross-domain transfer experiment, once: data, split, metrics, baselines.

Every cross-domain run asks one question -- has the classifier learned
"machine-generated-ness", or has it learned *this corpus, written by this
generator*? -- and answers it the same way: train on each corpus in turn,
evaluate on all of them, and read the source x target accuracy matrix.

    diagonal      -> in-domain, held-out grouped test split (the usual number)
    off-diagonal  -> transfer: never saw this corpus / this generator

What varies between runs is only *what* is put through that matrix: a feature
set, an embedding, an algorithm, a threshold on a detector score. That axis
belongs to the caller; everything below it is identical every time and lives
here, so the runner (``epai/ai_detection/train/run_clf_cross_domain.py``) and the
hyper-parameter search (``run_sweep_cross_domain.py``) cannot drift apart on the
split, the missing-value rule, the group exclusion or the metric definitions.

Every cell is reported under several metrics, not just accuracy: accuracy and
macro F1 from the predicted labels, AUROC and TPR at a fixed FPR from the
continuous score behind them. The last of those is reported *twice*, once with
the threshold fit on the cell being tested and once with the source's threshold
transported to it, and the difference between those two readings is where the
transfer failure this experiment hunts actually shows up. See the "Score
metrics" section below for what each one claims.

Three invariants this module exists to hold:

  * **Binarised target.** The corpora do not agree on generator names (``gpt``
    in the original Ghostbuster sets, ``<model>:<prompt_id>`` in the ones this
    repo generates), so only the human-vs-machine view is comparable across them.
  * **Grouped split.** A document and its rewrites share one ``group_col`` value
    and never straddle the train/test cut.
  * **Group exclusion under transfer.** ``essay`` and ``essay_gpt54mini`` are
    built from the same human sources and share their ids, so target rows whose
    group was seen while training are dropped from the evaluation (and counted).

A corpus is a :class:`~epai.utils.corpus_utils.CorpusSpec` -- a parquet plus an
optional ``domain`` -- so six datasets can be six slices of two files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from epai.utils.corpus_utils import CorpusSpec, dataset_specs, read_corpus
from epai.utils.preflight_utils import RULE, Report
from epai.utils.train_utils import CLF_TARGET_COL, is_missing

# Binarised target: human -> 0, any generator -> 1.
BINARY_CLASSES = ["human", "machine"]

# Chance level of a rate metric on the binarised task. The classes are balanced,
# so accuracy, macro F1 and AUROC are all read against 0.5. TPR at a fixed FPR is
# NOT -- its chance level is the FPR itself; see MetricSpec.reference.
CD_CHANCE_RATE = 0.5

# Embedding columns are stored under this prefix. It is only ever stripped, for
# display: a config names its columns literally, since a feature set may mix an
# embedding with a scalar score and prefixing one for you would corrupt the other.
EMB_PREFIX = "emb_"

# A feature whose name ends in BIN_SUFFIX is not read from the parquet: it is
# derived from the raw text column of the same stem by binarising human vs
# machine (see binarize_text_column). It lets a text label such as generated_by
# be dropped into the design matrix as a 0/1 feature -- a leakage sanity check.
BIN_SUFFIX = "__bin"

# Zero-shot score baselines: no model is fitted, only a fixed threshold on one
# scalar column. Each output lands in its own subdirectory, alongside the
# variants.
DEFAULT_BINOX_COL = "binoculars_score"

# The LLM judge writes P(human) per row -- 1.0 => human, 0.0 => machine -- so a
# LOW score means machine, the same direction as Binoculars.
DEFAULT_JUDGE_COL = "llm_judge_score_gpt56luna_v2"


# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------


def raw_feature_col(feature: str) -> str:
    """The parquet column a feature reads from (strips the derived BIN_SUFFIX)."""
    return feature[: -len(BIN_SUFFIX)] if feature.endswith(BIN_SUFFIX) else feature


def binarize_text_column(series: pd.Series, human_label: str) -> pd.Series:
    """``human_label`` -> 0.0, any other non-null value -> 1.0, missing -> NaN.

    The same human-vs-machine rule as the target ``y``, so binarising
    ``generated_by`` yields the label itself -- which is the point: it is a
    leakage sanity check / positive control, a feature the model should be able
    to learn to ~100% in-domain and under transfer. Missing values become NaN so
    the usual group-aware missing-row drop removes them.
    """
    missing = series.map(is_missing)
    result = (series != human_label).astype(float)
    result[missing] = np.nan
    return result


def binary_text_columns(cfg: dict) -> list[str]:
    """The raw text columns a config asks to binarise, from ``extra_binary_text``."""
    return [c for c in (cfg.get("extra_binary_text") or []) if isinstance(c, str)]


def binary_text_features(cfg: dict) -> list[str]:
    """Derived ``<col>__bin`` feature names for every ``extra_binary_text`` column."""
    return [f"{c}{BIN_SUFFIX}" for c in binary_text_columns(cfg)]


# ---------------------------------------------------------------------------
# Zero-shot score baselines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineKind:
    """One family of zero-shot score baselines, as a config block describes it.

    The families differ in exactly three things -- which config block turns them
    on, which column holds the score, and which end of that score means
    *machine*. Everything else (the grouped evaluation, the payload shape) is
    shared, so a third detector is one more entry here.

    A family contributes one variant PER COLUMN it is given, and every cutoff is
    taken from the config: nothing about a baseline is fitted. The headline
    metrics are AUROC and TPR at a fixed FPR, and neither reads the declared
    cutoff -- an ROC sweeps every threshold. Calibrating one on the source
    (Youden's J, as earlier versions did) therefore bought nothing but a second
    row that was bit-identical under every metric the run is read on, so the
    cutoff now only has to be a documented, fixed operating point for the
    label-derived numbers (accuracy, macro F1, the confusion matrix) to mean
    something.

    Several columns of one family is the normal case once a detector has been
    run with more than one configuration -- Binoculars scored with two model
    pairs, the judge with two judges. Those columns are the SAME detector read
    off different machinery, so they share the family's direction, style and
    colour, and differ only in which column they read and where they are
    operated. They are one family with several rows, not several families.
    """

    key: str  # names the output folder and the comparison row
    cfg_key: str  # the config block that configures it
    default_col: str
    # Shared stem of the family's columns, stripped to name the extra rows:
    # 'binoculars_score_qwen25_7b' -> the row 'binoculars__qwen25_7b'. Only ever
    # used for naming, never to find a column -- configs name those literally.
    column_prefix: str
    machine_is_high: bool  # True when a *high* score means machine-written
    label: str  # human-readable, for preflight messages

    @property
    def rule(self) -> str:
        """The decision rule in words, for logs and figure labels."""
        return "score >= threshold ⇒ machine" if self.machine_is_high else (
            "score <= threshold ⇒ machine"
        )


BASELINE_KINDS: tuple[BaselineKind, ...] = (
    BaselineKind(
        key="binoculars",
        cfg_key="binoculars_baseline",
        default_col=DEFAULT_BINOX_COL,
        column_prefix=DEFAULT_BINOX_COL,  # ..._qwen25_7b, ..._falcon3_7b, ...
        machine_is_high=False,  # Binoculars: a LOW score means machine
        label="Binoculars",
    ),
    BaselineKind(
        key="llm_judge",
        cfg_key="llm_judge_baseline",
        default_col=DEFAULT_JUDGE_COL,
        column_prefix="llm_judge_score",  # ..._<model>_<prompt>, one per judge
        machine_is_high=False,  # the judge reports P(human), so LOW means machine
        label="LLM judge",
    ),
)


def baseline_block(cfg: dict, kind: BaselineKind) -> dict | None:
    """One family's config block, when it is present and switched on.

    Every baseline is opt-in: a config asks for it with its ``cfg_key`` mapping
    (``enabled`` defaults to True when the key is present). Malformed values are
    caught by ``check_score_baselines``; here an off / absent / malformed
    baseline simply reads as None.
    """
    spec = cfg.get(kind.cfg_key)
    if not isinstance(spec, dict) or not spec.get("enabled", True):
        return None
    return spec


def baseline_slug(kind: BaselineKind, column: str) -> str:
    """Directory- and id-safe name for one column's baseline row.

    The family's default column keeps the bare family name, so a config that
    names one column writes exactly the paths and ids it always did; any other
    column is suffixed with what distinguishes it, which for the columns
    ``score_parquets.py`` writes is the scoring pair -- ``binoculars__qwen25_7b``.
    """
    if column == kind.default_col:
        return kind.key
    stem = column.removeprefix(kind.column_prefix).strip("_")
    stem = re.sub(r"[^0-9A-Za-z_.-]+", "_", stem).strip("_")
    return f"{kind.key}__{stem}" if stem else kind.key


def _as_threshold(value) -> float | None:
    """A cutoff read from the config, or None if it is not a plain number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _declared_columns(block: dict, kind: BaselineKind) -> list:
    """The block's column entries in declaration order, before validation.

    Two spellings, because one column is by far the common case and should not
    have to be written as a list: ``column:`` (a single name) or ``columns:``
    (a list of names, or of ``{column, threshold, label}`` mappings). Neither is
    normalised here -- whatever the config holds is passed through for
    ``baseline_entries`` to accept and ``check_baseline_block`` to complain
    about, so the two never disagree on what is malformed.
    """
    if "columns" in block:
        declared = block["columns"]
        return list(declared) if isinstance(declared, list) else [declared]
    return [block.get("column", kind.default_col)]


def baseline_entries(cfg: dict, kind: BaselineKind) -> list[dict]:
    """One normalised variant per column of a family, in declaration order.

    Each entry carries its own ``threshold``, falling back to a block-level one:
    a cutoff belongs to the *column*, since two Binoculars pairs put the score on
    different scales and a threshold refit for one says nothing about the other.
    The shared fallback exists for the single-column case and for a deliberate
    "same cutoff everywhere" comparison, not as an invitation to reuse Falcon's
    constants on a pair they were never calibrated for.

    Entries that are malformed, or that name no usable cutoff, are dropped here
    and reported by ``check_score_baselines``.
    """
    block = baseline_block(cfg, kind)
    if block is None:
        return []

    shared_threshold = _as_threshold(block.get("threshold"))
    entries: list[dict] = []
    for declared in _declared_columns(block, kind):
        if isinstance(declared, str):
            column, threshold, label = declared, shared_threshold, None
        elif isinstance(declared, dict):
            column = declared.get("column")
            threshold = _as_threshold(declared.get("threshold"))
            threshold = shared_threshold if threshold is None else threshold
            label = declared.get("label")
        else:
            continue

        if not isinstance(column, str) or not column.strip() or threshold is None:
            continue
        column = column.strip()
        name = baseline_slug(kind, column)
        if not isinstance(label, str) or not label.strip():
            # 'binoculars' for the family's own column, 'binoculars qwen25_7b'
            # for the rest -- the family name still leads, which is what the
            # figure styling falls back on when a payload predates baseline_kind.
            label = name.replace("__", " ")
        entries.append(
            {
                "kind": kind.key,
                "column": column,
                "machine_is_high": kind.machine_is_high,
                "threshold": threshold,
                "name": name,
                "label": label.strip(),
            }
        )

    # Two entries sharing a name would write into one another's output
    # directory. The duplicate is an error (check_baseline_block says so); here
    # the first one wins so the run still produces something readable.
    seen: set[str] = set()
    unique: list[dict] = []
    for entry in entries:
        if entry["name"] in seen:
            continue
        seen.add(entry["name"])
        unique.append(entry)
    return unique


def baseline_columns(cfg: dict) -> list[str]:
    """Every score column any enabled baseline reads, deduplicated."""
    cols = [e["column"] for k in BASELINE_KINDS for e in baseline_entries(cfg, k)]
    return list(dict.fromkeys(cols))


def format_threshold(value: float) -> str:
    """Compact rendering of a threshold, for labels and log lines."""
    return f"{float(value):g}"


def baseline_specs(cfg: dict) -> list[dict]:
    """Every baseline variant a config asks for, across all families, in order.

    A cutoff comes straight from the config and is applied unchanged to every
    source and target -- fit to nothing, the genuinely zero-shot reference the
    trained variants are measured against. A column with no usable ``threshold``
    contributes nothing (``check_score_baselines`` says why).
    """
    specs: list[dict] = []
    for kind in BASELINE_KINDS:
        specs.extend(baseline_entries(cfg, kind))
    return specs


def describe_variant(variant: dict) -> str:
    """One-line description of a baseline variant, for the run log.

    Spells out the direction as well as the cutoff: the two families read their
    threshold opposite ways, and a log line that only shows the number would be
    ambiguous between them.
    """
    rule = (
        "score>=thr ⇒ machine"
        if variant.get("machine_is_high")
        else "score<=thr ⇒ machine"
    )
    return (
        f"{variant['column']}, fixed threshold "
        f"{format_threshold(variant['threshold'])}, {rule}"
    )


def check_baseline_block(cfg: dict, kind: BaselineKind, report: Report) -> None:
    """Validate one baseline family's config block."""
    bx = cfg.get(kind.cfg_key)
    if bx is None:
        return
    if not isinstance(bx, dict):
        report.error(f"'{kind.cfg_key}' must be a mapping")
        return

    if "enabled" in bx and not isinstance(bx["enabled"], bool):
        report.error(f"'{kind.cfg_key}.enabled' must be a boolean")

    if "vanilla_threshold" in bx:
        # The fitted/vanilla split is gone: a baseline is one fixed cutoff now.
        report.error(
            f"'{kind.cfg_key}.vanilla_threshold' is no longer read -- the "
            f"baseline is a single fixed cutoff; rename it to "
            f"'{kind.cfg_key}.threshold' with one number"
        )

    if "column" in bx and "columns" in bx:
        # Which one wins would be invisible in the output: both spellings
        # produce rows named after the family.
        report.error(
            f"'{kind.cfg_key}' names both 'column' and 'columns' -- use one. "
            f"'column' is a single name, 'columns' a list of names or of "
            f"{{column, threshold, label}} mappings."
        )

    if "columns" in bx and not isinstance(bx["columns"], list):
        report.error(f"'{kind.cfg_key}.columns' must be a list")

    if not bx.get("enabled", True):
        return

    shared_threshold = _as_threshold(bx.get("threshold"))
    if "threshold" in bx and shared_threshold is None:
        report.error(f"'{kind.cfg_key}.threshold' must be a single number")

    seen: dict[str, str] = {}
    for i, declared in enumerate(_declared_columns(bx, kind)):
        where = f"{kind.cfg_key}.columns[{i}]" if "columns" in bx else f"{kind.cfg_key}.column"

        if isinstance(declared, str):
            col, threshold, label = declared, shared_threshold, None
        elif isinstance(declared, dict):
            col = declared.get("column")
            threshold = _as_threshold(declared.get("threshold"))
            if "threshold" in declared and threshold is None:
                report.error(f"'{where}.threshold' must be a single number")
            threshold = shared_threshold if threshold is None else threshold
            label = declared.get("label")
            if label is not None and not isinstance(label, str):
                report.error(f"'{where}.label' must be a string")
        else:
            report.error(
                f"'{where}' must be a column name or a {{column, threshold, label}} mapping"
            )
            continue

        if not isinstance(col, str) or not col.strip():
            report.error(f"'{where}' must name a non-empty column")
            continue
        col = col.strip()

        if threshold is None:
            # Every column is a different detector reading, so every one needs
            # its own operating point -- or one shared cutoff declared on the
            # block, which is only honest when the columns are on one scale.
            report.error(
                f"'{where}': no threshold for {col!r}. A cutoff is never fitted, so "
                f"it has to be named -- on the entry, or as "
                f"'{kind.cfg_key}.threshold' for every column of the block."
            )

        name = baseline_slug(kind, col)
        if name in seen:
            report.error(
                f"'{where}': {col!r} and {seen[name]!r} both name the baseline row "
                f"{name!r} and would overwrite each other's output. Give one of "
                f"them a distinct column name."
            )
        else:
            seen[name] = col


def check_score_baselines(cfg: dict, report: Report) -> None:
    """Validate every baseline block."""
    for kind in BASELINE_KINDS:
        check_baseline_block(cfg, kind, report)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def usable_mask(
    df: pd.DataFrame, columns: list[str], group_col: str
) -> tuple[np.ndarray, dict[str, int]]:
    """Rows usable for training, and how many each column knocked out.

    Same rule as ``train_utils.drop_missing_rows`` -- a missing value takes down
    its whole group, so a human document with no embedding does not leave its
    machine rewrites behind as orphaned half-groups -- but quiet, because this
    runs once per (feature set, dataset) pair and the counts are reported as a
    table instead.
    """
    bad = np.zeros(len(df), dtype=bool)
    per_col: dict[str, int] = {}
    for col in columns:
        col_bad = df[col].map(is_missing).to_numpy()
        n_bad = int(col_bad.sum())
        if n_bad:
            per_col[col] = n_bad
        bad |= col_bad

    if bad.any():
        bad_groups = df.loc[bad, group_col].unique()
        bad |= df[group_col].isin(bad_groups).to_numpy()
    return ~bad, per_col


def feature_dims(series: pd.Series) -> set[int]:
    """Widths this column contributes to the design matrix, one entry per row.

    A vector column contributes its length, a scalar column contributes 1. More
    than one value means the column is ragged and cannot be stacked.
    """
    dims = set()
    for value in series:
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            dims.add(len(value))
        else:
            dims.add(1)
    return dims


@dataclass
class Prepared:
    """One corpus, cleaned for one feature set."""

    name: str
    df: pd.DataFrame  # rows kept, with a binarised 'y' and an aliased 'group'
    n_file: int  # rows in the parquet -- ALL domains of a combined file
    n_class_dropped: int  # removed by drop_classes
    n_missing_dropped: int  # removed by the missing-value (group-aware) drop
    missing_per_col: dict[str, int]
    dims_per_col: dict[str, set[int]]
    n_filtered: int = 0  # rows of the file the `where` conditions excluded

    @property
    def n_corpus(self) -> int:
        """Rows of THIS corpus before its own filters -- the honest denominator.

        Not ``n_file``: one slice of a combined parquet is a third of the file,
        so a fraction taken over the file understates every drop by 3x and the
        thresholds keyed on it stop firing when they should.
        """
        return self.n_file - self.n_filtered

    @property
    def feature_dim(self) -> int:
        return sum(max(d) for d in self.dims_per_col.values())

    def dim_of(self, columns: list[str]) -> int:
        """Width contributed by ``columns`` only.

        The loader reads the feature set *and* every auxiliary column in one
        pass, so that a hybrid algorithm and a plain one train on identical rows.
        The auxiliary columns never enter the shared design matrix, though, so
        the width reported next to a variant has to exclude them.
        """
        return sum(max(d) for c, d in self.dims_per_col.items() if c in set(columns))

    @property
    def ragged_cols(self) -> list[str]:
        return [c for c, d in self.dims_per_col.items() if len(d) > 1]

    @property
    def class_counts(self) -> dict[str, int]:
        counts = self.df["y"].value_counts().to_dict()
        return {BINARY_CLASSES[c]: int(n) for c, n in sorted(counts.items())}

    @property
    def n_groups(self) -> int:
        return int(self.df["group"].nunique())


def load_prepared(
    spec: CorpusSpec,
    features: list[str],
    group_col: str,
    human_label: str,
    carry: list[str] = (),
) -> Prepared:
    """Read one corpus for one feature set and apply every filter the run applies.

    Only the needed columns are read: these parquets hold every embedder, so
    reading all of them would pull hundreds of MB per corpus per feature set.
    ``read_corpus`` handles the domain slice and drop_classes; what is left here
    is what depends on the feature set.

    ``carry`` names columns to read and keep beside the features without making
    them features: they are filtered with everything else, so they stay aligned
    row-for-row with the design matrix, but they never enter it, never widen it,
    and a missing value in one never drops a row. That is what a run needs to
    describe its own split -- "how much of the test half is essay" is answered
    with ``domain``, a column the model must not see.
    """
    # A ``<col>__bin`` feature is derived, not stored: read its raw text column.
    columns = list(
        dict.fromkeys(
            [group_col, CLF_TARGET_COL]
            + [raw_feature_col(f) for f in features]
            + list(carry)
        )
    )
    load = read_corpus(spec, columns=columns)
    df, n_file = load.df, load.n_file
    n_class_dropped = load.n_class_dropped
    n_filtered = load.n_filtered

    # Materialise the binarised text features so they stack and get missing-checked
    # like any other numeric column. Done after drop_classes so a dropped generator
    # never contributes a row here.
    for feat in features:
        if feat.endswith(BIN_SUFFIX):
            df[feat] = binarize_text_column(df[raw_feature_col(feat)], human_label)

    # The target and the group column join the filter, not just the features: a
    # row with no `generated_by` would binarise to "machine" (it is != the human
    # label) and a row with no group would be split as if it were its own
    # document. The single-feature runner always dropped these; the two swept
    # ones did not, and this is the stricter of the two behaviours.
    keep, missing_per_col = usable_mask(
        df, [group_col, CLF_TARGET_COL] + list(features), group_col
    )
    n_missing_dropped = int((~keep).sum())
    df = df[keep].reset_index(drop=True)

    dims_per_col = {col: feature_dims(df[col]) for col in features} if len(df) else {}

    # The corpora disagree on generator names, so only this binary view is
    # comparable across them. 'group' is a stable alias for the config's column.
    if len(df):
        df["y"] = (df[CLF_TARGET_COL] != human_label).astype(int)
        df["group"] = df[group_col]

    return Prepared(
        name=spec.name,
        df=df,
        n_file=n_file,
        n_class_dropped=n_class_dropped,
        n_missing_dropped=n_missing_dropped,
        missing_per_col=missing_per_col,
        dims_per_col=dims_per_col,
        n_filtered=n_filtered,
    )


# ---------------------------------------------------------------------------
# Split, metrics, aggregation
# ---------------------------------------------------------------------------


def grouped_split(
    y: np.ndarray, groups: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """The one train/test split of the whole experiment.

    Grouped, so a document and its rewrites never straddle the cut. The splitter
    only uses the row count of X, so a dummy stands in for the (large) design
    matrix -- which lets the preflight replay exactly this split without building
    one.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(np.zeros((len(y), 1)), y, groups))


def split_composition(
    df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    columns: list[str],
) -> dict:
    """How each side of the split is made up, per column, averaged over seeds.

    ``GroupShuffleSplit`` groups but does not stratify: it samples whole groups
    at random, so a column's proportions are preserved only *in expectation*.
    With ~900 test groups over three domains that is roughly +-1.6 points of
    sampling noise per domain -- small, but nothing in a run's output said what
    the actual split was, which is a gap once a model trains on several domains
    pooled together.

    The counts are means over the seeds because the split differs per seed, like
    every other number in a payload. Values are read as strings, and a missing
    one becomes ``"—"`` rather than vanishing from the table.
    """
    out: dict[str, dict] = {}
    for col in columns:
        if col not in df.columns:
            continue
        values = df[col].astype("object").where(df[col].notna(), "—").astype(str)
        per_value: dict[str, dict[str, float]] = {}
        for name, idx_of in (("train", 0), ("test", 1)):
            for split in splits:
                counts = values.iloc[split[idx_of]].value_counts()
                for value, n in counts.items():
                    per_value.setdefault(str(value), {"train": 0.0, "test": 0.0})
                    per_value[str(value)][name] += float(n)
        n_splits = max(len(splits), 1)
        for value in per_value:
            for side in ("train", "test"):
                per_value[value][side] /= n_splits
        # Shares alongside the counts: the counts say how big, the shares say
        # whether the two halves look alike, which is the actual question.
        for side in ("train", "test"):
            total = sum(v[side] for v in per_value.values()) or 1.0
            for value in per_value:
                per_value[value][f"{side}_share"] = per_value[value][side] / total
        out[col] = dict(sorted(per_value.items()))
    return out


def print_split_composition(composition: dict, indent: str = "  ") -> None:
    """Print what :func:`split_composition` measured, one block per column."""
    for col, per_value in composition.items():
        print(f"\n{indent}split by {col!r} (mean rows over seeds)")
        print(f"{indent}{'value':<24}{'train':>9}{'test':>9}{'train %':>10}{'test %':>9}")
        print(indent + "-" * 61)
        for value, c in per_value.items():
            print(
                f"{indent}{value:<24}{c['train']:>9.0f}{c['test']:>9.0f}"
                f"{c['train_share']*100:>9.1f}%{c['test_share']*100:>8.1f}%"
            )


# ---------------------------------------------------------------------------
# Score metrics: AUROC and TPR at a fixed FPR
# ---------------------------------------------------------------------------
#
# Accuracy answers "how often is the label right", which is not the question a
# deployed detector is graded on. Accusing a human of writing with an LLM is the
# expensive error, so the operating point that matters is a low false-positive
# one, and the number that matters there is the true-positive rate *at* it --
# TPR@FPR (Sarvazyan et al., arXiv:2412.05139, which reports TPR at FPR = 1% and
# finds values as low as 0% for detectors whose headline accuracy looks fine).
#
# Neither TPR@FPR nor AUROC is recoverable from a confusion matrix: both read the
# *ranking* of a continuous score, which is why the algorithms grew
# ``decision_score`` and why these numbers cannot be backfilled onto a matrix
# that was already run.
#
# TWO WAYS TO READ TPR@FPR, AND THEY ARE NOT THE SAME CLAIM
# ---------------------------------------------------------
# A TPR@FPR needs a threshold placed at that FPR, and *which rows the threshold
# is placed on* is the whole difference between a separability measure and a
# deployment measure. Both are reported, under distinct key prefixes:
#
#   (a) ``tpr_at_fpr_<level>``  -- the threshold is read off the evaluation
#       set's own ROC curve. This is the literature's number, including the
#       paper above. It is a point on the ROC curve and so measures separability
#       only: it assumes a calibrator with access to human text from the very
#       corpus being tested.
#
#   (b) ``tpr_at_srcfpr_<level>`` -- the threshold is calibrated to hit that FPR
#       on the SOURCE corpus's held-out split, then applied unchanged to the
#       target. ``fpr_at_srcfpr_<level>`` reports what FPR that threshold
#       actually produced there. This is the deployable number: a real detector
#       is calibrated once and then meets text it has never seen.
#
# (b) is the one this experiment exists to measure. A model can hold its AUROC
# under transfer -- the score still separates -- while its 1%-FPR threshold lands
# at 30% FPR on the target, because the score *distribution* shifted even though
# the ordering survived. That failure is invisible to (a), to accuracy and to
# AUROC alike, and it is the difference between "this detector ranks well" and
# "this detector can be deployed".
#
# On the diagonal the source's held-out split IS the evaluation set, so (b)
# degenerates to (a) by construction and its realized FPR is the nominal level.
# Only the off-diagonal cells of the (b) matrices carry new information; the
# diagonal is there so the two matrices are read on the same axes.
#
# A caveat that no key can carry: at FPR = 1% the threshold is set by the
# ~n_human/100 most machine-looking human rows -- 3 rows on a 300-human in-domain
# split, 10 on a 1000-human transfer cell. The estimate is correspondingly
# coarse, and the seed-to-seed spread understates its sampling error. 5% is the
# headline for that reason and 1% is the stress number.

#: FPR operating points every run reports, tightest first.
TPR_AT_FPR_LEVELS: tuple[float, ...] = (0.01, 0.05)


def level_tag(level: float) -> str:
    """A level as it appears in a metric key / filename: 0.01 -> ``'0.01'``."""
    return f"{float(level):g}"


def oracle_tpr_key(level: float) -> str:
    """Key for variant (a): TPR at a threshold fit on the evaluation set itself."""
    return f"tpr_at_fpr_{level_tag(level)}"


def transported_tpr_key(level: float) -> str:
    """Key for variant (b): TPR under the source-calibrated threshold."""
    return f"tpr_at_srcfpr_{level_tag(level)}"


def transported_fpr_key(level: float) -> str:
    """Key for variant (b): the FPR that threshold actually produced here."""
    return f"fpr_at_srcfpr_{level_tag(level)}"


def score_metric_keys(levels: tuple[float, ...] = TPR_AT_FPR_LEVELS) -> list[str]:
    """Every score-derived per-cell key, in reporting order.

    Drives the seed aggregation, the printed tables and the figures, so a new
    operating point is added in exactly one place (``TPR_AT_FPR_LEVELS``).
    """
    keys = ["auroc"]
    for level in levels:
        keys += [
            oracle_tpr_key(level),
            transported_tpr_key(level),
            transported_fpr_key(level),
        ]
    return keys


def auroc_score(y: np.ndarray, scores: np.ndarray) -> float:
    """AUROC with machine (``y == 1``) as the positive class. NaN if single-class.

    Deliberately *not* orientation-corrected. A detector whose score runs the
    wrong way scores below 0.5 and should read that way: flipping it to
    ``1 - auroc`` would present a broken detector as a working one, which is
    exactly how an inverted Binoculars score hides inside a fitted threshold.
    """
    y = np.asarray(y)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))


def roc_point_at_fpr(
    y: np.ndarray, scores: np.ndarray, level: float
) -> tuple[float, float]:
    """``(tpr, threshold)`` at the largest achievable FPR <= ``level``.

    The threshold is in ``score >= threshold ⇒ machine`` form, matching
    sklearn's ``roc_curve``. The FPR grid is discrete -- it can only step at an
    observed human score -- so the *achievable* point at or below the level is
    taken rather than an interpolated one, which is the conservative reading and
    the one that corresponds to a threshold you could actually deploy.

    When no threshold reaches the level (fewer than ``1/level`` human rows, or
    ties in the tail), ``roc_curve``'s leading point stands: TPR 0 at an
    infinite threshold, i.e. "this operating point is not reachable here".
    Returns ``(nan, nan)`` on a single-class set, where there is no ROC at all.
    """
    y = np.asarray(y)
    if np.unique(y).size < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y, np.asarray(scores, dtype=float))
    # fpr is non-decreasing and starts at 0, so this is never empty.
    idx = int(np.flatnonzero(fpr <= level)[-1])
    return float(tpr[idx]), float(thr[idx])


def rates_at_threshold(
    y: np.ndarray, scores: np.ndarray, threshold: float
) -> tuple[float, float]:
    """``(tpr, fpr)`` when a FIXED threshold is applied: ``score >= thr ⇒ machine``.

    The realized FPR is the point of the pair. A transported threshold is only
    as good as the FPR it actually delivers on the corpus it lands on, and a TPR
    quoted without it is meaningless -- 100% TPR at 100% FPR is a detector that
    calls everything machine.
    """
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(threshold):
        # An unreachable calibration point: nothing is called machine.
        pred = np.zeros(len(y), dtype=bool)
    else:
        pred = scores >= threshold
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    tpr = float(pred[y == 1].mean()) if n_pos else float("nan")
    fpr = float(pred[y == 0].mean()) if n_neg else float("nan")
    return tpr, fpr


def calibrate_thresholds(
    y: np.ndarray,
    scores: np.ndarray,
    levels: tuple[float, ...] = TPR_AT_FPR_LEVELS,
) -> dict[str, float]:
    """Per level, the threshold that hits that FPR on THESE rows. Keyed by tag.

    The calibration half of variant (b): fit here, then hand to
    :func:`score_metrics` as ``cal_thresholds`` for the rows the model has never
    seen.
    """
    return {
        level_tag(level): roc_point_at_fpr(y, scores, level)[1] for level in levels
    }


def score_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    cal_thresholds: dict[str, float] | None = None,
    levels: tuple[float, ...] = TPR_AT_FPR_LEVELS,
) -> dict:
    """AUROC plus both readings of TPR@FPR for one evaluation set.

    ``cal_thresholds`` is what :func:`calibrate_thresholds` returned on the
    source's held-out split. Without it only variant (a) is reported -- there is
    no transported threshold to apply -- and the ``*_at_srcfpr_*`` keys are
    absent rather than NaN, so a payload never claims a number it did not make.
    """
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    out: dict = {"auroc": auroc_score(y, scores)}

    for level in levels:
        tag = level_tag(level)
        tpr_a, thr_a = roc_point_at_fpr(y, scores, level)
        out[oracle_tpr_key(level)] = tpr_a
        # The threshold behind (a), so a cell can be traced without re-deriving
        # it, and so the two thresholds can be compared side by side.
        out[f"threshold_at_fpr_{tag}"] = thr_a

        if cal_thresholds is not None and tag in cal_thresholds:
            thr_b = cal_thresholds[tag]
            tpr_b, fpr_b = rates_at_threshold(y, scores, thr_b)
            out[transported_tpr_key(level)] = tpr_b
            out[transported_fpr_key(level)] = fpr_b
            out[f"threshold_at_srcfpr_{tag}"] = float(thr_b)
    return out


def metrics_dict(
    y: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
    cal_thresholds: dict[str, float] | None = None,
) -> dict:
    """The metrics of one evaluation set. Identical for every kind of variant.

    Trained classifiers, hybrids and threshold rules all end at a vector of
    predicted labels, so they all report the same keys -- which is what lets
    their matrices sit in one comparison and one figure.

    ``scores`` is the continuous ranking score behind those labels, where the
    variant has one (see ``ClfAlgorithm.decision_score``). It adds AUROC and the
    two TPR@FPR readings documented above; without it the label-only keys are
    all that is written, which is what an algorithm that cannot rank its rows
    should report.
    """
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "confusion": confusion_matrix(
            y, y_pred, labels=range(len(BINARY_CLASSES))
        ).tolist(),
        "n": int(len(y)),
        "class_balance": {
            BINARY_CLASSES[c]: int(n)
            for c, n in zip(*np.unique(y, return_counts=True))
        },
    }
    if scores is not None:
        out.update(score_metrics(y, scores, cal_thresholds))
    return out


def mean_std_over_seeds(runs: list[dict], key: str) -> dict:
    """``{key}_mean`` / ``{key}_std`` over the seeds that reported ``key``.

    Returns ``{}`` when no seed did, so a label-only variant simply has no AUROC
    keys instead of a column of NaNs that plots and rankings then have to guard.
    NaN values (a single-class evaluation set has no ROC) are averaged with
    ``nanmean``: one degenerate seed should not erase the others.
    """
    values = [float(r[key]) for r in runs if key in r]
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).any():
        return {f"{key}_mean": float("nan"), f"{key}_std": float("nan")}
    return {
        f"{key}_mean": float(np.nanmean(arr)),
        f"{key}_std": float(np.nanstd(arr)),
    }


def aggregate_seeds(
    per_seed: list[dict], names: list[str], mean_keys: tuple[str, ...] = ()
) -> dict:
    """Collapse ``per_seed[i][source][target]`` into one summary matrix.

    ``mean_keys`` names extra per-cell scalars to average as ``<key>_mean`` --
    the baselines' ``threshold``, which the classifiers do not have.
    Everything else that is constant across seeds (row counts, the number of
    excluded groups) is taken from the first seed rather than averaged.

    The score metrics (AUROC, both TPR@FPR readings, the thresholds behind them)
    are averaged when present and skipped when the variant produced no ranking
    score, so an algorithm without one keeps exactly the schema it had before.
    """
    score_keys = score_metric_keys() + [
        f"threshold_at_{side}_{level_tag(lvl)}"
        for lvl in TPR_AT_FPR_LEVELS
        for side in ("fpr", "srcfpr")
    ]
    summary: dict[str, dict] = {}
    for source in names:
        summary[source] = {}
        for target in names:
            runs = [s[source][target] for s in per_seed]
            accs = [r["accuracy"] for r in runs]
            f1s = [r["macro_f1"] for r in runs]
            cell = {
                "train_set": source,
                "test_set": target,
                "in_domain": source == target,
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "macro_f1_mean": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s)),
                "confusion_mean": np.mean(
                    [r["confusion"] for r in runs], axis=0
                ).tolist(),
                "train_n": runs[0]["train_n"],
                "n": runs[0]["n"],
                "excluded": runs[0]["excluded"],
                "class_balance": runs[0]["class_balance"],
            }
            for key in score_keys:
                cell.update(mean_std_over_seeds(runs, key))
            for key in mean_keys:
                cell[f"{key}_mean"] = float(np.mean([r[key] for r in runs]))
            summary[source][target] = cell
    return summary


@dataclass(frozen=True)
class MetricSpec:
    """One reported per-cell metric: how to name it, read it and colour it.

    Every consumer -- the seed aggregation, the printed tables, the per-variant
    matrices, the comparison figures and the sweep's objective list -- is driven
    off this list, so adding an operating point means adding entries here and
    nothing else.

    ``reference`` is the value the metric is read against, and it is NOT 0.5 for
    everything: a random detector scores 0.5 accuracy but only ``level`` TPR at
    FPR ``level``, so painting TPR@1%FPR on an accuracy scale would make every
    working detector look broken. For a realized FPR the reference is the
    nominal level the threshold was calibrated to and ``higher_is_better`` is
    False -- that cell is a miss in whichever direction it departs from it.

    ``floor`` / ``ceiling`` are the values the metric cannot pass. Every metric
    here is a rate or an AUROC, so both are the defaults -- but a figure that
    does not know them pads its axis past 1.0 and ends up ticking values the
    metric does not have, and never labels the ceiling the data is pressed
    against. See ``visualization_utils._cd_gutter``.
    """

    key: str  # per-cell stem; the summary holds <key>_mean and <key>_std
    label: str  # axis and colour-bar label
    short: str  # column header in the printed tables
    reference: float
    higher_is_better: bool = True
    diverging: bool = True  # diverging colour map anchored at `reference`
    note: str = ""  # one line under the figure, where the metric needs one
    floor: float = 0.0
    ceiling: float = 1.0

    @property
    def filename(self) -> str:
        """Figure stem: ``tpr_at_fpr_0.01`` -> ``cross_domain_tpr_at_fpr_0.01.png``."""
        return self.key


def cell_metric_specs(
    levels: tuple[float, ...] = TPR_AT_FPR_LEVELS
) -> list[MetricSpec]:
    """Every metric a cell can hold, in reporting order. See :class:`MetricSpec`."""
    specs = [
        MetricSpec(
            "accuracy", "Accuracy", "acc", CD_CHANCE_RATE,
            note="Both classes are balanced, so chance is 50%.",
        ),
        MetricSpec(
            "macro_f1", "Macro F1", "f1", CD_CHANCE_RATE,
            note="Accuracy alone hides a model that predicts one class on an "
                 "unseen corpus.",
        ),
        MetricSpec(
            "auroc", "AUROC", "auroc", CD_CHANCE_RATE,
            note="Ranking quality, independent of any threshold and of the class "
                 "balance. Below 0.5 means the score runs the wrong way — it is "
                 "left uncorrected on purpose.",
        ),
    ]
    for level in levels:
        tag = level_tag(level)
        pct = f"{level * 100:g}%"
        specs.append(
            MetricSpec(
                oracle_tpr_key(level),
                f"TPR @ {pct} FPR (target-calibrated)",
                f"tpr@{pct}",
                float(level),
                diverging=False,
                note=f"(a) The threshold is read off THIS cell's own ROC curve, so "
                     f"it assumes human text from the corpus being tested is "
                     f"available to calibrate on. Chance is {pct}.",
            )
        )
        specs.append(
            MetricSpec(
                transported_tpr_key(level),
                f"TPR @ {pct} source FPR (transported)",
                f"tprT@{pct}",
                float(level),
                diverging=False,
                note=f"(b) The threshold is calibrated to {pct} FPR on the SOURCE "
                     f"corpus's held-out split and applied unchanged here. Read it "
                     f"beside the realized FPR — a high TPR at a blown FPR is not "
                     f"a detection.",
            )
        )
        specs.append(
            MetricSpec(
                transported_fpr_key(level),
                f"Realized FPR @ {pct} source FPR",
                f"fprT@{pct}",
                float(level),
                higher_is_better=False,
                diverging=False,
                note=f"(b) What the source-calibrated threshold actually cost in "
                     f"false accusations here. It is {pct} on the diagonal by "
                     f"construction; every point above that is calibration that "
                     f"did not transfer.",
            )
        )
    return specs


def present_metric_specs(summary: dict, names: list[str]) -> list[MetricSpec]:
    """The metrics this summary actually holds, in reporting order.

    A variant whose algorithm has no ranking score, and any matrix written
    before the score metrics existed, carry accuracy and macro F1 only; every
    reader iterates this rather than the full list so both keep working.
    """
    if not names:
        return []
    first = summary[names[0]][names[0]]
    return [m for m in cell_metric_specs() if f"{m.key}_mean" in first]


def matrix_of(summary: dict, names: list[str], key: str) -> np.ndarray:
    """Source x target matrix of one ``<key>_mean``; NaN where a cell lacks it."""
    return np.array(
        [
            [float(summary[s][t].get(key, float("nan"))) for t in names]
            for s in names
        ],
        dtype=float,
    )


def split_diag(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(diagonal, off-diagonal)`` of a square matrix, both flattened."""
    off_mask = ~np.eye(mat.shape[0], dtype=bool)
    return np.diag(mat), mat[off_mask]


#: Key holding one variant's per-seed in-domain / transfer aggregates, as
#: ``<where>_<metric>_by_seed``: one float per seed, the mean over that seed's
#: diagonal (or off-diagonal) cells. Stored raw rather than as a mean and a
#: spread because the spread the figures draw -- an SD, a confidence interval at
#: whatever level the caller asked for -- is a plotting choice, and because the
#: cross-dataset rows have to recombine these across datasets before summarising.
BY_SEED_SUFFIX = "_by_seed"


#: Key holding the raw scores behind ONE seed's matrix, so the score
#: distributions can be drawn. Every metric in this module is a summary of these
#: numbers; the transported TPR in particular is a statement about where two
#: score distributions sit relative to one fixed cut, and no aggregate can show
#: that -- only the distributions can.
#:
#: Shape, under ``payload[SCORE_DIST_KEY]``::
#:
#:     {"seed": 0, "level": 0.01, "machine_is_high": true,
#:      "thresholds": {source: float},              # the transported cut
#:      "cells": {source: {target: {"scores": [...], "y": [0|1, ...]}}}}
#:
#: Scores are on the ORIENTED scale (high = machine) so the cut is always "at or
#: above this is called machine", whatever direction the raw column ran in.
#: ``thresholds[source]`` is fitted on the source's own held-out split -- the
#: diagonal cell of that row -- and applied unchanged across the row.
#:
#: One seed, not all of them: pooling seeds would blur the very thing the figure
#: is for (where one fixed cut fell), and the seed-to-seed movement is already
#: the error bar on every other figure. Raw rather than pre-binned, following
#: BY_SEED_SUFFIX above: the bin count is a plotting choice.
SCORE_DIST_KEY = "score_dist"

#: The operating point the stored distributions carry a cut for. One level, not
#: every level in TPR_AT_FPR_LEVELS: the figure exists to show a single line and
#: where it lands, and a second line on the same axes halves that.
SCORE_DIST_LEVEL = 0.01

#: Significant figures kept when a score vector is written to JSON. Six is well
#: past what any histogram can resolve and roughly halves the file.
SCORE_DIST_SIGFIGS = 6


def score_dist_cell(scores: np.ndarray, y: np.ndarray) -> dict:
    """One cell's oriented scores and binary labels, ready to serialise.

    ``scores`` must already be oriented (high = machine); rows whose score is
    not finite are dropped, with their labels, so the pair stays aligned.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y)
    keep = np.isfinite(scores)
    return {
        "scores": [
            float(f"{v:.{SCORE_DIST_SIGFIGS}g}") for v in scores[keep].tolist()
        ],
        "y": [int(v) for v in y[keep].tolist()],
    }


def score_dist_payload(
    cells: dict[str, dict[str, dict]],
    thresholds: dict[str, float],
    *,
    seed: int,
    level: float,
    machine_is_high: bool = True,
) -> dict:
    """Assemble :data:`SCORE_DIST_KEY`. See that constant for the shape."""
    return {
        "seed": int(seed),
        "level": float(level),
        "machine_is_high": bool(machine_is_high),
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "cells": cells,
    }


def aggregates_by_seed(
    per_seed: list[dict], names: list[str], key: str
) -> tuple[list[float], list[float]]:
    """``(in_domain, transfer)`` means of one metric, one entry per seed.

    The distinction that matters: this averages the cells WITHIN each seed and
    leaves the seeds apart, where :func:`summarise_variant`'s ``_std`` averages
    the seeds within each cell and leaves the cells apart. Both are spreads of
    the same numbers and they answer opposite questions -- "would another seed
    have said this" against "do the corpora agree" -- and only the first is an
    uncertainty on the mean the run is ranked by.

    It cannot be recovered from the per-cell means and SDs alone: that would need
    the covariance between cells across seeds, which the summary does not keep.
    Hence computing it here, from ``per_seed``, while it is still available.
    """
    in_domain: list[float] = []
    transfer: list[float] = []
    for seed in per_seed:
        diag, off = [], []
        for source in names:
            for target in names:
                try:
                    value = float(seed[source][target][key])
                except (KeyError, TypeError, ValueError):
                    value = float("nan")
                (diag if source == target else off).append(value)
        with np.errstate(invalid="ignore"):
            in_domain.append(float(np.nanmean(diag)) if diag else float("nan"))
            transfer.append(float(np.nanmean(off)) if off else float("nan"))
    return in_domain, transfer


def attach_by_seed(row: dict, variant_dir: Path) -> bool:
    """Back-fill one comparison row's ``_by_seed`` lists from its own results.

    A ``variant_comparison.json`` written before :func:`aggregates_by_seed`
    existed has no per-seed record, and nothing in it can be made to yield one.
    The record it was derived FROM is still on disk though -- ``per_seed`` in the
    variant's own ``cross_domain.json``, one directory down -- so a replot can
    recover the error bars for every run already finished, without retraining
    anything.

    Returns whether anything was added. A missing or unreadable file is not an
    error: that variant simply plots without a bar, which is the same outcome a
    single-seed run gets.
    """
    path = variant_dir / "cross_domain.json"
    if not path.is_file():
        return False
    try:
        with open(path) as f:
            payload = json.load(f)
        per_seed = payload.get("per_seed") or []
        names = payload["dataset_names"]
    except (OSError, ValueError, KeyError):
        return False
    if len(per_seed) < 2:
        return False

    added = False
    # Every metric the variant itself measured, not only the ones the row already
    # names: a pre-merge row carries the unqualified accuracy aliases and no
    # per-metric aggregates at all, and gets those filled in later
    # (``_ensure_row_aggregates``) from its matrix. Keying off the row here would
    # therefore attach nothing to exactly the payloads that need the back-fill
    # most. A ``_by_seed`` list for a metric the figures never draw costs
    # nothing; it is simply not read.
    for spec in present_metric_specs(payload.get("summary", {}), names):
        by_seed_in, by_seed_off = aggregates_by_seed(per_seed, names, spec.key)
        row[f"in_domain_{spec.key}{BY_SEED_SUFFIX}"] = by_seed_in
        row[f"transfer_{spec.key}{BY_SEED_SUFFIX}"] = by_seed_off
        added = True
    return added


def summarise_variant(payload: dict) -> dict:
    """Collapse one variant's matrix into the numbers the comparison ranks on.

    Every metric the summary holds gets the same aggregates -- its full matrix,
    the in-domain (diagonal) mean, the transfer (off-diagonal) mean, and the
    worst transfer cell -- under ``<metric>``, ``in_domain_<metric>_mean``,
    ``transfer_<metric>_mean`` and ``transfer_<metric>_worst``. "Worst" follows
    the metric's direction, so it is the minimum for a TPR and the *maximum* for
    a realized FPR, which is the cell that would get someone falsely accused.

    Where the payload has more than one seed it also gets
    ``in_domain_<metric>_by_seed`` and ``transfer_<metric>_by_seed``: those two
    means recomputed once per seed, which is what the figures put an error bar
    on (see :func:`aggregates_by_seed`).

    The accuracy aggregates keep their historical unqualified names
    (``in_domain_mean``, ``transfer_mean``, ``transfer_min``, ``gap``) as well,
    because the comparison figures and every ``variant_comparison.json`` already
    written read them.
    """
    names = payload["dataset_names"]
    summary = payload["summary"]
    specs = present_metric_specs(summary, names)
    # A zero-shot baseline is deterministic and carries no seeds, and a payload
    # written before this existed carries none either; both simply get no
    # ``_by_seed`` keys and no error bar on the figures.
    per_seed = payload.get("per_seed") or []

    row: dict = {
        "variant": payload.get("variant", payload.get("label", "")),
        "label": payload.get("label", payload.get("variant", "")),
        "feature_dim": payload["feature_dim"],
        "metrics": [m.key for m in specs],
        "baseline": bool(payload.get("baseline")),
    }

    for spec in specs:
        mat = matrix_of(summary, names, f"{spec.key}_mean")
        diag, off = split_diag(mat)
        # nan-aware: a single-class evaluation cell has no ROC and reports NaN
        # for the score metrics, and one such cell must not void the whole row.
        with np.errstate(invalid="ignore"):
            diag_mean = float(np.nanmean(diag)) if diag.size else float("nan")
            off_mean = float(np.nanmean(off)) if off.size else float("nan")
            worst = (
                (float(np.nanmin(off)) if spec.higher_is_better else float(np.nanmax(off)))
                if off.size and np.isfinite(off).any()
                else float("nan")
            )
        row[spec.key] = mat.tolist()
        row[f"{spec.key}_std"] = matrix_of(
            summary, names, f"{spec.key}_std"
        ).tolist()
        row[f"in_domain_{spec.key}_mean"] = diag_mean
        row[f"in_domain_{spec.key}_std"] = (
            float(np.nanstd(diag)) if diag.size else float("nan")
        )
        row[f"transfer_{spec.key}_mean"] = off_mean
        row[f"transfer_{spec.key}_std"] = (
            float(np.nanstd(off)) if off.size else float("nan")
        )
        row[f"transfer_{spec.key}_worst"] = worst
        # The seed spread of these two means, kept apart from the cell spread
        # above. See :func:`aggregates_by_seed` for why one cannot be got from
        # the other.
        if len(per_seed) > 1:
            by_seed_in, by_seed_off = aggregates_by_seed(per_seed, names, spec.key)
            row[f"in_domain_{spec.key}{BY_SEED_SUFFIX}"] = by_seed_in
            row[f"transfer_{spec.key}{BY_SEED_SUFFIX}"] = by_seed_off
        # In-domain minus transfer, in the metric's own direction: always "how
        # much was lost by leaving the corpus", never a sign that flips meaning.
        row[f"{spec.key}_gap"] = (
            diag_mean - off_mean if spec.higher_is_better else off_mean - diag_mean
        )

    # Historical accuracy aliases. Kept unqualified because they are what the
    # comparison figures, the ranking and every sweep config already name.
    row.update(
        accuracy_std=row.get("accuracy_std"),
        in_domain_mean=row["in_domain_accuracy_mean"],
        in_domain_std=row["in_domain_accuracy_std"],
        in_domain_min=float(np.nanmin(np.diag(np.array(row["accuracy"])))),
        transfer_mean=row["transfer_accuracy_mean"],
        transfer_std=row["transfer_accuracy_std"],
        transfer_min=row["transfer_accuracy_worst"],
        transfer_f1_mean=row.get("transfer_macro_f1_mean", float("nan")),
        gap=row["accuracy_gap"],
    )
    return row


def summarise_baseline(payload: dict) -> dict:
    """A baseline's comparison row: a variant row, relabelled and flagged.

    The label is the family's (``binoculars`` / ``llm_judge``), so two baselines
    never collapse into one indistinguishable "baseline" row in the ranking. The
    score column rides along because a family can now contribute several rows --
    one per column -- and the column is what tells them apart when two of them
    happen to draw identically.
    """
    row = summarise_variant(payload)
    row["baseline"] = True
    row["baseline_kind"] = payload.get("baseline_kind", BASELINE_KINDS[0].key)
    row["column"] = payload.get("column")
    return row


# ---------------------------------------------------------------------------
# Preflight checks on the data
# ---------------------------------------------------------------------------


def check_dataset(prep: Prepared, cfg: dict, tag_col: str, report: Report) -> bool:
    """Value-level checks for one (feature set, dataset) pair. False if unusable."""
    tag = f"{tag_col} | {prep.name}"
    min_rows: int = cfg.get("min_rows", 50)
    min_share: float = cfg.get("min_class_share", 0.10)

    if len(prep.df) == 0:
        report.error(
            f"{tag}: no rows left ({prep.n_corpus} in this corpus"
            + (f" of {prep.n_file} in the file" if prep.n_filtered else "")
            + f", {prep.n_class_dropped} dropped by drop_classes, "
            f"{prep.n_missing_dropped} by missing values)"
        )
        return False

    if prep.ragged_cols:
        for col in prep.ragged_cols:
            report.error(
                f"{tag}: {col} has inconsistent widths {sorted(prep.dims_per_col[col])} "
                f"— it cannot be stacked into a design matrix"
            )
        return False

    if len(prep.df) < min_rows:
        report.error(
            f"{tag}: only {len(prep.df)} usable row(s), below min_rows={min_rows} "
            f"({prep.n_missing_dropped} dropped for missing values: "
            f"{prep.missing_per_col})"
        )
        return False

    counts = prep.class_counts
    if len(counts) < 2:
        report.error(
            f"{tag}: only one class left after filtering ({counts}); "
            f"a binary classifier needs both human and machine rows"
        )
        return False

    n = len(prep.df)
    for cls, count in counts.items():
        if count / n < min_share:
            report.warn(
                f"{tag}: class {cls!r} is {100*count/n:.1f}% of the rows "
                f"({count}/{n}); the 50% chance line on the figures is misleading "
                f"when the classes are this skewed"
            )

    if prep.n_groups < 2:
        report.error(
            f"{tag}: only {prep.n_groups} group(s) in "
            f"{cfg['data']['group_col']!r}; a grouped split needs at least 2"
        )
        return False

    # A column that exists but is mostly Null trains on a fraction of the corpus
    # while still looking like a complete run in the results.
    if prep.n_missing_dropped / max(prep.n_corpus, 1) > 0.20:
        report.warn(
            f"{tag}: dropped {prep.n_missing_dropped}/{prep.n_corpus} rows "
            f"({100*prep.n_missing_dropped/max(prep.n_corpus, 1):.1f}%) for missing values "
            f"{prep.missing_per_col}"
        )

    return True


def check_baseline_datasets(cfg: dict, report: Report) -> None:
    """Per dataset, the rows each score baseline actually has a score for.

    A baseline is filtered on its own column, not on the feature set, so a partly
    filled column -- the LLM judge only scores the generators it was asked for --
    leaves it standing on fewer rows than the trained models. That is legitimate
    (a baseline can only be read where it has scores) but it must be visible, and
    a corpus whose scored rows are all one class cannot carry a comparison at
    all, which ``check_dataset`` turns into an error before anything runs.
    """
    specs = baseline_specs(cfg)
    if not specs:
        return

    group_col: str = cfg["data"]["group_col"]
    human_label: str = cfg["data"]["human_label"]
    columns = list(dict.fromkeys(s["column"] for s in specs))

    print()
    print("score baselines — the rows each one is actually scored on")
    print(f"{'baseline column':<30}{'dataset':<20}{'scored':>8}{'unscored':>10}  classes")
    print("-" * RULE)
    for col in columns:
        for spec in dataset_specs(cfg):
            try:
                prep = load_prepared(spec, [col], group_col, human_label)
            except Exception as exc:
                report.error(
                    f"{spec.name}: cannot read {col}: {type(exc).__name__}: {exc}"
                )
                continue
            print(
                f"{col:<30}{prep.name:<20}{len(prep.df):>8}"
                f"{prep.n_missing_dropped:>10}  "
                f"{prep.class_counts if len(prep.df) else '-'}"
            )
            check_dataset(prep, cfg, col, report)


def check_splits(
    preps: dict[str, Prepared], cfg: dict, tag_col: str, report: Report
) -> None:
    """Replay every split the run will take, without fitting anything.

    Catches the two failures that only appear at fit time: a training half with
    one class in it (the model cannot be fitted), and a transfer pair where
    every target group was already seen in training (nothing left to evaluate).
    """
    names = list(preps)
    base_seed: int = cfg["seed"]
    n_runs: int = cfg["model"]["n_runs"]
    test_size: float = cfg["model"].get("test_size", 0.30)

    for i in range(n_runs):
        seed = base_seed + i
        for source in names:
            df_s = preps[source].df
            y_s = df_s["y"].to_numpy()
            groups_s = df_s["group"].to_numpy()

            try:
                train_idx, test_idx = grouped_split(y_s, groups_s, test_size, seed)
            except Exception as exc:
                report.error(
                    f"{tag_col} | {source} (seed {seed}): grouped split failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            tag = f"{tag_col} | {source} (seed {seed})"
            if len(train_idx) == 0 or len(test_idx) == 0:
                report.error(
                    f"{tag}: split leaves {len(train_idx)} train / {len(test_idx)} "
                    f"test rows; adjust test_size={test_size}"
                )
                continue
            if np.unique(y_s[train_idx]).size < 2:
                report.error(
                    f"{tag}: the training half holds a single class — with "
                    f"{preps[source].n_groups} groups a grouped split cannot keep "
                    f"both classes; lower test_size or check "
                    f"{cfg['data']['group_col']!r}"
                )
            if np.unique(y_s[test_idx]).size < 2:
                report.warn(
                    f"{tag}: the in-domain test half holds a single class; its "
                    f"accuracy is not comparable to the transfer cells"
                )

            train_groups = set(groups_s[train_idx])
            for target in names:
                if target == source:
                    continue
                df_t = preps[target].df
                keep = ~df_t["group"].isin(train_groups).to_numpy()
                n_keep = int(keep.sum())
                pair = f"{tag_col} | {source} -> {target} (seed {seed})"
                if n_keep == 0:
                    report.error(
                        f"{pair}: every target group was seen while training; "
                        f"nothing left to evaluate on"
                    )
                    continue
                if n_keep < cfg.get("min_rows", 50):
                    report.warn(
                        f"{pair}: only {n_keep} target row(s) left after excluding "
                        f"groups seen in training"
                    )
                if np.unique(df_t.loc[keep, "y"].to_numpy()).size < 2:
                    report.warn(
                        f"{pair}: the evaluation rows hold a single class; accuracy "
                        f"there is a base rate, not a transfer result"
                    )


# ---------------------------------------------------------------------------
# Running a zero-shot score baseline
# ---------------------------------------------------------------------------


def oriented_scores(scores: np.ndarray, machine_is_high: bool) -> np.ndarray:
    """The detector score turned into a machine-ness score: higher = machine.

    Binoculars and the LLM judge both read LOW ⇒ machine, so they are negated to
    match the convention every score metric assumes (and that a trained
    classifier's ``decision_score`` already follows). This is a fixed
    re-orientation declared by the baseline's ``machine_is_high``, never one
    inferred from the data: flipping a detector because it happened to score
    below 0.5 AUROC on this corpus would launder a broken detector into a
    working one, and finding that it is broken is the point.
    """
    scores = np.asarray(scores, dtype=float)
    return scores if machine_is_high else -scores


def evaluate_score_rule(
    scores: np.ndarray,
    y: np.ndarray,
    threshold: float,
    machine_is_high: bool = False,
    cal_thresholds: dict[str, float] | None = None,
) -> dict:
    """Score the threshold rule, read in the metric's direction, on one eval set.

    The threshold rule gives the labels; the oriented score gives AUROC and both
    TPR@FPR readings. ``cal_thresholds`` are on the *oriented* scale, i.e. what
    :func:`calibrate_thresholds` returned for this source's held-out split.
    """
    y_pred = (
        (scores >= threshold) if machine_is_high else (scores <= threshold)
    ).astype(int)
    return metrics_dict(
        y,
        y_pred,
        scores=oriented_scores(scores, machine_is_high),
        cal_thresholds=cal_thresholds,
    )


def run_score_baseline(cfg: dict, out_dir: Path, variant: dict) -> dict:
    """Full source x target matrix for one zero-shot score-baseline variant.

    No classifier is trained and no threshold is fitted. ``variant`` comes from
    ``baseline_specs`` and carries the column and the direction as well as the
    cutoff, which is taken from the config and used unchanged for every source
    and target.

    It still takes the *same* grouped split, seeds and group exclusion as a
    trained variant, although it fits nothing: the diagonal must still be a
    held-out split and the off-diagonal must still drop target rows whose group
    was seen, or the rows would not line up with the trained models it is
    compared against. The payload is shaped identically to a trained variant's,
    so it drops into the same figures and comparison.

    One caveat this cannot paper over: a score column that is Null for some rows
    (the LLM judge only scores the generators it was asked for) makes those rows
    disappear from the baseline's evaluation set. The row counts printed here and
    in the preflight are what the baseline actually stands on.

    The declared cutoff enters exactly one metric family -- the confusion
    matrix, so accuracy and macro F1. The headline metrics never see it: AUROC
    sweeps every threshold, a target-calibrated TPR@FPR reads its cutoff off the
    tested cell's own ROC, and a transported one reads it off the source split's
    ROC at the nominal level. That also means the baseline is not zero-shot
    under those metrics -- placing a threshold at a nominal FPR needs human rows
    to place it on, and so does an ROC. Its claim to have seen no labels holds
    only for the label-derived numbers.
    """
    threshold = float(variant["threshold"])
    score_col: str = variant["column"]
    machine_is_high: bool = bool(variant.get("machine_is_high", False))

    base_seed: int = cfg["seed"]
    group_col: str = cfg["data"]["group_col"]
    human_label: str = cfg["data"]["human_label"]
    specs: list[CorpusSpec] = dataset_specs(cfg)
    n_runs: int = cfg["model"]["n_runs"]
    test_size: float = cfg["model"].get("test_size", 0.30)

    names = [spec.name for spec in specs]
    labels = {spec.name: spec.label for spec in specs}

    data = {}
    for spec in specs:
        prep = load_prepared(spec, [score_col], group_col, human_label)
        unscored = prep.n_missing_dropped
        note = f"  ({unscored} row(s) without a {score_col})" if unscored else ""
        tqdm.write(
            f"  {spec.name:<20} {len(prep.df):>6} rows  {prep.class_counts}{note}"
        )
        data[spec.name] = prep.df

    scores_all = {
        name: df[score_col].to_numpy(dtype=float) for name, df in data.items()
    }

    # per_seed[i][source][target] -> metrics
    per_seed: list[dict] = []
    # The first seed's raw scores, for the score-distribution figure. Oriented
    # here so the stored cut always reads "at or above => machine", whichever
    # way the raw column ran. See SCORE_DIST_KEY.
    dist_cells: dict[str, dict[str, dict]] = {}
    dist_thresholds: dict[str, float] = {}

    for i in tqdm(range(n_runs), desc="Seeds", leave=False):
        keep_scores = i == 0
        seed = base_seed + i
        seed_res: dict[str, dict] = {}

        for source in names:
            df_s = data[source]
            s_s, y_s = scores_all[source], df_s["y"].to_numpy()
            groups_s = df_s["group"].to_numpy()

            # Same grouped split the classifiers take. The baseline's cutoff
            # never looks at the training half; the split is kept only so its
            # evaluation rows are the ones every other row of the table stands on.
            train_idx, test_idx = grouped_split(y_s, groups_s, test_size, seed)
            train_groups = set(groups_s[train_idx])

            # The transported operating points of variant (b): thresholds fixed
            # to hit each nominal FPR on the SOURCE's held-out split, then
            # carried unchanged to every target. Calibrated on the held-out half
            # rather than the training half so that the trained variants and the
            # baselines transport a threshold fit on the same rows.
            cal_thresholds = calibrate_thresholds(
                y_s[test_idx], oriented_scores(s_s[test_idx], machine_is_high)
            )

            if keep_scores:
                dist_cells[source] = {}
                dist_thresholds[source] = float(
                    cal_thresholds[level_tag(SCORE_DIST_LEVEL)]
                )

            seed_res[source] = {}
            for target in names:
                if target == source:
                    # In-domain: the corpus's own held-out grouped test split.
                    res = evaluate_score_rule(
                        s_s[test_idx], y_s[test_idx], threshold, machine_is_high,
                        cal_thresholds=cal_thresholds,
                    )
                    res["excluded"] = 0
                    if keep_scores:
                        dist_cells[source][target] = score_dist_cell(
                            oriented_scores(s_s[test_idx], machine_is_high),
                            y_s[test_idx],
                        )
                else:
                    df_t = data[target]
                    # Target rows whose group was seen while training would be
                    # a free hit, exactly as for the trained models, so drop them.
                    keep = ~df_t["group"].isin(train_groups).to_numpy()
                    res = evaluate_score_rule(
                        scores_all[target][keep],
                        df_t.loc[keep, "y"].to_numpy(),
                        threshold,
                        machine_is_high,
                        cal_thresholds=cal_thresholds,
                    )
                    res["excluded"] = int((~keep).sum())
                    if keep_scores:
                        dist_cells[source][target] = score_dist_cell(
                            oriented_scores(scores_all[target][keep], machine_is_high),
                            df_t.loc[keep, "y"].to_numpy(),
                        )
                res["train_n"] = int(len(train_idx))
                res["threshold"] = float(threshold)
                seed_res[source][target] = res

        per_seed.append(seed_res)

    # 'threshold_mean' is the config's cutoff in every cell, but is kept per cell
    # for a uniform schema.
    summary = aggregate_seeds(per_seed, names, mean_keys=("threshold",))

    # Shaped like a trained variant's payload so plot_clf_cd_result works
    # unchanged. `features` and `model.ml_alg` are what the figure subtitles read.
    bx_cfg = dict(cfg)
    bx_cfg["features"] = [score_col]
    bx_cfg["model"] = dict(cfg["model"])
    bx_cfg["model"]["ml_alg"] = (
        f"{variant['kind']} (fixed threshold {format_threshold(threshold)})"
    )
    bx_cfg["alg_id"] = f"{cfg['alg_id']}__{variant['name']}"
    bx_cfg["alg_label"] = f"{cfg['alg_label']} — {variant['label']}"

    payload = {
        "alg_id": bx_cfg["alg_id"],
        "alg_label": bx_cfg["alg_label"],
        "variant": variant["name"],
        "label": variant["label"],
        "baseline": True,
        "baseline_kind": variant["kind"],
        "machine_is_high": machine_is_high,
        "column": score_col,
        "threshold": threshold,
        "feature_dim": 1,
        "dataset_names": names,
        "dataset_labels": labels,
        "classes": BINARY_CLASSES,
        "config": bx_cfg,
        "summary": summary,
        "per_seed": per_seed,
        # machine_is_high True regardless of the column's own direction: the
        # scores stored above were oriented on the way in.
        SCORE_DIST_KEY: score_dist_payload(
            dist_cells, dist_thresholds,
            seed=base_seed, level=SCORE_DIST_LEVEL, machine_is_high=True,
        ),
    }
    with open(out_dir / "cross_domain.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_matrix(alg_label: str, summary: dict, names: list[str]) -> None:
    """Print the source x target matrix of every metric the summary holds.

    One block per metric, all on the same axes, so accuracy and TPR@1%FPR are
    read against each other rather than in separate reports -- which is the
    whole reason for adding the second one.
    """
    w = max(len(n) for n in names) + 2
    specs = present_metric_specs(summary, names)

    for spec in specs:
        mat = matrix_of(summary, names, f"{spec.key}_mean")
        diag, off = split_diag(mat)
        print(f"\n{'='*72}")
        print(
            f"  {alg_label} — {spec.label.upper()}  "
            f"(rows = TRAINED ON, cols = TESTED ON)"
        )
        print("  [diagonal] = in-domain held-out; off-diagonal = transfer")
        print(f"{'='*72}\n")

        print(" " * w + "".join(f"{n:>18}" for n in names))
        for i, source in enumerate(names):
            row = f"{source:<{w}}"
            for j in range(len(names)):
                cell = f"{mat[i, j]*100:5.1f}%"
                cell = f"[{cell}]" if i == j else f" {cell} "
                row += f"{cell:>18}"
            print(row)

        with np.errstate(invalid="ignore"):
            worst = (
                float(np.nanmin(off)) if spec.higher_is_better else float(np.nanmax(off))
            )
            print(f"\n  in-domain (diagonal) mean : {np.nanmean(diag)*100:5.1f}%")
            print(f"  transfer  (off-diag) mean : {np.nanmean(off)*100:5.1f}%")
            print(
                f"  transfer  (off-diag) "
                f"{'min' if spec.higher_is_better else 'max'}  : {worst*100:5.1f}%"
            )
        print(f"  reference                 : {spec.reference*100:5.1f}%")
        if spec.note:
            print(f"\n  {spec.note}")

    print(
        "\n  A transfer mean near the reference means the in-domain score is "
        "corpus-specific,\n  not 'AI-ness'."
    )


def print_ranking(alg_label: str, rows: list[dict], noun: str = "variant") -> None:
    """Rank by transfer accuracy -- the point of the whole run.

    Two tables: the accuracy ranking as before, then the same rows under every
    other metric, transfer-side only. Splitting them keeps the first table the
    width it always was while still putting AUROC and the two TPR@FPR readings
    next to the accuracy that a reader would otherwise take at face value.
    """
    order = sorted(rows, key=lambda r: r["transfer_mean"], reverse=True)
    width = max(28, max((len(r["label"]) for r in order), default=0) + 4)
    print()
    print("=" * RULE)
    print(f"  {alg_label} — {noun.upper()}S RANKED BY TRANSFER (off-diagonal) ACCURACY")
    print("=" * RULE)
    print(
        f"{noun:<{width}}{'dim':>6}{'in-domain':>12}{'transfer':>11}"
        f"{'worst':>9}{'gap':>8}"
    )
    print("-" * RULE)
    for r in order:
        mark = "★ " if r.get("baseline") else ("◆ " if r.get("hybrid") else "  ")
        print(
            f"{mark + r['label']:<{width}}{r['feature_dim']:>6}"
            f"{r['in_domain_mean']*100:>11.1f}%{r['transfer_mean']*100:>10.1f}%"
            f"{r['transfer_min']*100:>8.1f}%{r['gap']*100:>7.1f}"
        )
    print("-" * RULE)
    print("  in-domain = mean of the diagonal (held-out, same corpus)")
    print("  transfer  = mean of the off-diagonal (unseen corpus); chance is 50%")
    print("  gap       = in-domain − transfer, i.e. what does not survive transfer")
    print_metric_ranking(order, noun=noun, width=width)
    if any(r.get("baseline") for r in order):
        print("  ★ zero-shot baseline: no classifier trained")
    hybrids = [r for r in order if r.get("hybrid")]
    if hybrids:
        cols = sorted({r.get("aux_column", "?") for r in hybrids})
        print(
            f"  ◆ hybrid: trained on the shared features PLUS {', '.join(cols)}, so "
            f"it is not ranked on equal information"
        )


def print_metric_ranking(
    order: list[dict], noun: str = "variant", width: int = 32
) -> None:
    """The transfer side of every metric, one column each, rows already ordered.

    Only the transfer (off-diagonal) numbers: the in-domain ones are near-perfect
    for everything here and printing both would double the width to say nothing.
    A row missing a metric prints ``—`` rather than a NaN, which is what a
    variant whose algorithm has no ranking score genuinely has.
    """
    specs = [
        m
        for m in cell_metric_specs()
        if any(f"transfer_{m.key}_mean" in r for r in order)
    ]
    # Accuracy already had its own table above; this one exists for the rest.
    specs = [m for m in specs if m.key != "accuracy"]
    if not specs or not order:
        return

    col = 12
    header = f"{noun:<{width}}" + "".join(f"{m.short:>{col}}" for m in specs)
    print()
    print("=" * max(RULE, len(header)))
    print("  TRANSFER (off-diagonal) UNDER EVERY OTHER METRIC")
    print("=" * max(RULE, len(header)))
    print(header)
    print("-" * max(RULE, len(header)))
    for r in order:
        mark = "★ " if r.get("baseline") else ("◆ " if r.get("hybrid") else "  ")
        line = f"{mark + r['label']:<{width}}"
        for m in specs:
            value = r.get(f"transfer_{m.key}_mean")
            line += (
                f"{'—':>{col}}"
                if value is None or not np.isfinite(value)
                else f"{value*100:>{col-1}.1f}%"
            )
        print(line)
    print("-" * max(RULE, len(header)))
    for m in specs:
        arrow = "higher is better" if m.higher_is_better else "LOWER is better"
        print(f"  {m.short:<10} {m.label} — reference {m.reference*100:g}%, {arrow}")
    print(
        "  tpr@X   (a) threshold fit on the cell being tested — separability only.\n"
        "  tprT@X  (b) threshold fit to X% FPR on the SOURCE's held-out split and\n"
        "          carried over unchanged — the deployable number. fprT@X is what\n"
        "          that threshold actually cost in false positives on the target."
    )
