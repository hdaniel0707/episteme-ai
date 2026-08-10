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

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from epai.utils.corpus_utils import CorpusSpec, dataset_specs, read_corpus
from epai.utils.preflight_utils import RULE, Report
from epai.utils.train_utils import CLF_TARGET_COL, is_missing

# Binarised target: human -> 0, any generator -> 1.
BINARY_CLASSES = ["human", "machine"]

# Embedding columns are stored under this prefix. It is only ever stripped, for
# display: a config names its columns literally, since a feature set may mix an
# embedding with a scalar score and prefixing one for you would corrupt the other.
EMB_PREFIX = "emb_"

# A feature whose name ends in BIN_SUFFIX is not read from the parquet: it is
# derived from the raw text column of the same stem by binarising human vs
# machine (see binarize_text_column). It lets a text label such as generated_by
# be dropped into the design matrix as a 0/1 feature -- a leakage sanity check.
BIN_SUFFIX = "__bin"

# Zero-shot score baselines: no model is fitted, only a threshold on one scalar
# column. Each output lands in its own subdirectory, alongside the variants.
DEFAULT_BINOX_COL = "binoculars_score"

# The LLM judge writes P(human) per row -- 1.0 => human, 0.0 => machine -- so a
# LOW score means machine, the same direction as Binoculars. 0.5 is the natural
# fit-nothing cutoff.
DEFAULT_JUDGE_COL = "llm_judge_score_gpt54_mini"
DEFAULT_JUDGE_VANILLA = (0.5,)


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

    The two families differ in exactly three things -- which config block turns
    them on, which column holds the score, and which end of that score means
    *machine*. Everything else (the fitted/vanilla split, the grouped evaluation,
    the payload shape) is shared, so a third detector is one more entry here.
    """

    key: str  # names the output folders: <key>_fitted, <key>_vanilla_<thr>
    cfg_key: str  # the config block that configures it
    default_col: str
    machine_is_high: bool  # True when a *high* score means machine-written
    label: str  # human-readable, for preflight messages
    default_vanilla: tuple[float, ...] = ()  # cutoffs used when none is given

    @property
    def fitted_name(self) -> str:
        return f"{self.key}_fitted"

    @property
    def vanilla_prefix(self) -> str:
        return f"{self.key}_vanilla_"

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
        machine_is_high=False,  # Binoculars: a LOW score means machine
        label="Binoculars",
    ),
    BaselineKind(
        key="llm_judge",
        cfg_key="llm_judge_baseline",
        default_col=DEFAULT_JUDGE_COL,
        machine_is_high=False,  # the judge reports P(human), so LOW means machine
        label="LLM judge",
        default_vanilla=DEFAULT_JUDGE_VANILLA,
    ),
)


def baseline_col(cfg: dict, kind: BaselineKind) -> str | None:
    """The score column for one baseline family, or None if it is off.

    Every baseline is opt-in: a config asks for it with its ``cfg_key`` mapping
    (``enabled`` defaults to True when the key is present, ``column`` to the
    kind's default). Malformed values are caught by ``check_score_baselines``;
    here an off / absent / malformed baseline simply reads as None.
    """
    spec = cfg.get(kind.cfg_key)
    if not isinstance(spec, dict) or not spec.get("enabled", True):
        return None
    col = spec.get("column", kind.default_col)
    return col if isinstance(col, str) and col.strip() else None


def baseline_columns(cfg: dict) -> list[str]:
    """Every score column any enabled baseline reads, deduplicated."""
    cols = [baseline_col(cfg, k) for k in BASELINE_KINDS]
    return list(dict.fromkeys(c for c in cols if c))


def format_threshold(value: float) -> str:
    """Compact rendering of a threshold, for labels and output-folder names."""
    return f"{float(value):g}"


def vanilla_thresholds(cfg: dict, kind: BaselineKind) -> list[float]:
    """The fixed cutoffs one baseline asks for, from its ``vanilla_threshold``.

    Scalar or list. An absent key falls back to the kind's ``default_vanilla``
    (empty for Binoculars, whose published cutoffs are corpus-specific and must
    be named explicitly; 0.5 for the judge, whose scale fixes the neutral point),
    and an explicit ``vanilla_threshold: []`` skips the vanilla variant. Malformed
    entries are dropped here and reported by ``check_score_baselines`` instead.
    """
    spec = cfg.get(kind.cfg_key)
    if not isinstance(spec, dict):
        return []
    if "vanilla_threshold" not in spec:
        return list(kind.default_vanilla)
    value = spec.get("vanilla_threshold")
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]


def baseline_kind_specs(cfg: dict, kind: BaselineKind) -> list[dict]:
    """The variants of ONE baseline family, or [] when that family is off.

    Two variants, and the distinction matters when reading the comparison:

      * ``<kind>_fitted`` -- the cutoff is calibrated per source on its
        *training* split (Youden's J). No classifier is trained, but the
        threshold is still fit on data, so it is not free of the source corpus;
      * ``<kind>_vanilla_<threshold>`` -- a cutoff taken straight from the
        config and applied everywhere, fit to nothing. The genuinely zero-shot
        reference, and the only row in the table that saw no labels at all.

    The fitted variant always runs; one vanilla variant runs per threshold, so a
    config can compare several published cutoffs.
    """
    col = baseline_col(cfg, kind)
    if not col:
        return []

    common = {
        "kind": kind.key,
        "column": col,
        "machine_is_high": kind.machine_is_high,
    }
    specs = [
        {
            **common,
            "variant": "fitted",
            "threshold": None,
            "name": kind.fitted_name,
            "label": kind.fitted_name,
        }
    ]
    for thr in vanilla_thresholds(cfg, kind):
        name = f"{kind.vanilla_prefix}{format_threshold(thr)}"
        specs.append(
            {
                **common,
                "variant": "vanilla",
                "threshold": float(thr),
                "name": name,
                "label": name,
            }
        )
    return specs


def baseline_specs(cfg: dict) -> list[dict]:
    """Every baseline variant a config asks for, across all families, in order."""
    specs: list[dict] = []
    for kind in BASELINE_KINDS:
        specs.extend(baseline_kind_specs(cfg, kind))
    return specs


def describe_variant(variant: dict) -> str:
    """One-line description of a baseline variant, for the run log.

    Spells out the direction as well as the cutoff: the two families read their
    threshold opposite ways, and a log line that only shows the number would be
    ambiguous between them.
    """
    cutoff = (
        "threshold fit per source"
        if variant["threshold"] is None
        else f"fixed threshold {format_threshold(variant['threshold'])}"
    )
    rule = (
        "score>=thr ⇒ machine"
        if variant.get("machine_is_high")
        else "score<=thr ⇒ machine"
    )
    return f"{variant['column']}, {cutoff}, {rule}"


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

    col = bx.get("column", kind.default_col)
    if not isinstance(col, str) or not col.strip():
        report.error(f"'{kind.cfg_key}.column' must be a non-empty string")

    thr = bx.get("vanilla_threshold")
    if thr is not None:
        values = thr if isinstance(thr, (list, tuple)) else [thr]
        # An explicit empty list is the documented way to skip the variant.
        if isinstance(thr, (list, tuple)) and not values:
            return
        ok = all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
        )
        if not ok:
            report.error(
                f"'{kind.cfg_key}.vanilla_threshold' must be a number or a "
                "list of numbers"
            )
        elif len({format_threshold(v) for v in values}) != len(values):
            # Each threshold names an output folder; two that render the same
            # would overwrite one another.
            report.error(
                f"'{kind.cfg_key}.vanilla_threshold' has entries that are "
                f"indistinguishable once formatted: {values}"
            )


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


def metrics_dict(y: np.ndarray, y_pred: np.ndarray) -> dict:
    """The metrics of one evaluation set. Identical for every kind of variant.

    Trained classifiers, hybrids and threshold rules all end at a vector of
    predicted labels, so they all report the same keys -- which is what lets
    their matrices sit in one comparison and one figure.
    """
    y_pred = np.asarray(y_pred)
    return {
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


def aggregate_seeds(
    per_seed: list[dict], names: list[str], mean_keys: tuple[str, ...] = ()
) -> dict:
    """Collapse ``per_seed[i][source][target]`` into one summary matrix.

    ``mean_keys`` names extra per-cell scalars to average as ``<key>_mean`` --
    the baselines' fitted ``threshold``, which the classifiers do not have.
    Everything else that is constant across seeds (row counts, the number of
    excluded groups) is taken from the first seed rather than averaged.
    """
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
            for key in mean_keys:
                cell[f"{key}_mean"] = float(np.mean([r[key] for r in runs]))
            summary[source][target] = cell
    return summary


def summarise_variant(payload: dict) -> dict:
    """Collapse one variant's matrix into the numbers the comparison ranks on."""
    names = payload["dataset_names"]
    summary = payload["summary"]

    acc = np.array(
        [[summary[s][t]["accuracy_mean"] for t in names] for s in names], dtype=float
    )
    acc_std = np.array(
        [[summary[s][t]["accuracy_std"] for t in names] for s in names], dtype=float
    )
    f1 = np.array(
        [[summary[s][t]["macro_f1_mean"] for t in names] for s in names], dtype=float
    )

    diag = np.diag(acc)
    off = acc[~np.eye(len(names), dtype=bool)]
    f1_off = f1[~np.eye(len(names), dtype=bool)]

    return {
        "variant": payload.get("variant", payload.get("label", "")),
        "label": payload.get("label", payload.get("variant", "")),
        "feature_dim": payload["feature_dim"],
        "accuracy": acc.tolist(),
        "accuracy_std": acc_std.tolist(),
        "macro_f1": f1.tolist(),
        "in_domain_mean": float(diag.mean()),
        "in_domain_std": float(diag.std()),
        "in_domain_min": float(diag.min()),
        "transfer_mean": float(off.mean()),
        "transfer_std": float(off.std()),
        "transfer_min": float(off.min()),
        "transfer_f1_mean": float(f1_off.mean()),
        # How much of the in-domain score does not survive leaving the corpus.
        "gap": float(diag.mean() - off.mean()),
        "baseline": bool(payload.get("baseline")),
    }


def summarise_baseline(payload: dict) -> dict:
    """A baseline's comparison row: a variant row, relabelled and flagged.

    The label is the variant's (``binoculars_fitted`` / ``llm_judge_vanilla_0.5``
    / ...), so variants never collapse into one indistinguishable "baseline" row
    in the ranking.
    """
    row = summarise_variant(payload)
    row["baseline"] = True
    row["baseline_variant"] = payload.get("baseline_variant", "fitted")
    row["baseline_kind"] = payload.get("baseline_kind", BASELINE_KINDS[0].key)
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


def fit_score_threshold(
    scores: np.ndarray, y: np.ndarray, machine_is_high: bool = False
) -> float:
    """Youden's-J optimal cutoff for a detector score, fit on a training split.

    Positive class is machine (``y == 1``). The ROC needs a decision score that
    is *higher* for machine, so a score that is *low* for machine goes in as
    ``-score`` and the cutoff is flipped back ("predict machine when
    ``score <= cutoff``"). Both current baselines are that way round: Binoculars
    by construction, the LLM judge because it reports P(human). A detector that
    scored P(machine) would pass ``machine_is_high=True`` and go in as is
    ("predict machine when ``score >= cutoff``"). This is the rule used in
    ``epai/ai_detection/analyse/analyse_binox_human_vs_ai.py``, except the
    threshold is calibrated here on the *training* split rather than the
    evaluation set, so the diagonal stays an honest held-out number and the
    off-diagonal is a real transfer of the source's threshold to an unseen corpus.

    Falls back to the score median if the split holds a single class (no ROC).
    """
    if np.unique(y).size < 2:
        return float(np.median(scores))
    sign = 1.0 if machine_is_high else -1.0
    fpr, tpr, thr = roc_curve(y, sign * scores)
    j = tpr - fpr
    return float(sign * thr[int(np.argmax(j))])


def evaluate_score_rule(
    scores: np.ndarray,
    y: np.ndarray,
    threshold: float,
    machine_is_high: bool = False,
) -> dict:
    """Score the threshold rule, read in the metric's direction, on one eval set."""
    y_pred = (
        (scores >= threshold) if machine_is_high else (scores <= threshold)
    ).astype(int)
    return metrics_dict(y, y_pred)


def run_score_baseline(cfg: dict, out_dir: Path, variant: dict) -> dict:
    """Full source x target matrix for one zero-shot score-baseline variant.

    No classifier is trained either way. ``variant`` comes from
    ``baseline_specs`` and carries the column and the direction as well as the
    cutoff; with ``threshold: None`` the cutoff is calibrated per source on its
    training split, and with a number it is that fixed value, used unchanged for
    every source and target.

    Both take the *same* grouped split, seeds and group exclusion as a trained
    variant, even the fixed-threshold one which fits nothing: the diagonal must
    still be a held-out split and the off-diagonal must still drop target rows
    whose group was seen, or the rows would not line up with the trained models
    they are compared against. The payload is shaped identically to a trained
    variant's, so it drops into the same figures and comparison.

    One caveat this cannot paper over: a score column that is Null for some rows
    (the LLM judge only scores the generators it was asked for) makes those rows
    disappear from the baseline's evaluation set. The row counts printed here and
    in the preflight are what the baseline actually stands on.
    """
    fixed_threshold = variant["threshold"]
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

    for i in tqdm(range(n_runs), desc="Seeds", leave=False):
        seed = base_seed + i
        seed_res: dict[str, dict] = {}

        for source in names:
            df_s = data[source]
            s_s, y_s = scores_all[source], df_s["y"].to_numpy()
            groups_s = df_s["group"].to_numpy()

            # Same grouped split the classifiers take, so the calibration set is
            # the same rows they train on.
            train_idx, test_idx = grouped_split(y_s, groups_s, test_size, seed)
            train_groups = set(groups_s[train_idx])
            # The vanilla variant never looks at the training split -- it is kept
            # only so the evaluation rows match every other row of the table.
            threshold = (
                fit_score_threshold(s_s[train_idx], y_s[train_idx], machine_is_high)
                if fixed_threshold is None
                else float(fixed_threshold)
            )

            seed_res[source] = {}
            for target in names:
                if target == source:
                    # In-domain: the corpus's own held-out grouped test split.
                    res = evaluate_score_rule(
                        s_s[test_idx], y_s[test_idx], threshold, machine_is_high
                    )
                    res["excluded"] = 0
                else:
                    df_t = data[target]
                    # Target rows whose group was seen while calibrating would be
                    # a free hit, exactly as for the trained models, so drop them.
                    keep = ~df_t["group"].isin(train_groups).to_numpy()
                    res = evaluate_score_rule(
                        scores_all[target][keep],
                        df_t.loc[keep, "y"].to_numpy(),
                        threshold,
                        machine_is_high,
                    )
                    res["excluded"] = int((~keep).sum())
                res["train_n"] = int(len(train_idx))
                res["threshold"] = float(threshold)
                seed_res[source][target] = res

        per_seed.append(seed_res)

    # 'threshold_mean' is the same for every target of a source (and for every
    # cell of a vanilla variant), but is kept per cell for a uniform schema.
    summary = aggregate_seeds(per_seed, names, mean_keys=("threshold",))

    # Shaped like a trained variant's payload so plot_clf_cd_result works
    # unchanged. `features` and `model.ml_alg` are what the figure subtitles read.
    bx_cfg = dict(cfg)
    bx_cfg["features"] = [score_col]
    bx_cfg["model"] = dict(cfg["model"])
    bx_cfg["model"]["ml_alg"] = f"{variant['kind']} ({variant['variant']})"
    bx_cfg["alg_id"] = f"{cfg['alg_id']}__{variant['name']}"
    bx_cfg["alg_label"] = f"{cfg['alg_label']} — {variant['label']}"

    payload = {
        "alg_id": bx_cfg["alg_id"],
        "alg_label": bx_cfg["alg_label"],
        "variant": variant["name"],
        "label": variant["label"],
        "baseline": True,
        "baseline_variant": variant["variant"],
        "baseline_kind": variant["kind"],
        "machine_is_high": machine_is_high,
        "column": score_col,
        # None for the fitted variant, whose cutoff is per source and lives in
        # each cell's 'threshold_mean' instead.
        "threshold": None if fixed_threshold is None else float(fixed_threshold),
        "feature_dim": 1,
        "dataset_names": names,
        "dataset_labels": labels,
        "classes": BINARY_CLASSES,
        "config": bx_cfg,
        "summary": summary,
        "per_seed": per_seed,
    }
    with open(out_dir / "cross_domain.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_matrix(alg_label: str, summary: dict, names: list[str]) -> None:
    """Print the source x target accuracy matrix; the diagonal is in-domain."""
    w = max(len(n) for n in names) + 2
    print(f"\n{'='*72}")
    print(f"  {alg_label} — ACCURACY  (rows = TRAINED ON, cols = TESTED ON)")
    print("  [diagonal] = in-domain held-out; off-diagonal = transfer")
    print(f"{'='*72}\n")

    print(" " * w + "".join(f"{n:>18}" for n in names))
    for source in names:
        row = f"{source:<{w}}"
        for target in names:
            s = summary[source][target]
            cell = f"{s['accuracy_mean']*100:5.1f}%"
            cell = f"[{cell}]" if s["in_domain"] else f" {cell} "
            row += f"{cell:>18}"
        print(row)

    off = [summary[s][t]["accuracy_mean"] for s in names for t in names if s != t]
    diag = [summary[s][s]["accuracy_mean"] for s in names]
    print(f"\n  in-domain (diagonal) mean : {np.mean(diag)*100:5.1f}%")
    print(f"  transfer  (off-diag) mean : {np.mean(off)*100:5.1f}%")
    print(f"  transfer  (off-diag) min  : {np.min(off)*100:5.1f}%")
    print("\n  Chance is 50% (both classes are balanced). A transfer mean near")
    print("  chance means the in-domain score is corpus-specific, not 'AI-ness'.")


def print_ranking(alg_label: str, rows: list[dict], noun: str = "variant") -> None:
    """Rank by transfer accuracy -- the point of the whole run."""
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
    if any(r.get("baseline") for r in order):
        print("  ★ zero-shot baseline: no classifier trained")
    hybrids = [r for r in order if r.get("hybrid")]
    if hybrids:
        cols = sorted({r.get("aux_column", "?") for r in hybrids})
        print(
            f"  ◆ hybrid: trained on the shared features PLUS {', '.join(cols)}, so "
            f"it is not ranked on equal information"
        )
