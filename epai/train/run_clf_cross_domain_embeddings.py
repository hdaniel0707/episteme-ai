"""Cross-domain transfer control, repeated for every embedding in the config.

``epai/train/run_clf_cross_domain.py`` answers "has the classifier learned
machine-generated-ness, or has it learned this corpus?" for *one* feature set.
This script asks the same question of every embedder in turn: each entry of
``embeddings`` in a config gets its own source x target accuracy matrix, its own
figures, and a row in a final comparison that ranks the embedders by how well
they transfer.

Configs are the YAML files in ``CFG_DIR``, one experiment each, exactly as in
``run_clf_cross_domain.py``. The only difference in their shape is that
``features`` is replaced by ``embeddings`` (a list, one matrix per entry) plus
an optional ``extra_features`` appended to all of them.

The per-embedding run is identical to the single-feature script -- grouped
split, binarised human-vs-machine target, target rows whose group was seen in
training excluded -- so the numbers are directly comparable to it. What is new:

  * one matrix per embedding, written to ``<experiment>/<alg_id>/<emb_col>/``;
  * a zero-shot Binoculars baseline (see below), one more matrix directly
    comparable to the embeddings, written to ``<experiment>/<alg_id>/binoculars/``;
  * a comparison figure over embeddings (in-domain vs transfer, the gap between
    them, and every source->target cell as a heatmap), with the baseline marked;
  * a preflight check that runs *before any training*, over *every* config.

The Binoculars baseline trains no classifier. For each source corpus a single
decision threshold on the Binoculars score is calibrated on the source's
*training* split (Youden's J, as in
``epai/analyse/analyse_binox_human_vs_ai.py``: a low score means machine, so
predict machine when ``score <= threshold``) and then applied unchanged to
every corpus's evaluation rows -- the held-out grouped test split on the
diagonal, the unseen corpora off it. The split, seeds and group exclusion are
the classifiers' exactly, so the baseline's numbers drop straight into the same
matrix and comparison. It is opt-in per config via the ``binoculars_baseline``
key.

The preflight matters because the interesting failures here are slow ones. A
run is fifteen embedders x nine matrix cells x three seeds; a column that is
missing from the third corpus, or a Null-heavy embedding that leaves one class
with no rows, would otherwise surface twenty minutes in. So everything that can
be checked without fitting a model is checked first: each config's shape and
values, that every parquet exists and holds every column, that the embedding
dimensions agree across corpora, that enough rows survive the missing-value
drop, and that every grouped split the run will actually take leaves both
classes in the training half and leaves something to evaluate on. Errors from
all configs are collected and reported together, not one per run.

The parquets carry every embedder at once and are hundreds of MB, so only the
columns a given embedding needs are ever read.

Run:
    python -m epai.train.run_clf_cross_domain_embeddings
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from epai.train.run_clf_cross_domain import BINARY_CLASSES, evaluate, print_matrix
from epai.utils.config_utils import RESET, YELLOW, collect_configs
from epai.utils.train_utils import (
    CLF_TARGET_COL,
    build_clf_model,
    is_missing,
    stack_features,
)
from epai.utils.visualization_utils import (
    plot_clf_cd_emb_comparison,
    plot_clf_cd_result,
)

BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "clf_cd_emb"

### INPUTS ###############################
CFG_DIR = BASE_DIR / "cfg" / "train" / "clf_cd_emb"
##########################################

EMB_PREFIX = "emb_"

# Zero-shot Binoculars baseline: no model is fitted, only a threshold on this
# scalar score. The output lands in this subdirectory, alongside the embeddings.
DEFAULT_BINOX_COL = "binoculars_score"
BINOX_DIR_NAME = "binoculars"

# Classifiers build_clf_model knows about. Checked up front so a typo does not
# survive until the first fit.
ML_ALGS = ("gradient_boosting", "logistic", "svm", "mlp")

RED = "\033[91m"
GREEN = "\033[92m"

RULE = 78


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def normalise_embedding(name: str) -> str:
    """``specter2`` and ``emb_specter2`` both mean the column ``emb_specter2``."""
    return name if name.startswith(EMB_PREFIX) else EMB_PREFIX + name


def embedding_columns(cfg: dict) -> list[str]:
    """The ``emb_*`` columns a config asks for, in order."""
    return [
        normalise_embedding(e)
        for e in cfg.get("embeddings", []) or []
        if isinstance(e, str)
    ]


def binoculars_baseline_col(cfg: dict) -> str | None:
    """The Binoculars column for the zero-shot baseline, or None if it is off.

    The baseline is opt-in: a config asks for it with a ``binoculars_baseline``
    mapping (``enabled`` defaults to True when the key is present, ``column`` to
    ``binoculars_score``). Malformed values are caught by ``check_config``; here
    an off / absent / malformed baseline simply reads as None.
    """
    spec = cfg.get("binoculars_baseline")
    if not isinstance(spec, dict) or not spec.get("enabled", True):
        return None
    col = spec.get("column", DEFAULT_BINOX_COL)
    return col if isinstance(col, str) and col.strip() else None


@dataclass
class Report:
    """Problems found by the preflight check.

    Errors abort the run before any model is fitted; warnings are printed and
    the run continues. ``context`` is the config file every message is about --
    several configs are checked in one pass, so an unattributed message would
    not say which file to fix.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    context: str = ""

    def _tag(self, message: str) -> str:
        return f"[{self.context}] {message}" if self.context else message

    def error(self, message: str) -> None:
        self.errors.append(self._tag(message))

    def warn(self, message: str) -> None:
        self.warnings.append(self._tag(message))

    @property
    def ok(self) -> bool:
        return not self.errors


def check_config(cfg: dict, report: Report) -> None:
    """Validate one config's shape and values, without touching any file."""
    if not isinstance(cfg, dict):
        report.error(f"config must be a YAML mapping, got {type(cfg).__name__}")
        return

    if "features" in cfg and "embeddings" not in cfg:
        # The single-feature key from cfg/train/clf_cd/*.yaml. Silently reading
        # it would run one embedding and call it a comparison.
        report.error(
            "config uses 'features'; this script takes 'embeddings' (a list, one "
            "cross-domain matrix per entry) plus an optional 'extra_features'"
        )

    for key in ("alg_id", "alg_label", "seed", "embeddings", "data", "model"):
        if key not in cfg:
            report.error(f"missing top-level key {key!r}")
    if report.errors:
        return  # everything below indexes into these keys

    if not isinstance(cfg["alg_id"], str) or not cfg["alg_id"].strip():
        report.error("'alg_id' must be a non-empty string")
    if not isinstance(cfg["seed"], int):
        report.error(f"'seed' must be an int, got {type(cfg['seed']).__name__}")

    embeddings = cfg["embeddings"]
    if not isinstance(embeddings, (list, tuple)) or not embeddings:
        report.error("'embeddings' must be a non-empty list")
    elif not all(isinstance(e, str) and e.strip() for e in embeddings):
        report.error("'embeddings' must hold non-empty strings")
    else:
        cols = embedding_columns(cfg)
        dupes = sorted({c for c in cols if cols.count(c) > 1})
        if dupes:
            report.error(f"duplicate embeddings after normalising: {dupes}")

    extra = cfg.get("extra_features") or []
    if not isinstance(extra, (list, tuple)) or not all(isinstance(c, str) for c in extra):
        report.error("'extra_features' must be a list of column names")

    bx = cfg.get("binoculars_baseline")
    if bx is not None:
        if not isinstance(bx, dict):
            report.error("'binoculars_baseline' must be a mapping")
        else:
            if "enabled" in bx and not isinstance(bx["enabled"], bool):
                report.error("'binoculars_baseline.enabled' must be a boolean")
            col = bx.get("column", DEFAULT_BINOX_COL)
            if not isinstance(col, str) or not col.strip():
                report.error("'binoculars_baseline.column' must be a non-empty string")

    data = cfg["data"]
    if not isinstance(data, dict):
        report.error("'data' must be a mapping")
    else:
        for key in ("human_label", "group_col", "datasets"):
            if key not in data:
                report.error(f"data is missing {key!r}")
        specs = data.get("datasets") or []
        if not isinstance(specs, (list, tuple)) or len(specs) < 2:
            report.error(
                f"cross-domain needs at least 2 datasets, got {len(specs)}"
            )
        else:
            names = []
            for i, spec in enumerate(specs):
                if not isinstance(spec, dict):
                    report.error(f"datasets[{i}] must be a mapping")
                    continue
                for key in ("name", "parquet_path"):
                    if key not in spec:
                        report.error(f"datasets[{i}] is missing {key!r}")
                drop = spec.get("drop_classes") or []
                if not isinstance(drop, (list, tuple)):
                    report.error(f"datasets[{i}] 'drop_classes' must be a list")
                names.append(spec.get("name"))
            dupes = sorted({n for n in names if names.count(n) > 1})
            if dupes:
                report.error(f"dataset names must be unique, repeated: {dupes}")

    model = cfg["model"]
    if not isinstance(model, dict):
        report.error("'model' must be a mapping")
        return

    ml_alg = model.get("ml_alg")
    if ml_alg not in ML_ALGS:
        report.error(f"unknown ml_alg {ml_alg!r}. Choose from: {list(ML_ALGS)}")
    else:
        try:
            build_clf_model(ml_alg, cfg.get("seed", 0))
        except Exception as exc:  # a known name that still cannot be built here
            report.error(f"ml_alg {ml_alg!r} cannot be built: {exc}")

    n_runs = model.get("n_runs")
    if not isinstance(n_runs, int) or n_runs < 1:
        report.error(f"'n_runs' must be an int >= 1, got {n_runs!r}")

    test_size = model.get("test_size", 0.30)
    if not isinstance(test_size, (int, float)) or not 0.0 < float(test_size) < 1.0:
        report.error(f"'test_size' must be in (0, 1), got {test_size!r}")

    min_rows = cfg.get("min_rows", 50)
    if not isinstance(min_rows, int) or min_rows < 1:
        report.error(f"'min_rows' must be an int >= 1, got {min_rows!r}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def check_files(cfg: dict, emb_cols: list[str], report: Report) -> None:
    """Every parquet exists, opens, and holds every column the run will ask for."""
    import pyarrow.parquet as pq

    group_col = cfg["data"]["group_col"]
    extra = list(cfg.get("extra_features") or [])
    binox = binoculars_baseline_col(cfg)
    needed = [group_col, CLF_TARGET_COL] + emb_cols + extra + ([binox] if binox else [])

    print(f"{'dataset':<22}{'rows':>9}{'cols':>7}  file")
    print("-" * RULE)
    for spec in cfg["data"]["datasets"]:
        path = BASE_DIR / spec["parquet_path"]
        if not path.is_file():
            report.error(f"{spec['name']}: parquet not found: {path}")
            print(f"{spec['name']:<22}{'-':>9}{'-':>7}  {path}  MISSING")
            continue
        try:
            # Footer only -- no row is read, so this stays cheap on 500MB files.
            pf = pq.ParquetFile(path)
            columns = list(pf.schema_arrow.names)
            n_rows = pf.metadata.num_rows
        except Exception as exc:
            report.error(
                f"{spec['name']}: cannot read {path}: {type(exc).__name__}: {exc}"
            )
            print(f"{spec['name']:<22}{'-':>9}{'-':>7}  {path}  UNREADABLE")
            continue

        print(f"{spec['name']:<22}{n_rows:>9}{len(columns):>7}  {path}")
        missing = [c for c in needed if c not in columns]
        if missing:
            # Split the message: a missing embedding is a config/parquet mismatch
            # worth naming column by column, the others are structural.
            missing_emb = [c for c in missing if c in emb_cols]
            missing_other = [c for c in missing if c not in emb_cols]
            if missing_other:
                report.error(f"{spec['name']}: missing column(s) {missing_other}")
            if missing_emb:
                report.error(
                    f"{spec['name']}: missing embedding column(s) {missing_emb} "
                    f"— run epai/embed/add_embeddings.py on this parquet, or drop "
                    f"them from 'embeddings' in the config"
                )


def usable_mask(
    df: pd.DataFrame, columns: list[str], group_col: str
) -> tuple[np.ndarray, dict[str, int]]:
    """Rows usable for training, and how many each column knocked out.

    Same rule as ``train_utils.drop_missing_rows`` -- a missing value takes down
    its whole group, so a human document with no embedding does not leave its
    machine rewrites behind as orphaned half-groups -- but quiet, because this
    runs once per (embedding, dataset) pair and the counts are reported as a
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
    n_file: int  # rows in the parquet
    n_class_dropped: int  # removed by drop_classes
    n_missing_dropped: int  # removed by the missing-value (group-aware) drop
    missing_per_col: dict[str, int]
    dims_per_col: dict[str, set[int]]

    @property
    def feature_dim(self) -> int:
        return sum(max(d) for d in self.dims_per_col.values())

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
    spec: dict, features: list[str], group_col: str, human_label: str
) -> Prepared:
    """Read one corpus for one feature set and apply every filter the run applies.

    Only the needed columns are read: these parquets hold every embedder, so
    reading all of them would pull hundreds of MB per corpus per embedding.
    """
    path = BASE_DIR / spec["parquet_path"]
    columns = list(dict.fromkeys([group_col, CLF_TARGET_COL] + features))
    df = pd.read_parquet(path, columns=columns)
    n_file = len(df)

    drop_classes = spec.get("drop_classes") or []
    if drop_classes:
        df = df[~df[CLF_TARGET_COL].isin(drop_classes)].reset_index(drop=True)
    n_class_dropped = n_file - len(df)

    keep, missing_per_col = usable_mask(df, features, group_col)
    n_missing_dropped = int((~keep).sum())
    df = df[keep].reset_index(drop=True)

    dims_per_col = {col: feature_dims(df[col]) for col in features} if len(df) else {}

    # The corpora disagree on generator names, so only this binary view is
    # comparable across them. 'group' is a stable alias for the config's column.
    if len(df):
        df["y"] = (df[CLF_TARGET_COL] != human_label).astype(int)
        df["group"] = df[group_col]

    return Prepared(
        name=spec["name"],
        df=df,
        n_file=n_file,
        n_class_dropped=n_class_dropped,
        n_missing_dropped=n_missing_dropped,
        missing_per_col=missing_per_col,
        dims_per_col=dims_per_col,
    )


def check_dataset(prep: Prepared, cfg: dict, emb_col: str, report: Report) -> bool:
    """Value-level checks for one (embedding, dataset) pair. False if unusable."""
    tag = f"{emb_col} | {prep.name}"
    min_rows: int = cfg.get("min_rows", 50)
    min_share: float = cfg.get("min_class_share", 0.10)

    if len(prep.df) == 0:
        report.error(
            f"{tag}: no rows left ({prep.n_file} in file, "
            f"{prep.n_class_dropped} dropped by drop_classes, "
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
    if prep.n_missing_dropped / prep.n_file > 0.20:
        report.warn(
            f"{tag}: dropped {prep.n_missing_dropped}/{prep.n_file} rows "
            f"({100*prep.n_missing_dropped/prep.n_file:.1f}%) for missing values "
            f"{prep.missing_per_col}"
        )

    return True


def check_splits(
    preps: dict[str, Prepared], cfg: dict, emb_col: str, report: Report
) -> None:
    """Replay every split the run will take, without fitting anything.

    Catches the two failures that only appear at fit time: a training half with
    one class in it (the model cannot be fitted), and a transfer pair where
    every target group was already seen in training (nothing left to evaluate,
    which the single-feature script raises on).
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

            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=seed
            )
            try:
                # The splitter only uses the row count of X, so a dummy avoids
                # building the (large) design matrix during the check.
                train_idx, test_idx = next(
                    splitter.split(np.zeros((len(y_s), 1)), y_s, groups_s)
                )
            except Exception as exc:
                report.error(
                    f"{emb_col} | {source} (seed {seed}): grouped split failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            tag = f"{emb_col} | {source} (seed {seed})"
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
                pair = f"{emb_col} | {source} -> {target} (seed {seed})"
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


def preflight_config(cfg: dict, emb_cols: list[str], report: Report) -> None:
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
    binox = binoculars_baseline_col(cfg)
    n_feature_sets = len(emb_cols) + (1 if binox else 0)
    n_fits = n_feature_sets * cfg["model"]["n_runs"] * len(cfg["data"]["datasets"])
    baseline_note = f" + binoculars baseline ({binox})" if binox else ""
    print(
        f"  {len(cfg['data']['datasets'])} datasets, {len(emb_cols)} embeddings"
        f"{baseline_note}, {cfg['model']['ml_alg']}, {cfg['model']['n_runs']} seeds "
        f" → {n_fits} fits"
    )

    print("\n[2/3] parquet files")
    n_errors = len(report.errors)
    check_files(cfg, emb_cols, report)
    if len(report.errors) > n_errors:
        # Without the columns there is nothing to load in stage 3.
        print("  unusable — skipping the data checks for this config")
        return

    print("\n[3/3] data, dimensions and splits")
    group_col: str = cfg["data"]["group_col"]
    human_label: str = cfg["data"]["human_label"]
    extra = list(cfg.get("extra_features") or [])
    specs = cfg["data"]["datasets"]

    print(
        f"{'feature set':<22}{'dataset':<20}{'rows':>7}{'dropped':>9}"
        f"{'dim':>7}{'groups':>8}  classes"
    )
    print("-" * RULE)

    # The Binoculars baseline is checked as one more feature set: a single scalar
    # column, no extra_features appended (it is a threshold rule, not a model).
    feature_sets: list[tuple[str, list[str]]] = [(e, [e] + extra) for e in emb_cols]
    binox = binoculars_baseline_col(cfg)
    if binox:
        feature_sets.append((binox, [binox]))

    for feat_label, features in feature_sets:
        preps: dict[str, Prepared] = {}
        for spec in specs:
            try:
                prep = load_prepared(spec, features, group_col, human_label)
            except Exception as exc:
                report.error(
                    f"{feat_label} | {spec['name']}: failed to load: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            ok = check_dataset(prep, cfg, feat_label, report)
            dropped = prep.n_class_dropped + prep.n_missing_dropped
            print(
                f"{feat_label:<22}{prep.name:<20}{len(prep.df):>7}{dropped:>9}"
                f"{(prep.feature_dim if prep.dims_per_col else 0):>7}"
                f"{(prep.n_groups if len(prep.df) else 0):>8}  "
                f"{prep.class_counts if len(prep.df) else '-'}"
            )
            if ok:
                # Keep only what the split check needs; the vectors are large and
                # the training loop re-reads them one embedding at a time.
                preps[prep.name] = Prepared(
                    name=prep.name,
                    df=prep.df[["y", "group"]].copy(),
                    n_file=prep.n_file,
                    n_class_dropped=prep.n_class_dropped,
                    n_missing_dropped=prep.n_missing_dropped,
                    missing_per_col=prep.missing_per_col,
                    dims_per_col=prep.dims_per_col,
                )
            del prep

        if len(preps) != len(specs):
            continue  # a dataset already failed; its splits are not meaningful

        # Features must live in one shared space to be transferable: the same
        # columns, in the same order, and the same width for every corpus.
        dims = {name: p.feature_dim for name, p in preps.items()}
        if len(set(dims.values())) != 1:
            report.error(
                f"{feat_label}: datasets disagree on feature dimensionality: {dims} "
                f"— the model trained on one corpus cannot be applied to another"
            )
            continue

        check_splits(preps, cfg, feat_label, report)


def preflight(cfg_paths: list[Path]) -> tuple[Report, list[tuple[Path, dict, list[str]]]]:
    """Check every config in the folder. Returns the report and the loaded configs."""
    report = Report()
    loaded: list[tuple[Path, dict, list[str]]] = []

    print("=" * RULE)
    print(f"PREFLIGHT — {len(cfg_paths)} config(s) in {CFG_DIR}")
    print("=" * RULE)

    for cfg_path in cfg_paths:
        report.context = cfg_path.name
        print(f"\n{'-' * RULE}\n{cfg_path.name}\n{'-' * RULE}")
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
        except Exception as exc:
            report.error(f"cannot parse YAML: {type(exc).__name__}: {exc}")
            continue

        emb_cols = embedding_columns(cfg) if isinstance(cfg, dict) else []
        preflight_config(cfg, emb_cols, report)
        loaded.append((cfg_path, cfg, emb_cols))

    # Each alg_id names an output directory, so a repeat would have two configs
    # writing into the same place.
    report.context = ""
    alg_ids = [cfg.get("alg_id") for _, cfg, _ in loaded if isinstance(cfg, dict)]
    dupes = sorted({a for a in alg_ids if a is not None and alg_ids.count(a) > 1})
    if dupes:
        report.error(f"alg_id must be unique across {CFG_DIR}, repeated: {dupes}")

    return report, loaded


def print_report(report: Report) -> None:
    print()
    print("=" * RULE)
    if report.warnings:
        print(f"{YELLOW}Warnings ({len(report.warnings)}){RESET}")
        for message in report.warnings:
            print(f"  {YELLOW}⚠️  {message}{RESET}")
        print()
    if report.errors:
        print(f"{RED}Preflight FAILED — {len(report.errors)} problem(s){RESET}")
        for message in report.errors:
            print(f"  {RED}✗ {message}{RESET}")
        print(f"{RED}Nothing was trained; fix the above and re-run.{RESET}")
    else:
        print(
            f"{GREEN}Preflight OK — every config, dataset, embedding and split "
            f"checks out.{RESET}"
        )
    print("=" * RULE)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_embedding(emb_col: str, cfg: dict, emb_dir: Path) -> dict:
    """Full source x target matrix for one embedding. Returns the saved payload."""
    base_seed: int = cfg["seed"]
    group_col: str = cfg["data"]["group_col"]
    human_label: str = cfg["data"]["human_label"]
    specs: list[dict] = cfg["data"]["datasets"]
    ml_alg: str = cfg["model"]["ml_alg"]
    n_runs: int = cfg["model"]["n_runs"]
    test_size: float = cfg["model"].get("test_size", 0.30)
    features = [emb_col] + list(cfg.get("extra_features") or [])

    names = [spec["name"] for spec in specs]
    labels = {spec["name"]: spec.get("label", spec["name"]) for spec in specs}

    data = {}
    for spec in specs:
        prep = load_prepared(spec, features, group_col, human_label)
        tqdm.write(f"  {spec['name']:<20} {len(prep.df):>6} rows  {prep.class_counts}")
        data[spec["name"]] = prep.df

    X_all = {name: stack_features(df, features)[0] for name, df in data.items()}
    feature_dim = next(iter(X_all.values())).shape[1]

    # per_seed[i][source][target] -> metrics
    per_seed: list[dict] = []

    for i in tqdm(range(n_runs), desc="Seeds", leave=False):
        seed = base_seed + i
        seed_model_dir = emb_dir / str(i) / "model"
        seed_model_dir.mkdir(parents=True, exist_ok=True)
        seed_res: dict[str, dict] = {}

        for source in names:
            df_s = data[source]
            X_s, y_s = X_all[source], df_s["y"].to_numpy()
            groups_s = df_s["group"].to_numpy()

            # Grouped split: a document and its rewrites never straddle the cut.
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=seed
            )
            train_idx, test_idx = next(splitter.split(X_s, y_s, groups_s))
            train_groups = set(groups_s[train_idx])

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_s[train_idx])
            model = build_clf_model(ml_alg, seed)
            model.fit(X_train, y_s[train_idx])

            joblib.dump(model, seed_model_dir / f"{source}_model.joblib")
            joblib.dump(scaler, seed_model_dir / f"{source}_scaler.joblib")

            seed_res[source] = {}
            for target in names:
                if target == source:
                    # In-domain: the corpus's own held-out grouped test split.
                    res = evaluate(model, scaler, X_s[test_idx], y_s[test_idx])
                    res["excluded"] = 0
                else:
                    df_t = data[target]
                    # `essay` and `essay_gpt54mini` share their human documents
                    # (same sources, same ids). Any target row whose group was
                    # trained on would be a free hit, so drop it.
                    keep = ~df_t["group"].isin(train_groups).to_numpy()
                    res = evaluate(
                        model,
                        scaler,
                        X_all[target][keep],
                        df_t.loc[keep, "y"].to_numpy(),
                    )
                    res["excluded"] = int((~keep).sum())
                res["train_n"] = int(len(train_idx))
                seed_res[source][target] = res

        per_seed.append(seed_res)

    # Aggregate over seeds.
    summary: dict[str, dict] = {}
    for source in names:
        summary[source] = {}
        for target in names:
            runs = [s[source][target] for s in per_seed]
            accs = [r["accuracy"] for r in runs]
            f1s = [r["macro_f1"] for r in runs]
            summary[source][target] = {
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

    # Shaped like the single-feature script's payload so the existing
    # cross-domain figures can be reused unchanged.
    emb_cfg = dict(cfg)
    emb_cfg["features"] = features
    emb_cfg["alg_id"] = f"{cfg['alg_id']}__{emb_col}"
    emb_cfg["alg_label"] = f"{cfg['alg_label']} — {emb_col}"

    payload = {
        "alg_id": emb_cfg["alg_id"],
        "alg_label": emb_cfg["alg_label"],
        "embedding": emb_col,
        "feature_dim": int(feature_dim),
        "dataset_names": names,
        "dataset_labels": labels,
        "classes": BINARY_CLASSES,
        "config": emb_cfg,
        "summary": summary,
        "per_seed": per_seed,
    }
    with open(emb_dir / "cross_domain.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def summarise_embedding(payload: dict) -> dict:
    """Collapse one embedding's matrix into the numbers the comparison ranks on."""
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
        "embedding": payload["embedding"],
        "label": payload["embedding"].replace(EMB_PREFIX, "", 1),
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


# ---------------------------------------------------------------------------
# Zero-shot Binoculars baseline
# ---------------------------------------------------------------------------


def fit_binoculars_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Youden's-J optimal Binoculars-score cutoff, fit on a source training split.

    Positive class is machine (``y == 1``). A *low* Binoculars score means
    machine, so the decision score fed to the ROC is ``-score`` (higher ⇒ more
    machine-like) and the returned cutoff is on the raw score: predict machine
    when ``score <= cutoff``. This is the rule used in
    ``epai/analyse/analyse_binox_human_vs_ai.py``, except the threshold is
    calibrated here on the *training* split rather than the evaluation set, so
    the diagonal stays an honest held-out number and the off-diagonal is a real
    transfer of the source's threshold to an unseen corpus.

    Falls back to the score median if the split holds a single class (no ROC).
    """
    if np.unique(y).size < 2:
        return float(np.median(scores))
    fpr, tpr, thr = roc_curve(y, -scores)
    j = tpr - fpr
    return float(-thr[int(np.argmax(j))])


def evaluate_binoculars(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """Score the ``score <= threshold ⇒ machine`` rule on one evaluation set.

    Same keys as ``run_clf_cross_domain.evaluate`` so the payload is identical in
    shape to an embedding's and the existing figures read it unchanged.
    """
    y_pred = (scores <= threshold).astype(int)
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


def run_binoculars_baseline(binox_col: str, cfg: dict, out_dir: Path) -> dict:
    """Full source x target matrix for the zero-shot Binoculars baseline.

    No classifier is trained. For each source a threshold is calibrated on its
    training split and applied to every corpus's evaluation rows -- the held-out
    grouped test split on the diagonal, the unseen corpora off it. The split,
    seeds and group exclusion match ``run_embedding`` exactly, and the payload is
    shaped identically, so it drops straight into the same figures and comparison.
    """
    base_seed: int = cfg["seed"]
    group_col: str = cfg["data"]["group_col"]
    human_label: str = cfg["data"]["human_label"]
    specs: list[dict] = cfg["data"]["datasets"]
    n_runs: int = cfg["model"]["n_runs"]
    test_size: float = cfg["model"].get("test_size", 0.30)

    names = [spec["name"] for spec in specs]
    labels = {spec["name"]: spec.get("label", spec["name"]) for spec in specs}

    data = {}
    for spec in specs:
        prep = load_prepared(spec, [binox_col], group_col, human_label)
        tqdm.write(f"  {spec['name']:<20} {len(prep.df):>6} rows  {prep.class_counts}")
        data[spec["name"]] = prep.df

    scores_all = {
        name: df[binox_col].to_numpy(dtype=float) for name, df in data.items()
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
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=seed
            )
            train_idx, test_idx = next(
                splitter.split(np.zeros((len(y_s), 1)), y_s, groups_s)
            )
            train_groups = set(groups_s[train_idx])
            threshold = fit_binoculars_threshold(s_s[train_idx], y_s[train_idx])

            seed_res[source] = {}
            for target in names:
                if target == source:
                    # In-domain: the corpus's own held-out grouped test split.
                    res = evaluate_binoculars(s_s[test_idx], y_s[test_idx], threshold)
                    res["excluded"] = 0
                else:
                    df_t = data[target]
                    # Target rows whose group was seen while calibrating would be
                    # a free hit, exactly as for the trained models, so drop them.
                    keep = ~df_t["group"].isin(train_groups).to_numpy()
                    res = evaluate_binoculars(
                        scores_all[target][keep],
                        df_t.loc[keep, "y"].to_numpy(),
                        threshold,
                    )
                    res["excluded"] = int((~keep).sum())
                res["train_n"] = int(len(train_idx))
                res["threshold"] = float(threshold)
                seed_res[source][target] = res

        per_seed.append(seed_res)

    # Aggregate over seeds (same shape as run_embedding's summary).
    summary: dict[str, dict] = {}
    for source in names:
        summary[source] = {}
        for target in names:
            runs = [s[source][target] for s in per_seed]
            accs = [r["accuracy"] for r in runs]
            f1s = [r["macro_f1"] for r in runs]
            summary[source][target] = {
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
                # Mean of the source-fit cutoffs over seeds; the same for every
                # target of a source, but kept per cell for a uniform schema.
                "threshold_mean": float(np.mean([r["threshold"] for r in runs])),
            }

    # Shaped like run_embedding's payload so plot_clf_cd_result works unchanged.
    bx_cfg = dict(cfg)
    bx_cfg["features"] = [binox_col]
    bx_cfg["model"] = dict(cfg["model"])
    bx_cfg["model"]["ml_alg"] = "binoculars (zero-shot)"
    bx_cfg["alg_id"] = f"{cfg['alg_id']}__binoculars"
    bx_cfg["alg_label"] = f"{cfg['alg_label']} — binoculars (zero-shot)"

    payload = {
        "alg_id": bx_cfg["alg_id"],
        "alg_label": bx_cfg["alg_label"],
        "embedding": binox_col,
        "baseline": True,
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


def summarise_baseline(payload: dict) -> dict:
    """The baseline's comparison row: an embedding row, relabelled and flagged."""
    row = summarise_embedding(payload)
    row["label"] = "binoculars (0-shot)"
    row["baseline"] = True
    return row


def print_ranking(alg_label: str, rows: list[dict]) -> None:
    """Rank the embeddings by transfer accuracy -- the point of the whole run."""
    order = sorted(rows, key=lambda r: r["transfer_mean"], reverse=True)
    print()
    print("=" * RULE)
    print(f"  {alg_label} — EMBEDDINGS RANKED BY TRANSFER (off-diagonal) ACCURACY")
    print("=" * RULE)
    print(
        f"{'embedding':<22}{'dim':>6}{'in-domain':>12}{'transfer':>11}"
        f"{'worst':>9}{'gap':>8}"
    )
    print("-" * RULE)
    for r in order:
        print(
            f"{r['label']:<22}{r['feature_dim']:>6}"
            f"{r['in_domain_mean']*100:>11.1f}%{r['transfer_mean']*100:>10.1f}%"
            f"{r['transfer_min']*100:>8.1f}%{r['gap']*100:>7.1f}"
        )
    print("-" * RULE)
    print("  in-domain = mean of the diagonal (held-out, same corpus)")
    print("  transfer  = mean of the off-diagonal (unseen corpus); chance is 50%")
    print("  gap       = in-domain − transfer, i.e. what does not survive transfer")


def run_config(cfg_path: Path, cfg: dict, emb_cols: list[str], alg_dir: Path) -> None:
    """Every embedding of one config, plus its comparison figures."""
    alg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, alg_dir / "config.yaml")
    with open(alg_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    tqdm.write("\n" + "=" * RULE)
    tqdm.write(f"  ALG ID     : {cfg['alg_id']}")
    tqdm.write(
        f"  ML ALG     : {cfg['model']['ml_alg']}  ({cfg['model']['n_runs']} seeds, "
        f"base seed {cfg['seed']})"
    )
    tqdm.write(f"  EMBEDDINGS : {len(emb_cols)}")
    tqdm.write(
        f"  SPLIT      : grouped by {cfg['data']['group_col']!r}, "
        f"test_size={cfg['model'].get('test_size', 0.30)}"
    )
    tqdm.write(f"  OUTPUT     : {alg_dir}")
    tqdm.write("=" * RULE)

    rows: list[dict] = []
    for emb_col in tqdm(emb_cols, desc="Embeddings", unit="emb", leave=False):
        tqdm.write(f"\n→ {emb_col}")
        emb_dir = alg_dir / emb_col
        emb_dir.mkdir(parents=True, exist_ok=True)

        payload = run_embedding(emb_col, cfg, emb_dir)
        print_matrix(payload["alg_label"], payload["summary"], payload["dataset_names"])

        # Plot as we go: an interrupted run still leaves finished embeddings
        # with their figures.
        plot_clf_cd_result(emb_dir)
        rows.append(summarise_embedding(payload))

    # Zero-shot Binoculars baseline: no model, just a source-fit threshold. Run
    # after the embeddings so it shows up as one more row in the comparison.
    binox_col = binoculars_baseline_col(cfg)
    if binox_col:
        tqdm.write(f"\n→ {binox_col} (zero-shot Binoculars baseline)")
        binox_dir = alg_dir / BINOX_DIR_NAME
        binox_dir.mkdir(parents=True, exist_ok=True)

        payload = run_binoculars_baseline(binox_col, cfg, binox_dir)
        print_matrix(payload["alg_label"], payload["summary"], payload["dataset_names"])
        plot_clf_cd_result(binox_dir)
        rows.append(summarise_baseline(payload))

    specs = cfg["data"]["datasets"]
    comparison = {
        "alg_id": cfg["alg_id"],
        "alg_label": cfg["alg_label"],
        "dataset_names": [s["name"] for s in specs],
        "dataset_labels": {s["name"]: s.get("label", s["name"]) for s in specs},
        "classes": BINARY_CLASSES,
        "config": cfg,
        "embeddings": rows,
    }
    with open(alg_dir / "embedding_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print_ranking(cfg["alg_label"], rows)
    plot_clf_cd_emb_comparison(alg_dir, comparison)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg_paths = collect_configs(CFG_DIR)

    # Every config is checked before any of them trains: a folder-wide run
    # should not get half-way through and then stop on the last file.
    report, loaded = preflight(cfg_paths)
    print_report(report)
    if not report.ok:
        sys.exit(1)

    experiment_id = "clf_cd_emb_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = RESULTS_DIR / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    for cfg_path, cfg, emb_cols in tqdm(loaded, desc="Configs", unit="cfg"):
        run_config(cfg_path, cfg, emb_cols, exp_dir / cfg["alg_id"])

    print(f"\nDone. Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()
