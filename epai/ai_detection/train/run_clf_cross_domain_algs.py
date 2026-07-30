"""Cross-domain transfer control across ML algorithms, on one fixed feature set.

Sibling of ``epai/ai_detection/train/run_clf_cross_domain_embeddings.py``. That script fixes
the algorithm and sweeps the embedding; this one fixes the feature set and
sweeps the *algorithm*: each entry of ``algorithms`` in a config gets its own
source x target accuracy matrix, its own figures, and a row in a comparison that
ranks the algorithms by how well they transfer.

The per-algorithm run is identical to the embeddings script -- grouped split,
binarised human-vs-machine target, target rows whose group was seen in training
excluded -- so the numbers are directly comparable both to each other and to the
embedding sweep. What is new:

  * the *algorithm* varies, not the features: every algorithm is trained on the
    same ``features`` design matrix (loaded and stacked once, reused across
    algorithms), so any difference in the matrix is the algorithm's;
  * algorithms are pluggable ``ClfAlgorithm`` subclasses under ``epai/train/algs``
    (logistic, svm, gradient_boosting, simple_mlp, binox_mlp), selected by
    ``name`` and configured with per-entry ``params``. Each owns its own
    preprocessing, and each declares which ``params`` it reads: an entry naming
    anything else is a preflight error rather than a silently ignored key, so the
    hyper-parameters ``sweep_clf_cross_domain.py`` finds can be transcribed into a
    config here and either take effect or fail loudly. The resolved set (the
    entry's params plus the algorithm's defaults) is printed before each matrix
    and stored in its ``cross_domain.json``;
  * an entry may set ``binoculars`` to hand that algorithm the Binoculars score
    as a second input beside the shared design matrix -- a *hybrid*, marked with
    a ◆ in the ranking and on the figures because it is not ranked on the same
    information as the rest. The column stays out of ``features`` (so the plain
    algorithms are unaffected) but joins the missing-value filter (so every
    algorithm still trains and is scored on identical rows);
  * one matrix per algorithm, written to ``<experiment>/<alg_id>/<algo_id>/``;
  * optional zero-shot score baselines (reused from the embeddings script), each
    one more matrix marked with a ★ in the comparison. Two families, configured
    by ``binoculars_baseline`` and ``llm_judge_baseline``, each contributing a
    ``<kind>_fitted`` variant whose cutoff is calibrated per source and one
    ``<kind>_vanilla_<threshold>`` per fixed cutoff, fit to nothing at all. The
    families read their score in opposite directions -- Binoculars is *low* for
    machine text, the LLM judge is P(machine) so it is *high* -- and the judge's
    default vanilla cutoff is 0.5;
  * a comparison figure over algorithms (in-domain vs transfer, the scatter, and
    every source->target cell as a heatmap);
  * a preflight check that runs *before any training*, over *every* config.

Run:
    python -m epai.ai_detection.train.run_clf_cross_domain_algs
"""

from datetime import datetime
from pathlib import Path
import json
import shutil
import sys

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from epai.train.algs import ALGORITHMS, build_algorithm
from epai.ai_detection.train.run_clf_cross_domain import BINARY_CLASSES, print_matrix
from epai.ai_detection.train.run_clf_cross_domain_embeddings import (
    DEFAULT_BINOX_COL,
    Prepared,
    Report,
    baseline_columns,
    baseline_specs,
    check_baseline_datasets,
    check_dataset,
    check_score_baselines,
    check_splits,
    describe_variant,
    load_prepared,
    run_score_baseline,
    summarise_baseline,
    summarise_embedding,
)
from epai.utils.config_utils import RESET, YELLOW, collect_configs
from epai.utils.train_utils import CLF_TARGET_COL, stack_features
from epai.utils.visualization_utils import (
    plot_clf_cd_alg_comparison,
    plot_clf_cd_result,
)

BASE_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "clf_cd_algs"

### INPUTS ###############################
CFG_DIR = BASE_DIR / "cfg" / "ai_detection" / "train" / "clf_cd_algs"
##########################################

RED = "\033[91m"
GREEN = "\033[92m"
RULE = 78


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def spec_binox_col(entry: dict) -> str | None:
    """The auxiliary Binoculars column one ``algorithms`` entry asks for, or None.

    ``binoculars: true`` takes the default column, a string names one, and
    ``false`` / absent means the entry is a plain algorithm on ``features``
    alone. A non-None value makes the entry a *hybrid*: it sees the score on
    top of the shared feature set, which the ranking and the figures mark.
    """
    value = entry.get("binoculars")
    if value is True:
        return DEFAULT_BINOX_COL
    if isinstance(value, str) and value.strip():
        return value
    return None


def algorithm_specs(cfg: dict) -> list[dict]:
    """Normalise ``algorithms`` to ``{id, name, label, params, binoculars}`` dicts.

    An entry may be a bare string (``"logistic"``) or a mapping with ``name``
    and optional ``id`` (output-folder name, so two entries of the same
    algorithm with different params do not collide), ``label`` (shown on figures),
    ``params`` (hyper-parameters passed to the algorithm) and ``binoculars`` (an
    auxiliary column handed to the algorithm beside the design matrix -- only
    algorithms whose class sets ``uses_aux`` accept one).
    """
    specs: list[dict] = []
    for entry in cfg.get("algorithms", []) or []:
        if isinstance(entry, str):
            specs.append(
                {
                    "id": entry,
                    "name": entry,
                    "label": entry,
                    "params": {},
                    "binoculars": None,
                }
            )
        elif isinstance(entry, dict) and isinstance(entry.get("name"), str):
            name = entry["name"]
            specs.append(
                {
                    "id": entry.get("id", name),
                    "name": name,
                    "label": entry.get("label", name),
                    "params": entry.get("params") or {},
                    "binoculars": spec_binox_col(entry),
                }
            )
    return specs


def resolved_params(spec: dict, seed: int) -> dict:
    """Every hyper-parameter one entry will actually train with, defaults included.

    A config carries only what it overrides, so ``spec["params"]`` alone does not
    say which model ran -- and the interesting settings now come from
    ``sweep_clf_cross_domain.py``, transcribed by hand. Asking the algorithm to
    resolve them makes that visible: it is printed before each matrix and stored
    in the payload as ``resolved_params``, so a results directory records the
    model that was fitted rather than the diff against defaults.

    Constructing the algorithm is cheap (no network is built until ``fit``), and
    it never raises here: the preflight has already rejected anything unbuildable,
    and a run that got this far must not die over a display string.
    """
    try:
        return build_algorithm(spec["name"], seed, **spec["params"]).resolved_params()
    except Exception:
        return dict(spec["params"])


def aux_columns(cfg: dict) -> list[str]:
    """Every auxiliary column any algorithm in this config asks for, deduplicated.

    These are *not* part of ``features``: they never enter the shared design
    matrix, so the plain algorithms are unaffected by their presence. They do
    join the missing-value filter, though -- see ``load_config_data`` -- so that
    every algorithm in the config trains and is scored on an identical row set.
    """
    cols = [s["binoculars"] for s in algorithm_specs(cfg) if s["binoculars"]]
    return list(dict.fromkeys(cols))


def load_columns(cfg: dict) -> list[str]:
    """Columns every dataset is read and missing-filtered on: features + auxiliary.

    One list, used by the preflight and by the run alike, so the rows a hybrid
    algorithm trains on are exactly the rows the plain ones train on. Filtering
    per algorithm instead would score them on different subsets and quietly make
    the comparison table meaningless.
    """
    return list(dict.fromkeys(list(cfg["features"]) + aux_columns(cfg)))


def feature_only_dim(prep: Prepared, features: list[str]) -> int:
    """Width of the shared design matrix, excluding any auxiliary columns."""
    return sum(max(d) for c, d in prep.dims_per_col.items() if c in features)


def check_config(cfg: dict, report: Report) -> None:
    """Validate one config's shape and values, without touching any file."""
    if not isinstance(cfg, dict):
        report.error(f"config must be a YAML mapping, got {type(cfg).__name__}")
        return

    if "embeddings" in cfg and "algorithms" not in cfg:
        # The key from the sibling embeddings script; silently reading it would
        # run zero algorithms and call it a comparison.
        report.error(
            "config uses 'embeddings'; this script takes 'algorithms' (a list, one "
            "cross-domain matrix per entry) plus a fixed 'features' list"
        )

    for key in ("alg_id", "alg_label", "seed", "algorithms", "features", "data", "model"):
        if key not in cfg:
            report.error(f"missing top-level key {key!r}")
    if report.errors:
        return  # everything below indexes into these keys

    if not isinstance(cfg["alg_id"], str) or not cfg["alg_id"].strip():
        report.error("'alg_id' must be a non-empty string")
    if not isinstance(cfg["seed"], int):
        report.error(f"'seed' must be an int, got {type(cfg['seed']).__name__}")

    features = cfg["features"]
    if not isinstance(features, (list, tuple)) or not features:
        report.error("'features' must be a non-empty list")
    elif not all(isinstance(c, str) and c.strip() for c in features):
        report.error("'features' must hold non-empty column names")

    specs = algorithm_specs(cfg)
    raw = cfg["algorithms"]
    if not isinstance(raw, (list, tuple)) or not raw:
        report.error("'algorithms' must be a non-empty list")
    elif len(specs) != len(raw):
        report.error(
            "every 'algorithms' entry must be a name string or a mapping with a "
            "string 'name'"
        )
    else:
        ids = [s["id"] for s in specs]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            report.error(
                f"algorithm ids must be unique (set an 'id' to disambiguate): {dupes}"
            )
        for entry, s in zip(raw, specs):
            if s["name"] not in ALGORITHMS:
                report.error(
                    f"unknown algorithm {s['name']!r}. Choose from: {sorted(ALGORITHMS)}"
                )
                continue
            if not isinstance(s["params"], dict):
                report.error(f"{s['id']}: 'params' must be a mapping")
                continue

            # An auxiliary column and an algorithm that takes one must agree, in
            # both directions: a 'binoculars' key on an algorithm that ignores it
            # would silently not be a hybrid, and a hybrid without one cannot fit.
            uses_aux = bool(getattr(ALGORITHMS[s["name"]], "uses_aux", False))
            asked = isinstance(entry, dict) and entry.get("binoculars") is not None
            if asked and not s["binoculars"]:
                report.error(
                    f"{s['id']}: 'binoculars' must be true or a column name, got "
                    f"{entry.get('binoculars')!r}"
                )
            elif s["binoculars"] and not uses_aux:
                report.error(
                    f"{s['id']}: {s['name']!r} does not take an auxiliary column, so "
                    f"'binoculars' would be ignored — drop it, or use an algorithm "
                    f"that sets uses_aux"
                )
            elif uses_aux and not s["binoculars"]:
                report.error(
                    f"{s['id']}: {s['name']!r} needs an auxiliary column — set "
                    f"'binoculars: {DEFAULT_BINOX_COL}' on this entry"
                )

            # Build it and check its hyper-parameter *values* here, so a name the
            # algorithm does not read (a mistranscribed sweep winner) or an
            # out-of-range value fails now rather than inside the first fit --
            # or, worse, trains the default model while the config claims
            # otherwise. See ClfAlgorithm's docstring.
            try:
                alg = build_algorithm(s["name"], cfg.get("seed", 0), **s["params"])
                alg.validate_params()
            except Exception as exc:  # a known name that still cannot be built
                report.error(f"{s['id']}: cannot build {s['name']!r}: {exc}")

    check_score_baselines(cfg, report)

    data = cfg["data"]
    if not isinstance(data, dict):
        report.error("'data' must be a mapping")
    else:
        for key in ("human_label", "group_col", "datasets"):
            if key not in data:
                report.error(f"data is missing {key!r}")
        ds = data.get("datasets") or []
        if not isinstance(ds, (list, tuple)) or len(ds) < 2:
            report.error(f"cross-domain needs at least 2 datasets, got {len(ds)}")
        else:
            names = []
            for i, spec in enumerate(ds):
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


def check_files(cfg: dict, report: Report) -> None:
    """Every parquet exists, opens, and holds every column the run will ask for."""
    import pyarrow.parquet as pq

    group_col = cfg["data"]["group_col"]
    needed = list(
        dict.fromkeys(
            [group_col, CLF_TARGET_COL]
            + load_columns(cfg)  # features + every algorithm's auxiliary column
            + baseline_columns(cfg)
        )
    )

    print(f"{'dataset':<22}{'rows':>9}{'cols':>7}  file")
    print("-" * RULE)
    for spec in cfg["data"]["datasets"]:
        path = BASE_DIR / spec["parquet_path"]
        if not path.is_file():
            report.error(f"{spec['name']}: parquet not found: {path}")
            print(f"{spec['name']:<22}{'-':>9}{'-':>7}  {path}  MISSING")
            continue
        try:
            pf = pq.ParquetFile(path)  # footer only -- cheap on large files
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
            report.error(f"{spec['name']}: missing column(s) {missing}")


def preflight_config(cfg: dict, report: Report) -> None:
    """Everything that can be checked about one config before a model is fitted."""
    print("\n[1/3] config")
    n_errors = len(report.errors)
    check_config(cfg, report)
    if len(report.errors) > n_errors:
        print("  invalid — skipping the data checks for this config")
        return
    specs = algorithm_specs(cfg)
    baseline_variants = baseline_specs(cfg)
    n_fits = (
        (len(specs) + len(baseline_variants))
        * cfg["model"]["n_runs"]
        * len(cfg["data"]["datasets"])
    )
    baseline_note = (
        " + score baselines "
        + ", ".join(f"{v['label']} ({v['column']})" for v in baseline_variants)
        if baseline_variants
        else ""
    )
    print(
        f"  {len(cfg['data']['datasets'])} datasets, {len(specs)} algorithms"
        f"{baseline_note}, features={cfg['features']}, "
        f"{cfg['model']['n_runs']} seeds  → {n_fits} fits"
    )
    hybrids = [s for s in specs if s["binoculars"]]
    if hybrids:
        # These see more than the feature set, so say so before anything runs.
        print(
            f"  {len(hybrids)} hybrid algorithm(s) also take an auxiliary column: "
            + ", ".join(f"{s['id']} ({s['binoculars']})" for s in hybrids)
        )
        print(
            f"  every algorithm is filtered on {load_columns(cfg)} so they all "
            f"train and score on the same rows"
        )

    print("\n[2/3] parquet files")
    n_errors = len(report.errors)
    check_files(cfg, report)
    if len(report.errors) > n_errors:
        print("  unusable — skipping the data checks for this config")
        return

    print("\n[3/3] data, dimensions and splits")
    group_col: str = cfg["data"]["group_col"]
    human_label: str = cfg["data"]["human_label"]
    features = list(cfg["features"])
    columns = load_columns(cfg)
    datasets = cfg["data"]["datasets"]

    print(
        f"{'dataset':<22}{'rows':>7}{'dropped':>9}{'dim':>7}{'groups':>8}  classes"
    )
    print("-" * RULE)

    # The feature set is fixed, so the data checks run once (not per algorithm).
    preps: dict = {}
    for spec in datasets:
        try:
            prep = load_prepared(spec, columns, group_col, human_label)
        except Exception as exc:
            report.error(
                f"{spec['name']}: failed to load: {type(exc).__name__}: {exc}"
            )
            continue

        ok = check_dataset(prep, cfg, "features", report)
        dropped = prep.n_class_dropped + prep.n_missing_dropped
        # 'dim' is the shared design matrix only; auxiliary columns are handed to
        # their algorithm separately and never widen it.
        print(
            f"{prep.name:<22}{len(prep.df):>7}{dropped:>9}"
            f"{(feature_only_dim(prep, features) if prep.dims_per_col else 0):>7}"
            f"{(prep.n_groups if len(prep.df) else 0):>8}  "
            f"{prep.class_counts if len(prep.df) else '-'}"
        )
        if ok:
            preps[prep.name] = prep

    # The baselines are filtered on their own score column instead of `features`,
    # so their row set is checked separately -- a partly filled column (the judge
    # only scores the generators it was asked for) means they stand on fewer rows
    # than the trained algorithms, and that has to be visible before the run.
    check_baseline_datasets(cfg, report)

    if len(preps) != len(datasets):
        return  # a dataset already failed; its splits are not meaningful

    # Features must live in one shared space to be transferable across corpora.
    dims = {name: p.feature_dim for name, p in preps.items()}
    if len(set(dims.values())) != 1:
        report.error(
            f"datasets disagree on feature dimensionality: {dims} "
            f"— a model trained on one corpus cannot be applied to another"
        )
        return

    check_splits(preps, cfg, "features", report)


def preflight(cfg_paths: list[Path]) -> tuple[Report, list[tuple[Path, dict]]]:
    """Check every config in the folder. Returns the report and the loaded configs."""
    report = Report()
    loaded: list[tuple[Path, dict]] = []

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

        preflight_config(cfg, report)
        loaded.append((cfg_path, cfg))

    # Each alg_id names an output directory; a repeat would collide.
    report.context = ""
    alg_ids = [cfg.get("alg_id") for _, cfg in loaded if isinstance(cfg, dict)]
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
            f"{GREEN}Preflight OK — every config, dataset, algorithm and split "
            f"checks out.{RESET}"
        )
    print("=" * RULE)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def fit_alg(alg, X: np.ndarray, y: np.ndarray, Z: np.ndarray | None):
    """Fit, passing the auxiliary matrix only to algorithms that declare it.

    Keeps the plain ``fit(X, y)`` signature intact for every algorithm that does
    not set ``uses_aux``; ``Z`` is None for those and never reaches them.
    """
    return alg.fit(X, y, Z) if alg.uses_aux else alg.fit(X, y)


def predict_alg(alg, X: np.ndarray, Z: np.ndarray | None) -> np.ndarray:
    """Predict, mirroring ``fit_alg``'s two signatures."""
    return alg.predict(X, Z) if alg.uses_aux else alg.predict(X)


def evaluate_alg(alg, X: np.ndarray, y: np.ndarray, Z: np.ndarray | None = None) -> dict:
    """Metrics for one algorithm on one evaluation set.

    Same keys as ``run_clf_cross_domain.evaluate``, but the algorithm owns its
    own preprocessing, so there is no external scaler to apply here. ``Z`` holds
    the auxiliary columns for a hybrid algorithm, sliced to the same rows as
    ``X``, and is None for every other algorithm.
    """
    y_pred = np.asarray(predict_alg(alg, X, Z))
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


def run_algorithm(
    spec: dict,
    cfg: dict,
    X_all: dict,
    data: dict,
    names: list[str],
    labels: dict,
    features: list[str],
    out_dir: Path,
    save_models: bool = True,
) -> dict:
    """Full source x target matrix for one algorithm. Returns the saved payload.

    ``X_all`` / ``data`` are loaded once by ``run_config`` and shared across
    algorithms, since every algorithm sees the same fixed feature set.

    ``save_models`` writes one pickled model per (seed, source). The hyper-parameter
    sweep turns it off: it fits thousands of throwaway candidates and only the
    winner is ever reloaded, so keeping them all would cost gigabytes for nothing.
    """
    base_seed: int = cfg["seed"]
    n_runs: int = cfg["model"]["n_runs"]
    test_size: float = cfg["model"].get("test_size", 0.30)
    name, params = spec["name"], spec["params"]

    feature_dim = int(next(iter(X_all.values())).shape[1])

    # The auxiliary column, kept out of X so it can enter the network late. Per
    # algorithm, since two entries may name different columns; one scalar column
    # per corpus, so stacking it here rather than in run_config costs nothing.
    binox_col = spec["binoculars"]
    Z_all = (
        {n: stack_features(df, [binox_col])[0] for n, df in data.items()}
        if binox_col
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

            # Grouped split: a document and its rewrites never straddle the cut.
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=seed
            )
            train_idx, test_idx = next(splitter.split(X_s, y_s, groups_s))
            train_groups = set(groups_s[train_idx])

            # A fresh algorithm per (seed, source); each owns its preprocessing.
            alg = build_algorithm(name, seed, **params)
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

            seed_res[source] = {}
            for target in names:
                if target == source:
                    # In-domain: the corpus's own held-out grouped test split.
                    res = evaluate_alg(
                        alg,
                        X_s[test_idx],
                        y_s[test_idx],
                        Z_s[test_idx] if Z_s is not None else None,
                    )
                    res["excluded"] = 0
                else:
                    df_t = data[target]
                    # Target rows whose group was trained on would be a free hit.
                    keep = ~df_t["group"].isin(train_groups).to_numpy()
                    Z_t = Z_all[target]
                    res = evaluate_alg(
                        alg,
                        X_all[target][keep],
                        df_t.loc[keep, "y"].to_numpy(),
                        Z_t[keep] if Z_t is not None else None,
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
                        f"{YELLOW}  ⚠️  could not save {name} for {source} "
                        f"(seed {seed}): {type(exc).__name__}: {exc}{RESET}"
                    )

        per_seed.append(seed_res)

    # Aggregate over seeds (same shape as the embeddings script's summary).
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

    # Shaped like the embeddings payload so plot_clf_cd_result / summarise_embedding
    # work unchanged. 'embedding' carries the row's display label for the reused
    # comparison code; 'algorithm' / 'params' record what actually ran.
    alg_cfg = dict(cfg)
    alg_cfg["features"] = features
    alg_cfg["binoculars"] = binox_col
    alg_cfg["model"] = {**cfg["model"], "ml_alg": name}
    alg_cfg["alg_id"] = f"{cfg['alg_id']}__{spec['id']}"
    alg_cfg["alg_label"] = f"{cfg['alg_label']} — {spec['label']}"

    payload = {
        "alg_id": alg_cfg["alg_id"],
        "alg_label": alg_cfg["alg_label"],
        "embedding": spec["label"],
        "algorithm": name,
        "params": params,
        # What the config asked for is `params`; this is what the algorithm made
        # of it, defaults filled in. Keeping both means a matrix can be traced
        # back to the exact model without re-deriving any default.
        "resolved_params": resolved_params(spec, base_seed),
        "binoculars": binox_col,
        "feature_dim": feature_dim,
        "dataset_names": names,
        "dataset_labels": labels,
        "classes": BINARY_CLASSES,
        "config": alg_cfg,
        "summary": summary,
        "per_seed": per_seed,
    }
    with open(out_dir / "cross_domain.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def print_ranking(alg_label: str, rows: list[dict]) -> None:
    """Rank the algorithms by transfer accuracy -- the point of the whole run."""
    order = sorted(rows, key=lambda r: r["transfer_mean"], reverse=True)
    print()
    print("=" * RULE)
    print(f"  {alg_label} — ALGORITHMS RANKED BY TRANSFER (off-diagonal) ACCURACY")
    print("=" * RULE)
    print(
        f"{'algorithm':<26}{'in-domain':>12}{'transfer':>11}{'worst':>9}{'gap':>8}"
    )
    print("-" * RULE)
    for r in order:
        mark = "★ " if r.get("baseline") else ("◆ " if r.get("hybrid") else "  ")
        print(
            f"{mark + r['label']:<26}"
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


def run_config(cfg_path: Path, cfg: dict, alg_dir: Path) -> None:
    """Every algorithm of one config, plus its comparison figures."""
    alg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, alg_dir / "config.yaml")
    with open(alg_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    group_col: str = cfg["data"]["group_col"]
    human_label: str = cfg["data"]["human_label"]
    specs_ds: list[dict] = cfg["data"]["datasets"]
    features = list(cfg["features"])
    columns = load_columns(cfg)
    aux = aux_columns(cfg)
    names = [spec["name"] for spec in specs_ds]
    labels = {spec["name"]: spec.get("label", spec["name"]) for spec in specs_ds}
    specs = algorithm_specs(cfg)

    tqdm.write("\n" + "=" * RULE)
    tqdm.write(f"  ALG ID     : {cfg['alg_id']}")
    tqdm.write(f"  ALGORITHMS : {[s['id'] for s in specs]}")
    tqdm.write(f"  FEATURES   : {features}")
    if aux:
        tqdm.write(f"  AUXILIARY  : {aux}  (hybrid entries only)")
    tqdm.write(
        f"  SPLIT      : grouped by {group_col!r}, "
        f"test_size={cfg['model'].get('test_size', 0.30)}, "
        f"{cfg['model']['n_runs']} seeds (base {cfg['seed']})"
    )
    tqdm.write(f"  OUTPUT     : {alg_dir}")
    tqdm.write("=" * RULE)

    # Load and stack the fixed feature set once; every algorithm reuses it. The
    # missing-value filter runs over features + auxiliary columns together, so a
    # row a hybrid algorithm cannot use is absent for the plain ones too and the
    # comparison stays like-for-like.
    data = {}
    for spec in specs_ds:
        prep = load_prepared(spec, columns, group_col, human_label)
        tqdm.write(f"  {spec['name']:<20} {len(prep.df):>6} rows  {prep.class_counts}")
        data[spec["name"]] = prep.df
    X_all = {name: stack_features(df, features)[0] for name, df in data.items()}

    rows: list[dict] = []
    for spec in tqdm(specs, desc="Algorithms", unit="alg", leave=False):
        hybrid_note = f" + {spec['binoculars']}" if spec["binoculars"] else ""
        tqdm.write(f"\n→ {spec['id']}  ({spec['name']}{hybrid_note})")
        # The resolved set, not the config's diff against the defaults: when the
        # params came out of a sweep leaderboard this is the line that confirms
        # the transcription took effect.
        eff = resolved_params(spec, cfg["seed"])
        tqdm.write(
            "   params: "
            + ", ".join(f"{k}={v}" for k, v in sorted(eff.items()))
        )
        out_dir = alg_dir / spec["id"]
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = run_algorithm(
            spec, cfg, X_all, data, names, labels, features, out_dir
        )
        print_matrix(payload["alg_label"], payload["summary"], payload["dataset_names"])

        # Plot as we go: an interrupted run still leaves finished algorithms
        # with their figures.
        plot_clf_cd_result(out_dir)
        row = summarise_embedding(payload)
        if spec["binoculars"]:
            # Flags the row for the ranking and the figures: it saw the shared
            # feature set *and* the score, so it is not on equal footing with the
            # plain algorithms and must not read as though it were.
            row["hybrid"] = True
            row["aux_column"] = spec["binoculars"]
        rows.append(row)

    # Optional zero-shot score baselines (reused from the embeddings script): per
    # enabled family, the source-fit threshold plus a fixed one per configured
    # cutoff. Binoculars reads low ⇒ machine, the LLM judge high ⇒ machine; the
    # variant carries its own direction, so both are run by the same loop.
    for variant in baseline_specs(cfg):
        tqdm.write(f"\n→ {variant['label']}  ({describe_variant(variant)})")
        bl_dir = alg_dir / variant["name"]
        bl_dir.mkdir(parents=True, exist_ok=True)
        payload = run_score_baseline(cfg, bl_dir, variant)
        print_matrix(payload["alg_label"], payload["summary"], payload["dataset_names"])
        plot_clf_cd_result(bl_dir)
        rows.append(summarise_baseline(payload))

    comparison = {
        "alg_id": cfg["alg_id"],
        "alg_label": cfg["alg_label"],
        "dataset_names": names,
        "dataset_labels": labels,
        "classes": BINARY_CLASSES,
        "config": cfg,
        "algorithms": rows,
    }
    with open(alg_dir / "algorithm_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print_ranking(cfg["alg_label"], rows)
    plot_clf_cd_alg_comparison(alg_dir, comparison)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg_paths = collect_configs(CFG_DIR)

    # Every config is checked before any of them trains.
    report, loaded = preflight(cfg_paths)
    print_report(report)
    if not report.ok:
        sys.exit(1)

    experiment_id = "clf_cd_algs_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = RESULTS_DIR / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    for cfg_path, cfg in tqdm(loaded, desc="Configs", unit="cfg"):
        run_config(cfg_path, cfg, exp_dir / cfg["alg_id"])

    print(f"\nDone. Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()
