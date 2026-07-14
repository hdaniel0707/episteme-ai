import sys
from pathlib import Path
import yaml

YELLOW = "\033[93m"
RESET = "\033[0m"

REQUIRED_KEYS = {"alg_id", "alg_label", "seed", "data", "features", "model"}
REQUIRED_DATA_KEYS = {"parquet_path"}
REQUIRED_MODEL_KEYS = {"n_runs", "ml_alg"}
METRIC_KEYS = ["train_rmse", "train_r2", "val_rmse", "val_r2", "test_rmse", "test_r2"]

# Cross-domain configs train on one corpus and test on the others, so they carry
# a list of datasets instead of a single parquet_path.
REQUIRED_CD_DATA_KEYS = {"datasets", "group_col", "human_label"}
REQUIRED_CD_DATASET_KEYS = {"name", "parquet_path"}


def collect_configs(cfg_dir: Path) -> list[Path]:
    if not cfg_dir.is_dir():
        print(
            f"{YELLOW}ERROR: --cfg-dir '{cfg_dir}' is not a directory.{RESET}",
            file=sys.stderr,
        )
        sys.exit(1)
    paths = sorted(cfg_dir.glob("*.yaml")) + sorted(cfg_dir.glob("*.yml"))
    if not paths:
        print(
            f"{YELLOW}ERROR: No YAML files found in '{cfg_dir}'.{RESET}",
            file=sys.stderr,
        )
        sys.exit(1)
    return list(dict.fromkeys(paths))


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    missing = REQUIRED_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"{YELLOW}{path}: missing top-level keys: {missing}{RESET}")

    missing_data = REQUIRED_DATA_KEYS - cfg["data"].keys()
    if missing_data:
        raise ValueError(f"{YELLOW}{path}: missing data keys: {missing_data}{RESET}")

    missing_model = REQUIRED_MODEL_KEYS - cfg["model"].keys()
    if missing_model:
        raise ValueError(f"{YELLOW}{path}: missing model keys: {missing_model}{RESET}")

    return cfg


def load_cd_config(path: Path) -> dict:
    """Load and validate a cross-domain classification config.

    Same shape as a `clf` config, except `data` holds a list of `datasets` (each
    trained on in turn and evaluated against all the others) rather than a single
    `parquet_path`, plus the `group_col` and `human_label` needed to make the
    corpora comparable.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    missing = REQUIRED_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"{YELLOW}{path}: missing top-level keys: {missing}{RESET}")

    missing_data = REQUIRED_CD_DATA_KEYS - cfg["data"].keys()
    if missing_data:
        raise ValueError(f"{YELLOW}{path}: missing data keys: {missing_data}{RESET}")

    datasets = cfg["data"]["datasets"]
    if len(datasets) < 2:
        raise ValueError(
            f"{YELLOW}{path}: cross-domain needs at least 2 datasets, "
            f"got {len(datasets)}.{RESET}"
        )
    for ds in datasets:
        missing_ds = REQUIRED_CD_DATASET_KEYS - ds.keys()
        if missing_ds:
            raise ValueError(
                f"{YELLOW}{path}: dataset {ds!r}: missing keys: {missing_ds}{RESET}"
            )
    names = [ds["name"] for ds in datasets]
    if len(set(names)) != len(names):
        raise ValueError(f"{YELLOW}{path}: dataset names must be unique: {names}{RESET}")

    missing_model = REQUIRED_MODEL_KEYS - cfg["model"].keys()
    if missing_model:
        raise ValueError(f"{YELLOW}{path}: missing model keys: {missing_model}{RESET}")

    return cfg
