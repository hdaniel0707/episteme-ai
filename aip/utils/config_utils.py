import sys
from pathlib import Path
import yaml

YELLOW = "\033[93m"
RESET = "\033[0m"

REQUIRED_KEYS = {"run_id", "run_label", "seed", "data", "features", "model"}
REQUIRED_DATA_KEYS = {"parquet_path"}
REQUIRED_MODEL_KEYS = {"n_runs", "ml_alg"}
METRIC_KEYS = ["train_rmse", "train_r2", "val_rmse", "val_r2", "test_rmse", "test_r2"]


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
