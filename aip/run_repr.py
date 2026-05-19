from datetime import datetime
from pathlib import Path
import shutil
import json
from tqdm import tqdm

from aip.autoencoder import AERegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from aip.utils.config_utils import collect_configs, load_config
from aip.utils.parquet_utils import load_parquet
from aip.utils.train_utils import build_features, build_model
from aip.utils.cuda_util import check_cuda
from aip.analyse_exp import run_individual_analysis
from aip.analyse_exp_compare import run_compared_analysis

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results" / "repr"

### INPUTS ###############################
CFG_DIR = BASE_DIR / "cfg" / "train" / "repr" / "hd" / "alma"


def run_experiments() -> str:

    experiment_id = "repr_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    EXP_DIR = RESULTS_DIR / experiment_id
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    cfg_paths = collect_configs(CFG_DIR)

    for cfg_path in tqdm(cfg_paths, desc="Configs"):

        cfg = load_config(cfg_path)

        print(f"\n{'='*60}")
        print(f"  ALG ID: {cfg['alg_id']}")
        print(f"  ML ALG: {cfg['model']['ml_alg']}")
        print(f"{'='*60}")

        ALG_DIR = EXP_DIR / cfg["alg_id"]
        ALG_DIR.mkdir(parents=True, exist_ok=True)

        shutil.copy(cfg_path, ALG_DIR / "config.yaml")
        with open(ALG_DIR / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        base_seed: int = cfg["seed"]
        parquet_path = BASE_DIR / cfg["data"]["parquet_path"]
        df = load_parquet(parquet_path)
        X, y, feature_labels = build_features(df, cfg["features"])

        m = cfg["model"]
        n_runs: int = m["n_runs"]
        ml_alg: str = m["ml_alg"]

        for i in tqdm(range(n_runs), desc="Seeds", leave=False):
            SEED_DIR = ALG_DIR / str(i)
            SEED_DIR.mkdir(parents=True, exist_ok=True)
            SEED_MODEL_DIR = SEED_DIR / "model"
            SEED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            SEED_RES_DIR = SEED_DIR / "results"
            SEED_RES_DIR.mkdir(parents=True, exist_ok=True)

            X_train, X_tmp, y_train, y_tmp = train_test_split(
                X, y, test_size=0.30, random_state=base_seed + i
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_tmp, y_tmp, test_size=0.50, random_state=base_seed + i
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            model: AERegressor = build_model(ml_alg, base_seed + i)
            model.fit(X_train, y_train, X_val=X_val)

            results: dict = {
                "train_losses": model.train_losses_,
                "val_losses": model.val_losses_,
            }

            with open(SEED_RES_DIR / "result.json", "w") as f:
                json.dump(results, f, indent=2)

            import joblib

            joblib.dump(model, SEED_MODEL_DIR / "model.joblib")
            joblib.dump(scaler, SEED_MODEL_DIR / "scaler.joblib")

    return experiment_id


def main() -> None:
    if check_cuda():
        experiment_id = run_experiments()
        run_individual_analysis(RESULTS_DIR, experiment_id)
        run_compared_analysis(RESULTS_DIR, experiment_id)


if __name__ == "__main__":
    main()
