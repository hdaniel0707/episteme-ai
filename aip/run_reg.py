from datetime import datetime
from pathlib import Path
import shutil
import json
import numpy as np
import joblib
from tqdm import tqdm

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from aip.utils.config_utils import collect_configs, load_config
from aip.utils.parquet_utils import load_parquet
from aip.utils.train_utils import build_features, build_model
from aip.utils.cuda_util import check_cuda
from aip.analyse_exp import run_individual_analysis
from aip.analyse_exp_compare import run_compared_analysis

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results" / "reg"

### INPUTS ###############################
CFG_DIR = BASE_DIR / "cfg" / "train" / "reg" / "hd" / "banan"


def run_experiments() -> str:

    # Generate Eyperiment ID
    experiment_id = "reg_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results fodler
    EXP_DIR = RESULTS_DIR / experiment_id
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    # Load config files paths
    cfg_paths = collect_configs(CFG_DIR)

    for cfg_path in tqdm(cfg_paths, desc="Configs"):

        # Load config file
        cfg = load_config(cfg_path)

        # Log
        print(f"\n{'='*60}")
        print(f"  ALG ID: {cfg['alg_id']}")
        print(f"  ML ALG: {cfg['model']['ml_alg']}")
        print(f"{'='*60}")

        # Create ALG DIR
        ALG_DIR = EXP_DIR / cfg["alg_id"]
        ALG_DIR.mkdir(parents=True, exist_ok=True)

        # Copy CFG
        shutil.copy(cfg_path, ALG_DIR / "config.yaml")
        with open(ALG_DIR / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        # Get data and build features
        base_seed: int = cfg["seed"]
        parquet_path = BASE_DIR / cfg["data"]["parquet_path"]
        df = load_parquet(parquet_path)
        X, y, feature_labels = build_features(df, cfg["features"])

        # Get Params
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

            # Split dataset
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

            # Build model
            model = build_model(ml_alg, base_seed + i)

            # Train model
            model.fit(X_train, y_train)

            # Get results
            if isinstance(model, GradientBoostingRegressor):
                train_losses = [
                    float(mean_squared_error(y_train, p))
                    for p in model.staged_predict(X_train)
                ]
                val_losses = [
                    float(mean_squared_error(y_val, p))
                    for p in model.staged_predict(X_val)
                ]
            else:
                train_losses, val_losses = [], []

            results: dict = {}
            for split, X_s, y_s in [
                ("train", X_train, y_train),
                ("val", X_val, y_val),
                ("test", X_test, y_test),
            ]:
                y_pred = model.predict(X_s)
                results[f"{split}_rmse"] = float(
                    np.sqrt(mean_squared_error(y_s, y_pred))
                )
                results[f"{split}_r2"] = float(r2_score(y_s, y_pred))

            results["train_losses"] = train_losses
            results["val_losses"] = val_losses
            results["y_test"] = y_test.tolist()
            results["y_pred"] = model.predict(X_test).tolist()
            results["feature_importances"] = (
                model.feature_importances_.tolist()
                if isinstance(model, GradientBoostingRegressor)
                else []
            )

            # Save results
            with open(SEED_RES_DIR / "result.json", "w") as f:
                json.dump(results, f, indent=2)

            # Save model
            if model is not None:
                joblib.dump(model, SEED_MODEL_DIR / "model.joblib")
                joblib.dump(scaler, SEED_MODEL_DIR / "scaler.joblib")
    return experiment_id


def main() -> None:
    if check_cuda():
        experiment_id = run_experiments()
        # experiment_id = "reg_20260519_160026"

        run_individual_analysis(RESULTS_DIR, experiment_id)
        run_compared_analysis(RESULTS_DIR, experiment_id)


if __name__ == "__main__":
    main()
