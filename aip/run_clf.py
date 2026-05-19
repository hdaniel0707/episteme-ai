from datetime import datetime
from pathlib import Path
import shutil
import json
import joblib
from tqdm import tqdm

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from aip.utils.config_utils import collect_configs, load_config
from aip.utils.parquet_utils import load_parquet
from aip.utils.train_utils import build_clf_features, build_clf_model
from aip.utils.visualization_utils import plot_clf_results

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results" / "clf"

### INPUTS ###############################
CFG_DIR = BASE_DIR / "cfg" / "train" / "clf" / "hd" / "alma"
##########################################


def run_classification() -> str:
    experiment_id = "clf_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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

        drop_classes = cfg["data"].get("drop_classes", [])
        X, y, le = build_clf_features(df, cfg["features"], drop_classes)

        print(f"Classes: {le.classes_.tolist()}")
        with open(ALG_DIR / "label_mapping.json", "w") as f:
            json.dump({i: c for i, c in enumerate(le.classes_.tolist())}, f, indent=2)

        n_runs: int = cfg["model"]["n_runs"]
        ml_alg: str = cfg["model"]["ml_alg"]

        for i in tqdm(range(n_runs), desc="Seeds", leave=False):
            SEED_DIR = ALG_DIR / str(i)
            SEED_MODEL_DIR = SEED_DIR / "model"
            SEED_RES_DIR = SEED_DIR / "results"
            SEED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            SEED_RES_DIR.mkdir(parents=True, exist_ok=True)

            X_train, X_tmp, y_train, y_tmp = train_test_split(
                X, y, test_size=0.30, random_state=base_seed + i, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_tmp, y_tmp, test_size=0.50, random_state=base_seed + i, stratify=y_tmp
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            model = build_clf_model(ml_alg, base_seed + i)
            model.fit(X_train, y_train)

            results: dict = {}
            for split, X_s, y_s in [
                ("train", X_train, y_train),
                ("val", X_val, y_val),
                ("test", X_test, y_test),
            ]:
                y_pred = model.predict(X_s)
                results[f"{split}_accuracy"] = float(accuracy_score(y_s, y_pred))
                results[f"{split}_report"] = classification_report(
                    y_s, y_pred, target_names=le.classes_.tolist(), output_dict=True
                )
                results[f"y_{split}"] = y_s.tolist()
                results[f"y_{split}_pred"] = y_pred.tolist()

            with open(SEED_RES_DIR / "result.json", "w") as f:
                json.dump(results, f, indent=2)

            joblib.dump(model, SEED_MODEL_DIR / "model.joblib")
            joblib.dump(scaler, SEED_MODEL_DIR / "scaler.joblib")
            joblib.dump(le, SEED_MODEL_DIR / "label_encoder.joblib")

    return experiment_id


def main() -> None:
    experiment_id = run_classification()
    print(f"\nDone. Results saved to: {RESULTS_DIR / experiment_id}")
    plot_clf_results(RESULTS_DIR / experiment_id)


if __name__ == "__main__":
    main()
