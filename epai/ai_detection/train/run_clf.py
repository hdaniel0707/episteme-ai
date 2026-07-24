from datetime import datetime
from pathlib import Path
import shutil
import json
import joblib
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from epai.utils.config_utils import RESET, YELLOW, collect_configs, load_config
from epai.utils.parquet_utils import load_parquet
from epai.utils.train_utils import (
    CLF_TARGET_COL,
    build_clf_features,
    build_clf_model,
    drop_missing_rows,
    split_train_val_test,
)
from epai.utils.visualization_utils import plot_clf_results

BASE_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "clf"

### INPUTS ###############################
CFG_DIR = BASE_DIR / "cfg" / "ai_detection" / "train" / "clf" / "hd2"
##########################################


def run_classification() -> str:
    experiment_id = "clf_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    EXP_DIR = RESULTS_DIR / experiment_id
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    cfg_paths = collect_configs(CFG_DIR)

    for cfg_path in tqdm(cfg_paths, desc="Configs"):
        cfg = load_config(cfg_path)

        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"  ALG ID: {cfg['alg_id']}")
        tqdm.write(f"  ML ALG: {cfg['model']['ml_alg']}")
        tqdm.write(f"  {'-'*30}")

        ALG_DIR = EXP_DIR / cfg["alg_id"]
        ALG_DIR.mkdir(parents=True, exist_ok=True)

        shutil.copy(cfg_path, ALG_DIR / "config.yaml")
        with open(ALG_DIR / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        base_seed: int = cfg["seed"]
        parquet_path = BASE_DIR / cfg["data"]["parquet_path"]
        df = load_parquet(parquet_path)

        # Rows sharing a group_col value (for the Ghostbuster corpora: the human
        # original and every LLM rewrite of the same source document share one
        # `id`) must land in the same split, or the test set contains rewrites of
        # documents the model already saw. Set data.group_col: null to opt out.
        group_col = cfg["data"].get("group_col")
        required = list(cfg["features"]) + [CLF_TARGET_COL]
        if group_col:
            required.append(group_col)
        df = drop_missing_rows(df, required, group_col=group_col)

        drop_classes = cfg["data"].get("drop_classes", [])
        X, y, le, groups, feature_labels = build_clf_features(
            df, cfg["features"], drop_classes, group_col
        )

        tqdm.write(f"  {'-'*30}")
        tqdm.write(f"  FEATURES: {cfg['features']} ({X.shape[1]} dims)")
        tqdm.write(f"  CLASSES: {le.classes_.tolist()}")
        if groups is None:
            tqdm.write(
                f"{YELLOW}  SPLIT: random (ungrouped) — paraphrase pairs may "
                f"straddle train/test{RESET}"
            )
        else:
            tqdm.write(
                f"  SPLIT: grouped by {group_col!r} "
                f"({len(set(groups))} groups over {len(y)} rows)"
            )
        tqdm.write(f"{'='*60}")

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

            X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
                X, y, seed=base_seed + i, groups=groups
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            model = build_clf_model(ml_alg, base_seed + i)
            model.fit(X_train, y_train)

            results: dict = {
                "split_scheme": f"grouped:{group_col}" if groups is not None else "random",
                "split_sizes": {
                    "train": int(len(y_train)),
                    "val": int(len(y_val)),
                    "test": int(len(y_test)),
                },
                # The grouped split cannot stratify — check these stay balanced.
                "class_balance": {
                    split: {
                        le.classes_[c]: int(n)
                        for c, n in zip(*np.unique(y_s, return_counts=True))
                    }
                    for split, y_s in [
                        ("train", y_train),
                        ("val", y_val),
                        ("test", y_test),
                    ]
                },
            }
            for split, X_s, y_s in [
                ("train", X_train, y_train),
                ("val", X_val, y_val),
                ("test", X_test, y_test),
            ]:
                y_pred = model.predict(X_s)
                results[f"{split}_accuracy"] = float(accuracy_score(y_s, y_pred))
                results[f"{split}_report"] = classification_report(
                    y_s, y_pred, target_names=le.classes_.tolist(), output_dict=True, zero_division=0
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
    plot_clf_results(RESULTS_DIR / experiment_id, show_random=False)

    # plot_clf_results(RESULTS_DIR / "clf_20260519_155000", show_random=False)


if __name__ == "__main__":
    main()
