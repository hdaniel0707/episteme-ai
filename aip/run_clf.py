from datetime import datetime
from pathlib import Path
import json
import numpy as np
import joblib
from tqdm import tqdm

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix as compute_cm,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from aip.utils.parquet_utils import load_parquet

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"

### INPUTS ###############################
PARQUET_PATH = BASE_DIR / "data" / "paquet" / "0515_A_500_AI_260.parquet"
N_RUNS = 3
BASE_SEED = 42

# Classes to exclude from classification (e.g. ["gpt:aggressive_phrasing"])
DROP_CLASSES: list[str] = ["gpt-4o-mini:aggressive"]

MODELS = {
    "gradient_boosting": lambda seed: GradientBoostingClassifier(random_state=seed),
    "logistic": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "svm": lambda seed: SVC(random_state=seed),
    "mlp": lambda seed: MLPClassifier(
        hidden_layer_sizes=(516, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=seed,
    ),
}
##########################################


def run_classification() -> str:
    experiment_id = "clf_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    EXP_DIR = RESULTS_DIR / experiment_id
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    df = load_parquet(PARQUET_PATH)

    if DROP_CLASSES:
        df = df[~df["generated_by"].isin(DROP_CLASSES)].reset_index(drop=True)
        print(f"Dropped classes: {DROP_CLASSES}")

    X = np.array(df["emb_specter2"].tolist(), dtype=np.float32)

    le = LabelEncoder()
    y = le.fit_transform(df["generated_by"].tolist())
    print(f"Classes: {le.classes_.tolist()}")

    # Save label mapping
    with open(EXP_DIR / "label_mapping.json", "w") as f:
        json.dump({i: c for i, c in enumerate(le.classes_.tolist())}, f, indent=2)

    for model_name, model_factory in tqdm(MODELS.items(), desc="Models"):
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*60}")

        MODEL_DIR = EXP_DIR / model_name
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(N_RUNS), desc="Seeds", leave=False):
            SEED_DIR = MODEL_DIR / str(i)
            SEED_MODEL_DIR = SEED_DIR / "model"
            SEED_RES_DIR = SEED_DIR / "results"
            SEED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            SEED_RES_DIR.mkdir(parents=True, exist_ok=True)

            X_train, X_tmp, y_train, y_tmp = train_test_split(
                X, y, test_size=0.30, random_state=BASE_SEED + i, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_tmp, y_tmp, test_size=0.50, random_state=BASE_SEED + i, stratify=y_tmp
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            model = model_factory(BASE_SEED + i)
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


def plot_results(experiment_id: str) -> None:
    import matplotlib.pyplot as plt

    EXP_DIR = RESULTS_DIR / experiment_id

    with open(EXP_DIR / "label_mapping.json") as f:
        label_mapping = json.load(f)
    class_names = [label_mapping[str(i)] for i in range(len(label_mapping))]
    n_classes = len(class_names)

    model_names = sorted([d.name for d in EXP_DIR.iterdir() if d.is_dir()])
    data: dict = {}
    for model_name in model_names:
        model_dir = EXP_DIR / model_name
        seeds = sorted([d.name for d in model_dir.iterdir() if d.is_dir()], key=int)
        data[model_name] = []
        for seed in seeds:
            with open(model_dir / seed / "results" / "result.json") as f:
                data[model_name].append(json.load(f))

    splits = ["train", "val", "test"]
    n_models = len(model_names)
    n_splits = len(splits)

    def get_y(result, split):
        if f"y_{split}" in result and f"y_{split}_pred" in result:
            return result[f"y_{split}"], result[f"y_{split}_pred"]
        # backward compat: old format saved "y_test" + "y_pred"
        if split == "test" and "y_test" in result and "y_pred" in result:
            return result["y_test"], result["y_pred"]
        return None, None

    # ── Figure 1: Accuracy ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, n_models * 2.5), 5))
    x = np.arange(n_models)
    w = 0.25
    for j, (split, color) in enumerate(
        zip(splits, ["steelblue", "orange", "seagreen"])
    ):
        means, stds = [], []
        for mn in model_names:
            accs = [r[f"{split}_accuracy"] for r in data[mn]]
            means.append(float(np.mean(accs)))
            stds.append(float(np.std(accs)))
        ax.bar(x + j * w, means, w, label=split, color=color, yerr=stds, capsize=4)
    ax.axhline(
        1 / n_classes, linestyle="--", color="red", linewidth=1.2, label="random"
    )
    ax.set_xticks(x + w)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Model and Split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(EXP_DIR / "accuracy.png", dpi=150)

    # ── Figure 2: Confusion matrices ──────────────────────────────────────
    tick_fs = max(6, 10 - n_classes // 3)
    fig, axes = plt.subplots(n_models, n_splits, figsize=(5 * n_splits, 4.5 * n_models))
    axes = np.array(axes).reshape(n_models, n_splits)
    for i, mn in enumerate(model_names):
        for j, split in enumerate(splits):
            ax = axes[i, j]
            cms = []
            for r in data[mn]:
                yt, yp = get_y(r, split)
                if yt is not None and yp is not None:
                    cms.append(compute_cm(yt, yp))
            if not cms:
                ax.text(
                    0.5,
                    0.5,
                    "predictions\nnot saved",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{mn} – {split}")
                continue
            avg_cm = np.mean(cms, axis=0)
            norm_cm = avg_cm / avg_cm.sum(axis=1, keepdims=True)
            im = ax.imshow(norm_cm, vmin=0, vmax=1, cmap="Blues")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=tick_fs)
            ax.set_yticklabels(class_names, fontsize=tick_fs)
            for r_idx in range(n_classes):
                for c_idx in range(n_classes):
                    ax.text(
                        c_idx,
                        r_idx,
                        f"{avg_cm[r_idx, c_idx]:.0f}",
                        ha="center",
                        va="center",
                        fontsize=tick_fs,
                        color="white" if norm_cm[r_idx, c_idx] > 0.5 else "black",
                    )
            ax.set_title(f"{mn} – {split}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "confusion_matrices.png", dpi=150)

    # ── Figure 3: Per-class F1 ────────────────────────────────────────────
    fig, axes = plt.subplots(
        n_models, n_splits, figsize=(max(6, n_classes) * n_splits * 0.6, 4 * n_models)
    )
    axes = np.array(axes).reshape(n_models, n_splits)
    for i, mn in enumerate(model_names):
        for j, split in enumerate(splits):
            ax = axes[i, j]
            f1s = {c: [] for c in class_names}
            for r in data[mn]:
                for c in class_names:
                    if c in r[f"{split}_report"]:
                        f1s[c].append(r[f"{split}_report"][c]["f1-score"])
            means = [float(np.mean(f1s[c])) if f1s[c] else 0.0 for c in class_names]
            stds = [float(np.std(f1s[c])) if f1s[c] else 0.0 for c in class_names]
            ax.bar(range(n_classes), means, yerr=stds, capsize=4, color="steelblue")
            ax.axhline(
                1 / n_classes,
                linestyle="--",
                color="red",
                linewidth=1.2,
                label="random",
            )
            ax.set_xticks(range(n_classes))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=tick_fs)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("F1 Score")
            ax.set_title(f"{mn} – {split}")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(EXP_DIR / "f1_scores.png", dpi=150)

    plt.show()
    print(f"Plots saved to: {EXP_DIR}")


def main() -> None:
    experiment_id = run_classification()
    print(f"\nDone. Results saved to: {RESULTS_DIR / experiment_id}")
    plot_results(experiment_id)

    # plot_results("clf_20260518_223639")


if __name__ == "__main__":
    main()
