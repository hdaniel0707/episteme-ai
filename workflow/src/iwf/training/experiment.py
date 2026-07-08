"""Multi-seed experiment runner for the embedding/metadata workflows.

Trains an impact head over ``n_runs`` model seeds on the SHARED split (split fixed,
only the model seed varies), evaluates train/val/test, and writes every seed directly
into the per-workflow folder — the same layout the text-transformer trainer uses:

    results/<wf_type>_N/seed_<i>/model/{model.joblib, scaler.joblib}
    results/<wf_type>_N/seed_<i>/result.json

``result.json`` root = flat test metrics (for aggregation) + per-split dicts with
y_true/y_pred + training history + flat all-rows prediction arrays (for the UMAP plots).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from workflow.src.iwf.config import Settings
from workflow.src.iwf.data.dataset import split_masks
from workflow.src.iwf.evaluation.metrics import classification_metrics, regression_metrics
from workflow.src.iwf.features.feature_builder import NeutralScaler, build_xy
from workflow.src.iwf.models.classifier import TorchMLPClassifier
from workflow.src.iwf.models.regressor import TorchMLPRegressor
from workflow.src.iwf.parallel import run_jobs, torch_cuda_available


def _add_class_ids(df: pd.DataFrame, class_names) -> pd.DataFrame:
    """Map class_label -> fixed integer ids in the configured ascending-impact order."""
    mapping = {name: i for i, name in enumerate(class_names)}
    df = df.copy()
    df["class_label_id"] = df["class_label"].map(mapping)
    return df


def _train_one_seed(payload: dict) -> dict:
    """Train + evaluate + persist ONE seed. Module-level so it can run in a worker process.

    Shared by the sequential and parallel paths, so both produce identical per-seed
    output (each seed is fully determined by base_seed + i, independent of the others)."""
    seed_dir = Path(payload["out_dir"]) / f"seed_{payload['i']}"
    i = payload["i"]
    task = payload["task"]
    settings: Settings = payload["settings"]
    Xs = payload["Xs"]
    y = payload["y"]
    masks = payload["masks"]
    scaler = payload["scaler"]
    class_names = list(payload["class_names"])

    (seed_dir / "model").mkdir(parents=True, exist_ok=True)

    common = dict(hidden=settings.hidden, dropout=settings.dropout, lr=settings.lr,
                  weight_decay=settings.weight_decay, max_epochs=settings.max_epochs,
                  patience=settings.patience, batch_size=settings.batch_size,
                  seed=settings.model_seed + i, activation=settings.activation,
                  arch=settings.model_arch)
    if task == "clf":
        model = TorchMLPClassifier(n_classes=len(class_names),
                                   class_weight=settings.class_weight,
                                   effective_beta=settings.effective_beta,
                                   loss=settings.loss, focal_gamma=settings.focal_gamma,
                                   label_smoothing=settings.label_smoothing, **common)
    else:
        model = TorchMLPRegressor(**common)

    model.fit(Xs[masks["train"]], y[masks["train"]],
              Xs[masks["val"]], y[masks["val"]])

    result: dict = {"seed": i, "history": model.history_,
                    "best_val_metric": model.history_.get("best_val_metric")}
    for split in ("train", "val", "test"):
        m = masks[split]
        y_true, y_pred = y[m], model.predict(Xs[m])
        metrics_d = (classification_metrics(y_true, y_pred, class_names) if task == "clf"
                     else regression_metrics(y_true, y_pred))
        metrics_d["y_true"] = y_true.tolist()
        metrics_d["y_pred"] = y_pred.tolist()
        result[split] = metrics_d
        if split == "test":
            # flat copies at the root: scalar metrics for aggregation, per_class_f1 /
            # confusion for the class plots
            result.update({k: v for k, v in metrics_d.items()
                           if k not in ("y_true", "y_pred")})
            if task == "clf":
                result["proba_test"] = model.predict_proba(Xs[m]).tolist()

    # flat all-rows arrays so UMAP plots align predictions to the parquet without a reload
    result["y_all"] = y.tolist()
    result["y_pred_all"] = model.predict(Xs).tolist()
    result["split_all"] = payload.get("split_all", [])

    (seed_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Move the net to CPU before pickling so the saved model loads on any machine.
    if getattr(model, "net_", None) is not None:
        model.net_.to("cpu")
    joblib.dump(model, seed_dir / "model" / "model.joblib")
    joblib.dump(scaler, seed_dir / "model" / "scaler.joblib")
    test_flat = {k: round(v, 4) for k, v in result["test"].items()
                 if isinstance(v, (int, float))}
    print(f"  [{seed_dir.parent.name}] seed {i}: test={test_flat}")
    return test_flat


def run_experiment(
    df: pd.DataFrame,
    task: str,                 # "clf" | "reg"
    settings: Settings,
    scalar_cols: list[str],
    out_dir: Path,
    label: str = "",
    bypass_guard: bool = False,
) -> Path:
    """Train ``settings.n_runs`` seeds and write them under ``out_dir/seed_<i>/``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if task == "clf":
        df = _add_class_ids(df, settings.class_names)
        df = df[df["class_label_id"].notna()].reset_index(drop=True)
        df["class_label_id"] = df["class_label_id"].astype(int)

    # Features are identical across seeds; only the model seed varies.
    X, y, names, n_value_cols = build_xy(df, task, scalar_cols, bypass_guard=bypass_guard)
    if y is None:
        raise ValueError(f"No target/label found for task={task!r}.")
    masks = split_masks(df)

    (out_dir / "experiment.json").write_text(json.dumps({
        "label": label, "task": task, "n_features": len(names),
        "feature_names_head": names[:8], "scalar_cols": scalar_cols,
        "bypass_leakage_guard": bypass_guard,
    }, indent=2), encoding="utf-8")

    # The scaler is fit on TRAIN only and is seed-independent, so fit it once and share it.
    scaler = NeutralScaler(n_value_cols).fit(X[masks["train"]])
    Xs = scaler.transform(X)

    split_all = df["split"].tolist() if "split" in df.columns else []
    payloads = [
        {"out_dir": str(out_dir), "i": i, "task": task,
         "settings": settings, "Xs": Xs, "y": y, "masks": masks,
         "scaler": scaler, "class_names": tuple(settings.class_names),
         "split_all": split_all}
        for i in range(settings.n_runs)
    ]
    # MLP/ResMLP heads are tiny on the GPU; the FT-Transformer is O(features^2), so reserve
    # more VRAM per worker for it (fewer share the card) to avoid out-of-memory.
    vram = 2500 if str(settings.model_arch).lower() == "ft_transformer" else 900
    run_jobs(_train_one_seed, payloads, use_cuda=torch_cuda_available(),
             vram_per_worker_mib=vram, label=f"{out_dir.name} seeds")

    return out_dir
