"""Cross-domain transfer control for the human-vs-machine classifier.

The in-domain numbers (~97-99%) survive a grouped split, a permuted-label
control and a length-only baseline, so the pipeline is sound and the signal is
real. The open question is what the signal *is*: has the classifier learned
"machine-generated-ness", or has it learned "this corpus, written by this
particular generator"? Those two are indistinguishable in-domain and come apart
under transfer.

Each config lists several datasets. Every dataset is trained on in turn and
evaluated on all of them, producing a source x target accuracy matrix:

    diagonal      -> in-domain, held-out grouped test split (the usual number)
    off-diagonal  -> transfer: never saw this corpus / this generator

A classifier that has learned a general notion of machine text keeps a healthy
off-diagonal. One that has learned corpus artifacts collapses towards 50%.

Two things this controls for that a naive transfer eval gets wrong:

  * Labels are binarised to human vs machine. The corpora do not share class
    names (`gpt` in the Ghostbuster sets, `gpt_plain` in the gpt54mini ones), so
    the raw label spaces are not comparable.
  * `<corpus>` and `<corpus>_gpt54mini` are built from the *same* human source
    documents and share their ids. Evaluating one on the other would score the
    model on human rows it trained on. Target rows whose group appears in the
    source's training split are therefore excluded, and the count is reported.

A corpus is a ``CorpusSpec`` -- a parquet plus an optional ``domain`` -- so the
six datasets of a full matrix are six slices of two combined files rather than
six separate parquets. See ``epai/utils/corpus_utils.py``.

Run:
    uv run python epai/ai_detection/train/run_clf_cross_domain.py
    uv run python epai/ai_detection/train/run_clf_cross_domain.py \
        --cfg cfg/ai_detection/train/clf_cd/gpt35t_luar_crud.yaml
    uv run python epai/ai_detection/train/run_clf_cross_domain.py \
        --cfg-dir cfg/ai_detection/train/clf_cd --no-plots
"""

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from epai.utils.config_utils import (
    add_config_args,
    load_cd_config,
    rel,
    resolve_configs,
)
from epai.utils.corpus_utils import (
    CorpusSpec,
    check_corpus_files,
    check_dataset_specs,
    dataset_specs,
    read_corpus,
)
from epai.utils.preflight_utils import (
    RULE,
    Report,
    check_identity,
    check_model_block,
    load_yaml,
    print_report,
)
from epai.utils.train_utils import (
    CLF_TARGET_COL,
    build_clf_model,
    drop_missing_rows,
    stack_features,
)
from epai.utils.visualization_utils import plot_clf_cd_results

BASE_DIR = Path(__file__).parent.parent.parent.parent
DEFAULT_CFG_DIR = BASE_DIR / "cfg" / "ai_detection" / "train" / "clf_cd" / "ghostbuster"
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "clf_cd"

# Binarised target: human -> 0, any generator -> 1.
BINARY_CLASSES = ["human", "machine"]


def load_dataset(
    spec: CorpusSpec, features: list[str], group_col: str, human_label: str
) -> pd.DataFrame:
    """Load one corpus, drop unusable rows, and binarise the target.

    The domain filter and drop_classes are applied by ``read_corpus`` first, so
    a row this run will not use cannot take its group down in the missing-value
    drop below.
    """
    load = read_corpus(spec)
    df = drop_missing_rows(
        load.df, features + [CLF_TARGET_COL, group_col], group_col=group_col
    )

    # The corpora disagree on generator names, so only this binary view is
    # comparable across them.
    df["y"] = (df[CLF_TARGET_COL] != human_label).astype(int)

    counts = df[CLF_TARGET_COL].value_counts().to_dict()
    tqdm.write(f"  {spec.name:<17} {len(df):>5} rows  {counts}  ← {spec.source}")
    return df


def evaluate(model, scaler, X: np.ndarray, y: np.ndarray) -> dict:
    y_pred = model.predict(scaler.transform(X))
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "confusion": confusion_matrix(y, y_pred, labels=range(len(BINARY_CLASSES))).tolist(),
        "n": int(len(y)),
        "class_balance": {
            BINARY_CLASSES[c]: int(n) for c, n in zip(*np.unique(y, return_counts=True))
        },
    }


def preflight(cfg_paths: list[Path]) -> tuple[Report, list[tuple[Path, dict]]]:
    """Check every config before the first fit: shape, files, columns, domains."""
    report = Report()
    loaded: list[tuple[Path, dict]] = []

    print("=" * RULE)
    print(f"PREFLIGHT — {len(cfg_paths)} config(s)")
    print("=" * RULE)

    for cfg_path in cfg_paths:
        report.context = cfg_path.name
        print(f"\n{cfg_path}")
        cfg = load_yaml(cfg_path, report)
        if cfg is None:
            continue

        n_errors = len(report.errors)
        check_identity(cfg, report)
        if "features" not in cfg:
            report.error("missing top-level key 'features'")
        elif not isinstance(cfg["features"], (list, tuple)) or not cfg["features"]:
            report.error("'features' must be a non-empty list of column names")
        check_dataset_specs(cfg, report)
        check_model_block(cfg, report)
        if len(report.errors) > n_errors:
            print("  invalid — skipping the file checks for this config")
            continue

        specs = dataset_specs(cfg, BASE_DIR)
        needed = list(cfg["features"]) + [
            CLF_TARGET_COL, cfg["data"]["group_col"]
        ]
        check_corpus_files(specs, needed, report, rule=RULE)
        loaded.append((cfg_path, cfg))

    report.context = ""
    return report, loaded


def run_cross_domain(
    cfg_paths: list[Path], results_dir: Path, experiment_id: str
) -> str:
    EXP_DIR = results_dir / experiment_id
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    for cfg_path in tqdm(cfg_paths, desc="Configs"):
        cfg = load_cd_config(cfg_path)

        base_seed: int = cfg["seed"]
        features: list[str] = cfg["features"]
        group_col: str = cfg["data"]["group_col"]
        human_label: str = cfg["data"]["human_label"]
        specs: list[CorpusSpec] = dataset_specs(cfg, BASE_DIR)
        ml_alg: str = cfg["model"]["ml_alg"]
        n_runs: int = cfg["model"]["n_runs"]
        test_size: float = cfg["model"].get("test_size", 0.30)

        ALG_DIR = EXP_DIR / cfg["alg_id"]
        ALG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg_path, ALG_DIR / "config.yaml")
        with open(ALG_DIR / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        tqdm.write(f"\n{'='*72}")
        tqdm.write(f"  ALG ID: {cfg['alg_id']}")
        tqdm.write(f"  ML ALG: {ml_alg}  ({n_runs} seeds, base seed {base_seed})")
        tqdm.write(f"  FEATURES: {features}")
        tqdm.write(f"  SPLIT: grouped by {group_col!r}, test_size={test_size}")
        tqdm.write(f"  {'-'*30}")
        tqdm.write(f"{'='*72}")

        data = {
            spec.name: load_dataset(spec, features, group_col, human_label)
            for spec in specs
        }
        names = [spec.name for spec in specs]
        labels = {spec.name: spec.label for spec in specs}

        # Features must live in one shared space to be transferable: the same
        # columns, in the same order, for every corpus.
        X_all = {name: stack_features(df, features)[0] for name, df in data.items()}
        dims = {name: X.shape[1] for name, X in X_all.items()}
        if len(set(dims.values())) != 1:
            raise ValueError(f"Datasets disagree on feature dimensionality: {dims}")

        # per_seed[i][source][target] -> metrics
        per_seed: list[dict] = []

        for i in tqdm(range(n_runs), desc="Seeds", leave=False):
            seed = base_seed + i
            SEED_MODEL_DIR = ALG_DIR / str(i) / "model"
            SEED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            seed_res: dict[str, dict] = {}

            for source in names:
                df_s = data[source]
                X_s, y_s = X_all[source], df_s["y"].to_numpy()
                groups_s = df_s[group_col].to_numpy()

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

                joblib.dump(model, SEED_MODEL_DIR / f"{source}_model.joblib")
                joblib.dump(scaler, SEED_MODEL_DIR / f"{source}_scaler.joblib")

                seed_res[source] = {}
                for target in names:
                    if target == source:
                        # In-domain: the corpus's own held-out grouped test split.
                        res = evaluate(model, scaler, X_s[test_idx], y_s[test_idx])
                        res["excluded"] = 0
                        res["train_n"] = int(len(train_idx))
                    else:
                        df_t = data[target]
                        # `essay` and `essay_gpt54mini` share their human documents
                        # (same source files, same ids). Any target row whose group
                        # was trained on would be a free hit, so drop it.
                        keep = ~df_t[group_col].isin(train_groups).to_numpy()
                        n_excluded = int((~keep).sum())
                        if not keep.any():
                            raise ValueError(
                                f"{source} -> {target}: every target group was seen "
                                f"in training; nothing left to evaluate on."
                            )
                        res = evaluate(
                            model,
                            scaler,
                            X_all[target][keep],
                            df_t.loc[keep, "y"].to_numpy(),
                        )
                        res["excluded"] = n_excluded
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

        with open(ALG_DIR / "cross_domain.json", "w") as f:
            json.dump(
                {
                    "alg_id": cfg["alg_id"],
                    "alg_label": cfg["alg_label"],
                    "dataset_names": names,
                    "dataset_labels": labels,
                    "classes": BINARY_CLASSES,
                    "config": cfg,
                    "summary": summary,
                    "per_seed": per_seed,
                },
                f,
                indent=2,
            )

        print_matrix(cfg["alg_label"], summary, names)

    return experiment_id


def print_matrix(alg_label: str, summary: dict, names: list[str]) -> None:
    """Print the source x target accuracy matrix; the diagonal is in-domain."""
    w = max(len(n) for n in names) + 2
    print(f"\n{'='*72}")
    print(f"  {alg_label} — ACCURACY  (rows = TRAINED ON, cols = TESTED ON)")
    print("  [diagonal] = in-domain held-out; off-diagonal = transfer")
    print(f"{'='*72}\n")

    print(" " * w + "".join(f"{n:>18}" for n in names))
    for source in names:
        row = f"{source:<{w}}"
        for target in names:
            s = summary[source][target]
            cell = f"{s['accuracy_mean']*100:5.1f}%"
            cell = f"[{cell}]" if s["in_domain"] else f" {cell} "
            row += f"{cell:>18}"
        print(row)

    off = [summary[s][t]["accuracy_mean"] for s in names for t in names if s != t]
    diag = [summary[s][s]["accuracy_mean"] for s in names]
    print(f"\n  in-domain (diagonal) mean : {np.mean(diag)*100:5.1f}%")
    print(f"  transfer  (off-diag) mean : {np.mean(off)*100:5.1f}%")
    print(f"  transfer  (off-diag) min  : {np.min(off)*100:5.1f}%")
    print("\n  Chance is 50% (both classes are balanced). A transfer mean near")
    print("  chance means the in-domain score is corpus-specific, not 'AI-ness'.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_args(p, DEFAULT_CFG_DIR, DEFAULT_RESULTS_DIR)
    p.add_argument("--no-plots", action="store_true",
                   help="Write the matrices and JSON but skip the figures.")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Start training without checking the configs first.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_paths = resolve_configs(args)

    if not args.skip_preflight:
        report, _ = preflight(cfg_paths)
        print_report(report, "every config, dataset and column")
        if not report.ok:
            sys.exit(1)

    experiment_id = "clf_cd_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_cross_domain(cfg_paths, args.results_dir, experiment_id)

    exp_dir = args.results_dir / experiment_id
    print(f"\nDone. Results saved to: {rel(exp_dir)}")
    if not args.no_plots:
        plot_clf_cd_results(exp_dir)


if __name__ == "__main__":
    main()
