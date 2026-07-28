"""Leave-one-out (pooled) transfer eval for the human-vs-machine classifier.

The source x target matrix in `run_clf_cross_domain.py` answers "train on ONE
corpus, how well does it transfer?". That is deliberately harsh: a single-source
model never sees any domain or generator diversity, so it has every incentive to
memorise corpus artifacts.

This script asks the complementary, and arguably fairer, question: if the model
is allowed to train on EVERYTHING EXCEPT the held-out slice — so it does see
diversity — how well does it generalise to a genuinely unseen slice?

Four numbers are produced per held-out slice, and the figure plots exactly these:

  model in-domain accuracy
      A held-out slice of the TRAINING pool: same domains, same generators as
      training, unseen documents. This is the ceiling — what the model scores
      when nothing has shifted — and it is what the out-of-domain number has to
      be read against. Without it a low transfer number is ambiguous: it could
      be a transfer failure or simply a weak model.

  model out-of-domain accuracy
      The held-out slice itself. The transfer number.

  binoculars_fitted
      Zero-shot Binoculars with the cutoff calibrated by Youden's J on the
      training pool and applied unchanged to the test rows. No classifier is
      trained, but the threshold still comes from labelled data.

  binoculars_vanilla_<threshold>
      Binoculars with a cutoff taken straight from the config (e.g. the value
      published in the paper), fit to nothing at all. The only row that saw no
      labels whatsoever, and the strictest reference.

Every one of the four is an accuracy on the same rows, so the bars are
like-for-like.

Two folds are produced:

  leave-one-domain-out (LODO)
      Hold out a topic/genre (essay | reuter | wp); train on the other topics
      (both generators). Tests transfer to an unseen *topic*.

  leave-one-generator-out (LOGO)
      Hold out a generator (gpt | gpt54mini); train on the other generator(s),
      all topics. Tests transfer to an unseen *model*.

Leakage control — three pools now have to stay disjoint (fit, in-domain test,
out-of-domain test), and the two axes need different handling:

  * Document ids are domain-prefixed and globally unique ("essay:1" never
    collides with "reuter:1"), so distinct domains share no rows. LODO therefore
    needs no id exclusion for the out-of-domain slice: the held-out domain's rows
    are simply absent from training.

  * A domain and its rewrite ("essay" and "essay_gpt54mini") are built from the
    SAME human sources and share every human id. Splitting by generator would
    otherwise put identical human rows in both train and test. LOGO therefore
    first partitions the document ids into disjoint train/test id sets, and only
    then selects the held-out generator's rows from the test ids.

  * The in-domain test set is carved out of the training pool BY DOCUMENT ID for
    both folds, so a document never contributes rows to both the fit pool and the
    in-domain test pool — which it otherwise would, once per generator variant.

  * Wherever a pooled set spans several generators of one domain, its human rows
    are duplicated (one copy per generator dataset). Human rows are de-duplicated
    by id in every pooled set so the class balance and the counts stay honest;
    machine rows are never de-duplicated (a gpt rewrite and a gpt54mini rewrite
    of the same source are genuinely different examples).

Run:
    python -m epai.ai_detection.train.run_clf_leave_one_out
"""

from datetime import datetime
from pathlib import Path
import shutil
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from epai.utils.config_utils import collect_configs, load_cd_config
from epai.utils.train_utils import build_clf_model, stack_features
from epai.utils.visualization_utils import plot_clf_loo_results
from epai.ai_detection.train.run_clf_cross_domain import (
    BINARY_CLASSES,
    evaluate,
    load_dataset,
)

# The zero-shot reference shares its threshold rule and its naming with the
# cross-domain comparison, so `binoculars_vanilla_0.901531` means the same cutoff
# in both places and the two sets of results can be read side by side.
from epai.ai_detection.train.run_clf_cross_domain_embeddings import (
    DEFAULT_BINOX_COL,
    binoculars_baseline_specs,
    evaluate_binoculars,
    fit_binoculars_threshold,
)

BASE_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "clf_loo"

### INPUTS ###############################
CFG_DIR = BASE_DIR / "cfg" / "ai_detection" / "train" / "clf_loo" / "hd2"
##########################################

# The two evaluation pools every fold produces, in the order they are reported.
SPLITS = ("in_domain", "out_domain")


def require_tags(specs: list[dict]) -> None:
    """Every dataset must declare a `domain` and a `generator` to build the folds."""
    for spec in specs:
        missing = {"domain", "generator"} - spec.keys()
        if missing:
            raise ValueError(
                f"dataset {spec['name']!r}: leave-one-out needs the keys {missing}. "
                f"Tag each dataset with its `domain` and `generator`."
            )


def resolve_binoculars(cfg: dict) -> tuple[str | None, list[dict]]:
    """The Binoculars score column and the baseline variants to run.

    Configs declare the baseline the same way as the cross-domain scripts, with a
    top-level `binoculars_baseline` mapping. The older `data.binoculars_col` key
    is still honoured (fitted variant only) so existing configs keep running.
    """
    specs = binoculars_baseline_specs(cfg)
    if specs:
        baseline = cfg["binoculars_baseline"]
        return baseline.get("column", DEFAULT_BINOX_COL), specs

    legacy = cfg["data"].get("binoculars_col")
    if not legacy:
        return None, []
    return legacy, [
        {
            "variant": "fitted",
            "threshold": None,
            "name": "binoculars_fitted",
            "label": "binoculars_fitted",
        }
    ]


def build_pool(
    specs: list[dict],
    data: dict[str, pd.DataFrame],
    X_all: dict[str, np.ndarray],
    binoculars_col: str | None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Concatenate every dataset into one design matrix + aligned metadata frame.

    The returned `meta` has a clean RangeIndex that is row-aligned with `X_big`,
    so a boolean mask (or an index array) selects the same rows from both.
    """
    X_parts, meta_parts = [], []
    for spec in specs:
        name = spec["name"]
        df, X = data[name], X_all[name]
        bino = (
            df[binoculars_col].to_numpy(dtype=float)
            if binoculars_col
            else np.full(len(df), np.nan)
        )
        meta_parts.append(
            pd.DataFrame(
                {
                    "id": df["id"].to_numpy(),
                    "domain": spec["domain"],
                    "generator": spec["generator"],
                    "y": df["y"].to_numpy(),
                    "bino": bino,
                }
            )
        )
        X_parts.append(X)
    X_big = np.vstack(X_parts)
    meta = pd.concat(meta_parts, ignore_index=True)
    return X_big, meta


def pool_indices(meta: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Row indices for a pooled set, de-duplicating human rows by id.

    Machine rows are kept as-is; only human rows (y == 0), which are shared
    verbatim across a domain's generator variants, are collapsed to one per id.
    """
    sub = meta[mask]
    humans = sub[sub["y"] == 0].drop_duplicates("id", keep="first")
    machines = sub[sub["y"] == 1]
    idx = np.concatenate([humans.index.to_numpy(), machines.index.to_numpy()])
    idx.sort()
    return idx


def split_ids(
    all_ids: np.ndarray, test_size: float, seed: int, salt: int = 0
) -> tuple[set, set]:
    """Partition the unique document ids into disjoint train / test id sets.

    `salt` keeps the two nested splits of one seed independent: the outer LOGO
    split and the inner in-domain split are drawn from different id populations,
    but they should not be driven by the same permutation either.
    """
    uniq = np.unique(all_ids)
    rng = np.random.default_rng([seed, salt])
    perm = rng.permutation(uniq)
    n_test = max(1, int(round(len(uniq) * test_size)))
    return set(perm[n_test:]), set(perm[:n_test])


def assert_disjoint(meta: pd.DataFrame, pools: dict[str, np.ndarray]) -> None:
    """No document id may appear in two of the fit / test pools of a fold."""
    names = list(pools)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = np.intersect1d(
                meta.loc[pools[a], "id"].to_numpy(), meta.loc[pools[b], "id"].to_numpy()
            )
            if shared.size:
                raise AssertionError(
                    f"id leakage: {shared.size} document id(s) appear in both the "
                    f"{a!r} and {b!r} pool of this fold, e.g. {list(shared[:5])}."
                )


def baseline_fold(
    variant: dict,
    bino: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: dict[str, np.ndarray],
) -> dict | None:
    """One Binoculars variant scored on every evaluation pool of a fold.

    The fitted variant calibrates its cutoff on the FIT pool only — the same rows
    the classifier saw — so the in-domain test set is unseen by the baseline too
    and the two in-domain numbers are comparable. The vanilla variant reads its
    cutoff from the config and looks at no labels at all. Rows with a missing
    score are dropped from the baseline only, never from the model's evaluation.
    """
    finite = np.isfinite(bino)
    fit_ok = fit_idx[finite[fit_idx]]
    if variant["threshold"] is None:
        if fit_ok.size == 0 or np.unique(y[fit_ok]).size < 2:
            return None
        threshold = fit_binoculars_threshold(bino[fit_ok], y[fit_ok])
    else:
        threshold = float(variant["threshold"])

    res: dict = {
        "variant": variant["variant"],
        "label": variant["label"],
        "threshold": float(threshold),
    }
    for split, idx in eval_idx.items():
        ok = idx[finite[idx]]
        res[split] = (
            evaluate_binoculars(bino[ok], y[ok], threshold) if ok.size else None
        )
    return res


def run_fold(
    X_big: np.ndarray,
    meta: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    ml_alg: str,
    seed: int,
    indomain_test_size: float,
    baseline_specs: list[dict],
) -> dict:
    """Fit on the training pool and evaluate in-domain and out-of-domain.

    The training pool is split by document id into a fit pool and an in-domain
    test pool; `test_mask` supplies the out-of-domain rows. All three are checked
    to be id-disjoint before anything is fitted.
    """
    pool_idx = pool_indices(meta, train_mask)
    fit_ids, in_ids = split_ids(
        meta.loc[pool_idx, "id"].to_numpy(), indomain_test_size, seed, salt=1
    )
    ids = meta["id"]
    fit_idx = pool_indices(meta, train_mask & ids.isin(fit_ids).to_numpy())
    in_idx = pool_indices(meta, train_mask & ids.isin(in_ids).to_numpy())
    out_idx = pool_indices(meta, test_mask)

    assert_disjoint(meta, {"fit": fit_idx, "in_domain": in_idx, "out_domain": out_idx})

    y = meta["y"].to_numpy()
    scaler = StandardScaler()
    model = build_clf_model(ml_alg, seed)
    model.fit(scaler.fit_transform(X_big[fit_idx]), y[fit_idx])

    eval_idx = {"in_domain": in_idx, "out_domain": out_idx}
    fold: dict = {
        split: evaluate(model, scaler, X_big[idx], y[idx])
        for split, idx in eval_idx.items()
    }
    fold["fit_n"] = int(fit_idx.size)
    fold["baselines"] = {}
    bino = meta["bino"].to_numpy()
    for variant in baseline_specs:
        res = baseline_fold(variant, bino, y, fit_idx, eval_idx)
        if res is not None:
            fold["baselines"][variant["name"]] = res
    return fold


def mean_std(values: list) -> dict:
    """Mean / std over seeds, tolerating folds where a number was unavailable."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(clean)), "std": float(np.std(clean))}


def aggregate(runs: list[dict], baseline_specs: list[dict]) -> dict:
    """Mean / std over seeds for one held-out slice."""
    agg: dict = {
        "fit_n": runs[0]["fit_n"],
        "baselines": {},
    }
    for split in SPLITS:
        agg[f"model_{split}_accuracy"] = mean_std([r[split]["accuracy"] for r in runs])
        agg[f"model_{split}_macro_f1"] = mean_std([r[split]["macro_f1"] for r in runs])
        agg[f"{split}_n"] = runs[0][split]["n"]
        agg[f"{split}_class_balance"] = runs[0][split]["class_balance"]

    for variant in baseline_specs:
        name = variant["name"]
        present = [r["baselines"][name] for r in runs if name in r["baselines"]]
        if not present:
            continue
        entry = {
            "variant": variant["variant"],
            "label": variant["label"],
            "threshold": mean_std([p["threshold"] for p in present]),
        }
        for split in SPLITS:
            entry[f"{split}_accuracy"] = mean_std(
                [p[split]["accuracy"] if p[split] else None for p in present]
            )
            entry[f"{split}_n"] = next(
                (p[split]["n"] for p in present if p[split]), None
            )
        agg["baselines"][name] = entry
    return agg


def run_leave_one_out() -> str:
    experiment_id = "clf_loo_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    EXP_DIR = RESULTS_DIR / experiment_id
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    for cfg_path in tqdm(collect_configs(CFG_DIR), desc="Configs"):
        cfg = load_cd_config(cfg_path)

        base_seed: int = cfg["seed"]
        features: list[str] = cfg["features"]
        group_col: str = cfg["data"]["group_col"]
        human_label: str = cfg["data"]["human_label"]
        specs: list[dict] = cfg["data"]["datasets"]
        ml_alg: str = cfg["model"]["ml_alg"]
        n_runs: int = cfg["model"]["n_runs"]
        logo_test_size: float = cfg["model"].get("logo_test_size", 0.30)
        indomain_test_size: float = cfg["model"].get("indomain_test_size", 0.20)

        require_tags(specs)
        binoculars_col, baseline_specs = resolve_binoculars(cfg)

        ALG_DIR = EXP_DIR / cfg["alg_id"]
        ALG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg_path, ALG_DIR / "config.yaml")

        baseline_note = (
            f"{binoculars_col} → {', '.join(v['label'] for v in baseline_specs)}"
            if baseline_specs
            else "off"
        )
        tqdm.write(f"\n{'='*72}")
        tqdm.write(f"  ALG ID: {cfg['alg_id']}   ML ALG: {ml_alg}  ({n_runs} seeds)")
        tqdm.write(f"  FEATURES: {features}")
        tqdm.write(f"  IN-DOMAIN TEST: {indomain_test_size:.0%} of the training ids")
        tqdm.write(f"  BINOCULARS REF: {baseline_note}")
        tqdm.write(f"{'='*72}")

        # Carry the binoculars column through the load so the baselines can use it.
        load_cols = list(features)
        if binoculars_col:
            load_cols = features + [binoculars_col]
        data = {
            spec["name"]: load_dataset(spec, load_cols, group_col, human_label)
            for spec in specs
        }
        X_all = {name: stack_features(df, features)[0] for name, df in data.items()}

        dims = {name: X.shape[1] for name, X in X_all.items()}
        if len(set(dims.values())) != 1:
            raise ValueError(f"Datasets disagree on feature dimensionality: {dims}")

        X_big, meta = build_pool(specs, data, X_all, binoculars_col)

        domains = list(dict.fromkeys(spec["domain"] for spec in specs))
        generators = list(dict.fromkeys(spec["generator"] for spec in specs))

        results: dict[str, dict] = {"lodo": {}, "logo": {}}

        # ---- leave-one-domain-out -------------------------------------------
        if len(domains) >= 2:
            for held in tqdm(domains, desc="LODO", leave=False):
                train_mask = (meta["domain"] != held).to_numpy()
                test_mask = (meta["domain"] == held).to_numpy()
                runs = [
                    run_fold(
                        X_big, meta, train_mask, test_mask, ml_alg, base_seed + i,
                        indomain_test_size, baseline_specs,
                    )
                    for i in range(n_runs)
                ]
                results["lodo"][held] = aggregate(runs, baseline_specs)

        # ---- leave-one-generator-out ----------------------------------------
        if len(generators) >= 2:
            for held in tqdm(generators, desc="LOGO", leave=False):
                runs = []
                for i in range(n_runs):
                    # A fresh id partition per seed; the held-out generator's rows
                    # come only from the test ids, the rest only from the train ids,
                    # so no document (human or machine) straddles the split. The
                    # in-domain test set is then carved out of the train ids inside
                    # run_fold, which keeps all three pools id-disjoint.
                    train_ids, test_ids = split_ids(
                        meta["id"].to_numpy(), logo_test_size, base_seed + i
                    )
                    train_mask = (
                        (meta["generator"] != held) & meta["id"].isin(train_ids)
                    ).to_numpy()
                    test_mask = (
                        (meta["generator"] == held) & meta["id"].isin(test_ids)
                    ).to_numpy()
                    runs.append(
                        run_fold(
                            X_big, meta, train_mask, test_mask, ml_alg, base_seed + i,
                            indomain_test_size, baseline_specs,
                        )
                    )
                results["logo"][held] = aggregate(runs, baseline_specs)

        with open(ALG_DIR / "leave_one_out.json", "w") as f:
            json.dump(
                {
                    "schema": 2,
                    "alg_id": cfg["alg_id"],
                    "alg_label": cfg["alg_label"],
                    "classes": BINARY_CLASSES,
                    "domains": domains,
                    "generators": generators,
                    "baselines": [
                        {k: v for k, v in spec.items()} for spec in baseline_specs
                    ],
                    "config": cfg,
                    "results": results,
                },
                f,
                indent=2,
            )

        print_report(cfg["alg_label"], results, baseline_specs)

    return experiment_id


def _fmt(stat: dict | None) -> str:
    if stat is None or stat["mean"] is None:
        return "   —  "
    return f"{stat['mean']*100:5.1f}%"


def print_report(alg_label: str, results: dict, baseline_specs: list[dict]) -> None:
    """In-domain vs out-of-domain accuracy, against every Binoculars baseline.

    Baseline columns are the OUT-OF-DOMAIN accuracy — the same rows as the
    out-of-domain model column, so the two are directly comparable.
    """
    for mode, title in [
        ("lodo", "LEAVE-ONE-DOMAIN-OUT"),
        ("logo", "LEAVE-ONE-GENERATOR-OUT"),
    ]:
        rows = results.get(mode, {})
        if not rows:
            continue
        print(f"\n{'='*100}")
        print(f"  {alg_label} — {title}")
        print("  in-domain = held-out ids of the training pool; out-domain = the held-out slice")
        print("  binoculars columns are accuracies on the out-domain rows")
        print(f"{'='*100}\n")

        names = [v["name"] for v in baseline_specs]
        short = {v["name"]: v["label"].replace("binoculars_", "binox ") for v in baseline_specs}
        header = f"  {'held-out':<16}{'fit_n':>8}{'in_n':>7}{'out_n':>7}"
        header += f"{'in acc':>10}{'out acc':>10}"
        for name in names:
            header += f"{short[name]:>24}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        in_accs, out_accs = [], []
        for held, agg in rows.items():
            in_accs.append(agg["model_in_domain_accuracy"]["mean"])
            out_accs.append(agg["model_out_domain_accuracy"]["mean"])
            line = f"  {held:<16}{agg['fit_n']:>8}{agg['in_domain_n']:>7}{agg['out_domain_n']:>7}"
            line += f"{_fmt(agg['model_in_domain_accuracy']):>10}"
            line += f"{_fmt(agg['model_out_domain_accuracy']):>10}"
            for name in names:
                entry = agg["baselines"].get(name)
                line += f"{_fmt(entry['out_domain_accuracy'] if entry else None):>24}"
            print(line)

        print("  " + "-" * (len(header) - 2))
        print(
            f"  {'mean':<16}{'':>24}"
            f"{np.mean(in_accs)*100:>9.1f}%{np.mean(out_accs)*100:>9.1f}%"
        )
    print(
        "\n  The in-domain column is the ceiling: the drop from it to the out-domain\n"
        "  column IS the transfer problem. A baseline column above the out-domain\n"
        "  column means the trained model is not worth its training data there."
    )


def main() -> None:
    experiment_id = run_leave_one_out()
    print(f"\nDone. Results saved to: {RESULTS_DIR / experiment_id}")
    plot_clf_loo_results(RESULTS_DIR / experiment_id)


if __name__ == "__main__":
    main()
