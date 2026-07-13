"""Offline smoke test for the workflow (NO API keys, NO model downloads).

Exercises every code path that does not need network/keys, on small synthetic data:
config.yaml loading + per-run overrides + the class-count knobs, identity, the shared
split, targets, classes, metadata, feature_builder + NeutralScaler (neutral-0 + mask),
intrinsic-dim, encoders (AE/PCA/identity), dim selection, the torch heads + trainer,
metrics, novelty, full `run_experiment` runs for reg/meta/clf (training + saving +
reload), the workflow/phase table, the registry, the progress display, a mocked
text-transformer run, and the aggregate pipeline.

Run:
    uv run python -m workflow.src.tests.smoke
Exit code 0 = all passed.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PASS, FAIL = [], []


def check(name: str, fn) -> None:
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:  # noqa: BLE001
        FAIL.append(name)
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------- synthetic data
def make_corpus(n: int = 160, dim: int = 32, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    # a target loosely correlated with the first embedding dim, in [0,1]
    raw = emb[:, 0] + 0.3 * rng.normal(size=n)
    pct = (raw - raw.min()) / (raw.max() - raw.min())
    df = pd.DataFrame({
        "paper_id": [f"W{i:05d}" for i in range(n)],
        "s2_id": [f"W{i:05d}" for i in range(n)],
        "title": [f"Paper {i}" for i in range(n)],
        "abstract": [f"Abstract about topic {i % 7}" for i in range(n)],
        "year": rng.integers(2010, 2023, size=n),
        "venue": rng.choice(["VenueA", "VenueB", "VenueC", None], size=n),
        "citation_count": rng.integers(0, 500, size=n),
        "author_count": rng.integers(1, 8, size=n),
        "best_author_h_index": rng.choice([5, 10, 20, np.nan], size=n),
        "subfield": rng.choice(["Subfield X", "Subfield Y"], size=n),
        "domain": "Computer Science",
        "fwci": rng.gamma(2.0, 1.0, size=n),
        "citation_normalized_percentile": pct.astype(np.float64),
        "emb_specter2": [row.tolist() for row in emb],
    })
    return df


# ---------------------------------------------------------------- tests
def t_config():
    from workflow.src.iwf.config import Settings, load_settings
    from workflow.src.iwf.targets import ALL_TARGET_KINDS
    s = Settings()
    assert abs(sum(s.split_ratios) - 1.0) < 1e-9
    assert s.target_kind in ALL_TARGET_KINDS
    assert isinstance(s.hidden, list) and len(s.hidden) >= 1
    # the real config.yaml loads and resolves into a valid Settings
    s2 = load_settings()
    s2.validate()
    assert len(s2.class_names) == len(s2.class_thresholds) + 1


def t_config_overrides():
    """Per-run overrides + aliases (workflow / embedder / num_classes) resolve correctly."""
    from workflow.src.iwf.config import load_settings, resolve_class_thresholds
    s = load_settings(overrides={"workflow": "emb_meta_reg", "embedder": "scibert",
                                 "num_classes": 5, "lr": "0.001", "hidden": [64, 32]})
    assert s.workflow_type == "emb_meta_reg"
    assert s.embedders == ["scibert"] and s.embed_for_encoder == "scibert"
    assert len(s.class_thresholds) == 4 and len(s.class_names) == 5
    assert s.class_names[0] == "Q5" and s.class_names[-1] == "Q1"
    assert abs(s.lr - 0.001) < 1e-12          # string coerced to float
    assert s.hidden == [64, 32]
    # the three class-count knobs, in priority order
    assert resolve_class_thresholds({"class_distribution": [15, 35, 35, 15]}) == \
        [0.15, 0.5, 0.85]
    assert resolve_class_thresholds({"num_classes": 4}) == [0.25, 0.5, 0.75]
    assert resolve_class_thresholds({"class_thresholds": [0.5, 0.9]}) == [0.5, 0.9]
    for bad in ({"num_classes": 1}, {"num_classes": 26},
                {"class_distribution": [100]}):
        try:
            resolve_class_thresholds(bad)
        except ValueError:
            continue
        raise AssertionError(f"class-count knob {bad} should have been rejected")


def t_expand_runs():
    """run_all_workflows expands the runs queue: workflow lists, per-run overrides."""
    from workflow.run_all_workflows import expand_runs
    cfg = {"runs": [
        {"name": "a", "workflow": ["emb_clf", "emb_reg"], "num_classes": 5},
        {"name": "b", "workflow": "tfm_reg", "text_transformer_epochs": 1},
    ]}
    runs = expand_runs(cfg, None)
    assert [wf for wf, _, _ in runs] == ["emb_clf", "emb_reg", "tfm_reg"]
    assert runs[0][1] == "a" and len(runs[0][2].class_names) == 5
    assert runs[2][2].text_transformer_epochs == 1
    for bad_cfg in ({}, {"runs": []},
                    {"runs": [{"name": "x", "workflow": "nope"}]}):
        try:
            expand_runs(bad_cfg, None)
        except ValueError:
            continue
        raise AssertionError(f"expand_runs should have rejected {bad_cfg}")


def t_identity():
    from workflow.src.iwf.data.identity import ensure_paper_id, keyed_merge, normalize_doi
    df = pd.DataFrame({"title": ["a", "b"], "abstract": ["x", "y"]})
    out = ensure_paper_id(df)
    assert out["paper_id"].notna().all() and out["paper_id"].is_unique
    left = pd.DataFrame({"paper_id": ["1", "2"], "v": [10, 20]})
    right = pd.DataFrame({"paper_id": ["2", "1"], "w": [200, 100]})  # different order
    merged = keyed_merge(left, right, ["w"])
    assert merged.loc[merged.paper_id == "1", "w"].iloc[0] == 100  # joined by key, not row
    assert normalize_doi("https://doi.org/10.1/AbC") == "10.1/abc"


def t_split():
    from workflow.src.iwf.data.dataset import add_split_column, make_split, split_masks
    df = make_corpus()
    df["strat_label"] = np.where(df["citation_normalized_percentile"] > 0.5, "Good", "Weak")
    a = add_split_column(df, seed=42, ratios=[0.7, 0.15, 0.15], stratify_col="strat_label")
    b = add_split_column(df, seed=42, ratios=[0.7, 0.15, 0.15], stratify_col="strat_label")
    assert (a["split"] == b["split"]).all(), "same seed must give same split"
    c = add_split_column(df, seed=7, ratios=[0.7, 0.15, 0.15], stratify_col="strat_label")
    assert not (a["split"] == c["split"]).all(), "different seed must differ"
    m = split_masks(a)
    assert m["train"].sum() + m["val"].sum() + m["test"].sum() == len(df)
    assert set(np.unique(a["split"])) == {"train", "val", "test"}
    # tiny / degenerate stratify must not crash (falls back)
    make_split(6, 1, [0.7, 0.15, 0.15], stratify=np.array(list("AABBCC")))


def t_targets():
    from workflow.src.iwf.targets import ALL_TARGET_KINDS, compute_target
    df = make_corpus()
    # uniform contract: every kind returns a [0,1] Series aligned to df.index
    for kind in ALL_TARGET_KINDS:
        v = compute_target(kind, df, min_refset=30)
        assert isinstance(v, pd.Series) and v.index.equals(df.index), f"{kind} index"
        assert v.dropna().between(0, 1).all(), f"{kind} out of [0,1]"
    # citation-based kinds cover ~all rows (citation_count is always present)
    for kind in ("field_year_percentile", "normalized_percentile", "bayesian_impact"):
        assert compute_target(kind, df, min_refset=30).notna().mean() > 0.9, kind
    a = compute_target("fwci_percentile", df)
    assert a.notna().all()
    # fallback path: drop the ready percentile, force FWCI-rank fallback
    df2 = df.drop(columns=["citation_normalized_percentile"])
    a2 = compute_target("fwci_percentile", df2)
    assert a2.notna().mean() > 0.9


def t_classes():
    from workflow.src.iwf.targets import CLASS_NAMES, to_classes
    s = pd.Series(np.linspace(0, 1, 100))
    cp = to_classes(s, "percentile", [0.5, 0.9])
    assert set(cp.unique()) <= set(CLASS_NAMES)
    assert (cp == "Weak").sum() > 0 and (cp == "Good").sum() > 0
    cs = to_classes(s, "css")
    assert set(cs.dropna().unique()) <= set(CLASS_NAMES)


def t_metadata():
    from workflow.src.iwf.features.metadata import build_metadata
    df = make_corpus()
    meta = build_metadata(df, ["year", "best_author_h_index", "author_count", "venue"])
    assert "meta_year" in meta and "meta_venue_freq" in meta
    assert meta["meta_h_index"].isna().any()  # some NaN preserved for masking


def t_feature_builder():
    from workflow.src.iwf.features.feature_builder import NeutralScaler, build_xy
    n, k = 50, 8
    df = pd.DataFrame({
        "z": [list(np.random.rand(k)) for _ in range(n)],
        "target": np.random.rand(n),
        "novelty": [0.5] * n,
    })
    df.loc[:4, "novelty"] = np.nan  # force some missing
    X, y, names, nval = build_xy(df, "reg", scalar_cols=["novelty"])
    assert X.shape == (n, k + 1 + 1)  # z + novelty + novelty_missing
    assert nval == k + 1
    assert names[-1] == "novelty_missing"
    assert X[:5, -1].sum() == 5  # mask flags the missing rows
    sc = NeutralScaler(nval).fit(X[:40])
    Xs = sc.transform(X)
    assert np.isfinite(Xs).all(), "missing must become finite neutral-0"
    assert abs(float(Xs[:40, :nval].mean())) < 0.5
    # leakage guard: reserved target-input columns are refused as features
    from workflow.src.iwf.features.feature_builder import drop_reserved
    assert drop_reserved(["novelty", "citation_count", "meta_year", "target_x"]) == ["novelty"]
    df["citation_count"] = 1  # present but must never become a feature
    X2, _, names2, _ = build_xy(df, "reg", scalar_cols=["novelty", "citation_count"])
    assert "citation_count" not in names2 and X2.shape[1] == k + 1 + 1
    # ...unless the guard is bypassed on purpose (emb_all_meta_* upper-bound test)
    X3, _, names3, _ = build_xy(df, "reg", scalar_cols=["novelty", "citation_count"],
                                bypass_guard=True)
    assert "citation_count" in names3


def t_intrinsic_dim():
    from workflow.src.iwf.encoder.intrinsic_dim import estimate_intrinsic_dim
    X = np.random.RandomState(0).normal(size=(120, 10)).astype(np.float32)
    d = estimate_intrinsic_dim(X)
    assert "twonn" in d and "mle" in d


def t_encoders():
    from workflow.src.iwf.encoder.autoencoder import AutoencoderEncoder
    from workflow.src.iwf.encoder.pca_umap import IdentityEncoder, PCAEncoder
    X = np.random.RandomState(0).normal(size=(120, 32)).astype(np.float32)
    ae = AutoencoderEncoder(latent_dim=8, max_epochs=8, patience=3, seed=0).fit(X)
    z = ae.transform(X)
    assert z.shape == (120, 8) and z.dtype == np.float32
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "enc.pt"
        ae.save(p)
        ae2 = AutoencoderEncoder(latent_dim=8).load(p)
        assert np.allclose(ae2.transform(X), z, atol=1e-5)
    assert PCAEncoder(8).fit(X).transform(X).shape == (120, 8)
    assert IdentityEncoder().fit(X).transform(X).shape == (120, 32)


def t_dim_selector():
    from workflow.src.iwf.encoder.dim_selector import select_dim
    X = np.random.RandomState(1).normal(size=(150, 32)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    tr = np.zeros(150, bool); tr[:100] = True
    va = np.zeros(150, bool); va[100:130] = True
    rep = select_dim(X, tr, va, y, candidate_dims=[4, 8, 16], task="reg",
                     encoder_kind="autoencoder", max_epochs=6, seed=0)
    assert rep["k_star"] in (4, 8, 16)
    assert len(rep["curve"]) == 3


def t_models():
    from workflow.src.iwf.models.classifier import TorchMLPClassifier
    from workflow.src.iwf.models.regressor import TorchMLPRegressor
    n, k = 200, 12
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n, k)).astype(np.float32)
    yreg = ((X[:, 0] - X[:, 0].min()) / np.ptp(X[:, 0])).astype(np.float32)
    reg = TorchMLPRegressor(hidden=[32, 16], max_epochs=20, patience=6, seed=0)
    reg.fit(X[:150], yreg[:150], X[150:], yreg[150:])
    p = reg.predict(X[150:])
    assert p.shape == (50,) and np.isfinite(p).all() and (0 <= p).all() and (p <= 1).all()
    yclf = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0.5).astype(int)  # 0/1/2
    clf = TorchMLPClassifier(n_classes=3, hidden=[32, 16], max_epochs=20, patience=6, seed=0)
    clf.fit(X[:150], yclf[:150], X[150:], yclf[150:])
    proba = clf.predict_proba(X[150:])
    assert proba.shape == (50, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert clf.predict(X[150:]).shape == (50,)


def t_architectures():
    """Every registered model block builds, trains and yields valid 4-class probabilities;
    the class-balanced compensation is a no-op when the classes are balanced."""
    from workflow.src.iwf.models.architectures import available_archs
    from workflow.src.iwf.models.classifier import TorchMLPClassifier, class_weights
    archs = available_archs()
    assert {"mlp", "resmlp", "ft_transformer"} <= set(archs)
    n, k, C = 240, 16, 4
    rng = np.random.RandomState(1)
    X = rng.normal(size=(n, k)).astype(np.float32)
    y = rng.randint(0, C, size=n)
    for arch in archs:
        clf = TorchMLPClassifier(n_classes=C, hidden=[24, 12], max_epochs=8, patience=4,
                                 seed=0, arch=arch)
        clf.fit(X[:180], y[:180], X[180:], y[180:])
        proba = clf.predict_proba(X[180:])
        assert proba.shape == (60, C)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4), f"{arch} proba"
    # class-balanced (effective) weights: uniform when balanced, up-weighted for the rare class
    bal = class_weights(np.array([0, 0, 1, 1, 2, 2, 3, 3]), C, "effective", 0.99)
    assert np.allclose(bal, 1.0, atol=1e-6)
    imb = class_weights(np.array([0] * 100 + [1, 2, 3]), C, "effective", 0.99)
    assert imb[1] > imb[0] and imb[2] > imb[0] and imb[3] > imb[0]


def t_metrics():
    from workflow.src.iwf.evaluation.metrics import (
        classification_metrics, generated_vs_human, regression_metrics,
    )
    rm = regression_metrics(np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.4, 0.8]))
    assert {"rmse", "mse", "mae", "r2", "spearman", "n"} <= rm.keys()
    cm = classification_metrics(np.array([0, 1, 2, 1]), np.array([0, 1, 1, 1]),
                                ["Weak", "Average", "Good"])
    assert "macro_f1" in cm and len(cm["confusion"]) == 3
    assert len(cm["per_class_f1"]) == 3
    g = generated_vs_human(np.array([0.5, 0.6]), {"mirai": np.array([0.3, 0.4])})
    assert "human_test" in g and "mirai" in g


def t_novelty():
    from workflow.src.iwf.features.novelty import novelty_feasibility
    Z = np.random.RandomState(0).normal(size=(80, 16)).astype(np.float32)
    nov, feas = novelty_feasibility(Z, k=5)
    assert nov.shape == (80,) and feas.shape == (80,)
    assert np.isfinite(nov).all() and np.isfinite(feas).all()


def _reduced_df(n=160, k=16, seed=0):
    """A synthetic post-reduce dataframe: z + target + class_label + split."""
    from workflow.src.iwf.data.dataset import add_split_column
    from workflow.src.iwf.targets import to_classes
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, k)).astype(np.float32)
    raw = z[:, 0] + 0.3 * rng.normal(size=n)
    target = (raw - raw.min()) / (raw.max() - raw.min())
    df = pd.DataFrame({
        "paper_id": [f"W{i:05d}" for i in range(n)],
        "z": [row.tolist() for row in z],
        "target": target.astype(np.float64),
        "novelty": rng.random(n),
        "meta_h_index": rng.choice([5.0, 10.0, np.nan], size=n),
    })
    df["strat_label"] = to_classes(df["target"], "percentile", [0.5, 0.9])
    df = add_split_column(df, seed=42, ratios=[0.7, 0.15, 0.15], stratify_col="strat_label")
    df["class_label"] = to_classes(df["target"], "percentile", [0.5, 0.9])
    return df


def t_experiment_reg():
    import json
    from workflow.src.iwf import paths
    from workflow.src.iwf.config import Settings
    from workflow.src.iwf.training.experiment import run_experiment
    df = _reduced_df()
    s = Settings(n_runs=1, max_epochs=15, patience=5, hidden=[32, 16])
    out_dir = paths.RESULTS_DIR / "emb_reg_1"
    run_experiment(df, "reg", settings=s, scalar_cols=[], out_dir=out_dir, label="z->impact")
    res = json.loads((out_dir / "seed_0" / "result.json").read_text())
    assert {"rmse", "r2", "spearman"} <= res["test"].keys()
    assert {"rmse", "r2", "spearman"} <= res.keys()          # flat root copies
    assert res["train"]["y_true"] and res["y_all"]           # arrays for plots
    assert (out_dir / "seed_0" / "model" / "model.joblib").exists()


def t_experiment_meta_and_clf():
    from workflow.src.iwf import paths
    from workflow.src.iwf.config import Settings
    from workflow.src.iwf.features.feature_builder import scalar_feature_columns
    from workflow.src.iwf.training.experiment import run_experiment
    df = _reduced_df()
    s = Settings(n_runs=1, max_epochs=15, patience=5, hidden=[32, 16],
                 class_thresholds=[0.5, 0.9], class_names=[])
    cols = scalar_feature_columns(df, True, True, False, [])  # meta_h_index + novelty
    assert "meta_h_index" in cols and "novelty" in cols
    run_experiment(df, "reg", settings=s, scalar_cols=cols,
                   out_dir=paths.RESULTS_DIR / "emb_meta_reg_1", label="z+meta")
    s_clf = Settings(n_runs=1, max_epochs=15, patience=5, hidden=[32, 16],
                     class_thresholds=[0.5, 0.9],
                     class_names=["Weak", "Average", "Good"])
    run_experiment(df, "clf", settings=s_clf, scalar_cols=[],
                   out_dir=paths.RESULTS_DIR / "emb_clf_1", label="z->class")


def t_saved_model_predicts():
    """Mirror predict_generated's scoring: find the latest reg model, predict on new z."""
    import joblib
    from workflow.src.iwf import workflow_results
    from workflow.src.iwf.features.feature_builder import build_xy
    reg_dir = workflow_results.latest_model_dir(task="reg")
    assert reg_dir is not None, "latest_model_dir should find the t_experiment_reg output"
    sd = sorted(reg_dir.glob("seed_*"))[0]
    model = joblib.load(sd / "model" / "model.joblib")
    scaler = joblib.load(sd / "model" / "scaler.joblib")
    gen = pd.DataFrame({"paper_id": ["g1", "g2"],
                        "z": [list(np.random.rand(16)), list(np.random.rand(16))]})
    X, y, _, _ = build_xy(gen, "reg", scalar_cols=[])
    assert y is None  # generated papers have no target
    pred = model.predict(scaler.transform(X))
    assert pred.shape == (2,) and np.isfinite(pred).all()


def t_validate():
    """Settings.validate() accepts a good config and raises ValueError on each violation."""
    from workflow.src.iwf.config import Settings
    Settings().validate()                                      # defaults are valid
    for bad in (
        Settings(workflow_type="bogus"),
        Settings(target_kind="not_a_kind"),
        Settings(model_arch="not_registered"),
        Settings(n_runs=0),
        Settings(use_metadata=True, metadata_fields=["year", "venue"]),   # leakage guard
        Settings(workflow_type="tfm_clf", text_transformer_arch="nope"),
    ):
        try:
            bad.validate()
        except ValueError:
            continue
        raise AssertionError(
            f"validate() should have rejected {bad.workflow_type}/{bad.target_kind}")


def t_workflow_table():
    """All eight topologies are present and route to the documented phase lists."""
    from workflow.src.iwf.config import WORKFLOWS, WORKFLOW_TYPES, is_text_workflow
    assert len(WORKFLOW_TYPES) == 8
    assert WORKFLOWS["emb_reg"]["phases"] == [0, 1, 2, 3, 4, 6, 7]
    assert WORKFLOWS["emb_clf"]["phases"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert WORKFLOWS["tfm_clf"]["phases"] == [0, 1, 5, 6, 7]
    assert WORKFLOWS["tfm_reg"]["phases"] == [0, 1, 6, 7]
    assert WORKFLOWS["emb_all_meta_clf"]["features"] == "all_meta"
    # text workflows must never schedule embed/encode/reduce
    for wf in ("tfm_clf", "tfm_reg"):
        assert is_text_workflow(wf)
        assert not ({2, 3, 4} & set(WORKFLOWS[wf]["phases"]))
    # every emb workflow's task matches its name suffix
    for wf, spec in WORKFLOWS.items():
        assert spec["task"] == ("clf" if wf.endswith("_clf") else "reg")


def t_registry():
    """The module registry exposes the real slot implementations, incl. the 4 text arches."""
    from workflow.src.iwf.registry import available_modules, get_module
    assert {"specter2_ft", "scibert_ft", "specter2_lora", "scibert_lora"} <= set(
        available_modules("text_transformer"))
    assert "identity" in available_modules("encoder")
    assert {"mlp", "resmlp", "ft_transformer"} <= set(available_modules("impact_predictor"))
    cls = get_module("text_transformer", "specter2_ft")
    assert cls.BACKBONE_NAME == "allenai/specter2_base"


def t_progress():
    """MultiWorkflowProgress instantiates, runs its queue/display lifecycle, renders."""
    import io
    from workflow.src.iwf.progress import MultiWorkflowProgress
    buf = io.StringIO()
    prog = MultiWorkflowProgress(["emb_reg", "tfm_reg"], stream=buf)
    prog.start()
    prog.update("emb_reg", phase=(6, 7), percent=40.0, seed=(2, 5),
                metric=("val_spearman", 0.412))
    prog.done("emb_reg", metric=("best_spearman", 0.431))
    prog.error("tfm_reg", "synthetic error for the smoke test")
    prog.stop()
    text = buf.getvalue()
    assert "emb_reg" in text and "DONE" in text and "ERROR" in text


class _FakeTokenizer:
    """Offline stand-in for a HF tokenizer: hashes words to ids, no vocab/download."""

    def __init__(self, vocab: int = 100):
        self.vocab = vocab

    def __call__(self, titles, abstracts=None, truncation=True, max_length=32,
                 padding=True, return_tensors="pt"):
        import torch
        pairs = zip(titles, abstracts) if abstracts is not None else ((t, "") for t in titles)
        texts = [f"{t} {a}" for t, a in pairs]
        length = min(max_length, max(4, max(len(x.split()) for x in texts) + 2))
        ids, mask = [], []
        for x in texts:
            toks = [1] + [(hash(w) % (self.vocab - 2)) + 2 for w in x.split()][: length - 2]
            m = [1] * len(toks)
            while len(toks) < length:
                toks.append(0); m.append(0)
            ids.append(toks[:length]); mask.append(m[:length])
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}

    def save_pretrained(self, path):  # no-op for the mock
        return None


def _text_df(n=120, seed=0):
    """Synthetic prepared-text dataframe: title/abstract/target/class_label/split."""
    from workflow.src.iwf.data.dataset import add_split_column
    from workflow.src.iwf.targets import to_classes
    rng = np.random.default_rng(seed)
    topics = rng.integers(0, 5, size=n)
    target = (topics / 4.0 + 0.1 * rng.standard_normal(n)).clip(0, 1)
    df = pd.DataFrame({
        "paper_id": [f"W{i:05d}" for i in range(n)],
        "title": [f"Study of topic {t} number {i}" for i, t in enumerate(topics)],
        "abstract": [f"We investigate aspect {t} with method {i % 3} in detail"
                     for i, t in enumerate(topics)],
        "target": target.astype(np.float64),
    })
    df["strat_label"] = to_classes(df["target"], "percentile", [0.25, 0.5, 0.75])
    df = add_split_column(df, seed=42, ratios=[0.7, 0.15, 0.15], stratify_col="strat_label")
    df["class_label"] = to_classes(df["target"], "percentile", [0.25, 0.5, 0.75],
                                   names=["Q4", "Q3", "Q2", "Q1"])
    return df


def t_tfm_mock():
    """Text-transformer training path on a tiny mocked BERT (no download), reg + clf."""
    from transformers import BertConfig, BertModel
    from workflow.src.iwf import paths
    from workflow.src.iwf.config import Settings
    from workflow.src.iwf.models.text_transformers import TextTransformerBase
    from workflow.src.iwf.training.text_trainer import TextTransformerTrainer

    df = _text_df()
    class_names = ["Q4", "Q3", "Q2", "Q1"]
    s = Settings(text_transformer_epochs=2, text_transformer_batch_size=8,
                 text_transformer_max_len=24, hidden=[16], n_runs=1,
                 class_names=class_names, class_thresholds=[0.25, 0.5, 0.75])
    bert_cfg = BertConfig(hidden_size=24, num_hidden_layers=2, num_attention_heads=2,
                          intermediate_size=48, vocab_size=100, max_position_embeddings=64)
    tok = _FakeTokenizer(vocab=100)

    # regression
    reg_model = TextTransformerBase(s, 1, "reg", backbone=BertModel(bert_cfg), tokenizer=tok)
    reg = TextTransformerTrainer(s, "reg", model=reg_model)
    m_reg = reg.run(df, seed=0, output_dir=paths.RESULTS_DIR / "tfm_reg_1" / "seed_0",
                    class_names=class_names)
    assert {"spearman", "mae", "r2"} <= m_reg.keys()
    assert (paths.RESULTS_DIR / "tfm_reg_1" / "seed_0" / "result.json").is_file()
    assert (paths.RESULTS_DIR / "tfm_reg_1" / "seed_0" / "checkpoint").is_dir()

    # classification (4 ordinal classes)
    clf_model = TextTransformerBase(s, len(class_names), "clf",
                                    backbone=BertModel(bert_cfg), tokenizer=tok)
    clf = TextTransformerTrainer(s, "clf", model=clf_model)
    m_clf = clf.run(df, seed=0, output_dir=paths.RESULTS_DIR / "tfm_clf_1" / "seed_0",
                    class_names=class_names)
    assert {"macro_f1", "accuracy", "per_class_f1"} <= m_clf.keys()
    assert len(m_clf["per_class_f1"]) == len(class_names)
    assert "confusion" in m_clf     # shared metrics give tfm seeds a confusion matrix too


def t_novelty_kinds():
    """Spherical novelty is percentile-uniform [0,1]; legacy kind + dispatcher work."""
    from workflow.src.iwf.features.novelty import compute_novelty, novelty_spherical
    Z = np.random.RandomState(1).normal(size=(90, 12)).astype(np.float32)
    nov = novelty_spherical(Z, k=6)
    assert nov.shape == (90,) and (nov >= 0).all() and (nov <= 1).all()
    assert 0.4 < float(np.mean(nov)) < 0.6          # percentile transform => ~uniform
    legacy = compute_novelty(Z, k=6, kind="cosine_max")
    assert legacy.shape == (90,) and np.isfinite(legacy).all()
    try:
        compute_novelty(Z, kind="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("compute_novelty should reject an unknown kind")


def t_feasibility_composite():
    """Composite feasibility is uniform [0,1]; year-less rows and n<=2 are safe."""
    from workflow.src.iwf.features.feasibility import feasibility
    rng = np.random.RandomState(2)
    Z = rng.normal(size=(100, 10)).astype(np.float32)
    years = rng.randint(2005, 2024, size=100).astype(float)
    years[:7] = np.nan
    f = feasibility(Z, years, k=6)
    assert f.shape == (100,) and (f >= 0).all() and (f <= 1).all()
    assert 0.4 < float(np.mean(f)) < 0.6
    assert feasibility(Z[:2], years[:2]).shape == (2,)
    f2 = feasibility(Z, None, k=6)                   # no years at all -> still valid
    assert np.isfinite(f2).all()


def t_baseline_metrics_and_plots():
    """baseline_compare: metrics table + both figures render from synthetic scores."""
    from workflow.src.iwf.evaluation.baseline_compare import (
        baseline_metrics, plot_baseline_agreement, plot_baseline_correlation,
    )
    rng = np.random.RandomState(3)
    y = rng.rand(200)
    scores = {
        "ours": np.clip(y + 0.1 * rng.randn(200), 0, 1),
        "naip": np.clip(y + 0.25 * rng.randn(200), 0, 1),
        "venue_ais": rng.rand(200),                   # uninformative baseline
    }
    thresholds = [0.25, 0.5, 0.75]
    m = baseline_metrics(y, scores, thresholds)
    assert set(m["scorer"]) == set(scores)
    ours = m.set_index("scorer")
    assert ours.loc["ours", "spearman"] > ours.loc["venue_ais", "spearman"]
    assert 0 <= ours.loc["ours", "exact_class_pct"] <= 100
    with tempfile.TemporaryDirectory() as d:
        plot_baseline_agreement(m, 4, "smoke_wf", Path(d) / "agree.png")
        frame = pd.DataFrame({"target": y, **scores})
        plot_baseline_correlation(frame, None, "smoke_wf", Path(d) / "corr.png")
        assert (Path(d) / "agree.png").is_file() and (Path(d) / "corr.png").is_file()


def t_nf_analysis():
    """nf_analysis: correlation matrix + binned-impact figures from a baselines frame."""
    from workflow.src.iwf.evaluation.nf_analysis import (
        nf_binned_impact, nf_correlation_matrix, partial_spearman,
    )
    rng = np.random.RandomState(4)
    n = 300
    target = rng.rand(n)
    df = pd.DataFrame({
        "target": target,
        "novelty": np.clip(target + 0.4 * rng.randn(n), 0, 1),
        "feasibility": rng.rand(n),
        "meta_h_index": rng.rand(n) * 40,
        "meta_venue_freq": rng.rand(n) * 100,
    })
    with tempfile.TemporaryDirectory() as d:
        stats = nf_correlation_matrix(df, "smoke_wf", Path(d) / "m.png")
        assert "novelty" in stats["columns"] and (Path(d) / "m.png").is_file()
        assert abs(stats["partial_vs_target"]["novelty"]) > 0.2   # real signal survives
        binned = nf_binned_impact(df, "smoke_wf", Path(d) / "b.png")
        assert "novelty" in binned and (Path(d) / "b.png").is_file()
    x = rng.rand(100)
    assert abs(partial_spearman(x, x, rng.rand(100, 2))) > 0.95


def t_external_model_registry():
    """All six external scorers resolve; unknown names raise; the kNN baseline scores."""
    from workflow.src.iwf.features.external_impact import (
        Specter2KnnBaseline, get_external_models, known_external_models,
    )
    assert {"naip", "specter2_knn", "llm_judge", "venue_ais"} <= set(known_external_models())
    models = get_external_models(["specter2_knn"])
    assert models[0].name == "specter2_knn"
    try:
        get_external_models(["nope"])
    except ValueError:
        pass
    else:
        raise AssertionError("unknown external model must raise")
    df = make_corpus(n=80)
    df["target"] = df["citation_normalized_percentile"]
    df["split"] = ["train"] * 60 + ["val"] * 10 + ["test"] * 10
    with tempfile.TemporaryDirectory() as d:
        s = Specter2KnnBaseline(k=5).score_dataframe(df, Path(d) / "c.json")
    assert s.notna().all() and s.between(0, 1).all()
    # the retrieval baseline must carry real signal on this correlated corpus
    from scipy.stats import spearmanr
    assert spearmanr(df["target"], s).correlation > 0.15


def t_fetchers_offline():
    """Extended-DB schema, insert/dedup, OpenAlex row mapping — all offline."""
    from workflow.src.iwf.fetchers import base as fbase
    from workflow.src.iwf.fetchers.openalex import reconstruct_abstract, work_to_row
    from workflow.src.iwf.fetchers.uefiscdi import normalize_journal

    inv = {"impact": [2], "Predicting": [0], "paper": [1]}
    assert reconstruct_abstract(inv) == "Predicting paper impact"
    assert reconstruct_abstract(None) is None

    work = {
        "id": "https://openalex.org/W123", "doi": "https://doi.org/10.1/AbC",
        "title": "T", "publication_year": 2020, "cited_by_count": 7, "fwci": 1.5,
        "abstract_inverted_index": inv,
        "citation_normalized_percentile": {"value": 0.9, "is_in_top_1_percent": False,
                                           "is_in_top_10_percent": True},
        "primary_location": {"source": {"display_name": "VenueX", "issn": ["1234-5678"]}},
        "primary_topic": {"domain": {"display_name": "Physical Sciences"},
                          "field": {"display_name": "Computer Science"},
                          "subfield": {"display_name": "AI"}},
        "authorships": [{"author": {"id": "https://openalex.org/A9", "display_name": "N"}}],
    }
    row = work_to_row(work)
    assert row["paper_id"] == "OA:W123" and row["doi"] == "10.1/abc"
    assert row["abstract"] == "Predicting paper impact" and row["venue"] == "VenueX"
    assert row["citation_normalized_percentile"] == 0.9
    assert row["authors"][0]["author_id"] == "A9"

    with tempfile.TemporaryDirectory() as d:
        conn = fbase.open_db(Path(d) / "x.db")
        dedup = fbase.Deduper(conn)
        assert not dedup.seen(row)
        fbase.insert_paper(conn, row)
        dedup.add(row)
        assert dedup.seen(row)
        assert dedup.seen({"paper_id": "other", "doi": "10.1/ABC"})     # DOI dedup
        assert dedup.seen({"paper_id": "other2", "title": "T", "year": 2020})
        n = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        assert n == 1
        fbase.set_cursor(conn, "openalex:all", "abc", 42)
        assert fbase.get_cursor(conn, "openalex:all") == "abc"
        conn.close()
    # parenthetical qualifiers like "(Print)"/"(Online)" are dropped on purpose
    assert normalize_journal("The Journal of AI-Systems (Print)") == "journal of ai systems"


def t_generated_sources_file():
    """generated_sources loads a local export and normalizes columns + paper ids."""
    from workflow.src.iwf.data import generated_sources
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "gen.csv"
        pd.DataFrame({"paper_title": ["G1", "G2"],
                      "summary": ["a1", "a2"]}).to_csv(p, index=False)
        gen = generated_sources.load_from_file(p, source="mirai")
    assert list(gen["title"]) == ["G1", "G2"]
    assert (gen["source"] == "mirai").all()
    assert gen["paper_id"].is_unique


def t_workflow_results_aggregate():
    """Per-workflow aggregate + cross-workflow comparison are produced from result.json."""
    from workflow.src.iwf import paths, workflow_results
    # emb_reg_1 (t_experiment_reg) and tfm_reg_1 (t_tfm_mock) seed folders exist already.
    summary = workflow_results.aggregate("emb_reg", paths.RESULTS_DIR / "emb_reg_1")
    assert summary is not None and summary["primary_metric"] == "spearman"
    assert (paths.RESULTS_DIR / "emb_reg_1" / "aggregate" / "metrics_summary.json").is_file()
    summary_t = workflow_results.aggregate("tfm_reg", paths.RESULTS_DIR / "tfm_reg_1")
    assert summary_t is not None and summary_t["n_seeds"] == 1
    comparison = workflow_results.cross_workflow_comparison()
    assert "emb_reg_1" in comparison and "tfm_reg_1" in comparison
    assert (paths.RESULTS_DIR / "summary" / "cross_workflow_comparison.json").is_file()


def t_auto_increment():
    """next_workflow_dir returns sequential _1, _2, _3 folders (no overwrites)."""
    from workflow.src.iwf import paths, workflow_results
    d1 = workflow_results.next_workflow_dir("test_wf")
    assert d1 == paths.RESULTS_DIR / "test_wf_1"
    d1.mkdir(parents=True, exist_ok=True)
    d2 = workflow_results.next_workflow_dir("test_wf")
    assert d2 == paths.RESULTS_DIR / "test_wf_2"
    d2.mkdir(parents=True, exist_ok=True)
    d3 = workflow_results.next_workflow_dir("test_wf")
    assert d3 == paths.RESULTS_DIR / "test_wf_3"
    # Different wf_type starts at _1 independently
    d_other = workflow_results.next_workflow_dir("other_wf")
    assert d_other == paths.RESULTS_DIR / "other_wf_1"


# ---------------------------------------------------------------- palette
def t_palette_category_of() -> None:
    from workflow.src.iwf.evaluation import palette

    cases = [
        ("target", "target"),
        ("ours", "target"),
        ("emb_clf", "emb_clf"),
        ("emb_clf_3", "emb_clf"),
        ("emb_reg_12", "emb_reg"),
        ("tfm_clf", "tfm_clf"),
        ("tfm_reg_5", "tfm_reg"),
        ("emb_meta_1", "meta"),
        ("emb_all_meta_2", "meta"),
        ("naip", "external"),
        ("naip_local", "external"),
        ("specter2_knn", "external"),
        ("llm_judge", "external"),
        ("venue_ais", "external"),
        ("forecite", "external"),
        ("cited_potential", "external"),
        ("unknown_model", "external"),
    ]
    for name, expected in cases:
        result = palette.category_of(name)
        assert result == expected, f"category_of('{name}') = '{result}', expected '{expected}'"


def t_palette_colors() -> None:
    from workflow.src.iwf.evaluation import palette

    # Test color_for returns hex
    for name in ["emb_clf", "emb_reg", "meta", "external", "target", "naip", "ours"]:
        color = palette.color_for(name)
        assert isinstance(color, str) and color.startswith("#"), \
            f"color_for('{name}') = {color}, expected hex string"

    # Test ordinal_colors
    for n in [1, 3, 6, 10]:
        colors = palette.ordinal_colors(n)
        assert len(colors) == n, f"ordinal_colors({n}) returned {len(colors)} colors"
        for c in colors:
            assert isinstance(c, str) and c.startswith("#"), f"Got non-hex color: {c}"

    # Test register_olive_seq
    name = palette.register_olive_seq()
    assert name == "olive_seq", f"Expected 'olive_seq', got '{name}'"

    # Test constants are hex
    for const in ["OLIVE", "SAGE", "SAND", "INK", "TERRACOTTA", "DUSK_BLUE", "MAUVE", "GOLD", "GRAY"]:
        color = getattr(palette, const)
        assert isinstance(color, str) and color.startswith("#"), \
            f"{const} = {color}, expected hex string"


def main() -> int:
    import shutil
    from workflow.src.iwf import paths

    print("Running offline smoke tests (no API keys)...\n")
    # Redirect experiment outputs to a throwaway dir so tests never touch real results/.
    _real_results = paths.RESULTS_DIR
    paths.RESULTS_DIR = Path(tempfile.mkdtemp(prefix="smoke_results_"))
    try:
        return _run_all()
    finally:
        shutil.rmtree(paths.RESULTS_DIR, ignore_errors=True)
        paths.RESULTS_DIR = _real_results


def _run_all() -> int:
    for name, fn in [
        ("config", t_config),
        ("config_overrides", t_config_overrides),
        ("expand_runs", t_expand_runs),
        ("identity", t_identity),
        ("split", t_split),
        ("targets", t_targets),
        ("classes", t_classes),
        ("metadata", t_metadata),
        ("feature_builder", t_feature_builder),
        ("intrinsic_dim", t_intrinsic_dim),
        ("encoders", t_encoders),
        ("dim_selector", t_dim_selector),
        ("models", t_models),
        ("architectures", t_architectures),
        ("metrics", t_metrics),
        ("novelty", t_novelty),
        ("novelty_kinds", t_novelty_kinds),
        ("feasibility_composite", t_feasibility_composite),
        ("baseline_metrics_and_plots", t_baseline_metrics_and_plots),
        ("nf_analysis", t_nf_analysis),
        ("external_model_registry", t_external_model_registry),
        ("fetchers_offline", t_fetchers_offline),
        ("generated_sources_file", t_generated_sources_file),
        ("experiment_reg", t_experiment_reg),
        ("experiment_meta_and_clf", t_experiment_meta_and_clf),
        ("saved_model_predicts", t_saved_model_predicts),
        ("validate", t_validate),
        ("workflow_table", t_workflow_table),
        ("registry", t_registry),
        ("progress", t_progress),
        ("tfm_mock", t_tfm_mock),
        ("workflow_results_aggregate", t_workflow_results_aggregate),
        ("auto_increment", t_auto_increment),
        ("palette_category_of", t_palette_category_of),
        ("palette_colors", t_palette_colors),
    ]:
        check(name, fn)

    print(f"\n{'=' * 50}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
