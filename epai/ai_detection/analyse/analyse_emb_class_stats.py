r"""Class geometry of the embeddings: why is in-domain detection ~98%?

The companion question to ``analyse_domain_pca.py``. That script *shows* the two
clouds; this one measures them, and specifically measures the one quantity the
scatter plot cannot show: how much of the in-domain score comes from a single
direction (one systematic shift between human and machine text) versus from the
767 other dimensions the classifier is free to combine.

Built for the ``*_gpt54mini_*`` parquets, which carry exactly two classes
(``human`` / ``gpt_plain``) and — because the machine texts are regenerations of
the same source documents — are PAIRED by ``id``. The pairing is the strongest
lever available here and section 4 below exploits it.

WHAT IS COMPUTED, PER CORPUS
----------------------------
1. Per-class summary — n, vector norms, centroid norm, mean per-dimension std,
   mean distance to centroid. Cosine discards magnitude, so norms are reported
   separately: for embedders that do not emit unit vectors, and corpora with a
   human/machine length gap, the norm can carry signal on its own.

2. Cosine distance populations — within-human, within-machine, and between.
   This is the honest version of "how far apart are the classes". If the
   between-class distances barely exceed the within-class ones, the clouds
   overlap heavily in the full space *despite* a 98% classifier, which means the
   score lives in a thin direction rather than in separated clouds.
   Also reported as trace(S_B)/trace(S_W), the Fisher ratio.

3. The centroid direction — every point projected onto the unit vector
   Δ = μ_machine − μ_human, reduced to one number per document. Cohen's d and
   AUC on that projection say how much separation a SINGLE direction buys.
   Section 5 turns this into an accuracy directly comparable to the classifier.

4. Paired difference vectors — δᵢ = emb(machineᵢ) − emb(humanᵢ) over shared
   ``id``s. Pairing cancels topic variance, which is the dominant noise in 2 and
   3. The number to read is the concentration ‖δ̄‖ / mean‖δᵢ‖:
       near 1  -> a near-rigid translation of the whole cloud; every document
                  moves the same way. The signature of a mechanical artifact.
       near 0  -> each document changes idiosyncratically.

5. Separability, on the SAME grouped split the real experiment uses
   (``GroupShuffleSplit`` on ``id``, ``StandardScaler`` fit on train only,
   ``LogisticRegression``). Three models on identical folds:
       full      — logistic on all dimensions; reproduces the diagonal number
       centroid  — threshold on the 1-D Δ projection (Δ and threshold both
                   fitted on train only, so this is honestly held out)
       shuffled  — logistic on permuted labels; a null / leakage control
   `full` minus `centroid` is the payload: it is the share of the in-domain
   score that a single systematic shift does NOT explain.

   NOTE ON THE SHUFFLE CONTROL — it is a leakage check, not a capacity check.
   Because accuracy is measured on a held-out grouped split, high-dimensional
   capacity (d ≳ n, where a separating hyperplane exists for almost any
   labelling) inflates TRAIN accuracy only; it cannot by itself produce a high
   held-out score. Train accuracy is therefore reported alongside test. A
   shuffled score near 50% says the grouped split is doing its job and no ``id``
   leaks across it; a shuffled score well above 50% means the split is broken
   and nothing else in this report can be trusted.

6. Across corpora — the pairwise cosine matrix over every (corpus, class)
   centroid, and the cosine matrix between the corpora's Δ directions. The
   second one is the transfer diagnostic: a NEGATIVE cos(Δ_c, Δ_c') means the
   two corpora disagree about which way "machine" points, which is what a
   below-chance transfer cell looks like from the geometry side.

TWO SPACES
----------
``run_clf_cross_domain.py`` fits a ``StandardScaler`` on the training split, so
the classifier does not see raw embedding geometry — it sees per-dimension
standardized space, a diagonal whitening that UPWEIGHTS low-variance
dimensions. A token-level artifact (a consistent nudge in a few dimensions that
otherwise barely move) is exactly the thing that transformation amplifies.
Sections 1-4 are therefore reported in both spaces: a separation that is
mediocre raw and large standardized localizes the mechanism.

The scaler used for the DESCRIPTIVE sections is fitted on the whole corpus,
which is fine for a description of the data but is not a held-out quantity.
Section 5 refits per split, train-only, and is the one to quote.

Outputs saved to results/emb_class_stats/<name>/:
  - emb_class_stats.log            — everything printed to the terminal
  - class_summary.csv              — per (corpus, class, space) statistics
  - distance_summary.csv           — within/between cosine distance populations
  - direction_summary.csv          — Δ effect size, paired-δ concentration
  - separability.csv               — full vs centroid vs shuffled accuracies
  - centroid_cosine.csv            — (corpus, class) x (corpus, class) matrix
  - delta_cosine.csv               — corpus x corpus Δ-direction matrix
  - *.png                          — the figures for the above

Usage:
    # one combined parquet = three corpora, one per domain
    uv run python epai/ai_detection/analyse/analyse_emb_class_stats.py \
        data/parquet/ghostbuster_gpt54mini.parquet

    # one domain of it, with a different embedder (the geometry changes with it)
    uv run python epai/ai_detection/analyse/analyse_emb_class_stats.py \
        data/parquet/ghostbuster_gpt54mini.parquet --domains wp \
        --emb-col emb_luar_crud

    # does the separation survive truncation? run the length ablation variants
    # and compare the `centroid` / `full` accuracies across them
    uv run python epai/ai_detection/analyse/analyse_emb_class_stats.py \
        data/parquet/wp_gpt54mini_binox0_red_200w_binox0.parquet \
        --name wp_200w
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from epai.utils.color_utils import group_color
from epai.utils.corpus_utils import CorpusSpec, read_corpus, specs_from_paths
from epai.utils.train_utils import CLF_TARGET_COL, drop_missing_rows

BASE_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "emb_class_stats"

ID_COL = "id"
GROUP_COL = CLF_TARGET_COL  # "generated_by"

HUMAN_LABEL = "human"
DEFAULT_EMB_COL = "emb_specter2"

# Group *families*: a generated_by label is kept when it starts with one of
# these, so gpt_plain counts as gpt (same longest-prefix idea as cfg/colors.yaml).
DEFAULT_FAMILIES = ("human", "gpt")

# The two spaces every descriptive section is computed in. See the module
# docstring: "std" is the geometry the classifier actually sees.
SPACES = ("raw", "std")
SPACE_LABELS = {"raw": "raw embedding", "std": "per-dim standardized"}

# Cosine-distance populations, in plot/report order.
POPULATIONS = ("within_human", "within_machine", "between")
POPULATION_LABELS = {
    "within_human": "within human",
    "within_machine": "within machine",
    "between": "between classes",
}
POPULATION_COLORS = {
    "within_human": "#2563eb",  # blue
    "within_machine": "#dc2626",  # red
    "between": "#6b7280",  # grey
}

# Models compared on identical folds in the separability experiment.
MODELS = ("full", "centroid", "shuffled")
MODEL_LABELS = {
    "full": "logistic (all dims)",
    "centroid": "threshold on Δ (1-D)",
    "shuffled": "logistic, shuffled labels",
}
MODEL_COLORS = {"full": "#7c3aed", "centroid": "#059669", "shuffled": "#9ca3af"}

PERCENTILES = (5, 25, 50, 75, 95)


class Tee:
    """Collect printed lines so they can be echoed to stdout and a log file."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def print(self, line: str = "") -> None:
        print(line)
        self._lines.append(line)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._lines) + "\n")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class Corpus:
    """One parquet, reduced to the human-vs-machine view this script analyses.

    ``X`` is the embedding matrix, ``y`` is 0 for human / 1 for machine (the same
    binarisation the cross-domain configs use), ``ids`` is the group column that
    pairs a human original with its machine rewrite.
    """

    def __init__(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        ids: np.ndarray,
        machine_labels: list[str],
    ) -> None:
        self.name = name
        self.X = X
        self.y = y
        self.ids = ids
        self.machine_labels = machine_labels
        # Standardized once per corpus, for the descriptive sections only.
        self.X_std = StandardScaler().fit_transform(X)

    def space(self, space: str) -> np.ndarray:
        return self.X if space == "raw" else self.X_std

    @property
    def n_human(self) -> int:
        return int((self.y == 0).sum())

    @property
    def n_machine(self) -> int:
        return int((self.y == 1).sum())


def select_groups(
    df: pd.DataFrame, families: tuple[str, ...], out: Tee
) -> tuple[pd.DataFrame, list[str]]:
    """Keep only rows whose ``generated_by`` belongs to one of ``families``."""
    labels = sorted(str(g) for g in df[GROUP_COL].dropna().unique())

    kept: list[str] = []
    for fam in families:
        kept.extend(lbl for lbl in labels if lbl.lower().startswith(fam.lower()))
    kept = list(dict.fromkeys(kept))  # dedupe, keep family order

    dropped = [lbl for lbl in labels if lbl not in kept]
    out.print(f"  classes found:   {labels}")
    out.print(f"  classes kept:    {kept}")
    if dropped:
        out.print(f"  classes dropped: {dropped}")

    if not kept:
        raise SystemExit(
            f"None of the families {list(families)} matched any {GROUP_COL} "
            f"value (available: {labels})."
        )
    return df[df[GROUP_COL].isin(kept)].copy(), kept


def load_corpus(
    spec: CorpusSpec, emb_col: str, families: tuple[str, ...], out: Tee
) -> Corpus:
    """Read one corpus -- a parquet, or one domain of it -- into the binary view."""
    path = spec.path
    out.print(f"\n{spec.source}")
    df = read_corpus(spec, quiet=True).df
    out.print(f"  loaded {len(df)} rows")

    for col in (emb_col, GROUP_COL, ID_COL):
        if col not in df.columns:
            raise SystemExit(
                f"Column {col!r} missing from {path.name}. "
                f"Available: {sorted(df.columns)}"
            )

    df, kept = select_groups(df, families, out)
    # A missing embedding takes down its whole id-group, so the human original
    # and its rewrite never get separated (same rule as the training pipeline).
    df = drop_missing_rows(df, [emb_col], group_col=ID_COL)

    machine_labels = [lbl for lbl in kept if not lbl.lower().startswith(HUMAN_LABEL)]
    if not machine_labels:
        raise SystemExit(f"{path.name} has no machine class among {kept}.")
    if not any(lbl.lower().startswith(HUMAN_LABEL) for lbl in kept):
        raise SystemExit(f"{path.name} has no {HUMAN_LABEL!r} class among {kept}.")
    if len(machine_labels) > 1:
        out.print(
            f"  note: {len(machine_labels)} machine classes {machine_labels} are "
            "pooled into one 'machine' class (the binarisation the cross-domain "
            "configs use). Pass --groups to narrow this."
        )

    X = np.array(df[emb_col].tolist(), dtype=np.float64)
    y = (~df[GROUP_COL].str.lower().str.startswith(HUMAN_LABEL)).to_numpy().astype(int)
    ids = df[ID_COL].to_numpy()

    out.print(f"  usable: {len(df)} rows, {X.shape[1]} dims, "
              f"human={int((y == 0).sum())} machine={int((y == 1).sum())}")
    return Corpus(spec.name, X, y, ids, machine_labels)


# ---------------------------------------------------------------------------
# 1. Per-class summary
# ---------------------------------------------------------------------------


def _unit(X: np.ndarray) -> np.ndarray:
    """Row-normalize, leaving zero rows alone."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def class_summary(corpus: Corpus, space: str) -> list[dict]:
    """n, norms, spread and centroid geometry for each class."""
    X = corpus.space(space)
    rows = []
    for cls, label in ((0, HUMAN_LABEL), (1, "machine")):
        Xc = X[corpus.y == cls]
        if Xc.size == 0:
            continue
        centroid = Xc.mean(axis=0)
        to_centroid = np.linalg.norm(Xc - centroid, axis=1)
        norms = np.linalg.norm(Xc, axis=1)
        rows.append(
            {
                "corpus": corpus.name,
                "space": space,
                "class": label,
                "n": int(Xc.shape[0]),
                "dim": int(Xc.shape[1]),
                "norm_mean": float(norms.mean()),
                "norm_std": float(norms.std(ddof=1)) if norms.size > 1 else 0.0,
                "centroid_norm": float(np.linalg.norm(centroid)),
                # Mean per-dimension std: the typical size of one coordinate's
                # spread. Squared and summed it is the within-class scatter.
                "per_dim_std_mean": float(Xc.std(axis=0, ddof=1).mean())
                if Xc.shape[0] > 1
                else 0.0,
                "dist_to_centroid_mean": float(to_centroid.mean()),
                "dist_to_centroid_std": float(to_centroid.std(ddof=1))
                if to_centroid.size > 1
                else 0.0,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 2. Cosine distance populations + Fisher ratio
# ---------------------------------------------------------------------------


def sample_cosine_distances(
    A: np.ndarray,
    B: np.ndarray | None,
    rng: np.random.Generator,
    max_pairs: int,
) -> np.ndarray:
    """Cosine distances over pairs, exhaustive when cheap, sampled when not.

    ``B is None`` means within-set pairs (i < j); otherwise every pair is one
    row from ``A`` and one from ``B``.
    """
    within = B is None
    U = _unit(A)
    V = U if within else _unit(B)
    na, nb = U.shape[0], V.shape[0]

    total = na * (na - 1) // 2 if within else na * nb
    if total <= 0:
        return np.empty(0)

    if total <= max_pairs:
        if within:
            i, j = np.triu_indices(na, k=1)
        else:
            i = np.repeat(np.arange(na), nb)
            j = np.tile(np.arange(nb), na)
    elif within:
        # Oversample and drop the i == j hits rather than looping.
        i = rng.integers(0, na, max_pairs * 2)
        j = rng.integers(0, na, max_pairs * 2)
        keep = i != j
        i, j = i[keep][:max_pairs], j[keep][:max_pairs]
    else:
        i = rng.integers(0, na, max_pairs)
        j = rng.integers(0, nb, max_pairs)

    return 1.0 - np.sum(U[i] * V[j], axis=1)


def distance_populations(
    corpus: Corpus, space: str, rng: np.random.Generator, max_pairs: int
) -> dict[str, np.ndarray]:
    """The three cosine-distance populations: within each class, and between."""
    X = corpus.space(space)
    H, M = X[corpus.y == 0], X[corpus.y == 1]
    return {
        "within_human": sample_cosine_distances(H, None, rng, max_pairs),
        "within_machine": sample_cosine_distances(M, None, rng, max_pairs),
        "between": sample_cosine_distances(H, M, rng, max_pairs),
    }


def fisher_ratio(corpus: Corpus, space: str) -> float:
    """trace(S_B) / trace(S_W) — between-class scatter over within-class scatter.

    The traces are used rather than the full 768x768 scatter matrices on
    purpose: with ~1000 rows per class those matrices are singular (rank <= n-1),
    so anything requiring an inverse — Mahalanobis, LDA, a whitened separation
    — is not estimable here without shrinkage. The traces are.
    """
    X = corpus.space(space)
    grand = X.mean(axis=0)
    n = X.shape[0]

    s_w = 0.0
    s_b = 0.0
    for cls in (0, 1):
        Xc = X[corpus.y == cls]
        if Xc.size == 0:
            continue
        mu = Xc.mean(axis=0)
        s_w += float(np.sum((Xc - mu) ** 2))
        s_b += float(Xc.shape[0] * np.sum((mu - grand) ** 2))
    return s_b / s_w if s_w > 0 else float("nan")


# ---------------------------------------------------------------------------
# 3. The centroid direction
# ---------------------------------------------------------------------------


def delta_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unit vector from the human centroid to the machine centroid."""
    delta = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
    norm = np.linalg.norm(delta)
    return delta / norm if norm > 0 else delta


def direction_stats(corpus: Corpus, space: str) -> dict:
    """Effect size of the single best-known direction, Δ = μ_machine − μ_human.

    Descriptive only: Δ is fitted on all rows here. The held-out version is in
    the separability experiment.
    """
    X = corpus.space(space)
    d_hat = delta_direction(X, corpus.y)
    proj = X @ d_hat

    p_h, p_m = proj[corpus.y == 0], proj[corpus.y == 1]
    n_h, n_m = p_h.size, p_m.size
    delta_norm = float(
        np.linalg.norm(X[corpus.y == 1].mean(axis=0) - X[corpus.y == 0].mean(axis=0))
    )
    if n_h < 2 or n_m < 2:
        return {
            "delta_norm": delta_norm,
            "cohens_d": float("nan"),
            "auc": float("nan"),
            "proj_human": p_h,
            "proj_machine": p_m,
        }

    v_h, v_m = p_h.var(ddof=1), p_m.var(ddof=1)
    pooled = np.sqrt(((n_h - 1) * v_h + (n_m - 1) * v_m) / (n_h + n_m - 2))
    cohens_d = (p_m.mean() - p_h.mean()) / pooled if pooled > 0 else float("nan")

    return {
        "delta_norm": delta_norm,
        "cohens_d": float(cohens_d),
        "auc": float(roc_auc_score(corpus.y, proj)),
        "proj_human": p_h,
        "proj_machine": p_m,
    }


# ---------------------------------------------------------------------------
# 4. Paired difference vectors
# ---------------------------------------------------------------------------


def paired_deltas(corpus: Corpus, space: str) -> np.ndarray:
    """δᵢ = mean machine embedding − mean human embedding, per shared ``id``.

    Ids carrying only one of the two classes are skipped. Where an id has
    several rows on a side (a corpus with more than one generator), that side is
    averaged first, so every id contributes exactly one δ.
    """
    X = corpus.space(space)
    df = pd.DataFrame({"id": corpus.ids, "y": corpus.y})

    deltas = []
    for _, idx in df.groupby("id").groups.items():
        idx = np.asarray(idx)
        ys = corpus.y[idx]
        if not (ys == 0).any() or not (ys == 1).any():
            continue
        deltas.append(X[idx[ys == 1]].mean(axis=0) - X[idx[ys == 0]].mean(axis=0))

    return np.array(deltas) if deltas else np.empty((0, X.shape[1]))


def paired_stats(deltas: np.ndarray) -> dict:
    """Is the human->machine move one shared shift, or per-document noise?"""
    if deltas.shape[0] < 2:
        return {
            "n_pairs": int(deltas.shape[0]),
            "concentration": float("nan"),
            "cos_to_mean_mean": float("nan"),
            "cos_to_mean_std": float("nan"),
            "frac_positive_cos": float("nan"),
            "cos_to_mean": np.empty(0),
        }

    mean_delta = deltas.mean(axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    mean_norm = float(np.linalg.norm(mean_delta))

    # ‖mean δ‖ / mean‖δ‖: 1 when every document moves the same way, ~0 when the
    # individual moves cancel.
    concentration = mean_norm / float(norms.mean()) if norms.mean() > 0 else float("nan")

    unit_mean = mean_delta / mean_norm if mean_norm > 0 else mean_delta
    cos_to_mean = _unit(deltas) @ unit_mean

    return {
        "n_pairs": int(deltas.shape[0]),
        "delta_norm_mean": float(norms.mean()),
        "mean_delta_norm": mean_norm,
        "concentration": float(concentration),
        "cos_to_mean_mean": float(cos_to_mean.mean()),
        "cos_to_mean_std": float(cos_to_mean.std(ddof=1)),
        "frac_positive_cos": float((cos_to_mean > 0).mean()),
        "cos_to_mean": cos_to_mean,
    }


# ---------------------------------------------------------------------------
# 5. Separability on the real split
# ---------------------------------------------------------------------------


def best_threshold(proj: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Threshold on a 1-D score maximizing accuracy, and that accuracy.

    Decision rule is ``proj > t -> machine``. Evaluated at every gap between
    consecutive sorted values, vectorized rather than looped.
    """
    order = np.argsort(proj, kind="mergesort")
    ps, ys = proj[order], y[order]
    n = ys.size
    n_pos = int(ys.sum())

    # k = number of smallest values predicted human, for k = 1..n
    cum_pos = np.cumsum(ys)  # positives that would be wrongly called human
    k = np.arange(1, n + 1)
    acc = ((k - cum_pos) + (n_pos - cum_pos)) / n  # correct humans + correct machines

    acc_all_machine = n_pos / n  # the k = 0 case: everything above the threshold
    best = int(np.argmax(acc))
    if acc_all_machine >= acc[best]:
        return float(ps[0] - 1.0), float(acc_all_machine)

    t = ps[best] if best == n - 1 else 0.5 * (ps[best] + ps[best + 1])
    return float(t), float(acc[best])


def separability(
    corpus: Corpus, seed: int, n_runs: int, test_size: float, out: Tee
) -> list[dict]:
    """Full-dim logistic vs a 1-D centroid threshold vs a shuffled-label null.

    Mirrors run_clf_cross_domain.py: GroupShuffleSplit on ``id`` so a document
    and its rewrite never straddle the cut, StandardScaler fitted on train only,
    LogisticRegression(max_iter=1000). Every model sees identical folds.
    """
    gss = GroupShuffleSplit(n_splits=n_runs, test_size=test_size, random_state=seed)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for run, (train_idx, test_idx) in enumerate(
        gss.split(corpus.X, corpus.y, groups=corpus.ids)
    ):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(corpus.X[train_idx])
        X_te = scaler.transform(corpus.X[test_idx])
        y_tr, y_te = corpus.y[train_idx], corpus.y[test_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            out.print(f"  run {run}: a split lost a class, skipped.")
            continue

        # full — all dimensions
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_tr, y_tr)
        rows.append(
            {
                "corpus": corpus.name,
                "run": run,
                "model": "full",
                "train_acc": float(accuracy_score(y_tr, clf.predict(X_tr))),
                "test_acc": float(accuracy_score(y_te, clf.predict(X_te))),
                "n_train": int(y_tr.size),
                "n_test": int(y_te.size),
            }
        )

        # centroid — one direction, fitted on train only
        d_hat = delta_direction(X_tr, y_tr)
        proj_tr, proj_te = X_tr @ d_hat, X_te @ d_hat
        t, acc_tr = best_threshold(proj_tr, y_tr)
        rows.append(
            {
                "corpus": corpus.name,
                "run": run,
                "model": "centroid",
                "train_acc": acc_tr,
                "test_acc": float(accuracy_score(y_te, (proj_te > t).astype(int))),
                "n_train": int(y_tr.size),
                "n_test": int(y_te.size),
            }
        )

        # shuffled — null / leakage control
        y_perm = rng.permutation(y_tr)
        clf_null = LogisticRegression(max_iter=1000, random_state=seed)
        clf_null.fit(X_tr, y_perm)
        rows.append(
            {
                "corpus": corpus.name,
                "run": run,
                "model": "shuffled",
                "train_acc": float(accuracy_score(y_perm, clf_null.predict(X_tr))),
                "test_acc": float(accuracy_score(y_te, clf_null.predict(X_te))),
                "n_train": int(y_tr.size),
                "n_test": int(y_te.size),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# 6. Across corpora
# ---------------------------------------------------------------------------


def centroid_cosine_matrix(
    corpora: list[Corpus], space: str
) -> tuple[np.ndarray, list[str]]:
    """Pairwise cosine similarity between every (corpus, class) centroid."""
    centroids, names = [], []
    for c in corpora:
        X = c.space(space)
        for cls, label in ((0, HUMAN_LABEL), (1, "machine")):
            Xc = X[c.y == cls]
            if Xc.size == 0:
                continue
            centroids.append(Xc.mean(axis=0))
            names.append(f"{c.name}/{label}")
    U = _unit(np.array(centroids))
    return U @ U.T, names


def delta_cosine_matrix(
    corpora: list[Corpus], space: str
) -> tuple[np.ndarray, list[str]]:
    """Cosine between each corpus's Δ direction — the transfer diagnostic.

    A negative entry means two corpora disagree about which way "machine"
    points: a model trained on one is systematically inverted on the other,
    which is what a below-chance transfer cell looks like from here.
    """
    deltas = [delta_direction(c.space(space), c.y) for c in corpora]
    U = _unit(np.array(deltas))
    return U @ U.T, [c.name for c in corpora]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _finish(fig: plt.Figure, fig_dir: Path, fname: str) -> None:
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  [OK] {fname}")


def _panels(n: int, title: str, width: float = 5.2, height: float = 4.2):
    fig, axes = plt.subplots(1, n, figsize=(width * n, height), squeeze=False)
    fig.suptitle(title, fontsize=13)
    return fig, list(axes[0])


def plot_distance_populations(
    per_corpus: dict[str, dict[str, np.ndarray]], space: str, fig_dir: Path
) -> None:
    """Within-class vs between-class cosine distance, one panel per corpus."""
    names = list(per_corpus)
    fig, axes = _panels(
        len(names),
        f"Cosine distance: within class vs between classes — {SPACE_LABELS[space]}",
    )

    for ax, name in zip(axes, names):
        pops = per_corpus[name]
        pooled = np.concatenate([v for v in pops.values() if v.size])
        if pooled.size == 0:
            ax.axis("off")
            continue
        bins = np.linspace(*np.percentile(pooled, [0.5, 99.5]), 50)

        for pop in POPULATIONS:
            values = pops[pop]
            if values.size == 0:
                continue
            ax.hist(
                values, bins=bins, density=True, alpha=0.5,
                color=POPULATION_COLORS[pop],
                label=f"{POPULATION_LABELS[pop]} (μ={values.mean():.3f})",
            )
            ax.axvline(values.mean(), color=POPULATION_COLORS[pop],
                       linestyle="--", linewidth=1.4)

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("cosine distance")
        ax.set_ylabel("density")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    _finish(fig, fig_dir, f"distance_populations_{space}.png")


def plot_delta_projection(
    per_corpus: dict[str, dict], space: str, fig_dir: Path
) -> None:
    """Every document as one number: its projection on Δ, split by class."""
    names = list(per_corpus)
    fig, axes = _panels(
        len(names),
        f"Projection on Δ = μ_machine − μ_human (1-D) — {SPACE_LABELS[space]}",
    )

    for ax, name in zip(axes, names):
        st = per_corpus[name]
        p_h, p_m = st.get("proj_human"), st.get("proj_machine")
        if p_h is None or p_h.size == 0 or p_m is None or p_m.size == 0:
            ax.axis("off")
            continue
        pooled = np.concatenate([p_h, p_m])
        bins = np.linspace(pooled.min(), pooled.max(), 45)

        ax.hist(p_h, bins=bins, density=True, alpha=0.55,
                color=group_color(HUMAN_LABEL), label=f"human (n={p_h.size})")
        ax.hist(p_m, bins=bins, density=True, alpha=0.55,
                color=group_color("gpt"), label=f"machine (n={p_m.size})")

        ax.set_title(
            f"{name}\nCohen's d = {st['cohens_d']:.2f}, AUC = {st['auc']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("projection on Δ")
        ax.set_ylabel("density")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    _finish(fig, fig_dir, f"delta_projection_{space}.png")


def plot_paired_deltas(
    per_corpus: dict[str, dict], space: str, fig_dir: Path
) -> None:
    """Do all documents move the same way? cos(δᵢ, δ̄) per matched pair."""
    names = list(per_corpus)
    fig, axes = _panels(
        len(names),
        f"Paired shift δᵢ = machineᵢ − humanᵢ, alignment to the mean shift — "
        f"{SPACE_LABELS[space]}",
    )

    for ax, name in zip(axes, names):
        st = per_corpus[name]
        cos = st.get("cos_to_mean")
        if cos is None or cos.size == 0:
            ax.axis("off")
            continue

        ax.hist(cos, bins=np.linspace(-1, 1, 60), color="#7c3aed", alpha=0.75)
        ax.axvline(0, color="black", linestyle=":", linewidth=1)
        ax.axvline(float(cos.mean()), color="#dc2626", linestyle="--", linewidth=1.5,
                   label=f"mean = {cos.mean():.3f}")
        ax.set_title(
            f"{name}\nconcentration ‖δ̄‖/mean‖δ‖ = {st['concentration']:.3f} "
            f"(n={st['n_pairs']} pairs)",
            fontsize=10,
        )
        ax.set_xlabel("cos(δᵢ, δ̄)")
        ax.set_ylabel("pairs")
        ax.set_xlim(-1, 1)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    _finish(fig, fig_dir, f"paired_deltas_{space}.png")


def _heatmap(
    ax: plt.Axes, M: np.ndarray, names: list[str], vmin: float, vmax: float, cmap: str
) -> None:
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(M[i, j]) > 0.6 * max(abs(vmin), abs(vmax))
                    else "black")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_matrices(
    centroid_M: np.ndarray,
    centroid_names: list[str],
    delta_M: np.ndarray,
    delta_names: list[str],
    space: str,
    fig_dir: Path,
) -> None:
    """Centroid cosine over (corpus, class), and Δ-direction cosine over corpora."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0 + 0.5 * len(centroid_names), 6.0))
    fig.suptitle(f"Centroid and Δ-direction geometry — {SPACE_LABELS[space]}", fontsize=13)

    _heatmap(axes[0], centroid_M, centroid_names, -1.0, 1.0, "coolwarm")
    axes[0].set_title("cosine between class centroids", fontsize=10)

    _heatmap(axes[1], delta_M, delta_names, -1.0, 1.0, "coolwarm")
    axes[1].set_title("cosine between Δ directions\n(negative = inverted transfer)",
                      fontsize=10)

    _finish(fig, fig_dir, f"cosine_matrices_{space}.png")


def plot_separability(sep: pd.DataFrame, fig_dir: Path) -> None:
    """The payload figure: full-dim vs 1-D vs null, on identical held-out folds."""
    names = list(dict.fromkeys(sep["corpus"]))
    fig, ax = plt.subplots(figsize=(2.6 * len(names) + 3.5, 4.6))

    width = 0.25
    x = np.arange(len(names))
    for k, model in enumerate(MODELS):
        sub = sep[sep["model"] == model]
        means = [sub.loc[sub["corpus"] == n, "test_acc"].mean() for n in names]
        errs = [sub.loc[sub["corpus"] == n, "test_acc"].std(ddof=0) for n in names]
        bars = ax.bar(x + (k - 1) * width, means, width, yerr=errs, capsize=3,
                      color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

    ax.axhline(0.5, color="black", linestyle=":", linewidth=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("held-out accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title(
        "In-domain separability on the grouped split\n"
        "gap between the first two bars = what a single direction does NOT explain",
        fontsize=11,
    )
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    _finish(fig, fig_dir, "separability.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _dist_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n_pairs": 0, "mean": float("nan"), "std": float("nan")} | {
            f"p{p}": float("nan") for p in PERCENTILES
        }
    s = {
        "n_pairs": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    }
    for p, q in zip(PERCENTILES, np.percentile(values, PERCENTILES)):
        s[f"p{p}"] = float(q)
    return s


def report_header(paths: list[Path], emb_col: str, args, out: Tee) -> None:
    out.print("=" * 78)
    out.print("Embedding class geometry — why is in-domain detection so high?")
    out.print(f"Generated:  {datetime.now():%Y-%m-%d %H:%M:%S}")
    out.print(f"Embedding:  {emb_col}")
    out.print(f"Parquets:   {len(paths)}")
    for p in paths:
        out.print(f"              {p}")
    out.print(f"Split:      GroupShuffleSplit on '{ID_COL}', "
              f"test_size={args.test_size}, n_runs={args.n_runs}, seed={args.seed}")
    out.print("=" * 78)


def report_class_summary(rows: list[dict], out: Tee) -> None:
    out.print("\n" + "-" * 78)
    out.print("1. PER-CLASS SUMMARY")
    out.print("-" * 78)
    header = (f"  {'corpus':<28}{'space':<6}{'class':<9}{'n':>6}{'‖x‖ mean':>11}"
              f"{'‖μ‖':>10}{'per-dim σ':>11}{'dist→μ':>10}")
    out.print(header)
    out.print("  " + "-" * (len(header) - 2))
    for r in rows:
        out.print(
            f"  {r['corpus'][:27]:<28}{r['space']:<6}{r['class']:<9}{r['n']:>6}"
            f"{r['norm_mean']:>11.3f}{r['centroid_norm']:>10.3f}"
            f"{r['per_dim_std_mean']:>11.4f}{r['dist_to_centroid_mean']:>10.3f}"
        )


def report_distances(rows: list[dict], out: Tee) -> None:
    out.print("\n" + "-" * 78)
    out.print("2. COSINE DISTANCE: WITHIN CLASS vs BETWEEN CLASSES")
    out.print("   Between barely above within => the clouds overlap and the")
    out.print("   classifier's score lives in a thin direction, not in separation.")
    out.print("-" * 78)
    header = (f"  {'corpus':<28}{'space':<6}{'population':<16}{'pairs':>9}"
              f"{'mean':>9}{'std':>9}{'p5':>9}{'p50':>9}{'p95':>9}")
    out.print(header)
    out.print("  " + "-" * (len(header) - 2))
    for r in rows:
        out.print(
            f"  {r['corpus'][:27]:<28}{r['space']:<6}{r['population']:<16}"
            f"{r['n_pairs']:>9}{r['mean']:>9.4f}{r['std']:>9.4f}"
            f"{r['p5']:>9.4f}{r['p50']:>9.4f}{r['p95']:>9.4f}"
        )

    out.print("\n  separation ratio and Fisher ratio per corpus / space:")
    seen = set()
    for r in rows:
        key = (r["corpus"], r["space"])
        if key in seen or r["population"] != "between":
            continue
        seen.add(key)
        within = [
            x["mean"] for x in rows
            if (x["corpus"], x["space"]) == key and x["population"].startswith("within")
        ]
        mean_within = float(np.mean(within)) if within else float("nan")
        out.print(
            f"    {r['corpus'][:27]:<28}{r['space']:<6}"
            f"between/within = {r['mean'] / mean_within:.4f}   "
            f"trace(S_B)/trace(S_W) = {r['fisher_ratio']:.4f}"
        )


def report_directions(rows: list[dict], out: Tee) -> None:
    out.print("\n" + "-" * 78)
    out.print("3/4. THE Δ DIRECTION, AND THE PAIRED SHIFT")
    out.print("   concentration ‖δ̄‖/mean‖δ‖ near 1 = every document moves the same")
    out.print("   way: a near-rigid translation, the signature of an artifact.")
    out.print("-" * 78)
    header = (f"  {'corpus':<28}{'space':<6}{'‖Δ‖':>9}{'Cohen d':>10}{'AUC':>8}"
              f"{'pairs':>8}{'concentr':>10}{'cos(δ,δ̄)':>11}{'frac>0':>9}")
    out.print(header)
    out.print("  " + "-" * (len(header) - 2))
    for r in rows:
        out.print(
            f"  {r['corpus'][:27]:<28}{r['space']:<6}{r['delta_norm']:>9.3f}"
            f"{r['cohens_d']:>10.3f}{r['auc']:>8.3f}{r['n_pairs']:>8}"
            f"{r['concentration']:>10.3f}{r['cos_to_mean_mean']:>11.3f}"
            f"{r['frac_positive_cos']:>9.3f}"
        )


def report_separability(sep: pd.DataFrame, out: Tee) -> None:
    out.print("\n" + "-" * 78)
    out.print("5. SEPARABILITY ON THE GROUPED SPLIT  (the number to quote)")
    out.print("   full - centroid = the share of in-domain accuracy that a single")
    out.print("   systematic shift does NOT explain.")
    out.print("   shuffled ~ 0.50 confirms the grouped split leaks no ids; it is a")
    out.print("   leakage check, not a capacity check (see the module docstring).")
    out.print("-" * 78)
    header = (f"  {'corpus':<28}{'model':<26}{'train acc':>11}{'test acc':>11}"
              f"{'± std':>9}")
    out.print(header)
    out.print("  " + "-" * (len(header) - 2))

    for name in dict.fromkeys(sep["corpus"]):
        sub = sep[sep["corpus"] == name]
        for model in MODELS:
            m = sub[sub["model"] == model]
            if m.empty:
                continue
            out.print(
                f"  {name[:27]:<28}{MODEL_LABELS[model]:<26}"
                f"{m['train_acc'].mean():>11.4f}{m['test_acc'].mean():>11.4f}"
                f"{m['test_acc'].std(ddof=0):>9.4f}"
            )
        full = sub.loc[sub["model"] == "full", "test_acc"].mean()
        cen = sub.loc[sub["model"] == "centroid", "test_acc"].mean()
        out.print(f"  {'':<28}→ full − centroid = {full - cen:+.4f}")


def report_matrices(
    centroid_M: np.ndarray,
    centroid_names: list[str],
    delta_M: np.ndarray,
    delta_names: list[str],
    space: str,
    out: Tee,
) -> None:
    out.print("\n" + "-" * 78)
    out.print(f"6. ACROSS CORPORA  ({SPACE_LABELS[space]})")
    out.print("-" * 78)

    out.print("\n  cosine between (corpus, class) centroids:")
    width = max(len(n) for n in centroid_names) + 2
    out.print("    " + " " * width + "".join(f"{i:>8}" for i in range(len(centroid_names))))
    for i, name in enumerate(centroid_names):
        out.print(f"    {name:<{width}}" + "".join(f"{v:>8.3f}" for v in centroid_M[i]))

    out.print("\n  cosine between Δ directions — NEGATIVE means the two corpora")
    out.print("  disagree on which way 'machine' points (inverted transfer):")
    width = max(len(n) for n in delta_names) + 2
    out.print("    " + " " * width + "".join(f"{i:>8}" for i in range(len(delta_names))))
    for i, name in enumerate(delta_names):
        out.print(f"    {name:<{width}}" + "".join(f"{v:>8.3f}" for v in delta_M[i]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("parquets", nargs="+", type=Path,
                   help="Parquet file(s) to analyse. A combined file becomes one "
                        "corpus per domain unless --domains narrows it.")
    p.add_argument("--domains", nargs="*", default=None, metavar="DOMAIN",
                   help="Domains to take from each parquet (default: every domain "
                        "the file has; a file with no domain column is used whole).")
    p.add_argument("--emb-col", default=DEFAULT_EMB_COL,
                   help="Embedding column to analyse. Statistics are only comparable "
                        "within one embedder — raw magnitudes differ between columns.")
    p.add_argument("--groups", nargs="+", default=list(DEFAULT_FAMILIES),
                   help="generated_by families to keep, matched by prefix. Everything "
                        "not starting with 'human' is pooled as the machine class.")
    p.add_argument("--space", choices=[*SPACES, "both"], default="both",
                   help="Geometry for the descriptive sections. 'std' is what the "
                        "classifier sees (run_clf_cross_domain fits a StandardScaler).")
    p.add_argument("--max-pairs", type=int, default=200_000,
                   help="Cap on sampled pairs per cosine-distance population.")
    p.add_argument("--n-runs", type=int, default=3, help="Splits in the separability experiment.")
    p.add_argument("--test-size", type=float, default=0.30, help="Held-out fraction.")
    p.add_argument("--seed", type=int, default=42, help="Seed for splits and pair sampling.")
    p.add_argument("--name", default=None,
                   help="Output directory name under results/emb_class_stats/ "
                        "(default: derived from the parquet stems).")
    return p.parse_args()


def output_name(specs: list[CorpusSpec], emb_col: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    parts = [s.name for s in specs]
    stems = "__".join(parts)
    if len(stems) > 100:
        stems = f"{parts[0]}__and_{len(parts) - 1}_more"
    return f"{stems}__{emb_col}"


def main() -> None:
    args = parse_args()
    for p in args.parquets:
        if not p.is_file():
            raise SystemExit(f"Parquet file not found: {p}")

    specs = specs_from_paths(args.parquets, args.domains, base_dir=BASE_DIR)
    spaces = list(SPACES) if args.space == "both" else [args.space]
    fig_dir = RESULTS_DIR / output_name(specs, args.emb_col, args.name)

    out = Tee()
    report_header(args.parquets, args.emb_col, args, out)

    corpora = [
        load_corpus(spec, args.emb_col, tuple(args.groups), out) for spec in specs
    ]

    class_rows: list[dict] = []
    distance_rows: list[dict] = []
    direction_rows: list[dict] = []
    dist_pops: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    dir_stats: dict[str, dict[str, dict]] = {}

    for space in spaces:
        dist_pops[space] = {}
        dir_stats[space] = {}
        for corpus in corpora:
            rng = np.random.default_rng(args.seed)
            class_rows.extend(class_summary(corpus, space))

            pops = distance_populations(corpus, space, rng, args.max_pairs)
            dist_pops[space][corpus.name] = pops
            ratio = fisher_ratio(corpus, space)
            for pop in POPULATIONS:
                distance_rows.append(
                    {"corpus": corpus.name, "space": space, "population": pop,
                     "fisher_ratio": ratio} | _dist_stats(pops[pop])
                )

            direction = direction_stats(corpus, space)
            paired = paired_stats(paired_deltas(corpus, space))
            dir_stats[space][corpus.name] = direction | paired
            direction_rows.append(
                {"corpus": corpus.name, "space": space}
                | {k: v for k, v in (direction | paired).items()
                   if not isinstance(v, np.ndarray)}
            )

    out.print("\n" + "=" * 78)
    report_class_summary(class_rows, out)
    report_distances(distance_rows, out)
    report_directions(direction_rows, out)

    out.print("\n" + "-" * 78)
    out.print("5. running the separability experiment ...")
    out.print("-" * 78)
    sep_rows: list[dict] = []
    for corpus in corpora:
        sep_rows.extend(separability(corpus, args.seed, args.n_runs, args.test_size, out))
    sep = pd.DataFrame(sep_rows)
    report_separability(sep, out)

    matrices = {}
    for space in spaces:
        centroid_M, centroid_names = centroid_cosine_matrix(corpora, space)
        delta_M, delta_names = delta_cosine_matrix(corpora, space)
        matrices[space] = (centroid_M, centroid_names, delta_M, delta_names)
        report_matrices(centroid_M, centroid_names, delta_M, delta_names, space, out)

    # --- figures ---------------------------------------------------------
    print()
    for space in spaces:
        plot_distance_populations(dist_pops[space], space, fig_dir)
        plot_delta_projection(dir_stats[space], space, fig_dir)
        plot_paired_deltas(dir_stats[space], space, fig_dir)
        plot_matrices(*matrices[space], space, fig_dir)
    plot_separability(sep, fig_dir)

    # --- tables ----------------------------------------------------------
    fig_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "class_summary.csv": pd.DataFrame(class_rows),
        "distance_summary.csv": pd.DataFrame(distance_rows),
        "direction_summary.csv": pd.DataFrame(direction_rows),
        "separability.csv": sep,
    }
    for space in spaces:
        centroid_M, centroid_names, delta_M, delta_names = matrices[space]
        tables[f"centroid_cosine_{space}.csv"] = pd.DataFrame(
            centroid_M, index=centroid_names, columns=centroid_names
        )
        tables[f"delta_cosine_{space}.csv"] = pd.DataFrame(
            delta_M, index=delta_names, columns=delta_names
        )

    for fname, table in tables.items():
        keep_index = fname.startswith(("centroid_cosine", "delta_cosine"))
        table.to_csv(fig_dir / fname, index=keep_index)
        print(f"  [OK] {fname}")

    out.print("\n" + "=" * 78)
    out.save(fig_dir / "emb_class_stats.log")
    print(f"\nAll outputs saved to {fig_dir}")


if __name__ == "__main__":
    main()
