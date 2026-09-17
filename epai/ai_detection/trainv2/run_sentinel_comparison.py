r"""Training-based comparison on MG-GB / MG-Sci / MG-AI-Sci (Section 5.2
"Training-based comparison", ``latex/ICLR_AIDetect/iclr2027_conference.tex``).

Replicates the head architecture from Chen et al. 2023 ("GPT-Sentinel",
arXiv:2305.07969): a frozen RoBERTa encoder feeding a two-layer fully
connected MLP with GELU ("RoBERTa-Sentinel" in that paper). This uses
``epai/algs/simple_mlp.py`` with the ``activation="gelu"`` / ``scheduler=
"cosine"`` options added to ``epai/algs/_torch_common.py`` for this script (both
default to the old ``relu`` / no-scheduler behaviour everywhere else, so no
other experiment in this codebase is affected). Reported throughout alongside
the existing ``roberta_logistic`` baseline (a plain logistic probe on the same
embedding, from ``_trained.TRAINED_SPECS``) -- same embedding, two heads.

Fidelity to the paper, and where this deviates
-----------------------------------------------
The paper's Table 2 gives training hyperparameters, which are matched
exactly: 15 epochs, batch size 512, lr 1e-4, weight decay 1e-3, AdamW,
cross-entropy loss, cosine-annealing LR schedule. It states no dropout
anywhere, so dropout is 0 here (this package's ``simple_mlp`` default is 0.1);
it trains a fixed 15 epochs with no mention of a validation split, so early
stopping is disabled (``patience=0``, ``val_frac=0``) rather than this
package's own defaults.

Two things the paper does NOT state, so cannot be replicated exactly:
  * the MLP's hidden width -- the paper only ever writes the symbolic
    ``T[CLS] in R^H``, no number. ``SENTINEL_HIDDEN`` below is this script's
    own choice, not a number out of the paper.
  * which RoBERTa variant (base vs. large) -- also never named.

One thing this deviates from on purpose, because fixing it would mean
re-embedding the whole corpus (out of scope here): the paper pools the
[CLS] token; this repo's ``emb_roberta`` column (used by every trained
detector across this benchmark, not just this script) is roberta-base,
MEAN-pooled over the last hidden state (see ``epai/utils/embedding_utils.py``).

Scope
-----
Only MG-GB and MG-Sci are ever trained on. MG-AI-Sci is eval-only everywhere
(its own "Eval-only" row in the benchmark-summary table; also far too small
for LODO -- ~8-40 rows per domain, same reason ``run_lodo_performance.py``
already excludes it there). GB-orig is out of scope entirely, per the
benchmark table this script targets (MG-GB / MG-Sci / MG-AI-Sci only).

Three regimes, one CSV, one row per (regime, level, kind, detector,
train_split, eval_split, fpr):

  in_domain  grouped 70/30 within MG-GB / MG-Sci separately, N seeds --
             the ``run_in_domain_performance.compute_trained`` pattern.
  lodo       leave-one-domain-out within MG-GB / MG-Sci separately (3 domains
             each), N seeds per fold -- the ``run_lodo_performance.
             compute_trained`` pattern. This is the "close" OOD reading.
  far_ood    fit on the FULL MG-GB (all 3 domains, no held-out split), score
             on the full MG-Sci, and vice versa -- cross-BENCHMARK transfer,
             the "far" OOD reading (MG-GB / MG-Sci share zero document ids --
             see ``run_shift_axis_transfer.py``'s module docstring).

Every regime also scores its fitted model(s), unchanged, on the full MG-AI-Sci
corpus (never trained on).

Zero-shot picks (``kind="zeroshot"`` rows, ``--zs-columns`` / ``--no-
zeroshot``, on by default): the two hand-picked columns from
``cfg/ai_detection/trainv2/best_zeroshot.yaml`` this repo's own paper leads
with -- Binoculars' strongest backbone (``binoculars_score_llama31_8b`` ->
``binox/llama31-8b``) and this paper's own LogJudge readout for gpt-5.6-luna
(``llm_judge_score_gpt56luna_v3_wlp`` -> ``judge/gpt56luna_wlp``). Each is
scored on EXACTLY the same rows the trained detectors were evaluated on in
that regime -- the same seed's grouped 70/30 test partition (in_domain), the
same held-out-domain fold (lodo), or the same full target corpus (far_ood) --
using :mod:`epai.ai_detection.trainv2._zeroshot`'s column-scoring helpers, so
a trained-vs-zero-shot comparison is never confounded by different eval rows.
A zero-shot pick has no training corpus, so its rows carry ``train_split=""``
in in_domain/lodo; in far_ood its rows are still tagged by which trained
model's scenario they accompany (``train_split="mg_sci"``/``"mg_gb"``), even
though the zero-shot SCORE itself doesn't depend on training -- this keeps
the schema uniform across ``kind`` for
:mod:`plot_sentinel_scenarios`'s per-scenario grouping.

Reuses ``epai.ai_detection.trainv2._trained``'s ``fit`` / ``eval_on_fprs`` /
``load_multi_features`` / ``load_multi_ood`` and ``core``'s ``grouped_split``
/ ``goodness`` (AT-HM) / ``eff_fpr_level`` unchanged, on ``embeddings=
(EMB_COL,)`` -- neither trained detector here uses the judge cue-vector
auxiliary input, so ``load_multi_features``/``load_multi_ood`` (embeddings-
only, no ``AUX_COL`` dependency) are used instead of ``_trained.
load_features``. Does NOT import ``TRAINED_SPECS`` and mutate it with a new
entry -- see ``_trained.py``'s module docstring ("no entry in TRAINED_SPECS
sets aux") and ``run_shift_axis_transfer.py``'s precedent for writing sibling
compute functions instead of extending the shared registry other scripts
depend on; ``SPECS`` below is this script's own, separate dict.

compute -> CSV -> printed table (no plotting here -- see
``plot_sentinel_comparison.py`` for the per-regime scatters in the same
design as the paper's main figure, and ``plot_sentinel_scenarios.py`` for the
one all-scenarios-x-all-algorithms comparison figure):

    results/trainv2/sentinel_comparison/
        performance.csv   regime, level, kind, detector, train_split,
                          eval_split, held_out_domain, fpr, n_train, n_test,
                          n_test_human, n_seeds, auroc(_std), tpr_at_fpr(_std),
                          tpr_eff(_std), eff_fpr, goodness(_std)

Run:
    uv run python epai/ai_detection/trainv2/run_sentinel_comparison.py
    uv run python epai/ai_detection/trainv2/run_sentinel_comparison.py --seeds 0 1 2
    uv run python epai/ai_detection/trainv2/run_sentinel_comparison.py \
        --regimes far_ood --fpr 0.01 0.05
    uv run python epai/ai_detection/trainv2/run_sentinel_comparison.py --no-zeroshot
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from epai.ai_detection.trainv2._trained import (
    EMB_COL,
    TRAINED_SPECS,
    eval_on_fprs,
    fit,
    load_multi_features,
    load_multi_ood,
)
from epai.ai_detection.trainv2._zeroshot import eval_column_fprs, orient, resolve_specs
from epai.ai_detection.trainv2.core import eff_fpr_level, goodness, grouped_split
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "sentinel_comparison"

#: MG-GB / MG-Sci only -- see the module docstring's "Scope" section.
TRAIN_SPLITS = ["mg_gb", "mg_sci"]
OOD_SPLIT = "mg_aisci"

#: column names in every MANUGRAM parquet (matches run_lodo_performance.py).
DOMAIN_COL = "domain"
GROUP_COL = "id"

EMBS: tuple[str, ...] = (EMB_COL,)

#: This script's own choice, NOT a number from the paper -- see the module
#: docstring. One hidden layer + the output layer = the paper's "two fully
#: connected layers".
SENTINEL_HIDDEN = 256

#: key -> spec, this script's own dict (see the module docstring for why this
#: is not folded into ``_trained.TRAINED_SPECS``).
SPECS: dict[str, dict] = {
    "roberta_logistic": TRAINED_SPECS["roberta_logistic"],
    "roberta_sentinel": {
        "label": "roberta·sentinel_mlp", "alg": "simple_mlp", "aux": False,
        "embeddings": EMBS,
        "params": {
            "hidden_dims": [SENTINEL_HIDDEN],
            "activation": "gelu",
            # Not stated in the paper -> off, rather than this package's own
            # (unrelated) defaults.
            "dropout": 0.0,
            "input_dropout": 0.0,
            "norm_layer": "none",
            # Table 2 (RoBERTa-Sentinel row), matched exactly.
            "epochs": 15,
            "batch_size": 512,
            "lr": 1e-4,
            "weight_decay": 1e-3,
            "scheduler": "cosine",
            # The paper trains a fixed 15 epochs on no stated validation split.
            "patience": 0,
            "val_frac": 0.0,
        },
    },
}

#: the two zero-shot picks this comparison uses -- Binoculars' strongest
#: backbone and this paper's own LogJudge readout, both from
#: cfg/ai_detection/trainv2/best_zeroshot.yaml. See the module docstring.
ZS_COLUMNS: list[str] = ["binoculars_score_llama31_8b", "llm_judge_score_gpt56luna_v3_wlp"]

DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_FPRS = [0.01]

CSV_FIELDS = [
    "regime", "level", "kind", "detector", "train_split", "eval_split", "held_out_domain",
    "fpr", "n_train", "n_test", "n_test_human", "n_seeds",
    "auroc", "auroc_std", "tpr_at_fpr", "tpr_at_fpr_std",
    "tpr_eff", "tpr_eff_std", "eff_fpr", "goodness", "goodness_std",
]


def _std(vals: list[float]) -> float:
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.std(vals)) if len(vals) > 1 else 0.0


def _nanmean(vals) -> float:
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _row(regime: str, level: str, detector: str, train_split: str, eval_split: str,
         held_out_domain, cell: dict, kind: str = "trained") -> dict:
    return {
        "regime": regime, "level": level, "kind": kind, "detector": detector,
        "train_split": train_split, "eval_split": eval_split,
        "held_out_domain": held_out_domain, "fpr": cell["fpr"],
        "n_train": cell.get("n_train", ""), "n_test": cell.get("n_test", ""),
        "n_test_human": cell.get("n_test_human", ""), "n_seeds": cell["n_seeds"],
        "auroc": cell["auroc"], "auroc_std": cell["auroc_std"],
        "tpr_at_fpr": cell["tpr_at_fpr"], "tpr_at_fpr_std": cell["tpr_at_fpr_std"],
        "tpr_eff": cell["tpr_eff"], "tpr_eff_std": cell["tpr_eff_std"],
        "eff_fpr": cell["eff_fpr"],
        "goodness": cell["goodness"], "goodness_std": cell["goodness_std"],
    }


# ---------------------------------------------------------------------------
# Loading -- thin wrappers over _trained.load_multi_features/load_multi_ood
# that always return a trailing ``raw`` dict (``{}`` when no score columns
# were requested), so every call site below has one fixed tuple shape
# regardless of whether zero-shot columns are in play.
# ---------------------------------------------------------------------------


def _load_features(path: Path, group_col: str, human_label: str, *,
                   domain_col: str | None = None, score_cols: tuple[str, ...] = ()):
    out = load_multi_features(path, group_col, human_label, EMBS,
                              domain_col=domain_col, score_cols=score_cols)
    return out if score_cols else (*out, {})


def _load_ood(path: Path, human_label: str, score_cols: tuple[str, ...] = ()):
    out = load_multi_ood(path, human_label, EMBS, score_cols=score_cols)
    return out if score_cols else (*out, {})


def resolve_zs_specs(paths: list[Path], columns: list[str]) -> list[tuple[str, str, bool]]:
    """``resolve_specs`` restricted to columns present in EVERY one of
    ``paths`` -- so a zero-shot pick plotted for one regime/split is the same
    pick everywhere else, never silently dropped on just one of them."""
    if not columns:
        return []
    common: set[str] | None = None
    for p in paths:
        names = set(pq.read_schema(p).names)
        common = names if common is None else common & names
    return resolve_specs(sorted(common or set()), Path("."), columns=columns)


# ---------------------------------------------------------------------------
# Regime 1: in-domain (grouped 70/30)
# ---------------------------------------------------------------------------


def compute_in_domain(train_splits, ood_path, seeds, test_size, fprs,
                      group_col, human_label, combine, zs_specs=()) -> list[dict]:
    zs_cols = tuple(c for c, _, _ in zs_specs)
    oX, oy, on_h, _, o_raw = _load_ood(ood_path, human_label, zs_cols)
    print(f"  OOD eval corpus {OOD_SPLIT!r}: {len(oy)} rows, {on_h} human (never trained on)")

    zs_ood_read = {col: eval_column_fprs(orient(o_raw[col], mih), oy, fprs)
                  for col, _, mih in zs_specs}

    rows: list[dict] = []
    for split_key, path in tqdm(train_splits, desc="in-domain", unit="split"):
        X, y, groups, _, raw = _load_features(path, group_col, human_label, score_cols=zs_cols)
        for spec_key, spec in SPECS.items():
            detector = f"trained/{spec['label']}"
            a: list[float] = []
            t = {f: [] for f in fprs}
            te_ = {f: [] for f in fprs}
            oa = {f: [] for f in fprs}
            ot = {f: [] for f in fprs}
            ote = {f: [] for f in fprs}
            n_tr = n_te = n_te_h = 0
            for seed in seeds:
                tr, te = grouped_split(y, groups, test_size, seed)
                clf = fit(spec, X, y, None, groups, tr, seed)
                au, per = eval_on_fprs(clf, spec, X[te], y[te], None, fprs)
                a.append(au)
                for f in fprs:
                    t[f].append(per[f][0])
                    te_[f].append(per[f][1])
                n_tr, n_te, n_te_h = len(tr), len(te), int((y[te] == 0).sum())

                oau, oper = eval_on_fprs(clf, spec, oX, oy, None, fprs)
                for f in fprs:
                    oa[f].append(oau)
                    ot[f].append(oper[f][0])
                    ote[f].append(oper[f][1])

            for f in fprs:
                g = [goodness(ai, ei, combine, alpha=f) for ai, ei in zip(a, te_[f])]
                cell = {
                    "fpr": f, "n_train": n_tr, "n_test": n_te, "n_test_human": n_te_h,
                    "n_seeds": len(seeds),
                    "auroc": float(np.mean(a)), "auroc_std": _std(a),
                    "tpr_at_fpr": float(np.mean(t[f])), "tpr_at_fpr_std": _std(t[f]),
                    "tpr_eff": float(np.mean(te_[f])), "tpr_eff_std": _std(te_[f]),
                    "eff_fpr": eff_fpr_level(n_te_h, f),
                    "goodness": float(np.mean(g)), "goodness_std": _std(g),
                }
                rows.append(_row("in_domain", "split", detector, split_key, split_key, "", cell))

                og = [goodness(ai, ei, combine, alpha=f) for ai, ei in zip(oa[f], ote[f])]
                ocell = {
                    "fpr": f, "n_train": n_tr, "n_test": "", "n_test_human": on_h,
                    "n_seeds": len(seeds),
                    "auroc": float(np.mean(oa[f])), "auroc_std": _std(oa[f]),
                    "tpr_at_fpr": float(np.mean(ot[f])), "tpr_at_fpr_std": _std(ot[f]),
                    "tpr_eff": float(np.mean(ote[f])), "tpr_eff_std": _std(ote[f]),
                    "eff_fpr": eff_fpr_level(on_h, f),
                    "goodness": float(np.mean(og)), "goodness_std": _std(og),
                }
                rows.append(_row("in_domain", "split", detector, split_key, OOD_SPLIT, "", ocell))

        # Zero-shot picks, on the SAME seed-by-seed test partitions.
        for col, det, mih in zs_specs:
            a = []
            t = {f: [] for f in fprs}
            te_ = {f: [] for f in fprs}
            n_te_h = 0
            for seed in seeds:
                tr, te = grouped_split(y, groups, test_size, seed)
                n_te_h = int((y[te] == 0).sum())
                au, per = eval_column_fprs(orient(raw[col][te], mih), y[te], fprs)
                a.append(au)
                for f in fprs:
                    t[f].append(per[f][0])
                    te_[f].append(per[f][1])
            for f in fprs:
                g = [goodness(ai, ei, combine, alpha=f) for ai, ei in zip(a, te_[f])
                    if np.isfinite(ai) and np.isfinite(ei)]
                cell = {
                    "fpr": f, "n_train": "", "n_test": "", "n_test_human": n_te_h,
                    "n_seeds": len(seeds),
                    "auroc": _nanmean(a), "auroc_std": _std(a),
                    "tpr_at_fpr": _nanmean(t[f]), "tpr_at_fpr_std": _std(t[f]),
                    "tpr_eff": _nanmean(te_[f]), "tpr_eff_std": _std(te_[f]),
                    "eff_fpr": eff_fpr_level(n_te_h, f),
                    "goodness": _nanmean(g), "goodness_std": _std(g),
                }
                rows.append(_row("in_domain", "split", det, "", split_key, "", cell,
                                 kind="zeroshot"))

    # Zero-shot OOD read: one deterministic full-MG-AI-Sci score per pick,
    # computed once above -- not re-derived per training split.
    for col, det, mih in zs_specs:
        oau, oper = zs_ood_read[col]
        for f in fprs:
            t_ood, te_ood = oper[f]
            gd = (goodness(oau, te_ood, combine, alpha=f)
                 if np.isfinite(oau) and np.isfinite(te_ood) else float("nan"))
            ocell = {
                "fpr": f, "n_train": "", "n_test": "", "n_test_human": on_h,
                "n_seeds": len(seeds),
                "auroc": oau, "auroc_std": float("nan"),
                "tpr_at_fpr": t_ood, "tpr_at_fpr_std": float("nan"),
                "tpr_eff": te_ood, "tpr_eff_std": float("nan"),
                "eff_fpr": eff_fpr_level(on_h, f),
                "goodness": gd, "goodness_std": float("nan"),
            }
            rows.append(_row("in_domain", "split", det, "", OOD_SPLIT, "", ocell,
                             kind="zeroshot"))
    return rows


# ---------------------------------------------------------------------------
# Regime 2: LODO ("close" OOD)
# ---------------------------------------------------------------------------


def _lodo_fold(spec, X, y, groups, tr, te, seeds, fprs, oX, oy) -> dict:
    """One held-out-domain fold, averaged over seeds -- also scored on the
    MG-AI-Sci OOD corpus every seed. Mirrors ``run_lodo_performance._fold``,
    narrowed to this script's ``SPECS`` (no judge cue-vector aux)."""
    a: list[float] = []
    t = {f: [] for f in fprs}
    te_ = {f: [] for f in fprs}
    oa = {f: [] for f in fprs}
    ot = {f: [] for f in fprs}
    ote = {f: [] for f in fprs}
    for seed in seeds:
        clf = fit(spec, X, y, None, groups, tr, seed)
        au, per = eval_on_fprs(clf, spec, X[te], y[te], None, fprs)
        a.append(au)
        for f in fprs:
            t[f].append(per[f][0])
            te_[f].append(per[f][1])
        oau, oper = eval_on_fprs(clf, spec, oX, oy, None, fprs)
        for f in fprs:
            oa[f].append(oau)
            ot[f].append(oper[f][0])
            ote[f].append(oper[f][1])
    n_h = int((y[te] == 0).sum())
    return {
        "auroc": float(np.mean(a)), "n_test": int(te.sum()), "n_test_human": n_h,
        "by_fpr": {f: {"tpr_at_fpr": float(np.mean(t[f])), "tpr_eff": float(np.mean(te_[f])),
                       "eff_fpr": eff_fpr_level(n_h, f)} for f in fprs},
        "ood_by_fpr": {f: {"auroc": float(np.mean(oa[f])), "tpr_at_fpr": float(np.mean(ot[f])),
                           "tpr_eff": float(np.mean(ote[f]))} for f in fprs},
    }


def compute_lodo(train_splits, ood_path, seeds, fprs, human_label, combine,
                 min_domain_rows: int = 100, zs_specs=()) -> list[dict]:
    zs_cols = tuple(c for c, _, _ in zs_specs)
    oX, oy, on_h, _, o_raw = _load_ood(ood_path, human_label, zs_cols)
    print(f"  OOD eval corpus {OOD_SPLIT!r}: {len(oy)} rows, {on_h} human (never trained on)")

    zs_ood_read = {col: eval_column_fprs(orient(o_raw[col], mih), oy, fprs)
                  for col, _, mih in zs_specs}

    rows: list[dict] = []
    for split_key, path in tqdm(train_splits, desc="LODO", unit="split"):
        X, y, groups, dom, _, raw = _load_features(
            path, GROUP_COL, human_label, domain_col=DOMAIN_COL, score_cols=zs_cols)
        counts = {d: int((dom == d).sum()) for d in np.unique(dom)}
        domains = sorted(d for d, n in counts.items()
                         if n >= min_domain_rows and np.unique(y[dom == d]).size == 2)
        if len(domains) < 2:
            print(f"  {split_key}: only {len(domains)} usable domain(s) ({counts}) — skipped")
            continue
        print(f"\n  {split_key}: {len(domains)} domains {domains}")

        for spec_key, spec in SPECS.items():
            detector = f"trained/{spec['label']}"
            folds = []
            for d in domains:
                tr, te = dom != d, dom == d
                assert d not in set(dom[tr]) and set(dom[te]) == {d}
                assert set(groups[tr]).isdisjoint(groups[te]), "a group straddled the fold"
                fd = _lodo_fold(spec, X, y, groups, tr, te, seeds, fprs, oX, oy)
                folds.append(fd)
                for f in fprs:
                    bf = fd["by_fpr"][f]
                    cell = {
                        "fpr": f, "n_train": "", "n_test": fd["n_test"],
                        "n_test_human": fd["n_test_human"], "n_seeds": len(seeds),
                        "auroc": fd["auroc"], "auroc_std": "",
                        "tpr_at_fpr": bf["tpr_at_fpr"], "tpr_at_fpr_std": "",
                        "tpr_eff": bf["tpr_eff"], "tpr_eff_std": "",
                        "eff_fpr": bf["eff_fpr"],
                        "goodness": goodness(fd["auroc"], bf["tpr_eff"], combine, alpha=f),
                        "goodness_std": "",
                    }
                    rows.append(_row("lodo", "fold", detector, split_key, split_key, d, cell))

            for f in fprs:
                gf = [goodness(fd["auroc"], fd["by_fpr"][f]["tpr_eff"], combine, alpha=f)
                     for fd in folds]
                cell = {
                    "fpr": f, "n_train": "",
                    "n_test": int(round(float(np.mean([fd["n_test"] for fd in folds])))),
                    "n_test_human": int(round(float(np.mean([fd["n_test_human"] for fd in folds])))),
                    "n_seeds": len(seeds),
                    "auroc": float(np.mean([fd["auroc"] for fd in folds])),
                    "auroc_std": _std([fd["auroc"] for fd in folds]),
                    "tpr_at_fpr": float(np.mean([fd["by_fpr"][f]["tpr_at_fpr"] for fd in folds])),
                    "tpr_at_fpr_std": _std([fd["by_fpr"][f]["tpr_at_fpr"] for fd in folds]),
                    "tpr_eff": float(np.mean([fd["by_fpr"][f]["tpr_eff"] for fd in folds])),
                    "tpr_eff_std": _std([fd["by_fpr"][f]["tpr_eff"] for fd in folds]),
                    "eff_fpr": float(np.mean([fd["by_fpr"][f]["eff_fpr"] for fd in folds])),
                    "goodness": float(np.mean(gf)), "goodness_std": _std(gf),
                }
                rows.append(_row("lodo", "split", detector, split_key, split_key, "", cell))

                oa = [fd["ood_by_fpr"][f]["auroc"] for fd in folds]
                ot = [fd["ood_by_fpr"][f]["tpr_at_fpr"] for fd in folds]
                ote = [fd["ood_by_fpr"][f]["tpr_eff"] for fd in folds]
                og = [goodness(ai, ei, combine, alpha=f) for ai, ei in zip(oa, ote)]
                ocell = {
                    "fpr": f, "n_train": "", "n_test": "", "n_test_human": on_h,
                    "n_seeds": len(seeds),
                    "auroc": float(np.mean(oa)), "auroc_std": _std(oa),
                    "tpr_at_fpr": float(np.mean(ot)), "tpr_at_fpr_std": _std(ot),
                    "tpr_eff": float(np.mean(ote)), "tpr_eff_std": _std(ote),
                    "eff_fpr": eff_fpr_level(on_h, f),
                    "goodness": float(np.mean(og)), "goodness_std": _std(og),
                }
                rows.append(_row("lodo", "split", detector, split_key, OOD_SPLIT, "", ocell))

        # Zero-shot picks, on the SAME held-out-domain folds (deterministic --
        # no seed loop; the fold partition itself doesn't depend on seed).
        for col, det, mih in zs_specs:
            fold_a: list[float] = []
            fold_t = {f: [] for f in fprs}
            fold_te = {f: [] for f in fprs}
            fold_nh: list[int] = []
            for d in domains:
                te = dom == d
                fold_nh.append(int((y[te] == 0).sum()))
                au, per = eval_column_fprs(orient(raw[col][te], mih), y[te], fprs)
                fold_a.append(au)
                for f in fprs:
                    fold_t[f].append(per[f][0])
                    fold_te[f].append(per[f][1])
            for f in fprs:
                g = [goodness(ai, ei, combine, alpha=f) for ai, ei in zip(fold_a, fold_te[f])
                    if np.isfinite(ai) and np.isfinite(ei)]
                cell = {
                    "fpr": f, "n_train": "", "n_test": "",
                    "n_test_human": int(round(_nanmean(fold_nh))) if fold_nh else "",
                    "n_seeds": len(seeds),
                    "auroc": _nanmean(fold_a), "auroc_std": _std(fold_a),
                    "tpr_at_fpr": _nanmean(fold_t[f]), "tpr_at_fpr_std": _std(fold_t[f]),
                    "tpr_eff": _nanmean(fold_te[f]), "tpr_eff_std": _std(fold_te[f]),
                    "eff_fpr": _nanmean([eff_fpr_level(nh, f) for nh in fold_nh]) if fold_nh else f,
                    "goodness": _nanmean(g), "goodness_std": _std(g),
                }
                rows.append(_row("lodo", "split", det, "", split_key, "", cell, kind="zeroshot"))

    for col, det, mih in zs_specs:
        oau, oper = zs_ood_read[col]
        for f in fprs:
            t_ood, te_ood = oper[f]
            gd = (goodness(oau, te_ood, combine, alpha=f)
                 if np.isfinite(oau) and np.isfinite(te_ood) else float("nan"))
            ocell = {
                "fpr": f, "n_train": "", "n_test": "", "n_test_human": on_h,
                "n_seeds": len(seeds),
                "auroc": oau, "auroc_std": float("nan"),
                "tpr_at_fpr": t_ood, "tpr_at_fpr_std": float("nan"),
                "tpr_eff": te_ood, "tpr_eff_std": float("nan"),
                "eff_fpr": eff_fpr_level(on_h, f),
                "goodness": gd, "goodness_std": float("nan"),
            }
            rows.append(_row("lodo", "split", det, "", OOD_SPLIT, "", ocell, kind="zeroshot"))
    return rows


# ---------------------------------------------------------------------------
# Regime 3: far OOD (train on one full corpus, score on the others)
# ---------------------------------------------------------------------------


def compute_far_ood(train_splits, ood_path, seeds, fprs,
                    group_col, human_label, combine, zs_specs=()) -> list[dict]:
    zs_cols = tuple(c for c, _, _ in zs_specs)
    loaded = {key: _load_features(path, group_col, human_label, score_cols=zs_cols)
             for key, path in train_splits}
    oX, oy, on_h, _, o_raw = _load_ood(ood_path, human_label, zs_cols)
    print(f"  OOD eval corpus {OOD_SPLIT!r}: {len(oy)} rows, {on_h} human (never trained on)")

    rows: list[dict] = []
    keys = [k for k, _ in train_splits]

    # Zero-shot: each full-corpus read is independent of any trained model,
    # so it's computed once per corpus and reused across every scenario that
    # references it (train=mg_gb's far-OOD read on mg_sci is the SAME number
    # as train=mg_sci's own full-corpus zero-shot score).
    zs_full: dict[str, dict[str, tuple]] = {}
    zs_n_human: dict[str, int] = {}
    for key in keys:
        _, y_k, _, _, raw_k = loaded[key]
        zs_n_human[key] = int((y_k == 0).sum())
        zs_full[key] = {col: eval_column_fprs(orient(raw_k[col], mih), y_k, fprs)
                       for col, _, mih in zs_specs}
    zs_n_human[OOD_SPLIT] = on_h
    zs_full[OOD_SPLIT] = {col: eval_column_fprs(orient(o_raw[col], mih), oy, fprs)
                          for col, _, mih in zs_specs}

    for train_key in tqdm(keys, desc="far-OOD", unit="split"):
        X, y, groups, _, _ = loaded[train_key]
        cross_targets = [(k, loaded[k][0], loaded[k][1]) for k in keys if k != train_key]
        cross_targets.append((OOD_SPLIT, oX, oy))
        idx_all = np.arange(len(y))

        for spec_key, spec in SPECS.items():
            detector = f"trained/{spec['label']}"
            per_target = {name: {"a": [], "t": {f: [] for f in fprs}, "te": {f: [] for f in fprs}}
                         for name, _, _ in cross_targets}
            for seed in seeds:
                clf = fit(spec, X, y, None, groups, idx_all, seed)
                for name, eX, ey in cross_targets:
                    au, per = eval_on_fprs(clf, spec, eX, ey, None, fprs)
                    per_target[name]["a"].append(au)
                    for f in fprs:
                        per_target[name]["t"][f].append(per[f][0])
                        per_target[name]["te"][f].append(per[f][1])

            for name, eX, ey in cross_targets:
                n_h = int((ey == 0).sum())
                stats = per_target[name]
                for f in fprs:
                    g = [goodness(ai, ei, combine, alpha=f)
                        for ai, ei in zip(stats["a"], stats["te"][f])]
                    cell = {
                        "fpr": f, "n_train": len(y), "n_test": len(ey), "n_test_human": n_h,
                        "n_seeds": len(seeds),
                        "auroc": float(np.mean(stats["a"])), "auroc_std": _std(stats["a"]),
                        "tpr_at_fpr": float(np.mean(stats["t"][f])), "tpr_at_fpr_std": _std(stats["t"][f]),
                        "tpr_eff": float(np.mean(stats["te"][f])), "tpr_eff_std": _std(stats["te"][f]),
                        "eff_fpr": eff_fpr_level(n_h, f),
                        "goodness": float(np.mean(g)), "goodness_std": _std(g),
                    }
                    rows.append(_row("far_ood", "split", detector, train_key, name, "", cell))

        # Zero-shot picks: same cross-corpus targets, cached full-corpus reads.
        eval_names = [k for k in keys if k != train_key] + [OOD_SPLIT]
        for col, det, mih in zs_specs:
            for name in eval_names:
                au, per = zs_full[name][col]
                n_h = zs_n_human[name]
                for f in fprs:
                    t_, te_ = per[f]
                    gd = (goodness(au, te_, combine, alpha=f)
                         if np.isfinite(au) and np.isfinite(te_) else float("nan"))
                    cell = {
                        "fpr": f, "n_train": "", "n_test": "", "n_test_human": n_h,
                        "n_seeds": len(seeds),
                        "auroc": au, "auroc_std": float("nan"),
                        "tpr_at_fpr": t_, "tpr_at_fpr_std": float("nan"),
                        "tpr_eff": te_, "tpr_eff_std": float("nan"),
                        "eff_fpr": eff_fpr_level(n_h, f),
                        "goodness": gd, "goodness_std": float("nan"),
                    }
                    rows.append(_row("far_ood", "split", det, train_key, name, "", cell,
                                     kind="zeroshot"))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'regime':<10}{'kind':<9}{'detector':<26}{'train':<9}{'eval':<10}"
          f"{'domain':<10}{'FPR':>6}{'auroc':>15}{'TPR':>15}{'AT-HM':>10}")
    for r in rows:
        def fmt(v, sd):
            if v == "" or (isinstance(v, float) and not np.isfinite(v)):
                return "--"
            bare = (sd in ("", None) or (isinstance(sd, float) and not np.isfinite(sd)))
            return f"{v:.3f}" if bare else f"{v:.3f}±{sd:.3f}"
        print(f"{r['regime']:<10}{r['kind']:<9}{r['detector']:<26}{r['train_split']:<9}"
              f"{r['eval_split']:<10}{str(r['held_out_domain']):<10}"
              f"{float(r['fpr']) * 100:>5g}%"
              f"{fmt(r['auroc'], r['auroc_std']):>15}"
              f"{fmt(r['tpr_at_fpr'], r['tpr_at_fpr_std']):>15}"
              f"{fmt(r['goodness'], r['goodness_std']):>10}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=TRAIN_SPLITS, metavar="KEY",
                    help=f"training splits (default: {TRAIN_SPLITS}) -- MG-GB / MG-Sci only")
    ap.add_argument("--ood-split", default=OOD_SPLIT, metavar="KEY",
                    help=f"eval-only corpus, never trained on (default: {OOD_SPLIT!r})")
    ap.add_argument("--regimes", nargs="+", choices=("in_domain", "lodo", "far_ood"),
                    default=["in_domain", "lodo", "far_ood"])
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--test-size", type=float, default=0.30,
                    help="in-domain regime only (default: 0.30)")
    ap.add_argument("--fpr", nargs="+", type=float, default=DEFAULT_FPRS, metavar="LEVEL",
                    help=f"oracle FPR level(s) (default: {DEFAULT_FPRS})")
    ap.add_argument("--combine", choices=("hmean", "gmean", "amean"), default="hmean",
                    help="AT-HM = this mean of Gini(AUROC) and oracle TPR (default: hmean)")
    ap.add_argument("--group-col", default="id")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--min-domain-rows", type=int, default=100,
                    help="LODO regime only: a domain below this row count is dropped "
                         "(matches run_lodo_performance.py's default)")
    ap.add_argument("--zs-columns", nargs="*", default=ZS_COLUMNS, metavar="COLUMN",
                    help=f"zero-shot score column(s) to score on the same rows as the "
                         f"trained detectors, in every regime (default: {ZS_COLUMNS})")
    ap.add_argument("--no-zeroshot", action="store_true",
                    help="skip the zero-shot picks entirely (kind='trained' rows only)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"default: a fresh {DEFAULT_OUT_DIR.name}_<YYYYMMDD_HHMM> "
                         f"directory under {DEFAULT_OUT_DIR.parent}/")
    args = ap.parse_args()
    out_dir = args.out_dir or (
        DEFAULT_OUT_DIR.parent / f"{DEFAULT_OUT_DIR.name}_{datetime.now():%Y%m%d_%H%M}")

    train_splits = resolve_splits(args.splits, PARQUET_DIR)
    [(_, ood_path)] = resolve_splits([args.ood_split], PARQUET_DIR)
    missing = [str(p) for _, p in [*train_splits, (args.ood_split, ood_path)] if not p.exists()]
    if missing:
        raise SystemExit(f"parquet(s) not found: {missing}")

    zs_columns = [] if args.no_zeroshot else args.zs_columns
    zs_specs = resolve_zs_specs([*(p for _, p in train_splits), ood_path], zs_columns)

    print(f"detectors: {[s['label'] for s in SPECS.values()]}")
    if zs_specs:
        print(f"zero-shot: {[det for _, det, _ in zs_specs]}")
    elif zs_columns:
        print(f"zero-shot: none of {zs_columns} found in every split — skipped")
    print(f"train on : {[s for s, _ in train_splits]}   OOD eval: {args.ood_split!r}")
    print(f"seeds    : {args.seeds}   fpr: {args.fpr}   regimes: {args.regimes}\n")

    rows: list[dict] = []
    if "in_domain" in args.regimes:
        print("=== in-domain (grouped 70/30) ===")
        rows += compute_in_domain(train_splits, ood_path, args.seeds, args.test_size,
                                  args.fpr, args.group_col, args.human_label, args.combine,
                                  zs_specs)
    if "lodo" in args.regimes:
        print("\n=== LODO (\"close\" OOD) ===")
        rows += compute_lodo(train_splits, ood_path, args.seeds, args.fpr,
                             args.human_label, args.combine, args.min_domain_rows, zs_specs)
    if "far_ood" in args.regimes:
        print("\n=== far OOD (train on the full corpus, score on the others) ===")
        rows += compute_far_ood(train_splits, ood_path, args.seeds, args.fpr,
                                args.group_col, args.human_label, args.combine, zs_specs)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "performance.csv"
    pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
    _print_table(rows)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
