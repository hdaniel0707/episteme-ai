r"""Training-based comparison on MG-GB / MG-Sci / MG-AI-Sci (Section 5.2
"Training-based comparison", ``latex/_ICLR_2027__AI_Detect/iclr2027_conference.tex``).

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

Design: a fixed evaluation set per scenario
--------------------------------------------
Earlier versions of this script (and every other trainv2 runner) evaluated a
zero-shot pick on whatever subset of a corpus the TRAINED detector happened
to be scored on that regime -- the seed's grouped 70/30 test partition, the
held-out-domain fold, or the full target corpus. Since a zero-shot detector
has no training step, that subset never served a fairness purpose; it only
injected sampling noise into a number that should be stable, and made a
zero-shot pick's reading appear to "move" between scenarios that were
supposed to be testing only the TRAINED detector's degradation.

This version instead fixes ONE evaluation set per corpus up front and reuses
it, unchanged, everywhere that corpus is an eval target:

  1. For MG-GB and MG-Sci separately, draw one grouped 70/30 split
     (``--split-seed``, deterministic -- not redrawn per model-fit seed).
     The 30% eval half is fixed for the rest of the run; only the 70% train
     half's composition changes across scenarios.
  2. **In-Domain**: fit on the full 70%, score on that corpus's own fixed
     30%.
  3. **Near-OOD**: for each of the 3 domains d, fit on (70% minus domain d),
     score on (domain d's slice of the fixed 30%); average the 3 folds.
  4. **Far-OOD**: reuse the EXACT SAME fit as In-Domain (same 70%, no domain
     exclusion) -- just score it on the OTHER corpus's fixed 30% instead.
     Far-OOD and In-Domain are therefore the same trained classifier scored
     on two different eval sets, not two differently-trained classifiers --
     the only thing that changes between them is the eval target, not also
     the train-set size (the old design trained Far-OOD on the FULL,
     un-split corpus, confounding "different eval corpus" with "different/
     larger train set").
  5. Every scenario also scores against MG-AI-Sci (full corpus, never
     trained on, no subsampling -- it never had a "moving eval set" problem).

Run this whole procedure once with MG-GB as the "primary" corpus and once
with MG-Sci as primary; each scenario's headline reading is the mean of the
two runs (``plot_sentinel_scenarios.py``'s existing unfiltered averaging
already does this -- see that module's docstring).

A zero-shot pick is scored exactly once per fixed 30% set (plus once on the
full MG-AI-Sci) and that same number is looked up everywhere it's needed. A
direct consequence, not a bug: since there are only two fixed eval sets
total (MG-GB's 30%, MG-Sci's 30%) plus MG-AI-Sci, and every scenario's
headline number averages over BOTH corpus orientations, a zero-shot
detector's In-Domain, Near-OOD, and Far-OOD readings are all the SAME
number -- mean(zeroshot-on-MG-GB-30%, zeroshot-on-MG-Sci-30%,
zeroshot-on-MG-AI-Sci, zeroshot-on-MG-AI-Sci) -- regardless of scenario.
:mod:`plot_sentinel_scenarios` draws this as one star-shaped point per
zero-shot algorithm rather than three overlapping scenario markers.

Only MG-GB and MG-Sci are ever trained on (exactly 2 splits -- the
fixed-eval-set pairing requires exactly this). GB-orig is out of scope
entirely, per the benchmark table this script targets.

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
depend on; ``SPECS`` below is this script's own, separate dict. This
deliberately diverges from :mod:`epai.ai_detection.trainv2._zeroshot`'s
documented default (overlay zero-shot on the trained model's own seed/fold
rows), which the OTHER trainv2 runners still use.

compute -> CSV -> printed table (no plotting here -- see
``plot_sentinel_scenarios.py`` for the one all-scenarios-x-all-algorithms
comparison figure, and ``build_sentinel_scenarios_table.py`` for the
matching LaTeX table):

    results/trainv2/sentinel_comparison_<timestamp>/
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
    uv run python epai/ai_detection/trainv2/run_sentinel_comparison.py --split-seed 1
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

#: MG-GB / MG-Sci only, exactly 2 -- see the module docstring's "Design" section.
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

#: the three zero-shot picks this comparison uses -- Binoculars' strongest
#: backbone and this paper's own LogJudge/FusionJudge readouts, all from
#: cfg/ai_detection/trainv2/best_zeroshot.yaml (the current ord2-based
#: LogJudge column, not the old `_wlp` one this script used before
#: FusionJudge existed).
ZS_COLUMNS: list[str] = ["binoculars_score_llama31_8b", "logjudge_ord2_gpt56luna_v3",
                         "fusionjudge_score"]

#: model-fit seeds (init/SGD variance only -- the data SPLIT is fixed by
#: --split-seed, decoupled from these).
DEFAULT_SEEDS = [0, 1, 2]
#: the ONE seed that determines the fixed 70/30 split for both corpora --
#: deliberately separate from DEFAULT_SEEDS. See the module docstring.
SPLIT_SEED = 0
DEFAULT_FPRS = [0.01]
REGIMES = ("in_domain", "near_ood", "far_ood")

CSV_FIELDS = [
    "regime", "level", "kind", "detector", "train_split", "eval_split", "held_out_domain",
    "fpr", "n_train", "n_test", "n_test_human", "n_seeds",
    "auroc", "auroc_std", "tpr_at_fpr", "tpr_at_fpr_std",
    "tpr_eff", "tpr_eff_std", "eff_fpr", "goodness", "goodness_std",
]


def _std(vals: list[float]) -> float:
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.std(vals)) if len(vals) > 1 else 0.0


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
# Fixed data: one grouped 70/30 split per corpus, drawn once.
# ---------------------------------------------------------------------------


def load_fixed_corpora(train_splits, group_col: str, human_label: str,
                       split_seed: int, test_size: float,
                       zs_cols: tuple[str, ...]) -> dict[str, dict]:
    """``{corpus_key: {X, y, groups, dom, raw, tr, ev}}`` -- ``tr``/``ev`` are
    the ONE grouped 70/30 split for that corpus (``split_seed``, not redrawn
    per model-fit seed or per scenario). Every regime below reads its
    train/eval rows out of this fixed ``tr``/``ev`` pair -- see the module
    docstring's "Design" section."""
    data: dict[str, dict] = {}
    for key, path in train_splits:
        X, y, groups, dom, _, raw = _load_features(
            path, group_col, human_label, domain_col=DOMAIN_COL, score_cols=zs_cols)
        tr, ev = grouped_split(y, groups, test_size, split_seed)
        data[key] = dict(X=X, y=y, groups=groups, dom=dom, raw=raw, tr=tr, ev=ev)
        n_h = int((y[ev] == 0).sum())
        print(f"  {key}: fixed split (seed={split_seed}) -> {len(tr)} train / "
              f"{len(ev)} eval rows ({n_h} human in eval)")
    return data


def compute_zs_fixed_reads(data: dict[str, dict], oy, o_raw, on_h,
                           zs_specs, fprs) -> tuple[dict, dict]:
    """One fixed zero-shot read per ``(corpus, column)`` -- each corpus's
    30% eval half (never the 70%, never the full corpus) -- plus one for the
    full MG-AI-Sci. Computed ONCE; every scenario below reuses these numbers
    verbatim rather than rescoring on some regime-specific subset. See the
    module docstring's "Design" section for why."""
    zs_fixed: dict[str, dict[str, tuple]] = {}
    zs_n_human: dict[str, int] = {}
    for key, d in data.items():
        ev, y_ev = d["ev"], d["y"][d["ev"]]
        zs_n_human[key] = int((y_ev == 0).sum())
        zs_fixed[key] = {col: eval_column_fprs(orient(d["raw"][col][ev], mih), y_ev, fprs)
                         for col, _, mih in zs_specs}
    zs_n_human[OOD_SPLIT] = on_h
    zs_fixed[OOD_SPLIT] = {col: eval_column_fprs(orient(o_raw[col], mih), oy, fprs)
                           for col, _, mih in zs_specs}
    return zs_fixed, zs_n_human


def zs_rows_for_corpus(primary: str, other: str, zs_specs, zs_fixed, zs_n_human,
                       fprs, combine, regimes: set[str]) -> list[dict]:
    """Zero-shot rows for one primary/other corpus orientation. In-Domain and
    Near-OOD both read ``zs_fixed[primary]`` -- the SAME number, since both
    target the same fixed 30% eval set and a zero-shot pick has no training
    set to shrink for Near-OOD's domain exclusion. Far-OOD reads
    ``zs_fixed[other]`` -- the one case where the eval TARGET itself
    changes. Every regime also carries the fixed MG-AI-Sci read."""
    def cell_rows(regime: str, target: str) -> list[dict]:
        rows = []
        for col, det, _ in zs_specs:
            au, per = zs_fixed[target][col]
            n_h = zs_n_human[target]
            for f in fprs:
                t_, te_ = per[f]
                gd = (goodness(au, te_, combine, alpha=f)
                     if np.isfinite(au) and np.isfinite(te_) else float("nan"))
                cell = {
                    "fpr": f, "n_train": "", "n_test": "", "n_test_human": n_h,
                    "n_seeds": 1,
                    "auroc": au, "auroc_std": float("nan"),
                    "tpr_at_fpr": t_, "tpr_at_fpr_std": float("nan"),
                    "tpr_eff": te_, "tpr_eff_std": float("nan"),
                    "eff_fpr": eff_fpr_level(n_h, f),
                    "goodness": gd, "goodness_std": float("nan"),
                }
                rows.append(_row(regime, "split", det, primary, target, "", cell,
                                 kind="zeroshot"))
        return rows

    rows: list[dict] = []
    if "in_domain" in regimes:
        rows += cell_rows("in_domain", primary)
        rows += cell_rows("in_domain", OOD_SPLIT)
    if "near_ood" in regimes:
        rows += cell_rows("near_ood", primary)  # reused In-Domain number, see docstring
        rows += cell_rows("near_ood", OOD_SPLIT)
    if "far_ood" in regimes:
        rows += cell_rows("far_ood", other)
        rows += cell_rows("far_ood", OOD_SPLIT)
    return rows


# ---------------------------------------------------------------------------
# Trained detectors: In-Domain + Far-OOD share one fit (same fixed 70%);
# Near-OOD refits per excluded domain, still within the fixed split.
# ---------------------------------------------------------------------------


def compute_trained_for_corpus(primary: str, other: str, data: dict[str, dict],
                               oX, oy, on_h, seeds, fprs, combine,
                               min_domain_rows: int, regimes: set[str]) -> list[dict]:
    dp, do = data[primary], data[other]
    X, y, groups, dom, tr, ev = dp["X"], dp["y"], dp["groups"], dp["dom"], dp["tr"], dp["ev"]
    Xo, yo, ev_o = do["X"], do["y"], do["ev"]

    rows: list[dict] = []
    for spec_key, spec in tqdm(SPECS.items(), desc=f"trained[{primary}]", unit="alg"):
        detector = f"trained/{spec['label']}"

        # --- In-Domain + Far-OOD: ONE fit per seed on the full fixed 70%,
        # scored on the own 30%, the OTHER corpus's fixed 30%, and MG-AI-Sci.
        if "in_domain" in regimes or "far_ood" in regimes:
            a: list[float] = []
            t = {f: [] for f in fprs}
            te_ = {f: [] for f in fprs}
            fa: list[float] = []
            ft = {f: [] for f in fprs}
            fte = {f: [] for f in fprs}
            oa = {f: [] for f in fprs}
            ot = {f: [] for f in fprs}
            ote = {f: [] for f in fprs}
            for seed in seeds:
                clf = fit(spec, X, y, None, groups, tr, seed)
                au, per = eval_on_fprs(clf, spec, X[ev], y[ev], None, fprs)
                a.append(au)
                for f in fprs:
                    t[f].append(per[f][0])
                    te_[f].append(per[f][1])

                fau, fper = eval_on_fprs(clf, spec, Xo[ev_o], yo[ev_o], None, fprs)
                fa.append(fau)
                for f in fprs:
                    ft[f].append(fper[f][0])
                    fte[f].append(fper[f][1])

                oau, oper = eval_on_fprs(clf, spec, oX, oy, None, fprs)
                for f in fprs:
                    oa[f].append(oau)
                    ot[f].append(oper[f][0])
                    ote[f].append(oper[f][1])

            n_tr, n_ev, n_ev_h = len(tr), len(ev), int((y[ev] == 0).sum())
            n_evo, n_evo_h = len(ev_o), int((yo[ev_o] == 0).sum())
            for f in fprs:
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

                if "in_domain" in regimes:
                    g = [goodness(ai, ei, combine, alpha=f) for ai, ei in zip(a, te_[f])]
                    cell = {
                        "fpr": f, "n_train": n_tr, "n_test": n_ev, "n_test_human": n_ev_h,
                        "n_seeds": len(seeds),
                        "auroc": float(np.mean(a)), "auroc_std": _std(a),
                        "tpr_at_fpr": float(np.mean(t[f])), "tpr_at_fpr_std": _std(t[f]),
                        "tpr_eff": float(np.mean(te_[f])), "tpr_eff_std": _std(te_[f]),
                        "eff_fpr": eff_fpr_level(n_ev_h, f),
                        "goodness": float(np.mean(g)), "goodness_std": _std(g),
                    }
                    rows.append(_row("in_domain", "split", detector, primary, primary, "", cell))
                    rows.append(_row("in_domain", "split", detector, primary, OOD_SPLIT, "",
                                     ocell))

                if "far_ood" in regimes:
                    fg = [goodness(ai, ei, combine, alpha=f) for ai, ei in zip(fa, fte[f])]
                    fcell = {
                        "fpr": f, "n_train": n_tr, "n_test": n_evo, "n_test_human": n_evo_h,
                        "n_seeds": len(seeds),
                        "auroc": float(np.mean(fa)), "auroc_std": _std(fa),
                        "tpr_at_fpr": float(np.mean(ft[f])), "tpr_at_fpr_std": _std(ft[f]),
                        "tpr_eff": float(np.mean(fte[f])), "tpr_eff_std": _std(fte[f]),
                        "eff_fpr": eff_fpr_level(n_evo_h, f),
                        "goodness": float(np.mean(fg)), "goodness_std": _std(fg),
                    }
                    rows.append(_row("far_ood", "split", detector, primary, other, "", fcell))
                    rows.append(_row("far_ood", "split", detector, primary, OOD_SPLIT, "",
                                     ocell))

        # --- Near-OOD: separate refit per excluded domain, still drawn from
        # the SAME fixed tr/ev -- train on (70% minus domain d), score on
        # (domain d's slice of the fixed 30%).
        if "near_ood" not in regimes:
            continue
        counts = {d: int((dom[ev] == d).sum()) for d in np.unique(dom[ev])}
        domains = sorted(d for d, n in counts.items()
                         if n >= min_domain_rows and np.unique(y[ev][dom[ev] == d]).size == 2)
        if len(domains) < 2:
            print(f"  {primary}: only {len(domains)} usable domain(s) in the fixed eval "
                  f"set ({counts}) — near-OOD skipped")
            continue

        fold_a: list[float] = []
        fold_t = {f: [] for f in fprs}
        fold_te = {f: [] for f in fprs}
        fold_n: list[int] = []
        fold_nh: list[int] = []
        fold_oa = {f: [] for f in fprs}
        fold_ot = {f: [] for f in fprs}
        fold_ote = {f: [] for f in fprs}
        for d in domains:
            tr_d = tr[dom[tr] != d]
            ev_d = ev[dom[ev] == d]
            assert d not in set(dom[tr_d])
            assert set(groups[tr_d]).isdisjoint(groups[ev_d]), "a group straddled the fold"
            a_: list[float] = []
            t_ = {f: [] for f in fprs}
            te__ = {f: [] for f in fprs}
            oa_ = {f: [] for f in fprs}
            ot_ = {f: [] for f in fprs}
            ote_ = {f: [] for f in fprs}
            for seed in seeds:
                clf = fit(spec, X, y, None, groups, tr_d, seed)
                au, per = eval_on_fprs(clf, spec, X[ev_d], y[ev_d], None, fprs)
                a_.append(au)
                for f in fprs:
                    t_[f].append(per[f][0])
                    te__[f].append(per[f][1])
                oau, oper = eval_on_fprs(clf, spec, oX, oy, None, fprs)
                for f in fprs:
                    oa_[f].append(oau)
                    ot_[f].append(oper[f][0])
                    ote_[f].append(oper[f][1])
            n_h = int((y[ev_d] == 0).sum())
            fold_a.append(float(np.mean(a_)))
            fold_n.append(len(ev_d))
            fold_nh.append(n_h)
            for f in fprs:
                fold_t[f].append(float(np.mean(t_[f])))
                fold_te[f].append(float(np.mean(te__[f])))
                fold_oa[f].append(float(np.mean(oa_[f])))
                fold_ot[f].append(float(np.mean(ot_[f])))
                fold_ote[f].append(float(np.mean(ote_[f])))
            for f in fprs:
                fold_cell = {
                    "fpr": f, "n_train": len(tr_d), "n_test": len(ev_d), "n_test_human": n_h,
                    "n_seeds": len(seeds),
                    "auroc": float(np.mean(a_)), "auroc_std": "",
                    "tpr_at_fpr": float(np.mean(t_[f])), "tpr_at_fpr_std": "",
                    "tpr_eff": float(np.mean(te__[f])), "tpr_eff_std": "",
                    "eff_fpr": eff_fpr_level(n_h, f),
                    "goodness": goodness(float(np.mean(a_)), float(np.mean(te__[f])),
                                        combine, alpha=f),
                    "goodness_std": "",
                }
                rows.append(_row("near_ood", "fold", detector, primary, primary, d, fold_cell))

        for f in fprs:
            gf = [goodness(ai, ei, combine, alpha=f) for ai, ei in zip(fold_a, fold_te[f])]
            cell = {
                "fpr": f, "n_train": "",
                "n_test": int(round(float(np.mean(fold_n)))),
                "n_test_human": int(round(float(np.mean(fold_nh)))),
                "n_seeds": len(seeds),
                "auroc": float(np.mean(fold_a)), "auroc_std": _std(fold_a),
                "tpr_at_fpr": float(np.mean(fold_t[f])), "tpr_at_fpr_std": _std(fold_t[f]),
                "tpr_eff": float(np.mean(fold_te[f])), "tpr_eff_std": _std(fold_te[f]),
                "eff_fpr": float(np.mean([eff_fpr_level(nh, f) for nh in fold_nh])),
                "goodness": float(np.mean(gf)), "goodness_std": _std(gf),
            }
            rows.append(_row("near_ood", "split", detector, primary, primary, "", cell))

            ogf = [goodness(ai, ei, combine, alpha=f) for ai, ei in zip(fold_oa[f], fold_ote[f])]
            ocell = {
                "fpr": f, "n_train": "", "n_test": "", "n_test_human": on_h,
                "n_seeds": len(seeds),
                "auroc": float(np.mean(fold_oa[f])), "auroc_std": _std(fold_oa[f]),
                "tpr_at_fpr": float(np.mean(fold_ot[f])), "tpr_at_fpr_std": _std(fold_ot[f]),
                "tpr_eff": float(np.mean(fold_ote[f])), "tpr_eff_std": _std(fold_ote[f]),
                "eff_fpr": eff_fpr_level(on_h, f),
                "goodness": float(np.mean(ogf)), "goodness_std": _std(ogf),
            }
            rows.append(_row("near_ood", "split", detector, primary, OOD_SPLIT, "", ocell))
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
    ap.add_argument("--splits", nargs=2, default=TRAIN_SPLITS, metavar="KEY",
                    help=f"exactly 2 training splits, paired as primary/other in both "
                         f"orientations (default: {TRAIN_SPLITS})")
    ap.add_argument("--ood-split", default=OOD_SPLIT, metavar="KEY",
                    help=f"eval-only corpus, never trained on (default: {OOD_SPLIT!r})")
    ap.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                    help="model-fit seeds (init/SGD variance only -- the data split "
                         "itself is fixed by --split-seed, see --split-seed)")
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED,
                    help=f"the ONE seed for the fixed 70/30 split per corpus, drawn once "
                         f"and reused across every scenario/model-fit seed (default: "
                         f"{SPLIT_SEED})")
    ap.add_argument("--test-size", type=float, default=0.30,
                    help="eval-set fraction of the fixed split (default: 0.30)")
    ap.add_argument("--fpr", nargs="+", type=float, default=DEFAULT_FPRS, metavar="LEVEL",
                    help=f"oracle FPR level(s) (default: {DEFAULT_FPRS})")
    ap.add_argument("--combine", choices=("hmean", "gmean", "amean"), default="hmean",
                    help="AT-HM = this mean of Gini(AUROC) and oracle TPR (default: hmean)")
    ap.add_argument("--group-col", default="id")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--min-domain-rows", type=int, default=100,
                    help="Near-OOD regime only: a domain below this row count within the "
                         "fixed 30% eval set is dropped (default: 100)")
    ap.add_argument("--zs-columns", nargs="*", default=ZS_COLUMNS, metavar="COLUMN",
                    help=f"zero-shot score column(s), each scored once per fixed 30% eval "
                         f"set and reused across every regime (default: {ZS_COLUMNS})")
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
    zs_cols = tuple(c for c, _, _ in zs_specs)

    print(f"detectors: {[s['label'] for s in SPECS.values()]}")
    if zs_specs:
        print(f"zero-shot: {[det for _, det, _ in zs_specs]}")
    elif zs_columns:
        print(f"zero-shot: none of {zs_columns} found in every split — skipped")
    print(f"corpora  : {[s for s, _ in train_splits]}   OOD eval: {args.ood_split!r}")
    print(f"seeds    : {args.seeds}   split-seed: {args.split_seed}   fpr: {args.fpr}   "
          f"regimes: {args.regimes}\n")

    print("=== fixed 70/30 split per corpus ===")
    data = load_fixed_corpora(train_splits, args.group_col, args.human_label,
                              args.split_seed, args.test_size, zs_cols)
    oX, oy, on_h, _, o_raw = _load_ood(ood_path, args.human_label, zs_cols)
    print(f"  OOD eval corpus {args.ood_split!r}: {len(oy)} rows, {on_h} human "
          f"(never trained on)")

    zs_fixed, zs_n_human = (
        compute_zs_fixed_reads(data, oy, o_raw, on_h, zs_specs, args.fpr)
        if zs_specs else ({}, {}))

    keys = [k for k, _ in train_splits]
    regimes = set(args.regimes)
    rows: list[dict] = []
    for primary, other in [(keys[0], keys[1]), (keys[1], keys[0])]:
        print(f"\n=== primary={primary!r}, other={other!r} ===")
        rows += compute_trained_for_corpus(primary, other, data, oX, oy, on_h, args.seeds,
                                           args.fpr, args.combine, args.min_domain_rows,
                                           regimes)
        if zs_specs:
            rows += zs_rows_for_corpus(primary, other, zs_specs, zs_fixed, zs_n_human,
                                       args.fpr, args.combine, regimes)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "performance.csv"
    pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
    _print_table(rows)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
