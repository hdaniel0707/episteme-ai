"""Baseline comparison plots + metrics (plan §2): our model vs the frozen baselines.

Consumes ``results/<run>/baselines.parquet`` (written by main_train) + the best seed's
predictions, and produces into ``results/<run>/aggregate/baselines/``:

  * baseline_agreement_<run>.png    — % exact / ±1 class agreement with the TRUE class
                                       per scorer (+ quadratic-weighted kappa), test split
  * baseline_correlation_<run>.png  — pairwise Spearman heatmap + ours-vs-best-baseline
                                       scatter, test split
  * baseline_metrics.csv            — spearman/pearson/kendall/mae/agreement per scorer

``baseline_leaderboard`` (called by run_all_workflows after a queue) writes one grouped
bar chart of test Spearman for every trained run vs the shared baselines.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau, pearsonr, spearmanr  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

from . import palette

OURS = "ours"


# ------------------------------------------------------------------ helpers

def to_classes_by_thresholds(scores: np.ndarray, thresholds: list[float],
                             reference: np.ndarray | None = None) -> np.ndarray:
    """Bin scores into ordinal class ids using percentile thresholds computed on the
    scorer's own distribution (``reference`` defaults to ``scores``), so every scorer is
    compared on the same footing regardless of its absolute scale."""
    ref = np.asarray(reference if reference is not None else scores, dtype=float)
    ref = ref[np.isfinite(ref)]
    cuts = np.quantile(ref, thresholds) if len(ref) else np.array(thresholds)
    return np.digitize(np.asarray(scores, dtype=float), cuts)


def _weighted_kappa(y_true_cls, pred_cls) -> float:
    try:
        return float(cohen_kappa_score(y_true_cls, pred_cls, weights="quadratic"))
    except ValueError:
        return float("nan")


def baseline_metrics(y_true: np.ndarray, scores_by_scorer: dict[str, np.ndarray],
                     thresholds: list[float]) -> pd.DataFrame:
    """One row per scorer: rank correlations + class-agreement statistics vs y_true."""
    y = np.asarray(y_true, dtype=float)
    true_cls = to_classes_by_thresholds(y, thresholds)
    rows = []
    for name, s in scores_by_scorer.items():
        s = np.asarray(s, dtype=float)
        m = np.isfinite(y) & np.isfinite(s)
        if m.sum() < 3:
            continue
        pred_cls = to_classes_by_thresholds(s[m], thresholds)
        tc = true_cls[m]
        rows.append({
            "scorer": name,
            "n": int(m.sum()),
            "spearman": float(spearmanr(y[m], s[m]).correlation),
            "pearson": float(pearsonr(y[m], s[m])[0]),
            "kendall": float(kendalltau(y[m], s[m]).correlation),
            "mae": float(np.mean(np.abs(y[m] - s[m]))),
            "exact_class_pct": float((pred_cls == tc).mean() * 100.0),
            "within1_class_pct": float((np.abs(pred_cls - tc) <= 1).mean() * 100.0),
            "weighted_kappa": _weighted_kappa(tc, pred_cls),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures

def plot_baseline_agreement(metrics: pd.DataFrame, n_classes: int, wf_name: str,
                            path: Path) -> None:
    """Grouped bars: % exact same class + % within ±1 class per scorer (vs TRUE class)."""
    if metrics.empty:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scorers = metrics["scorer"].tolist()
    x = np.arange(len(scorers))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(scorers) * 1.6), 5))
    palette.apply_style()

    # Per-scorer colors: exact = full alpha, ±1 = reduced alpha
    colors_exact = [palette.color_for(s) for s in scorers]
    b1 = ax.bar(x - w / 2, metrics["exact_class_pct"], w, color=colors_exact,
                alpha=0.85, label="exact same class")
    b2 = ax.bar(x + w / 2, metrics["within1_class_pct"], w, color=colors_exact,
                alpha=0.45, label="within ±1 class")
    for bars in (b1, b2):
        for rect in bars:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.8,
                    f"{rect.get_height():.0f}%", ha="center", va="bottom", fontsize=8)
    ax.axhline(100.0 / max(1, n_classes), color="gray", ls="--", lw=1.2,
               label=f"random agreement ({100.0 / max(1, n_classes):.0f}%)")
    labels = [f"{s}\nκ={k:.2f}" for s, k in zip(scorers, metrics["weighted_kappa"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% of test papers in the same impact class as the TRUE label")
    ax.set_ylim(0, 105)
    ax.set_title(f"Class agreement with the true impact class — {wf_name} vs baselines "
                 f"({n_classes} classes, test split)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {path.name}")


def plot_baseline_correlation(scores: pd.DataFrame, class_labels: np.ndarray | None,
                              wf_name: str, path: Path) -> None:
    """Left: pairwise Spearman heatmap of {target, ours, baselines}; right: ours vs the
    strongest external baseline scatter — the 'good abstract is good everywhere' figure."""
    cols = [c for c in scores.columns if scores[c].notna().sum() >= 3]
    if len(cols) < 2:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub = scores[cols].astype(float)

    corr = np.full((len(cols), len(cols)), np.nan)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            m = sub[a].notna() & sub[b].notna()
            if m.sum() >= 3:
                corr[i, j] = spearmanr(sub.loc[m, a], sub.loc[m, b]).correlation

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    im = ax1.imshow(corr, cmap=palette.DIVERGING_CMAP, vmin=-1, vmax=1,
                    interpolation="nearest")
    for i in range(len(cols)):
        for j in range(len(cols)):
            if np.isfinite(corr[i, j]):
                ax1.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if abs(corr[i, j]) > 0.6 else "black")
    ax1.set_xticks(range(len(cols)))
    ax1.set_xticklabels(cols, rotation=35, ha="right", fontsize=9)
    ax1.set_yticks(range(len(cols)))
    ax1.set_yticklabels(cols, fontsize=9)
    ax1.set_title("Pairwise Spearman ρ (test split)")
    fig.colorbar(im, ax=ax1, shrink=0.85)

    # scatter vs the strongest external baseline (prefer naip when present)
    externals = [c for c in cols if c not in ("target", OURS)]
    if OURS in cols and externals:
        i_ours = cols.index(OURS)
        best = "naip" if "naip" in externals else max(
            externals, key=lambda c: abs(corr[i_ours, cols.index(c)])
            if np.isfinite(corr[i_ours, cols.index(c)]) else -1)
        m = sub[OURS].notna() & sub[best].notna()
        rho = corr[i_ours, cols.index(best)]
        if class_labels is not None and len(class_labels) == len(sub):
            cls = pd.Series(class_labels)[m.to_numpy()]
            for c in cls.dropna().unique():
                sel = (cls == c).to_numpy()
                ax2.scatter(sub.loc[m, OURS].to_numpy()[sel],
                            sub.loc[m, best].to_numpy()[sel], s=14, alpha=0.55, label=str(c))
            ax2.legend(title="true class", fontsize=8)
        else:
            ax2.scatter(sub.loc[m, OURS], sub.loc[m, best], s=14, alpha=0.55,
                        color=palette.DUSK_BLUE)
        lim = [0, 1]
        ax2.plot(lim, lim, color="gray", ls="--", lw=1)
        ax2.set_xlabel(f"our model score ({wf_name})")
        ax2.set_ylabel(f"{best} score")
        ax2.set_title(f"Ours vs {best}  (Spearman ρ = {rho:.3f})")
        ax2.grid(True, alpha=0.3)
    fig.suptitle(f"Baseline correlation — {wf_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ------------------------------------------------------------------ per-run entry

def _best_seed_result(wf_dir: Path, task: str = "reg",
                      class_thresholds: list[float] | None = None) -> dict | None:
    """The best seed's result.json, by TEST correctness % - the same selection rule every
    other "best seed" plot uses."""
    from workflow.src.iwf.evaluation.plots import _pick_best_seed
    seeds = []
    for sd in sorted(Path(wf_dir).glob("seed_*")):
        rj = sd / "result.json"
        if not rj.is_file():
            continue
        seeds.append(json.loads(rj.read_text(encoding="utf-8")))
    return _pick_best_seed(seeds, task, class_thresholds)


def _split_preds_from_result(best: dict | None, split: str
                             ) -> tuple[np.ndarray, np.ndarray] | None:
    """(y_true, y_pred) for one split from a result.json dict (reg scale)."""
    if not best or split not in best:
        return None
    t = best[split]
    if not isinstance(t, dict) or not t.get("y_true") or not t.get("y_pred"):
        return None
    return np.asarray(t["y_true"], dtype=float), np.asarray(t["y_pred"], dtype=float)


def _best_seed_test_preds(wf_dir: Path, task: str = "reg",
                          class_thresholds: list[float] | None = None
                          ) -> tuple[np.ndarray, np.ndarray] | None:
    """(y_true, y_pred) on the test split from the best seed's result.json (reg scale)."""
    return _split_preds_from_result(_best_seed_result(wf_dir, task, class_thresholds), "test")


BASELINE_COLS = ("naip", "specter2_knn", "llm_judge", "venue_ais",
                 "forecite", "cited_potential")


def _split_scores(split_df: pd.DataFrame, ours: np.ndarray | None,
                  task: str, n_classes: int) -> tuple[np.ndarray, dict]:
    """(y_true, {scorer: scores}) for one split's baselines.parquet slice."""
    y_true = pd.to_numeric(split_df["target"], errors="coerce").to_numpy(dtype=float)
    scores: dict[str, np.ndarray] = {}
    if ours is not None and len(ours) == len(split_df):
        o = ours / max(1, n_classes - 1) if task == "clf" else ours
        scores[OURS] = np.asarray(o, dtype=float)
    for col in BASELINE_COLS:
        if col in split_df.columns and split_df[col].notna().sum() >= 3:
            scores[col] = pd.to_numeric(split_df[col], errors="coerce").to_numpy(dtype=float)
    return y_true, scores


def generate_baseline_plots(wf_type: str, task: str, wf_dir: Path, settings) -> None:
    """Produce the comparison figures + metrics CSV for one run, for every split. The CSV
    ``baseline_metrics.csv`` gets a ``split`` column (train/val/test) so the model can be
    benchmarked against the baselines on each split, not just test. Never raises.

    ``spearman`` here is Spearman's rank correlation ρ between a scorer's ranking of the
    papers and the true impact-target ranking on that split: +1 = identical ordering,
    0 = unrelated, −1 = reversed. It is the headline "does this scorer rank papers by
    impact the way the ground truth does" number. (This is ρ, not a p-value.)
    """
    wf_dir = Path(wf_dir)
    bp = wf_dir / "baselines.parquet"
    if not bp.is_file():
        return
    df = pd.read_parquet(bp)
    if "split" not in df.columns or "target" not in df.columns:
        return

    thresholds = [float(t) for t in
                  (getattr(settings, "class_thresholds", None) or [0.25, 0.5, 0.75])]
    n_classes = len(thresholds) + 1
    out = wf_dir / "aggregate" / "baselines"
    out.mkdir(parents=True, exist_ok=True)

    best = _best_seed_result(wf_dir, task, thresholds)

    # Per-split metrics table (train/val/test) → one CSV
    per_split_metrics: list[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        split_df = df[df["split"].astype(str) == split].reset_index(drop=True)
        if len(split_df) < 5:
            continue
        preds = _split_preds_from_result(best, split)
        ours = preds[1] if preds is not None else None
        y_true, scores = _split_scores(split_df, ours, task, n_classes)
        if not scores:
            continue
        m = baseline_metrics(y_true, scores, thresholds)
        if m.empty:
            continue
        m.insert(0, "split", split)
        per_split_metrics.append(m)

    if not per_split_metrics:
        return
    metrics_all = pd.concat(per_split_metrics, ignore_index=True)
    metrics_all.to_csv(out / "baseline_metrics.csv", index=False)

    # Figures are drawn on the test split (the headline comparison)
    test = df[df["split"].astype(str) == "test"].reset_index(drop=True)
    if len(test) < 5:
        return
    preds = _split_preds_from_result(best, "test")
    ours = preds[1] if preds is not None else None
    y_true, scores = _split_scores(test, ours, task, n_classes)
    if not scores:
        return
    test_metrics = metrics_all[metrics_all["split"] == "test"].drop(columns=["split"])

    try:
        plot_baseline_agreement(test_metrics, n_classes, wf_dir.name,
                                out / f"baseline_agreement_{wf_dir.name}.png")
    except Exception as e:
        print(f"[baselines] {wf_dir.name} agreement plot skipped ({e})")
    try:
        frame = pd.DataFrame({"target": y_true, **scores})
        cls = test["class_label"].to_numpy() if "class_label" in test.columns else None
        plot_baseline_correlation(frame, cls, wf_dir.name,
                                  out / f"baseline_correlation_{wf_dir.name}.png")
    except Exception as e:
        print(f"[baselines] {wf_dir.name} correlation plot skipped ({e})")


# ------------------------------------------------------------------ cross-run summary

def _leaderboard_panel(ax, runs: dict[str, pd.DataFrame], scorers: list[str],
                       split: str) -> bool:
    """Draw one grouped-bar panel of Spearman ρ per run×scorer for a given split.
    Returns True if any bars were drawn."""
    x = np.arange(len(runs))
    w = 0.8 / max(1, len(scorers))
    drew = False
    for si, scorer in enumerate(scorers):
        vals = []
        for m in runs.values():
            sel = m[(m["scorer"] == scorer) & (m["split"] == split)]
            vals.append(float(sel["spearman"].iloc[0]) if len(sel) else np.nan)
        if np.isfinite(vals).any():
            drew = True
        ax.bar(x + (si - (len(scorers) - 1) / 2) * w, vals, w,
               color=palette.color_for(scorer), alpha=0.85, label=scorer)
    ax.set_xticks(x)
    ax.set_xticklabels(list(runs), rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Spearman ρ vs true impact target")
    ax.set_title(f"{split} split")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    return drew


def baseline_leaderboard(results_dir: Path) -> None:
    """Grouped bars of Spearman-vs-target for every run's model + its baselines, one panel
    per split (train / val / test), from each run's aggregate/baselines/baseline_metrics.csv.
    Also writes ``summary/baseline_leaderboard.csv`` — the tidy (run, split, scorer, metric)
    table behind the figure. Never raises.

    Spearman ρ is the rank correlation between a scorer's ordering of the papers and the
    true impact-target ordering (+1 best, 0 unrelated); it is the headline benchmark number.
    """
    results_dir = Path(results_dir)
    runs: dict[str, pd.DataFrame] = {}
    for d in sorted(results_dir.iterdir()) if results_dir.is_dir() else []:
        csv = d / "aggregate" / "baselines" / "baseline_metrics.csv"
        if csv.is_file():
            try:
                m = pd.read_csv(csv)
                if "split" not in m.columns:      # backward-compat: old test-only CSVs
                    m["split"] = "test"
                runs[d.name] = m
            except Exception:
                continue
    if not runs:
        return

    out = results_dir / "summary"
    out.mkdir(parents=True, exist_ok=True)

    # Tidy summary CSV (statistics only — one row per run×split×scorer)
    tidy = []
    for run_name, m in runs.items():
        for _, r in m.iterrows():
            tidy.append({"run": run_name, **r.to_dict()})
    pd.DataFrame(tidy).to_csv(out / "baseline_leaderboard.csv", index=False)

    scorers: list[str] = []
    for m in runs.values():
        for s in m["scorer"]:
            if s not in scorers:
                scorers.append(s)
    splits = [sp for sp in ("train", "val", "test")
              if any((m["split"] == sp).any() for m in runs.values())]
    if not splits:
        return

    palette.apply_style()
    fig, axes = plt.subplots(1, len(splits), figsize=(max(8, len(runs) * 1.8) * len(splits) / 3 + 4, 5.5),
                             squeeze=False)
    for ax, split in zip(axes[0], splits):
        _leaderboard_panel(ax, runs, scorers, split)
    axes[0][-1].legend(fontsize=8, ncols=min(4, len(scorers)))
    fig.suptitle("Model-vs-baseline leaderboard — Spearman ρ with the true impact target "
                 "(higher = better)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "baseline_leaderboard.png", dpi=150)
    plt.close(fig)
    print(f"[baselines] leaderboard -> {out / 'baseline_leaderboard.png'} "
          f"(+ baseline_leaderboard.csv)")
