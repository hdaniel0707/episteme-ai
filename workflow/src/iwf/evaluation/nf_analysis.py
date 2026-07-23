"""Novelty/feasibility statistics (plan §3.3): are they related to impact, and do they
help the metadata models?

Reads ``results/<run>/baselines.parquet`` and writes into
``results/<run>/aggregate/nf/``:

  * nf_correlation_matrix.png — Spearman heatmap of {target, novelty, feasibility,
    external scores, metadata} with significance stars, + partial Spearman of
    novelty/feasibility vs target controlling for venue frequency and h-index.
  * nf_binned_impact.png      — mean target (95% CI) per novelty / feasibility decile
    (the classic inverted-U novelty check, Uzzi 2013).
  * nf_stats.json             — the raw numbers behind both figures.

The ablation figure (``nf_ablation_delta.png``, results/summary/) compares runs whose
run_config differs only in use_novelty/use_feasibility — see ``nf_ablation``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

from . import palette

CANDIDATE_COLS = ["target", "novelty", "feasibility", "naip", "specter2_knn",
                  "llm_judge", "venue_ais", "meta_h_index", "meta_venue_freq"]


def _stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    """Rank-based partial correlation of x,y given controls (residualized ranks)."""
    def ranks(a):
        return rankdata(a, method="average").astype(float)

    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(controls), axis=1)
    if m.sum() < 5:
        return float("nan")
    rx, ry = ranks(x[m]), ranks(y[m])
    C = np.column_stack([np.ones(m.sum()),
                         *[ranks(controls[m, j]) for j in range(controls.shape[1])]])

    def resid(v):
        beta, *_ = np.linalg.lstsq(C, v, rcond=None)
        return v - C @ beta

    ex, ey = resid(rx), resid(ry)
    if np.std(ex) == 0 or np.std(ey) == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def nf_correlation_matrix(df: pd.DataFrame, wf_name: str, path: Path) -> dict:
    """Spearman heatmap with significance stars; returns the numbers as a dict."""
    cols = [c for c in CANDIDATE_COLS
            if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 5]
    if len(cols) < 2:
        return {}
    X = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in cols}
    n = len(cols)
    rho = np.full((n, n), np.nan)
    pval = np.full((n, n), np.nan)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            m = np.isfinite(X[a]) & np.isfinite(X[b])
            if m.sum() >= 5:
                r = spearmanr(X[a][m], X[b][m])
                rho[i, j], pval[i, j] = r.correlation, r.pvalue

    fig, ax = plt.subplots(figsize=(1.1 * n + 3, 1.0 * n + 2))
    im = ax.imshow(rho, cmap=palette.DIVERGING_CMAP, vmin=-1, vmax=1, interpolation="nearest")
    for i in range(n):
        for j in range(n):
            if np.isfinite(rho[i, j]):
                ax.text(j, i, f"{rho[i, j]:.2f}{_stars(pval[i, j])}", ha="center",
                        va="center", fontsize=8,
                        color="white" if abs(rho[i, j]) > 0.6 else "black")
    ax.set_xticks(range(n)); ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(cols, fontsize=9)
    ax.set_title(f"Spearman correlations — {wf_name} (test split)\n"
                 "* p<.05  ** p<.01  *** p<.001")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # partial correlations vs target, controlling for venue_freq + h_index
    partials = {}
    ctrl_cols = [c for c in ("meta_venue_freq", "meta_h_index") if c in cols]
    if "target" in cols and ctrl_cols:
        C = np.column_stack([X[c] for c in ctrl_cols])
        for feat in ("novelty", "feasibility"):
            if feat in cols:
                partials[feat] = partial_spearman(X[feat], X["target"], C)
        if partials:
            txt = "partial ρ vs target | " + ", ".join(ctrl_cols) + ":  " + "  ".join(
                f"{k}={v:.3f}" for k, v in partials.items())
            fig.text(0.5, 0.005, txt, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return {"columns": cols, "spearman": np.where(np.isfinite(rho), rho, None).tolist(),
            "partial_vs_target": partials}


def nf_binned_impact(df: pd.DataFrame, wf_name: str, path: Path,
                     n_bins: int = 10) -> dict:
    """Mean target (95% CI) per novelty / feasibility decile — the inverted-U check."""
    feats = [c for c in ("novelty", "feasibility") if c in df.columns]
    if "target" not in df.columns or not feats:
        return {}
    y = pd.to_numeric(df["target"], errors="coerce")
    stats: dict = {}
    fig, axes = plt.subplots(1, len(feats), figsize=(6 * len(feats), 4.5), squeeze=False)
    for ax, feat in zip(axes[0], feats):
        x = pd.to_numeric(df[feat], errors="coerce")
        m = x.notna() & y.notna()
        if m.sum() < n_bins * 3:
            ax.set_title(f"{feat}: not enough data")
            continue
        bins = pd.qcut(x[m], q=n_bins, labels=False, duplicates="drop")
        centers, means, cis = [], [], []
        for b in sorted(bins.dropna().unique()):
            vals = y[m][bins == b]
            centers.append((b + 0.5) / n_bins)
            means.append(float(vals.mean()))
            cis.append(1.96 * float(vals.std()) / max(1.0, np.sqrt(len(vals))))
        ax.errorbar(centers, means, yerr=cis, fmt="o-", color=palette.DUSK_BLUE, capsize=3)
        ax.set_xlabel(f"{feat} decile (0-1)")
        ax.set_ylabel("mean impact target (95% CI)")
        ax.set_title(f"Impact vs {feat}")
        ax.grid(True, alpha=0.3)
        stats[feat] = {"bin_centers": centers, "mean_target": means, "ci95": cis}
    fig.suptitle(f"Impact by novelty/feasibility decile — {wf_name} (test split)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return stats


def generate_nf_analysis(wf_type: str, wf_dir: Path, settings) -> None:
    """Phase-7 entry: figures + nf_stats.json for one run. Never raises upstream."""
    wf_dir = Path(wf_dir)
    bp = wf_dir / "baselines.parquet"
    if not bp.is_file():
        return
    df = pd.read_parquet(bp)
    if "split" in df.columns:
        test = df[df["split"].astype(str) == "test"].reset_index(drop=True)
        if len(test) >= 30:  # tiny test splits: fall back to all rows (label-free features)
            df = test
    if "novelty" not in df.columns and "feasibility" not in df.columns:
        return
    out = wf_dir / "aggregate" / "nf"
    stats = {
        "correlations": nf_correlation_matrix(df, wf_dir.name,
                                              out / "nf_correlation_matrix.png"),
        "binned": nf_binned_impact(df, wf_dir.name, out / "nf_binned_impact.png"),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "nf_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ ablation summary

def _task_of_wf(wf: str) -> str:
    return "clf" if "clf" in str(wf) else "reg"


def nf_ablation(results_dir: Path) -> None:
    """Do the novelty/feasibility (and metadata) features help?

    The old version bucketed runs into the four (novelty, feasibility) combos and grouped
    by workflow type. When every meta run enables both features, all bars collapse into the
    single "both" bucket and the figure is uninformative. This version fixes the category
    choice by plotting a **delta over the plain-embedding baseline** of the same task
    (emb_clf/emb_reg with no fused features): each meta run shows how much its feature set
    lifts the primary metric above that no-feature reference. Only feature combos that
    actually occur in the run set are drawn, and a tidy CSV is written alongside.
    """
    results_dir = Path(results_dir)
    entries = []
    baselines: dict[str, list[float]] = {"clf": [], "reg": []}
    for d in sorted(results_dir.iterdir()) if results_dir.is_dir() else []:
        rc, ms = d / "run_config.json", d / "aggregate" / "metrics_summary.json"
        if not (rc.is_file() and ms.is_file()):
            continue
        try:
            cfg = json.loads(rc.read_text(encoding="utf-8"))
            summ = json.loads(ms.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        wf = str(cfg.get("workflow_type", ""))
        primary = summ.get("primary_metric", "spearman")
        mean = summ.get(f"mean_{primary}")
        if mean is None:
            continue
        task = _task_of_wf(wf)
        is_meta = "meta" in wf
        # Plain-embedding reference run for its task. use_novelty/use_feasibility are
        # config-global fields inherited by EVERY run's settings snapshot, but they only
        # take effect for meta feature-builds (feature_builder.py only reads them when the
        # 'novelty'/'feasibility' columns exist, which the z-only workflows never build) —
        # so gating on wf identity (plain emb_clf/emb_reg), not those flags, is what
        # actually identifies the no-fused-features baseline.
        if wf in ("emb_clf", "emb_reg"):
            # plain-embedding reference run for its task
            baselines[task].append(float(mean))
        if is_meta:
            entries.append({"run": d.name, "wf": wf, "task": task,
                            "novelty": bool(cfg.get("use_novelty")),
                            "feasibility": bool(cfg.get("use_feasibility")),
                            "metadata": bool(cfg.get("use_metadata")),
                            "metric": primary, "mean": float(mean),
                            "std": float(summ.get(f"std_{primary}") or 0.0)})
    if not entries:
        return
    df = pd.DataFrame(entries)

    def _combo(row) -> str:
        parts = [p for p, on in (("novelty", row["novelty"]),
                                 ("feasibility", row["feasibility"]),
                                 ("metadata", row["metadata"])) if on]
        return "+".join(parts) if parts else "meta(none)"

    df["combo"] = df.apply(_combo, axis=1)
    base_mean = {t: (float(np.mean(v)) if v else np.nan) for t, v in baselines.items()}
    df["baseline"] = df["task"].map(base_mean)
    df["delta"] = df["mean"] - df["baseline"]

    out = results_dir / "summary"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "nf_ablation_delta.csv", index=False)

    # Only combos that actually occur; group along workflow type
    combos = list(dict.fromkeys(df["combo"]))
    wfs = sorted(df["wf"].unique())
    x = np.arange(len(wfs))
    w = 0.8 / max(1, len(combos))
    colors = palette.class_colors(len(combos))
    have_baseline = df["baseline"].notna().any()
    yvals = "delta" if have_baseline else "mean"

    fig, ax = plt.subplots(figsize=(max(7, len(wfs) * 2.2), 5))
    for ci, combo in enumerate(combos):
        means, stds = [], []
        for wf in wfs:
            sel = df[(df["wf"] == wf) & (df["combo"] == combo)]
            means.append(float(sel[yvals].iloc[-1]) if len(sel) else np.nan)
            stds.append(float(sel["std"].iloc[-1]) if len(sel) else 0.0)
        ax.bar(x + (ci - (len(combos) - 1) / 2) * w, means, w, yerr=stds,
               color=colors[ci], alpha=0.9, capsize=3, label=combo)
    ax.axhline(0.0, color=palette.GRAY, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(wfs, fontsize=9)
    if have_baseline:
        ax.set_ylabel("Δ primary metric vs plain-embedding baseline (same task)")
        ax.set_title("Do fused novelty/feasibility/metadata features help?\n"
                     "(lift over the no-feature embedding baseline)")
    else:
        ax.set_ylabel("mean primary test metric across seeds")
        ax.set_title("Fused-feature runs by feature set\n"
                     "(no plain-embedding baseline found for a delta)")
    ax.legend(title="fused features", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "nf_ablation_delta.png", dpi=150)
    plt.close(fig)
    print(f"[nf] ablation figure -> {out / 'nf_ablation_delta.png'} (+ nf_ablation_delta.csv)")
