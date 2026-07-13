"""ALL evaluation plots — the single plotting module for every workflow.

Every workflow type gets the same plot suite from :func:`generate_workflow_plots`
(called by ``workflow_results.finalize_workflow``); only the classification /
regression variants differ:

**Classification (emb_clf, emb_meta_clf, emb_all_meta_clf, tfm_clf):**
  correctness_bars, learning_curve, confusion_matrix, class_boxplot,
  umap3d_correctness, umap3d_classes, umap3d_domain_*, nn_weights

**Regression (emb_reg, emb_meta_reg, emb_all_meta_reg, tfm_reg):**
  correctness_bars (class-binned), mse_bars, learning_curve, confusion_matrix
  (class-binned), spearman_distribution, predicted_vs_actual, umap3d_error,
  umap3d_domain_*, nn_weights

UMAP and NN-weight plots need z embeddings / saved joblib models; they are skipped
gracefully for text-transformer workflows that have neither. A few standalone
figures used by single phases live at the bottom (dim curve, generated-vs-human,
text-transformer training curves).
"""

from __future__ import annotations

import json
import os
import re
import webbrowser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402  (registers the '3d' projection)

from . import palette

MAX_UMAP_POINTS = 6000

# Re-export palette constants for backwards compatibility
CORRECT_COLOR = palette.CORRECT_COLOR
WRONG_COLOR = palette.WRONG_COLOR
ERROR_COLORSCALE = palette.ERROR_COLORSCALE_PLOTLY
ERROR_CMAP = palette.ERROR_CMAP_MATPLOTLIB
SPLIT_SHADES = palette.SPLIT_SHADES

# Weights below this fraction of a layer's max |weight| are not drawn (keeps the
# NN-weight HTML responsive; near-zero connections add nothing but lag).
NN_WEIGHT_DISPLAY_THRESHOLD = 0.15
NN_MAX_HTML_LINES_PER_LAYER = 4000


# ===================================================================== shared helpers

def umap_3d(Z: np.ndarray, seed: int = 42) -> np.ndarray:
    import umap
    return np.asarray(
        umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=seed)
        .fit_transform(Z), dtype=float,
    )


def _ordinal_color(idx: int, total: int) -> str:
    colors = palette.ordinal_colors(total)
    return colors[min(idx, len(colors) - 1)]


def _impact_sizes(values, lo=2.0, hi=6.5):
    v = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0).clip(0, 1)
    return lo + (hi - lo) * v


def _open_browser(path: Path) -> None:
    if os.getenv("UMAP_OPEN_BROWSER", "true").strip().lower() in ("1", "true", "yes", "on"):
        try:
            webbrowser.open(path.resolve().as_uri())
        except Exception:
            pass


def _safe_domain_name(dom: str) -> str:
    return re.sub(r"[^\w\-]", "_", dom)[:60]


def _save_3d_static(groups, title: str, path: Path) -> None:
    """``groups`` = list of (label, coords[N,3], color, diameters[N])."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for label, c, color, sizes in groups:
        if len(c):
            ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=np.square(np.asarray(sizes)) * 0.7,
                       c=color, alpha=0.5, edgecolors="none", label=label)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.set_zlabel("UMAP-3")
    ax.legend(loc="upper right", markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _save_interactive(fig_plotly, path: Path) -> None:
    fig_plotly.write_html(str(path))
    print(f"[plots] interactive 3-D UMAP -> {path}")
    _open_browser(path)


def _get_domain(df: pd.DataFrame) -> pd.Series:
    def first(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return str(v[0]) if len(v) else "unknown"
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "unknown"
        s = str(v)
        return s if s and s.lower() != "nan" else "unknown"

    if "domains" in df.columns:
        return df["domains"].map(first)
    if "domain" in df.columns:
        return df["domain"].fillna("unknown").astype(str)
    if "subfield" in df.columns:
        return df["subfield"].fillna("unknown").astype(str)
    return pd.Series(["unknown"] * len(df), index=df.index)


# ===================================================================== data extraction

def _extract_split_preds(seed_data: dict) -> dict[str, dict] | None:
    """y_true / y_pred arrays for every available split from a seed's result.json.

    Return structure: ``{split: {"y_true": np.ndarray, "y_pred": np.ndarray}}``.
    """
    result: dict[str, dict] = {}
    for sp in ("train", "val", "test"):
        sp_data = seed_data.get(sp, {})
        if isinstance(sp_data, dict) and "y_true" in sp_data and "y_pred" in sp_data:
            yt = np.asarray(sp_data["y_true"])
            yp = np.asarray(sp_data["y_pred"])
            if len(yt) > 0 and len(yp) > 0:
                result[sp] = {"y_true": yt, "y_pred": yp}
    return result or None


def _get_umap_data(seed_data: dict, df: pd.DataFrame, seed: int) -> dict | None:
    """All-rows predictions + 3-D UMAP coordinates of the z embedding.

    Uses the flat ``y_all`` / ``y_pred_all`` / ``split_all`` arrays saved in result.json
    (emb/meta workflows). Falls back to stitching per-split predictions (text-transformer
    workflows). Returns None when no predictions exist at all.
    """
    if "y_all" in seed_data and "y_pred_all" in seed_data:
        y_true_all = np.asarray(seed_data["y_all"])
        y_pred_all = np.asarray(seed_data["y_pred_all"])
    else:
        split_preds = _extract_split_preds(seed_data)
        if split_preds is None:
            return None
        y_true_all = np.concatenate([split_preds[sp]["y_true"] for sp in split_preds])
        y_pred_all = np.concatenate([split_preds[sp]["y_pred"] for sp in split_preds])

    coords, idx = None, None
    if "z" in df.columns:
        try:
            z_arr = np.array(df["z"].tolist(), dtype=np.float32)
            n = len(z_arr)
            rng = np.random.default_rng(seed)
            idx_full = (rng.choice(n, MAX_UMAP_POINTS, replace=False)
                        if n > MAX_UMAP_POINTS else np.arange(n))
            coords = umap_3d(z_arr[idx_full], seed=seed)
            # Only usable when the prediction arrays align 1:1 with the parquet rows.
            idx = idx_full if len(y_true_all) == n else None
        except Exception as e:
            print(f"[plots] UMAP computation failed: {e}")

    return {"coords": coords, "idx": idx, "y_true": y_true_all, "y_pred": y_pred_all,
            "df_len": len(df)}


def _load_best_seed_data(wf_dir: Path, task: str) -> dict | None:
    """Find the best-performing seed (by best_val_metric, falling back to the primary
    test metric) and return its result.json contents."""
    metric_key = "macro_f1" if task == "clf" else "spearman"
    best, best_val = None, -np.inf
    for sd in sorted(Path(wf_dir).glob("seed_*")):
        rj = sd / "result.json"
        if not rj.is_file():
            continue
        r = json.loads(rj.read_text(encoding="utf-8"))
        val = r.get("best_val_metric")
        if not isinstance(val, (int, float)):
            val = r.get(metric_key, -np.inf)
        if isinstance(val, (int, float)) and val > best_val:
            best_val = val
            best = r
            best["_seed_dir"] = str(sd)
            best["_seed_name"] = sd.name
    return best


# ===================================================================== split bars

def _correctness_pct_bars(splits: list[str], correct: list[int], totals: list[int],
                          agg: Path, wf_type: str, title: str) -> None:
    """One olive-shaded bar per split showing % correct, item count in the x-label."""
    pct = [100.0 * c / max(t, 1) for c, t in zip(correct, totals)]
    x = np.arange(len(splits))
    colors = [SPLIT_SHADES.get(sp, "#8FA35E") for sp in splits]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x, pct, color=colors, edgecolor=palette.INK, linewidth=0.6)
    for i, p in enumerate(pct):
        ax.text(i, p + 2, f"{p:.2f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{sp} ({totals[i]})" for i, sp in enumerate(splits)])
    ax.set_ylabel("Correct Predictions (%)")
    ax.set_ylim(0, 108)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(agg / f"correctness_bars_{wf_type}.png", dpi=120)
    plt.close(fig)


def correctness_bars(split_preds: dict, agg: Path, wf_type: str, task: str,
                     thresholds: list[float] | None = None) -> None:
    """% correct per split. clf: exact class match; reg: prediction lands in the true
    impact-class bin (same thresholds as the classifier — directly comparable)."""
    splits = [s for s in ("train", "val", "test") if s in split_preds]
    if not splits:
        return
    correct, totals = [], []
    for sp in splits:
        yt = np.asarray(split_preds[sp]["y_true"])
        yp = np.asarray(split_preds[sp]["y_pred"])
        if task == "reg":
            thr = sorted(float(t) for t in (thresholds or []))
            yt = np.digitize(yt.astype(float), thr)
            yp = np.digitize(yp.astype(float), thr)
        correct.append(int((yt == yp).sum()))
        totals.append(len(yt))
    title = ("Prediction Accuracy by Split" if task == "clf"
             else "Category Accuracy by Split")
    _correctness_pct_bars(splits, correct, totals, agg, wf_type, title)


def mse_bars(split_preds: dict, agg: Path, wf_type: str) -> None:
    """Bar chart: MSE for train / val / test (regression only)."""
    splits = [s for s in ("train", "val", "test") if s in split_preds]
    if not splits:
        return
    mses = [float(np.mean((np.asarray(split_preds[sp]["y_true"], dtype=float)
                           - np.asarray(split_preds[sp]["y_pred"], dtype=float)) ** 2))
            for sp in splits]
    colors = [SPLIT_SHADES.get(sp, "#8FA35E") for sp in splits]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(splits, mses, color=colors, edgecolor=palette.INK, linewidth=0.6)
    for b, v in zip(bars, mses):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom",
                fontsize=9)
    ax.set_ylabel("MSE")
    ax.set_title("MSE by Split (best seed)")
    fig.tight_layout()
    fig.savefig(agg / "mse_bars.png", dpi=120)
    plt.close(fig)


# ===================================================================== learning curve

def learning_curve(history: dict, agg: Path, wf_type: str, task: str) -> None:
    """Training loss (left axis) + validation metric (right axis) vs epoch — identical
    layout for clf and reg; only the metric label differs."""
    metric_label = ("Validation macro-F1 (0-1, higher=better)" if task == "clf"
                    else "Validation Spearman/R² (0-1, higher=better)")
    tl = list(history.get("train_loss", []))
    vm = list(history.get("val_metric", []))
    # Text-transformer stores per-step loss; average per epoch for consistency.
    if not tl and "step_loss" in history:
        sl = history["step_loss"]
        epochs = len(vm) if vm else 1
        chunk = max(1, len(sl) // epochs)
        tl = [float(np.mean(sl[i * chunk:(i + 1) * chunk])) for i in range(epochs)]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    if tl:
        ax1.plot(range(1, len(tl) + 1), tl, color=palette.DUSK_BLUE,
                 label="Training loss (lower=better)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color=palette.DUSK_BLUE)
    ax2 = ax1.twinx()
    if vm:
        ax2.plot(range(1, len(vm) + 1), vm, color=palette.OLIVE, label=metric_label)
        best = int(np.argmax(vm))
        ax2.axvline(best + 1, color="gray", ls=":", label=f"Best epoch ({best + 1})")
    ax2.set_ylabel(metric_label, color=palette.OLIVE)
    # twinx legends don't combine automatically — merge the handles of both axes
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    if l1 or l2:
        ax1.legend(l1 + l2, lab1 + lab2, loc="best", fontsize=8)
    ax1.set_title("Training Progress by Epoch")
    fig.tight_layout()
    fig.savefig(agg / f"learning_curve_{wf_type}.png", dpi=120)
    plt.close(fig)


# ===================================================================== seed-spread figures

def _render_confusion_matrix(cm: np.ndarray, class_names: list[str], seed_label: str,
                             agg: Path, wf_type: str) -> None:
    """Confusion-matrix heatmap; ticks auto-adjust to the configured class count (2..25)."""
    n = cm.shape[0]
    ticks = class_names if len(class_names) == n else [str(i) for i in range(n)]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap=palette.register_olive_seq())
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    ax.set_xticks(range(n)); ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(ticks)
    ax.set_xlabel("Predicted Impact Class")
    ax.set_ylabel("Actual Impact Class")
    ax.set_title("Confusion Matrix (Test Set)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(agg / f"confusion_matrix_{wf_type}_{seed_label}.png", dpi=120)
    plt.close(fig)


def plot_seed_spread_figures(wf_type: str, task: str, seeds: list[dict], agg: Path,
                             class_names: list[str], class_thresholds: list[float]) -> None:
    """Across-seed figures written into aggregate/: metric spread, best-seed confusion
    matrix (clf and class-binned reg), class boxplot, predicted-vs-actual scatter."""
    class_names = class_names or []
    thr = sorted(float(t) for t in (class_thresholds or []))
    if task == "reg":
        rhos = [s.get("spearman") for s in seeds if isinstance(s.get("spearman"), (int, float))]
        if rhos:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(rhos, bins=min(10, max(3, len(rhos))), color=palette.DUSK_BLUE, alpha=0.8)
            ax.axvline(float(np.mean(rhos)), color="red", ls="--",
                       label=f"mean={np.mean(rhos):.3f}")
            ax.set_xlabel("Spearman rho (test)")
            ax.set_ylabel("seeds")
            ax.set_title(f"{wf_type}: Spearman across seeds")
            ax.legend()
            fig.tight_layout(); fig.savefig(agg / "spearman_distribution.png", dpi=120)
            plt.close(fig)
        seed_p = next((s for s in seeds
                       if isinstance(s.get("test"), dict) and s["test"].get("y_true")), None)
        if seed_p is not None:
            yt = np.asarray(seed_p["test"]["y_true"], dtype=float)
            yp = np.asarray(seed_p["test"]["y_pred"], dtype=float)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(yt, yp, s=10, alpha=0.5)
            ax.plot([0, 1], [0, 1], "r--", lw=1)
            ax.set_xlabel("true percentile"); ax.set_ylabel("predicted")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f"{wf_type}: predicted vs actual (best seed)")
            fig.tight_layout(); fig.savefig(agg / "predicted_vs_actual.png", dpi=120)
            plt.close(fig)
            # Class-binned confusion matrix: same thresholds as the classifier, so
            # regression and classification workflows are directly comparable.
            if thr and len(class_names) == len(thr) + 1:
                from sklearn.metrics import confusion_matrix as sk_cm
                cm = sk_cm(np.digitize(yt, thr), np.digitize(yp, thr),
                           labels=list(range(len(class_names))))
                _render_confusion_matrix(cm.astype(float), class_names,
                                         seed_p.get("_seed_name", "seed_0"), agg, wf_type)
    else:
        per_class = [s.get("per_class_f1") for s in seeds if s.get("per_class_f1")]
        if per_class:
            arr = [list(map(float, p)) for p in per_class if p]
            width = max(len(p) for p in arr)
            cols = [[row[j] for row in arr if len(row) > j] for j in range(width)]
            labels = (class_names if len(class_names) == width
                      else [f"class {j}" for j in range(width)])
            fig, ax = plt.subplots(figsize=(max(6, width * 1.1), 4))
            bp = ax.boxplot(cols, tick_labels=labels, patch_artist=True, showmeans=True,
                            medianprops=dict(color=palette.INK),
                            meanprops=dict(marker="D", markerfacecolor=palette.OLIVE,
                                           markeredgecolor="white", markersize=5))
            for patch in bp["boxes"]:
                patch.set_facecolor(palette.SAND)
                patch.set_alpha(0.85)
            ax.set_ylabel("F1 Score (0-1)")
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", alpha=0.3)
            ax.set_title("F1 Score by Class (across seeds)")
            fig.tight_layout()
            fig.savefig(agg / f"class_boxplot_{wf_type}.png", dpi=120)
            plt.close(fig)
        seed_c = next((s for s in seeds if s.get("confusion")), None)
        if seed_c is not None:
            _render_confusion_matrix(np.asarray(seed_c["confusion"], dtype=float),
                                     class_names, seed_c.get("_seed_name", "seed_0"),
                                     agg, wf_type)


# ===================================================================== 3-D UMAP plots

def clf_umap_correctness(coords, correct_mask, target, agg: Path, wf_type: str) -> None:
    """3-D UMAP: gold = correctly classified, purple = wrong. Size ~ impact."""
    import plotly.graph_objects as go
    sizes = _impact_sizes(target)
    fig = go.Figure()
    groups = []
    for mask, name, color in ((correct_mask, "correct", CORRECT_COLOR),
                              (~correct_mask, "wrong", WRONG_COLOR)):
        if not mask.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
            mode="markers", name=name,
            marker=dict(size=sizes[mask], color=color, opacity=0.55,
                        line=dict(width=0.2, color="white")),
        ))
        groups.append((name, coords[mask], color, sizes[mask]))
    fig.update_layout(title="UMAP: Correct vs Wrong Predictions",
                      scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2",
                                 zaxis_title="UMAP-3", aspectmode="data"),
                      margin=dict(l=0, r=0, b=0, t=40))
    _save_interactive(fig, agg / f"umap3d_correctness_{wf_type}.html")
    _save_3d_static(groups, "Correct vs Wrong Predictions",
                    agg / f"umap3d_correctness_{wf_type}.png")


def clf_umap_classes(coords, class_labels, target, class_names, agg: Path,
                     wf_type: str) -> None:
    """3-D UMAP coloured by impact class (ascending Q_n .. Q1), size ~ impact value."""
    import plotly.graph_objects as go
    sizes = _impact_sizes(target)
    fig = go.Figure()
    groups = []
    total = len(class_names)
    for j, cn in enumerate(class_names):
        m = class_labels == cn
        if not m.any():
            continue
        color = _ordinal_color(j, total)
        fig.add_trace(go.Scatter3d(
            x=coords[m, 0], y=coords[m, 1], z=coords[m, 2],
            mode="markers", name=cn,
            marker=dict(size=sizes[m], color=color, opacity=0.55,
                        line=dict(width=0.2, color="white")),
        ))
        groups.append((cn, coords[m], color, sizes[m]))
    fig.update_layout(title="UMAP: Impact Class Distribution",
                      scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2",
                                 zaxis_title="UMAP-3", aspectmode="data"),
                      legend_title_text="Impact class",
                      margin=dict(l=0, r=0, b=0, t=40))
    _save_interactive(fig, agg / f"umap3d_classes_{wf_type}.html")
    _save_3d_static(groups, "Impact Class Distribution",
                    agg / f"umap3d_classes_{wf_type}.png")


def clf_umap_domains(coords, correct_mask, class_labels, class_names, domains, agg: Path,
                     wf_type: str) -> None:
    """Per-domain 3-D UMAP: combined PNG subplot grid + one interactive HTML per domain."""
    import plotly.graph_objects as go
    unique_domains = sorted(set(domains))
    if len(unique_domains) < 2:
        return

    class_order = {cn: i for i, cn in enumerate(class_names)}
    rank = np.array([class_order.get(cl, 0) for cl in class_labels], dtype=float)
    sizes_arr = 2.0 + 5.0 * rank / max(len(class_names) - 1, 1)

    n = len(unique_domains)
    ncols = min(4, n)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig = plt.figure(figsize=(5 * ncols, 4.5 * nrows))
    for idx_d, dom in enumerate(unique_domains):
        ax = fig.add_subplot(nrows, ncols, idx_d + 1, projection="3d")
        mask_d = np.array([d == dom for d in domains])
        n_d = mask_d.sum()
        if n_d == 0:
            ax.set_title(dom, fontsize=9)
            continue
        c_ok = mask_d & correct_mask
        c_bad = mask_d & ~correct_mask
        if c_ok.any():
            ax.scatter(coords[c_ok, 0], coords[c_ok, 1], coords[c_ok, 2],
                       s=np.square(sizes_arr[c_ok]) * 0.7, c=CORRECT_COLOR, alpha=0.5,
                       edgecolors="none", label="correct")
        if c_bad.any():
            ax.scatter(coords[c_bad, 0], coords[c_bad, 1], coords[c_bad, 2],
                       s=np.square(sizes_arr[c_bad]) * 0.7, c=WRONG_COLOR, alpha=0.5,
                       edgecolors="none", label="wrong")
        ax.set_title(f"{dom} ({int(c_ok.sum())}/{n_d})", fontsize=9)
        ax.tick_params(labelsize=6)
    fig.suptitle("Per-Domain Correctness (size = class rank)", fontsize=11)
    fig.tight_layout()
    fig.savefig(agg / f"umap3d_domain_correctness_{wf_type}.png", dpi=120)
    plt.close(fig)

    for dom in unique_domains:
        mask_d = np.array([d == dom for d in domains])
        n_d = int(mask_d.sum())
        if n_d == 0:
            continue
        c_ok = mask_d & correct_mask
        c_bad = mask_d & ~correct_mask
        n_ok = int(c_ok.sum())
        fig_h = go.Figure()
        for m, name, color in ((c_ok, "correct", CORRECT_COLOR),
                               (c_bad, "wrong", WRONG_COLOR)):
            if m.any():
                fig_h.add_trace(go.Scatter3d(
                    x=coords[m, 0], y=coords[m, 1], z=coords[m, 2],
                    mode="markers", name=name,
                    marker=dict(size=sizes_arr[m], color=color, opacity=0.6,
                                line=dict(width=0.2, color="white")),
                ))
        fig_h.update_layout(
            title=f"{dom}: {n_ok}/{n_d} correct ({100 * n_ok / max(n_d, 1):.1f}%)",
            scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3",
                       aspectmode="data"),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        out_path = agg / f"umap3d_domain_{_safe_domain_name(dom)}_correctness_{wf_type}.html"
        fig_h.write_html(str(out_path))
        print(f"[plots] {wf_type}: domain HTML -> {out_path.name}")


def reg_umap_error(coords, y_true, y_pred, agg: Path, wf_type: str) -> None:
    """3-D UMAP with continuous gold→purple colouring by per-sample squared error."""
    import plotly.graph_objects as go
    errors = (np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2
    err_cap = max(float(np.percentile(errors, 95)), 1e-6)
    err_norm = np.clip(errors / err_cap, 0, 1)
    fig = go.Figure(data=[go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers",
        marker=dict(size=3, color=errors, colorscale=ERROR_COLORSCALE,
                    cmin=0, cmax=err_cap, colorbar=dict(title="MSE"), opacity=0.6),
        customdata=np.stack([y_true, y_pred, errors], axis=1),
        hovertemplate="true=%{customdata[0]:.3f}<br>pred=%{customdata[1]:.3f}"
                      "<br>MSE=%{customdata[2]:.4f}<extra></extra>",
    )])
    fig.update_layout(title="UMAP: Prediction Error (gold=accurate, purple=off)",
                      scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2",
                                 zaxis_title="UMAP-3", aspectmode="data"),
                      margin=dict(l=0, r=0, b=0, t=40))
    _save_interactive(fig, agg / f"umap3d_error_{wf_type}.html")
    fig_s = plt.figure(figsize=(8, 7))
    ax = fig_s.add_subplot(111, projection="3d")
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                    c=err_norm, cmap=ERROR_CMAP, s=4, alpha=0.5, edgecolors="none")
    fig_s.colorbar(sc, ax=ax, label="normalised error", shrink=0.6)
    ax.set_title("Prediction Error")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.set_zlabel("UMAP-3")
    fig_s.tight_layout()
    fig_s.savefig(agg / f"umap3d_error_{wf_type}.png", dpi=130)
    plt.close(fig_s)


def reg_umap_domains(coords, y_true, y_pred, domains, agg: Path, wf_type: str) -> None:
    """Per-domain 3-D UMAP: combined PNG subplot grid + one interactive HTML per domain."""
    import plotly.graph_objects as go
    unique_domains = sorted(set(domains))
    if len(unique_domains) < 2:
        return

    errors = (np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2
    err_cap = max(float(np.percentile(errors, 95)), 1e-6)
    err_norm = np.clip(errors / err_cap, 0, 1)
    impact = np.nan_to_num(np.asarray(y_true, dtype=float), nan=0.0).clip(0, 1)
    sizes = 2.0 + 5.0 * impact

    n = len(unique_domains)
    ncols = min(4, n)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig = plt.figure(figsize=(5 * ncols, 4.5 * nrows))
    for idx_d, dom in enumerate(unique_domains):
        ax = fig.add_subplot(nrows, ncols, idx_d + 1, projection="3d")
        mask_d = np.array([d == dom for d in domains])
        if not mask_d.any():
            ax.set_title(dom, fontsize=9)
            continue
        ax.scatter(coords[mask_d, 0], coords[mask_d, 1], coords[mask_d, 2],
                   c=err_norm[mask_d], cmap=ERROR_CMAP, s=np.square(sizes[mask_d]) * 0.7,
                   alpha=0.5, edgecolors="none", vmin=0, vmax=1)
        ax.set_title(f"{dom} (MSE={float(errors[mask_d].mean()):.4f})", fontsize=9)
        ax.tick_params(labelsize=6)
    fig.suptitle("Per-Domain Error (gold=accurate, purple=off; size = impact)", fontsize=11)
    fig.tight_layout()
    fig.savefig(agg / f"umap3d_domain_error_{wf_type}.png", dpi=120)
    plt.close(fig)

    for dom in unique_domains:
        mask_d = np.array([d == dom for d in domains])
        if not mask_d.any():
            continue
        mean_err = float(errors[mask_d].mean())
        fig_h = go.Figure(data=[go.Scatter3d(
            x=coords[mask_d, 0], y=coords[mask_d, 1], z=coords[mask_d, 2],
            mode="markers",
            marker=dict(size=sizes[mask_d], color=errors[mask_d],
                        colorscale=ERROR_COLORSCALE, cmin=0, cmax=err_cap,
                        colorbar=dict(title="MSE"), opacity=0.6),
            customdata=np.stack([y_true[mask_d], y_pred[mask_d], errors[mask_d]], axis=1),
            hovertemplate="true=%{customdata[0]:.3f}<br>pred=%{customdata[1]:.3f}"
                          "<br>MSE=%{customdata[2]:.4f}<extra></extra>",
        )])
        fig_h.update_layout(
            title=f"{dom}: mean MSE={mean_err:.4f}",
            scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3",
                       aspectmode="data"),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        out_path = agg / f"umap3d_domain_{_safe_domain_name(dom)}_error_{wf_type}.html"
        fig_h.write_html(str(out_path))
        print(f"[plots] {wf_type}: domain HTML -> {out_path.name}")


# ===================================================================== NN weight diagram

def extract_linear_weights(net) -> list[np.ndarray]:
    """Weight matrices of all Linear layers in the module (DFS order)."""
    import torch.nn as nn
    return [m.weight.detach().cpu().numpy() for m in net.modules()
            if isinstance(m, nn.Linear) and m.weight.numel() > 0]


def _build_nn_html(weights: list[np.ndarray], layer_sizes: list[int], n_z: int) -> str:
    """Interactive HTML SVG showing neural-network connection strengths."""
    LAYER_SPACING = 320
    MAX_NEURONS_DISPLAY = 400
    NEURON_SPACING = max(4, int(3000 / max(layer_sizes)))
    RADIUS = max(2, NEURON_SPACING // 3)

    n_layers = len(layer_sizes)
    display_n = [min(n, MAX_NEURONS_DISPLAY) for n in layer_sizes]
    total_h = max(display_n) * NEURON_SPACING + 120
    total_w = (n_layers - 1) * LAYER_SPACING + 160

    xs = [80 + i * LAYER_SPACING for i in range(n_layers)]

    def neuron_y(li: int, ni: int) -> float:
        n = display_n[li]
        top = (total_h - 80 - n * NEURON_SPACING) / 2 + 40
        return top + ni * NEURON_SPACING

    lines_svg: list[str] = []
    for li, w in enumerate(weights):
        out_n, in_n = w.shape
        max_w = np.abs(w).max()
        if max_w < 1e-9:
            continue
        threshold = NN_WEIGHT_DISPLAY_THRESHOLD * max_w
        x1, x2 = xs[li], xs[li + 1]
        dn_in, dn_out = display_n[li], display_n[li + 1]
        step_in = max(1, in_n // dn_in)
        step_out = max(1, out_n // dn_out)
        pairs = []
        for ni_raw, ni_disp in enumerate(range(0, min(in_n, dn_in * step_in), step_in)):
            for nj_raw, nj_disp in enumerate(range(0, min(out_n, dn_out * step_out), step_out)):
                abs_w = abs(w[nj_raw * step_out if step_out > 1 else nj_raw,
                              ni_raw * step_in if step_in > 1 else ni_raw])
                if abs_w > threshold:
                    pairs.append((abs_w / max_w, ni_disp, nj_disp))
        if len(pairs) > NN_MAX_HTML_LINES_PER_LAYER:
            pairs.sort(key=lambda t: t[0], reverse=True)
            pairs = pairs[:NN_MAX_HTML_LINES_PER_LAYER]
        pairs.sort(key=lambda t: t[0])  # weakest first so strong ones draw on top
        for alpha, ni_d, nj_d in pairs:
            y1 = neuron_y(li, ni_d)
            y2 = neuron_y(li + 1, nj_d)
            lines_svg.append(
                f'<line x1="{x1}" y1="{y1:.1f}" x2="{x2}" y2="{y2:.1f}" '
                f'stroke="{palette.INK}" stroke-opacity="{alpha:.3f}" stroke-width="0.6"/>'
            )

    circles_svg: list[str] = []
    for li, n in enumerate(display_n):
        for ni in range(n):
            y = neuron_y(li, ni)
            x = xs[li]
            if li == 0:
                n_meta_disp = display_n[0] - min(n_z, display_n[0])
                fill = palette.TERRACOTTA if ni < max(n_meta_disp, 0) else palette.DUSK_BLUE
            else:
                fill = palette.GRAY
            circles_svg.append(f'<circle cx="{x}" cy="{y:.1f}" r="{RADIUS}" fill="{fill}"/>')

    labels_svg: list[str] = []
    layer_names = ["Input"] + [f"Hidden {i + 1}" for i in range(n_layers - 2)] + ["Output"]
    for li, name in enumerate(layer_names):
        labels_svg.append(
            f'<text x="{xs[li]}" y="{total_h - 30}" text-anchor="middle" '
            f'font-size="13" fill="#222">{name} ({layer_sizes[li]})</text>'
        )
    if 0 < n_z < layer_sizes[0]:
        n_meta_disp = display_n[0] - min(n_z, display_n[0])
        if n_meta_disp > 0:
            y_mid_meta = (neuron_y(0, 0) + neuron_y(0, n_meta_disp - 1)) / 2
            labels_svg.append(
                f'<text x="{xs[0] - RADIUS - 6}" y="{y_mid_meta + 4}" '
                f'text-anchor="end" font-size="10" fill="{palette.TERRACOTTA}">metadata</text>'
            )
        n_z_disp = min(n_z, display_n[0])
        y_mid_z = (neuron_y(0, n_meta_disp) + neuron_y(0, n_meta_disp + n_z_disp - 1)) / 2
        labels_svg.append(
            f'<text x="{xs[0] - RADIUS - 6}" y="{y_mid_z + 4}" '
            f'text-anchor="end" font-size="10" fill="{palette.DUSK_BLUE}">z-emb.</text>'
        )

    svg_body = "\n".join(lines_svg + circles_svg + labels_svg)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Neural Network Weights</title>
<style>
body{{background:white;margin:0;padding:8px;font-family:sans-serif}}
#container{{overflow:auto;width:100%;height:calc(100vh - 50px)}}
#controls{{padding:6px;background:rgba(255,255,255,.95);border:1px solid #ccc;
           border-radius:4px;display:inline-flex;gap:6px;align-items:center;margin-bottom:6px}}
button{{padding:4px 10px;cursor:pointer}}
</style>
</head>
<body>
<div id="controls">
  <b>Connection Strengths</b>
  <button onclick="zoom(1.5)">+</button>
  <button onclick="zoom(0.67)">−</button>
  <button onclick="resetView()">Reset</button>
  <small>scroll=zoom &nbsp; drag=pan &nbsp;
         <span style="color:{palette.TERRACOTTA}">&#9632;</span>metadata &nbsp;
         <span style="color:{palette.DUSK_BLUE}">&#9632;</span>z-embedding &nbsp;
         <span style="color:{palette.INK}">&#8212;</span>connection strength (opacity)</small>
</div>
<div id="container">
<svg id="nn" viewBox="0 0 {total_w} {total_h}" width="{total_w}" height="{total_h}"
     xmlns="http://www.w3.org/2000/svg" style="cursor:grab">
{svg_body}
</svg>
</div>
<script>
var scale=1,tx=0,ty=0,drag=false,lx,ly;
var nn=document.getElementById('nn');
nn.addEventListener('wheel',function(e){{
  e.preventDefault();
  scale*=e.deltaY<0?1.12:0.89;
  applyT();
}},{{passive:false}});
nn.addEventListener('mousedown',function(e){{drag=true;lx=e.clientX;ly=e.clientY;nn.style.cursor='grabbing';}});
document.addEventListener('mouseup',function(){{drag=false;nn.style.cursor='grab';}});
document.addEventListener('mousemove',function(e){{
  if(!drag)return;
  tx+=e.clientX-lx;ty+=e.clientY-ly;lx=e.clientX;ly=e.clientY;applyT();
}});
function applyT(){{nn.style.transform='translate('+tx+'px,'+ty+'px) scale('+scale+')';nn.style.transformOrigin='0 0';}}
function zoom(f){{scale*=f;applyT();}}
function resetView(){{scale=1;tx=0;ty=0;applyT();}}
</script>
</body>
</html>"""


def _save_nn_png(weights: list[np.ndarray], layer_sizes: list[int],
                 n_z: int, wf_type: str, path: Path) -> None:
    """High-resolution PNG of the NN connection-strength diagram."""
    from matplotlib.collections import LineCollection

    LAYER_SP = 2.5
    N_SP = 0.12
    MAX_N = 300
    MAX_LINES = 8000

    n_layers = len(layer_sizes)
    display_n = [min(n, MAX_N) for n in layer_sizes]
    max_d = max(display_n)
    height = max_d * N_SP
    width = (n_layers - 1) * LAYER_SP

    fig, ax = plt.subplots(figsize=(max(8, n_layers * 3.5), max(6, max_d * 0.04 + 1)))
    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_xlim(-0.3, width + 0.3)
    ax.set_ylim(-N_SP, height + N_SP)

    def pos(li, ni):
        n = display_n[li]
        x = li * LAYER_SP
        top = (height - n * N_SP) / 2
        return x, top + ni * N_SP

    for li, w in enumerate(weights):
        out_n, in_n = w.shape
        dn_in, dn_out = display_n[li], display_n[li + 1]
        step_in = max(1, in_n // dn_in)
        step_out = max(1, out_n // dn_out)
        max_w = np.abs(w).max()
        if max_w < 1e-9:
            continue
        thr = NN_WEIGHT_DISPLAY_THRESHOLD * max_w
        pairs = []
        for ni_d in range(dn_in):
            ni_r = ni_d * step_in
            for nj_d in range(dn_out):
                abs_w = abs(w[nj_d * step_out, ni_r])
                if abs_w > thr:
                    pairs.append((abs_w, ni_d, nj_d))
        if len(pairs) > MAX_LINES:
            pairs.sort(reverse=True)
            pairs = pairs[:MAX_LINES]
        pairs.sort()  # weakest first
        if not pairs:
            continue
        lines, colors = [], []
        for abs_w, ni_d, nj_d in pairs:
            x1, y1 = pos(li, ni_d)
            x2, y2 = pos(li + 1, nj_d)
            lines.append([(x1, y1), (x2, y2)])
            colors.append((0.0, 0.0, 0.545, abs_w / max_w))
        ax.add_collection(LineCollection(lines, linewidths=0.4, colors=colors, zorder=1))

    for li, n in enumerate(display_n):
        for ni in range(n):
            x, y = pos(li, ni)
            if li == 0:
                n_meta_d = display_n[0] - min(n_z, display_n[0])
                c = palette.TERRACOTTA if ni < max(n_meta_d, 0) else palette.DUSK_BLUE
            else:
                c = palette.GRAY
            ax.plot(x, y, "o", ms=max(1.5, N_SP * 5), color=c, zorder=3)

    layer_names = ["Input"] + [f"H{i + 1}" for i in range(n_layers - 2)] + ["Output"]
    for li, name in enumerate(layer_names):
        ax.text(li * LAYER_SP, -N_SP * 0.8, f"{name}\n({layer_sizes[li]})",
                ha="center", va="top", fontsize=8)
    if 0 < n_z < layer_sizes[0]:
        n_meta_d = display_n[0] - min(n_z, display_n[0])
        if n_meta_d > 0:
            _, ym = pos(0, n_meta_d // 2)
            ax.text(-0.15, ym, "meta", ha="right", va="center", fontsize=7, color=palette.TERRACOTTA)
        _, yz = pos(0, n_meta_d + min(n_z, display_n[0]) // 2)
        ax.text(-0.15, yz, "z-emb", ha="right", va="center", fontsize=7, color=palette.DUSK_BLUE)

    ax.set_title("Neural Network Connection Strengths\n"
                 f"(dark blue=strong, transparent=weak  |  "
                 f"{'red=metadata, blue=z-embedding at input' if n_z > 0 else 'input neurons'})",
                 fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plots] {wf_type}: NN weights PNG -> {path.name}")


def plot_nn_weights(wf_type: str, agg: Path, seed_data: dict, df: pd.DataFrame) -> None:
    """NN connection-strength diagram (interactive HTML + PNG) for fully-connected models.

    Only weights above ``NN_WEIGHT_DISPLAY_THRESHOLD`` of the layer max are drawn. Input
    neurons are ordered metadata (red, top) then z-embedding (blue, bottom). Skipped for
    transformer architectures and workflows without a saved joblib model.
    """
    model_path = Path(seed_data.get("_seed_dir", "")) / "model" / "model.joblib"
    if not model_path.is_file():
        print(f"[plots] {wf_type}: no model.joblib found, skipping NN weight viz.")
        return

    try:
        import joblib
        model = joblib.load(model_path)
        net = getattr(model, "net_", None)
        if net is None:
            return
    except Exception as e:
        print(f"[plots] {wf_type}: could not load model for NN viz ({e}).")
        return

    core = getattr(net, "core", net)
    arch_name = type(core).__name__
    if "Transformer" in arch_name or "FT" in arch_name:
        print(f"[plots] {wf_type}: NN weight viz not supported for {arch_name}.")
        return

    weights = [w for w in extract_linear_weights(net) if w.shape[0] > 0 and w.shape[1] > 0]
    if not weights:
        return

    n_z = 0
    if "z" in df.columns:
        try:
            first_z = df["z"].iloc[0]
            n_z = len(first_z) if hasattr(first_z, "__len__") else 0
        except Exception:
            n_z = 0

    # Input ordering: [metadata+mask, z_dims] so metadata appears at the top
    in_features = weights[0].shape[1]
    if 0 < n_z < in_features:
        n_rest = in_features - n_z
        perm = list(range(n_rest, in_features)) + list(range(n_rest))
        weights[0] = weights[0][:, perm]

    layer_sizes = [weights[0].shape[1]] + [w.shape[0] for w in weights]

    try:
        html_path = agg / f"nn_weights_{wf_type}.html"
        html_path.write_text(_build_nn_html(weights, layer_sizes, n_z), encoding="utf-8")
        print(f"[plots] {wf_type}: NN weights HTML -> {html_path.name}")
    except Exception as e:
        print(f"[plots] {wf_type}: NN weights HTML failed ({e}).")

    try:
        _save_nn_png(weights, layer_sizes, n_z, wf_type, agg / f"nn_weights_{wf_type}.png")
    except Exception as e:
        print(f"[plots] {wf_type}: NN weights PNG failed ({e}).")


# ===================================================================== main entry

def _umap_suite(wf_type, task, agg, seed_data, df, class_names, seed) -> None:
    """The 3-D UMAP family (correctness/classes/error + per-domain), clf and reg."""
    umap_data = _get_umap_data(seed_data, df, seed)
    if umap_data is None or umap_data.get("coords") is None:
        print(f"[plots] {wf_type}: no z embedding data, skipping UMAP plots.")
        return
    coords = umap_data["coords"]
    idx = umap_data["idx"]
    aligned = idx is not None and len(umap_data["y_true"]) == umap_data["df_len"]
    if aligned:
        y_true = umap_data["y_true"][idx]
        y_pred = umap_data["y_pred"][idx]
    else:
        y_true = umap_data["y_true"][:len(coords)]
        y_pred = umap_data["y_pred"][:len(coords)]

    target_vals = pd.to_numeric(df["target"], errors="coerce").to_numpy(dtype=float)
    target = target_vals[idx] if aligned else np.zeros(len(coords))
    domains = _get_domain(df).to_numpy()[idx] if aligned else None

    if task == "clf":
        correct_mask = np.asarray(y_true) == np.asarray(y_pred)
        try:
            clf_umap_correctness(coords, correct_mask, target, agg, wf_type)
        except Exception as e:
            print(f"[plots] {wf_type} umap_correctness skipped: {e}")
        if class_names and "class_label" in df.columns and aligned:
            class_labels = df["class_label"].to_numpy()[idx]
            try:
                clf_umap_classes(coords, class_labels, target, class_names, agg, wf_type)
            except Exception as e:
                print(f"[plots] {wf_type} umap_classes skipped: {e}")
            if domains is not None and len(set(domains)) >= 2:
                try:
                    clf_umap_domains(coords, correct_mask, class_labels, class_names,
                                     domains, agg, wf_type)
                except Exception as e:
                    print(f"[plots] {wf_type} umap_domains skipped: {e}")
    else:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        try:
            reg_umap_error(coords, y_true, y_pred, agg, wf_type)
        except Exception as e:
            print(f"[plots] {wf_type} umap_error skipped: {e}")
        if domains is not None and len(set(domains)) >= 2:
            try:
                reg_umap_domains(coords, y_true, y_pred, domains, agg, wf_type)
            except Exception as e:
                print(f"[plots] {wf_type} umap_domains skipped: {e}")


def generate_workflow_plots(wf_type: str, task: str, wf_dir: Path, settings) -> None:
    """Generate the full evaluation plot suite for one workflow run. Never raises."""
    agg = wf_dir / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)

    seed_data = _load_best_seed_data(wf_dir, task)
    if seed_data is None:
        print(f"[plots] {wf_type}: no seed data found, skipping plots.")
        return

    from workflow.src.iwf import paths
    from workflow.src.iwf.data.dataset import load_parquet

    class_names = list(getattr(settings, "class_names", []) or [])
    class_thresholds = list(getattr(settings, "class_thresholds", []) or [])
    class_scheme = getattr(settings, "class_scheme", "percentile")
    split_seed = int(getattr(settings, "split_seed", 42))

    ds_id = getattr(settings, "dataset_id", "wf01")
    parquet_path = paths.reduced_parquet(ds_id)
    if not parquet_path.is_file():
        parquet_path = paths.prepared_parquet(ds_id)
    if not parquet_path.is_file():
        print(f"[plots] {wf_type}: no parquet found, skipping plots.")
        return
    df = load_parquet(parquet_path)

    # Regression correctness bars use the same class boundaries as the classifier; for
    # the css scheme the boundaries are data-derived.
    thresholds = list(class_thresholds)
    if task == "reg" and (class_scheme or "percentile").lower() == "css" \
            and "target" in df.columns:
        try:
            from workflow.src.iwf.targets import css_boundaries
            tv = pd.to_numeric(df["target"], errors="coerce").dropna().to_numpy()
            if len(tv) > 0:
                thresholds = css_boundaries(tv, len(class_names))
        except Exception:
            pass

    split_preds = _extract_split_preds(seed_data)
    if split_preds:
        try:
            correctness_bars(split_preds, agg, wf_type, task, thresholds)
        except Exception as e:
            print(f"[plots] {wf_type} correctness_bars skipped: {e}")
        if task == "reg":
            try:
                mse_bars(split_preds, agg, wf_type)
            except Exception as e:
                print(f"[plots] {wf_type} mse_bars skipped: {e}")
    else:
        print(f"[plots] {wf_type}: no split predictions, skipping split bars.")

    try:
        learning_curve(seed_data.get("history", {}), agg, wf_type, task)
    except Exception as e:
        print(f"[plots] {wf_type} learning_curve skipped: {e}")

    try:
        _umap_suite(wf_type, task, agg, seed_data, df, class_names, split_seed)
    except Exception as e:
        print(f"[plots] {wf_type} UMAP suite skipped: {e}")

    try:
        plot_nn_weights(wf_type, agg, seed_data, df)
    except Exception as e:
        print(f"[plots] {wf_type} nn_weights skipped: {e}")


# ===================================================================== standalone figures

def plot_dim_curve(curve: list[dict], baseline: float, k_star: int, path: Path) -> None:
    """Latent-dimension selection curve (main_encoder)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ks = [c["k"] for c in curve]
    ms = [c["metric"] for c in curve]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, ms, "o-", label="probe metric")
    if baseline is not None:
        ax.axhline(baseline, color="gray", ls="--", label="full-embedding baseline")
    ax.axvline(k_star, color="red", ls=":", label=f"k* = {k_star}")
    ax.set_xlabel("latent dim k")
    ax.set_ylabel("val metric")
    ax.set_title("Latent-dimension selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_generated_distribution(human_scores, generated: dict, path: Path) -> None:
    """Predicted-impact distribution: human test split vs LLM-generated sources."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)
    h = np.asarray(human_scores, dtype=float)
    ax.hist(h[~np.isnan(h)], bins=bins, alpha=0.5, density=True, label="human test")
    for name, arr in generated.items():
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size:
            ax.hist(a, bins=bins, alpha=0.5, density=True, label=name)
    ax.set_xlabel("Predicted normalized impact")
    ax.set_ylabel("Density")
    ax.set_title("Predicted impact: human test vs LLM-generated")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_tfm_training_curve(history: dict, path: Path,
                            title: str = "Text-transformer training curve") -> None:
    """Validation metric vs epoch with the best epoch marked (per-seed, text workflows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vm = list(history.get("val_metric", []))
    best_epoch = int(history.get("best_epoch", 0) or 0)
    fig, ax = plt.subplots(figsize=(7, 4))
    if vm:
        ax.plot(range(1, len(vm) + 1), vm, "o-", color=palette.OLIVE, label="val metric")
        if best_epoch:
            ax.axvline(best_epoch, color="gray", ls=":", label=f"best @ epoch {best_epoch}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val metric (Spearman / macro-F1)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _smooth(values, window: int = 25):
    a = np.asarray(values, dtype=float)
    if a.size < 2:
        return a
    window = max(1, min(window, a.size))
    kernel = np.ones(window) / window
    return np.convolve(a, kernel, mode="valid")


def plot_tfm_loss_lr(history: dict, path: Path,
                     title: str = "Train loss (smoothed) + LR schedule") -> None:
    """Per-step train loss (smoothed) with the LR schedule overlaid (text workflows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    loss = list(history.get("step_loss", []))
    lr = list(history.get("step_lr", []))
    fig, ax1 = plt.subplots(figsize=(7, 4))
    if loss:
        sm = _smooth(loss)
        ax1.plot(range(1, len(sm) + 1), sm, color=palette.DUSK_BLUE, label="train loss (smoothed)")
    ax1.set_xlabel("optimizer step")
    ax1.set_ylabel("train loss", color=palette.DUSK_BLUE)
    ax2 = ax1.twinx()
    if lr:
        ax2.plot(range(1, len(lr) + 1), lr, color=palette.GOLD, lw=1, label="learning rate")
    ax2.set_ylabel("learning rate", color=palette.GOLD)
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
