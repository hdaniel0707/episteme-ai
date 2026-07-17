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

# NN-weight diagram sparsity: a connection is drawn only when it is BOTH above this
# fraction of the layer's max |weight| AND among the strongest NN_TOP_K_PER_NEURON
# incoming connections of its target neuron. This keeps only the genuinely important
# connections visible instead of a uniform hairball.
NN_WEIGHT_DISPLAY_THRESHOLD = 0.35
NN_TOP_K_PER_NEURON = 3
NN_MAX_HTML_LINES_PER_LAYER = 1200


def _select_important_connections(w: np.ndarray, dn_in: int, dn_out: int
                                  ) -> list[tuple[float, float, int, int]]:
    """Pick the connections worth drawing for one weight matrix ``w`` [out, in].

    Returns (rel_strength 0..1, signed_weight, in_display_idx, out_display_idx),
    weakest first. Selection: per displayed OUTPUT neuron keep its top-K strongest
    incoming displayed connections, then drop any below the global threshold."""
    out_n, in_n = w.shape
    step_in = max(1, in_n // dn_in)
    step_out = max(1, out_n // dn_out)
    sub = w[::step_out, ::step_in][:dn_out, :dn_in]          # displayed submatrix
    max_w = float(np.abs(sub).max())
    if max_w < 1e-9:
        return []
    thr = NN_WEIGHT_DISPLAY_THRESHOLD * max_w
    pairs: list[tuple[float, float, int, int]] = []
    k = NN_TOP_K_PER_NEURON
    for nj in range(sub.shape[0]):
        row = np.abs(sub[nj])
        top = np.argsort(row)[-k:] if len(row) > k else np.arange(len(row))
        for ni in top:
            if row[ni] >= thr:
                pairs.append((row[ni] / max_w, float(sub[nj, ni]), int(ni), nj))
    pairs.sort(key=lambda t: t[0])   # weakest first so strong ones draw on top
    return pairs


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
        if not s or s.lower() == "nan":
            return "unknown"
        # domains from SQLite arrives as a JSON-array string (occasionally
        # double-encoded by a stray json.dumps on a copy path) — unwrap up to
        # two layers before falling back to the raw string.
        for _ in range(2):
            try:
                decoded = json.loads(s)
            except (json.JSONDecodeError, TypeError):
                break
            if isinstance(decoded, list):
                return str(decoded[0]) if decoded else "unknown"
            if isinstance(decoded, str):
                s = decoded
                continue
            break
        return s

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
    """Two stacked panels sharing the epoch axis — no dual-axis tricks:

    top:    training loss  (lower is better ↓), best epoch marked
    bottom: validation metric (higher is better ↑), best epoch dot + value label
    """
    metric_name = "macro-F1" if task == "clf" else "Spearman"
    tl = list(history.get("train_loss", []))
    vm = list(history.get("val_metric", []))
    # Text-transformer stores per-step loss; average per epoch for consistency.
    if not tl and "step_loss" in history:
        sl = history["step_loss"]
        epochs = len(vm) if vm else 1
        chunk = max(1, len(sl) // epochs)
        tl = [float(np.mean(sl[i * chunk:(i + 1) * chunk])) for i in range(epochs)]
    if not tl and not vm:
        return
    best = int(np.argmax(vm)) if vm else None

    palette.apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True,
                                   layout="constrained")
    if tl:
        ax1.plot(range(1, len(tl) + 1), tl, color=palette.DUSK_BLUE, lw=1.6)
        ax1.fill_between(range(1, len(tl) + 1), tl, min(tl),
                         color=palette.DUSK_BLUE, alpha=0.10)
    ax1.set_ylabel("Training loss\n(lower is better ↓)")
    ax1.grid(True, alpha=0.3)
    if vm:
        ax2.plot(range(1, len(vm) + 1), vm, color=palette.OLIVE, lw=1.8)
        ax2.fill_between(range(1, len(vm) + 1), vm, min(vm),
                         color=palette.OLIVE, alpha=0.12)
        ax2.plot(best + 1, vm[best], "o", ms=8, color=palette.GOLD,
                 mec=palette.INK, mew=1.2, zorder=5)
        ax2.annotate(f"best: {vm[best]:.3f}\n(epoch {best + 1})",
                     xy=(best + 1, vm[best]), xytext=(8, -4),
                     textcoords="offset points", fontsize=9, color=palette.INK)
        for ax in (ax1, ax2):
            ax.axvline(best + 1, color=palette.GRAY, ls=":", lw=1)
    ax2.set_ylabel(f"Validation {metric_name}\n(higher is better ↑)")
    ax2.set_xlabel("Epoch")
    ax2.grid(True, alpha=0.3)
    ax1.set_title("Training Progress — loss falls, validation quality rises",
                  fontsize=11)
    fig.align_ylabels((ax1, ax2))
    fig.savefig(agg / f"learning_curve_{wf_type}.png", dpi=130)
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
        x1, x2 = xs[li], xs[li + 1]
        pairs = _select_important_connections(w, display_n[li], display_n[li + 1])
        if len(pairs) > NN_MAX_HTML_LINES_PER_LAYER:
            pairs = pairs[-NN_MAX_HTML_LINES_PER_LAYER:]   # already sorted weakest first
        for rel, signed_w, ni_d, nj_d in pairs:
            y1 = neuron_y(li, ni_d)
            y2 = neuron_y(li + 1, nj_d)
            # quadratic emphasis: weak survivors stay faint, the strongest pop
            alpha = 0.08 + 0.82 * rel * rel
            width = 0.4 + 1.6 * rel * rel
            color = palette.INK if signed_w >= 0 else palette.TERRACOTTA
            lines_svg.append(
                f'<line x1="{x1}" y1="{y1:.1f}" x2="{x2}" y2="{y2:.1f}" '
                f'stroke="{color}" stroke-opacity="{alpha:.3f}" '
                f'stroke-width="{width:.2f}"/>'
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
         <span style="color:{palette.TERRACOTTA}">&#9632;</span>metadata input &nbsp;
         <span style="color:{palette.DUSK_BLUE}">&#9632;</span>z-embedding input &nbsp;
         <span style="color:{palette.INK}">&#8212;</span>positive weight &nbsp;
         <span style="color:{palette.TERRACOTTA}">&#8212;</span>negative weight &nbsp;
         (only each neuron's top-{NN_TOP_K_PER_NEURON} strongest connections shown)</small>
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

    from matplotlib.colors import to_rgb
    rgb_pos, rgb_neg = to_rgb(palette.INK), to_rgb(palette.TERRACOTTA)
    for li, w in enumerate(weights):
        pairs = _select_important_connections(w, display_n[li], display_n[li + 1])
        if len(pairs) > MAX_LINES:
            pairs = pairs[-MAX_LINES:]   # already sorted weakest first
        if not pairs:
            continue
        lines, colors, widths = [], [], []
        for rel, signed_w, ni_d, nj_d in pairs:
            x1, y1 = pos(li, ni_d)
            x2, y2 = pos(li + 1, nj_d)
            lines.append([(x1, y1), (x2, y2)])
            rgb = rgb_pos if signed_w >= 0 else rgb_neg
            colors.append((*rgb, 0.08 + 0.82 * rel * rel))
            widths.append(0.3 + 1.4 * rel * rel)
        ax.add_collection(LineCollection(lines, linewidths=widths, colors=colors, zorder=1))

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

    ax.set_title("Neural Network — Most Important Connections\n"
                 f"(top-{NN_TOP_K_PER_NEURON} strongest inputs per neuron; "
                 "dark olive=positive, terracotta=negative; thicker=stronger"
                 f"{' | red=metadata, blue=z-embedding inputs' if n_z > 0 else ''})",
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


# ===================================================================== dataset analysis plots

def plot_counts_year(counts: pd.DataFrame, path: Path) -> None:
    """Bar chart: papers per year."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette.apply_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(counts["year"], counts["n"], color=palette.OLIVE, alpha=0.85, edgecolor=palette.INK, linewidth=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Papers")
    ax.set_title("Papers per Year")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_counts_domain(counts: pd.DataFrame, path: Path) -> None:
    """Horizontal bar chart: papers per domain (top N, sorted descending)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette.apply_style()
    fig, ax = plt.subplots(figsize=(8, max(5, len(counts) * 0.2)))
    # Sort by count descending (already done in data_analysis.py)
    y_pos = np.arange(len(counts))
    ax.barh(y_pos, counts["n"], color=palette.SAGE, alpha=0.85, edgecolor=palette.INK, linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(counts["domain"], fontsize=9)
    ax.set_xlabel("Number of Papers")
    ax.set_title("Papers per Domain (Top Domains)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_year_domain_heatmap(pivot: pd.DataFrame, path: Path) -> None:
    """Heatmap: years × domains."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette.apply_style()
    # Limit size for readability
    max_size = 25
    if pivot.shape[0] > max_size or pivot.shape[1] > max_size:
        pivot = pivot.iloc[:max_size, :max_size]
    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.5), max(6, pivot.shape[0] * 0.4)))
    im = ax.imshow(pivot.values, cmap=palette.register_olive_seq(), aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Domain")
    ax.set_ylabel("Year")
    ax.set_title("Papers: Year × Domain Heatmap")
    # Annotate if small enough
    if pivot.shape[0] <= 25 and pivot.shape[1] <= 25:
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if val > 0:
                    ax.text(j, i, str(int(val)), ha="center", va="center", fontsize=7, color="white")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_completeness(comp: pd.DataFrame, path: Path) -> None:
    """Horizontal bars: completeness % per field."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette.apply_style()
    fig, ax = plt.subplots(figsize=(10, len(comp) * 0.3))
    y_pos = np.arange(len(comp))
    colors = palette.ordinal_colors(len(comp))
    ax.barh(y_pos, comp["pct_nonnull"], color=colors, alpha=0.85, edgecolor=palette.INK, linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comp["field"], fontsize=9)
    ax.set_xlabel("Completeness (%)")
    ax.set_title("Field Completeness")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_completeness_by_year(pivot_pct: pd.DataFrame, path: Path) -> None:
    """Heatmap: field × year completeness %."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette.apply_style()
    fig, ax = plt.subplots(figsize=(max(10, pivot_pct.shape[1] * 0.4), max(8, pivot_pct.shape[0] * 0.3)))
    im = ax.imshow(pivot_pct.values, cmap=palette.register_olive_seq(), aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(np.arange(pivot_pct.shape[1]))
    ax.set_yticks(np.arange(pivot_pct.shape[0]))
    ax.set_xticklabels(pivot_pct.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(pivot_pct.index, fontsize=8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Field")
    ax.set_title("Field Completeness (%) by Year")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Completeness (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_abstract_length_hist(lengths: pd.Series, path: Path) -> None:
    """Histogram: abstract character lengths with median vline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette.apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=50, color=palette.SAND, alpha=0.75, edgecolor=palette.INK, linewidth=0.5)
    median_val = lengths.median()
    ax.axvline(median_val, color=palette.OLIVE, linestyle="--", linewidth=2, label=f"Median: {median_val:.0f} chars")
    ax.set_xlabel("Abstract Length (characters)")
    ax.set_ylabel("Frequency")
    ax.set_title("Abstract Length Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_source_productivity(sources: pd.DataFrame, path: Path) -> None:
    """Horizontal grouped bars per data source: records contributed (sand) vs records
    usable for training (olive), with the usable % annotated — 'which source was most
    productive'. Expects columns source, n_total, n_usable, pct_usable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette.apply_style()
    d = sources.sort_values("n_total")           # largest on top for horizontal bars
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.55 * len(d) + 1.5)))
    ax.barh(y + 0.2, d["n_total"], height=0.38, color=palette.SAND,
            edgecolor=palette.INK, linewidth=0.5, label="all records")
    ax.barh(y - 0.2, d["n_usable"], height=0.38, color=palette.OLIVE,
            label="usable for training")
    for yi, (tot, use, pct) in enumerate(zip(d["n_total"], d["n_usable"], d["pct_usable"])):
        ax.text(tot, yi + 0.2, f" {tot:,}", va="center", fontsize=8, color=palette.INK)
        ax.text(use, yi - 0.2, f" {use:,} ({pct:.0f}%)", va="center", fontsize=8,
                color=palette.INK)
    ax.set_yticks(y)
    ax.set_yticklabels(d["source"])
    ax.set_xlabel("papers")
    ax.set_title("Data-Source Productivity — contributed vs training-usable records")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame, path: Path) -> None:
    """Spearman correlation heatmap of the numeric metadata columns (diverging BrBG)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette.apply_style()
    cols = list(corr.columns)
    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 2.5, 1.0 * len(cols) + 2))
    im = ax.imshow(corr.to_numpy(dtype=float), cmap=palette.DIVERGING_CMAP,
                   vmin=-1, vmax=1)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=9)
    ax.set_title("Metadata Correlations (Spearman)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_score_distributions(scores: pd.DataFrame, path: Path) -> None:
    """Score distributions of the usable corpus: the [0,1] training target (should be
    ~uniform by construction), log-scaled citation counts, and FWCI when present."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette.apply_style()
    have_fwci = "fwci" in scores.columns and scores["fwci"].notna().sum() >= 10
    n_panels = 3 if have_fwci else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.2))

    t = scores["target"].dropna()
    axes[0].hist(t, bins=40, color=palette.OLIVE, alpha=0.85)
    axes[0].set_xlabel("impact target (field-year percentile, 0-1)")
    axes[0].set_ylabel("papers")
    axes[0].set_title("Training-target distribution")

    c = scores["citation_count"].dropna()
    c = c[c >= 0]
    axes[1].hist(np.log10(c + 1), bins=40, color=palette.DUSK_BLUE, alpha=0.85)
    axes[1].set_xlabel("log10(citations + 1)")
    axes[1].set_title(f"Citation distribution (median {c.median():.0f})")

    if have_fwci:
        f = scores["fwci"].dropna().clip(upper=scores["fwci"].quantile(0.99))
        axes[2].hist(f, bins=40, color=palette.GOLD, alpha=0.85)
        axes[2].set_xlabel("FWCI (clipped at p99)")
        axes[2].set_title("FWCI distribution")

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Score Distributions (training-usable corpus)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ===================================================================== benchmark plots

def plot_benchmark_run(metrics: pd.DataFrame, bench_name: str, run_name: str,
                       path: Path) -> None:
    """Grouped bars: one group per metric, bar per scorer (ours first, then alpha)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette.apply_style()

    all_metrics = metrics[metrics["domain"] == "ALL"]
    if all_metrics.empty:
        return

    metric_names = sorted(all_metrics["metric"].unique())
    scorers = sorted(all_metrics["scorer"].unique())
    # Sort so "ours" comes first
    if "ours" in scorers:
        scorers.remove("ours")
        scorers = ["ours"] + sorted(scorers)

    n_metrics = len(metric_names)
    n_scorers = len(scorers)
    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 1.5), 5))

    x = np.arange(n_metrics)
    bar_width = 0.8 / max(1, n_scorers)

    for i, scorer in enumerate(scorers):
        values = []
        for metric_name in metric_names:
            row = all_metrics[(all_metrics["scorer"] == scorer) & (all_metrics["metric"] == metric_name)]
            if not row.empty:
                values.append(row.iloc[0]["value"])
            else:
                values.append(np.nan)

        color = palette.color_for(scorer if scorer != "ours" else run_name)
        bars = ax.bar(x + (i - (n_scorers - 1) / 2) * bar_width, values, bar_width,
                      label=scorer, color=color, alpha=0.85, edgecolor=palette.INK, linewidth=0.5)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width() / 2, height,
                        f"{height:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Benchmark: {bench_name} ({run_name})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_benchmark_overview(all_metrics: pd.DataFrame, path: Path) -> None:
    """Lollipop chart: all runs sorted by spearman (domain=ALL)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette.apply_style()

    all_domain = all_metrics[all_metrics["domain"] == "ALL"]
    if all_domain.empty:
        return

    # Filter to one metric (primary)
    metric_df = all_domain[all_domain["metric"] == "spearman"]
    if metric_df.empty:
        metric_df = all_domain  # fallback to any metric

    # Sort by value
    metric_df = metric_df.sort_values("value")

    fig, ax = plt.subplots(figsize=(10, max(6, len(metric_df) * 0.3)))

    y_pos = np.arange(len(metric_df))
    x_vals = metric_df["value"].values
    labels = [f"{row['run']}:{row['scorer']}" for _, row in metric_df.iterrows()]
    colors = [palette.color_for(row["scorer"]) for _, row in metric_df.iterrows()]

    # Lollipop chart
    ax.hlines(y_pos, 0, x_vals, colors=palette.GRAY, alpha=0.3, linewidth=1)
    ax.scatter(x_vals, y_pos, c=colors, s=60, alpha=0.85, edgecolors=palette.INK, linewidth=0.5, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Spearman ρ")
    ax.set_title("Benchmark Overview - All Runs")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_benchmark_heatmap(all_metrics: pd.DataFrame, path: Path) -> None:
    """Heatmap: rows=models, cols=(benchmark, metric), values=score."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette.apply_style()

    all_domain = all_metrics[all_metrics["domain"] == "ALL"]
    if all_domain.empty:
        return

    # Create pivot: rows=run:scorer, cols=benchmark:metric
    all_domain["model"] = all_domain["run"] + ":" + all_domain["scorer"]
    all_domain["bench_metric"] = all_domain["benchmark"] + "\n" + all_domain["metric"]

    pivot = all_domain.pivot_table(index="model", columns="bench_metric", values="value")

    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 0.6), max(8, pivot.shape[0] * 0.3)))

    im = ax.imshow(pivot.values, cmap=palette.register_olive_seq(), aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(pivot.index, fontsize=8)

    # Annotate with values
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white")

    ax.set_title("Benchmark Heatmap: Model x Benchmark-Metric")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Score")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_per_domain_bars(metrics_df: pd.DataFrame, run_name: str, path: Path) -> None:
    """Per-domain metrics bar chart: bars = domains (desc by value), vline = ALL-domain value."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette.apply_style()

    if metrics_df.empty:
        return

    all_row = metrics_df[metrics_df["domain"] == "ALL"]
    bars_df = metrics_df[metrics_df["domain"] != "ALL"]
    if bars_df.empty:
        return

    # Sorted so the strongest domain renders at the top of a horizontal bar chart
    bars_df = bars_df.sort_values("value", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(bars_df) * 0.35)))

    y_pos = np.arange(len(bars_df))
    color = palette.color_for(run_name)

    ax.barh(y_pos, bars_df["value"], color=color, alpha=0.85, edgecolor=palette.INK, linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{d} (n={n})" for d, n in zip(bars_df["domain"], bars_df["n"])], fontsize=8)
    ax.set_xlabel(bars_df.iloc[0]["metric"])
    ax.set_title(f"Per-Domain Metrics - {run_name}")

    lo = min(0.0, float(bars_df["value"].min()))
    hi = max(1.0, float(bars_df["value"].max()))
    ax.set_xlim([lo, hi])

    if not all_row.empty:
        ax.axvline(float(all_row.iloc[0]["value"]), color=palette.INK, linestyle="--",
                   linewidth=1, alpha=0.7, label=f"ALL (n={int(all_row.iloc[0]['n'])})")
        ax.legend(loc="lower right", fontsize=8)

    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cross_domain_matrix(matrix: pd.DataFrame, path: Path) -> None:
    """LODO matrix: runs vs domains heatmap."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette.apply_style()

    if matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(max(10, matrix.shape[1] * 0.5), max(8, matrix.shape[0] * 0.3)))

    im = ax.imshow(matrix.values, cmap=palette.register_olive_seq(), aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)

    # Annotate with values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white")

    ax.set_title("Cross-Domain Matrix: Run x Domain")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Metric Value")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
