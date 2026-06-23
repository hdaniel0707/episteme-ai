"""
Analyse how well Binoculars detector scores (from ``epai.add_binoculars_scores``)
separate human-written from AI-generated abstracts.

Detection framing
-----------------
Positive class = AI-generated (``generated_by != "human"``); negative = human.
A *low* Binoculars score (PPL / cross-PPL) indicates AI text, so the decision
score used for ROC/PR is ``-binoculars_score`` (higher → more AI-like). Each
``binoculars_*`` column is one candidate detector and is evaluated independently.

Figures + tables saved to results/binoculars/<parquet_stem>/:
  - roc_curves.png             — ROC of every predictor overlaid, AUC in legend
  - pr_curves.png              — precision–recall of every predictor, AP in legend
  - auc_by_predictor.png       — bar chart ranking predictors by ROC-AUC
  - auc_per_generator.png      — heatmap: predictor × AI-generator detection AUC
  - score_dist_<predictor>.png — human vs AI score histograms + optimal threshold
  - detection_metrics.csv      — AUC / AP / best-threshold acc & F1 / TPR@FPR
  - per_generator_auc.csv       — predictor × generator AUC (human vs each generator)

Usage:
    python -m epai.analyse_binoculars
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = Path(__file__).parent.parent

### INPUTS ############################################################
PARQUET_PATH = BASE_DIR / "data" / "parquet" / "0515_A_500_AI_260_naip_binox.parquet"
FIG_DIR = BASE_DIR / "results" / "binoculars"

GROUP_COL = "generated_by"
HUMAN_GROUP = "human"  # the negative (non-AI) class
TARGET_FPRS = [0.01, 0.05, 0.10]  # report TPR at these false-positive rates
#######################################################################


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def predictor_columns(df: pd.DataFrame) -> list[str]:
    """All binoculars score columns present in the dataframe, sorted."""
    return sorted(c for c in df.columns if c.startswith("binoculars_"))


def _pretty(col: str) -> str:
    """'binoculars_falcon_rw_1b' -> 'falcon_rw_1b'."""
    return col.removeprefix("binoculars_")


def _detection_arrays(df: pd.DataFrame, score_col: str):
    """Return (y_ai, decision_score) over rows with a valid score & group.

    y_ai = 1 for AI-generated, 0 for human.
    decision_score = -binoculars_score so that higher ⇒ more AI-like.
    """
    mask = df[score_col].notna() & df[GROUP_COL].notna()
    sub = df.loc[mask]
    y_ai = (sub[GROUP_COL] != HUMAN_GROUP).to_numpy(int)
    score = sub[score_col].to_numpy(float)
    return y_ai, -score, sub[GROUP_COL].to_numpy(str)


def _ai_generators(df: pd.DataFrame) -> list[str]:
    return sorted(g for g in df[GROUP_COL].dropna().unique() if g != HUMAN_GROUP)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def detection_metrics(df: pd.DataFrame, score_col: str) -> dict:
    """ROC-AUC, average precision, best-threshold accuracy/F1 and TPR@FPR."""
    y, s, _ = _detection_arrays(df, score_col)
    if y.sum() == 0 or y.sum() == len(y):  # need both classes
        return {}

    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)

    fpr, tpr, thr = roc_curve(y, s)
    # Youden's J optimal operating point.
    j = tpr - fpr
    best = int(np.argmax(j))
    best_decision_thr = thr[best]
    y_pred = (s >= best_decision_thr).astype(int)
    acc = float((y_pred == y).mean())
    f1 = float(f1_score(y, y_pred))
    # Map the decision threshold back to the original binoculars score.
    best_score_thr = -best_decision_thr

    out = {
        "predictor": _pretty(score_col),
        "n": int(len(y)),
        "n_ai": int(y.sum()),
        "n_human": int(len(y) - y.sum()),
        "roc_auc": float(auc),
        "avg_precision": float(ap),
        "best_score_threshold": float(best_score_thr),
        "best_acc": acc,
        "best_f1": f1,
    }
    for t in TARGET_FPRS:
        out[f"tpr@fpr={t:g}"] = float(np.interp(t, fpr, tpr))
    return out


def per_generator_auc(df: pd.DataFrame) -> pd.DataFrame:
    """Detection AUC of each predictor for human-vs-<each AI generator>.

    Isolates which predictor catches which generator: for generator g the
    positives are only g's rows, negatives are human's rows.
    """
    generators = _ai_generators(df)
    rows = []
    for col in predictor_columns(df):
        row = {"predictor": _pretty(col)}
        for g in generators:
            mask = df[col].notna() & df[GROUP_COL].isin([HUMAN_GROUP, g])
            sub = df.loc[mask]
            y = (sub[GROUP_COL] == g).to_numpy(int)
            if y.sum() == 0 or y.sum() == len(y):
                row[g] = np.nan
                continue
            row[g] = float(roc_auc_score(y, -sub[col].to_numpy(float)))
        rows.append(row)
    return pd.DataFrame(rows).set_index("predictor")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_roc_curves(df: pd.DataFrame, fig_dir: Path) -> None:
    cols = predictor_columns(df)
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, col in enumerate(cols):
        y, s, _ = _detection_arrays(df, col)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        ax.plot(
            fpr,
            tpr,
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"{_pretty(col)}  (AUC={auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="chance (AUC=0.5)")
    ax.set_xlabel("False positive rate  (human flagged as AI)")
    ax.set_ylabel("True positive rate  (AI detected)")
    ax.set_title("AI-text detection ROC — Binoculars predictors")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_curves.png", dpi=150)
    plt.close(fig)
    print("  [OK] roc_curves.png")


def plot_pr_curves(df: pd.DataFrame, fig_dir: Path) -> None:
    cols = predictor_columns(df)
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, ax = plt.subplots(figsize=(7, 7))
    baseline = None
    for i, col in enumerate(cols):
        y, s, _ = _detection_arrays(df, col)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        prec, rec, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        baseline = y.mean()  # prevalence of AI
        ax.plot(
            rec,
            prec,
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"{_pretty(col)}  (AP={ap:.3f})",
        )
    if baseline is not None:
        ax.axhline(
            baseline,
            color="k",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            label=f"chance (AP={baseline:.2f})",
        )
    ax.set_xlabel("Recall  (AI detected)")
    ax.set_ylabel("Precision")
    ax.set_title("AI-text detection precision–recall — Binoculars predictors")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "pr_curves.png", dpi=150)
    plt.close(fig)
    print("  [OK] pr_curves.png")


def plot_auc_by_predictor(metrics: pd.DataFrame, fig_dir: Path) -> None:
    m = metrics.sort_values("roc_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(m) + 1)))
    bars = ax.barh(m["predictor"], m["roc_auc"], color="steelblue", alpha=0.8)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="chance")
    for bar, v in zip(bars, m["roc_auc"]):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.4, 1.02)
    ax.set_xlabel("ROC-AUC  (human vs AI)")
    ax.set_title("Detection power per Binoculars predictor")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "auc_by_predictor.png", dpi=150)
    plt.close(fig)
    print("  [OK] auc_by_predictor.png")


def plot_auc_per_generator(auc_df: pd.DataFrame, fig_dir: Path) -> None:
    if auc_df.empty:
        return
    data = auc_df.to_numpy(float)
    fig, ax = plt.subplots(
        figsize=(max(7, 1.2 * auc_df.shape[1] + 3), max(3, 0.7 * auc_df.shape[0] + 2))
    )
    im = ax.imshow(data, vmin=0.5, vmax=1.0, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ROC-AUC")
    ax.set_xticks(range(auc_df.shape[1]))
    ax.set_xticklabels(auc_df.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(auc_df.shape[0]))
    ax.set_yticklabels(auc_df.index, fontsize=9)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            if np.isnan(v):
                continue
            ax.text(
                c, r, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if v < 0.78 else "black",
            )
    ax.set_title("Detection AUC by predictor × AI generator  (vs human)")
    fig.tight_layout()
    fig.savefig(fig_dir / "auc_per_generator.png", dpi=150)
    plt.close(fig)
    print("  [OK] auc_per_generator.png")


def plot_score_distribution(df: pd.DataFrame, score_col: str, fig_dir: Path) -> None:
    """Overlaid human vs AI score histograms with the optimal threshold."""
    y, s, _ = _detection_arrays(df, score_col)
    if y.sum() == 0 or y.sum() == len(y):
        return
    score = -s  # back to original binoculars score
    human_scores = score[y == 0]
    ai_scores = score[y == 1]

    lo, hi = np.percentile(score, [1, 99])
    bins = np.linspace(lo, hi, 50)

    met = detection_metrics(df, score_col)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(human_scores, bins=bins, color="#888888", alpha=0.6, label=f"human (n={len(human_scores)})", density=True)
    ax.hist(ai_scores, bins=bins, color="#d1495b", alpha=0.6, label=f"AI (n={len(ai_scores)})", density=True)
    ax.axvline(
        met["best_score_threshold"],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"threshold={met['best_score_threshold']:.3f}",
    )
    ax.set_xlabel("Binoculars score  (PPL / cross-PPL) — lower ⇒ AI")
    ax.set_ylabel("Density")
    ax.set_title(
        f"{_pretty(score_col)} — human vs AI separation\n"
        f"AUC={met['roc_auc']:.3f}  AP={met['avg_precision']:.3f}  "
        f"best acc={met['best_acc']:.3f}  F1={met['best_f1']:.3f}"
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fname = f"score_dist_{_pretty(score_col)}.png"
    fig.savefig(fig_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  [OK] {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    df = pd.read_parquet(PARQUET_PATH)
    cols = predictor_columns(df)
    if not cols:
        raise SystemExit(
            f"No 'binoculars_*' columns found in {PARQUET_PATH.name} — "
            "did you run epai.add_binoculars_scores first?"
        )

    print(f"Loaded {len(df)} rows from {PARQUET_PATH.name}")
    n_human = (df[GROUP_COL] == HUMAN_GROUP).sum()
    print(f"  human: {n_human}   AI: {len(df) - n_human}   ({len(_ai_generators(df))} generators)")
    print(f"  predictors: {[_pretty(c) for c in cols]}")

    fig_dir = FIG_DIR / PARQUET_PATH.stem
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- metrics tables ---
    metrics = pd.DataFrame(
        [m for m in (detection_metrics(df, c) for c in cols) if m]
    )
    metrics.to_csv(fig_dir / "detection_metrics.csv", index=False)
    auc_df = per_generator_auc(df)
    auc_df.to_csv(fig_dir / "per_generator_auc.csv")
    print("  [OK] detection_metrics.csv / per_generator_auc.csv")

    print("\nDetection performance (human vs AI):")
    for _, r in metrics.sort_values("roc_auc", ascending=False).iterrows():
        tpr_str = "  ".join(f"TPR@{t:g}={r[f'tpr@fpr={t:g}']:.2f}" for t in TARGET_FPRS)
        print(
            f"  {r['predictor']:<40} AUC={r['roc_auc']:.3f}  AP={r['avg_precision']:.3f}  "
            f"acc={r['best_acc']:.3f}  F1={r['best_f1']:.3f}  {tpr_str}"
        )

    # --- figures ---
    print()
    plot_roc_curves(df, fig_dir)
    plot_pr_curves(df, fig_dir)
    plot_auc_by_predictor(metrics, fig_dir)
    plot_auc_per_generator(auc_df, fig_dir)
    for col in cols:
        plot_score_distribution(df, col, fig_dir)

    print(f"\nAll figures + tables saved to {fig_dir}")


if __name__ == "__main__":
    main()
