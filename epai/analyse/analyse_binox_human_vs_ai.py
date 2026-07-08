"""Compare human vs AI Binoculars scores *within* parquets — one violin per file.

Every ``id`` in a ``*_binox*`` parquet appears once per ``generated_by`` group
(``human``, ``gpt``, ``claude``, ...). For each parquet this script pairs, per
``id``, the ``human`` row against a chosen generator's row (default ``gpt``) and
plots the per-id score differences. Multiple parquets are shown side by side —
one violin each — and every parquet can be given its own target generator.

Pairing & distance
-------------------
For a target generator ``g`` the ``human`` and ``g`` subsets are merged on ``id``
(unique within each generator). For each matched id the score vector across the
``binoculars_*`` columns is taken for human and for ``g``; the pairwise distance
is the Euclidean distance ``|human - g|`` (abs diff for the single-column case),
and the signed difference ``human - g`` keeps direction. MSE = mean over all valid
element pairs of ``(human - g)**2``. Ids present for only one side are reported.

Invalid pairs
-------------
If *either* element of a pair is null or not a real number, the pair is skipped,
recorded, and reported in the closing summary statistics.

Targets
-------
Each parquet is given as ``PATH[:TARGET]``; ``TARGET`` defaults to ``gpt`` when
omitted, so every file can be compared against a different generator.

Figures + tables saved to results/binox_human_vs_ai/<parquet_stems>/:
  - human_vs_ai_distance_violin.png    — |human − g| per parquet, MSE annotated
  - human_vs_ai_signed_diff_violin.png — signed human − g (which side scores higher)
  - human_vs_ai_score_distributions.png — per parquet: human vs AI score histograms
  - human_vs_ai_roc.png                — ROC per parquet overlaid, AUC in legend
  - human_vs_ai_pr.png                 — precision–recall per parquet overlaid, AP in legend
  - human_vs_ai_confusion.png          — per-parquet 2×2 confusion matrix at best threshold
  - human_vs_ai_summary.csv            — per-parquet dist stats + AUC/AP/acc/F1/confusion
  - human_vs_ai_summary.log            — full text summary incl. skipped-pair details

Classification framing
----------------------
Two classes: positive = AI (target), negative = human. Binoculars scores are
lower for AI, so the rule is "predict AI when score <= threshold" (equivalently
decision score = -score). The operating threshold is Youden's J optimum on the
ROC curve. Classes are balanced (paired), so the PR chance baseline is 0.5.

Usage:
    uv run python epai/analyse/analyse_binox_human_vs_ai.py data/parquet/wp_binox0.parquet
    uv run python epai/analyse/analyse_binox_human_vs_ai.py \
        data/parquet/wp_binox1.parquet:gpt \
        data/parquet/essay_binox1.parquet:gpt \
        data/parquet/reuter_binox1.parquet:gpt \
        data/parquet/wp_binox0.parquet:gpt \
        data/parquet/essay_binox0.parquet:gpt \
        data/parquet/reuter_binox0.parquet:gpt

    
    uv run python epai/analyse/analyse_binox_human_vs_ai.py \
        data/parquet/wp_binox0.parquet:gpt \
        data/parquet/essay_binox0.parquet:gpt \
        data/parquet/reuter_binox0.parquet:gpt \
        data/parquet/wp2_binox0.parquet:gpt_plain \
        data/parquet/essay2_binox0.parquet:gpt_plain \
        data/parquet/reuter2_binox0.parquet:gpt_plain
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
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "binox_human_vs_ai"

ID_COL = "id"
GROUP_COL = "generated_by"
HUMAN_GROUP = "human"
DEFAULT_TARGET = "gpt"


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
# Core computation
# ---------------------------------------------------------------------------


def _binox_columns(df: pd.DataFrame) -> list[str]:
    """Binoculars score columns present in the frame, sorted."""
    return sorted(c for c in df.columns if c.startswith("binoculars_"))


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerce the given columns to floats; non-numeric / null become NaN."""
    return df[cols].apply(pd.to_numeric, errors="coerce")


class PairResult:
    """Human-vs-<target> comparison for one parquet file."""

    def __init__(self, parquet: str, target: str, cols: list[str]) -> None:
        self.parquet = parquet  # file stem, used for labels
        self.target = target
        self.cols = cols
        self.n_pairs = 0  # ids present for both human and target
        self.n_only_human = 0  # ids with a human row but no target row
        self.n_only_target = 0  # ids with a target row but no human row
        self.distances = np.empty(0)  # Euclidean / |human-g| — magnitude only
        self.signed_diffs = np.empty(0)  # human - g (mean over cols) — keeps sign
        self.human_scores = np.empty(0)  # human score per valid pair (mean over cols)
        self.target_scores = np.empty(0)  # target score per valid pair (mean over cols)
        self.mse = float("nan")
        self.skipped: list[dict] = []  # id + raw values of skipped pairs

        # --- classification (human vs AI, positive = AI, AI ⇔ score < threshold) ---
        self.has_clf = False
        self.roc_fpr = np.empty(0)
        self.roc_tpr = np.empty(0)
        self.roc_auc = float("nan")
        self.pr_precision = np.empty(0)
        self.pr_recall = np.empty(0)
        self.avg_precision = float("nan")
        self.clf_threshold = float("nan")  # Binoculars score cut; score < cut ⇒ AI
        self.accuracy = float("nan")
        self.f1 = float("nan")
        self.cm = np.zeros((2, 2), int)  # [[TN, FP], [FN, TP]] over [human, AI]

    @property
    def label(self) -> str:
        return f"{self.parquet}\nvs {self.target}"

    @property
    def n_valid(self) -> int:
        return int(self.distances.size)

    @property
    def n_skipped(self) -> int:
        return len(self.skipped)


def compute_for_spec(path: Path, target: str, out: Tee) -> PairResult:
    """Load one parquet and compute human-vs-``target`` per-id differences."""
    if not path.exists():
        raise SystemExit(f"Missing parquet: {path}")

    df = pd.read_parquet(path)
    for col in (ID_COL, GROUP_COL):
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' missing from {path.name}.")
    cols = _binox_columns(df)
    if not cols:
        raise SystemExit(f"No 'binoculars_*' columns found in {path.name}.")

    human = df[df[GROUP_COL] == HUMAN_GROUP]
    tgt = df[df[GROUP_COL] == target]
    if human.empty:
        raise SystemExit(f"No '{HUMAN_GROUP}' rows found in {path.name}.")
    if tgt.empty:
        raise SystemExit(
            f"generator '{target}' not found in {path.name} "
            f"(available: {sorted(df[GROUP_COL].dropna().unique())})"
        )

    # id must be unique within each generator subset to pair unambiguously.
    for grp_label, sub in ((HUMAN_GROUP, human), (target, tgt)):
        dup = sub.duplicated(subset=[ID_COL]).sum()
        if dup:
            raise SystemExit(
                f"[{path.name}] id is not unique within '{grp_label}' rows "
                f"({dup} duplicates) — cannot pair unambiguously."
            )

    left = pd.concat(
        [human[[ID_COL]], _to_numeric(human, cols).add_suffix("_h")], axis=1
    )
    right = pd.concat(
        [tgt[[ID_COL]], _to_numeric(tgt, cols).add_suffix("_t")], axis=1
    )
    merged = left.merge(right, on=ID_COL, how="outer", indicator=True)

    res = PairResult(path.stem, target, cols)
    res.n_only_human = int((merged["_merge"] == "left_only").sum())
    res.n_only_target = int((merged["_merge"] == "right_only").sum())
    paired = merged[merged["_merge"] == "both"].reset_index(drop=True)
    res.n_pairs = len(paired)

    a = paired[[f"{c}_h" for c in cols]].to_numpy(float)  # human
    b = paired[[f"{c}_t" for c in cols]].to_numpy(float)  # target

    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    for i in np.nonzero(~valid)[0]:
        res.skipped.append(
            {
                "id": paired.loc[i, ID_COL],
                "human": a[i].tolist(),
                target: b[i].tolist(),
            }
        )

    av, bv = a[valid], b[valid]
    diff = av - bv  # (n_valid, k)
    res.distances = np.sqrt((diff**2).sum(axis=1))  # Euclidean; |human-g| for k=1
    res.signed_diffs = diff.mean(axis=1)  # signed; == human - g for k=1
    res.human_scores = av.mean(axis=1)  # == human score for k=1
    res.target_scores = bv.mean(axis=1)  # == target score for k=1
    res.mse = float((diff**2).mean()) if diff.size else float("nan")

    compute_classification(res)
    return res


def compute_classification(res: PairResult) -> None:
    """Score human-vs-AI classification from the paired Binoculars scores.

    Positive class = AI (the target generator). A *low* Binoculars score means
    AI, so the model is "predict AI when score < threshold"; equivalently the
    decision score fed to sklearn is ``-score`` (higher ⇒ more AI-like). The
    operating threshold is Youden's J optimum on the ROC curve.
    """
    if res.human_scores.size == 0 or res.target_scores.size == 0:
        return

    scores = np.concatenate([res.human_scores, res.target_scores])
    y = np.concatenate(
        [np.zeros(res.human_scores.size, int), np.ones(res.target_scores.size, int)]
    )
    decision = -scores  # higher ⇒ more AI-like

    res.roc_fpr, res.roc_tpr, thr = roc_curve(y, decision)
    res.roc_auc = float(roc_auc_score(y, decision))
    res.pr_precision, res.pr_recall, _ = precision_recall_curve(y, decision)
    res.avg_precision = float(average_precision_score(y, decision))

    # Youden's J optimal operating point → threshold on the raw Binoculars score.
    j = res.roc_tpr - res.roc_fpr
    best_decision_thr = thr[int(np.argmax(j))]
    res.clf_threshold = float(-best_decision_thr)  # predict AI when score <= this
    y_pred = (scores <= res.clf_threshold).astype(int)
    res.accuracy = float((y_pred == y).mean())
    res.f1 = float(f1_score(y, y_pred, zero_division=0))
    res.cm = confusion_matrix(y, y_pred, labels=[0, 1])
    res.has_clf = True


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _violin_figure(
    usable: list[PairResult],
    data: list[np.ndarray],
    *,
    ylabel: str,
    title: str,
    annotate,
    zero_line: bool,
    fname: str,
    fig_dir: Path,
) -> None:
    """Draw one violin per parquet and save it.

    ``annotate(r)`` returns the short text placed above each violin.
    """
    positions = list(range(1, len(usable) + 1))
    fig, ax = plt.subplots(figsize=(max(6, 2.4 * len(usable) + 2), 6))
    parts = ax.violinplot(
        data, positions=positions, showmeans=True, showextrema=True, widths=0.8
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_edgecolor("black")
        pc.set_alpha(0.6)
    if "cmeans" in parts:
        parts["cmeans"].set_color("#d1495b")

    if zero_line:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    labels = []
    for x, d, r in zip(positions, data, usable):
        top = float(d.max())
        pad = 0.01 * (abs(top) or 1) + 0.005
        ax.text(
            x, top + pad, annotate(r),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
        labels.append(
            f"{r.label}\nn={r.n_valid}"
            + (f"  (skipped {r.n_skipped})" if r.n_skipped else "")
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.margins(y=0.12)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  [OK] {fname}")


def plot_violins(results: list[PairResult], fig_dir: Path) -> None:
    usable = [r for r in results if r.n_valid > 0]
    if not usable:
        print("  [skip] no valid pairs for any parquet — nothing to plot")
        return

    # Figure 1 — magnitude of human/AI score gap, annotated with MSE.
    _violin_figure(
        usable,
        [r.distances for r in usable],
        ylabel="Pairwise distance  |human − AI|",
        title="Human vs AI Binoculars distance (magnitude) per parquet",
        annotate=lambda r: f"MSE={r.mse:.4g}",
        zero_line=False,
        fname="human_vs_ai_distance_violin.png",
        fig_dir=fig_dir,
    )

    # Figure 2 — signed difference, to see which side scores higher.
    _violin_figure(
        usable,
        [r.signed_diffs for r in usable],
        ylabel="Signed difference  (human − AI)",
        title="Human vs AI signed difference per parquet  (>0 ⇒ human larger)",
        annotate=lambda r: f"mean={r.signed_diffs.mean():+.4g}",
        zero_line=True,
        fname="human_vs_ai_signed_diff_violin.png",
        fig_dir=fig_dir,
    )


def plot_distributions(results: list[PairResult], fig_dir: Path) -> None:
    """Per parquet, overlay the human and AI Binoculars score distributions."""
    usable = [r for r in results if r.n_valid > 0]
    if not usable:
        print("  [skip] no valid pairs for any parquet — no distributions to plot")
        return

    ncols = min(3, len(usable))
    nrows = int(np.ceil(len(usable) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False
    )
    flat = axes.flatten()

    for ax, r in zip(flat, usable):
        both = np.concatenate([r.human_scores, r.target_scores])
        lo, hi = np.percentile(both, [1, 99])
        if lo == hi:  # degenerate range guard
            lo, hi = both.min(), both.max() or 1.0
        bins = np.linspace(lo, hi, 40)
        ax.hist(
            r.human_scores, bins=bins, color="#4c72b0", alpha=0.6, density=True,
            label=f"human (n={r.human_scores.size})",
        )
        ax.hist(
            r.target_scores, bins=bins, color="#d1495b", alpha=0.6, density=True,
            label=f"{r.target} (n={r.target_scores.size})",
        )
        ax.axvline(r.human_scores.mean(), color="#4c72b0", linestyle="--", linewidth=1.2)
        ax.axvline(r.target_scores.mean(), color="#d1495b", linestyle="--", linewidth=1.2)
        ax.set_title(f"{r.parquet}\nhuman vs {r.target}", fontsize=10)
        ax.set_xlabel("Binoculars score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    for ax in flat[len(usable):]:  # hide unused axes
        ax.axis("off")

    fig.suptitle("Human vs AI Binoculars score distributions", fontsize=13)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fname = "human_vs_ai_score_distributions.png"
    fig.savefig(fig_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  [OK] {fname}")


def plot_roc(results: list[PairResult], fig_dir: Path) -> None:
    """Overlay the human-vs-AI ROC curve of every parquet, AUC in the legend."""
    usable = [r for r in results if r.has_clf]
    if not usable:
        print("  [skip] no classifiable parquet — no ROC to plot")
        return
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, r in enumerate(usable):
        ax.plot(
            r.roc_fpr, r.roc_tpr, color=colors[i % len(colors)], linewidth=2,
            label=f"{r.parquet} vs {r.target}  (AUC={r.roc_auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="chance (AUC=0.5)")
    ax.set_xlabel("False positive rate  (human classified as AI)")
    ax.set_ylabel("True positive rate  (AI detected)")
    ax.set_title("Human vs AI detection ROC (Binoculars score)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "human_vs_ai_roc.png", dpi=150)
    plt.close(fig)
    print("  [OK] human_vs_ai_roc.png")


def plot_pr(results: list[PairResult], fig_dir: Path) -> None:
    """Overlay the human-vs-AI precision–recall curve of every parquet, AP in legend."""
    usable = [r for r in results if r.has_clf]
    if not usable:
        print("  [skip] no classifiable parquet — no PR curve to plot")
        return
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, r in enumerate(usable):
        ax.plot(
            r.pr_recall, r.pr_precision, color=colors[i % len(colors)], linewidth=2,
            label=f"{r.parquet} vs {r.target}  (AP={r.avg_precision:.3f})",
        )
    # Classes are balanced (paired), so chance precision = 0.5.
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1, alpha=0.6, label="chance (AP=0.5)")
    ax.set_xlabel("Recall  (AI detected)")
    ax.set_ylabel("Precision")
    ax.set_title("Human vs AI detection precision–recall (Binoculars score)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "human_vs_ai_pr.png", dpi=150)
    plt.close(fig)
    print("  [OK] human_vs_ai_pr.png")


def plot_confusion(results: list[PairResult], fig_dir: Path) -> None:
    """Per parquet, the 2×2 confusion matrix at its Youden-optimal threshold."""
    usable = [r for r in results if r.has_clf]
    if not usable:
        print("  [skip] no classifiable parquet — no confusion matrices to plot")
        return

    ncols = min(3, len(usable))
    nrows = int(np.ceil(len(usable) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.4 * ncols, 4.0 * nrows), squeeze=False
    )
    flat = axes.flatten()
    ticks = ["human", "AI"]

    for ax, r in zip(flat, usable):
        cm = r.cm
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        thresh = cm.max() / 2 if cm.max() else 0
        for row in range(2):
            for col in range(2):
                ax.text(
                    col, row, f"{cm[row, col]}", ha="center", va="center",
                    fontsize=12,
                    color="white" if cm[row, col] > thresh else "black",
                )
        ax.set_xticks([0, 1], labels=ticks)
        ax.set_yticks([0, 1], labels=ticks)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(
            f"{r.parquet} vs {r.target}\n"
            f"score<{r.clf_threshold:.3f}⇒AI  acc={r.accuracy:.3f}  F1={r.f1:.3f}",
            fontsize=9,
        )

    for ax in flat[len(usable):]:  # hide unused axes
        ax.axis("off")

    fig.suptitle("Human vs AI confusion matrices (Youden-optimal threshold)", fontsize=13)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "human_vs_ai_confusion.png", dpi=150)
    plt.close(fig)
    print("  [OK] human_vs_ai_confusion.png")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _dist_stats(d: np.ndarray) -> dict:
    if d.size == 0:
        return {k: float("nan") for k in ("mean", "std", "min", "median", "max")}
    return {
        "mean": float(d.mean()),
        "std": float(d.std()),
        "min": float(d.min()),
        "median": float(np.median(d)),
        "max": float(d.max()),
    }


def report(results: list[PairResult], fig_dir: Path, out: Tee) -> None:
    out.print("=" * 70)
    out.print("Human vs AI Binoculars comparison (per parquet)")
    out.print(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    out.print("=" * 70)

    rows = []
    for r in results:
        s = _dist_stats(r.distances)
        out.print(f"\n[{r.parquet}: human vs {r.target}]  columns: {r.cols}")
        out.print(
            f"  matched pairs={r.n_pairs}  valid={r.n_valid}  skipped={r.n_skipped}"
            f"  (unmatched: only human={r.n_only_human}, only {r.target}={r.n_only_target})"
        )
        out.print(
            f"  distance  mean={s['mean']:.4g}  std={s['std']:.4g}  "
            f"min={s['min']:.4g}  median={s['median']:.4g}  max={s['max']:.4g}"
        )
        mean_signed = float(r.signed_diffs.mean()) if r.n_valid else float("nan")
        larger = (
            "human larger" if mean_signed > 0
            else f"{r.target} larger" if mean_signed < 0
            else "equal"
        )
        out.print(f"  mean signed diff (human - {r.target})={mean_signed:+.4g}  → {larger}")
        out.print(f"  MSE={r.mse:.6g}")
        tn, fp, fn, tp = r.cm.ravel()
        out.print(
            f"  classification (AI positive): ROC-AUC={r.roc_auc:.3f}  "
            f"AP={r.avg_precision:.3f}  acc={r.accuracy:.3f}  F1={r.f1:.3f}"
        )
        out.print(
            f"    threshold: score<={r.clf_threshold:.4g} ⇒ AI   "
            f"confusion  TP={tp} FP={fp} FN={fn} TN={tn}"
        )
        rows.append(
            {
                "parquet": r.parquet,
                "target": r.target,
                "columns": ",".join(r.cols),
                "n_pairs": r.n_pairs,
                "n_valid": r.n_valid,
                "n_skipped": r.n_skipped,
                "n_only_human": r.n_only_human,
                "n_only_target": r.n_only_target,
                "dist_mean": s["mean"],
                "dist_std": s["std"],
                "dist_min": s["min"],
                "dist_median": s["median"],
                "dist_max": s["max"],
                "mean_signed_diff": mean_signed,
                "mse": r.mse,
                "roc_auc": r.roc_auc,
                "avg_precision": r.avg_precision,
                "clf_threshold": r.clf_threshold,
                "accuracy": r.accuracy,
                "f1": r.f1,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }
        )

    # --- skipped-pair detail ---
    out.print("\n" + "-" * 70)
    out.print("Skipped pairs (element null / non-numeric in human or target):")
    total_skipped = sum(r.n_skipped for r in results)
    if total_skipped == 0:
        out.print("  (none — every pair had numeric scores on both sides)")
    else:
        for r in results:
            if not r.skipped:
                continue
            out.print(f"\n  [{r.parquet}: human vs {r.target}] {r.n_skipped} skipped:")
            for sk in r.skipped:
                out.print(
                    f"    id={sk['id']}  human={sk['human']}  "
                    f"{r.target}={sk[r.target]}"
                )
        out.print(f"\n  TOTAL skipped across parquets: {total_skipped}")

    out.print("\n" + "=" * 70)

    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_path = fig_dir / "human_vs_ai_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  [OK] {csv_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_spec(spec: str) -> tuple[Path, str]:
    """Split a ``PATH[:TARGET]`` spec; TARGET defaults to ``gpt``.

    The target is taken from the substring after the last ':' only when that
    substring is non-empty and contains no path separator (so Windows drive
    letters / ordinary paths without a target are left intact).
    """
    if ":" in spec:
        head, sep, tail = spec.rpartition(":")
        if head and tail and "/" not in tail and "\\" not in tail:
            return Path(head), tail
    return Path(spec), DEFAULT_TARGET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "specs",
        nargs="+",
        help="One or more parquet specs 'PATH[:TARGET]'. TARGET defaults to "
        f"'{DEFAULT_TARGET}'. Each parquet becomes one violin.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = [_parse_spec(s) for s in args.specs]

    out = Tee()
    results: list[PairResult] = []
    for path, target in specs:
        print(f"Comparing human vs '{target}' in {path.name} ...")
        results.append(compute_for_spec(path, target, out))

    # Directory named after the parquet stems involved (deduped, order kept).
    stems = list(dict.fromkeys(r.parquet for r in results))
    fig_dir = RESULTS_DIR / "__".join(stems)
    print()
    plot_violins(results, fig_dir)
    plot_distributions(results, fig_dir)
    plot_roc(results, fig_dir)
    plot_pr(results, fig_dir)
    plot_confusion(results, fig_dir)
    report(results, fig_dir, out)

    out.save(fig_dir / "human_vs_ai_summary.log")
    print(f"\nAll outputs saved to {fig_dir}")


if __name__ == "__main__":
    main()
