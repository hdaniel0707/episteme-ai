"""
Analyse NAIP scores stored in data/parquet/0515_A_500_AI_260_naip.parquet.

Produces figures saved to results/naip/<parquet_stem>/:
  - overall_distribution.png              — histograms of raw and final score across all rows
  - grouped_raw.png                       — raw score distributions per generated_by group
  - grouped_final.png                     — final score distributions per generated_by group
  - grouped_stats_{raw,final}.png         — mean ± std bar chart per group
  - corr_raw_citations.png                — scatter: naip_raw_score vs log(1+citation_count)
  - corr_final_citations.png              — scatter: naip_final_score vs log(1+citation_count)
  - corr_raw_yearly_citations.png         — scatter: naip_raw_score vs log(1+yearly_citations)
  - corr_final_yearly_citations.png       — scatter: naip_final_score vs log(1+yearly_citations)

Usage:
    python -m aip.analyse_naip
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Colors per source / LLM family live in cfg/colors.yaml (see color_utils).
from epai.utils.color_utils import group_color as _group_color

BASE_DIR = Path(__file__).parent.parent
PARQUET_PATH = BASE_DIR / "data" / "parquet" / "0523_A_10_AI_naip.parquet"
FIG_DIR = BASE_DIR / "results" / "naip"

SCORE_COLS = ["naip_raw_score", "naip_final_score"]
SCORE_LABELS = {
    "naip_raw_score": "Raw score",
    "naip_final_score": "Final score (TNCSI_sp)",
}
N_BINS = 40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_overall_distribution(df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col in zip(axes, SCORE_COLS):
        vals = df[col].dropna()
        ax.hist(
            vals,
            bins=N_BINS,
            color="steelblue",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.85,
        )
        ax.axvline(
            vals.mean(),
            color="firebrick",
            linewidth=1.5,
            linestyle="--",
            label=f"mean={vals.mean():.3f}",
        )
        ax.axvline(
            vals.median(),
            color="darkorange",
            linewidth=1.5,
            linestyle=":",
            label=f"median={vals.median():.3f}",
        )
        ax.set_xlabel(SCORE_LABELS[col])
        ax.set_ylabel("Count")
        ax.set_title(f"Overall distribution — {SCORE_LABELS[col]}")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"NAIP score distributions  (n={len(df)})", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "overall_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] overall_distribution.png")


def plot_grouped_distributions(df: pd.DataFrame, fig_dir: Path) -> None:
    groups = sorted(df["generated_by"].dropna().unique())
    colors = [_group_color(g) for g in groups]

    for col, fname in [
        (SCORE_COLS[0], "grouped_raw.png"),
        (SCORE_COLS[1], "grouped_final.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for group, color in zip(groups, colors):
            vals = df.loc[df["generated_by"] == group, col].dropna()
            ax.hist(
                vals,
                bins=N_BINS,
                alpha=0.55,
                color=color,
                edgecolor="white",
                linewidth=0.3,
                label=f"{group} (n={len(vals)})",
            )
        ax.set_xlabel(SCORE_LABELS[col])
        ax.set_ylabel("Count")
        ax.set_title(f"{SCORE_LABELS[col]} — grouped by generated_by")
        ax.legend(fontsize=8, title="generated_by")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  [OK] {fname}")


def plot_grouped_stats(df: pd.DataFrame, fig_dir: Path) -> None:
    groups = sorted(df["generated_by"].dropna().unique())
    x = np.arange(len(groups))
    colors = [_group_color(g) for g in groups]

    for col, fname in [
        (SCORE_COLS[0], "grouped_stats_raw.png"),
        (SCORE_COLS[1], "grouped_stats_final.png"),
    ]:
        means, stds = [], []
        for group in groups:
            vals = df.loc[df["generated_by"] == group, col].dropna()
            means.append(vals.mean())
            stds.append(vals.std())

        fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.4), 6))
        ax.bar(
            x,
            means,
            0.5,
            yerr=stds,
            color=colors,
            alpha=0.8,
            error_kw={"elinewidth": 1.5, "capsize": 4},
        )
        for xi, (m, s) in enumerate(zip(means, stds)):
            ax.text(xi, m + s + 0.005, f"{m:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Score (mean ± std)")
        ax.set_title(f"{SCORE_LABELS[col]} — mean ± std per generated_by group")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  [OK] {fname}")


def plot_correlation_with_citations(
    df: pd.DataFrame,
    fig_dir: Path,
    cit_col: str = "citation_count",
    cit_label: str = "citation_count",
    stem_suffix: str = "citations",
) -> None:
    import plotly.graph_objects as go
    from scipy.stats import pearsonr

    groups = sorted(df["generated_by"].dropna().unique())

    for col, stem in [
        ("naip_raw_score", f"corr_raw_{stem_suffix}"),
        ("naip_final_score", f"corr_final_{stem_suffix}"),
    ]:
        mask = df[col].notna() & df[cit_col].notna()
        sub = df[mask].copy()
        sub["log_citations"] = np.log1p(sub[cit_col])

        x_all = sub[col].to_numpy(dtype=float)
        y_all = sub["log_citations"].to_numpy(dtype=float)
        r, _ = pearsonr(x_all, y_all)
        ols_slope, ols_intercept = np.polyfit(x_all, y_all, 1)
        x_ols = np.linspace(x_all.min(), x_all.max(), 200)
        y_ols = ols_slope * x_ols + ols_intercept

        cit_type = "Yearly" if "yearly" in stem_suffix else "Total"

        fig = go.Figure()

        for group in groups:
            gdf = sub[sub["generated_by"] == group]
            if gdf.empty:
                continue
            color = _group_color(group)
            fig.add_trace(
                go.Scatter(
                    x=gdf[col].tolist(),
                    y=gdf["log_citations"].tolist(),
                    mode="markers",
                    name=f"{group} (n={len(gdf)})",
                    marker=dict(size=6, opacity=0.5, color=color),
                    hovertemplate=f"{group}<br>{SCORE_LABELS[col]}: %{{x:.3f}}<br>log(1+{cit_label}): %{{y:.2f}}<extra></extra>",
                )
            )

        # binned mean + smoothed trend (same approach as visualization_utils)
        n_bins = 20
        bins = np.linspace(x_all.min(), x_all.max(), n_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_means = np.array(
            [
                (
                    y_all[(x_all >= lo) & (x_all < hi)].mean()
                    if ((x_all >= lo) & (x_all < hi)).any()
                    else np.nan
                )
                for lo, hi in zip(bins[:-1], bins[1:])
            ]
        )
        valid = ~np.isnan(bin_means)
        bx, by = bin_centers[valid], bin_means[valid]

        fig.add_trace(
            go.Scatter(
                x=bx.tolist(),
                y=by.tolist(),
                mode="lines+markers",
                name="Mean per bin",
                line=dict(color="salmon", width=2),
                marker=dict(size=7),
                hovertemplate=f"Score bin: %{{x:.3f}}<br>Mean log(1+{cit_label}): %{{y:.2f}}<extra></extra>",
            )
        )

        if len(bx) >= 4:
            window = max(3, len(bx) // 5)
            pad = window // 2
            padded = np.pad(by, pad, mode="edge")
            ma = np.convolve(padded, np.ones(window) / window, mode="valid")
            fig.add_trace(
                go.Scatter(
                    x=bx.tolist(),
                    y=ma.tolist(),
                    mode="lines",
                    name="Smooth mean",
                    line=dict(color="firebrick", width=4),
                    hovertemplate=f"Score bin: %{{x:.3f}}<br>Smooth mean: %{{y:.2f}}<extra></extra>",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=x_ols.tolist(),
                y=y_ols.tolist(),
                mode="lines",
                name=f"OLS (slope={ols_slope:.3f})",
                line=dict(color="black", width=2, dash="dash"),
                hovertemplate=f"OLS: %{{y:.2f}}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"[{cit_type}] {SCORE_LABELS[col]} vs log(1 + {cit_label})  (r = {r:.3f})",
            xaxis_title=SCORE_LABELS[col],
            yaxis_title=f"log(1 + {cit_label})",
            hovermode="closest",
            margin=dict(l=40, r=20, t=50, b=40),
        )

        html_path = fig_dir / f"{stem}.html"
        png_path = fig_dir / f"{stem}.png"
        fig.write_html(str(html_path))
        fig.write_image(str(png_path), scale=3)
        print(f"  [OK] {stem}.html / .png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df)} rows from {PARQUET_PATH.name}")
    print(f"  generated_by values: {sorted(df['generated_by'].dropna().unique())}")
    for col in SCORE_COLS:
        n_valid = df[col].notna().sum()
        print(
            f"  {col}: {n_valid} non-null  mean={df[col].mean():.4f}  std={df[col].std():.4f}"
        )

    fig_dir = FIG_DIR / PARQUET_PATH.stem
    fig_dir.mkdir(parents=True, exist_ok=True)

    current_year = 2026
    df["yearly_citations"] = df["citation_count"] / (current_year - df["year"]).clip(
        lower=1
    )

    plot_overall_distribution(df, fig_dir)
    plot_grouped_distributions(df, fig_dir)
    plot_grouped_stats(df, fig_dir)
    plot_correlation_with_citations(df, fig_dir)
    plot_correlation_with_citations(
        df,
        fig_dir,
        cit_col="yearly_citations",
        cit_label="yearly_citations",
        stem_suffix="yearly_citations",
    )

    print(f"\nAll figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
