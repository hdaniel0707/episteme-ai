"""Word- and token-count summary statistics per ``generated_by`` group.

Works on the "family of" wp / essay / reuter parquets (``*_binox*.parquet``),
which all carry ``id`` / ``text`` / ``generated_by`` columns. Only the ``human``
and ``gpt`` groups are kept by default — matched as *families* by prefix, so
``gpt_plain``, ``gpt_prompt1``, ... are included and reported under their own
label, while ``claude`` / ``gemini`` rows are dropped.

Two length measures per row:
  - words  — whitespace-separated chunks (``text.split()``)
  - tokens — by default a built-in regex tokenizer (``\\w+`` plus single
    punctuation marks), which needs no model download. Pass ``--tokenizer
    <hf-model-id>`` (e.g. ``allenai/specter2_base``, the model behind the
    ``emb_specter2`` column) to count with a real HuggingFace tokenizer instead.

Outputs saved to results/text_lengths/<corpus>/, one directory per corpus
(a domain of a combined parquet, or a whole per-corpus file):
  - text_lengths_summary.log       — everything printed to the terminal
  - text_lengths_summary.csv       — one row per (group, metric) with the stats
  - text_lengths_per_row.csv       — id / group / word_count / token_count
  - text_lengths_hist.png          — density histograms, words | tokens
  - text_lengths_violin.png        — violin + inner box per group, words | tokens
  - text_lengths_ecdf.png          — empirical CDFs per group, words | tokens

Usage:
    # every domain of a combined parquet, one report each
    uv run python epai/ai_detection/analyse/analyse_text_lengths.py \
        data/parquet/ghostbuster_gpt54mini.parquet

    # one domain, counted with a real tokenizer
    uv run python epai/ai_detection/analyse/analyse_text_lengths.py \
        data/parquet/ghostbuster_gpt54mini.parquet --domains reuter \
        --tokenizer allenai/specter2_base

    # a per-corpus file (no domain column) is analysed whole
    uv run python epai/ai_detection/analyse/analyse_text_lengths.py \
        data/parquet/wp_gpt54mini_binox0_red_200w.parquet
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epai.utils.color_utils import group_color
from epai.utils.corpus_utils import CorpusSpec, read_corpus, specs_from_paths

BASE_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "text_lengths"

ID_COL = "id"
TEXT_COL = "text"
GROUP_COL = "generated_by"

# Group *families*: a generated_by label is kept when it starts with one of
# these (same longest-prefix idea as cfg/colors.yaml), so gpt_plain counts as
# gpt. Families are listed in plot/report order.
DEFAULT_FAMILIES = ("human", "gpt")

# Built-in tokenizer: word characters runs, or a single non-space symbol.
TOKEN_RE = re.compile(r"\w+|[^\w\s]")

METRICS = ("word_count", "token_count")
METRIC_LABELS = {"word_count": "Words", "token_count": "Tokens"}
PERCENTILES = (25, 50, 75, 90)


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


def select_groups(
    df: pd.DataFrame, families: tuple[str, ...], out: Tee
) -> tuple[pd.DataFrame, list[str]]:
    """Keep only rows whose group belongs to one of ``families``.

    Returns the filtered frame and the group labels in family order (``human``
    before every ``gpt*`` variant, variants sorted alphabetically).
    """
    labels = sorted(str(g) for g in df[GROUP_COL].dropna().unique())

    kept: list[str] = []
    for fam in families:
        kept.extend(lbl for lbl in labels if lbl.lower().startswith(fam.lower()))
    kept = list(dict.fromkeys(kept))  # dedupe, keep family order

    dropped = [lbl for lbl in labels if lbl not in kept]
    out.print(f"Groups found:  {labels}")
    out.print(f"Groups kept:   {kept}   (families: {list(families)})")
    out.print(f"Groups dropped:{dropped if dropped else ' (none)'}")

    if not kept:
        raise SystemExit(
            f"None of the families {list(families)} matched any {GROUP_COL} "
            f"value (available: {labels})."
        )

    return df[df[GROUP_COL].isin(kept)].copy(), kept


def count_tokens(texts: list[str], tokenizer_spec: str) -> list[int]:
    """Token count per text, via the built-in regex or a HuggingFace tokenizer."""
    if tokenizer_spec == "regex":
        return [len(TOKEN_RE.findall(t)) for t in texts]

    from transformers import AutoTokenizer

    print(f"Loading tokenizer '{tokenizer_spec}' ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )["input_ids"]
    return [len(ids) for ids in encoded]


def compute_lengths(
    df: pd.DataFrame, tokenizer_spec: str, out: Tee
) -> pd.DataFrame:
    """Attach word_count / token_count columns; drop and report empty texts."""
    texts = df[TEXT_COL]
    empty = texts.isna() | texts.fillna("").astype(str).str.strip().eq("")
    n_empty = int(empty.sum())
    if n_empty:
        ids = df.loc[empty, ID_COL].tolist() if ID_COL in df.columns else []
        out.print(
            f"Skipped {n_empty} row(s) with missing/empty '{TEXT_COL}'"
            + (f" (ids: {ids[:20]}{' ...' if len(ids) > 20 else ''})" if ids else "")
        )

    kept = df.loc[~empty].copy()
    texts = kept[TEXT_COL].astype(str).tolist()
    kept["word_count"] = [len(t.split()) for t in texts]
    kept["token_count"] = count_tokens(texts, tokenizer_spec)
    return kept


def stats_for(values: np.ndarray) -> dict:
    """Summary statistics for one group / metric."""
    if values.size == 0:
        keys = ["n", "mean", "std", "min", "max", "sum"]
        keys += [f"p{p}" for p in PERCENTILES]
        return {k: float("nan") for k in keys} | {"n": 0}

    s = {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "sum": float(values.sum()),
    }
    for p, q in zip(PERCENTILES, np.percentile(values, PERCENTILES)):
        s[f"p{p}"] = float(q)
    return s


def series_by_group(
    df: pd.DataFrame, groups: list[str], metric: str
) -> dict[str, np.ndarray]:
    """Metric values per group, in the given group order."""
    return {
        g: df.loc[df[GROUP_COL] == g, metric].to_numpy(dtype=float) for g in groups
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _panel_grid(title: str) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """One panel per metric, sharing nothing (words and tokens differ in scale)."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6.0 * len(METRICS), 4.6))
    fig.suptitle(title, fontsize=13)
    return fig, dict(zip(METRICS, np.atleast_1d(axes)))


def _finish(fig: plt.Figure, fig_dir: Path, fname: str) -> None:
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  [OK] {fname}")


def plot_histograms(
    df: pd.DataFrame, groups: list[str], fig_dir: Path, stem: str
) -> None:
    """Overlaid density histograms — the shape of each group's length spread."""
    fig, axes = _panel_grid(f"Text length distributions — {stem}")

    for metric, ax in axes.items():
        data = series_by_group(df, groups, metric)
        pooled = np.concatenate([v for v in data.values() if v.size]) if data else None
        if pooled is None or pooled.size == 0:
            ax.axis("off")
            continue
        lo, hi = np.percentile(pooled, [0, 99])
        if lo == hi:
            lo, hi = float(pooled.min()), float(pooled.max()) or 1.0
        bins = np.linspace(lo, hi, 40)

        for g, values in data.items():
            if values.size == 0:
                continue
            color = group_color(g)
            ax.hist(
                values, bins=bins, density=True, color=color, alpha=0.55,
                label=f"{g} (n={values.size})",
            )
            ax.axvline(values.mean(), color=color, linestyle="--", linewidth=1.4)

        ax.set_xlabel(METRIC_LABELS[metric])
        ax.set_ylabel("Density")
        ax.set_title(f"{METRIC_LABELS[metric]} per text  (dashed = mean)", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    _finish(fig, fig_dir, "text_lengths_hist.png")


def plot_violins(
    df: pd.DataFrame, groups: list[str], fig_dir: Path, stem: str
) -> None:
    """Violin per group with an inner box — spread plus median/quartiles."""
    fig, axes = _panel_grid(f"Text length per group — {stem}")

    for metric, ax in axes.items():
        data = series_by_group(df, groups, metric)
        usable = [(g, v) for g, v in data.items() if v.size]
        if not usable:
            ax.axis("off")
            continue
        positions = list(range(1, len(usable) + 1))
        values = [v for _, v in usable]

        parts = ax.violinplot(
            values, positions=positions, showmeans=False, showextrema=False, widths=0.8
        )
        for pc, (g, _) in zip(parts["bodies"], usable):
            pc.set_facecolor(group_color(g))
            pc.set_edgecolor("white")
            pc.set_linewidth(2)  # 2px surface gap between adjacent fills
            pc.set_alpha(0.65)

        ax.boxplot(
            values, positions=positions, widths=0.12, showfliers=False,
            medianprops=dict(color="black", linewidth=1.6),
            boxprops=dict(color="black", linewidth=1.0),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
        )

        for x, (_, v) in zip(positions, usable):
            ax.text(
                x, float(v.max()), f"mean {v.mean():,.0f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        ax.set_xticks(positions)
        ax.set_xticklabels([f"{g}\nn={v.size}" for g, v in usable], fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]} per text", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.margins(y=0.12)

    _finish(fig, fig_dir, "text_lengths_violin.png")


def plot_ecdf(df: pd.DataFrame, groups: list[str], fig_dir: Path, stem: str) -> None:
    """Empirical CDFs — where the groups separate, without binning artefacts."""
    fig, axes = _panel_grid(f"Text length ECDF — {stem}")

    for metric, ax in axes.items():
        data = series_by_group(df, groups, metric)
        plotted = False
        for g, values in data.items():
            if values.size == 0:
                continue
            xs = np.sort(values)
            ys = np.arange(1, xs.size + 1) / xs.size
            ax.step(
                xs, ys, where="post", color=group_color(g), linewidth=2,
                label=f"{g} (n={xs.size})",
            )
            plotted = True

        if not plotted:
            ax.axis("off")
            continue

        ax.axhline(0.5, color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlabel(METRIC_LABELS[metric])
        ax.set_ylabel("Cumulative fraction of texts")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{METRIC_LABELS[metric]} ECDF  (dotted = median)", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    _finish(fig, fig_dir, "text_lengths_ecdf.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(
    df: pd.DataFrame,
    groups: list[str],
    spec: CorpusSpec,
    tokenizer_spec: str,
    fig_dir: Path,
    out: Tee,
) -> None:
    out.print("\n" + "=" * 78)
    out.print(f"Word / token count summary: {spec.source}")
    out.print(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    out.print(f"Tokenizer: {tokenizer_spec}")
    out.print("=" * 78)

    rows = []
    for metric in METRICS:
        data = series_by_group(df, groups, metric)
        out.print(f"\n[{METRIC_LABELS[metric]}]  ({metric})")
        header = (
            f"  {'group':<14}{'n':>7}{'mean':>11}{'std':>11}{'min':>9}"
            + "".join(f"{'p' + str(p):>9}" for p in PERCENTILES)
            + f"{'max':>9}{'total':>13}"
        )
        out.print(header)
        out.print("  " + "-" * (len(header) - 2))

        for g, values in data.items():
            s = stats_for(values)
            out.print(
                f"  {g:<14}{s['n']:>7}{s['mean']:>11.1f}{s['std']:>11.1f}"
                f"{s['min']:>9.0f}"
                + "".join(f"{s['p' + str(p)]:>9.0f}" for p in PERCENTILES)
                + f"{s['max']:>9.0f}{s['sum']:>13,.0f}"
            )
            rows.append({"corpus": spec.name, "group": g, "metric": metric} | s)

        # Human-vs-each-generator deltas, the comparison this script exists for.
        if groups and groups[0].lower().startswith("human"):
            human = data[groups[0]]
            for g in groups[1:]:
                other = data[g]
                if human.size == 0 or other.size == 0:
                    continue
                diff = other.mean() - human.mean()
                ratio = other.mean() / human.mean() if human.mean() else float("nan")
                out.print(
                    f"  → {g} vs {groups[0]}: mean {diff:+.1f} "
                    f"({ratio:.2f}x), median "
                    f"{np.median(other) - np.median(human):+.1f}"
                )

    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = fig_dir / "text_lengths_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  [OK] {csv_path.name}")

    per_row_cols = [c for c in (ID_COL, GROUP_COL, *METRICS) if c in df.columns]
    per_row_path = fig_dir / "text_lengths_per_row.csv"
    df[per_row_cols].to_csv(per_row_path, index=False)
    print(f"  [OK] {per_row_path.name}")

    out.print("\n" + "=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("parquets", nargs="+", type=Path,
                        help="Parquet file(s) to analyse. A combined file is "
                             "analysed once per domain unless --domains narrows it.")
    parser.add_argument("--domains", nargs="*", default=None, metavar="DOMAIN",
                        help="Domains to analyse from each parquet (default: every "
                             "domain the file has; a file with no domain column is "
                             "analysed whole).")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_FAMILIES),
        help="Group families to keep, matched by prefix on generated_by "
        f"(default: {' '.join(DEFAULT_FAMILIES)}).",
    )
    parser.add_argument(
        "--tokenizer",
        default="regex",
        help="'regex' for the built-in tokenizer (default, no download), or a "
        "HuggingFace model id such as 'allenai/specter2_base'.",
    )
    return parser.parse_args()


def analyse_one(spec: CorpusSpec, args: argparse.Namespace) -> None:
    """One corpus: its own figures, report and output directory."""
    df = read_corpus(spec, quiet=True).df
    for col in (TEXT_COL, GROUP_COL):
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' missing from {spec.path.name}.")

    out = Tee()
    out.print(f"Loaded {len(df)} rows ← {spec.source}")

    subset, groups = select_groups(df, tuple(args.groups), out)
    subset = compute_lengths(subset, args.tokenizer, out)
    out.print(f"Analysing {len(subset)} rows across {len(groups)} group(s).")

    fig_dir = RESULTS_DIR / spec.name
    print()
    plot_histograms(subset, groups, fig_dir, spec.name)
    plot_violins(subset, groups, fig_dir, spec.name)
    plot_ecdf(subset, groups, fig_dir, spec.name)
    report(subset, groups, spec, args.tokenizer, fig_dir, out)

    out.save(fig_dir / "text_lengths_summary.log")
    print(f"\nAll outputs saved to {fig_dir}")


def main() -> None:
    args = parse_args()
    for spec in specs_from_paths(args.parquets, args.domains, base_dir=BASE_DIR):
        analyse_one(spec, args)


if __name__ == "__main__":
    main()
