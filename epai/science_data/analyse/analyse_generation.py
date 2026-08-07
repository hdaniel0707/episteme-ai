"""Stage 5d — is the generated corpus a fair human-vs-AI comparison?

Runs on the final parquet and looks for the ways this task can be accidentally
trivial. A detector scoring well on a corpus with any of these problems has
learned an artifact, not authorship.

  - **Length.** Stage 4 targets each source abstract's own word count, so the
    two classes should overlap closely. The length-only ROC AUC is the number
    to read: at ~0.5 length carries no signal, and anything above ~0.65 means a
    trivial baseline competes with your real models, so every later result has
    to be read against it.
  - **Verbatim overlap.** Under the Ghostbuster protocol the generator never
    sees the human abstract, so overlap should be near the floor set by shared
    technical vocabulary. A heavy tail means stage 3 leaked (cross-check
    ``analyse_prompts.py``).
  - **Formatting tells.** Markdown, section labels and stock LLM openers occur
    in one class only and are learnable without any authorship signal.
  - **Degenerate output.** Refusals, truncation and near-empty generations.

Run (--cfg defaults to the example in cfg/science_data/):
    uv run python epai/science_data/analyse/analyse_generation.py --variant gpt54mini_0807A
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from epai.science_data.analyse._report import Report, describe, overlap_fraction
from epai.science_data.config import (
    BASE_DIR,
    DEFAULT_CFG_PATH,
    final_path,
    generated_by_label,
    load_pipeline_config,
)
from epai.utils.color_utils import group_color
from epai.utils.parquet_utils import load_parquet

HUMAN = "human"

# Surface patterns that should not distinguish the classes. Stock openers are
# not errors on their own — a real abstract can say "In this paper" — but a
# large gap in their rate between classes is a learnable tell.
TELL_PATTERNS = {
    "markdown": re.compile(r"[*_#`]"),
    "section_label": re.compile(
        r"\b(background|objectives?|methods?|results|conclusions?):", re.IGNORECASE
    ),
    "stock_opener": re.compile(
        r"^\s*(in this (paper|study|work)|this (paper|study|work) (presents|investigates|explores|examines))",
        re.IGNORECASE,
    ),
    "em_dash": re.compile(r"—"),
    "delve_moreover": re.compile(r"\b(delve|moreover|furthermore|notably)\b", re.IGNORECASE),
    "first_person_plural": re.compile(r"\b(we|our)\b", re.IGNORECASE),
}

REFUSAL_RE = re.compile(
    r"^\s*(i'm sorry|i am sorry|i cannot|i can't|as an ai|unable to)", re.IGNORECASE
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cfg",
        type=Path,
        default=DEFAULT_CFG_PATH,
        help=f"pipeline config YAML (default: {DEFAULT_CFG_PATH.relative_to(BASE_DIR)})",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="filename suffix used at generation time (default: the single "
        "generator's label with ':' replaced, else 'multi')",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(args.cfg)
    generators = cfg["generate"]["generators"]
    ai_labels = [generated_by_label(g["model"], cfg) for g in generators]
    variant = args.variant or (
        ai_labels[0].replace(":", "_") if len(ai_labels) == 1 else "multi"
    )

    df = load_parquet(final_path(cfg, variant))
    rep = Report("generation", cfg["dataset_id"])
    df["words"] = df["text"].fillna("").str.split().str.len()

    # ---------------------------------------------------------- composition --
    rep.header("Composition")
    rep.line(f"{len(df)} rows, {df['id'].nunique()} papers, variant '{variant}'")
    rep.table(
        df.groupby(["domain", "generated_by"]).size().reset_index(name="rows"),
        "composition",
    )

    incomplete = df.groupby("id")["generated_by"].nunique()
    n_incomplete = int((incomplete < len(ai_labels) + 1).sum())
    if n_incomplete:
        rep.line("")
        rep.line(
            f"WARNING: {n_incomplete} id(s) lack a full set of rows. The per-id "
            "human-vs-AI analyses will silently skip them."
        )

    # -------------------------------------------------------------- length --
    rep.header("Length by class")
    rep.table(
        pd.DataFrame(
            [describe(g["words"], label) for label, g in df.groupby("generated_by", sort=False)]
        ),
        "length_by_class",
    )

    human = df[df["generated_by"] == HUMAN].set_index("id")
    rep.line("")
    for label in ai_labels:
        ai = df[df["generated_by"] == label].set_index("id")
        shared = human.index.intersection(ai.index)
        if shared.empty:
            continue
        ratio = ai.loc[shared, "words"] / human.loc[shared, "words"].replace(0, np.nan)
        rep.line(
            f"{label}: per-paper length ratio (AI/human) median "
            f"{ratio.median():.3f}, IQR {ratio.quantile(0.25):.3f}-{ratio.quantile(0.75):.3f}"
        )

    # ------------------------------------------------- length-only baseline --
    rep.header("Length-only baseline (the sanity floor)")
    baseline_rows = []
    for label in ai_labels:
        pair = df[df["generated_by"].isin([HUMAN, label])]
        if pair["generated_by"].nunique() < 2:
            continue
        y = (pair["generated_by"] == label).astype(int)
        auc = roc_auc_score(y, pair["words"])
        baseline_rows.append(
            {"generator": label, "length_only_auc": round(float(auc), 4)}
        )
    baseline = pd.DataFrame(baseline_rows)
    rep.table(baseline, "length_only_baseline")
    rep.line("")
    worst = baseline["length_only_auc"].map(lambda a: abs(a - 0.5)).max() if not baseline.empty else 0
    if worst > 0.15:
        rep.line(
            "WARNING: length alone separates the classes (AUC far from 0.5). Any "
            "detector result on this corpus must be reported against this "
            "baseline, or length-match the classes before training."
        )
    else:
        rep.line("Length carries little signal — the classes are well matched.")

    # ------------------------------------------------------------- overlap --
    rep.header("Verbatim overlap with the human abstract")
    overlap_rows = []
    for label in ai_labels:
        ai = df[df["generated_by"] == label].set_index("id")
        shared = human.index.intersection(ai.index)
        vals = [
            overlap_fraction(human.loc[i, "text"], ai.loc[i, "text"], n=5)
            for i in tqdm(shared, desc=f"overlap {label}", unit="paper")
        ]
        overlap_rows.append(describe(pd.Series(vals), label))
        pd.DataFrame({"id": shared, "overlap_5gram": vals}).to_csv(
            rep.dir / f"overlap_{label}.csv", index=False
        )
    rep.table(pd.DataFrame(overlap_rows), "overlap_by_generator")
    rep.line("")
    rep.line(
        "Under the Ghostbuster protocol the generator never saw the human text, "
        "so this should sit near the floor set by shared terminology. A long "
        "right tail points back at stage 3 leakage."
    )

    # -------------------------------------------------------------- tells --
    rep.header("Formatting tells (rate per class)")
    tell_rows = []
    for label, group in df.groupby("generated_by", sort=False):
        row = {"generated_by": label, "n": len(group)}
        for name, pat in TELL_PATTERNS.items():
            row[name] = round(100.0 * group["text"].fillna("").str.contains(pat).mean(), 2)
        tell_rows.append(row)
    tells = pd.DataFrame(tell_rows)
    rep.table(tells, "formatting_tells")
    rep.line("")
    rep.line("Values are the percentage of rows matching. Large gaps are learnable artifacts.")

    gaps = []
    human_row = tells[tells["generated_by"] == HUMAN]
    for name in TELL_PATTERNS:
        if human_row.empty:
            break
        h = float(human_row.iloc[0][name])
        for label in ai_labels:
            ai_row = tells[tells["generated_by"] == label]
            if ai_row.empty:
                continue
            gap = abs(float(ai_row.iloc[0][name]) - h)
            if gap > 30:
                gaps.append(f"{name}: {h:.1f}% human vs {float(ai_row.iloc[0][name]):.1f}% {label}")
    if gaps:
        rep.line("")
        rep.line("WARNING: >30-point gaps, effectively free classifier features:")
        for g in gaps:
            rep.line(f"  {g}")

    # --------------------------------------------------------- degenerate --
    rep.header("Degenerate generations")
    degen_rows = []
    for label in ai_labels:
        group = df[df["generated_by"] == label]
        text = group["text"].fillna("")
        degen_rows.append(
            {
                "generator": label,
                "empty": int((text.str.strip() == "").sum()),
                "under_50_words": int((group["words"] < 50).sum()),
                "refusal_like": int(text.str.contains(REFUSAL_RE).sum()),
                "ends_mid_sentence": int((~text.str.rstrip().str.endswith((".", "?", "!"))).sum()),
            }
        )
    rep.table(pd.DataFrame(degen_rows), "degenerate_generations")

    # --------------------------------------------------------------- plots --
    labels = [HUMAN, *ai_labels]
    present = [l for l in labels if (df["generated_by"] == l).any()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for label in present:
        vals = df.loc[df["generated_by"] == label, "words"]
        axes[0].hist(vals, bins=40, alpha=0.55, label=label, color=group_color(label), density=True)
    axes[0].set_xlabel("abstract length (words)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Length by class")
    axes[0].legend()

    axes[1].violinplot(
        [df.loc[df["generated_by"] == l, "words"].to_numpy() for l in present],
        showmedians=True,
    )
    axes[1].set_xticks(range(1, len(present) + 1))
    axes[1].set_xticklabels(present, rotation=20, ha="right")
    axes[1].set_ylabel("words")
    axes[1].set_title("Length spread")
    rep.figure(fig, "length_by_class")

    rep.close()


if __name__ == "__main__":
    main()
