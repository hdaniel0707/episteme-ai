"""Stage 5c — did the prompt specs leak the human abstract?

This is the check that makes the ``llm_spec`` prompt mode defensible. That mode
was chosen because it keeps the generated abstract about the same *work* as the
human one, which is what the per-id human-vs-AI analyses depend on — but the
cost is that a spec quoting the source would let the generator inherit human
phrasing, and the AI class would then be part-paraphrase rather than
independently written.

The spec prompt forbids verbatim phrases and numeric results. This script
measures whether that held:

  - **5-gram overlap** — the share of the spec's 5-grams that appear verbatim in
    the source abstract. Some overlap is unavoidable (technical terms), but the
    tail is what matters.
  - **Longest verbatim run** — a single long quoted span is a leak even when the
    averaged overlap looks fine.
  - **Digit carry-over** — a coverage check, not a leak check, and the one
    metric here you want to be *high*. The spec is required to carry the
    paper's quantitative results, because a generator left to invent figures
    produces a detectable artifact of its own. A low share means the specs are
    vague and the AI abstracts are making their numbers up.

The flagged-ids CSV lets you drop or regenerate the offending papers rather
than discarding the whole run.

Run (--cfg defaults to the example in cfg/science_data/):
    uv run python epai/science_data/analyse/analyse_prompts.py [--overlap-threshold 0.10]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from epai.science_data.analyse._report import (
    Report,
    describe,
    longest_common_run,
    overlap_fraction,
)
from epai.science_data.config import BASE_DIR, DEFAULT_CFG_PATH, load_pipeline_config, prompts_path
from epai.utils.parquet_utils import load_parquet

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cfg",
        type=Path,
        default=DEFAULT_CFG_PATH,
        help=f"pipeline config YAML (default: {DEFAULT_CFG_PATH.relative_to(BASE_DIR)})",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.10,
        help="flag specs whose 5-gram overlap with the source exceeds this",
    )
    parser.add_argument(
        "--run-threshold",
        type=int,
        default=8,
        help="flag specs sharing a verbatim run of at least this many words",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(args.cfg)
    df = load_parquet(prompts_path(cfg))
    rep = Report("prompts", cfg["dataset_id"])

    mode = cfg["prompt"].get("mode", "llm_spec")
    rep.line(f"prompt mode: {mode}")
    rep.line(f"{len(df)} prompts across {df['domain'].nunique()} domains")

    # ------------------------------------------------------------ lengths --
    rep.header("Lengths")
    df["spec_words"] = df["spec"].fillna("").str.split().str.len()
    df["prompt_words"] = df["prompt"].str.split().str.len()
    rep.table(
        pd.DataFrame(
            [
                describe(df["spec_words"], "spec_words"),
                describe(df["prompt_words"], "prompt_words"),
                describe(df["target_words"], "target_words (source abstract)"),
            ]
        ),
        "prompt_lengths",
    )

    if mode == "title_only":
        rep.line("")
        rep.line(
            "Prompt mode is title_only: no spec was written, so there is nothing "
            "to leak and the overlap checks below are skipped. Note that these "
            "prompts are vaguer than the essay/wp prompts, which is a confound "
            "when comparing domains."
        )
        rep.close()
        return

    # ------------------------------------------------------------ overlap --
    rep.header("Leakage: spec vs. source abstract")
    records = df.to_dict("records")
    overlaps, runs, shared_numbers = [], [], []
    for row in tqdm(records, desc="measuring overlap", unit="paper"):
        spec, source = row.get("spec") or "", row["text"]
        overlaps.append(overlap_fraction(source, spec, n=5))
        runs.append(longest_common_run(source, spec))
        spec_nums = set(_NUMBER_RE.findall(spec))
        shared_numbers.append(len(spec_nums & set(_NUMBER_RE.findall(source))))

    df["overlap_5gram"] = overlaps
    df["longest_run"] = runs
    df["shared_numbers"] = shared_numbers

    rep.table(
        pd.DataFrame(
            [
                describe(df["overlap_5gram"], "overlap_5gram"),
                describe(df["longest_run"], "longest_verbatim_run_words"),
                describe(df["shared_numbers"], "shared_numbers"),
            ]
        ),
        "leakage_stats",
    )

    n_num = int((df["shared_numbers"] > 0).sum())
    rep.line("")
    rep.line(
        f"{n_num} spec(s) ({100.0 * n_num / max(len(df), 1):.1f}%) carry at least "
        "one number over from the source abstract — which is what the spec "
        "prompt asks for, so this should be HIGH."
    )
    if len(df) and n_num / len(df) < 0.5:
        rep.line("")
        rep.line(
            "WARNING: fewer than half the specs carry a number. The generator is "
            "then inventing figures, which is a detectable artifact of its own "
            "and makes the human and AI rows of a pair differ in content as well "
            "as style. Check that the spec prompt still requires the quantitative "
            "results, and that the source abstracts actually contain any."
        )

    # ------------------------------------------------------------ flagged --
    rep.header("Flagged specs")
    flagged = df[
        (df["overlap_5gram"] > args.overlap_threshold)
        | (df["longest_run"] >= args.run_threshold)
    ]
    pct = 100.0 * len(flagged) / max(len(df), 1)
    rep.line(
        f"{len(flagged)} of {len(df)} ({pct:.1f}%) exceed overlap "
        f">{args.overlap_threshold:.0%} or a verbatim run of >={args.run_threshold} words"
    )
    rep.table(
        flagged[["id", "domain", "overlap_5gram", "longest_run", "shared_numbers"]]
        .sort_values("overlap_5gram", ascending=False)
        .head(50),
        "flagged_specs",
        show=False,
    )
    if pct > 5:
        rep.line("")
        rep.line(
            "WARNING: leakage is above 5%. The generated abstracts will inherit "
            "human phrasing, which weakens the human-vs-AI contrast. Either "
            "tighten spec_system_prompt in the prompt config, give it a new id, "
            "and regenerate, or "
            "drop the flagged ids before training."
        )

    rep.header("Per-domain overlap")
    rep.table(
        pd.DataFrame(
            [describe(g["overlap_5gram"], d) for d, g in df.groupby("domain", sort=False)]
        ),
        "overlap_per_domain",
    )

    # -------------------------------------------------------------- plots --
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["overlap_5gram"], bins=40, color="#4f81bd")
    axes[0].axvline(args.overlap_threshold, color="#c0504d", linestyle="--", label="threshold")
    axes[0].set_xlabel("5-gram overlap with source")
    axes[0].legend()
    axes[1].hist(df["longest_run"], bins=range(0, max(int(df["longest_run"].max()) + 2, 12)), color="#4f81bd")
    axes[1].axvline(args.run_threshold, color="#c0504d", linestyle="--", label="threshold")
    axes[1].set_xlabel("longest verbatim run (words)")
    axes[1].legend()
    for ax in axes:
        ax.set_ylabel("specs")
    fig.suptitle("Prompt-spec leakage from the human abstract")
    rep.figure(fig, "spec_leakage")

    rep.close()


if __name__ == "__main__":
    main()
