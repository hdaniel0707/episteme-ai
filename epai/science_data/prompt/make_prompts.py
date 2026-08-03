"""Stage 3 — build one generation prompt per paper.

The Ghostbuster essay/wp prompts are content-derived specs, not topic labels
(``data/ghostbuster-data-2/essay/prompts/`` reads like "Discuss the concept of
perishability in service marketing and its implications for supply and demand
regulation..."), so this stage mirrors that: a model writes a short description
of what the paper does, and that description — not the abstract — is what the
generator sees in stage 4.

Two reasons it is built that way rather than from the title alone:

  - **Comparability.** A title-only prompt is systematically vaguer than the
    prompts behind essay/wp/reuter. Cross-domain results would then partly
    measure prompt specificity rather than domain transfer.
  - **Pairing.** The per-id human-vs-AI analyses compare the two rows sharing an
    ``id``. Holding the content roughly fixed means the residual difference is
    style, which is the thing being detected.

The cost is leakage risk: a spec that quotes the abstract would let the
generated text inherit human phrasing. That is contained on two sides — the
spec prompt forbids verbatim phrases and numeric results, and
``analyse_prompts.py`` measures the n-gram overlap that actually resulted, so
the assumption is checked rather than trusted.

``prompt.mode: title_only`` switches to the deterministic, leak-free variant
for ablation.

The target length is the source abstract's own word count, carried into the
prompt. Without it, generated abstracts converge on a characteristic length and
a length-only classifier scores well for reasons that have nothing to do with
authorship.

Output: data/science/<dataset_id>_prompts.parquet
        data/science/shards/<dataset_id>_spec.jsonl  (resumable checkpoint)

Run:
    python -m epai.science_data.prompt.make_prompts \
        --cfg cfg/science_data/science_cfg_example.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from epai.science_data.config import (
    human_path,
    load_pipeline_config,
    prompts_path,
    shard_path,
)
from epai.science_data.llm import Shard, openai_text_call, run_resumable
from epai.utils.parquet_utils import load_parquet, save_parquet

load_dotenv()

SPEC_SYSTEM_PROMPT = """You are summarising research papers so that another writer can reconstruct them.

Given a paper's title and abstract, write a 2-3 sentence description of what the paper investigates, what approach it takes, and what kind of finding it reports.

Requirements:
- Describe the work at a conceptual level: the question, the approach, the type of result.
- Do NOT include any numerical results, percentages, sample sizes, dataset sizes, or performance figures.
- Do NOT copy any phrase of five or more consecutive words from the abstract.
- Do NOT use the words "abstract" or "the paper states".
- Write plain prose, no bullet points, no headings.
- Stay under 80 words."""

GENERATION_PROMPT_TEMPLATE = """Write the abstract of a research paper.

Title: {title}
Field: {domain}
What the paper is about: {spec}

Write approximately {target_words} words. Output only the abstract text, with no title, no headings, and no section labels."""

TITLE_ONLY_TEMPLATE = """Write the abstract of a research paper.

Title: {title}
Field: {domain}

Write approximately {target_words} words. Output only the abstract text, with no title, no headings, and no section labels."""

OUT_COLUMNS = [
    "id",
    "text",
    "generated_by",
    "title",
    "year",
    "venue",
    "citation_count",
    "influential_citation_count",
    "fields_of_study",
    "domain",
    "query",
    "also_matched_domains",
    "spec",
    "prompt",
    "target_words",
]


def build_specs(df: pd.DataFrame, cfg: dict, retry_failed: bool) -> dict[str, str]:
    """Get a content spec per paper, resuming from the shard."""
    client = OpenAI()
    model = cfg["prompt"]["model"]
    shard = Shard(shard_path(cfg, "spec"))

    tasks = []
    for row in df.to_dict("records"):
        user = f"Title: {row['title']}\n\nAbstract: {row['text']}"
        tasks.append(
            (row["id"], openai_text_call(client, model, SPEC_SYSTEM_PROMPT, user))
        )

    run_resumable(
        tasks,
        shard,
        max_workers=cfg["prompt"].get("max_workers", 8),
        desc="writing specs",
        retry_failed=retry_failed,
        payload_of=lambda _k, text: {"spec": text},
    )
    return {k: r["spec"] for k, r in shard.ok_records().items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, required=True, help="pipeline config YAML")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry records previously recorded as failures",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(args.cfg)
    mode = cfg["prompt"].get("mode", "llm_spec")

    df = load_parquet(human_path(cfg))
    # Length target comes from the paper this prompt was derived from, so the
    # generated abstract is matched per-paper rather than to a corpus average.
    df["target_words"] = df["text"].str.split().str.len()

    if mode == "llm_spec":
        specs = build_specs(df, cfg, args.retry_failed)
        missing = set(df["id"]) - set(specs)
        if missing:
            print(
                f"\n{len(missing)} paper(s) have no spec and are dropped here. "
                "Re-run with --retry-failed to attempt them again."
            )
            df = df[df["id"].isin(specs)].copy()
        df["spec"] = df["id"].map(specs)
        df["prompt"] = [
            GENERATION_PROMPT_TEMPLATE.format(
                title=r["title"],
                domain=r["domain"],
                spec=r["spec"],
                target_words=r["target_words"],
            )
            for r in df.to_dict("records")
        ]
    else:
        df["spec"] = ""
        df["prompt"] = [
            TITLE_ONLY_TEMPLATE.format(
                title=r["title"], domain=r["domain"], target_words=r["target_words"]
            )
            for r in df.to_dict("records")
        ]

    save_parquet(df[OUT_COLUMNS].reset_index(drop=True), prompts_path(cfg))
    print(f"\nPrompt mode: {mode}")
    print(f"Prompts built for {len(df)} papers across {df['domain'].nunique()} domains")


if __name__ == "__main__":
    main()
