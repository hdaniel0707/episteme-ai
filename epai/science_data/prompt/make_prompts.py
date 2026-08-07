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
spec prompt forbids any run of five or more consecutive words from the source,
and ``analyse_prompts.py`` measures the overlap that actually resulted, so the
assumption is checked rather than trusted.

**Facts cross, wording does not.** The spec is *required* to carry the paper's
quantitative results — effect sizes, performance figures, sample sizes — rather
than being stripped of them. Withholding them does not reduce the leak that
matters (that leak is phrasing) and it costs twice over: the generator has to
invent figures, and invented figures are a detectable artifact in their own
right, since models reach for round and implausibly clean numbers. It also
makes the two rows of a pair differ in content, when style is the only
difference the classifier is meant to have access to.

``prompt.mode: title_only`` switches to the deterministic, leak-free variant
for ablation — no spec, so nothing derived from the abstract reaches the
generator at all.

The target length is the source abstract's own word count, carried into the
prompt. Without it, generated abstracts converge on a characteristic length and
a length-only classifier scores well for reasons that have nothing to do with
authorship.

Output: data/science/<dataset_id>_prompts.parquet
        data/science/<dataset_id>_spec.jsonl  (resumable checkpoint)

Run (--cfg defaults to the example in cfg/science_data/):
    uv run python epai/science_data/prompt/make_prompts.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from epai.science_data.config import (
    BASE_DIR,
    DEFAULT_CFG_PATH,
    human_path,
    load_pipeline_config,
    load_prompt_config,
    prompt_config_path,
    prompts_path,
    shard_path,
)
from epai.science_data.llm import Shard, openai_text_call, run_resumable
from epai.science_data.rules import OPTIONAL_NUMERIC_COLUMNS
from epai.utils.parquet_utils import coerce_dtypes, load_parquet, save_parquet

load_dotenv()


def build_prompt(row: dict, mode: str, prompts: dict) -> str:
    """Render the generation prompt for one paper.

    ``title_only`` drops the spec block and the results instruction with it,
    which is what makes it the leak-free arm of the ablation: nothing derived
    from the abstract reaches the generator.
    """
    spec_block = (
        prompts["spec_block"].format(spec=row["spec"]) if mode == "llm_spec" else ""
    )
    return prompts["generation_prompt_template"].format(
        title=row["title"],
        domain=row["domain"],
        spec_block=spec_block,
        target_words=row["target_words"],
        results_instruction=(
            prompts["results_instruction"] if mode == "llm_spec" else ""
        ),
    )


OUT_COLUMNS = [
    "id",
    "text",
    "generated_by",
    "doi",
    "title",
    "author_ids",
    "author_count",
    "year",
    "venue",
    "citation_count",
    "fwci",
    "language",
    "fields_of_study",
    "domain",
    "query",
    "also_matched_domains",
    "spec",
    "prompt",
    "target_words",
]


def build_specs(
    df: pd.DataFrame, cfg: dict, prompts: dict, retry_failed: bool
) -> dict[str, str]:
    """Get a content spec per paper, resuming from the shard."""
    client = OpenAI()
    model = cfg["prompt"]["model"]
    shard = Shard(shard_path(cfg, "spec"))

    tasks = []
    for row in df.to_dict("records"):
        user = f"Title: {row['title']}\n\nAbstract: {row['text']}"
        tasks.append(
            (
                row["id"],
                openai_text_call(
                    client, model, prompts["spec_system_prompt"], user
                ),
            )
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
    parser.add_argument(
        "--cfg",
        type=Path,
        default=DEFAULT_CFG_PATH,
        help=f"pipeline config YAML (default: {DEFAULT_CFG_PATH.relative_to(BASE_DIR)})",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry records previously recorded as failures",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(args.cfg)
    prompts = load_prompt_config(cfg)
    mode = cfg["prompt"].get("mode", "llm_spec")
    print(f"Prompt config: {prompt_config_path(cfg).relative_to(BASE_DIR)}")

    df = load_parquet(human_path(cfg))
    # Length target comes from the paper this prompt was derived from, so the
    # generated abstract is matched per-paper rather than to a corpus average.
    df["target_words"] = df["text"].str.split().str.len()

    if mode == "llm_spec":
        specs = build_specs(df, cfg, prompts, args.retry_failed)
        missing = set(df["id"]) - set(specs)
        if missing:
            print(
                f"\n{len(missing)} paper(s) have no spec and are dropped here. "
                "Re-run with --retry-failed to attempt them again."
            )
            df = df[df["id"].isin(specs)].copy()
        df["spec"] = df["id"].map(specs)
    else:
        df["spec"] = ""

    df["prompt"] = [build_prompt(r, mode, prompts) for r in df.to_dict("records")]

    # reindex, not strict selection: a corpus filtered before `doi` /
    # `author_ids` / `author_count` existed carries none of them. Null is the
    # right value for a metadata column no stage reads; failing the run over it
    # is not.
    out = coerce_dtypes(
        df.reindex(columns=OUT_COLUMNS), OPTIONAL_NUMERIC_COLUMNS
    )
    save_parquet(out.reset_index(drop=True), prompts_path(cfg))
    print(f"\nPrompt mode: {mode}")
    print(f"Prompts built for {len(df)} papers across {df['domain'].nunique()} domains")


if __name__ == "__main__":
    main()
