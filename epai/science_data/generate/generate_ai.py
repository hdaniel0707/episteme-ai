"""Stage 4 — generate the AI abstract for each prompt, and export the corpus.

Each generator sees only the prompt from stage 3 — never the human abstract.
That is the Ghostbuster protocol, and it is what separates this corpus from the
rephrasing setup in ``epai/utils/genai_utils.py``, where the model rewrites the
human text and the AI class inherits its content and structure.

Work is keyed by ``(id, model, prompt_name)`` and checkpointed per call, so:

  - a crash costs only the calls in flight;
  - adding a generator to the config later issues only that generator's calls,
    leaving the existing rows untouched and unpaid-for a second time.

``require_all_generators`` is applied here at export, not during generation. A
paper is kept only if every configured generator produced text for it, so all
generators see the same set of papers — but because the check runs at export it
re-evaluates correctly when the generator set changes, instead of baking in
whichever set happened to run first.

The exported frame carries one human row and one row per generator for each
id, matching the ``id`` / ``text`` / ``generated_by`` schema of the essay, wp
and reuter corpora. ``generated_by`` uses each generator's configured ``label``
(default ``gpt_plain``), which puts it in the "gpt" family that
``analyse_text_lengths.py`` and ``cfg/colors.yaml`` key on by prefix.

Output: data/parquet/<dataset_id>_all_<variant>.parquet
        data/parquet/<dataset_id>_<domain>_<variant>.parquet
        data/science/shards/<dataset_id>_generate.jsonl  (resumable checkpoint)

Run:
    python -m epai.science_data.generate.generate_ai \
        --cfg cfg/science_data/science_cfg_example.yaml
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from epai.science_data.config import (
    final_path,
    load_pipeline_config,
    prompts_path,
    shard_path,
)
from epai.science_data.llm import Shard, openai_text_call, run_resumable
from epai.utils.parquet_utils import load_parquet, save_parquet

load_dotenv()

GENERATION_SYSTEM_PROMPT = """You are a researcher writing the abstract of your own paper.

Write in the register and structure of a real published abstract in the stated field. Output the abstract text only: no title, no "Abstract:" heading, no section labels such as "Background:" or "Methods:", no markdown formatting, no bullet points."""

# The human abstracts had these stripped by rules.clean_abstract, so leaving
# them in the generated text would create a formatting asymmetry between the
# classes — a tell that has nothing to do with authorship.
_LEADING_LABEL_RE = re.compile(r"^\s*(abstract|summary)\b[\s:.\-—]*", re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"[*_#`]+")
_WS_RE = re.compile(r"\s+")

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
]


def clean_generated(text: str) -> str:
    """Normalise generated text the same way the human abstracts were."""
    text = _LEADING_LABEL_RE.sub("", text)
    text = _MARKDOWN_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def _key(paper_id: str, model: str, prompt_name: str) -> str:
    return f"{paper_id}|{model}|{prompt_name}"


def generate(df: pd.DataFrame, cfg: dict, retry_failed: bool) -> Shard:
    client = OpenAI()
    shard = Shard(shard_path(cfg, "generate"))

    tasks = []
    for gen in cfg["generate"]["generators"]:
        model, prompt_name = gen["model"], gen["prompt_name"]
        for row in df.to_dict("records"):
            tasks.append(
                (
                    _key(row["id"], model, prompt_name),
                    openai_text_call(
                        client, model, GENERATION_SYSTEM_PROMPT, row["prompt"]
                    ),
                )
            )

    run_resumable(
        tasks,
        shard,
        max_workers=cfg["generate"].get("max_workers", 8),
        desc="generating abstracts",
        retry_failed=retry_failed,
        payload_of=lambda _k, text: {"abstract": clean_generated(text)},
    )
    return shard


def export(df: pd.DataFrame, shard: Shard, cfg: dict) -> pd.DataFrame:
    """Assemble human + generated rows, enforcing generator completeness."""
    generators = cfg["generate"]["generators"]
    ok = shard.ok_records()

    if cfg["generate"].get("require_all_generators", True):
        complete = {
            row_id
            for row_id in df["id"]
            if all(
                _key(row_id, g["model"], g["prompt_name"]) in ok for g in generators
            )
        }
        dropped = len(df) - len(complete)
        if dropped:
            print(
                f"\n{dropped} paper(s) dropped: not every generator produced text. "
                "Re-run with --retry-failed first if that count looks high."
            )
        df = df[df["id"].isin(complete)].copy()

    human = df.copy()
    human["generated_by"] = "human"
    frames = [human[OUT_COLUMNS]]

    print("\nExported rows per class:")
    print(f"  {'human':<20} {len(human):>6} rows")

    for gen in generators:
        model, prompt_name, label = gen["model"], gen["prompt_name"], gen["label"]
        rows = df.copy()
        texts = [
            ok.get(_key(row_id, model, prompt_name), {}).get("abstract")
            for row_id in rows["id"]
        ]
        rows["text"] = texts
        rows["generated_by"] = label
        # Only incomplete ids survive to here when require_all_generators is off.
        rows = rows[rows["text"].notna()]
        frames.append(rows[OUT_COLUMNS])
        print(f"  {label:<20} {len(rows):>6} rows")

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, required=True, help="pipeline config YAML")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry calls previously recorded as failures",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="filename suffix, e.g. 'gpt54mini'. Defaults to the single "
        "generator's label, or 'multi' when several are configured.",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(args.cfg)
    generators = cfg["generate"]["generators"]
    variant = args.variant or (
        generators[0]["label"] if len(generators) == 1 else "multi"
    )

    df = load_parquet(prompts_path(cfg))
    shard = generate(df, cfg, args.retry_failed)

    final = export(df, shard, cfg)

    save_parquet(final, final_path(cfg, variant))
    for domain, group in final.groupby("domain", sort=False):
        save_parquet(group.reset_index(drop=True), final_path(cfg, variant, domain))

    print(f"\nTotal {len(final)} rows, {final['id'].nunique()} papers")
    print(final.groupby(["domain", "generated_by"]).size().to_string())


if __name__ == "__main__":
    main()
