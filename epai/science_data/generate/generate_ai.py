"""Stage 4 — generate the AI abstract for each prompt, and export the corpus.

Each generator sees only the prompt from stage 3 — never the human abstract.
That is the Ghostbuster protocol, and it is what separates this corpus from the
rephrasing setup in ``epai/utils/genai_utils.py``, where the model rewrites the
human text and the AI class inherits its content and structure.

Work is keyed by ``(id, generated_by)`` and checkpointed per call, so:

  - a crash costs only the calls in flight;
  - adding a generator to the config later issues only that generator's calls,
    leaving the existing rows untouched and unpaid-for a second time.

``require_all_generators`` is applied here at export, not during generation. A
paper is kept only if every configured generator produced text for it, so all
generators see the same set of papers — but because the check runs at export it
re-evaluates correctly when the generator set changes, instead of baking in
whichever set happened to run first.

The exported frame carries one human row and one row per generator for each
id, matching the ``uid`` / ``id`` / ``text`` / ``generated_by`` schema of the
essay, wp and reuter corpora.

``prompt`` is carried through from stage 3 onto the generated rows only, and is
null on the human row — a human abstract was not written to a prompt, and
filling the column there with the prompt derived FROM it would invite exactly
the reading the Ghostbuster protocol exists to prevent. On a generated row it is
the verbatim user message that produced the text, so the corpus can be read
without going back to ``<dataset_id>_prompts.parquet``. The system prompt is not
stored per row: it is one string for the whole run, and ``generated_by`` already
names the prompt config it came from.

``generated_by`` is DERIVED, never configured: ``<model_slug>:<prompt_config>``,
e.g. ``gpt54mini:0807A``. A label that can be typed by hand is the one field in
the corpus that can disagree with what actually produced the row. It still
begins with the model family, so ``cfg/colors.yaml`` resolves it by
longest-prefix exactly as before.

``uid`` (``<id>:<generated_by>``) is written here, as soon as a row knows its
label. ``id`` stays the PAIR key that splits group on.

Output: data/science/<dataset_id>_<variant>.parquet — one file, every domain
        and both classes, in ghostbuster column order. The last stage, and the
        only artifact meant to leave data/science/: move it to data/parquet/ by
        hand once the run looks good.
        data/science/<dataset_id>_generate.jsonl  (resumable checkpoint)

Run (--cfg defaults to the example in cfg/science_data/):
    uv run python epai/science_data/generate/generate_ai.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from epai.science_data.config import (
    BASE_DIR,
    DEFAULT_CFG_PATH,
    final_path,
    generated_by_label,
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

# The human abstracts had these stripped by rules.clean_abstract, so leaving
# them in the generated text would create a formatting asymmetry between the
# classes — a tell that has nothing to do with authorship.
_LEADING_LABEL_RE = re.compile(r"^\s*(abstract|summary)\b[\s:.\-—]*", re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"[*_#`]+")
_WS_RE = re.compile(r"\s+")

# `uid` is unique per row, `id` is the PAIR key shared by a paper's human row
# and every generated row derived from it. Both are needed and they do
# different jobs: splits group on `id`, so that a paper never lands on both
# sides of a split, while `uid` is what identifies a single row. Same
# convention and same separator as combine_domains.py.
ID_SEP = ":"

# Ghostbuster convention: uid, id, domain, text, generated_by first, then
# metadata, then any emb_* a later stage adds. Same front block as
# `combine_domains.order_columns`, so a science corpus and a ghostbuster one can
# be eyeballed with the same `df.iloc[:, :5]`.
#
# `prompt` goes last rather than beside `text`: it is long enough to make any
# `df.head()` unreadable, and putting it after the metadata keeps the front
# block identical to the other corpora.
OUT_COLUMNS = [
    "uid",
    "id",
    "domain",
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
    "query",
    "also_matched_domains",
    "prompt",
]


def clean_generated(text: str) -> str:
    """Normalise generated text the same way the human abstracts were."""
    text = _LEADING_LABEL_RE.sub("", text)
    text = _MARKDOWN_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def _key(paper_id: str, label: str) -> str:
    """Checkpoint key for one call.

    Keyed on the *label* (``gpt54mini:0807A``) rather than the model alone, so
    the prompt config is part of the identity of a cached call. Editing the
    prompts and pointing `prompt_config` at a new id therefore re-issues every
    call, instead of silently reusing text generated under the old prompts —
    which is the failure this whole labelling scheme exists to prevent.
    """
    return f"{paper_id}|{label}"


def generate(df: pd.DataFrame, cfg: dict, prompts: dict, retry_failed: bool) -> Shard:
    client = OpenAI()
    shard = Shard(shard_path(cfg, "generate"))

    tasks = []
    for gen in cfg["generate"]["generators"]:
        label = generated_by_label(gen["model"], cfg)
        for row in df.to_dict("records"):
            tasks.append(
                (
                    _key(row["id"], label),
                    openai_text_call(
                        client,
                        gen["model"],
                        prompts["generation_system_prompt"],
                        row["prompt"],
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
    labels = [generated_by_label(g["model"], cfg) for g in generators]
    ok = shard.ok_records()

    if cfg["generate"].get("require_all_generators", True):
        complete = {
            row_id
            for row_id in df["id"]
            if all(_key(row_id, label) in ok for label in labels)
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
    # The prompt travels with the row it produced, and it produced no human
    # row. `df` carries the column because stage 3 wrote it beside the human
    # abstract it was derived from; leaving that value here would say the
    # abstract was written to the prompt, when the truth is the reverse.
    human["prompt"] = None
    # reindex, not strict selection: `doi`, `author_ids` and `author_count` are
    # metadata no stage reads, and a corpus built before they existed carries
    # none of them — see the same guard in make_prompts.py.
    frames = [human.reindex(columns=OUT_COLUMNS)]

    print("\nExported rows per class:")
    print(f"  {'human':<24} {len(human):>6} rows")

    for label in labels:
        rows = df.copy()
        rows["text"] = [ok.get(_key(i, label), {}).get("abstract") for i in rows["id"]]
        rows["generated_by"] = label
        # Only incomplete ids survive to here when require_all_generators is off.
        rows = rows[rows["text"].notna()]
        frames.append(rows.reindex(columns=OUT_COLUMNS))
        print(f"  {label:<24} {len(rows):>6} rows")

    # Pin the optional columns after the concat: the AI frames inherit whatever
    # the human frame had, and an all-null column here would otherwise reach
    # parquet as Arrow type `null` and refuse to be read beside a corpus that
    # has values.
    out = coerce_dtypes(pd.concat(frames, ignore_index=True), OPTIONAL_NUMERIC_COLUMNS)
    # Same reasoning for `prompt`, which coerce_dtypes cannot help with because
    # it is text: null on every human row, and null on ALL rows if a run
    # exported nothing but human rows. Naming the dtype makes it Arrow `string`
    # either way, instead of `null` in the degenerate case.
    out["prompt"] = out["prompt"].astype("string")

    # uid last, once every row knows its generated_by. Assigned here rather
    # than in the clean stage so the row key exists as soon as the row does.
    out["uid"] = out["id"].astype(str) + ID_SEP + out["generated_by"].astype(str)

    duplicated = out["uid"].duplicated()
    if duplicated.any():
        # Two rows with one uid means a paper appeared twice under the same
        # label — a duplicate that survived the filter stage, not a pair.
        raise SystemExit(
            f"{int(duplicated.sum())} duplicate uid(s), e.g. "
            f"{out.loc[duplicated, 'uid'].head(3).tolist()}. Each (paper, "
            "generator) must be one row; check the filter stage's dedupe."
        )

    return out.reindex(columns=OUT_COLUMNS)


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
        help="retry calls previously recorded as failures",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="filename suffix. Defaults to the single generator's label with "
        "the ':' replaced (gpt54mini_0807A), or 'multi' when several are "
        "configured.",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(args.cfg)
    prompts = load_prompt_config(cfg)
    generators = cfg["generate"]["generators"]
    # ':' is legal in a POSIX filename but confuses shell completion, scp-style
    # host:path parsing and Windows entirely, so the label's separator becomes
    # '_' in the filename only. The label itself keeps the colon.
    variant = args.variant or (
        generated_by_label(generators[0]["model"], cfg).replace(":", "_")
        if len(generators) == 1
        else "multi"
    )
    print(f"Prompt config: {prompt_config_path(cfg).relative_to(BASE_DIR)}")

    df = load_parquet(prompts_path(cfg))
    shard = generate(df, cfg, prompts, args.retry_failed)

    final = export(df, shard, cfg)

    save_parquet(final, final_path(cfg, variant))

    print(f"\nTotal {len(final)} rows, {final['id'].nunique()} papers")
    print(final.groupby(["domain", "generated_by"]).size().to_string())


if __name__ == "__main__":
    main()
