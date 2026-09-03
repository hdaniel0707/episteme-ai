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
without going back to ``<dataset_id>_prompts_<prompt_config>.parquet``. The
value is read from that file rather than from the shard, so the two must
describe the same arm; they do, because both paths carry ``prompt_config``. The
system prompt is not
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
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from epai.science_data.config import (
    BASE_DIR,
    DEFAULT_CFG_PATH,
    default_variant,
    final_path,
    generated_by_label,
    generator_endpoint,
    load_pipeline_config,
    load_prompt_config,
    prompt_config_path,
    prompts_path,
    shard_path,
    strip_structured_labels_enabled,
)
from epai.science_data.llm import Shard, openai_text_call, run_resumable
from epai.science_data.rules import OPTIONAL_NUMERIC_COLUMNS, strip_structured_labels
from epai.utils.parquet_utils import coerce_dtypes, load_parquet, save_parquet

load_dotenv()

# The human abstracts had these stripped by rules.clean_abstract, so leaving
# them in the generated text would create a formatting asymmetry between the
# classes — a tell that has nothing to do with authorship.
_LEADING_LABEL_RE = re.compile(r"^\s*(abstract|summary)\b[\s:.\-—]*", re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"[*_#`]+")
_WS_RE = re.compile(r"\s+")

# Open-weight generators (Qwen, DeepSeek-R1, gpt-oss) can wrap the answer in a
# reasoning block. Left in `text` it is a token the human class can never carry
# — the exact kind of source artifact this corpus exists to avoid. Only paired
# <think>…</think> is removed; a malformed unclosed one falls through to the
# empty-response check in call_with_retry and is recorded as a failed call.
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)

# A shared open-weight endpoint (GenAI4Science / SZTAKI) sheds load by rejecting
# a fraction of calls with a transient 400, and `call_with_retry`'s in-call
# retries — four attempts over ~15s — do not always outlast a busy spell. After
# the first generation sweep, `generate` re-runs just the failures a few more
# times; this is the pause between those passes, long enough for a queued
# endpoint to drain rather than be hit again immediately.
_RETRY_PASS_COOLDOWN = 60

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


def clean_generated(text: str, *, strip_labels: bool = False) -> str:
    """Normalise generated text the same way the human abstracts were.

    :param strip_labels: remove structured-abstract section labels. OFF by
        default, and read from the same ``clean.strip_structured_labels`` key
        the fetch stage reads for the human abstracts -- the two classes are
        cleaned identically or not at all, which is the whole point.

    Markdown removal has no counterpart on the human side because OpenAlex
    abstracts do not carry markdown. It is one-directional for the same reason
    the rest of this is symmetric: it removes a token only one class could ever
    produce. (It is also why it must not be applied to the human text -- the
    pattern would eat the underscores in ``k_B`` and ``H_2``.)
    """
    # Reasoning block first: it can itself contain markdown and section labels,
    # and stripping those out of it before removing the block would leave debris.
    text = _THINK_RE.sub("", text)
    # Markdown FIRST. Both label patterns anchor on a word boundary that `**`
    # defeats, so "**Background:**" would survive the structured-label strip and
    # then lose only its asterisks -- leaving a bare "Background:" in the AI
    # class that the human class never contains. Which is the exact artifact
    # this function exists to prevent.
    text = _MARKDOWN_RE.sub("", text)
    text = _LEADING_LABEL_RE.sub("", text)
    if strip_labels:
        text = strip_structured_labels(text)
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
    shard = Shard(shard_path(cfg, "generate"))
    # Same key oa_fetch reads for the human abstracts. Bound once here rather
    # than inside the lambda so the setting cannot differ between calls.
    strip_labels = strip_structured_labels_enabled(cfg)

    # One client per distinct endpoint. Generators sharing a base_url share a
    # client (the SDK is thread-safe); a run that lists a generator it never
    # actually calls — every row already in the shard — still needs that
    # provider's key, which is the honest failure: don't list what you can't run.
    clients: dict[tuple[str | None, str], OpenAI] = {}
    request_timeout = cfg["generate"].get("request_timeout", 300)

    def client_for(gen: dict) -> OpenAI:
        base_url, api_key_env = generator_endpoint(gen)
        cache_key = (base_url, api_key_env)
        if cache_key not in clients:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise SystemExit(
                    f"{api_key_env} is not set — needed by generator {gen['model']!r}."
                )
            clients[cache_key] = OpenAI(
                api_key=api_key, base_url=base_url, timeout=request_timeout
            )
        return clients[cache_key]

    tasks = []
    for gen in cfg["generate"]["generators"]:
        label = generated_by_label(gen["model"], cfg)
        client = client_for(gen)
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

    def sweep(retry: bool) -> None:
        run_resumable(
            tasks,
            shard,
            max_workers=cfg["generate"].get("max_workers", 8),
            desc="generating abstracts",
            retry_failed=retry,
            payload_of=lambda _k, text: {
                "abstract": clean_generated(text, strip_labels=strip_labels)
            },
        )

    def n_failed() -> int:
        return sum(1 for r in shard.records.values() if not r.get("ok"))

    sweep(retry_failed)

    # Whole-corpus retry sweeps over the failures only, so a transient endpoint
    # error does not need a manual --retry-failed re-run. Stop as soon as a pass
    # converts nothing (the rest are failing deterministically — a content
    # filter, an over-length prompt) or no failures remain. Failure counts only
    # ever decrease across passes, since a retried call either succeeds or stays
    # failed, so `now >= prev` is an exact "made no progress" test.
    retry_passes = cfg["generate"].get("retry_passes", 3)
    prev = n_failed()
    for i in range(retry_passes):
        if prev == 0:
            break
        print(
            f"\n  Auto-retry {i + 1}/{retry_passes}: {prev} failure(s) left; "
            f"pausing {_RETRY_PASS_COOLDOWN}s for the endpoint to recover"
        )
        time.sleep(_RETRY_PASS_COOLDOWN)
        sweep(True)
        now = n_failed()
        if now >= prev:
            print(f"  Auto-retry made no progress ({now} still failing) — stopping.")
            break
        prev = now

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
        "the ':' replaced (gpt54mini_0807A), or 'multi_<prompt_config>' when "
        "several are configured.",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(args.cfg)
    prompts = load_prompt_config(cfg)
    # ':' is legal in a POSIX filename but confuses shell completion, scp-style
    # host:path parsing and Windows entirely, so the label's separator becomes
    # '_' in the filename only. The label itself keeps the colon.
    variant = args.variant or default_variant(cfg)
    print(f"Prompt config: {prompt_config_path(cfg).relative_to(BASE_DIR)}")

    df = load_parquet(prompts_path(cfg))
    shard = generate(df, cfg, prompts, args.retry_failed)

    final = export(df, shard, cfg)

    save_parquet(final, final_path(cfg, variant))

    print(f"\nTotal {len(final)} rows, {final['id'].nunique()} papers")
    print(final.groupby(["domain", "generated_by"]).size().to_string())


if __name__ == "__main__":
    main()
