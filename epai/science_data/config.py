"""Loading and validation for the single science_data pipeline config.

Every stage reads the same YAML file (see
``cfg/science_data/science_cfg_example.yaml`` for the annotated template). Two
things live outside it, each so it can be swapped independently:

  - the domain/query lists, in ``queries_*.yaml`` — the same split the
    citation-impact pipeline uses;
  - **the prompts**, in ``cfg/science_data/prompts/<id>.yaml``. Those are the
    experiment's independent variable, and their id is stamped into every
    generated row (see :func:`generated_by_label`), so a corpus names the exact
    prompts that produced it.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from epai.science_data.rules import FilterConfig

BASE_DIR = Path(__file__).parent.parent.parent

# Defaults for the stage CLIs, so the pipeline runs with no arguments. Both are
# absolute, so a stage behaves the same whatever directory it is invoked from.
DEFAULT_CFG_PATH = BASE_DIR / "cfg" / "science_data" / "science_cfg_example.yaml"
DEFAULT_QUERIES_PATH = BASE_DIR / "cfg" / "science_data" / "queries_example_debug.yaml"

PROMPTS_DIR = BASE_DIR / "cfg" / "science_data" / "prompts"

REQUIRED_TOP_KEYS = {
    "dataset_id",
    "prompt_config",
    "fetch",
    "filter",
    "prompt",
    "generate",
}
TRIM_MODES = {"head", "sample"}
SEARCH_MODES = {"relevance", "bulk"}
PROMPT_MODES = {"llm_spec", "title_only"}

# Every prompt the pipeline sends must come from the prompt config, so that the
# id stamped into `generated_by` accounts for all of them. A missing key is a
# hard error rather than a fallback to something hardcoded — a silent default
# would be a prompt the id does not describe.
REQUIRED_PROMPT_KEYS = {
    "id",
    "spec_system_prompt",
    "generation_prompt_template",
    "spec_block",
    "results_instruction",
    "generation_system_prompt",
}


def load_pipeline_config(path: Path) -> dict:
    """Load and validate the pipeline config; returns the raw dict.

    Validation is strict and up-front. Every stage after fetch costs either API
    calls or an hour of wall-clock, so a typo in a key name should fail on line
    one rather than after 4000 requests.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: config must be a mapping")

    missing = REQUIRED_TOP_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"{path}: missing top-level keys: {sorted(missing)}")

    fetch = cfg["fetch"]
    if fetch.get("target_per_domain", 0) < 1:
        raise ValueError(f"{path}: fetch.target_per_domain must be >= 1")
    if fetch.get("overfetch", 1.0) < 1.0:
        raise ValueError(
            f"{path}: fetch.overfetch must be >= 1.0 (got {fetch.get('overfetch')}); "
            "fetching fewer records than the target cannot reach the target"
        )
    if fetch.get("search_mode", "relevance") not in SEARCH_MODES:
        raise ValueError(f"{path}: fetch.search_mode must be one of {sorted(SEARCH_MODES)}")

    filt = cfg["filter"]
    if filt.get("trim_mode", "head") not in TRIM_MODES:
        raise ValueError(f"{path}: filter.trim_mode must be one of {sorted(TRIM_MODES)}")
    # Raises on unknown keys, so a mistyped threshold cannot silently take the
    # default and desynchronise fetch from filter.
    FilterConfig.from_dict(
        {k: v for k, v in filt.items() if k not in {"trim_mode", "seed"}}
    )

    if cfg["prompt"].get("mode", "llm_spec") not in PROMPT_MODES:
        raise ValueError(f"{path}: prompt.mode must be one of {sorted(PROMPT_MODES)}")

    if not str(cfg.get("prompt_config") or "").strip():
        raise ValueError(
            f"{path}: prompt_config must name a file in cfg/science_data/prompts/ "
            "(without the .yaml), e.g. prompt_config: \"0807A\""
        )
    # Load it here rather than lazily in stage 3: a bad prompt id should fail
    # before the fetch, not four thousand API calls later.
    load_prompt_config(cfg)

    generators = cfg["generate"].get("generators") or []
    if not generators:
        raise ValueError(f"{path}: generate.generators must list at least one generator")
    for g in generators:
        if not g.get("model"):
            raise ValueError(f"{path}: generator {g!r} needs a model")
    # `generated_by` is derived as <model_slug>:<prompt_config>, so two
    # generators are distinguishable only if their models are — and two models
    # whose slugs collide ("gpt-5.4-mini" vs "gpt_5_4_mini") would silently
    # write rows under one label.
    models = [g["model"] for g in generators]
    slugs = [model_slug(m) for m in models]
    if len(set(slugs)) != len(slugs):
        raise ValueError(
            f"{path}: generator models must have distinct slugs, got "
            f"{dict(zip(models, slugs))}. Labels are derived from the slug, so "
            "these generators would share a generated_by value."
        )
    if "human" in slugs:
        raise ValueError(f"{path}: 'human' is reserved for the original abstracts")

    return cfg


def filter_config(cfg: dict) -> FilterConfig:
    """Build the row-local rule thresholds from the pipeline config."""
    return FilterConfig.from_dict(
        {k: v for k, v in cfg["filter"].items() if k not in {"trim_mode", "seed"}}
    )


def load_queries(path: Path) -> dict[str, list[str]]:
    """Load the ``queries_domains`` mapping (same shape as the fetch pipeline).

    Domain order is preserved, and it matters: cross-domain duplicates are
    resolved first-domain-wins, so a stable order makes the assignment
    reproducible.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    queries_domains = data.get("queries_domains")
    if not queries_domains:
        raise ValueError(f"{path}: missing 'queries_domains'")
    for domain, queries in queries_domains.items():
        if not queries:
            raise ValueError(f"{path}: domain {domain!r} has no queries")

    return queries_domains


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
#
# EVERY stage reads and writes inside data/science/. Nothing here writes to
# data/parquet/, which holds finished corpora that other pipelines load — the
# final stage's output is promoted there by hand, once, when a run is judged
# good. That keeps a half-finished or abandoned run from ever being picked up
# by an ai_detection config that globs data/parquet/.
#
# Every stage writes ONE parquet holding every domain, with `domain` as a
# column. There are no per-domain files and no shard directory: a corpus is a
# slice of a file, not a file — which is what `epai.utils.corpus_utils` has
# expected since combine_domains.py folded essay/reuter/wp together, and what
# the ghostbuster parquets already look like. Selecting one domain downstream
# is `where: {domain: "Biology"}`, not a different path.

SCIENCE_DIR = BASE_DIR / "data" / "science"


def raw_path(cfg: dict) -> Path:
    """Stage 1 output: every record the API returned, admitted or not."""
    return SCIENCE_DIR / f"{cfg['dataset_id']}_raw.parquet"


def human_path(cfg: dict) -> Path:
    """Stage 2 output: the filtered human corpus, all domains."""
    return SCIENCE_DIR / f"{cfg['dataset_id']}_human.parquet"


def prompts_path(cfg: dict) -> Path:
    """Stage 3 output: the human corpus plus a spec and a prompt per paper."""
    return SCIENCE_DIR / f"{cfg['dataset_id']}_prompts.parquet"


def final_path(cfg: dict, variant: str) -> Path:
    """Stage 4 output: human + generated rows, all domains.

    The last stage, and the only artifact meant to leave data/science/. Move it
    to data/parquet/ by hand when the run is good; nothing in this pipeline
    does that for you.
    """
    return SCIENCE_DIR / f"{cfg['dataset_id']}_{variant}.parquet"


# ---------------------------------------------------------------------------
# Prompts and labels
# ---------------------------------------------------------------------------


def prompt_config_path(cfg: dict) -> Path:
    return PROMPTS_DIR / f"{cfg['prompt_config']}.yaml"


def load_prompt_config(cfg: dict) -> dict:
    """Load the prompt file named by ``prompt_config``.

    Its ``id`` must match the filename. They are two spellings of the same
    thing — the filename is how it is found, the ``id`` is what gets stamped
    into ``generated_by`` — and a copied-but-not-renamed file would otherwise
    label a whole corpus with the id of the config it was copied from, which is
    undetectable afterwards.
    """
    path = prompt_config_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f"prompt_config {cfg['prompt_config']!r} not found at {path}.\n"
            f"  Available: {sorted(p.stem for p in PROMPTS_DIR.glob('*.yaml')) or 'none'}"
        )

    with open(path) as f:
        prompts = yaml.safe_load(f)

    if not isinstance(prompts, dict):
        raise ValueError(f"{path}: prompt config must be a mapping")

    missing = REQUIRED_PROMPT_KEYS - prompts.keys()
    if missing:
        raise ValueError(f"{path}: missing prompt key(s): {sorted(missing)}")

    if prompts["id"] != cfg["prompt_config"]:
        raise ValueError(
            f"{path}: id is {prompts['id']!r} but the file is named "
            f"{cfg['prompt_config']!r}.yaml. These must match — the id is what "
            "gets written into every generated row, so a mismatch mislabels the "
            "whole corpus. If this file was copied, change its id."
        )

    return prompts


def model_slug(model: str) -> str:
    """Compact a model name for use in a label: ``gpt-5.4-mini`` -> ``gpt54mini``.

    Punctuation is dropped rather than mapped to underscores so the result
    stays a single token, and the label's one ``:`` separator keeps its meaning
    as the model/prompt-config boundary.
    """
    return re.sub(r"[^a-z0-9]", "", model.lower())


def generated_by_label(model: str, cfg: dict) -> str:
    """The ``generated_by`` value for a generator: ``gpt54mini:0807A``.

    Derived, never configured. The point of the label is to say which model and
    which prompts produced a row, so letting it be typed by hand would make it
    the one field in the corpus that can disagree with what actually happened.
    """
    return f"{model_slug(model)}:{cfg['prompt_config']}"


def shard_path(cfg: dict, stage: str) -> Path:
    """Append-only JSONL checkpoint for a resumable LLM stage.

    Not a shard of the corpus and not a parquet — this is the record of which
    paid API calls have already been made, and deleting it means paying for
    them again. It sits beside the other stage files rather than in a
    subdirectory of its own.
    """
    return BASE_DIR / "data" / "science" / f"{cfg['dataset_id']}_{stage}.jsonl"
