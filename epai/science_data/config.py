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
from epai.utils.embed_select import validate_embedders
from epai.utils.text_standardize import WHITESPACE_MODES

BASE_DIR = Path(__file__).parent.parent.parent

# Defaults for the stage CLIs, so the pipeline runs with no arguments. Both are
# absolute, so a stage behaves the same whatever directory it is invoked from.
DEFAULT_CFG_PATH = BASE_DIR / "cfg" / "science_data" / "science_cfg_example.yaml"
DEFAULT_QUERIES_PATH = BASE_DIR / "cfg" / "science_data" / "queries_3domain.yaml"

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
#
# `mode` is in here, rather than in the pipeline config where it used to live,
# for exactly that reason. It decides whether the generator is shown a content
# spec, which is a difference in what the prompt SAYS — and the prompt config's
# id is the corpus's only record of what the prompt said. Split across two
# files, the mode was the one input to a generated row that its label did not
# name: `0811A` under `llm_spec` produced rows labelled gpt56luna:0811A whose
# prompts carried a spec, indistinguishable afterwards from ones that did not.
# Here, the id accounts for it like it accounts for every prompt string.
REQUIRED_PROMPT_KEYS = {
    "id",
    "mode",
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

    # Optional block, and everything in it defaults OFF: these are the
    # transforms that delete words rather than substituting characters.
    clean = cfg.get("clean") or {}
    if not isinstance(clean, dict):
        raise ValueError(f"{path}: clean must be a mapping")
    if not isinstance(clean.get("strip_structured_labels", False), bool):
        raise ValueError(f"{path}: clean.strip_structured_labels must be true or false")

    # Optional block: a config written before this stage existed still loads and
    # gets the stage with its defaults. Absent means enabled — a corpus whose
    # two classes spell their apostrophes differently is the case that should
    # need a deliberate opt-out, not the other way round.
    std = cfg.get("standardize") or {}
    if not isinstance(std, dict):
        raise ValueError(f"{path}: standardize must be a mapping")
    if not isinstance(std.get("enabled", True), bool):
        raise ValueError(f"{path}: standardize.enabled must be true or false")
    if not isinstance(std.get("keep_raw", True), bool):
        raise ValueError(f"{path}: standardize.keep_raw must be true or false")
    ws = std.get("whitespace", "flat")
    if ws not in WHITESPACE_MODES:
        raise ValueError(
            f"{path}: standardize.whitespace must be one of "
            f"{sorted(WHITESPACE_MODES)}, got {ws!r}"
        )

    if not str(cfg.get("prompt_config") or "").strip():
        raise ValueError(
            f"{path}: prompt_config must name a file in cfg/science_data/prompts/ "
            "(without the .yaml), e.g. prompt_config: \"0807A\""
        )
    # Load it here rather than lazily in stage 3: a bad prompt id should fail
    # before the fetch, not four thousand API calls later.
    prompts = load_prompt_config(cfg)

    # `mode` moved into the prompt config (see REQUIRED_PROMPT_KEYS). A copy
    # left behind here is neither ignored nor obeyed — it has to agree, or the
    # run stops. Ignoring it silently would be worse than the split it replaced:
    # a config that plainly reads `mode: llm_spec` would build a title-only
    # corpus, and the person reading the config would have no reason to doubt it.
    declared = cfg["prompt"].get("mode")
    if declared is not None and declared != prompts["mode"]:
        raise ValueError(
            f"{path}: prompt.mode is {declared!r}, but prompt config "
            f"{cfg['prompt_config']!r} declares mode {prompts['mode']!r}.\n"
            "  The mode belongs to the prompt config now — it decides what the "
            "prompt says, so it is part of what the id names. Either delete "
            "prompt.mode from this file, or point prompt_config at the id whose "
            "mode you actually meant."
        )

    generators = cfg["generate"].get("generators") or []
    if not generators:
        raise ValueError(f"{path}: generate.generators must list at least one generator")
    allowed_generator_keys = {"model", "provider", "base_url", "api_key_env"}
    for g in generators:
        if not g.get("model"):
            raise ValueError(f"{path}: generator {g!r} needs a model")
        unknown = g.keys() - allowed_generator_keys
        if unknown:
            raise ValueError(
                f"{path}: generator {g['model']!r} has unknown key(s): {sorted(unknown)}"
            )
        # Resolves the provider shorthand and checks base_url/api_key_env agree;
        # a bad provider name should fail here, not after the fetch.
        try:
            generator_endpoint(g)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
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

    # Optional, and only read by run_full_pipeline.py's embed step — the four
    # corpus stages never touch it. Absent means "whatever add_embeddings.py
    # defaults to", which is how every config written before the embed step
    # existed continues to work.
    embed = cfg.get("embed")
    if embed is not None:
        if not isinstance(embed, dict):
            raise ValueError(f"{path}: embed must be a mapping")
        validate_embedders(embed.get("embedders"), f"{path}: embed.embedders")
        for key in ("batch_size", "limit"):
            value = embed.get(key)
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"{path}: embed.{key} must be null or an integer >= 1")
        if not isinstance(embed.get("recompute", False), bool):
            raise ValueError(f"{path}: embed.recompute must be true or false")

    return cfg


def embed_config(cfg: dict) -> dict:
    """The optional ``embed`` block, or an empty mapping when it is absent."""
    return cfg.get("embed") or {}


def strip_structured_labels_enabled(cfg: dict) -> bool:
    """Whether both cleaners should remove structured-abstract section labels.

    One function, read by the fetch stage for the human abstracts and by the
    generate stage for the machine ones. Two call sites reading the same key
    through the same helper is what stops the two classes being cleaned
    differently, which is the failure the whole symmetry argument is about.
    """
    return bool((cfg.get("clean") or {}).get("strip_structured_labels", False))


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

    Keys are slugged here, once, so that the name a domain has in the config is
    turned into the name it has in the corpus in exactly one place —
    ``Computer Science`` in the YAML, ``computer_science`` in the ``domain``
    column and in every id from here on.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    queries_domains = data.get("queries_domains")
    if not queries_domains:
        raise ValueError(f"{path}: missing 'queries_domains'")

    slugged: dict[str, list[str]] = {}
    for domain, queries in queries_domains.items():
        if not queries:
            raise ValueError(f"{path}: domain {domain!r} has no queries")
        slug = domain_slug(domain)
        if slug in slugged:
            raise ValueError(
                f"{path}: domain {domain!r} slugs to {slug!r}, which another "
                "domain in this file already uses — two domains sharing a name "
                "would merge into one in the corpus. Rename one of them."
            )
        slugged[slug] = queries

    return slugged


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
# is `where: {domain: "biology"}`, not a different path.

SCIENCE_DIR = BASE_DIR / "data" / "science"


def raw_path(cfg: dict) -> Path:
    """Stage 1 output: every record the API returned, admitted or not."""
    return SCIENCE_DIR / f"{cfg['dataset_id']}_raw.parquet"


def human_path(cfg: dict) -> Path:
    """Stage 2 output: the filtered human corpus, all domains."""
    return SCIENCE_DIR / f"{cfg['dataset_id']}_human.parquet"


def prompts_path(cfg: dict) -> Path:
    """Stage 3 output: the human corpus plus a spec and a prompt per paper.

    Carries ``prompt_config`` in the name, as the stage-4 output does. It used
    to be ``<id>_prompts.parquet`` for every arm, which cost more than a
    clobbered intermediate: stage 4 takes ``text`` from the generate shard but
    ``prompt`` from THIS file, and the shard skips calls it has already made. So
    re-exporting an arm after another arm's prompt stage had run produced a
    corpus whose ``text`` came from the right prompts and whose ``prompt``
    column came from the wrong ones — a row that misreports its own input, with
    nothing in the file to show for it.
    """
    return SCIENCE_DIR / f"{cfg['dataset_id']}_prompts_{cfg['prompt_config']}.parquet"


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

    ``mode`` is validated here rather than in :func:`load_pipeline_config`
    because this is the file it belongs to: an unknown mode is a broken prompt
    config, not a broken pipeline config, and stage 3 reads it from the dict
    this function returns.
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

    if prompts["mode"] not in PROMPT_MODES:
        raise ValueError(
            f"{path}: mode is {prompts['mode']!r}, must be one of "
            f"{sorted(PROMPT_MODES)}"
        )

    return prompts


def domain_slug(domain: str) -> str:
    """Config domain name -> corpus domain name: ``Computer Science`` ->
    ``computer_science``.

    The domain is written into the ``id`` (``science:computer_science:W2143…``),
    and an id is split on ``:`` and read by eye in every report the pipeline
    prints, so a domain that carries a space or a colon would be a domain that
    breaks one of those. Lowercase and underscore rather than dropped
    punctuation (:func:`model_slug`'s rule) because a domain is a phrase and
    ``computerscience`` is a worse thing to read than a model name is.

    Idempotent: the slug of a slug is itself, which is what lets the filter
    stage apply it to a raw parquet whether or not the fetch that wrote it
    already had.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", str(domain).lower()).strip("_")
    if not slug:
        raise ValueError(f"domain {domain!r} has no alphanumeric characters")
    return slug


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


# The generate stage sends only ``model`` and ``messages``, so any
# OpenAI-compatible endpoint serves. ``provider`` is a shorthand for the two
# hosts this project uses; a bare ``base_url`` + ``api_key_env`` covers any
# other. Absent both, a generator is plain OpenAI, exactly as before.


def generator_endpoint(g: dict) -> tuple[str | None, str]:
    """``(base_url, api_key_env)`` for one ``generators`` entry.

    ``base_url`` ``None`` means OpenAI's own endpoint — the SDK's default — so
    an entry with neither ``provider`` nor ``base_url`` behaves as it did before
    this routing existed. ``api_key_env`` is always a name, never a key: the
    value is read from the environment when the client is built.
    """
    provider = g.get("provider")
    base_url = g.get("base_url")
    api_key_env = g.get("api_key_env")

    if provider is None:
        if base_url and not api_key_env:
            raise ValueError(
                f"generator {g.get('model')!r}: base_url is set but api_key_env "
                "is not — name the env var holding that endpoint's key"
            )
        return base_url, api_key_env or "OPENAI_API_KEY"
    if provider == "openai":
        return base_url, api_key_env or "OPENAI_API_KEY"
    if provider == "genai4science":
        from epai.utils.env_utils import genai4science_endpoint

        default_base_url, default_key_env = genai4science_endpoint()
        return (base_url or default_base_url, api_key_env or default_key_env)
    raise ValueError(
        f"generator {g.get('model')!r}: unknown provider {provider!r}; use "
        "'openai', 'genai4science', or set base_url + api_key_env directly"
    )


def default_variant(cfg: dict) -> str:
    """Filename suffix for the generate output when ``--variant`` is not given.

    One generator: its label with the ``:`` swapped for ``_`` (``gpt56luna_0811A``).
    Several: ``multi_<prompt_config>``, so the file still names its prompt arm
    even though no single model owns it.
    """
    generators = cfg["generate"]["generators"]
    if len(generators) == 1:
        return generated_by_label(generators[0]["model"], cfg).replace(":", "_")
    return f"multi_{cfg['prompt_config']}"


def shard_path(cfg: dict, stage: str) -> Path:
    """Append-only JSONL checkpoint for a resumable LLM stage.

    Not a shard of the corpus and not a parquet — this is the record of which
    paid API calls have already been made, and deleting it means paying for
    them again. It sits beside the other stage files rather than in a
    subdirectory of its own.
    """
    return BASE_DIR / "data" / "science" / f"{cfg['dataset_id']}_{stage}.jsonl"
