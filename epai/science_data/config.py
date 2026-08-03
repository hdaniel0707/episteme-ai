"""Loading and validation for the single science_data pipeline config.

All five stages read the same YAML file (see
``cfg/science_data/science_cfg_example.yaml`` for the annotated template). The
queries themselves live in a separate file so the large domain/query lists can
be swapped independently of the thresholds — the same split the citation-impact
pipeline uses.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from epai.science_data.rules import FilterConfig

BASE_DIR = Path(__file__).parent.parent.parent

# Defaults for the stage CLIs, so the pipeline runs with no arguments. Both are
# absolute, so a stage behaves the same whatever directory it is invoked from.
DEFAULT_CFG_PATH = BASE_DIR / "cfg" / "science_data" / "science_cfg_example.yaml"
DEFAULT_QUERIES_PATH = BASE_DIR / "cfg" / "science_data" / "queries_example_debug.yaml"

REQUIRED_TOP_KEYS = {"dataset_id", "fetch", "filter", "prompt", "generate"}
TRIM_MODES = {"head", "sample"}
SEARCH_MODES = {"relevance", "bulk"}
PROMPT_MODES = {"llm_spec", "title_only"}


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

    generators = cfg["generate"].get("generators") or []
    if not generators:
        raise ValueError(f"{path}: generate.generators must list at least one generator")
    labels = [g.get("label") for g in generators]
    for g in generators:
        if not g.get("model") or not g.get("prompt_name") or not g.get("label"):
            raise ValueError(f"{path}: generator {g!r} needs model, prompt_name and label")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{path}: generator labels must be unique: {labels}")
    if "human" in labels:
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


def raw_path(cfg: dict) -> Path:
    return BASE_DIR / "data" / "science" / f"{cfg['dataset_id']}_raw.parquet"


def human_path(cfg: dict, domain: str | None = None) -> Path:
    suffix = "all" if domain is None else _slug(domain)
    return BASE_DIR / "data" / "parquet" / f"{cfg['dataset_id']}_{suffix}.parquet"


def prompts_path(cfg: dict) -> Path:
    return BASE_DIR / "data" / "science" / f"{cfg['dataset_id']}_prompts.parquet"


def final_path(cfg: dict, variant: str, domain: str | None = None) -> Path:
    suffix = "all" if domain is None else _slug(domain)
    return (
        BASE_DIR
        / "data"
        / "parquet"
        / f"{cfg['dataset_id']}_{suffix}_{variant}.parquet"
    )


def shard_path(cfg: dict, stage: str) -> Path:
    """Append-only JSONL checkpoint for a resumable LLM stage."""
    return BASE_DIR / "data" / "science" / "shards" / f"{cfg['dataset_id']}_{stage}.jsonl"


def _slug(domain: str) -> str:
    return domain.strip().lower().replace(" ", "_")
