"""Configuration loading — the two-file control surface.

* ``workflow/config.yaml``  — WHAT to train (workflow, target, classes, model,
  hyperparameters, and the ``runs:`` queue). Parsed here into :class:`Settings`.
* ``workflow/.env.workflow`` — HOW this machine runs it (parallelism, GPU batch
  size, force flags). Read via the ``get_*`` env accessors, never via Settings.

Secrets / API keys are read from the repo-root ``.env`` (loaded into os.environ)
via :func:`secret`. The root ``.env`` is never modified.

Per-run overrides (from ``config.yaml``'s ``runs:`` list) are applied as plain
dicts through :func:`load_settings` — no environment mutation anywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from workflow.src.iwf import paths

_LOADED = False

# ---- workflow topology vocabulary — the ONE table every module derives from ----
# task:     what the head predicts (clf = impact class, reg = impact percentile)
# features: model input — z (latent only) | meta (z + curated scalars)
#           | all_meta (z + ALL numeric columns, intentional leakage upper bound)
#           | text (raw title+abstract into a fine-tuned transformer)
#           | rag (LLM-as-judge: RAG store + evolving notes as the model)
# phases:   ordered phase ids run by run_pipeline (see run_pipeline.PHASE_NAMES)
WORKFLOWS: dict[str, dict] = {
    "emb_clf":          {"task": "clf", "features": "z",        "phases": [0, 1, 2, 3, 4, 5, 6, 7]},
    "emb_reg":          {"task": "reg", "features": "z",        "phases": [0, 1, 2, 3, 4, 6, 7]},
    "emb_meta_clf":     {"task": "clf", "features": "meta",     "phases": [0, 1, 2, 3, 4, 5, 6, 7]},
    "emb_meta_reg":     {"task": "reg", "features": "meta",     "phases": [0, 1, 2, 3, 4, 6, 7]},
    "emb_all_meta_clf": {"task": "clf", "features": "all_meta", "phases": [0, 1, 2, 3, 4, 5, 6, 7]},
    "emb_all_meta_reg": {"task": "reg", "features": "all_meta", "phases": [0, 1, 2, 3, 4, 6, 7]},
    "tfm_clf":          {"task": "clf", "features": "text",     "phases": [0, 1, 5, 6, 7]},
    "tfm_reg":          {"task": "reg", "features": "text",     "phases": [0, 1, 6, 7]},
    "llm_clf":          {"task": "clf", "features": "rag",      "phases": [0, 1, 2, 5, 6, 7]},
    "llm_reg":          {"task": "reg", "features": "rag",      "phases": [0, 1, 2, 6, 7]},
}
WORKFLOW_TYPES = tuple(WORKFLOWS)
VALID_WORKFLOW_TYPES = (*WORKFLOW_TYPES, "all")
TEXT_TRANSFORMER_ARCHS = ("specter2_ft", "scibert_ft", "scibert_lora", "specter2_lora")

# Columns that build the impact percentile and must never be features (leakage guard).
RESERVED_METADATA_FIELDS = {
    "year", "citation_count", "influential_citation_count", "domains",
}

# Number of ordinal impact classes (Q_n .. Q1), clamped to this range.
MIN_CLASSES = 2
MAX_CLASSES = 25


def is_text_workflow(workflow_type: str) -> bool:
    """True for the text-transformer topologies, which skip embed/encode/reduce."""
    return WORKFLOWS.get((workflow_type or "").lower(), {}).get("features") == "text"


def is_judge_workflow(workflow_type: str) -> bool:
    """True for the LLM-as-judge workflows (llm_clf/llm_reg), which use RAG features."""
    return WORKFLOWS.get((workflow_type or "").lower(), {}).get("features") == "rag"


def load_env() -> None:
    """Load the repo-root .env (secrets) and workflow/.env.workflow (machine knobs).

    Root .env is loaded first; .env.workflow is loaded with override=False so it can
    only *add* keys, never clobber a secret already provided by the root .env.
    """
    global _LOADED
    if _LOADED:
        return
    if paths.ROOT_ENV.is_file():
        load_dotenv(paths.ROOT_ENV)
    if paths.WORKFLOW_ENV.is_file():
        load_dotenv(paths.WORKFLOW_ENV, override=False)
    _LOADED = True


# ---------- typed env accessors (machine/ops knobs from .env.workflow only) ----------

def _get(key: str, default: str | None = None) -> str | None:
    load_env()
    v = os.getenv(key)
    if v is None:
        return default
    v = v.strip()
    return v if v != "" else default


def get_str(key: str, default: str) -> str:
    v = _get(key)
    return v if v is not None else default


def get_int(key: str, default: int) -> int:
    v = _get(key)
    return int(v) if v is not None else default


def get_float(key: str, default: float) -> float:
    v = _get(key)
    return float(v) if v is not None else default


def get_bool(key: str, default: bool) -> bool:
    v = _get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def secret(key: str) -> str | None:
    """Read an API key / secret from the environment (repo-root .env). May be None."""
    load_env()
    return os.getenv(key)


# Placeholder secret values treated as "not set" (so a bogus key is never sent).
_PLACEHOLDER_SECRETS = {"", "xxx", "changeme", "none", "null"}


def api_key(*names: str) -> str | None:
    """First real (non-placeholder) API key among names, or None."""
    for name in names:
        v = secret(name)
        if v is None:
            continue
        v = v.strip()
        low = v.lower()
        if low in _PLACEHOLDER_SECRETS or low.startswith("your-") or low.startswith("your_"):
            continue
        return v
    return None


def use_llm_abstracts_for_training() -> bool:
    """Whether LLM-filled abstracts (source='llm') may enter training (config.yaml, default False)."""
    try:
        cfg = load_config_yaml()
    except Exception:
        cfg = {}
    if "use_llm_abstracts_for_training" in cfg and cfg["use_llm_abstracts_for_training"] is not None:
        return _coerce("use_llm_abstracts_for_training", "bool", cfg["use_llm_abstracts_for_training"])
    return get_bool("USE_LLM_ABSTRACTS_FOR_TRAINING", False)


# Raw `source` labels grouped into acquisition channels. The DB stores variants
# ("s2_original_0512_A", "uefiscdi_journal"), so every lookup goes through this one table.
SOURCE_ALIASES = {
    "openalex": ("openalex", "uefiscdi_journal"),
    "s2": ("s2",),
    "crossref": ("crossref",),
    "arxiv": ("arxiv",),
    "pdf": ("pdf", "uefiscdi_pdf", "jcr"),
    "llm": ("llm",),
}


def canonical_source(value) -> str:
    """Acquisition channel of a raw source label ('unknown' when empty)."""
    s = str(value or "").strip().lower()
    if not s or s in ("none", "null"):
        return "unknown"
    for channel, prefixes in SOURCE_ALIASES.items():
        if any(s.startswith(p) or p in s for p in prefixes):
            return channel
    return "other"


def training_sources() -> list[str]:
    """Channels allowed into training (config.yaml, empty/'any' = every source)."""
    try:
        cfg = load_config_yaml()
    except Exception:
        cfg = {}
    raw = cfg.get("training_sources")
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        raw = [raw]
    wanted = [str(v).strip().lower() for v in raw if str(v).strip()]
    if not wanted or "any" in wanted or "all" in wanted:
        return []
    unknown = [w for w in wanted if w not in SOURCE_ALIASES]
    if unknown:
        raise ValueError(f"training_sources has unknown channel(s) {unknown}; "
                         f"valid: {sorted(SOURCE_ALIASES)} or 'any'.")
    return wanted


def analyse_sources() -> list[str]:
    """Channels included in the dataset-analysis plots/CSVs (.env.workflow ANALYSE_SOURCES,
    empty/'any' = every source). Independent of training_sources (config.yaml) — this only
    filters what main_analyse_dataset describes, never what a run trains on."""
    raw = _get("ANALYSE_SOURCES")
    if not raw:
        return []
    wanted = [v.strip().lower() for v in raw.split(",") if v.strip()]
    if not wanted or "any" in wanted or "all" in wanted:
        return []
    unknown = [w for w in wanted if w not in SOURCE_ALIASES]
    if unknown:
        raise ValueError(f"ANALYSE_SOURCES has unknown channel(s) {unknown}; "
                         f"valid: {sorted(SOURCE_ALIASES)} or 'any'.")
    return wanted


def max_abstract_chars() -> int:
    """Longest abstract kept for training in characters (config.yaml, 0 = no limit)."""
    try:
        cfg = load_config_yaml()
    except Exception:
        cfg = {}
    if cfg.get("max_abstract_chars") is not None:
        return max(0, _coerce("max_abstract_chars", "int", cfg["max_abstract_chars"]))
    return max(0, get_int("MAX_ABSTRACT_CHARS", 0))


def target_kind() -> str:
    """Active impact-target kind (config.yaml, default field_year_percentile)."""
    try:
        cfg = load_config_yaml()
    except Exception:
        cfg = {}
    return str(cfg.get("target_kind") or "field_year_percentile").lower()


def target_min_refset() -> int:
    """Min (field x year) cell size before pooling up a level (config.yaml, default 30)."""
    try:
        cfg = load_config_yaml()
    except Exception:
        cfg = {}
    if cfg.get("target_min_refset") is not None:
        return max(1, _coerce("target_min_refset", "int", cfg["target_min_refset"]))
    return 30


def require_known_domain_for_training() -> bool:
    """Whether rows without a domain are dropped from training (config.yaml, default False)."""
    try:
        cfg = load_config_yaml()
    except Exception:
        cfg = {}
    if "require_known_domain_for_training" in cfg and \
            cfg["require_known_domain_for_training"] is not None:
        return _coerce("require_known_domain_for_training", "bool",
                       cfg["require_known_domain_for_training"])
    return get_bool("REQUIRE_KNOWN_DOMAIN_FOR_TRAINING", False)


# ---------- class-count resolution (config.yaml knobs) ----------

def _distribution_to_thresholds(pcts: list[float]) -> list[float]:
    """Cumulative cut points from per-class shares, e.g. [15,35,35,15] -> [0.15,0.50,0.85]."""
    total = sum(pcts)
    if total <= 0:
        raise ValueError("class_distribution percentages must sum to a positive number.")
    cum = 0.0
    thresholds = []
    for p in pcts[:-1]:
        cum += p
        thresholds.append(round(cum / total, 6))
    return thresholds


def resolve_class_thresholds(cfg: dict) -> list[float]:
    """Resolve class-boundary thresholds from the config dict.

    Priority (first one set wins):
      1. class_distribution — per-class percentages (need not be equal), e.g. [15,35,35,15].
      2. num_classes        — integer count -> equal-sized classes, e.g. 9 -> Q9..Q1.
      3. class_thresholds   — manual boundary list (N thresholds -> N+1 classes).
    Class count is always clamped to [MIN_CLASSES, MAX_CLASSES].
    """
    dist = cfg.get("class_distribution")
    if dist:
        dist = [float(x) for x in dist]
        n = len(dist)
        if not (MIN_CLASSES <= n <= MAX_CLASSES):
            raise ValueError(
                f"class_distribution defines {n} classes; must be between "
                f"{MIN_CLASSES} and {MAX_CLASSES}."
            )
        return _distribution_to_thresholds(dist)

    n_classes = int(cfg.get("num_classes") or 0)
    if n_classes:
        if not (MIN_CLASSES <= n_classes <= MAX_CLASSES):
            raise ValueError(
                f"num_classes={n_classes} must be between {MIN_CLASSES} and {MAX_CLASSES}."
            )
        return [round(i / n_classes, 6) for i in range(1, n_classes)]

    thr = cfg.get("class_thresholds")
    return [float(t) for t in thr] if thr else [0.25, 0.50, 0.75]


@dataclass
class Settings:
    """Resolved training parameters (config.yaml top level + per-run overrides)."""

    # identity / corpus
    dataset_id: str = "wf01"
    year_from: int | None = None
    year_to: int | None = None
    max_papers: int | None = None        # cap corpus size for quick tests (None = all)
    max_abstract_chars: int = 0          # drop longer abstracts (0 = keep every length)
    # acquisition channels allowed into training; empty = every collected source
    training_sources: list[str] = field(default_factory=list)

    # shared split (ONE split for ALL stages)
    split_seed: int = 42
    split_ratios: list[float] = field(default_factory=lambda: [0.70, 0.15, 0.15])
    domain_holdout: str = ""  # leave-one-domain-out: domain name to exclude from training

    # fixed-holdout split (data-size sweep): when holdout_val_n/holdout_test_n are both > 0,
    # this REPLACES split_ratios — val/test are a fixed-size holdout picked once and stay
    # identical across every train_n step; train is a seeded, nested prefix of the rest.
    holdout_val_n: int = 0
    holdout_test_n: int = 0
    holdout_seed: int = 0        # 0 = reuse split_seed
    train_n: int = 0             # 0 = use every row remaining after the holdout
    train_select_seed: int = 0   # 0 = reuse split_seed

    # target
    target_kind: str = "field_year_percentile"
    target_min_refset: int = 30             # min (field x year) cell before pooling
    class_scheme: str = "percentile"        # percentile | css | quantile
    class_thresholds: list[float] = field(default_factory=lambda: [0.25, 0.50, 0.75])
    # ordinal class names, ascending impact; empty -> auto Q_n..Q1
    class_names: list[str] = field(default_factory=list)

    # embedders / encoder
    embedders: list[str] = field(default_factory=lambda: ["specter2"])
    embed_for_encoder: str = "specter2"     # which embedding column feeds the encoder
    encoder_kind: str = "identity"          # identity | autoencoder | pca | umap
    dim_grid: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    dim_tolerance: float = 0.01             # probe-metric tolerance (no-reconstruction case)
    dim_recon_target: float = 0.95          # AE: smallest k with reconstruction R^2 >= this
    encoder_max_epochs: int = 600
    encoder_patience: int = 40

    # metadata feature toggles (meta workflows)
    use_metadata: bool = True
    use_novelty: bool = True
    use_feasibility: bool = True
    novelty_kind: str = "spherical"          # spherical | cosine_max (legacy)
    novelty_k: int = 10                      # kNN size for novelty/feasibility geometry
    external_impact: list[str] = field(default_factory=list)
    metadata_fields: list[str] = field(
        default_factory=lambda: ["best_author_h_index", "author_count", "venue"]
    )
    benchmarks: list[str] = field(default_factory=list)  # benchmark names to evaluate on
    # True only for emb_all_meta_* (set automatically from the workflow type): include ALL
    # numeric columns as features, incl. target inputs — the intentional upper-bound test.
    bypass_leakage_guard: bool = False

    # model / training (head)
    model_arch: str = "mlp"                  # mlp | resmlp | ft_transformer
    hidden: list[int] = field(default_factory=lambda: [256, 128])
    activation: str = "gelu"
    dropout: float = 0.5
    lr: float = 5e-4
    weight_decay: float = 0.02
    max_epochs: int = 2000
    patience: int = 60
    batch_size: int = 128
    n_runs: int = 5
    model_seed: int = 42
    # outlier-seed exclusion for seed-averaged plots (accuracy-by-split, confusion matrix,
    # train/test correctness bars) — never triggers below seed_outlier_min_n seeds in a run:
    seed_outlier_enabled: bool = True
    seed_outlier_min_n: int = 5
    seed_outlier_gap_pct: float = 15.0
    # classifier-only loss knobs (ignored by the regressor):
    class_weight: str = "effective"          # none | balanced | effective (Cui 2019)
    effective_beta: float = 0.999
    loss: str = "ce"                         # ce | focal
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05

    # workflow topology selector (one of WORKFLOW_TYPES; "all" runs every one)
    workflow_type: str = "emb_clf"

    # text-transformer workflow settings (tfm_clf / tfm_reg only)
    text_transformer_arch: str = "specter2_ft"
    text_transformer_lr: float = 2e-5
    text_transformer_epochs: int = 5
    text_transformer_warmup_ratio: float = 0.06
    text_transformer_batch_size: int = 16     # micro-batch; effective batch stays 64
    text_transformer_max_len: int = 256
    text_transformer_lora_r: int = 8
    text_transformer_lora_alpha: int = 16
    text_transformer_lora_dropout: float = 0.1
    text_transformer_freeze_layers: int = 0   # freeze first N encoder layers (0 = train all)
    text_transformer_n_runs: int = 2

    # LLM-as-judge workflow (llm_clf / llm_reg): the LLM is called exactly once per record
    # against a fixed rubric — see src/main_train_judge.py. Machine-level LLM connection
    # settings (endpoint, key, context window) stay in .env.workflow — see iwf/llm/client.py.
    judge_max_train_papers: int = 0   # cap how many TRAIN papers build the RAG store and get
                                       # judged for train metrics (0 = use every TRAIN paper)

    def __post_init__(self) -> None:
        if not self.class_names:
            from workflow.src.iwf.targets import default_class_names
            self.class_names = default_class_names(len(self.class_thresholds) + 1)

    def update(self, overrides: dict[str, Any]) -> "Settings":
        """Apply a flat override dict (Settings field names), coercing to field types."""
        types = {f.name: f.type for f in fields(self)}
        for k, v in overrides.items():
            if k in types:
                setattr(self, k, _coerce(k, types[k], v))
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    def copy(self) -> "Settings":
        import copy as _copy
        return _copy.deepcopy(self)

    def validate(self) -> "Settings":
        """Fail fast on an inconsistent configuration; raises one ValueError listing
        every violation instead of crashing deep inside a phase."""
        from workflow.src.iwf.targets import ALL_TARGET_KINDS

        errors: list[str] = []

        wt = (self.workflow_type or "").lower()
        if wt not in VALID_WORKFLOW_TYPES:
            errors.append(
                f"workflow={self.workflow_type!r} is invalid; choose one of "
                f"{list(VALID_WORKFLOW_TYPES)}."
            )

        if wt in ("tfm_clf", "tfm_reg", "all") and \
                (self.text_transformer_arch or "").lower() not in TEXT_TRANSFORMER_ARCHS:
            errors.append(
                f"text_transformer_arch={self.text_transformer_arch!r} is invalid; choose one "
                f"of {list(TEXT_TRANSFORMER_ARCHS)}."
            )

        if (self.target_kind or "").lower() not in ALL_TARGET_KINDS:
            errors.append(
                f"target_kind={self.target_kind!r} is invalid; choose one of {ALL_TARGET_KINDS}."
            )

        # Embedder names, checked here so a typo fails before fetch/prepare rather than deep
        # in phase 2. Imported lazily: embedders pulls in torch/epai, config does not.
        try:
            from workflow.src.iwf.embedding.embedders import KNOWN_EMBEDDERS
        except Exception:                       # torch/epai missing -> skip, phase 2 will say
            KNOWN_EMBEDDERS = None
        if KNOWN_EMBEDDERS:
            for name in [*(self.embedders or []), self.embed_for_encoder]:
                if name and name not in KNOWN_EMBEDDERS:
                    errors.append(
                        f"embedder={name!r} is unknown; choose from "
                        f"{sorted(KNOWN_EMBEDDERS)}."
                    )

        n_classes = len(self.class_thresholds) + 1
        if not (MIN_CLASSES <= n_classes <= MAX_CLASSES):
            errors.append(
                f"Resolved class count is {n_classes} (from class_distribution / num_classes / "
                f"class_thresholds); must be between {MIN_CLASSES} and {MAX_CLASSES}."
            )
        if self.class_names and len(self.class_names) != n_classes:
            errors.append(
                f"class_names has {len(self.class_names)} entries but {n_classes} classes are "
                "configured; fix class_names or the class-count knob."
            )

        try:
            from workflow.src.iwf.models.architectures import available_archs
            valid_archs = [*available_archs(), "logreg", "linreg"]
            arch = (self.model_arch or "").lower()
            if arch not in valid_archs:
                errors.append(f"model_arch={self.model_arch!r} is not registered; "
                              f"choose from {valid_archs}.")
            elif arch == "logreg" and WORKFLOWS.get(wt, {}).get("task") == "reg":
                errors.append("model_arch='logreg' (scikit-learn baseline) only supports "
                              "classification workflows, not regression.")
            elif arch == "linreg" and WORKFLOWS.get(wt, {}).get("task") == "clf":
                errors.append("model_arch='linreg' (scikit-learn baseline) only supports "
                              "regression workflows, not classification.")
        except Exception as e:  # importing torch must never make validation unusable
            print(f"[config] (note) could not verify model_arch against the registry: {e}")

        if self.n_runs < 1:
            errors.append(f"n_runs must be >= 1 (got {self.n_runs}).")

        holdout_active = self.holdout_val_n > 0 or self.holdout_test_n > 0
        if holdout_active and (self.holdout_val_n <= 0 or self.holdout_test_n <= 0):
            errors.append(
                "holdout_val_n and holdout_test_n must both be > 0 to enable the "
                f"fixed-holdout split (got holdout_val_n={self.holdout_val_n}, "
                f"holdout_test_n={self.holdout_test_n}).")
        if not holdout_active and self.train_n:
            errors.append(
                "train_n is only meaningful with the fixed-holdout split; set "
                "holdout_val_n/holdout_test_n > 0, or leave train_n at 0.")
        if holdout_active and self.max_papers is not None and \
                self.holdout_val_n + self.holdout_test_n > self.max_papers:
            errors.append(
                f"holdout_val_n+holdout_test_n ({self.holdout_val_n + self.holdout_test_n}) "
                f"exceeds max_papers ({self.max_papers}).")

        if self.seed_outlier_min_n < 2:
            errors.append(f"seed_outlier_min_n must be >= 2 (got {self.seed_outlier_min_n}).")
        if self.seed_outlier_gap_pct <= 0:
            errors.append(
                f"seed_outlier_gap_pct must be > 0 (got {self.seed_outlier_gap_pct}).")

        if (self.class_scheme or "").lower() not in ("percentile", "css", "quantile"):
            errors.append(f"class_scheme={self.class_scheme!r} is invalid; choose "
                          "percentile, css, or quantile.")

        if (self.novelty_kind or "").lower() not in ("spherical", "cosine_max"):
            errors.append(f"novelty_kind={self.novelty_kind!r} is invalid; choose "
                          "spherical or cosine_max.")

        try:
            from workflow.src.iwf.features.external_impact import known_external_models
            unknown = [m for m in self.external_impact
                       if m.lower() not in known_external_models()]
            if unknown:
                errors.append(f"external_impact contains unknown model(s) {unknown}; "
                              f"choose from {known_external_models()}.")
        except Exception:
            pass  # never let an import problem make validation unusable

        if self.use_metadata and not self.bypass_leakage_guard:
            offending = sorted(RESERVED_METADATA_FIELDS.intersection(
                f.strip().lower() for f in self.metadata_fields
            ))
            if offending:
                errors.append(
                    "use_metadata is true but metadata_fields contains reserved target-input "
                    f"column(s) {offending}; these build the impact target and must never be "
                    "features."
                )

        if errors:
            raise ValueError("Invalid workflow configuration:\n  - " + "\n  - ".join(errors))
        return self


# ---------- yaml loading + coercion ----------

def _coerce(name: str, ftype, value):
    """Best-effort coercion of a yaml/override value to the Settings field type."""
    if value is None:
        return None
    t = str(ftype)
    if "list" in t:
        if isinstance(value, str):                        # "256,128" -> [256, 128]
            value = [tok.strip() for tok in value.split(",") if tok.strip()]
        if "int" in t:
            return [int(v) for v in value]
        if "float" in t:
            return [float(v) for v in value]
        return [str(v) for v in value]
    if "bool" in t:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if "int" in t:
        return int(value)
    if "float" in t:
        return float(value)                               # also handles yaml "2e-5" strings
    return str(value)


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config_yaml(config_path: str | Path | None = None) -> dict:
    """Read config.yaml and return its top-level keys, lower-cased (``runs`` included)."""
    path = Path(config_path) if config_path else paths.WORKFLOW_CONFIG
    if not path.is_file():
        raise FileNotFoundError(
            f"Training config not found: {path}\n"
            "workflow/config.yaml is the single training-configuration file — restore it."
        )
    cfg = load_yaml(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level.")
    return {str(k).lower(): v for k, v in cfg.items()}


# Keys that are handled specially and are not Settings fields.
_META_KEYS = {"runs", "name", "num_classes", "class_distribution", "workflow", "embedder"}


def _apply_aliases(cfg: dict) -> dict:
    """Resolve convenience aliases in place: workflow -> workflow_type,
    embedder -> embedders + embed_for_encoder, class-count knobs -> class_thresholds."""
    cfg = dict(cfg)
    if "workflow" in cfg and cfg["workflow"]:
        cfg["workflow_type"] = cfg["workflow"]
    if cfg.get("embedder"):
        emb = str(cfg["embedder"])
        cfg.setdefault("embedders", [emb])
        cfg.setdefault("embed_for_encoder", emb)
    if cfg.get("class_distribution") or cfg.get("num_classes"):
        cfg["class_thresholds"] = resolve_class_thresholds(cfg)
        # a new class count invalidates explicitly-set names unless given alongside
        cfg.setdefault("class_names", [])
    return cfg


def load_settings(config_path: str | Path | None = None,
                  overrides: dict | None = None) -> Settings:
    """Build Settings from config.yaml, then apply an optional per-run override dict.

    ``overrides`` keys are Settings field names (or the aliases ``workflow`` /
    ``embedder`` / ``num_classes`` / ``class_distribution``), exactly as written
    in a ``runs:`` entry of config.yaml.
    """
    load_env()
    cfg = load_config_yaml(config_path)
    if overrides:
        cfg.update({str(k).lower(): v for k, v in overrides.items()})
    cfg = _apply_aliases(cfg)
    cfg["class_thresholds"] = resolve_class_thresholds(cfg)
    flat = {k: v for k, v in cfg.items() if k not in _META_KEYS}
    names = flat.pop("class_names", None) or []
    s = Settings()
    s.update(flat)
    if names:
        s.class_names = [str(n) for n in names]
    else:  # auto-derive so the names always match the resolved class count
        from workflow.src.iwf.targets import default_class_names
        s.class_names = default_class_names(len(s.class_thresholds) + 1)
    return s
