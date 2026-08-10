"""Loading, validation and paths for the single ghostbuster pipeline config.

Every stage of ``run_full_pipeline.py`` reads the same YAML file (see
``cfg/ghostbuster/gb_cfg_example.yaml`` for the annotated template), for the
same reason science_data does: the model name decides every filename downstream,
so it has to be stated once.

**Nothing here writes, copies or generates the shared inputs.** ``human/``,
``prompts/`` and reuter's ``gpt/<author>/headlines/`` are seeded into
``external/ghostbuster/data/`` by hand, once, and reused by every model variant
— that is what makes two variants comparable rather than merely similar. This
module only checks that they are there (:func:`missing_seeds`) and refuses to
let a config ask the fork to regenerate them.

The variant is derived from the generator model, not configured:
``gpt-5.6-luna`` -> ``gpt56luna``, giving ``data/parquet/essay_gpt56luna.parquet``
and ``data/parquet/ghostbuster_gpt56luna.parquet``, which is the naming the
existing ``*_gpt54mini`` files already use.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# The one thing worth sharing with the science pipeline: how a model name
# becomes a filename token. Two pipelines spelling `gpt-5.4-mini` differently
# would put the same generator under two names in data/parquet/.
from epai.science_data.config import model_slug
from epai.utils.embed_select import validate_embedders

BASE_DIR = Path(__file__).parent.parent.parent

DEFAULT_CFG_PATH = BASE_DIR / "cfg" / "ghostbuster" / "gb_cfg_example.yaml"

# --- the two submodules, each with its own uv environment -------------------
GHOSTBUSTER_PROJECT = BASE_DIR / "external" / "ghostbuster"
GB_DATA_DIR = GHOSTBUSTER_PROJECT / "data"
# Bare filenames: both are run with cwd set to the fork, because generate.py
# does `import utils` and reads/writes "data/<domain>/..." relative to the
# process's working directory. Nothing about that is configurable there.
GENERATE_SCRIPT = "generate.py"
FIX_EMPTY_SCRIPT = "fix_empty_generations.py"

BINOCULARS_PROJECT = BASE_DIR / "external" / "Binoculars"
BINOCULARS_SCRIPT = BINOCULARS_PROJECT / "score_parquets.py"

# --- in-repo stages --------------------------------------------------------
PREPARE_DIR = "epai/ai_detection/prepare"
COMBINE_SCRIPT = f"{PREPARE_DIR}/combine_domains.py"
PARQUET_DIR = BASE_DIR / "data" / "parquet"

# Pipeline order, and the order every report prints. A config's `domains` list
# is filtered through this, not used as given, so two configs that name the same
# domains in a different order still produce byte-identical commands.
DOMAINS = ("essay", "wp", "reuter")

PREPARE_SCRIPTS = {
    "essay": f"{PREPARE_DIR}/prepare_essay.py",
    "wp": f"{PREPARE_DIR}/prepare_wp.py",
    "reuter": f"{PREPARE_DIR}/prepare_reuter.py",
}

# Mirrors generate.py's PROMPT_TYPE_INDICES plus its one claude type. Kept as a
# literal rather than imported: the fork lives in another environment and cannot
# be imported from here, so this list is a copy that has to be checked against
# it when the fork gains a generator.
GPT_TYPES = (
    "gpt",
    "gpt_prompt1",
    "gpt_prompt2",
    "gpt_writing",
    "gpt_semantic",
    "gpt_plain",
)
ALL_TYPES = GPT_TYPES + ("claude",)

# Also copies of the fork's constants, and also unimportable for the same
# reason. generate.py loops ``range(1, (limit or 1000) + 1)`` for essay and wp,
# and over (author, 1..20) pairs for reuter; the collect stage's audit needs
# those numbers to say whether a run produced everything it should have.
DEFAULT_DOMAIN_ITEMS = 1000
REUTER_ARTICLES_PER_AUTHOR = 20

# What a domain needs on disk before anything can be generated for it, relative
# to external/ghostbuster/data/<domain>/. Reuter has no prompts/ directory: its
# prompts are headlines, written under gpt/<author>/headlines/ whatever variant
# is being generated (see generate.py's --reuter_prompts block).
SEED_SUBDIRS = {
    "essay": ("human", "prompts"),
    "wp": ("human", "prompts"),
    "reuter": ("human", "gpt"),
}

# combine_domains.py's --drop-incomplete choices, copied so a bad value fails
# here instead of after the corpus has been prepared.
INCOMPLETE_MODES = ("emb", "optional", "none")

# The suffix this pipeline gives the scored corpus. score_parquets.py writes in
# place unless -o is given, so the pipeline passes -o explicitly and this is the
# only place the name is decided.
SCORED_SUFFIX = "_binox0"

REQUIRED_TOP_KEYS = {"dataset_id", "generate", "prepare", "combine", "score", "embed"}

# Directories under a domain that hold the seeded inputs rather than a
# generator's output. prepare_*.py skip them too; the audit in the collect stage
# uses this to know what it is allowed to count.
NON_GENERATOR_DIRS = {"prompts", "logprobs", "headlines"}


def load_pipeline_config(path: Path) -> dict:
    """Load and validate the pipeline config; returns the raw dict.

    Strict and up-front, for the same reason as science_data: the generate stage
    costs API calls and hours, so a mistyped key belongs in an error on line one
    rather than in a corpus nobody notices is wrong.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: config must be a mapping")

    missing = REQUIRED_TOP_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"{path}: missing top-level keys: {sorted(missing)}")

    if not str(cfg.get("dataset_id") or "").strip():
        raise ValueError(f"{path}: dataset_id must be a non-empty string")

    gen = cfg["generate"]

    types = list(gen.get("types") or [])
    if not types:
        raise ValueError(
            f"{path}: generate.types must list at least one generator, "
            f"e.g. [\"gpt_plain\"]. Choose from {list(ALL_TYPES)}"
        )
    # human/prompts are named separately from the other unknowns because asking
    # for them is a specific, plausible mistake: they are directories in the
    # same tree, generate.py has flags for them, and running those flags
    # overwrites the seeded inputs that every variant of this corpus shares.
    seeded = [t for t in types if t in ("human", "prompts")]
    if seeded:
        raise ValueError(
            f"{path}: generate.types must not contain {seeded}. Those are the "
            "seeded inputs, shared by every model variant and placed by hand — "
            "regenerating them would change the human text and the prompts this "
            "corpus is supposed to hold fixed."
        )
    unknown = [t for t in types if t not in ALL_TYPES]
    if unknown:
        raise ValueError(
            f"{path}: unknown generate.types {unknown}. Choose from {list(ALL_TYPES)}"
        )
    if len(set(types)) != len(types):
        raise ValueError(f"{path}: generate.types contains duplicates: {types}")

    domains = list(gen.get("domains") or [])
    if not domains:
        raise ValueError(
            f"{path}: generate.domains must list at least one domain, "
            f"choose from {list(DOMAINS)}"
        )
    unknown = [d for d in domains if d not in DOMAINS]
    if unknown:
        raise ValueError(
            f"{path}: unknown generate.domains {unknown}. Choose from {list(DOMAINS)}"
        )
    if len(set(domains)) != len(domains):
        raise ValueError(f"{path}: generate.domains contains duplicates: {domains}")

    # The model is what the variant, and therefore every path, is derived from,
    # so a run that generates gpt_* rows without naming a model cannot be given
    # a filename at all.
    if any(t in GPT_TYPES for t in types) and not str(gen.get("gpt_model") or "").strip():
        raise ValueError(
            f"{path}: generate.gpt_model is required — generate.types includes "
            f"{[t for t in types if t in GPT_TYPES]}"
        )
    if "claude" in types and not str(gen.get("claude_model") or "").strip():
        raise ValueError(
            f"{path}: generate.claude_model is required — generate.types includes 'claude'"
        )

    limit = gen.get("limit")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError(f"{path}: generate.limit must be null or an integer >= 1")
    seed = gen.get("seed", 0)
    if not isinstance(seed, int):
        raise ValueError(f"{path}: generate.seed must be an integer")
    if not isinstance(gen.get("fix_empty", True), bool):
        raise ValueError(f"{path}: generate.fix_empty must be true or false")

    if not isinstance(cfg["prepare"].get("embeddings", False), bool):
        raise ValueError(f"{path}: prepare.embeddings must be true or false")

    comb = cfg["combine"]
    mode = comb.get("drop_incomplete", "emb")
    if mode not in INCOMPLETE_MODES:
        raise ValueError(
            f"{path}: combine.drop_incomplete must be one of {list(INCOMPLETE_MODES)}"
        )
    frac = comb.get("max_drop_frac", 0.05)
    if not isinstance(frac, (int, float)) or not 0 <= frac <= 1:
        raise ValueError(f"{path}: combine.max_drop_frac must be between 0 and 1")
    labels = comb.get("drop_generated_by", [])
    if labels is None or not isinstance(labels, list):
        raise ValueError(
            f"{path}: combine.drop_generated_by must be a list (use [] to keep "
            "every label)"
        )

    # The one cross-stage coupling worth enforcing rather than silently fixing.
    # prepare_*.py always writes an emb_specter2 column; with --no-embedding it
    # is entirely Null, and combine's default mode drops any row with a hole in
    # an emb_* column — which is every row, so the whole corpus would be dropped
    # and the run would die on --max-drop-frac with a message about a column
    # that "was never computed". True, but three stages away from the cause.
    if not cfg["prepare"].get("embeddings", False) and mode != "none":
        raise ValueError(
            f"{path}: prepare.embeddings is false, so combine.drop_incomplete "
            f"must be \"none\" (got {mode!r}). Without embeddings the "
            "emb_specter2 column is all-Null, and any other mode would drop "
            "every row of every domain."
        )

    score = cfg["score"]
    if not str(score.get("text_column") or "").strip():
        raise ValueError(f"{path}: score.text_column must be a non-empty string")
    batch = score.get("batch_size", 1)
    if not isinstance(batch, int) or batch < 1:
        raise ValueError(f"{path}: score.batch_size must be an integer >= 1")
    slimit = score.get("limit")
    if slimit is not None and (not isinstance(slimit, int) or slimit < 1):
        raise ValueError(f"{path}: score.limit must be null or an integer >= 1")
    # Two halves of one detector. Swapping only one makes the scores
    # incomparable with every other file in data/parquet/ while still looking
    # like a Binoculars score.
    if bool(score.get("observer")) != bool(score.get("performer")):
        raise ValueError(
            f"{path}: set both score.observer and score.performer, or neither. "
            "One model of the pair changed is a different detector."
        )

    embed = cfg["embed"]
    if not str(embed.get("text_col") or "").strip():
        raise ValueError(f"{path}: embed.text_col must be a non-empty string")
    validate_embedders(embed.get("embedders"), f"{path}: embed.embedders")
    ebatch = embed.get("batch_size")
    if ebatch is not None and (not isinstance(ebatch, int) or ebatch < 1):
        raise ValueError(f"{path}: embed.batch_size must be null or an integer >= 1")
    elimit = embed.get("limit")
    if elimit is not None and (not isinstance(elimit, int) or elimit < 1):
        raise ValueError(f"{path}: embed.limit must be null or an integer >= 1")
    if not isinstance(embed.get("recompute", False), bool):
        raise ValueError(f"{path}: embed.recompute must be true or false")

    return cfg


# ---------------------------------------------------------------------------
# Derived names
# ---------------------------------------------------------------------------


def domains(cfg: dict) -> list[str]:
    """The configured domains, in pipeline order."""
    return [d for d in DOMAINS if d in cfg["generate"]["domains"]]


def types(cfg: dict) -> list[str]:
    """The configured generator types, in the fork's own order."""
    return [t for t in ALL_TYPES if t in cfg["generate"]["types"]]


def variant(cfg: dict) -> str:
    """The filename token for this run: ``gpt56luna``.

    Derived from the model that produced the machine rows, never configured —
    the point of the token is to say which generator a corpus came from, so a
    hand-typed value would be the one part of the name that can lie. A run that
    mixes an OpenAI and an Anthropic generator has no single answer, and gets
    ``multi``, the same convention science_data uses.
    """
    gen = cfg["generate"]
    selected = types(cfg)
    has_gpt = any(t in GPT_TYPES for t in selected)
    has_claude = "claude" in selected

    if has_gpt and has_claude:
        return "multi"
    if has_claude:
        return model_slug(gen["claude_model"])
    return model_slug(gen["gpt_model"])


def collected_dir(cfg: dict, var: str) -> Path:
    """Where the collect stage copies the fork's text tree.

    Beside the corpora it is a new generation of: ``data/ghostbuster-data`` is
    the original release, ``data/ghostbuster-data-gpt54`` the previous variant,
    so ``dataset_id: ghostbuster`` puts this one at
    ``data/ghostbuster-data-<variant>``.
    """
    return BASE_DIR / "data" / f"{cfg['dataset_id']}-data-{var}"


def domain_source_dir(cfg: dict, var: str, domain: str) -> Path:
    """The collected text tree for one domain — what prepare_*.py reads."""
    return collected_dir(cfg, var) / domain


def domain_parquet(cfg: dict, var: str, domain: str) -> Path:
    """One domain's parquet: ``data/parquet/essay_gpt56luna.parquet``."""
    return PARQUET_DIR / f"{domain}_{var}.parquet"


def combined_path(cfg: dict, var: str) -> Path:
    """The unioned corpus: ``data/parquet/ghostbuster_gpt56luna.parquet``."""
    return PARQUET_DIR / f"{cfg['dataset_id']}_{var}.parquet"


def scored_path(cfg: dict, var: str) -> Path:
    """The scored corpus: ``..._gpt56luna_binox0.parquet``.

    A separate file rather than an in-place update, which is score_parquets.py's
    default: the combine stage's output is cheap to keep and is the thing to
    re-score when the detector changes.
    """
    combined = combined_path(cfg, var)
    return combined.with_name(f"{combined.stem}{SCORED_SUFFIX}{combined.suffix}")


def generate_flags(cfg: dict, domain: str) -> list[str]:
    """The ``--<domain>_<type>`` flags for one domain, e.g. ``--essay_gpt_plain``."""
    return [f"--{domain}_{t}" for t in types(cfg)]


# ---------------------------------------------------------------------------
# Checks over the fork's data tree
# ---------------------------------------------------------------------------


def missing_seeds(cfg: dict) -> list[Path]:
    """Seeded input directories that are absent or empty, for the preflight.

    Absent *or empty*, because an empty directory is what a half-finished manual
    copy leaves behind, and generate.py's failure mode for it is an unhandled
    ``FileNotFoundError`` a thousand items into a paid run.
    """
    missing = []
    for domain in domains(cfg):
        for sub in SEED_SUBDIRS[domain]:
            path = GB_DATA_DIR / domain / sub
            if not path.is_dir() or not any(path.iterdir()):
                missing.append(path)
    return missing


def is_empty_txt(path: Path) -> bool:
    """Whether a generated file came back blank.

    Mirrors fix_empty_generations.py's own check, so the audit and the fix agree
    on what "empty" means — a 0-byte file and a file of whitespace are the same
    failed API call.
    """
    if path.stat().st_size == 0:
        return True
    return path.read_text(encoding="utf-8", errors="replace").strip() == ""


def count_txt(root: Path) -> tuple[int, int]:
    """``(files, empty)`` under ``root``, skipping the non-generator subdirs.

    Recursive, so it works for reuter's ``<author>/<n>.txt`` layout as well as
    the flat domains, and it skips ``logprobs``/``headlines`` for the same
    reason prepare_*.py do: those are not documents.
    """
    if not root.is_dir():
        return 0, 0
    files = empty = 0
    for path in root.rglob("*.txt"):
        if not NON_GENERATOR_DIRS.isdisjoint(path.relative_to(root).parts[:-1]):
            continue
        files += 1
        if is_empty_txt(path):
            empty += 1
    return files, empty


def expected_documents(cfg: dict, domain: str, limit: int | None) -> int:
    """How many documents a complete run should have produced for one domain.

    The fork's own loop bounds, not the number of human files: generate.py
    iterates ``1..(limit or 1000)`` reading ``prompts/<idx>.txt`` for essay and
    wp, and (author, ``1..20``) pairs for reuter. Those are different numbers —
    the seeded ``essay/human`` tree holds 1001 files against 1000 prompts, and
    counting humans would report a permanent shortfall of one.

    Capped by what is actually seeded, so a deliberately small seed (200
    prompts, say) is measured against itself rather than against 1000.
    """
    if domain == "reuter":
        human = GB_DATA_DIR / domain / "human"
        authors = [p for p in human.iterdir() if p.is_dir()] if human.is_dir() else []
        available = len(authors) * REUTER_ARTICLES_PER_AUTHOR
    else:
        available, _ = count_txt(GB_DATA_DIR / domain / "prompts")

    baseline = DEFAULT_DOMAIN_ITEMS if limit is None else limit
    return min(available, baseline)
