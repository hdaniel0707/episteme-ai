"""All filesystem locations used by the workflow.

Everything the workflow writes lives under `workflow/` (data/, results/, models/).
The original `epai` store is read-only and referenced via `EPAI_STORE_DIR`.
"""

from pathlib import Path

# .../workflow/src/iwf/paths.py
#   parents[0] = iwf, parents[1] = src, parents[2] = workflow, parents[3] = repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

# Original epai store (read-only source of papers).
EPAI_STORE_DIR = REPO_ROOT / "store"
EPAI_SQL_DIR = EPAI_STORE_DIR / "data" / "sql"

# Workflow-owned output locations.
DATA_DIR = WORKFLOW_DIR / "data"
PARQUET_DIR = DATA_DIR / "parquet"
SQL_DIR = DATA_DIR / "sql"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = WORKFLOW_DIR / "results"
MODELS_DIR = WORKFLOW_DIR / "models"
GENERATED_RESULTS_DIR = RESULTS_DIR / "generated"

# The two user-editable configuration files + the repo-root secrets file.
ROOT_ENV = REPO_ROOT / ".env"                      # secrets / API keys (never edited)
WORKFLOW_ENV = WORKFLOW_DIR / ".env.workflow"      # machine / operational knobs
WORKFLOW_CONFIG = WORKFLOW_DIR / "config.yaml"     # training configuration + runs queue


def ensure_dirs() -> None:
    """Create the workflow output directories if they do not exist."""
    for d in (PARQUET_DIR, SQL_DIR, CACHE_DIR, RESULTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---- artifact name helpers (one dataset id flows through every phase) ----

def raw_parquet(dataset_id: str) -> Path:
    return PARQUET_DIR / f"{dataset_id}_raw.parquet"


def prepared_parquet(dataset_id: str) -> Path:
    return PARQUET_DIR / f"{dataset_id}.parquet"


def embedded_parquet(dataset_id: str) -> Path:
    return PARQUET_DIR / f"{dataset_id}_emb.parquet"


def reduced_parquet(dataset_id: str) -> Path:
    return PARQUET_DIR / f"{dataset_id}_z.parquet"


def generated_parquet(dataset_id: str) -> Path:
    return PARQUET_DIR / f"{dataset_id}_generated.parquet"


def encoder_path(dataset_id: str) -> Path:
    return MODELS_DIR / f"{dataset_id}_encoder.pt"


def encoder_meta_path(dataset_id: str) -> Path:
    return MODELS_DIR / f"{dataset_id}_encoder.json"
