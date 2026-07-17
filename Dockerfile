# EpistemeAI workflow — training/analysis container (CUDA-ready, CPU fallback).
#
#   docker build -t epistemeai-workflow .
#   docker run --gpus all -v $PWD/workflow/data:/app/workflow/data \
#              -v $PWD/workflow/results:/app/workflow/results \
#              --env-file .env epistemeai-workflow \
#              python -m workflow.run_all_workflows
#
# Data collection (run BEFORE training, separately):
#   docker run ... epistemeai-workflow python -m workflow.src.main_extend_db bootstrap --target-rows 100000
#   docker run ... epistemeai-workflow python -m workflow.src.main_extend_db extend
#
# The image installs dependencies from pyproject.toml/uv.lock only; data, models,
# results and external/ stay in mounted volumes (see .dockerignore).

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UMAP_OPEN_BROWSER=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv (package manager) + Python toolchain
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# dependency layer (cached unless the lock changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# project code (data/results/models/external are excluded via .dockerignore and
# expected as volume mounts)
COPY epai/ epai/
COPY workflow/ workflow/
COPY cfg/ cfg/

ENV PATH="/app/.venv/bin:${PATH}"

# offline self-check by default; override the command for real runs
CMD ["python", "-m", "workflow.src.tests.smoke"]
