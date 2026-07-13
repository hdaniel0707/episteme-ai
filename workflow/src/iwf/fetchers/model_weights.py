"""Download frozen model weights used by external scorers. USER-run only — never
invoked automatically by any phase script.

    python -m workflow.src.iwf.fetchers.model_weights
"""

from __future__ import annotations

from pathlib import Path

from workflow.src.iwf import paths

# Verified via https://huggingface.co/ssocean/NAIPv2 (LoRA adapter over a Llama-3
# backbone; AAAI'25, arXiv 2408.03934). The adapter's own adapter_config.json records
# which base model to pull, so callers never need to name it separately.
NAIP_DEFAULT_REPO = "ssocean/NAIPv2"


def download_naip_weights(dest: Path | None = None, repo_id: str | None = None) -> Path:
    """Download the NAIP v2 LoRA adapter from HuggingFace Hub.

    Needs ``huggingface_hub`` (already installed) and, since the adapter's base model
    is a gated Llama-3 checkpoint, an HF_TOKEN in the repo-root .env with access granted
    at https://huggingface.co/meta-llama/Meta-Llama-3-8B (request access, wait for
    approval, then generate a read token at https://huggingface.co/settings/tokens).
    Running inference additionally needs `pip install peft` (ask before installing).
    """
    from huggingface_hub import snapshot_download

    from workflow.src.iwf import config

    dest = Path(dest) if dest else paths.MODELS_DIR / "naip"
    repo_id = repo_id or NAIP_DEFAULT_REPO
    dest.mkdir(parents=True, exist_ok=True)

    token = config.secret("HF_TOKEN")
    if not token:
        print("[model_weights] no HF_TOKEN in .env — this will fail unless the repo is "
              "public and you already have Llama-3 access cached locally.")

    local_path = snapshot_download(repo_id=repo_id, local_dir=str(dest), token=token)
    print(f"[model_weights] {repo_id} -> {local_path}")
    print(f"[model_weights] set NAIP_WEIGHTS_DIR={local_path} in workflow/.env.workflow "
          "to enable the naip_local external_impact scorer.")
    return Path(local_path)


if __name__ == "__main__":
    download_naip_weights()
