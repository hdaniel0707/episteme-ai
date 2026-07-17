"""One-shot installer for EXTERNAL impact-prediction systems (comparison baselines).

Pulls every registered external repo into workflow/external/ (gitignored), creates an
isolated uv virtual-environment per repo, installs its requirements there, and downloads
frozen model weights. Nothing here touches the project's own .venv or requirements.

    python -m workflow.src.setup_external_models            # install everything
    python -m workflow.src.setup_external_models --list     # show the registry
    python -m workflow.src.setup_external_models --only naip

Requires: git + uv on PATH (https://docs.astral.sh/uv/). Weights that live behind a
gated HuggingFace repo (Llama-3 base for NAIP) additionally need HF_TOKEN in the
repo-root .env with access granted on the model page.

Idempotent: existing clones are `git pull`-ed, existing venvs/weights are kept.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from workflow.src.iwf import paths

EXTERNAL_DIR = paths.WORKFLOW_DIR / "external"

# Registry of external impact-prediction systems used as comparison baselines.
#   repo        git URL cloned into external/<name>/
#   extras      packages the repo needs beyond its own requirements file
#   weights     callable(dest_dir) that fetches frozen weights; None = repo only
#   env_hint    .env.workflow keys the user must set afterwards
EXTERNAL_SYSTEMS: dict[str, dict] = {
    "naip": {
        "repo": "https://github.com/ssocean/NAIP",
        "requirements": "requirements.txt",     # relative inside the clone, if present
        "extras": ["peft", "accelerate"],
        "weights": "naip",                      # handled below via model_weights
        "env_hint": "NAIP_WEIGHTS_DIR=<printed weight path>",
    },
}


def _run(cmd: list[str], cwd: Path | None = None) -> bool:
    print(f"  $ {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [FAIL] {e}")
        return False


def _have(tool: str) -> bool:
    return shutil.which(tool) is not None


def install_one(name: str, spec: dict) -> bool:
    print(f"\n=== {name} ===")
    dest = EXTERNAL_DIR / name
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 1) clone or update the repo
    if (dest / ".git").is_dir():
        _run(["git", "-C", str(dest), "pull", "--ff-only"])
    else:
        if not _run(["git", "clone", "--depth", "1", spec["repo"], str(dest)]):
            return False

    # 2) isolated uv venv + the repo's requirements (never the project venv)
    venv = dest / ".venv"
    if not venv.exists():
        if not _run(["uv", "venv", str(venv)]):
            return False
    py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    req = dest / spec.get("requirements", "requirements.txt")
    if req.is_file():
        _run(["uv", "pip", "install", "--python", str(py), "-r", str(req)])
    extras = spec.get("extras") or []
    if extras:
        _run(["uv", "pip", "install", "--python", str(py), *extras])

    # 3) frozen weights
    if spec.get("weights") == "naip":
        try:
            from workflow.src.iwf.fetchers.model_weights import download_naip_weights
            download_naip_weights(dest / "weights")
        except Exception as e:
            print(f"  [weights] skipped ({e}) — rerun after fixing HF_TOKEN/access; "
                  "the clone + venv are already in place.")

    hint = spec.get("env_hint")
    if hint:
        print(f"  [next] set in workflow/.env.workflow: {hint}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="show the registry and exit")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of systems (default: all)")
    args = ap.parse_args()

    if args.list:
        for n, s in EXTERNAL_SYSTEMS.items():
            print(f"{n:12s} {s['repo']}")
        return 0

    for tool in ("git", "uv"):
        if not _have(tool):
            print(f"[setup] '{tool}' not found on PATH — install it first "
                  f"({'https://docs.astral.sh/uv/' if tool == 'uv' else 'https://git-scm.com'}).",
                  file=sys.stderr)
            return 1

    names = ([n.strip().lower() for n in args.only.split(",")] if args.only
             else list(EXTERNAL_SYSTEMS))
    ok = True
    for name in names:
        spec = EXTERNAL_SYSTEMS.get(name)
        if spec is None:
            print(f"[setup] unknown system {name!r}; have {list(EXTERNAL_SYSTEMS)}",
                  file=sys.stderr)
            ok = False
            continue
        ok = install_one(name, spec) and ok

    print("\n[setup] done." if ok else "\n[setup] finished with errors.", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
