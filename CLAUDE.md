# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Rules
- Do NOT install any Python packages or modify requirements.txt / pyproject.toml
- Do NOT run `pip install`, `uv add`, or any package manager commands
- Ask me first if you need a new dependency
- Do NOT run scripts only write them if you are asked.
- Do NOT delete, recreate, or modify anything inside any `.venv/` — this repo's or a
  submodule's. Treat every `.venv/` as read-only.

## The `.venv` rule is about side effects, not just `rm`

The way this gets broken is not `rm -rf .venv`. It is a command that looked read-only:

- `uv run` and `uv sync` **silently delete and recreate** a `.venv` when its base
  interpreter is missing — which is normal here, because these venvs are built against
  an interpreter that does not exist in the sandbox.
- `uv run --no-sync` does **not** prevent this. It only skips *reinstalling packages*,
  so it deletes the venv, recreates it, and leaves it **empty**. This already happened
  to `external/Binoculars/.venv` and cost a multi-GB torch reinstall.

So: assume any `uv` command can destroy a venv, whatever flags it carries.

- To inspect an environment, **read files** — `.venv/pyvenv.cfg`, or list
  `.venv/lib/*/site-packages`. Do not probe it by running something through `uv`.
- Ask me before running any `uv` command at all, including ones that only look like
  they read (`uv run python -c ...`, `uv pip list`, version checks).
- Writing a *script* that calls `uv run` is fine — that is the script's job at runtime.
  The rule is about what you execute yourself.

> **Note**: Codebase is currently being restructured. Some modules may be incomplete or not yet wired up to the entry point scripts.
