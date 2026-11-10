# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Rules

## Packages and environments
- Do NOT install, upgrade, or remove any package, and do NOT modify requirements.txt /
  pyproject.toml. Ask me first if you need a new dependency.
- Do NOT run `uv` in any form (`uv run`, `uv sync`, `uv pip ...`, `uv run --no-sync`,
  version checks). `uv` silently deletes and recreates a `.venv` whose base interpreter
  is missing, and `--no-sync` just leaves it empty; this already cost a multi-GB torch
  reinstall of `external/Binoculars/.venv`. Writing a script that calls `uv run` is
  fine -- that is the script's job at runtime.
- Treat every `.venv/` (this repo's and every submodule's) as read-only. To inspect one,
  read its files (`pyvenv.cfg`, `lib/*/site-packages`).

## Running code
- You may run Python scripts, for smoke tests and for the release work below. Run them
  with this repo's venv interpreter directly, never through `uv`:
  `.venv/bin/python path/to/script.py` (run from the repo root; its base interpreter,
  /usr/bin/python3.12, exists in the sandbox).
- Do not run anything with a submodule's venv (e.g. `external/Binoculars/.venv`).
- `permissions.blockReadsOutsideWorkingDirectories` is on: every path in a Bash command
  must be provable from the command text alone, or it stops for approval. So:
  - Use absolute paths (`/home/daniel/projects/episteme-ai/...`). Never start with
    `cd X;` / `cd X &&` -- the working directory is already the repo root.
  - No paths built at run time: no `$(...)`, `$VAR`, globs-in-subshells, or paths
    computed inside `python -c` (e.g. `os.listdir` to pick a file). Look the file up
    first (`ls`), then use its literal path in the next command.
  - Prefer one simple command per call over long `;`-chained one-liners; for anything
    non-trivial in Python, write a script to the scratchpad and run it.
  - Use the Read tool rather than `cat`/`sed -n` for plain file reads.
- Ask first before running anything slow, GPU-bound, or that calls a paid/remote API
  (judge capture, generation, Binoculars/Fast-DetectGPT scoring, embedding).
- Write outputs only under `../logjudge-manugram/`, `released_data/`, or your scratchpad /
  `$TMPDIR`. The master parquets in `data/parquet/` are read-only: never write, rewrite,
  or reorder them.

> **Note**: Codebase is currently being restructured. Some modules may be incomplete or not yet wired up to the entry point scripts.

## What this repo is

This repo is the working codebase and paper source for a submission to **ICLR 2027**
on AI-text detection. The paper draft — theory, framing, related work, experiment
writeups — lives in LaTeX in the `overleaf/` git submodule, which is the Overleaf
project itself (remote `https://git.overleaf.com/...`, branch `main`). Main file
`overleaf/iclr2027_conference.tex`; bibliography `iclr2027_conference.bib`; figures in
`overleaf/figs/`. `0_iclr2027_conference.tex` is the untouched ICLR formatting
template, not the paper. The old copy-pasted `latex/` folder is gone; ignore any
`latex/_ICLR_2027__AI_Detect/...` path still mentioned in code or docstrings
(it now means `overleaf/...`).

When you need to understand the motivation, terminology, or claims behind an
experiment or script in this repo, check the paper source first — it's the
authoritative statement of the theory and argument the code is built to support.

### Paper workflow (Overleaf submodule)
- Overleaf is the source of truth; the user also edits there in the browser. Before
  reading or editing the paper, run `git -C /home/daniel/projects/episteme-ai/overleaf pull`
  so you see the current text (if it fails on auth, tell the user rather than retrying).
- Edit files under `overleaf/` directly and commit inside the submodule
  (`git -C .../overleaf commit`). Do NOT `git push` to Overleaf without asking — a push
  shows up live in the shared Overleaf project.
- Figures for the paper go in `overleaf/figs/`; they reach Overleaf only once committed
  and pushed there.
- After pushing, the parent repo's pointer moves: `git add overleaf` and commit here
  (only when the user asks for a commit).

## Release for review (anonymized code + data)

- `released_data/{full,lite}/` holds the reviewer parquets (`<stem>.parquet`,
  `<stem>_lite.parquet` without embeddings), built by
  `epai/ai_detection/score/build_release_parquets.py` from the `release:` block of
  `cfg/ai_detection/core_parquet_columns.yaml`. Only paper-backed columns. Also on OSF
  (anonymized view-only link in the release repo's `.env_sample`).
- `../logjudge-manugram/` (sibling folder, its own git repo, own uv venv on
  /usr/bin/python3.12) is the anonymized code repo for the submission. Two steps:
  `scripts/compute_all.py` (release parquets -> `outputs/<result>/*.csv`, the only step
  that reads data; package `compute/`) and `scripts/plot_all.py` (CSVs only -> figures;
  package `plotting/`, kept separable). One output folder per paper result. 1%-FPR only.
- `released_data/` is gitignored here.
- Nothing in `../logjudge-manugram/` may identify the authors: no personal names or emails,
  no HUN-REN / SZTAKI / GENAI4Science, no hostnames, inference-endpoint URLs, API keys,
  or `/home/...` paths.
