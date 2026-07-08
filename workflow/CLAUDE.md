# CLAUDE.md (workflow/)

Scoped guidance for working inside `workflow/`. The repo-root `CLAUDE.md` still applies
(no package installs, ask before new dependencies, don't run scripts unless asked) — this
file only adds workflow-specific layout rules.

## Layout

- `run_pipeline.py`, `run_all_workflows.py` — the only two entry points at this root.
  Everything else importable/runnable lives under `src/`.
- **Two user-editable config files, two jobs:**
  - `config.yaml` — WHAT to train: workflow type, target, class-count knobs, model
    architecture, hyperparameters, and the `runs:` queue for multi-training sessions.
    A run entry may override any top-level key for that run only.
  - `.env.workflow` — HOW this machine runs it: parallelism, GPU batch size, force
    flags. Nothing here may change the science of a run.
  - Secrets stay in the repo-root `.env`; never put keys in either workflow file.
- `src/` — phase scripts (`main_*.py`), the `iwf/` package (the actual logic), and
  `src/tests/smoke.py` (offline suite). Import as `workflow.src.iwf....`.
- `data/`, `models/`, `results/` — generated artifacts, safe to delete and regenerate.
- `README.md` — architecture + phase reference, paired with `WORKFLOW.drawio`.
- `WORKFLOW_REFERENCE.md` — every `config.yaml` / `.env.workflow` key explained.
- `mathematical_documentation.md` — the equations behind targets, metrics, classes.

## Rules

- `epai/` (repo root) is the read-only base library — reuse and import from it, never
  edit it. Only edit inside `workflow/`.
- The topology table `WORKFLOWS` in `src/iwf/config.py` is the single source of truth
  for task / feature-set / phase-list per workflow type — extend it there, never
  hard-code workflow names elsewhere.
- ONE implementation per concern: metrics in `iwf/evaluation/metrics.py`, all plots in
  `iwf/evaluation/plots.py`, the multi-seed loop in `iwf/training/experiment.py`. Do not
  add per-workflow copies; add a task-variant inside the shared module instead.
- Per-run parameters travel as override dicts into `Settings` (`load_settings(...,
  overrides=...)`) — never mutate `os.environ` to pass parameters.
- Every workflow writes the same results layout (`results/<wf>_N/seed_<i>/result.json`);
  keep new outputs inside that contract so aggregation/plots stay workflow-agnostic.
- When adding a phase script, put it in `src/` and wire it into `run_pipeline._run_phase`
  + the `WORKFLOWS` phase lists, not as a new top-level entry point.
