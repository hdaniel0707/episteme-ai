r"""Progress of every planned rebuttal run, read from the sidecars on disk.
Makes no calls. The sidecars are checkpointed every 200 calls, so a running job
shows up to 200 calls behind its progress bar.

    bash epai/ai_detection/rebuttal/status.sh
"""

from __future__ import annotations

import time

import pandas as pd

from epai.ai_detection.rebuttal.conditions import build_subset, load_core
from epai.ai_detection.rebuttal.capture import _sidecar

# (script, model, condition, subset, reps) -- must match the run_*.sh scripts
PLAN = [
    ("main", "gpt56luna", "base_hp4", "full", 1),
    ("main", "gpt56luna", "base_free", "full", 1),
    ("main", "gpt56luna", "base_v3", "full", 1),
    ("main", "gpt56luna", "lj_v3", "near", 3),
    ("main", "gpt56luna", "base_v3", "near", 3),
    ("extra", "gpt56luna", "lj_swap", "small", 1),
    ("extra", "gpt56luna", "lj_alt", "small", 1),
    ("extra", "gpt56luna", "lj_letters", "small", 1),
    ("extra", "gpt56luna", "base_v3_t1", "small", 3),
    ("gemma", "gemma4-31b", "base_hp4", "humans_plus_third", 1),
    ("gemma", "gemma4-31b", "base_free", "humans_plus_third", 1),
    ("gemma", "gemma4-31b", "base_v3", "humans_plus_third", 1),
    ("default", "gpt56luna", "base_default", "near", 3),
]


def main() -> None:
    core = load_core()
    core = core[core["text"].astype(str).str.strip().str.len() > 0]
    sizes = {s: len(build_subset(core, s)) for s in {p[3] for p in PLAN}}
    print(f"{'script':6} {'model':11} {'condition':11} {'subset':18} "
          f"{'done':>7} / {'total':<7} {'%':>5}  {'failed':>6}  last write")
    for script, model, cond, subset, reps in PLAN:
        total = sizes[subset] * reps
        path = _sidecar(model, cond, subset)
        if not path.is_file():
            print(f"{script:6} {model:11} {cond:11} {subset:18} {0:>7} / {total:<7} "
                  f"{0:>5.1f}  {0:>6}  not started")
            continue
        df = pd.read_parquet(path, columns=["answer", "error"])
        done = int(df["answer"].notna().sum())
        failed = int(df["answer"].isna().sum())
        when = time.strftime("%m-%d %H:%M", time.localtime(path.stat().st_mtime))
        print(f"{script:6} {model:11} {cond:11} {subset:18} {done:>7} / {total:<7} "
              f"{100 * done / total:>5.1f}  {failed:>6}  {when}")


if __name__ == "__main__":
    main()
