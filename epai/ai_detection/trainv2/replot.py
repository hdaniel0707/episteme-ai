"""Redraw the figures and rewrite the CSVs of an existing ``run_in_domain.py``
result directory — from its ``summary.jsonl`` only. No parquet read, no
training, no seeds refit.

Use this after tweaking a plot (a colour, an axis, a label) to update a run
you already have, instead of retraining it — or to add the CSVs to a result
directory written before ``run_in_domain.py`` wrote them.

(``run_embedding_effectiveness.py`` has its own ``--replot`` — it reads its
``summary.csv`` directly.)

Run:
    uv run python epai/ai_detection/trainv2/replot.py results/trainv2/<run_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import write_csv


def _load_jsonl(path: Path) -> tuple[dict, list[dict]]:
    """A trainv2 ``.jsonl`` file: its leading ``_meta`` row, and the rest."""
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    if not rows or "_meta" not in rows[0]:
        raise ValueError(f"{path}: expected a leading {{'_meta': ...}} row")
    return rows[0]["_meta"], rows[1:]


def replot(run_dir: Path) -> None:
    summary_path = run_dir / "summary.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{run_dir}: no summary.jsonl — not a trainv2 result directory"
        )
    meta, summary_rows = _load_jsonl(summary_path)
    fig_dir = run_dir / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary_rows, fig_dir / "summary.csv")

    from .plots import scatter_grids

    paths = scatter_grids(summary_rows, meta, fig_dir)
    print(f"Replotted (generator effectiveness): {run_dir}")
    print("  " + ", ".join(f"fig/{p.name}" for p in paths) + ", fig/summary.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", type=Path, help="a results/trainv2/<name>_<timestamp> directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    replot(args.run_dir)


if __name__ == "__main__":
    main()
