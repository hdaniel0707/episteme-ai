"""Regenerate leave-one-out figures from saved JSON, without retraining.

Walks a results folder and redraws every leave-one-out figure it can from the
data already on disk -- use it after tweaking anything in the plotting code.
It dispatches on the JSON filename written by
``epai/ai_detection/train/run_clf_leave_one_out.py``:

  * every ``leave_one_out.json`` -> the per-fold figures (the grouped LODO/LOGO
    bars and the model-vs-Binoculars gap chart), via ``plot_clf_loo_result`` on
    its directory.

The input can be any folder from a single result directory up to the whole
``results/`` tree; matching files are found recursively, so the scope replotted
is exactly the folder you point it at.

Run:
    uv run python epai/ai_detection/train/replot_leave_one_out.py results/clf_loo
    uv run python epai/ai_detection/train/replot_leave_one_out.py results/clf_loo/clf_loo_20260728_120000
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

from epai.utils.visualization_utils import (  # noqa: E402  (after backend set)
    plot_clf_loo_result,
)

LEAVE_ONE_OUT = "leave_one_out.json"


def replot_folder(root: Path) -> dict[str, int]:
    """Redraw every figure derivable from the leave-one-out JSON under ``root``.

    Returns per-kind counts (including how many files failed). ``rglob`` matches
    files directly inside ``root`` as well as in any subdirectory, so pointing at
    a single result directory works just as well as pointing at the whole tree.
    """
    counts = {"folds": 0, "errors": 0}

    # One leave_one_out.json per algorithm directory; plot_clf_loo_result reads
    # the file from the directory it is given.
    for path in sorted(root.rglob(LEAVE_ONE_OUT)):
        print(f"[loo]  {path.parent}")
        try:
            plot_clf_loo_result(path.parent)
            counts["folds"] += 1
        except Exception as exc:
            print(f"  [ERROR] {path}: {type(exc).__name__}: {exc}")
            counts["errors"] += 1

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "folder",
        type=Path,
        help="Results folder to replot (a single result dir, an experiment dir, "
        "or the whole results/ tree). Searched recursively.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.folder
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    print(f"Replotting from: {root}\n")
    counts = replot_folder(root)

    if counts["folds"] == 0 and counts["errors"] == 0:
        print(
            f"\nNo leave-one-out JSON found under {root}. Point this at a "
            f"results/clf_loo/<experiment>/ folder or one of its subdirectories."
        )
        return

    print(f"\nDone. Replotted {counts['folds']} leave-one-out result(s).")
    if counts["errors"]:
        print(f"{counts['errors']} file(s) failed — see the [ERROR] lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
