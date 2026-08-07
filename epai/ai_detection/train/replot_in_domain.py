"""Regenerate in-domain figures from saved JSON, without retraining.

The counterpart of ``replot_cross_domain.py``, for what
``run_clf_in_domain.py`` writes. Walks a results folder and redraws every figure
it can from the data already on disk -- use it after tweaking the plotting code,
or after a run made with ``--no-plots``. It dispatches on the filenames:

  * every ``in_domain.json`` -> that variant's confusion matrix, via
    ``plot_clf_id_result`` on its directory. This covers each variant directory
    and each zero-shot baseline directory.
  * every ``in_domain_comparison.json`` -> the ranked comparison figure, via
    ``plot_clf_id_comparison``.

The two filenames are distinct from the cross-domain ones on purpose: pointing
either replot script at the whole ``results/`` tree redraws only its own
experiments and silently ignores the other's.

The input can be any folder from a single result directory up to the whole
``results/`` tree; matching files are found recursively, so the scope replotted
is exactly the folder you point it at.

Run:
    uv run python epai/ai_detection/train/replot_in_domain.py results/clf_id
    uv run python epai/ai_detection/train/replot_in_domain.py \
        results/clf_id/clf_id_20260807_101500
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

from epai.utils.visualization_utils import (  # noqa: E402  (after backend set)
    plot_clf_id_comparison,
    plot_clf_id_result,
)

RESULT = "in_domain.json"
COMPARISON = "in_domain_comparison.json"


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def replot_folder(root: Path) -> dict[str, int]:
    """Redraw every figure derivable from the JSON under ``root``.

    Returns per-kind counts (including how many files failed). ``rglob`` matches
    files directly inside ``root`` as well as in any subdirectory, so pointing at
    a single result directory works just as well as pointing at the whole tree.
    """
    counts = {"results": 0, "comparisons": 0, "errors": 0}

    for path in sorted(root.rglob(RESULT)):
        print(f"[result] {path.parent}")
        try:
            plot_clf_id_result(path.parent)
            counts["results"] += 1
        except Exception as exc:
            print(f"  [ERROR] {path}: {type(exc).__name__}: {exc}")
            counts["errors"] += 1

    for path in sorted(root.rglob(COMPARISON)):
        print(f"[cmp]    {path.parent}")
        try:
            plot_clf_id_comparison(path.parent, _load(path))
            counts["comparisons"] += 1
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

    if counts["results"] + counts["comparisons"] == 0 and counts["errors"] == 0:
        print(
            f"\nNo in-domain JSON found under {root}. Point this at a "
            f"results/clf_id/<experiment>/ folder or one of its subdirectories."
        )
        return

    print(
        f"\nDone. Replotted {counts['results']} result(s) and "
        f"{counts['comparisons']} comparison(s)."
    )
    if counts["errors"]:
        print(f"{counts['errors']} file(s) failed — see the [ERROR] lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
