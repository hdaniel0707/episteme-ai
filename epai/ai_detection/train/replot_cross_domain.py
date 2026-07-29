"""Regenerate cross-domain figures from saved JSON, without retraining.

Walks a results folder and redraws every figure it can from the data already on
disk -- use it after tweaking anything in the plotting code. It works for both
result types (``run_clf_cross_domain_embeddings.py`` and
``run_clf_cross_domain_algs.py``) because it dispatches on the JSON filenames
they write:

  * every ``cross_domain.json`` -> the per-matrix figures (accuracy / macro-F1 /
    summary), via ``plot_clf_cd_result`` on its directory. This covers each
    embedding / algorithm directory and the ``binoculars`` baseline directory.
  * every ``embedding_comparison.json`` -> the embedding comparison figures, via
    ``plot_clf_cd_emb_comparison``.
  * every ``algorithm_comparison.json`` -> the algorithm comparison figures, via
    ``plot_clf_cd_alg_comparison``.

The input can be any folder from a single result directory up to the whole
``results/`` tree; matching files are found recursively, so the scope replotted
is exactly the folder you point it at.

Run:
    uv run python epai/ai_detection/train/replot_cross_domain.py results/clf_cd_algs
    uv run python epai/ai_detection/train/replot_cross_domain.py results/clf_cd_emb
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

from epai.utils.visualization_utils import (  # noqa: E402  (after backend set)
    plot_clf_cd_alg_comparison,
    plot_clf_cd_emb_comparison,
    plot_clf_cd_result,
)

CROSS_DOMAIN = "cross_domain.json"
EMB_COMPARISON = "embedding_comparison.json"
ALG_COMPARISON = "algorithm_comparison.json"


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def replot_folder(root: Path) -> dict[str, int]:
    """Redraw every figure derivable from the JSON under ``root``.

    Returns per-kind counts (including how many files failed). ``rglob`` matches
    files directly inside ``root`` as well as in any subdirectory, so pointing at
    a single result directory works just as well as pointing at the whole tree.
    """
    counts = {
        "matrices": 0,
        "embedding_comparison": 0,
        "algorithm_comparison": 0,
        "errors": 0,
    }

    # Per-matrix figures: one cross_domain.json per embedding / algorithm /
    # baseline directory. plot_clf_cd_result reads the file from the directory.
    for path in sorted(root.rglob(CROSS_DOMAIN)):
        print(f"[matrix]  {path.parent}")
        try:
            plot_clf_cd_result(path.parent)
            counts["matrices"] += 1
        except Exception as exc:
            print(f"  [ERROR] {path}: {type(exc).__name__}: {exc}")
            counts["errors"] += 1

    # Comparison figures: the two scripts write different filenames, so each maps
    # unambiguously to its own plotting function.
    for path in sorted(root.rglob(EMB_COMPARISON)):
        print(f"[emb cmp] {path.parent}")
        try:
            plot_clf_cd_emb_comparison(path.parent, _load(path))
            counts["embedding_comparison"] += 1
        except Exception as exc:
            print(f"  [ERROR] {path}: {type(exc).__name__}: {exc}")
            counts["errors"] += 1

    for path in sorted(root.rglob(ALG_COMPARISON)):
        print(f"[alg cmp] {path.parent}")
        try:
            plot_clf_cd_alg_comparison(path.parent, _load(path))
            counts["algorithm_comparison"] += 1
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

    total = counts["matrices"] + counts["embedding_comparison"] + counts["algorithm_comparison"]
    if total == 0 and counts["errors"] == 0:
        print(
            f"\nNo cross-domain JSON found under {root}. Point this at a "
            f"results/<experiment>/ folder or one of its subdirectories."
        )
        return

    print(
        f"\nDone. Replotted {counts['matrices']} matrix set(s), "
        f"{counts['embedding_comparison']} embedding comparison(s), "
        f"{counts['algorithm_comparison']} algorithm comparison(s)."
    )
    if counts["errors"]:
        print(f"{counts['errors']} file(s) failed — see the [ERROR] lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
