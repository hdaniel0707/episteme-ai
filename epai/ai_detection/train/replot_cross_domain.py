"""Regenerate cross-domain figures from saved JSON, without retraining.

Walks a results folder and redraws every figure it can from the data already on
disk -- use it after tweaking anything in the plotting code, or after a run made
with ``--no-plots``. It dispatches on the JSON filenames
``run_clf_cross_domain.py`` writes:

  * every ``cross_domain.json`` -> the per-matrix figures (accuracy / macro-F1 /
    summary), via ``plot_clf_cd_result`` on its directory. This covers each
    variant directory and each zero-shot baseline directory.
  * every ``variant_comparison.json`` -> the comparison figures, via
    ``plot_clf_cd_comparison``.

The two legacy comparison filenames are still recognised, so result trees
written before the three runners merged (``results/clf_cd_emb/``,
``results/clf_cd_algs/``) redraw exactly as they did:

  * ``embedding_comparison.json`` -> ``plot_clf_cd_emb_comparison``
  * ``algorithm_comparison.json`` -> ``plot_clf_cd_alg_comparison``

The input can be any folder from a single result directory up to the whole
``results/`` tree; matching files are found recursively, so the scope replotted
is exactly the folder you point it at.

Run:
    uv run python epai/ai_detection/train/replot_cross_domain.py results/clf_cd
    uv run python epai/ai_detection/train/replot_cross_domain.py \
        results/clf_cd/clf_cd_20260807_101500
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

from epai.utils.visualization_utils import (  # noqa: E402  (after backend set)
    plot_clf_cd_alg_comparison,
    plot_clf_cd_comparison,
    plot_clf_cd_emb_comparison,
    plot_clf_cd_result,
)

CROSS_DOMAIN = "cross_domain.json"

#: Comparison files, and the function that draws each. A filename maps
#: unambiguously to a payload shape, so dispatching on it needs no sniffing --
#: which is also how the two pre-merge shapes keep working: they are simply two
#: more rows, pointed at the legacy readers that understand their row key.
COMPARISONS: tuple[tuple[str, str, object], ...] = (
    ("variant_comparison.json", "cmp", plot_clf_cd_comparison),
    ("embedding_comparison.json", "emb cmp", plot_clf_cd_emb_comparison),
    ("algorithm_comparison.json", "alg cmp", plot_clf_cd_alg_comparison),
)


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def replot_folder(root: Path) -> dict[str, int]:
    """Redraw every figure derivable from the JSON under ``root``.

    Returns per-kind counts (including how many files failed). ``rglob`` matches
    files directly inside ``root`` as well as in any subdirectory, so pointing at
    a single result directory works just as well as pointing at the whole tree.
    """
    counts = {"matrices": 0, "comparisons": 0, "errors": 0}

    # Per-matrix figures: one cross_domain.json per variant / baseline directory.
    # plot_clf_cd_result reads the file from the directory.
    for path in sorted(root.rglob(CROSS_DOMAIN)):
        print(f"[matrix]  {path.parent}")
        try:
            plot_clf_cd_result(path.parent)
            counts["matrices"] += 1
        except Exception as exc:
            print(f"  [ERROR] {path}: {type(exc).__name__}: {exc}")
            counts["errors"] += 1

    for filename, tag, draw in COMPARISONS:
        for path in sorted(root.rglob(filename)):
            print(f"[{tag}] {path.parent}")
            try:
                draw(path.parent, _load(path))
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

    total = counts["matrices"] + counts["comparisons"]
    if total == 0 and counts["errors"] == 0:
        print(
            f"\nNo cross-domain JSON found under {root}. Point this at a "
            f"results/<experiment>/ folder or one of its subdirectories."
        )
        return

    print(
        f"\nDone. Replotted {counts['matrices']} matrix set(s) and "
        f"{counts['comparisons']} comparison(s)."
    )
    if counts["errors"]:
        print(f"{counts['errors']} file(s) failed — see the [ERROR] lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
