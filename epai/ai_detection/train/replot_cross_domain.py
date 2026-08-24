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
  * every ``suite_comparison.json`` -> the cross-dataset figures of a run whose
    config declared ``data.suites``, via ``plot_clf_cd_suite_comparison``. It
    sits one level ABOVE the per-dataset directories, so pointing this at a run
    directory redraws the datasets and the comparison across them in one pass.

The two legacy comparison filenames are still recognised, so result trees
written before the three runners merged (``results/clf_cd_emb/``,
``results/clf_cd_algs/``) redraw exactly as they did:

  * ``embedding_comparison.json`` -> ``plot_clf_cd_emb_comparison``
  * ``algorithm_comparison.json`` -> ``plot_clf_cd_alg_comparison``

The input can be any folder from a single result directory up to the whole
``results/`` tree; matching files are found recursively, so the scope replotted
is exactly the folder you point it at.

Figures are written over the ones already there, under the same names. That
matters for the ranked comparison figure -- ``<noun>_comparison_<metric>.png``,
so ``algorithm_comparison_auroc.png`` and the rest -- because it no longer draws
what it drew before. It used to be paired bars with a ±1 SD whisker; it is now
one row per item showing every source->target transfer cell, beside the same
cells as a paired difference against the best item. Any PNG of the old shape
still on disk is stale, and a replot is what clears it. Nothing about the JSON
changed, so no run needs repeating.

Run:
    uv run python epai/ai_detection/train/replot_cross_domain.py results/clf_cd

    # one experiment (this also redraws the cross-dataset figures above it)
    uv run python epai/ai_detection/train/replot_cross_domain.py \
        results/clf_cd/clf_cd_20260824_225250

    # one dataset of one experiment
    uv run python epai/ai_detection/train/replot_cross_domain.py \
        results/clf_cd/clf_cd_20260824_113244/all_datasets/science_v3
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

from epai.utils.cd_core import attach_by_seed  # noqa: E402
from epai.utils.suite_utils import mean_by_seed_across  # noqa: E402
from epai.utils.visualization_utils import (  # noqa: E402  (after backend set)
    plot_clf_cd_alg_comparison,
    plot_clf_cd_comparison,
    plot_clf_cd_emb_comparison,
    plot_clf_cd_result,
    plot_clf_cd_suite_comparison,
)

CROSS_DOMAIN = "cross_domain.json"

#: Row keys that name a variant's own sub-directory, in the order to try them.
#: ``variant`` is the id the runner names the directory after; ``label`` is the
#: fallback for a payload old enough not to carry one.
VARIANT_ID_KEYS = ("variant", "label")

#: Comparison files, and the function that draws each. A filename maps
#: unambiguously to a payload shape, so dispatching on it needs no sniffing --
#: which is also how the two pre-merge shapes keep working: they are simply two
#: more rows, pointed at the legacy readers that understand their row key.
COMPARISONS: tuple[tuple[str, str, object], ...] = (
    ("variant_comparison.json", "cmp", plot_clf_cd_comparison),
    ("suite_comparison.json", "ds cmp", plot_clf_cd_suite_comparison),
    ("embedding_comparison.json", "emb cmp", plot_clf_cd_emb_comparison),
    ("algorithm_comparison.json", "alg cmp", plot_clf_cd_alg_comparison),
)


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _variant_index(base: Path) -> dict[str, Path]:
    """Every name a variant under ``base`` answers to, mapped to its directory.

    A row cannot simply be looked up by ``row["variant"]``: the pre-merge
    payloads in ``results/clf_cd_algs/`` carry no ``variant`` key at all, and
    their ``label`` ("Logistic Regression") is not the directory name
    ("logistic"). So the index is built from the directories themselves -- each
    one's ``cross_domain.json`` states both -- and a row is matched against
    whichever of its own names is in it.

    A name claimed by two directories is dropped rather than guessed at: pointing
    a row at the wrong variant's seeds would put a plausible and wrong error bar
    on the figure, which is worse than the bar being absent.
    """
    index: dict[str, Path] = {}
    clashes: set[str] = set()

    def claim(name, directory: Path) -> None:
        if not name:
            return
        name = str(name)
        if index.get(name, directory) != directory:
            clashes.add(name)
        index.setdefault(name, directory)

    for directory in sorted(d for d in base.iterdir() if d.is_dir()):
        path = directory / CROSS_DOMAIN
        if not path.is_file():
            continue
        claim(directory.name, directory)
        try:
            payload = _load(path)
        except (OSError, ValueError):
            continue
        for key in VARIANT_ID_KEYS:
            claim(payload.get(key), directory)
        # The pre-merge payloads carry neither, and their directory name is an
        # algorithm id the comparison row never mentions. What they do carry is
        # `alg_label`, whose last em-dash segment IS the row's label -- the
        # runner built the row's label from it. That is the only link between the
        # two files, so it is worth following.
        claim(payload.get("algorithm"), directory)
        alg_label = payload.get("alg_label")
        if alg_label:
            claim(str(alg_label).rsplit("—", 1)[-1].strip(), directory)

    for name in clashes:
        index.pop(name, None)
    return index


def _variant_dir(index: dict[str, Path], row: dict) -> Path | None:
    """Where one comparison row's own results live, if they can be identified."""
    for key in VARIANT_ID_KEYS:
        directory = index.get(str(row.get(key)))
        if directory is not None:
            return directory
    return None


def _backfill_by_seed(path: Path, payload: dict) -> int:
    """Recover the per-seed aggregates a pre-existing comparison JSON lacks.

    The error bars on the comparison figures are the spread of each seed's own
    in-domain / transfer mean. Runs finished before that was recorded have no
    such field in their comparison JSON, but they do still have every seed in the
    per-variant ``cross_domain.json`` files sitting one directory down -- so the
    numbers are recoverable here, and no run has to be repeated to get a figure
    with error bars on it.

    Works in memory, on the payload about to be drawn: the JSON on disk is left
    exactly as the run wrote it. A ``suite_comparison.json`` is two levels up
    from its variants, so its rows are filled per dataset first and then
    recombined across datasets the same way the runner does it.

    Returns how many rows gained anything.
    """
    rows = (
        payload.get("variants")
        or payload.get("algorithms")
        or payload.get("embeddings")
        or []
    )
    suite_names = payload.get("suite_names")
    base = path.parent
    if not base.is_dir():
        return 0
    index = {
        suite: _variant_index(base / suite)
        for suite in (suite_names or [])
        if (base / suite).is_dir()
    } if suite_names else _variant_index(base)
    filled = 0

    for row in rows:
        if suite_names is None:
            variant_dir = _variant_dir(index, row)
            if variant_dir is not None and attach_by_seed(row, variant_dir):
                filled += 1
            continue

        per_suite = row.setdefault("per_suite", {})
        touched = False
        for suite in suite_names:
            if suite not in index:
                continue
            cell = per_suite.setdefault(suite, {})
            variant_dir = _variant_dir(index[suite], row)
            if variant_dir is not None and attach_by_seed(cell, variant_dir):
                touched = True
        if not touched:
            continue
        present = [per_suite[s] for s in suite_names if per_suite.get(s)]
        for key in row.get("metrics", []):
            for stem in (f"in_domain_{key}", f"transfer_{key}"):
                merged = mean_by_seed_across(present, stem)
                if merged is not None:
                    row[f"{stem}_by_seed"] = merged
        filled += 1

    return filled


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
                payload = _load(path)
                filled = _backfill_by_seed(path, payload)
                if filled:
                    print(f"  recovered per-seed spread for {filled} row(s)")
                draw(path.parent, payload)
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
