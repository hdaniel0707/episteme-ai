r"""Several datasets in one config: the suite level, and the comparison across it.

A cross-domain config has always had one list of corpora, and the matrix between
them is the result: the diagonal is in-domain, the off-diagonal is transfer. That
answers "does this detector survive leaving its domain" for ONE dataset. It does
not answer the question after it -- does the ranking it produced hold on a
different corpus, or against a different generator -- because that needs four
such matrices and there was no way to ask for more than one.

A SUITE is one of those datasets: a parquet, and the corpora cut out of it that
become the axes of its matrix. ``data.suites`` is a list of them, and everything
else in the config -- ``feature_sets``, ``algorithms``, the baselines, ``model``
-- is shared by all of them, which is the whole point: a variant is trained the
same way on every suite, so the four transfer numbers it produces are comparable
and can be ranked together.

    data:
      human_label: "human"        # shared: the run's binarisation
      group_col: "id"             # shared: the split's grouping column
      suites:
        - name: "ghostbuster_gpt35t"
          parquet_path: "data/parquet/ghostbuster_gpt35t.parquet"
          datasets:               # ← unchanged: the axes of THIS suite's matrix
            - {name: "essay",  where: {domain: "essay"}}
            - {name: "reuter", where: {domain: "reuter"}}
        - name: "science_v3"
          parquet_path: "data/parquet/science_v3_gpt56luna_0811A.parquet"
          datasets:
            - {name: "computer_science", where: {domain: "computer_science"}}
            - {name: "mathematics",      where: {domain: "mathematics"}}

THE IMPLEMENTATION IS A SPLIT, NOT A NEW PIPELINE. :func:`suite_configs` turns
one config with N suites into N ordinary configs, each with the flat
``data.datasets`` the runner has always taken. Everything downstream -- the
preflight, the loader, the matrix, the per-variant figures, the within-suite
ranking -- runs unchanged on each of them and cannot tell the difference. What is
genuinely new is only this module's second half: collapsing the N within-suite
comparisons into one cross-suite comparison.

WHICH KEYS ARE SHARED. ``human_label`` and ``group_col`` describe the run's
question (what counts as human, what must not be split across the train/test
boundary) and are set once under ``data``. ``parquet_path``, ``where`` and
``drop_classes`` keep the "set a default, override where needed" behaviour they
already had, gaining one more level to be set at: ``data`` → suite → dataset,
innermost wins. ``datasets`` is per suite and only per suite -- it *is* the suite.

A CONFIG WITHOUT ``suites`` IS UNCHANGED. The two shapes are told apart by
:func:`has_suites`, and the flat one takes exactly the path it always did, down
to the directory layout. Nothing here runs for it.

AGGREGATION. Each (variant, suite) pair contributes the row
``cd_core.summarise_variant`` already builds -- its in-domain mean, its transfer
mean, its worst transfer cell -- and :func:`suite_comparison_rows` stacks those
into one row per variant. The cross-suite aggregates are stored under the SAME
key names (``transfer_auroc_mean`` and friends), so every reader written for a
within-suite comparison row reads a cross-suite one without changes; what the
keys mean is one level up, and the two facts that need saying are said by the
figures and by :func:`print_suite_ranking`:

  * the mean is UNWEIGHTED over suites -- each dataset counts once, whatever its
    row count and however many corpora its matrix has. A weighted mean would let
    the largest corpus decide the ranking, which is the opposite of the question;
  * ``transfer_<metric>_std`` is the spread ACROSS SUITES, not across the cells
    of one matrix as it is within a suite. It is the number that says whether a
    variant's transfer is a property of the variant or of the dataset it was
    measured on;
  * ``transfer_<metric>_worst`` is likewise the WORST SUITE, not the worst cell.
    A mean over four datasets hides one collapsed dataset exactly as a mean over
    six off-diagonal cells hides one collapsed corpus pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from epai.utils.cd_core import BY_SEED_SUFFIX, cell_metric_specs
from epai.utils.corpus_utils import check_dataset_specs, check_where
from epai.utils.preflight_utils import RULE

#: Where the suites live in the config, and what a config is recognised by.
SUITE_KEY = "suites"

#: ``data`` keys that describe the run rather than one suite. Set once, under
#: ``data``; a suite that set them would be describing a different experiment,
#: not a different dataset, and its matrix would not be comparable with the rest.
SHARED_DATA_KEYS: tuple[str, ...] = ("human_label", "group_col")

#: ``data`` keys a suite may set for itself, falling back to the ``data`` level.
#: The same defaulting the datasets already have, one level out.
SUITE_DATA_KEYS: tuple[str, ...] = ("parquet_path", "where", "drop_classes")

#: Keys of a suite entry that are the suite's identity rather than its data.
SUITE_META_KEYS: tuple[str, ...] = ("name", "label", "datasets")


# ---------------------------------------------------------------------------
# Splitting one config into one config per suite
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Suite:
    """One suite, and the ordinary cross-domain config that runs it."""

    name: str
    label: str
    cfg: dict


def has_suites(cfg) -> bool:
    """True if this config declares suites rather than a flat ``data.datasets``."""
    return (
        isinstance(cfg, dict)
        and isinstance(cfg.get("data"), dict)
        and bool(cfg["data"].get(SUITE_KEY))
    )


def suite_entries(cfg: dict) -> list[dict]:
    """The raw ``data.suites`` entries, or an empty list."""
    data = cfg.get("data")
    if not isinstance(data, dict):
        return []
    entries = data.get(SUITE_KEY) or []
    return list(entries) if isinstance(entries, (list, tuple)) else []


def suite_configs(cfg: dict) -> list[Suite]:
    """One config per suite: the shared axes, this suite's data block.

    The returned configs are what the runner and the preflight actually see, and
    each is a complete, ordinary cross-domain config -- which is why nothing
    downstream needs to know suites exist. ``alg_id`` and ``alg_label`` carry the
    suite so that a per-variant figure drawn deep inside a suite directory still
    says which dataset it is about; the parent's own ``alg_id`` is what names the
    output directory and is left to the runner.

    The parent config is not mutated and is not shared by reference beyond its
    immutable parts: only ``data`` is rebuilt, which is the only part that
    differs.
    """
    data = cfg.get("data") or {}
    suites: list[Suite] = []

    for entry in suite_entries(cfg):
        if not isinstance(entry, dict) or "name" not in entry:
            continue  # check_suites reports this; building on it would only crash
        name = str(entry["name"])
        label = str(entry.get("label") or name)

        # data-level defaults first, then the suite's own, so the suite wins on
        # any key it sets -- the same precedence a dataset entry has over `data`.
        suite_data = {key: data[key] for key in SHARED_DATA_KEYS if key in data}
        for key in SUITE_DATA_KEYS:
            if data.get(key) is not None:
                suite_data[key] = data[key]
            if entry.get(key) is not None:
                suite_data[key] = entry[key]
        # A malformed entry still yields a usable config object -- the empty
        # list is what check_suites has already reported, and building on it
        # lets the rest of the preflight finish instead of raising here.
        suite_data["datasets"] = entry.get("datasets") or []

        sub = dict(cfg)
        sub["data"] = suite_data
        sub["alg_id"] = f"{cfg.get('alg_id', 'run')}__{name}"
        sub["alg_label"] = f"{cfg.get('alg_label', cfg.get('alg_id', 'run'))} — {label}"
        # Read by nothing that fits a model; they exist so a suite's own JSON
        # says which suite it is without having to parse its alg_id back apart.
        sub["suite"] = name
        sub["suite_label"] = label
        suites.append(Suite(name=name, label=label, cfg=sub))

    return suites


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def check_suites(cfg: dict, report, min_suites: int = 1) -> None:
    """Validate ``data.suites``, then each suite as the config it becomes.

    Passed to ``variant_utils.check_config`` as its ``data_check`` when the
    config declares suites, so the shared axes are still checked exactly once
    while the data block is checked once per suite. The per-suite errors are
    tagged with the suite name: with four datasets in one file, "missing column
    emb_luar_crud" without one would not say which parquet to fix.
    """
    data = cfg.get("data")
    if not isinstance(data, dict):
        report.error("'data' must be a mapping")
        return

    for key in SHARED_DATA_KEYS:
        if key not in data:
            report.error(f"data is missing {key!r}")

    if "datasets" in data:
        # Both shapes at once: one of them would silently not run.
        report.error(
            "data sets both 'datasets' and 'suites' — 'datasets' is the flat "
            "single-dataset shape and 'suites' is the multi-dataset one, so keep "
            "only one. To run the flat config as a suite, move its 'datasets' "
            "list (and its 'parquet_path') into a suite entry."
        )
        return

    entries = data.get(SUITE_KEY)
    if not isinstance(entries, (list, tuple)):
        report.error(
            f"'data.suites' must be a list of suite entries, got "
            f"{type(entries).__name__ if entries is not None else 'nothing'}"
        )
        return
    if len(entries) < min_suites:
        report.error(
            f"'data.suites' needs at least {min_suites} suite(s), got {len(entries)}"
        )
        return

    check_where(data.get("where"), "data", report)
    if data.get("drop_classes") is not None and not isinstance(
        data["drop_classes"], (list, tuple)
    ):
        report.error("data 'drop_classes' must be a list when given")

    names: list[str] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            report.error(f"suites[{i}] must be a mapping")
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            report.error(f"suites[{i}] is missing a non-empty 'name'")
        else:
            names.append(name)
        if not isinstance(entry.get("datasets"), (list, tuple)):
            report.error(
                f"suites[{i}] ({name!r}) is missing 'datasets' — a suite is the "
                f"corpora its own transfer matrix runs between"
            )
        check_where(entry.get("where"), f"suites[{i}]", report)
        if entry.get("drop_classes") is not None and not isinstance(
            entry["drop_classes"], (list, tuple)
        ):
            report.error(f"suites[{i}] 'drop_classes' must be a list when given")
        unknown = sorted(
            set(entry) - set(SUITE_META_KEYS) - set(SUITE_DATA_KEYS)
        )
        if unknown:
            report.error(
                f"suites[{i}] ({name!r}) has unknown key(s) {unknown}; a suite "
                f"entry takes {sorted(SUITE_META_KEYS + SUITE_DATA_KEYS)}"
            )

    # Each suite names an output directory under the run's, so a repeat would
    # have two datasets writing into one place.
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        report.error(f"suite names must be unique, repeated: {dupes}")

    # Now the part that is genuinely a dataset check, per suite, through exactly
    # the same function the flat shape uses.
    outer = report.context
    for suite in suite_configs(cfg):
        if not suite.cfg["data"]["datasets"]:
            continue  # already reported above; a second error would only repeat it
        report.context = f"{outer} · {suite.name}" if outer else suite.name
        check_dataset_specs(suite.cfg, report)
    report.context = outer


# ---------------------------------------------------------------------------
# Aggregation across suites
# ---------------------------------------------------------------------------


def _nan_stat(values: np.ndarray, fn) -> float:
    """``fn`` over the finite entries, NaN when there are none."""
    finite = values[np.isfinite(values)]
    return float(fn(finite)) if finite.size else float("nan")


def mean_by_seed_across(rows: list[dict | None], stem: str) -> list[float] | None:
    """One ``<stem>_by_seed`` list averaged elementwise over several datasets.

    A dataset's ``_by_seed`` entry ``i`` is seed ``i``'s aggregate, and the run
    uses the same seed list for every dataset, so entry ``i`` means the same
    thing in all of them and they can be averaged position by position. The
    result is what seed ``i`` scored over the whole suite, which is the thing the
    cross-dataset figure needs an error bar on -- as opposed to averaging each
    dataset's spread, which would answer a different and much smaller question.

    None when no dataset carries the key, or when they disagree on how many
    seeds they ran: entry ``i`` would then not be one seed, and an error bar
    drawn from it would be fiction.
    """
    lists = [
        r[f"{stem}{BY_SEED_SUFFIX}"]
        for r in rows
        if r is not None and isinstance(r.get(f"{stem}{BY_SEED_SUFFIX}"), list)
    ]
    if not lists or len({len(v) for v in lists}) != 1:
        return None
    stacked = np.array(lists, dtype=float)
    with np.errstate(invalid="ignore"):
        return [float(v) for v in np.nanmean(stacked, axis=0)]


def suite_comparison_rows(
    per_suite: dict[str, list[dict]], suite_names: list[str]
) -> list[dict]:
    """One row per variant, holding its per-suite numbers and their aggregates.

    ``per_suite`` maps a suite name to the comparison rows that suite produced
    (``cd_core.summarise_variant`` / ``summarise_baseline`` output). Variants are
    matched across suites by their ``variant`` id, which is shared by
    construction: the ``feature_sets`` x ``algorithms`` axes are declared once
    for the whole run. A variant absent from a suite still gets a row, with that
    suite's cells NaN, so a partial run reads as a gap rather than as a silently
    shorter mean.

    The aggregate keys deliberately match the within-suite ones -- see the module
    docstring -- so ``print_metric_ranking`` and the comparison figures read this
    row unchanged. ``per_suite`` on each row is where the individual datasets
    live, and is what the cross-suite figures draw.
    """
    specs = {m.key: m for m in cell_metric_specs()}

    # First-seen order, so the rows follow the config's `algorithms` list rather
    # than an alphabetical order nobody wrote.
    order: list[str] = []
    seen: dict[str, dict] = {}
    for name in suite_names:
        for row in per_suite.get(name, []):
            vid = row.get("variant") or row.get("label", "")
            if vid not in seen:
                order.append(vid)
                seen[vid] = row

    rows: list[dict] = []
    for vid in order:
        first = seen[vid]
        by_suite = {
            name: next(
                (
                    r
                    for r in per_suite.get(name, [])
                    if (r.get("variant") or r.get("label", "")) == vid
                ),
                None,
            )
            for name in suite_names
        }
        present = [r for r in by_suite.values() if r is not None]

        # Every metric any suite reported for this variant. Union rather than
        # intersection: a metric missing from one suite is a NaN cell there, and
        # dropping the metric entirely would lose the suites that do have it.
        metric_keys = [
            key
            for key in specs
            if any(f"transfer_{key}_mean" in r for r in present)
        ]

        row: dict = {
            "variant": vid,
            "label": first.get("label", vid),
            "feature_dim": first.get("feature_dim", 0),
            "metrics": metric_keys,
            "baseline": bool(first.get("baseline")),
            "n_suites": len(present),
            "per_suite": {},
        }
        if first.get("baseline_kind"):
            row["baseline_kind"] = first["baseline_kind"]
        if first.get("column"):
            # Which score a baseline row reads -- a family contributes one row
            # per column, so this is what keeps two of them apart downstream.
            row["column"] = first["column"]
        if first.get("hybrid"):
            row["hybrid"] = True
            row["aux_column"] = first.get("aux_column")

        for name in suite_names:
            src = by_suite[name]
            cell: dict = {}
            for key in metric_keys:
                for stem in (
                    f"in_domain_{key}_mean",
                    f"in_domain_{key}_std",
                    f"transfer_{key}_mean",
                    f"transfer_{key}_std",
                    f"transfer_{key}_worst",
                    f"{key}_gap",
                ):
                    cell[stem] = (
                        float(src[stem])
                        if src is not None and stem in src
                        else float("nan")
                    )
                # Carried through unchanged so a dataset's own line on the
                # cross-dataset figure gets the same error bar as anything else.
                for stem in (f"in_domain_{key}", f"transfer_{key}"):
                    values = (src or {}).get(f"{stem}{BY_SEED_SUFFIX}")
                    if isinstance(values, list):
                        cell[f"{stem}{BY_SEED_SUFFIX}"] = list(values)
            row["per_suite"][name] = cell

        for key in metric_keys:
            spec = specs[key]
            in_dom = np.array(
                [row["per_suite"][n][f"in_domain_{key}_mean"] for n in suite_names]
            )
            transfer = np.array(
                [row["per_suite"][n][f"transfer_{key}_mean"] for n in suite_names]
            )
            worst_cells = np.array(
                [row["per_suite"][n][f"transfer_{key}_worst"] for n in suite_names]
            )

            in_mean = _nan_stat(in_dom, np.mean)
            tr_mean = _nan_stat(transfer, np.mean)
            row[f"in_domain_{key}_mean"] = in_mean
            row[f"in_domain_{key}_std"] = _nan_stat(in_dom, np.std)
            row[f"transfer_{key}_mean"] = tr_mean
            # Across suites, not across cells: how much the dataset decides.
            row[f"transfer_{key}_std"] = _nan_stat(transfer, np.std)
            # The worst SUITE. A mean over four datasets hides one that collapsed
            # exactly as a mean over six cells hides one corpus pair.
            row[f"transfer_{key}_worst"] = _nan_stat(
                transfer, np.min if spec.higher_is_better else np.max
            )
            # ...and the worst single cell anywhere in the run, which is the one
            # a deployed detector would actually meet.
            row[f"transfer_{key}_worst_cell"] = _nan_stat(
                worst_cells, np.min if spec.higher_is_better else np.max
            )
            row[f"{key}_gap"] = (
                in_mean - tr_mean if spec.higher_is_better else tr_mean - in_mean
            )
            # The suite-level seed aggregate: seed i's score over every dataset,
            # not the datasets' spreads pooled. See mean_by_seed_across.
            for stem in (f"in_domain_{key}", f"transfer_{key}"):
                merged = mean_by_seed_across(present, stem)
                if merged is not None:
                    row[f"{stem}{BY_SEED_SUFFIX}"] = merged

        # The unqualified accuracy aliases every existing reader still names.
        if "accuracy" in metric_keys:
            row.update(
                in_domain_mean=row["in_domain_accuracy_mean"],
                in_domain_std=row["in_domain_accuracy_std"],
                transfer_mean=row["transfer_accuracy_mean"],
                transfer_std=row["transfer_accuracy_std"],
                transfer_min=row["transfer_accuracy_worst"],
                gap=row["accuracy_gap"],
            )
        rows.append(row)

    return rows


def suite_metric_keys(rows: list[dict]) -> list[str]:
    """Metrics these cross-suite rows can be reported under, in reporting order."""
    return [
        m.key
        for m in cell_metric_specs()
        if any(f"transfer_{m.key}_mean" in r for r in rows)
    ]


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _fmt(value, width: int = 8) -> str:
    """A rate as a percentage, or an em dash where there is nothing to print."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{'—':>{width}}"
    if not np.isfinite(number):
        return f"{'—':>{width}}"
    return f"{number*100:>{width-1}.1f}%"


def print_suite_ranking(
    alg_label: str,
    rows: list[dict],
    suite_names: list[str],
    suite_labels: dict,
    noun: str = "variant",
) -> None:
    """Rank the variants across every suite, one table per metric.

    The first table is accuracy and carries the full aggregate set (mean, worst
    suite, spread across suites, in-domain, gap); the rest are transfer only,
    since the in-domain column is near-perfect for every trained variant and
    printing it nine times over would say nothing nine times.

    Reading it: a variant whose per-suite columns are flat has a property of the
    variant, and one whose columns swing has a property of the datasets. The
    ``sd`` column is that reading as a number, and it is the reason the mean is
    never printed on its own.
    """
    if not rows:
        return

    specs = [m for m in cell_metric_specs() if m.key in set(suite_metric_keys(rows))]
    if not specs:
        return

    width = max(26, max((len(r["label"]) for r in rows), default=0) + 4)
    col = min(18, max(11, max((len(n) for n in suite_names), default=0) + 2))
    # Headers are the suite NAMES, not their labels: the names are unique by
    # construction and the labels frequently are not once trimmed to a column
    # ("Ghostbuster (gpt-3.5-turbo)" and "Ghostbuster (gpt-5.4-mini)" cut to the
    # same header, which would silently mislabel three of four columns). Trimming
    # keeps both ends for the same reason -- what tells these apart is the tail.
    heads = {n: _short(n, col) for n in suite_names}
    partial = [r for r in rows if r.get("n_suites", len(suite_names)) < len(suite_names)]

    print()
    print("=" * RULE)
    print(f"  {alg_label} — {len(suite_names)} DATASETS, EACH WITH ITS OWN "
          f"IN-DOMAIN / TRANSFER SPLIT")
    print("=" * RULE)
    for name in suite_names:
        print(f"  {heads[name]:<{col}}  {name:<26} {suite_labels.get(name, name)}")
    print(
        "\n  No model is trained on one dataset and tested on another: each column "
        "is that\n  dataset's own cross-domain matrix collapsed to a single "
        "transfer number."
    )

    for spec in specs:
        key = spec.key
        order = sorted(
            rows,
            key=lambda r: (
                _num(r.get(f"transfer_{key}_mean")) if spec.higher_is_better
                else -_num(r.get(f"transfer_{key}_mean"))
            ),
            reverse=True,
        )
        header = (
            f"{noun:<{width}}"
            + "".join(f"{heads[n]:>{col}}" for n in suite_names)
            + f"{'MEAN':>9}{'worst':>8}{'sd':>7}"
            + (f"{'in-dom':>9}{'gap':>7}" if key == "accuracy" else "")
        )
        rule = max(RULE, len(header))
        print()
        print("=" * rule)
        print(
            f"  {alg_label} — {noun.upper()}S ACROSS {len(suite_names)} DATASETS: "
            f"TRANSFER {spec.label.upper()}"
        )
        print("=" * rule)
        print(header)
        print("-" * rule)
        for r in order:
            mark = "★ " if r.get("baseline") else ("◆ " if r.get("hybrid") else "  ")
            # A mean over fewer datasets than the run covered is not the same
            # number as the rows around it, so it never sits unmarked among them.
            name_cell = mark + r["label"] + (
                f"  (!{r.get('n_suites')}/{len(suite_names)})"
                if r in partial else ""
            )
            line = f"{name_cell:<{width}}"
            for name in suite_names:
                line += _fmt(r["per_suite"].get(name, {}).get(f"transfer_{key}_mean"), col)
            line += _fmt(r.get(f"transfer_{key}_mean"), 9)
            line += _fmt(r.get(f"transfer_{key}_worst"), 8)
            line += _fmt(r.get(f"transfer_{key}_std"), 7)
            if key == "accuracy":
                line += _fmt(r.get(f"in_domain_{key}_mean"), 9)
                line += _fmt(r.get(f"{key}_gap"), 7)
            print(line)
        print("-" * rule)
        arrow = "higher is better" if spec.higher_is_better else "LOWER is better"
        print(
            f"  per-dataset columns = that dataset's own transfer (off-diagonal) mean; "
            f"reference {spec.reference*100:g}%, {arrow}"
        )
        print(
            "  MEAN  = unweighted mean over datasets — each counts once, whatever "
            "its size\n"
            "  worst = the worst DATASET, not the worst corpus pair inside one\n"
            "  sd    = spread ACROSS datasets: small means the variant decides the "
            "score, large means the dataset does"
        )
        if spec.note:
            print(f"\n  {spec.note}")

    if partial:
        print(
            f"\n  ! covers fewer than the {len(suite_names)} datasets of the run, so "
            f"its MEAN is not taken over the same\n    evidence as the rows around "
            f"it. A trained variant cannot end up here (the axes are\n    declared "
            f"once for every dataset); a ★ baseline whose score column is missing "
            f"from a\n    parquet is stopped by the preflight rather than reduced "
            f"to this."
        )
    if any(r.get("baseline") for r in rows):
        print("\n  ★ zero-shot baseline: no classifier trained")
    hybrids = [r for r in rows if r.get("hybrid")]
    if hybrids:
        cols = sorted({r.get("aux_column") or "?" for r in hybrids})
        print(
            f"  ◆ hybrid: trained on the shared features PLUS {', '.join(cols)}, so "
            f"it is not ranked on equal information"
        )


def _short(text: str, width: int) -> str:
    """A header trimmed to fit, keeping BOTH ends.

    Dataset names here differ by their suffix -- ``ghostbuster_gpt35t`` against
    ``ghostbuster_gpt54mini`` -- so trimming from the right is exactly the wrong
    end to lose. The ellipsis goes in the middle instead.
    """
    text = str(text)
    keep = width - 1
    if len(text) <= keep:
        return text
    head = (keep - 1) // 2
    return text[:head] + "…" + text[-(keep - 1 - head):]


def _num(value, default: float = float("nan")) -> float:
    """One aggregate as a float; missing or unparseable becomes NaN."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
