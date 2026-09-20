r"""Full-schema sanity check for the 4 core AI-detection parquets
(``ghostbuster_orig``, ``manugram_gb``, ``manugram_sci``,
``manugram_aiscientist``), against the canonical layout in
``cfg/ai_detection/core_parquet_columns.yaml`` (see
``epai/ai_detection/score/core_columns.py`` for how that's built).

``core_parquet_columns.yaml`` defines ``canonical_columns()``: the exact,
ordered, stem-agnostic list of columns every one of the 4 parquets must
carry, identically spelled. Nothing here is shape/pattern-matched -- a
column not on that list, and not in that stem's ``per_stem_extra`` allowlist
either, is EXTRA, full stop, regardless of whether it happens to *look like*
a judge column (a dropped experimental variant such as ``..._smin_0p05`` or
a differently-abbreviated ad hoc name is still shaped like one). This also
means name-matching across the 4 parquets falls out for free from checks
2-4 below -- a column required to be identical everywhere either is, or one
of those checks already catches it; there's no separate cross-parquet pass.

Six checks per parquet, schema/metadata first (cheap), data reads only
where a real answer needs one:

  1. DUPLICATE COLUMNS -- any column name repeated in the schema.
  2. ORDER -- canonical_columns(), restricted to columns actually present;
     everything else (per_stem_extra or genuinely unrecognized) is appended
     in whatever order it already has, not position-checked.
  3. MISSING -- a canonical_columns() name absent from the file.
  4. EXTRA -- a column present that's neither in canonical_columns() nor in
     this stem's per_stem_extra allowlist. One WARN with the count, then an
     INFO line naming each one.
  5. DUPLICATE ROWS -- the primary key (``uid``) must be unique.
  6. NULL / EMPTY -- every column's null count. Scalar columns read it from
     the parquet footer statistics (no data read); list-typed columns
     (embeddings, judge logprob windows, cue vectors) and string columns
     are read directly, since "list is null" vs "list is empty" and empty
     string can't be told apart from footer stats alone. A column that is
     entirely null/empty is an ERROR, UNLESS it's in core_parquet_columns.yaml's
     ``allow_null`` list (null on every row *on purpose* -- e.g.
     ``fastdetect_prob_*``, which only has real values for one pair; the
     actual detector score everything reads is ``fastdetect_crit_*``, which
     has none), in which case it's INFO instead. A partially-null column is
     always just reported for the count (not necessarily a bug -- some
     metadata is legitimately optional).

Plus two checks that aren't per-parquet:

  - SUITE ORDER DRIFT -- the embedders_order / model_pair_order /
    detectors_order lists in core_parquet_columns.yaml must cover exactly
    the same models as cfg/detector_suite.yaml (that file decides
    *membership*, this one decides column *order* -- see the yaml's header
    comment for why they're split).
  - CROSS-PARQUET NAME MATCH (runs last, needs all 4 parquets -- skipped
    under --only) -- belt-and-suspenders on top of checks 3-4: for every
    column name seen anywhere across the 4 files, it must be in all 4, or
    be declared in some stem's per_stem_extra. This should already follow
    from checks 3+4 passing on every stem individually, but it verifies
    that directly, on the raw counts, rather than trusting the per-stem
    checks agree with each other.

Run:
    uv run python epai/ai_detection/score/check_core_columns.py
    uv run python epai/ai_detection/score/check_core_columns.py --only manugram_sci
    uv run python epai/ai_detection/score/check_core_columns.py --fail-on warn
    uv run python epai/ai_detection/score/check_core_columns.py --drop-extra
    uv run python epai/ai_detection/score/check_core_columns.py --drop-extra --fix-order
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from epai.ai_detection.score.core_columns import (
    ALL_STEMS,
    ALLOW_NULL,
    CORE_STEMS,
    PRIMARY_KEY,
    allowed_extra,
    canonical_columns,
    expected_order,
    suite_order_drift,
)

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"

_SEVERITY = {"ERROR": 2, "WARN": 1, "INFO": 0}

_COLOR = {"ERROR": "\033[31m", "WARN": "\033[33m", "OK": "\033[32m"}  # red, yellow, green
_RESET = "\033[0m"
_USE_COLOR = sys.stdout.isatty()


def _level_tag(level: str) -> str:
    if _USE_COLOR and level in _COLOR:
        return f"{_COLOR[level]}[{level}]{_RESET}"
    return f"[{level}]"


def _issue(issues: list[tuple[str, str]], level: str, msg: str) -> None:
    issues.append((level, msg))
    print(f"  {_level_tag(level)} {msg}")


def _ok(msg: str) -> None:
    print(f"  {_level_tag('OK')} {msg}")


def _mark_if_clean(issues: list[tuple[str, str]], before: int, label: str) -> None:
    """Print an [OK] line iff the check just run (``len(issues)`` compared
    against its count ``before`` that check ran) found nothing to flag."""
    if len(issues) == before:
        _ok(label)


def _physical_column_index(pf: pq.ParquetFile, name: str) -> int:
    """Row-group physical column index for a flat (non-nested) field.

    Parquet's row-group columns are physical leaves, not logical Arrow
    fields -- a nested field (e.g. a list<struct> like the judge logprob
    windows) expands into >1 physical column, so a positional index built
    from ``schema_arrow.names`` silently drifts for every flat column that
    comes after one. Match by ``path_in_schema`` instead; a flat field's
    path is exactly its name (no ``.``).
    """
    rg = pf.metadata.row_group(0)
    for i in range(rg.num_columns):
        if rg.column(i).path_in_schema == name:
            return i
    raise KeyError(f"{name}: no flat physical column found (unexpectedly nested?)")


def _footer_null_count(pf: pq.ParquetFile, name: str) -> int:
    col_idx = _physical_column_index(pf, name)
    total = 0
    for i in range(pf.metadata.num_row_groups):
        st = pf.metadata.row_group(i).column(col_idx).statistics
        total += st.null_count if st is not None and st.null_count is not None else 0
    return total


def _extra_columns(names: list[str], stem: str) -> list[str]:
    """Columns present in the file that are neither in canonical_columns()
    nor in this stem's per_stem_extra allowlist."""
    allowed = set(canonical_columns()) | allowed_extra(stem)
    return [c for c in names if c not in allowed]


def _check_order(names: list[str], stem: str, issues: list[tuple[str, str]]) -> None:
    extra = _extra_columns(names, stem)
    expected = expected_order(stem, names) + extra
    if expected != names:
        _issue(issues, "WARN", "columns are not in the canonical order "
                                "(see cfg/ai_detection/core_parquet_columns.yaml)")
        # first few positions where the actual file and the canonical order
        # disagree -- enough to see where a column drifted without dumping
        # the full 60-80 column list.
        mismatches = [(i, a, e) for i, (a, e) in enumerate(zip(names, expected)) if a != e]
        for i, actual_col, expected_col in mismatches[:8]:
            _issue(issues, "INFO", f"  position {i}: has {actual_col!r}, expected {expected_col!r}")
        if len(mismatches) > 8:
            _issue(issues, "INFO", f"  ... and {len(mismatches) - 8} more position(s)")


def _check_extra(names: list[str], stem: str, issues: list[tuple[str, str]]) -> None:
    extra = _extra_columns(names, stem)
    if extra:
        _issue(issues, "WARN",
               f"{len(extra)} extra column(s) not covered by the canonical schema "
               f"(not meta/suite, not an llm_judge* column)")
        for c in extra:
            _issue(issues, "INFO", f"  {c}")


def _check_missing(names: list[str], issues: list[tuple[str, str]]) -> None:
    present = set(names)
    for c in canonical_columns():
        if c not in present:
            _issue(issues, "WARN", f"{c}: MISSING -- in the canonical schema but not in this file")


def _check_duplicate_columns(names: list[str], issues: list[tuple[str, str]]) -> None:
    seen: dict[str, int] = {}
    for c in names:
        seen[c] = seen.get(c, 0) + 1
    for c, n in seen.items():
        if n > 1:
            _issue(issues, "ERROR", f"{c}: column name appears {n} times in the schema")


def _check_duplicate_rows(src: Path, issues: list[tuple[str, str]]) -> None:
    table = pq.read_table(src, columns=[PRIMARY_KEY])
    n = table.num_rows
    col = table.column(PRIMARY_KEY)
    n_unique = pc.count_distinct(col).as_py()
    if n_unique < n:
        n_dupes = n - n_unique
        vc = pc.value_counts(col)
        dupes = [(row["values"].as_py(), row["counts"].as_py())
                 for row in vc if row["counts"].as_py() > 1]
        sample = ", ".join(f"{v!r}x{c}" for v, c in dupes[:5])
        _issue(issues, "ERROR",
               f"{PRIMARY_KEY}: {n_dupes} duplicate row(s) ({len(dupes)} distinct "
               f"{PRIMARY_KEY} value(s) repeated) -- e.g. {sample}")


def _check_nulls(src: Path, names: list[str], issues: list[tuple[str, str]]) -> None:
    pf = pq.ParquetFile(src)
    n_rows = pf.metadata.num_rows
    if n_rows == 0:
        return
    schema = pf.schema_arrow

    for name in names:
        field = schema.field(name)
        is_listy = pa.types.is_list(field.type) or pa.types.is_large_list(field.type)
        is_stringy = pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
        allowed = name in ALLOW_NULL
        entirely_level = "INFO" if allowed else "ERROR"
        entirely_suffix = " (expected -- see core_parquet_columns.yaml allow_null)" if allowed else ""

        if not is_listy and not is_stringy:
            null_count = _footer_null_count(pf, name)
            if null_count == n_rows:
                _issue(issues, entirely_level, f"{name}: entirely null ({n_rows} row(s)){entirely_suffix}")
            elif null_count > 0:
                _issue(issues, "INFO", f"{name}: {null_count}/{n_rows} null")
            continue

        col = pq.read_table(src, columns=[name]).column(name).combine_chunks()
        null_count = col.null_count
        if is_listy:
            lens = pc.list_value_length(col)
            empty_count = pc.sum(pc.equal(lens, 0)).as_py() or 0
        else:
            non_null = pc.fill_null(col, "\x00__NULL__\x00")
            empty_count = pc.sum(pc.equal(non_null, "")).as_py() or 0

        empty_kind = "empty-list" if is_listy else "empty-string"
        if null_count == n_rows:
            _issue(issues, entirely_level, f"{name}: entirely null ({n_rows} row(s)){entirely_suffix}")
        elif null_count + empty_count == n_rows and empty_count > 0:
            _issue(issues, entirely_level,
                   f"{name}: entirely null-or-{empty_kind} "
                   f"({null_count} null, {empty_count} {empty_kind}, {n_rows} row(s))"
                   f"{entirely_suffix}")
        else:
            if null_count > 0:
                _issue(issues, "INFO", f"{name}: {null_count}/{n_rows} null")
            if empty_count > 0:
                _issue(issues, "INFO", f"{name}: {empty_count}/{n_rows} {empty_kind}")


def _prompt_yes_no(question: str) -> bool:
    try:
        return input(question).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _rewrite_columns(src: Path, order: list[str]) -> None:
    """Rewrite ``src`` with exactly ``order`` as its columns (a subset
    drops the rest, any order permutes) -- write-to-temp-then-atomic-rename,
    so a crash or Ctrl-C mid-write can't leave a truncated/corrupt parquet
    in place of the original."""
    table = pq.read_table(src).select(order)
    tmp = src.with_name(src.name + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(src)


def _maybe_drop_extra(src: Path, names: list[str], stem: str, args: argparse.Namespace) -> list[str]:
    """Returns the column list ``names`` should be treated as from here on
    (unchanged if nothing was dropped, so later steps in this run --
    ``_maybe_fix_order`` -- see the file as it now actually is without a
    disk re-read)."""
    if not args.drop_extra:
        return names
    extra = _extra_columns(names, stem)
    if not extra:
        return names
    print()
    if not _prompt_yes_no(f"  Delete {len(extra)} extra column(s) from {src}? [y/N] "):
        print("  skipped -- no changes made")
        return names
    kept = [c for c in names if c not in extra]
    _rewrite_columns(src, kept)
    _ok(f"dropped {len(extra)} column(s) -- {src} now has {len(kept)} columns")
    return kept


def _target_order(names: list[str], stem: str) -> list[str]:
    return expected_order(stem, names) + _extra_columns(names, stem)


def _maybe_fix_order(src: Path, names: list[str], stem: str, args: argparse.Namespace) -> None:
    if not args.fix_order:
        return
    target = _target_order(names, stem)
    if target == names:
        return
    print()
    if not _prompt_yes_no(f"  Reorder columns in {src} to canonical order? [y/N] "):
        print("  skipped -- no changes made")
        return
    _rewrite_columns(src, target)
    _ok(f"reordered columns -- {src} now matches the canonical order")


def check_stem(stem: str, args: argparse.Namespace) -> tuple[list[tuple[str, str]], list[str]]:
    src = PARQUET_DIR / f"{stem}.parquet"
    print(f"\n{'-' * 74}\n{stem}  ({src})")
    issues: list[tuple[str, str]] = []
    if not src.is_file():
        _issue(issues, "ERROR", f"not on disk: {src}")
        return issues, []

    names = list(pq.read_schema(src).names)
    print(f"  {len(names)} columns")

    before = len(issues)
    _check_duplicate_columns(names, issues)
    _mark_if_clean(issues, before, "no duplicate column names")

    before = len(issues)
    _check_order(names, stem, issues)
    _mark_if_clean(issues, before, "columns are in canonical order")

    before = len(issues)
    _check_missing(names, issues)
    _mark_if_clean(issues, before, "no missing canonical columns")

    before = len(issues)
    _check_extra(names, stem, issues)
    _mark_if_clean(issues, before, "no extra columns")

    before = len(issues)
    _check_duplicate_rows(src, issues)
    _mark_if_clean(issues, before, f"{PRIMARY_KEY} is unique (no duplicate rows)")

    before = len(issues)
    _check_nulls(src, names, issues)
    _mark_if_clean(issues, before, "no null/empty columns")

    n_err = sum(1 for lvl, _ in issues if lvl == "ERROR")
    n_warn = sum(1 for lvl, _ in issues if lvl == "WARN")
    n_info = sum(1 for lvl, _ in issues if lvl == "INFO")
    if n_err == 0 and n_warn == 0:
        _ok("all checks clean" + (f" ({n_info} info)" if n_info else ""))
    else:
        print(f"  {n_err} error(s), {n_warn} warning(s), {n_info} info")

    names = _maybe_drop_extra(src, names, stem, args)
    _maybe_fix_order(src, names, stem, args)
    return issues, names


def check_suite_drift() -> list[tuple[str, str]]:
    print(f"\n{'-' * 74}\nsuite order vs. detector_suite.yaml membership")
    issues: list[tuple[str, str]] = []
    drift = suite_order_drift()
    if not drift:
        _ok("embedders_order / model_pair_order / detectors_order match detector_suite.yaml")
        return issues
    for group, (missing, extra) in drift.items():
        if missing:
            _issue(issues, "ERROR",
                   f"{group}: in detector_suite.yaml but missing from its "
                   f"*_order list in core_parquet_columns.yaml: {sorted(missing)}")
        if extra:
            _issue(issues, "ERROR",
                   f"{group}: in an *_order list but not in detector_suite.yaml "
                   f"anymore: {sorted(extra)}")
    return issues


def _known_variable_columns() -> set[str]:
    """Every column name declared in some stem's per_stem_extra -- the only
    names :func:`check_cross_parquet_names` doesn't require in all 4 files."""
    return set().union(*(allowed_extra(s) for s in CORE_STEMS))


def check_cross_parquet_names(stem_names: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Every column name should either appear in all 4 parquets, or be
    declared in some stem's per_stem_extra -- anything else present in only
    some of the 4 is a name that drifted (a typo'd rename, an ad hoc variant
    added to one corpus and not backfilled to the others, ...)."""
    print(f"\n{'-' * 74}\ncolumn names across all 4 parquets")
    issues: list[tuple[str, str]] = []
    expected_variation = _known_variable_columns()
    stems = list(stem_names)

    all_columns = sorted(set().union(*(set(v) for v in stem_names.values())))
    for col in all_columns:
        present = [s for s in stems if col in stem_names[s]]
        if len(present) == len(stems) or col in expected_variation:
            continue
        missing = [s for s in stems if s not in present]
        _issue(issues, "ERROR",
               f"{col}: in {len(present)}/{len(stems)} parquet(s) "
               f"({', '.join(present)}) -- missing from {', '.join(missing)}")

    _mark_if_clean(issues, 0, f"all non-per_stem_extra column names match across "
                               f"the {len(stems)} parquets")
    return issues


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="+", choices=ALL_STEMS, metavar="STEM")
    ap.add_argument("--fail-on", choices=["error", "warn"], default="error",
                     help="exit 1 threshold (default: error only)")
    ap.add_argument("--drop-extra", action="store_true",
                     help="after checking a parquet, if it has extra columns "
                          "(check 4), ask [y/N] whether to rewrite the file "
                          "without them. Prompts once per parquet, at the end "
                          "of that parquet's report; 'n' or Ctrl-D leaves it "
                          "untouched. Rewrites in place (write-to-temp then "
                          "atomic rename).")
    ap.add_argument("--fix-order", action="store_true",
                     help="after checking a parquet (and after --drop-extra's "
                          "prompt, if also given), if its columns aren't in "
                          "canonical order, ask [y/N] whether to rewrite the "
                          "file with them reordered. Any column not in the "
                          "canonical schema (still extra, because --drop-extra "
                          "wasn't given or its prompt was declined) is placed "
                          "at the end, same as the order check's diff does.")
    args = ap.parse_args()

    all_issues: list[tuple[str, str]] = check_suite_drift()
    stem_names: dict[str, list[str]] = {}
    for stem in (args.only or CORE_STEMS):
        issues, names = check_stem(stem, args)
        all_issues += issues
        stem_names[stem] = names

    if args.only:
        print(f"\n{'-' * 74}\ncolumn names across all 4 parquets")
        print(f"  skipped -- needs all {len(CORE_STEMS)} parquets, --only was given")
    else:
        all_issues += check_cross_parquet_names(stem_names)

    print(f"\n{'=' * 74}")
    n_err = sum(1 for lvl, _ in all_issues if lvl == "ERROR")
    n_warn = sum(1 for lvl, _ in all_issues if lvl == "WARN")
    n_info = sum(1 for lvl, _ in all_issues if lvl == "INFO")
    print(f"{n_err} error(s), {n_warn} warning(s), {n_info} info across "
          f"{len(args.only or CORE_STEMS)} parquet(s)")

    threshold = 1 if args.fail_on == "warn" else 2
    sys.exit(1 if any(_SEVERITY[lvl] >= threshold for lvl, _ in all_issues) else 0)


if __name__ == "__main__":
    main()
