#!/usr/bin/env python3
"""Is this parquet fully and sanely populated? One report, any schema, an exit code.

Column by column: nulls, empty strings / arrays, non-finite numbers, and — the
check a null count misses — **degenerate columns**: a score or embedding column
that came out constant, near-constant, all-zero, or ragged. That is the shape a
flaky scoring endpoint leaves behind (every row "filled" with 0.5), and nothing
downstream notices until a figure comes out at AUROC 0.5.

Not tied to the training schema (that is ``epai/ai_detection/analyse/audit_corpus.py``,
which also checks pairing, balance and text quality). This one answers the
narrower question: did every stage that was supposed to write a column actually
write a real value into every row.

Runs on the repo interpreter — ``pyarrow`` + ``numpy``, one column in memory at
a time (~0.8 GB, ~20 s on a 320 MB / 6000-row corpus).

    .venv/bin/python epai/scripts/check_parquet_completeness.py data/parquet/denario_hc.parquet

    # does this file have every column science_v3 has?
    .venv/bin/python epai/scripts/check_parquet_completeness.py data/parquet/denario_hc.parquet \
        --reference data/parquet/science_v3_gpt56luna_0811A.parquet

    # gate a pipeline: exit 1 on any ERROR (or on WARN too, with --strict)
    .venv/bin/python epai/scripts/check_parquet_completeness.py data/parquet/denario_hc.parquet \
        --fail-on error  --json results/denario_hc/completeness.json

Exit code: 0 if clean (or only WARN, unless --strict / --fail-on warn), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Bounded score columns — P(...) that must sit in [0, 1]. Anything outside is an
# ERROR, not a WARN.
BOUNDED_SCORE_PREFIXES = ("llm_judge_score_", "detector_score_")
# Every column a scorer/embedder fills with a meaningful value: a constant or
# near-constant one of these is a scorer that fell back to a default.
SCORE_PREFIXES = BOUNDED_SCORE_PREFIXES + (
    "binoculars_score", "fastdetect_crit_",
)

# Fixed-width vector columns: ragged / all-zero / NaN-inside are errors here.
# Other list columns (`*_cues_*`, `author_ids`, `queries`) are legitimately
# variable length.
VECTOR_PREFIXES = ("emb_",)
VECTOR_SUFFIXES = ("_cuevec_gpt56luna_v3",)
VECTOR_CONTAINS = ("_cuevec_",)

# Allowed to be entirely null — upstream does not always fill them and says so
# where they are written.
OPTIONAL_NULL_PREFIXES = ("fastdetect_prob_", "fastdetect_ntok_")

# Share of a score column's rows sitting on one value: at/above CONSTANT it is
# an ERROR (the scorer wrote a default), at/above SUSPICIOUS a WARN (could be a
# genuinely clumpy judge — worth a look, not necessarily broken).
CONSTANT_FRAC = 0.99
SUSPICIOUS_FRAC = 0.80

# Text columns whose emptiness is a real problem, and whose length is worth a
# distribution.
TEXTISH = ("text", "text_raw", "text_preclean", "abstract", "abstract_raw")
TITLEISH = ("title",)

# Columns where a partial null is a real gap. Everywhere else a partial null is
# an expected consequence of the two classes carrying different metadata (only
# machine rows have a `prompt`, only some works have a `doi`), so it is a WARN.
REQUIRED_COLS = ("uid", "id", "text", "generated_by", "domain", "y", "group")


def _partial_null_is_error(name: str) -> bool:
    return name in REQUIRED_COLS or name.startswith(SCORE_PREFIXES) or _is_vector(name)


def _null_line(name: str, nulls: int, n: int) -> str:
    pct = 100 * nulls / n
    shown = f"{pct:.1f}%" if pct >= 0.1 else f"{pct:.2g}%"
    return f"{name}: {nulls}/{n} null ({shown}) — partial fill"


def _is_vector(name: str) -> bool:
    return (name.startswith(VECTOR_PREFIXES)
            or name.endswith(VECTOR_SUFFIXES)
            or any(s in name for s in VECTOR_CONTAINS))


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warns: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        print(f"    {RED}ERROR{RESET} {msg}")

    def warn(self, msg: str) -> None:
        self.warns.append(msg)
        print(f"    {YELLOW}WARN {RESET} {msg}")

    def ok(self, msg: str) -> None:
        print(f"    {GREEN}ok   {RESET}{DIM} {msg}{RESET}")

    def detail(self, msg: str) -> None:
        print(f"    {DIM}       {msg}{RESET}")


# ---------------------------------------------------------------------------


def _fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "NA"
    return f"{x:.4g}"


def _is_optional_null(name: str) -> bool:
    return any(name.startswith(p) for p in OPTIONAL_NULL_PREFIXES)


def check_scalar(name: str, arr: pa.ChunkedArray, n: int, rep: Report) -> dict:
    """Nulls, distinctness and — for numbers — range and non-finite values."""
    nulls = arr.null_count
    info: dict = {"nulls": nulls}
    is_score = name.startswith(SCORE_PREFIXES)

    if nulls == n:
        if _is_optional_null(name):
            rep.warn(f"{name}: entirely null ({n} rows) — expected for this column")
        elif is_score:
            rep.error(f"{name}: entirely null ({n} rows) — the scorer wrote nothing")
        else:
            rep.warn(f"{name}: entirely null ({n} rows) — unused column?")
        return info
    if nulls:
        (rep.error if _partial_null_is_error(name) else rep.warn)(_null_line(name, nulls, n))

    non_null = arr.drop_null()
    n_distinct = len(pa.compute.unique(non_null))
    info["distinct"] = n_distinct

    if pa.types.is_floating(arr.type) or pa.types.is_integer(arr.type):
        v = non_null.to_numpy(zero_copy_only=False).astype(float)
        n_nan = int(np.isnan(v).sum())
        n_inf = int(np.isinf(v).sum())
        finite = v[np.isfinite(v)]
        info.update(
            nan=n_nan, inf=n_inf,
            min=float(finite.min()) if finite.size else None,
            max=float(finite.max()) if finite.size else None,
            mean=float(finite.mean()) if finite.size else None,
            std=float(finite.std()) if finite.size else None,
            zeros=int((finite == 0).sum()),
        )
        if n_nan:
            rep.error(f"{name}: {n_nan} NaN value(s) in a non-null column")
        if n_inf:
            rep.error(f"{name}: {n_inf} infinite value(s)")

        bounded = name.startswith(BOUNDED_SCORE_PREFIXES)
        if bounded and finite.size and (finite.min() < 0 or finite.max() > 1):
            rep.error(
                f"{name}: values outside [0, 1] "
                f"(min {_fmt(info['min'])}, max {_fmt(info['max'])}) — "
                f"a P(...) column should not"
            )

        if is_score:
            top_frac = _top_value_fraction(non_null)
            info["top_value_frac"] = top_frac
            if n_distinct == 1:
                rep.error(f"{name}: CONSTANT — every row is {_fmt(finite[0])}. "
                          f"The scorer wrote one value into all {len(finite)} rows.")
            elif (info["std"] is not None and info["std"] < 1e-9) or top_frac >= CONSTANT_FRAC:
                rep.error(f"{name}: effectively constant — {top_frac:.0%} of rows "
                          f"share one value (std {_fmt(info['std'])})")
            elif top_frac >= SUSPICIOUS_FRAC:
                rep.warn(f"{name}: {top_frac:.0%} of rows share one value "
                         f"({n_distinct} distinct total) — check this scorer ran "
                         f"properly (or it is a very clumpy judge)")
        rep.detail(f"{name}: {n_distinct} distinct, "
                   f"[{_fmt(info['min'])}, {_fmt(info['max'])}] "
                   f"mean {_fmt(info['mean'])} std {_fmt(info['std'])}"
                   + (f", {info['nan']} NaN" if info["nan"] else ""))
    else:  # string / bool / etc.
        if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
            _check_string(name, non_null, n, n_distinct, rep, info)
        else:
            rep.ok(f"{name}: {n_distinct} distinct")

    return info


def _top_value_fraction(non_null: pa.ChunkedArray) -> float:
    vc = non_null.value_counts()
    counts = vc.field("counts").to_numpy()
    return float(counts.max() / counts.sum()) if counts.size else 0.0


def _check_string(name: str, non_null, n: int, n_distinct: int,
                  rep: Report, info: dict) -> None:
    s = non_null.to_pylist()
    empty = sum(1 for x in s if x == "" or (isinstance(x, str) and not x.strip()))
    info.update(empty_str=empty, distinct=n_distinct)

    if name in TEXTISH:
        if empty:
            rep.error(f"{name}: {empty} empty / whitespace-only document(s)")
        lengths = np.array([len(x) for x in s], dtype=float)
        words = np.array([len(x.split()) for x in s], dtype=float)
        short = int((words < 20).sum())
        truncated = sum(1 for x in s if x.rstrip().endswith(("…", "...")))
        info.update(len_min=int(lengths.min()), len_med=int(np.median(lengths)),
                    len_max=int(lengths.max()), words_med=int(np.median(words)))
        if short:
            rep.warn(f"{name}: {short} document(s) under 20 words")
        if truncated:
            rep.warn(f"{name}: {truncated} document(s) end in an ellipsis (truncated?)")
        rep.detail(f"{name}: {info['words_med']} words median, "
                   f"chars [{info['len_min']}, {info['len_max']}], {n_distinct} distinct")
    elif name in TITLEISH:
        if empty:
            rep.warn(f"{name}: {empty} empty title(s)")
        rep.detail(f"{name}: {n_distinct} distinct")
    elif name == "uid":
        dupes = len(s) - n_distinct
        if dupes:
            rep.error(f"{name}: {dupes} duplicate value(s) — not unique per row")
        else:
            rep.ok(f"{name}: {n_distinct} unique")
    elif name == "generated_by":
        vc = non_null.value_counts()
        pairs = {vc.field("values")[i].as_py(): vc.field("counts")[i].as_py()
                 for i in range(len(vc))}
        rep.ok(f"{name}: " + ", ".join(f"{k}={v}" for k, v in pairs.items()))
        if len(pairs) < 2:
            rep.warn(f"{name}: single class — fine for an eval-only holdout, "
                     f"not for a training corpus")
    else:
        if empty and n_distinct == 1:
            rep.warn(f"{name}: every non-null value is empty")
        rep.detail(f"{name}: {n_distinct} distinct"
                   + (f", {empty} empty" if empty else ""))


def check_list(name: str, arr: pa.ChunkedArray, n: int, rep: Report) -> dict:
    """List columns. Fixed-width vectors (emb_*, *_cuevec_*) get the full
    embedding sanity check; variable-length lists just get null / empty."""
    nulls = arr.null_count
    info: dict = {"nulls": nulls}
    vector = _is_vector(name)
    if nulls == n:
        (rep.error if vector else rep.warn)(f"{name}: entirely null")
        return info
    if nulls:
        (rep.error if vector else rep.warn)(_null_line(name, nulls, n))

    flat = arr.drop_null().to_pylist()
    lengths = np.array([len(x) if x is not None else 0 for x in flat])
    empty = int((lengths == 0).sum())
    widths = sorted(set(lengths[lengths > 0].tolist()))
    info.update(empty=empty, widths=widths)

    if not vector:
        if empty:
            rep.detail(f"{name}: {empty} empty list(s) (variable-length column)")
        rep.detail(f"{name}: list<{arr.type.value_type}>, "
                   f"lengths {lengths.min()}..{lengths.max()}")
        return info

    if empty:
        rep.error(f"{name}: {empty} empty vector(s)")
    if len(widths) > 1:
        rep.error(f"{name}: ragged — widths {widths} (a vector column must be one width)")
        return info
    if not widths:
        return info

    dim = widths[0]
    M = np.vstack([np.asarray(x, dtype=float) for x in flat if x])
    n_nan = int(np.isnan(M).any(axis=1).sum())
    finite_rows = ~np.isnan(M).any(axis=1)
    n_zero = int((finite_rows & (np.abs(M).sum(axis=1) == 0)).sum())
    norms = np.linalg.norm(np.nan_to_num(M), axis=1)
    info.update(dim=dim, nan_vectors=n_nan, zero_vectors=n_zero,
                norm_min=float(norms.min()), norm_max=float(norms.max()))
    if n_nan:
        rep.error(f"{name}: {n_nan} vector(s) contain NaN")
    if n_zero:
        rep.error(f"{name}: {n_zero} all-zero vector(s) — the embedder produced "
                  f"nothing for those rows")
    # Identical high-dim vectors are a real red flag (two docs, one embedding);
    # for a low-dim / binary vector (the 11-dim cue vector) collisions are
    # expected and mean nothing.
    if dim > 32:
        n_identical = M.shape[0] - len({m.tobytes() for m in M})
        if n_identical:
            rep.warn(f"{name}: {n_identical} vector(s) byte-identical to another row")
    rep.detail(f"{name}: dim {dim}, norm [{_fmt(info['norm_min'])}, "
               f"{_fmt(info['norm_max'])}]")
    return info


def check_file(path: Path, reference_cols: set[str] | None,
               args: argparse.Namespace) -> Report:
    rep = Report()
    print(f"\n{BOLD}{'=' * 78}{RESET}\n{BOLD}{path}{RESET}\n{'=' * 78}")

    if not path.is_file():
        rep.error("file does not exist")
        return rep
    size_mb = path.stat().st_size / 1e6

    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    schema = pf.schema_arrow
    print(f"  {n} rows, {len(schema)} columns, {size_mb:.1f} MB")
    if n == 0:
        rep.error("file has 0 rows")
        return rep

    if reference_cols is not None:
        have = set(schema.names)
        missing = sorted(reference_cols - have)
        extra = sorted(have - reference_cols)
        if missing:
            rep.error(f"missing {len(missing)} column(s) the reference has: {missing}")
        if extra:
            rep.warn(f"has {len(extra)} column(s) the reference does not: {extra}")

    col_info: dict[str, dict] = {}
    print(f"\n  {BOLD}columns{RESET}")
    for field in schema:
        arr = pq.read_table(path, columns=[field.name]).column(0)
        if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
            col_info[field.name] = check_list(field.name, arr, n, rep)
        else:
            col_info[field.name] = check_scalar(field.name, arr, n, rep)

    # verdict
    ne, nw = len(rep.errors), len(rep.warns)
    print(f"\n  {BOLD}verdict{RESET}: ", end="")
    if ne:
        print(f"{RED}{ne} error(s){RESET}, {nw} warning(s)")
    elif nw:
        print(f"{YELLOW}{nw} warning(s){RESET}, no errors")
    else:
        print(f"{GREEN}clean{RESET}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "path": str(path), "rows": n, "columns": len(schema),
            "errors": rep.errors, "warnings": rep.warns,
            "column_info": col_info,
        }, indent=2, default=str))
        print(f"  {DIM}json → {args.json}{RESET}")
    return rep


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("parquets", nargs="+", type=Path)
    ap.add_argument("--reference", type=Path, default=None,
                    help="another parquet whose column set every input must contain")
    ap.add_argument("--fail-on", choices=("error", "warn", "never"), default="error",
                    help="which findings set exit code 1 (default: error)")
    ap.add_argument("--strict", action="store_true", help="alias for --fail-on warn")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the machine-readable report here (last file wins)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.strict:
        args.fail_on = "warn"

    reference_cols = None
    if args.reference:
        reference_cols = set(pq.ParquetFile(args.reference).schema_arrow.names)
        print(f"{DIM}reference: {args.reference} ({len(reference_cols)} columns){RESET}")

    worst_error = worst_warn = False
    for path in args.parquets:
        rep = check_file(path, reference_cols, args)
        worst_error |= bool(rep.errors)
        worst_warn |= bool(rep.warns)

    if args.fail_on == "error" and worst_error:
        sys.exit(1)
    if args.fail_on == "warn" and (worst_error or worst_warn):
        sys.exit(1)


if __name__ == "__main__":
    main()
