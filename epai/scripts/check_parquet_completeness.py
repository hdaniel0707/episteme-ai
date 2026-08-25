#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["duckdb"]
# ///
"""
Check parquet files for missing/empty/null/zero values, column by column.

Usage:
    uv run epai/scripts/check_parquet_completeness.py data/parquet/ghostbuster_gpt35t.parquet
    uv run epai/scripts/check_parquet_completeness.py data/parquet/ghostbuster_gpt35ts.parquet
    uv run epai/scripts/check_parquet_completeness.py data/parquet/ghostbuster_gpt54mini.parquet
    uv run epai/scripts/check_parquet_completeness.py data/parquet/ghostbuster_gpt56luna.parquet
    uv run epai/scripts/check_parquet_completeness.py data/parquet/science_v3_gpt56luna_0811A.parquet
"""
import sys
import duckdb


def check_file(con, path):
    print(f"\n{'=' * 90}\n{path}\n{'=' * 90}")

    total = con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
    print(f"rows: {total}")
    if total == 0:
        print("  (empty file)")
        return

    cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()

    header = f"{'column':<32}{'type':<14}{'nulls':>10}{'null%':>8}{'extra':>34}"
    print(header)
    print("-" * len(header))

    for name, dtype, *_ in cols:
        q = f'"{name}"'
        null_count = con.execute(
            f"SELECT count(*) FROM read_parquet('{path}') WHERE {q} IS NULL"
        ).fetchone()[0]
        null_pct = 100.0 * null_count / total

        extra = ""
        if dtype == "VARCHAR":
            empty = con.execute(
                f"SELECT count(*) FROM read_parquet('{path}') WHERE {q} = ''"
            ).fetchone()[0]
            literal_zero = con.execute(
                f"SELECT count(*) FROM read_parquet('{path}') WHERE {q} = '0'"
            ).fetchone()[0]
            extra = f"empty_str={empty} literal_'0'={literal_zero}"
        elif dtype.endswith("[]"):
            empty_arr = con.execute(
                f"SELECT count(*) FROM read_parquet('{path}') "
                f"WHERE {q} IS NOT NULL AND len({q}) = 0"
            ).fetchone()[0]
            extra = f"empty_array={empty_arr}"
        elif dtype in ("DOUBLE", "FLOAT", "BIGINT", "INTEGER", "DECIMAL"):
            row = con.execute(
                f"SELECT count(*) FILTER (WHERE {q} = 0), "
                f"min({q}), max({q}), avg({q}) "
                f"FROM read_parquet('{path}')"
            ).fetchone()
            zero_count, mn, mx, avg = row
            avg_s = f"{avg:.4f}" if avg is not None else "NA"
            mn_s = f"{mn:.4f}" if mn is not None else "NA"
            mx_s = f"{mx:.4f}" if mx is not None else "NA"
            extra = f"zeros={zero_count} min={mn_s} max={mx_s} avg={avg_s}"

        flag = ""
        if null_count > 0:
            flag = "  <-- NULLS"
        print(f"{name:<32}{dtype:<14}{null_count:>10}{null_pct:>7.2f}%{extra:>34}{flag}")


def main():
    files = sys.argv[1:]
    if not files:
        print(__doc__)
        sys.exit(1)
    con = duckdb.connect()
    for f in files:
        check_file(con, f)


if __name__ == "__main__":
    main()
