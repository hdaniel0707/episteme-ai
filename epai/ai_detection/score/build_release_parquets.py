r"""Write the reviewer copy of each base parquet into ``released_data/``, keeping
only the columns listed in the ``release`` block of
``cfg/ai_detection/core_parquet_columns.yaml`` -- the columns the ICLR 2027
paper reports a result from.

    core stems      release.core  +  release.per_stem_extra[stem]
    MG-Reph stems   release.rephrase

    <stem>.parquet        full release (same file name as the master, so moved
                          scripts only need their data directory changed)
    <stem>_lite.parquet   --lite: the same minus every emb_* column (MG-Reph has
                          none, so its lite file equals its full one -- written
                          anyway so the lite set is complete on its own)

Output goes to released_data/full/ (or released_data/lite/ with --lite). The
base parquets are only read, never modified, and the output directory may
not be data/parquet/ -- the full release files share the masters' names. A
stem missing any release column is reported and skipped -- no partial release
file is written. Output column order is exactly the yaml's order. The pandas
schema metadata is stripped, since it would otherwise still list every dropped
column's name.

Run:
    uv run python epai/ai_detection/score/build_release_parquets.py --dry-run
    uv run python epai/ai_detection/score/build_release_parquets.py
    uv run python epai/ai_detection/score/build_release_parquets.py --lite
    uv run python epai/ai_detection/score/build_release_parquets.py --only manugram_gb
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow.parquet as pq

from epai.ai_detection.score.core_columns import (
    ALL_STEMS,
    BASE_DIR,
    CORE_CFG,
    REPHRASE_STEMS,
)

PARQUET_DIR = BASE_DIR / "data" / "parquet"
RELEASE_DIR = BASE_DIR / "released_data"
RELEASE = CORE_CFG["release"]


def release_columns(stem: str, lite: bool = False) -> list[str]:
    if stem in REPHRASE_STEMS:
        cols = list(RELEASE["rephrase"])
    else:
        cols = list(RELEASE["core"]) + list(RELEASE.get("per_stem_extra", {}).get(stem, []))
    return [c for c in cols if not c.startswith("emb_")] if lite else cols


def process_stem(stem: str, args: argparse.Namespace) -> bool:
    src = PARQUET_DIR / f"{stem}.parquet"
    out = args.out_dir / (f"{stem}_lite.parquet" if args.lite else f"{stem}.parquet")
    print(f"\n{'-' * 74}\n{stem}")
    if not src.is_file():
        print(f"  not on disk: {src} -- skipped")
        return False

    keep = release_columns(stem, args.lite)
    names = pq.read_schema(src).names
    missing = [c for c in keep if c not in names]
    if missing:
        print(f"  [ERROR] {len(missing)} release column(s) missing -- skipped, nothing written:")
        for c in missing:
            print(f"    {c}")
        return False

    keep_set = set(keep)
    dropped = [c for c in names if c not in keep_set]
    print(f"  keep {len(keep)} / {len(names)} columns, drop {len(dropped)}:")
    for c in dropped:
        print(f"    - {c}")

    if args.dry_run:
        print("  [dry run] -- nothing written")
        return True
    if out.exists() and not args.overwrite:
        print(f"  {out} exists -- pass --overwrite; skipped")
        return False

    table = pq.read_table(src, columns=keep).select(keep).replace_schema_metadata(None)
    tmp = out.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, out)

    written = pq.read_schema(out).names
    if written != keep:
        print(f"  [ERROR] written schema does not match the release list: {out}")
        return False
    print(f"  wrote {table.num_rows} rows x {len(written)} columns -> {out}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="+", choices=ALL_STEMS, metavar="STEM")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: released_data/full/ or, with "
                         "--lite, released_data/lite/; never data/parquet/)")
    ap.add_argument("--lite", action="store_true",
                    help="write <stem>_lite.parquet without the emb_* columns")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing release file")
    ap.add_argument("--dry-run", action="store_true",
                    help="print kept/dropped columns per stem, write nothing")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = RELEASE_DIR / ("lite" if args.lite else "full")
    if args.out_dir.resolve() == PARQUET_DIR.resolve():
        ap.error("--out-dir may not be data/parquet/: release files share the masters' names")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stems = args.only or ALL_STEMS
    failed = [s for s in stems if not process_stem(s, args)]
    print(f"\n{'=' * 74}\n{len(stems) - len(failed)}/{len(stems)} stem(s) ok"
          + (f"; failed/skipped: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
