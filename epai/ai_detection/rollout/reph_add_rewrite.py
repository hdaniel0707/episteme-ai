r"""Add the gpt56luna REWRITE rows into MG-Reph, alongside the existing rephrase.

``manugram_reph_gb.parquet`` / ``manugram_reph_sci.parquet`` hold human text
paired with one gpt-5.6-luna REPHRASE (``generated_by`` ``gpt56luna:0908R``,
see ``rephrase_build.py``) — a content-preserving edit that sees the human
text verbatim. The same human documents already have a gpt-5.6-luna FULL-TEXT
REWRITE under the Ghostbuster protocol (the generator never sees the human
text), sitting in ``manugram_gb.parquet`` / ``manugram_sci.parquet`` as
``gpt56luna:0701A`` / ``gpt56luna:0811A`` respectively — already scored
(embeddings, zero-shot, all suite judges) as part of those core splits.

This pulls those existing rewrite rows in by ``id``, no new model calls — a
pure column-conforming concat, like ``rephrase_build.step_extract``. It turns
MG-Reph into a 2x2 at fixed model identity (gpt-5.6-luna):

               original              rephrased
    human      (existing)            gpt56luna:0908R  (existing)
    gpt56luna  gpt56luna:0701A/0811A  <-- this script    (reph_rephrase_rewrite.py
                                                           adds the 4th cell)

Inner-joined on ``id``: only rewrite rows whose id already has a surviving
human+rephrase pair in the MG-Reph file are pulled in (a handful of ids were
dropped from MG-Reph by ``rephrase_build.py``'s length-runaway QC gate).

    python epai/ai_detection/rollout/reph_add_rewrite.py --dry-run
    python epai/ai_detection/rollout/reph_add_rewrite.py
    python epai/ai_detection/rollout/reph_add_rewrite.py --only gb
"""

from __future__ import annotations

import argparse
import shutil

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from epai.ai_detection.rollout._common import (
    BASE_DIR, DIM, GREEN, RED, RESET, YELLOW, banner, rel,
)
from epai.ai_detection.rollout.rephrase_build import QC_COLS

PARQUET_DIR = BASE_DIR / "data" / "parquet"

#: fam key -> its rewrite source (already-scored, multi-generator) and the
#: MG-Reph half it feeds. Matches split_manugram_reph.py's FAMILIES mapping.
FAMILIES = {
    "gb": {
        "source_family": "ghostbuster",
        "rewrite_src": PARQUET_DIR / "manugram_gb.parquet",
        "target": PARQUET_DIR / "manugram_reph_gb.parquet",
    },
    "sci": {
        "source_family": "science",
        "rewrite_src": PARQUET_DIR / "manugram_sci.parquet",
        "target": PARQUET_DIR / "manugram_reph_sci.parquet",
    },
}


def _rewrite_label(gen_values: list[str]) -> str:
    """The single ``gpt56luna:<prompt_id>`` label in a rewrite source file."""
    labels = sorted({g for g in gen_values if g.startswith("gpt56luna:")})
    if len(labels) != 1:
        raise SystemExit(
            f"expected exactly one gpt56luna: label in the rewrite source, "
            f"found {labels}"
        )
    return labels[0]


def _merge_type(a: pa.DataType, b: pa.DataType) -> pa.DataType:
    """Resolve two Arrow types for the same field name into one shared type.

    ``pa.unify_schemas`` refuses these outright even though the value-level
    cast is safe and lossless: string vs large_string (a metadata column built
    by two different pipelines), and int vs float (an integer metadata column
    -- author_count, citation_count, year -- that null-heavy rows promoted to
    double in one file but not the other). Always widens, never narrows.
    """
    if a == b:
        return a
    if pa.types.is_string(a) or pa.types.is_large_string(a) or \
       pa.types.is_string(b) or pa.types.is_large_string(b):
        return pa.large_string()
    if pa.types.is_integer(a) and pa.types.is_floating(b):
        return b
    if pa.types.is_floating(a) and pa.types.is_integer(b):
        return a
    if pa.types.is_integer(a) and pa.types.is_integer(b):
        return a if a.bit_width >= b.bit_width else b
    if pa.types.is_floating(a) and pa.types.is_floating(b):
        return a if a.bit_width >= b.bit_width else b
    raise SystemExit(f"don't know how to merge field types {a} and {b}")


def _unify_schemas(a: pa.Schema, b: pa.Schema) -> pa.Schema:
    """Like ``pa.unify_schemas`` but resolves the mismatches this merge hits
    (see ``_merge_type``) instead of raising on them."""
    types = {f.name: f.type for f in a}
    for f in b:
        types[f.name] = _merge_type(types[f.name], f.type) if f.name in types else f.type
    seen = {f.name for f in a}
    names = [f.name for f in a] + [f.name for f in b if f.name not in seen]
    return pa.schema([pa.field(n, types[n]) for n in names])


def _conform(t: pa.Table, schema: pa.Schema) -> pa.Table:
    """Cast/pad ``t`` onto ``schema``, filling any column it lacks with nulls."""
    cols = [
        t.column(f.name).cast(f.type) if f.name in t.column_names
        else pa.nulls(t.num_rows, f.type)
        for f in schema
    ]
    return pa.table(cols, schema=schema)


def add_rewrite(fam: str, *, force: bool, dry_run: bool) -> int:
    cfg = FAMILIES[fam]
    target, rewrite_src, source_family = (
        cfg["target"], cfg["rewrite_src"], cfg["source_family"]
    )
    for p in (target, rewrite_src):
        if not p.is_file():
            print(f"{RED}not found: {rel(p)}{RESET}")
            return 1

    tgt = pq.read_table(target)
    src = pq.read_table(rewrite_src)
    rw_label = _rewrite_label(src.column("generated_by").to_pylist())

    gen = tgt.column("generated_by").to_pylist()
    if rw_label in gen:
        n = sum(1 for g in gen if g == rw_label)
        if not force:
            print(f"{RED}{rel(target)} already has {n} row(s) under "
                  f"{rw_label!r}.{RESET}\n  Pass --force to drop and re-merge them.")
            return 1
        tgt = tgt.filter(pc.not_equal(tgt.column("generated_by"), rw_label))
        print(f"{YELLOW}  dropped {n} existing {rw_label!r} row(s) (--force){RESET}")

    tgt_ids = set(tgt.column("id").to_pylist())
    src_mask = pc.and_(
        pc.equal(src.column("generated_by"), rw_label),
        pc.is_in(src.column("id"), value_set=pa.array(sorted(tgt_ids))),
    )
    matched = src.filter(src_mask)
    missing = tgt_ids - set(matched.column("id").to_pylist())
    if missing:
        print(f"{YELLOW}  {len(missing)} MG-Reph id(s) have no {rw_label!r} row "
              f"in {rel(rewrite_src)} — left unmatched:{RESET}")
        for mid in sorted(missing)[:5]:
            print(f"{DIM}    {mid}{RESET}")

    n = matched.num_rows
    if n == 0:
        print(f"{RED}nothing matched.{RESET}")
        return 1

    # Match the target's existing field types exactly (e.g. large_string, not
    # string) -- unify_schemas is strict about the Arrow string-width variant.
    matched = matched.append_column(
        "source_family",
        pa.array([source_family] * n, tgt.schema.field("source_family").type),
    )
    for c in QC_COLS:
        matched = matched.append_column(c, pa.nulls(n, tgt.schema.field(c).type))

    unified = _unify_schemas(tgt.schema, matched.schema)
    combined = pa.concat_tables([_conform(tgt, unified), _conform(matched, unified)])
    order = pc.sort_indices(
        combined, sort_keys=[("id", "ascending"), ("generated_by", "ascending")]
    )
    combined = combined.take(order)

    print(f"  {rel(target):45s} +{n} {rw_label!r} row(s) -> {combined.num_rows} total")
    print(f"  classes now: {sorted(set(combined.column('generated_by').to_pylist()))}")

    if dry_run:
        print(f"{DIM}dry-run — nothing written{RESET}")
        return 0

    bak = target.with_suffix(target.suffix + ".bak")
    shutil.copy2(target, bak)
    print(f"{DIM}  backup -> {rel(bak)}{RESET}")
    pq.write_table(combined, target)
    print(f"{GREEN}wrote {combined.num_rows} rows{RESET} -> {rel(target)}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="+", choices=sorted(FAMILIES), default=None,
                    help="restrict to one or both families (default: both)")
    ap.add_argument("--force", action="store_true",
                    help="drop and re-merge if the rewrite label is already present")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fams = args.only or sorted(FAMILIES)
    failed = False
    for fam in fams:
        banner("add-rewrite", fam)
        code = add_rewrite(fam, force=args.force, dry_run=args.dry_run)
        if code != 0:
            failed = True
    if failed:
        raise SystemExit(1)
    if not args.dry_run:
        print(f"\n{GREEN}Done.{RESET} Next: reph_rephrase_rewrite.py, "
              f"then reph_fill_scores.py.")


if __name__ == "__main__":
    main()
