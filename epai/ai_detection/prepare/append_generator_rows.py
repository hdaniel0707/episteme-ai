r"""Append one generator's rows from a fresh generate run into a scored corpus.

The multi-generator science corpora (``science_v3f_multi_0811A.parquet`` and its
siblings) are the end of a chain — ``generate`` → ``promote`` → ``standardize``
→ the ``rebuild_corpus.py`` dedup passes → ``run_holdout_pipeline.py --from
embed`` — and re-running that whole chain to add a fourth generator would
re-touch every existing row. This does the surgical thing instead: it takes the
new generator's rows out of the raw generate output, folds their text the same
way ``standardize_text.py`` did the rest of the corpus, attaches every
detector-suite column as Null, and appends them.

After this, ``run_holdout_pipeline.py --from embed`` fills those Null columns —
its scoring stages skip rows that already have a value, so only the appended
rows cost anything (and a judge column the corpus never had is computed for
every row, backfilling the existing ones).

WHAT COMES ACROSS
-----------------
The corpus (non-suite) columns are copied verbatim from the source row —
``generate_ai.py`` already filled them from the paper's human row. ``text`` is
replaced with its standardised form and the pre-standardisation string is kept
in ``text_raw``, matching what the rest of the corpus carries. Every other
column present in the target (``emb_*``, ``binoculars_score*``,
``fastdetect_*``, ``detector_*``, ``llm_judge_*``) is added as Null.

THE PAPER-SET GAP
-----------------
The target is a *rebuilt* corpus: a few dozen of its papers were backfilled
during dedup and are not in the generate run's paper set, and a few dozen of
the generate run's papers were dropped from the target. Only the intersection
can be appended. Both gaps are reported; ``--require-coverage`` makes a
target paper with no new row fatal instead of a warning.

    python epai/ai_detection/prepare/append_generator_rows.py \
        --source data/science/science_v3_multi_0811A.parquet \
        --target data/parquet/science_v3f_multi_0811A.parquet \
        --label gemma431b:0811A

    # see what it would do, write nothing
    python epai/ai_detection/prepare/append_generator_rows.py ... --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from epai.utils.text_standardize import standardize_text

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
DIM = "\033[2m"
RESET = "\033[0m"

# Columns copied straight from the source generate output. Everything else in
# the target is a detector-suite output and is appended as Null for the scoring
# stages to fill. `text` and `text_raw` are handled separately (standardised).
CORPUS_COLS = [
    "uid", "id", "domain", "generated_by", "doi", "title", "author_ids",
    "author_count", "year", "venue", "citation_count", "fwci", "language",
    "fields_of_study", "query", "also_matched_domains", "prompt",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", type=Path, required=True,
                    help="raw generate output holding the new generator's rows "
                         "(e.g. data/science/science_v3_multi_0811A.parquet)")
    ap.add_argument("--target", type=Path, required=True,
                    help="the scored corpus to append into, rewritten in place "
                         "unless -o is given")
    ap.add_argument("--label", required=True,
                    help="generated_by value of the rows to move, e.g. "
                         "gemma431b:0811A")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="write here instead of overwriting --target")
    ap.add_argument("--whitespace", default="flat", choices=["flat", "lines"],
                    help="standardize_text whitespace mode (science corpora: "
                         "flat; ghostbuster: lines). Default flat.")
    ap.add_argument("--require-coverage", action="store_true",
                    help="fail if any target paper has no row under --label")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the <target>.bak copy made before an in-place write")
    ap.add_argument("--dry-run", action="store_true",
                    help="report and write nothing")
    args = ap.parse_args()

    for p in (args.source, args.target):
        if not p.is_file():
            raise SystemExit(f"{RED}not found: {p}{RESET}")

    # pyarrow throughout, not pandas: the target's score columns are typed
    # (int64 ntok counts, large_string rationales) and a pandas round-trip that
    # introduces a null upcasts int64 -> double for the whole column. Appending
    # pa.nulls of the schema's own type leaves every existing column untouched.
    src = pq.read_table(args.source)
    tgt = pq.read_table(args.target)
    schema = tgt.schema

    labels = src.column("generated_by").to_pylist()
    idx = [i for i, v in enumerate(labels) if v == args.label]
    if not idx:
        raise SystemExit(
            f"{RED}no rows with generated_by == {args.label!r} in "
            f"{args.source}{RESET}\n  Present labels: "
            f"{sorted(set(labels))}"
        )
    tgt_labels = set(tgt.column("generated_by").to_pylist())
    if args.label in tgt_labels:
        n = sum(v == args.label for v in tgt.column("generated_by").to_pylist())
        raise SystemExit(
            f"{RED}{args.target} already has {n} row(s) under {args.label!r}.{RESET}\n"
            f"  Nothing to do — or drop them first if this is a re-append."
        )

    missing = [c for c in CORPUS_COLS + ["text"] if c not in src.column_names]
    if missing:
        raise SystemExit(f"{RED}source is missing column(s): {missing}{RESET}")
    if "text_raw" not in schema.names:
        raise SystemExit(f"{RED}target has no 'text_raw' column{RESET}")

    new = src.take(idx)

    # Only papers the target actually holds can be appended.
    tgt_ids = set(tgt.column("id").to_pylist())
    new_ids = new.column("id").to_pylist()
    keep_pos = [i for i, pid in enumerate(new_ids) if pid in tgt_ids]
    keep = new.take(keep_pos)

    dropped = len(new_ids) - len(keep_pos)
    uncovered = sorted(tgt_ids - set(new_ids))

    print(f"\ntarget   {args.target}  ({tgt.num_rows} rows, {len(tgt_ids)} papers)")
    print(f"source   {args.source}")
    print(f"label    {args.label}")
    print(f"\n  {len(idx)} source row(s) under the label")
    print(f"  {keep.num_rows} appended  ·  {dropped} dropped (paper not in target)")
    if uncovered:
        msg = (f"{RED}  {len(uncovered)} target paper(s) will have NO {args.label} "
               f"row{RESET}" if args.require_coverage else
               f"{YELLOW}  {len(uncovered)} target paper(s) have no {args.label} "
               f"row — appended arm is {keep.num_rows}/{len(tgt_ids)}{RESET}")
        print(msg)
        print(f"{DIM}    e.g. {uncovered[:3]}{RESET}")
        if args.require_coverage:
            raise SystemExit(1)

    raw_text = [str(t) for t in keep.column("text").to_pylist()]
    std_text = [standardize_text(t, whitespace=args.whitespace) for t in raw_text]
    changed = sum(a != b for a, b in zip(raw_text, std_text))
    print(f"\n  standardize_text ({args.whitespace}): {changed}/{keep.num_rows} "
          f"row(s) changed")

    # One array per target column, in target order and target type.
    n = keep.num_rows
    cols = []
    for field in schema:
        c = field.name
        if c == "text":
            cols.append(pa.array(std_text, type=field.type))
        elif c == "text_raw":
            cols.append(pa.array(raw_text, type=field.type))
        elif c in CORPUS_COLS:
            cols.append(keep.column(c).cast(field.type).combine_chunks())
        else:
            cols.append(pa.nulls(n, type=field.type))
    appended = pa.table(cols, schema=schema)

    out = args.out or args.target
    if args.dry_run:
        print(f"\n{DIM}dry-run — would write {tgt.num_rows + n} rows → {out}{RESET}")
        return

    combined = pa.concat_tables([tgt, appended])
    order = pc.sort_indices(
        combined, sort_keys=[("id", "ascending"), ("generated_by", "ascending")]
    )
    combined = combined.take(order)

    if out == args.target and not args.no_backup:
        bak = args.target.with_suffix(args.target.suffix + ".bak")
        shutil.copy2(args.target, bak)
        print(f"{DIM}  backup → {bak}{RESET}")

    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, out)
    print(f"\n{GREEN}appended {n} row(s){RESET} → {out}  ({combined.num_rows} rows)")
    print(f"  next: run_holdout_pipeline.py --from embed -o {out} "
          f"--gpu <id> --yes --skip-existing")


if __name__ == "__main__":
    main()
