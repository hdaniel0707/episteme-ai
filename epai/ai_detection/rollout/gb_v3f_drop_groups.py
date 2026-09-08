r"""Drop whole ``id`` groups from the cleaned Ghostbuster v3f corpus.

``gb_v3f_clean.py`` only removes empty-``text`` groups and >=4x runaways. Two
bad groups slip past it and are removed here instead:

  essay:561   the upstream human file embeds whitespace-padded formatting, so
              the fork's old word count (``split(" ")``) read it as ~7350 words
              and every generator was prompted for that. gpt-oss-120b refused
              ("I'm not able to generate ... 7,350 words"), saved as an 89-word
              body; gpt-5.6-luna wrote a 10k-word runaway. The bug is fixed in
              the fork now, but this group was built before the fix.

  essay:819   the upstream human "essay" is a 6-word topic heading
              ("Behavioral Economics Biases/heuristics in Apple Products"), not
              prose. round_to_50(6) -> 50, so all four generators produced
              ~50-word blurbs. Same defect family as the six blank seeds.

Each id is removed as a whole 5-row group (human + the four generators), so
every surviving group keeps its full set. Writes in place after saving a
one-time ``.bak`` next to the file.

    python epai/ai_detection/rollout/gb_v3f_drop_groups.py --dry-run
    python epai/ai_detection/rollout/gb_v3f_drop_groups.py --yes
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from epai.ai_detection.rollout._common import (
    BASE_DIR, COMPLETENESS_SCRIPT, DIM, GREEN, RED, RESET, YELLOW, banner, rel, run,
)

CORPUS = BASE_DIR / "data" / "parquet" / "ghostbuster_v3f_multi_clean.parquet"
DROP_IDS = ["essay:561", "essay:819"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("parquet", type=Path, nargs="?", default=CORPUS,
                    help=f"corpus to edit in place (default {rel(CORPUS)})")
    ap.add_argument("--ids", nargs="+", default=DROP_IDS, metavar="ID",
                    help=f"id groups to drop (default: {' '.join(DROP_IDS)})")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="write here instead of overwriting the input")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the completeness report afterwards")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing .bak / output")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src: Path = args.parquet
    if not src.is_file():
        raise SystemExit(f"{RED}not a file: {rel(src)}{RESET}")
    out: Path = args.out or src

    t = pq.read_table(src)
    for col in ("id", "generated_by", "text"):
        if col not in t.column_names:
            raise SystemExit(f"{RED}no {col!r} column in {rel(src)}{RESET}")

    id_col = [str(x) for x in t.column("id").to_pylist()]
    gen_col = [str(x) for x in t.column("generated_by").to_pylist()]
    words = [len((s or "").split()) for s in t.column("text").to_pylist()]

    banner("groups to drop", rel(src))
    want = list(dict.fromkeys(args.ids))
    missing = [i for i in want if i not in id_col]
    if missing:
        raise SystemExit(f"{RED}id(s) not in the corpus: {', '.join(missing)}{RESET}")

    for rid in want:
        rows = [k for k in range(len(id_col)) if id_col[k] == rid]
        print(f"{rid}  ({len(rows)} rows)")
        for k in rows:
            print(f"  {DIM}{gen_col[k]:<18} {words[k]:>7}w{RESET}")

    keep = pc.invert(pc.is_in(t.column("id"),
                              value_set=pa.array(want, type=pa.string())))
    n_keep = pc.sum(pc.cast(keep, pa.int64())).as_py()
    print(f"\nrows: {t.num_rows} -> {GREEN}{n_keep}{RESET} "
          f"(drop {RED}{t.num_rows - n_keep}{RESET} across {len(want)} group(s))")

    if args.dry_run:
        print(f"\n{DIM}--dry-run: would write -> {rel(out)}{RESET}")
        return
    if not args.yes:
        if input(f"{YELLOW}write {n_keep} rows -> {rel(out)}?{RESET} [y/N] ") \
                .strip().lower() not in ("y", "yes"):
            raise SystemExit("aborted.")

    if out == src:
        bak = src.with_suffix(src.suffix + ".bak")
        if bak.exists() and not args.force:
            raise SystemExit(f"{RED}{rel(bak)} exists (pass --force){RESET}")
        shutil.copy2(src, bak)
        print(f"{DIM}backup -> {rel(bak)}{RESET}")

    pq.write_table(t.filter(keep), out)
    print(f"{GREEN}wrote{RESET} {rel(out)}  ({n_keep} rows, {t.num_columns} columns)")

    if not args.no_check:
        banner("completeness report", rel(out))
        run([sys.executable, BASE_DIR / COMPLETENESS_SCRIPT, out,
             "--max-essay-human-blanks"], False)


if __name__ == "__main__":
    main()
