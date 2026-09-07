r"""Check the merged Ghostbuster corpus and drop the rows nothing can fix.

Reads   data/parquet/ghostbuster_v3f_multi.parquet        (read-only)
Writes  data/parquet/ghostbuster_v3f_multi_clean.parquet  (new file)

Runs ``check_parquet_completeness.py`` for the full picture, then removes two
classes of bad row — detected the same way the checker detects them, and always
as whole ``id`` groups so every surviving group keeps its full human + generator
set:

  blank text   a null / empty / whitespace-only ``text`` cell. The upstream
               Ghostbuster essay release ships six blank HUMAN seed files, and
               every generator's answer to those prompts is written against a
               zero-word target. Nothing regenerates a human row (no prompt), so
               these groups can only be dropped.

  runaway      a machine row far longer than its human partner AND over the
               corpus's absolute word bar — a decoding loop. Flagged at the same
               >= 4.0x ratio the checker ERRORs on (``--ratio`` to change;
               ``--drop-long`` also removes the 2.0-4.0x WARN band).

DROPPING A RUNAWAY TAKES GOOD ROWS WITH IT. The runaway is one generator's row,
but its ``id`` group also holds the (fine) human original and the other three
generators' answers to the same prompt — 4 extra rows per group here. Keeping
those means regenerating just the bad row instead, which needs the gpt56luna
generator and is a separate operation; this script does not do it. Pass
``--no-runaways`` to leave the runaway groups in place.

What this does NOT touch: unfilled suite columns (embeddings, Binoculars,
Fast-DetectGPT, detector, judges). A partial-fill column is not a bad row — it
is work ``fill_suite.py`` has not done yet. Run that next.

    uv run python epai/ai_detection/rollout/gb_v3f_clean.py --yes
    uv run python epai/ai_detection/rollout/gb_v3f_clean.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from epai.ai_detection.rollout._common import (
    BASE_DIR, BOLD, DIM, GREEN, RED, RESET, YELLOW, banner, rel,
)
from epai.ai_detection.rollout.gemma_ghostbuster_new import COMBINED

COMPLETENESS = BASE_DIR / "epai/scripts/check_parquet_completeness.py"

# Mirrors check_parquet_completeness.py's runaway detector (LEN_* constants).
RATIO_ERROR = 4.0
RATIO_WARN = 2.0
ABS_MIN_WORDS = 900
ABS_MULT = 3.0
MIN_REF_WORDS = 20


def run_checker(path: Path) -> int:
    """The full completeness report. Its exit code is informational here."""
    return subprocess.run(
        [sys.executable, str(COMPLETENESS), str(path),
         "--max-essay-human-blanks", "--verbose"],
        cwd=BASE_DIR,
    ).returncode


def blank_ids(t: pa.Table) -> set[str]:
    text = t.column("text")
    stripped = pc.utf8_trim_whitespace(text)
    blank = pc.or_(pc.is_null(text), pc.fill_null(pc.equal(stripped, ""), False))
    return set(str(x) for x in pc.unique(t.column("id").filter(blank)).to_pylist())


def runaway_hits(t: pa.Table, ratio: float, drop_long: bool) -> list[tuple]:
    """(id, generated_by, words, human_words, ratio) for each runaway machine row."""
    ids = [str(x) for x in t.column("id").to_pylist()]
    gen = [str(x) for x in t.column("generated_by").to_pylist()]
    words = [len((s or "").split()) for s in t.column("text").to_pylist()]

    human_w = {ids[i]: words[i] for i in range(len(ids)) if gen[i] == "human"}
    if not human_w:
        return []
    abs_bar = max(ABS_MIN_WORDS, ABS_MULT * float(np.median(list(human_w.values()))))
    floor = ratio if not drop_long else RATIO_WARN

    hits = []
    for i in range(len(ids)):
        if gen[i] == "human":
            continue
        h = human_w.get(ids[i])
        if not h or h < MIN_REF_WORDS or words[i] < abs_bar:
            continue
        r = words[i] / h
        if r >= floor:
            hits.append((ids[i], gen[i], words[i], h, r))
    return hits


def _by(t: pa.Table, mask: pa.Array, col: str) -> dict:
    if col not in t.column_names:
        return {}
    return dict(Counter(str(v) for v in t.column(col).filter(mask).to_pylist()))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("parquet", type=Path, nargs="?", default=COMBINED,
                    help=f"merged corpus to clean (default {rel(COMBINED)})")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output (default: <stem>_clean.parquet next to the input)")
    ap.add_argument("--ratio", type=float, default=RATIO_ERROR,
                    help=f"runaway machine/human word ratio to drop at "
                         f"(default {RATIO_ERROR})")
    ap.add_argument("--drop-long", action="store_true",
                    help=f"also drop the {RATIO_WARN}-{RATIO_ERROR}x WARN band")
    ap.add_argument("--no-runaways", action="store_true",
                    help="drop only blank-text groups, leave runaways in place")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the completeness report before and after")
    ap.add_argument("--force", action="store_true", help="overwrite the output")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src: Path = args.parquet
    if not src.is_file():
        raise SystemExit(f"{RED}not a file: {rel(src)}{RESET}")
    out: Path = args.out or src.with_name(f"{src.stem}_clean{src.suffix}")
    if out.resolve() == src.resolve():
        raise SystemExit(f"{RED}refusing to write over the input: {rel(out)}{RESET}")
    if out.exists() and not args.force and not args.dry_run:
        raise SystemExit(f"{RED}output exists (pass --force): {rel(out)}{RESET}")

    if not args.no_check:
        banner("completeness report", rel(src))
        run_checker(src)

    t = pq.read_table(src)
    for col in ("id", "generated_by", "text"):
        if col not in t.column_names:
            raise SystemExit(f"{RED}no {col!r} column in {rel(src)}{RESET}")

    banner("bad rows", "detected as whole id groups")
    blanks = blank_ids(t)
    b_mask = pc.is_in(t.column("id"),
                      value_set=pa.array(sorted(blanks), type=pa.string()))
    print(f"{BOLD}blank text{RESET}: {len(blanks)} id group(s)")
    if blanks:
        print(f"  by generated_by: {_by(t, b_mask, 'generated_by')}")
        print(f"  by domain:       {_by(t, b_mask, 'domain')}")
        print(f"  {DIM}{', '.join(sorted(blanks))}{RESET}")

    hits = [] if args.no_runaways else runaway_hits(t, args.ratio, args.drop_long)
    runaways = {h[0] for h in hits}
    print(f"\n{BOLD}runaway{RESET}: {len(runaways)} id group(s) "
          f"(ratio >= {RATIO_WARN if args.drop_long else args.ratio})")
    for rid, g, w, hw, r in sorted(hits, key=lambda x: -x[2]):
        print(f"  {rid:<16} {g:<18} {w:>6}w = {r:>5.1f}x its {hw}w human partner")
    if runaways:
        keep_labels = sorted(set(str(x) for x in t.column("generated_by").to_pylist())
                             - {"human"} - {h[1] for h in hits})
        print(f"  {YELLOW}NOTE{RESET} dropping these groups also removes the human "
              f"original and the\n       {', '.join(keep_labels)} rows for the same "
              f"prompt(s). To keep those,\n       re-run with --no-runaways and "
              f"regenerate the flagged rows instead.")

    drop = blanks | runaways
    if not drop:
        print(f"\n{GREEN}nothing to drop.{RESET} Run fill_suite.py on {rel(src)}.")
        return

    keep = pc.invert(pc.is_in(t.column("id"),
                              value_set=pa.array(sorted(drop), type=pa.string())))
    n_keep = pc.sum(pc.cast(keep, pa.int64())).as_py()
    print(f"\nrows: {t.num_rows} -> {GREEN}{n_keep}{RESET}  "
          f"(drop {RED}{t.num_rows - n_keep}{RESET} across {len(drop)} id group(s))")

    if args.dry_run:
        print(f"\n{DIM}--dry-run: would write -> {rel(out)}{RESET}")
        return
    if not args.yes:
        if input(f"{YELLOW}write {n_keep} rows -> {rel(out)}?{RESET} [y/N] ") \
                .strip().lower() not in ("y", "yes"):
            raise SystemExit("aborted.")

    cleaned = t.filter(keep)
    pq.write_table(cleaned, out)
    print(f"{GREEN}wrote{RESET} {rel(out)}  ({cleaned.num_rows} rows, "
          f"{cleaned.num_columns} columns)")

    if not args.no_check:
        banner("re-check", rel(out))
        left_blank = blank_ids(cleaned)
        left_run = {h[0] for h in
                    ([] if args.no_runaways else runaway_hits(cleaned, args.ratio, args.drop_long))}
        if left_blank or left_run:
            print(f"{RED}still present: blank={left_blank} runaway={left_run}{RESET}")
            raise SystemExit(1)
        print(f"{GREEN}no blank-text or runaway rows remain.{RESET}")
        print(f"{DIM}Partial-fill errors from the full checker are expected here "
              f"and are filled by:{RESET}")
        print(f"  uv run python epai/ai_detection/rollout/fill_suite.py {rel(out)} "
              f"--gpu <id> --yes")


if __name__ == "__main__":
    main()
