r"""Rebuild ``uid`` from ``id`` and ``generated_by``, where they have drifted apart.

THE DRIFT THIS REPAIRS
----------------------
``combine_domains.py`` defines the row key as ``uid = "{id}:{generated_by}"`` and
builds it once, as the corpus is assembled. Nothing rebuilds it afterwards --
and ``rename_column_values.py`` exists precisely to rewrite ``generated_by``
afterwards, turning a placeholder generator id into the canonical
``<model_slug>:<prompt_id>`` label. Run that on a finished corpus and the uid
keeps the OLD label for ever::

    id           essay:1
    generated_by gpt54mini:0701A      <- renamed
    uid          essay:1:gpt_plain    <- not renamed, and now a lie

Half the rows of ``ghostbuster_gpt35t`` and ``ghostbuster_gpt54mini`` are in
exactly that state (``essay:1:gpt``, ``essay:1:gpt_plain``), which is what
``check_standardization.py`` reports as "rows whose uid is not
'{id}:{generated_by}'". The right fix is not to relax the rule: the uid is the
row key, and a key that disagrees with the columns it is derived from breaks the
join back to the per-domain parquets and hides the relabelling from anything
that reads it.

WHY THIS IS SAFE ON A CORPUS YOU DO NOT WANT TO TOUCH
------------------------------------------------------
It rewrites one metadata column and nothing else. ``binoculars_score`` and every
``emb_*`` are derived from ``text``, which this never reads -- so unlike
``standardize_text.py`` there is nothing to re-score and nothing to re-embed
afterwards. A benchmark corpus can have its uid repaired without becoming a
different benchmark.

UNIQUENESS IS CHECKED BEFORE ANYTHING IS WRITTEN
-------------------------------------------------
A uid identifies a row, so the rebuild is refused outright if it would produce
two rows with the same one -- which is what happens when a rename COLLAPSES two
generator labels onto a third and the pairs behind them were distinct only by
label. That is a real corpus problem and it must not be discovered later, as a
duplicate the checker reports on a file nobody remembers rewriting.

``id`` is deliberately left alone. It is the PAIR key -- ``essay:1`` names a
source prompt, and its human and machine answers both carry it -- so folding the
label into it would put the target into the row identifier. See the header of
``combine_domains.py``.

Nothing is written until the plan is printed and confirmed. The write is
atomic: a temporary file in the same directory, moved over the original only
after it is complete, so an interrupted run cannot leave half a parquet.

NOTE: ``.gitignore`` holds ``/data/``, so these parquets are NOT in git and there
is no ``git checkout`` to undo an unwanted write. Use ``--dry-run`` first, and
``-o`` when you are not certain.

Run:
    # look, decide nothing
    uv run python epai/scripts/rebuild_uid.py data/parquet/ghostbuster_gpt35t.parquet --dry-run

    # both drifted corpora, one confirmation
    uv run python epai/scripts/rebuild_uid.py \
        data/parquet/ghostbuster_gpt35t.parquet \
        data/parquet/ghostbuster_gpt54mini.parquet

    # into a copy, leaving the input untouched
    uv run python epai/scripts/rebuild_uid.py data/parquet/ghostbuster_gpt35t.parquet \
        -o data/parquet/ghostbuster_gpt35t_uidfix.parquet
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

from epai.ai_detection.prepare.combine_domains import ID_COL, TARGET_COL, UID_COL
from epai.utils.label_utils import LABEL_SEP

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

RULE = 78
MAX_LISTED = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("parquets", nargs="+", type=Path, metavar="PARQUET",
                    help="Corpus parquet(s) whose uid should be rebuilt.")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output path (default: rewrite each input in place). "
                         "Only valid with a single input file.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and stop. Nothing is asked, nothing "
                         "is written.")
    ap.add_argument("--yes", action="store_true",
                    help="Skip the confirmation prompt (for pipelines).")
    ap.add_argument("--list", type=int, default=MAX_LISTED, metavar="N",
                    help=f"Example rows to show per file (default: {MAX_LISTED}).")
    return ap.parse_args(argv)


def expected_uid(df: pd.DataFrame) -> pd.Series:
    """``"{id}:{generated_by}"`` per row -- the definition, in one place."""
    return df[ID_COL].astype(str) + LABEL_SEP + df[TARGET_COL].astype(str)


def plan_for(path: Path, df: pd.DataFrame, limit: int) -> dict:
    """What would change in one file, and whether it is allowed to.

    The duplicate check runs on the REBUILT column rather than the current one:
    a file whose uid is currently unique can still have a rebuild that collides,
    and that is the only moment the collision is cheap to refuse.
    """
    wanted = expected_uid(df)
    current = df[UID_COL].astype(str)
    changed = current != wanted

    dupes = wanted[wanted.duplicated(keep=False)]
    # islice over a generator, not a full list sliced afterwards: the frame is
    # thousands of rows and only `limit` of them are ever printed.
    examples = list(itertools.islice(
        ((c, w) for c, w, ch in zip(current, wanted, changed) if ch), limit
    ))

    return {
        "path": path,
        "rows": len(df),
        "n_changed": int(changed.sum()),
        "examples": examples,
        "duplicate_uids": Counter(dupes.tolist()),
        "wanted": wanted,
    }


def print_plan(plan: dict, out_path: Path, limit: int) -> None:
    path, n, total = plan["path"], plan["n_changed"], plan["rows"]
    print("-" * RULE)
    print(f"{BOLD}{path}{RESET}")
    print(f"  rows            : {total}")
    if not n:
        print(f"  {GREEN}uid already matches '{{id}}{LABEL_SEP}{{generated_by}}' "
              f"in every row — nothing to do{RESET}")
        return

    share = f"{n} ({100.0 * n / total:.2f}%)" if total else str(n)
    print(f"  uid to rebuild  : {share} of rows")
    print(f"  {BOLD}before{RESET}  ->  {BOLD}after{RESET}")
    for before, after in plan["examples"][:limit]:
        print(f"    {before}\n      -> {after}")
    if n > len(plan["examples"]):
        print(f"    ... and {n - len(plan['examples'])} more")

    dupes = plan["duplicate_uids"]
    if dupes:
        print(f"  {RED}rebuilt uid would NOT be unique: "
              f"{len(dupes)} value(s) on more than one row{RESET}")
        for uid, count in list(dupes.most_common(limit)):
            print(f"    {uid}  x{count}")
    print(f"  write           : {out_path}"
          + ("  (in place)" if out_path == path else ""))


def write_parquet(df: pd.DataFrame, target: Path) -> None:
    """Write ``df`` to ``target`` atomically (temp file in the same dir + rename).

    The same routine as ``rename_column_values.py``, and for the same reason:
    these files are hundreds of megabytes, a partial write is indistinguishable
    from a corrupt corpus, and there is no git history to recover one from.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def confirm(prompt: str) -> bool:
    """Ask for an explicit ``yes``. Anything else — including EOF — is a no."""
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        print("\nNo input available (stdin is not a terminal). Aborting.")
        return False
    return answer in {"yes", "y"}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    missing = [p for p in args.parquets if not p.is_file()]
    if missing:
        print("Not found: " + ", ".join(str(p) for p in missing), file=sys.stderr)
        return 1
    if args.out and len(args.parquets) > 1:
        print("--out names one output path but "
              f"{len(args.parquets)} input files were given. Rewrite in place, "
              "or run one file at a time.", file=sys.stderr)
        return 2

    frames: dict[Path, pd.DataFrame] = {}
    plans: list[dict] = []
    for path in args.parquets:
        df = pd.read_parquet(path)
        missing_cols = [c for c in (UID_COL, ID_COL, TARGET_COL)
                        if c not in df.columns]
        if missing_cols:
            print(f"{path}: missing column(s) {missing_cols}; this is not a "
                  f"corpus parquet. It has: {list(df.columns)}", file=sys.stderr)
            return 2
        frames[path] = df
        plans.append(plan_for(path, df, args.list))

    print("=" * RULE)
    for plan in plans:
        print_plan(plan, args.out or plan["path"], args.list)
    print("=" * RULE)

    blocked = [p for p in plans if p["duplicate_uids"] and p["n_changed"]]
    if blocked:
        print(f"\n{RED}Refusing: the rebuild would produce duplicate uids in "
              f"{len(blocked)} file(s).{RESET}")
        print("  A uid identifies a row. Two rows sharing one is the same "
              "document twice to everything downstream.")
        print("  This usually means a rename collapsed two generator labels "
              "onto one. Fix the labels first.")
        return 1

    total = sum(p["n_changed"] for p in plans)
    touched = sum(1 for p in plans if p["n_changed"])
    if not total:
        print(f"\n{GREEN}Every uid already matches. Nothing written.{RESET}")
        return 0

    print(f"\nTotal: {total} uid(s) rebuilt across {touched} of "
          f"{len(plans)} file(s).")
    if args.dry_run:
        print("\n--dry-run: nothing was written.")
        return 0
    if not args.yes and not confirm(
        f"\n{YELLOW}Rewrite {touched} file(s)?{RESET} type 'yes' to confirm: "
    ):
        print("Aborted. No file written.")
        return 1

    print()
    for plan in plans:
        if not plan["n_changed"]:
            continue
        path = plan["path"]
        df = frames[path]
        df[UID_COL] = plan["wanted"]
        target = args.out or path
        write_parquet(df, target)
        print(f"  {target}: {plan['n_changed']} uid(s) rebuilt")

    print(f"\n{GREEN}Done — {total} uid(s) rebuilt.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
