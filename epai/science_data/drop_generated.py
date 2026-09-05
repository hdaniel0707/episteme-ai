r"""Remove one generator's rows from a science generate shard (and/or parquet).

Use it when a generator has to be re-run under different settings — e.g.
qwen3.8:27b generated with its ``<think>`` pass on, now to be regenerated with
it off — and the already-produced rows must go so the corpus is consistent.

  shard    ``data/science/<id>_generate.jsonl`` — the append-only checkpoint
           ``generate_ai.py`` resumes from. Dropping a label's keys here is what
           makes the next run re-issue those calls; without it every one is
           skipped as "already done".
  parquet  optional. ``<id>_multi_<cfg>.parquet`` only carries a generator's
           rows once a run has finished the export step, so it usually has
           nothing to drop after an interrupted run — pass it anyway to be sure.

The label is the ``generated_by`` value, ``<model_slug>:<prompt_config>`` —
``qwen3827b:0811A``. A ``.bak`` copy is made before anything is rewritten.

    python epai/science_data/drop_generated.py --label qwen3827b:0811A
    python epai/science_data/drop_generated.py --label qwen3827b:0811A \
        --parquet data/science/science_v3_multi_0811A.parquet --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SHARD = BASE_DIR / "data/science/science_v3_generate.jsonl"

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
DIM = "\033[2m"
RESET = "\033[0m"


def _backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    print(f"{DIM}  backup → {bak}{RESET}")


def drop_from_shard(shard: Path, label: str, dry_run: bool) -> int:
    """Rewrite the JSONL shard without records whose key ends in ``|<label>``."""
    suffix = f"|{label}"
    kept: list[str] = []
    dropped = 0
    ok_dropped = 0
    with open(shard, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                kept.append(line.rstrip("\n"))
                continue
            if str(rec.get("key", "")).endswith(suffix):
                dropped += 1
                ok_dropped += 1 if rec.get("ok") else 0
            else:
                kept.append(line.rstrip("\n"))

    print(f"  {shard.name}: {dropped} record(s) match {label} "
          f"({ok_dropped} ok, {dropped - ok_dropped} failed) — "
          f"{len(kept)} kept")
    if dropped and not dry_run:
        _backup(shard)
        shard.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        print(f"{GREEN}  rewrote {shard.name}{RESET}")
    return dropped


def drop_from_parquet(path: Path, label: str, dry_run: bool) -> int:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    if "generated_by" not in t.column_names:
        print(f"{YELLOW}  {path.name}: no 'generated_by' column — skipped{RESET}")
        return 0
    mask = pc.not_equal(t.column("generated_by"), label)
    kept = t.filter(mask)
    dropped = t.num_rows - kept.num_rows
    print(f"  {path.name}: {dropped} row(s) match {label} — "
          f"{kept.num_rows} kept")
    if dropped and not dry_run:
        _backup(path)
        pq.write_table(kept, path)
        print(f"{GREEN}  rewrote {path.name}{RESET}")
    return dropped


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--label", required=True,
                    help="generated_by value to remove, e.g. qwen3827b:0811A")
    ap.add_argument("--shard", type=Path, default=DEFAULT_SHARD,
                    help=f"generate shard JSONL (default {DEFAULT_SHARD.name}); "
                         f"pass 'none' to skip the shard")
    ap.add_argument("--parquet", type=Path, nargs="+", default=[],
                    help="parquet file(s) to also drop the label's rows from")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    total = 0
    if str(args.shard).lower() != "none":
        if not args.shard.is_file():
            raise SystemExit(f"{RED}shard not found: {args.shard}{RESET}")
        total += drop_from_shard(args.shard, args.label, args.dry_run)

    for p in args.parquet:
        if not p.is_file():
            print(f"{YELLOW}not found, skipped: {p}{RESET}")
            continue
        total += drop_from_parquet(p, args.label, args.dry_run)

    if total == 0:
        print(f"\n{GREEN}Nothing matched {args.label}.{RESET}")
    elif args.dry_run:
        print(f"\n{YELLOW}dry-run — {total} record/row(s) would be removed.{RESET}")
    else:
        print(f"\n{GREEN}Removed {total} record/row(s) for {args.label}.{RESET}\n"
              f"  Re-run generate_ai.py to regenerate them.")


if __name__ == "__main__":
    main()
