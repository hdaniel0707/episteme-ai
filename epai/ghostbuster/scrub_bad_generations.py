r"""Delete generated .txt files that look like a failed call, not a document.

``generate.py`` only records a failure when the API call itself raises or
returns nothing — a refusal ("I'm sorry, but I can't help with that") or an
error body the endpoint returned as if it were content is a normal-looking,
non-empty reply, so it is written to disk and counted as a success. Nothing in
the existing pipeline catches that until ``check_parquet_completeness.py``
scans the finished corpus, by which point the offending row is buried in a
300 MB parquet instead of being one named .txt file.

This scans the same text tree ``generate.py`` writes to and deletes any file
whose content is unambiguously a failed call rather than prose — the same two
ERROR-tier signatures ``check_parquet_completeness.py`` uses (a structural
marker anywhere, e.g. ``<think>`` or ``"error":``; or a refusal/error phrase in
the first 80 characters), not its WARN-tier "phrase anywhere in the document",
which is too permissive to auto-delete on: a document that merely *mentions*
rate limits midway through is not a failed call.

Deleting the file is the fix, not a report: the fork's own generate.py and
fix_empty_generations.py both skip a file that already exists, so the next
``--only generate`` (or ``run_full_pipeline.py``'s own retry-pass loop) simply
re-issues that one call and refills the gap. Blank files are
``fix_empty_generations.py``'s job and are left alone here.

Usage:
    uv run python epai/ghostbuster/scrub_bad_generations.py --cfg cfg/ghostbuster/gb_gptoss120b.yaml
    uv run python epai/ghostbuster/scrub_bad_generations.py --cfg cfg/ghostbuster/gb_gptoss120b.yaml --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from epai.ghostbuster.config import (
    BASE_DIR,
    DEFAULT_CFG_PATH,
    GB_DATA_DIR,
    domains,
    load_pipeline_config,
    output_dir_name,
    types,
)
from epai.scripts.check_parquet_completeness import (
    LEADING_WINDOW,
    TEXT_ERROR_SIGNATURES_PHRASE,
    TEXT_ERROR_SIGNATURES_STRUCTURAL,
)

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
DIM = "\033[2m"
RESET = "\033[0m"


def looks_like_failed_call(text: str) -> str | None:
    """The matched signature, or None if the text reads as real prose.

    Only the two tiers ``check_parquet_completeness.py`` treats as ERROR, not
    its "phrase anywhere in the document" WARN tier — see the module
    docstring for why the WARN tier is left for a human to read, not deleted
    on automatically.
    """
    low = text.lower()
    lead = low[:LEADING_WINDOW]
    for sig in TEXT_ERROR_SIGNATURES_STRUCTURAL:
        if sig in low:
            return sig
    for sig in TEXT_ERROR_SIGNATURES_PHRASE:
        if sig in lead:
            return sig
    if low.lstrip()[:1] in ("{", "<"):
        return "leading '{' or '<'"
    return None


def scan_dir(root: Path) -> list[tuple[Path, str]]:
    """``(path, matched signature)`` for every offending file under ``root``.

    Recursive, so it works for reuter's ``<author>/<n>.txt`` layout as well as
    essay/wp's flat one. Blank files are skipped — they are
    ``fix_empty_generations.py``'s job, and an empty string matches none of
    the signatures anyway.
    """
    hits: list[tuple[Path, str]] = []
    if not root.is_dir():
        return hits
    for path in sorted(root.rglob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        sig = looks_like_failed_call(text)
        if sig:
            hits.append((path, sig))
    return hits


def scrub(cfg: dict, dry_run: bool = False) -> int:
    """Delete every offending file under this config's generator directories.

    Only the directories this config's ``generate.domains``/``generate.types``
    would themselves write to — never the seeded ``human``/``prompts`` inputs,
    and never another variant's directory sitting alongside them in the fork's
    shared tree.
    """
    total = 0
    for domain in domains(cfg):
        for gen_type in types(cfg):
            root = GB_DATA_DIR / domain / output_dir_name(cfg, gen_type)
            for path, sig in scan_dir(root):
                try:
                    rel = path.relative_to(BASE_DIR)
                except ValueError:
                    rel = path
                verb = "would delete" if dry_run else "deleted"
                print(f"  {RED}{verb}{RESET} {rel}  {DIM}[{sig!r}]{RESET}")
                if not dry_run:
                    path.unlink()
                total += 1
    return total


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cfg", type=Path, default=DEFAULT_CFG_PATH,
                    help=f"pipeline config YAML (default: "
                         f"{DEFAULT_CFG_PATH.relative_to(BASE_DIR)})")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be deleted without deleting it")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_pipeline_config(args.cfg)
    n = scrub(cfg, args.dry_run)
    if n:
        verb = "Would delete" if args.dry_run else "Deleted"
        print(f"\n{YELLOW}{verb} {n} file(s) that looked like a failed call.{RESET}")
        if not args.dry_run:
            print(
                "  Re-run generate to refill them (it skips what already "
                "exists):\n"
                "    uv run python epai/ghostbuster/run_full_pipeline.py --only generate"
            )
    else:
        print(f"{GREEN}Clean — nothing looked like a failed call.{RESET}")


if __name__ == "__main__":
    main()
