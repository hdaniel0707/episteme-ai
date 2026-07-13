"""Dataset analysis CLI: generate summary statistics and plots from a database.

Usage:
    python -m workflow.src.main_analyse_dataset
    python -m workflow.src.main_analyse_dataset --db extended --out /path/to/output
    python -m workflow.src.main_analyse_dataset --db training
    python -m workflow.src.main_analyse_dataset --db /absolute/path/to/custom.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from workflow.src.iwf import paths
from workflow.src.iwf.evaluation.data_analysis import analyse


def resolve_db_path(db_spec: str) -> Path:
    """Resolve --db argument to an actual database path.

    Accepts:
      - "extended", "training", "raw" → paths.SQL_DIR / "{db_spec}.db"
      - bare filename (no /) → paths.SQL_DIR / filename
      - absolute path → used as-is
    """
    if "/" in db_spec or "\\" in db_spec:
        # Absolute or relative path
        p = Path(db_spec)
        if p.is_absolute():
            return p
        return Path.cwd() / p
    else:
        # Check if it's a known literal
        if db_spec in ("extended", "training", "raw"):
            return paths.SQL_DIR / f"{db_spec}.db"
        # Treat as filename under SQL_DIR
        return paths.SQL_DIR / db_spec


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Analyze a dataset database: generate summary tables and plots."
    )
    ap.add_argument(
        "--db",
        default="extended",
        help="Database to analyze: 'extended', 'training', 'raw', filename, or absolute path "
             "(default: extended)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output directory (default: workflow/data/analysis)",
    )
    args = ap.parse_args()

    db_path = resolve_db_path(args.db)
    if not db_path.is_file():
        print(f"[analyse] error: database not found at {db_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else paths.DATA_DIR / "analysis"

    print(f"[analyse] database: {db_path}")
    print(f"[analyse] output: {out_dir}")

    result = analyse(db_path, out_dir)

    print(f"\n[analyse] results:")
    print(f"  Total papers: {result['n_papers']:,}")
    print(f"  Training-ready: {result['n_training_complete']:,}")
    print(f"  Output: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
