r"""Fold ``capture_baseline_judge.py`` sidecars back into the real parquets.

Each ``capture_baseline_judge.py --model X`` run leaves
``data/parquet/tmp/judge_scores/X/<stem>.parquet`` = ``uid`` + one
``llm_judge_score_*`` column. This joins every sidecar column onto the matching
``data/parquet/<stem>.parquet`` by ``uid`` -- the one expensive read+write per
file, done once, after every model has finished.

The parsed-mode twin of ``merge_captured_logprobs.py``; same guards. Refuses to
overwrite a column the real parquet already has (``--overwrite`` to allow it),
refuses if a sidecar ``uid`` is missing from the real file, refuses below
95 % coverage (``--allow-low`` / ``--min-coverage``).

    uv run python epai/ai_detection/score/merge_captured_judge_scores.py --dry-run
    uv run python epai/ai_detection/score/merge_captured_judge_scores.py
    uv run python epai/ai_detection/score/merge_captured_judge_scores.py \
        --workspaces data/parquet/tmp/judge_scores/gpt56luna --only manugram_den
    uv run python epai/ai_detection/score/merge_captured_judge_scores.py --out-dir data/parquet/tmp/merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from epai.utils.parquet_utils import load_parquet, save_parquet

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
CAPTURE_DIR = PARQUET_DIR / "tmp" / "judge_scores"
CORE_STEMS = ["ghostbuster_orig", "manugram_gb", "manugram_sci", "manugram_den"]
KEY_COL = "uid"
PREFIX = "llm_judge_score_"


def _sidecar_columns(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    cols = [c for c in df.columns if c.startswith(PREFIX)]
    if KEY_COL not in df.columns or not cols:
        raise SystemExit(f"{path}: expected {KEY_COL!r} + {PREFIX}* column(s), "
                         f"got {list(df.columns)}")
    if not df[KEY_COL].is_unique:
        raise SystemExit(f"{path}: {KEY_COL!r} is not unique")
    return df[[KEY_COL, *cols]]


def merge_stem(stem: str, workspaces: list[Path], args) -> None:
    src = PARQUET_DIR / f"{stem}.parquet"
    out = (Path(args.out_dir) / f"{stem}.parquet") if args.out_dir else src
    sidecars = [(w, w / f"{stem}.parquet") for w in workspaces
                if (w / f"{stem}.parquet").is_file()]
    print(f"\n{'-' * 74}\n{stem}   {len(sidecars)} sidecar(s)")
    if not sidecars:
        print("  no sidecar for this stem -- skipped")
        return
    if not src.is_file():
        print(f"  {src} not on disk -- skipped")
        return

    new: dict[str, pd.DataFrame] = {}
    for w, path in sidecars:
        part = _sidecar_columns(path)
        for c in [x for x in part.columns if x != KEY_COL]:
            if c in new:
                raise SystemExit(f"{c} appears in two workspaces ({w} and another)")
            new[c] = part[[KEY_COL, c]]
    print("  columns: " + ", ".join(sorted(new)))

    real_keys = set(pq.read_table(src, columns=[KEY_COL]).column(0).to_pylist())
    existing = set(pq.read_schema(src).names)
    collide = sorted(c for c in new if c in existing)
    if collide and not args.overwrite:
        raise SystemExit(f"  {stem}: already has {collide} -- pass --overwrite to replace")

    plan: list[tuple[str, pd.DataFrame]] = []
    for c, part in new.items():
        missing = set(part[KEY_COL]) - real_keys
        if missing:
            raise SystemExit(f"  {c}: {len(missing)} sidecar uid(s) not in {stem} "
                             f"(e.g. {list(missing)[:3]})")
        cov = int(part[c].notna().sum())
        frac = cov / len(real_keys) if real_keys else 0.0
        flag = "  LOW" if frac < args.min_coverage else ""
        print(f"    {c}: {cov}/{len(real_keys)} rows scored ({frac:.0%}){flag}")
        if flag and not args.allow_low:
            raise SystemExit(f"  {c}: coverage {frac:.0%} < {args.min_coverage:.0%} "
                             "-- finish the capture, or pass --allow-low")
        plan.append((c, part))

    if args.dry_run:
        print("  [dry run] -- nothing written")
        return

    df = load_parquet(src)
    for c, part in plan:
        merged = df[[KEY_COL]].merge(part, on=KEY_COL, how="left")
        df[c] = merged[c].to_numpy()
    save_parquet(df, out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--workspaces", nargs="+", type=Path, default=None,
                    help="capture dirs (default: every dir under "
                         "data/parquet/tmp/judge_scores/)")
    ap.add_argument("--only", nargs="+", choices=CORE_STEMS, metavar="STEM")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write merged <stem>.parquet here instead of in place")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an llm_judge_score_* column the real file already has")
    ap.add_argument("--min-coverage", type=float, default=0.95,
                    help="warn/refuse below this scored fraction (default 0.95)")
    ap.add_argument("--allow-low", action="store_true",
                    help="merge even if a column is below --min-coverage")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    workspaces = args.workspaces or (
        sorted(d for d in CAPTURE_DIR.iterdir() if d.is_dir())
        if CAPTURE_DIR.is_dir() else []
    )
    if not workspaces:
        raise SystemExit(f"no capture workspaces found under {CAPTURE_DIR}")
    print(f"workspaces: {[str(w) for w in workspaces]}")
    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for stem in (args.only or CORE_STEMS):
        merge_stem(stem, workspaces, args)

    print(f"\n{'=' * 74}\ndone")


if __name__ == "__main__":
    main()
