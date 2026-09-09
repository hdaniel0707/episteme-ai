r"""Split ``data/parquet/manugram_reph.parquet`` into its two source families.

MG-Reph is the human rows of the modern Ghostbuster suite and the scholarly
suite, each paired with ONE gpt-5.6-luna rephrase. So it is human vs. one
generator, and only the ``gpt56luna`` judge is scored on both classes -- the
gemma4 / qwen3 / gptoss / llama33 judges are null on the rephrase rows. That
makes it a poor fit for the pooled calibrations and the cross-generator /
multi-judge runs, so it is kept out of the default sweep (:data:`splits.DEFAULT`
/ ``rebuild_all.py``) and used only in targeted experiments -- on the whole
corpus, or on one of these halves:

    manugram_reph_sci.parquet   source_family == 'science'
                                (computer_science / mathematics / sociology)
    manugram_reph_gb.parquet    source_family == 'ghostbuster'
                                (essay / reuter / wp)

Every column is carried through unchanged; only the rows are partitioned. The
two halves share no document ``id``. They are registered as the opt-in splits
``mg_reph_sci`` / ``mg_reph_gb``:

    uv run python epai/ai_detection/trainv2/split_manugram_reph.py
    uv run python epai/ai_detection/trainv2/split_manugram_reph.py --dry-run
    uv run python epai/ai_detection/trainv2/split_manugram_reph.py --force

    uv run python epai/ai_detection/trainv2/run_zeroshot_performance.py --splits mg_reph_sci mg_reph_gb
    uv run python epai/ai_detection/trainv2/run_embedding_effectiveness.py --splits mg_reph_gb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE / "data" / "parquet"

GEN_COL = "generated_by"
FAMILY_COL = "source_family"
HUMAN_LABEL = "human"

#: source_family value -> output basename (== the splits.py registry key).
FAMILIES = {"science": "manugram_reph_sci", "ghostbuster": "manugram_reph_gb"}


def _judge_coverage(part: pd.DataFrame) -> list[str]:
    """The judge score columns with at least one non-null value in BOTH the
    human and the machine half -- the only ones usable for detection here."""
    h = part[part[GEN_COL] == HUMAN_LABEL]
    m = part[part[GEN_COL] != HUMAN_LABEL]
    return [
        c.removeprefix("llm_judge_score_").replace("_v3", "")
        for c in sorted(part.columns)
        if c.startswith("llm_judge_score_")
        and h[c].notna().any() and m[c].notna().any()
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in", dest="src", type=Path,
                    default=PARQUET_DIR / "manugram_reph.parquet")
    ap.add_argument("--out-dir", type=Path, default=PARQUET_DIR)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even when the output parquets already exist "
                         "(otherwise the run is a no-op)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the partition, write nothing")
    args = ap.parse_args()

    outs = {fam: args.out_dir / f"{stem}.parquet" for fam, stem in FAMILIES.items()}
    have = [p for p in outs.values() if p.exists()]
    if have and not (args.force or args.dry_run):
        if len(have) == len(outs):
            print("outputs already present, nothing to do (pass --force to rebuild): "
                  + ", ".join(p.name for p in outs.values()))
            return 0
        raise SystemExit(
            "some outputs exist, some are missing "
            + f"({[p.name for p in have]}); pass --force to rebuild both"
        )

    if not args.src.exists():
        raise SystemExit(f"not found: {args.src}")

    df = pd.read_parquet(args.src)
    if FAMILY_COL not in df.columns:
        raise SystemExit(f"{args.src.name}: no {FAMILY_COL!r} column to split on")

    seen = set(df[FAMILY_COL].dropna().unique())
    unknown = seen - set(FAMILIES)
    if unknown:
        raise SystemExit(f"unexpected {FAMILY_COL} value(s): {sorted(unknown)}")
    if df[FAMILY_COL].isna().any():
        raise SystemExit(f"{int(df[FAMILY_COL].isna().sum())} row(s) with null {FAMILY_COL!r}")

    print(f"source: {args.src}  ({len(df)} rows, {df['id'].nunique()} ids)\n")
    for fam, out in outs.items():
        part = df[df[FAMILY_COL] == fam].reset_index(drop=True)
        if part.empty:
            print(f"  {fam:11}  0 rows -- nothing to write")
            continue
        doms = ", ".join(sorted(part["domain"].unique()))
        gens = ", ".join(f"{g}={n}" for g, n in part[GEN_COL].value_counts().items())
        print(f"  {fam:11}  {len(part):6} rows  {part['id'].nunique()} ids  [{doms}]")
        print(f"  {'':11}  {gens}")
        print(f"  {'':11}  judges on both classes: {', '.join(_judge_coverage(part)) or 'NONE'}")
        if args.dry_run:
            print(f"  {'':11}  -> would write {out}\n")
            continue
        args.out_dir.mkdir(parents=True, exist_ok=True)
        part.to_parquet(out, index=False)
        print(f"  {'':11}  -> wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
