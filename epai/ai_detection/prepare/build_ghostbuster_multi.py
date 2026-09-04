r"""Fold the per-generator ghostbuster corpora into one multi-generator file.

The mirror of ``science_v3_multi_0811A`` on the essay/wp/reuter side: a single
parquet whose ``human`` class appears once and whose machine classes are one
per generator, told apart by ``generated_by``. Downstream keeps or drops a
generator with a ``generated_by`` filter -- nothing here has to be re-run to
change the mix.

WHAT GOES IN
------------
    class                 source parquet                       text
    human                 ghostbuster_gpt56luna    (human rows)  standardised
    gpt56luna:0701A       ghostbuster_gpt56luna    (ai rows)     standardised
    gpt54mini:0701A       ghostbuster_gpt54mini    (ai rows)     standardised
    gpt35t:0000A          ghostbuster_gpt35t       (ai rows)     RAW (unstd.)
    gpt35ts:0000A         ghostbuster_gpt35ts      (ai rows)     standardised

The human class is taken from ONE source (gpt56luna, 3000 rows, standardised) --
the other files' human rows are the same texts by a different route (2994 rows,
and raw in gpt35t) and copying them would only add duplicates and a schema seam.

``gpt35t`` and ``gpt35ts`` are the same generations -- ``gpt35ts`` is
``standardize_text.py`` already run on ``gpt35t`` -- and both ship
``generated_by == "gpt35t:0000A"``. The standardised one is relabelled to
``gpt35ts:0000A`` here so the two are separable by the same ``generated_by``
filter everything else uses. Keep both, filter later.

WHAT IS AND IS NOT DONE
----------------------
Rows are concatenated with their detector-suite columns exactly as each source
carries them -- every source is already row-aligned with itself, so this is a
union + NaN-fill, not a carry-by-uid. It does NOT score anything, does NOT
standardise the raw ``gpt35t`` rows, and does NOT touch the ``strip_boilerplate``
provenance difference between the classes (gpt35t/gpt54mini were built with it
on, gpt56luna with it off -- a documented corpus property, not this script's
job). Filling the resulting holes is the separate ``--from embed`` scoring pass.

    uv run python epai/ai_detection/prepare/build_ghostbuster_multi.py \
        -o data/parquet/ghostbuster_multi.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from epai.utils.parquet_utils import coerce_dtypes, load_parquet, save_parquet

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
DIM = "\033[2m"
RESET = "\033[0m"

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"

ID_COL = "id"
UID_COL = "uid"
TEXT_COL = "text"
RAW_COL = "text_raw"
TARGET_COL = "generated_by"
DOMAIN_COL = "domain"

META_COLS = [UID_COL, ID_COL, DOMAIN_COL, TEXT_COL, RAW_COL, TARGET_COL]

# Score columns are scalar floats and may be present in one source and absent in
# another; Arrow writes an absent-then-unioned column as type ``null`` unless it
# is coerced first (see parquet_utils.coerce_dtypes). Embedding / reason / cue
# columns are lists or strings and are left as object.
SCORE_PREFIXES = (
    "binoculars_score",
    "fastdetect_crit_", "fastdetect_prob_", "fastdetect_ntok_",
    "detector_score_", "detector_ntok_",
    "llm_judge_score_",
)

# (source stem, which rows, relabel generated_by to)
SOURCES = [
    ("ghostbuster_gpt56luna", "human", None),
    ("ghostbuster_gpt56luna", "ai", None),
    ("ghostbuster_gpt54mini", "ai", None),
    ("ghostbuster_gpt35t", "ai", None),
    ("ghostbuster_gpt35ts", "ai", "gpt35ts:0000A"),
]


def _is_score_col(c: str) -> bool:
    return any(c.startswith(p) for p in SCORE_PREFIXES)


def take_rows(df: pd.DataFrame, which: str) -> pd.DataFrame:
    human = df[TARGET_COL] == "human"
    return df[human].copy() if which == "human" else df[~human].copy()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-o", "--out", type=Path,
                    default=PARQUET_DIR / "ghostbuster_multi.parquet")
    ap.add_argument("--parquet-dir", type=Path, default=PARQUET_DIR,
                    help="where the ghostbuster_<gen>.parquet sources live")
    ap.add_argument("--source", action="append", nargs="+", metavar="ARG",
                    help="Override the hardcoded SOURCES list above. Repeatable: "
                         "--source STEM WHICH [RELABEL], e.g. "
                         "--source ghostbuster_gptoss120b ai. STEM is a "
                         "data/parquet/<stem>.parquet file (no extension), WHICH "
                         "is 'human' or 'ai', RELABEL (optional) renames "
                         "generated_by for that source's rows. Given at all, this "
                         "replaces SOURCES entirely rather than adding to it.")
    args = ap.parse_args()

    if args.source:
        sources = []
        for parts in args.source:
            if len(parts) not in (2, 3):
                ap.error(f"--source takes 2 or 3 values (stem, which[, relabel]), "
                         f"got {parts}")
            stem, which, *rest = parts
            if which not in ("human", "ai"):
                ap.error(f"--source {stem}: WHICH must be 'human' or 'ai', got {which!r}")
            sources.append((stem, which, rest[0] if rest else None))
    else:
        sources = SOURCES

    frames: list[pd.DataFrame] = []
    for stem, which, relabel in sources:
        path = args.parquet_dir / f"{stem}.parquet"
        if not path.is_file():
            raise SystemExit(f"{RED}not found: {path}{RESET}")
        df = take_rows(load_parquet(path), which)

        if RAW_COL not in df.columns:
            df[RAW_COL] = df[TEXT_COL]          # raw gpt35t: text already IS raw

        if relabel is not None:
            df[TARGET_COL] = relabel
        # uid is {id}:{generated_by}; rebuild so a relabelled class gets a fresh,
        # collision-free uid rather than keeping the source one.
        df[UID_COL] = df[ID_COL].astype(str) + ":" + df[TARGET_COL].astype(str)

        label = relabel or df[TARGET_COL].iloc[0]
        print(f"  {stem:24s} {which:5s} -> {label:18s} {len(df):5d} rows, "
              f"{sum(_is_score_col(c) or c.startswith('emb_') for c in df.columns)} suite col(s)")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)

    if merged[UID_COL].duplicated().any():
        dup = merged.loc[merged[UID_COL].duplicated(), UID_COL].unique()[:5]
        raise SystemExit(f"{RED}duplicate uid after merge: {list(dup)}{RESET}")

    # order columns: meta first, then suite columns alphabetically
    suite = sorted(c for c in merged.columns if c not in META_COLS)
    merged = merged[META_COLS + suite]

    score_cols = {c: "float64" for c in merged.columns if _is_score_col(c)}
    merged = coerce_dtypes(merged, score_cols)

    save_parquet(merged, args.out)

    # ---- coverage: which (class x column) cells are filled ------------------
    print(f"\n{GREEN}{len(merged)} rows{RESET} -> {args.out}")
    counts = merged[TARGET_COL].value_counts()
    print("\nclass counts:")
    for k, v in counts.items():
        print(f"  {k:20s} {v}")

    cov_cols = [c for c in merged.columns if _is_score_col(c) or c.startswith("emb_")
                or c.startswith("llm_judge_")]
    grp = merged.groupby(TARGET_COL)
    print(f"\ncoverage (% non-null per class) {DIM}-- what the fill pass must "
          f"compute{RESET}:")
    header = "  " + f"{'column':32s}" + "".join(f"{k[:12]:>13s}" for k in counts.index)
    print(header)
    for c in cov_cols:
        frac = grp[c].apply(lambda s: s.notna().mean())
        cells = "".join(
            f"{GREEN if frac[k] > 0.99 else (YELLOW if frac[k] > 0 else RED)}"
            f"{frac[k]*100:12.0f}%{RESET}" for k in counts.index
        )
        print(f"  {c:32s}{cells}")

    print(f"\nnext: run_holdout_pipeline.py --from embed on {args.out.name} "
          f"(scores only the empty cells above)")


if __name__ == "__main__":
    main()
