r"""Split the 3-class ``ghostbuster_orig.parquet`` into its two benchmark
variants: the RAW GPT-3.5 pairing and the STANDARDISED one.

WHAT IS IN THE SOURCE
--------------------
``ghostbuster_orig.parquet`` ships one human row and TWO machine rows per source
document (``build_ghostbuster_multi.py``: "keep both, filter later"):

    generated_by      text the features were computed on
    human             standardised
    gpt35t:0000A      RAW  (== text_raw; standardize_text.py never ran on it)
    gpt35ts:0000A     standardised (the same generation, standardize_text.py run)

``gpt35t`` and ``gpt35ts`` are the identical GPT-3.5 generation -- byte-identical
``text_raw`` -- differing only in whether the typographic scrub reached the
``text`` the scorers read. The human class exists ONLY standardised, but its
``text_raw`` is verbatim the Ghostbuster original: ``essay:1``'s ``text_raw``
equals ``open('external/ghostbuster/data/essay/human/1.txt').read()`` byte for
byte, no restructure truncation. So ``human text <- text_raw`` IS the combine
with the submodule's raw human data -- no separate merge step.

WHAT THIS WRITES
---------------
    ghostbuster_orig_3class.parquet   an untouched copy of the source, kept
                                      because the gpt35ts rows and the raw human
                                      text survive nowhere else. Made once; not
                                      overwritten.

    ghostbuster_orig_stand.parquet    human + gpt35ts, both standardised -- the
                                      internally consistent pairing, and the one
                                      every other MANUGRAM split matches.
                                      ``gpt35ts:0000A`` is relabelled to
                                      ``gpt35t:0000A`` so the generator name is
                                      sane and the two variants line up
                                      column-for-column. Every feature is carried
                                      through unchanged -- the standardised
                                      machine text was already scored, so there
                                      is nothing to recompute. THIS FILE IS DONE.

    ghostbuster_orig.parquet          human + gpt35t, both RAW -- the "online
                                      Ghostbuster benchmark, unchanged" variant.
                                      The gpt35t machine rows are carried through
                                      unchanged (already raw-scored). The human
                                      rows get ``text <- text_raw`` and EVERY
                                      computed feature column NULLED, because
                                      they were scored on standardised text.
                                      They must be re-scored: follow
                                      GHOSTBUSTER_ORIG_SPLIT.md.

Run:
    uv run python epai/ai_detection/prepare/split_ghostbuster_orig.py --dry-run
    uv run python epai/ai_detection/prepare/split_ghostbuster_orig.py
    uv run python epai/ai_detection/prepare/split_ghostbuster_orig.py --variant stand
    uv run python epai/ai_detection/prepare/split_ghostbuster_orig.py --force
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.parquet_utils import coerce_dtypes, load_parquet, save_parquet

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
DIM = "\033[2m"
RESET = "\033[0m"

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"

HUMAN = "human"
RAW_GEN = "gpt35t:0000A"     # machine, features on raw text
STD_GEN = "gpt35ts:0000A"    # machine, features on standardised text
CANON_GEN = "gpt35t:0000A"   # the label both variants ship in the end

META_COLS = ["uid", "id", "domain", "text", "text_raw", "generated_by"]

#: Every non-meta column must start with one of these. The scalar-score families
#: are additionally coerced to float64 on write so a half-nulled column is not
#: stored as Arrow ``null`` type (see parquet_utils.coerce_dtypes).
FEATURE_PREFIXES = ("emb_", "binoculars_", "fastdetect_", "detector_", "llm_judge_")
SCALAR_SCORE_PREFIXES = (
    "binoculars_score",
    "fastdetect_crit_", "fastdetect_prob_", "fastdetect_ntok_",
    "detector_score_", "detector_ntok_",
    "llm_judge_score_",
)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    feat = [c for c in df.columns if c not in META_COLS]
    stray = [c for c in feat if not c.startswith(FEATURE_PREFIXES)]
    if stray:
        raise SystemExit(
            f"{RED}non-meta column(s) match no known feature prefix "
            f"{FEATURE_PREFIXES}: {stray}{RESET}\n"
            f"add the prefix to FEATURE_PREFIXES (and, if it is a scalar score, "
            f"to SCALAR_SCORE_PREFIXES) once you know it should be nulled for the "
            f"raw human rows."
        )
    return feat


def _rebuild_uid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["uid"] = df["id"].astype(str) + ":" + df["generated_by"].astype(str)
    if df["uid"].duplicated().any():
        dup = df.loc[df["uid"].duplicated(), "uid"].unique()[:5]
        raise SystemExit(f"{RED}duplicate uid after rebuild: {list(dup)}{RESET}")
    return df


def _validate_source(df: pd.DataFrame) -> None:
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{RED}source is missing meta column(s): {missing}{RESET}")
    gens = set(df["generated_by"].unique())
    expected = {HUMAN, RAW_GEN, STD_GEN}
    if gens != expected:
        raise SystemExit(
            f"{RED}expected exactly {sorted(expected)} in 'generated_by', "
            f"got {sorted(gens)}{RESET}\nThis script splits the 3-class file; "
            f"run it on the source before any earlier split has trimmed it."
        )
    counts = df["generated_by"].value_counts()
    spread = counts.max() - counts.min()
    if spread > 0.02 * counts.max():
        print(f"{YELLOW}NOTE: class counts differ by {spread} "
              f"({counts.to_dict()}) — a per-id class is missing somewhere.{RESET}")
    span = df.groupby("id")["generated_by"].nunique()
    if (span != 3).any():
        n = int((span != 3).sum())
        print(f"{YELLOW}NOTE: {n} id(s) do not carry all 3 classes.{RESET}")


def _fill_report(df: pd.DataFrame, cols: list[str]) -> str:
    def filled(s: pd.Series) -> float:
        return float(s.map(lambda v: not (
            v is None
            or (isinstance(v, float) and pd.isna(v))
            or (hasattr(v, "__len__") and not isinstance(v, str) and len(v) == 0)
        )).mean())

    part = [c for c in cols if 0.0 < filled(df[c]) < 0.999]
    empty = [c for c in cols if filled(df[c]) == 0.0]
    bits = []
    if empty:
        bits.append(f"{RED}{len(empty)} empty{RESET}")
    if part:
        bits.append(f"{YELLOW}{len(part)} partial{RESET}: {', '.join(part[:6])}"
                    + (" …" if len(part) > 6 else ""))
    return "  ".join(bits) or f"{GREEN}all {len(cols)} filled{RESET}"


def build_stand(src: pd.DataFrame, out: Path, dry: bool) -> None:
    part = src[src["generated_by"].isin([HUMAN, STD_GEN])].reset_index(drop=True)
    part = part.copy()
    part.loc[part["generated_by"] == STD_GEN, "generated_by"] = CANON_GEN
    part = _rebuild_uid(part)
    feat = _feature_cols(part)
    print(f"\n{GREEN}ghostbuster_orig_stand{RESET}  "
          f"{len(part)} rows  ({int((part['generated_by'] == HUMAN).sum())} human / "
          f"{int((part['generated_by'] == CANON_GEN).sum())} {CANON_GEN})")
    print(f"  relabelled {STD_GEN} -> {CANON_GEN}")
    print(f"  features: {_fill_report(part, feat)}")
    if dry:
        print(f"  {DIM}-> would write {out}{RESET}")
        return
    part = coerce_dtypes(part, {c: "float64" for c in feat
                                if c.startswith(SCALAR_SCORE_PREFIXES)})
    save_parquet(part[META_COLS + [c for c in part.columns if c not in META_COLS]],
                 out)


def build_orig(src: pd.DataFrame, out: Path, dry: bool) -> None:
    part = src[src["generated_by"].isin([HUMAN, RAW_GEN])].reset_index(drop=True)
    part = _rebuild_uid(part.copy())
    feat = _feature_cols(part)
    human = (part["generated_by"] == HUMAN).to_numpy()

    changed = int((part.loc[human, "text"] != part.loc[human, "text_raw"]).sum())
    n_human = int(human.sum())

    if not dry:
        part["text"] = np.where(human, part["text_raw"], part["text"])
        for c in feat:
            col = part[c].to_numpy(dtype=object, copy=True)
            col[human] = None
            part[c] = col
        part = coerce_dtypes(part, {c: "float64" for c in feat
                                    if c.startswith(SCALAR_SCORE_PREFIXES)})

    print(f"\n{GREEN}ghostbuster_orig{RESET} (raw)  {len(part)} rows  "
          f"({n_human} human / {int((~human).sum())} {RAW_GEN})")
    print(f"  human text <- text_raw  ({changed}/{n_human} rows actually differ)")
    print(f"  nulled {len(feat)} feature column(s) for the {n_human} human rows "
          f"— re-score per GHOSTBUSTER_ORIG_SPLIT.md")
    print(f"  {RAW_GEN} rows carried through unchanged: "
          f"{_fill_report(part[part['generated_by'] == RAW_GEN], feat)}")
    if dry:
        print(f"  {DIM}-> would write {out}{RESET}")
        return
    save_parquet(part[META_COLS + [c for c in part.columns if c not in META_COLS]],
                 out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in", dest="src", type=Path,
                    default=PARQUET_DIR / "ghostbuster_orig.parquet",
                    help="the 3-class source (default: %(default)s)")
    ap.add_argument("--backup", type=Path,
                    default=PARQUET_DIR / "ghostbuster_orig_3class.parquet",
                    help="untouched copy of the source (default: %(default)s)")
    ap.add_argument("--out-dir", type=Path, default=PARQUET_DIR)
    ap.add_argument("--drop-cols", nargs="+", default=[], metavar="COL",
                    help="columns to drop from BOTH variants before writing "
                         "(e.g. llm_judge_score_sonnet5_v3 — see "
                         "GHOSTBUSTER_ORIG_SPLIT.md)")
    ap.add_argument("--variant", choices=("both", "stand", "orig"), default="both")
    ap.add_argument("--force", action="store_true",
                    help="overwrite outputs that already exist")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the partition, write nothing")
    args = ap.parse_args()

    out_stand = args.out_dir / "ghostbuster_orig_stand.parquet"
    out_orig = args.out_dir / "ghostbuster_orig.parquet"
    wants_orig = args.variant in ("both", "orig")
    wants_stand = args.variant in ("both", "stand")

    # Read from the backup when it exists — the source may already have been
    # trimmed to the raw variant by a prior run.
    read_from = args.src
    if args.backup.is_file():
        read_from = args.backup
        print(f"{DIM}reading from the existing backup {args.backup.name} "
              f"(source may already be trimmed){RESET}")
    if not read_from.is_file():
        raise SystemExit(f"{RED}not found: {read_from}{RESET}")

    targets = ([out_stand] if wants_stand else []) + ([out_orig] if wants_orig else [])
    clash = [p for p in targets if p.is_file() and p != args.src]
    # out_orig overwriting the source in place is expected, not a clash.
    if clash and not (args.force or args.dry_run):
        raise SystemExit(
            f"{RED}output(s) exist: {[p.name for p in clash]} — pass --force "
            f"to overwrite{RESET}")

    df = load_parquet(read_from)
    _validate_source(df)
    print(f"\nsource: {read_from}  ({len(df)} rows, {df['id'].nunique()} ids, "
          f"{df['generated_by'].nunique()} classes)")

    if args.drop_cols:
        absent = [c for c in args.drop_cols if c not in df.columns]
        if absent:
            print(f"{YELLOW}--drop-cols not in source (ignored): {absent}{RESET}")
        drop = [c for c in args.drop_cols if c in df.columns]
        if drop:
            df = df.drop(columns=drop)
        print(f"dropping from both variants: {drop or '(none)'}")

    # Back up BEFORE build_orig overwrites ghostbuster_orig.parquet in place.
    if wants_orig and not args.dry_run and not args.backup.is_file():
        shutil.copy2(args.src, args.backup)
        print(f"{GREEN}backed up{RESET} {args.src.name} -> {args.backup.name}")

    if wants_stand:
        build_stand(df, out_stand, args.dry_run)
    if wants_orig:
        build_orig(df, out_orig, args.dry_run)

    if not args.dry_run:
        print(f"\n{GREEN}done.{RESET}")
        if wants_stand:
            print(f"  {out_stand.name}: finished — verify with "
                  f"check_parquet_completeness.py")
        if wants_orig:
            print(f"  {out_orig.name}: human rows need re-scoring — "
                  f"see GHOSTBUSTER_ORIG_SPLIT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
