r"""Copy the detector-suite columns a sibling corpus has already scored.

``science_v3_multi_0811A`` shares its ``human`` and ``gpt56luna:0811A`` rows,
verbatim, with ``science_v3_gpt56luna_0811A`` — same source papers, same human
abstracts, same gpt-5.6-luna generations (the multi generate run reused the
generate shard, so nothing was regenerated). Those 6000 rows were scored once
already; this copies their scores across by ``uid`` so only the new
open-weight-generator rows have to be computed.

WHAT IS COPIED, AND WHAT IS NOT
------------------------------
The suite's *output* columns that the source actually carries:

    emb_*  binoculars_score*  fastdetect_crit/prob/ntok_*  detector_score/ntok_*
    llm_judge_score_<judge>_<ver>   (for every judge still in the suite)
    llm_judge_reason_/cues_human_/cues_machine_/other_<cue_judge>_<ver>

**Not** ``llm_judge_cuevec_<cue_judge>_<ver>`` — that is derived, and
``add_cue_features.py`` rebuilds it from the cue lists (which ARE copied), so it
covers the new rows too. A suite column the source lacks
(``fastdetect_crit_llama31_8b``, the GenAI4Science judges, …) is reported and
left for the scoring stages to fill on every row.

ORDER
-----
Run AFTER ``standardize_text.py`` — the copied scores describe standardised
text, and this corpus's text must have been through the same transform first —
and BEFORE ``run_holdout_pipeline.py --from embed``, whose score stages then
skip every row this filled.

Re-run safe: a column that already holds values in the target is left alone
unless ``--overwrite``, so a carry after a partial scoring pass will not clobber
the freshly-computed rows.

    python epai/ai_detection/prepare/carry_suite_columns.py \
        data/parquet/science_v3_multi_0811A.parquet \
        --source data/parquet/science_v3_gpt56luna_0811A.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from epai.utils.parquet_utils import load_parquet, save_parquet

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
DIM = "\033[2m"
RESET = "\033[0m"

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SUITE = BASE_DIR / "cfg" / "detector_suite.yaml"

UID_COL = "uid"


def _keys(suite: dict, group: str) -> list[str]:
    return [i["key"] if isinstance(i, dict) else str(i) for i in (suite.get(group) or [])]


def _stem(name: str) -> str:
    return name.replace("-", "_").replace(".", "_")


def suite_output_columns(suite: dict) -> list[str]:
    """Every column the detector-suite scoring stages fill, by the same naming
    rules ``run_holdout_pipeline.py`` uses. The cue vector is deliberately absent
    — it is rebuilt, not carried."""
    ver = suite.get("judge_prompt_version", "v3")
    cue = _stem(suite.get("cue_judge", "gpt56luna"))
    cols: list[str] = [f"emb_{e}" for e in _keys(suite, "embedders")]

    for pair in _keys(suite, "binoculars_pairs"):
        cols.append("binoculars_score" if pair == "falcon-7b"
                    else f"binoculars_score_{_stem(pair)}")
    for pair in _keys(suite, "fastdetect_pairs"):
        p = _stem(pair)
        cols += [f"fastdetect_crit_{p}", f"fastdetect_prob_{p}", f"fastdetect_ntok_{p}"]
    for det in _keys(suite, "detectors"):
        cols += [f"detector_score_{det}", f"detector_ntok_{det}"]
    for judge in _keys(suite, "judges"):
        cols.append(f"llm_judge_score_{_stem(judge)}_{ver}")
    cols += [
        f"llm_judge_reason_{cue}_{ver}",
        f"llm_judge_cues_human_{cue}_{ver}",
        f"llm_judge_cues_machine_{cue}_{ver}",
        f"llm_judge_other_{cue}_{ver}",
    ]
    # de-dup, keep order
    return list(dict.fromkeys(cols))


def _is_null(v) -> bool:
    if v is None:
        return True
    if isinstance(v, (list, np.ndarray)):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def has_values(series: pd.Series) -> bool:
    return any(not _is_null(v) for v in series)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("target", type=Path, help="corpus to fill (standardised, unscored)")
    ap.add_argument("--source", type=Path, required=True,
                    help="corpus that already carries the scored columns")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="write here instead of rewriting the target in place")
    ap.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace target columns that already hold values "
                         "(default: leave them, so a carry after a scoring pass "
                         "is safe)")
    args = ap.parse_args()

    for p in (args.target, args.source, args.suite):
        if not p.is_file():
            raise SystemExit(f"{RED}not found: {p}{RESET}")

    suite = yaml.safe_load(args.suite.read_text()) or {}
    want = suite_output_columns(suite)

    src_cols = set(pq.ParquetFile(args.source).schema_arrow.names)
    present = [c for c in want if c in src_cols]
    absent = [c for c in want if c not in src_cols]

    target = load_parquet(args.target)
    if UID_COL not in target.columns:
        raise SystemExit(f"{RED}target has no {UID_COL!r} column{RESET}")
    source = pd.read_parquet(args.source, columns=[UID_COL] + present)
    if source[UID_COL].duplicated().any():
        raise SystemExit(f"{RED}source {UID_COL} is not unique{RESET}")

    # A column present in the source but entirely null (the uncalibrated
    # fastdetect_prob_* pairs) has nothing to carry — the scoring stage leaves
    # it null too, so skip it rather than create an all-null column.
    carry = [c for c in present if has_values(source[c])]
    empty = [c for c in present if c not in carry]

    print(f"target  {args.target}")
    print(f"source  {args.source}")
    print(f"suite   {len(want)} output column(s); "
          f"{len(carry)} to carry, {len(empty)} null in source, {len(absent)} absent")
    if absent:
        print(f"{YELLOW}  computed on every row by the scoring stages: "
              f"{absent}{RESET}")
    if empty:
        print(f"{DIM}  null in source, skipped: {empty}{RESET}")

    src_uids = set(source[UID_COL])
    matched = target[UID_COL].isin(src_uids).sum()
    print(f"\n{matched}/{len(target)} target row(s) match a source row "
          f"(the rest are scored fresh)")

    lookup = {c: dict(zip(source[UID_COL], source[c])) for c in carry}
    del source
    copied, skipped = [], []
    for c in carry:
        if c in target.columns and has_values(target[c]) and not args.overwrite:
            skipped.append(c)
            continue
        target[c] = [lookup[c].get(u) for u in target[UID_COL]]
        copied.append(c)

    dest = args.out or args.target
    save_parquet(target, dest)

    print(f"\n{GREEN}carried {len(copied)} column(s){RESET} → {dest}")
    if skipped:
        print(f"{DIM}  left alone (already had values; --overwrite to replace): "
              f"{skipped}{RESET}")
    # sanity: a carried column should be filled for exactly `matched` rows
    for c in copied[:1] + [c for c in copied if c.startswith("llm_judge_score_")][:1]:
        filled = int(sum(not _is_null(v) for v in target[c]))
        flag = "" if filled == matched else f"  {YELLOW}(expected {matched}){RESET}"
        print(f"{DIM}  {c}: filled for {filled} row(s){flag}{RESET}")
    print(f"\nnext: run_holdout_pipeline.py --from embed  (scores only the "
          f"{len(target) - matched} unmatched rows + the {len(absent)} new column(s))")


if __name__ == "__main__":
    main()
