r"""Guard: confirm the ai_scientist corpus shares NO paper -- human or AI --
with the existing Denario corpus (``manugram_den.parquet``), before either the
scoring pipeline or the merge runs.

Two entirely different worries, both checked:

  HUMAN side   ``ai_scientist_hc_raw.parquet``'s 20 matched human abstracts
               (OpenAlex, domain 'computer_science') vs. ``manugram_den``'s 80
               human rows (OpenAlex, Denario's 9 domains -- none of which is
               'computer_science', so this should come back clean by
               construction; checked rather than assumed). Same 3-key method
               ``exclude.py`` already uses: OpenAlex work id, DOI, normalised
               title -- plus a normalised-abstract check as a fourth,
               independent net (catches a re-titled reprint neither id/doi/
               title would).

  AI side      ``ai_scientist_detect.parquet``'s 20 machine abstracts (Sakana /
               CycleResearcher / Zochi) vs. ``manugram_den``'s 80 ``denario``
               rows. No id/DOI on either side (neither is a real publication),
               so this is normalised-title + normalised-abstract only. There
               is no plausible causal path for these to actually collide --
               entirely different generating systems -- so this is a sanity
               net, not a real expectation of finding anything.

Exits non-zero (and prints exactly which pair collided) if either check finds
a hit. A clean run prints nothing to worry about and exits 0 -- safe to chain
before the pipeline / the merge in a shell ``&&``.

Run:
    uv run python -m epai.science_data_denario.check_ai_scientist_overlap
    uv run python -m epai.science_data_denario.check_ai_scientist_overlap \
        --denario data/parquet/manugram_den.parquet \
        --ai-scientist-detect data/parquet/ai_scientist_detect.parquet \
        --ai-scientist-hc-raw data/parquet/ai_scientist_hc_raw.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from epai.science_data.rules import normalise_abstract, normalise_title

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"

BASE_DIR = Path(__file__).resolve().parents[2]
PARQUET_DIR = BASE_DIR / "data" / "parquet"

DEFAULT_DENARIO = PARQUET_DIR / "manugram_den.parquet"
DEFAULT_AI_DETECT = PARQUET_DIR / "ai_scientist_detect.parquet"
DEFAULT_AI_HC_RAW = PARQUET_DIR / "ai_scientist_hc_raw.parquet"


def _norm_oa_id(value: object) -> str:
    import re
    m = re.search(r"W\d+", str(value or ""), re.IGNORECASE)
    return m.group(0).upper() if m else ""


def _titles(df: pd.DataFrame, col: str = "title") -> set[str]:
    return {normalise_title(v) for v in df.get(col, pd.Series(dtype="object")).dropna()} - {""}


def _abstracts(df: pd.DataFrame, col: str = "text") -> set[str]:
    return {normalise_abstract(v) for v in df.get(col, pd.Series(dtype="object")).dropna()} - {""}


def _oa_ids(df: pd.DataFrame, col: str = "oa_id") -> set[str]:
    if col not in df.columns:
        return set()
    return {_norm_oa_id(v) for v in df[col].dropna()} - {""}


def _report_set_overlap(
    label: str, a_keys: set[str], b_keys: set[str],
    a_lookup: dict[str, str], b_lookup: dict[str, str],
) -> list[str]:
    """``a_lookup``/``b_lookup`` map key -> a human-readable label (title), for
    printing which actual papers collided rather than just the folded key."""
    hits = a_keys & b_keys
    if not hits:
        return []
    lines = [f"{RED}  {label}: {len(hits)} collision(s){RESET}"]
    for h in sorted(hits)[:20]:
        lines.append(f"    {a_lookup.get(h, h)!r}  <->  {b_lookup.get(h, h)!r}")
    if len(hits) > 20:
        lines.append(f"    ... and {len(hits) - 20} more")
    return lines


def check(denario: pd.DataFrame, ai_detect: pd.DataFrame, ai_hc_raw: pd.DataFrame) -> int:
    problems: list[str] = []

    den_human = denario[denario["generated_by"] == "human"]
    den_machine = denario[denario["generated_by"] != "human"]
    ai_human = ai_hc_raw[ai_hc_raw["generated_by"] == "human"]
    ai_machine = ai_hc_raw[ai_hc_raw["generated_by"] != "human"]

    print(f"denario:               {len(den_human)} human, {len(den_machine)} AI")
    print(f"ai_scientist (raw hc): {len(ai_human)} human, {len(ai_machine)} AI")
    print(f"ai_scientist_detect:   {len(ai_detect)} AI rows (pre-match; cross-checked too)")
    print()

    # --- HUMAN side ---------------------------------------------------------
    den_oa, ai_oa = _oa_ids(den_human), _oa_ids(ai_human)
    oa_hits = den_oa & ai_oa
    if oa_hits:
        problems.append(f"{RED}  oa_id: {len(oa_hits)} shared OpenAlex work id(s): "
                         f"{sorted(oa_hits)[:20]}{RESET}")

    den_title_map = {normalise_title(t): t for t in den_human.get("title", pd.Series(dtype="object")).dropna()}
    ai_title_map = {normalise_title(t): t for t in ai_human.get("title", pd.Series(dtype="object")).dropna()}
    problems += _report_set_overlap("human title", set(den_title_map), set(ai_title_map),
                                     den_title_map, ai_title_map)

    den_abs_map = {normalise_abstract(t): t[:70] for t in den_human.get("text", pd.Series(dtype="object")).dropna()}
    ai_abs_map = {normalise_abstract(t): t[:70] for t in ai_human.get("text", pd.Series(dtype="object")).dropna()}
    problems += _report_set_overlap("human abstract", set(den_abs_map), set(ai_abs_map),
                                     den_abs_map, ai_abs_map)

    # --- AI side (check both the matched-hc machine rows AND the pre-match
    #     detect file, in case a row got dropped/renamed between the two) ----
    for ai_frame, tag in ((ai_machine, "hc-raw"), (ai_detect, "detect")):
        den_title_map = {normalise_title(t): t for t in den_machine.get("title", pd.Series(dtype="object")).dropna()}
        ai_title_map = {normalise_title(t): t for t in ai_frame.get("title", pd.Series(dtype="object")).dropna()}
        problems += _report_set_overlap(f"AI title ({tag})", set(den_title_map), set(ai_title_map),
                                         den_title_map, ai_title_map)

        den_abs_map = {normalise_abstract(t): t[:70] for t in den_machine.get("text", pd.Series(dtype="object")).dropna()}
        ai_abs_map = {normalise_abstract(t): t[:70] for t in ai_frame.get("text", pd.Series(dtype="object")).dropna()}
        problems += _report_set_overlap(f"AI abstract ({tag})", set(den_abs_map), set(ai_abs_map),
                                         den_abs_map, ai_abs_map)

    if problems:
        print(f"{RED}OVERLAP FOUND -- do not run the pipeline / merge yet:{RESET}")
        for p in problems:
            print(p)
        return 1

    print(f"{GREEN}No overlap: 0 shared OpenAlex ids, 0 shared titles, "
          f"0 shared abstracts, human or AI, either direction.{RESET}")
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--denario", type=Path, default=DEFAULT_DENARIO)
    ap.add_argument("--ai-scientist-detect", type=Path, default=DEFAULT_AI_DETECT)
    ap.add_argument("--ai-scientist-hc-raw", type=Path, default=DEFAULT_AI_HC_RAW)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    missing = [p for p in (args.denario, args.ai_scientist_detect, args.ai_scientist_hc_raw)
               if not p.is_file()]
    if missing:
        print(f"{RED}not found: {[str(p) for p in missing]}{RESET}", file=sys.stderr)
        return 2

    import pyarrow.parquet as pq

    den_cols = set(pq.read_schema(args.denario).names)
    want = [c for c in ("generated_by", "title", "text", "oa_id") if c in den_cols]
    denario = pd.read_parquet(args.denario, columns=want)
    ai_detect = pd.read_parquet(args.ai_scientist_detect)
    ai_hc_raw = pd.read_parquet(args.ai_scientist_hc_raw)

    return check(denario, ai_detect, ai_hc_raw)


if __name__ == "__main__":
    sys.exit(main())
