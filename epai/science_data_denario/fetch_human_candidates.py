r"""Stage 1 — a candidate pool of real human abstracts for the Denario control.

One OpenAlex query set per Denario domain (cfg/science_data_denario/queries.yaml),
fetched through the SAME admission rules as the science_data fetch
(`epai.science_data.rules`): cleaned abstract, English heuristic, citation-count
and FWCI present, and — the load-bearing one — published no later than
``--year-max`` (2020 by default). An abstract written after ChatGPT may itself be
LLM-polished, and one of those in the "human" class is worse than a smaller pool.

This is only the POOL. build_denario_hc.py picks exactly one candidate per
Denario paper, matched on topic and length, so this stage overfetches
(``--pool-factor`` per domain) and does no matching of its own.

Deduplication spans all domains (one shared ``SeenIndex``), so a paper that
matches two domains' queries is claimed once — the same rule the science fetch
uses, for the same reason (a row under two domain labels leaks across folds).

Output: data/science_denario/human_candidates.parquet — every kept candidate,
with its Denario domain slug, cleaned abstract, word count and the OpenAlex work
id. Rewritten after each domain, so a crash costs only the domain in flight.

Run:
    python -m epai.science_data_denario.fetch_human_candidates --yes
    python -m epai.science_data_denario.fetch_human_candidates \
        --pool-factor 40 --year-max 2020 --yes
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from epai.science_data.rules import (
    OA_SELECT,
    FilterConfig,
    SeenIndex,
    check_record,
    normalise_abstract,
    normalise_record_oa,
    normalise_title,
)
from epai.science_data_denario.exclude import Exclusion, load_exclusion
from epai.utils.oa_utils import _session, check_service, iter_search
from epai.utils.parquet_utils import coerce_dtypes, save_parquet

load_dotenv()

YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_QUERIES = BASE_DIR / "cfg" / "science_data_denario" / "queries.yaml"
DEFAULT_DENARIO = BASE_DIR / "data" / "parquet" / "denario_detect.parquet"
DEFAULT_OUT = BASE_DIR / "data" / "science_denario" / "human_candidates.parquet"

OUT_COLUMNS = [
    "oa_id", "doi", "title", "abstract", "abstract_raw", "cleaning_fixes",
    "year", "venue", "citation_count", "fwci", "fields_of_study",
    "domain", "query", "n_words",
]


def denario_domain_counts(path: Path) -> dict[str, int]:
    """How many Denario papers each domain slug carries — the per-domain target."""
    return (
        pd.read_parquet(path, columns=["domain"])["domain"]
        .value_counts()
        .to_dict()
    )


def make_session(assume_yes: bool):
    """OpenAlex session from the environment, with the same fallbacks as the
    science fetch: a rejected key is dropped, an unreachable API is confirmed."""
    mailto = os.environ.get("OPENALEX_MAILTO")
    api_key = os.environ.get("OPENALEX_API_KEY")
    session = _session(mailto, api_key)
    status, message = check_service(session)
    print(f"OpenAlex: {status} — {message}")
    if status == "auth_failed":
        print(f"{YELLOW}  dropping the rejected key, continuing on the free tier{RESET}")
        return _session(mailto, None)
    if status != "ok":
        ok = assume_yes or (
            sys.stdin.isatty()
            and input("Continue anyway? [y/N]: ").strip().lower() in ("y", "yes")
        )
        if not ok:
            sys.exit(1)
    return session


def save_candidates(records: list[dict], out: Path) -> pd.DataFrame:
    out.parent.mkdir(parents=True, exist_ok=True)
    df = coerce_dtypes(
        pd.DataFrame.from_records(records, columns=OUT_COLUMNS),
        {"fwci": "float64", "year": "float64", "citation_count": "float64"},
    )
    save_parquet(df, out)
    return df


def fetch_domain(
    slug: str,
    queries: list[str],
    want: int,
    session,
    cfg: FilterConfig,
    seen: SeenIndex,
    excl: Exclusion,
    records: list[dict],
    strip_labels: bool,
    delay: float,
    excluded: list[int],
) -> int:
    """Round-robin the domain's queries until ``want`` admissible candidates are
    kept. Returns the count kept for this domain. ``excluded[0]`` is incremented
    for every candidate dropped for being in a training corpus."""
    filters = {
        "from_publication_date": f"{cfg.year_min}-01-01",
        "to_publication_date": f"{cfg.year_max}-12-31",
        "has_abstract": "true",
        "type": "article",
        "is_retracted": "false",
        "language": "en",
    }
    live = {q: iter_search(q, session, OA_SELECT, filters) for q in queries}
    kept = 0
    bar = tqdm(total=want, desc=slug, unit="cand", leave=False)
    while live and kept < want:
        for q in list(live):
            try:
                work = next(live[q])
            except StopIteration:
                del live[q]
                continue
            row = normalise_record_oa(work, slug, q, strip_labels=strip_labels)
            norm_title = normalise_title(row["title"])
            norm_abs = normalise_abstract(row["abstract"])
            if seen.lookup(row["paper_id"], norm_title, norm_abs) is not None:
                continue
            if check_record(row, cfg) is not None:
                continue
            if excl.hit(row["paper_id"], row["doi"], row["title"]) is not None:
                # In a training corpus — a detector would have seen it. Claim it
                # so a later query cannot re-offer the same paper.
                seen.claim(row["paper_id"], norm_title, -1, norm_abs)
                excluded[0] += 1
                continue
            seen.claim(row["paper_id"], norm_title, len(records), norm_abs)
            records.append({
                "oa_id": row["paper_id"],
                "doi": row["doi"],
                "title": row["title"],
                "abstract": row["abstract"],
                "abstract_raw": row["abstract_raw"],
                "cleaning_fixes": row["cleaning_fixes"],
                "year": row["year"],
                "venue": row["venue"],
                "citation_count": row["citation_count"],
                "fwci": row["fwci"],
                "fields_of_study": row["fields_of_study"],
                "domain": slug,
                "query": q,
                "n_words": len(row["abstract"].split()),
            })
            kept += 1
            bar.update(1)
            if kept >= want:
                break
        time.sleep(delay)
    bar.close()
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--denario", type=Path, default=DEFAULT_DENARIO,
                    help="denario_detect.parquet — read only for its per-domain counts")
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pool-factor", type=int, default=40,
                    help="candidates to fetch per Denario paper in a domain "
                         "(default 40) — the match then keeps 1 in pool-factor")
    ap.add_argument("--min-pool", type=int, default=60,
                    help="floor on the per-domain pool, for the small domains")
    ap.add_argument("--year-min", type=int, default=2005)
    ap.add_argument("--year-max", type=int, default=2020,
                    help="latest publication year — keep it pre-ChatGPT")
    ap.add_argument("--min-words", type=int, default=120,
                    help="candidate abstract length window, lower bound. The "
                         "Denario abstracts run ~110-350 words; the per-row match "
                         "tightens this, this only keeps the pool in range.")
    ap.add_argument("--max-words", type=int, default=380)
    ap.add_argument("--strip-structured-labels", action="store_true",
                    help="remove 'Background:'/'Methods:' section labels from "
                         "candidate abstracts (both classes end up without them). "
                         "OFF by default to match science_v3; turn on if the "
                         "build step warns about labels in the medicine / "
                         "biomedical_informatics rows.")
    ap.add_argument("--exclude-parquet", type=Path, action="append", default=None,
                    help="training corpus a candidate must NOT appear in "
                         "(repeatable; default: science_v3). Matched on OpenAlex "
                         "work id, DOI and normalised title.")
    ap.add_argument("--no-exclude-check", action="store_true",
                    help="skip the training-overlap check entirely — only if no "
                         "training corpus exists yet")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    queries = yaml.safe_load(args.queries.read_text())["queries_domains"]
    targets = denario_domain_counts(args.denario)
    cfg = FilterConfig(
        year_min=args.year_min, year_max=args.year_max,
        min_words=args.min_words, max_words=args.max_words,
    )
    excl = Exclusion() if args.no_exclude_check else load_exclusion(args.exclude_parquet)
    print(f"training-overlap guard: {len(excl.work_ids)} work ids, "
          f"{len(excl.dois)} DOIs, {len(excl.titles)} titles"
          + ("  (DISABLED)" if args.no_exclude_check else ""))
    session = make_session(args.yes)

    seen = SeenIndex()
    records: list[dict] = []
    excluded = [0]
    for slug, n_need in sorted(targets.items(), key=lambda kv: -kv[1]):
        qs = queries.get(slug)
        if not qs:
            print(f"{YELLOW}  {slug}: no queries in {args.queries.name} — "
                  f"{n_need} Denario paper(s) will be unmatched{RESET}")
            continue
        want = max(args.min_pool, n_need * args.pool_factor)
        kept = fetch_domain(slug, qs, want, session, cfg, seen, excl, records,
                            args.strip_structured_labels, args.delay, excluded)
        print(f"  {slug:<26} {kept:>4} candidates  (need {n_need})")
        save_candidates(records, args.out)

    df = save_candidates(records, args.out)
    print(f"\n{GREEN}{len(df)} candidates → {args.out}{RESET}  "
          f"({excluded[0]} dropped for training overlap)")

    got = df["domain"].value_counts().to_dict()
    short = [(s, got.get(s, 0), n) for s, n in targets.items() if got.get(s, 0) < n]
    if short:
        print(f"\n{YELLOW}SHORTFALL — the match will not fill these domains:{RESET}")
        for s, g, n in short:
            print(f"  {s}: {g} candidates < {n} needed — broaden its queries "
                  f"or raise --pool-factor")
        sys.exit(1)


if __name__ == "__main__":
    main()
