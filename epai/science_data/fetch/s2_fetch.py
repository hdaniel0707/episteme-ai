"""Stage 1 — fetch scientific abstracts from Semantic Scholar.

The defining feature of this stage is that it stops on the *post-filter* count.
Each incoming record is run through the same row-local admission rules the
filter stage uses (``epai.science_data.rules``), and a domain finishes the
moment it has kept ``target_per_domain × overfetch`` admissible records. So the
target is set after filtering, as intended: no overshoot, and no second scrape
to make up a shortfall.

Two mechanisms hold that guarantee up:

  - **Round-robin over a domain's queries.** Records are pulled one at a time
    from each query in turn rather than giving each query a fixed quota. If one
    query runs dry early, the others absorb its share automatically; fixed
    quotas cannot do that and would leave the domain short.
  - **A global seen-index.** Deduplication spans all domains, not just the
    current one, because the same paper appearing under two domain labels would
    leak across LODO/LOGO folds downstream. First domain wins (config order, so
    it is deterministic) and the loser is recorded on the winning row's
    ``also_matched_domains``.

Every record the API returned is written out, including the rejected ones, each
carrying ``kept`` and ``drop_reason``. Rejected rows cost nothing to store and
mean that tightening a filter later is a re-run of stage 2 over local data
rather than another few thousand API calls.

A domain that cannot reach its target fails the run loudly. Finding that out
here is far cheaper than discovering it three stages downstream.

Output: data/science/<dataset_id>_raw.parquet
        data/science/shards/<dataset_id>_raw_<domain>.parquet  (per-domain, crash-safe)

Run (both configs default to the examples in cfg/science_data/):
    python -m epai.science_data.fetch.s2_fetch

    # unauthenticated smoke test, e.g. while waiting on an API key
    python -m epai.science_data.fetch.s2_fetch --no-api-key

    python -m epai.science_data.fetch.s2_fetch \
        --cfg cfg/science_data/science_cfg_example.yaml \
        --queries cfg/science_data/queries_example.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from epai.science_data.config import (
    BASE_DIR,
    DEFAULT_CFG_PATH,
    DEFAULT_QUERIES_PATH,
    filter_config,
    load_pipeline_config,
    load_queries,
    raw_path,
    _slug,
)
from epai.science_data.rules import (
    PAPER_FIELDS,
    SeenIndex,
    check_record,
    normalise_record,
    normalise_title,
)
from epai.utils.env_utils import doublecheck_env
from epai.utils.parquet_utils import save_parquet
from epai.utils.s2_utils import (
    S2AuthError,
    S2RateLimitError,
    _session,
    check_credentials,
    iter_search,
    iter_search_bulk,
)

load_dotenv()

YELLOW = "\033[93m"
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

RAW_COLUMNS = [
    "paper_id",
    "title",
    "abstract",
    "abstract_raw",
    "cleaning_fixes",
    "year",
    "venue",
    "citation_count",
    "influential_citation_count",
    "fwci",
    "fields_of_study",
    "domain",
    "query",
    "also_matched_domains",
    "kept",
    "drop_reason",
]


def _round_robin(iterators: dict[str, object], rate_limited: set[str]):
    """Yield ``(query, record)`` one record at a time, cycling over queries.

    Exhausted queries are dropped from the rotation, so the remaining ones
    automatically take over the shortfall. Queries dropped because of a rate
    limit are recorded in *rate_limited*, so the caller can tell a genuine
    shortage of papers from a throttled run.
    """
    live = dict(iterators)
    while live:
        for query in list(live):
            try:
                yield query, next(live[query])  # type: ignore[arg-type]
            except StopIteration:
                del live[query]
            except S2AuthError:
                # Not a per-query problem: every other query will fail
                # identically, so abort rather than grind through them all and
                # report five empty domains.
                raise
            except S2RateLimitError as e:
                tqdm.write(f"{YELLOW}  [{query}] rate limited: {e}{RESET}")
                rate_limited.add(query)
                del live[query]
            except Exception as e:  # network hiccup on one query only
                tqdm.write(f"{YELLOW}  [{query}] iterator failed: {e}{RESET}")
                del live[query]


def _confirm(question: str, assume_yes: bool) -> bool:
    """Ask the user a yes/no question; default is no.

    ``--yes`` skips the prompt. When stdin is not a terminal (cron, a pipe)
    there is nobody to answer, so the answer is no rather than a hang.
    """
    if assume_yes:
        print(f"{question} [--yes given, continuing]")
        return True
    if not sys.stdin.isatty():
        print(f"{question} — stdin is not a terminal; pass --yes to proceed.")
        return False
    return input(f"{question} [y/N]: ").strip().lower() in ("y", "yes")


def resolve_session(assume_yes: bool, no_api_key: bool = False):
    """Build the API session, checking the credentials before any real work.

    Returns the session to fetch with, or exits. A key that the API rejects is
    worse than no key at all — the header makes every request fail — so when
    the probe reports ``auth_failed`` the offer is to continue *without* it.

    ``no_api_key`` ignores whatever is in the environment and goes
    unauthenticated. Useful while waiting on a replacement key: it needs no
    edit to ``.env`` (a secrets file worth leaving alone) and nothing to undo
    once the new key arrives.
    """
    api_key = None if no_api_key else os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

    if no_api_key:
        print(
            f"\n{YELLOW}--no-api-key given: ignoring SEMANTIC_SCHOLAR_API_KEY and "
            f"running unauthenticated.{RESET}\n"
            "  Shared rate-limit pool, so expect 429s; the fetcher retries them "
            "with backoff. Fine for a small smoke test, slow for a full run."
        )
        return _session(None)

    if not api_key:
        print(
            f"\n{YELLOW}No SEMANTIC_SCHOLAR_API_KEY is set.{RESET}\n"
            "  Unauthenticated requests share one rate-limit pool with every "
            "other anonymous caller, so expect frequent 429s and a slow fetch.\n"
            "  Get a key at https://www.semanticscholar.org/product/api#api-key-form"
        )
        if not _confirm("Continue unauthenticated?", assume_yes):
            print("Aborted.")
            sys.exit(1)
        return _session(None)

    session = _session(api_key)
    status, message = check_credentials(session)

    if status == "ok":
        print(f"{GREEN}Semantic Scholar: {message}.{RESET}")
        return session

    if status == "auth_failed":
        print(
            f"\n{RED}Semantic Scholar rejected the API key ({message}).{RESET}\n"
            "  SEMANTIC_SCHOLAR_API_KEY is set but invalid, expired, or revoked. "
            "Every request will fail while it is being sent.\n"
            "  Request a new one at "
            "https://www.semanticscholar.org/product/api#api-key-form\n"
            "  Unauthenticated access still works, but shares a rate-limit pool "
            "with all anonymous callers and will be slow."
        )
        if not _confirm("Continue WITHOUT the key (unauthenticated)?", assume_yes):
            print("Aborted — fix the key and re-run.")
            sys.exit(1)
        return _session(None)

    print(
        f"\n{YELLOW}Could not verify credentials: {message}.{RESET}\n"
        "  This is NOT an invalid key — a rejected key returns 401/403 and is "
        "reported separately. Semantic Scholar throttles valid keys too, and\n"
        "  heavily at times, so this is most likely a transient rate limit. The "
        "fetcher retries 429s and 5xx errors with backoff as it goes, and\n"
        "  continuing is usually the right call; expect a slow run, and see "
        "--allow-shortfall if domains come up under quota."
    )
    if not _confirm("Continue anyway?", assume_yes):
        print("Aborted.")
        sys.exit(1)
    return session


def fetch_domain(
    domain: str,
    queries: list[str],
    session,
    cfg: dict,
    seen: SeenIndex,
    records: list[dict],
) -> tuple[int, int, set[str]]:
    """Fetch one domain until its quota is met; append rows to *records*.

    Returns ``(kept, seen_total, rate_limited_queries)``. Rows are appended to
    the shared *records* list so that ``also_matched_domains`` can be written
    back onto rows already claimed by an earlier domain.
    """
    fetch_cfg = cfg["fetch"]
    rules_cfg = filter_config(cfg)
    quota = math.ceil(fetch_cfg["target_per_domain"] * fetch_cfg.get("overfetch", 1.2))

    search = iter_search_bulk if fetch_cfg.get("search_mode") == "bulk" else iter_search
    # Server-side year filter saves bandwidth only; check_record re-tests it, so
    # correctness never depends on the API honouring this.
    extra_params = {"year": f"{rules_cfg.year_min}-{rules_cfg.year_max}"}

    kept = 0
    seen_total = 0
    rate_limited: set[str] = set()
    # position=1 keeps this under the domain bar; without an explicit position
    # the two can end up rendering onto the same terminal line.
    bar = tqdm(
        total=quota,
        desc=domain,
        unit="kept",
        leave=False,
        position=1,
        dynamic_ncols=True,
    )

    # Throttled stretches can run tens of seconds without a single record
    # arriving. Surfacing the retry count in the bar's postfix is what
    # distinguishes "waiting out a 429" from "hung", which otherwise look
    # identical from the outside.
    def on_status(msg: str) -> None:
        bar.set_postfix_str(msg, refresh=True)

    iterators = {
        q: search(q, session, PAPER_FIELDS, extra_params, on_status) for q in queries
    }

    for query, paper in _round_robin(iterators, rate_limited):
        seen_total += 1
        row = normalise_record(paper, domain, query)

        prior = seen.lookup(row["paper_id"], normalise_title(row["title"]))
        if prior is not None:
            # Already claimed — record the cross-domain match for auditing and
            # move on. Only worth noting when the domains actually differ.
            if records[prior]["domain"] != domain:
                existing = records[prior]["also_matched_domains"]
                parts = [p for p in existing.split(",") if p]
                if domain not in parts:
                    parts.append(domain)
                    records[prior]["also_matched_domains"] = ",".join(parts)
            row["kept"] = False
            row["drop_reason"] = "duplicate"
            records.append(row)
            continue

        reason = check_record(row, rules_cfg)
        row["kept"] = reason is None
        row["drop_reason"] = reason or ""
        records.append(row)

        if reason is None:
            # Claim only admissible records: a rejected paper should not block
            # a later, cleaner record of the same work from a different query.
            seen.claim(row["paper_id"], normalise_title(row["title"]), len(records) - 1)
            kept += 1
            bar.update(1)
            if kept >= quota:
                break

    bar.close()
    return kept, seen_total, rate_limited


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cfg",
        type=Path,
        default=DEFAULT_CFG_PATH,
        help=f"pipeline config YAML (default: {DEFAULT_CFG_PATH.relative_to(BASE_DIR)})",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help=f"queries_domains YAML (default: {DEFAULT_QUERIES_PATH.relative_to(BASE_DIR)})",
    )
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="warn instead of failing when a domain cannot reach its quota",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="answer yes to the credential prompts (for non-interactive runs)",
    )
    parser.add_argument(
        "--no-api-key",
        action="store_true",
        help="ignore SEMANTIC_SCHOLAR_API_KEY and fetch unauthenticated "
        "(slower, shared rate limit) — useful while waiting on a new key",
    )
    args = parser.parse_args()

    # Echo the .env settings this pipeline depends on, masked. Absolute path so
    # the check works whatever directory the script is invoked from.
    doublecheck_env(str(BASE_DIR / ".env"))

    cfg = load_pipeline_config(args.cfg)
    queries_domains = load_queries(args.queries)

    session = resolve_session(args.yes, args.no_api_key)

    target = cfg["fetch"]["target_per_domain"]
    quota = math.ceil(target * cfg["fetch"].get("overfetch", 1.2))
    shard_dir = BASE_DIR / "data" / "science" / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    seen = SeenIndex()
    shortfalls: list[tuple[str, int]] = []
    throttled: set[str] = set()
    interrupted = False

    domain_bar = tqdm(
        queries_domains.items(), desc="domains", position=0, dynamic_ncols=True
    )
    try:
        for domain, queries in domain_bar:
            domain_bar.set_description(f"domain: {domain}")
            start = len(records)

            try:
                kept, seen_total, rate_limited = fetch_domain(
                    domain, queries, session, cfg, seen, records
                )
            except S2AuthError as e:
                domain_bar.close()
                print(f"\n{RED}ERROR: {e}{RESET}", file=sys.stderr)
                sys.exit(1)

            throttled |= rate_limited
            yield_pct = 100.0 * kept / seen_total if seen_total else 0.0
            tqdm.write(
                f"  [{domain}] kept {kept}/{quota} from {seen_total} fetched "
                f"({yield_pct:.1f}% yield)"
            )
            if kept < quota:
                shortfalls.append((domain, kept))

            # Per-domain shard: a crash in domain 9 must not cost domains 1-8.
            shard = pd.DataFrame.from_records(records[start:])
            if not shard.empty:
                shard.to_parquet(shard_dir / f"{cfg['dataset_id']}_raw_{_slug(domain)}.parquet", index=False)

            time.sleep(cfg["fetch"].get("query_delay", 1.5))
    except KeyboardInterrupt:
        # Ctrl-C during a long backoff is expected, not a crash. Keep whatever
        # was fetched instead of discarding it behind a traceback.
        domain_bar.close()
        interrupted = True
        print(f"\n{YELLOW}Interrupted — saving the {len(records)} record(s) fetched so far.{RESET}")

    df = pd.DataFrame.from_records(records, columns=RAW_COLUMNS)
    out = raw_path(cfg)
    save_parquet(df, out)

    if interrupted:
        print(f"Partial raw corpus → {out}")
        print(
            "This run is incomplete, so do not filter from it. Re-run the fetch "
            "when the API is available."
        )
        sys.exit(130)

    n_kept = int(df["kept"].sum())
    print(f"\nFetched {len(df)} records, {n_kept} admissible, across {len(queries_domains)} domains")
    print(f"Raw corpus → {out}")

    if shortfalls:
        lines = "\n".join(
            f"  {d}: {k} kept, needed {quota} (target {target} × overfetch)"
            for d, k in shortfalls
        )
        msg = f"{len(shortfalls)} domain(s) did not reach quota:\n{lines}"
        if args.allow_shortfall:
            print(f"{YELLOW}WARNING: {msg}{RESET}")
        else:
            print(f"{RED}ERROR: {msg}{RESET}", file=sys.stderr)
            if throttled:
                # Rate limiting and a genuine shortage of papers look identical
                # in the counts, but the fixes are completely different — so say
                # which one happened rather than defaulting to query advice.
                print(
                    f"\n{len(throttled)} query/queries were cut short by rate "
                    "limiting, so this is a throttling problem, NOT a query "
                    "problem. Do not rewrite the queries.\n"
                    "  Use a valid SEMANTIC_SCHOLAR_API_KEY, or retry later when "
                    "the shared unauthenticated pool is less busy.",
                    file=sys.stderr,
                )
            else:
                print(
                    "No rate limiting was hit, so these domains genuinely ran out "
                    "of matching papers. Add or broaden this domain's queries, "
                    "lower fetch.target_per_domain, or set fetch.search_mode: bulk "
                    "to lift the 1000-hit-per-query cap.",
                    file=sys.stderr,
                )
            print(
                "The raw parquet was still written, so re-running only refetches.",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
