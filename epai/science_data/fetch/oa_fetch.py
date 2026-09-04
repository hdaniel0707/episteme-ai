"""Stage 1 — fetch scientific abstracts from OpenAlex.

Stops on the *post-filter* count: each incoming record is run through the same
row-local admission rules the filter stage uses (``epai.science_data.rules``),
and a domain finishes the moment it has kept ``target_per_domain × overfetch``
admissible records. The target is set after filtering, as intended — no
overshoot, and no second scrape to make up a shortfall.

Two mechanisms hold that guarantee up:

  - **Round-robin over a domain's queries.** Records are pulled one at a time
    from each query in turn rather than giving each query a fixed quota. If one
    query runs dry early, the others absorb its share automatically.
  - **A global seen-index.** Deduplication spans all domains, because the same
    paper appearing under two domain labels would leak across LODO/LOGO folds
    downstream. First domain wins (config order, so it is deterministic) and the
    loser is recorded on the winning row's ``also_matched_domains``.

Every record the API returned is written out, including the rejected ones, each
carrying ``kept`` and ``drop_reason``.

**Per record**: the fields the rules read — ``paper_id``, ``title``,
``abstract``, ``year``, ``citation_count``, ``fwci`` and ``language`` — plus
``doi``, ``author_ids``, ``author_count``, ``venue`` and ``fields_of_study``
as metadata. ``author_ids`` is a genuine list column of OpenAlex author ids,
not names: a name is not an identity, whereas ``A5011953697`` is stable across
a person's papers and resolves at ``/authors/A5011953697``.

``fwci`` and ``language`` are *admission rules*, not just columns: a work
OpenAlex cannot place in a field has no FWCI to normalise against and is
dropped, as is one whose language label is not English. Both are still stored,
so the drop report can distinguish a ``wrong_language`` rejection from a
``not_english`` one (the latter is the stopword heuristic disagreeing with
OpenAlex's own label).

**Why OpenAlex.** Its rate limit is an actual limit rather than load shedding,
and a key is optional rather than load-bearing: an ``OPENALEX_API_KEY``
(Premium) raises the limit and is used automatically when set, but the free
tier serves the same works with the same fields, so a lapsed or missing key
degrades the run instead of ending it. ``OPENALEX_MAILTO``, a plain contact
address, selects the faster "polite pool" and is not a secret.

One quirk of the source worth knowing: abstracts are not served as prose but as
an inverted index, so ``abstract_raw`` is a reconstruction and the original
inter-word whitespace is lost. Irrelevant after cleaning, which collapses
whitespace anyway, but it means ``abstract_raw`` is not a verbatim publisher
string.

Output: data/science/<dataset_id>_raw.parquet — one file, every domain, with
``domain`` as a column. It is rewritten after each domain completes, so a crash
in domain 9 does not cost domains 1-8 and there are no per-domain pieces to
reassemble afterwards.

Run (both configs default to the examples in cfg/science_data/):
    uv run python epai/science_data/fetch/oa_fetch.py

    uv run python epai/science_data/fetch/oa_fetch.py \
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
    strip_structured_labels_enabled,
)
from epai.science_data.rules import (
    OA_SELECT,
    OPTIONAL_NUMERIC_COLUMNS,
    SeenIndex,
    check_record,
    normalise_abstract,
    normalise_record_oa,
    normalise_title,
)
from epai.utils.env_utils import doublecheck_env, summarize_value
from epai.utils.parquet_utils import coerce_dtypes, save_parquet
from epai.utils.oa_utils import (
    OARateLimitError,
    OARequestError,
    _session,
    check_service,
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
    "doi",
    "title",
    "abstract",
    "abstract_raw",
    "cleaning_fixes",
    "author_ids",
    "author_count",
    "year",
    "venue",
    "citation_count",
    "fwci",
    "language",
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
            except OARequestError:
                # Not a per-query problem. The filter/select expressions and
                # the API key are shared by every query, so a rejection of
                # either fails all of them identically. Abort rather than grind
                # through them all and report five empty domains. Catches
                # OAAuthError too — it is a subclass for exactly this reason.
                raise
            except OARateLimitError as e:
                tqdm.write(f"{YELLOW}  [{query}] rate limited: {e}{RESET}")
                rate_limited.add(query)
                del live[query]
            except Exception as e:  # network hiccup on one query only
                tqdm.write(f"{YELLOW}  [{query}] iterator failed: {e}{RESET}")
                del live[query]


def save_raw(records: list[dict], out: Path, quiet: bool = False) -> pd.DataFrame:
    """Write every record fetched so far to the one raw parquet.

    Called after each domain as a checkpoint and once more at the end, so the
    file on disk is always a complete, readable corpus of everything fetched up
    to that point — never a set of pieces that have to be found and stitched
    back together.

    ``columns=RAW_COLUMNS`` pins the column order and, more importantly, keeps
    it stable when the first domain happens to contain no records at all.
    """
    df = coerce_dtypes(
        pd.DataFrame.from_records(records, columns=RAW_COLUMNS),
        OPTIONAL_NUMERIC_COLUMNS,
    )
    if quiet:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
    else:
        save_parquet(df, out)
    return df


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


def resolve_session(assume_yes: bool, no_mailto: bool = False, no_api_key: bool = False):
    """Build the API session, checking the service is reachable before real work.

    Two identifiers, neither required, and they are not a pair:

      - ``OPENALEX_API_KEY`` — an OpenAlex Premium key. A real secret. Raises
        the rate limit. Optional: the free tier serves the same works with the
        same fields.
      - ``OPENALEX_MAILTO`` — a contact address that selects OpenAlex's "polite
        pool", a separate and more consistently served queue. Not a credential;
        it travels in plain text in the query string. Also optional, and
        **redundant once a key is set** — the key is the stronger identifier
        and brings its own limit, so this only matters to a keyless run.

    Both are sent when present, because sending both costs nothing. A key that
    the API rejects is worse than no key at all — it makes every request fail —
    so when the probe reports ``auth_failed`` the offer is to continue *without*
    it.

    ``no_mailto`` / ``no_api_key`` ignore whatever is in the environment. They
    need no edit to ``.env`` (a secrets file worth leaving alone) and nothing
    to undo afterwards.
    """
    mailto = None if no_mailto else os.environ.get("OPENALEX_MAILTO")
    api_key = None if no_api_key else os.environ.get("OPENALEX_API_KEY")

    if no_api_key:
        print(
            f"{YELLOW}--no-api-key given: ignoring OPENALEX_API_KEY and running "
            f"on the free tier.{RESET}"
        )
    elif api_key:
        # Masked, never echoed in full — the .env summary above prints it the
        # same way. "Sending", not "using Premium": whether OpenAlex recognises
        # the key is not visible in a successful response, and the free tier
        # returns identical data, so a run that silently ignored the key would
        # look exactly like one that honoured it.
        print(f"Sending OPENALEX_API_KEY={summarize_value(api_key)}.")

    if no_mailto:
        print(
            f"{YELLOW}--no-mailto given: ignoring OPENALEX_MAILTO.{RESET}"
        )
    elif not mailto and not api_key:
        # Only worth stopping for when there is no key either. A key is the
        # stronger identifier of the two and carries its own rate limit, so
        # adding a contact address on top changes nothing; without one, the
        # run falls back to the shared anonymous pool, which is slow enough to
        # be worth confirming before spending an hour on it.
        print(
            f"\n{YELLOW}Neither OPENALEX_API_KEY nor OPENALEX_MAILTO is set.{RESET}\n"
            "  Requests go to the common pool, shared with every other "
            "anonymous caller and throttled accordingly.\n"
            "  OPENALEX_MAILTO=you@example.com is the cheap fix: it is not a "
            "secret, not an account, and not required — OpenAlex simply serves "
            "identified callers from a faster queue.\n"
            "  See https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication"
        )
        if not _confirm("Continue anonymously?", assume_yes):
            print("Aborted.")
            sys.exit(1)

    session = _session(mailto, api_key)
    status, message = check_service(session)

    if status == "ok":
        print(f"{GREEN}OpenAlex: {message}.{RESET}")
        return session

    if status == "auth_failed":
        print(
            f"\n{RED}OpenAlex rejected the API key ({message}).{RESET}\n"
            "  OPENALEX_API_KEY is set but invalid, expired, or the Premium "
            "subscription has lapsed. Every request will fail while it is "
            "being sent.\n"
            "  OpenAlex serves the same works and the same fields without a "
            "key — dropping it costs rate limit and data freshness, nothing "
            "else."
        )
        if not _confirm("Continue WITHOUT the key?", assume_yes):
            print("Aborted — fix the key and re-run.")
            sys.exit(1)
        return _session(mailto, None)

    if status == "bad_request":
        # Nothing the user can answer here: the probe sends a fixed, minimal
        # query, so a refusal means the API contract moved under us.
        print(
            f"\n{RED}OpenAlex refused the probe request ({message}).{RESET}\n"
            "  The probe uses a fixed minimal query, so this is not a config "
            "problem — most likely the API's filter syntax has changed.\n"
            "  Check https://docs.openalex.org/api-entities/works/filter-works",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"\n{YELLOW}Could not reach OpenAlex: {message}.{RESET}\n"
        "  This says nothing about the queries — it is a network problem, an "
        "outage, or a spent daily quota (100,000 requests, resets at UTC\n"
        "  midnight). The fetcher retries 429s and 5xx errors with backoff as "
        "it goes, so continuing is reasonable if you expect it to clear."
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
    # Same key the generate stage reads for the machine abstracts, so both
    # classes are cleaned the same way or neither is.
    strip_labels = strip_structured_labels_enabled(cfg)
    quota = math.ceil(fetch_cfg["target_per_domain"] * fetch_cfg.get("overfetch", 1.2))

    search = iter_search_bulk if fetch_cfg.get("search_mode") == "bulk" else iter_search

    # Server-side filters save bandwidth only; check_record re-tests every one
    # of these, so correctness never depends on the API honouring them. They
    # are what keeps the fetch yield high: `has_abstract` alone removes the
    # bulk of what would otherwise be fetched and immediately dropped.
    filters = {
        "from_publication_date": f"{rules_cfg.year_min}-01-01",
        "to_publication_date": f"{rules_cfg.year_max}-12-31",
        "has_abstract": "true",
        "type": "article",
        "is_retracted": "false",
    }
    if rules_cfg.require_language:
        filters["language"] = rules_cfg.require_language
    if rules_cfg.require_fwci:
        # OpenAlex has no "is not null" filter — `fwci:!null` is a 500 and
        # there is no `has_fwci` field — but FWCI is never negative, so a
        # ">-1" range excludes exactly the nulls. Deliberately not ">0":
        # a zero FWCI is the common case (most works have one), and dropping
        # those would bias the corpus toward cited work in exactly the way
        # `require_citation_count` is careful not to.
        filters["fwci"] = ">-1"

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

    # A backoff can run tens of seconds without a record arriving. Surfacing
    # the retry count in the bar's postfix is what distinguishes "waiting out a
    # 429" from "hung", which otherwise look identical from the outside.
    def on_status(msg: str) -> None:
        bar.set_postfix_str(msg, refresh=True)

    iterators = {
        q: search(q, session, OA_SELECT, filters, on_status) for q in queries
    }

    for query, work in _round_robin(iterators, rate_limited):
        seen_total += 1
        row = normalise_record_oa(work, domain, query, strip_labels=strip_labels)

        prior = seen.lookup(
            row["paper_id"],
            normalise_title(row["title"]),
            normalise_abstract(row["abstract"]),
        )
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
            seen.claim(
                row["paper_id"],
                normalise_title(row["title"]),
                len(records) - 1,
                normalise_abstract(row["abstract"]),
            )
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
        help="answer yes to the startup prompts (for non-interactive runs)",
    )
    parser.add_argument(
        "--no-mailto",
        action="store_true",
        help="ignore OPENALEX_MAILTO and use the common pool instead of the "
        "polite one (slower, throttled harder)",
    )
    parser.add_argument(
        "--no-api-key",
        action="store_true",
        help="ignore OPENALEX_API_KEY and fetch on the free tier — useful for "
        "checking whether a failure is key-related",
    )
    args = parser.parse_args()

    # Echo the .env settings this pipeline depends on, masked. Absolute path so
    # the check works whatever directory the script is invoked from.
    doublecheck_env(str(BASE_DIR / ".env"))

    cfg = load_pipeline_config(args.cfg)
    queries_domains = load_queries(args.queries)

    # The raw corpus is keyed only by `dataset_id`, so a re-run silently
    # replaces the previous one. Ask first: a full fetch is an hour you do not
    # get back, and a corpus built under different queries or thresholds is not
    # recoverable from the new one.
    out = raw_path(cfg)
    if out.exists():
        print(
            f"\n{YELLOW}{out.relative_to(BASE_DIR)} already exists and will be "
            f"overwritten.{RESET}\n"
            "  Give this run its own dataset_id in the config to keep both."
        )
        if not _confirm("Overwrite it?", args.yes):
            print("Aborted.")
            sys.exit(1)

    session = resolve_session(args.yes, args.no_mailto, args.no_api_key)

    target = cfg["fetch"]["target_per_domain"]
    quota = math.ceil(target * cfg["fetch"].get("overfetch", 1.2))

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

            try:
                kept, seen_total, rate_limited = fetch_domain(
                    domain, queries, session, cfg, seen, records
                )
            except OARequestError as e:
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

            # Checkpoint into the one output file rather than a per-domain
            # shard: a crash in domain 9 must not cost domains 1-8, but that
            # does not require a second set of files to reassemble afterwards.
            # Rewriting the whole parquet each time is affordable because it
            # happens once per domain, not once per record.
            save_raw(records, out, quiet=True)

            time.sleep(cfg["fetch"].get("query_delay", 1.5))
    except KeyboardInterrupt:
        # Ctrl-C during a long backoff is expected, not a crash. Keep whatever
        # was fetched instead of discarding it behind a traceback.
        domain_bar.close()
        interrupted = True
        print(f"\n{YELLOW}Interrupted — saving the {len(records)} record(s) fetched so far.{RESET}")

    df = save_raw(records, out)

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
                    "  Set OPENALEX_MAILTO to use the polite pool, or wait for "
                    "the daily quota to reset at UTC midnight.",
                    file=sys.stderr,
                )
            else:
                print(
                    "No rate limiting was hit, so these domains genuinely ran out "
                    "of matching papers. Add or broaden this domain's queries, "
                    "lower fetch.target_per_domain, or set fetch.search_mode: bulk "
                    "to lift the 10,000-hit-per-query cap.",
                    file=sys.stderr,
                )
            print(
                "The raw parquet was still written, so re-running only refetches.",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
