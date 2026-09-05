"""OpenAlex fetcher — the workhorse toward ~100k rows (plan §4.2).

Cursor-paged ``/works`` queries (200/page, polite pool via OPENALEX_MAILTO), filtered to
journal articles with an abstract; the abstract is reconstructed from
``abstract_inverted_index``. fwci + citation_normalized_percentile arrive for free and
map directly onto the existing ``fwci_percentile`` / ``normalized_percentile`` targets.
Author h-indexes are batch-filled afterwards (1 request per 100 author ids).
Resume-safe: the page cursor is persisted in the DB's ``fetch_state`` table.
"""

from __future__ import annotations

import gc
import sqlite3
import time
from typing import Iterator

import requests

from workflow.src.iwf import config
from workflow.src.iwf.data.identity import normalize_doi
from workflow.src.iwf.fetchers.base import BaseFetcher, QuotaExhausted, get_cursor, set_cursor

_QUOTA_RETRY_AFTER_S = 60

WORKS_URL = "https://api.openalex.org/works"
AUTHORS_URL = "https://api.openalex.org/authors"
PER_PAGE = 200
SELECT = ("id,doi,title,publication_year,primary_location,abstract_inverted_index,"
          "cited_by_count,fwci,citation_normalized_percentile,primary_topic,authorships")


def _auth_params() -> dict:
    """Auth/politeness query params (api_key when set, plus mailto)."""
    params: dict = {}
    key = config.api_key("OPENALEX_API_KEY")
    if key:
        params["api_key"] = key
    m = config.get_str("OPENALEX_MAILTO", "")
    if m:
        params["mailto"] = m
    return params


# Body words that mean a daily-credit limit (not a transient burst 429).
_DAILY_LIMIT_MARKERS = ("daily", "credit", "quota", "insufficient", "limit reached",
                        "rate limit", "exceeded")


def _looks_like_daily_limit(text: str) -> bool:
    """True when the response body reads like a daily-limit message."""
    t = (text or "").lower()
    return any(m in t for m in _DAILY_LIMIT_MARKERS)


def _get(url: str, params: dict) -> dict | None:
    """GET one OpenAlex page; raise QuotaExhausted on a daily limit."""
    for attempt in range(4):
        try:
            r = requests.get(url, params={**params, **_auth_params()}, timeout=60)
            if r.status_code == 429:
                retry_after = int(r.headers.get("retry-after", 0) or 0)
                if retry_after > _QUOTA_RETRY_AFTER_S or _looks_like_daily_limit(r.text):
                    raise QuotaExhausted(
                        f"OpenAlex daily/rate limit (429, retry-after={retry_after}s)")
                time.sleep(3.0 * (attempt + 1))
                continue
            if r.status_code in (402, 403) and _looks_like_daily_limit(r.text):
                raise QuotaExhausted(
                    f"OpenAlex daily/credit limit (HTTP {r.status_code}): {r.text[:150]}")
            if r.ok:
                return r.json()
            print(f"[openalex] HTTP {r.status_code}: {r.text[:200]}")
            return None
        except requests.RequestException as e:
            print(f"[openalex] request failed ({e}); retrying...")
            time.sleep(2.0 * (attempt + 1))
    return None


def reconstruct_abstract(inv: dict | None) -> str | None:
    """abstract_inverted_index {word: [positions]} -> plain text."""
    if not inv:
        return None
    slots: dict[int, str] = {}
    for word, positions in inv.items():
        for p in positions:
            slots[int(p)] = word
    if not slots:
        return None
    return " ".join(slots[i] for i in sorted(slots))


def work_to_row(work: dict, source: str = "openalex") -> dict:
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    cnp = work.get("citation_normalized_percentile") or {}
    topic = work.get("primary_topic") or {}
    authors = []
    for a in (work.get("authorships") or [])[:50]:
        author = a.get("author") or {}
        aid = (author.get("id") or "").rsplit("/", 1)[-1]
        if aid:
            authors.append({"author_id": aid, "name": author.get("display_name")})
    domain = (topic.get("domain") or {}).get("display_name")
    field = (topic.get("field") or {}).get("display_name")
    issns = src.get("issn") or ([src.get("issn_l")] if src.get("issn_l") else [])
    return {
        "paper_id": "OA:" + (work.get("id") or "").rsplit("/", 1)[-1],
        "oa_id": work.get("id"),
        "doi": normalize_doi(work.get("doi")),
        "title": work.get("title"),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "year": work.get("publication_year"),
        "venue": src.get("display_name"),
        "venue_issn": issns[0] if issns else None,
        "citation_count": work.get("cited_by_count"),
        "fwci": work.get("fwci"),
        "citation_normalized_percentile": cnp.get("value"),
        "is_in_top_1_percent": cnp.get("is_in_top_1_percent"),
        "is_in_top_10_percent": cnp.get("is_in_top_10_percent"),
        "domains": [d for d in (domain, field) if d],
        "subfield": (topic.get("subfield") or {}).get("display_name"),
        "authors": authors,
        "source": source,
    }


class OpenAlexFetcher(BaseFetcher):
    """Domain-balanced /works crawl, RANDOMLY sampled. ``domain_filters`` = list of
    OpenAlex filter strings (e.g. 'primary_topic.field.id:fields/17'); each is drawn from
    round-robin up to ``per_filter_cap`` so the field×year reference sets stay dense.

    Uses OpenAlex's own ``sample``+``seed`` mode (a fixed, seeded 10,000-work random draw
    per filter, walked page-by-page) instead of a plain cursor crawl. Without an explicit
    ``sort``, OpenAlex's cursor pagination returns works ORDERED BY CITED_BY_COUNT
    DESCENDING for this filter shape (confirmed empirically) - a plain crawl silently
    collects only the most-cited "blockbuster" papers per field, never anything typical.
    ``sample`` fixes that: same seed -> same 10,000-work pool every time, so resuming after
    a daily-limit wait (or a later, separate collection run) walks FURTHER into that same
    pool via an advancing page number, never re-drawing already-seen pages."""

    name = "openalex"
    SAMPLE_SIZE = 10000     # OpenAlex's own hard cap for sample=
    SAMPLE_SEED = 42        # fixed -> same 10,000-work random pool per filter, every run

    def __init__(self, year_from: int | None = None, year_to: int | None = None,
                 domain_filters: list[str] | None = None,
                 per_filter_cap: int | None = None):
        self.year_from = year_from
        self.year_to = year_to
        self.domain_filters = domain_filters or [""]
        self.per_filter_cap = per_filter_cap

    def _base_filter(self) -> str:
        parts = ["type:article", "has_abstract:true"]
        if self.year_from and self.year_to:
            parts.append(f"publication_year:{self.year_from}-{self.year_to}")
        elif self.year_from:
            parts.append(f"from_publication_date:{self.year_from}-01-01")
        elif self.year_to:
            parts.append(f"to_publication_date:{self.year_to}-12-31")
        return ",".join(parts)

    def rows(self, conn: sqlite3.Connection) -> Iterator[dict]:
        total_pages = -(-self.SAMPLE_SIZE // PER_PAGE)   # ceil division
        for df in self.domain_filters:
            filt = self._base_filter() + (f",{df}" if df else "")
            state_key = f"{self.name}:{df or 'all'}"
            page = int(get_cursor(conn, state_key) or "1")
            fetched = 0
            while page <= total_pages:
                if self.per_filter_cap and fetched >= self.per_filter_cap:
                    break
                data = _get(WORKS_URL, {"filter": filt, "per-page": PER_PAGE, "page": page,
                                        "sample": self.SAMPLE_SIZE, "seed": self.SAMPLE_SEED,
                                        "select": SELECT})
                if data is None:
                    break
                results = data.get("results", [])
                if not results:
                    break
                for work in results:
                    fetched += 1
                    yield work_to_row(work)
                page += 1
                set_cursor(conn, state_key, str(page), fetched)
                time.sleep(0.11)  # ~9 req/s: inside the polite-pool rate limit


def _wait_for_daily_reset(hours: float, already_elapsed: float = 0.0) -> None:
    """Idle for the daily-credit reset in 5-minute chunks, freeing memory first."""
    gc.collect()
    total_s = max(0, int(hours * 3600 - already_elapsed))
    chunk = 300
    resume = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + total_s))
    print(f"[openalex] idling {total_s / 3600:.1f}h for the quota reset "
          f"(resume ~{resume}); minimal-memory wait.", flush=True)
    waited = 0
    while waited < total_s:
        time.sleep(min(chunk, total_s - waited))
        waited += chunk
        if waited % 3600 < chunk:
            print(f"[openalex] waiting for daily reset: ~{(total_s - waited) / 3600:.1f}h left",
                  flush=True)


def collect_new_papers(conn: sqlite3.Connection, fetcher: "OpenAlexFetcher",
                       target: int | None, *, wait_days: int = 0, wait_hours: float = 25.0,
                       stall_limit: int = 2000, on_wait=None) -> int:
    """Collect up to target new papers, waiting out daily limits up to wait_days times.

    One session at a time (each frees its own Deduper). On a daily limit before the target,
    wait ~wait_hours and continue, at most wait_days times; stop early without waiting when the
    query is exhausted or stalls on duplicates. ``on_wait`` runs useful work at the start of
    each wait (e.g. LLM fills) and counts against the wait time.
    """
    total = 0
    days_left = int(wait_days)
    while target is None or total < target:
        remaining = None if target is None else target - total
        total += fetcher.run(conn, limit=remaining, stall_limit=stall_limit)
        reason = getattr(fetcher, "stop_reason", "exhausted")

        if target is not None and total >= target:
            print(f"[openalex] target reached: {total} new papers")
            break
        if reason == "limit":
            break
        if reason in ("stalled", "exhausted"):
            print(f"[openalex] no more new papers from this query ({reason}); stopping at "
                  f"{total}"
                  + (f"/{target}" if target else "")
                  + " — a daily-reset wait would not help.")
            break
        if reason == "quota":
            if days_left <= 0:
                print(f"[openalex] daily limit reached and wait-day budget spent; stopping at "
                      f"{total}" + (f"/{target}" if target else "") + " new papers.")
                break
            days_left -= 1
            print(f"[openalex] daily limit at {total}"
                  + (f"/{target}" if target else "")
                  + f"; {days_left} wait-day(s) remaining after this one.")
            elapsed = 0.0
            if on_wait is not None:
                t0 = time.time()
                try:
                    on_wait()
                except Exception as e:
                    print(f"[openalex] wait-time fill skipped: {type(e).__name__}: {e}")
                elapsed = time.time() - t0
            _wait_for_daily_reset(wait_hours, already_elapsed=elapsed)
            continue
        break
    return total


def fill_author_h_index(conn: sqlite3.Connection, batch: int = 100) -> int:
    """Batch-fill authors.h_index for authors that lack one (1 request / ``batch`` ids)."""
    ids = [r[0] for r in conn.execute(
        "SELECT author_id FROM authors WHERE h_index IS NULL")]
    ids = [i for i in ids if i.startswith("A")]  # OpenAlex author ids only
    filled = 0
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]
        data = _get(AUTHORS_URL, {
            "filter": "ids.openalex:" + "|".join(chunk),
            "per-page": batch, "select": "id,summary_stats"})
        for a in (data or {}).get("results", []):
            aid = (a.get("id") or "").rsplit("/", 1)[-1]
            h = (a.get("summary_stats") or {}).get("h_index")
            if aid and h is not None:
                conn.execute("UPDATE authors SET h_index = ? WHERE author_id = ?",
                             (int(h), aid))
                filled += 1
        conn.commit()
        time.sleep(0.11)
    print(f"[openalex] filled h_index for {filled} authors")
    return filled
