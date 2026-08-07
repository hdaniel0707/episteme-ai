import random
import requests
import time
from typing import Callable, Iterator

from tqdm import tqdm

REQUEST_DELAY = 1.5
PAGE_SIZE = 100
MAX_RESULTS = 1000

# 429 and 5xx need opposite strategies here, so they get separate budgets.
#
# Semantic Scholar sheds load rather than policing cadence: measured against a
# valid key, roughly 25-30% of requests succeed and that fraction is flat
# whether calls are issued back-to-back or 8s apart. So a 429 is not a
# complaint about our rate and backing off does not improve the next attempt's
# odds — it only adds latency. Retries are therefore near-constant, spaced to
# respect S2's documented ~1 req/s for keyed callers, with jitter so parallel
# queries do not lock into lockstep.
#
# The budget has to absorb that hit rate: at ~27% per request, 6 attempts drop
# ~15% of pages outright, which reads downstream as "this query ran dry" and
# silently shrinks a domain. 25 attempts puts that below 0.1%.
THROTTLE_RETRIES = 25
THROTTLE_WAIT = 1.1
THROTTLE_JITTER = 0.4

# 5xx is the opposite case: a struggling server genuinely benefits from being
# left alone, so these keep exponential backoff and a tight budget.
MAX_RETRIES = 6
RETRY_BACKOFF = 4.0
RETRY_MAX_WAIT = 60.0
SERVER_ERROR_STATUSES = frozenset({500, 502, 503, 504})
RETRY_STATUSES = SERVER_ERROR_STATUSES | {429}

# Only report a stall once it is out of the ordinary. With a ~27% success rate
# a handful of 429s is the normal path to every page; logging each one buries
# the run in noise that looks like failure while it is working fine.
THROTTLE_REPORT_AFTER = 8

# Attempts for the startup credential probe. Fewer than THROTTLE_RETRIES: the
# point is to see through a throttled moment, not to sit at a prompt for
# minutes.
CHECK_ATTEMPTS = 6

API_BASE = "https://api.semanticscholar.org/graph/v1"


class S2AuthError(RuntimeError):
    """The API rejected our credentials (401/403).

    Its own type because it is neither retryable nor per-query: every
    subsequent request will fail the same way, so callers should abort the
    whole run rather than log it once per query and carry on to an empty
    corpus.
    """


class S2RateLimitError(RuntimeError):
    """Retries were exhausted against a rate limit or a failing server.

    Distinct from "this query has no more results", and the distinction is the
    whole point: both leave a domain short of its quota, but one is fixed by
    waiting or using a key and the other by writing better queries. Reporting
    a rate limit as "add or broaden this domain's queries" sends you off
    rewriting a config that was never the problem.
    """


def _session(api_key: str | None) -> requests.Session:
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    if api_key:
        s.headers["x-api-key"] = api_key
    return s


def _tag(params: dict) -> str:
    """Short label for the query a message belongs to.

    Round-robin means several queries are in flight at once, so an untagged
    "HTTP 429 — retrying" says nothing about which one stalled.
    """
    query = str(params.get("query", "?"))
    return query if len(query) <= 40 else query[:37] + "..."


def _get_with_retry(
    session: requests.Session,
    url: str,
    params: dict,
    timeout: int,
    on_status: Callable[[str], None] | None = None,
):
    """GET *url*, retrying throttling and transient server errors.

    Returns the response, or ``None`` on a non-retryable error. Raises
    :class:`S2AuthError` immediately on 401/403 — retrying a bad key only wastes
    time, and silently treating it as "no results" hides the real cause behind
    an empty corpus.

    429 and 5xx are budgeted separately and deliberately: see the constants at
    the top of this module for why a 429 here is retried flat-and-often while a
    5xx is backed off exponentially.

    *on_status* receives a short human-readable string whenever the request is
    stalled, and ``""`` once it clears. Waiting out a throttle can take tens of
    seconds during which a caller's progress bar would otherwise sit frozen with
    no indication it is still alive.
    """
    throttled = 0
    server_errors = 0
    reported = False

    while True:
        resp = session.get(url, params=params, timeout=timeout)

        if resp.status_code in (401, 403):
            raise S2AuthError(
                f"HTTP {resp.status_code} from Semantic Scholar. The API key is "
                "invalid, expired, or revoked. Request a new one at "
                "https://www.semanticscholar.org/product/api#api-key-form, or "
                "unset SEMANTIC_SCHOLAR_API_KEY to run unauthenticated (slower, "
                "shared rate limit)."
            )

        if resp.ok:
            if on_status:
                on_status("")
            if reported:
                tqdm.write(
                    f"    [{_tag(params)}] recovered after {throttled + server_errors} "
                    "throttled attempt(s)"
                )
            return resp

        if resp.status_code not in RETRY_STATUSES:
            if on_status:
                on_status("")
            tqdm.write(f"    [{_tag(params)}] HTTP {resp.status_code} — giving up on this query")
            return None

        # Honour Retry-After whenever the server sends one; it beats guessing.
        retry_after = resp.headers.get("Retry-After")
        try:
            explicit_wait = float(retry_after) if retry_after else None
        except ValueError:
            explicit_wait = None

        if resp.status_code in SERVER_ERROR_STATUSES:
            server_errors += 1
            if server_errors >= MAX_RETRIES:
                break
            wait = explicit_wait or RETRY_BACKOFF * (2 ** (server_errors - 1))
            wait = min(wait, RETRY_MAX_WAIT)
        else:
            throttled += 1
            if throttled >= THROTTLE_RETRIES:
                break
            wait = explicit_wait or THROTTLE_WAIT + random.uniform(0, THROTTLE_JITTER)

        if on_status:
            on_status(f"throttled {throttled + server_errors}")
        if throttled + server_errors >= THROTTLE_REPORT_AFTER and not reported:
            reported = True
            tqdm.write(
                f"    [{_tag(params)}] HTTP {resp.status_code} — still retrying after "
                f"{throttled + server_errors} attempts; Semantic Scholar is shedding load"
            )

        time.sleep(wait)

    if on_status:
        on_status("")
    attempts = throttled + server_errors
    raise S2RateLimitError(
        f"HTTP {resp.status_code} after {attempts} attempts. The request kept "
        "being refused — with no API key this usually means the shared "
        "unauthenticated pool is exhausted, not that anything is wrong with the "
        "query."
    )


def check_credentials(
    session: requests.Session, attempts: int = CHECK_ATTEMPTS
) -> tuple[str, str]:
    """Probe the API with a cheap request; return ``(status, message)``.

    ``status`` is one of:
      ``"ok"``          — the request succeeded.
      ``"auth_failed"`` — 401/403, i.e. the key is invalid, expired or revoked.
      ``"unavailable"`` — rate-limited or unreachable; says nothing about the key.

    Worth the extra call at startup: a rejected key otherwise looks exactly like
    "these queries returned nothing", and you only find out after paging through
    every domain.

    The probe retries, because it must not be stricter than the fetching it
    guards. Semantic Scholar throttles hard enough that a *valid* key sees 429s
    on a large fraction of requests; :func:`_get_with_retry` rides those out, so
    a one-shot probe would abort runs that were going to succeed. Only 401/403
    is conclusive on the first response — retrying a rejected key proves nothing.
    """
    last = "no response"

    for attempt in range(attempts):
        try:
            resp = session.get(
                f"{API_BASE}/paper/search",
                params={"query": "machine learning", "fields": "paperId", "limit": 1},
                timeout=30,
            )
        except Exception as e:
            last = f"could not reach the API: {type(e).__name__}: {e}"
        else:
            if resp.status_code in (401, 403):
                return "auth_failed", f"HTTP {resp.status_code} — the API key was rejected"
            if resp.ok:
                return "ok", "credentials accepted"
            last = f"HTTP {resp.status_code} — API not answering normally"

        if attempt < attempts - 1:
            # Flat, not exponential, for the same reason the fetch path is: the
            # next attempt's odds do not improve by waiting longer.
            time.sleep(THROTTLE_WAIT + random.uniform(0, THROTTLE_JITTER))

    return "unavailable", f"{last} (after {attempts} attempts)"


def iter_search(
    query: str,
    session: requests.Session,
    paper_fields: str,
    extra_params: dict | None = None,
    on_status: Callable[[str], None] | None = None,
) -> Iterator[dict]:
    """Paginate through paper/search results for *query*.

    Relevance-ranked search. Capped at ``MAX_RESULTS`` (1000) hits per query by
    the API itself, so this is the right endpoint when a few hundred papers per
    query is enough; use :func:`iter_search_bulk` when it is not.

    ``extra_params`` is merged into the query string, for server-side filters
    such as ``{"year": "-2020"}`` (up to and including 2020). Filters applied
    here only save bandwidth — callers must not rely on them, because the
    row-local rules re-check every field anyway.
    """
    offset = 0
    while offset < MAX_RESULTS:
        resp = _get_with_retry(
            session,
            f"{API_BASE}/paper/search",
            {
                "query": query,
                "fields": paper_fields,
                "limit": PAGE_SIZE,
                "offset": offset,
                **(extra_params or {}),
            },
            timeout=30,
            on_status=on_status,
        )
        if resp is None:
            break

        data = resp.json()
        batch = data.get("data") or []
        if not batch:
            break

        yield from batch

        total = data.get("total", 0)
        offset += PAGE_SIZE
        if offset >= total:
            break

        time.sleep(REQUEST_DELAY)


def iter_search_bulk(
    query: str,
    session: requests.Session,
    paper_fields: str,
    extra_params: dict | None = None,
    on_status: Callable[[str], None] | None = None,
) -> Iterator[dict]:
    """Paginate through paper/search/bulk results for *query*.

    The bulk endpoint lifts the 1000-hit ceiling of :func:`iter_search` and
    paginates by continuation ``token`` rather than ``offset``. Note that it
    matches differently: the query string is a boolean/term match rather than a
    natural-language relevance ranking, and results are not relevance-ordered.
    Prefer :func:`iter_search` unless a domain genuinely cannot reach its target
    within 1000 hits per query.
    """
    token: str | None = None
    while True:
        params = {
            "query": query,
            "fields": paper_fields,
            **(extra_params or {}),
        }
        if token:
            params["token"] = token

        resp = _get_with_retry(
            session,
            f"{API_BASE}/paper/search/bulk",
            params,
            timeout=60,
            on_status=on_status,
        )
        if resp is None:
            break

        data = resp.json()
        batch = data.get("data") or []
        if not batch:
            break

        yield from batch

        token = data.get("token")
        if not token:
            break

        time.sleep(REQUEST_DELAY)
