"""OpenAlex API client — the drop-in counterpart to :mod:`epai.utils.s2_utils`.

Same surface as the Semantic Scholar client (``_session``, ``check_service``,
``iter_search``, ``iter_search_bulk``, two exception types), so the fetch stage
built on one can be pointed at the other with almost no change. The differences
below are the ones that actually matter when reading either file.

**Two identifiers, neither required, and they are not the same thing.**

  - ``OPENALEX_API_KEY`` — an OpenAlex Premium key, sent as the ``api_key``
    parameter. This *is* a secret. It raises the rate limit and gives access to
    fresher data. Optional: the free tier serves the same works with the same
    fields, so the fetch stage works identically without it.
  - ``OPENALEX_MAILTO`` — an email address in the ``mailto`` parameter, which
    puts the caller in the "polite pool": a separate, faster, more consistently
    served queue. A courtesy identifier, not a secret, and never a requirement
    — it travels in the query string in plain text and losing it costs nothing.
    Redundant when a key is set, since the key identifies the caller already.

Both are sent when present, and either works alone.

**Rate limits are real limits, not load shedding.** Semantic Scholar refuses a
large fraction of well-behaved requests regardless of cadence, which is why
``s2_utils`` retries 429s flat and often. OpenAlex publishes 10 requests/second
and 100,000/day and largely honours it: a 429 here means we genuinely went too
fast, so backing off is the correct response and it works. Hence exponential
backoff and a small retry budget — the opposite of the S2 policy, for the
opposite reason.

**Search targets title + abstract.** The bare ``search=`` parameter also
searches full text, which drags in papers that merely mention a term in a
footnote. ``filter=title_and_abstract.search:...`` keeps the match where a
domain label can be justified from it.

**Pagination has two modes**, mirroring S2's two search endpoints:
:func:`iter_search` uses page/per-page (relevance-ranked, capped by OpenAlex at
10,000 results per query — ten times S2's ceiling), and
:func:`iter_search_bulk` uses cursor paging, which has no ceiling at all.
"""

import random
import time
from typing import Callable, Iterator

import requests
from tqdm import tqdm

API_BASE = "https://api.openalex.org"

# OpenAlex allows 10 req/s. 0.2s between pages leaves headroom for the round
# robin in the fetch stage, which has several queries in flight at once.
REQUEST_DELAY = 0.2

# 200 is the documented maximum for per-page.
PAGE_SIZE = 200

# Hard ceiling on page-based paging: OpenAlex refuses page × per-page > 10,000.
# Cursor paging (iter_search_bulk) is not subject to it.
MAX_RESULTS = 10_000

# A 429 from OpenAlex means we exceeded a published rate, so waiting genuinely
# fixes it and a short exponential backoff clears it. Contrast s2_utils, where
# a 429 says nothing about our cadence and the budget has to be 25 attempts to
# absorb the refusal rate.
MAX_RETRIES = 6
RETRY_BACKOFF = 2.0
RETRY_MAX_WAIT = 60.0
RETRY_JITTER = 0.3

SERVER_ERROR_STATUSES = frozenset({500, 502, 503, 504})
RETRY_STATUSES = SERVER_ERROR_STATUSES | {429}

# Retries for the startup reachability probe. Small on purpose: the point is to
# see through one bad moment, not to sit at a prompt for minutes.
CHECK_ATTEMPTS = 4

USER_AGENT = "episteme-ai/0.1 (https://github.com/; python-requests)"


class OARequestError(RuntimeError):
    """OpenAlex rejected the request itself (HTTP 400).

    Its own type because it is neither retryable nor per-query: the ``filter``
    and ``select`` strings come from one place and are identical for every
    query, so if OpenAlex will not parse them once it will not parse them ever.
    Callers should abort the run and fix the parameters rather than log the
    same error for each domain and finish with an empty corpus.
    """


class OAAuthError(OARequestError):
    """OpenAlex rejected our credentials (HTTP 401/403).

    A subclass, because it shares the property that makes the parent fatal —
    the same key goes out on every request, so one rejection means all of them
    — and callers that only want "abort the run" can catch the parent. It is a
    separate type anyway because the *fix* is completely different: a malformed
    filter needs a config edit, a rejected key needs a new key or a run without
    one.
    """


class OARateLimitError(RuntimeError):
    """Retries were exhausted against a rate limit or a failing server.

    Distinct from "this query has no more results", and the distinction is the
    whole point: both leave a domain short of its quota, but one is fixed by
    slowing down or waiting for the daily quota to roll over, and the other by
    writing better queries.
    """


def _session(mailto: str | None, api_key: str | None = None) -> requests.Session:
    """Build a session carrying whichever identifiers are available.

    *mailto* selects the polite pool; *api_key* is an OpenAlex Premium key.
    They are independent — either, both, or neither.

    ``mailto`` goes in both the query string (which is what OpenAlex documents)
    and the User-Agent, because their routing has honoured both over time and
    sending both costs nothing. The key goes only in the query string, where
    OpenAlex expects it, and deliberately *not* into the User-Agent: that
    string is echoed in error messages and logs, and a secret does not belong
    in either.
    """
    s = requests.Session()
    s.headers.update({"Accept": "application/json", "User-Agent": USER_AGENT})
    # requests merges session-level params into every request's own params.
    params = {}
    if mailto:
        s.headers["User-Agent"] = f"{USER_AGENT} mailto:{mailto}"
        params["mailto"] = mailto
    if api_key:
        params["api_key"] = api_key
    s.params = params
    return s


def _safe_params(params: dict) -> dict:
    """Copy of *params* with any credential masked, for error messages.

    A 400 is worth reporting with the parameters that caused it, but those
    parameters may carry the key — and an exception message ends up in logs,
    tracebacks and pasted bug reports.
    """
    if "api_key" not in params:
        return params
    return {**params, "api_key": "***"}


def _escape(value: str) -> str:
    """Make *value* safe to embed in a ``filter=`` expression.

    OpenAlex splits the filter string on commas (AND) and pipes (OR), so a
    query containing either would silently be parsed as two filters and match
    something entirely different. Both are replaced with spaces, which is
    harmless inside a search term.
    """
    return str(value).replace(",", " ").replace("|", " ").strip()


def _filter_string(query: str | None, filters: dict | None) -> str:
    """Build the comma-joined ``filter=`` expression.

    *query*, when given, becomes ``title_and_abstract.search:<query>``. Values
    in *filters* may be lists, which are joined with ``|`` (OpenAlex's OR).
    """
    parts = []
    if query:
        parts.append(f"title_and_abstract.search:{_escape(query)}")
    for key, value in (filters or {}).items():
        if isinstance(value, (list, tuple, set)):
            value = "|".join(_escape(v) for v in value)
        else:
            value = _escape(value)
        parts.append(f"{key}:{value}")
    return ",".join(parts)


def _tag(query: str | None) -> str:
    """Short label for the query a message belongs to.

    Round-robin means several queries are in flight at once, so an untagged
    "HTTP 429 — retrying" says nothing about which one stalled.
    """
    q = str(query or "?")
    return q if len(q) <= 40 else q[:37] + "..."


def _get_with_retry(
    session: requests.Session,
    url: str,
    params: dict,
    timeout: int,
    on_status: Callable[[str], None] | None = None,
    tag: str | None = None,
):
    """GET *url*, retrying rate limits and transient server errors.

    Returns the response, or ``None`` on a non-retryable error. Raises
    :class:`OAAuthError` on 401/403 and :class:`OARequestError` on 400, both
    immediately: one means the key is bad, the other that the filter or select
    expression is malformed, and retrying either only wastes time while hiding
    the cause behind an empty corpus.

    *on_status* receives a short human-readable string whenever the request is
    stalled, and ``""`` once it clears, so a caller's progress bar can show
    that a long wait is a backoff rather than a hang.
    """
    attempts = 0
    reported = False

    while True:
        resp = session.get(url, params=params, timeout=timeout)

        if resp.status_code in (400, 401, 403):
            # OpenAlex puts a usable explanation in the body; without it the
            # message is just "400" and every filter key is a suspect.
            try:
                detail = resp.json().get("message") or resp.text[:200]
            except Exception:
                detail = resp.text[:200]

            if resp.status_code in (401, 403):
                raise OAAuthError(
                    f"HTTP {resp.status_code} from OpenAlex: {detail}\n"
                    "  OPENALEX_API_KEY is set but was rejected — invalid, "
                    "expired, or the Premium subscription has lapsed.\n"
                    "  OpenAlex serves the same works without a key, so "
                    "unsetting it (or passing --no-api-key) is a working "
                    "fallback, just at the free rate limit."
                )
            raise OARequestError(
                f"HTTP 400 from OpenAlex: {detail}\n"
                f"  request params: {_safe_params(params)}\n"
                "  This is a malformed filter/select expression, not a transient "
                "failure — every query would fail identically. See "
                "https://docs.openalex.org/api-entities/works/filter-works"
            )

        if resp.ok:
            if on_status:
                on_status("")
            if reported:
                tqdm.write(f"    [{_tag(tag)}] recovered after {attempts} retry/retries")
            return resp

        if resp.status_code not in RETRY_STATUSES:
            if on_status:
                on_status("")
            tqdm.write(
                f"    [{_tag(tag)}] HTTP {resp.status_code} — giving up on this query"
            )
            return None

        attempts += 1
        if attempts >= MAX_RETRIES:
            break

        # Honour Retry-After whenever the server sends one; it beats guessing.
        retry_after = resp.headers.get("Retry-After")
        try:
            wait = float(retry_after) if retry_after else None
        except ValueError:
            wait = None
        if wait is None:
            wait = RETRY_BACKOFF * (2 ** (attempts - 1)) + random.uniform(0, RETRY_JITTER)
        wait = min(wait, RETRY_MAX_WAIT)

        if on_status:
            on_status(f"backing off {attempts}")
        if not reported:
            reported = True
            kind = (
                "rate limited"
                if resp.status_code == 429
                else f"server error {resp.status_code}"
            )
            tqdm.write(
                f"    [{_tag(tag)}] {kind} — waiting {wait:.1f}s and retrying "
                f"(attempt {attempts}/{MAX_RETRIES})"
            )

        time.sleep(wait)

    if on_status:
        on_status("")
    raise OARateLimitError(
        f"HTTP {resp.status_code} after {attempts} attempts. OpenAlex allows "
        "10 requests/second and 100,000/day; a persistent 429 means either the "
        "cadence is too high or the daily quota is spent (it resets at UTC "
        "midnight). Setting OPENALEX_MAILTO moves the run to the polite pool, "
        "which is served more consistently."
    )


def check_service(
    session: requests.Session, attempts: int = CHECK_ATTEMPTS
) -> tuple[str, str]:
    """Probe the API with a cheap request; return ``(status, message)``.

    ``status`` is one of:
      ``"ok"``          — the request succeeded.
      ``"auth_failed"`` — 401/403, i.e. the Premium key was rejected.
      ``"bad_request"`` — 400, i.e. OpenAlex refused the request itself.
      ``"unavailable"`` — rate-limited or unreachable.

    Worth the one extra call: both a rejected key and an outage otherwise look
    exactly like "these queries returned nothing", and you only find that out
    after paging through every domain. Only 401/403 is conclusive on the first
    response — retrying a rejected key proves nothing — so unlike the other
    outcomes it returns without using the remaining attempts.
    """
    last = "no response"

    for attempt in range(attempts):
        try:
            resp = session.get(
                f"{API_BASE}/works",
                params={"filter": "title_and_abstract.search:machine learning",
                        "select": "id",
                        "per-page": 1},
                timeout=30,
            )
        except Exception as e:
            last = f"could not reach the API: {type(e).__name__}: {e}"
        else:
            if resp.status_code in (401, 403):
                return "auth_failed", f"HTTP {resp.status_code} — the API key was rejected"
            if resp.status_code == 400:
                return "bad_request", "HTTP 400 — request refused"
            if resp.ok:
                count = (resp.json().get("meta") or {}).get("count")
                # Name the identifiers sent, not a tier. A 200 does not report
                # which pool served it, so anything more specific than this
                # would be invention.
                sent = []
                if session.params.get("api_key"):
                    sent.append("api_key")
                if session.params.get("mailto"):
                    sent.append("mailto")
                how = "+".join(sent) if sent else "anonymous"
                return "ok", f"reachable ({how}, {count:,} works match the probe)"
            last = f"HTTP {resp.status_code} — API not answering normally"

        if attempt < attempts - 1:
            time.sleep(RETRY_BACKOFF * (2**attempt) + random.uniform(0, RETRY_JITTER))

    return "unavailable", f"{last} (after {attempts} attempts)"


def iter_search(
    query: str,
    session: requests.Session,
    select: str,
    filters: dict | None = None,
    on_status: Callable[[str], None] | None = None,
) -> Iterator[dict]:
    """Paginate through /works for *query*, relevance-ranked.

    OpenAlex sorts by ``relevance_score`` by default whenever a search is part
    of the filter, so this is the counterpart to S2's ``paper/search``. Page
    paging is capped by the API at ``MAX_RESULTS`` (10,000) hits per query —
    generous enough that the cap is rarely the binding constraint; use
    :func:`iter_search_bulk` when it is.

    *filters* is merged into the ``filter=`` expression, for server-side
    narrowing such as ``{"from_publication_date": "1900-01-01"}``. Filters
    applied here only save bandwidth — callers must not rely on them, because
    the row-local rules re-check every field anyway.
    """
    filter_str = _filter_string(query, filters)
    fetched = 0
    page = 1

    while fetched < MAX_RESULTS:
        resp = _get_with_retry(
            session,
            f"{API_BASE}/works",
            {
                "filter": filter_str,
                "select": select,
                "per-page": min(PAGE_SIZE, MAX_RESULTS - fetched),
                "page": page,
            },
            timeout=30,
            on_status=on_status,
            tag=query,
        )
        if resp is None:
            break

        data = resp.json()
        batch = data.get("results") or []
        if not batch:
            break

        yield from batch

        fetched += len(batch)
        page += 1
        total = (data.get("meta") or {}).get("count", 0)
        if fetched >= total:
            break

        time.sleep(REQUEST_DELAY)


def iter_search_bulk(
    query: str,
    session: requests.Session,
    select: str,
    filters: dict | None = None,
    on_status: Callable[[str], None] | None = None,
) -> Iterator[dict]:
    """Paginate through /works for *query* using cursor paging.

    Cursor paging lifts the 10,000-result ceiling of :func:`iter_search` and
    walks the result set by continuation token instead of page number. The
    match itself is identical — same filter, same relevance ordering — which is
    a real advantage over the S2 equivalent, where the bulk endpoint changes
    the matching semantics as well as the paging.

    Prefer :func:`iter_search` anyway: cursor paging keeps a server-side
    position, so it costs OpenAlex more, and 10,000 relevance-ranked hits per
    query is already far past the point where relevance has decayed to noise.
    """
    filter_str = _filter_string(query, filters)
    cursor: str | None = "*"

    while cursor:
        resp = _get_with_retry(
            session,
            f"{API_BASE}/works",
            {
                "filter": filter_str,
                "select": select,
                "per-page": PAGE_SIZE,
                "cursor": cursor,
            },
            timeout=60,
            on_status=on_status,
            tag=query,
        )
        if resp is None:
            break

        data = resp.json()
        batch = data.get("results") or []
        if not batch:
            break

        yield from batch

        cursor = (data.get("meta") or {}).get("next_cursor")
        if not cursor:
            break

        time.sleep(REQUEST_DELAY)


def get_work(session: requests.Session, work_id: str, select: str | None = None) -> dict:
    """Fetch a single work by OpenAlex ID, DOI, PMID, or MAG ID.

    OpenAlex resolves all of them on the same path, e.g. ``W2741809807``,
    ``doi:10.1038/nature12373``, or a full ``https://doi.org/...`` URL.
    """
    params = {"select": select} if select else {}
    resp = _get_with_retry(
        session, f"{API_BASE}/works/{work_id}", params, timeout=30, tag=work_id
    )
    if resp is None:
        raise OARequestError(f"OpenAlex returned no usable response for {work_id!r}")
    return resp.json()
