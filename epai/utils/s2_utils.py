import requests
import time
from typing import Iterator

REQUEST_DELAY = 1.5
PAGE_SIZE = 100
MAX_RESULTS = 1000

# Rate limits are the normal failure mode here, especially unauthenticated,
# where the quota is shared across every anonymous caller and is often already
# exhausted. Waits are capped so a long backoff chain stays interruptible.
MAX_RETRIES = 6
RETRY_BACKOFF = 4.0
RETRY_MAX_WAIT = 60.0
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

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


def _get_with_retry(session: requests.Session, url: str, params: dict, timeout: int):
    """GET *url*, retrying rate limits and transient server errors.

    Returns the response, or ``None`` once the retries are exhausted. Raises
    :class:`S2AuthError` immediately on 401/403 — retrying a bad key only wastes
    time, and silently treating it as "no results" hides the real cause behind
    an empty corpus.
    """
    for attempt in range(MAX_RETRIES):
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
            return resp

        if resp.status_code not in RETRY_STATUSES:
            print(f"    HTTP {resp.status_code} — giving up on this query")
            return None

        if attempt < MAX_RETRIES - 1:
            # Honour Retry-After when the server sends one, else back off.
            retry_after = resp.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else RETRY_BACKOFF * (2**attempt)
            except ValueError:
                wait = RETRY_BACKOFF * (2**attempt)
            wait = min(wait, RETRY_MAX_WAIT)
            print(
                f"    HTTP {resp.status_code} — retrying in {wait:.0f}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(wait)

    raise S2RateLimitError(
        f"HTTP {resp.status_code} after {MAX_RETRIES} attempts. The request kept "
        "being refused — with no API key this usually means the shared "
        "unauthenticated pool is exhausted, not that anything is wrong with the "
        "query."
    )


def check_credentials(session: requests.Session) -> tuple[str, str]:
    """Probe the API with one cheap request; return ``(status, message)``.

    ``status`` is one of:
      ``"ok"``          — the request succeeded.
      ``"auth_failed"`` — 401/403, i.e. the key is invalid, expired or revoked.
      ``"unavailable"`` — rate-limited or unreachable; says nothing about the key.

    Worth the single extra call at startup: a rejected key otherwise looks
    exactly like "these queries returned nothing", and you only find out after
    paging through every domain.
    """
    try:
        resp = session.get(
            f"{API_BASE}/paper/search",
            params={"query": "machine learning", "fields": "paperId", "limit": 1},
            timeout=30,
        )
    except Exception as e:
        return "unavailable", f"could not reach the API: {type(e).__name__}: {e}"

    if resp.status_code in (401, 403):
        return "auth_failed", f"HTTP {resp.status_code} — the API key was rejected"
    if resp.ok:
        return "ok", "credentials accepted"
    return "unavailable", f"HTTP {resp.status_code} — API not answering normally"


def iter_search(
    query: str,
    session: requests.Session,
    paper_fields: str,
    extra_params: dict | None = None,
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
            session, f"{API_BASE}/paper/search/bulk", params, timeout=60
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
