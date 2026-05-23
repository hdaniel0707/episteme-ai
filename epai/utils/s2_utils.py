import requests
import time
from typing import Iterator

REQUEST_DELAY = 1.5
PAGE_SIZE = 100
MAX_RESULTS = 1000

API_BASE = "https://api.semanticscholar.org/graph/v1"


def _session(api_key: str | None) -> requests.Session:
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    if api_key:
        s.headers["x-api-key"] = api_key
    return s


def iter_search(
    query: str, session: requests.Session, paper_fields: str
) -> Iterator[dict]:
    """Paginate through paper/search results for *query*."""
    offset = 0
    while offset < MAX_RESULTS:
        resp = session.get(
            f"{API_BASE}/paper/search",
            params={
                "query": query,
                "fields": paper_fields,
                "limit": PAGE_SIZE,
                "offset": offset,
            },
            timeout=30,
        )
        if not resp.ok:
            print(f"    HTTP {resp.status_code} at offset {offset} — stopping")
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
