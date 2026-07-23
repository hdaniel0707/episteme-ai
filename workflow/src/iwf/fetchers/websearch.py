"""Look up a paper's abstract by title from free, no-key scholarly APIs (extend-phase fill).

Used by ``llm_fill.llm_fill_abstracts`` as a trustworthy first source before the LLM: given a
title, return the real abstract, which the LLM then verifies/cleans. Providers (all free, no
registration): ``crossref``, ``europepmc``; an optional keyed ``api`` (Tavily/Serper) is off
unless configured. Every call is best-effort and returns None on any failure.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import requests

from workflow.src.iwf import config

_JATS_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _clean(text: str | None) -> str:
    """Strip HTML/JATS tags and collapse whitespace."""
    if not text:
        return ""
    return _WS.sub(" ", _JATS_TAG.sub(" ", str(text))).strip()


def _norm_title(t: str | None) -> str:
    """Lowercase, keep only letters and digits."""
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def _similar(a: str | None, b: str | None) -> float:
    """Similarity ratio between two titles (0..1)."""
    na, nb = _norm_title(a), _norm_title(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _timeout() -> float:
    return config.get_float("WEBSEARCH_TIMEOUT", 15.0)


def _min_similarity() -> float:
    return config.get_float("WEBSEARCH_MIN_SIMILARITY", 0.82)


def _crossref(title: str, year: int | None) -> str | None:
    """Search Crossref by title and return a matching abstract."""
    params = {"query.bibliographic": title, "rows": 3,
              "select": "title,abstract,published"}
    mailto = config.get_str("OPENALEX_MAILTO", "")
    if mailto:
        params["mailto"] = mailto
    try:
        r = requests.get("https://api.crossref.org/works", params=params, timeout=_timeout())
        if not r.ok:
            return None
        items = (r.json().get("message") or {}).get("items") or []
    except (requests.RequestException, ValueError):
        return None
    for it in items:
        cand_title = (it.get("title") or [""])[0]
        if _similar(title, cand_title) < _min_similarity():
            continue
        abstract = _clean(it.get("abstract"))
        if len(abstract) > 40:
            return abstract
    return None


def _europepmc(title: str, year: int | None) -> str | None:
    """Search Europe PMC by title and return a matching abstract."""
    query = f'TITLE:"{title}"'
    if year:
        query += f" AND PUB_YEAR:{year}"
    params = {"query": query, "resultType": "core", "format": "json", "pageSize": 3}
    try:
        r = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params=params, timeout=_timeout())
        if not r.ok:
            return None
        results = ((r.json().get("resultList") or {}).get("result")) or []
    except (requests.RequestException, ValueError):
        return None
    for it in results:
        if _similar(title, it.get("title")) < _min_similarity():
            continue
        abstract = _clean(it.get("abstractText"))
        if len(abstract) > 40:
            return abstract
    return None


def _api_provider(title: str, year: int | None) -> str | None:
    """Optional keyed web search (Tavily/Serper); off without a key."""
    provider = config.get_str("WEBSEARCH_API_PROVIDER", "").strip().lower()
    key = config.secret("WEBSEARCH_API_KEY")
    if not provider or not key:
        return None
    q = f"{title} abstract"
    try:
        if provider == "tavily":
            r = requests.post("https://api.tavily.com/search", timeout=_timeout(), json={
                "api_key": key, "query": q, "max_results": 3, "include_answer": False})
            if not r.ok:
                return None
            snippets = [_clean(x.get("content")) for x in (r.json().get("results") or [])]
        elif provider == "serper":
            r = requests.post("https://google.serper.dev/search", timeout=_timeout(),
                              headers={"X-API-KEY": key}, json={"q": q, "num": 3})
            if not r.ok:
                return None
            snippets = [_clean(x.get("snippet")) for x in (r.json().get("organic") or [])]
        else:
            return None
    except (requests.RequestException, ValueError):
        return None
    text = " ".join(s for s in snippets if s)
    return text if len(text) > 60 else None


_PROVIDERS = {
    "crossref": _crossref,
    "europepmc": _europepmc,
    "api": _api_provider,
}


def is_enabled() -> bool:
    """True when web search is turned on (ENABLE_LLM_WEBSEARCH)."""
    return config.get_bool("ENABLE_LLM_WEBSEARCH", False)


def search_abstract(title: str, year: int | None = None) -> tuple[str, str] | None:
    """Return (abstract, provider) for the title, or None if not found."""
    if not title or not title.strip():
        return None
    order = [p.strip().lower() for p in
             config.get_str("WEBSEARCH_PROVIDERS", "crossref,europepmc").split(",")
             if p.strip()]
    for name in order:
        fn = _PROVIDERS.get(name)
        if fn is None:
            continue
        try:
            hit = fn(title, year)
        except Exception:
            hit = None
        if hit:
            return hit, name
    return None
