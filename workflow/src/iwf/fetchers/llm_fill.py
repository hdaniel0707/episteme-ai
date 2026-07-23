"""Extend-phase LLM auto-fill: pick a domain, or reconstruct a missing abstract/year.

Domain fill is harmless (fills a missing label, never marks ``source``, always trained).
Abstract fill tries a web search first (``websearch``), has the LLM verify/clean the hit, and
falls back to the LLM's own knowledge; either way it marks ``source='llm'`` so the abstract
can be excluded from training. Row caps come from config.yaml (``llm_fill_domain_max`` /
``llm_fill_abstract_max``); gates come from .env.workflow (``ENABLE_LLM_FILL`` /
``ENABLE_LLM_ABSTRACT_FILL`` / ``ENABLE_LLM_WEBSEARCH``). Every call is best-effort.
"""

from __future__ import annotations

import json
import sqlite3

from workflow.src.iwf import config
from workflow.src.iwf.config import load_config_yaml
from workflow.src.iwf.domains import ALLOWED_DOMAINS
from workflow.src.iwf.fetchers import websearch
from workflow.src.iwf.llm import client as llm_client

_DEFAULT_DOMAIN_MAX = 60
_DEFAULT_ABSTRACT_MAX = 50


def _cfg_int(key: str, default: int) -> int:
    """Read an integer knob from config.yaml, or return default."""
    try:
        v = load_config_yaml().get(key)
    except Exception:
        v = None
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


DOMAIN_SYSTEM = (
    "You are a research-domain classifier. Given a paper's title and abstract, choose the "
    "single best-matching field from a fixed list. Reply with ONLY that exact field name."
)


def llm_fill_domains(conn: sqlite3.Connection, limit: int | None = None) -> int:
    """Fill a domain from ALLOWED_DOMAINS for rows that have none."""
    if limit is None:
        limit = _cfg_int("llm_fill_domain_max", _DEFAULT_DOMAIN_MAX)
    if limit <= 0:
        print("[llm_fill] domain fill disabled (llm_fill_domain_max <= 0)")
        return 0
    rows = conn.execute("""
        SELECT paper_id, title, abstract FROM papers
        WHERE title IS NOT NULL AND TRIM(title) != ''
          AND (domains IS NULL OR TRIM(domains) IN ('', '[]'))
        LIMIT ?
    """, (limit,)).fetchall()
    if not rows:
        print("[llm_fill] no rows need a domain")
        return 0
    print(f"[llm_fill] domain: {len(rows)} candidate rows (cap {limit})")
    options = "\n".join(f"- {d}" for d in ALLOWED_DOMAINS)
    filled = 0
    for idx, (paper_id, title, abstract) in enumerate(rows):
        prompt = (
            f"Choose the single best field for this paper from the list below:\n{options}\n\n"
            f"Title: {title}\nAbstract: {abstract or '(not available)'}\n\n"
            '''Respond as JSON: {"domain": "<exact field name from the list>"}'''
        )
        resp = llm_client.complete(prompt, system=DOMAIN_SYSTEM, as_json=True)
        picked = resp.get("domain") if isinstance(resp, dict) else None
        if picked not in ALLOWED_DOMAINS:
            continue
        conn.execute("UPDATE papers SET domains = ? WHERE paper_id = ?",
                     (json.dumps([picked]), paper_id))
        filled += 1
        if (idx + 1) % 100 == 0:
            conn.commit()
            print(f"[llm_fill] domain: {idx + 1}/{len(rows)}, filled {filled}")
    conn.commit()
    print(f"[llm_fill] domain complete: filled {filled}/{len(rows)}")
    return filled


ABSTRACT_SYSTEM = (
    "You are a scientific-literature assistant helping reconstruct missing metadata for a "
    "paper. Accuracy matters more than coverage: only answer if reasonably confident, "
    "otherwise say you don't know rather than guessing."
)

_VERIFY_INSTRUCTION = (
    "A web search returned the candidate abstract below. Decide whether it genuinely belongs "
    "to the paper with this exact title. If it clearly matches, return it cleaned up (plain "
    "prose, no HTML). If it does not match, set known=false."
)

_KNOWLEDGE_INSTRUCTION = (
    "If you recognize this paper (or a very likely match) from your training data, provide "
    "its abstract and publication year from your own knowledge. If you are not reasonably "
    "confident, set known=false instead of guessing."
)


def llm_fill_abstracts(conn: sqlite3.Connection, limit: int | None = None) -> int:
    """Fill a missing abstract/year (web-search-first, then LLM knowledge)."""
    if limit is None:
        limit = _cfg_int("llm_fill_abstract_max", _DEFAULT_ABSTRACT_MAX)
    if limit <= 0:
        print("[llm_fill] abstract fill disabled (llm_fill_abstract_max <= 0)")
        return 0

    websearch_on = websearch.is_enabled()
    print(f"[llm_fill] abstract/year: web search {'ON' if websearch_on else 'OFF'} "
          f"(providers={config.get_str('WEBSEARCH_PROVIDERS', 'crossref,europepmc')})")

    rows = conn.execute("""
        SELECT paper_id, title, year, citation_count FROM papers
        WHERE title IS NOT NULL AND TRIM(title) != ''
          AND citation_count IS NOT NULL
          AND (abstract IS NULL OR TRIM(abstract) = '' OR year IS NULL)
        LIMIT ?
    """, (limit,)).fetchall()
    if not rows:
        print("[llm_fill] no rows need abstract/year fill")
        return 0
    print(f"[llm_fill] abstract/year: {len(rows)} candidate rows (cap {limit})")
    filled = 0
    n_web = 0
    for idx, (paper_id, title, year, citation_count) in enumerate(rows):
        retrieved = provider = None
        if websearch_on:
            hit = websearch.search_abstract(title, year)
            if hit:
                retrieved, provider = hit

        if retrieved:
            prompt = (
                f"Paper title: {title}\n"
                + (f"Known publication year: {year}\n" if year else "")
                + f"\n{_VERIFY_INSTRUCTION}\n\nCandidate abstract (from web search "
                  f"'{provider}'):\n{retrieved}\n\n"
                  '''Respond as JSON: {"known": <true|false>, "abstract": <cleaned string '''
                  '''or null>, "year": <int or null>}'''
            )
        else:
            prompt = (
                f"Paper title: {title}\n"
                + (f"Known publication year: {year}\n" if year else "")
                + f"\n{_KNOWLEDGE_INSTRUCTION}\n\n"
                  '''Respond as JSON: {"known": <true|false>, "abstract": <string or null>, '''
                  '''"year": <int or null>}'''
            )
        resp = llm_client.complete(prompt, system=ABSTRACT_SYSTEM, as_json=True)
        if not isinstance(resp, dict) or not resp.get("known"):
            continue
        updates: dict = {}
        abstract = resp.get("abstract")
        if isinstance(abstract, str) and len(abstract.strip()) > 20:
            updates["abstract"] = abstract.strip()
        yr = resp.get("year")
        if yr and not year:
            try:
                updates["year"] = int(yr)
            except (TypeError, ValueError):
                pass
        if not updates:
            continue
        if "abstract" in updates:
            updates["source"] = "llm"
            if retrieved:
                n_web += 1
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(f"UPDATE papers SET {set_clause} WHERE paper_id = ?",
                     (*updates.values(), paper_id))
        filled += 1
        if (idx + 1) % 25 == 0:
            conn.commit()
            print(f"[llm_fill] abstract/year: {idx + 1}/{len(rows)}, filled {filled} "
                  f"({n_web} web-verified)")
    conn.commit()
    print(f"[llm_fill] abstract/year complete: filled {filled}/{len(rows)} "
          f"({n_web} web-verified, {filled - n_web} from LLM knowledge)")
    return filled
