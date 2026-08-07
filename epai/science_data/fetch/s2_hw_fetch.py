"""
Hello-world script for the Semantic Scholar API — lookup by DOI.

Takes a DOI rather than an S2 paper ID: the API accepts "DOI:<doi>" as a paper
identifier on the same /paper/ endpoint. Standalone on purpose (raw `requests`,
no project imports) so it can be copied out and run anywhere.

See oa_hw_fetch.py for the same dump against OpenAlex — run both on one DOI to
compare coverage before choosing a source for stage 1.

Usage:
    python -m epai.science_data.fetch.s2_hw_fetch
    python -m epai.science_data.fetch.s2_hw_fetch <doi>

Example:
    python -m epai.science_data.fetch.s2_hw_fetch 10.1038/s41562-025-02244-z
"""

import os
import sys
import time
import textwrap
from collections import defaultdict
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.semanticscholar.org/graph/v1"

api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
headers = {"x-api-key": api_key} if api_key else {}

PAPER_FIELDS = ",".join(
    [
        "title",
        "year",
        "publicationDate",
        "abstract",
        "authors",
        "venue",
        "publicationVenue",
        "publicationTypes",
        "citationCount",
        "influentialCitationCount",
        "referenceCount",
        "isOpenAccess",
        "openAccessPdf",
        "fieldsOfStudy",
        "s2FieldsOfStudy",
        "externalIds",
        "tldr",
        "embedding",
    ]
)


# ── helpers ───────────────────────────────────────────────────────────────────


def get(url, params, retries=5):
    for attempt in range(retries):
        r = requests.get(url, params=params, headers=headers)
        if r.status_code == 429:
            wait = 2**attempt
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        time.sleep(1)
        return r.json()
    raise RuntimeError("Too many retries")


def sep(char="─", width=72):
    print(char * width)


def section(title):
    sep()
    print(f"  {title}")
    sep()


def wrap(text, indent=4, width=72):
    prefix = " " * indent
    return textwrap.fill(
        text, width=width, initial_indent=prefix, subsequent_indent=prefix
    )


# ── fetchers ──────────────────────────────────────────────────────────────────


def get_paper(doi):
    """Fetch paper by DOI using S2's DOI:<doi> identifier syntax."""
    return get(f"{BASE_URL}/paper/DOI:{doi}", {"fields": PAPER_FIELDS})


def get_citations(doi, limit=100):
    data = get(
        f"{BASE_URL}/paper/DOI:{doi}/citations",
        {"limit": limit, "fields": "title,year,citationCount,intents,isInfluential"},
    )
    return data.get("data", [])


def get_references(doi, limit=100):
    data = get(
        f"{BASE_URL}/paper/DOI:{doi}/references",
        {"limit": limit, "fields": "title,year,citationCount,intents,isInfluential"},
    )
    return data.get("data", [])


# ── pretty printer ────────────────────────────────────────────────────────────


def print_paper(doi):
    print()
    section(f"Fetching paper by DOI: {doi}")

    p = get_paper(doi)

    # ── Identity ──────────────────────────────────────────────────────────────
    section("IDENTITY")
    print(f"  Title       : {p.get('title', 'N/A')}")
    print(f"  Year        : {p.get('year', 'N/A')}")
    print(f"  Published   : {p.get('publicationDate', 'N/A')}")

    types = p.get("publicationTypes") or []
    print(f"  Pub. Types  : {', '.join(types) or 'N/A'}")

    venue = p.get("venue") or ""
    pub_venue = p.get("publicationVenue") or {}
    venue_name = pub_venue.get("name") or venue or "N/A"
    venue_type = pub_venue.get("type", "")
    print(f"  Venue       : {venue_name}" + (f"  [{venue_type}]" if venue_type else ""))

    # ── Authors ───────────────────────────────────────────────────────────────
    section("AUTHORS")
    for a in p.get("authors", []):
        affs = a.get("affiliations") or []
        aff_str = f"  ({', '.join(affs)})" if affs else ""
        print(f"  • {a.get('name', '?')}{aff_str}  [id: {a.get('authorId', '?')}]")

    # ── External IDs ──────────────────────────────────────────────────────────
    section("EXTERNAL IDs")
    ext = p.get("externalIds") or {}
    if ext:
        for k, v in ext.items():
            print(f"  {k:<12}: {v}")
    else:
        print("  None")

    open_access = p.get("isOpenAccess", False)
    pdf = (p.get("openAccessPdf") or {}).get("url", "")
    print(
        f"  Open Access : {'Yes' if open_access else 'No'}"
        + (f"  →  {pdf}" if pdf else "")
    )

    # ── Fields of Study ───────────────────────────────────────────────────────
    section("FIELDS OF STUDY")
    fos = p.get("fieldsOfStudy") or []
    print(f"  MAG         : {', '.join(fos) or 'N/A'}")

    s2fos = p.get("s2FieldsOfStudy") or []
    if s2fos:
        for f in s2fos:
            src = f.get("source", "?")
            cat = f.get("category", "?")
            print(f"  S2  [{src:<9}]: {cat}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    section("METRICS")
    total_citations = p.get("citationCount", 0) or 0
    influential = p.get("influentialCitationCount", 0) or 0
    year = p.get("year") or 0
    age = max(2026 - year, 1) if year else 1

    print(f"  Citations             : {total_citations:>8,}")
    print(f"  Influential citations : {influential:>8,}")
    print(f"  References            : {p.get('referenceCount', 0):>8,}")
    print(
        f"  Influence rate        : {influential / max(total_citations, 1) * 100:>7.1f}%"
    )
    print(f"  Citations / year      : {total_citations / age:>8.1f}")

    # ── TLDR ──────────────────────────────────────────────────────────────────
    tldr = p.get("tldr") or {}
    if tldr.get("text"):
        section("TLDR  (auto-generated)")
        print(wrap(tldr["text"]))

    # ── Abstract ──────────────────────────────────────────────────────────────
    section("ABSTRACT")
    abstract = p.get("abstract") or "N/A"
    print(wrap(abstract))

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_obj = p.get("embedding") or {}
    section("EMBEDDING")
    if embedding_obj:
        model = embedding_obj.get("model", "unknown model")
        vector = embedding_obj.get("vector", [])
        dim = len(vector)
        preview = ", ".join(f"{v:.5f}" for v in vector[:8])
        print(f"  Model     : {model}")
        print(f"  Dimension : {dim}")
        print(f"  Preview   : [{preview}, ...]")
        norm = sum(x * x for x in vector) ** 0.5
        print(f"  L2 norm   : {norm:.6f}")
    else:
        print(
            "  Not available (may require an API key or field not supported for this paper)"
        )

    # ── Citing papers ─────────────────────────────────────────────────────────
    section("CITING PAPERS  (top 5 per intent category)")
    citations = get_citations(doi)
    if citations:
        by_intent = defaultdict(list)
        for entry in citations:
            intents = entry.get("intents") or []
            if intents:
                for intent in intents:
                    by_intent[intent].append(entry)
            else:
                by_intent["no intent"].append(entry)

        for intent_label, entries in sorted(by_intent.items()):
            top5 = sorted(
                entries,
                key=lambda e: (e.get("citingPaper") or {}).get("citationCount") or 0,
                reverse=True,
            )[:5]
            print(f"\n  [{intent_label.upper()}]  ({len(entries)} total)")
            for entry in top5:
                cp = entry.get("citingPaper", {})
                star = "⭐" if entry.get("isInfluential") else "  "
                print(f"    {star} [{cp.get('year', '?')}] {cp.get('title', '?')}")
                print(f"         {cp.get('citationCount') or 0:,} citations")
    else:
        print("  None found")

    # ── References ────────────────────────────────────────────────────────────
    section("REFERENCES  (top 5 per intent category)")
    references = get_references(doi)
    if references:
        by_intent = defaultdict(list)
        for entry in references:
            intents = entry.get("intents") or []
            if intents:
                for intent in intents:
                    by_intent[intent].append(entry)
            else:
                by_intent["no intent"].append(entry)

        for intent_label, entries in sorted(by_intent.items()):
            top5 = sorted(
                entries,
                key=lambda e: (e.get("citedPaper") or {}).get("citationCount") or 0,
                reverse=True,
            )[:5]
            print(f"\n  [{intent_label.upper()}]  ({len(entries)} total)")
            for entry in top5:
                rp = entry.get("citedPaper", {})
                star = "⭐" if entry.get("isInfluential") else "  "
                print(f"    {star} [{rp.get('year', '?')}] {rp.get('title', '?')}")
                print(f"         {rp.get('citationCount') or 0:,} citations")
    else:
        print("  None found")

    # ── Citation network quality metrics ──────────────────────────────────────
    influential_citations = [e for e in citations if e.get("isInfluential")]
    if citations:
        infl_pct = len(influential_citations) / len(citations) * 100
        avg_cites_of_citers = sum(
            (e.get("citingPaper") or {}).get("citationCount") or 0 for e in citations
        ) / len(citations)
        methodology_count = sum(
            1 for e in citations if "methodology" in (e.get("intents") or [])
        )
        section("CITATION NETWORK QUALITY")
        print(f"  Influential citers % : {infl_pct:>7.1f}%")
        print(f"  Avg citer citations  : {avg_cites_of_citers:>8.1f}")
        print(f"  Methodology citations: {methodology_count:>8,}")

    # ── Top 5 influential citations ───────────────────────────────────────────
    section("TOP 5 INFLUENTIAL CITING PAPERS")
    if influential_citations:
        top5 = sorted(
            influential_citations,
            key=lambda e: (e.get("citingPaper") or {}).get("citationCount") or 0,
            reverse=True,
        )[:5]
        for entry in top5:
            cp = entry.get("citingPaper") or {}
            print(f"  ⭐ [{cp.get('year', '?')}] {cp.get('title', '?')}")
            print(f"       {cp.get('citationCount') or 0:,} citations")
    else:
        print("  None found")

    # ── Top 5 influential references ──────────────────────────────────────────
    section("TOP 5 INFLUENTIAL REFERENCES")
    influential_references = [e for e in (references or []) if e.get("isInfluential")]
    if influential_references:
        top5 = sorted(
            influential_references,
            key=lambda e: (e.get("citedPaper") or {}).get("citationCount") or 0,
            reverse=True,
        )[:5]
        for entry in top5:
            rp = entry.get("citedPaper") or {}
            print(f"  ⭐ [{rp.get('year', '?')}] {rp.get('title', '?')}")
            print(f"       {rp.get('citationCount') or 0:,} citations")
    else:
        print("  None found")

    sep("═")
    print()


if __name__ == "__main__":
    doi = sys.argv[1] if len(sys.argv) > 1 else "10.1038/s41562-025-02244-z"
    print_paper(doi)