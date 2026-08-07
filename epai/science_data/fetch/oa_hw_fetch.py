"""
Hello-world script for the OpenAlex API — lookup by DOI.

Dumps everything OpenAlex holds about one paper, sectioned, so you can see what
is available before deciding what oa_fetch.py should store. Standalone on
purpose (raw `requests`, no project imports) so it can be copied out and run
anywhere.

Credentials, both optional and independent:
    OPENALEX_MAILTO   a contact address that selects the "polite pool", a
                      faster and more consistently served queue. Not a secret.
    OPENALEX_API_KEY  an OpenAlex Premium key, sent as `api_key`. A real
                      secret, and used automatically when set. The free tier
                      returns the same work with the same fields, so every
                      section below prints identically without it.

Well beyond what stage 1 keeps: the full topic hierarchy (domain → field →
subfield → topic) with scores, SDG mappings, keywords, per-year citation
counts, FWCI, institution affiliations with ROR ids, grants, open-access
routing, and the citing/referenced/related networks. What oa_fetch.py actually
stores is the short list in `rules.OA_SELECT`; everything else printed here is
one `select=` edit away.

Usage:
    uv run python epai/science_data/fetch/oa_hw_fetch.py
    uv run python epai/science_data/fetch/oa_hw_fetch.py <doi>

Example:
    uv run python epai/science_data/fetch/oa_hw_fetch.py 10.1038/s41562-025-02244-z
"""

import os
import sys
import time
import textwrap
from collections import defaultdict

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.openalex.org"

mailto = os.getenv("OPENALEX_MAILTO")
api_key = os.getenv("OPENALEX_API_KEY")

headers = {
    "Accept": "application/json",
    # `mailto` is documented as a query parameter; this is a belt-and-braces
    # second copy. The key deliberately does NOT go here — the User-Agent shows
    # up in server logs and error messages.
    "User-Agent": "episteme-ai-hello-world/0.1"
    + (f" mailto:{mailto}" if mailto else ""),
}

common_params = {}
if mailto:
    common_params["mailto"] = mailto
if api_key:
    common_params["api_key"] = api_key

# `select` trims the response to the fields this script prints. Without it a
# single work carries every authorship, location and yearly count OpenAlex
# holds, which is a lot of bytes to download and discard.
WORK_FIELDS = ",".join(
    [
        "id",
        "doi",
        "ids",
        "display_name",
        "title",
        "publication_year",
        "publication_date",
        "type",
        "language",
        "authorships",
        "primary_location",
        "locations_count",
        "open_access",
        "abstract_inverted_index",
        "cited_by_count",
        "counts_by_year",
        "fwci",
        "citation_normalized_percentile",
        "referenced_works",
        "referenced_works_count",
        "related_works",
        "primary_topic",
        "topics",
        "keywords",
        "sustainable_development_goals",
        # `awards` + `funders`, not `grants`: OpenAlex retired the latter and
        # rejects it outright. Worth knowing that `select` is validated
        # strictly and the 400 body lists every accepted field — that error is
        # the fastest way to check this list against the live API.
        "awards",
        "funders",
        "is_retracted",
        "is_paratext",
    ]
)

# Lighter projection for the citing/cited lists, where only a one-line summary
# of each neighbour gets printed.
NEIGHBOUR_FIELDS = "id,doi,display_name,publication_year,cited_by_count,type,primary_location"

# OpenAlex accepts at most 50 values in one `|`-joined OR filter.
OR_FILTER_MAX = 50


# ── helpers ───────────────────────────────────────────────────────────────────


def get(url, params=None, retries=5):
    """GET with backoff on 429/5xx.

    OpenAlex's limits (10/s, 100k/day) are real limits rather than load
    shedding, so a 429 means we genuinely went too fast and backing off
    actually clears it. Hence long waits and a small retry budget.
    """
    params = {**common_params, **(params or {})}
    for attempt in range(retries):
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 429 or r.status_code in (500, 502, 503, 504):
            wait = 2**attempt
            print(f"  HTTP {r.status_code}, waiting {wait}s...")
            time.sleep(wait)
            continue
        if r.status_code in (400, 401, 403):
            # OpenAlex explains itself in the body; the bare status does not.
            try:
                detail = r.json().get("message", r.text[:200])
            except Exception:
                detail = r.text[:200]
            hint = ""
            if r.status_code in (401, 403) and api_key:
                hint = ("\n  OPENALEX_API_KEY was rejected. Unset it to run on "
                        "the free tier — the output is the same.")
            raise RuntimeError(f"HTTP {r.status_code} from OpenAlex: {detail}{hint}")
        r.raise_for_status()
        time.sleep(0.2)
        return r.json()
    raise RuntimeError("Too many retries")


def short_id(value):
    """`https://openalex.org/W2741809807` → `W2741809807`."""
    return str(value or "").rsplit("/", 1)[-1]


def invert_abstract(inverted_index):
    """Rebuild plain text from OpenAlex's `abstract_inverted_index`.

    OpenAlex ships abstracts as `{word: [positions]}` rather than prose, which
    reportedly sidesteps the redistribution terms on the abstracts it
    aggregates. Word order is recovered exactly; the original whitespace is not
    recorded, so tokens are rejoined with single spaces.
    """
    if not inverted_index:
        return ""
    positions = [
        (pos, word)
        for word, where in inverted_index.items()
        for pos in (where or [])
    ]
    positions.sort(key=lambda p: p[0])
    return " ".join(word for _, word in positions)


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


def venue_of(work):
    """Source name for a work, or 'N/A'."""
    source = (work.get("primary_location") or {}).get("source") or {}
    return source.get("display_name") or "N/A"


def describe(entry, indent=4):
    """One-line summary of a dict whose exact shape is not guaranteed.

    Used for `funders` and `awards`, which replaced the retired `grants` field.
    Nothing downstream reads either, and the naming here has already moved once,
    so this prints whatever scalar keys actually arrive rather than hard-coding
    names that would turn the next rename into a KeyError. Nested values are
    summarised by type — a hello-world should show that they exist without
    dumping a page of JSON.
    """
    prefix = " " * indent
    if not isinstance(entry, dict):
        print(f"{prefix}• {entry}")
        return
    label = (
        entry.get("display_name")
        or entry.get("award_id")
        or short_id(entry.get("id"))
        or "?"
    )
    scalars = ", ".join(
        f"{k}={v}"
        for k, v in entry.items()
        if k not in {"display_name", "id", "award_id"}
        and not isinstance(v, (list, dict))
    )
    nested = ", ".join(
        f"{k}[{len(v)}]" for k, v in entry.items() if isinstance(v, (list, dict))
    )
    detail = "  ".join(p for p in (scalars, nested) if p)
    print(f"{prefix}• {label}" + (f"  ({detail})" if detail else ""))


# ── fetchers ──────────────────────────────────────────────────────────────────


def get_work(doi):
    """Fetch a work by DOI using OpenAlex's `doi:` identifier syntax.

    OpenAlex resolves DOIs, PMIDs, MAG ids and its own Wxxxx ids on this same
    path, so swapping the identifier type needs no other change.
    """
    return get(f"{BASE_URL}/works/doi:{doi}", {"select": WORK_FIELDS})


def get_citing(work_id, limit=100):
    """Works that cite *work_id*, most-cited first.

    `cites:` is a filter over the whole /works index rather than a dedicated
    endpoint, which is why it takes the same sort and select parameters as any
    other search.
    """
    data = get(
        f"{BASE_URL}/works",
        {
            "filter": f"cites:{short_id(work_id)}",
            "select": NEIGHBOUR_FIELDS,
            "sort": "cited_by_count:desc",
            "per-page": min(limit, 200),
        },
    )
    return data.get("results", [])


def get_referenced(referenced_ids, limit=100):
    """Hydrate `referenced_works` ids into full records.

    The work object lists references as bare ids only, so each has to be looked
    up. Batching them into `|`-joined OR filters turns 100 lookups into two
    requests.
    """
    ids = [short_id(i) for i in (referenced_ids or [])][:limit]
    out = []
    for start in range(0, len(ids), OR_FILTER_MAX):
        chunk = ids[start : start + OR_FILTER_MAX]
        data = get(
            f"{BASE_URL}/works",
            {
                "filter": f"openalex:{'|'.join(chunk)}",
                "select": NEIGHBOUR_FIELDS,
                "per-page": OR_FILTER_MAX,
            },
        )
        out.extend(data.get("results", []))
    return out


# ── pretty printer ────────────────────────────────────────────────────────────


def print_work(doi):
    print()
    section(f"Fetching work by DOI: {doi}")
    # Report what is being *sent*, not which tier we think we are on. Nothing
    # in a successful response says whether the key was recognised, so any
    # claim beyond "we sent it" would be a guess dressed up as a status line.
    sent = []
    sent.append("api_key" if api_key else "no api_key")
    sent.append("mailto (polite pool)" if mailto else "no mailto")
    print(f"  Sending: {', '.join(sent)}")
    if api_key:
        print("  A rejected key returns 403 and is reported below; the free tier"
              " returns the same fields either way.")

    w = get_work(doi)

    # ── Identity ──────────────────────────────────────────────────────────────
    section("IDENTITY")
    print(f"  Title       : {w.get('display_name') or w.get('title') or 'N/A'}")
    print(f"  Year        : {w.get('publication_year', 'N/A')}")
    print(f"  Published   : {w.get('publication_date', 'N/A')}")
    print(f"  Type        : {w.get('type', 'N/A')}")
    print(f"  Language    : {w.get('language', 'N/A')}")

    location = w.get("primary_location") or {}
    source = location.get("source") or {}
    venue_type = source.get("type", "")
    print(f"  Venue       : {venue_of(w)}" + (f"  [{venue_type}]" if venue_type else ""))
    if source.get("host_organization_name"):
        print(f"  Publisher   : {source['host_organization_name']}")
    print(f"  Locations   : {w.get('locations_count', 0)} (versions/repositories)")

    flags = []
    if w.get("is_retracted"):
        flags.append("RETRACTED")
    if w.get("is_paratext"):
        flags.append("paratext (front matter, not a research article)")
    if flags:
        print(f"  Flags       : {', '.join(flags)}")

    # ── Authors ───────────────────────────────────────────────────────────────
    section("AUTHORS")
    authorships = w.get("authorships") or []
    for a in authorships:
        author = a.get("author") or {}
        institutions = [
            i.get("display_name", "?") for i in (a.get("institutions") or [])
        ]
        aff_str = f"  ({', '.join(institutions)})" if institutions else ""
        star = "✉ " if a.get("is_corresponding") else "  "
        print(
            f"  {star}• {author.get('display_name', '?')}{aff_str}"
            f"  [id: {short_id(author.get('id'))}]"
        )
        if author.get("orcid"):
            print(f"       ORCID: {author['orcid']}")
    if not authorships:
        print("  None")

    # Institution roll-up: OpenAlex resolves raw affiliation strings to ROR
    # ids and country codes, so this is a real roll-up, not string matching.
    countries = defaultdict(int)
    for a in authorships:
        for i in a.get("institutions") or []:
            if i.get("country_code"):
                countries[i["country_code"]] += 1
    if countries:
        listed = ", ".join(
            f"{c} ({n})" for c, n in sorted(countries.items(), key=lambda kv: -kv[1])
        )
        print(f"\n  Countries   : {listed}")

    # ── External IDs ──────────────────────────────────────────────────────────
    section("EXTERNAL IDs")
    ids = w.get("ids") or {}
    if ids:
        for k, v in ids.items():
            print(f"  {k:<12}: {v}")
    else:
        print("  None")

    # ── Open access ───────────────────────────────────────────────────────────
    section("OPEN ACCESS")
    oa = w.get("open_access") or {}
    is_oa = oa.get("is_oa", False)
    print(f"  Open Access : {'Yes' if is_oa else 'No'}")
    # oa_status is the "colour" taxonomy: gold/green/hybrid/bronze/closed.
    print(f"  OA status   : {oa.get('oa_status', 'N/A')}")
    if oa.get("oa_url"):
        print(f"  OA URL      : {oa['oa_url']}")
    print(f"  In repository: {'Yes' if oa.get('any_repository_has_fulltext') else 'No'}")

    # ── Topics ────────────────────────────────────────────────────────────────
    section("TOPICS  (OpenAlex hierarchy: domain → field → subfield → topic)")
    primary = w.get("primary_topic") or {}
    if primary:
        print(f"  Primary     : {primary.get('display_name', '?')}"
              f"  (score {primary.get('score', 0):.3f})")
        for level in ("domain", "field", "subfield"):
            node = primary.get(level) or {}
            print(f"    {level:<9}: {node.get('display_name', 'N/A')}")
    topics = w.get("topics") or []
    if len(topics) > 1:
        print("\n  All topics  :")
        for t in topics:
            field = (t.get("field") or {}).get("display_name", "?")
            print(f"    • {t.get('display_name', '?')} "
                  f"[{field}]  score {t.get('score', 0):.3f}")
    if not topics and not primary:
        print("  None")

    keywords = w.get("keywords") or []
    if keywords:
        print("\n  Keywords    : "
              + ", ".join(k.get("display_name", "?") for k in keywords))

    sdgs = w.get("sustainable_development_goals") or []
    if sdgs:
        print("\n  UN SDGs     :")
        for s in sdgs:
            print(f"    • {s.get('display_name', '?')}  (score {s.get('score', 0):.3f})")

    # ── Metrics ───────────────────────────────────────────────────────────────
    section("METRICS")
    total_citations = w.get("cited_by_count", 0) or 0
    year = w.get("publication_year") or 0
    age = max(2026 - year, 1) if year else 1

    print(f"  Citations             : {total_citations:>8,}")
    print(f"  References            : {w.get('referenced_works_count', 0):>8,}")
    print(f"  Citations / year      : {total_citations / age:>8.1f}")

    # FWCI: citations relative to what similar-age, same-field work receives.
    # 1.0 is exactly average, which is what makes it comparable across domains
    # and years where the raw count is not. Stored by oa_fetch.py as its own
    # `fwci` column. Null for very recent works and for works OpenAlex cannot
    # place in a field.
    fwci = w.get("fwci")
    if fwci is not None:
        print(f"  FWCI                  : {fwci:>8.2f}   (1.00 = field average)")
    else:
        print("  FWCI                  :      N/A   (too recent, or no field)")
    percentile = (w.get("citation_normalized_percentile") or {}).get("value")
    if percentile is not None:
        print(f"  Citation percentile   : {percentile * 100:>7.1f}%")

    counts = w.get("counts_by_year") or []
    if counts:
        print("\n  Citations by year:")
        peak = max((c.get("cited_by_count", 0) for c in counts), default=0) or 1
        for c in sorted(counts, key=lambda c: c.get("year", 0)):
            n = c.get("cited_by_count", 0)
            bar = "█" * max(1, round(n / peak * 40)) if n else ""
            print(f"    {c.get('year', '?')}  {n:>6,}  {bar}")

    # ── Funding ───────────────────────────────────────────────────────────────
    funders = w.get("funders") or []
    awards = w.get("awards") or []
    if funders or awards:
        section("FUNDING")
        if funders:
            print("  Funders:")
            for f in funders:
                describe(f)
        if awards:
            print("  Awards:")
            for a in awards:
                describe(a)

    # ── Abstract ──────────────────────────────────────────────────────────────
    section("ABSTRACT  (reconstructed from the inverted index)")
    abstract = invert_abstract(w.get("abstract_inverted_index"))
    if abstract:
        print(wrap(abstract))
        print(f"\n  ({len(abstract.split()):,} words)")
    else:
        # Common enough to call out: some publishers (Elsevier most visibly)
        # are excluded from OpenAlex's abstract coverage.
        print("  N/A — OpenAlex has no abstract for this work.")
        print("  Filter with has_abstract:true to skip these during a bulk fetch.")

    # ── Citing works ──────────────────────────────────────────────────────────
    section("CITING WORKS  (top 10 by citation count)")
    citing = get_citing(w["id"])
    if citing:
        for c in citing[:10]:
            print(f"    [{c.get('publication_year', '?')}] {c.get('display_name', '?')}")
            print(f"         {c.get('cited_by_count') or 0:,} citations  ·  {venue_of(c)}")
    else:
        print("  None found")

    # ── References ────────────────────────────────────────────────────────────
    section("REFERENCES  (top 10 by citation count)")
    references = get_referenced(w.get("referenced_works"))
    if references:
        top = sorted(references, key=lambda r: r.get("cited_by_count") or 0, reverse=True)
        for r in top[:10]:
            print(f"    [{r.get('publication_year', '?')}] {r.get('display_name', '?')}")
            print(f"         {r.get('cited_by_count') or 0:,} citations  ·  {venue_of(r)}")
    else:
        print("  None found")

    # ── Citation network shape ────────────────────────────────────────────────
    # Summarised by year and venue. OpenAlex records no citation *intent*
    # (background/methodology/result), so a citation is a citation here.
    if citing:
        section("CITATION NETWORK")
        by_year = defaultdict(int)
        for c in citing:
            by_year[c.get("publication_year") or 0] += 1
        avg_cites_of_citers = sum(
            c.get("cited_by_count") or 0 for c in citing
        ) / len(citing)
        print(f"  Citing works sampled  : {len(citing):>8,}")
        print(f"  Avg citer citations   : {avg_cites_of_citers:>8.1f}")
        print(f"  Distinct citing venues: {len({venue_of(c) for c in citing}):>8,}")
        print("\n  Citing works by year:")
        for y in sorted(by_year):
            print(f"    {y or '?'}  {by_year[y]:>4,}")

    # ── Related works ─────────────────────────────────────────────────────────
    section("RELATED WORKS  (OpenAlex topic-similarity, not citation-based)")
    related = w.get("related_works") or []
    if related:
        for rid in related[:5]:
            print(f"  • {short_id(rid)}  →  {BASE_URL}/works/{short_id(rid)}")
        print(f"\n  ({len(related)} total; ids only — hydrate them like references above)")
    else:
        print("  None")

    # ── Known coverage gaps ───────────────────────────────────────────────────
    # Worth stating explicitly: each of these is a thing you might reasonably
    # expect a scholarly API to expose, and looking for it in the JSON above
    # turns up nothing rather than an empty field.
    section("NOT AVAILABLE FROM OPENALEX")
    print("  Citation intents      : none (no background/methodology/result split)")
    print("  Influential-citation  : none; FWCI above normalises volume, and says")
    print("                          nothing about whether a citation mattered")
    print("  TLDR summaries        : none")
    print("  Text embeddings       : none — embed the abstract yourself")
    print("  Verbatim abstracts    : reconstructed from an inverted index, so")
    print("                          inter-word whitespace is lost (see ABSTRACT)")

    sep("═")
    print()


if __name__ == "__main__":
    doi = sys.argv[1] if len(sys.argv) > 1 else "10.1038/s41562-025-02244-z"
    print_work(doi)
