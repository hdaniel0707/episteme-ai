"""Row-local admission rules shared by the fetch and filter stages.

Every rule in this module decides on a *single* record, using only that
record's own fields. That is a deliberate constraint, and it is what lets the
fetch stage guarantee an exact post-filter target: because admissibility never
depends on the rest of the corpus, the fetcher can test each incoming record as
it arrives and stop the moment it has kept enough of them. No overshoot, no
second scrape.

Two consequences worth keeping in mind if you add a rule:

  - No corpus-relative thresholds. "Drop the shortest 5%" cannot be evaluated
    on a record in isolation, and adding such a rule would silently break the
    exactness guarantee. Use absolute bounds instead (see ``FilterConfig``).
  - Deduplication is the one non-row-local check, and it is made row-local by
    threading a caller-owned ``SeenIndex`` of already-claimed ids and titles.

The filter stage re-runs the identical rules over the raw parquet, so the two
stages cannot drift: there is one implementation, imported twice.

Cleaning happens *before* judging. ``clean_abstract`` strips publisher
boilerplate, HTML, and structured-abstract labels, and the length/language
rules then apply to the cleaned text — which is also the text that gets stored.
Stripping the structured labels ("Background: ... Methods: ...") matters more
than it looks: those labels appear only in the human class, so leaving them in
would hand a downstream classifier a formatting tell that has nothing to do
with authorship.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class FilterConfig:
    """Absolute, corpus-independent thresholds for the admission rules."""

    year_min: int = 1900
    # Pre-ChatGPT cutoff: abstracts published after this may themselves be
    # LLM-written or LLM-polished, which would poison the "human" class.
    year_max: int = 2020

    min_words: int = 80
    max_words: int = 400

    # Titles that are a bare word or two are usually parse errors upstream.
    min_title_words: int = 3

    # A record with no citationCount field at all is dropped; a genuine zero is
    # kept. Dropping zeros would bias the corpus toward well-cited work, which
    # correlates with venue prestige and writing polish.
    require_citation_count: bool = True

    # English heuristic (no language-detection dependency in this project).
    min_english_stopwords: int = 4
    max_non_latin_ratio: float = 0.10

    @classmethod
    def from_dict(cls, d: dict | None) -> "FilterConfig":
        d = d or {}
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        unknown = set(d) - set(valid)
        if unknown:
            raise ValueError(f"Unknown filter config keys: {sorted(unknown)}")
        return cls(**valid)


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")

# Leading "Abstract" / "ABSTRACT" / "Abstract." / "Abstract:" heading.
_LEADING_ABSTRACT_RE = re.compile(r"^\s*abstract\b[\s:.\-—]*", re.IGNORECASE)

# Structured-abstract section labels, e.g. "Background:", "MATERIALS AND
# METHODS:". Matched at a sentence/segment boundary and removed, keeping the
# prose that follows.
_STRUCTURED_LABELS = (
    "background",
    "objective",
    "objectives",
    "aim",
    "aims",
    "purpose",
    "introduction",
    "context",
    "methods",
    "method",
    "materials and methods",
    "patients and methods",
    "design",
    "setting",
    "participants",
    "intervention",
    "interventions",
    "measurements",
    "main outcome measures",
    "results",
    "findings",
    "discussion",
    "conclusion",
    "conclusions",
    "significance",
    "implications",
    "keywords",
)
_STRUCTURED_RE = re.compile(
    r"(?:(?<=^)|(?<=[.!?\s]))(?:" + "|".join(_STRUCTURED_LABELS) + r")\s*:\s*",
    re.IGNORECASE,
)

# Publisher / copyright boilerplate. Everything from the marker to the end of
# the abstract is dropped, since these blocks are always trailing.
_BOILERPLATE_RE = re.compile(
    r"(©|\(c\)\s*\d{4}|copyright\s+\d{4}"
    r"|all rights reserved"
    r"|published by (elsevier|springer|wiley|taylor|sage|oxford)"
    r"|this article is protected by copyright"
    r"|\bwiley periodicals\b"
    r"|\bspringer-verlag\b)",
    re.IGNORECASE,
)

_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z']+")

# Frequent English function words — presence of several is a cheap, dependency
# -free signal that the text is English prose.
_ENGLISH_STOPWORDS = frozenset(
    """the of and to in a is that for we this with are as was by on it be from
    an at which has have not or their its these can our were also been but
    more than when using used such other into between both during each""".split()
)


def is_missing(value) -> bool:
    """True for ``None`` and for NaN.

    Both spellings of "absent" have to be handled, because the fetch stage
    tests records straight from the API (where a missing field is ``None``)
    while the filter stage tests the same records after a parquet round-trip,
    which turns a null in a numeric column into ``NaN``. Since ``NaN is None``
    is false and ``int(NaN)`` raises, checking only for ``None`` would let
    null-citation rows pass the filter after the fetcher had rejected them —
    precisely the fetch/filter drift that sharing this module is meant to rule
    out. The ``value != value`` test identifies NaN without needing numpy.
    """
    return value is None or value != value


def as_text(value) -> str:
    """Coerce a possibly-null field to a string.

    Guards the same round-trip: a null in a string column can arrive as NaN,
    which is *truthy*, so ``(value or "")`` would pass it through unchanged and
    the following ``.strip()`` would raise.
    """
    return "" if is_missing(value) else str(value)


def clean_abstract(raw: str | None) -> tuple[str, list[str]]:
    """Normalise an S2 abstract; return ``(cleaned_text, applied_fixes)``.

    ``applied_fixes`` names each transformation that actually changed the text,
    so the filter analysis can report how dirty the raw corpus was.
    """
    text = as_text(raw)
    if not text:
        return "", []

    fixes: list[str] = []

    unescaped = html.unescape(text)
    if unescaped != text:
        fixes.append("html_entities")
        text = unescaped

    stripped_tags = _TAG_RE.sub(" ", text)
    if stripped_tags != text:
        fixes.append("html_tags")
        text = stripped_tags

    cut = _BOILERPLATE_RE.search(text)
    if cut is not None:
        text = text[: cut.start()]
        fixes.append("publisher_boilerplate")

    without_heading = _LEADING_ABSTRACT_RE.sub("", text)
    if without_heading != text:
        fixes.append("abstract_heading")
        text = without_heading

    without_labels = _STRUCTURED_RE.sub("", text)
    if without_labels != text:
        fixes.append("structured_labels")
        text = without_labels

    collapsed = _WS_RE.sub(" ", text).strip()
    if collapsed != text:
        fixes.append("whitespace")
    return collapsed, fixes


def normalise_title(title: str | None) -> str:
    """Fold a title to a dedupe key.

    Preprint and published versions of the same paper carry different
    ``paperId``s but near-identical titles, so id-only deduplication is not
    enough. Accents are folded, punctuation dropped, whitespace collapsed.
    """
    title = as_text(title)
    if not title:
        return ""
    folded = unicodedata.normalize("NFKD", title)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.lower()
    folded = re.sub(r"[^a-z0-9\s]", " ", folded)
    return _WS_RE.sub(" ", folded).strip()


def looks_english(text: str, cfg: FilterConfig) -> bool:
    """Heuristic English check (no language-detection dependency available).

    Two signals: enough distinct English function words, and few non-Latin
    characters. Deliberately permissive — it is aimed at obviously non-English
    abstracts, not at borderline cases.
    """
    if not text:
        return False

    letters = [c for c in text if c.isalpha()]
    if letters:
        non_latin = sum(1 for c in letters if ord(c) > 0x24F)
        if non_latin / len(letters) > cfg.max_non_latin_ratio:
            return False

    words = set(_WORD_RE.findall(text.lower()))
    return len(words & _ENGLISH_STOPWORDS) >= cfg.min_english_stopwords


# --------------------------------------------------------------------------
# Deduplication state
# --------------------------------------------------------------------------


class SeenIndex:
    """Caller-owned set of already-claimed paper ids and normalised titles.

    Shared across *all* domains on purpose. A paper matching queries in two
    domains would otherwise land in two per-domain datasets, and the same text
    appearing under two domain labels leaks across LODO/LOGO folds. First
    domain wins (domains are processed in config order, so the outcome is
    deterministic) and the loser is recorded on the winning row's
    ``also_matched_domains`` for auditing.
    """

    def __init__(self) -> None:
        self.by_paper_id: dict[str, int] = {}
        self.by_title: dict[str, int] = {}

    def lookup(self, paper_id: str, norm_title: str) -> int | None:
        """Return the row index that already claimed this paper, if any."""
        if paper_id and paper_id in self.by_paper_id:
            return self.by_paper_id[paper_id]
        if norm_title and norm_title in self.by_title:
            return self.by_title[norm_title]
        return None

    def claim(self, paper_id: str, norm_title: str, row_index: int) -> None:
        if paper_id:
            self.by_paper_id[paper_id] = row_index
        if norm_title:
            self.by_title[norm_title] = row_index


# --------------------------------------------------------------------------
# The admission test
# --------------------------------------------------------------------------

# Every reason `check_record` can return, in report order.
DROP_REASONS = (
    "missing_paper_id",
    "missing_title",
    "short_title",
    "missing_abstract",
    "missing_year",
    "year_out_of_range",
    "missing_citation_count",
    "truncated_abstract",
    "abstract_too_short",
    "abstract_too_long",
    "not_english",
    "duplicate",
)


def check_record(record: dict, cfg: FilterConfig) -> str | None:
    """Return a drop reason for *record*, or ``None`` if it is admissible.

    Expects a cleaned record as produced by :func:`normalise_record` (i.e. the
    abstract already run through :func:`clean_abstract`). Duplicates are *not*
    checked here — that needs the caller's :class:`SeenIndex`.
    """
    if not as_text(record.get("paper_id")):
        return "missing_paper_id"

    title = as_text(record.get("title")).strip()
    if not title:
        return "missing_title"
    if len(title.split()) < cfg.min_title_words:
        return "short_title"

    abstract = as_text(record.get("abstract"))
    if not abstract:
        return "missing_abstract"

    year = record.get("year")
    if is_missing(year):
        return "missing_year"
    if not (cfg.year_min <= int(year) <= cfg.year_max):
        return "year_out_of_range"

    if cfg.require_citation_count and is_missing(record.get("citation_count")):
        return "missing_citation_count"

    # S2 truncates some abstracts mid-sentence; an ellipsis terminator is the
    # reliable marker. Missing terminal punctuation alone is too noisy to use.
    if abstract.endswith("…") or abstract.endswith("..."):
        return "truncated_abstract"

    n_words = len(abstract.split())
    if n_words < cfg.min_words:
        return "abstract_too_short"
    if n_words > cfg.max_words:
        return "abstract_too_long"

    if not looks_english(abstract, cfg):
        return "not_english"

    return None


def normalise_record(paper: dict, domain: str, query: str) -> dict:
    """Flatten one S2 API paper into the raw-corpus row shape."""
    cleaned, fixes = clean_abstract(paper.get("abstract"))
    title = (paper.get("title") or "").strip()
    return {
        "paper_id": paper.get("paperId") or "",
        "title": title,
        "abstract": cleaned,
        "abstract_raw": paper.get("abstract") or "",
        "cleaning_fixes": ",".join(fixes),
        "year": paper.get("year"),
        "venue": paper.get("venue") or "",
        "citation_count": paper.get("citationCount"),
        "influential_citation_count": paper.get("influentialCitationCount"),
        "fields_of_study": ",".join(paper.get("fieldsOfStudy") or []),
        "domain": domain,
        "query": query,
        "also_matched_domains": "",
    }


# Fields requested from the S2 API. Kept next to `normalise_record` so the two
# stay in step.
PAPER_FIELDS = (
    "paperId,title,abstract,year,venue,"
    "citationCount,influentialCitationCount,fieldsOfStudy"
)
