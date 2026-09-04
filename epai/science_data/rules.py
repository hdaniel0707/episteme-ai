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
    threading a caller-owned ``SeenIndex`` of already-claimed ids, titles and
    abstracts.

The filter stage re-runs the identical rules over the raw parquet, so the two
stages cannot drift: there is one implementation, imported twice.

Cleaning happens *before* judging. ``clean_abstract`` unescapes entities, drops
HTML tags, cuts publisher boilerplate and collapses whitespace, and the
length/language rules then apply to the cleaned text — which is also the text
that gets stored. The publisher's own text survives beside it in
``abstract_raw``, so nothing here is unrecoverable.

Structured-abstract labels ("Background: ... Methods: ...") are a special case
and are left alone by default. They *are* a formatting tell — they appear only
in the human class, since the prompts forbid the generator from emitting them —
but removing them means deleting words by keyword, which is the one class of
transform this codebase has been bitten by. ``clean.strip_structured_labels``
turns it on, for both classes at once; see :func:`strip_structured_labels`.
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

    # A record with no citation count field at all is dropped; a genuine zero
    # is kept. Dropping zeros would bias the corpus toward well-cited work,
    # which correlates with venue prestige and writing polish.
    require_citation_count: bool = True

    # Same rule, same reasoning, for Field-Weighted Citation Impact: null means
    # OpenAlex could not place the work in a field, so there is no expectation
    # to normalise against. A genuine 0.0 is kept — most works have exactly
    # that, and dropping them would reintroduce the very bias the citation-count
    # rule avoids.
    require_fwci: bool = True

    # The language OpenAlex assigned to the work. Enforced as a rule rather
    # than trusted from the server-side filter, on the same principle as every
    # other threshold here. Empty string disables the check.
    require_language: str = "en"

    # English heuristic over the abstract text itself (no language-detection
    # dependency in this project). Not redundant with `require_language`: that
    # one tests OpenAlex's metadata label, this one tests the prose, and a
    # mislabelled record fails only the second.
    min_english_stopwords: int = 4
    max_non_latin_ratio: float = 0.10

    # Function-word share below which a 40+ word passage is not English prose —
    # a wrong-language abstract that OpenAlex mislabelled "en", or raw LaTeX.
    # Real English abstracts here sit at 0.15+; non-English ones at 0.06 and
    # below, so the threshold has a wide margin either side.
    min_english_fw_ratio: float = 0.12

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

# A scrape that grabbed the publisher's *page* instead of the abstract leaves
# fragments a real abstract never contains: a journal's article-nav strip, a
# database's chrome, a paywall. Matched anywhere, case-folded.
_NON_ABSTRACT_SIGNATURES = (
    "previous article next article", "close ebsco",
    "research databases, ebooks", "discovery service contact",
    "sign in librarians", "pdfbibtexsection", "this website uses cookies",
    "accept all cookies", "skip to main content", "institutional login",
    "purchase access", "add to cart", "javascript is disabled",
    "enable javascript to",
)
# Book front matter — a table of contents or acknowledgements page in place of
# an abstract. Only ever at the very start, so anchored there: a real abstract
# may *mention* a preface, it does not *open* with the word.
_FRONT_MATTER_OPENER = re.compile(
    r"^[\s*.\-–]*(?:"
    r"previous article|next article|"
    r"table of contents|brief table of contents|contents\b|"
    r"list of (?:figures|tables|contributors|illustrations|abbreviations|boxes|plates)|"
    r"acknowledge?ments|notes on contributors|about the (?:author|editor|contributor)s?|"
    r"preface\b|foreword\b"
    r")",
    re.IGNORECASE,
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


def strip_structured_labels(text: str) -> str:
    """Remove structured-abstract section labels, keeping the prose after them.

    OFF by default wherever it is wired in (``clean.strip_structured_labels``).
    This is the only cleaning step in the pipeline that removes text by
    KEYWORD rather than substituting characters, and that class of transform
    has a bad record here: the Ghostbuster fork's ``strip_boilerplate`` deleted
    the opening paragraph of ~9% of generated essays because "ensure" contains
    "sure". So it is opt-in, and the corpus keeps its words unless someone asks
    otherwise.

    The colon is required (see :data:`_STRUCTURED_RE`), and that is what keeps
    this safe when it *is* enabled -- "Background: we study X" is a label,
    "Background radiation was measured" is a sentence, and only the first has
    one. The cost of the strict rule is coverage: an all-caps or
    period-separated label, or one whose colon was carried by an HTML tag that
    ``clean_abstract`` already replaced with a space, is not matched.

    Public and shared on purpose. Both :func:`clean_abstract` (human) and
    ``generate_ai.clean_generated`` (AI) call it, reading one config key, so
    the two classes cannot end up cleaned differently -- a label the human
    class contains and the AI class never does is a token a classifier learns
    instantly, and it says nothing about authorship.
    """
    return _STRUCTURED_RE.sub("", text)


def clean_abstract(
    raw: str | None, *, strip_labels: bool = False
) -> tuple[str, list[str]]:
    """Normalise an abstract; return ``(cleaned_text, applied_fixes)``.

    ``applied_fixes`` names each transformation that actually changed the text,
    so the filter analysis can report how dirty the raw corpus was.

    :param strip_labels: also remove structured-abstract section labels.
        OFF by default, and it is the one transform here that deletes words
        rather than rewriting characters -- see
        :func:`strip_structured_labels`. Whatever this is set to, it must match
        what ``generate_ai.clean_generated`` was given, or the two classes are
        cleaned differently; both read one config key, ``clean
        .strip_structured_labels``, so that they cannot disagree.
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

    if strip_labels:
        without_labels = strip_structured_labels(text)
        if without_labels != text:
            fixes.append("structured_labels")
            text = without_labels

    collapsed = _WS_RE.sub(" ", text).strip()
    if collapsed != text:
        fixes.append("whitespace")
    return collapsed, fixes


def _fold_key(text: str) -> str:
    """Accent-fold, lowercase, drop punctuation, collapse whitespace.

    The shared reduction behind both dedupe keys below: two strings that differ
    only in casing, accents, punctuation or spacing fold to the same key.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.lower()
    folded = re.sub(r"[^a-z0-9\s]", " ", folded)
    return _WS_RE.sub(" ", folded).strip()


def normalise_title(title: str | None) -> str:
    """Fold a title to a dedupe key.

    Preprint and published versions of the same paper carry different
    ``paperId``s but near-identical titles, so id-only deduplication is not
    enough.
    """
    return _fold_key(as_text(title))


def normalise_abstract(abstract: str | None) -> str:
    """Fold a cleaned abstract to a dedupe key.

    The same article deposited twice (preprint plus published, or two
    aggregator feeds) can carry different ``paperId``s *and* titles divergent
    enough to survive :func:`normalise_title` — an added subtitle, a reworded
    translation — while the abstract text is identical bar casing and
    punctuation. Keying on the folded abstract as well catches those; the text
    is the training signal, so a collision here is exactly the leak that
    matters. Expects the *cleaned* abstract, so publisher boilerplate and HTML
    are already gone.
    """
    return _fold_key(as_text(abstract))


def looks_english(text: str, cfg: FilterConfig) -> bool:
    """Heuristic English check (no language-detection dependency available).

    Three signals: few non-Latin characters, enough distinct English function
    words, and — for a passage long enough to measure — a function-word *share*
    in the English range. The share is what catches a French or Indonesian
    abstract that OpenAlex mislabelled ``"en"``: it clears the distinct-word bar
    on chance collisions (``a``, ``on``, ``or``) but its density of English
    function words is a third of real English prose.
    """
    if not text:
        return False

    letters = [c for c in text if c.isalpha()]
    if letters:
        non_latin = sum(1 for c in letters if ord(c) > 0x24F)
        if non_latin / len(letters) > cfg.max_non_latin_ratio:
            return False

    tokens = _WORD_RE.findall(text.lower())
    if len(set(tokens) & _ENGLISH_STOPWORDS) < cfg.min_english_stopwords:
        return False
    if len(tokens) >= 40:
        return _function_word_ratio(text) >= cfg.min_english_fw_ratio
    return True


# Function words of the languages that turn up mislabelled "en" in OpenAlex —
# French, German, Spanish, Italian, Dutch, Norwegian, Indonesian, Czech. Two or
# more in a title, and more of them than English ones, means the title is not
# English. One-character words are left out: they collide with maths notation
# (``$Z_4$`` tokenises to ``z``) far more than they catch anything.
_NON_ENGLISH_FUNCTION_WORDS = frozenset(
    """les des du de un une et ou dans sur pour par aux avec sans cette
    und der die das den ein eine von zu im mit für über auch
    dan yang dalam pada untuk dengan dari ini adalah di ke
    og på av som til med det
    van een het op voor naar tussen zijn
    los las una en con para por del
    il per con che della delle nel
    uma do da ao
    ve se je při což""".split()
)
_LATEX_RE = re.compile(r"\$[^$]*\$|\\[a-zA-Z]+|[{}^_]")


def title_looks_english(title: str, cfg: FilterConfig) -> bool:
    """Is the title English? A title-only generation prompt hands the model
    nothing else, so a French title yields a French abstract paired with the
    paper's English one — a language artefact the classifier learns instantly.

    A short string, so the abstract's function-word *ratio* is too noisy here;
    instead: reject on a non-Latin script, on an accented-letter share a real
    English title never reaches, or on non-English function words outnumbering
    English ones (see :data:`_NON_ENGLISH_FUNCTION_WORDS`).
    """
    core = _TAG_RE.sub(" ", html.unescape(as_text(title)))
    core = _LATEX_RE.sub(" ", core)
    letters = [c for c in core if c.isalpha()]
    if not letters:
        return True  # nothing to judge — a symbol-only title is caught elsewhere

    if sum(1 for c in letters if ord(c) > 0x24F) / len(letters) > 0.30:
        return False

    def _fold(c: str) -> str:
        return "".join(x for x in unicodedata.normalize("NFKD", c)
                       if not unicodedata.combining(x))

    accented = sum(1 for c in letters
                   if ord(c) < 0x250 and c.lower() != _fold(c).lower()) / len(letters)

    # `van der Waals`, `de Rham`, `von Neumann` — a foreign particle next to a
    # capitalised word (not the title's first) is part of a name, not prose.
    cased = re.findall(r"[^\W\d_]+", core, re.UNICODE)
    _NAME_PARTICLES = {"van", "der", "von", "de", "del", "da", "di", "la", "le"}
    name_pos = {
        i for i, t in enumerate(cased)
        if t.lower() in _NAME_PARTICLES
        and ((i > 1 and cased[i - 1][:1].isupper())
             or (i + 1 < len(cased) and cased[i + 1][:1].isupper()))
    }
    tokens = [t.lower() for i, t in enumerate(cased) if i not in name_pos
              and len(t) >= 2]
    ne = sum(t in _NON_ENGLISH_FUNCTION_WORDS for t in tokens)
    en = sum(t in _ENGLISH_STOPWORDS for t in tokens)

    if ne >= 2 and ne > en:
        return False
    if ne >= 1 and en == 0 and accented > 0.02:
        return False
    if accented > 0.12 and en == 0:
        return False
    return True


def _function_word_ratio(text: str) -> float:
    """Share of tokens that are common English function words.

    Running prose sits near 0.15-0.4; a menu, a reference list or LaTeX source
    is content-word (or symbol) heavy and falls well below that. Uses the same
    word list as :func:`looks_english`, as a *ratio* rather than a distinct
    count — a 300-word nav dump clears the distinct-word bar easily.
    """
    tokens = _WORD_RE.findall(text.lower())
    if not tokens:
        return 0.0
    return sum(t in _ENGLISH_STOPWORDS for t in tokens) / len(tokens)


def abstract_is_boilerplate(text: str) -> bool:
    """True when *text* is not an abstract at all: a scrape of the publisher's
    page (site chrome, an article-nav strip, a paywall) or a book's table of
    contents / front matter.

    Every trigger is something a real abstract never contains, because
    :func:`check_record` drops on this and a false positive silently loses a
    genuine paper. Wrong-language and raw-LaTeX text is *not* judged here — it
    fails :func:`looks_english` instead; keeping the two apart is what lets the
    drop report say which it was.
    """
    if not text:
        return False
    low = text.lower()
    if any(s in low for s in _NON_ABSTRACT_SIGNATURES):
        return True
    if _FRONT_MATTER_OPENER.match(text.lstrip()):
        return True
    return False


# --------------------------------------------------------------------------
# Deduplication state
# --------------------------------------------------------------------------


class SeenIndex:
    """Caller-owned set of already-claimed paper ids, normalised titles and
    normalised abstracts.

    Shared across *all* domains on purpose. A paper matching queries in two
    domains would otherwise land in two per-domain datasets, and the same text
    appearing under two domain labels leaks across LODO/LOGO folds. First
    domain wins (domains are processed in config order, so the outcome is
    deterministic) and the loser is recorded on the winning row's
    ``also_matched_domains`` for auditing.

    The abstract key catches the case id and title both miss: the same article
    deposited twice under different ``paperId``s and titles, but with identical
    abstract text (see :func:`normalise_abstract`). ``norm_abstract`` is
    optional so callers that have no cleaned abstract to hand can still use the
    id/title keys alone.
    """

    def __init__(self) -> None:
        self.by_paper_id: dict[str, int] = {}
        self.by_title: dict[str, int] = {}
        self.by_abstract: dict[str, int] = {}

    def lookup(self, paper_id: str, norm_title: str,
               norm_abstract: str = "") -> int | None:
        """Return the row index that already claimed this paper, if any."""
        if paper_id and paper_id in self.by_paper_id:
            return self.by_paper_id[paper_id]
        if norm_title and norm_title in self.by_title:
            return self.by_title[norm_title]
        if norm_abstract and norm_abstract in self.by_abstract:
            return self.by_abstract[norm_abstract]
        return None

    def claim(self, paper_id: str, norm_title: str, row_index: int,
              norm_abstract: str = "") -> None:
        if paper_id:
            self.by_paper_id[paper_id] = row_index
        if norm_title:
            self.by_title[norm_title] = row_index
        if norm_abstract:
            self.by_abstract[norm_abstract] = row_index


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
    "missing_fwci",
    "wrong_language",
    "truncated_abstract",
    "abstract_too_short",
    "abstract_too_long",
    "non_english_title",
    "not_english",
    "boilerplate_abstract",
    "duplicate",
)


def check_record(record: dict, cfg: FilterConfig) -> str | None:
    """Return a drop reason for *record*, or ``None`` if it is admissible.

    Expects a cleaned record as produced by :func:`normalise_record_oa` (i.e.
    the abstract already run through :func:`clean_abstract`). Duplicates are
    *not* checked here — that needs the caller's :class:`SeenIndex`.
    """
    if not as_text(record.get("paper_id")):
        return "missing_paper_id"

    title = as_text(record.get("title")).strip()
    if not title:
        return "missing_title"
    if len(title.split()) < cfg.min_title_words:
        return "short_title"
    if not title_looks_english(title, cfg):
        return "non_english_title"

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

    if cfg.require_fwci and is_missing(record.get("fwci")):
        return "missing_fwci"

    if cfg.require_language and as_text(record.get("language")) != cfg.require_language:
        return "wrong_language"

    # Some sources truncate abstracts mid-sentence; an ellipsis terminator is
    # the reliable marker. Missing terminal punctuation alone is too noisy.
    if abstract.endswith("…") or abstract.endswith("..."):
        return "truncated_abstract"

    n_words = len(abstract.split())
    if n_words < cfg.min_words:
        return "abstract_too_short"
    if n_words > cfg.max_words:
        return "abstract_too_long"

    # A scrape of the publisher's page or a book's contents page — checked
    # before the language rule so an English one is not mislabelled
    # ``not_english`` (a nav dump has almost no function words either).
    if abstract_is_boilerplate(abstract):
        return "boilerplate_abstract"

    if not looks_english(abstract, cfg):
        return "not_english"

    return None


# Numeric columns that can arrive null, with the dtype every stage must write
# them as. `fwci` is the only one, and it is null-bearing despite
# `require_fwci`: the *raw* parquet holds every record the API returned,
# rejected ones included, and a record rejected for `missing_fwci` is precisely
# one whose `fwci` is null.
#
# An all-null column serialises to Arrow type `null`, which then cannot be read
# alongside a file where the same column has values —
# `ArrowNotImplementedError: Unsupported cast from double to null`. Pinning the
# dtype at write time keeps every shard schema-compatible. See
# `epai.utils.parquet_utils.coerce_dtypes`.
OPTIONAL_NUMERIC_COLUMNS = {
    "fwci": "float64",
}


# --------------------------------------------------------------------------
# OpenAlex
# --------------------------------------------------------------------------
#
# The flattening lives here rather than in `oa_utils` because the API field
# list (`OA_SELECT`) and the code reading it have to stay in step, so they sit
# next to each other. `oa_utils` stays a transport layer that knows nothing
# about the row shape.


def invert_abstract(inverted_index: dict | None) -> str:
    """Rebuild plain text from an OpenAlex ``abstract_inverted_index``.

    OpenAlex does not ship abstracts as prose. It ships ``{word: [positions]}``,
    an inverted index, which reportedly sidesteps the redistribution terms on
    the abstracts it aggregates. Reconstruction is exact for word order but not
    for whitespace: the original spacing between tokens is not recorded, so
    tokens are rejoined with single spaces. That is harmless here because
    `clean_abstract` collapses whitespace anyway.

    Two things it does *not* recover, both worth knowing before treating
    ``abstract_raw`` as the publisher's text:

      - Punctuation attached to words survives (it is part of the token), but
        anything the indexer dropped is gone for good.
      - Gaps in the position sequence are left as-is; the words are emitted in
        position order regardless, so a missing index shortens the text rather
        than shifting it.
    """
    if not inverted_index:
        return ""

    positions: list[tuple[int, str]] = []
    for word, where in inverted_index.items():
        for pos in where or []:
            positions.append((pos, word))

    positions.sort(key=lambda p: p[0])
    return " ".join(word for _, word in positions)


def _oa_id(value: str | None) -> str:
    """Strip the URL prefix from an OpenAlex id: a full URL down to ``W12345``.

    The short form is what the dedupe index and every report want to show, and
    it round-trips back to the API unchanged (``/works/W12345`` resolves, as
    does ``/authors/A5011953697``).
    """
    text = as_text(value)
    return text.rsplit("/", 1)[-1] if text else ""


def _doi(value: str | None) -> str:
    """Strip the resolver prefix from an OpenAlex DOI.

    OpenAlex returns ``https://doi.org/10.1038/nature12373``; the bare
    ``10.1038/nature12373`` is the DOI itself, which is what a reader pastes
    into a resolver and what every other DOI-bearing table in this project
    stores. Empty for the ~8% of works that carry no DOI.
    """
    text = as_text(value).strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.lower().startswith(prefix):
            return text[len(prefix):]
    return text


def author_ids_oa(work: dict) -> list[str]:
    """OpenAlex author ids for one work, in authorship order.

    Ids rather than names because a name is not an identity: authors publish
    under varying forms ("W. G. Zumft" / "Walter G. Zumft"), and different
    people share names. ``A5011953697`` is stable across a person's papers and
    resolves at ``/authors/A5011953697``, so it is the only form worth grouping
    or joining on.

    Returned as a list, and stored in parquet as a genuine list column rather
    than a delimiter-joined string — there is no separator that cannot appear
    inside a value to worry about, and no re-splitting for a consumer to get
    wrong.

    **The list can be shorter than the author count.** Roughly 5% of
    authorships are ones OpenAlex could not resolve to an author entity; they
    carry a name but no id, and are skipped here because there is no id to
    record. That is why :func:`normalise_record_oa` also stores
    ``author_count``, taken from the authorship list before any of this
    filtering: comparing the two is the only way to tell a genuinely
    single-author paper from a ten-author one whose authors went unresolved.

    ``authorships`` is the single most expensive field in ``OA_SELECT`` — it
    carries institutions, countries and affiliation strings per author, none of
    which this pipeline stores. It is requested anyway because there is no
    ids-only projection: OpenAlex selects whole top-level fields.
    """
    ids = []
    for authorship in work.get("authorships") or []:
        author_id = _oa_id((authorship.get("author") or {}).get("id"))
        if author_id:
            ids.append(author_id)
    return ids


def normalise_record_oa(
    work: dict, domain: str, query: str, *, strip_labels: bool = False
) -> dict:
    """Flatten one OpenAlex work into the raw-corpus row shape.

    The fields that need explaining rather than just reading:

      - ``fwci`` — Field-Weighted Citation Impact: citations received against
        the expectation for work of the same field, type and age, so 1.0 is
        exactly average. Unlike ``citation_count`` it is comparable across
        domains and across publication years, which is what makes it worth
        carrying next to the raw count rather than instead of it. Null when
        OpenAlex cannot place the work in a field, and ``check_record`` drops
        those under ``require_fwci``.
      - ``language`` — OpenAlex's own language label, required to equal
        ``require_language`` (``"en"``). Kept as a column and not merely
        filtered on, so the ``not_english`` rejections can be told apart from
        the ``wrong_language`` ones in the drop report.
      - ``author_ids`` — see :func:`author_ids_oa`; a list of OpenAlex author
        ids in authorship order, which can be shorter than ``author_count``.
      - ``author_count`` — authorships on the work, counted before unresolved
        ones are dropped from ``author_ids``. The honest author count.
      - ``doi`` — bare, no resolver prefix (see :func:`_doi`). Empty for the
        works that carry none; not required, since a missing DOI says nothing
        about whether the abstract is usable.
      - ``fields_of_study`` — the *field* level of the OpenAlex topic hierarchy
        (domain → field → subfield → topic), which is the level whose labels
        are broad subject areas ("Biology", "Computer Science"). The finer
        levels are dropped.

    Of these only ``fwci`` and ``language`` are read by a rule. The rest are
    corpus metadata, kept so a downstream analysis can slice on them without a
    refetch.
    """
    # Reconstructed once and reused: the inverted index of a 300-word abstract
    # is a few hundred dict entries, and both the cleaned and the raw column
    # need the same reconstruction.
    raw_abstract = invert_abstract(work.get("abstract_inverted_index"))
    cleaned, fixes = clean_abstract(raw_abstract, strip_labels=strip_labels)
    title = as_text(work.get("display_name") or work.get("title")).strip()

    source = (work.get("primary_location") or {}).get("source") or {}

    # dict.fromkeys, not a set: topics arrive in descending score order and
    # keeping that order makes the column stable across runs.
    fields = dict.fromkeys(
        as_text((t.get("field") or {}).get("display_name"))
        for t in (work.get("topics") or [])
    )
    fields.pop("", None)

    return {
        "paper_id": _oa_id(work.get("id")),
        "doi": _doi(work.get("doi")),
        "title": title,
        "abstract": cleaned,
        "abstract_raw": raw_abstract,
        "cleaning_fixes": ",".join(fixes),
        "author_ids": author_ids_oa(work),
        "author_count": len(work.get("authorships") or []),
        "year": work.get("publication_year"),
        "venue": as_text(source.get("display_name")),
        "citation_count": work.get("cited_by_count"),
        "fwci": work.get("fwci"),
        "language": as_text(work.get("language")),
        "fields_of_study": ",".join(fields),
        "domain": domain,
        "query": query,
        "also_matched_domains": "",
    }


# Fields requested from the OpenAlex API (`select=`). Kept next to
# `normalise_record_oa` so the two stay in step: every entry here is read by
# that function, and every field it reads is listed here.
#
# Trimming this list is worth it — a bare /works response carries all
# locations, counts_by_year, related works and referenced works for every hit,
# an order of magnitude more bytes than the fields below. `authorships` is the
# one expensive field kept anyway, because it is the only way to get author
# names (OpenAlex selects whole top-level fields, so there is no names-only
# projection); it roughly doubles the per-record payload of a page.
OA_SELECT = (
    "id,doi,display_name,publication_year,abstract_inverted_index,"
    "authorships,cited_by_count,fwci,language,primary_location,topics"
)
