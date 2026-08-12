r"""Character-level text standardisation, shared by every corpus pipeline.

WHAT THIS IS FOR
----------------
A detector corpus has two classes whose text reached the parquet by different
routes: the human side through whatever the source publisher, scraper or
detokeniser did to it, the AI side straight out of a chat model. Anything one
route does and the other does not becomes a feature -- and a feature that is
about the route, not about who wrote the text.

The measured case that prompted this module: in the Ghostbuster corpus the
human loops normalise quotes in ``wp`` but not in ``essay``, and the Reuters
source is plain ASCII to begin with. Curly-quote rate on the human side came
out at 4.3% (wp), 90.2% (essay), 0.0% (reuter), against an AI side produced by
one model at a roughly constant rate. So "curly quote" means *human* in essay
and *AI* in reuter -- the same feature with the sign reversed between domains,
which is what drives cross-domain transfer BELOW chance rather than merely to
chance.

Mapping both classes onto one spelling of each character removes the tell and
leaves the writing intact.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
**It never removes text based on words.** No preamble stripping, no section
labels, no "references" truncation, no title dropping. Those are content
decisions, they are how ``strip_boilerplate`` silently deleted 9% of essay
opening paragraphs, and they belong in a stage that says so in its name. Every
transform here is a character or whitespace substitution that a reader would
call the same text set differently.

**It is applied to both classes, always.** The pipelines run it over a whole
``text`` column, never per class -- a standardisation applied to one side is
just a new artifact pointing the other way.

RELATION TO ``prepare/normalize_text.py``
-----------------------------------------
That script is the *research* tool: opt-in, run by hand on a finished parquet,
with aggressive optional transforms (``reflow``, ``flatten``, ``shuffle_words``)
for attributing an effect to a feature. It imports :data:`UNICODE_GROUPS` from
here, so the table has one definition and the experiment cannot silently
diverge from what the pipelines produce.

This module is the *pipeline* half: the conservative subset that should be true
of every corpus before anyone scores or embeds it.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# The character table
# ---------------------------------------------------------------------------

# Grouped rather than flat so a caller can report which class of substitution
# actually fired -- normalize_text.py prints exactly that inventory.
#
# An explicit table rather than ``unicodedata.normalize("NFKC")``: NFKC leaves
# curly quotes exactly as they are (it is not a punctuation folder) while
# rewriting a pile of things nobody asked it to touch.
UNICODE_GROUPS: dict[str, dict[str, str]] = {
    "single_quotes": {
        "‘": "'",  # ' left single
        "’": "'",  # ' right single -- THE artifact
        "‚": "'",  # ‚ single low-9
        "‛": "'",  # ‛ single high-reversed-9
        "′": "'",  # ′ prime
        "´": "'",  # ´ acute accent used as apostrophe
        "`": "'",  # ` backtick used as apostrophe
    },
    "double_quotes": {
        "“": '"',  # " left double
        "”": '"',  # " right double
        "„": '"',  # „ double low-9
        "‟": '"',  # ‟ double high-reversed-9
        "″": '"',  # ″ double prime
        "«": '"',  # « left guillemet
        "»": '"',  # » right guillemet
    },
    "dashes": {
        "‐": "-",   # ‐ hyphen
        "‑": "-",   # ‑ non-breaking hyphen
        "‒": "-",   # ‒ figure dash
        "–": "-",   # – en dash
        "—": "--",  # — em dash
        "―": "--",  # ― horizontal bar
        "−": "-",   # − minus sign
    },
    "ellipsis": {
        "…": "...",  # …
    },
    "spaces": {
        " ": " ", " ": " ", " ": " ", " ": " ",
        " ": " ", " ": " ", " ": " ", " ": " ",
        " ": " ", " ": " ", " ": " ", " ": " ",
        " ": " ", " ": " ", " ": " ", "　": " ",
    },
    "invisible": {
        "­": "",  # soft hyphen
        "​": "",  # zero-width space
        "‌": "",  # zero-width non-joiner
        "‍": "",  # zero-width joiner
        "⁠": "",  # word joiner
        "﻿": "",  # BOM / zero-width no-break space
    },
    "bullets": {
        "•": "*", "·": "*", "●": "*", "▪": "*",
    },
    "ligatures": {
        "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
        "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
        "Œ": "OE", "œ": "oe", "Æ": "AE", "æ": "ae",
    },
    "misc": {
        "™": "(TM)", "®": "(R)", "©": "(C)",
        "⁄": "/",  # fraction slash
        "‹": "<", "›": ">",
    },
}

UNICODE_MAP: dict[str, str] = {
    ch: repl for group in UNICODE_GROUPS.values() for ch, repl in group.items()
}
UNICODE_TABLE = str.maketrans(UNICODE_MAP)


# ---------------------------------------------------------------------------
# Whitespace
# ---------------------------------------------------------------------------

_TRAILING_WS_RE = re.compile(r"(?m)[ \t]+$")
_LEADING_WS_RE = re.compile(r"(?m)^[ \t]+")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
# Two or more, down to one: see WHITESPACE_MODES["lines"].
_NEWLINE_RUN_RE = re.compile(r"\n{2,}")
_ANY_WS_RE = re.compile(r"\s+")

WHITESPACE_MODES = {
    "lines": (
        "keep one line per paragraph; collapse runs of blank lines to a single "
        "newline. Matches what generate.py's human loops already do in all "
        "three Ghostbuster domains -- essay and reuter collapse blank lines "
        "once, wp does it TWICE, while the AI side did it once. That mismatch "
        "made blank-line rate a class label in wp; this removes it by taking "
        "every document to the same convention."
    ),
    "flat": (
        "the document becomes one line: every run of whitespace, newlines "
        "included, becomes a single space. For corpora that are single "
        "paragraphs anyway -- science abstracts, where rules.clean_abstract "
        "and generate_ai.clean_generated both already end in this exact "
        "collapse, so it is a backstop rather than a change."
    ),
}


def standardize_text(text: str, *, whitespace: str = "lines") -> str:
    """Return *text* with typographic characters and layout standardised.

    Idempotent: running it twice changes nothing the first pass did not, which
    is what makes it safe to re-run a pipeline stage over its own output.

    :param whitespace: one of :data:`WHITESPACE_MODES`.
    """
    if whitespace not in WHITESPACE_MODES:
        raise ValueError(
            f"unknown whitespace mode {whitespace!r}; "
            f"choose from {sorted(WHITESPACE_MODES)}"
        )

    text = text.translate(UNICODE_TABLE)

    # Before anything counts newlines. A CRLF corpus would otherwise keep a
    # stray \r on every line and hand the classifier a perfect class label if
    # only one side had it.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if whitespace == "flat":
        return _ANY_WS_RE.sub(" ", text).strip()

    text = _TRAILING_WS_RE.sub("", text)
    text = _LEADING_WS_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _NEWLINE_RUN_RE.sub("\n", text)
    return text.strip()


def substitution_counts(text: str) -> dict[str, int]:
    """Characters per :data:`UNICODE_GROUPS` group that *text* contains.

    For reporting what a corpus looked like BEFORE standardisation -- the
    per-class breakdown is the evidence that an asymmetry existed, and it is
    unrecoverable once the text is rewritten.
    """
    counts: dict[str, int] = {}
    for group, mapping in UNICODE_GROUPS.items():
        n = sum(text.count(ch) for ch in mapping)
        if n:
            counts[group] = n
    return counts
