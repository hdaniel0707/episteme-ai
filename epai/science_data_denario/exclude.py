"""Training-set overlap guard for the Denario human control.

The 80 human abstracts in ``denario_hc`` must not be anything a detector was
trained on, or its recall on that row is a free hit. The training corpora carry
the OpenAlex work id inside their group key — ``science:<domain>:<Wxxxx>`` — plus
a ``doi`` and a ``title`` column, so three keys are available to match on:

  work id      exact, and the strongest — ``W2798701005`` is stable across a
               paper's OpenAlex history.
  doi          exact, catches a work re-ingested under a different OpenAlex id.
  norm title   fuzzy (``rules.normalise_title``), catches preprint-vs-published
               pairs that share neither id nor doi.

Both ``fetch_human_candidates`` (drop the candidate) and ``build_denario_hc``
(hard-fail if one slipped through) call :func:`load_exclusion` on the same
default parquet, so the check cannot be silently skipped by running only one.

NOTE — this only guards the Denario side. A NEW training corpus (science_v5,
Track 2) that adds Astronomy / Neuroscience overlaps these domains far more, and
must exclude the ids in ``denario_hc_raw.parquet``'s ``oa_id`` column from ITS
fetch. That is the other end and is not done here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from epai.science_data.rules import normalise_title

BASE_DIR = Path(__file__).resolve().parents[2]

#: What the human control is checked against by default. A list, so science_v5
#: can be appended without a code change once it exists.
DEFAULT_EXCLUDE = [BASE_DIR / "data" / "parquet" / "science_v3_gpt56luna_0811A.parquet"]

_OA_ID_RE = re.compile(r"W\d+", re.IGNORECASE)


@dataclass
class Exclusion:
    work_ids: set[str] = field(default_factory=set)
    dois: set[str] = field(default_factory=set)
    titles: set[str] = field(default_factory=set)

    def hit(self, work_id: str | None, doi: str | None, title: str | None) -> str | None:
        """Return which key matched (``'work_id'`` / ``'doi'`` / ``'title'``) or None."""
        if work_id and _norm_id(work_id) in self.work_ids:
            return "work_id"
        if doi and _norm_doi(doi) in self.dois:
            return "doi"
        if title and normalise_title(title) in self.titles:
            return "title"
        return None

    def __len__(self) -> int:
        return len(self.work_ids)


def _norm_id(value: str) -> str:
    m = _OA_ID_RE.search(str(value))
    return m.group(0).upper() if m else ""


def _norm_doi(value: str) -> str:
    text = str(value or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text


def load_exclusion(paths: list[Path] | None = None) -> Exclusion:
    """Union of work ids, DOIs and normalised titles across the given parquets.

    Reads only the id / doi / title columns, so it stays cheap on a scored
    400 MB training parquet.
    """
    excl = Exclusion()
    for path in paths or DEFAULT_EXCLUDE:
        path = Path(path)
        if not path.is_file():
            raise SystemExit(
                f"exclusion parquet not found: {path}\n"
                f"  pass --exclude-parquet, or (only if you really have no "
                f"training corpus yet) --no-exclude-check"
            )
        cols = pd.read_parquet(path).columns
        want = [c for c in ("id", "doi", "title") if c in cols]
        df = pd.read_parquet(path, columns=want)
        if "id" in df:
            excl.work_ids |= {_norm_id(v) for v in df["id"].dropna()} - {""}
        if "doi" in df:
            excl.dois |= {_norm_doi(v) for v in df["doi"].dropna()} - {""}
        if "title" in df:
            excl.titles |= {normalise_title(v) for v in df["title"].dropna()} - {""}
    return excl
