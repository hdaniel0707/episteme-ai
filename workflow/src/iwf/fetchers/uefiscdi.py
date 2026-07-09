"""UEFISCDI journal lists (plan §4.3): AIS table loader + red/yellow-zone paper seeds.

UEFISCDI publishes yearly public xlsx lists of red (Q1) / yellow (Q2) zone journals and
their Article Influence Score ("scor relativ de influență"), per JCR year
(uefiscdi.gov.ro -> Scientometrie -> Reviste; CNFIS mirrors 2013-2024). Those lists are
JOURNAL-level, not paper-level, so they are used two ways:

  1. ``load_ais_table``  -> venue_ais baseline (features.external_impact.VenueAISBaseline)
  2. ``UefiscdiSeedFetcher`` -> per-ISSN OpenAlex paper fetches; every paper inherits the
     journal's AIS + zone (an independent human-committee quality signal).

Place the downloaded xlsx files under ``workflow/data/uefiscdi/`` (any filename). Column
names differ between years, so the loader matches them fuzzily; the JCR year is taken
from a year-like column when present, else parsed from the filename.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterator

import pandas as pd

from workflow.src.iwf import paths
from workflow.src.iwf.fetchers.base import BaseFetcher
from workflow.src.iwf.fetchers.openalex import PER_PAGE, SELECT, WORKS_URL, _get, work_to_row

UEFISCDI_DIR = paths.DATA_DIR / "uefiscdi"

# fuzzy header matching: first alias found (case/diacritic-insensitive substring) wins
_COL_ALIASES = {
    "journal": ("revista", "journal", "titlu", "publicatie", "publication"),
    "issn": ("issn",),
    "ais": ("scor relativ de influenta", "article influence", "ais",
            "influence score", "sri"),
    "zone": ("zona", "zone", "cuartila", "quartile", "clasificare"),
    "year": ("an", "year", "jcr"),
}

_DIACRITICS = str.maketrans("ăâîșşțţ", "aaisstt")


def _norm_header(h: str) -> str:
    return str(h).strip().lower().translate(_DIACRITICS)


def normalize_journal(name) -> str:
    """Canonical journal-name key for joining venues across sources."""
    s = str(name or "").lower().translate(_DIACRITICS)
    s = re.sub(r"\(.*?\)", " ", s)          # drop parenthetical qualifiers
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\bthe\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _find_col(df: pd.DataFrame, key: str) -> str | None:
    for col in df.columns:
        h = _norm_header(col)
        for alias in _COL_ALIASES[key]:
            if alias in h:
                return col
    return None


def _year_from_name(path: Path) -> int | None:
    m = re.search(r"(20\d{2})", path.name)
    return int(m.group(1)) if m else None


def load_ais_table(xlsx_dir: str | Path | None = None) -> pd.DataFrame:
    """All xlsx files under ``xlsx_dir`` -> DataFrame[journal, journal_norm, issn, year,
    ais, zone]. Raises FileNotFoundError when the folder has no readable list."""
    xlsx_dir = Path(xlsx_dir) if xlsx_dir else UEFISCDI_DIR
    files = sorted(xlsx_dir.glob("*.xlsx")) + sorted(xlsx_dir.glob("*.xls"))
    if not files:
        raise FileNotFoundError(
            f"No UEFISCDI journal lists under {xlsx_dir} — download the yearly AIS xlsx "
            "from uefiscdi.gov.ro (Scientometrie -> Reviste) into that folder.")
    frames = []
    for f in files:
        try:
            raw = pd.read_excel(f)
        except Exception as e:
            print(f"[uefiscdi] {f.name} unreadable ({e}); skipped")
            continue
        jcol, icol = _find_col(raw, "journal"), _find_col(raw, "issn")
        acol, zcol = _find_col(raw, "ais"), _find_col(raw, "zone")
        ycol = _find_col(raw, "year")
        if jcol is None or acol is None:
            print(f"[uefiscdi] {f.name}: no journal/AIS columns recognised; skipped")
            continue
        part = pd.DataFrame({
            "journal": raw[jcol].astype(str).str.strip(),
            "issn": raw[icol].astype(str).str.strip() if icol else None,
            "ais": pd.to_numeric(
                raw[acol].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"),
            "zone": raw[zcol].astype(str).str.strip().str.lower() if zcol else None,
        })
        if ycol is not None:
            part["year"] = pd.to_numeric(raw[ycol], errors="coerce")
        else:
            part["year"] = float("nan")
        part["year"] = part["year"].fillna(_year_from_name(f) or 0).astype(int)
        frames.append(part[part["ais"].notna() & (part["journal"] != "")])
    if not frames:
        raise FileNotFoundError(f"No usable UEFISCDI list could be parsed in {xlsx_dir}.")
    out = pd.concat(frames, ignore_index=True)
    out["journal_norm"] = out["journal"].map(normalize_journal)
    out = out.drop_duplicates(subset=["journal_norm", "year"], keep="last")
    print(f"[uefiscdi] AIS table: {len(out)} journal-year rows from {len(files)} file(s)")
    return out


class UefiscdiSeedFetcher(BaseFetcher):
    """Fetch papers of red/yellow-zone journals from OpenAlex by ISSN; each paper
    inherits venue_ais + venue_zone from the UEFISCDI list."""

    name = "uefiscdi_journal"

    def __init__(self, year_from: int | None = None, year_to: int | None = None,
                 per_journal_cap: int = 300, xlsx_dir: str | Path | None = None):
        self.year_from = year_from
        self.year_to = year_to
        self.per_journal_cap = int(per_journal_cap)
        self.xlsx_dir = xlsx_dir

    def rows(self, conn: sqlite3.Connection) -> Iterator[dict]:
        import time

        ais = load_ais_table(self.xlsx_dir)
        # newest AIS per journal, ISSN required for the OpenAlex filter
        ais = ais.sort_values("year").drop_duplicates("journal_norm", keep="last")
        ais = ais[ais["issn"].notna() & ais["issn"].astype(str).str.contains(r"\d")]
        print(f"[{self.name}] seeding from {len(ais)} journals with an ISSN")

        for _, jr in ais.iterrows():
            issn = re.sub(r"[^0-9Xx-]", "", str(jr["issn"]).split(";")[0].strip())
            if not issn:
                continue
            filt = f"primary_location.source.issn:{issn},type:article,has_abstract:true"
            if self.year_from and self.year_to:
                filt += f",publication_year:{self.year_from}-{self.year_to}"
            cursor, fetched = "*", 0
            while cursor and fetched < self.per_journal_cap:
                data = _get(WORKS_URL, {"filter": filt, "per-page": PER_PAGE,
                                        "cursor": cursor, "select": SELECT})
                if data is None:
                    break
                for work in data.get("results", []):
                    row = work_to_row(work, source=self.name)
                    row["venue_ais"] = float(jr["ais"])
                    row["venue_zone"] = jr.get("zone")
                    fetched += 1
                    yield row
                cursor = (data.get("meta") or {}).get("next_cursor")
                time.sleep(0.11)


def print_download_hints() -> None:
    """The lists move around on the site; print the entry pages instead of dead links."""
    print("Download the yearly AIS / red-yellow journal lists (xlsx) from:")
    print("  * https://uefiscdi.gov.ro/scientometrie-reviste")
    print("  * https://www.cnfis.ro/clasificare-reviste-isi/   (2013-2024 mirror)")
    print(f"and place them under {UEFISCDI_DIR} (any filename; the JCR year is read "
          "from a year column or the filename).")
