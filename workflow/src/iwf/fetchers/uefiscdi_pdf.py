"""Universal UEFISCDI/JCR PDF data extractor.

Reads all PDFs from workflow/data/pdf/, automatically detects table structures,
extracts journal data (name, ISSN, AIS/influence scores, quartiles).
Handles various PDF formats and gracefully skips PDFs without relevant data.
"""

import re
import sqlite3
from pathlib import Path
from typing import Iterator, Optional

import pdfplumber

from workflow.src.iwf import paths


def fuzzy_match_header(header_str: str, keywords: list[str], threshold: float = 0.6) -> bool:
    """Check if header matches any keyword (case-insensitive, partial match)."""
    h = str(header_str).lower().strip()
    for kw in keywords:
        if kw.lower() in h or h in kw.lower():
            return True
    return False


def extract_float(value: str) -> Optional[float]:
    """Safely extract float from string, handling various formats."""
    if not value or value.strip() == "":
        return None
    try:
        # Remove common characters
        v = str(value).strip().replace(",", ".").replace("%", "").strip()
        if v:
            return float(v)
    except (ValueError, TypeError):
        pass
    return None


def extract_from_table(table: list, pdf_name: str) -> Iterator[dict]:
    """Extract journal data from a single table.

    Tries multiple column detection strategies to find:
    journal name, ISSN, AIS/influence score, quartile/zone.
    """
    if not table or len(table) < 2:
        return

    header = table[0]
    if not header:
        return

    # Strategy 1: Fuzzy match on header
    col_indices = {}
    for idx, cell in enumerate(header):
        cell_str = str(cell).lower().strip() if cell else ""
        if not cell_str or len(cell_str) < 2:
            continue

        if fuzzy_match_header(cell_str, ["journal", "title", "name", "revue", "revista"]):
            col_indices["journal"] = idx
        elif fuzzy_match_header(cell_str, ["issn", "issn-l", "issn_l"]):
            col_indices["issn"] = idx
        elif fuzzy_match_header(cell_str, ["ais", "article influence", "influence score", "jif", "impact factor"]):
            col_indices["ais"] = idx
        elif fuzzy_match_header(cell_str, ["quartile", "q1", "q2", "q3", "q4", "zone", "rosu", "galben", "zona"]):
            col_indices["quartile"] = idx

    # If no headers matched, try position-based heuristics
    if not col_indices:
        # Common layout: journal | issn | ais | quartile
        if len(header) >= 2:
            col_indices["journal"] = 0
        if len(header) >= 3:
            col_indices["issn"] = 1
        if len(header) >= 4:
            col_indices["ais"] = 2
            col_indices["quartile"] = 3

    if "journal" not in col_indices:
        return  # Can't extract without journal name column

    # Extract rows
    for row in table[1:]:
        if not row or len(row) < 1:
            continue

        try:
            # Get journal name (required)
            j_idx = col_indices.get("journal", 0)
            journal_name = str(row[j_idx]).strip() if j_idx < len(row) else None
            if not journal_name or journal_name.lower() in ["journal", "title", "revue", ""]:
                continue

            # Get optional fields
            issn = None
            if "issn" in col_indices and col_indices["issn"] < len(row):
                issn_val = str(row[col_indices["issn"]]).strip()
                if issn_val and issn_val.lower() not in ["issn", ""]:
                    issn = issn_val

            ais = None
            if "ais" in col_indices and col_indices["ais"] < len(row):
                ais = extract_float(row[col_indices["ais"]])

            quartile = None
            if "quartile" in col_indices and col_indices["quartile"] < len(row):
                quartile_val = str(row[col_indices["quartile"]]).strip()
                if quartile_val and quartile_val.lower() not in ["quartile", "zone", ""]:
                    quartile = quartile_val

            # Only yield if we have at least journal name + one numeric field
            if journal_name and (ais is not None or issn):
                yield {
                    "journal_name": journal_name,
                    "issn": issn,
                    "ais_score": ais,
                    "quartile": quartile,
                    "source_file": pdf_name,
                }
        except Exception as e:
            continue  # Skip malformed rows


def extract_from_pdf(pdf_path: Path) -> Iterator[dict]:
    """Extract all journal data from a single PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            extracted_count = 0
            for page_num, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables()
                    if not tables:
                        continue

                    for table_num, table in enumerate(tables):
                        for record in extract_from_table(table, pdf_path.name):
                            yield record
                            extracted_count += 1
                except Exception as e:
                    continue

            if extracted_count > 0:
                print(f"[pdf] {pdf_path.name}: extracted {extracted_count} journals")
            else:
                print(f"[pdf] {pdf_path.name}: no journal data found (may not contain UEFISCDI/JCR tables)")

    except Exception as e:
        print(f"[pdf] ERROR reading {pdf_path.name}: {type(e).__name__}")


def extract_all_pdfs() -> Iterator[dict]:
    """Extract journal data from all PDFs in workflow/data/pdf/."""
    pdf_dir = paths.DATA_DIR / "pdf"

    if not pdf_dir.exists():
        print(f"[pdf] Directory not found: {pdf_dir}")
        return

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[pdf] No PDF files found in {pdf_dir}")
        return

    print(f"[pdf] Found {len(pdf_files)} PDF files")
    for pdf_path in pdf_files:
        for record in extract_from_pdf(pdf_path):
            yield record


def import_journal_data(conn: sqlite3.Connection) -> int:
    """Import journal data from all PDFs into extended DB journal cache."""
    # Create journal cache table if not exists
    conn.execute("""
    CREATE TABLE IF NOT EXISTS journal_cache (
        journal_name TEXT PRIMARY KEY,
        issn TEXT,
        ais_score REAL,
        quartile TEXT,
        source_file TEXT,
        imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    imported = 0
    duplicates = 0

    for journal_data in extract_all_pdfs():
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM journal_cache WHERE journal_name = ?",
                (journal_data["journal_name"],)
            )
            if cursor.fetchone()[0] > 0:
                duplicates += 1
                continue

            conn.execute("""
            INSERT INTO journal_cache
            (journal_name, issn, ais_score, quartile, source_file)
            VALUES (?, ?, ?, ?, ?)
            """, (
                journal_data["journal_name"],
                journal_data["issn"],
                journal_data["ais_score"],
                journal_data["quartile"],
                journal_data["source_file"],
            ))
            imported += 1

            if imported % 50 == 0:
                conn.commit()
                print(f"[pdf] Processed {imported} journals...")

        except Exception as e:
            continue

    conn.commit()
    print(f"[pdf] Import complete: {imported} journals added, {duplicates} duplicates skipped")
    return imported


if __name__ == "__main__":
    conn = sqlite3.connect(paths.SQL_DIR / "extended.db")
    count = import_journal_data(conn)
    conn.close()
    print(f"[pdf] Total: {count} journals in cache")
