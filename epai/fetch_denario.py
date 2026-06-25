#!/usr/bin/env python3
"""
Download Denario example papers, extract title + abstract, store in SQLite.

Source: https://github.com/AstroPilot-AI/DenarioExamplePapers

Usage:
    pip install pypdf requests
    python denario_fetch.py                 # download everything + build DB
    python denario_fetch.py --no-download    # just (re)build DB from already-downloaded PDFs
    python denario_fetch.py --out my_papers  # change output folder
    python denario_fetch.py --debug          # download only the first 5 papers

Outputs:
    papers/<Category>/<paper>.pdf   one PDF per paper
    denario_papers.db               SQLite DB, table `papers`
"""

import argparse
import os
import re
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

README_RAW = "https://raw.githubusercontent.com/AstroPilot-AI/DenarioExamplePapers/main/README.md"
HEADERS = {"User-Agent": "denario-fetch/1.0"}

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "data" / "sql" / "denario_papers.db"
PAPERS_DIR = BASE_DIR / "data" / "papers" / "denario"


# --------------------------------------------------------------------------- #
# 1. Parse the README into a list of papers
# --------------------------------------------------------------------------- #
def fetch_readme() -> str:
    req = urllib.request.Request(README_RAW, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def parse_papers(readme_text: str) -> list[dict]:
    """Return [{category, title, pdf_url, files_url}, ...]."""
    section = readme_text.split("## Papers", 1)[1].split("## Downloading", 1)[0]
    papers, buf, cat = [], None, "Uncategorized"

    line_re = re.compile(
        r"^(.*?)\s*&rArr;\s*\[pdf\]\((.*?)\)\s*\|\s*\[files\]\((.*?)\)"
    )

    def flush():
        nonlocal buf
        if not buf:
            return
        joined = re.sub(r"\s+", " ", " ".join(buf)).strip()
        m = line_re.search(joined)
        if m:
            papers.append(
                {
                    "category": cat,
                    "title": m.group(1).strip(),
                    "pdf_url": m.group(2).strip(),
                    "files_url": m.group(3).strip(),
                }
            )
        buf = None

    for ln in section.splitlines():
        h = re.match(r"^#{2,4}\s+(.*)", ln.strip())
        if h:
            flush()
            cat = h.group(1).strip()
        elif ln.lstrip().startswith("- "):
            flush()
            buf = [ln.lstrip()[2:]]
        elif buf is not None:
            if ln.strip() == "":
                flush()
            else:
                buf.append(ln.strip())
    flush()
    return papers


# --------------------------------------------------------------------------- #
# 2. Download PDFs
# --------------------------------------------------------------------------- #
def safe_name(title: str, maxlen: int = 120) -> str:
    s = re.sub(r"[^\w\s-]", "", title).strip()
    s = re.sub(r"\s+", "_", s)
    return s[:maxlen] or "untitled"


def download_pdf(url: str, dest: str, retries: int = 3) -> bool:
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        return True  # already have it
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                f.write(r.read())
            if os.path.getsize(dest) > 1024:
                return True
        except Exception as e:
            print(f"    attempt {attempt} failed: {e}")
            time.sleep(2 * attempt)
    return False


# --------------------------------------------------------------------------- #
# 3. Extract abstract from a PDF
# --------------------------------------------------------------------------- #
def extract_abstract(pdf_path: str) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        return f"[could not read PDF: {e}]"

    # Grab text from the first 2 pages (abstract is always there)
    text = ""
    for page in reader.pages[:2]:
        text += "\n" + (page.extract_text() or "")
    text = re.sub(r"[ \t]+", " ", text)

    # Abstract starts at the word ABSTRACT, ends at the Introduction heading
    start = re.search(r"\bA\s*B\s*S\s*T\s*R\s*A\s*C\s*T\b", text, re.IGNORECASE)
    if not start:
        return ""
    body = text[start.end():]
    end = re.search(
        r"\n\s*(?:\d+\.?\s+)?(?:I\s*N\s*T\s*R\s*O\s*D\s*U\s*C\s*T\s*I\s*O\s*N"
        r"|Keywords|Key\s*words)\b",
        body,
        re.IGNORECASE,
    )
    if end:
        body = body[: end.start()]
    # Tidy: collapse hyphenation at line breaks, then newlines into spaces
    body = re.sub(r"-\n", "", body)
    body = re.sub(r"\s+", " ", body).strip()
    return body


# --------------------------------------------------------------------------- #
# 4. SQLite storage
# --------------------------------------------------------------------------- #
def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            category     TEXT,
            title        TEXT,
            abstract     TEXT,
            pdf_url      TEXT UNIQUE,
            files_url    TEXT,
            local_path   TEXT,
            downloaded_at TEXT
        )
        """
    )
    conn.commit()
    return conn


def upsert(conn: sqlite3.Connection, p: dict):
    conn.execute(
        """
        INSERT INTO papers (category, title, abstract, pdf_url, files_url, local_path, downloaded_at)
        VALUES (:category, :title, :abstract, :pdf_url, :files_url, :local_path, :downloaded_at)
        ON CONFLICT(pdf_url) DO UPDATE SET
            category=excluded.category,
            title=excluded.title,
            abstract=excluded.abstract,
            files_url=excluded.files_url,
            local_path=excluded.local_path,
            downloaded_at=excluded.downloaded_at
        """,
        p,
    )
    conn.commit()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PAPERS_DIR), help="folder for downloaded PDFs")
    ap.add_argument("--db", default=str(DB_PATH), help="SQLite DB path")
    ap.add_argument("--no-download", action="store_true", help="skip downloading")
    ap.add_argument("--debug", action="store_true", help="only process the first 5 papers")
    args = ap.parse_args()

    print("Fetching README ...")
    papers = parse_papers(fetch_readme())
    print(f"Found {len(papers)} papers.\n")

    if args.debug:
        papers = papers[:5]
        print(f"[debug] limited to {len(papers)} papers.\n")

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(args.db)
    ok = fail = 0

    for i, p in enumerate(papers, 1):
        dest = os.path.join(args.out, safe_name(p["category"]), safe_name(p["title"]) + ".pdf")
        print(f"[{i:>2}/{len(papers)}] {p['title'][:70]}")

        got = os.path.exists(dest)
        if not args.no_download:
            got = download_pdf(p["pdf_url"], dest)

        abstract, local_path = "", None
        if got and os.path.exists(dest):
            abstract = extract_abstract(dest)
            local_path = dest
            ok += 1
        else:
            fail += 1
            print("    -> download failed, storing metadata only")

        upsert(
            conn,
            {
                **p,
                "abstract": abstract,
                "local_path": local_path,
                "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )
        time.sleep(0.3)  # be polite to the server

    conn.close()
    print(f"\nDone. PDFs OK: {ok}, failed: {fail}")
    print(f"Database: {args.db}  (table 'papers')")
    print(f"PDFs:     {args.out}/")


if __name__ == "__main__":
    main()