r"""Fetch more AI-generated papers (beyond Denario's 80) into the same
ai_detection "detect" schema as ``denario_detect.parquet``.

Denario's `mg_den` class is only 80 abstracts, all from one framework with an
undisclosed generator mix (see MODEL_BIAS.md). This pulls in every other
downloadable corpus of *actually agent-generated* research papers found by
searching:

    ai_author          system                  n   generator
    --------------     ----------------------- --  ---------------------------
    sakana             AI-Scientist v1         10  Claude-3.5-Sonnet / GPT-4o /
                                                    DeepSeek Coder / Llama-3.1-
                                                    405B (varies per paper, not
                                                    recoverable per-example)
    sakanav2           AI-Scientist-v2 /            Sakana's tree-search agent
                       ICLR2025-Workshop-Exp.   3   (backend undisclosed per-paper)
    cycleresearcher    CycleResearcher          6   CycleResearcher-12B/72B/123B
                                                    (fixed, disclosed model family)
    zochi              Zochi / Intology AI      1   undisclosed/proprietary

20 rows total. Small, but it roughly doubles Denario's AI-only pool and adds
generator diversity Denario doesn't have (see per-source note above -- most of
these at least don't hide which *system* produced them, even where the exact
LLM checkpoint is unknown).

--------------------------------------------------------------------------------
WHAT THIS DOES NOT DO
--------------------------------------------------------------------------------
This is the fetch + abstract-extraction stage ONLY -- the equivalent of
``fetch_denario.py`` + ``prepare_denario.py`` + ``prepare_denario_detect.py``
collapsed into one script, stopping at the same place ``denario_detect.parquet``
does: an eval-only, machine-only, UN-standardised, UN-scored corpus. Embedding,
standardisation and detector scoring are a separate later pass (mirror
``run_holdout_pipeline.py``'s stage list once this exists).

--------------------------------------------------------------------------------
SCHEMA
--------------------------------------------------------------------------------
Same 7 columns as ``prepare_denario_detect.py`` writes, plus two more:

    uid, id, domain, text, generated_by, title, year,  ai_author, source_url

``generated_by`` is set to the SAME value as ``ai_author`` (not a single
constant like Denario's ``"denario"``) -- every existing trainv2 script that
breaks a result down `--by-generator` (e.g. ``run_judge_by_generator.py``,
or the ad-hoc per-generator AUROC check in MODEL_BIAS.md) already groups on
``generated_by``, and each of these 4 sources is exactly the kind of distinct
"generator" that breakdown exists for. If you wanted one pooled label instead,
change ``GENERATED_BY_IS_AI_AUTHOR`` below.

``domain`` is the coarse constant ``"computer_science"`` for every row, not a
per-paper topic guess. Denario ships an explicit ``category`` field to slugify;
none of these sources do, and every one of them IS an ML/AI-systems paper, so
a coarse, honest label beats a fabricated fine-grained one.

``title`` is HARD-CODED only for the 4 papers whose exact title was confirmed
from an authoritative source (arXiv listing, or directly quoted in the
workshop-experiment repo's README) -- Zochi's Tempest paper and all 3
Sakana-v2 workshop papers. Every other title is extracted at runtime: the 10
Sakana v1 papers from their LaTeX ``\title{...}`` (reliable -- it's the actual
source macro, not a layout guess); the 6 CycleResearcher papers from PDF text
(best-effort heuristic: the text preceding the "Abstract" heading, since
CycleResearcher ships no .tex source). Every auto-extracted title is printed
in Stage 3 for a manual look -- treat the CycleResearcher ones as unverified
until skimmed. ``source_url`` is carried through so every row is traceable
back to where it came from, for citation.

--------------------------------------------------------------------------------
STAGES, EACH VERIFIED BEFORE THE NEXT RUNS
--------------------------------------------------------------------------------
  1. discover   resolve every registry entry to a concrete download URL (GitHub
                Contents API for repo files; the arXiv id needs no resolution).
                Verifies: every entry resolved, no duplicate resolved URLs.
                -> data/papers/ai_scientist/manifest.json
  2. download   fetch each PDF / arXiv Atom record to disk (skips what's already
                there and looks valid -- rerun-safe).
                Verifies: every file exists, is above a minimum size, and a PDF
                really starts with %PDF (catches an HTML error page saved with
                a .pdf name -- the exact failure mode fetch_denario.py's own
                size check exists for).
                -> data/papers/ai_scientist/<ai_author>/<id>.pdf|.tex|.xml
                -> data/papers/ai_scientist/download_status.json
  3. extract    pull title + abstract out of each file -- three methods, one
                per source kind: arXiv (parse the Atom XML directly), LaTeX
                source (regex out \title{...} / \begin{abstract}...\end{abstract},
                sakana v1 only -- cleaner than PDF text, no ligature/hyphenation
                noise), PDF (regex out the Abstract...Introduction span, same
                method as fetch_denario.py's extract_abstract, for sakanav2 and
                cycleresearcher, which ship no .tex source).
                Verifies: abstract non-empty and within a plausible word-count
                band for every paper; prints the word-count table and the list
                of auto-extracted titles to eyeball.
                -> data/papers/ai_scientist/extracted.json
  4. assemble   build the final DataFrame, verify the schema invariants
                (columns, no duplicate id, no null text, no null generated_by),
                print the same value_counts report prepare_denario_detect.py
                prints, and save.
                -> data/parquet/ai_scientist_detect.parquet

Each stage's output is cached, so a later stage can be re-run alone
(``--from download``) without repeating the ones before it. Nothing here mutates
in place -- rerunning a stage overwrites only that stage's cache file.

A stage that finds a real problem (a paper that won't resolve, won't download,
or extracts empty) STOPS the pipeline there by default, printing exactly which
paper and why -- this is 20 rows; a silently-dropped one is 5% of the corpus
gone unnoticed. Pass ``--allow-partial`` to instead skip just that paper and
continue (Stage 4 then also refuses to run if MORE than a couple came up
partial, since at that point something systemic is wrong).

Run:
    uv run python epai/ai_detection/fetch/fetch_ai_scientist_papers.py --dry-run
    uv run python epai/ai_detection/fetch/fetch_ai_scientist_papers.py
    uv run python epai/ai_detection/fetch/fetch_ai_scientist_papers.py --from extract
    uv run python epai/ai_detection/fetch/fetch_ai_scientist_papers.py --from assemble -o data/parquet/ai_scientist_detect.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
CACHE_DIR = BASE_DIR / "data" / "papers" / "ai_scientist"
DEFAULT_OUT = BASE_DIR / "data" / "parquet" / "ai_scientist_detect.parquet"

HEADERS = {"User-Agent": "episteme-ai-fetch-ai-scientist/1.0"}
GITHUB_API = "https://api.github.com"
ARXIV_API = "http://export.arxiv.org/api/query"

DOMAIN = "computer_science"
GENERATED_BY_IS_AI_AUTHOR = True  # see module docstring, SCHEMA section

MIN_PDF_BYTES = 4096
MIN_TEX_BYTES = 500
MIN_ABSTRACT_WORDS = 30
MAX_ABSTRACT_WORDS = 800

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
DIM = "\033[2m"
RESET = "\033[0m"


# --------------------------------------------------------------------------- #
# Registry -- every paper this script knows how to fetch
# --------------------------------------------------------------------------- #
@dataclass
class PaperSpec:
    ai_author: str
    year: int | None          # approximate: the framework's release year, NOT
                               # a confirmed per-example generation date
    kind: str                 # "github_pdf" | "arxiv"
    title: str | None = None  # only set where confirmed from an authoritative
                               # source; None means "extract it in Stage 3"
    repo: str | None = None
    path: str | None = None       # exact repo path, if known
    folder: str | None = None     # OR a folder to search for the one PDF in it
    latex_dir: str | None = None  # OR a folder to search for the one .tex source in it
    arxiv_id: str | None = None

    @property
    def key(self) -> str:
        """Stable id for filenames/caching, independent of the resolved URL."""
        raw = self.path or self.folder or self.latex_dir or self.arxiv_id or self.title or "?"
        raw = re.sub(r"\.pdf$", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"/latex$", "", raw, flags=re.IGNORECASE)
        return f"{self.ai_author}_{re.sub(r'[^a-z0-9]+', '_', raw.lower()).strip('_')}"


REGISTRY: list[PaperSpec] = [
    # -- Sakana AI-Scientist v1 --------------------------------------------- #
    # github.com/SakanaAI/AI-Scientist, example_papers/ -- 10 papers across the
    # NanoGPT / 2D-diffusion / grokking templates. The paper reports 4 base
    # LLMs (Claude 3.5 Sonnet, GPT-4o, DeepSeek Coder, Llama-3.1-405B) across
    # its runs but does not tag which example came from which -- so
    # ai_author names the SYSTEM, not the underlying model.
    #
    # Each `example_papers/<name>/` also ships `latex/template.tex` -- the
    # actual LaTeX source the PDF was compiled from (verified on
    # layerwise_lr_grokking: GitHub's contents API just doesn't recurse into
    # subfolders by default, which made it look empty on a shallow listing).
    # Title + abstract come straight out of that source (kind="github_latex"),
    # not out of PDF-extracted text -- no ligature/hyphenation noise, no
    # header/footer bleed, and the title is the actual `\title{...}`, not a
    # heuristic guess.
    *[
        PaperSpec("sakana", 2024, "github_latex",
                  repo="SakanaAI/AI-Scientist",
                  latex_dir=f"example_papers/{fname}/latex")
        for fname in [
            "adaptive_dual_scale_denoising",
            "data_augmentation_grokking",
            "dual_expert_denoiser",
            "gan_diffusion",
            "grid_based_noise_adaptation",
            "layerwise_lr_grokking",
            "mdl_grokking_correlation",
            "multi_style_adapter",
            "rl_lr_adaptation",
            "weight_initialization_grokking",
        ]
    ],

    # -- Sakana AI-Scientist-v2 workshop-accepted output --------------------- #
    # github.com/SakanaAI/AI-Scientist-ICLR2025-Workshop-Experiment -- 3 real
    # ICLR-2025-workshop submissions, one per folder (a 4th folder,
    # `ai-reviewing/`, is demo code, not a paper -- excluded). The PDF filename
    # inside each folder is NOT uniform (it's `annotated_paper.pdf`, alongside
    # a `sakana_review.pdf` of the reviews), so it's resolved by listing the
    # folder in Stage 1 rather than hardcoded, and the reviews file is
    # filtered out by name.
    PaperSpec("sakanav2", 2025, "github_pdf",
              title="Compositional Regularization: Unexpected Obstacles in "
                    "Enhancing Neural Network Generalization",
              repo="SakanaAI/AI-Scientist-ICLR2025-Workshop-Experiment",
              folder="compositional-regularization"),
    PaperSpec("sakanav2", 2025, "github_pdf",
              title="Unveiling the Impact of Label Noise on Model Calibration "
                    "in Deep Learning",
              repo="SakanaAI/AI-Scientist-ICLR2025-Workshop-Experiment",
              folder="label-noise"),
    PaperSpec("sakanav2", 2025, "github_pdf",
              title="Real-World Challenges in Pest Detection Using Deep Learning",
              repo="SakanaAI/AI-Scientist-ICLR2025-Workshop-Experiment",
              folder="pest-detection"),
    # NOTE: the repo also has an `ai-reviewing/` folder, but it is Sakana's
    # review-generation demo code (notebooks + perform_review.py) -- not a
    # submitted paper -- so it is deliberately NOT in this registry.

    # -- CycleResearcher ------------------------------------------------------ #
    # github.com/zhu-minjun/Researcher, generated_paper/ -- 6 PDFs. Written by
    # the CycleResearcher-12B/72B/123B model family: a purpose-built,
    # open-weight paper-writing model, fixed and disclosed (unlike the other
    # three sources here).
    *[
        PaperSpec("cycleresearcher", 2024, "github_pdf",
                  repo="zhu-minjun/Researcher", path=f"generated_paper/{fname}.pdf")
        for fname in ["Hark_reward", "World_GPT", "code2reward", "compress", "gss", "reversed"]
    ],

    # -- Zochi / Intology AI --------------------------------------------------- #
    # The only source with a real arXiv listing -- fetched directly via the
    # arXiv API, no PDF, no extraction heuristics.
    PaperSpec("zochi", 2025, "arxiv",
              title="Tempest: Autonomous Multi-Turn Jailbreaking of Large "
                    "Language Models with Tree Search",
              arxiv_id="2503.10619"),
]


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def _get(url: str, retries: int = 3, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 403 and "rate limit" in e.read(500).decode("utf-8", "ignore").lower():
                raise SystemExit(
                    f"{RED}GitHub API rate limit hit fetching {url}.{RESET} "
                    f"Set a GITHUB_TOKEN env var and rerun, or wait an hour."
                )
            last_exc = e
        except Exception as e:  # noqa: BLE001
            last_exc = e
        if attempt < retries:
            time.sleep(2 * attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_exc}")


def _github_contents(repo: str, path: str) -> list[dict] | dict:
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
    return json.loads(_get(url))


# --------------------------------------------------------------------------- #
# Stage 1 -- discover
# --------------------------------------------------------------------------- #
MANIFEST_PATH = CACHE_DIR / "manifest.json"


def discover() -> list[dict]:
    print("=" * 92)
    print("STAGE 1: discover")
    print("=" * 92)
    manifest: list[dict] = []
    for spec in REGISTRY:
        entry = {
            "key": spec.key, "ai_author": spec.ai_author, "title": spec.title,
            "year": spec.year, "kind": spec.kind,
        }
        if spec.kind == "arxiv":
            entry["download_url"] = f"{ARXIV_API}?id_list={spec.arxiv_id}"
            entry["source_url"] = f"https://arxiv.org/abs/{spec.arxiv_id}"
            entry["ext"] = "xml"
        elif spec.path:
            info = _github_contents(spec.repo, spec.path)
            entry["download_url"] = info["download_url"]
            entry["source_url"] = info["html_url"]
            entry["ext"] = "pdf"
        elif spec.latex_dir:
            listing = _github_contents(spec.repo, spec.latex_dir)
            tex_files = [f for f in listing if f["name"].lower().endswith(".tex")]
            if not tex_files:
                raise SystemExit(
                    f"{RED}discover: no .tex file in {spec.repo}/{spec.latex_dir} "
                    f"(saw {[f['name'] for f in listing]}){RESET}"
                )
            if len(tex_files) > 1:
                print(f"  {YELLOW}{spec.repo}/{spec.latex_dir}: {len(tex_files)} .tex "
                      f"files, picking {tex_files[0]['name']!r} out of "
                      f"{[f['name'] for f in tex_files]}{RESET}")
            chosen = tex_files[0]
            entry["download_url"] = chosen["download_url"]
            entry["source_url"] = chosen["html_url"]
            entry["ext"] = "tex"
        elif spec.folder:
            listing = _github_contents(spec.repo, spec.folder)
            pdfs = [f for f in listing if f["name"].lower().endswith(".pdf")]
            # "review" files (the reviewer feedback, e.g. sakana_review.pdf)
            # are never the paper -- excluded outright, ordering-independent.
            non_review = [f for f in pdfs if "review" not in f["name"].lower()]
            if not non_review:
                raise SystemExit(
                    f"{RED}discover: no non-review PDF found in "
                    f"{spec.repo}/{spec.folder} (saw {[f['name'] for f in pdfs]}){RESET}"
                )
            # Prefer a clean copy over an annotated one where both exist.
            preferred = [f for f in non_review if "annotat" not in f["name"].lower()]
            chosen = (preferred or non_review)[0]
            print(f"  {spec.repo}/{spec.folder}: picked {chosen['name']!r} "
                  f"out of {[f['name'] for f in pdfs]}")
            entry["download_url"] = chosen["download_url"]
            entry["source_url"] = chosen["html_url"]
            entry["ext"] = "pdf"
        else:
            raise SystemExit(
                f"{RED}discover: {spec.key} has neither path, folder, latex_dir, "
                f"nor arxiv_id{RESET}"
            )
        manifest.append(entry)

    urls = [e["download_url"] for e in manifest]
    if len(urls) != len(set(urls)):
        dupes = {u for u in urls if urls.count(u) > 1}
        raise SystemExit(f"{RED}discover: duplicate resolved URL(s): {dupes}{RESET}")

    print(f"\nresolved {len(manifest)}/{len(REGISTRY)} entries, 0 duplicates")
    for e in manifest:
        print(f"  [{e['ai_author']:<16}] {e['key']}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {MANIFEST_PATH}")
    return manifest


# --------------------------------------------------------------------------- #
# Stage 2 -- download
# --------------------------------------------------------------------------- #
STATUS_PATH = CACHE_DIR / "download_status.json"


def _dest_for(entry: dict) -> Path:
    return CACHE_DIR / entry["ai_author"] / f"{entry['key']}.{entry['ext']}"


def download(manifest: list[dict], allow_partial: bool) -> list[dict]:
    print("=" * 92)
    print("STAGE 2: download")
    print("=" * 92)
    statuses: list[dict] = []
    for entry in manifest:
        dest = _dest_for(entry)
        dest.parent.mkdir(parents=True, exist_ok=True)
        ok, reason = True, ""
        min_bytes = {"pdf": MIN_PDF_BYTES, "tex": MIN_TEX_BYTES}.get(entry["ext"], 1)
        if dest.is_file() and dest.stat().st_size >= min_bytes:
            reason = "cached"
        else:
            try:
                data = _get(entry["download_url"])
                dest.write_bytes(data)
                reason = "downloaded"
            except Exception as e:  # noqa: BLE001
                ok, reason = False, f"download failed: {e}"

        if ok and entry["ext"] == "pdf":
            head = dest.read_bytes()[:5] if dest.is_file() else b""
            if head != b"%PDF-":
                ok, reason = False, f"not a PDF (starts {head!r}) -- likely an HTML error page"
            elif dest.stat().st_size < MIN_PDF_BYTES:
                ok, reason = False, f"only {dest.stat().st_size} bytes"
        elif ok and entry["ext"] == "tex":
            body = dest.read_text(encoding="utf-8", errors="replace")
            if r"\documentclass" not in body and r"\begin{document}" not in body:
                ok, reason = False, "doesn't look like LaTeX source (no \\documentclass / \\begin{document})"

        status = {**entry, "dest": str(dest), "ok": ok, "reason": reason}
        statuses.append(status)
        mark = f"{GREEN}OK{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{mark}] {entry['key']:<45} {reason}")

    failed = [s for s in statuses if not s["ok"]]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(statuses, indent=2))
    print(f"\n{len(statuses) - len(failed)}/{len(statuses)} OK, wrote {STATUS_PATH}")

    if failed and not allow_partial:
        names = ", ".join(s["key"] for s in failed)
        raise SystemExit(
            f"{RED}download: {len(failed)} paper(s) failed: {names}{RESET}\n"
            f"Fix the URL/network issue and rerun `--from download`, or pass "
            f"--allow-partial to drop them and continue."
        )
    return [s for s in statuses if s["ok"]]


# --------------------------------------------------------------------------- #
# Stage 3 -- extract
# --------------------------------------------------------------------------- #
EXTRACTED_PATH = CACHE_DIR / "extracted.json"

BOUNDARY_END = re.compile(
    r"\n\s*(?:\d+\.?\s+)?(?:I\s*N\s*T\s*R\s*O\s*D\s*U\s*C\s*T\s*I\s*O\s*N"
    r"|Keywords|Key\s*words|CCS\s+Concepts)\b",
    re.IGNORECASE,
)
ABSTRACT_START = re.compile(r"\bA\s*B\s*S\s*T\s*R\s*A\s*C\s*T\b", re.IGNORECASE)


def _pdf_first_pages_text(path: Path, n_pages: int = 3) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text = ""
    for page in reader.pages[:n_pages]:
        text += "\n" + (page.extract_text() or "")
    return re.sub(r"[ \t]+", " ", text)


def _extract_abstract_from_pdf(text: str) -> str:
    start = ABSTRACT_START.search(text)
    if not start:
        return ""
    body = text[start.end():]
    end = BOUNDARY_END.search(body)
    if end:
        body = body[: end.start()]
    body = re.sub(r"-\n", "", body)
    body = re.sub(r"\s+", " ", body).strip()
    return body


def _extract_title_from_pdf(text: str) -> str:
    """Best-effort: the non-empty lines before the Abstract heading, minus
    anything that reads like an author/affiliation line. NOT authoritative --
    Stage 3 prints every auto-extracted title for a manual look."""
    start = ABSTRACT_START.search(text)
    head = text[: start.start()] if start else text[:800]
    lines = [ln.strip() for ln in head.splitlines() if ln.strip()]
    candidate_lines = []
    for ln in lines:
        if "@" in ln or re.search(r"\buniversity\b|\binstitute\b|\bdepartment\b", ln, re.IGNORECASE):
            break
        candidate_lines.append(ln)
        if len(" ".join(candidate_lines)) > 40:
            break
    title = " ".join(candidate_lines).strip()
    return title[:300] if title else "[title extraction failed]"


def _balanced_brace(text: str, open_pos: int) -> str:
    """``text[open_pos]`` must be ``{``. Return the content up to its match,
    handling one level of nesting (``\\title{Foo \\textbf{bar} baz}``)."""
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_pos + 1:i]
    return text[open_pos + 1:]  # unterminated -- return what there is


def _clean_latex(text: str) -> str:
    """Strip the LaTeX noise an abstract/title commonly carries. Not a full
    detexer -- just enough for the plain prose these templates produce."""
    text = re.sub(r"(?<!\\)%.*", "", text)                     # comments
    text = re.sub(r"\\(?:emph|textbf|textit|underline|texttt)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:emph|textbf|textit|underline|texttt)\{([^{}]*)\}", r"\1", text)  # nested, once more
    text = re.sub(r"\\cite[a-zA-Z]*\{[^{}]*\}", "", text)
    text = re.sub(r"\\ref\{[^{}]*\}", "", text)
    text = text.replace(r"\%", "%").replace(r"\&", "&").replace(r"\_", "_")
    text = text.replace("~", " ").replace("\\\\", " ")
    text = re.sub(r"\\[a-zA-Z]+", "", text)   # any remaining bare commands
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_from_latex(tex: str) -> dict:
    title = ""
    m = re.search(r"\\title\{", tex)
    if m:
        title = _clean_latex(_balanced_brace(tex, m.end() - 1))

    abstract = ""
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL)
    if m:
        abstract = _clean_latex(m.group(1))
    else:
        m = re.search(r"\\abstract\{", tex)
        if m:
            abstract = _clean_latex(_balanced_brace(tex, m.end() - 1))
    return {"title": title, "abstract": abstract}


def _extract_arxiv(path: Path) -> dict:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(path.read_text())
    entry = root.find("a:entry", ns)
    if entry is None:
        return {"title": "", "abstract": ""}
    title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
    summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
    summary = re.sub(r"\s+", " ", summary)
    return {"title": title, "abstract": summary}


def extract(statuses: list[dict], allow_partial: bool) -> list[dict]:
    print("=" * 92)
    print("STAGE 3: extract")
    print("=" * 92)
    results: list[dict] = []
    auto_titles: list[tuple[str, str]] = []
    for s in statuses:
        dest = Path(s["dest"])
        if s["ext"] == "xml":
            parsed = _extract_arxiv(dest)
            title = s["title"] or parsed["title"]
            abstract = parsed["abstract"]
        elif s["ext"] == "tex":
            parsed = _extract_from_latex(dest.read_text(encoding="utf-8", errors="replace"))
            abstract = parsed["abstract"]
            # The \title{...} macro is authoritative (not a PDF-layout guess),
            # but still worth a look -- multi-line titles or an unusual macro
            # can come out empty or truncated.
            title = s["title"] or parsed["title"] or "[title extraction failed]"
            if not s["title"]:
                auto_titles.append((s["key"], title))
        else:
            text = _pdf_first_pages_text(dest)
            abstract = _extract_abstract_from_pdf(text)
            if s["title"]:
                title = s["title"]
            else:
                title = _extract_title_from_pdf(text)
                auto_titles.append((s["key"], title))

        n_words = len(abstract.split())
        ok = n_words >= MIN_ABSTRACT_WORDS
        warn = ok and not (MIN_ABSTRACT_WORDS <= n_words <= MAX_ABSTRACT_WORDS)
        results.append({**s, "title": title, "abstract": abstract, "n_words": n_words, "ok": ok})

        mark = f"{GREEN}OK{RESET}" if ok else f"{RED}FAIL{RESET}"
        flag = f" {YELLOW}(unusual length){RESET}" if warn else ""
        print(f"  [{mark}] {s['key']:<45} {n_words:>4} words{flag}")

    if auto_titles:
        print(f"\n{YELLOW}auto-extracted titles -- verify these manually before using "
              f"them in anything published:{RESET}")
        for key, title in auto_titles:
            print(f"  {key:<45} {title!r}")

    failed = [r for r in results if not r["ok"]]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n{len(results) - len(failed)}/{len(results)} OK, wrote {EXTRACTED_PATH}")

    if failed and not allow_partial:
        names = ", ".join(r["key"] for r in failed)
        raise SystemExit(
            f"{RED}extract: {len(failed)} paper(s) produced no usable abstract: "
            f"{names}{RESET}\nInspect the cached PDF under {CACHE_DIR}, fix the "
            f"regex or the file, and rerun `--from extract`, or pass "
            f"--allow-partial to drop them."
        )
    return [r for r in results if r["ok"]]


# --------------------------------------------------------------------------- #
# Stage 4 -- assemble
# --------------------------------------------------------------------------- #
OUT_COLUMNS = ["uid", "id", "domain", "text", "generated_by", "title", "year",
               "ai_author", "source_url"]


def make_id(download_url: str) -> str:
    return "aisci:" + hashlib.sha1(download_url.encode("utf-8")).hexdigest()


def assemble(records: list[dict]) -> pd.DataFrame:
    print("=" * 92)
    print("STAGE 4: assemble")
    print("=" * 92)
    rows = []
    for r in records:
        generated_by = r["ai_author"] if GENERATED_BY_IS_AI_AUTHOR else "ai_scientist"
        paper_id = make_id(r["download_url"])
        rows.append({
            "uid": f"{paper_id}:{generated_by}",
            "id": paper_id,
            "domain": DOMAIN,
            "text": r["abstract"],
            "generated_by": generated_by,
            "title": r["title"],
            "year": r["year"],
            "ai_author": r["ai_author"],
            "source_url": r["source_url"],
        })
    df = pd.DataFrame(rows, columns=OUT_COLUMNS)

    # Verification -- same discipline prepare_denario_detect.py applies.
    problems = []
    if df["id"].duplicated().any():
        problems.append(f"{int(df['id'].duplicated().sum())} duplicate id(s)")
    if df["text"].isna().any() or (df["text"].str.strip() == "").any():
        problems.append("empty/null text row(s)")
    if df["generated_by"].isna().any():
        problems.append("null generated_by row(s)")
    missing_cols = [c for c in OUT_COLUMNS if c not in df.columns]
    if missing_cols:
        problems.append(f"missing column(s) {missing_cols}")
    if problems:
        raise SystemExit(f"{RED}assemble: {'; '.join(problems)}{RESET}")

    print(f"\n{len(df)} rows, {df['id'].nunique()} groups")
    print(f"  ai_author    : {df['ai_author'].value_counts().to_dict()}")
    print(f"  generated_by : {df['generated_by'].value_counts().to_dict()}")
    print(f"  domain       : {df['domain'].value_counts().to_dict()}")
    print(
        "\nAll machine, by construction -- eval-only, same as denario_detect.parquet. "
        "\nNot standardised, embedded or scored yet -- do that before any config reads it."
    )
    return df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--from", dest="from_stage", default="discover",
                   choices=["discover", "download", "extract", "assemble"],
                   help="resume from this stage, reading the previous stage's cache")
    p.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--allow-partial", action="store_true",
                   help="drop papers that fail download/extraction instead of stopping")
    p.add_argument("--dry-run", action="store_true",
                   help="run every stage, print everything, write no parquet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    order = ["discover", "download", "extract", "assemble"]
    start = order.index(args.from_stage)

    manifest = statuses = extracted = None
    if start <= order.index("discover"):
        manifest = discover()
    else:
        manifest = json.loads(MANIFEST_PATH.read_text())

    if start <= order.index("download"):
        statuses = download(manifest, args.allow_partial)
    else:
        statuses = [s for s in json.loads(STATUS_PATH.read_text()) if s["ok"]]

    if start <= order.index("extract"):
        extracted = extract(statuses, args.allow_partial)
    else:
        extracted = [r for r in json.loads(EXTRACTED_PATH.read_text()) if r["ok"]]

    df = assemble(extracted)

    if args.dry_run:
        print(f"\nDry run -- nothing written to {args.out}")
        return

    from epai.utils.parquet_utils import save_parquet
    save_parquet(df, args.out)
    print(
        "\nNext: standardize_text.py -> add_embeddings.py -> the detector suite, "
        "\nthe same order run_holdout_pipeline.py uses for denario_detect.parquet."
    )


if __name__ == "__main__":
    main()
