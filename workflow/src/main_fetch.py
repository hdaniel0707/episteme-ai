"""Phase 0 - collect the corpus.

Reads the epai SQLite DB (PRIMARY source of papers), then gap-fills missing impact fields
from OpenAlex (and S2 where needed), merging on paper_id. Writes the enriched corpus as a
parquet under workflow/data. The original epai DB is never modified (plan section 6.1).

    python -m workflow.src.main_fetch
"""

from __future__ import annotations

from workflow.src.iwf import paths
from workflow.src.iwf.config import Settings, load_settings
from workflow.src.iwf.data import sources
from workflow.src.iwf.data.dataset import save_parquet

# ===== CONFIG (defaults come from .env.workflow; edit here for a one-off run) =====
ENRICH_OPENALEX = True
# =================================================================================


def main(settings: Settings | None = None) -> None:
    s = settings or load_settings()
    paths.ensure_dirs()
    print(f"[fetch] dataset_id={s.dataset_id}  years={s.year_from}-{s.year_to}  "
          f"max_papers={s.max_papers}")

    corpus = sources.assemble(
        year_from=s.year_from, year_to=s.year_to, max_papers=s.max_papers,
        enrich_openalex=ENRICH_OPENALEX,
    )
    out = paths.raw_parquet(s.dataset_id)
    save_parquet(corpus, out)
    have_fwci = corpus["fwci"].notna().sum() if "fwci" in corpus.columns else 0
    print(f"[fetch] {len(corpus)} papers  |  fwci present: {have_fwci}")


if __name__ == "__main__":
    main()
