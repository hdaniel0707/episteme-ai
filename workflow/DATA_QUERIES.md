# Data collection queries

Every query / API filter the collector uses to gather papers, and how to widen coverage.
This documents the code in `src/iwf/fetchers/` + `src/main_extend_db.py`; it is a reference,
not a config file. Source on/off switches live in `.env.workflow` (`ENABLE_OPENALEX`,
`ENABLE_S2`, `ENABLE_PDF`, `ENABLE_CROSSREF`, `ENABLE_ARXIV`, `ENABLE_UEFISCDI`,
`ENABLE_FILL`) and are honoured whether you run the deploy scripts or `main_extend_db`
directly.

## Cross-domain coverage — the important part

The primary OpenAlex crawl filters on `type:article,has_abstract:true` **with no field
restriction by default** (`fetch_domains: []` in `config.yaml`). An empty `fetch_domains`
means one unbucketed query across *all* OpenAlex fields — i.e. the corpus already spans the
full range of scientific domains, which is exactly what the cross-domain impact models need.

To force a *balanced* number of papers per field instead of whatever the global stream
returns, set `fetch_domains` to one OpenAlex filter string per bucket (each becomes its own
cursor-paged crawl, capped at `remaining / n_buckets`). Field ids are listed at
<https://api.openalex.org/fields> (e.g. computer science = `fields/17`, medicine =
`fields/27`):

```yaml
# config.yaml — one entry per domain bucket for a balanced cross-domain corpus
fetch_domains:
  - primary_topic.field.id:fields/17   # Computer Science
  - primary_topic.field.id:fields/27   # Medicine
  # ... add the fields you want balanced; see the /fields endpoint for all ids
```

`fetch_year_from` / `fetch_year_to` (config.yaml, fetch time) bound the crawl years;
`TRAIN_YEAR_FROM` / `TRAIN_YEAR_TO` (.env.workflow, default 1700–2024) bound which of the
collected years survive into `training.db`.

## OpenAlex — primary crawl (`src/iwf/fetchers/openalex.py`)

- Base filter (`_base_filter`, `openalex.py:121-129`):
  `type:article,has_abstract:true[,publication_year:<from>-<to>]`, with each `fetch_domains`
  entry appended as `,<filter>` (`rows`, `:131-133`).
- Selected fields (`SELECT`) and paging use the works endpoint `https://api.openalex.org/works`
  with cursor paging (`per-page`, `next_cursor`), ~9 req/s polite-pool rate (`:140-149`).
- Abstracts are reconstructed from OpenAlex's inverted index; `fwci` and
  `citation_normalized_percentile` arrive as free impact labels.
- Author h-index gap-fill batches author ids to the authors endpoint (`fill_author_h_index`,
  `openalex.py:152+`).

## Semantic Scholar (S2) — gap-fill by DOI (`src/iwf/fetchers/s2.py`)

- Batch endpoint `https://api.semanticscholar.org/graph/v1/paper/batch`
  (`BATCH_URL`, `s2.py:19`), fields `externalIds,influentialCitationCount,embedding`
  (`FIELDS`, `s2.py:20`), 100 ids/request keyed as `DOI:<doi>` (`fill_from_s2`, `:29-48`).
- Only rows already sourced from `openalex` / `uefiscdi_journal` with a DOI and a missing
  `influential_citation_count` or `embedding` are queried (`s2.py:33-37`) — S2 fills gaps, it
  does not seed new papers here.

## Metadata gap-fill by title search (`src/iwf/fetchers/enrich_metadata.py`)

- OpenAlex title search: `title.search:<title>` on the works endpoint
  (`search_openalex_by_title`, `enrich_metadata.py:38-92`).
- S2 title search: the paper-search endpoint
  `https://api.semanticscholar.org/graph/v1/paper/search` with the title as the query
  (`search_s2_by_title`, `:95-146`).
- Titles are normalized (`_normalize_title`, `:28-35`) for match verification; used to fill
  DOI/citations/year on incomplete archive rows before `training.db` is derived.

## UEFISCDI / JCR journal seeding (`src/iwf/fetchers/uefiscdi.py`)

- Per-journal OpenAlex ISSN query:
  `primary_location.source.issn:<issn>,type:article,has_abstract:true[,publication_year:...]`
  (`uefiscdi.py:143-145`), capped at `fetch_per_journal_cap` (config.yaml, default 300).
- The red/yellow-zone journal ISSNs come from the yearly AIS xlsx lists dropped into
  `workflow/data/uefiscdi/`; the PDF extraction step (`uefiscdi_pdf.import_journal_data`)
  populates the `journal_data` table (AIS / quartile), which also feeds the `venue_ais`
  baseline.

## Crossref (bootstrap) (`src/iwf/fetchers/crossref.py`)

- `type:journal-article` with year filters (`crossref.py:37-41`).

## arXiv (bootstrap) (`src/iwf/fetchers/arxiv.py`)

- Categories crawled: `cs,stat,physics,q-bio,econ,math` (`arxiv.py:31`), one query per
  category as `search_query=cat:<category>*` (`arxiv.py:48`).

## LLM auto-fill (extend phase, off/on by default per sub-step — see `.env.workflow`)

`src/iwf/fetchers/llm_fill.py`, called from `cmd_extend` after `fill`:
- **Domain fill** (`ENABLE_LLM_FILL`, default on): rows with a title but no `domains` get one
  picked by the LLM from `src/iwf/domains.py`'s `ALLOWED_DOMAINS` (OpenAlex's ~26-field
  taxonomy — the same partition `fetch_domains` filters against). Never changes `source`.
- **Abstract/year fill** (`ENABLE_LLM_ABSTRACT_FILL`, default off): rows with a title +
  citation_count but a missing abstract/year get one from the LLM's own knowledge (no live
  web search — `ENABLE_LLM_WEBSEARCH` is a stub, no search tool is wired in). An abstract
  fill sets `source='llm'`; excluded from `training.db` unless `USE_LLM_ABSTRACTS_FOR_TRAINING=true`.

Every call degrades to a no-op (row left unchanged) without `LLM_JUDGE_PROVIDER`/
`LLM_JUDGE_MODEL`/`LLM_JUDGE_API_KEY`/`LOCAL_LLM_URL` configured in `.env.workflow` — the
extend phase never fails for a missing LLM endpoint.

## Provenance (`source` column)

Every row records where it came from: `openalex`, `s2` / `s2_original[_<db>]`,
`uefiscdi_journal`, `arxiv`, `crossref`, `pdf` (UEFISCDI/JCR import), and — when LLM
auto-fill is enabled (see `.env.workflow`) — `llm` for LLM-supplied abstracts. The analysis
"source productivity" plot collapses these to the three real acquisition channels:
**s2**, **openalex**, **pdf**.
