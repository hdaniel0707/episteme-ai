# PLAN — Baselines, Novelty/Feasibility, and Database Extension for the `workflow/` pipeline

> Status: SPECIFICATION ONLY — no code written. Target branch: `workflows` (all new code goes
> under `workflow/`; the professor's `epai/` code is READ-ONLY and must never be edited).
> This document is written as a code-generation spec: a cheaper LLM should be able to
> implement each module from its section alone.

---

## 0. Findings — what already exists (read before implementing)

### 0.1 Professor's repo (`main` branch, `epai/`)

| Asset | Path | What it is |
|---|---|---|
| **NAID v2.1 scorer ("NAIP")** | `epai/score/add_naip_scores.py` | Calls the HuggingFace Gradio Space `ssocean/Newborn_Article_Impact_Predict` (model versions `v1` / `v2.1`) with **title + abstract only**. Returns `naip_raw_score` and `naip_final_score` (= normalized **TNCSI_sp**, 0–1). Resume-safe parquet caching. **This is a frozen, externally trained impact predictor — exactly the baseline class we need.** |
| NAIP analysis plots | `epai/analyse/analyse_naip.py` | Histograms per `generated_by` group + Pearson-r scatter of NAIP score vs `log(1+citations)` and yearly citations. Good style reference for our comparison plots. |
| Binoculars scorer | `epai/score/add_binoculars_scores.py` | AI-text **detection**, NOT impact — irrelevant as an impact baseline. |
| Fetcher (Semantic Scholar) | `epai/fetch/fetch.py`, `epai/utils/s2_utils.py` | S2 Graph API search → SQLite. Fields: paperId, title, abstract, SPECTER embedding, authors(+hIndex), citationCount, influentialCitationCount, year, venue. |
| DB schema | `epai/utils/sqlite_utils.py` | Tables `papers(paper_id PK, title, abstract, embedding JSON, citation_count, influential_citation_count, year, venue, queries JSON, domains JSON, generated_by)`, `authors(author_id PK, name, h_index, citation_count, paper_count, affiliations)`, `paper_authors(paper_id, author_id)`. |

**Answer to the main question:** YES — the professor's repo contains one integrated,
not-our-trained impact predictor: **NAID/NAIP v2.1** (AAAI'25 "From Words to Worth",
LLaMA-3 + LoRA fine-tuned on the NAID dataset, TNCSI_sp target, title+abstract input).
No other external impact model is integrated on `main`.

### 0.2 Our `workflows` branch (`workflow/`)

Already present (do NOT re-implement):
- `workflow/src/iwf/features/external_impact.py` — pluggable frozen scorer registry:
  `naip` (fully wired, same HF Space), `forecite` + `cited_potential` (generic REST stubs
  enabled via `FORECITE_URL` / `CITED_POTENTIAL_URL` env). JSON score cache per `paper_id`.
- `workflow/src/iwf/features/novelty.py` — `novelty_feasibility(Z, k)` = kNN cosine
  geometry on the reduced latent space: novelty = 1 − max-sim, feasibility = mean-sim.
- `workflow/src/iwf/features/feature_builder.py` — fuses z + metadata + novelty/feasibility
  + external scores with a neutral-zero + `*_missing`-mask contract and a leakage guard.
- `workflow/config.yaml` — knobs `use_novelty`, `use_feasibility`,
  `external_impact: []`, `metadata_fields`, 8 workflow topologies, `runs:` queue.
- `workflow/src/iwf/targets.py` — 4 impact targets in [0,1]: `field_year_percentile`,
  `fwci_percentile`, `normalized_percentile`, `bayesian_impact`; percentile/CSS class binning.
- `workflow/src/iwf/evaluation/plots.py` — full plot suite (correctness bars, MSE bars,
  confusion matrices, UMAP suites, seed spread). **No baseline-comparison plots yet.**

**Known defects to fix first (P0):**
1. `workflow/src/iwf/data/` package is **missing** from the tree, yet it is imported by
   `main_fetch.py` (`iwf.data.sources`, `iwf.data.dataset`) and `external_impact.py`
   (`iwf.data.identity.PAPER_ID`). Phase 0 and external scoring currently cannot run.
2. `external_impact: []` default means no run ever computes baseline scores; no plot
   consumes them.

---

## 1. Baseline models (frozen, title+abstract-only) — the comparison set

Selection criteria: (a) never trained by us, (b) accepts title+abstract only,
(c) reproducible (public weights/API/deterministic), (d) outputs a scalar mappable to [0,1].

| # | Baseline id | What | Access | Output → [0,1] mapping |
|---|---|---|---|---|
| 1 | `naip` | NAID v2.1 (LLaMA-3 LoRA, AAAI'25) | HF Space `ssocean/Newborn_Article_Impact_Predict` (already wired); weights at github.com/ssocean/NAIP for local batch | `naip_final_score` already = TNCSI_sp ∈ [0,1] |
| 2 | `specter2_knn` | SPECTER2 frozen embedding + **label-free kNN citation-percentile transfer**: score = mean target-percentile of the k most similar TRAIN papers (embedding frozen, no learned head — a standard "retrieval baseline") | `allenai/specter2` + `specter2_regression` adapter (HF) | rank/CDF over corpus → [0,1] |
| 3 | `llm_judge` | Zero-shot LLM impact rating: fixed prompt "rate expected scholarly impact 0–100 of this title+abstract", temperature 0, one fixed model+version (e.g. `claude-haiku-4-5` or local Llama) — the standard "LLM-as-judge" baseline in recent SDQP papers | API (batchable, cached) | score/100 → [0,1] |
| 4 | `venue_ais` | Journal-level Article Influence Score of the paper's venue (UEFISCDI "scor relativ de influență" lists / JCR AIS). Pure scientometric no-content baseline — the null hypothesis "venue explains everything" | UEFISCDI public xlsx lists (see §4) joined on ISSN/venue name | AIS percentile within year → [0,1] |
| 5 | `meta_gbm` *(optional, weakest claim to "not trained by us" — mark clearly in plots)* | Metadata-only gradient boosting (h-index, author_count, venue_freq, year) — the classic scientometric baseline every impact paper reports | trained on our TRAIN split only, no text | model output is percentile target |

Notes:
- SChuBERT/MultiSChuBERT (Groningen) predicts log-citations but needs full text /
  visual render for its headline numbers and has no turnkey weights → **excluded** from the
  default set; keep the `_EndpointModel` REST stub so it can be added later if hosted.
- ForeCite / CitedPotential stubs stay as-is (enabled only when someone hosts an endpoint).

### 1.1 Module spec — extend `workflow/src/iwf/features/external_impact.py`

Add three classes to the existing `_REGISTRY` (same `ExternalImpactModel` contract:
`score(title, abstract) -> float | None`, cached via `score_dataframe`):

```
class Specter2KnnBaseline(ExternalImpactModel):
    name = "specter2_knn"
    # NOT a per-row remote call. Override score_dataframe(df, cache_path):
    #   1. needs columns: specter2 embedding (reuse phase-1 cache via iwf.embedding), split, target
    #   2. fit sklearn NearestNeighbors(metric="cosine") on TRAIN rows only
    #   3. score(row) = mean(target of k=15 nearest TRAIN neighbours)  [row itself excluded]
    #   4. deterministic, no network. Config key: baseline_knn_k: 15
class LLMJudgeBaseline(ExternalImpactModel):
    name = "llm_judge"
    # POST to provider chosen by env LLM_JUDGE_PROVIDER (anthropic|openai|ollama),
    # model pinned by LLM_JUDGE_MODEL. Prompt template = module-level constant
    # JUDGE_PROMPT (verbatim, versioned "v1"). temperature=0, max_tokens=8.
    # Parse int 0..100 -> /100. On parse failure return None. Cache key includes prompt version.
class VenueAISBaseline(ExternalImpactModel):
    name = "venue_ais"
    # Needs the UEFISCDI AIS table (see fetchers §4.3): venue/ISSN -> AIS per JCR year.
    # score_dataframe override: join df.venue (normalized lowercase, fuzzy fallback via
    # rapidfuzz token_sort_ratio>=95) x df.year -> AIS; then percentile-rank within year.
    # Missing venue -> None (mask bit handles it).
```

Config: `external_impact: [naip, specter2_knn, llm_judge, venue_ais]` becomes the
documented default for `emb_meta_*` runs; each entry both (a) becomes a fused metadata
feature via `feature_builder.EXTERNAL_COLS` (append the new names) and (b) is logged
to `results/<run>/baselines.parquet` (columns: paper_id, split, one column per baseline)
for the comparison plots. That satisfies "baselines added to metadata and all-metadata models".

---

## 2. Baseline comparison plots (phase 7 additions)

New module: `workflow/src/iwf/evaluation/baseline_compare.py`.
Called from `generate_workflow_plots()` when `results/<run>/baselines.parquet` exists.
All figures PNG (dpi=150, matplotlib, same style as `plots.py`) into
`results/<run>/aggregate/baselines/`.

Per trained model (i.e. per workflow run), exactly **2 headline figures**:

**Figure 1 — `baseline_agreement_<run>.png` (the "% same category" plot)**
- Bin every score (ours + each baseline) into the run's class bins (`class_thresholds`,
  default quartiles Q4..Q1) computed **on the test split**.
- Grouped bar chart, x = scorer (`ours`, `naip`, `specter2_knn`, `llm_judge`, `venue_ais`),
  bars = (a) % exact same class as TRUE class, (b) % within ±1 class; annotate each bar
  with the % value and Cohen's quadratic-weighted κ underneath.
- Include a horizontal dashed line = random-agreement level (1/n_classes).
- Function: `plot_baseline_agreement(y_true_cls, pred_cls_by_scorer: dict[str, np.ndarray], class_names, path)`.

**Figure 2 — `baseline_correlation_<run>.png` (the "does good = good elsewhere" plot)**
- Left panel: heatmap of pairwise **Spearman ρ** among {true target, our model score,
  each baseline score} on the test split (annotated cells, diverging colormap, range −1..1).
- Right panel: scatter our-score (x) vs best-external-baseline (y = naip), colored by true
  class, with identity line + ρ in title. This is the direct visual proof that "a good
  abstract under one predictor is good under another".
- Function: `plot_baseline_correlation(scores: pd.DataFrame, class_labels, path)` where
  `scores` columns = [target, ours, naip, specter2_knn, llm_judge, venue_ais].

Cross-run summary (run by `run_all_workflows` after the queue):
`results/summary/baseline_leaderboard.png` — one grouped bar chart: x = every trained
model in `runs:`, grouped bars = Spearman ρ with true target for that model and each
baseline (baselines constant across runs → horizontal reference lines). Shows in one
image whether each of our models beats every baseline.

Metrics helper (same module):
`baseline_metrics(y_true, scores_by_scorer) -> pd.DataFrame` with rows per scorer and
columns [spearman, pearson, kendall, mae, exact_class_pct, within1_class_pct, weighted_kappa]
→ saved as `baseline_metrics.csv` next to the figures (feeds the thesis tables).

---

## 3. Novelty & feasibility (refine existing + correlation analysis)

### 3.1 Novelty — upgrade to spherical geometry (keep old as fallback)

Current: `novelty = 1 − max cosine sim to kNN`. Requested: **geodesic (spherical) distance**
in the reduced embedding space. Literature anchor: Shibayama, Yin & Matsumoto (2021)
*Measuring novelty in science with word embedding* (PLoS One) — novelty as embedding-space
distance; and the Uzzi et al. atypical-combinations tradition.

Spec — modify `workflow/src/iwf/features/novelty.py` (backward compatible):
```
def novelty_spherical(Z, k=10) -> np.ndarray:
    # 1. L2-normalize rows -> points on unit hypersphere
    # 2. kNN (cosine); geodesic distance d = arccos(clip(sim, -1, 1)) in [0, pi]
    # 3. raw_novelty_i = mean geodesic distance to the k nearest neighbours
    # 4. return percentile-rank (scipy.stats.rankdata / n) -> uniform [0,1]
```
Config: `novelty_kind: spherical  # spherical | cosine_max (legacy)` ; k stays `novelty_k: 10`.
Percentile transform guarantees the required "0–1 scale so every case is comparable".

### 3.2 Feasibility — replace density proxy with a grounded composite

Requested meaning: "how plausible is it that the paper's claims can be realized and tested".
No off-the-shelf frozen model outputs this from title+abstract; the closest scientific
traditions are (a) literature-supported claim feasibility (Matter-of-Fact benchmark, 2025:
feasibility = consistency with prior published findings) and (b) replicability prediction
(structured-groups PNAS work). Both reduce to: **claims near well-established prior work are
feasible; claims far from any precedent are risky.**

Composite spec — new file `workflow/src/iwf/features/feasibility.py`:
```
def feasibility(Z, years, k=10, weights=(0.5, 0.3, 0.2)) -> np.ndarray:
    # component 1: PRECEDENT (w=0.5) - mean cosine sim to the k nearest neighbours
    #              among papers with year < own year ("prior art exists").
    #              First-year papers fall back to global kNN.
    # component 2: DENSITY (w=0.3)  - current mean-sim measure (well-populated region
    #              = methods/community exist). Reuse novelty.py internals.
    # component 3: MATURITY (w=0.2) - fraction of the k neighbours older than the paper
    #              (a paper surrounded mostly by older work builds on settled ground).
    # each component percentile-ranked to [0,1] first, then weighted sum, then
    # percentile-ranked AGAIN -> final feasibility in uniform [0,1].
```
Optional (flagged `feasibility_llm: false`): add an `llm_judge`-style 0–100 rated
"testability of the abstract's claims" as a 4th component — only if API budget allows;
keep OFF by default so the metric stays free and deterministic.
`feature_builder` keeps consuming a column named `feasibility` — no contract change.
Label-free ⇒ leakage-safe ⇒ computable on all rows (same guarantee as now).

### 3.3 Correlation analysis — are novelty/feasibility useful?

New module `workflow/src/iwf/evaluation/nf_analysis.py`, run in phase 7:
1. `nf_correlation_matrix.png` — Spearman heatmap over test split of
   {target, novelty, feasibility, naip, meta_h_index, meta_venue_freq} + significance
   stars (p<.05/.01/.001, scipy). Also **partial** Spearman of novelty↔target and
   feasibility↔target controlling for venue_freq + h_index (pingouin or manual
   rank-regression residuals) — this answers "is the signal real or just venue".
2. `nf_binned_impact.png` — mean target (with 95% CI) per novelty decile and per
   feasibility decile, two panels; the classic inverted-U novelty check (Uzzi).
3. **Ablation** (config-level, no new code): document in `runs:` recipe three entries —
   `emb_meta_*` with `use_novelty/use_feasibility` = (false,false), (true,false),
   (true,true) — and add `nf_ablation_delta.png` to the cross-run summary: bars of
   Δmetric vs the (false,false) run. Answers "do these help metadata models predict impact".

---

## 4. Database extension to ~100k rows (fetchers)

Principles: professor's DBs are never modified; new DB `workflow/data/extended.db` uses the
**identical** `papers/authors/paper_authors` schema (copy the DDL from
`epai/utils/sqlite_utils.py`, do not import it — repo isolation) **plus** new nullable
columns on `papers`: `doi TEXT`, `source TEXT` (s2|openalex|uefiscdi_journal), `fwci REAL`,
`citation_normalized_percentile REAL`, `is_in_top_1_percent INT`, `is_in_top_10_percent INT`,
`subfield TEXT`, `venue_issn TEXT`, `venue_ais REAL`, `oa_id TEXT`. Same schema ⇒ the whole
existing pipeline (and epai tools) read it unchanged. Real records only — every row comes
from an API response; no generated/imputed titles or abstracts, rows without abstract are skipped.

New package `workflow/src/iwf/fetchers/`:

### 4.1 `base.py`
```
class BaseFetcher(ABC):
    def rows(self) -> Iterator[dict]         # yields normalized paper dicts
    # shared: insert_paper(conn, row) with INSERT OR IGNORE on paper_id,
    # dedup ALSO by lowercase DOI and by (normalized title, year) hash;
    # progress tqdm; resume via a fetch_state table (source, cursor, fetched_at).
```
Normalized dict contract (the ONLY thing fetchers produce):
`{paper_id, doi, title, abstract, year, venue, venue_issn, citation_count,
influential_citation_count, fwci, citation_normalized_percentile, is_in_top_1_percent,
is_in_top_10_percent, domains: list[str], subfield, authors: list[{author_id, name,
h_index, citation_count, paper_count}], source}`.

### 4.2 `openalex.py` — the workhorse to 100k
- Endpoint `https://api.openalex.org/works`, cursor paging (`cursor=*`), 200/page,
  `mailto` param from env `OPENALEX_MAILTO` (polite pool), ~10 req/s cap, retry 429 w/ backoff.
- `select=` id, doi, title, publication_year, primary_location, abstract_inverted_index,
  cited_by_count, fwci, citation_normalized_percentile, topics, authorships.
- Abstract reconstruction from `abstract_inverted_index` (positions → word list → join).
- Filters from a new config block (`workflow/config.yaml` §7 DATA EXTENSION):
  `fetch_year_from/to`, `fetch_domains` (OpenAlex topic/field ids mirroring the corpus
  domains), `fetch_per_domain_cap` (default 100000 // n_domains — keeps field balance so
  `field_year_percentile` ref-sets stay dense), `has_abstract:true`, type=article.
- Author h-index: 1 extra batched call per 100 unique author ids to `/authors?filter=ids...`
  → `summary_stats.h_index`. (Fills `best_author_h_index` metadata coverage.)
- **Bonus impact labels come free**: fwci + citation_normalized_percentile map directly to
  the existing `fwci_percentile` / `normalized_percentile` targets.

### 4.3 `uefiscdi.py` — Romanian classified-journal seed lists
- UEFISCDI publishes yearly public xlsx: red/yellow zone journals (Q1/Q2 by IF and by AIS)
  and "scor relativ de influență" (Article Influence Score) per journal, 2013→present
  (uefiscdi.gov.ro → Scientometrie → Reviste; CNFIS mirrors 2013–2024).
- These are **journal-level**, not paper-level ⇒ two uses:
  1. `load_ais_table(xlsx_dir) -> DataFrame[issn, journal_name, year, ais, zone]`
     (feeds the `venue_ais` baseline §1). Files are committed under
     `workflow/data/uefiscdi/` (small, public); a `download_uefiscdi.py --list-urls`
     helper prints the URLs but manual download is acceptable (site layout shifts).
  2. Seed fetching: for each red/yellow ISSN → OpenAlex
     `filter=primary_location.source.issn:<issn>,publication_year:X` → papers inherit
     `venue_ais` + `zone`; `source='uefiscdi_journal'`. Gives a quality-stratified subcorpus
     (zone = an independent, human-committee quality label usable for extra validation plots).
- No paper-level "recommended papers" list exists publicly with abstracts — do NOT fabricate;
  journal zone is the honest usable signal.

### 4.4 `s2.py` — thin Semantic Scholar fetcher (same API as epai's but living in workflow/,
uses the normalized dict; only for gap-filling `influential_citation_count` and SPECTER
embeddings for OpenAlex-sourced rows; batch endpoint `/paper/batch`, key from `S2_API_KEY`).

### 4.5 Entry point `workflow/src/main_extend_db.py`
```
python -m workflow.src.main_extend_db --sources openalex,uefiscdi --target-rows 100000
# phases: 1) copy prof DB rows into extended.db (provenance source='s2_original')
#         2) uefiscdi journal seeds  3) openalex domain-balanced fill to target
#         4) s2 gap-fill  5) report: rows per source/domain/year histogram printed +
#            saved workflow/data/extended_db_report.png
```
`main_fetch.py` gets one new setting `db_path: extended` (default stays the prof DB —
old runs reproduce byte-identically).

---

## 5. Integration order & work packages (each is one cheap-LLM prompt)

| WP | Content | Files | Depends on |
|---|---|---|---|
| **WP0** | Recreate missing `iwf/data/` package: `identity.py` (PAPER_ID="paper_id"), `dataset.py` (save/load_parquet), `sources.py` (read epai sqlite → DataFrame, optional OpenAlex enrich) — reverse-engineer exact call signatures from `main_fetch.py`, `main_prepare.py`, `external_impact.py` imports | `workflow/src/iwf/data/{__init__,identity,dataset,sources}.py` | — |
| WP1 | 3 new baseline classes + registry + `baselines.parquet` dump in phase 5/7 | `features/external_impact.py` (+small hook in `main_train.py` result saving) | WP0 |
| WP2 | Comparison plots + metrics CSV + leaderboard | `evaluation/baseline_compare.py`, hook in `plots.generate_workflow_plots`, `run_all_workflows` summary | WP1 |
| WP3 | Spherical novelty + composite feasibility + config knobs | `features/novelty.py`, `features/feasibility.py`, `config.py`, `config.yaml` | WP0 |
| WP4 | Correlation matrix, binned-impact, ablation summary plot | `evaluation/nf_analysis.py` | WP3 |
| WP5 | Fetcher package + extended DB + entry point + UEFISCDI loader | `fetchers/*`, `main_extend_db.py`, config §7 | WP0 |
| WP6 | Smoke tests: every WP gets an offline test in `src/tests/smoke.py` style (fake 50-row DataFrame, no network — external models mocked to return constants) | `src/tests/` | all |

Constraints for every WP prompt: pure additions under `workflow/`; imports of `epai.*`
forbidden (copy small snippets, cite origin in a comment); every scalar feature follows the
neutral-0 + `*_missing` mask contract; every score exposed to stats is percentile-[0,1];
all network calls cached + resume-safe (JSON/parquet cache keyed by paper_id, like NAIP).

---

## 6. References
- NAID/NAIP: arxiv.org/abs/2408.03934 (AAAI'25), github.com/ssocean/NAIP, HF Space ssocean/Newborn_Article_Impact_Predict
- SPECTER2: huggingface.co/allenai/specter2 (+ `specter2_regression` adapter), arxiv.org/abs/2004.07180
- SChuBERT/MultiSChuBERT (excluded, rationale §1): arxiv.org/abs/2308.07971
- GPT/text-embedding citation prediction: arxiv.org/abs/2407.19942
- Novelty: Shibayama, Yin, Matsumoto 2021 (PLoS One, word-embedding novelty); Uzzi et al. 2013 atypical combinations; review arxiv.org/abs/2501.17456
- Feasibility grounding: Matter-of-Fact (arxiv.org/abs/2506.04410), replicability prediction (PNAS structured groups, PMC10245209)
- UEFISCDI AIS / red-yellow lists: uefiscdi.gov.ro/scientometrie-reviste (xlsx per JCR year), cnfis.ro/clasificare-reviste-isi (2013–2024 mirror)
- OpenAlex work object (fwci, citation_normalized_percentile, abstract_inverted_index): docs.openalex.org / github.com/ourresearch/openalex-docs
