# `denario_hc` — Denario human-control holdout

Builds `data/parquet/denario_hc.parquet`: the 80 Denario AI-written paper
abstracts **plus 80 matched real human abstracts** (one per Denario paper),
balanced like every training corpus so the holdout yields a real ROC / AUROC /
FPR instead of a detection-rate-only table.

`hc` = **human control**.

## Why

The existing `denario` holdout (`data/parquet/denario_detect.parquet`) is
all-machine, so it can only report recall — no FPR, no ROC — and every
"detection collapses on real AI papers" claim in `info/iclr27_plan.html` is
unfalsifiable on it. This adds the negative class.

Two asymmetries between the classes are removed so the detector reads *authorship*
and not *provenance*:

* **extraction pipeline** — the Denario abstracts came out of PDFs (run-together
  words, lost ligatures, `*` markdown, spaces before punctuation). `ocr_clean.py`
  repairs these in **both** classes, like `standardize_text.py`.
* **typography / whitespace** — `standardize_text.py --whitespace flat`, the same
  stage and setting `science_v3` used.

The 80 human abstracts are checked against the training corpus (`science_v3` by
default) on OpenAlex work id + DOI + normalised title, at fetch time and again at
build time — a matched human that is in a training corpus is a hard failure.

> A **new** training corpus (`science_v5` / Track 2, which adds Astronomy and
> Neuroscience) must exclude these 80 `oa_id`s from *its own* fetch. That is the
> other end and is not handled here. The ids are in
> `data/parquet/denario_hc_raw.parquet`, column `oa_id` (non-empty rows).

## Matching method

A **per-abstract 1:1 match**, not a distributional one: every Denario abstract
gets one designated human counterpart.

*"slug"* — a lowercased, ASCII, punctuation-free domain name, safe for
filenames / `--where` filters / figure labels, produced by
`prepare_denario_detect.py::slugify_domain` (`"Cosmology & Astrophysics"` →
`cosmology_astrophysics`). It is the value of the `domain` column. There are 9.

### `build_denario_hc.py`

1. Load the 80 Denario abstracts (each carries a domain slug and a word count)
   and the candidate pool.
2. Embed all 80 + every candidate with SPECTER2, L2-normalise.
3. **Group by domain slug.** Everything below runs within one slug, so a
   cosmology Denario paper can only be matched to a candidate fetched under
   cosmology queries — the domain histogram is identical between the two classes
   (17 vs 17, 15 vs 15, …).
4. For a slug with `N` Denario papers and `M` candidates, build the `N×M` cost
   matrix

   ```
   cost[i,j] = w_length · |wordsᵢ − wordsⱼ| / len_scale  +  w_topic · (1 − cos(embᵢ, embⱼ))
             = 1.0     · |Δwords|            / 40         +  1.0     · (1 − SPECTER2 cosine)
   ```
5. `scipy.optimize.linear_sum_assignment` (Hungarian algorithm) → the globally
   optimal one-to-one assignment: every Denario paper gets exactly one human, no
   human reused, total cost minimised. It is optimal over the slug, not
   greedy-nearest — an individual pair may not be the single closest in length
   if that human is better spent on another paper.
6. Concatenate the 9 slugs → 80 machine + 80 human rows.

`results/denario_hc/match_report.csv` records `len_delta` and `cosine` per pair.

### What each pair is

| dimension | guarantee |
|---|---|
| **domain** | exact, per pair (matching is within-slug) |
| **length** | close, per pair — globally optimised, not greedy. Mock run: median \|Δwords\| ≈ 4–5, p90 ≈ 12, worst ≈ 31. Tune with `--w-length` / `--len-scale`. |
| **topic** | nearest available in the slug by SPECTER2 cosine. Slugs are coarse and the pool is pre-2021 while Denario is 2024–25, so an exact topic twin often does not exist — it picks the closest pre-2021 paper in the same subfield. Audit via the `cosine` column. |
| **year** | *not* matched — humans 2005–2020, Denario ~2024–25. Deliberate: a post-ChatGPT abstract may be LLM-polished and would poison the "human" label. |

### Candidate pool size

`fetch_human_candidates.py` fetches **`max(min_pool, n_need × pool_factor)` per
domain** (`min_pool=60`, `pool_factor=40`), i.e. proportional — a 17-paper
domain pulls ~680 candidates, a 4-paper domain ~160, so every Denario paper gets
the same depth of choice (`pool_factor`). More than a few dozen per target rarely
changes the assignment and only costs fetch time; raise `--pool-factor` (and
re-fetch) if the match report shows weak cosines for a domain. A `SHORTFALL`
warning means OpenAlex genuinely has too few pre-2021 papers for that domain's
queries — fix the queries or accept `--allow-shortfall`.

## Files

| file | stage |
|---|---|
| `fetch_human_candidates.py` | OpenAlex candidate pool, one query set per Denario domain (`cfg/science_data_denario/queries.yaml`), pre-2021, training-overlap excluded |
| `build_denario_hc.py` | 1:1 match (SPECTER2 topic + abstract length, optimal assignment), assemble the balanced parquet |
| `ocr_clean.py` | repair PDF-extraction artifacts in both classes |
| `exclude.py` | training-overlap guard (work id / DOI / title) |

Step 5 scores with the shared **detector suite** — `cfg/detector_suite.yaml`
(models) / `cfg/detector_suite.md` (rationale). Every dataset uses the same set,
so `denario_hc` numbers are comparable with `science_v3` / `ghostbuster_*`.

## Run

```bash
# ── 1. candidate pool from OpenAlex ────────────────────────────────────────────
#    per Denario domain, pre-2021, excludes anything already in science_v3
#    (~10-20 min; anonymous works, OPENALEX_MAILTO in .env speeds it up)
.venv/bin/python -m epai.science_data_denario.fetch_human_candidates --yes
#    -> data/science_denario/human_candidates.parquet
#    if a small domain (biology/chemistry/medicine) reports SHORTFALL:
#    re-run with --pool-factor 80  or widen that domain in cfg/science_data_denario/queries.yaml

# ── 2. match 80 humans 1:1 to the 80 Denario papers, assemble the parquet ──────
#    SPECTER2 (downloads the model on first run), optimal length+topic assignment.
#    --strip-structured-labels removes "Background:"/"Methods:" from both classes
#    (a no-op on the Denario side); keep it on — the medicine/biomedical domains
#    always carry a few, and the machine class never does.
.venv/bin/python -m epai.science_data_denario.build_denario_hc --gpu 0 --strip-structured-labels
#    -> data/parquet/denario_hc_raw.parquet
#    -> results/denario_hc/match_report.csv   ← review: |len delta| and cosine per pair

# ── 3. OCR / PDF-artifact repair (both classes) ───────────────────────────────
#    DRY RUN FIRST — it prints every word-split it would make; eyeball them.
#    Pin any real word it splits wrongly with --keep-word (repeatable).
.venv/bin/python -m epai.science_data_denario.ocr_clean data/parquet/denario_hc_raw.parquet \
    --extra-text-parquet data/science_denario/human_candidates.parquet \
    --keep-word transferability --dry-run
#    then write (same flags, add -o):
.venv/bin/python -m epai.science_data_denario.ocr_clean data/parquet/denario_hc_raw.parquet \
    --extra-text-parquet data/science_denario/human_candidates.parquet \
    --keep-word transferability \
    -o data/parquet/denario_hc_clean.parquet

# ── 4. typographic standardisation (same stage/setting science_v3 used) ───────
.venv/bin/python epai/ai_detection/prepare/standardize_text.py \
    data/parquet/denario_hc_clean.parquet --whitespace flat \
    -o data/parquet/denario_hc.parquet

# ── 5. embed + score with the full detector suite ────────────────────────────
#    --from embed runs, in order: embed (12 embedders from cfg/detector_suite.yaml)
#    → binoculars ×5 (falcon-7b, falcon3-7b, qwen2.5-7b, mistral-v0.3, llama3.1-8b)
#    → fast-detectgpt ×5 (same pairs) → radar → judges gpt56luna / qwen3-14b /
#    sonnet5 / gpt-oss:120b / llama3.3:70b → judge-reasons → cue-features.
#    First run downloads ~13 local models; GPU-bound; ~$3-5 frontier judge API.
#    Env: HF_TOKEN (Llama licence accepted) for llama3.1-8b binox+fastdetect;
#    GENAI4SCIENCE_API_KEY for the gpt-oss / llama3.3 judges (rate-limited, slow).
#    holdout has no blank text, so ANY Null in a scored column is a real gap:
#    the runner re-runs that stage (skipping filled rows) up to --fill-retries
#    times with a --fill-cooldown pause, then fails. Bump --fill-retries for the
#    flaky GenAI4Science endpoint.
.venv/bin/python epai/ai_detection/prepare/run_holdout_pipeline.py \
    --in data/parquet/denario_hc.parquet -o data/parquet/denario_hc.parquet \
    --from embed --yes --gpu 1 --cpu-threads 16 --fill-retries 4
#    a stage that dies → resume:  --from <stage>   (embed / binox / fastdetect / …)
#    just the two flaky judges:   --only judge-gptoss-120b judge-llama33-70b --fill-retries 6
#    the runner now fails a stage that leaves Null rows OR whose score column
#    came out constant / 97%-one-value; the genai4science judge also prints a
#    structured-vs-free-text-fallback breakdown per run.

# ── 5b. sanity-check the scored file (a redundant backstop now, still worth it) ─
#    per-column nulls, out-of-range, near-constant (80% WARN), missing columns.
#    RUN AS ONE LINE (a broken \ splits the args).
.venv/bin/python epai/scripts/check_parquet_completeness.py data/parquet/denario_hc.parquet --reference data/parquet/science_v3_gpt56luna_0811A.parquet
#    look hardest at llm_judge_score_gptoss_120b_v3 / _llama33_70b_v3

# ── 6. wire into the config ──────────────────────────────────────────────────
#    cfg/ai_detection/train/clf_id/clf_id_science_v3.yaml, under data.eval_only.datasets
#    (keep the existing `denario` entry, add this one):
#      - name: denario_hc
#        label: "Denario + human control"
#        parquet_path: data/parquet/denario_hc.parquet
.venv/bin/python epai/ai_detection/train/run_clf_in_domain.py \
    --cfg cfg/ai_detection/train/clf_id/clf_id_science_v3.yaml --dry-run
```

## Outputs

| path | what |
|---|---|
| `data/science_denario/human_candidates.parquet` | the OpenAlex candidate pool |
| `data/parquet/denario_hc_raw.parquet` | 80+80, pre-clean; `oa_id` column = the training-exclusion list |
| `data/parquet/denario_hc_clean.parquet` | after `ocr_clean` (`text_preclean` kept) |
| `data/parquet/denario_hc.parquet` | after `standardize_text` + scoring — the holdout |
| `results/denario_hc/match_report.csv` | per-pair length delta and topic cosine |

## Notes / knobs

* **Balance** — 80/80 by construction. The holdout FPR floor is 1/80 ≈ 1.25 %; read
  the transported-threshold TPR as the headline, as the plan already does.
* **`--exclude-parquet`** (repeatable, both stage 1 and 2) — training corpora the
  humans must not appear in. Default: `science_v3`. Add `science_v5` here once it
  exists. `--no-exclude-check` disables it (only valid before any training corpus).
* **`ocr_clean` word segmentation** — a run is split only when every piece is a
  real word and the run is not itself a real word / `<prefix>+word`; uses
  `/usr/share/dict/words` when present. Always `--dry-run` first and read the
  "word splits applied" list. A real term the dictionary is missing that gets
  wrongly split (e.g. `transferability`) — pin it with `--keep-word
  transferability` (repeatable). Genuine long words left whole show up under
  "spans left unsegmented" — that is correct, not an error.
* **Length control** — the per-row match already tracks Denario length. For an
  explicit fixed-length condition (Figure 6), truncate the finished holdout:
  `epai/ai_detection/prepare/truncate_text.py data/parquet/denario_hc.parquet
  --length 175 --unit words --drop-short`, re-embed/score, add as a second
  `eval_only` entry.
