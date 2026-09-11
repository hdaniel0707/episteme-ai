# trainv2 — AI-detection experiments over the MANUGRAM benchmark

A small, from-scratch rebuild of the AI-detection training/analysis runners. The
old `epai/ai_detection/train` tree answered many questions in one run, so no
single run was legible. `trainv2` restarts from the opposite end:

* **one experiment per script**, each over the four core splits of the
  **MANUGRAM** benchmark (`splits.py`) — the MG-Reph family is opt-in;
* **two metrics** — AUROC (machine = positive, *not* orientation-corrected) and
  TPR@1%FPR. Where there is no calibration slice the TPR is **oracle**: the
  threshold is fit on the evaluation rows themselves (an upper bound on the
  operating point);
* **one split convention** — a grouped 70/30 train/test split keyed on the
  document `id`, so a human text and every rewrite of it never straddle the cut;
  seeded and repeated; reported as mean ± SD over seeds;
* **compute → CSV → plot-from-CSV** — most runners write a CSV, then draw their
  figures *from that CSV*, and take a `--replot <csv>` that redraws without
  recomputing. A PNG can never disagree with its table.

Reused from the old code: only `epai/algs` (the pluggable classifier registry)
and `epai/utils/cd_core` (metric primitives). Everything else here is
self-contained.

> Run everything from the repo root. The lines below use `uv run`;
> `.venv/bin/python` works too. Per repo policy, **ask before running any `uv`
> command yourself** — the lines here are for the maintainer to run.

---

## Scripts at a glance

| script | what it does | § |
|--------|--------------|---|
| `rebuild_all.py` | orchestrate the whole rebuild — dependency order + bounded parallelism | [Full rebuild](#full-rebuild) |
| `calibrate_judge_baseline.py`<br>`calibrate_binox_baseline.py`<br>`calibrate_fastdetect_baseline.py`<br>`calibrate_detector_baseline.py` | one fixed **1%-FPR + Youden** threshold per score column → a calibrated baseline YAML | A |
| `run_judge_youden_calibration.py`<br>`run_binox_youden_calibration.py`<br>`run_fastdetect_youden_calibration.py` | per-domain + pooled **Youden** operating points → CSV for downstream reads | B |
| `run_embedding_effectiveness.py` | which `emb_*` column separates human from AI best, per split + mean | C |
| `run_zeroshot_performance.py` | zero-shot AUROC vs. oracle TPR@1%FPR, one scatter per detector family | D |
| `run_best_zeroshot_performance.py` | the **hand-picked** best of each family (`best_zeroshot.yaml`), one figure | D |
| `run_judge_by_generator.py` | judge × generator detection matrix (AUROC + TPR@1%FPR) | E |
| `run_in_domain_performance.py` | trained detectors, random grouped 70/30 + eval-only OOD | F |
| `run_lodo_performance.py` | trained detectors, leave-one-domain-out | F |
| `split_manugram_reph.py` | split MG-Reph into its **science** / **ghostbuster** halves | [MG-Reph](#mg-reph-experiments-opt-in) |
| `plot_score_histograms.py` | 3-class (human / rephrase / rewritten) score histograms, one figure per column, panels MG-GB / MG-Sci / MG-All | [MG-Reph](#mg-reph-experiments-opt-in) |
| `rephrase_zeroshot_performance.py` | binox / fastdetect zero-shot AUROC vs. oracle TPR@1%FPR, **rephrase counted as human** | [MG-Reph](#mg-reph-experiments-opt-in) |
| `rephrase_best_zeroshot_performance.py` | best-of-family (llama31 binox/fastdetect + gpt56luna judge) on one figure, rephrase = human | [MG-Reph](#mg-reph-experiments-opt-in) |
| `rephrase_in_domain_performance.py` | trained detection (in-domain 70/30) on the rephrase corpora, rephrase = human | [MG-Reph](#mg-reph-experiments-opt-in) |
| `rephrase_lodo_performance.py` | trained detection (leave-one-domain-out) on the rephrase corpora, rephrase = human | [MG-Reph](#mg-reph-experiments-opt-in) |
| `rebuild_rephrase.py` | orchestrate the whole MG-Reph pipeline (the split + the 5 above) | [MG-Reph](#mg-reph-experiments-opt-in) |

Not run directly — `__init__.py` (rationale), `splits.py` (the registry),
`core.py` (metrics / split / seed aggregation), `data.py` (`load_multi`),
`calibrate_common.py` (the calibrate-wrapper engine), `_trained.py`
(`TRAINED_SPECS`), `_reph_trained.py` (the rephrase feature loader), `plots.py`
(every figure function).

The three trained detectors in `_trained.py`:

| key                     | plot label       | algorithm    | aux input |
|-------------------------|------------------|--------------|-----------|
| `roberta_logistic`      | `roberta·logistic` | `logistic` on `emb_roberta` | — |
| `roberta_fusion_cuevec` | `roberta⊕cuevec` | `fusion_mlp` (concatenation fusion) | `llm_judge_cuevec_gpt56luna_v3` (11-dim) |
| `roberta_gated_cuevec`  | `roberta+gated`  | `gated_mlp` (learned gate on the embedding branch) | `llm_judge_cuevec_gpt56luna_v3` |

---

## The MANUGRAM benchmark

One registry (`splits.py`), imported by every script that works across the
benchmark. The short key is what a `--splits` / `--corpora` flag takes and what
lands in a CSV's `split` column; the display name is for figure labels.

**Core splits** (`splits.DEFAULT`) — what a bare `--splits` / `--corpora`, and
`rebuild_all.py`, sweep:

| key       | parquet (`data/parquet/`)   | display  | generators (non-human)                              | rows   | domains                              | role |
|-----------|-----------------------------|----------|----------------------------------------------------|--------|--------------------------------------|------|
| `gb_orig` | `ghostbuster_orig.parquet`  | GB-orig  | `gpt35t`, `gpt35ts` (GPT-3.5, legacy)               | ~9.0k  | wp, reuter, essay                    | multi-generator |
| `mg_gb`   | `manugram_gb.parquet`       | MG-GB    | `gpt56luna`, `gptoss120b`, `gemma431b`, `llama3370b`| ~15.0k | wp, reuter, essay                    | multi-generator |
| `mg_sci`  | `manugram_sci.parquet`      | MG-Sci   | `gpt56luna`, `gptoss120b`, `gemma431b`, `llama3370b`| ~15.0k | computer_science, mathematics, sociology | multi-generator |
| `mg_aisci`  | `manugram_aiscientist.parquet`      | MG-AI-Sci   | `denario`, `sakana`, `sakanav2`, `cycleresearcher`, `zochi` | 200 (100 human) | 9 scientific domains + computer_science, ~1–80 rows each | multi-generator, 1:1-matched, **eval-only OOD** |

**Opt-in splits** — registered, but never in a default sweep; pass by name only:

| key           | parquet                        | display     | contents |
|---------------|--------------------------------|-------------|----------|
| `mg_reph`     | `manugram_reph.parquet`        | MG-Reph     | **Manugram Rephrase** — the human rows of MG-GB + MG-Sci, each paired with one moderate `gpt56luna:0908R` rephrase. ~12.0k rows, 6 domains, `source_family ∈ {ghostbuster, science}` |
| `mg_reph_sci` | `manugram_reph_sci.parquet`    | MG-Reph-Sci | the `source_family == science` half — CS / math / sociology, ~6.0k rows (3.0k pairs) |
| `mg_reph_gb`  | `manugram_reph_gb.parquet`     | MG-Reph-GB  | the `source_family == ghostbuster` half — essay / reuter / wp, ~6.0k rows (~3.0k pairs) |

`manugram_reph_{sci,gb}.parquet` are produced by `split_manugram_reph.py`
([MG-Reph experiments](#mg-reph-experiments-opt-in)).

Fixed left-to-right order everywhere: `gb_orig → mg_gb → mg_sci → mg_reph →
mg_reph_sci → mg_reph_gb → mg_aisci`, then the pooled column (`all` / `mean`) last.

**Single-generator caveat.** The `MG-Reph` family has one non-human label
each, so the multi-generator scripts skip it: `run_judge_by_generator.py`
needs ≥2 generators, and `data.load_multi` rejects a file with fewer than two.
`MG-AI-Sci` technically has five now (`denario`, `sakana`, `sakanav2`,
`cycleresearcher`, `zochi`, since the 2026-09-11 merge with the ai_scientist
corpus) but stays out of the multi-generator scripts by default too — it is a
1:1-matched eval-only holdout (one human per ONE specific generator, not one
human x every generator), so most of those five have only a handful of rows
(zochi: 1, sakanav2: 3, cycleresearcher: 6) — too sparse for a real
per-generator matrix. Column-based scripts (calibration, embedding
effectiveness, zero-shot, trained) treat it as *human vs. all-AI* and run
fine. `MG-AI-Sci`'s 100 human rows (was 80 pre-merge) sit right at the
reachability boundary for a 1%-FPR threshold (`n_human * fpr >= 1.0`) rather
than clearly below it — its TPR@1%FPR is still effectively a **floor**, not a
stable measurement, even though it may now register as `reachable = True` in
the CSV.

**Why MG-Reph is opt-in.** One generator, and only the **`gpt56luna` judge** is
scored on both classes — the `gemma4_31b` / `qwen3_14b` / `gptoss_120b` /
`llama33_70b` judge columns are populated on the carried human rows but **null
on the rephrase rows**, so a human-vs-rephrase reading with those judges has no
machine class. Binoculars, Fast-DetectGPT, RADAR, the `gpt56luna` judge, every
`emb_*`, and the `gpt56luna` cue vector are all fully populated. Folding MG-Reph
into the pooled calibrations or the multi-judge runs would pull in those null
columns, so it stays out of the default sweep and is used only in the targeted
experiments below.

---

## Experiments

### A. Baseline threshold calibration → `cfg/ai_detection/trainv2/*_baseline_calibrated.yaml`

One **fixed** threshold per score column, pooled over domains, so the *same*
cutoff transports to every corpus a later comparison looks at. Two operating
points per column: the deployable **1%-FPR** cut (written as `threshold`, read
by `cd_core.baseline_entries()`) and **Youden's J** (`threshold_youden`,
recorded). Output block shape is the one `cd_core.BASELINE_KINDS` already reads,
so each YAML drops into a real `run_clf_cross_domain.py` training config
unchanged.

| script | family | column prefix | machine = | YAML block |
|--------|--------|---------------|-----------|------------|
| `calibrate_binox_baseline.py`      | Binoculars pairs   | `binoculars_score`  | low  | `binoculars_baseline` |
| `calibrate_fastdetect_baseline.py` | Fast-DetectGPT pairs | `fastdetect_crit_` | high | `fastdetect_baseline` |
| `calibrate_detector_baseline.py`   | pretrained detector | `detector_score_`  | high | `detector_baseline` |
| `calibrate_judge_baseline.py`      | LLM judges          | `llm_judge_score_` | low  | `llm_judge_baseline` |

```bash
# Default: calibrate on manugram_sci.parquet, pooled over its domains.
uv run python epai/ai_detection/trainv2/calibrate_judge_baseline.py

# --corpora all: calibrate per core split AND on all of them pooled ('all'),
# print a per-corpus table, drop one distribution PNG per column, write the YAML
# from the pooled 'all' row. This is the mode a full rebuild uses.
uv run python epai/ai_detection/trainv2/calibrate_judge_baseline.py      --corpora all
uv run python epai/ai_detection/trainv2/calibrate_binox_baseline.py      --corpora all
uv run python epai/ai_detection/trainv2/calibrate_fastdetect_baseline.py --corpora all
uv run python epai/ai_detection/trainv2/calibrate_detector_baseline.py   --corpora all

# calibrate on a specific corpus (incl. an MG-Reph half) → own -o, not the shared yaml
uv run python epai/ai_detection/trainv2/calibrate_judge_baseline.py \
    --corpora mg_reph_sci --cols llm_judge_score_gpt56luna_v3 \
    -o cfg/ai_detection/trainv2/reph_sci_judge_baseline.yaml

# --generators: restrict the machine class to one generator model (matched on
# the part before ':'). Reference calibration comparable to the single-generator
# rephrase experiments; writes to a separate dir, never touches cfg/.
uv run python epai/ai_detection/trainv2/calibrate_judge_baseline.py      --corpora all --generators gpt56luna
uv run python epai/ai_detection/trainv2/calibrate_binox_baseline.py      --corpora all --generators gpt56luna
uv run python epai/ai_detection/trainv2/calibrate_fastdetect_baseline.py --corpora all --generators gpt56luna
uv run python epai/ai_detection/trainv2/calibrate_detector_baseline.py   --corpora all --generators gpt56luna

# other options: a single positional parquet, --cols, --fpr-target 0.05, --no-plots
```

Outputs: the YAML under `cfg/ai_detection/trainv2/`, plus
`results/trainv2/baseline_calibration/<script-name>/thresholds.csv` and one
`<column>.png` per score column. `sonnet5` is excluded from the judge baseline
by default (absent on MG-Sci); pass it via `--cols` to include it. With
`--generators gpt56luna` everything (CSV, PNGs **and** YAML) goes to
`results/trainv2/baseline_calibration_[flag=gpt56luna]/<script-name>/` instead —
the shared `cfg/` YAML is left untouched.

### B. Per-domain Youden calibration (CSV of operating points)

The older-style calibration: for one positional parquet, fit Youden's J per
domain **and** pooled, report AUROC / threshold / realised TPR-FPR, plot the
score distributions, and write one CSV row per `(column, domain)` — meant to be
*read* by a later performance run so the operating point is fit once here rather
than drifting inside every experiment. Calibration, not evaluation: fit and
reported on the same rows.

| script | family | output dir (`results/trainv2/`) |
|--------|--------|--------------------------------|
| `run_binox_youden_calibration.py`      | Binoculars     | `binox_calibration/` |
| `run_fastdetect_youden_calibration.py` | Fast-DetectGPT | `fastdetect_calibration/` |
| `run_judge_youden_calibration.py`      | LLM judges     | `judge_calibration/` |

```bash
uv run python epai/ai_detection/trainv2/run_judge_youden_calibration.py data/parquet/manugram_sci.parquet
uv run python epai/ai_detection/trainv2/run_binox_youden_calibration.py data/parquet/manugram_gb.parquet --cols binoculars_score binoculars_score_qwen25_7b
uv run python epai/ai_detection/trainv2/run_fastdetect_youden_calibration.py data/parquet/manugram_aiscientist.parquet --no-plots
```

Outputs: `<stem>_thresholds.csv` and `fig/<stem>/<column>.png`.

### C. Representation analysis — `run_embedding_effectiveness.py`

Which `emb_*` column separates human from AI best, on each split and on average.
Algorithm fixed (L2 logistic on standardised features); every embedding common
to the chosen splits is swept; one binary problem per split (human vs. every AI
row pooled); grouped 70/30, N seeds, AUROC and TPR@1%FPR as mean ± SD.

```bash
uv run python epai/ai_detection/trainv2/run_embedding_effectiveness.py
uv run python epai/ai_detection/trainv2/run_embedding_effectiveness.py --splits mg_sci mg_gb --seeds 0 1 2 3 4
uv run python epai/ai_detection/trainv2/run_embedding_effectiveness.py --replot results/trainv2/embedding_effectiveness/summary.csv
```

Outputs (`results/trainv2/embedding_effectiveness/`): `per_seed.csv`,
`summary.csv` (per split + a `mean` row whose SD is the spread *across* splits),
`by_split.png`, `ranking.png`.

### D. Zero-shot detector performance

No training, no calibrated cutoff. AUROC (oriented) vs. **oracle** TPR@FPR
(threshold refit on each split's own ROC).

All four performance runners (this section + F) take `--fpr` as a **list** of
oracle FPR levels — default `0.01 0.05`. The `performance.csv` gains an `fpr`
column and every figure is emitted once per level, suffixed `_1pct` / `_5pct`.
For the trained runners the models are fit once; the extra level is only a
different ROC readout. (`run_dr_transfer.py` keeps a scalar `--fpr`.)

**`run_zeroshot_performance.py`** — every score column of each family
(Binoculars / Fast-DetectGPT / judge / pretrained detector), one scatter per
family: colour = detector version, marker = split, bold circle = mean across
splits.

```bash
uv run python epai/ai_detection/trainv2/run_zeroshot_performance.py
uv run python epai/ai_detection/trainv2/run_zeroshot_performance.py --families judge --splits mg_gb gb_orig
uv run python epai/ai_detection/trainv2/run_zeroshot_performance.py --replot results/trainv2/zeroshot_performance/performance.csv
```
Outputs (`results/trainv2/zeroshot_performance/`): `performance.csv`, and per
FPR level `{binoculars,fastdetect,judge,detector}_{1pct,5pct}.png` (+ `_goodness_`).

**`run_best_zeroshot_performance.py`** — the **hand-picked** best of each family
(a flat `columns:` list in `cfg/ai_detection/trainv2/best_zeroshot.yaml`), all
on one figure.

```bash
uv run python epai/ai_detection/trainv2/run_best_zeroshot_performance.py
uv run python epai/ai_detection/trainv2/run_best_zeroshot_performance.py --replot results/trainv2/best_zeroshot_performance/performance.csv
```
Outputs (`results/trainv2/best_zeroshot_performance/`): `performance.csv`,
`best_zeroshot_{1pct,5pct}.png` (+ `_goodness_`).

### E. Judge × generator — `run_judge_by_generator.py`

For every `(judge, generator)` pair on the three multi-generator splits
(`gb_orig`, `mg_gb`, `mg_sci`): human rows vs. *that one* generator, every other
generator excluded. Two oracle matrices — AUROC and TPR@1%FPR. Cells where judge
and generator are the same model family are starred (`*`) — the reason the
matrix exists.

```bash
uv run python epai/ai_detection/trainv2/run_judge_by_generator.py
uv run python epai/ai_detection/trainv2/run_judge_by_generator.py --splits mg_gb mg_sci
uv run python epai/ai_detection/trainv2/run_judge_by_generator.py --replot results/trainv2/judge_by_generator/matrix.csv
```
Outputs (`results/trainv2/judge_by_generator/`): `matrix.csv`, `overall.csv`,
`auroc.png`, `tpr_at_1pct_fpr.png`. `sonnet5` dropped by default; `--cols` to
include.

### F. Trained detection

The three `_trained.py` detectors, AUROC vs. oracle TPR@1%FPR, with the
`best_zeroshot.yaml` picks overlaid. Every fitted model is also scored,
unchanged, on each `--ood-splits` corpus in full (default `mg_aisci`) — never
trained on.

The zero-shot picks are scored on **the same rows as the trained models** —
each seed's 70/30 test partition (in-domain) or the held-out domain fold
(LODO), the full corpus per OOD split — not on the whole dataset. So every
per-corpus point (bar the OOD zero-shot one, a single deterministic reading)
carries a seed / fold SD, and the goodness bar folds AUROC + TPR into one score
*per replicate* so its bar carries an SD too (`goodness` / `goodness_std` in the
CSV).

**`run_in_domain_performance.py`** — random grouped 70/30 within each training
split (default `gb_orig mg_gb mg_sci`), N seeds, human vs. all-AI pooled.

```bash
uv run python epai/ai_detection/trainv2/run_in_domain_performance.py
uv run python epai/ai_detection/trainv2/run_in_domain_performance.py --splits mg_gb mg_sci --ood-splits mg_aisci --seeds 0 1 2 3 4
uv run python epai/ai_detection/trainv2/run_in_domain_performance.py --replot results/trainv2/in_domain_performance/performance.csv
```
Outputs (`results/trainv2/in_domain_performance/`): `performance.csv`,
`in_domain_{1pct,5pct}.png` (+ `_goodness_`, `_denario_`).

**`run_lodo_performance.py`** — leave-one-domain-out instead of a random split:
each domain of a split is held out in turn, trained on the rest, scored on the
held-out domain in full. Per-split SD is the spread *across* domain folds
(domain-transfer robustness) — for the zero-shot picks too, scored on each of
those same folds. Only splits with ≥2 domains of ≥`--min-domain-rows`
(default 100) rows qualify — so `gb_orig mg_gb mg_sci`, 3 domains each.

```bash
uv run python epai/ai_detection/trainv2/run_lodo_performance.py
uv run python epai/ai_detection/trainv2/run_lodo_performance.py --seeds 0 1 2
uv run python epai/ai_detection/trainv2/run_lodo_performance.py --replot results/trainv2/lodo_performance/performance.csv
```
Outputs (`results/trainv2/lodo_performance/`): `performance.csv`,
`lodo_{1pct,5pct}.png` (+ `_goodness_`, `_denario_`).

### G. Domain randomization → MG-AI-Sci transfer (opt-in)

**`run_dr_transfer.py`** — an opt-in add-on, never in `rebuild_all.py`. Re-runs
both transfer measurements above (in-domain 70/30 and LODO, scored on `mg_aisci`)
and for every `_trained.py` detector also trains a **domain-randomized twin**:
the same algorithm fit on training embeddings perturbed with Gaussian noise
(config `cfg/ai_detection/trainv2/domain_randomization.yaml` — `sigma`,
`per_feature_std`, `n_copies`; `enabled: false` or `kind: none` makes the twin a
null control). Noise touches **training rows only**; every eval set is scored
clean. Only the RoBERTa embedding is noised — the cue vector is tiled verbatim.
Nothing in the main pipeline is touched; when the script is not run the pipeline
is byte-identical.

```bash
.venv/bin/python epai/ai_detection/trainv2/run_dr_transfer.py
.venv/bin/python epai/ai_detection/trainv2/run_dr_transfer.py --regimes in_domain --seeds 0 1
.venv/bin/python epai/ai_detection/trainv2/run_dr_transfer.py --replot results/trainv2/dr_transfer/performance.csv
```
Outputs (`results/trainv2/dr_transfer/`): `performance.csv`, `dr_transfer.png`
(per regime a panel; base = filled circle, twin = hollow ring of the same
colour, joined by a connector). MG-AI-Sci has 100 human rows now (was 80,
pre-merge with the ai_scientist corpus) — right at the 1%-FPR reachability
boundary rather than clearly below it, so its oracle TPR@1%FPR is still a
noisy single-threshold reading rather than a stable one — read the
**base → twin delta**, not the absolute.

---

## MG-Reph experiments (opt-in)

MG-Reph is never in a default sweep or in `rebuild_all.py` (see *Why MG-Reph is
opt-in* above). It has its own self-contained pipeline — six steps, all counting
rephrased text as **human** ground truth and using **only gpt56luna** (no
gemma). `rebuild_rephrase.py` runs the lot; each step also runs standalone.

Everything writes to a dedicated `results/trainv2/rephrase_*` directory:

| step | script | → |
|------|--------|---|
| split | `split_manugram_reph.py` | `data/parquet/manugram_reph_{sci,gb}.parquet` |
| histograms | `plot_score_histograms.py` | `rephrase_score_histograms/` |
| zero-shot | `rephrase_zeroshot_performance.py` | `rephrase_zeroshot_performance/` |
| best zero-shot | `rephrase_best_zeroshot_performance.py` | `rephrase_best_zeroshot_performance/` |
| in-domain trained | `rephrase_in_domain_performance.py` | `rephrase_in_domain_performance/` |
| LODO trained | `rephrase_lodo_performance.py` | `rephrase_lodo_performance/` |

### 1. Split the corpus — `split_manugram_reph.py`

Partitions `manugram_reph.parquet` by `source_family` into
`manugram_reph_sci.parquet` (CS / math / sociology) and
`manugram_reph_gb.parquet` (essay / reuter / wp). Every column carried through
unchanged; the two halves share no document `id`. **Idempotent** — a no-op once
both halves exist; `--force` rebuilds. Prints, per half, the domain / generator
counts and which judge columns are scored on both classes (only `gpt56luna`).

```bash
uv run python epai/ai_detection/trainv2/split_manugram_reph.py            # writes both halves (or no-op)
uv run python epai/ai_detection/trainv2/split_manugram_reph.py --dry-run  # report only, write nothing
uv run python epai/ai_detection/trainv2/split_manugram_reph.py --force    # rebuild
```
Options: `--in <parquet>`, `--out-dir <dir>`. The halves are also registered as
the opt-in splits `mg_reph_sci` / `mg_reph_gb` (whole corpus: `mg_reph`) for
ad-hoc use with the core scripts.

### 2. Distribution look — `plot_score_histograms.py`

A 3-class view of where the rephrase sits, laid out **like the baseline
calibration figures**: one figure per score column, a panel per corpus.

Three classes, taken per corpus:

| corpus | human | rephrase | rewritten |
|--------|-------|----------|-----------|
| MG-GB  | `human` (MG-GB)  | `gpt56luna:0908R` (MG-Reph-GB)  | `gpt56luna:0701A` (MG-GB)  |
| MG-Sci | `human` (MG-Sci) | `gpt56luna:0908R` (MG-Reph-Sci) | `gpt56luna:0811A` (MG-Sci) |

The `*_reph_*` human rows are the same documents (same `id`, scores copied
whole), so humans come from the full corpus. Colours: human blue, rephrase
orange, rewritten red.

**One figure per score column** — every `binoculars_score*` (5), every
`fastdetect_crit_*` (5), `detector_score_radar`, and
`llm_judge_score_gpt56luna_v3` (the other judges are null on the rephrase rows)
— with **three panels: MG-GB, MG-Sci, MG-All (pooled)**, each an overlaid
density histogram of the three classes. Every panel marks two Youden's-J cuts
fit on just the two classes involved: human\|rephrase (dotted) and
human\|rewritten (dashed). Not a calibration — writes no config.

```bash
uv run python epai/ai_detection/trainv2/plot_score_histograms.py                        # all columns, all corpora
uv run python epai/ai_detection/trainv2/plot_score_histograms.py --families binoculars fastdetect
uv run python epai/ai_detection/trainv2/plot_score_histograms.py --no-thresholds
```
Options: `--bins`, `--out-dir`.
Outputs (`results/trainv2/rephrase_score_histograms/`): one `<column>.png` per
score column (3 panels each) and `summary.csv` (one row per corpus × column ×
class — corpus ∈ MG-GB / MG-Sci / MG-All: `n`, `mean`, `p1`/`p50`/`p99`, the two
cuts, the two human-vs-other AUROCs).

### 3. Zero-shot performance, rephrase = human — `rephrase_zeroshot_performance.py`

Like `run_zeroshot_performance.py` (oriented AUROC vs. **oracle** TPR@1%FPR, one
scatter per family, colour = version, shape = corpus, bold circle = mean across
MG-GB and MG-Sci) but the ground truth is **the full gpt56luna generation
only**: `rewritten` → `y=1`, **`human` and `rephrase` both → `y=0`**. So it
measures how well each Binoculars / Fast-DetectGPT pair still catches full
generations once light rephrases are folded into the human class.

```bash
uv run python epai/ai_detection/trainv2/rephrase_zeroshot_performance.py
uv run python epai/ai_detection/trainv2/rephrase_zeroshot_performance.py \
    --replot results/trainv2/rephrase_zeroshot_performance/performance.csv
```
Options: `--fpr` (list, default `0.01 0.05`), `--out-dir`, `--no-plots`.
Outputs (`results/trainv2/rephrase_zeroshot_performance/`):
`{binoculars,fastdetect}_{1pct,5pct}.png`, `performance.csv` (row per family ×
version × split × fpr — split ∈ `mg_gb` / `mg_sci` / `all` / `mean`; `all` is
pooled and kept in the CSV only).

### 4. Best-of-family on one figure, rephrase = human — `rephrase_best_zeroshot_performance.py`

The rephrase-GT sibling of `run_best_zeroshot_performance.py`. Reads the same
hand-picked column list from `--config`
(`cfg/ai_detection/trainv2/best_zeroshot.yaml`) **minus any gemma judge** (null
on the rephrase rows) — so `binoculars_score_llama31_8b`,
`fastdetect_crit_llama31_8b`, `llm_judge_score_gpt56luna_v3` — and scores each
with the same rephrase = human GT, all on **one** AUROC-vs-oracle-TPR@1%FPR
scatter (colour = `<family>/<version>`, shape = corpus, bold circle = mean
across MG-GB and MG-Sci).

```bash
uv run python epai/ai_detection/trainv2/rephrase_best_zeroshot_performance.py
uv run python epai/ai_detection/trainv2/rephrase_best_zeroshot_performance.py \
    --replot results/trainv2/rephrase_best_zeroshot_performance/performance.csv
```
Options: `--config`, `--fpr` (list, default `0.01 0.05`), `--out-dir`,
`--no-plots`. Columns in the config that the rephrase parquets weren't scored on
(the logprob-window / ensemble judges) are dropped automatically.
Outputs (`results/trainv2/rephrase_best_zeroshot_performance/`):
`best_zeroshot_{1pct,5pct}.png`, `performance.csv`.

### 5. Trained detection, rephrase = human — `rephrase_{in_domain,lodo}_performance.py`

The rephrase-GT siblings of `run_in_domain_performance.py` /
`run_lodo_performance.py`. **The training procedure is identical** — same three
`_trained.py` detectors (`emb_roberta` → logistic, and the two `emb_roberta` +
gpt56luna-cuevec hybrids; **no gemma anywhere**), same grouped 70/30 / LODO,
same N seeds, same AUROC + oracle TPR@1%FPR, same figure. Only the data and
label change:

* corpora: MG-GB (= MG-GB + MG-Reph-GB) and MG-Sci (= MG-Sci + MG-Reph-Sci); a
  document's human / rephrase / full-generation rows share one `id`
* label: `y=1` the full gpt56luna generation, **`y=0` human AND rephrase**
* OOD (default `mg_aisci`, `--ood-splits`): scored normally (human vs. all-AI)
* zero-shot overlay: the `best_zeroshot.yaml` picks minus any gemma judge —
  `binoculars_score_llama31_8b`, `fastdetect_crit_llama31_8b`,
  `llm_judge_score_gpt56luna_v3` — rephrase-GT on the training corpora, normal
  GT on the OOD split; scored on the **same eval rows as the trained models**
  (per-seed test partition / per-fold held-out domain), so they carry the same
  seed / fold SD (OOD point excepted)

```bash
uv run python epai/ai_detection/trainv2/rephrase_in_domain_performance.py
uv run python epai/ai_detection/trainv2/rephrase_in_domain_performance.py --seeds 0 1 2
uv run python epai/ai_detection/trainv2/rephrase_lodo_performance.py
uv run python epai/ai_detection/trainv2/rephrase_lodo_performance.py \
    --replot results/trainv2/rephrase_lodo_performance/performance.csv
```
Options: `--splits` (`mg_gb` / `mg_sci`), `--ood-splits`, `--seeds`,
`--test-size` (in-domain), `--min-domain-rows` (LODO), `--fpr` (list, default
`0.01 0.05`), `--zeroshot-config`, `--no-zeroshot`, `--out-dir`, `--no-plots`.
Outputs: `results/trainv2/rephrase_in_domain_performance/{performance.csv,in_domain_{1pct,5pct}.png}`
and `results/trainv2/rephrase_lodo_performance/{performance.csv,lodo_{1pct,5pct}.png}`.

### 6. Run the whole pipeline — `rebuild_rephrase.py`

The rephrase counterpart of `rebuild_all.py`. Runs the split (a no-op if the
halves exist) then the five standalone scripts — independent steps concurrently,
`in_domain` / `lodo` throttled to one at a time. With `--jobs > 1` each step is
logged to `results/trainv2/_rebuild_logs/<step>.log` (tail on failure); a step
whose dependency failed is skipped; exit is non-zero if anything failed.

```bash
uv run python epai/ai_detection/trainv2/rebuild_rephrase.py
uv run python epai/ai_detection/trainv2/rebuild_rephrase.py --dry-run
uv run python epai/ai_detection/trainv2/rebuild_rephrase.py --jobs 1        # serial, live output
uv run python epai/ai_detection/trainv2/rebuild_rephrase.py --only in_domain lodo
```

Steps: `split_manugram_reph`, `score_histograms`, `zeroshot`, `best_zeroshot`,
`in_domain`, `lodo`. `--heavy-jobs N` (default 1) caps the two trained steps.

`run_judge_by_generator.py` does **not** apply to MG-Reph — one generator.

---

## Full rebuild

The dataset renaming + Manugram Rephrase means every CSV/PNG under
`results/trainv2/` is stale.

**`rebuild_all.py` runs the whole sequence** — the steps are independent, so
they run concurrently, with the memory/compute-heavy ones throttled to one at a
time. It uses the **core splits only**; the MG-Reph split + plot are run
separately (above).

```bash
uv run python epai/ai_detection/trainv2/rebuild_all.py            # everything, ~4-wide
uv run python epai/ai_detection/trainv2/rebuild_all.py --dry-run  # print the plan, run nothing
uv run python epai/ai_detection/trainv2/rebuild_all.py --jobs 1   # serial, live output
uv run python epai/ai_detection/trainv2/rebuild_all.py --only run_lodo_performance   # that step + its deps
uv run python epai/ai_detection/trainv2/rebuild_all.py --skip-youden                 # drop experiment B
uv run python epai/ai_detection/trainv2/rebuild_all.py --list      # step names
```

With `--jobs > 1` each step is logged to `results/trainv2/_rebuild_logs/<step>.log`
(tail shown on failure); a step whose dependency failed is skipped; exit is
non-zero if anything failed or was skipped. `--heavy-jobs N` (default 1) caps the
RAM/compute-heavy steps (`run_embedding_effectiveness`,
`run_in_domain_performance`, `run_lodo_performance`) — bump it if the box has
headroom.

The steps it runs, all independent:

```bash
# A — baseline calibration (writes cfg/ai_detection/trainv2/*_baseline_calibrated.yaml)
uv run python epai/ai_detection/trainv2/calibrate_judge_baseline.py      --corpora all
uv run python epai/ai_detection/trainv2/calibrate_binox_baseline.py      --corpora all
uv run python epai/ai_detection/trainv2/calibrate_fastdetect_baseline.py --corpora all
uv run python epai/ai_detection/trainv2/calibrate_detector_baseline.py   --corpora all

# C, D, E — zero-shot analyses
uv run python epai/ai_detection/trainv2/run_embedding_effectiveness.py
uv run python epai/ai_detection/trainv2/run_zeroshot_performance.py
uv run python epai/ai_detection/trainv2/run_best_zeroshot_performance.py
uv run python epai/ai_detection/trainv2/run_judge_by_generator.py

# F — trained detectors
uv run python epai/ai_detection/trainv2/run_in_domain_performance.py
uv run python epai/ai_detection/trainv2/run_lodo_performance.py

# B — per-domain Youden calibration (optional; --skip-youden drops these)
uv run python epai/ai_detection/trainv2/run_judge_youden_calibration.py      data/parquet/manugram_sci.parquet
uv run python epai/ai_detection/trainv2/run_binox_youden_calibration.py      data/parquet/manugram_sci.parquet
uv run python epai/ai_detection/trainv2/run_fastdetect_youden_calibration.py data/parquet/manugram_sci.parquet
```

Common flags across the runners: `--fpr` / `--fpr-target` (default `0.01`; the
four D/F performance runners take a **list**, default `0.01 0.05`, and emit a
`_1pct` / `_5pct` figure set per level), `--seeds`, `--test-size` (default
`0.30`), `--human-label` (default `human`),
`--group-col` (default `id`), `--no-plots` / `--no-plot`, `--replot <csv>`.
