# Detector suite

`cfg/detector_suite.yaml` is the one place that says which detector models this
project scores its corpora with. Every dataset — `ghostbuster_*`, `science_v3`,
`science_v5`, `denario_hc`, whatever comes next — carries the **same** embedding,
Binoculars, Fast-DetectGPT, detector and judge columns, so a number from one is
comparable with a number from another.

## What reads it

| consumer | uses |
|---|---|
| `epai/ai_detection/prepare/run_holdout_pipeline.py` | all five groups, as its `--embedders` / `--binox-pairs` / `--fastdetect-pairs` / `--detectors` / `--judges` defaults + `judge_prompt_version`, `cue_judge`. `--from embed` on an already-standardised corpus makes it a general suite scorer, not just a holdout one |
| `epai/ai_detection/prepare/carry_suite_columns.py` | the exact output-column names, to copy from a sibling corpus that already scored them (so only new rows are computed) |
| `cfg/science_data/*.yaml` `embed.embedders` | must equal `embedders` here (kept in sync by hand for now) |
| `cfg/ai_detection/train/**/*.yaml` baseline blocks | choose *which* of these columns become rows in a given figure — a per-experiment call — but never a model outside this file |

If the YAML is missing or unreadable, `run_holdout_pipeline.py` falls back to the
constants hard-coded in it, so a partial checkout still runs.

## The groups

### embedders (12)

`emb_<name>` columns. Grouped by what the model encodes, which is what makes them
a spectrum for detection: **content** (`specter2`, `scincl`, `scibert`) →
**general semantic** (`mpnet`, `minilm`, `e5`, `bge`) → **raw LM** (`roberta`) →
**authorship / style** (`styledistance`, `style_wegmann`, `luar`, `luar_crud`).
The training configs currently read only `emb_luar_crud`; the rest are scored so
the feature axis can be widened later without a re-score. `qwen3`, `openai_small`
and `openai_large` are deliberately **out** — the OpenAI ones are billed per row.

### binoculars_pairs (5) and fastdetect_pairs (5)

The same five model pairs, in **detector-reference-model capability order** —
Falcon-7B (2023) → Falcon3-7B → Qwen2.5-7B → Mistral-7B v0.3 → Llama-3.1-8B —
which is the x-axis of Figures 1 and 2. The two lists are kept **identical** on
purpose: Binoculars and Fast-DetectGPT are the same pair with one statistic
swapped, so a `binoculars__<pair>` row without its `fastdetect__<pair>` twin is a
hole in the comparison Leg 1 is built on.

**Each pair is base + instruct of one family** — observer/performer for
Binoculars, sampling/scoring for Fast-DetectGPT — sharing a tokenizer, because
the text is encoded once and both models are fed the same ids. `models:` in the
YAML lists the exact HF ids; the source of truth is
`external/Binoculars/model_pairs.py` / `external/fast-detect-gpt/detector_pairs.py`.

**Two of the five are not a byte-identical match, and this matters:**

- **`falcon3-7b`** — `Falcon3-7B-Base` and `Falcon3-7B-Instruct` share a
  vocabulary but the instruct tokenizer adds control tokens, so the two are
  *similar, not identical*. `score_parquets.py`'s strict tokenizer check rejects
  this at startup; `run_holdout_pipeline.py` handles it by giving `falcon3-7b`
  its **own stage** carrying `--allow-token-mismatch`, so the flag does not also
  switch the check off for the four pairs that pass it honestly (the check is
  what stands between an "extended" tokenizer and a genuinely `INCOMPATIBLE`
  remapping that would fill the column with meaningless numbers). Flagged in the
  YAML as `extended_tokenizer: true`.
- **`llama31-8b` on the Fast-DetectGPT side** — upstream ships calibration
  constants for Llama-**3**, not Llama-3.1, so that row's declared crit threshold
  is *borrowed* from a similar-but-different model. The Binoculars `llama31-8b`
  row has no real threshold either (placeholder 0.95). Both are marked
  `calibrated: false`.

Only `falcon-7b` carries real calibration constants (Binoculars: 0.9015 accuracy
/ 0.8536 at 1% FPR; Fast-DetectGPT crit: 1.43). The other four carry **placeholder
thresholds** in the training configs — a declared operating point for
reproducibility, not a fitted one. That is fine for a holdout with a human class
(AUROC and oracle TPR come off the holdout's own ROC) and is called out on every
figure.

`falcon-7b` **must stay first** in `binoculars_pairs`: it writes the legacy
unsuffixed `binoculars_score` column that the configs' calibrated threshold names.

`llama31-8b` (both families) is **gated** — needs `HF_TOKEN` with an accepted
Llama licence. Its weights are shared between the Binoculars and Fast-DetectGPT
runs, so adding it to one when the other already has it is nearly free.

### detectors (1)

Pretrained supervised detectors, `detector_score_<name>`. `radar` only —
`pangram_roberta` is gated on the Hub and access is pending.

### judges (5)

`llm_judge_score_<model>_<prompt>`, one scoring process each, in capability-ladder
order:

- **`qwen3-14b`** — local 14B, bf16 on the A100. The low rung — inverted (below
  chance) on science prose, which is the punchline of Leg 1.
- **`qwen38-27b`** — GenAI4Science, a third open-weight family; 27B, a rung below
  `llama33-70b`. Holds a dedicated GPU on the perf endpoint (unlike the
  downgraded `llama33-70b`), so it actually serves 4 concurrent calls in parallel.
- **`llama33-70b`** — GenAI4Science (HUN-REN SZTAKI) open-weight endpoint.
- **`gptoss-120b`** — GenAI4Science, the top open-weight rung. gpt-oss and
  llama3.3 are also the Track-1/2 open-weight *generators*, so a judge and a
  generator here can be the exact same weights.
- **`gpt56luna`** — API, same model family as the generator. Listed **first** in
  the YAML (not by capability) because the cue vector
  (`llm_judge_cuevec_gpt56luna_v3`) is built from its rationales and the
  cue-feature stage must run after it.

**`sonnet5` was removed (2026-09-03)** — the cross-vendor control ("one of your
judges is the generator judging itself"), but too expensive to run on every
corpus. `llm_judge_score_sonnet5_v3` stays in the files that already carry it
(`science_v3_gpt56luna_0811A`, `denario_hc`) and is not recomputed elsewhere.
`haiku45` and `opus-5` are the planned frontier additions (experiment 8).

The three GenAI4Science judges reach an **OpenAI-compatible endpoint**. Set
`GENAI4SCIENCE_PERFORMANCE_API_KEY` (dedicated capacity, max 4 concurrent — a
5th call trips a ~1-minute key block) and the code uses the performance
endpoint automatically; otherwise it falls back to `GENAI4SCIENCE_API_KEY` on
the shared endpoint. Either base URL is overridable
(`GENAI4SCIENCE_*_BASE_URL`); see `epai/utils/env_utils.py :: genai4science_endpoint`.
Keep `--g4s-workers` at 4 or below. `add_llm_judge_scores.py` tries structured output first and falls
back to a plain completion parsed with `parse_probability` (stripping any
`<think>…</think>` and preferring the number after an "answer"/"probability"
marker). It prints a per-run breakdown of structured vs fallback vs
parse-retry rows, so a run on the fragile free-text path is visible. The
endpoint is rate-limited and drops a large fraction of calls under batch load —
budget several retry passes on a full corpus.

`judge_prompt_version: v3` and `cue_judge: gpt56luna` are part of the column
names — changing either changes which columns every config reads.

## Adding a model

In this order, or a config's preflight will abort on a corpus that lacks the
column:

1. **Register the pair / model** in the external submodule if it is new —
   `external/Binoculars/model_pairs.py`, `external/fast-detect-gpt/detector_pairs.py`
   — and, for a Binoculars pair, run `external/Binoculars/check_pairs.py` to see
   whether its tokenizer is `extended` (goes in `binoculars_pairs` with
   `extended_tokenizer: true`) or genuinely `INCOMPATIBLE` (do not add it).
2. **Add it to `cfg/detector_suite.yaml`** in the right capability position.
3. **Backfill every existing dataset parquet** with the new column — the
   relevant `epai/ai_detection/score/` or `embed/` script over each
   `data/parquet/*` corpus. Until every corpus a training config names has the
   column, step 4 fails the preflight (by design).
4. **Add the baseline row** to the training configs that should plot it, with a
   real threshold if the model ships one or a documented placeholder if not.
5. For an embedder, also add it to `cfg/science_data/*.yaml` `embed.embedders`.
