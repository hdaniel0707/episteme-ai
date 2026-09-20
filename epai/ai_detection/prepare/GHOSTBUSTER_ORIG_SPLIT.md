# Splitting `ghostbuster_orig` into the raw + standardised variants

`ghostbuster_orig.parquet` (a `CORE_STEM`, one of the four MANUGRAM benchmark
splits) currently ships **three** row-classes per source document:

| `generated_by`  | what it is                          | `text` scored on |
| --------------- | ----------------------------------- | ---------------- |
| `human`         | GB essays / wp / reuter             | standardised     |
| `gpt35t:0000A`  | the GPT-3.5 generation              | **raw**          |
| `gpt35ts:0000A` | the *same* generation, `standardize_text.py` run | standardised |

`gpt35t` and `gpt35ts` are the identical generation — byte-identical `text_raw` —
differing only in whether the typographic scrub reached the `text` the scorers
read. The human class exists **only standardised**. Every trainv2 runner does
"human vs. everything else", so today `gb_orig`'s machine class is `gpt35t` +
`gpt35ts` pooled: doubled, 1 : 2 human : machine, two feature regimes mixed.

We split it into two 2-class benchmarks:

* **`ghostbuster_orig.parquet`** — `human` + `gpt35t`, both **raw**. The "online
  Ghostbuster benchmark, unchanged" variant. The `gpt35t` machine rows are
  already raw-scored; the **human rows are re-scored on raw text** (step 3).
* **`ghostbuster_orig_stand.parquet`** — `human` + `gpt35ts` (relabelled to
  `gpt35t:0000A`), both **standardised**. Internally consistent, matches every
  other MANUGRAM split. **Finished after step 2** — nothing to recompute.

`ghostbuster_orig_3class.parquet` is kept as the untouched source (the `gpt35ts`
rows and the raw human `text` survive nowhere else).

### Where the raw human data comes from

There is **no separate "combine" step** — the raw human text is already in the
parquet as `text_raw`, and it is **byte-identical to the
`external/ghostbuster/data/<domain>/human/<n>.txt` files** (verified:
`essay:1`'s `text_raw` == `open('external/ghostbuster/data/essay/human/1.txt').read()`,
same 3855 chars, no restructure truncation). `split_ghostbuster_orig.py --variant
orig` sets human `text ← text_raw` and nulls every human feature column; the
pipeline then re-scores those rows. So `ghostbuster_orig.parquet` ends up as the
Ghostbuster human originals + the raw `gpt35t` machine rows, scored on the same
(un-standardised) text.

---

## Prerequisites

* The fresh `ghostbuster_orig.parquet` from the server. 3 classes, ~2994
  rows each.
* A GPU box for the re-score: embeddings (12 models), Binoculars / Fast-DetectGPT
  (5 pairs), radar, the local `qwen3-14b` judge.
* API access: OpenAI (`gpt56luna`), GenAI4Science (`gptoss-120b`, `llama33-70b`,
  `gemma4-31b`).
* All commands run from the repo root, `uv run python …`.

## Known column quirks (not caused by the split)

* **`fastdetect_prob_{falcon3_7b,mistral_v03,qwen25_7b,llama31_8b}`** are null for
  every class already — only the calibrated `falcon-7b` pair gets a `_prob_`
  column. `check_parquet_completeness.py` knows this; carry them through.
* **`llm_judge_cues_{human,machine}_gpt56luna_v3`** are lists that are legitimately
  empty on rows where the judge named no cue of that polarity. Partial ≠ broken.
* **`llm_judge_score_sonnet5_v3`** — sonnet5 was retired 2026-09-03 and was
  **not** run on the `gpt35ts` rows. It would be 50 % null in `_stand`, and in
  `_orig` the human half would need re-scoring. Excluded from trainv2's default
  judge family — **drop it** (below).
* **`llm_judge_score_gemma4_31b_v3_lp`** — the gemma logprob-mode judge. The
  re-score pipeline does **not** run it (`run_gemma_logprob_judge.py` removed),
  so the human half of this column would stay null and fail the final `verify`.
  Also excluded from trainv2's judge family — **drop it** too. (`_wlp` / `_slp` /
  `_sbc` are kept: the pipeline recomputes them from the logprob windows.)

## Caveat: boilerplate lineage

The `gpt35t` machine rows were built with `strip_boilerplate` ON; the human rows
came from `ghostbuster_gpt56luna` with it OFF (see `build_ghostbuster_multi.py`).
`text_raw` for humans is verbatim the submodule `.txt` (essay / wp / reuter, where
"boilerplate" — publisher headers, reference lists — is close to a non-issue), so
this is expected to be immaterial. If a `_orig` vs `_stand` comparison shows a
surprising gap, this is the first thing to rule out.

---

## Step 0 — sync + sanity-check the source

```bash
# bring data/parquet/ghostbuster_orig.parquet from the server, then:
uv run python epai/scripts/check_parquet_completeness.py data/parquet/ghostbuster_orig.parquet
```

Expect: 3 classes (`human` / `gpt35t:0000A` / `gpt35ts:0000A`), ~2994 each, every
column full except the known quirks above.

## Step 1 — split

```bash
DROP="llm_judge_score_sonnet5_v3 llm_judge_score_gemma4_31b_v3_lp"

uv run python epai/ai_detection/prepare/split_ghostbuster_orig.py --dry-run --drop-cols $DROP

uv run python epai/ai_detection/prepare/split_ghostbuster_orig.py --drop-cols $DROP
```

Writes:

```
data/parquet/ghostbuster_orig_3class.parquet   backup — keep it, do not commit
data/parquet/ghostbuster_orig_stand.parquet    DONE
data/parquet/ghostbuster_orig.parquet          raw; human feature cols NULLED
```

Re-running is safe: once the backup exists the script reads from it. Add
`--force` to also rewrite `_stand` on a re-run.

## Step 2 — verify `ghostbuster_orig_stand.parquet` (no re-score)

```bash
uv run python epai/scripts/check_parquet_completeness.py \
    data/parquet/ghostbuster_orig_stand.parquet

# cross-check it is standardised the same way as a sibling split
uv run python epai/ai_detection/analyse/check_standardization.py \
    data/parquet/ghostbuster_orig_stand.parquet data/parquet/manugram_gb.parquet
```

`_stand` is finished. **Everything below is only for the raw
`ghostbuster_orig.parquet`.**

---

## Step 3 — re-score the raw human rows (the overnight pipeline)

`run_ghostbuster_orig_pipeline.py` runs the whole suite unattended — embeddings,
Binoculars, Fast-DetectGPT, radar, the 5 `v3` judges, the cue vector, the
logprob-window judges (`_wlp` / `_slp` / `_sbc`), the `base` control judge, and
the soft-min ensemble — on `data/parquet/ghostbuster_orig.parquet` in place.
The `gpt35t` machine rows are already scored, so every stage only touches the
~2994 nulled human rows. It does **not** run `standardize_text.py` (raw is the
point), it does **not** run the gemma `_lp` judge, and the final `verify` passes
`--allow-unstandardized` so an un-standardised-text finding is a WARN, not a
stop.

Each stage is a separate process; one that exits 0 having filled nothing is
caught by the column check after it and re-run (scorers skip filled rows) up to
`--fill-retries` times with a `--fill-cooldown` pause. Set
`NTFY_PIPELINE_TOPIC` in `.env` for a phone push at every stage boundary.

```bash
# dry-run first — prints every command, runs nothing
uv run python epai/ai_detection/prepare/run_ghostbuster_orig_pipeline.py --dry-run

# the real run (assumes step 1 has produced the 2-class raw file)
uv run python epai/ai_detection/prepare/run_ghostbuster_orig_pipeline.py --yes --gpu 0

# or do the split + re-score in one go
uv run python epai/ai_detection/prepare/run_ghostbuster_orig_pipeline.py --split --yes --gpu 0

# resume after a failure (the summary prints this line for you)
uv run python epai/ai_detection/prepare/run_ghostbuster_orig_pipeline.py \
    --from logprob-capture-gpt56luna --yes --gpu 0
```

Notes:

* **`--split`** prepends the split stage *and wipes* the pipeline's capture
  workspaces under `data/parquet/tmp/gb_orig_pipe/` — a fresh split invalidates
  the human logprob / baseline captures. Without it the split is assumed done and
  the workspaces persist for a resume.
* The capture → merge stages for the logprob windows and the `base` judge use
  **isolated workspaces** and `--overwrite`. `merge_*` replaces the whole column
  by a `uid` left-join, so the sidecars must cover every `uid`; `capture_*`
  therefore re-captures the `gpt35t` rows too (redundant, same text/model —
  harmless).
* The pipeline stops on a genuine failure (a degenerate score column, a judge
  that could not finish after retries). It does **not** stop on the
  standardisation WARN.

## Step 4 — verify

The pipeline's last stage already runs this. To re-check by hand:

```bash
uv run python epai/scripts/check_parquet_completeness.py \
    data/parquet/ghostbuster_orig.parquet --allow-unstandardized --fail-on error
```

Expect: 2 classes (`human` / `gpt35t:0000A`), ~2994 each, every column full
(minus the known quirks), no degenerate columns, the standardisation finding a
WARN. **Do not** run `check_standardization.py` on this file — it is meant to
fail that check.

Sanity-diff the two variants:

```bash
uv run python -c "
import pandas as pd
a = pd.read_parquet('data/parquet/ghostbuster_orig.parquet')
b = pd.read_parquet('data/parquet/ghostbuster_orig_stand.parquet')
for df,n in [(a,'orig'),(b,'stand')]:
    m = df[df.generated_by!='human']
    print(n, 'machine binox_llama31 mean', round(m.binoculars_score_llama31_8b.mean(),3),
          'judge gpt56luna mean', round(m.llm_judge_score_gpt56luna_v3.mean(),3))
"
```

## Step 5 — wire into trainv2

1. `epai/ai_detection/trainv2/splits.py`:
   * `SPLITS`: add `"gb_orig_stand": "ghostbuster_orig_stand.parquet"` after
     `gb_orig`.
   * `DISPLAY`: add `"gb_orig_stand": "GB-orig-std"`.
   * `MARKER`: add `"gb_orig_stand": "P"` (or any free marker).
   * **Do not** add it to `DEFAULT` — it is opt-in, like the `mg_reph` keys.
2. `epai/ai_detection/trainv2/README.md` line ~78: `gb_orig` is now `human` +
   `gpt35t` (raw); add a `gb_orig_stand` row (`human` + `gpt35t`, standardised).
3. The `CORE_STEMS` lists in `epai/ai_detection/score/*.py` still name
   `ghostbuster_orig` — unchanged, still correct.
4. Re-run everything that reads `gb_orig`:

```bash
uv run python epai/ai_detection/trainv2/rebuild_all.py
```

Every `gb_orig` number changes (1 : 2 → 1 : 1 balance, half the machine rows).
Compare `gb_orig` vs `gb_orig_stand` to decide which the paper uses.
