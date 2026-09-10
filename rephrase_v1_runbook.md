# rephrase_v1 — runbook

Build the **rephrase arm**: ~5,992 human texts across six domains (essay, news,
creative writing, computer science, mathematics, sociology), each paired with one
moderate `gpt-5.6-luna` rephrase, brought to the detector-suite schema with the
`gpt56luna` judge only. The content-preserving point on the LM-involvement
spectrum, opposite the full-text-generation v3f corpora.

- Output: `data/parquet/rephrase_v1_multi.parquet`
- Label: `generated_by = "gpt56luna:0908R"` (prompt config `cfg/ai_detection/rephrase/rephrase_v1.yaml`)
- Generation checkpoint: `data/science/rephrase_v1_generate.jsonl` (resumable)
- Plan: `~/.claude/plans/i-will-write-an-cozy-moonbeam.md`

## Inputs are read-only

`data/parquet/ghostbuster_v3f_multi.parquet` and
`data/parquet/science_v3f_multi_0811A.parquet` are **only ever read**. Every write
targets `rephrase_v1_multi.parquet` (+ its `.bak`) and the generation shard.
`check_standardization.py` and `rephrase_qc.py` (without `--write`) are read-only.

## Prerequisites

- `OPENAI_API_KEY` in `.env` (the `generate` step).
- A free GPU for the `score` step; pick its id for `<ID>` below.
- Run everything via `uv run` from the repo root.

## Pipeline

`rephrase_build.py` runs five steps with `--from <step>` resume:

| step | what it does | writes |
|---|---|---|
| `extract` | the `human` rows from both v3f corpora → the new parquet, copied whole (so every suite score/embedding/judge comes with them), original `uid` preserved, `source_family` + QC columns added, schemas unioned | `rephrase_v1_multi.parquet` |
| `generate` | one rephrase per human text via `gpt-5.6-luna`, domain-aware system prompt. Resumable — a re-run only re-issues failures | `rephrase_v1_generate.jsonl` |
| `append` | rephrase rows onto the parquet (text standardised per source family, suite columns Null, QC columns filled). Length-runaway rows reported and left out | `rephrase_v1_multi.parquet` (+ `.bak`) |
| `verify` | `check_standardization.py` — the standardise gate. Confirms every `text` is a fixpoint of `standardize_text`, per domain and per class, with no residue asymmetric across human vs rephrase. (Replaces the normal pipelines' column-wide standardize stage, which can't run here once the carried human scores are present.) | a report under `results/standardization/` |
| `score` | `run_holdout_pipeline.py --from embed --judges gpt56luna` — emb / Binoculars / Fast-DetectGPT / detector / gpt56luna judge + cue vector over the Null (rephrase) rows only | `rephrase_v1_multi.parquet` |

Then `rephrase_qc.py` (separate script) is the retrospective calibration check.

## Run lines

```bash
# 0. Dry run — validates extract (incl. the pa.unify_schemas merge) + generate
#    wiring, writes nothing. Stops at 'append' with an error because
#    rephrase_v1_multi.parquet does not exist yet — expected.
uv run python epai/ai_detection/rollout/rephrase_build.py --dry-run

# 1. Whole pipeline: extract → generate → append → verify → score.
#    Extract is fast; generate issues ~5,992 gpt-5.6-luna calls (resumable).
#    Safe to Ctrl-C during generate and resume.
uv run python epai/ai_detection/rollout/rephrase_build.py --gpu <ID> --yes

# --- or run it in controlled stages, resuming from each step ---
uv run python epai/ai_detection/rollout/rephrase_build.py --from generate --yes   # after eyeballing the shard
uv run python epai/ai_detection/rollout/rephrase_build.py --from append --yes
uv run python epai/ai_detection/rollout/rephrase_build.py --from verify --yes
uv run python epai/ai_detection/rollout/rephrase_build.py --from score --gpu <ID> --yes

# 2. Calibration check (read-only unless --write).
uv run python epai/ai_detection/analyse/rephrase_qc.py
uv run python epai/ai_detection/analyse/rephrase_qc.py --write   # also stores the semantic/style cosines on the rephrase rows
```

## Options

- `--from generate --limit 12 --yes` — cheap end-to-end smoke test on 12 texts.
  It appends 12 rows under `gpt56luna:0908R`, so before a full run either restore
  `rephrase_v1_multi.parquet` from `.bak` and delete the shard, or just re-run
  `--from extract`.
- `--retry-failed` (on a `--from generate` resume) — re-issue only the calls the
  shard recorded as failed.
- `--workers N` — concurrent OpenAI calls in `generate` (default 8).
- `--cpu-threads N` — cap CPU pools on a shared box (passed to the score stages).
- `--per-band N` — cue-vector coverage cap (default 20000; must exceed the row
  count, ~11,984).

## Verification checklist

1. **Dry run** prints ~5,992 humans, 6 domains, and the resolved `--from embed`
   command.
2. **After `generate`** — check the shard success rate; read ~10 human/rephrase
   pairs per domain (meaning preserved, wording clearly changed, no preamble
   leakage, length sane).
3. **After `append`** — total rows ≈ 11,984; `llm_judge_score_gpt56luna_v3`
   filled for exactly the ~5,992 human rows, null for the ~5,992 rephrase rows.
4. **`verify` exits 0** — every `text` a fixpoint of `standardize_text`, no
   asymmetric residue. Expected non-fatal WARNs on this combined file:
   non-canonical column order, science ids not domain-prefixed like ghostbuster
   ids.
5. **After `score`** — `run_holdout_pipeline.py` exits 0; its per-stage
   completeness + degenerate-column checks pass. The four unused judge columns
   being null on rephrase rows is by design (not a failure of these stages).
6. **QC gate** — `rephrase_qc.py`: per domain, semantic cosine (e5) median
   > ~0.85, surface Jaccard median in ~[0.25, 0.60], length ratio in [0.7, 1.4]
   for > 95% of rows. A domain that fails high (barely changed) or low (drifted)
   is fixed by revising that domain's hint in a **new** prompt id
   (`rephrase_v2.yaml`, id `0908S`…) and regenerating just that domain.
7. **Sanity result** — the human / rephrase / full-gen means table shows
   rephrase between human and full-gen.
