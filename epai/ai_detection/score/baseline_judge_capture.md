# Baseline-judge capture

Score the four core benchmark splits with an LLM judge under the **`base`
prompt** — the control for the cue-loaded `v3` judge already in the corpora —
without rewriting an 800 MB parquet until the one merge at the end.

The parsed-mode twin of `logprob_capture.md`: same
capture → sidecar → merge shape. Where that pipeline stores a ranked token
window, this one stores a single number: the probability the judge writes.

Two scripts:

| script | what it does |
|---|---|
| `capture_baseline_judge.py --model X` | calls judge `X` once per row under the `base` prompt, writes a small **sidecar** `data/parquet/tmp/judge_scores/X/<stem>.parquet` (`uid` + one score column). Resumable. Never touches the big parquets. |
| `merge_captured_judge_scores.py` | joins every `llm_judge_score_*` sidecar column under `data/parquet/tmp/judge_scores/` back onto `data/parquet/<stem>.parquet` by `uid` — the one expensive read+write per file, done once. |

Nothing here touches `add_llm_judge_scores.py`'s existing columns or any rebuild.

## The `base` prompt

`PROMPTS["base"]` in `add_llm_judge_scores.py` is `v3` with **only the two
evidence lists removed** (the human-cue bullets and the machine-cue bullets).
Same role, same `JUDGE RELATIVE TO THE GENRE` framing, same
`WHAT IS NEVER EVIDENCE` section, same `CALIBRATION` block — verbatim. So
`llm_judge_score_<model>_base` beside `llm_judge_score_<model>_v3` isolates one
thing: **how much of the judge's signal is the cue checklist versus the model's
own priors.**

One prompt covers every genre (essay / wire / fiction / abstract), exactly as
`v3` does — there is no per-corpus judge prompt.

## What gets captured

| model | endpoint | path |
|---|---|---|
| `gemma4-31b` | GenAI4Science (open, no $) | chat completions, structured output → free-text + `parse_judge_probability` fallback |
| `gpt56luna` | OpenAI (**billed**) | Responses API, the same parsed path that produced `llm_judge_score_gpt56luna_v3` |

One column per model, parsed mode → **no mode suffix**:

    llm_judge_score_gemma4_31b_base
    llm_judge_score_gpt56luna_base

A row that kept failing every retry is left Null. Empty-text rows are skipped.
Scoring reuses `ApiJudge` + `judge_row` from `add_llm_judge_scores`, so it is the
same tested path (structured output, free-text fallback, clip to `[0, 1]`).

## 1. Smoke test first

`--limit N` judges only `N` eligible rows per file, drawn **round-robin across
`domain` × `generated_by`**, so a small run is balanced and deterministic — the
same rows every time, so two models are compared on identical documents.

```bash
# always look before leaping — counts only, no calls
uv run python epai/ai_detection/score/capture_baseline_judge.py --model gemma4-31b --dry-run

# smoke: 40 rows/file — writes to data/parquet/tmp/judge_scores/<model>/
uv run python epai/ai_detection/score/capture_baseline_judge.py --model gemma4-31b --limit 40 --yes
uv run python epai/ai_detection/score/capture_baseline_judge.py --model gpt56luna  --limit 40 --yes

# or one whole small file — MG-Den is 160 rows, trivial cost:
uv run python epai/ai_detection/score/capture_baseline_judge.py --model gpt56luna --only manugram_den --yes
```

Each file prints a summary: rows scored, mean / median, **distinct values and
largest tie** (the parsed `v3` scores clump on round numbers; `base` uses the
same "two decimals, no round values" calibration, so watch this), and the value
range. At the end of the run, `gemma4-31b` also prints how many rows came
through structured output vs the free-text fallback -- a run that is slow or
stalls partway is usually one where the endpoint stopped honouring
`response_format` under load and fell back to a 2048-token generation per row.

The smoke writes to the **same** workspace the full run uses, so smoke rows
carry over — re-running without `--limit` resumes and only the remaining rows
are sent. Nothing to clean up.

## 2. Full run

The two `capture` runs are independent — **run them at the same time**, one per
terminal / machine:

```bash
# free, open endpoint -- keep --workers low, the endpoint throttles
uv run python epai/ai_detection/score/capture_baseline_judge.py --model gemma4-31b --yes

# billed -- prompts "type 'yes'" once; ~39,102 short completions. OpenAI is fast:
uv run python epai/ai_detection/score/capture_baseline_judge.py --model gpt56luna --workers 16
```

Row counts per model: `ghostbuster_orig` 8,982 · `manugram_gb` 14,960 ·
`manugram_sci` 15,000 · `manugram_den` 160 → **39,102**.

Both resume: re-run the same command and only unscored rows are sent. `Ctrl-C`
writes what was captured so far.

Then fold the sidecars back in — once, after both models finish:

```bash
uv run python epai/ai_detection/score/merge_captured_judge_scores.py --dry-run   # coverage / collisions
uv run python epai/ai_detection/score/merge_captured_judge_scores.py             # writes data/parquet/*.parquet
```

`merge` refuses to overwrite an existing column (`--overwrite`), refuses if a
sidecar `uid` is missing from the real file, and refuses below 95 % coverage
(`--allow-low` to force, `--min-coverage` to change the bar). `--out-dir` writes
the merged files elsewhere instead of in place.

## Flags

**capture** — `--model` (required), `--prompt` (default `base` — any registered
prompt; names the column), `--only STEM+`, `--workspace DIR`, `--limit N`,
`--workers N` (default 4 — GenAI4Science throttles; OpenAI takes 16+),
`--retries N` (default 3), `--save-every N` (default 200), `--max-chars N`
(default: send the whole document), `--dry-run`, `--yes`.

**merge** — `--workspaces DIR+` (default: every dir under
`data/parquet/tmp/judge_scores/`), `--only STEM+`, `--out-dir DIR`,
`--overwrite`, `--min-coverage F` (default 0.95), `--allow-low`, `--dry-run`.

## Notes

- **Cost.** `gpt56luna` full run ≈ 39k calls — the `base` prompt (~400 tok) +
  the whole document (~150–1500 tok) + ~5 output tok. Roughly half the input
  tokens of a `v3` run (the two evidence lists are gone). `--max-chars` caps
  very long documents if the bill needs it (leave it off for comparability with
  the `v3` columns).
- **No big-file checkpoint problem.** Checkpoints write the few-KB sidecar, not
  the 800 MB parquet. The 800 MB write happens once, at merge.
- **Prerequisites in `.env`:** `OPENAI_API_KEY` for `gpt56luna`,
  `GENAI4SCIENCE_PERFORMANCE_API_KEY` (or `GENAI4SCIENCE_API_KEY`) for
  `gemma4-31b`.
- **Completeness checker.** `check_parquet_completeness.py` treats
  `llm_judge_score_*_base` as an ordinary bounded score column (range `[0, 1]`,
  degeneracy / clumping check) after merge.
- **Downstream.** The `llm_judge_score_` prefix means the trainv2 judge sweep
  (`run_zeroshot_performance.py`, `calibrate_judge_baseline.py`,
  `run_judge_by_generator.py`, the goodness bars) picks the new columns up on
  re-run with no config change — label `gemma4_31b_base`, `gpt56luna_base`. Add
  to `cfg/ai_detection/trainv2/best_zeroshot.yaml` by hand if you want them in
  the curated "best" figure.
