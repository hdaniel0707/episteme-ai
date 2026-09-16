# LLM-judge logprob capture

Capture an LLM judge's **answer-token log-probabilities** for the four core
benchmark splits, so a detection score can be computed *afterwards* — off the
stored distribution, with whatever aggregation function — instead of being
baked in at scoring time.

Two scripts:

| script | what it does |
|---|---|
| `capture_judge_logprobs.py --model X` | calls judge `X` once per row, writes a small **sidecar** `data/parquet/tmp/X/<stem>.parquet` (`uid` + the new columns). The dir is created if missing (`data/` is gitignored). Never touches the 800 MB parquets. |
| `merge_captured_logprobs.py` | joins every sidecar column under `data/parquet/tmp/` back onto `data/parquet/<stem>.parquet` by `uid` — the one expensive read+write per file, done once. |

Nothing here touches `add_llm_judge_scores.py` or any existing column, and it is
not part of any rebuild.

## What gets captured

Two judges × two answer formats:

| model | endpoint | path |
|---|---|---|
| `gemma4-31b` | GenAI4Science (open, no $) | chat completions, `logprobs=True, top_logprobs=20` |
| `gpt56luna` | OpenAI (**billed**) | Responses API, `reasoning={"effort":"none"}` + `include=["message.output_text.logprobs"]` — the only path that returns logprobs for the GPT-5.x line |

| mode | answer | column |
|---|---|---|
| `words` | one word — `human` / `machine` (the existing `LOGPROB_ANSWER_BLOCK`) | `llm_judge_logprobs_<model>_v3_words` |
| `scale` | one letter — **A = almost-certainly-human … E = almost-certainly-machine** | `llm_judge_logprobs_<model>_v3_scale` |

`<model>` is the registry column stem: `gemma4_31b`, `gpt56luna`. Every column is
`list<struct{token: string, logprob: double}>` — the ranked top-k window at the
answer position. `window[0]` is the sampled token (`temperature=0`). A row that
kept failing is left empty.

**Why the scale runs A→E (human→machine).** LLMs bias toward the first answer
letter when uncertain. With `A = human`, that bias nudges borderline cases *up*
the P(human) axis, so borderline **humans stay above the detector threshold** —
the safe direction for the oracle TPR@1%FPR, which is dominated by the low-human
tail.

**Why capture the raw window, not a score.** The paid calls run once; the
scoring function is then free to change. The `words` window supports the
`P(human)/(P(human)+P(machine))` ratio (and variants — window-edge floor, etc.);
the `scale` window supports a probability-weighted rating
(`Σ P(k)·value(k)` over A–E) and anything else. None of that needs a re-run.

## 1. Smoke test first

`--limit N` judges only `N` eligible rows per file, drawn **round-robin across
`domain` × `generated_by`**, so a small run is balanced (both classes, every
domain) and the same rows are picked every time — two models are then compared
on identical documents.

```bash
# always look before leaping
uv run python epai/ai_detection/score/capture_judge_logprobs.py --model gemma4-31b --dry-run

# smoke: 40 rows/file × both modes -- writes to data/parquet/tmp/<model>/
uv run python epai/ai_detection/score/capture_judge_logprobs.py --model gemma4-31b --limit 40 --yes
uv run python epai/ai_detection/score/capture_judge_logprobs.py --model gpt56luna  --limit 40 --yes

# or one whole small file -- MG-AI-Sci is 200 rows, trivial cost:
uv run python epai/ai_detection/score/capture_judge_logprobs.py --model gpt56luna --only manugram_aiscientist --yes
```

Each run prints a per-(file, mode) summary: rows captured, **mean window size**
(and how many are size-1), **argmax in-format** fraction, **saturated > 0.98**
fraction, and the argmax token distribution. That is what tells you whether the
mode produces a graded signal on real documents.

The smoke writes to the **same** workspace the full run uses
(`data/parquet/tmp/<model>/`), so the smoke rows carry over — re-running without
`--limit` resumes and only the remaining rows are sent. Nothing to clean up.
Pass `--workspace <dir>` if you want a throwaway location instead.

To inspect a smoke sidecar directly:

```bash
uv run python epai/ai_detection/score/merge_captured_logprobs.py \
    --only manugram_aiscientist --dry-run  # coverage + column report, writes nothing
```

## 2. Full run

The two `capture` runs are independent — **run them at the same time**, one per
terminal / machine:

```bash
# free, open endpoint
uv run python epai/ai_detection/score/capture_judge_logprobs.py --model gemma4-31b --yes

# billed — prompts "type 'yes'" once; ~78k short completions (39,102 rows × 2 modes)
uv run python epai/ai_detection/score/capture_judge_logprobs.py --model gpt56luna
```

Row counts per model × mode: `ghostbuster_orig` 8,982 · `manugram_gb` 14,960 ·
`manugram_sci` 15,000 · `manugram_den` 160 → **39,102**.

Both resume: re-run the same command and only unfilled rows are sent. An
interrupt (`Ctrl-C`) writes what was captured so far.

Then fold the sidecars back in — once, after both models finish:

```bash
uv run python epai/ai_detection/score/merge_captured_logprobs.py --dry-run   # coverage / collisions
uv run python epai/ai_detection/score/merge_captured_logprobs.py             # writes data/parquet/*.parquet
```

`merge` refuses to overwrite an existing column (`--overwrite`), refuses if a
sidecar `uid` is missing from the real file, and refuses below 95 % coverage
(`--allow-low` to force, `--min-coverage` to change the bar). `--out-dir` writes
the merged files elsewhere instead of in place.

## Flags

**capture** — `--model` (required), `--modes {words,scale}+` (default both),
`--only STEM+`, `--workspace DIR`, `--limit N`, `--workers N` (default 4),
`--retries N` (default 3), `--save-every N` (default 200 — the sidecar is small,
so this is cheap), `--max-chars N` (default: send the whole document),
`--dry-run`, `--yes`.

**merge** — `--workspaces DIR+` (default: every dir under
`data/parquet/tmp/`), `--only STEM+`, `--out-dir DIR`, `--overwrite`,
`--min-coverage F` (default 0.95), `--allow-low`, `--dry-run`.

## Notes

- **Cost.** `gpt56luna` full run ≈ 78k calls; each is the `v3` evidence prompt
  (~600 tok) + the whole document (~150–1500 tok) + ~5 output tok — roughly
  90–140M input tokens. Price it at gpt-5.6-luna's rate. `--max-chars` caps very
  long documents if the bill needs it (the existing judge columns send whole
  documents, so leave it off for comparability).
- **No big-file checkpoint problem.** Checkpoints write the few-MB sidecar, not
  the 800 MB parquet. The 800 MB write happens once, at merge.
- **Prerequisites in `.env`:** `OPENAI_API_KEY` for `gpt56luna`,
  `GENAI4SCIENCE_PERFORMANCE_API_KEY` (or `GENAI4SCIENCE_API_KEY`) for
  `gemma4-31b`.
- **Completeness checker.** `check_parquet_completeness.py` will null/empty-check
  the new `list<struct>` columns after merge; nothing logprob-specific.
- **Scoring the windows** is downstream and not decided here: `words` →
  `P(human)/(P(human)+P(machine))` and variants; `scale` → `Σ P(k)·value(k)`
  over A–E (values `1.0, 0.75, 0.5, 0.25, 0.0`), renormalised over the letters
  present. Related work: `info/citations.md` §11.
