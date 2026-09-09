# gemma4-31b logprob judge — run lines

Adds `llm_judge_score_gemma4_31b_v3_lp` to the four core benchmark splits, scoring
the `gemma4-31b` judge in `--mode logprob`: the score is `P(human) / (P(human) +
P(machine))` read off the one-word answer token's distribution, instead of a
probability the model writes out. No round-number ties — the fix for gemma4's low
TPR@1%FPR (0.43 vs gpt56luna's 0.57 in `results/trainv2/judge_by_generator`).

Full state and design notes: `info/genai4science_judge_logprob_handoff.md`.

## Code

- `epai/ai_detection/score/add_llm_judge_scores.py` — `LOGPROB_API_MODELS`,
  `ApiJudge._score_genai4science_logprob()`, `_logprob_diagnostics()`.
- `epai/ai_detection/score/run_gemma_logprob_judge.py` — the loop below.
- `epai/ai_detection/score/probe_genai4science_logprobs.py` — capability probe.

## Pre-flight

Nothing else may be using `GENAI4SCIENCE_PERFORMANCE_API_KEY` during the run — it
shares a 4-concurrent-call cap with the ghostbuster / science generation pipelines.

Verified working 2026-09-09:

```
.venv/bin/python epai/ai_detection/score/probe_genai4science_logprobs.py --models gemma4-31b
# -> logprobs populated, token 0 = 'machine' (a real answer word)

.venv/bin/python epai/ai_detection/score/run_gemma_logprob_judge.py --only manugram_den --out-dir /tmp/lp_smoke --limit 60 --yes
# -> 60/60 scored, 0 failed, answer mass mean 1.000, 53/60 distinct, ROC-AUC 0.932
```

## Full pass — all 4 files, in place

```
.venv/bin/python epai/ai_detection/score/run_gemma_logprob_judge.py --yes --save-every 2000
```

- ~39,100 rows (ghostbuster_orig 8982 + manugram_gb 14960 + manugram_sci 15000 + manugram_den 160), ~8 rows/s, ≈ 1.5–2 h with parquet writes.
- Resumable: re-run the same line, filled rows are skipped; a per-file failure is
  recorded and the loop continues to the next file.
- `--save-every 2000` keeps the 500–830 MB `ghostbuster_orig` / `manugram_gb`
  re-serialisations from dominating; lower it if interruptions are likely.

### Background

```
nohup .venv/bin/python epai/ai_detection/score/run_gemma_logprob_judge.py --yes --save-every 2000 > /tmp/gemma_lp.log 2>&1 &
tail -f /tmp/gemma_lp.log
```

### One file at a time

```
.venv/bin/python epai/ai_detection/score/run_gemma_logprob_judge.py --yes --save-every 2000 --only manugram_den
.venv/bin/python epai/ai_detection/score/run_gemma_logprob_judge.py --yes --save-every 2000 --only ghostbuster_orig
.venv/bin/python epai/ai_detection/score/run_gemma_logprob_judge.py --yes --save-every 2000 --only manugram_gb
.venv/bin/python epai/ai_detection/score/run_gemma_logprob_judge.py --yes --save-every 2000 --only manugram_sci
```

## After it finishes

```
.venv/bin/python epai/ai_detection/trainv2/run_judge_by_generator.py
```

Auto-discovers the new `_lp` column (no code change) — shows whether logprob mode
closes gemma4's TPR@1%FPR gap against `gpt56luna`.

## What to watch per file

- `failed : 0` in the summary.
- `answer word:` split and `answer mass:` mean near 1.0 (as in the smoke run).
- A cluster of failures = a row shape the per-call checks reject (no distribution
  returned / token 0 not `human`/`machine` / answer mass < 0.10); re-running the
  line picks them back up.
