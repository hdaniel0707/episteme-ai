# Rebuttal API experiments — runbook

These are the experiments in TODO block A of `reviews/review_summary.html`. The code lives in
`conditions.py` (prompts, label folds, row subsets) and `capture.py` (the runner).
Outputs are sidecars in `data/parquet/tmp/rebuttal/<model>/<condition>__<subset>.parquet`.
The master parquets are only read. Each sidecar stores the raw answer and the top-20
log-probabilities of every output token, so the scorer for block B can recompute anything
without new calls. Re-running a command resumes it: completed rows are skipped and failed
rows are retried.

Run everything from the repo root with `.venv/bin/python -m epai.ai_detection.rebuttal.capture …`.
`--list` shows the conditions. `--dry-run` shows the row counts and the estimated input
tokens, and makes no calls.

## Read first: the paper's BaseJudge was not called like LogJudge

`add_llm_judge_scores.py::_score_openai` produced the existing luna BaseJudge column
(`llm_judge_score_gpt56luna_v3`). It used **Chat Completions with a JSON schema**, the
model's **default reasoning effort**, and the **default temperature**. LogJudge
(`capture_judge_logprobs.py`) used the **Responses API with reasoning effort `none`** at
**temperature 0**. Appendix "Judge Prompts and API Settings" says reasoning effort `none`
for luna without distinguishing the two calls.

So the paper's BaseJudge vs LogJudge comparison differs in call settings as well as in
the requested output, which is exactly xghj's point 2. `base_v3` fixes this: the same v3
BaseJudge prompt, called exactly like LogJudge. Check git history or old logs for which
effort the original BaseJudge run actually used, and correct the appendix text either way.

## Conditions

| condition | TODO | what changes vs. the paper | default subset × reps |
|---|---|---|---|
| `base_v3` | A1 control, A2, A3 | BaseJudge v3 prompt verbatim; called like LogJudge (effort none, T=0, plain text, logprobs on) | humans_plus_third × 1 |
| `base_hp4` | A1, A2 | as `base_v3`, but **four** decimal places (one sentence changed) | humans_plus_third × 1 |
| `base_free` | A1, A2 | as `base_v3`, but **unrestricted** precision ("as many decimal places as your judgement supports"), no example numbers | humans_plus_third × 1 |
| `lj_v3` | A3 | the paper's LogJudge, unchanged, repeated | near × 3 |
| `lj_swap` | A5 | answer words in the other order (machine first) | small × 1 |
| `lj_alt` | A5 | other words: `person` / `AI` | small × 1 |
| `lj_letters` | A5 | letters, `A` = machine / `B` = human (opposite of ord5's A = human) | small × 1 |
| `base_v3_t1` | A4 | `base_v3` at temperature 1, 3 samples to average | small × 3 |

Subsets: `humans_plus_third` = every human row + ⅓ of the machine rows per
(split, generator), with MG-AI-Sci kept whole (18,172 rows). The human-side 1%-FPR threshold
is exact; TPR comes from a random machine sample. `near` = the most machine-like 5% of each
split's humans plus up to 400 machine texts near the luna LogJudge/FusionJudge 1%-FPR
thresholds plus 100 random controls per split (2,337 rows). `small` = per split 1000 human
+ 1000 machine rows, stratified by domain × generator, plus all of MG-AI-Sci (6,200 rows).
`full` = all 36,148 rows.

## One script per step

| script | what it runs, in order |
|---|---|
| `run_probes.sh` | 12-call probes of `base_free`, `base_hp4`, `base_v3`, `lj_alt`, `lj_letters` (nothing saved) |
| `run_main.sh` | `base_hp4`, `base_free`, `base_v3`, `lj_v3`, `base_v3 --subset near --reps 3` (steps 1–4) |
| `run_extra.sh` | `lj_swap`, `lj_alt`, `lj_letters`, `base_v3_t1` (steps 5–6) |
| `run_gemma.sh` | gemma4-31b `base_hp4`, `base_free`, `base_v3`, with 4 workers |

Run them as `bash epai/ai_detection/rebuttal/run_main.sh` from any directory. Extra arguments
are passed to every call: `--workers 4` for rate limits, `--dry-run` for counts only. Each
script logs to `data/parquet/tmp/rebuttal/logs/`. A script stops at the first run that
crashes; a re-run skips everything already captured.

## Order and size (gpt-5.6-luna; multiply by your per-token price)

| step | command (after `.venv/bin/python -m epai.ai_detection.rebuttal.capture`) | calls | input tokens |
|---|---|---|---|
| 0 probe | `--model gpt56luna --condition base_hp4 --probe 12` and the same for `base_free`, `base_v3`, `lj_alt`, `lj_letters` | 5 × 12 | ~0 |
| 1 A1/A2 | `--model gpt56luna --condition base_hp4 --yes` | 18.2k | 26.6 M |
| 1b A1/A2 | `--model gpt56luna --condition base_free --yes` | 18.2k | ~27 M |
| 2 A1 control/A2 | `--model gpt56luna --condition base_v3 --yes` | 18.2k | 26.6 M |
| 3 A3 | `--model gpt56luna --condition lj_v3 --yes` | 7.0k | 10.0 M |
| 4 A3 (fusion side) | `--model gpt56luna --condition base_v3 --subset near --reps 3 --yes` | 7.0k | ~10 M |
| 5 A5 | `--model gpt56luna --condition lj_swap --yes`, then `lj_alt`, then `lj_letters` | 3 × 6.2k | 3 × 8.6 M |
| 6 A4 | `--model gpt56luna --condition base_v3_t1 --yes` | 18.6k | ~27 M |

Steps 1–4 are the minimum; both critical reviewers lead with these. Steps 1, 1b and 2 total
about 80M input tokens, and the whole table about 177M.

**Why two precision conditions.** "Why four digits?" has no good answer on its own. The claim
we test is about *resolution*: the number of distinct values the judge actually writes, and
the size of the tied groups at the 1%-FPR threshold. That number should not grow with the
allowed digits. `base_hp4` (fixed, higher) and `base_free` (unlimited) bracket the instruction.
A2's expected value, computed from the same call's token probabilities, is the
continuous limit of the stated number and needs no extra calls.

**Same models as the paper.** `--model` accepts only `gpt56luna` (`gpt-5.6-luna`) and
`gemma4-31b` (`gemma4:31b`), the registry entries that produced the paper's scores. Every row
records the version the API says served it (`model_version`), and the run summary warns if
more than one version appears. Compare it with the original run's version if old logs have it. Run `--dry-run` first: it prints the
exact figure.

**Probe first (step 0).** Look at three things:
- **Number tokenization.** How does luna split `0.0731`: `0` `.` `073` `1`, or something else?
  The A2 same-call expected value is computed from the first fractional-digit position, and
  its exact rule is fixed after seeing this.
- **Bare-number answers.** The answer should be a bare number. If it isn't, tell Claude
  before the full run.
- **Fold coverage for the alternative labels.** For `lj_alt` and `lj_letters`, the first
  token must fold to one of the condition's labels (`person` / `ai`, `a` / `b`).

**gemma-4-31b (free, institutional endpoint, 4 concurrent calls, ~3 s/call):** the same
steps with `--model gemma4-31b --workers 4`. Do steps 1–2 on `humans_plus_third`. That is
about 18k calls each, roughly 4 h per condition. They can run in parallel with the luna jobs.

## Running many jobs

The luna jobs write separate sidecars, so two or three can run at once in separate
terminals or `tmux` windows. Keep the total `--workers` within the account's rate limit.
`Ctrl-C` writes what has been captured, and the same command resumes later.

## After capture

The scorer (`score.py`, block B, CPU only) reads the sidecars, joins them with the master
parquets on `uid`, and computes AUROC / TPR@1%FPR / AT-HM, tie tables and the A2
expected-value readout. All comparisons use the same rows for every detector.
It is written after the probe settles the token rule.
