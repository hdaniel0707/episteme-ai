# Scoring the LLM-judge logprob windows

`score_logprob_windows.py` turns the captured window columns
(`llm_judge_logprobs_<model>_v3_{words,scale}`, see `logprob_capture.md`) into
plain **P(human)** score columns:

| from | to | readout |
|---|---|---|
| `llm_judge_logprobs_<model>_v3_words` | `llm_judge_score_<model>_v3_wlp` | words, **contrast** |
| `llm_judge_logprobs_<model>_v3_scale` | `llm_judge_score_<model>_v3_slp` | scale, **expected value** |
| `llm_judge_logprobs_<model>_v3_scale` | `llm_judge_score_<model>_v3_sbc` | scale, **contrast** |
| `llm_judge_logprobs_<model>_v3_words` | `llm_judge_score_<model>_v3_logjudge_t0000` | words, **contrast** (= `wlp`) |
| `llm_judge_logprobs_<model>_v3_scale` | `llm_judge_score_<model>_v3_logjudge_t0001` | scale, **contrast**, full ladder |

`<model>` ∈ `gemma4_31b`, `gpt56luna` → **10 columns per parquet**, across the 4
core splits. All are P(human): **1.0 = human, 0.0 = machine**,
`machine_is_high=False`, same `llm_judge_score_` prefix as every other judge, so
`run_zeroshot_performance.py`, `calibrate_judge_baseline.py`,
`run_judge_youden_calibration.py`, `run_judge_by_generator.py` pick them up on
re-run with no config change (labels `gemma4_31b_wlp`, `gpt56luna_sbc`, …).

## `logjudge_<paramid>` -- the naming convention for LogJudge columns

Every column that instantiates the paper's general LogJudge formula (not
`wlp`/`slp`/`sbc`, which predate this and are kept unchanged) gets the suffix
`_v3_logjudge_<paramid>`, `<paramid>` a sequential id (`t0000`, `t0001`, ...)
assigned once, permanently, the moment a (prompt, lambda, v) configuration
becomes a real column -- ids are never reused or renumbered, and exploring a
config in `search_logjudge_params.py`'s random search does not by itself
spend one; only promoting a trial to a permanent column does. The `logjudge_`
tag also disambiguates these from the *other*, differently-computed
`llm_judge_score_*` columns on the same parquets (`sonnet5`, `gptoss_120b`,
`llama33_70b`, `qwen3_14b`, the `_base` columns, …), which are not LogJudge at
all.

**`epai/ai_detection/score/logjudge_params.csv` is the registry** -- the only
place a paramid's exact configuration is decoded from: `prompt`,
`lambda_family`, `eps` (if the threshold readout), and `v_1..v_5` (positional
values for the labels in scheme order, so both the 2-label and 5-label schemes
fit one schema), plus a free-text `note`.

The paper's two reported variants are the first two entries:

- **`t0000`** (paper's `ord2`) -- `L = {human, machine}`, `v = (1, 0)`. This is
  `wlp` under the new convention; same formula, same input, added as its own
  column so the paper's variant resolves directly to a parquet column instead
  of requiring a `t0000 == wlp` footnote everywhere downstream.
- **`t0001`** (paper's `ord5`) -- `L = {A,B,C,D,E}`, `v = (1,.75,.5,.25,0)`.
  This is a *new* formula, distinct from both existing scale-prompt scores:
  unlike `slp` it never renormalises by `P(x)` (so no clumping at exact letter
  values), and unlike `sbc` it keeps all five letters and their ordinal
  weights instead of collapsing to a binary `{A,B}` vs `{D,E}` split. It is
  exactly `wlp`'s formula applied to the five-letter ladder instead of the
  two-word one.

`t0000` vs `t0001` is therefore the paper's controlled comparison: same
readout, only the scheme's resolution changes. `wlp` / `slp` / `sbc` are left
in place unchanged -- `slp` and `sbc` are the earlier (readout, prompt)
exploration this ablation grew out of (see "Why `sbc` exists" below), not
superseded data.

## The window

Each cell is `list<struct{token, logprob}>` — the endpoint's ranked top-k at the
answer position, `window[0]` the sampled token (temperature 0), `logprob` a
natural-log probability (≤ 0). `[]` means every retry for that row failed.

- **gemma4-31b** (GenAI4Science, `top_logprobs=20`): always a 20-wide window.
  On `words` the losing answer word is usually present; on `scale` 3–5 letters
  are present and hold essentially all the mass.
- **gpt56luna** (OpenAI Responses API): a *short* window, 1–5 entries. On
  `words` ~94 % of rows return only the sampled token; on `scale` typically 1–2
  letters. The answer token still holds ~all the probability mass — the model is
  confident, the runner-up just falls outside the returned window.

Measured answer mass (5th percentile) is ≥ 0.98 on every file, both modes. That
single fact is what makes the formulation below simple: **the leftover
(non-answer) mass is small, so how we treat it barely matters** — but it must be
treated *consistently*, or two near-identical model states get very different
scores.

## The contrast readout (`wlp`, `sbc`)

Both use the same function. Given the probability on the "human" outcome
(`p_h`) and the "machine" outcome (`p_m`):

```
s = (1 + p_h − p_m) / 2
```

**Derivation.** Read the answer position as a distribution over three outcomes —
human (value 1), machine (value 0), and *everything else* (a newline, a quote, a
middle rating, a refusal token: value 0.5, no evidence). The expected value is

```
s = p_h·1 + p_m·0 + (1 − p_h − p_m)·0.5  =  (1 + p_h − p_m) / 2
```

Properties: `s = 0.5` when neither outcome has mass; `s = (1 + p_h)/2` when only
the human side is present (a confident 0.99 → 0.995, a hesitant 0.60 → 0.80 — it
is **not** pinned to 1.0 just because the losing token fell outside a short
window); within rounding of the renormalised ratio `p_h / (p_h + p_m)` whenever
the two outcomes hold all the mass (they do here); always in `[0, 1]`,
continuous, monotone. No window-edge floor, no branching, no clamp (the clip in
code is float-noise insurance).

Why not renormalise (`p_h / (p_h + p_m)`, missing side = 0)? It sends every
one-outcome window to exactly 0 or 1 — the `_v3_lp` clumping problem — which for
gpt56luna is almost every row.

**`wlp`** — `p_h`, `p_m` are the summed probability of the `human` / `machine`
tokens, over casing / leading-space spellings (`human`, ` human`, `Human`, …
all fold to `human`).

**`sbc`** — the scale prompt, scored the same way: `p_h = p_A + p_B`,
`p_m = p_D + p_E`. The middle rating `C` and any non-letter mass fall into the
`(1 − p_h − p_m)` term, i.e. count as 0.5 — which is exactly what "C = genuinely
uncertain" should mean. Drops the ordinal ruler entirely.

## The expected-value readout (`slp`)

Sum the probability of each letter, renormalise over the letters present, take
the expected value against the fixed rating scale:

```
p_k = Σ exp(logprob)   over tokens normalising to letter k,   k ∈ {A,B,C,D,E}
q_k = p_k / Σ_j p_j
s   = Σ_k q_k · v(k),        v = (A 1.0, B 0.75, C 0.5, D 0.25, E 0.0)
```

`s = 0.5` when no A–E is in the window. `v` runs A→E = human→machine (the capture
prompt's ordering; A leads so any answer-letter bias leans *human*). The linear
`v` is a convention (G-Eval, arXiv:2303.16634; arXiv:2403.00998 notes results
move with the option→scalar map).

Renormalised (not the contrast's "leftover = 0.5") because a scale window's
leftover mass is ~0. The consequence: a gpt56luna row that returns a **single**
letter collapses to that letter's exact value (only `E` → `s = 0.0`), so
`gpt56luna slp` takes ~5 values and clumps ~45 %. `sbc` does not have this
problem — its residual term keeps single-letter windows continuous.

## Why `sbc` exists — the zero-shot comparison

`slp` was a bet that the ordered rating would de-clump the score and beat the
binary readout. On the zero-shot judge sweep it did not:

| judge | mean AUROC | mean TPR@1%FPR |
|---|---|---|
| gemma `wlp` | 0.941 | 0.667 |
| gemma `slp` | 0.919 | 0.671 |
| gpt56luna `wlp` | **0.981** | **0.678** |
| gpt56luna `slp` | 0.968 | 0.552 — *and 2 of 4 splits: threshold unreachable, TPR 0* |

- **gpt56luna `slp` is broken at the operating point**: 1–2 letters per window
  → a 5-valued score → an atom at exactly 0.0 shared by machine rows and >1 %
  of human rows → no 1%-FPR threshold exists (`threshold_fpr = −inf`).
- **gemma `slp` ranks worse** (AUROC 0.773 vs `wlp` 0.832 on MG-Sci): the linear
  ruler caps non-extreme letters at 0.25 / 0.75 and the A-leads bias pulls
  borderline machines toward the human end — a *centred, non-discriminative*
  compression that shrinks the correct extreme scores as much as the wrong ones.

`sbc` keeps the scale *prompt* (a possibly different / complementary elicitation)
but scores it with the `wlp` formula, so a `sbc` vs `wlp` gap is **purely the
prompt** — 5-level rating vs binary. If `sbc ≈ wlp`, the ruler was the whole
problem. If `sbc` still trails `wlp`, the 5-level elicitation is genuinely weaker
and `slp`/`sbc` can both be dropped.

## Edge cases (all readouts)

| window | score |
|---|---|
| empty `[]` (capture failed) | **Null** — not 0.5; a pile of exact-0.5 rows would be an artefact in the detector histogram |
| present, answer mass < `MIN_ANSWER_MASS` (0.05) | 0.5 — answered, no readable evidence (e.g. a scale row that returned only `C`) |
| a `logprob` that is null / non-finite | that entry skipped (the completeness checker flags the column separately) |

`check_parquet_completeness.py` sees `wlp` / `slp` / `sbc` as ordinary bounded
`llm_judge_score_*` columns (range `[0, 1]`, degeneracy check). The suffixes end
in `p` / `c` but not `_lp`, so they do **not** trip the logprob-mode `_lp`
tie-free check — that is for the parsed `llm_judge_score_<model>_v3_lp` column.

## Run

```bash
# distributions only, writes nothing — always look first
.venv/bin/python epai/ai_detection/score/score_logprob_windows.py --dry-run

# write the score columns into the 4 core parquets, in place
# (a column that already exists is skipped unless --overwrite)
.venv/bin/python epai/ai_detection/score/score_logprob_windows.py

# one file / a scratch copy / recompute
.venv/bin/python epai/ai_detection/score/score_logprob_windows.py --only manugram_aiscientist
.venv/bin/python epai/ai_detection/score/score_logprob_windows.py --out-dir data/parquet/tmp/scored
.venv/bin/python epai/ai_detection/score/score_logprob_windows.py --overwrite
```

Deterministic, no API calls, one read + write per file.

## What the dry runs showed

`wlp` / `slp` (first pass):
- `gpt56luna wlp`: clump 2–13 % — the contrast form spreads the one-token
  windows; no collapse to {0, 1}.
- `gemma4_31b wlp`: clump 30–64 % on GB-orig / MG-GB — real judge confidence on
  machine-heavy corpora, not an artefact; matches the parsed `_v3_lp` column.
- `gemma4_31b slp`: clump 7 % (MG-Sci), 18 % (MG-Den) — graded, but see the
  ranking result above.
- `gpt56luna slp`: clump ~45 %, coarse (single-letter windows).

`sbc`:
- `gpt56luna sbc`: clump **2–3 %** (vs `slp` 43–55 %) — the residual term
  de-clumps the single-letter windows completely.
- `gemma4_31b sbc`: clump 11–36 %, close to `wlp` — as expected, same formula.
- 0 Null, ≤ 1 row at exactly 0.5 per file.

Run `run_zeroshot_performance.py --families judge` and
`calibrate_judge_baseline.py --corpora all` after writing `sbc` for the AUROC /
TPR@1%FPR verdict.

## Ensembling across models

`wlp` is the per-model column; the two models are **complementary by domain** —
gemma owns prose (GB-orig, MG-GB), gpt56luna owns science (MG-Sci) — and their
`wlp` scores correlate only r ≈ 0.2–0.4 within class. So there is real ensemble
headroom, but it is **disjunctive** ("trust whichever model is more alarmed"),
not additive: any symmetric *average* (arithmetic, geometric = mean log-odds =
product-of-experts, rank, z-scored) lets a confidently-wrong model cancel a
correct one and plateaus at TPR@1%FPR ≈ 0.60. A trained *linear* combiner is
worse still (≈ 0.45) — it cannot express "min".

`ensemble_judge_scores.py` writes the **soft-min**:

```
s = −T · log( (1/M) Σ_m exp( −s_m / T ) )     over the member P(human) columns
```

Symmetric in the members, extends to any M, and — like mean-log-odds — robust to
adding a weak member (a near-chance score is rarely the minimum). `T → 0` is the
hard min, `T → ∞` the mean; **T ≈ 0.1** is the knee.

Mean over the 4 splits (AUROC / mean TPR@1%FPR, TPR incl. the MG-Den floor):

| combine (on `gemma_wlp`, `gpt56luna_wlp`) | M=2 | M=4 (+ parsed `_v3`) | M=6 (+ 2 near-chance judges) |
|---|---|---|---|
| mean log-odds ("sum the logprobs") | 0.974 / 0.60 | 0.977 / 0.63 | 0.968 / 0.58 |
| arithmetic / rank / z-mean | 0.975 / 0.59–0.61 | 0.979 / 0.67–0.74 | **breaks: 0.44–0.52** |
| **soft-min, T = 0.1** | 0.977 / 0.59 | 0.982 / 0.65 | 0.977 / 0.64 |
| hard min | 0.983 / 0.71 | 0.984 / 0.71 | 0.981 / 0.71 |
| *best single (`gpt56luna_wlp`)* | *0.981 / 0.68* | | |

Hard `min` gives the highest mean TPR but drags MG-Sci's low-FPR operating point
(per-split TPR 0.26 vs 0.68 for `gpt56luna_wlp` alone — gemma is noisy on MG-Sci
humans). Soft-min at T ≈ 0.1 recovers most of that (MG-Sci ≈ 0.39) and stays
robust as members are added.

Output column `llm_judge_score_<name>_v3_smin_<tau>` (`.` → `p` in `<tau>`:
`smin_0p1`), P(human), auto-picked by the trainv2 `llm_judge_score_` prefix.

## Next

Raw feature-vector columns (`words`: `[p_h, p_m, logit_diff, answer_mass, …]`;
`scale`: `[q_A…q_E, entropy, answer_mass]`) for a trained combiner — deferred;
the scalars above are the zero-shot judge columns.
