# Scoring the LLM-judge logprob windows

`score_logprob_windows.py` turns the captured window columns
(`llm_judge_logprobs_<model>_v3_{words,scale}`, see `logprob_capture.md`) into
plain **P(human)** score columns:

| from | to | readout |
|---|---|---|
| `llm_judge_logprobs_<model>_v3_words` | `llm_judge_score_<model>_v3_wlp` | words, **contrast** |
| `llm_judge_logprobs_<model>_v3_scale` | `llm_judge_score_<model>_v3_slp` | scale, **expected value** |
| `llm_judge_logprobs_<model>_v3_scale` | `llm_judge_score_<model>_v3_sbc` | scale, **contrast** |

`<model>` ∈ `gemma4_31b`, `gpt56luna` → **6 columns per parquet**, across the 4
core splits. All are P(human): **1.0 = human, 0.0 = machine**,
`machine_is_high=False`, same `llm_judge_score_` prefix as every other judge, so
`run_zeroshot_performance.py`, `calibrate_judge_baseline.py`,
`run_judge_youden_calibration.py`, `run_judge_by_generator.py` pick them up on
re-run with no config change (labels `gemma4_31b_wlp`, `gpt56luna_sbc`, …).

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
.venv/bin/python epai/ai_detection/score/score_logprob_windows.py --only manugram_den
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

## Next

Raw feature-vector columns (`words`: `[p_h, p_m, logit_diff, answer_mass, …]`;
`scale`: `[q_A…q_E, entropy, answer_mass]`) for a trained combiner — deferred;
the scalars above are the zero-shot judge columns.
