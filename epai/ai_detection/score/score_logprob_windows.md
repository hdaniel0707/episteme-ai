# Scoring the LLM-judge logprob windows

`score_logprob_windows.py` turns the four captured window columns
(`llm_judge_logprobs_<model>_v3_{words,scale}`, see `logprob_capture.md`) into
four plain **P(human)** score columns:

| from | to | answer read |
|---|---|---|
| `llm_judge_logprobs_<model>_v3_words` | `llm_judge_score_<model>_v3_wlp` | `human` / `machine` |
| `llm_judge_logprobs_<model>_v3_scale` | `llm_judge_score_<model>_v3_slp` | `A` (human) … `E` (machine) |

`<model>` ∈ `gemma4_31b`, `gpt56luna` → 8 columns across the 4 core parquets.
All are P(human): **1.0 = human, 0.0 = machine**, `machine_is_high=False`, same
orientation and `llm_judge_score_` prefix as every other judge, so
`run_zeroshot_performance.py`, `calibrate_judge_baseline.py`,
`run_judge_youden_calibration.py`, `run_judge_by_generator.py` pick them up on
re-run with no config change (labels `gemma4_31b_wlp`, `gpt56luna_slp`, …).

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

## `words` → `wlp`

Sum the probability of each answer word over its casing / leading-space
spellings (`human`, ` human`, `Human`, … all fold to `human`):

```
p_h = Σ exp(logprob)   over tokens normalising to "human"
p_m = Σ exp(logprob)   over tokens normalising to "machine"
```

Score:

```
s = (1 + p_h − p_m) / 2
```

**Derivation.** Read the answer position as a distribution over three outcomes —
`human` (value 1), `machine` (value 0), and *everything else* (a newline, a
quote, a refusal token: value 0.5, i.e. no evidence). The expected value is

```
s = p_h·1 + p_m·0 + (1 − p_h − p_m)·0.5  =  (1 + p_h − p_m) / 2
```

Properties:

- **no answer words in the window** → `p_h = p_m = 0` → `s = 0.5`
- **only `human`** (`p_m = 0`) → `s = (1 + p_h)/2`; a confident 0.99 → 0.995, a
  hesitant 0.60 → 0.80. The score is not pinned to 1.0 just because the losing
  word fell outside a short window — this is what keeps gpt56luna's `words`
  score from collapsing to {0, 1} on 94 % of its rows.
- **only `machine`** → `s = (1 − p_m)/2`
- **both present** → within rounding of the renormalised ratio
  `p_h / (p_h + p_m)` whenever the two words hold all the mass (they do here)
- always in `[0, 1]`, continuous, monotone ↑ in `p_h` and ↓ in `p_m`. No
  window-edge floor, no branching on which words are present, no clamp needed
  (the clip in the code is float-noise insurance only).

Why not renormalise (`p_h / (p_h + p_m)`, missing side = 0)? It sends every
one-word window to exactly 0 or 1 — the `_v3_lp` clumping problem — which for
gpt56luna is almost every row.

## `scale` → `slp`

Sum the probability of each letter, renormalise over the letters present, take
the expected value against the fixed rating scale:

```
p_k = Σ exp(logprob)   over tokens normalising to letter k,   k ∈ {A,B,C,D,E}
q_k = p_k / Σ_j p_j
s   = Σ_k q_k · v(k),        v = (A 1.0, B 0.75, C 0.5, D 0.25, E 0.0)
```

- **no A–E in the window** → `s = 0.5`
- `v` runs A→E = human→machine (the capture prompt's ordering; A leads so any
  answer-letter bias leans *human*, the safe side for a low-FPR operating point)
- the linear `v` is a convention (G-Eval, arXiv:2303.16634; arXiv:2403.00998
  notes results move with the option→scalar map). It is fine as a zero-shot
  score; a *trained* combiner would take `q_A…q_E` directly. Not built here —
  ask if you want the raw-vector column.

Renormalisation (rather than the words-style "leftover = 0.5") because a scale
window's leftover mass is ~0 — the letters already hold everything. The one
consequence: a gpt56luna row that returns a **single** letter collapses to that
letter's exact value (only `E` → `s = 0.0`), so `gpt56luna slp` is coarse
(clump ~45 %). `gemma4_31b slp`, with 3–5 letters per window, is the graded one
(clump 7–18 % on the hard splits) and is the column to prefer.

## Edge cases (both modes)

| window | score |
|---|---|
| empty `[]` (capture failed) | **Null** — not 0.5; a pile of exact-0.5 rows would be an artefact in the detector histogram |
| present, answer mass < `MIN_ANSWER_MASS` (0.05) | 0.5 — answered, no readable evidence |
| a `logprob` that is null / non-finite | that entry skipped (the completeness checker flags the column separately) |

`check_parquet_completeness.py` sees `wlp` / `slp` as ordinary bounded
`llm_judge_score_*` columns (range `[0,1]`, degeneracy check). The suffixes end
in `lp` but not `_lp`, so they do **not** trip the logprob-mode `_lp` tie-free
check — that is for the parsed `llm_judge_score_<model>_v3_lp` column, a
different thing.

## Run

```bash
# distributions only, writes nothing — always look first
.venv/bin/python epai/ai_detection/score/score_logprob_windows.py --dry-run

# write the 8 columns into the 4 core parquets, in place
.venv/bin/python epai/ai_detection/score/score_logprob_windows.py

# one file / a scratch copy
.venv/bin/python epai/ai_detection/score/score_logprob_windows.py --only manugram_den
.venv/bin/python epai/ai_detection/score/score_logprob_windows.py --out-dir data/parquet/tmp/scored

# recompute an existing column
.venv/bin/python epai/ai_detection/score/score_logprob_windows.py --overwrite
```

Deterministic, no API calls, one read + write per file. Refuses to replace an
existing `wlp` / `slp` column without `--overwrite`.

## What the dry run showed (first pass)

- `gpt56luna wlp`: clump 2–13 % — the `(1+p_h−p_m)/2` form spreads the
  one-token windows well; no collapse to {0, 1}.
- `gemma4_31b wlp`: clump 30–64 % on `ghostbuster_orig` / `manugram_gb` — real
  judge confidence on machine-heavy corpora (gemma decisively answers `machine`,
  losing word absent), not a scoring artefact; matches the parsed `_v3_lp`
  column's known behaviour. Well spread on `manugram_sci` / `manugram_den`.
- `gemma4_31b slp`: the graded signal — clump 7 % (mg_sci), 18 % (mg_den).
- `gpt56luna slp`: clump ~45 %, coarse, for the single-letter-window reason above.
- 0 Null, 0 (or 1) rows at exactly 0.5 — coverage is complete and no row
  answered fully off-format.

## Next

Raw feature-vector columns (`words`: `[p_h, p_m, logit_diff, answer_mass, …]`;
`scale`: `[q_A…q_E, entropy, answer_mass]`) for a trained combiner — deferred;
the 4 scalars above are the zero-shot judge columns.
