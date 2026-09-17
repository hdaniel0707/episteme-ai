# Prompt-diversity LogJudge pilot

An extension of LogJudge's `ord2` score (`s = (1 + p_human - p_machine) / 2`,
`score_logprob_windows.py`): instead of reading it off **one** evidence prompt
(`v3`), read it off **several** differently-framed evidence prompts and
average the results. The paper already flags "combining multiple judges" as a
natural extension it left for future work (`sec:method_logjudge`); this is the
prompt-side sibling of that — same judge model, several prompts, mean-pooled.

Status: **piloted and validated on MG-AI-Sci (200 rows, both judge models).
Scale-up to the 3 remaining core splits, `gpt56luna` only, is written and
ready to run — not yet run.**

## The pipeline, end to end

| stage | script | what it does |
|---|---|---|
| 1. candidates | `cfg/ai_detection/judge_prompts/candidates/cand01..10.yaml` | 10 hand-drafted evidence prompts, each shaped like `v3` (genre framing, human/machine cue lists, `CALIBRATION` heading) but with a different detection philosophy (process-forensics, predictability, editorial-reader, adversarial-generator, discourse-structure, voice/idiolect, error-forensics, field-convention, holistic/comparative, effort-allocation). Auto-registered into `add_llm_judge_scores.PROMPTS` at import — usable anywhere `--prompt v3` is. |
| 2. selection | `select_diverse_judge_prompts.py` | embeds `v3` + all 10 candidates with `roberta-base` (mean-pooled), picks the pair `{A, B}` maximising `min(d(v3,A), d(v3,B), d(A,B))` — the trio's worst-case pairwise distance. **Winner: `cand03` (editorial-reader) + `cand10` (effort-allocation).** |
| 3. pilot capture | `run_prompt_pilot_capture.py --prompts cand03 cand10` | parsed-mode sanity check + words-mode logprob capture, both judge models, MG-AI-Sci only. Already run. |
| 4. merge | `merge_captured_logprobs.py` | folds the capture sidecars into the real parquet. Already run for the pilot. |
| 5. score | `score_logprob_windows.py` | turns the new prompts' logprob windows into `..._wlp` (ord2) columns — generalized from a `v3`-only hardcode to any prompt id, so this now also handles `cand03`/`cand10` (and any future prompt) automatically. |
| 6. mean ensemble | `ensemble_judge_scores.py --combine mean` | new mode alongside the existing `softmin` (which combines across judge *models*): plain per-row mean across judge *prompts*, same model. |
| 7. scale-up | `run_prompt_pilot_scaleup.py` | steps 3–6 (capture → merge → score → ensemble) at full scale, `gpt56luna` only, on `ghostbuster_orig` / `manugram_gb` / `manugram_sci` (MG-AI-Sci is already done). **Not yet run.** |

## Column naming glossary

Given `<model>` ∈ {`gpt56luna`, `gemma4_31b`} and prompts `v3` (production),
`cand03`, `cand10`:

| column | meaning |
|---|---|
| `llm_judge_score_<model>_<promptid>` | **parsed** mode: the model writes the probability (structured output), one prompt |
| `llm_judge_score_<model>_<promptid>_wlp` | **logprob/ord2** mode: read off the `human`/`machine` answer-token logprobs, one prompt |
| `llm_judge_score_<model>_3p_promptmean` | **the new algorithm** — wlp mean of `{v3, cand03, cand10}`, all 3 prompts |
| `llm_judge_score_<model>_v3_c03_c10_parsed_promptmean` | parsed-mode counterpart of the above (extra comparison point, not the headline result) |
| `llm_judge_score_<model>_v3_c03_wlp_promptmean` | wlp mean of `{v3, cand03}` only — dropping `cand10`, to check whether 3-way beats 2-way |
| `llm_judge_score_<model>_v3_c03_parsed_promptmean` | parsed-mode counterpart of the above |

`_promptmean` means "averaged across **prompts**" — it says nothing about
which score type went in; that's carried entirely by which member columns
were passed to `--members`, encoded in `--name`. (This tripped us up once
already — see the naming glossary above before trusting a `_promptmean`
column's contents from its name alone; check what `--members` it was built
from.)

## Pilot results (MG-AI-Sci, `--fpr 0.01`, goodness = HM(2·AUROC−1, TPR@1%FPR))

| detector | AUROC | TPR@1%FPR | goodness |
|---|---|---|---|
| `gpt56luna_3p_promptmean` (wlp, v3+c03+c10) | 0.960 | 0.74 | **0.82** |
| `gpt56luna_v3_c03_c10_parsed_promptmean` (parsed, v3+c03+c10) | 0.957 | 0.59 | 0.71 |
| `gpt56luna_v3_c03_wlp_promptmean` (wlp, v3+c03 only) | 0.961 | 0.57 | 0.70 |
| `gpt56luna_v3_wlp` (plain v3, baseline) | 0.958 | 0.56 | 0.69 |
| `gpt56luna_cand03_wlp` (alone) | 0.954 | 0.54 | 0.68 |
| `gpt56luna_v3_c03_parsed_promptmean` (parsed, v3+c03 only) | 0.954 | 0.46 | 0.61 |
| `gpt56luna_v3` (plain v3, parsed baseline) | 0.940 | 0.32 | 0.46 |
| `gpt56luna_cand10_wlp` (alone) | 0.945 | 0.26 | 0.39 |
| `fastdetect/llama31-8b` | 0.855 | 0.26 | 0.37 |
| `binoculars/llama31-8b` | 0.845 | 0.22 | 0.33 |

Takeaways:
- The wlp mean-of-3 is the best detector on the pilot split, beating plain
  `v3_wlp`, both non-judge zero-shot baselines, and every individual prompt.
- The gain isn't purely a logprob artifact — the **parsed**-mode 3-way mean
  (0.71) also clearly beats the plain parsed baseline (0.46).
- 3-way beats 2-way (0.82 vs 0.70) even though `cand10_wlp` is the *weakest*
  prompt alone (0.39) — a "wisdom of crowds" effect, not just riding the
  strongest prompt.
- `gemma4-31b` was mixed/inconclusive on this pilot (individual candidates hit
  `TPR=0` at 1%FPR; at 5%FPR `cand10_wlp` alone was actually the best
  `gemma4-31b` detector, beating its own mean-of-3) — **this is why the
  scale-up is `gpt56luna`-only.**

Full 13/19-detector comparisons and charts: `results/trainv2/prompt_pilot/`
(all judge variants x both models) and `results/trainv2/prompt_pilot_gpt56luna/`
(6-detector focused: binoculars, fastdetect, `gpt56luna` parsed/wlp baselines,
3-way parsed/wlp means). Configs:
`cfg/ai_detection/trainv2/best_zeroshot_prompt_pilot.yaml` and
`..._gpt56luna.yaml`.

## Run lines — scale-up (not yet run)

Row counts: `ghostbuster_orig` 5,988 · `manugram_gb` 14,960 · `manugram_sci`
15,000 → **~35,950 rows**. Default is `gpt56luna` only, `cand03`+`cand10`
only, words-mode capture only (no parsed-mode pass — that's `--include-parsed`,
roughly doubles the OpenAI calls, off by default since only the wlp mean is
"the new algorithm").

```bash
# 1. sanity check the plan first -- no calls made
uv run python epai/ai_detection/score/run_prompt_pilot_scaleup.py --dry-run

# 2. the real run (one confirmation; --yes to skip it, e.g. for nohup)
uv run python epai/ai_detection/score/run_prompt_pilot_scaleup.py

# faster throughput if the rate limit allows it:
uv run python epai/ai_detection/score/run_prompt_pilot_scaleup.py --workers 8

# to also get the parsed-mode 3-way mean at full scale (extra cost):
uv run python epai/ai_detection/score/run_prompt_pilot_scaleup.py --include-parsed
```

It's resumable (every underlying script skips already-scored rows), and a
failure in one stage doesn't stop the rest of the batch.

## Run lines — after the scale-up finishes

Re-run the focused 6-detector comparison **without** `--splits`, so it covers
all 4 core splits (the config's columns already exist on every split once the
scale-up is done — same names as the pilot):

```bash
uv run python epai/ai_detection/trainv2/run_best_zeroshot_performance.py \
    --config cfg/ai_detection/trainv2/best_zeroshot_prompt_pilot_gpt56luna.yaml \
    --out-dir results/trainv2/prompt_pilot_gpt56luna_full
```

This gives the per-split breakdown plus the pooled `mean` row/bar across all
4 splits — the real test of whether the mean-of-3 result generalises past the
200-row MG-AI-Sci pilot. If it holds up, add
`llm_judge_score_gpt56luna_3p_promptmean` to
`cfg/ai_detection/trainv2/best_zeroshot.yaml` and `judge_baseline_calibrated.yaml`
to fold it into the paper's main tables/figures.
