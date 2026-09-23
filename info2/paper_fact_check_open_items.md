# ICLR 2027 paper — open fact-check items

Fact-check of `latex/_ICLR_2027__AI_Detect/iclr2027_conference.tex` against the result CSVs, done 2026-09-23.
Line numbers are from that date. Search for the quoted text if the file has changed since.
Metric used throughout: AT-HM = harmonic mean of Ã = max(0, 2·AUROC−1) and T̃ = max(0, (TPR−0.01)/0.99).

Already handled (skip these):
- **Sec. 4.4 rephrasing AT-HM numbers + Fig. 7:** fixed. The "before" is now luna-only (`plot_iclr_fig7.py`, `results/trainv2/iclr_fig7/rephrase_before.csv`).
- **MG-Reph "rewrite" wording:** the author decided to keep it. The machine-original class is really the `gpt56luna:0701A`/`0811A` prompt generation, but "rewrite" is meant as "regenerate".

Everything else in the paper checked out: the main table, the abstract and intro numbers, the readout-zoom statistics, the domain-shift table, backbone selection, Table 1, the judge–generator cells, and the Fast-DetectGPT rephrase AUROCs.

---

## A. Claims that are false or misleading

### A1. "LogJudge's lead comes from scientific and AI-Scientist abstracts" — Sec. 4.1, ~line 489
- **Claim:** "LogJudge's lead comes from scientific and AI-Scientist abstracts, where perplexity-based detectors are weak; perplexity-based detectors remain stronger on regenerated news, essays, and fiction."
- **Evidence:** LogJudge (gpt-5.6-luna) also leads by a large margin on GB-orig: AT-HM 0.883 vs 0.535 for Fast-DetectGPT and 0.611 for Binoculars (Table 2). With MG-AI-Sci dropped, LogJudge still leads, 0.792 vs 0.714 / 0.712, and the paper states this in the same paragraph.
- **Source:** `results/trainv2/best_zeroshot_performance/performance.csv` (fpr=0.01 rows).
- **Suggested fix:** "LogJudge's lead comes from GB-orig and the scientific and AI-Scientist abstracts; perplexity-based detectors remain stronger on MG-GB."

### A2. "that judge is at floor regardless of generator" — judge–generator appendix, ~line 1669
- **Claim:** on MG-Sci the gemma-4-31b judge is at floor regardless of generator.
- **Evidence:** in the gemma judge's MG-Sci row, the llama-3.3-70b cell has AUROC 0.9558 and TPR 0.177, so AT-HM ≈ 0.28. The other three cells are ≈ 0.00–0.005.
- **Source:** `results/trainv2/iclr_fig8/matrix.csv`, row `mg_sci,gemma4_31b,llama3370b:0811A`.
- **Suggested fix:** "on MG-Sci that judge is at floor on three of the four generators (AT-HM ≤ 0.01; 0.28 on llama-3.3-70b text)".

## B. Small inaccuracies or overstatements

### B1. "24k rephrased texts" — contributions list, line 122
MG-Reph-GB + MG-Reph-Sci = 11,958 + 12,000 = 23,958 texts. Only about half are rephrased: 2,987 + 3,000 human-rephrased and 2,987 + 3,000 machine-rephrased, about 11,974 in total. The rest are the originals.
- **Source:** `results/trainv2/rephrase_transfer/center_of_mass.csv` (n column).
- **Fix:** "a 24k-text rephrase corpus (12k rephrased texts plus their originals)".

### B2. "changes AUROC by at most 0.03" — Sec. 4.2, line 598
gemma-4-31b on MG-Sci goes from 0.799 to 0.832, a change of 0.033 (appendix readout table). Either write "at most 0.033" or "by about 0.03".

### B3. "gemma … near chance under either readout" — Sec. 4.2, line 598
- **Claim:** "The gain does not require a competent judge: gemma-4-31b gains on every split, including the two on which it is near chance under either readout."
- **Evidence:** on MG-Sci and MG-AI-Sci, gemma's AUROC is 0.80–0.87, which is not near chance. Only its TPR@1%FPR is near floor (0.00–0.19).
- **Fix:** "including the two on which its recall at 1% FPR is near zero".

### B4. "the other three's self-family cells" — backbone appendix, line 1535
qwen3-14b is not a Manugram generator, so it has no self-family cell. Only llama-3.3-70b and gpt-oss-120b do. Fix: "the other two's".

### B5. "every detector except RADAR reaches AT-HM of at least 0.89" on Reuter — per-domain appendix, line 1444
RADAR reaches 0.997 on GB-orig Reuter, so the exception holds only on MG-GB (0.617 there). Fix: "every detector (RADAR only on GB-orig)", or restrict the sentence to MG-GB.

### B6. "their human/AI distributions overlap almost entirely" — LLM-as-judge appendix, line 1616
This is said of llama-3.3-70b, gpt-oss-120b and qwen3-14b. But on GB-orig, llama-3.3-70b has AUROC 0.896 and gpt-oss-120b 0.859, so they do not overlap almost entirely there. It is roughly true on MG-Sci and MG-AI-Sci.
- **Source:** `results/trainv2/iclr_fig2_fig3/judge_backbone_performance.csv`.
- **Fix:** "overlap heavily outside GB-orig".

### B7. Mixed statistics — Sec. 4.1, line 489
- **Claim:** "within each MG-GB domain LogJudge catches 60–100% … a single threshold … yields 55%, whereas Fast-DetectGPT loses little (86% to 84%)".
- **Problem:** LogJudge is given as a range of per-domain values, but Fast-DetectGPT's 86% is the mean of its per-domain values (0.767, 0.892, 0.913). This is not false, but the two are presented differently.
- **Fix:** use means for both (LogJudge ≈ 83% → 55%, Fast-DetectGPT ≈ 86% → 84%), or ranges for both (LogJudge 60–100%, Fast-DetectGPT 77–91%).
- **Caveat:** these per-domain values are unverified (see C2).

### B8. "current detectors are near ceiling on the existing ones" — intro, line 115 (also the abstract)
This is only true for Binoculars with its original Falcon-7B backbone (AUROC 1.00 on GB-orig, Table 1). With the Llama-3.1-8B backbone used in the main table, Fast-DetectGPT and Binoculars reach only TPR@1%FPR 0.40 / 0.48 on GB-orig (AT-HM 0.535 / 0.611). BaseJudge gpt-5.6-luna reaches 0.54. A reviewer may notice this tension.
- **Fix:** qualify it, e.g. "the original Binoculars configuration is at ceiling on …".

## C. Could not verify (no result artifact found)

Most important: C1 and C2. These tables are in the paper, but no CSV reproduces them.

### C1. Embedding table (`tab:embedding-ranking`, around line 1746)
- **Paper values:** roberta 0.999 ± 0.001 / 0.988 ± 0.016, luar 0.997 / 0.958, …
- **Only run on disk:** `results/old_trainv2/embedding_effectiveness/summary.csv` (Sep 9). Its values do **not** match: roberta TPR 0.998, styledistance TPR 0.979 vs 0.904 in the paper, scibert 0.967 vs 0.929.
- **Why:** that run also uses the old splits: GB-orig with 8,982 texts (the old standardized version) and `mg_den` (160 texts) instead of MG-AI-Sci.
- **To do:** find where the paper numbers came from, or re-run `epai/ai_detection/trainv2/run_embedding_effectiveness.py` on the current splits and update the table. The prose after the table ("style-only encoders lose most … on MG-AI-Sci") would also need checking.

### C2. Per-domain table (`tab:general-results-domains`, ~lines 1436–1524)
- **Status:** the pooled "All" rows and the four-split mean match `performance.csv`. The per-domain rows (Essay / Reuter / WP / CS / Math / Soc) have no artifact.
- **Why:** `run_best_zeroshot_domains.py` should write `results/trainv2/best_zeroshot_performance/performance_by_domain.csv`, but that file does not exist.
- **Also depends on it:** the text claims "60–100%", "0.602–1.000", "0.941–1.000", "0.767–0.913" and "BaseJudge best on all three MG-Sci domains".
- **To do:** run that script and compare.

### C3. Log-probability reproducibility table (`tab:logprob_reproducibility`, ~line 1699)
The 26% / 13% / 10% / 8% window identity, 96–100% verdict agreement and max |Δs| 0.843 have no result file or script located. The limitations section's "4% of verdicts flip" depends on it.

### C4. "5th percentile of P(x) ≥ 0.98" — Sec. 2.1 (~line 339) and the LogJudge appendix (~line 802)
- **Status:** no artifact.
- **How to verify:** compute the answer mass P(x) from the `llm_judge_logprobs_*` window columns in the 4 core parquets, for both judges.

### C5. Readout-zoom ROC claims (~line 1414)
"FusionJudge ROC above both readouts from about 0.4% FPR up to 5%" and "29% vs 11% TPR at 0.1% FPR". `results/trainv2/iclr_readout_zoom/stats_gpt56luna_all.csv` only covers the 1% FPR point. Check against `plot_iclr_readout_zoom.py`.

### C6. "10 fields (8–40 texts each)" for MG-AI-Sci — Table 1 (line 163) and the per-domain table caption
Not checked. Count the domain values in the MG-AI-Sci parquet.

---

## Notes for whoever picks this up
- **Project rules:** read `CLAUDE.md` first. Do not run `uv` commands without asking. To run a script, `.venv/bin/python <script>` works (the base interpreter is `/usr/bin/python3.12`, which exists), with `MPLCONFIGDIR=$TMPDIR`.
- **Git:** `latex/` is gitignored, so edits there don't show in `git diff`.
