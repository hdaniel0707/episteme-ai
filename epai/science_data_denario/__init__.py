"""Build the Denario human-control holdout (`denario_hc`).

A three-stage companion to `epai/ai_detection/prepare/run_holdout_pipeline.py`.
The existing `denario` holdout is all-machine (80 AI-written paper abstracts), so
it can only report a detection rate — no FPR, no ROC, and every "everything
collapses" claim in info/iclr27_plan.html is unfalsifiable on it.

This package adds a matched human negative class:

    fetch_human_candidates  OpenAlex, one query set per Denario domain, pre-2021
                            so the "human" label is safe.
    build_denario_hc        one real human abstract per Denario paper, matched
                            within domain on SPECTER2 topic + abstract length
                            (optimal 1:1 assignment). 80 machine + 80 human.
    ocr_clean               repair PDF-extraction artifacts (run-together words,
                            lost ligatures, spaces before punctuation) in BOTH
                            classes — the Denario abstracts came out of PDFs, the
                            human ones came clean from OpenAlex, and left alone
                            that difference is a class label.

Then the standard chain takes over: standardize_text.py (whitespace: flat, the
same setting science_v3 used) then run_holdout_pipeline.py --from embed.

The 80 human OpenAlex work ids are written to the output so they can be excluded
from any future training fetch (science_v5 / Track 2 adds Astronomy and
Neuroscience, which overlap this holdout's domains).
"""
