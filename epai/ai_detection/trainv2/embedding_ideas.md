# Multi-embedding fusion / ensembling — ideas to come back to

Working notes from a discussion on 2026-09-11, kept for later.

**Status (2026-09-12): Stage 1 implemented, Stage 0 still open.** All four
architectures on the menu below (score-level ensemble, early fusion, N-branch
gate, attention fusion) are implemented as `epai/algs/emb_stacking.py` /
`emb_early_fusion.py` / `emb_gated_n.py` / `emb_attention_fusion.py`, wired
into `run_in_domain_performance.py` and `run_lodo_performance.py` via
`_trained.py`'s `MULTI_EMB_SPECS` (default embeddings: `emb_roberta` +
`emb_styledistance` + `emb_specter2` — see `_trained.py` for the reasoning,
overridable with `--multi-embeddings`). `roberta_fusion_cuevec` /
`roberta_gated_cuevec` were removed from `TRAINED_SPECS` in the same change
(see the README's trained-detector table). Stage 0's pairwise embedding-score
correlation diagnostic was NOT done first -- the diversity-family reasoning
below was used as the prior instead -- so it remains a good follow-up for
interpreting *why* a given architecture won or lost, not a blocker on what
already ran.

The motivating idea: instead of one embedding (`emb_roberta`),
give the detector several different "views" of the same text — concatenated
(early fusion) or combined as an ensemble (late fusion) — to see whether that
buys back some of the LODO / MG-AI-Sci robustness the single-embedding probes
are missing (see `improving_sota_ideas.md` for that diagnosis). The open
question going in was explicitly **how to do this without overfitting**,
given each embedding adds hundreds of dimensions against training splits of
only a few thousand rows.

## What's already available (no new data needed)

`run_embedding_effectiveness.py` already computed 12 embeddings on every
MANUGRAM split, including MG-AI-Sci (verified present in every core split's
parquet schema): `emb_roberta` (current default), `emb_scibert`, `emb_luar`,
`emb_luar_crud`, `emb_styledistance`, `emb_specter2`, `emb_e5`, `emb_scincl`,
`emb_style_wegmann`, `emb_bge`, `emb_mpnet`, `emb_minilm`.

They cluster into three families with genuinely different inductive biases —
worth treating as a hypothesis generator, not just "more views":

- **General-purpose LM embeddings** (mostly topical/semantic content):
  `roberta` (current default), `e5`, `bge`, `mpnet`, `minilm`.
- **Authorship/style embeddings** (built to capture *how* something is
  written, largely stripped of topic): `luar`, `luar_crud`, `styledistance`,
  `style_wegmann`. `gated_mlp`'s own module docstring already flags
  `luar_crud` as scoring near-ceiling in-domain and at-or-below-chance
  off-domain — an interesting, if risky, ensemble member.
- **Scientific-domain embeddings** (trained on scientific text, relevant given
  MG-Sci / MG-AI-Sci is exactly that domain): `scibert`, `specter2`, `scincl`.

Working hypothesis: `roberta`'s LODO fragility may come partly from leaning on
topical/content signal that shifts between domains. A style embedding, built
to ignore topic, might transfer better across the same domain shift — a more
targeted reason to combine embeddings than "more views = better."

`results/trainv2/embedding_effectiveness/summary.csv` currently only reports
in-domain 70/30 performance per embedding (every one lands 0.94-1.00 AUROC
there, unsurprisingly, since in-domain is where every embedding is easy) —
it does NOT currently score any embedding on LODO / MG-AI-Sci, so there is no
existing evidence yet about which embedding(s) actually hold up OOD. That's
part of the diagnostic work below, not already answered.

## The overfitting risk, stated precisely

Each embedding is likely a few hundred dimensions; naively concatenating all
12 could easily reach 5,000-8,000+ dims against training splits of only a few
thousand rows -- a recipe for a linear probe to memorize rather than
generalize. Mitigations, roughly in order of how much they should be reached
for by default:

- **Prefer late fusion (score-level ensembling) over early fusion.** Training
  N small, independent per-embedding probes and combining their *scores* has
  far lower effective capacity than one giant concatenated input — the same
  reason a random-forest-style ensemble overfits less than one huge tree.
- **Diversity over count.** `e5` / `bge` / `mpnet` / `minilm` are all
  general-purpose sentence embedding models trained similarly — throwing all
  four in probably adds correlated noise, not signal. A small, deliberately
  diverse subset (one general LM, one style, one scientific) is likely to
  beat "all 12."
- **PCA-reduce each embedding to a fixed small dim (fit on TRAIN only)**
  before any concatenation, if early fusion is used at all.
- **Embedding / branch dropout** — randomly zero out a whole embedding's
  block per training row (not per-feature, a "modality dropout"). Forces the
  model not to lean on one embedding — an overfitting control and a mild
  domain-robustness regularizer at once, since it can't count on any single
  view being reliable. (Baldi & Sadowski's dropout analysis and the general
  "modality dropout" idea used in multi-modal learning, e.g. Neverova et al.,
  "ModDrop: Adaptive Multi-Modal Gesture Recognition," *TPAMI* 2016.)
- **Reuse the calibration-percentile trick already built for the cascade
  fix** (`_ecdf_percentile` in `run_cascade_performance.py`) for combining
  scores that live on different scales — already implemented, debugged
  machinery, no new code needed for that part.

## Architecture menu, cheapest / lowest-risk first

1. **Score-level ensemble / stacking.** Train the existing linear-probe
   pipeline once per embedding (swap `EMB_COL`, nothing else changes), then
   combine per-row scores post-hoc: unweighted average of percentile ranks,
   or a small stacked logistic regression over the per-embedding scores (fit
   on a calibration slice, same discipline as the cascade). Lowest
   overfitting risk, cheapest to run, reuses almost all existing infra.
   Citations: D. Wolpert, "Stacked Generalization," *Neural Networks* 5(2)
   (1992); L. Breiman, "Bagging Predictors," *Machine Learning* 24 (1996) —
   the general "combine independently-trained weak-ish learners" framing.
2. **Early fusion (concatenation) with PCA + embedding dropout.** One joint
   linear/shallow probe over a reduced, concatenated vector. More expressive
   than pure stacking, higher overfitting risk, needs the mitigations above.
3. **Learned gate over N embedding branches.** A generalization of the
   existing `gated_mlp` (which already gates `emb_roberta` vs. the judge cue
   vector) to gate across several embeddings instead of one embedding vs. one
   cue vector. More expressive still (per-row adaptive weighting), but the
   gate itself is something else that can overfit or degenerate to "always
   trust roberta," especially since the training domains are more homogeneous
   than the OOD target. Mixture-of-experts framing: Jacobs et al., "Adaptive
   Mixtures of Local Experts," *Neural Computation* 3(1) (1991); Shazeer et
   al., "Outrageously Large Neural Networks: The Sparsely-Gated
   Mixture-of-Experts Layer," *ICLR* 2017.
4. **Attention/transformer fusion over embeddings-as-tokens.** Treat each
   embedding as one "token," let self-attention learn how to combine them.
   Interesting for a paper, but likely overkill for a few-thousand-row
   training set — a stretch goal, not a first move. (Multi-modal transformer
   fusion generally: Lu et al., "ViLBERT," *NeurIPS* 2019, as the
   cross-modal-attention framing to cite, even though the modalities here are
   several text embeddings rather than vision+text.)

## Proposed staged experiment plan

**Stage 0 — diagnostics (cheap, do first, no new training beyond what
`run_embedding_effectiveness.py` already computes).** For each pair of
embeddings, compute how correlated their per-row decision scores are on a
shared eval set (MG-AI-Sci is the one that matters). Low correlation between
two embeddings is exactly what makes ensembling them worth doing; high
correlation means redundant. This tells us which 3-4 embeddings to actually
combine before building anything, rather than guessing from the family
clustering alone. Also worth doing here: actually score each embedding's
existing simple probe on LODO / MG-AI-Sci (not just in-domain, which is all
`embedding_effectiveness` currently reports) -- we don't yet know which
single embedding, if any, holds up better than `roberta` OOD.

**Stage 1 — score-level ensemble** (architecture #1). Per-embedding LODO
probes (reusing `_trained.py` / `robust_mlp` unchanged, just a different
`EMB_COL`), combined via (a) percentile-average over a diversity-picked
subset, (b) a stacked logistic regression over all per-embedding scores, (c)
naive average of all 12 as a baseline to show "more isn't automatically
better." Scored the same way as everything else -- MG-AI-Sci and pooled mean
at 1% FPR, same CSV schema / plots as the rest of trainv2.

**Stage 2 — early fusion**, only if Stage 1 shows real signal: concatenate
the diversity-picked subset (not all 12), PCA-reduced, with embedding
dropout, same linear-probe philosophy as `robust_mlp`.

**Stage 3 — learned gate**, only if Stage 2 also looks promising: extend
`gated_mlp` to N embedding branches.

## Open question, unresolved when this was written

Whether to start with Stage 0's correlation diagnostic (cheapest, purely
informative, no training) or jump straight to Stage 1's score-ensemble on a
hand-picked diverse subset (roberta + one style embedding + one scientific
embedding), since the family clustering above already gives a reasonable
prior on which embeddings should be diverse. Decide this before implementing.
