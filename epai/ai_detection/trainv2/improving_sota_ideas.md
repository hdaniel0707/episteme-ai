# Improving on the trained detectors: three ideas for the ICLR submission

Working notes, kept for the article write-up. Diagnosis first (grounded in
`results/trainv2/lodo_performance/performance.csv`), then three concrete
methods, each with the citations we'll need. Nothing here is speculative —
every method is scoped to what the existing `_trained.py` / LODO harness
already gives us for free (three labeled training domains, a held-out OOD
corpus, per-row zero-shot judge scores).

## The gap this is chasing

LODO (held-out domain), 1% FPR:

| detector | AUROC (mean, 3 source splits) | TPR@1% (mean) | AUROC (MG-AI-Sci) | TPR@1% (MG-AI-Sci) |
|---|---|---|---|---|
| `trained/roberta·logistic` | 0.969 | **0.748** | 0.895 | **0.297** |
| `trained/roberta⊕cuevec` (fusion) | 0.965 | 0.677 | 0.901 | 0.302 |
| `trained/roberta+gated` | 0.972 | 0.761 | 0.902 | 0.320 |
| `judge/gpt56luna_wlp` (zero-shot, logprob window) | 0.981 | 0.771 | **0.958** | **0.560** |
| `judge/gpt56luna` (zero-shot, parsed) | 0.974 | 0.631 | 0.940 | 0.320 |
| `detector/radar` (zero-shot) | 0.608 | 0.164 | 0.529 | 0.020 |

Every trained probe's AUROC survives the domain jump reasonably (0.97 → 0.90
on MG-AI-Sci). Its TPR@1%FPR does not (0.75 → 0.30) — a much bigger relative
loss than AUROC's. The zero-shot judge's logprob-window score loses far less
on both axes (0.98 → 0.96 AUROC, 0.77 → 0.56 TPR). The cue-vector hybrids
(`fusion_mlp`, `gated_mlp`) that already exist don't close this: they're
within noise of plain `roberta_logistic` on MG-AI-Sci. `gated_mlp`'s own
module docstring already names this exact phenomenon as the open question the
package is "stuck on" — ranking (AUROC) survives the domain jump, the
low-FPR tail does not.

Reframing: this is a **calibration-under-shift** problem (the decision
threshold / score distribution moves under domain change) more than a
representation problem (the ranking mostly holds up). That points at what
kind of new method is worth trying — not a bigger embedding, but a training
objective and/or regularizer that specifically targets the tail and the
domain-shift itself, plus a system-level way to borrow the judge's more
robust tail without paying for an LLM call on every row.

## #1 — Tail-aware objective (TopPush / partial-AUC loss)

**Method.** Train a linear/shallow head on `emb_roberta` with a *partial-AUC
push loss* instead of plain cross-entropy: for each mini-batch, take the
hardest (highest-scoring) `push_frac` fraction of the batch's human rows, and
add a pairwise hinge term that pushes every machine row's score above that
hard-human set specifically — `softplus(margin - (s_machine - s_hard_human))`,
averaged over the machine × hard-human pairs. This is the direct surrogate
for TPR@1%FPR (the metric that's actually failing) rather than for full AUROC
(the metric that isn't). Combined additively with the standard CE term for
stability: `loss = CE + push_weight · push_loss`.

**Why here.** Standard ERM (cross-entropy) spends its gradient budget
uniformly over the ROC curve, which is why the easy 99% already transfers
fine and the hard top-1% (the part push-based losses target) doesn't. A
"P-Norm Push" / "TopPush"-style objective concentrates capacity exactly where
the domain-shift damage is concentrated.

**Implementation.** `epai/algs/robust_mlp.py::RobustMLPAlg`
(`loss_kind="toppush"`, `push_frac`, `push_weight`, `push_margin`), built on
the shared `epai/algs/_torch_common.py` harness so architecture stays
identical to a plain linear probe (`hidden_dims: []`) — only the loss changes.

**Citations.**
- C. Rudin, "The P-Norm Push: A Simple Convex Ranking Algorithm that
  Concentrates at the Top of the List," *JMLR* 10 (2009).
- N. Li, R. Jin, Z.-H. Zhou, "Top Rank Optimization in Linear Time" (TopPush),
  *NeurIPS* 2014.
- H. Narasimhan, S. Agarwal, "A Structural SVM Based Approach for Optimizing
  Partial AUC," *ICML* 2013.
- T. Yang et al., "AUC Maximization in the Presence of Long-Tail
  Noisy Labels," and the broader partial-AUC-optimization literature for the
  low-FPR regime specifically.

## #2 — Domain-invariance regularizer (CORAL / V-REx)

**Method.** The three training splits (`gb_orig`, `mg_gb`, `mg_sci`) — or, in
a single LODO fold, the domains still in the training union — are treated as
distinct "environments." Two regularizers, both computed on the penultimate
representation (the embedding itself, when the head is a linear probe):

- **Deep CORAL** (primary): a closed-form penalty on the *second-order
  statistics* — align the feature covariance matrix across every pair of
  domains present in a batch, `‖Cov(domain_i) − Cov(domain_j)‖²_F`, no
  adversarial machinery.
- **V-REx** (companion arm): penalize the *variance of the per-domain mean
  loss* within a batch — push the model toward doing equally well across
  domains, not just averagely well.

Both reuse the domain labels the LODO harness already computes (`dom` in
`run_lodo_performance.py::_load`) — no new data or annotation needed.

**Why here.** A model trained on pooled source domains can partly key off a
domain-specific "accent" that happens to correlate with the label in the
training mix but doesn't generalize (exactly what `gated_mlp`'s docstring
worries about — the embedding "inverts" off-domain). Forcing the
representation to look statistically similar across the *known* domains is a
generalization of the isotropic-noise domain randomization already running in
`run_dr_transfer.py`, but targets the actual shift directions instead of
guessing with Gaussian noise — a natural ablation: "structured invariance vs.
blind noise."

**Deliberately deferred:** IRMv1 (gradient-penalty invariance) and DANN
(adversarial domain-invariance) — both are known to be unstable/hard to tune,
and are a reasonable stretch goal only if CORAL/V-REx underperform.

**Implementation.** Same `RobustMLPAlg` (`domain_penalty="coral"|"vrex"`,
`domain_penalty_weight`), domains threaded through via a new
`ClfAlgorithm.set_fit_domains()` hook (mirrors the existing
`set_fit_groups()`), so every other algorithm in `epai/algs/` is unaffected.

**Citations.**
- B. Sun, K. Saenko, "Deep CORAL: Correlation Alignment for Deep Domain
  Adaptation," *ECCV Workshops* 2016.
- D. Krueger et al., "Out-of-Distribution Generalization via Risk
  Extrapolation (REx)," *ICML* 2021.
- M. Arjovsky, L. Bottou, I. Gulrajani, D. Lopez-Paz, "Invariant Risk
  Minimization," arXiv:1907.02893 (2019) — cited as the deferred alternative.
- Y. Ganin et al., "Domain-Adversarial Training of Neural Networks," *JMLR*
  17 (2016) — cited as the deferred alternative.

## #3 — Judge-anchored cascade (selective escalation)

**Method.** Don't touch the trained model at all. Fit it as usual, then use a
held-out **calibration slice of the training rows** (grouped, disjoint from
the fit rows — same discipline as the existing grouped 70/30 cut) to define
an *abstention band* around the operating threshold via **split-conformal
prediction**: rank calibration rows by their nonconformity (distance of the
trained probe's score from the 1%-FPR threshold), and escalate any eval row
whose margin falls in the bottom `α` fraction (least confident) to the
zero-shot judge (`llm_judge_score_gpt56luna_v3_wlp`, the strongest zero-shot
signal in the table above). Both the trained probe's score and the judge's
score are first transformed to their own percentile rank within the
calibration slice before blending — the two scores live on unrelated numeric
scales, so splicing one in for the other on the raw scale corrupts the joint
ranking AUROC reads rather than improving it (see the results note below). A
cheap fixed-margin-band variant, not yet implemented, is a natural baseline
routing rule to compare against later.

**Why here.** This is the one idea that doesn't try to make the cheap model
itself more robust — it accepts that the cheap model will keep getting the
hard, near-threshold cases wrong under shift, and cheaply outsources exactly
those cases (and only those) to the score that's empirically ~2× more robust
in the tail. Reports as a coverage-vs-TPR@1%FPR curve (escalation budget →
recovered TPR), which is a clean, reviewer-friendly systems contribution
alongside #1/#2's purely algorithmic one.

**Implementation.** `run_cascade_performance.py`, computed post-hoc from
scores the LODO/in-domain harness already produces — no new training.
Escalation budgets `{5%, 10%, 20%}` reported as pseudo-detectors
(`cascade@5pct`, ...) on the same figures as everything else.

**Citations.**
- V. Vovk, A. Gammerman, G. Shafer, *Algorithmic Learning in a Random World*,
  Springer, 2005 — split-conformal prediction.
- A. N. Angelopoulos, S. Bates, "A Gentle Introduction to Conformal
  Prediction and Distribution-Free Uncertainty Quantification,"
  arXiv:2107.07511 (2023).
- Y. Geifman, R. El-Yaniv, "Selective Classification for Deep Neural
  Networks," *NeurIPS* 2017 — the abstention/coverage framing.
- P. Viola, M. Jones, "Rapid Object Detection using a Boosted Cascade of
  Simple Features," *CVPR* 2001 — the classic cheap-filter-then-specialist
  cascade design this borrows the shape of.

## What's comparable to what

All new arms (#1, #2, and their `#1+#2` combination) are a **linear probe on
`emb_roberta`** — identical architecture to `trained/roberta·logistic` and to
each other — so the only thing that varies across the ablation grid is the
loss / regularizer, isolating that effect from architecture choice. One
caveat: `trained/roberta·logistic` itself is fit with sklearn
(L-BFGS/liblinear), while every new arm is a PyTorch linear layer trained
with AdamW + mini-batch SGD — a minor optimizer confound worth naming in the
write-up. `robust_mlp`'s `erm_torch` arm (`loss_kind="ce"`,
`domain_penalty="none"`) is the architecture- *and* optimizer-matched control
for the other `robust_mlp` arms, so within-family comparisons (`erm_torch` vs.
`toppush` vs. `coral` vs. `vrex` vs. `toppush+coral`) are clean; the
cross-family comparison against `roberta·logistic` carries the optimizer
caveat above.

Ablation grid run by `run_robust_performance.py`:

| spec key | loss | domain penalty | weight |
|---|---|---|---|
| `erm_torch` | CE | none | — |
| `tail_toppush` | CE + push | none | — |
| `domain_coral_w1` / `_w100` / `_w10000` | CE | CORAL | 1 / 100 / 10000 |
| `domain_vrex_w1` / `_w10` / `_w100` | CE | V-REx | 1 / 10 / 100 |
| `tail_toppush_coral` | CE + push | CORAL | 1 |

(Weight swept for CORAL/V-REx after the first run at weight=1 showed both
contributing ~nothing — see the results note below.)

Plus the existing `TRAINED_SPECS` (`roberta_logistic`, `roberta_fusion_cuevec`,
`roberta_gated_cuevec`) and the zero-shot picks from `best_zeroshot.yaml`, all
on the same figures — `run_robust_performance.py` reuses the exact
`zeroshot_scatter` / `goodness_bars` / `denario_scatter` plots that
`run_in_domain_performance.py` / `run_lodo_performance.py` already produce,
not a bespoke figure.

## Results notes (first run, 2026-09-11, LODO, 1% FPR)

**#1 + #2.** Almost the entire gain over `roberta·logistic` (0.297 → ~0.335
TPR@1% on MG-AI-Sci) is explained by `erm_torch` alone — i.e. by fitting a
PyTorch linear probe with a real held-out validation split early-stopped on
AUROC, not by anything method-specific. On top of that control: TopPush adds
a small but consistent increment (+0.008 to +0.013 TPR, both MG-AI-Sci and
the pooled mean — a real, if modest, effect). CORAL at weight=1 tied
`erm_torch` to 3 decimals on every split checked (and `tail_toppush_coral`,
weight=1, was numerically identical to `tail_toppush` alone) — the penalty is
almost certainly too small to move training at all, not evidence the method
doesn't work; see the CORAL weight-sweep note above. V-REx at weight=1 was
similarly negligible, once even slightly *worse* than the control. Read the
`_w100` / `_w10000` (CORAL) and `_w10` / `_w100` (V-REx) arms before drawing
any conclusion about domain-invariance regularization here.

**#3.** An early version of `run_cascade_performance.py` blended the trained
probe's raw score and the judge's raw score directly
(`np.where(mask, judge_score, trained_score)`). Result: cascade performance
was *worse than either signal alone* (e.g. MG-AI-Sci AUROC 0.895 trained-alone
/ 0.958 judge-alone vs. 0.631 at a 5% escalation budget), and it got
monotonically *worse* with a larger escalation budget — the signature of a
scale-mismatch bug (splicing raw judge values into a raw trained-score array
corrupts the array's joint ranking rather than improving it), not a negative
result about the escalation idea. Fixed by transforming both scores to their
percentile rank within the calibration slice before blending
(`_ecdf_percentile` in `run_cascade_performance.py`) — re-run needed to read
the corrected coverage-vs-TPR@1%FPR trade-off.
