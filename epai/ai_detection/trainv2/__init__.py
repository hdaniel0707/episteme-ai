"""trainv2 — a small, from-scratch rebuild of the AI-detection training runners.

The old ``epai/ai_detection/train`` tree grew a preflight report, four baseline
families, eval-only holdouts, hybrid/fusion algorithms, three training modes and
a pile of cross-corpus comparison tables. It answers a lot of questions at once,
which is the problem: no single run is legible.

trainv2 restarts from the opposite end. One experiment per script, over the
four splits of the **SCRIBE** benchmark (:mod:`.splits`), two metrics, one
figure family:

  * metrics — AUROC and TPR@1%FPR (**oracle** where there is no calibration
    slice: threshold fit on the evaluation rows themselves).
  * split  — where a classifier is trained, a grouped 70/30 split so a
    document and its rewrites never straddle the cut, seeded and repeated,
    reported as mean +- SD over seeds.
  * every script is compute -> CSV -> plot-from-CSV, with a ``--replot`` that
    redraws the figures from the CSV without recomputing.

The scripts:

  * ``calibrate_*_baseline.py`` — one fixed 1%-FPR + Youden threshold per
    score column, per split and pooled.
  * ``run_embedding_effectiveness.py`` — which embedding separates human from
    AI best, per split and on average (logistic, seeds, mean +- SD).
  * ``run_zeroshot_performance.py`` / ``run_best_zeroshot_performance.py`` —
    zero-shot AUROC vs. oracle TPR@1%FPR per family / for the hand-picked best.
  * ``run_in_domain_performance.py`` / ``run_lodo_performance.py`` — the
    trained headline (one fixed config), random 70/30 and leave-one-domain-out
    respectively, with the zero-shot picks overlaid.
  * ``run_judge_by_generator.py`` — judge x generator matrix, per split.
  * ``run_best_of_family_performance.py`` — auto-picked best of each family
    at its transported cutoff, on one figure.

Reused from the old code: only ``epai/algs`` (the pluggable classifier
registry). Everything else here is self-contained.
"""
