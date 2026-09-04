"""trainv2 — a small, from-scratch rebuild of the AI-detection training runners.

The old ``epai/ai_detection/train`` tree grew a preflight report, four baseline
families, eval-only holdouts, hybrid/fusion algorithms, three training modes and
a pile of cross-corpus comparison tables. It answers a lot of questions at once,
which is the problem: no single run is legible.

trainv2 restarts from the opposite end. One experiment at a time, one input
shape, two metrics, one figure family:

  * input  — a ``*_multi_*`` parquet only (human + >=2 generators, one row per
    ``(source document, generator)``, sharing a group id).
  * metrics — AUROC and TPR@1%FPR, both **oracle** (threshold fit on the
    evaluation rows themselves). No transported-threshold machinery.
  * split  — one grouped train/test split so a document and its rewrites never
    straddle the cut. Seeded, repeated, reported as mean +- std.

The first experiment (``run_in_domain.py``) evaluates **data-generator
effectiveness**: for every generator in the file, train human-vs-that-generator
over the embedding x algorithm grid, per domain and pooled, and plot each
training as a point in (AUROC, TPR@1%FPR) space.

Reused from the old code: only ``epai/algs`` (the pluggable classifier
registry). Everything else here is self-contained.
"""
