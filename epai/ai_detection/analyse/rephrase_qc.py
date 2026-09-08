r"""Did the rephrase arm land where it was meant to? The calibration check.

``rephrase_build.py`` runs one moderate, domain-aware rephrase of each human
text with no pilot. This is the check after the fact: is the rephrase strong
enough that surface form clearly moved, and restrained enough that the content
was preserved rather than the text regenerated?

Two families of metric, per (human, rephrase) pair matched on ``id``:

  surface   ``rephrase_surface_jaccard`` (token-set overlap, LOW = more
            changed), ``rephrase_seqmatch_ratio`` (difflib char ratio, LOW =
            more changed), ``rephrase_len_ratio`` — all written by the append
            step.
  semantic  cosine between the pair's ``emb_e5`` / ``emb_mpnet`` (content —
            should stay HIGH) and ``emb_luar_crud`` / ``emb_styledistance``
            (authorship/style — expected to MOVE). Computed here; ``--write``
            stores them on the rephrase rows as ``rephrase_semsim_e5`` etc.

GATE (rough, per domain):
  semantic cosine (e5) median  > 0.85     content preserved
  surface jaccard median       in [0.25, 0.60]   real rewrite, not a copy / not regenerated
  length ratio in [0.7, 1.4]   > 95% of rows

A domain that fails high (barely changed) or low (drifted) is fixed by revising
that domain's hint in a NEW prompt id and regenerating just that domain.

It also prints the first real result: mean ``llm_judge_score_gpt56luna_v3`` and
``binoculars_score`` for human vs rephrase, next to the full-text-generation
gpt56luna rows in the two v3f corpora — rephrase is expected to sit between
human and full-gen.

    python epai/ai_detection/analyse/rephrase_qc.py
    python epai/ai_detection/analyse/rephrase_qc.py --write
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from epai.utils.parquet_utils import load_parquet, save_parquet

BASE_DIR = Path(__file__).resolve().parents[3]
TARGET = BASE_DIR / "data/parquet/rephrase_v1_multi.parquet"
GB_SRC = BASE_DIR / "data/parquet/ghostbuster_v3f_multi.parquet"
SCI_SRC = BASE_DIR / "data/parquet/science_v3f_multi_0811A.parquet"

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

SEM_EMB = {"e5": "emb_e5", "mpnet": "emb_mpnet"}
STYLE_EMB = {"luar": "emb_luar_crud", "styledistance": "emb_styledistance"}
JUDGE_COL = "llm_judge_score_gpt56luna_v3"
BINOX_COL = "binoculars_score"

GATE_SEMSIM_E5 = 0.85
GATE_JACCARD = (0.25, 0.60)
GATE_LEN = (0.7, 1.4)


def cosine(a, b) -> float:
    if a is None or b is None:
        return np.nan
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(a @ b / (na * nb))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--parquet", type=Path, default=TARGET)
    ap.add_argument("--write", action="store_true",
                    help="store the semantic/style cosines on the rephrase rows")
    args = ap.parse_args()

    if not args.parquet.is_file():
        print(f"{RED}not found: {args.parquet}{RESET}")
        return 1

    df = load_parquet(args.parquet)
    human = df[df["generated_by"] == "human"].set_index("id")
    reph = df[df["generated_by"] != "human"].copy()
    if reph.empty:
        print(f"{RED}no rephrase rows in {args.parquet} — run the append step.{RESET}")
        return 1
    label = reph["generated_by"].iloc[0]
    print(f"{BOLD}rephrase QC{RESET}  {args.parquet.name}  "
          f"({len(human)} human, {len(reph)} {label})\n")

    # ---- per-pair cosines --------------------------------------------------
    sims: dict[str, list[float]] = {f"semsim_{k}": [] for k in SEM_EMB}
    sims |= {f"style_sim_{k}": [] for k in STYLE_EMB}
    for _, row in reph.iterrows():
        h = human.loc[row["id"]] if row["id"] in human.index else None
        for k, col in {**SEM_EMB, **STYLE_EMB}.items():
            key = ("semsim_" if k in SEM_EMB else "style_sim_") + k
            sims[key].append(np.nan if h is None else
                             cosine(h.get(col), row.get(col)))
    for key, vals in sims.items():
        reph[f"rephrase_{key}"] = vals

    # ---- per-domain report + gate ---------------------------------------- #
    metric_cols = (["rephrase_surface_jaccard", "rephrase_seqmatch_ratio",
                    "rephrase_len_ratio"]
                   + [f"rephrase_{k}" for k in sims])
    print(f"{BOLD}  per-domain medians{RESET}")
    hdr = "  " + f"{'domain':<18}" + "".join(f"{m.replace('rephrase_', '')[:13]:>15}"
                                             for m in metric_cols)
    print(hdr)
    gate_fail = []
    for dom, g in reph.groupby("domain"):
        cells = "".join(f"{g[m].median():>15.3f}" for m in metric_cols)
        print(f"  {dom:<18}{cells}")
        jac_med = g["rephrase_surface_jaccard"].median()
        e5_med = g["rephrase_semsim_e5"].median()
        len_ok = g["rephrase_len_ratio"].between(*GATE_LEN).mean()
        issues = []
        if not (e5_med > GATE_SEMSIM_E5):
            issues.append(f"e5 semsim {e5_med:.3f} <= {GATE_SEMSIM_E5} (content drift)")
        if jac_med > GATE_JACCARD[1]:
            issues.append(f"jaccard {jac_med:.3f} > {GATE_JACCARD[1]} (too weak)")
        if jac_med < GATE_JACCARD[0]:
            issues.append(f"jaccard {jac_med:.3f} < {GATE_JACCARD[0]} (too strong)")
        if len_ok < 0.95:
            issues.append(f"only {len_ok:.0%} within length band")
        if issues:
            gate_fail.append((dom, issues))

    print()
    if gate_fail:
        for dom, issues in gate_fail:
            print(f"{YELLOW}  {dom}: {'; '.join(issues)}{RESET}")
    else:
        print(f"{GREEN}  gate: all domains within band{RESET}")

    # ---- human vs rephrase vs full-gen ---------------------------------- #
    print(f"\n{BOLD}  mean score by class{RESET}  {DIM}(judge: P(human), 1=human; "
          f"binoculars: lower=more AI){RESET}")
    rows = [("human (this corpus)", df[df["generated_by"] == "human"]),
            (f"{label} (rephrase)", reph)]
    for name, src, lab in (("gpt56luna:0811A (science full-gen)", SCI_SRC, "gpt56luna:0811A"),
                           ("gpt56luna:0701A (ghostbuster full-gen)", GB_SRC, "gpt56luna:0701A")):
        if src.is_file():
            have = set(pq.ParquetFile(src).schema_arrow.names)
            cols = ["generated_by"] + [c for c in (JUDGE_COL, BINOX_COL) if c in have]
            fg = pd.read_parquet(src, columns=cols)
            rows.append((name, fg[fg["generated_by"] == lab]))
    print(f"  {'class':<38}{'judge':>10}{'binoculars':>14}{'n':>8}")
    for name, g in rows:
        j = g[JUDGE_COL].mean() if JUDGE_COL in g else np.nan
        b = g[BINOX_COL].mean() if BINOX_COL in g else np.nan
        print(f"  {name:<38}{j:>10.3f}{b:>14.3f}{len(g):>8}")

    if args.write:
        keep_cols = [c for c in df.columns]
        for key in sims:
            col = f"rephrase_{key}"
            if col not in df.columns:
                df[col] = np.nan
                keep_cols.append(col)
            df.loc[reph.index, col] = reph[col]
        save_parquet(df[keep_cols], args.parquet)
        print(f"\n{GREEN}wrote{RESET} {len(sims)} cosine column(s) → {args.parquet}")

    return 1 if gate_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
