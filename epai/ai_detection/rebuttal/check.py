r"""Quality check of every planned rebuttal sidecar (``status.PLAN``) before it is
evaluated. Makes no calls; reads the sidecars and the master parquets.

    .venv/bin/python -m epai.ai_detection.rebuttal.check
    .venv/bin/python -m epai.ai_detection.rebuttal.check --model gpt56luna -v

Per run (FAIL = do not evaluate yet; WARN = look at it, may be fine):

  FAIL  sidecar missing; expected (stem, uid, rep) rows without an answer
        (never sent, or failed: re-run the run_*.sh script); duplicate keys;
        answered rows with no token windows; prob runs: no number in [0, 1] in the
        answer, or the stored quick score differs from the re-parsed answer;
        label runs: no score.
  WARN  rows not in the planned subset; answers that are not a bare number;
        label runs: first answer token not one of the labels, or < 5% label mass
        at position 0 (scored 1/2); top-window shorter than 20 at the first
        answer position; more than one served model version; the expected number
        of decimals (2 for base_v3, 4 for base_hp4) missing on > 5% of rows.
  INFO  decimals written, distinct values, A2 captured mass, repeat agreement.

Writes ``<out>/check.csv`` (one row per run) and exits 1 if any run FAILs.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from epai.ai_detection.rebuttal.capture import _as_written
from epai.ai_detection.rebuttal.conditions import CONDITIONS, WORKSPACE, build_subset, load_core
from epai.ai_detection.rebuttal.readouts import (
    MIN_ANSWER_MASS,
    add_readouts,
    first_token_is_label,
    is_bare_number,
    n_decimals,
)
from epai.ai_detection.rebuttal.status import PLAN

EXPECTED_DECIMALS = {"base_v3": 2, "base_v3_t1": 2, "base_hp4": 4,
                     "base_default": 2, "base_default_hp4": 4}


def check_run(model: str, cond_name: str, subset: str, reps: int, expected: pd.DataFrame,
              workspace: Path) -> dict:
    cond = CONDITIONS[cond_name]
    path = workspace / model / f"{cond_name}__{subset}.parquet"
    rec = {"model": model, "condition": cond_name, "subset": subset, "reps": reps,
           "expected": len(expected) * reps, "fail": [], "warn": [], "info": []}
    if not path.is_file():
        rec["fail"].append("sidecar missing")
        return rec
    df = pd.read_parquet(path)
    key = ["stem", "uid", "rep"]
    want = pd.MultiIndex.from_tuples(
        [(s, u, r) for s, u in zip(expected["stem"], expected["uid"]) for r in range(reps)],
        names=key)
    have = pd.MultiIndex.from_frame(df[key].astype({"rep": int}))
    ok = df["answer"].notna().to_numpy()
    rec["rows"] = len(df)

    dup = int(have.duplicated().sum())
    if dup:
        rec["fail"].append(f"{dup} duplicate (stem, uid, rep) keys")
    answered = have[ok]
    missing = want.difference(answered)
    rec["missing"] = len(missing)
    if len(missing):
        by = Counter(missing.get_level_values("stem"))
        rec["fail"].append(f"{len(missing)} expected rows without an answer "
                           f"({dict(by)}) -- re-run the script")
        errs = Counter(str(e)[:90] for e in df.loc[~ok, "error"].dropna())
        for msg, c in errs.most_common(3):
            rec["info"].append(f"error x{c}: {msg}")
    extra = have.difference(want)
    if len(extra):
        rec["warn"].append(f"{len(extra)} rows not in the planned subset (ignored by evaluate)")

    df = df[ok]
    paper_call = cond.call == "paper"   # the paper's BaseJudge call: no logprobs by design
    if not paper_call:
        no_tok = int(sum(t is None or len(t) == 0 for t in df["tokens"]))
        if no_tok:
            rec["fail"].append(f"{no_tok} answered rows without token windows")
        df = df[[t is not None and len(t) > 0 for t in df["tokens"]]]
    else:
        # rows captured before capture._as_written existed stored "0.4" for a written
        # 0.40: re-spell every answer from the raw JSON so decimals count the same way
        if "raw_answer" in df:
            df = df.assign(answer=[_as_written(float(v), raw) for v, raw in
                                   zip(df["score"], df["raw_answer"])])
        u = df[["input_tokens", "output_tokens", "reasoning_tokens"]].apply(
            pd.to_numeric, errors="coerce")
        rec["info"].append(f"tokens per call (median): input {u['input_tokens'].median():.0f}, "
                           f"output {u['output_tokens'].median():.0f}, reasoning "
                           f"{u['reasoning_tokens'].median():.0f} (mean "
                           f"{u['reasoning_tokens'].mean():.0f}, zero on "
                           f"{(u['reasoning_tokens'] == 0).mean():.1%} of calls)")
    df = add_readouts(df, cond_name)

    # luna returns fewer than the requested 20 alternatives (often 1-2), so this
    # is reported, not flagged
    win = np.array([len(t[0]["top"]) for t in df["tokens"] if len(t)])
    if len(win):
        rec["info"].append(f"top-window length at answer position 0: median "
                           f"{np.median(win):.0f}, min {win.min()}, max {win.max()}")

    versions = Counter(df["model_version"].fillna("?"))
    rec["model_versions"] = "; ".join(f"{v} x{c}" for v, c in versions.most_common())
    if len(versions) > 1:
        rec["warn"].append(f"{len(versions)} served model versions: {rec['model_versions']}")

    if cond.kind == "prob":
        bad = df["parsed"].isna()
        if bad.any():
            rec["fail"].append(f"{int(bad.sum())} answers with no number in [0, 1], e.g. "
                               f"{df.loc[bad, 'answer'].head(3).tolist()}")
        diff = (df["parsed"] - pd.to_numeric(df["score"], errors="coerce")).abs() > 1e-12
        if diff.any():
            rec["fail"].append(f"{int(diff.sum())} quick scores differ from the re-parsed answer")
        not_bare = ~df["answer"].map(is_bare_number)
        if not_bare.any():
            rec["warn"].append(f"{int(not_bare.sum())} answers are not a bare number, e.g. "
                               f"{df.loc[not_bare, 'answer'].head(3).tolist()}")
        dec = Counter(df["answer"].map(n_decimals))
        rec["info"].append("decimals written: " + ", ".join(
            f"{k}: {v / len(df):.1%}" for k, v in sorted(dec.items(), key=lambda kv: -kv[1])[:5]))
        want_dec = EXPECTED_DECIMALS.get(cond_name)
        if want_dec is not None and dec.get(want_dec, 0) / max(len(df), 1) < 0.95:
            rec["warn"].append(f"only {dec.get(want_dec, 0) / len(df):.1%} of answers have "
                               f"{want_dec} decimals")
        rec["info"].append(f"distinct written values: {df['parsed'].nunique()}"
                           + ("" if paper_call else
                              f"  distinct A2 values: {df['ev'].round(9).nunique()}"))
        m = df["mass"].dropna()
        if len(m) and not paper_call:
            rec["info"].append(f"A2 captured digit mass at 1st position: median {m.median():.3f}, "
                               f"5th pct {m.quantile(0.05):.3f}, < 0.5 on {(m < 0.5).mean():.1%}")
        if df["ev"].isna().any() and not paper_call:
            rec["warn"].append(f"{int(df['ev'].isna().sum())} rows without an A2 value")
    else:
        if df["lj"].isna().any():
            rec["fail"].append(f"{int(df['lj'].isna().sum())} rows without a LogJudge score")
        off = ~np.array([first_token_is_label(cond_name, t) for t in df["tokens"]])
        if off.any():
            ex = Counter(t[0]["token"] for t, o in zip(df["tokens"], off) if o).most_common(3)
            rec["warn"].append(f"first token not a label on {int(off.sum())} rows, e.g. {ex}")
        low = df["label_mass"] < MIN_ANSWER_MASS
        if low.any():
            rec["warn"].append(f"{int(low.sum())} rows with < 5% label mass (scored 1/2)")
        rec["info"].append(f"label mass at position 0: median {df['label_mass'].median():.4f}, "
                           f"min {df['label_mass'].min():.4f}")

    if reps > 1:
        col = "parsed" if cond.kind == "prob" else "lj"
        wide = df.pivot_table(index=["stem", "uid"], columns="rep", values=col)
        if wide.shape[1] > 1:
            same = (wide.nunique(axis=1) == 1).mean()
            rec["info"].append(f"identical {col} across all {wide.shape[1]} reps: {same:.1%} "
                               f"(T={'default' if cond.call == 'paper' else cond.temperature})")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="only this model's runs")
    ap.add_argument("--workspace", type=Path, default=WORKSPACE,
                    help="folder holding <model>/<condition>__<subset>.parquet")
    ap.add_argument("--out", type=Path, default=None, help="default: <workspace>/eval")
    ap.add_argument("-v", "--verbose", action="store_true", help="print INFO lines too")
    args = ap.parse_args()
    out = args.out or args.workspace / "eval"

    core = load_core()
    subsets = {}
    recs = []
    for _, model, cond, subset, reps in PLAN:
        if args.model and model != args.model:
            continue
        if subset not in subsets:   # the rows capture.py sent: subset first, then empty texts dropped
            rows = build_subset(core, subset)
            subsets[subset] = rows[rows["text"].astype(str).str.strip().str.len() > 0]
        recs.append(check_run(model, cond, subset, reps, subsets[subset], args.workspace))

    n_fail = 0
    for r in recs:
        verdict = "FAIL" if r["fail"] else ("WARN" if r["warn"] else "OK")
        n_fail += verdict == "FAIL"
        print(f"\n[{verdict:4}] {r['model']:11} {r['condition']:11} {r['subset']:18} "
              f"expected {r['expected']}, missing {r.get('missing', '-')}")
        for lvl in ("fail", "warn") + (("info",) if args.verbose else ()):
            for msg in r[lvl]:
                print(f"         {lvl.upper():4}  {msg}")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**{k: v for k, v in r.items() if k not in ("fail", "warn", "info")},
                   "verdict": "FAIL" if r["fail"] else ("WARN" if r["warn"] else "OK"),
                   "fail": " | ".join(r["fail"]), "warn": " | ".join(r["warn"]),
                   "info": " | ".join(r["info"])} for r in recs]).to_csv(out / "check.csv",
                                                                          index=False)
    print(f"\n{len(recs)} runs, {n_fail} FAIL -- wrote {out / 'check.csv'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
