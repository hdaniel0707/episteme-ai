r"""Compare two independent logprob captures of the SAME rows, to check
whether temperature=0 decoding is actually reproducible in practice.

Neither the OpenAI Responses API (gpt56luna) nor the GenAI4Science chat
endpoint (gemma4-31b) call is given a ``seed``, and both are shared, batched
production endpoints -- floating-point reduction order under batched GPU
inference isn't strictly deterministic even at temperature=0, so two captures
of the same rows can, in principle, disagree. This checks whether they do,
and whether it's large enough to matter.

Reads two sidecar directories written by ``capture_judge_logprobs.py`` (its
``--workspace`` flag) covering the same ``--only`` stem(s) and ``--limit``,
run close together in time, and diffs every
``llm_judge_logprobs_<model>_v3_{words,scale}`` column they share, row by row
(joined on ``uid``).

For each shared column this reports (over the rows both replicates actually
captured -- i.e. excluding rows outside the shared ``--limit`` selection,
which both correctly leave empty):
  * byte-identical windows -- same tokens, same logprobs, same order
  * same sampled (argmax) token, even if the rest of the top-k window differs
  * the resulting LogJudge score under the paper's own registered formula
    (``score_words`` = t0000, ``score_scale_contrast_full`` = t0001, both
    imported from ``score_logprob_windows.py`` rather than reimplemented)
  * same verdict -- same side of 0.5 (same human/machine call), regardless
    of how far apart the two scores are. This is usually the number that
    matters most in practice: a score wobble that never flips which side of
    0.5 a row lands on doesn't change a single classification, even if the
    raw |delta| looks large.
  A logit wobble that never changes which label wins, or moves the score by
  less than the shrinkage term already absorbs, is not a problem for the
  paper's results; one that flips the verdict, or moves several rows' scores
  by more than ~0.05, would be worth flagging.

Run (after two capture_judge_logprobs.py runs into separate --workspace dirs,
same --only stem(s) and --limit):

    .venv/bin/python epai/ai_detection/score/check_logprob_variance.py \
        --a data/parquet/tmp/variance_a_gpt56luna --b data/parquet/tmp/variance_b_gpt56luna
    .venv/bin/python epai/ai_detection/score/check_logprob_variance.py \
        --a data/parquet/tmp/variance_a_gemma4-31b --b data/parquet/tmp/variance_b_gemma4-31b

Read-only: writes nothing, makes no API calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from epai.ai_detection.score.score_logprob_windows import (
    score_scale_contrast_full,
    score_words,
)

KEY_COL = "uid"
IN_PREFIX = "llm_judge_logprobs_"
#: input suffix -> (mode label, the paper's registered scorer for that mode)
MODE_SCORER = {
    "_v3_words": ("words", score_words),
    "_v3_scale": ("scale", score_scale_contrast_full),
}


def _load(path: Path) -> tuple[list[str], dict[str, list]]:
    """``uid`` list + ``{column_name: window_list}`` for every
    ``llm_judge_logprobs_*`` column in this sidecar."""
    schema = pq.read_schema(path)
    cols = [c for c in schema.names if c.startswith(IN_PREFIX)]
    table = pq.read_table(path, columns=[KEY_COL, *cols])
    uids = table.column(KEY_COL).to_pylist()
    windows = {c: table.column(c).to_pylist() for c in cols}
    return uids, windows


def _window_key(window) -> tuple:
    """An order-sensitive fingerprint of a top-k window; ``()`` for an empty
    or missing window (a failed capture)."""
    if not window:
        return ()
    return tuple((w.get("token"), w.get("logprob")) for w in window)


def _verdict(score: float) -> str:
    """Which side of 0.5 a P(human) score falls on -- the actual detector
    call, regardless of margin. An exact 0.5 is its own bucket, so two 0.5s
    agree with each other but not with a 0.501 (a boundary case rare enough
    not to matter)."""
    return "human" if score > 0.5 else ("machine" if score < 0.5 else "tie")


def compare_column(name: str, uids_a: list[str], win_a: list,
                    uids_b: list[str], win_b: list) -> None:
    hit = next((suf for suf in MODE_SCORER if name.endswith(suf)), None)
    if hit is None:
        print(f"  {name}: unrecognised suffix -- skipped")
        return
    mode, scorer = MODE_SCORER[hit]

    a_by_uid = dict(zip(uids_a, win_a))
    b_by_uid = dict(zip(uids_b, win_b))
    shared = [u for u in a_by_uid if u in b_by_uid]
    print(f"  {name}  ({mode}, n={len(shared)} shared rows)")
    if not shared:
        print("    no shared uid between the two captures -- skipped")
        return

    n_identical = n_top_same = n_both_failed = n_one_failed = n_same_verdict = 0
    score_deltas: list[float] = []
    n_scored = 0
    for u in shared:
        wa, wb = a_by_uid[u], b_by_uid[u]
        ka, kb = _window_key(wa), _window_key(wb)
        if not ka and not kb:
            n_both_failed += 1
            continue
        if not ka or not kb:
            n_one_failed += 1
            continue
        n_identical += ka == kb
        n_top_same += wa[0].get("token") == wb[0].get("token")
        sa, sb = scorer(wa), scorer(wb)
        if sa is not None and sb is not None:
            n_scored += 1
            score_deltas.append(abs(sa - sb))
            n_same_verdict += _verdict(sa) == _verdict(sb)

    # `shared` is every uid present in both sidecars -- for a --limit run that
    # includes rows neither replicate ever attempted (both stay `[]`), which
    # is expected, not a failure. The real comparison population is the rows
    # both replicates actually captured something for.
    n_untouched = n_both_failed
    n_compared = len(shared) - n_untouched - n_one_failed
    if n_untouched:
        print(f"    not captured in either replicate: {n_untouched} "
              f"(outside --limit, or a genuine failure in both -- indistinguishable here)")
    if n_one_failed:
        print(f"    captured in only one replicate:   {n_one_failed} (a real capture gap)")
    if n_compared == 0:
        print("    no row was captured in both replicates -- nothing to compare")
        return
    print(f"    compared: {n_compared} rows")
    print(f"    byte-identical window:  {n_identical}/{n_compared}  ({n_identical / n_compared:.0%})")
    print(f"    same sampled token:     {n_top_same}/{n_compared}  ({n_top_same / n_compared:.0%})")
    if score_deltas:
        d = np.array(score_deltas)
        print(f"    same verdict (same side of 0.5, i.e. same human/machine "
              f"call regardless of margin): {n_same_verdict}/{n_scored} "
              f"({n_same_verdict / n_scored:.0%})")
        print(f"    |score_a - score_b|    n={len(d)}  mean={d.mean():.4f}  "
              f"max={d.max():.4f}  >0.01: {int((d > 0.01).sum())}  "
              f">0.05: {int((d > 0.05).sum())}")
    else:
        print("    no row had a usable score in both captures")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--a", type=Path, required=True, help="first capture workspace")
    ap.add_argument("--b", type=Path, required=True, help="second capture workspace")
    ap.add_argument("--stems", nargs="+", default=None,
                     help="restrict to these stems (default: every <stem>.parquet "
                          "present in both workspaces)")
    args = ap.parse_args()

    stems_a = {p.stem for p in args.a.glob("*.parquet")}
    stems_b = {p.stem for p in args.b.glob("*.parquet")}
    stems = sorted(args.stems) if args.stems else sorted(stems_a & stems_b)
    if not stems:
        raise SystemExit(f"no shared <stem>.parquet between {args.a} and {args.b}")

    for stem in stems:
        print(f"\n{'-' * 74}\n{stem}")
        uids_a, windows_a = _load(args.a / f"{stem}.parquet")
        uids_b, windows_b = _load(args.b / f"{stem}.parquet")
        shared_cols = sorted(set(windows_a) & set(windows_b))
        if not shared_cols:
            print("  no shared llm_judge_logprobs_* column -- skipped")
            continue
        for col in shared_cols:
            compare_column(col, uids_a, windows_a[col], uids_b, windows_b[col])


if __name__ == "__main__":
    main()
