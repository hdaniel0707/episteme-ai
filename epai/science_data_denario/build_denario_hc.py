r"""Stage 2 — one human abstract per Denario paper; assemble the holdout parquet.

Takes the candidate pool from fetch_human_candidates.py and, per Denario domain,
solves an optimal 1:1 assignment (``scipy.optimize.linear_sum_assignment``)
between the Denario papers and the candidates, minimising

    w_length * |len(denario) - len(human)| / len_scale  +  w_topic * (1 - cos)

where ``cos`` is SPECTER2 cosine similarity of the two abstracts. The result is
80 Denario (machine) rows and 80 matched human rows — balanced, like every
training corpus, so ``holdout_core`` gives back a real ROC / AUROC / FPR instead
of a detection-rate-only table.

The human ``text`` is the ``clean_abstract`` output already stored by stage 1 —
the same thing the science pipeline feeds to ``standardize_text`` — NOT
re-extracted from a PDF. The remaining machine-vs-human asymmetry (Denario text
came out of a PDF) is what ``ocr_clean.py`` removes next.

OUTPUT  data/parquet/denario_hc_raw.parquet (or wherever ``-o`` points)
    uid, id, domain, text, generated_by, title, year, oa_id

    machine rows  generated_by carried through as-is from ``--denario``'s own
                  column (so a multi-generator source like
                  ai_scientist_detect.parquet keeps each row's real generator
                  -- 'sakana' / 'sakanav2' / 'cycleresearcher' / 'zochi' --
                  rather than being collapsed to one label); id/uid/text from
                  the source (``text`` = ``text_raw`` if present, else ``text``)
    human rows    generated_by = 'human',   id = '<--id-prefix>:<domain>:<oa_id>'

``oa_id`` is kept on every row (empty for machine) so the 80 human OpenAlex work
ids can be excluded from a later training fetch — science_v5 / Track 2 adds
Astronomy and Neuroscience, which overlap this holdout.

    results/denario_hc/match_report.csv   per-pair length delta and cosine

Run:
    python -m epai.science_data_denario.build_denario_hc --gpu 0
    python -m epai.science_data_denario.build_denario_hc --w-topic 2.0 --allow-shortfall
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from epai.science_data.rules import _STRUCTURED_RE, strip_structured_labels
from epai.science_data_denario.exclude import Exclusion, load_exclusion
from epai.utils.embedding_utils import add_embedding
from epai.utils.parquet_utils import load_parquet, save_parquet

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DENARIO = BASE_DIR / "data" / "parquet" / "denario_detect.parquet"
DEFAULT_CANDIDATES = BASE_DIR / "data" / "science_denario" / "human_candidates.parquet"
DEFAULT_OUT = BASE_DIR / "data" / "parquet" / "denario_hc_raw.parquet"
REPORT_DIR = BASE_DIR / "results" / "denario_hc"

OUT_COLUMNS = ["uid", "id", "domain", "text", "generated_by", "title", "year", "oa_id"]


def embed_specter2(frames: list[pd.DataFrame], emb_type: str, batch_size: int) -> list[np.ndarray]:
    """One SPECTER2 pass over the concatenation of ``frames`` (each needs
    ``title`` and ``abstract``); returns L2-normalised matrices, split back."""
    sizes = [len(f) for f in frames]
    joined = pd.concat(
        [f[["title", "abstract"]].reset_index(drop=True) for f in frames],
        ignore_index=True,
    )
    joined = add_embedding(joined, emb_type, batch_size=batch_size)
    mat = np.vstack(joined[f"emb_{emb_type}"].to_numpy()).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    out, start = [], 0
    for n in sizes:
        out.append(mat[start:start + n])
        start += n
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--denario", type=Path, default=DEFAULT_DENARIO,
                    help="the machine-side detect-schema parquet (uid/id/domain/"
                         "text[/text_raw]/generated_by/title/year) -- despite the "
                         "flag name, any corpus in this schema works, not just "
                         "Denario's own.")
    ap.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--id-prefix", default="denhc",
                    help="prefix for the synthesized human row id, "
                         "'<prefix>:<domain>:<oa_id>' (default 'denhc' -- change "
                         "this for a different --denario source so its ids don't "
                         "collide with the Denario human-control run)")
    ap.add_argument("--report-dir", type=Path, default=REPORT_DIR,
                    help="where match_report.csv is written (default: "
                         f"{REPORT_DIR.relative_to(BASE_DIR)})")
    ap.add_argument("--emb-type", default="specter2",
                    help="topic embedding for the match (default specter2)")
    ap.add_argument("--w-length", type=float, default=1.0)
    ap.add_argument("--w-topic", type=float, default=1.0)
    ap.add_argument("--len-scale", type=float, default=40.0,
                    help="word-count difference that costs as much as a topic "
                         "cosine of 1.0 (default 40)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--gpu", default=None, help="sets CUDA_VISIBLE_DEVICES")
    ap.add_argument("--allow-shortfall", action="store_true",
                    help="match as many as possible when a domain's pool is "
                         "smaller than its Denario count, instead of aborting")
    ap.add_argument("--exclude-parquet", type=Path, action="append", default=None,
                    help="training corpus the matched humans must NOT appear in "
                         "(repeatable; default: science_v3)")
    ap.add_argument("--no-exclude-check", action="store_true")
    ap.add_argument("--strip-structured-labels", action="store_true",
                    help="remove 'Background:'/'Methods:' section labels from the "
                         "text of BOTH classes (rules.strip_structured_labels — "
                         "colon-required, keeps the prose). The Denario class has "
                         "no labels, so this only touches the human rows; use it "
                         "when the run below warns about them.")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    excl = Exclusion() if args.no_exclude_check else load_exclusion(args.exclude_parquet)

    machine_full = load_parquet(args.denario)
    # `text_raw` (pre-standardization) if the source already has it -- Denario's
    # does, by the time this stage runs, because run_holdout_pipeline.py's
    # standardize_text already ran on it for the main scoring pipeline (see the
    # module docstring). A source that hasn't been standardized yet (e.g. a
    # freshly-fetched corpus with no separate raw/standardized split) has no
    # such column, and its `text` IS already the raw text -- use that instead.
    text_col = "text_raw" if "text_raw" in machine_full.columns else "text"
    machine = machine_full[
        ["id", "uid", "domain", "title", "year", "generated_by", text_col]
    ].reset_index(drop=True)
    machine = machine.rename(columns={text_col: "text"})
    machine["abstract"] = machine["text"]
    machine["n_words"] = machine["text"].str.split().str.len()
    machine["_pos"] = np.arange(len(machine))

    cand = load_parquet(args.candidates).reset_index(drop=True)

    # Drop any candidate that is in a training corpus. fetch already does this,
    # so this is the second gate — it catches a pool built with a stale or
    # different --exclude-parquet, and it is cheap.
    doi_col = cand["doi"] if "doi" in cand.columns else pd.Series([""] * len(cand))
    hits = [
        excl.hit(o, d, t) is not None
        for o, d, t in zip(cand["oa_id"], doi_col, cand["title"])
    ]
    if any(hits):
        print(f"{YELLOW}  {sum(hits)} candidate(s) dropped here for training "
              f"overlap (fetch should have caught these){RESET}")
        cand = cand[[not h for h in hits]].reset_index(drop=True)
    cand["_pos"] = np.arange(len(cand))

    print(f"{len(machine)} Denario papers, {len(cand)} candidates")
    Em, Ec = embed_specter2([machine, cand], args.emb_type, args.batch_size)

    rows: list[dict] = []
    report: list[dict] = []
    for slug, grp in machine.groupby("domain", sort=False):
        pool = cand[cand["domain"] == slug]
        if len(pool) < len(grp):
            msg = f"{slug}: pool {len(pool)} < {len(grp)} Denario papers"
            if not args.allow_shortfall:
                raise SystemExit(
                    f"{RED}{msg}. Broaden that domain's queries or raise "
                    f"fetch --pool-factor, or pass --allow-shortfall.{RESET}"
                )
            print(f"{YELLOW}  {msg} — matching {len(pool)} of {len(grp)}{RESET}")

        m_pos = grp["_pos"].to_numpy()
        c_pos = pool["_pos"].to_numpy()
        len_cost = np.abs(
            grp["n_words"].to_numpy()[:, None] - pool["n_words"].to_numpy()[None, :]
        ) / args.len_scale
        topic_cost = 1.0 - Em[m_pos] @ Ec[c_pos].T
        cost = args.w_length * len_cost + args.w_topic * topic_cost

        r_idx, c_idx = linear_sum_assignment(cost)
        for ri, ci in zip(r_idx, c_idx):
            m = grp.iloc[ri]
            h = pool.iloc[ci]
            oa = str(h["oa_id"])
            hid = f"{args.id_prefix}:{slug}:{oa}"
            year = int(h["year"]) if pd.notna(h["year"]) else pd.NA
            rows.append({
                "uid": f"{hid}:human", "id": hid, "domain": slug,
                "text": h["abstract"], "generated_by": "human",
                "title": h["title"], "year": year, "oa_id": oa,
            })
            rows.append({
                "uid": m["uid"], "id": m["id"], "domain": slug,
                "text": m["text"], "generated_by": m["generated_by"],
                "title": m["title"], "year": m["year"], "oa_id": "",
            })
            report.append({
                "machine_id": m["id"], "oa_id": oa, "domain": slug,
                "len_machine": int(m["n_words"]), "len_human": int(h["n_words"]),
                "len_delta": int(abs(m["n_words"] - h["n_words"])),
                "cosine": float(Em[m["_pos"]] @ Ec[h["_pos"]]),
            })

    out = pd.DataFrame(rows, columns=OUT_COLUMNS)
    out["year"] = out["year"].astype("Int64")

    if args.strip_structured_labels:
        before = out["text"].apply(lambda t: bool(_STRUCTURED_RE.search(str(t)))).sum()
        out["text"] = out["text"].apply(lambda t: strip_structured_labels(str(t)))
        print(f"stripped structured-abstract labels from {before} row(s)")

    # Final gate: not one human row may be in a training corpus.
    leaked = [
        (r["id"], r["title"]) for r in rows
        if r["generated_by"] == "human"
        and excl.hit(r["oa_id"], None, r["title"]) is not None
    ]
    if leaked:
        raise SystemExit(
            f"{RED}{len(leaked)} matched human(s) are in a training corpus — "
            f"NOT writing. Re-run fetch_human_candidates.py with the right "
            f"--exclude-parquet.\n  " + "\n  ".join(f"{i}  {t[:70]}" for i, t in leaked)
            + f"{RESET}"
        )

    save_parquet(out, args.out)

    rep = pd.DataFrame(report)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    rep.to_csv(args.report_dir / "match_report.csv", index=False)

    n_h = int((out["generated_by"] == "human").sum())
    n_m = len(out) - n_h
    print(f"\n{GREEN}{len(out)} rows ({n_m} machine, {n_h} human) → {args.out}{RESET}")
    print(f"  machine generators: {out.loc[out['generated_by'] != 'human', 'generated_by'].value_counts().to_dict()}")
    print(f"  |len delta|  median {rep['len_delta'].median():.0f}  "
          f"p90 {rep['len_delta'].quantile(0.9):.0f}  max {rep['len_delta'].max():.0f}")
    print(f"  topic cosine median {rep['cosine'].median():.3f}  "
          f"min {rep['cosine'].min():.3f}")
    print(f"  match report → {args.report_dir / 'match_report.csv'}")

    labelled = out.loc[
        (out["generated_by"] == "human") & out["text"].apply(
            lambda t: bool(_STRUCTURED_RE.search(str(t)))
        )
    ]
    if len(labelled):
        print(f"\n{YELLOW}{len(labelled)} human row(s) contain structured-abstract "
              f"labels ('Background:' ...). The Denario class never has them — "
              f"re-run fetch_human_candidates.py with --strip-structured-labels "
              f"and rebuild.{RESET}")

    print(f"\n{YELLOW}Before training on any future corpus that could overlap "
          f"this run's domain(s), exclude these OpenAlex ids:{RESET}")
    print(f"  {args.out} → column 'oa_id' (non-empty rows)")


if __name__ == "__main__":
    main()
