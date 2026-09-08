r"""The rephrase arm — a content-preserving point on the LM-involvement spectrum.

Every machine row in ``science_v3f_multi_0811A`` and ``ghostbuster_v3f_multi`` is
full-text generation under the Ghostbuster protocol: the generator never sees the
human text. This builds the opposite — one moderate gpt-5.6-luna rephrase of each
human text, meaning preserved, wording changed — as a single self-contained
corpus, brought to the detector suite's column schema with the ``gpt56luna``
judge only (see ``cfg/ai_detection/rephrase/rephrase_v1.yaml``).

  extract   the ~5,992 ``human`` rows from both v3f corpora into
            data/parquet/rephrase_v1_multi.parquet — copied whole, so every
            suite score/embedding/judge they already carry comes with them (no
            separate carry step). Original ``uid`` preserved so a later concat
            with the v3f corpora dedups on it; a ``source_family`` column added.
  generate  one rephrase per human text via gpt-5.6-luna, domain-aware system
            prompt from rephrase_v1.yaml. Checkpointed to
            data/science/rephrase_v1_generate.jsonl — a re-run only re-issues
            failures.
  append    the rephrase rows into the parquet (text standardised the same way
            as its source family, suite columns Null), with the QC columns
            ``rephrase_len_ratio`` / ``rephrase_surface_jaccard`` /
            ``rephrase_seqmatch_ratio``. Rows whose length ran away from the
            human partner are reported and left out.
  verify    epai/ai_detection/analyse/check_standardization.py — the standardise
            gate. The normal pipelines run standardize_text as one column-wide
            stage over both classes; here the human rows arrive already
            standardised (from the v3f pipelines) and the rephrase rows are
            standardised row-wise in ``append``, so this is what confirms BOTH
            classes are a fixpoint of the current ``standardize_text``, with no
            residue that lands on one class of a domain and not the other.
  score     epai/ai_detection/prepare/run_holdout_pipeline.py --from embed
            --judges gpt56luna — emb / Binoculars / Fast-DetectGPT / detector /
            gpt56luna judge + cue vector over the Null (rephrase) rows.

Then run epai/ai_detection/analyse/rephrase_qc.py for the calibration check.

    python epai/ai_detection/rollout/rephrase_build.py --gpu 0 --yes
    python epai/ai_detection/rollout/rephrase_build.py --from generate --yes
    python epai/ai_detection/rollout/rephrase_build.py --dry-run
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from epai.ai_detection.rollout._common import (
    BASE_DIR, DIM, GREEN, RED, RESET, YELLOW, holdout_score, rel, run, run_steps,
)
from epai.science_data.config import model_slug
from epai.science_data.generate.generate_ai import (
    _key, _LEADING_LABEL_RE, _MARKDOWN_RE, _THINK_RE, clean_generated,
)
from epai.science_data.llm import Shard, openai_text_call, run_resumable
from epai.utils.text_standardize import standardize_text

GB_SRC = BASE_DIR / "data/parquet/ghostbuster_v3f_multi.parquet"
SCI_SRC = BASE_DIR / "data/parquet/science_v3f_multi_0811A.parquet"
TARGET = BASE_DIR / "data/parquet/rephrase_v1_multi.parquet"
SHARD = BASE_DIR / "data/science/rephrase_v1_generate.jsonl"
PROMPT_CFG = BASE_DIR / "cfg/ai_detection/rephrase/rephrase_v1.yaml"
CHECK_STD_SCRIPT = "epai/ai_detection/analyse/check_standardization.py"

MODEL = "gpt-5.6-luna"

STEPS = ["extract", "generate", "append", "verify", "score"]

# Copied from the paired human row onto the rephrase row (whichever the schema
# has). id / uid / domain / generated_by / text / text_raw / source_family and
# the QC columns are set explicitly; everything else is a suite output and is
# appended Null. `prompt` is left Null — the rephrase instruction is the cfg
# file named by the 0908R label, not a per-row string.
COPY_FROM_HUMAN = [
    "doi", "title", "author_ids", "author_count", "year", "venue",
    "citation_count", "fwci", "language", "fields_of_study", "query",
    "also_matched_domains",
]
QC_COLS = ["rephrase_len_ratio", "rephrase_surface_jaccard", "rephrase_seqmatch_ratio"]

# Columns carried on purpose beyond check_standardization.py's standard schema —
# the science paper metadata (from science_v3f), plus this corpus's own additions
# — so the gate reports them as declared rather than unexpected.
STD_EXTRA_COLS = [
    "doi", "title", "year", "venue", "fwci", "prompt", "author_ids",
    "author_count", "citation_count", "language", "fields_of_study", "query",
    "also_matched_domains", "source_family", *QC_COLS,
]

# Length-runaway gate on the rephrase, against its human partner. Mirrors the
# spirit of gb_v3f_clean.runaway_hits but two-sided — a rephrase should track the
# original's length, so a big move in EITHER direction is a failure.
LEN_RATIO_HI = 1.5
LEN_RATIO_LO = 0.6


def load_prompts() -> tuple[str, dict[str, str], str]:
    cfg = yaml.safe_load(PROMPT_CFG.read_text()) or {}
    return cfg["system_prompt"], cfg["domain_hints"], str(cfg["id"])


def _tok(text: str) -> set[str]:
    return {w for w in text.lower().split() if w}


def jaccard(a: str, b: str) -> float:
    sa, sb = _tok(a), _tok(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def clean_for(raw: str, family: str) -> tuple[str, str]:
    """``(text_raw, text)`` the way ``family``'s corpus was built.

    science  : ``clean_generated`` (markdown / label / <think> strip, then
               whitespace flattened — abstracts are one paragraph), matching
               ``generate_ai`` exactly.
    ghostbuster : the same token cleanup, but paragraph structure kept — an
               essay / news / story rephrase must carry newlines like the human
               side does, so only ``standardize_text(whitespace="lines")``
               touches the layout.
    """
    if family == "science":
        cleaned = clean_generated(raw)
        return cleaned, standardize_text(cleaned, whitespace="flat")
    stripped = _LEADING_LABEL_RE.sub("", _MARKDOWN_RE.sub("", _THINK_RE.sub("", raw))).strip()
    return stripped, standardize_text(stripped, whitespace="lines")


# --------------------------------------------------------------------------- #


def step_extract(args: argparse.Namespace) -> int:
    for p in (GB_SRC, SCI_SRC):
        if not p.is_file():
            print(f"{RED}not found: {rel(p)}{RESET}")
            return 1

    frames = []
    for src, family in ((GB_SRC, "ghostbuster"), (SCI_SRC, "science")):
        t = pq.read_table(src)
        t = t.filter(pc.equal(t.column("generated_by"), "human"))
        n = t.num_rows
        t = t.append_column("source_family", pa.array([family] * n, pa.string()))
        for c in QC_COLS:
            t = t.append_column(c, pa.nulls(n, pa.float64()))
        print(f"  {rel(src):55s} {n:5d} human row(s)")
        frames.append(t)

    unified = pa.unify_schemas([f.schema for f in frames])

    def conform(t: pa.Table) -> pa.Table:
        cols = [
            t.column(f.name).cast(f.type) if f.name in t.column_names
            else pa.nulls(t.num_rows, f.type)
            for f in unified
        ]
        return pa.table(cols, schema=unified)

    combined = pa.concat_tables([conform(f) for f in frames])
    combined = combined.take(pc.sort_indices(combined, sort_keys=[("id", "ascending")]))

    if args.dry_run:
        print(f"\n{DIM}dry-run — would write {combined.num_rows} human row(s) "
              f"→ {rel(TARGET)}{RESET}")
        return 0

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, TARGET)
    print(f"\n{GREEN}wrote {combined.num_rows} human row(s){RESET} → {rel(TARGET)}")
    return 0


def step_generate(args: argparse.Namespace) -> int:
    if not TARGET.is_file():
        print(f"{YELLOW}{rel(TARGET)} not found — run the extract step first.{RESET}")
        return 1

    system, hints, prompt_id = load_prompts()
    t = pq.read_table(TARGET, columns=["id", "domain", "generated_by", "text"])
    t = t.filter(pc.equal(t.column("generated_by"), "human"))
    ids = [str(x) for x in t.column("id").to_pylist()]
    domains = [str(x) for x in t.column("domain").to_pylist()]
    texts = [str(x) for x in t.column("text").to_pylist()]

    unknown = sorted({d for d in domains if d not in hints})
    if unknown:
        print(f"{RED}no domain_hint in {rel(PROMPT_CFG)} for: {unknown}{RESET}")
        return 1

    label = f"{model_slug(MODEL)}:{prompt_id}"
    if args.limit:
        ids, domains, texts = ids[:args.limit], domains[:args.limit], texts[:args.limit]
    print(f"  model  {MODEL}   label  {label}")
    print(f"  {len(ids)} human text(s) to rephrase")

    if args.dry_run:
        print(f"\n{DIM}dry-run — would issue up to {len(ids)} call(s), "
              f"checkpoint → {rel(SHARD)}{RESET}")
        return 0

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(timeout=300)

    tasks = [
        (_key(pid, label),
         openai_text_call(client, MODEL, f"{system}\n\n{hints[dom]}", txt,
                          base_url=None))
        for pid, dom, txt in zip(ids, domains, texts)
    ]
    # Store the raw model output; cleaning depends on the row's source family
    # and is done at append time (default payload_of is {"text": text}).
    run_resumable(
        tasks, Shard(SHARD), max_workers=args.workers, desc="rephrase",
        retry_failed=args.retry_failed,
    )
    return 0


def step_append(args: argparse.Namespace) -> int:
    if not TARGET.is_file():
        print(f"{YELLOW}{rel(TARGET)} not found — run extract + generate first.{RESET}")
        return 1
    _, _, prompt_id = load_prompts()
    label = f"{model_slug(MODEL)}:{prompt_id}"

    tgt = pq.read_table(TARGET)
    schema = tgt.schema
    gb = tgt.column("generated_by").to_pylist()
    if label in gb:
        n = sum(v == label for v in gb)
        print(f"{RED}{rel(TARGET)} already has {n} row(s) under {label!r}.{RESET}\n"
              f"  Drop them first if this is a re-append.")
        return 1

    id_col = tgt.column("id").to_pylist()
    text_col = tgt.column("text").to_pylist()
    fam_col = tgt.column("source_family").to_pylist()
    human_pos = {str(id_col[i]): i for i, g in enumerate(gb) if g == "human"}
    human_text = {pid: str(text_col[i]) for pid, i in human_pos.items()}
    family = {pid: str(fam_col[i]) for pid, i in human_pos.items()}

    ok = Shard(SHARD).ok_records()
    have = [pid for pid in human_pos if _key(pid, label) in ok]
    missing = [pid for pid in human_pos if pid not in have]
    if not have:
        print(f"{RED}no successful rephrases in {rel(SHARD)}.{RESET}")
        return 1
    if missing:
        print(f"{YELLOW}  {len(missing)} human text(s) have no rephrase yet — "
              f"run generate (--retry-failed) to close the gap.{RESET}")

    text_raw, std = {}, {}
    len_ratio, jac, seqm, runaway = {}, {}, {}, []
    for pid in have:
        text_raw[pid], std[pid] = clean_for(str(ok[_key(pid, label)]["text"]),
                                            family[pid])
        h, r = human_text[pid], std[pid]
        hw, rw = len(h.split()), len(r.split())
        ratio = rw / hw if hw else 0.0
        len_ratio[pid] = ratio
        jac[pid] = jaccard(h, r)
        seqm[pid] = difflib.SequenceMatcher(None, h, r).ratio()
        if hw >= 20 and (ratio > LEN_RATIO_HI or ratio < LEN_RATIO_LO):
            runaway.append(pid)

    if runaway:
        print(f"{YELLOW}  {len(runaway)} rephrase(s) ran away from the human "
              f"length (ratio outside [{LEN_RATIO_LO}, {LEN_RATIO_HI}]) — left "
              f"out:{RESET}")
        for pid in runaway[:5]:
            print(f"{DIM}    {pid}  ratio {len_ratio[pid]:.2f}{RESET}")
        print(f"{DIM}    to regenerate: delete their keys from {rel(SHARD)} and "
              f"re-run generate{RESET}")
    keep = [pid for pid in have if pid not in set(runaway)]
    pos = [human_pos[pid] for pid in keep]
    n = len(keep)

    def col_for(f: pa.Field) -> pa.Array:
        c = f.name
        if c == "text":
            return pa.array([std[pid] for pid in keep], type=f.type)
        if c == "text_raw":
            return pa.array([text_raw[pid] for pid in keep], type=f.type)
        if c == "generated_by":
            return pa.array([label] * n, type=f.type)
        if c == "uid":
            return pa.array([f"{pid}:{label}" for pid in keep], type=f.type)
        if c == "id":
            return pa.array(keep, type=f.type)
        if c == "domain" or c == "source_family" or c in COPY_FROM_HUMAN:
            return tgt.column(c).take(pos).cast(f.type).combine_chunks()
        if c == "rephrase_len_ratio":
            return pa.array([len_ratio[pid] for pid in keep], type=f.type)
        if c == "rephrase_surface_jaccard":
            return pa.array([jac[pid] for pid in keep], type=f.type)
        if c == "rephrase_seqmatch_ratio":
            return pa.array([seqm[pid] for pid in keep], type=f.type)
        return pa.nulls(n, type=f.type)

    appended = pa.table([col_for(f) for f in schema], schema=schema)

    if args.dry_run:
        print(f"\n{DIM}dry-run — would append {n} row(s) → {rel(TARGET)} "
              f"({tgt.num_rows + n} rows){RESET}")
        return 0

    bak = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, bak)
    print(f"{DIM}  backup → {rel(bak)}{RESET}")

    combined = pa.concat_tables([tgt, appended])
    order = pc.sort_indices(
        combined, sort_keys=[("id", "ascending"), ("generated_by", "ascending")]
    )
    pq.write_table(combined.take(order), TARGET)
    print(f"\n{GREEN}appended {n} rephrase row(s){RESET} → {rel(TARGET)} "
          f"({combined.num_rows} rows)")
    return 0


def step_verify(args: argparse.Namespace) -> int:
    if not args.dry_run and not TARGET.is_file():
        print(f"{RED}not found: {rel(TARGET)}{RESET}")
        return 1
    # Pin 'lines': the combined file has real newlines (ghostbuster rows), and a
    # 'flat' science row is trivially 'lines'-conformant too — so 'lines' is the
    # one convention the whole file must satisfy. --fail-on error trips on a
    # non-fixpoint text or residue that is asymmetric across the classes.
    cmd = [sys.executable, BASE_DIR / CHECK_STD_SCRIPT, TARGET,
           "--whitespace", "lines", "--extra-cols", *STD_EXTRA_COLS,
           "--fail-on", "error"]
    return run(cmd, args.dry_run)


def step_score(args: argparse.Namespace) -> int:
    return holdout_score(TARGET, ["gpt56luna"], gpu=args.gpu,
                         cpu_threads=args.cpu_threads, per_band=args.per_band,
                         yes=args.yes, dry_run=args.dry_run)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--from", dest="start_from", choices=STEPS, metavar="STEP",
                    help=f"start here instead of at the start: {', '.join(STEPS)}")
    ap.add_argument("--workers", type=int, default=8, metavar="N",
                    help="concurrent OpenAI calls in the generate step (default 8)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="generate: re-issue calls previously recorded as failed")
    ap.add_argument("--limit", type=int, default=0, metavar="N",
                    help="generate: cap the number of texts (a smoke test)")
    ap.add_argument("--gpu", default=None, metavar="ID")
    ap.add_argument("--cpu-threads", type=int, default=None, metavar="N")
    ap.add_argument("--per-band", type=int, default=20000,
                    help="cue-vector coverage cap; must exceed the row count")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    steps = STEPS[STEPS.index(args.start_from):] if args.start_from else list(STEPS)
    print(f"{DIM}  corpus   {rel(TARGET)}")
    print(f"  steps    {' → '.join(steps)}{RESET}")

    runners = {
        "extract": lambda: step_extract(args),
        "generate": lambda: step_generate(args),
        "append": lambda: step_append(args),
        "verify": lambda: step_verify(args),
        "score": lambda: step_score(args),
    }
    run_steps(steps, runners, "epai/ai_detection/rollout/rephrase_build.py")


if __name__ == "__main__":
    main()
