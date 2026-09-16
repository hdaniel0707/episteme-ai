r"""Rephrase the gpt56luna REWRITE rows added by ``reph_add_rewrite.py`` -- the
fourth cell of MG-Reph's 2x2 (gpt-5.6-luna x {original, rephrased}).

Exactly the ``rephrase_build.py`` recipe -- same model, same clean / QC /
length-runaway gate -- pointed at the gpt56luna REWRITE text instead of the
human text, using the prompt in
``cfg/ai_detection/rephrase/rephrase_v1_rewrite.yaml`` (id ``0916R``, the same
system prompt and domain hints as ``rephrase_v1.yaml``'s ``0908R``, copied
verbatim). A new prompt id is required even though the wording is identical:
the label has to say what was rephrased -- ``gpt56luna:0908R`` already means
"rephrase of human"; reusing it here would silently merge two different
source populations under one class.

  generate  one gpt-5.6-luna call per REWRITE text (not human), domain-aware
            system prompt from ``rephrase_v1_rewrite.yaml``. Checkpointed to
            ``data/science/rephrase_v1_rewrite_generate.jsonl`` -- a re-run
            only re-issues failures.
  append    the rephrase rows into the parquet, text standardised the same
            way as ``rephrase_build.clean_for`` (by ``source_family``), with
            QC columns ``rephrase_len_ratio`` / ``rephrase_surface_jaccard`` /
            ``rephrase_seqmatch_ratio`` computed against the REWRITE text --
            its immediate source, not the human original. Domain /
            source_family carried from the rewrite row; paper metadata
            (title / doi / ...) carried from the human row, same as
            ``rephrase_build.py``. Rows whose length ran away from the
            rewrite are reported and left out.

Then fill embed / zero-shot / judge columns on the new (Null) rows:

    python epai/ai_detection/rollout/reph_fill_scores.py --gpu 0 --yes

    python epai/ai_detection/rollout/reph_rephrase_rewrite.py --dry-run
    python epai/ai_detection/rollout/reph_rephrase_rewrite.py
    python epai/ai_detection/rollout/reph_rephrase_rewrite.py --from append
    python epai/ai_detection/rollout/reph_rephrase_rewrite.py --only gb
"""

from __future__ import annotations

import argparse
import difflib
import shutil

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from epai.ai_detection.rollout._common import (
    BASE_DIR, DIM, GREEN, RED, RESET, YELLOW, banner, rel, run_steps,
)
from epai.ai_detection.rollout.reph_add_rewrite import FAMILIES
from epai.ai_detection.rollout.rephrase_build import (
    COPY_FROM_HUMAN, LEN_RATIO_HI, LEN_RATIO_LO, MODEL, clean_for, jaccard,
    load_prompts as human_reph_load_prompts,
)
from epai.science_data.config import model_slug
from epai.science_data.generate.generate_ai import _key
from epai.science_data.llm import Shard, openai_text_call, run_resumable

PROMPT_CFG = BASE_DIR / "cfg/ai_detection/rephrase/rephrase_v1_rewrite.yaml"
SHARD = BASE_DIR / "data/science/rephrase_v1_rewrite_generate.jsonl"

STEPS = ["generate", "append"]


def load_prompts() -> tuple[str, dict[str, str], str]:
    cfg = yaml.safe_load(PROMPT_CFG.read_text()) or {}
    return cfg["system_prompt"], cfg["domain_hints"], str(cfg["id"])


def _label() -> str:
    _, _, prompt_id = load_prompts()
    return f"{model_slug(MODEL)}:{prompt_id}"


def _existing_rewrite_label(gen_values: list[str]) -> str:
    """The gpt56luna rewrite label already merged in by ``reph_add_rewrite.py``.

    The target file carries up to three gpt56luna labels once this script has
    run once: the human-rephrase (``0908R``), the rewrite
    (``0701A``/``0811A``), and this script's own (``0916R``) -- so the rewrite
    label is whichever is neither of the other two.
    """
    _, _, human_reph_id = human_reph_load_prompts()
    human_reph_label = f"{model_slug(MODEL)}:{human_reph_id}"
    my_label = _label()
    labels = sorted(
        {g for g in gen_values if g.startswith("gpt56luna:")}
        - {human_reph_label, my_label}
    )
    if len(labels) != 1:
        raise SystemExit(
            f"expected exactly one gpt56luna rewrite label besides "
            f"{human_reph_label!r} and {my_label!r}, found {labels} -- "
            f"run reph_add_rewrite.py first?"
        )
    return labels[0]


def step_generate(fam: str, args: argparse.Namespace) -> int:
    target = FAMILIES[fam]["target"]
    if not target.is_file():
        print(f"{RED}not found: {rel(target)}{RESET}")
        return 1

    system, hints, _ = load_prompts()
    label = _label()
    t = pq.read_table(target, columns=["id", "domain", "generated_by", "text"])
    gen = t.column("generated_by").to_pylist()
    rw_label = _existing_rewrite_label(gen)

    mask = [g == rw_label for g in gen]
    ids = [str(x) for x, m in zip(t.column("id").to_pylist(), mask) if m]
    domains = [str(x) for x, m in zip(t.column("domain").to_pylist(), mask) if m]
    texts = [str(x) for x, m in zip(t.column("text").to_pylist(), mask) if m]

    unknown = sorted({d for d in domains if d not in hints})
    if unknown:
        print(f"{RED}no domain_hint in {rel(PROMPT_CFG)} for: {unknown}{RESET}")
        return 1

    if args.limit:
        ids, domains, texts = ids[:args.limit], domains[:args.limit], texts[:args.limit]
    print(f"  family {fam}   source {rw_label!r}   model {MODEL}   label {label}")
    print(f"  {len(ids)} rewrite text(s) to rephrase")

    if args.dry_run:
        print(f"{DIM}dry-run — would issue up to {len(ids)} call(s), "
              f"checkpoint -> {rel(SHARD)}{RESET}")
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
    run_resumable(
        tasks, Shard(SHARD), max_workers=args.workers,
        desc=f"rephrase-of-rewrite:{fam}", retry_failed=args.retry_failed,
    )
    return 0


def step_append(fam: str, args: argparse.Namespace) -> int:
    target = FAMILIES[fam]["target"]
    if not target.is_file():
        print(f"{RED}not found: {rel(target)}{RESET}")
        return 1

    label = _label()
    tgt = pq.read_table(target)
    schema = tgt.schema
    gen = tgt.column("generated_by").to_pylist()
    if label in gen:
        n = sum(v == label for v in gen)
        print(f"{RED}{rel(target)} already has {n} row(s) under {label!r}.{RESET}\n"
              f"  Drop them first if this is a re-append.")
        return 1

    rw_label = _existing_rewrite_label(gen)
    id_col = tgt.column("id").to_pylist()
    text_col = tgt.column("text").to_pylist()
    fam_col = tgt.column("source_family").to_pylist()

    rw_pos = {str(id_col[i]): i for i, g in enumerate(gen) if g == rw_label}
    human_pos = {str(id_col[i]): i for i, g in enumerate(gen) if g == "human"}
    rw_text = {pid: str(text_col[i]) for pid, i in rw_pos.items()}
    family = {pid: str(fam_col[i]) for pid, i in rw_pos.items()}

    ok = Shard(SHARD).ok_records()
    have = [pid for pid in rw_pos if _key(pid, label) in ok]
    missing = [pid for pid in rw_pos if pid not in have]
    if not have:
        print(f"{RED}no successful rephrases in {rel(SHARD)}.{RESET}")
        return 1
    if missing:
        print(f"{YELLOW}  {len(missing)} rewrite text(s) have no rephrase yet — "
              f"run generate (--retry-failed) to close the gap.{RESET}")

    text_raw, std = {}, {}
    len_ratio, jac, seqm, runaway = {}, {}, {}, []
    for pid in have:
        text_raw[pid], std[pid] = clean_for(str(ok[_key(pid, label)]["text"]),
                                            family[pid])
        source, r = rw_text[pid], std[pid]
        sw, rwn = len(source.split()), len(r.split())
        ratio = rwn / sw if sw else 0.0
        len_ratio[pid] = ratio
        jac[pid] = jaccard(source, r)
        seqm[pid] = difflib.SequenceMatcher(None, source, r).ratio()
        if sw >= 20 and (ratio > LEN_RATIO_HI or ratio < LEN_RATIO_LO):
            runaway.append(pid)

    if runaway:
        print(f"{YELLOW}  {len(runaway)} rephrase(s) ran away from the rewrite "
              f"length (ratio outside [{LEN_RATIO_LO}, {LEN_RATIO_HI}]) — left "
              f"out:{RESET}")
        for pid in runaway[:5]:
            print(f"{DIM}    {pid}  ratio {len_ratio[pid]:.2f}{RESET}")
        print(f"{DIM}    to regenerate: delete their keys from {rel(SHARD)} and "
              f"re-run generate{RESET}")
    keep = [pid for pid in have if pid not in set(runaway)]
    pos = [rw_pos[pid] for pid in keep]        # domain / source_family from the rewrite row
    hpos = [human_pos[pid] for pid in keep]    # paper metadata from the human row
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
        if c == "domain" or c == "source_family":
            return tgt.column(c).take(pos).cast(f.type).combine_chunks()
        if c in COPY_FROM_HUMAN:
            return tgt.column(c).take(hpos).cast(f.type).combine_chunks()
        if c == "rephrase_len_ratio":
            return pa.array([len_ratio[pid] for pid in keep], type=f.type)
        if c == "rephrase_surface_jaccard":
            return pa.array([jac[pid] for pid in keep], type=f.type)
        if c == "rephrase_seqmatch_ratio":
            return pa.array([seqm[pid] for pid in keep], type=f.type)
        return pa.nulls(n, type=f.type)

    appended = pa.table([col_for(f) for f in schema], schema=schema)

    if args.dry_run:
        print(f"{DIM}dry-run — would append {n} row(s) -> {rel(target)} "
              f"({tgt.num_rows + n} rows){RESET}")
        return 0

    bak = target.with_suffix(target.suffix + ".bak")
    shutil.copy2(target, bak)
    print(f"{DIM}  backup -> {rel(bak)}{RESET}")

    combined = pa.concat_tables([tgt, appended])
    order = pc.sort_indices(
        combined, sort_keys=[("id", "ascending"), ("generated_by", "ascending")]
    )
    pq.write_table(combined.take(order), target)
    print(f"{GREEN}appended {n} rephrase-of-rewrite row(s){RESET} -> {rel(target)} "
          f"({combined.num_rows} rows)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="+", choices=sorted(FAMILIES), default=None,
                    help="restrict to one or both families (default: both)")
    ap.add_argument("--from", dest="start_from", choices=STEPS, metavar="STEP",
                    help=f"start here instead of at the start: {', '.join(STEPS)}")
    ap.add_argument("--workers", type=int, default=8, metavar="N",
                    help="concurrent OpenAI calls in the generate step (default 8)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="generate: re-issue calls previously recorded as failed")
    ap.add_argument("--limit", type=int, default=0, metavar="N",
                    help="generate: cap the number of texts (a smoke test)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fams = args.only or sorted(FAMILIES)
    steps = STEPS[STEPS.index(args.start_from):] if args.start_from else list(STEPS)

    for fam in fams:
        banner(f"rephrase-of-rewrite: {fam}")
        runners = {
            "generate": (lambda fam=fam: step_generate(fam, args)),
            "append": (lambda fam=fam: step_append(fam, args)),
        }
        run_steps(steps, runners,
                 "epai/ai_detection/rollout/reph_rephrase_rewrite.py")


if __name__ == "__main__":
    main()
