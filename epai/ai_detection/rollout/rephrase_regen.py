r"""Backfill the rephrase rows missing from ``rephrase_v1_multi.parquet``.

``check_parquet_completeness.py`` reports these as

    id: N group(s) missing a generator their domain has — {'gpt56luna:0908R': N}
      id=essay:120  missing ['gpt56luna:0908R']
      ...

i.e. a human text whose ``gpt-5.6-luna`` rephrase never made it in — the
``generate`` call failed / was refused, or ``rephrase_build.step_append`` left
the rephrase out because its length ran away from the human partner. Dropping
the group would take a good human row with it; this regenerates just the missing
rephrase, through the exact prompt / model / cleaning that ``rephrase_build.py``
used, and appends it.

WHAT IT DOES
  1. scan the parquet for ``id`` groups that have the ``human`` row but not
     ``gpt56luna:0908R`` — the same rule ``check_parquet_completeness.py``
     applies (every group carries its domain's full generator set). ``--ids`` to
     name them instead.
  2. print them and ask for confirmation — then one ``gpt-5.6-luna`` call per
     missing row (BILLED), same system prompt + domain hint as
     ``cfg/ai_detection/rephrase/rephrase_v1.yaml``. Checkpointed to
     ``data/science/rephrase_v1_regen.jsonl`` (resumable; ``--retry-failed``).
  3. clean / QC / length-runaway gate exactly as ``rephrase_build.step_append``.
  4. append the new rows to the parquet (+ ``.bak``), suite columns Null.

A fresh rephrase that is itself a length-runaway is reported, not spliced.

Then fill the suite columns on the new (Null) rows and re-check:

    uv run python epai/ai_detection/rollout/rephrase_build.py --from score --gpu <ID> --yes
    uv run python epai/ai_detection/rollout/rephrase_build.py --from verify --yes

    uv run python epai/ai_detection/rollout/rephrase_regen.py --dry-run
    uv run python epai/ai_detection/rollout/rephrase_regen.py --yes
    uv run python epai/ai_detection/rollout/rephrase_regen.py --ids essay:120 wp:865
"""

from __future__ import annotations

import argparse
import difflib
import shutil
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from epai.ai_detection.rollout._common import (
    BASE_DIR, DIM, GREEN, RED, RESET, YELLOW, banner, rel,
)
from epai.ai_detection.rollout.rephrase_build import (
    COPY_FROM_HUMAN, LEN_RATIO_HI, LEN_RATIO_LO, MODEL, TARGET, clean_for,
    jaccard, load_prompts,
)
from epai.science_data.config import model_slug
from epai.science_data.generate.generate_ai import _key
from epai.science_data.llm import Shard, openai_text_call, run_resumable

# Its own shard, not rephrase_build's data/science/rephrase_v1_generate.jsonl —
# this is a targeted top-up, and keeping the checkpoints separate means a later
# full rephrase_build run is unaffected by what happened here.
SHARD = BASE_DIR / "data/science/rephrase_v1_regen.jsonl"


def missing_rephrases(t: pa.Table, ids: list[str] | None, label: str) -> list[str]:
    """``id`` groups that carry ``human`` but not ``label``.

    Mirrors ``check_parquet_completeness.py``'s pairing rule: per domain, every
    group should hold the full set of generators the domain has. Here a domain
    has ``{human, <label>}``; a group with only ``human`` is missing the
    rephrase. ``label`` is the only thing this script can produce — a gap of
    anything else is reported and skipped.
    """
    id_col = [str(x) for x in t.column("id").to_pylist()]
    dom_col = [str(x) for x in t.column("domain").to_pylist()]
    gen_col = [str(x) for x in t.column("generated_by").to_pylist()]

    dom_labels: dict[str, set[str]] = {}
    grp_labels: dict[str, set[str]] = {}
    grp_domain: dict[str, str] = {}
    for rid, dom, gen in zip(id_col, dom_col, gen_col):
        dom_labels.setdefault(dom, set()).add(gen)
        grp_labels.setdefault(rid, set()).add(gen)
        grp_domain[rid] = dom

    want = set(ids) if ids else None
    out: list[str] = []
    other_gaps: Counter = Counter()
    for rid, present in sorted(grp_labels.items()):
        if want is not None and rid not in want:
            continue
        gap = dom_labels[grp_domain[rid]] - present
        if not gap:
            continue
        other_gaps.update(gap - {label})
        if label in gap:
            if "human" not in present:
                print(f"  {YELLOW}skip{RESET} {rid} — no human row to rephrase")
                continue
            out.append(rid)

    if other_gaps:
        print(f"  {YELLOW}note{RESET} some groups also miss non-regenerable "
              f"label(s): {dict(other_gaps)} — not handled here")
    if want is not None:
        nothing = want - set(out)
        if nothing:
            print(f"  {YELLOW}note{RESET} already have {label} (nothing to do): "
                  f"{sorted(nothing)}")
    return out


def generate(rids: list[str], human_text: dict[str, str], domains: dict[str, str],
             system: str, hints: dict[str, str], label: str,
             workers: int, retry_failed: bool) -> None:
    """One gpt-5.6-luna call per rid — the same wiring as rephrase_build.step_generate."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(timeout=300)
    tasks = [
        (_key(rid, label),
         openai_text_call(client, MODEL, f"{system}\n\n{hints[domains[rid]]}",
                          human_text[rid], base_url=None))
        for rid in rids
    ]
    run_resumable(tasks, Shard(SHARD), max_workers=workers, desc="rephrase",
                  retry_failed=retry_failed)


def splice(t: pa.Table, rids: list[str], label: str, family: dict[str, str],
           human_text: dict[str, str], human_pos: dict[str, int],
           ok: dict[str, dict], src: Path, out_path: Path) -> int:
    """Append the fresh rephrase rows onto the parquet (QC + runaway gate as append)."""
    schema = t.schema
    text_raw, std, len_ratio, jac, seqm = {}, {}, {}, {}, {}
    runaway, ready = [], []
    for rid in rids:
        key = _key(rid, label)
        if key not in ok:
            print(f"  {YELLOW}no rephrase yet{RESET} {rid} — re-run with "
                  f"--retry-failed, or drop the group")
            continue
        text_raw[rid], std[rid] = clean_for(str(ok[key]["text"]), family[rid])
        h, r = human_text[rid], std[rid]
        hw, rw = len(h.split()), len(r.split())
        ratio = rw / hw if hw else 0.0
        len_ratio[rid] = ratio
        jac[rid] = jaccard(h, r)
        seqm[rid] = difflib.SequenceMatcher(None, h, r).ratio()
        if hw >= 20 and (ratio > LEN_RATIO_HI or ratio < LEN_RATIO_LO):
            runaway.append(rid)
        else:
            ready.append(rid)

    if runaway:
        print(f"{YELLOW}  {len(runaway)} fresh rephrase(s) still run away from "
              f"the human length (ratio outside [{LEN_RATIO_LO}, {LEN_RATIO_HI}]) "
              f"— left out:{RESET}")
        for rid in runaway:
            print(f"{DIM}    {rid}  ratio {len_ratio[rid]:.2f}{RESET}")
        print(f"{DIM}    re-run (the model may do better on a retry), or drop the "
              f"group — its human row too — from the corpus{RESET}")
    if not ready:
        print(f"{RED}nothing to splice.{RESET}")
        return 1

    pos = [human_pos[rid] for rid in ready]
    n = len(ready)

    # Column-for-column, matching rephrase_build.step_append.col_for: identity
    # columns set explicitly, human metadata carried over, QC columns computed,
    # every suite output appended Null for the score stage to fill.
    def col_for(f: pa.Field) -> pa.Array:
        c = f.name
        if c == "text":
            return pa.array([std[rid] for rid in ready], type=f.type)
        if c == "text_raw":
            return pa.array([text_raw[rid] for rid in ready], type=f.type)
        if c == "generated_by":
            return pa.array([label] * n, type=f.type)
        if c == "uid":
            return pa.array([f"{rid}:{label}" for rid in ready], type=f.type)
        if c == "id":
            return pa.array(ready, type=f.type)
        if c == "domain" or c == "source_family" or c in COPY_FROM_HUMAN:
            return t.column(c).take(pos).cast(f.type).combine_chunks()
        if c == "rephrase_len_ratio":
            return pa.array([len_ratio[rid] for rid in ready], type=f.type)
        if c == "rephrase_surface_jaccard":
            return pa.array([jac[rid] for rid in ready], type=f.type)
        if c == "rephrase_seqmatch_ratio":
            return pa.array([seqm[rid] for rid in ready], type=f.type)
        return pa.nulls(n, type=f.type)

    appended = pa.table([col_for(f) for f in schema], schema=schema)
    for rid in ready:
        print(f"  {rid:<34} -> {len(std[rid].split()):>5}w   "
              f"ratio {len_ratio[rid]:.2f}   jaccard {jac[rid]:.2f}")

    bak = src.with_suffix(src.suffix + ".bak")
    shutil.copy2(src, bak)
    print(f"{DIM}  backup → {rel(bak)}{RESET}")

    combined = pa.concat_tables([t, appended])
    order = pc.sort_indices(
        combined, sort_keys=[("id", "ascending"), ("generated_by", "ascending")]
    )
    pq.write_table(combined.take(order), out_path)
    print(f"\n{GREEN}appended {n} rephrase row(s){RESET} → {rel(out_path)} "
          f"({combined.num_rows} rows)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("parquet", type=Path, nargs="?", default=TARGET,
                    help=f"corpus to patch (default {rel(TARGET)})")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="write here instead of patching the input in place")
    ap.add_argument("--ids", nargs="+", default=None, metavar="ID",
                    help="rows to regenerate (default: auto-detect the missing ones)")
    ap.add_argument("--workers", type=int, default=8, metavar="N",
                    help="concurrent OpenAI calls (default 8)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-issue calls the regen shard recorded as failed")
    ap.add_argument("--splice-only", action="store_true",
                    help="skip generation, splice what the regen shard already has")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src: Path = args.parquet
    if not src.is_file():
        raise SystemExit(f"{RED}not a file: {rel(src)}{RESET}")
    out_path: Path = args.out or src

    system, hints, prompt_id = load_prompts()
    label = f"{model_slug(MODEL)}:{prompt_id}"

    banner("scan", rel(src))
    t = pq.read_table(src)
    rids = missing_rephrases(t, args.ids, label)
    if not rids:
        print(f"{GREEN}no groups missing {label}.{RESET}")
        return

    id_col = [str(x) for x in t.column("id").to_pylist()]
    gen_col = [str(x) for x in t.column("generated_by").to_pylist()]
    text_col = t.column("text").to_pylist()
    dom_col = [str(x) for x in t.column("domain").to_pylist()]
    fam_col = [str(x) for x in t.column("source_family").to_pylist()]
    human_pos = {id_col[i]: i for i in range(len(id_col)) if gen_col[i] == "human"}
    human_text = {rid: str(text_col[human_pos[rid]]) for rid in rids}
    domains = {rid: dom_col[human_pos[rid]] for rid in rids}
    family = {rid: fam_col[human_pos[rid]] for rid in rids}

    unknown = sorted({domains[rid] for rid in rids if domains[rid] not in hints})
    if unknown:
        raise SystemExit(f"{RED}no domain_hint in the prompt cfg for: {unknown}{RESET}")

    print(f"\n  {len(rids)} group(s) missing {label}:")
    for rid in rids:
        print(f"    {rid:<34} {domains[rid]:<16} human {len(human_text[rid].split()):>5}w")

    if args.dry_run:
        already = Shard(SHARD).ok_records() if SHARD.is_file() else {}
        todo = [rid for rid in rids if _key(rid, label) not in already]
        print(f"\n{DIM}dry-run — would issue {len(todo)} gpt-5.6-luna call(s) "
              f"({len(rids) - len(todo)} already in {rel(SHARD)}), then append "
              f"up to {len(rids)} row(s) → {rel(out_path)}{RESET}")
        return

    if not args.splice_only:
        print(f"\n{YELLOW}{len(rids)} gpt-5.6-luna call(s) will be made (billed).{RESET}")
        if not args.yes:
            if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                raise SystemExit("aborted.")
        banner("generate", rel(SHARD))
        generate(rids, human_text, domains, system, hints, label,
                 args.workers, args.retry_failed)

    ok = Shard(SHARD).ok_records()

    banner("splice", rel(out_path))
    code = splice(t, rids, label, family, human_text, human_pos, ok, src, out_path)
    if code:
        raise SystemExit(code)

    print(f"\n  next — fill the suite columns on the new rows, then re-check:")
    print(f"    uv run python epai/ai_detection/rollout/rephrase_build.py --from score --gpu <ID> --yes")
    print(f"    uv run python epai/ai_detection/rollout/rephrase_build.py --from verify --yes")


if __name__ == "__main__":
    main()
