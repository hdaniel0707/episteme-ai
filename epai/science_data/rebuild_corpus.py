#!/usr/bin/env python3
r"""Rebuild a science multi-generator corpus with its bad human rows removed.

``check_parquet_completeness.py -v`` on ``science_v3_multi_0811A`` finds two
kinds of bad ``human`` row:

  - **duplicate abstracts** — 10 pairs whose abstract is byte-identical to
    another row's. Eight are the same paper deposited twice in OpenAlex under
    two work ids; two are different papers that ended up with the same abstract
    text. Either way the pair leaks across a train/test or LODO split, and every
    ``emb_*`` warning in that report is the same 10 pairs echoed — the embedders
    are deterministic.
  - **not an abstract at all** — ~37 rows whose "abstract" is a scrape of the
    publisher's page (an EBSCO nav dump, a journal's article-nav strip), a
    book's table of contents, or raw LaTeX. ``rules.abstract_is_boilerplate``.

Both are on the HUMAN side only: a generated row is written from a title-only
prompt, so it is never a scraped page and never a second deposit of anything.
When a human paper is dropped its whole ``id`` goes — the human row **and** its
three generated rows are simply never built for the new corpus, because the
rebuild starts from the human set and works forward.

This rebuilds the corpus at its original size:

  1. **detect**  — list the non-abstract rows (filter drops these on its own),
     then group the human rows by folded abstract (``normalise_abstract``), show
     every collision and ask per group what to keep. Default: keep the highest
     ``citation_count`` deposit, drop the rest. Groups whose titles disagree are
     flagged — those want a look, not the default.
  2. **raw**     — write ``data/science/<new-id>_raw.parquet`` = the original raw
     fetch minus the dropped duplicate papers, and ``cfg/science_data/<new-id>.yaml``.
     No new OpenAlex call: the fetch overfetched 1.2x, so the filter stage
     backfills the freed slots from the **same Aug-12 fetch** — same ranking,
     no drift. Every backfilled paper goes through the same ``filter.py`` gate,
     so it cannot be a dropped id, an id / title / abstract duplicate of a kept
     row, or itself a non-abstract — the defect cannot recur.
  3. **filter → prompts → generate** — the science_data stages under the new id.
     ``filter`` re-tests every row with the current rules, so the non-abstract
     and near-duplicate rows are dropped here and the trim backfills past them.
     ``generate`` reuses the old generate checkpoint verbatim, so only the
     handful of backfilled papers' AI rows are actually generated.
  4. **promote → standardize → carry** — copy to ``data/parquet/``, fold the
     text the same way, then ``carry_suite_columns.py`` copies every already
     computed embedding / score / judge column across by ``uid`` for every
     paper that stayed.
  5. **score**   — ``run_holdout_pipeline.py --from embed`` fills the rows that
     are still Null — the backfilled papers only — across the whole detector
     suite (needs a GPU box + the Binoculars sub-venv; printed, not run, unless
     ``--gpu`` is given).

The result is a NEW dataset id — ``science_v3d`` — because a changed paper set
gets a new id (see ``cfg/science_data/science_v3.yaml``). The source corpus is
never touched.

    # interactive, stops before the GPU stage
    .venv/bin/python epai/science_data/rebuild_corpus.py

    # everything, on a GPU box (answer the ~10 duplicate-group prompts first)
    .venv/bin/python epai/science_data/rebuild_corpus.py --gpu 0

    # everything, no prompts: --keep fixes the one group whose default is wrong
    # (its abstract belongs to the other paper); every other group's default is
    # correct and the two EBSCO rows are dropped by the boilerplate filter anyway
    .venv/bin/python epai/science_data/rebuild_corpus.py --gpu 0 --yes \
        --keep W2141899406

    # resume on the GPU box after an interactive run stopped at the score step
    .venv/bin/python epai/science_data/rebuild_corpus.py --from score --gpu 0

    .venv/bin/python epai/science_data/rebuild_corpus.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from epai.science_data.config import BASE_DIR, SCIENCE_DIR, load_pipeline_config
from epai.science_data.filter.filter import apply_rules
from epai.science_data.rules import (
    abstract_is_boilerplate,
    normalise_abstract,
    normalise_title,
)

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

PARQUET_DIR = BASE_DIR / "data" / "parquet"
PY = sys.executable  # the interpreter running this script — reused for subprocesses

SOURCE_CFG = BASE_DIR / "cfg" / "science_data" / "science_v3.yaml"
SOURCE_DATASET_ID = "science_v3"
PROMPT_CONFIG = "0811A"
VARIANT = f"multi_{PROMPT_CONFIG}"

STAGES = ("detect", "raw", "filter", "prompts", "generate",
          "promote", "carry", "score")


# --------------------------------------------------------------------------
# stage 1 — detect duplicate human abstracts, decide what to drop
# --------------------------------------------------------------------------


def _paper_id(row_id: str) -> str:
    """`science:sociology:W123` -> `W123`."""
    return row_id.rsplit(":", 1)[-1]


def _same_work(a: str, b: str) -> bool:
    """True when the two titles look like one paper deposited twice: identical
    once folded, or one a prefix of the other (a subtitle was added). A pair
    that fails this shares an abstract without sharing a title — a scrape
    artefact or a metadata mix-up, not a duplicate upload — and wants a look."""
    fa, fb = normalise_title(a), normalise_title(b)
    if not fa or not fb:
        return False
    short, long = sorted((fa, fb), key=len)
    return long.startswith(short) or short in long


def find_duplicate_groups(in_parquet: Path) -> list[list[dict]]:
    cols = ["id", "domain", "generated_by", "text", "title",
            "citation_count", "year", "fwci"]
    tbl = pq.read_table(in_parquet, columns=cols)
    human = [r for r in tbl.to_pylist() if r["generated_by"] == "human"]

    groups: dict[str, list[dict]] = {}
    for r in human:
        groups.setdefault(normalise_abstract(r["text"]), []).append(r)

    out = []
    for members in groups.values():
        if len(members) > 1:
            members.sort(key=lambda r: (-(r["citation_count"] or 0),
                                        r["year"] or 0, r["id"]))
            out.append(members)
    out.sort(key=lambda g: g[0]["domain"])
    return out


def report_boilerplate(in_parquet: Path) -> None:
    """List the human rows whose abstract is a page scrape / contents page /
    reference list. Not a decision — filter.py drops these on its own now
    (rules.abstract_is_boilerplate); this is just so the removal is visible."""
    tbl = pq.read_table(in_parquet, columns=["id", "domain", "generated_by",
                                             "title", "text"])
    junk = [r for r in tbl.to_pylist()
            if r["generated_by"] == "human" and abstract_is_boilerplate(r["text"])]
    if not junk:
        return
    by_dom = Counter(r["domain"] for r in junk)
    print(f"\n{BOLD}{len(junk)} non-abstract human row(s){RESET} "
          f"({', '.join(f'{d}:{n}' for d, n in sorted(by_dom.items()))}) — "
          f"the filter stage drops these automatically:")
    for r in junk:
        print(f"  {_paper_id(r['id']):12} [{r['domain'][:4]}] "
              f"{DIM}{(r['title'] or '')[:60]}{RESET}")
        print(f"      {DIM}{' '.join((r['text'] or '').split())[:105]}…{RESET}")


def _print_group(n: int, total: int, group: list[dict]) -> bool:
    keeper = group[0]
    suspect = any(not _same_work(keeper["title"], m["title"])
                  for m in group[1:])
    flag = f"  {YELLOW}titles disagree — check before accepting{RESET}" if suspect else ""
    print(f"\n{BOLD}── group {n}/{total} ── {group[0]['domain']} ──{RESET}{flag}")
    for i, m in enumerate(group, 1):
        mark = f"{GREEN}keep{RESET}" if i == 1 else f"{RED}drop{RESET}"
        junk = (f"  {YELLOW}← not a real abstract{RESET}"
                if abstract_is_boilerplate(m["text"] or "") else "")
        print(f"  [{i}] {mark}  {_paper_id(m['id']):12}  "
              f"cit {m['citation_count'] or 0:>6}  fwci {m['fwci']}{junk}")
        print(f"      {DIM}{(m['title'] or '')[:88]}{RESET}")
        print(f"      {DIM}{' '.join((m['text'] or '').split())[:110]}…{RESET}")
    return suspect


def resolve_groups(groups: list[list[dict]], *, assume_yes: bool,
                   keep_ids: set[str], drop_group_ids: set[str]) -> dict:
    """Per-group resolution. Returns ``{'drop': [paper_id, ...]}``.

    ``keep_ids`` / ``drop_group_ids`` are ``--keep`` / ``--drop-group`` from the
    command line: a group is matched by *any* of its members' paper ids, and a
    matched group is resolved with no prompt. Everything else is asked
    interactively unless ``assume_yes``; answering ``g`` at any prompt accepts
    the ``[1]`` default for that group and all the rest.
    """
    drop: list[str] = []
    auto = assume_yes
    auto_suspect = 0
    for n, group in enumerate(groups, 1):
        pids = [_paper_id(m["id"]) for m in group]
        suspect = _print_group(n, len(groups), group)

        if drop_group_ids & set(pids):
            drop += pids
            print(f"  {RED}--drop-group: dropping all {len(group)}{RESET}")
            continue
        if keep_ids & set(pids):
            keep = next(p for p in pids if p in keep_ids)
            drop += [p for p in pids if p != keep]
            print(f"  --keep: keep {keep}, drop {len(group) - 1}")
            continue

        choice = "" if auto else input(
            f"  {BOLD}Enter{RESET}=keep [1] · 1/2/…=keep that one · "
            f"a=drop whole group · s=skip · g=keep [1] here and for the rest : "
        ).strip().lower()
        if choice == "g":
            auto, choice = True, ""

        if choice == "s":
            print(f"  {DIM}skipped — both rows stay{RESET}")
            continue
        if choice == "a":
            drop += pids
            print(f"  {RED}dropping all {len(group)}{RESET}")
            continue
        keep_idx = (int(choice) - 1
                    if choice.isdigit() and 1 <= int(choice) <= len(group)
                    else 0)
        drop += [p for i, p in enumerate(pids) if i != keep_idx]
        auto_suspect += suspect and auto and keep_idx == 0
        print(f"  keep {pids[keep_idx]}, drop {len(group) - 1}")

    if auto_suspect:
        hint = ("" if "W2141899406" in keep_ids
                else " (for science_v3 the one wrong default is group 6 — its "
                     "abstract belongs to the other paper — use --keep W2141899406)")
        print(f"\n{YELLOW}accepted the default for {auto_suspect} flagged "
              f"group(s); pass --keep <paper_id> to override one{hint}.{RESET}")
    return {"drop": sorted(set(drop))}


# --------------------------------------------------------------------------
# stage 2 — new raw parquet + new pipeline config
# --------------------------------------------------------------------------


def build_new_raw(new_id: str, drop_ids: list[str], *, dry_run: bool) -> None:
    import pyarrow as pa
    import pyarrow.compute as pc

    src_raw = SCIENCE_DIR / f"{SOURCE_DATASET_ID}_raw.parquet"
    dst_raw = SCIENCE_DIR / f"{new_id}_raw.parquet"
    tbl = pq.read_table(src_raw)

    keep_mask = pc.invert(pc.is_in(tbl.column("paper_id"),
                                   value_set=pa.array(drop_ids)))
    kept = tbl.filter(keep_mask)

    # What filter.py will actually keep per domain — run its own apply_rules
    # (current rules: boilerplate + abstract-dedup both reject now) so the
    # headroom is the real number, not the count the old fetch left behind.
    cfg = load_pipeline_config(SOURCE_CFG)
    scored = apply_rules(kept.to_pandas(), cfg)
    admissible = Counter(
        scored.loc[scored["drop_reason"] == "", "domain"].astype(str)
    )

    id_to_domain = dict(zip(tbl.column("paper_id").to_pylist(),
                            tbl.column("domain").to_pylist()))
    dropped_per_domain = Counter(id_to_domain[p] for p in drop_ids
                                 if p in id_to_domain)

    target = cfg["fetch"]["target_per_domain"]
    print(f"\n  removing {len(drop_ids)} paper(s) from {src_raw.name}; "
          f"filter re-tested with current rules")
    short = False
    for dom in sorted(admissible):
        head = admissible[dom] - target
        short |= head < 0
        tag = f"{GREEN}ok{RESET}" if head >= 0 else f"{RED}SHORT {head}{RESET}"
        print(f"    {dom:18} you drop {dropped_per_domain[dom]:2}   "
              f"admissible {admissible[dom]:4}   headroom {head:+4}  {tag}")
    if short:
        raise SystemExit(
            f"{RED}A domain is below {target} admissible papers after the drop "
            f"— the overfetch surplus is exhausted. A real top-up fetch would "
            f"be needed, which this script does not do.{RESET}")

    if dry_run:
        print(f"  {DIM}[dry-run] would write {dst_raw}{RESET}")
    else:
        pq.write_table(kept, dst_raw)
        print(f"  wrote {dst_raw}  ({kept.num_rows} rows)")

    write_new_cfg(new_id, drop_ids, dry_run=dry_run)


def write_new_cfg(new_id: str, drop_ids: list[str], *, dry_run: bool) -> None:
    dst = BASE_DIR / "cfg" / "science_data" / f"{new_id}.yaml"
    text = SOURCE_CFG.read_text()
    old_line = f'dataset_id: "{SOURCE_DATASET_ID}"'
    new_line = f'dataset_id: "{new_id}"'
    if text.count(old_line) != 1:
        raise SystemExit(f"{RED}cannot find a single {old_line!r} in "
                         f"{SOURCE_CFG}{RESET}")
    header = (
        f"# GENERATED by epai/science_data/rebuild_corpus.py — do not hand-edit.\n"
        f"#\n"
        f"# {new_id} is {SOURCE_DATASET_ID} with {len(drop_ids)} duplicate-abstract\n"
        f"# human papers removed and the freed slots backfilled from the same\n"
        f"# fetch's overfetch surplus. Only the paper set changed; every fetch and\n"
        f"# filter threshold below is identical to {SOURCE_DATASET_ID}.yaml.\n"
        f"#\n"
        f"# Dropped paper ids: {', '.join(drop_ids)}\n"
        f"#\n"
    )
    out = header + text.replace(old_line, new_line, 1)
    if dry_run:
        print(f"  {DIM}[dry-run] would write {dst}{RESET}")
    else:
        dst.write_text(out)
        print(f"  wrote {dst}")


# --------------------------------------------------------------------------
# stages 3-8 — drive the existing pipeline CLIs
# --------------------------------------------------------------------------


def run(cmd: list[str], *, dry_run: bool) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"\n  {DIM}$ {printable}{RESET}")
    if dry_run:
        return
    subprocess.run([str(c) for c in cmd], check=True, cwd=BASE_DIR)


def stage_filter(new_cfg: Path, *, dry_run: bool) -> None:
    run([PY, "epai/science_data/filter/filter.py", "--cfg", new_cfg],
        dry_run=dry_run)


def stage_prompts(new_cfg: Path, *, dry_run: bool) -> None:
    run([PY, "epai/science_data/prompt/make_prompts.py", "--cfg", new_cfg],
        dry_run=dry_run)


def stage_generate(new_id: str, new_cfg: Path, *, dry_run: bool) -> None:
    src_shard = SCIENCE_DIR / f"{SOURCE_DATASET_ID}_generate.jsonl"
    dst_shard = SCIENCE_DIR / f"{new_id}_generate.jsonl"
    if not dst_shard.exists():
        print(f"  copy {src_shard.name} -> {dst_shard.name}  "
              f"{DIM}(reuse every paid generation for the papers that stayed){RESET}")
        if not dry_run:
            shutil.copy2(src_shard, dst_shard)
    run([PY, "epai/science_data/generate/generate_ai.py", "--cfg", new_cfg],
        dry_run=dry_run)

    if dry_run:
        return
    out = SCIENCE_DIR / f"{new_id}_{VARIANT}.parquet"
    tbl = pq.read_table(out, columns=["id", "generated_by"])
    per_id = Counter(tbl.column("id").to_pylist())
    incomplete = {i: c for i, c in per_id.items() if c != 4}
    if incomplete:
        print(f"\n{YELLOW}  {len(incomplete)} paper(s) do not have all 4 rows "
              f"(human + 3 generators):{RESET}")
        for i, c in list(incomplete.items())[:10]:
            print(f"    {i}  ({c} rows)")
        print(f"  a generator dropped them to rate limits. Re-run:\n"
              f"    {PY} epai/science_data/rebuild_corpus.py --from generate\n"
              f"  (generate_ai.py retries only the missing calls), then continue.")
        raise SystemExit(1)
    print(f"  {GREEN}{out.name}: {tbl.num_rows} rows, "
          f"{len(per_id)} papers, all complete{RESET}")


def stage_promote(new_id: str, *, dry_run: bool) -> None:
    src = SCIENCE_DIR / f"{new_id}_{VARIANT}.parquet"
    dst = PARQUET_DIR / f"{new_id}_{VARIANT}.parquet"
    print(f"  copy {src.name} -> data/parquet/")
    if not dry_run:
        shutil.copy2(src, dst)
    run([PY, "epai/ai_detection/prepare/standardize_text.py", dst],
        dry_run=dry_run)


def stage_carry(new_id: str, in_parquet: Path, *, dry_run: bool) -> None:
    dst = PARQUET_DIR / f"{new_id}_{VARIANT}.parquet"
    run([PY, "epai/ai_detection/prepare/carry_suite_columns.py",
         dst, "--source", in_parquet], dry_run=dry_run)


def stage_score(new_id: str, gpu: str | None, *, dry_run: bool) -> None:
    dst = PARQUET_DIR / f"{new_id}_{VARIANT}.parquet"
    cmd = [PY, "epai/ai_detection/prepare/run_holdout_pipeline.py",
           "--in", dst, "-o", dst, "--from", "embed"]
    if gpu is not None:
        cmd += ["--gpu", gpu]
    if gpu is None:
        print(f"\n  {YELLOW}the score stage needs a GPU box and the Binoculars "
              f"sub-venv (uv on PATH). Run there:{RESET}")
        print(f"    {' '.join(str(c) for c in cmd)} --yes")
        print(f"\n  then verify:\n"
              f"    {PY} epai/scripts/check_parquet_completeness.py {dst} -v")
        return
    run(cmd + ["--yes"], dry_run=dry_run)
    run([PY, "epai/scripts/check_parquet_completeness.py", dst, "-v"],
        dry_run=dry_run)


# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_parquet", type=Path,
                    default=PARQUET_DIR / f"{SOURCE_DATASET_ID}_{VARIANT}.parquet",
                    help="the scored corpus to rebuild (default: science_v3_multi_0811A)")
    ap.add_argument("--new-id", default="science_v3d",
                    help="dataset_id for the rebuilt corpus (default: science_v3d)")
    ap.add_argument("--from", dest="start", choices=STAGES, default="detect",
                    help="resume from this stage (default: detect)")
    ap.add_argument("--gpu", default=None, metavar="ID",
                    help="run the score stage here on this GPU; without it the "
                         "score command is printed, not run")
    ap.add_argument("--yes", action="store_true",
                    help="don't prompt: take the keep-highest-citation default "
                         "for every duplicate group (still prints them). Combine "
                         "with --keep / --drop-group to override individual groups")
    ap.add_argument("--keep", action="append", default=[], metavar="PAPER_ID",
                    help="in the duplicate group containing this paper id, keep "
                         "this row instead of the highest-citation default "
                         "(repeatable)")
    ap.add_argument("--drop-group", action="append", default=[], metavar="PAPER_ID",
                    help="drop every row of the duplicate group containing this "
                         "paper id (repeatable)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every action and command, change nothing")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    new_id = args.new_id
    new_cfg = BASE_DIR / "cfg" / "science_data" / f"{new_id}.yaml"
    plan_path = SCIENCE_DIR / f"{new_id}_dedupe_plan.json"
    start_at = STAGES.index(args.start)

    def active(stage: str) -> bool:
        return STAGES.index(stage) >= start_at

    if active("detect"):
        if not args.in_parquet.is_file():
            raise SystemExit(f"{RED}not found: {args.in_parquet}{RESET}")
        report_boilerplate(args.in_parquet)
        groups = find_duplicate_groups(args.in_parquet)
        print(f"\n{BOLD}{len(groups)} duplicate-abstract group(s) among the human "
              f"rows of {args.in_parquet.name}{RESET}")
        plan = (resolve_groups(groups, assume_yes=args.yes,
                               keep_ids=set(args.keep),
                               drop_group_ids=set(args.drop_group))
                if groups else {"drop": []})

        n = len(plan["drop"])
        print(f"\n{BOLD}plan{RESET}: drop {n} duplicate paper(s)"
              + (f" — {', '.join(plan['drop'])}" if n else "")
              + "; the filter stage additionally drops the non-abstract rows above")
        if not args.yes and input(f"{BOLD}build {new_id}?{RESET} [y/N] ")\
                .strip().lower() not in ("y", "yes"):
            print("aborted")
            return
        if not args.dry_run:
            plan_path.write_text(json.dumps(plan, indent=2))
            print(f"  wrote {plan_path}")
    else:
        if not plan_path.is_file():
            raise SystemExit(
                f"{RED}no plan at {plan_path} — run the detect stage first "
                f"(drop --from, or --from detect){RESET}")
        plan = json.loads(plan_path.read_text())
        print(f"{DIM}resuming from {args.start}; plan: drop "
              f"{len(plan['drop'])} paper(s){RESET}")

    drop_ids = plan["drop"]

    if active("raw"):
        print(f"\n{BOLD}== raw =={RESET}")
        build_new_raw(new_id, drop_ids, dry_run=args.dry_run)
    if active("filter"):
        print(f"\n{BOLD}== filter =={RESET}")
        stage_filter(new_cfg, dry_run=args.dry_run)
    if active("prompts"):
        print(f"\n{BOLD}== prompts =={RESET}")
        stage_prompts(new_cfg, dry_run=args.dry_run)
    if active("generate"):
        print(f"\n{BOLD}== generate =={RESET}")
        stage_generate(new_id, new_cfg, dry_run=args.dry_run)
    if active("promote"):
        print(f"\n{BOLD}== promote + standardize =={RESET}")
        stage_promote(new_id, dry_run=args.dry_run)
    if active("carry"):
        print(f"\n{BOLD}== carry suite columns =={RESET}")
        stage_carry(new_id, args.in_parquet, dry_run=args.dry_run)
    if active("score"):
        print(f"\n{BOLD}== score =={RESET}")
        stage_score(new_id, args.gpu, dry_run=args.dry_run)

    print(f"\n{GREEN}done{RESET} — {PARQUET_DIR / f'{new_id}_{VARIANT}.parquet'}")


if __name__ == "__main__":
    main()
