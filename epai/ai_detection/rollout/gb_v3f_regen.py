r"""Regenerate the runaway machine rows in the merged Ghostbuster corpus, in place.

A runaway is a decoding loop — one generator's answer to a prompt, many times
longer than the human original (see check_parquet_completeness.py). Dropping it
takes the whole ``id`` group with it (the human original and every other
generator's good answer to the same prompt). This regenerates just the bad row
instead: it asks the SAME generator for a fresh document, standardises it, drops
it back into the row, and nulls that row's suite columns so ``fill_suite.py``
re-scores only it.

WHAT IT CAN REGENERATE
    Rows whose ``generated_by`` is a label this script knows how to reproduce
    (``REGENERABLE`` below — currently ``gpt56luna:0701A`` via OpenAI) in the
    ``essay`` / ``wp`` domains, where the fork addresses a document as
    ``data/<domain>/<out_name>/<idx>.txt``. A ``reuter`` id or an unknown
    generator is reported and left for ``gb_v3f_clean.py`` to drop.

WHAT IT DOES
  1. find the runaway rows (same gates as the checker; ``--ids`` to name them)
  2. delete the fork's ``.txt`` for each, run ``generate.py`` in the fork's env
     (BILLED API calls — one per row), confirm the files came back
  3. standardise the new text (epai.utils.text_standardize, whitespace=lines)
  4. write ``text`` / ``text_raw`` back into the merged parquet and NULL every
     suite column for those rows

It never touches a row it did not regenerate, and never the source
``ghostbuster_<gen>.parquet`` files — only the merged corpus given as argument.
A fresh generation that is ITSELF a runaway is reported and NOT spliced.

    uv run python epai/ai_detection/rollout/gb_v3f_regen.py \
        data/parquet/ghostbuster_v3f_multi.parquet --yes
    uv run python epai/ai_detection/rollout/gb_v3f_regen.py \
        data/parquet/ghostbuster_v3f_multi.parquet --ids essay:800 essay:885 --dry-run

Next: fill_suite.py on the same file.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from dotenv import load_dotenv

from epai.ai_detection.rollout._common import (
    BASE_DIR, BOLD, DIM, GREEN, RED, RESET, YELLOW, banner, rel,
)
from epai.ai_detection.rollout.gb_v3f_clean import RATIO_WARN, runaway_hits
from epai.ai_detection.rollout.gemma_ghostbuster_new import COMBINED
from epai.utils.text_standardize import standardize_text

# GENAI4SCIENCE_*/OPENAI_API_KEY live in THIS repo's .env; the fork's subprocess
# inherits os.environ, so the key must be in this process first.
load_dotenv()

FORK = BASE_DIR / "external" / "ghostbuster"
COLLECTED = BASE_DIR / "data" / "ghostbuster-data-gpt56luna"

# generated_by label -> what the fork needs to reproduce it. Plain-OpenAI only
# for now (no --base_url / --api_key_env); add an entry with those keys for a
# GenAI4Science-hosted generator.
REGENERABLE: dict[str, dict] = {
    "gpt56luna:0701A": {"gpt_model": "gpt-5.6-luna", "out_name": "gpt56luna_0701A"},
}

# Everything that is not row identity gets nulled and recomputed by fill_suite.
KEEP_COLS = {"uid", "id", "domain", "text", "text_raw", "generated_by"}


def fork_env() -> dict:
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    env["CUDA_VISIBLE_DEVICES"] = ""          # pure API calls, keep off the GPU
    return env


def parse_id(rid: str) -> tuple[str, int] | None:
    """``essay:800`` -> ``("essay", 800)``; None for reuter / anything else."""
    parts = rid.split(":")
    if len(parts) == 2 and parts[0] in ("essay", "wp") and parts[1].isdigit():
        return parts[0], int(parts[1])
    return None


def targets(t: pa.Table, ids: list[str] | None, ratio: float) -> list[dict]:
    """The rows to regenerate, resolved to (id, domain, idx, label, spec)."""
    id_col = [str(x) for x in t.column("id").to_pylist()]
    gen_col = [str(x) for x in t.column("generated_by").to_pylist()]

    if ids:
        want = set(ids)
        wanted = {(id_col[i], gen_col[i]) for i in range(len(id_col))
                  if id_col[i] in want and gen_col[i] != "human"}
    else:
        wanted = {(rid, g) for rid, g, *_ in runaway_hits(t, ratio, False)}

    out: list[dict] = []
    for rid, label in sorted(wanted):
        if label not in REGENERABLE:
            print(f"  {YELLOW}skip{RESET} {rid}  {label} — no regen recipe; "
                  f"drop it with gb_v3f_clean.py")
            continue
        dom_idx = parse_id(rid)
        if dom_idx is None:
            print(f"  {YELLOW}skip{RESET} {rid} — only essay/wp ids can be "
                  f"regenerated here; drop it with gb_v3f_clean.py")
            continue
        domain, idx = dom_idx
        out.append({"id": rid, "domain": domain, "idx": idx,
                    "label": label, "spec": REGENERABLE[label]})
    return out


def check_seeds(tg: dict) -> str | None:
    for sub in ("human", "prompts"):
        p = FORK / "data" / tg["domain"] / sub / f"{tg['idx']}.txt"
        if not p.is_file():
            return f"missing seed input {rel(p)}"
    return None


def fork_txt(tg: dict) -> Path:
    return (FORK / "data" / tg["domain"] / tg["spec"]["out_name"]
            / f"{tg['idx']}.txt")


def collected_txt(tg: dict) -> Path:
    return (COLLECTED / tg["domain"] / tg["spec"]["out_name"]
            / f"{tg['idx']}.txt")


def regenerate(rows: list[dict], workers: int, dry_run: bool) -> int:
    for tg in rows:
        p = fork_txt(tg)
        if p.is_file():
            print(f"{DIM}$ rm {rel(p)}{RESET}")
            if not dry_run:
                p.unlink()

    combos = {(tg["domain"], tg["spec"]["out_name"], tg["spec"]["gpt_model"])
              for tg in rows}
    for domain, out_name, model in sorted(combos):
        cmd = ["uv", "run", "--project", str(FORK), "python", "generate.py",
               f"--{domain}_gpt_plain", "--out_name", out_name,
               "--gpt_model", model, "--seed", "0", "--workers", str(workers)]
        print(f"\n{DIM}$ {' '.join(cmd)}{RESET}{DIM}  (cwd {rel(FORK)}){RESET}")
        if dry_run:
            continue
        code = subprocess.run(cmd, cwd=FORK, env=fork_env()).returncode
        if code != 0:
            print(f"{RED}generate.py failed for {domain} (exit {code}).{RESET}")
            return code

    if dry_run:
        return 0

    missing = [tg["id"] for tg in rows
               if not (fork_txt(tg).is_file() and fork_txt(tg).read_text().strip())]
    if missing:
        print(f"{RED}no usable text came back for: {', '.join(missing)}{RESET}")
        return 1

    # Keep the collected tree in step with the fork tree, the way `collect`
    # would — so a later run_full_pipeline over gpt56luna sees the fix too.
    for tg in rows:
        dst = collected_txt(tg)
        if dst.parent.is_dir():
            dst.write_text(fork_txt(tg).read_text())
            print(f"{DIM}  synced {rel(dst)}{RESET}")
    return 0


def splice(src: Path, out: Path, rows: list[dict], ratio: float,
           force_runaway: bool) -> int:
    t = pq.read_table(src)
    id_col = [str(x) for x in t.column("id").to_pylist()]
    gen_col = [str(x) for x in t.column("generated_by").to_pylist()]
    words = [len((s or "").split()) for s in t.column("text").to_pylist()]
    human_w = {id_col[i]: words[i] for i in range(len(id_col))
               if gen_col[i] == "human"}

    new_text: dict[int, str] = {}
    new_raw: dict[int, str] = {}
    still_runaway: list[str] = []
    for tg in rows:
        raw = fork_txt(tg).read_text()
        std = standardize_text(raw, whitespace="lines")
        h = human_w.get(tg["id"])
        ratio_now = (len(std.split()) / h) if h else 0.0
        flag = f"{ratio_now:.1f}x human" if h else f"{len(std.split())}w"
        print(f"  {tg['id']:<14} {tg['label']:<16} -> {len(std.split()):>5}w "
              f"({flag})")
        if h and ratio_now >= RATIO_WARN:
            still_runaway.append(tg["id"])
        for i in range(len(id_col)):
            if id_col[i] == tg["id"] and gen_col[i] == tg["label"]:
                new_text[i], new_raw[i] = std, raw

    if still_runaway and not force_runaway:
        print(f"\n{RED}still a runaway after regeneration: "
              f"{', '.join(still_runaway)}{RESET}\n  Re-run (the model may do "
              f"better on a retry), or pass --force-runaway to splice anyway, "
              f"or drop these with gb_v3f_clean.py.")
        return 1
    if not new_text:
        print(f"{YELLOW}nothing to splice.{RESET}")
        return 1

    row_mask = pa.array([i in new_text for i in range(len(id_col))])
    n_targets = len(new_text)
    cols = []
    for name in t.column_names:
        col = t.column(name)
        if name == "text":
            vals = col.to_pylist()
            for i, v in new_text.items():
                vals[i] = v
            cols.append(pa.array(vals, type=col.type))
        elif name == "text_raw":
            vals = col.to_pylist()
            for i, v in new_raw.items():
                vals[i] = v
            cols.append(pa.array(vals, type=col.type))
        elif name in KEEP_COLS:
            cols.append(col)
        else:
            cols.append(pc.replace_with_mask(
                col, row_mask, pa.nulls(n_targets, type=col.type)))

    pq.write_table(pa.table(cols, schema=t.schema), out)
    nulled = [n for n in t.column_names if n not in KEEP_COLS]
    print(f"\n{GREEN}wrote{RESET} {rel(out)}  "
          f"({len(new_text)} row(s) rebuilt, {len(nulled)} suite column(s) "
          f"nulled on them)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("parquet", type=Path, nargs="?", default=COMBINED,
                    help=f"merged corpus to patch (default {rel(COMBINED)})")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="write here instead of patching the input in place")
    ap.add_argument("--ids", nargs="+", default=None, metavar="ID",
                    help="rows to regenerate (default: auto-detect runaways)")
    ap.add_argument("--ratio", type=float, default=4.0,
                    help="runaway ratio for auto-detect (default 4.0)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force-runaway", action="store_true",
                    help="splice even if the fresh generation is also a runaway")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src: Path = args.parquet
    if not src.is_file():
        raise SystemExit(f"{RED}not a file: {rel(src)}{RESET}")
    out: Path = args.out or src

    if not (FORK / "generate.py").is_file():
        raise SystemExit(f"{RED}fork not checked out: {rel(FORK / 'generate.py')}"
                         f"{RESET}\n  git submodule update --init")
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY") \
            and not (FORK / ".env").is_file():
        raise SystemExit(
            f"{RED}OPENAI_API_KEY not set and {rel(FORK / '.env')} absent.{RESET}\n"
            f"  gpt-5.6-luna is an OpenAI-direct model; put the key in this "
            f"repo's .env or the fork's .env before running.")

    banner("targets", rel(src))
    rows = targets(pq.read_table(src, columns=["id", "generated_by", "text"]),
                   args.ids, args.ratio)
    if not rows:
        print(f"{GREEN}no regenerable runaway rows.{RESET}")
        return

    for tg in rows:
        gap = check_seeds(tg)
        print(f"  {tg['id']:<14} {tg['label']:<16} "
              f"fork: data/{tg['domain']}/{tg['spec']['out_name']}/{tg['idx']}.txt"
              + (f"  {RED}{gap}{RESET}" if gap else ""))
        if gap:
            raise SystemExit(1)

    print(f"\n{YELLOW}{len(rows)} document(s) will be regenerated with billed "
          f"API calls.{RESET}")
    if not args.dry_run and not args.yes:
        if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit("aborted.")

    banner("regenerate", f"{rel(FORK)}/generate.py")
    code = regenerate(rows, args.workers, args.dry_run)
    if code != 0:
        raise SystemExit(code)
    if args.dry_run:
        print(f"\n{DIM}--dry-run: no files deleted, no calls made, "
              f"nothing spliced.{RESET}")
        return

    banner("splice", rel(out))
    code = splice(src, out, rows, args.ratio, args.force_runaway)
    if code != 0:
        raise SystemExit(code)
    print(f"\n  next: uv run python epai/ai_detection/rollout/fill_suite.py "
          f"{rel(out)} --gpu <id> --yes")


if __name__ == "__main__":
    main()
