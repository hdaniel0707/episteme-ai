r"""Regenerate the runaway machine rows in ``ghostbuster_gpt56luna.parquet``.

A runaway is a decoding loop — one generator's answer to a prompt, many times
longer than the human original (see check_parquet_completeness.py). Dropping it
takes the whole ``id`` group with it (the human original and the other three
generators' good answers to the same prompt). This regenerates just the bad row
instead, and does it **through the fork's own generate.py**, so the replacement
document goes through the exact prompt, model and post-processing that built the
rest of the corpus — nothing is reimplemented here.

WHAT IT DOES
  1. find the runaway rows (same gates as the checker; ``--ids`` to name them)
  2. delete the fork's ``.txt`` for each and run ``generate.py`` in the fork's
     environment (``uv run --project external/ghostbuster``) — BILLED API calls,
     one per row; the fork skips every file that already exists, so only the
     deleted ones are regenerated. The fork now picks ``max_tokens`` /
     ``max_completion_tokens`` from the endpoint, so the OpenAI call succeeds.
  3. standardise the new text (epai.utils.text_standardize, whitespace=lines —
     the pipeline's ``standardize`` stage, which generate.py output has not seen)
  4. write ``text`` / ``text_raw`` into ``ghostbuster_gpt56luna.parquet`` and
     NULL every suite column on those rows, for ``fill_suite.py`` to recompute

It only touches rows it regenerated. The regenerated ``.txt`` is also copied
into the collected tree, so a later ``run_full_pipeline.py`` over gpt56luna is
consistent. A fresh generation that is ITSELF a runaway is reported, not spliced.

Pipeline:  regen (this)  ->  gb_v3f_merge.py  ->  fill_suite.py  ->  gb_v3f_clean.py

    uv run python epai/ai_detection/rollout/gb_v3f_regen.py --yes
    uv run python epai/ai_detection/rollout/gb_v3f_regen.py \
        --ids essay:800 essay:885 --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from epai.ai_detection.rollout._common import (
    BASE_DIR, DIM, GREEN, RED, RESET, YELLOW, banner, rel,
)
from epai.ai_detection.rollout.gb_v3f_clean import RATIO_WARN, runaway_hits
from epai.utils.text_standardize import standardize_text

# The fork's subprocess inherits os.environ; the API key must be in it first.
for _env in (BASE_DIR / ".env", BASE_DIR / "external/ghostbuster/.env"):
    load_dotenv(_env)

FORK = BASE_DIR / "external" / "ghostbuster"
COLLECTED = BASE_DIR / "data" / "ghostbuster-data-gpt56luna"
SOURCE = BASE_DIR / "data" / "parquet" / "ghostbuster_gpt56luna.parquet"

# generated_by label -> what generate.py needs. ``base_url`` None means OpenAI's
# own API (no endpoint flags); a GenAI4Science generator sets base_url +
# api_key_env, passed through as --base_url / --api_key_env.
REGENERABLE: dict[str, dict] = {
    "gpt56luna:0701A": {
        "gpt_model": "gpt-5.6-luna", "out_name": "gpt56luna_0701A",
        "base_url": None, "api_key_env": "OPENAI_API_KEY",
    },
}

# Everything that is not row identity gets nulled and recomputed by fill_suite.
KEEP_COLS = {"uid", "id", "domain", "text", "text_raw", "generated_by"}


def fork_env() -> dict:
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)          # inherited from the outer `uv run`
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    env["CUDA_VISIBLE_DEVICES"] = ""      # pure API calls
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
    """Delete each target's .txt, then one generate.py run per (domain, spec)."""
    for tg in rows:
        p = fork_txt(tg)
        if p.is_file():
            print(f"{DIM}$ rm {rel(p)}{RESET}")
            if not dry_run:
                p.unlink()

    runs: dict[tuple, dict] = {}
    for tg in rows:
        s = tg["spec"]
        runs.setdefault((tg["domain"], s["out_name"], s["gpt_model"],
                         s["base_url"], s["api_key_env"]), s)

    for (domain, out_name, model, base_url, api_key_env), _ in runs.items():
        cmd = ["uv", "run", "--project", str(FORK), "python", "generate.py",
               f"--{domain}_gpt_plain", "--out_name", out_name,
               "--gpt_model", model, "--seed", "0", "--workers", str(workers)]
        if base_url is not None:
            cmd += ["--base_url", base_url, "--api_key_env", api_key_env]
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

    for tg in rows:                       # keep the collected tree in step
        dst = collected_txt(tg)
        if dst.parent.is_dir():
            dst.write_text(fork_txt(tg).read_text())
            print(f"{DIM}  synced {rel(dst)}{RESET}")
    return 0


def splice(src: Path, out: Path, rows: list[dict], force_runaway: bool) -> int:
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
              f"better on a retry), pass --force-runaway to splice anyway, or "
              f"drop these with gb_v3f_clean.py.")
        return 1
    if not new_text:
        print(f"{YELLOW}nothing to splice (no matching row for the target "
              f"id + label).{RESET}")
        return 1

    # Rebuild column by column. text / text_raw get the new document; every
    # suite column (scalar OR list<double> embedding) is nulled on the target
    # rows via a pylist round-trip — pyarrow's replace_with_mask has no kernel
    # for list types, and the parquet is only 6000 rows.
    pass_through = KEEP_COLS - {"text", "text_raw"}
    cols = []
    for name in t.column_names:
        col = t.column(name)
        if name in pass_through:
            cols.append(col)
            continue
        vals = col.to_pylist()
        if name == "text":
            for i, v in new_text.items():
                vals[i] = v
        elif name == "text_raw":
            for i, v in new_raw.items():
                vals[i] = v
        else:
            for i in new_text:
                vals[i] = None
        cols.append(pa.array(vals, type=col.type))

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
    ap.add_argument("parquet", type=Path, nargs="?", default=SOURCE,
                    help=f"per-generator parquet to patch (default {rel(SOURCE)})")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="write here instead of patching the input in place")
    ap.add_argument("--ids", nargs="+", default=None, metavar="ID",
                    help="rows to regenerate (default: auto-detect runaways)")
    ap.add_argument("--ratio", type=float, default=4.0,
                    help="runaway ratio for auto-detect (default 4.0)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force-runaway", action="store_true",
                    help="splice even if the fresh generation is also a runaway")
    ap.add_argument("--splice-only", action="store_true",
                    help="skip generation, splice the .txt already in the fork "
                         "tree (resume after a splice failure without re-billing)")
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

    banner("targets", rel(src))
    rows = targets(pq.read_table(src, columns=["id", "generated_by", "text"]),
                   args.ids, args.ratio)
    if not rows:
        print(f"{GREEN}no regenerable runaway rows.{RESET}")
        return

    missing = sorted({s["api_key_env"] for tg in rows for s in (tg["spec"],)
                      if not os.environ.get(s["api_key_env"])})
    if not args.dry_run and missing:
        raise SystemExit(
            f"{RED}not set: {', '.join(missing)}{RESET}\n  needed by the target "
            f"generator(s); put the key(s) in this repo's .env or "
            f"external/ghostbuster/.env.")

    for tg in rows:
        gap = check_seeds(tg)
        exists = fork_txt(tg).is_file()
        print(f"  {tg['id']:<14} {tg['label']:<16} "
              f"fork: data/{tg['domain']}/{tg['spec']['out_name']}/{tg['idx']}.txt"
              f"{'  (present)' if exists else ''}"
              + (f"  {RED}{gap}{RESET}" if gap else ""))
        if gap and not args.splice_only:
            raise SystemExit(1)
        if args.splice_only and not exists:
            raise SystemExit(f"{RED}--splice-only: {rel(fork_txt(tg))} is not "
                             f"there — nothing to splice.{RESET}")

    if not args.splice_only:
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
    elif args.dry_run:
        print(f"\n{DIM}--dry-run --splice-only: would splice the existing "
              f".txt; nothing written.{RESET}")
        return

    banner("splice", rel(out))
    code = splice(src, out, rows, args.force_runaway)
    if code != 0:
        raise SystemExit(code)
    print(f"\n  next: uv run python epai/ai_detection/rollout/gb_v3f_merge.py")


if __name__ == "__main__":
    main()
