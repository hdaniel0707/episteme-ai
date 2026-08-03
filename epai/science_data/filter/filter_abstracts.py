"""Stage 2 — filter the raw corpus down to exactly N human abstracts per domain.

This stage re-runs the *same* admission rules the fetcher used, over the raw
parquet, and then trims each domain's surplus back to ``target_per_domain``.
Re-running rather than trusting the ``kept`` flag written at fetch time is the
point: it means you can tighten a threshold in the config and rebuild the
corpus from local data, with no API calls at all.

Because the fetcher already stopped on the post-filter count, every domain
arrives here with at least its target (fetch fails loudly otherwise), so the
trim is a surplus trim and never a shortfall.

Output columns follow the AI-detection corpus convention (``id`` / ``text`` /
``generated_by``), so the result drops straight into Stage B of that pipeline:

    id            -> "science:<paperId>"  — shared by the human row and every
                     AI row generated from it, which is what the per-id
                     human-vs-AI analyses key on
    text          -> the cleaned abstract, alone (the title is a prompt input,
                     not part of the text being classified)
    generated_by  -> "human"
    title, year, venue, citation_count, domain, query  — carried along

Output: data/parquet/<dataset_id>_all.parquet     (every domain)
        data/parquet/<dataset_id>_<domain>.parquet (one per domain)
        results/science_data/filter/<dataset_id>/  (drop-reason report)

Run:
    python -m epai.science_data.filter.filter_abstracts \
        --cfg cfg/science_data/science_cfg_example.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from epai.science_data.config import (
    BASE_DIR,
    filter_config,
    human_path,
    load_pipeline_config,
    raw_path,
)
from epai.science_data.rules import (
    DROP_REASONS,
    SeenIndex,
    check_record,
    normalise_title,
)
from epai.utils.parquet_utils import load_parquet, save_parquet

RED = "\033[91m"
RESET = "\033[0m"

RESULTS_DIR = BASE_DIR / "results" / "science_data" / "filter"

OUT_COLUMNS = [
    "id",
    "text",
    "generated_by",
    "title",
    "year",
    "venue",
    "citation_count",
    "influential_citation_count",
    "fields_of_study",
    "domain",
    "query",
    "also_matched_domains",
]


def apply_rules(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Re-run the row-local rules plus deduplication over the raw corpus.

    Adds ``drop_reason`` (empty string when admissible). Rows are processed in
    fetch order so that first-domain-wins deduplication reproduces exactly what
    the fetcher did.
    """
    rules_cfg = filter_config(cfg)
    seen = SeenIndex()
    reasons: list[str] = []

    for i, row in enumerate(
        tqdm(df.to_dict("records"), desc="applying rules", unit="row")
    ):
        reason = check_record(row, rules_cfg)
        if reason is None:
            norm = normalise_title(row.get("title"))
            if seen.lookup(row.get("paper_id") or "", norm) is not None:
                reason = "duplicate"
            else:
                seen.claim(row.get("paper_id") or "", norm, i)
        reasons.append(reason or "")

    out = df.copy()
    out["drop_reason"] = reasons
    return out


def trim_to_target(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Cut each domain's surplus back to exactly ``target_per_domain``."""
    target = cfg["fetch"]["target_per_domain"]
    mode = cfg["filter"].get("trim_mode", "head")
    seed = cfg["filter"].get("seed", 42)

    parts = []
    short = []
    for domain, group in df.groupby("domain", sort=False):
        if len(group) < target:
            short.append((domain, len(group)))
            parts.append(group)
            continue
        if mode == "sample":
            parts.append(group.sample(n=target, random_state=seed))
        else:
            parts.append(group.head(target))

    if short:
        lines = "\n".join(f"  {d}: {n} admissible, need {target}" for d, n in short)
        print(
            f"{RED}ERROR: domain(s) below target after filtering:\n{lines}{RESET}",
            file=sys.stderr,
        )
        print(
            "The fetch stage should have caught this. Either it was run with "
            "--allow-shortfall, or filter thresholds were tightened since. "
            "Re-run fetch with the current config, or lower target_per_domain.",
            file=sys.stderr,
        )
        sys.exit(1)

    return pd.concat(parts, ignore_index=True)


def write_report(raw: pd.DataFrame, cfg: dict) -> None:
    """Save the drop-reason breakdown — overall, and per domain.

    The most useful artifact of the whole stage: it tells you what S2 actually
    returned and which rule is doing the work, which is exactly what you need
    to know before trusting the corpus.
    """
    out_dir = RESULTS_DIR / cfg["dataset_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = ["kept", *DROP_REASONS]
    reason = raw["drop_reason"].replace("", "kept")

    overall = reason.value_counts().reindex(labels, fill_value=0)
    overall_df = pd.DataFrame(
        {
            "reason": overall.index,
            "count": overall.to_numpy(),
            "pct": (100.0 * overall / max(len(raw), 1)).round(2).to_numpy(),
        }
    )
    overall_df.to_csv(out_dir / "drop_reasons.csv", index=False)

    per_domain = (
        pd.crosstab(raw["domain"], reason)
        .reindex(columns=labels, fill_value=0)
        .reset_index()
    )
    per_domain.to_csv(out_dir / "drop_reasons_per_domain.csv", index=False)

    # How dirty the raw abstracts were, i.e. how much clean_abstract had to do.
    fixes = (
        raw["cleaning_fixes"].fillna("").str.split(",").explode().replace("", pd.NA).dropna()
    )
    fix_counts = fixes.value_counts().rename_axis("fix").reset_index(name="count")
    fix_counts.to_csv(out_dir / "cleaning_fixes.csv", index=False)

    print(f"\nDrop reasons ({len(raw)} raw records):")
    for _, r in overall_df.iterrows():
        if r["count"]:
            print(f"  {r['reason']:<26} {int(r['count']):>7}  ({r['pct']:>5.2f}%)")
    print(f"\nReport → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, required=True, help="pipeline config YAML")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.cfg)
    raw = load_parquet(raw_path(cfg))

    scored = apply_rules(raw, cfg)
    write_report(scored, cfg)

    kept = scored[scored["drop_reason"] == ""].copy()
    print(f"\n{len(kept)} admissible of {len(scored)} raw records")

    kept["id"] = "science:" + kept["paper_id"].astype(str)
    kept["text"] = kept["abstract"]
    kept["generated_by"] = "human"

    final = trim_to_target(kept, cfg)[OUT_COLUMNS]

    save_parquet(final, human_path(cfg))
    for domain, group in final.groupby("domain", sort=False):
        save_parquet(group.reset_index(drop=True), human_path(cfg, domain))

    print("\nPer-domain counts:")
    print(final["domain"].value_counts().to_string())

    overlap = (final["also_matched_domains"].fillna("") != "").sum()
    print(
        f"\n{overlap} paper(s) also matched another domain's queries "
        f"({100.0 * overlap / max(len(final), 1):.1f}%) — assigned first-domain-wins. "
        "A high share here means the domains overlap more than the LODO setup assumes."
    )


if __name__ == "__main__":
    main()
