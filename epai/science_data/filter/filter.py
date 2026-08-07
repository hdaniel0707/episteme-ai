"""Stage 2 — reduce the raw corpus to exactly N human records per domain.

Three jobs, in order: re-run the admission rules over every raw record, trim
each domain's surplus back to ``target_per_domain``, and reshape what survives
into the AI-detection corpus schema.

Despite the name of the rules it applies, this is a *record* filter, not an
abstract filter. Only 5 of the 14 drop reasons look at the abstract text
(missing, truncated, too short, too long, not English); the other 9 reject on
the id, the title, the year, the citation count, the FWCI, the language label,
or a duplicate already claimed by an earlier domain. A record is admitted as a
whole or not at all.

Re-running the rules rather than trusting the ``kept`` flag written at fetch
time is the point: it means you can tighten a threshold in the config and
rebuild the corpus from local data, with no API calls at all.

Because the fetcher already stopped on the post-filter count, every domain
arrives here with at least its target (fetch fails loudly otherwise), so the
trim is a surplus trim and never a shortfall.

Output columns follow the AI-detection corpus convention (``id`` / ``text`` /
``generated_by``), so the result drops straight into Stage B of that pipeline:

    id            -> "science:<domain>:<paperId>"  — shared by the human row and
                     every AI row generated from it, which is what the per-id
                     human-vs-AI analyses key on. The domain is in there because
                     an id is the one part of a row that survives every join and
                     every report: a bare "science:<paperId>" says which corpus
                     a row came from but not which of its domains, and anything
                     reading the domain off the id prefix (combine_domains.py
                     does, for inputs with no domain column) collapses all five
                     into one. Slugged, so an id never carries a space.
    text          -> the cleaned abstract, alone (the title is a prompt input,
                     not part of the text being classified)
    generated_by  -> "human"
    doi, title, author_ids, author_count, year, venue, citation_count, fwci,
    language, fields_of_study, domain, query  — carried along

Output: data/science/<dataset_id>_human.parquet   (one file, every domain)
        results/science_data/filter/<dataset_id>/ (drop-reason report)

One parquet, with ``domain`` as a column — not one file per domain. Downstream,
``epai.utils.corpus_utils`` selects a domain with ``where: {domain: "biology"}``,
so a per-domain file would only be a second copy of rows that are already there.

Domain names are slugged (``Computer Science`` -> ``computer_science``). The
fetch stage already slugs them via :func:`load_queries`; it is re-applied here,
where it is idempotent, so that a raw parquet fetched before that was true still
produces a corpus with one spelling of each domain.

Run (the config defaults to the example in cfg/science_data/):
    uv run python epai/science_data/filter/filter.py

    uv run python epai/science_data/filter/filter.py \
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
    DEFAULT_CFG_PATH,
    domain_slug,
    filter_config,
    human_path,
    load_pipeline_config,
    raw_path,
)
from epai.science_data.rules import (
    DROP_REASONS,
    OPTIONAL_NUMERIC_COLUMNS,
    SeenIndex,
    check_record,
    normalise_title,
)
from epai.utils.parquet_utils import coerce_dtypes, load_parquet, save_parquet

RED = "\033[91m"
RESET = "\033[0m"

RESULTS_DIR = BASE_DIR / "results" / "science_data" / "filter"

# ``id`` = NAMESPACE : domain : paperId. The namespace says which corpus a row
# came from and keeps these ids from ever colliding with a ghostbuster one; the
# domain says which slice of it. Same separator as combine_domains.py, which
# splits ids on it.
ID_NAMESPACE = "science"
ID_SEP = ":"

OUT_COLUMNS = [
    "id",
    "text",
    "generated_by",
    "doi",
    "title",
    "author_ids",
    "author_count",
    "year",
    "venue",
    "citation_count",
    "fwci",
    "language",
    "fields_of_study",
    "domain",
    "query",
    "also_matched_domains",
]


def slug_domains(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise ``domain`` (and the audit trail in ``also_matched_domains``).

    A no-op for anything fetched since :func:`load_queries` started slugging its
    keys — the slug of a slug is itself. It runs anyway because re-filtering an
    older raw parquet from local data is a supported thing to do, and a corpus
    where half the rows say ``Computer Science`` and half say
    ``computer_science`` is two domains as far as any groupby is concerned.

    Applied before the rules rather than after, so that the drop-reason report
    and the corpus name their domains the same way.
    """
    out = df.copy()
    out["domain"] = out["domain"].astype(str).map(domain_slug)
    if "also_matched_domains" in out.columns:
        out["also_matched_domains"] = (
            out["also_matched_domains"]
            .fillna("")
            .astype(str)
            .map(lambda s: ",".join(domain_slug(p) for p in s.split(",") if p))
        )

    renamed = {
        before: after
        for before, after in zip(df["domain"].astype(str), out["domain"])
        if before != after
    }
    if renamed:
        pairs = ", ".join(f"{b} -> {a}" for b, a in sorted(renamed.items()))
        print(f"Domain names slugged: {pairs}")
    return out


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

    The most useful artifact of the whole stage: it tells you what OpenAlex
    actually returned and which rule is doing the work, which is exactly what
    you need to know before trusting the corpus.
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
    parser.add_argument(
        "--cfg",
        type=Path,
        default=DEFAULT_CFG_PATH,
        help=f"pipeline config YAML (default: {DEFAULT_CFG_PATH.relative_to(BASE_DIR)})",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(args.cfg)
    raw = slug_domains(load_parquet(raw_path(cfg)))

    scored = apply_rules(raw, cfg)
    write_report(scored, cfg)

    kept = scored[scored["drop_reason"] == ""].copy()
    print(f"\n{len(kept)} admissible of {len(scored)} raw records")

    kept["id"] = (
        ID_NAMESPACE
        + ID_SEP
        + kept["domain"].astype(str)
        + ID_SEP
        + kept["paper_id"].astype(str)
    )
    kept["text"] = kept["abstract"]
    kept["generated_by"] = "human"

    # reindex rather than plain selection: a raw parquet fetched before a
    # metadata column was introduced does not carry it (`doi`, `author_ids`
    # and `author_count` were all added after the first corpora were built).
    # Filling those with null is right; failing the whole stage over a metadata
    # column that no rule reads is not. A column the rules *do* read cannot go
    # missing unnoticed — check_record above would have dropped every row for
    # it first, and `fwci`/`language` are now among them.
    final = coerce_dtypes(
        trim_to_target(kept, cfg).reindex(columns=OUT_COLUMNS),
        OPTIONAL_NUMERIC_COLUMNS,
    )

    save_parquet(final, human_path(cfg))

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
