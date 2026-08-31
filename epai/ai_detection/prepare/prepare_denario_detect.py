r"""Bridge the Denario papers into the ai_detection schema, for eval-only use.

``prepare_denario.py`` writes the *citation_impact* schema -- ``paper_id`` /
``abstract`` / ``domains`` / ``emb_specter2`` -- which is the shape that side of
the project reads. Nothing in ``epai/ai_detection`` can read it: the training
configs, ``cd_core`` and every scoring script want ``id`` / ``text`` /
``generated_by``, plus ``domain`` and ``uid`` where ``combine_domains.py`` has
been. This script is that rename, and nothing else.

    paper_id   -> id      the group key. One paper per group here: unlike the
                          Ghostbuster corpora there is no human counterpart to
                          share an id with, so every group has exactly one row.
    <id>:<gen> -> uid     row identity, same convention combine_domains.py uses
    abstract   -> text    the column every detector reads
    domains[0] -> domain  slugified; the Denario categories are prose
                          ("Quantum physics + Cosmology")
    generated_by          carried through as "denario"
    title, year           carried for reading the exported .txt files back

WHAT THIS IS FOR, AND WHAT IT IS NOT
------------------------------------
The output is an EVAL-ONLY corpus (``data.eval_only`` in the training configs;
see ``epai/utils/holdout_core.py``). It holds machine text only -- Denario
papers are LLM-written end to end -- so nothing can be trained on it, nothing
can be split on it, and it is an error as a ``datasets`` entry. It exists to be
scored by detectors that were calibrated somewhere else.

This script does NOT score anything, and the file it writes is not usable until
the rest of the chain has run on it. Every column the training config reads has
to be present or the preflight refuses to start -- correctly, because a Denario
figure covering three detectors out of twelve is worse than a run that stops.
In order:

    uv run python epai/ai_detection/prepare/prepare_denario_detect.py
    uv run python epai/ai_detection/prepare/standardize_text.py \
        data/parquet/denario_detect.parquet
    uv run python epai/ai_detection/embed/add_embeddings.py \
        data/parquet/denario_detect.parquet --gpu 0
    uv run python epai/ai_detection/score/run_binox_pairs.py \
        data/parquet/denario_detect.parquet --gpu 0
    uv run python epai/ai_detection/score/run_fastdetect_pairs.py \
        data/parquet/denario_detect.parquet --gpu 0
    uv run python epai/ai_detection/score/add_detector_scores.py \
        data/parquet/denario_detect.parquet --gpu 0
    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/denario_detect.parquet --model gpt56luna
    uv run python epai/ai_detection/score/add_llm_judge_reasons.py \
        data/parquet/denario_detect.parquet --model gpt56luna
    uv run python epai/ai_detection/score/add_cue_features.py \
        data/parquet/denario_detect.parquet

``standardize_text.py`` first, and before the scoring: it rewrites ``text``, and
a Binoculars number computed on the pre-standardised text is a number about a
document that is no longer in the corpus. The training corpora went through the
same stage, so skipping it here would make the standardisation itself a
difference between the two.

LENGTH IS THE CONFOUND TO CHECK
--------------------------------
The Ghostbuster corpora were cut to roughly 200 words. A Denario abstract is not
necessarily that long, and a detection rate measured on documents of a different
length is not comparable to a transfer cell measured on 200-word ones -- length
would be doing the work that genre is being blamed for. ``--report-lengths``
prints the distribution so the comparison can be made before anything is scored;
``epai/ai_detection/prepare/truncate_text.py`` is what fixes it if it needs
fixing.

Run:
    uv run python epai/ai_detection/prepare/prepare_denario_detect.py
    uv run python epai/ai_detection/prepare/prepare_denario_detect.py \
        --in data/parquet/denario_papers.parquet \
        -o data/parquet/denario_detect.parquet --report-lengths
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from epai.utils.parquet_utils import load_parquet, save_parquet

BASE_DIR = Path(__file__).parent.parent.parent.parent

DEFAULT_IN = BASE_DIR / "data" / "parquet" / "denario_papers.parquet"
DEFAULT_OUT = BASE_DIR / "data" / "parquet" / "denario_detect.parquet"

#: Every Denario row is machine-written, so this is the whole label column. It
#: must not equal the configs' `human_label` ("human") -- that is what makes the
#: binarised target come out all-machine, which is the point of the corpus.
GENERATED_BY = "denario"

#: Written when a row has no usable category. A single bucket rather than a
#: guess: `domain` is only ever used to describe or slice this corpus, never as
#: a feature, so an honest "unknown" costs nothing and an invented label would
#: put rows in a group they do not belong to.
UNKNOWN_DOMAIN = "unknown"

#: The output schema, in order. Deliberately narrow, the same discipline
#: combine_domains.py applies: a column carried here for no reason is a column
#: every later stage has to decide what to do with.
OUT_COLUMNS = ["uid", "id", "domain", "text", "generated_by", "title", "year"]


def slugify_domain(value: object) -> str:
    """``"Quantum physics + Cosmology"`` -> ``"quantum_physics_cosmology"``.

    The Denario categories are prose with spaces and a ``+`` in them, and
    ``domain`` ends up in filenames, ``--where`` conditions and figure labels.
    Lowercased and reduced to ``[a-z0-9_]`` so it can be typed on a command line
    without quoting.
    """
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) else None
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return UNKNOWN_DOMAIN
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return slug or UNKNOWN_DOMAIN


def to_detection_schema(raw: pd.DataFrame) -> pd.DataFrame:
    """The citation_impact frame as an ai_detection corpus. A rename, no more."""
    missing = [c for c in ("paper_id", "abstract") if c not in raw.columns]
    if missing:
        raise KeyError(
            f"input parquet is missing {missing}; expected the output of "
            f"epai/ai_detection/prepare/prepare_denario.py "
            f"(columns: {sorted(raw.columns)})"
        )

    out = pd.DataFrame(index=range(len(raw)))
    out["id"] = raw["paper_id"].astype("string")
    out["text"] = raw["abstract"].astype("string")
    out["generated_by"] = GENERATED_BY
    # Same shape combine_domains.py writes: the label goes in `uid` and never in
    # `id`, so the group key stays free of the target.
    out["uid"] = out["id"] + f":{GENERATED_BY}"
    out["domain"] = [slugify_domain(v) for v in raw.get("domains", pd.Series([None] * len(raw)))]
    out["title"] = raw.get("title", pd.Series([None] * len(raw))).astype("string")
    out["year"] = raw.get("year", pd.Series([pd.NA] * len(raw)))
    return out[OUT_COLUMNS]


def drop_unusable(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows with no text. Reported, never silent.

    A row with an empty abstract would reach a detector as an empty string and
    come back with a score that means nothing, and on a corpus of 82 a handful
    of those moves every rate by points.
    """
    text = df["text"].fillna("").str.strip()
    keep = text.str.len() > 0
    return df[keep].reset_index(drop=True), int((~keep).sum())


def report_lengths(df: pd.DataFrame) -> None:
    """Word-count distribution, per domain and overall. See the module docstring."""
    words = df["text"].fillna("").str.split().str.len()
    print("\nword counts — compare these with the corpora this will be scored beside")
    print(f"{'domain':<32}{'n':>6}{'min':>8}{'median':>9}{'mean':>8}{'max':>8}")
    print("-" * 71)
    for domain, idx in df.groupby("domain").groups.items():
        w = words.loc[idx]
        print(
            f"{str(domain):<32}{len(w):>6}{int(w.min()):>8}{int(w.median()):>9}"
            f"{w.mean():>8.0f}{int(w.max()):>8}"
        )
    print("-" * 71)
    print(
        f"{'ALL':<32}{len(words):>6}{int(words.min()):>8}{int(words.median()):>9}"
        f"{words.mean():>8.0f}{int(words.max()):>8}"
    )
    print(
        "\nThe ghostbuster corpora were cut to ~200 words. If these are far off,\n"
        "truncate before scoring (epai/ai_detection/prepare/truncate_text.py) or\n"
        "length becomes the confound instead of genre."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in", dest="src", type=Path, default=DEFAULT_IN,
                   help="Denario parquet in the citation_impact schema.")
    p.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT,
                   help="Output parquet, in the ai_detection schema.")
    p.add_argument("--report-lengths", action="store_true",
                   help="Print the word-count distribution before writing.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be written, write nothing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_parquet(args.src)

    df = to_detection_schema(raw)
    df, dropped = drop_unusable(df)
    if dropped:
        print(f"Dropped {dropped} row(s) with no text.")

    dupes = int(df["id"].duplicated().sum())
    if dupes:
        # `id` is the group key, so duplicates would silently merge two papers
        # into one group. Worth saying rather than deduplicating on a guess.
        print(
            f"WARNING: {dupes} duplicate id(s) — those papers share a group and "
            f"will be scored as one. Check the source db's pdf_url column."
        )

    print(f"\n{len(df)} rows, {df['id'].nunique()} groups")
    print(f"  generated_by : {df['generated_by'].value_counts().to_dict()}")
    print(f"  domain       : {df['domain'].value_counts().to_dict()}")
    print(
        "\nAll machine, by construction — this is an EVAL-ONLY corpus. Name it "
        "under\n`data.eval_only` in a training config, never under `data.datasets`."
    )

    if args.report_lengths:
        report_lengths(df)

    if args.dry_run:
        print(f"\nDry run — nothing written to {args.out}")
        return

    save_parquet(df, args.out)
    print(
        "\nNot usable yet: standardize, embed and score it before any config "
        "reads it.\nSee the run block at the top of this file for the order."
    )


if __name__ == "__main__":
    main()
