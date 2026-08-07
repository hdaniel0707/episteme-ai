r"""Fold the per-domain parquets (essay, reuter, wp, ...) into a single file.

Three corpora in three files means every config, every script and every notebook
carries the same three paths and the same three-way loop. This script unions them
into one parquet with an explicit ``domain`` column, so a run over "all domains"
is one path and a run over one domain is a filter.

The output schema is deliberately narrow -- everything that is not one of these
is DROPPED, not carried along:

  uid           ADDED -- ``{id}:{generated_by}``, unique per row
  id            unchanged, still prefixed with the domain (``essay:1``,
                ``reut:WilliamKazer:9``); NOT unique -- it is the pair key
  domain        ADDED -- read back off the id prefix, aliased to a full name
                (``reut`` -> ``reuter``); never guessed from the file name
  text          unchanged
  generated_by  unchanged
  author        DROPPED -- folded into the id first (see below)
  binoculars_score           kept if present
  llm_judge_score*           kept if present
  emb_*                      kept if present
  --extra-cols               anything else worth keeping, named explicitly

``author`` (reuter only) is the reason this is not a plain ``pd.concat``. A column
that exists in one domain and is null in the other two is a column that any
groupby, any dropna and any "which features do I have" check has to special-case
forever. Reuter ids already carry the author (``reut:WilliamKazer:9``), so the
column is redundant there; where an id does NOT carry it, the author is spliced
into the id as the second segment before the column is dropped. Nothing is lost,
and the identity of a row stays a single string.

``id`` is not unique, and deliberately so. The ghostbuster corpora are paired:
``essay:1`` names a source prompt, and its human and machine answers both carry
it. Two keys, two jobs -- ``uid`` identifies a row, ``id`` identifies a pair --
because the pairing is worth keeping. A split that puts one member of a pair in
train and the other in test shows the model the same source content on both
sides, so anything trained on this file should group on ``id``
(``GroupKFold``, ``GroupShuffleSplit``), which is only possible while ``id``
still means what it means upstream.

The label therefore goes in ``uid`` and never in ``id``. Folding it into ``id``
would make the target part of the row identifier: everything that sorts, groups
or hashes an id would carry label information, joins back to the per-domain
parquets would break, and a row's identity would change whenever it was
relabelled. ``uid`` has the same string in it, but nothing is expected to
featurize ``uid``.

Rows are filtered on the way in: ``--drop-generated-by`` removes the labels it
names, defaulting to the prompt-variant generations (claude, gpt_prompt1,
gpt_prompt2, gpt_semantic, gpt_writing). Labels absent from a given input are
simply not there -- no error -- but a label that matched nothing in ANY input is
reported at the end, since a misspelled label is indistinguishable from a
successful no-op otherwise. Pass the flag with no values to keep everything.

``--drop-incomplete`` then removes rows with a hole where an embedding should
be -- six rows per ghostbuster corpus have neither embeddings nor scores. The
check runs per input, over the columns THAT input has, which is the only place
it can run safely: after the union, a column absent from one file is
indistinguishable from a row that failed, and requiring completeness there
would delete a whole domain instead of six rows. ``--max-drop-frac`` is the
guard against exactly that, per file.

The run ends with a statistics block: the row funnel per input, the class table
per domain, pair integrity, length by domain and class, score summaries and
embedding widths.

Partial coverage is the thing to watch and the reason for the coverage table.
Union-ing files with different columns fills the gaps with NaN, and a NaN
``llm_judge_score_gpt54_mini`` for one domain out of three is invisible until a
baseline silently trains on two thirds of the corpus. The report prints, per
column, which inputs actually had it; ``--require-full-coverage`` turns a partial
column into an error instead of a footnote.

Usage:
    # the three normalised corpora -> one file
    uv run python epai/ai_detection/prepare/combine_domains.py \
        data/parquet/essay_binox0.parquet \
        data/parquet/reuter_binox0.parquet \
        data/parquet/wp_binox0.parquet \
        -o data/parquet/ghostbuster_gpt35t.parquet

    uv run python epai/ai_detection/prepare/combine_domains.py \
        data/parquet/essay_gpt54mini_binox0.parquet \
        data/parquet/reuter_gpt54mini_binox0.parquet \
        data/parquet/wp_gpt54mini_binox0.parquet \
        -o data/parquet/ghostbuster_gpt54mini.parquet
    
    uv run python epai/ai_detection/prepare/combine_domains.py \
        data/parquet/essay_gpt54mini_binox0_red_200w_binox0.parquet \
        data/parquet/reuter_gpt54mini_binox0_red_200w_binox0.parquet \
        data/parquet/wp_gpt54mini_binox0_red_200w_binox0.parquet \
        -o data/parquet/ghostbuster_gpt54mini_200w.parquet

    # keep every generated_by label instead of the default drop list
    uv run python epai/ai_detection/prepare/combine_domains.py ... \
        -o out.parquet --drop-generated-by

    # drop a different set
    uv run python epai/ai_detection/prepare/combine_domains.py ... \
        -o out.parquet --drop-generated-by claude gpt_plain
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from epai.utils.parquet_utils import load_parquet, save_parquet

BASE_DIR = Path(__file__).parent.parent.parent.parent

ID_COL = "id"
UID_COL = "uid"
TEXT_COL = "text"
TARGET_COL = "generated_by"
DOMAIN_COL = "domain"
AUTHOR_COL = "author"

EMB_PREFIX = "emb_"
SCORE_PREFIXES = ("binoculars_score", "llm_judge_score")

# Columns every row must have; a file missing one of these is not a corpus this
# script can combine.
REQUIRED_COLS = (ID_COL, TEXT_COL, TARGET_COL)

# Optional columns, kept where present. Prefix-matched, so a second judge model
# (``llm_judge_score_<model>``) or a new embedding comes along for free.
OPTIONAL_PREFIXES = ("emb_", "binoculars_score", "llm_judge_score")

# Id prefix -> domain name. The prefix is what the prepare_*.py scripts wrote
# into the ids; the value is what the ``domain`` column should read. Unknown
# prefixes are kept verbatim (and reported), so a new corpus works without an
# edit here -- add an entry only to rename it.
DOMAIN_ALIASES = {
    "reut": "reuter",
}

# ``generated_by`` labels dropped unless --drop-generated-by says otherwise.
# These are the prompt-variant generations: keeping them turns the binary
# human-vs-machine question into a mix of generators with wildly different
# class balances per domain, which is not the corpus any of the configs expect.
# Pass --drop-generated-by with no values to keep every label.
DEFAULT_DROP_LABELS = (
    "claude",
    "gpt_prompt1",
    "gpt_prompt2",
    "gpt_semantic",
    "gpt_writing",
)

# Which columns a row must have a value in to be kept. ``emb`` is the default:
# a handful of rows in the ghostbuster corpora have no embedding (and no score
# either -- the same rows failed both), and a NaN vector is not something any
# feature block can use, so they are removed rather than carried as holes.
INCOMPLETE_MODES = ("emb", "optional", "none")

ID_SEP = ":"
# For deciding whether an id already carries its author: compare on alphanumerics
# only, so "William Kazer", "WilliamKazer" and "william_kazer" are one name.
NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------


def domain_of(row_id: str, aliases: dict[str, str]) -> str | None:
    """Domain of one id, read off the segment before the first ``:``.

    The id is the only place the domain is recorded per row; the file name is
    not, and trusting it would silently mislabel a file that already holds more
    than one domain (an earlier output of this script, for instance).
    """
    if not isinstance(row_id, str) or ID_SEP not in row_id:
        return None
    prefix = row_id.split(ID_SEP, 1)[0].strip()
    if not prefix:
        return None
    return aliases.get(prefix, prefix)


def domains_of(ids: pd.Series, aliases: dict[str, str]) -> pd.Series:
    return ids.map(lambda i: domain_of(i, aliases))


# ---------------------------------------------------------------------------
# Author -> id
# ---------------------------------------------------------------------------


def slugify(name: str) -> str:
    """``"William Kazer"`` -> ``"WilliamKazer"``, for use inside an id."""
    return "".join(ch for ch in str(name) if ch.isalnum())


def normalised(text: str) -> str:
    return NON_ALNUM_RE.sub("", str(text).lower())


def id_carries_author(row_id: str, author: str) -> bool:
    """True if any segment of the id already names this author."""
    target = normalised(author)
    if not target:
        return False
    return any(normalised(part) == target for part in str(row_id).split(ID_SEP))


def fold_author_into_id(row_id: str, author: object) -> str:
    """``reut:9`` + ``William Kazer`` -> ``reut:WilliamKazer:9``.

    A no-op when the id already carries the author, which is the common case --
    the reuter ids were built that way. Rows with no author are left alone
    rather than given an empty segment, so an id never grows a ``::``.
    """
    if author is None or (isinstance(author, float) and pd.isna(author)):
        return row_id
    slug = slugify(author)
    if not slug or id_carries_author(row_id, author):
        return row_id
    head, _, tail = str(row_id).partition(ID_SEP)
    if not tail:  # no domain prefix to insert after; append instead
        return f"{head}{ID_SEP}{slug}"
    return f"{head}{ID_SEP}{slug}{ID_SEP}{tail}"


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def is_wanted(col: str, extra: tuple[str, ...]) -> bool:
    if col in REQUIRED_COLS or col in extra:
        return True
    return any(col.startswith(p) for p in OPTIONAL_PREFIXES)


def order_columns(cols: list[str], extra: tuple[str, ...]) -> list[str]:
    """uid, id, domain, text, generated_by, scores, extras, then emb_* last.

    Embeddings last and everything else in a fixed order so the file can be
    eyeballed with ``df.iloc[:, :8]`` regardless of which inputs it came from.
    """
    front = [UID_COL, ID_COL, DOMAIN_COL, TEXT_COL, TARGET_COL]
    embeddings = sorted(c for c in cols if c.startswith(EMB_PREFIX))
    middle = [c for c in cols if c not in front and c not in embeddings]
    # scores before extras, each group in the order the inputs presented them
    scores = [c for c in middle if c not in extra]
    extras = [c for c in middle if c in extra]
    return [c for c in front if c in cols] + scores + extras + embeddings


def completeness_columns(df: pd.DataFrame, mode: str) -> list[str]:
    """Columns a row must have a value in, given --drop-incomplete.

    Only columns this file actually has are considered. That is the whole point
    of doing the check per input rather than after the union: after the union a
    column absent from one input looks exactly like a row that failed, and
    requiring completeness would delete a whole domain instead of six rows.
    """
    if mode == "none":
        return []
    cols = [c for c in df.columns if c.startswith(EMB_PREFIX)]
    if mode == "optional":
        cols += [c for c in df.columns if c.startswith(SCORE_PREFIXES)]
    return cols


def null_mask(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Rows with a missing value in any of ``cols``.

    ``notna`` is elementwise on object columns, so a list/ndarray embedding
    counts as present and only a true None/NaN counts as missing.
    """
    mask = pd.Series(False, index=df.index)
    for col in cols:
        mask |= df[col].isna()
    return mask


def embedding_dim(series: pd.Series) -> int | None:
    """Length of the first non-null vector in an ``emb_*`` column.

    Scans past holes rather than stopping at the first one, so a column whose
    first row failed still reports its width under --drop-incomplete none.
    """
    for value in series:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        try:
            return len(value)
        except TypeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------


def prepare_frame(
    path: Path,
    args: argparse.Namespace,
    aliases: dict[str, str],
    dropped_labels: Counter,
    funnel: list[dict],
) -> pd.DataFrame:
    """Load one input and cut it down to the output schema.

    ``dropped_labels`` is updated in place with the per-label row counts removed
    by --drop-generated-by, so main() can tell a label that dropped nothing
    anywhere (a typo, most likely) from one that simply was not in this file.
    ``funnel`` collects one row-count row per input for the closing statistics.
    """
    df = load_parquet(path)
    stage: dict[str, object] = {"file": path.name, "loaded": len(df)}
    funnel.append(stage)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name}: missing required column(s) {missing}.")

    df = df.copy()

    # --- unwanted generators ------------------------------------------------
    if args.drop_generated_by:
        present = Counter(df[TARGET_COL])
        hits = {lbl: present[lbl] for lbl in args.drop_generated_by if present[lbl]}
        dropped_labels.update(hits)
        if hits:
            keep_rows = ~df[TARGET_COL].isin(list(args.drop_generated_by))
            print(f"  {TARGET_COL}: dropped "
                  + ", ".join(f"{lbl}: {n}" for lbl, n in sorted(hits.items()))
                  + f"  ({len(df)} -> {int(keep_rows.sum())} rows)")
            df = df.loc[keep_rows].reset_index(drop=True)
        if df.empty:
            raise SystemExit(
                f"{path.name}: every row was dropped by --drop-generated-by "
                f"({sorted(present)} are the labels this file has)."
            )
        remaining = sorted(set(df[TARGET_COL]))
        if len(remaining) < 2:
            print(f"  WARNING  only one {TARGET_COL} label left in this file: "
                  f"{remaining} — nothing to classify unless another input "
                  "supplies the other class.")
    stage["after_labels"] = len(df)

    # --- author -> id, before anything is dropped ---------------------------
    if AUTHOR_COL in df.columns:
        before = df[ID_COL].astype(str)
        df[ID_COL] = [
            fold_author_into_id(i, a) for i, a in zip(before, df[AUTHOR_COL])
        ]
        changed = int((df[ID_COL] != before).sum())
        have_author = int(df[AUTHOR_COL].notna().sum())
        print(f"  {AUTHOR_COL!r}: {have_author} row(s) with an author; "
              f"{changed} id(s) rewritten, {have_author - changed} already carried it")

    # --- domain from the id -------------------------------------------------
    domains = domains_of(df[ID_COL].astype(str), aliases)
    unknown = int(domains.isna().sum())
    if unknown:
        sample = df.loc[domains.isna(), ID_COL].astype(str).head(5).tolist()
        raise SystemExit(
            f"{path.name}: {unknown} id(s) have no '{ID_SEP}'-separated domain "
            f"prefix, e.g. {sample}. The domain is read off the id, never off the "
            "file name — fix the ids in the prepare_*.py step that wrote this "
            "file."
        )
    df[DOMAIN_COL] = domains
    counts = Counter(domains)
    print(f"  {DOMAIN_COL}: " + ", ".join(f"{d}: {n}" for d, n in sorted(counts.items())))
    if len(counts) > 1:
        print(f"  NOTE  {path.name} already holds {len(counts)} domains; "
              "all of them are carried through.")

    # --- narrow to the output schema ---------------------------------------
    keep = [c for c in df.columns if c == DOMAIN_COL or is_wanted(c, args.extra_cols)]
    dropped = [c for c in df.columns if c not in keep]
    if dropped:
        print(f"  dropped {len(dropped)} column(s): {dropped}")
    df = df[keep].copy()

    # --- rows with holes ----------------------------------------------------
    check_cols = completeness_columns(df, args.drop_incomplete)
    if check_cols:
        holes = null_mask(df, check_cols)
        n_holes = int(holes.sum())
        if n_holes:
            per_col = {c: int(df[c].isna().sum()) for c in check_cols}
            per_col = {c: n for c, n in per_col.items() if n}
            frac = n_holes / max(len(df), 1)
            print(f"  incomplete rows: dropped {n_holes} of {len(df)} "
                  f"({100 * frac:.2f}%) with a missing value in "
                  f"{len(per_col)} of {len(check_cols)} checked column(s)")
            for col, n in sorted(per_col.items(), key=lambda kv: -kv[1])[:8]:
                print(f"    {col:<28} {n} missing")
            if frac > args.max_drop_frac:
                raise SystemExit(
                    f"{path.name}: that is more than --max-drop-frac "
                    f"({100 * args.max_drop_frac:.2f}%). A few failed rows are "
                    "expected; this many means a column was never computed for "
                    "this file.\nRe-run the embedding/score step, or pass "
                    "--drop-incomplete none to keep the holes."
                )
            df = df.loc[~holes].reset_index(drop=True)
    stage["after_incomplete"] = len(df)

    # --- uid: the unique row key -------------------------------------------
    # ``id`` stays the PAIR key -- a ghostbuster id names a source prompt, and
    # its human and machine answers share it. That is worth keeping: a split
    # that puts one member of a pair in train and the other in test is showing
    # the model the same source content on both sides. So uniqueness lives in a
    # separate column instead, and the label goes there rather than into ``id``,
    # where anything that groups or sorts by id would inherit it.
    df.insert(0, UID_COL, df[ID_COL].astype(str) + ID_SEP + df[TARGET_COL].astype(str))
    return df


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def coverage_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per optional column, which inputs had it and how many values are set."""
    all_cols: list[str] = []
    for frame in frames.values():
        for col in frame.columns:
            if col not in all_cols:
                all_cols.append(col)

    rows = []
    for col in all_cols:
        present = [name for name, f in frames.items() if col in f.columns]
        non_null = sum(int(f[col].notna().sum()) for f in frames.values() if col in f)
        total = sum(len(f) for f in frames.values())
        rows.append({
            "column": col,
            "files": f"{len(present)}/{len(frames)}",
            "rows_set": non_null,
            "rows_total": total,
            "missing_from": ", ".join(n for n in frames if n not in present),
        })
    return pd.DataFrame(rows)


def print_coverage(table: pd.DataFrame, collapse_emb: bool) -> None:
    shown = table
    if collapse_emb:
        emb = table[table["column"].str.startswith("emb_")]
        shown = table[~table["column"].str.startswith("emb_")]
        if not emb.empty:
            partial = emb[emb["rows_set"] < emb["rows_total"]]
            if partial.empty:
                print(f"  {len(emb)} emb_* column(s), all set for every row")
            else:
                worst = int((partial["rows_total"] - partial["rows_set"]).max())
                print(f"  {len(emb)} emb_* column(s), {len(partial)} of them with "
                      f"gaps (worst: {worst} row(s) of "
                      f"{int(partial['rows_total'].iloc[0])}): "
                      f"{partial['column'].tolist()}")
    width = max([len(c) for c in shown["column"]] + [10])
    print(f"  {'column':<{width}}  {'files':>7}  {'rows set':>12}  missing from")
    print("  " + "-" * (width + 40))
    for _, r in shown.iterrows():
        gap = "" if r["rows_set"] == r["rows_total"] else r["missing_from"]
        print(f"  {r['column']:<{width}}  {r['files']:>7}  "
              f"{r['rows_set']:>6}/{r['rows_total']:<5}  {gap}")


def print_dim_check(df: pd.DataFrame) -> None:
    """Embedding dimensions in the combined frame, one line per width."""
    dims: dict[str, int | None] = {}
    for col in sorted(c for c in df.columns if c.startswith(EMB_PREFIX)):
        dims[col] = embedding_dim(df[col])
    if not dims:
        return
    unknown = [c for c, d in dims.items() if d is None]
    print("\nEmbedding widths")
    for col, dim in dims.items():
        if dim is not None:
            print(f"  {col:<24} {dim}")
    if unknown:
        print(f"  WARNING  no non-null vector to measure in: {unknown}")


# ---------------------------------------------------------------------------
# Closing statistics
# ---------------------------------------------------------------------------


def print_funnel(funnel: list[dict]) -> None:
    """Rows in, rows out, and where the difference went."""
    print("\nRows per input")
    header = f"  {'file':<44}{'loaded':>9}{'-labels':>9}{'-holes':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for stage in funnel:
        print(f"  {str(stage['file'])[:44]:<44}{stage['loaded']:>9}"
              f"{stage['after_labels']:>9}{stage['after_incomplete']:>9}")
    totals = [sum(int(s[k]) for s in funnel)
              for k in ("loaded", "after_labels", "after_incomplete")]
    print(f"  {'TOTAL':<44}{totals[0]:>9}{totals[1]:>9}{totals[2]:>9}")


def print_class_table(df: pd.DataFrame) -> None:
    """Rows per domain x generated_by, with the machine share per domain.

    The share is the number to look at: a domain whose balance differs from the
    others' turns "which domain is this" into a partial answer to "is this
    machine", and any cross-domain result then has a shortcut available to it.
    """
    table = (df.groupby([DOMAIN_COL, TARGET_COL]).size()
             .unstack(fill_value=0).sort_index())
    labels = list(table.columns)
    width = max([len(str(d)) for d in table.index] + [8])
    print("\nRows per domain x " + TARGET_COL)
    head = f"  {'domain':<{width}}" + "".join(f"{str(l)[:14]:>15}" for l in labels)
    print(head + f"{'total':>10}{'machine %':>11}")
    print("  " + "-" * (len(head) + 19))
    for domain, row in table.iterrows():
        total = int(row.sum())
        machine = total - int(row.get("human", 0))
        print(f"  {str(domain):<{width}}"
              + "".join(f"{int(row[l]):>15}" for l in labels)
              + f"{total:>10}{100 * machine / max(total, 1):>10.1f}%")
    grand = int(table.values.sum())
    machine_all = grand - int(table["human"].sum()) if "human" in table else grand
    print(f"  {'ALL':<{width}}"
          + "".join(f"{int(table[l].sum()):>15}" for l in labels)
          + f"{grand:>10}{100 * machine_all / max(grand, 1):>10.1f}%")


def print_pair_report(df: pd.DataFrame) -> None:
    """How many ids carry a full set of labels, and how many are orphans.

    ``id`` is the pair key, so this is the check that the pairing survived the
    label filter and the incomplete-row drop: an id left with only its human
    row is a source prompt whose machine answer was dropped, and it contributes
    an unmatched example to whatever is trained next.
    """
    per_id = df.groupby(ID_COL)[TARGET_COL].nunique()
    sizes = Counter(per_id.values.tolist())
    n_labels = df[TARGET_COL].nunique()
    print("\nPairing (id = pair key, uid = row key)")
    print(f"  distinct ids            : {len(per_id)}")
    print(f"  distinct {TARGET_COL:<15}: {n_labels}  ({sorted(df[TARGET_COL].unique())})")
    for size in sorted(sizes):
        note = "  <- complete" if size == n_labels else ""
        print(f"  ids with {size} label(s)     : {sizes[size]}{note}")
    orphans = per_id[per_id < n_labels]
    if len(orphans):
        sample = orphans.index[:5].tolist()
        print(f"  NOTE  {len(orphans)} id(s) do not carry every label, e.g. {sample}. "
              "Expected if the\n        domains use different generators; a sign of "
              "an over-eager filter otherwise.")
    print("  Split on 'id' (GroupKFold/GroupShuffleSplit), not on 'uid' — the two "
          "halves of\n  a pair share a source prompt, and straddling a fold leaks it "
          "across train/test.")


def print_score_stats(df: pd.DataFrame) -> None:
    """Mean/std/range of the scalar scores, per domain, and any NaNs left."""
    cols = [c for c in df.columns if c.startswith(SCORE_PREFIXES)]
    if not cols:
        return
    print("\nScores")
    for col in cols:
        nan = int(df[col].isna().sum())
        print(f"  {col}  (NaN: {nan})")
        stats = df.groupby(DOMAIN_COL)[col].agg(["mean", "std", "min", "max"])
        for domain, row in stats.iterrows():
            print(f"    {str(domain):<10} mean {row['mean']:>8.3f}   "
                  f"sd {row['std']:>7.3f}   [{row['min']:.3f}, {row['max']:.3f}]")


def print_length_stats(df: pd.DataFrame) -> None:
    """Document length per domain and class, in words.

    Length is the confound that keeps coming back: if machine texts are
    systematically longer in one domain and shorter in another, a classifier
    can score well in-domain on length alone and then invert on transfer.
    """
    words = df[TEXT_COL].fillna("").astype(str).str.split().str.len()
    tmp = pd.DataFrame({DOMAIN_COL: df[DOMAIN_COL], TARGET_COL: df[TARGET_COL],
                        "words": words})
    table = tmp.groupby([DOMAIN_COL, TARGET_COL])["words"].mean().unstack()
    print("\nMean document length (words)")
    labels = list(table.columns)
    width = max([len(str(d)) for d in table.index] + [8])
    head = f"  {'domain':<{width}}" + "".join(f"{str(l)[:14]:>15}" for l in labels)
    gap = "human" in table.columns and len(labels) == 2
    print(head + (f"{'gap':>10}" if gap else ""))
    print("  " + "-" * (len(head) + 10))
    for domain, row in table.iterrows():
        line = f"  {str(domain):<{width}}" + "".join(
            f"{row[l]:>15.0f}" if pd.notna(row[l]) else f"{'-':>15}" for l in labels)
        if gap:
            other = [l for l in labels if l != "human"][0]
            line += f"{row[other] - row['human']:>+10.0f}"
        print(line)
    if gap:
        print("  gap = machine - human. A gap that differs in SIGN across domains is "
              "the\n  cheapest available explanation for good in-domain and poor "
              "transfer scores.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_aliases(pairs: list[str]) -> dict[str, str]:
    aliases = dict(DOMAIN_ALIASES)
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--domain-alias expects prefix=name, got {pair!r}.")
        prefix, name = pair.split("=", 1)
        aliases[prefix.strip()] = name.strip()
    return aliases


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parquets", nargs="+", type=Path,
                    help="Per-domain parquet files to combine.")
    ap.add_argument("-o", "--out", type=Path, required=True,
                    help="Output parquet path.")
    ap.add_argument("--extra-cols", nargs="*", default=[], metavar="COL",
                    help="Additional column names to keep (default: only id, "
                    "text, generated_by, binoculars_score, llm_judge_score*, "
                    "emb_*).")
    ap.add_argument("--drop-generated-by", nargs="*", metavar="LABEL",
                    default=list(DEFAULT_DROP_LABELS),
                    help="Drop rows whose generated_by is one of these, where "
                    "they exist (default: " + " ".join(DEFAULT_DROP_LABELS) +
                    "). Pass the flag with no values to keep every label.")
    ap.add_argument("--domain-alias", nargs="*", default=[], metavar="PREFIX=NAME",
                    help="Map an id prefix to a domain name, e.g. reut=reuter. "
                    "Unmapped prefixes are used as-is.")
    ap.add_argument("--require-full-coverage", action="store_true",
                    help="Fail if any kept column is missing from some input, "
                    "instead of reporting it and filling with NaN.")
    ap.add_argument("--drop-incomplete", choices=INCOMPLETE_MODES, default="emb",
                    help="Drop rows with a missing value in: 'emb' the emb_* "
                    "columns their own input file has (default), 'optional' "
                    "those plus the binoculars/llm-judge scores, 'none' keep "
                    "every row.")
    ap.add_argument("--max-drop-frac", type=float, default=0.05,
                    help="Fail if --drop-incomplete would remove more than this "
                    "fraction of any input (default: 0.05). A larger share means "
                    "a column was never computed, not that a few rows failed.")
    ap.add_argument("--drop-duplicate-rows", action="store_true",
                    help=f"Keep the first row of each duplicated "
                    f"({ID_COL}, {TARGET_COL}) pair instead of failing. Note that "
                    f"a duplicated {ID_COL} alone is normal — it is the pair key.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite the output file if it already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be written; write nothing.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.extra_cols = tuple(args.extra_cols)
    args.drop_generated_by = tuple(dict.fromkeys(args.drop_generated_by))
    aliases = parse_aliases(args.domain_alias)

    if args.out.exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"{args.out} exists — pass --overwrite to replace it.")
    if len(set(p.resolve() for p in args.parquets)) != len(args.parquets):
        raise SystemExit("The same input file was given more than once.")

    print(f"combine_domains.py  —  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 88)
    print(f"Output     : {args.out}"
          f"{'   (DRY RUN — not written)' if args.dry_run else ''}")
    drop_note = (", ".join(args.drop_generated_by)
                 or "(none — every generated_by label is kept)")
    print(f"Drop labels: {drop_note}")
    print("=" * 88)

    frames: dict[str, pd.DataFrame] = {}
    dropped_labels: Counter = Counter()
    funnel: list[dict] = []
    for path in args.parquets:
        if not path.is_file():
            raise SystemExit(f"Not found: {path}")
        print(f"\n{path}")
        frames[path.name] = prepare_frame(path, args, aliases, dropped_labels, funnel)

    never_hit = [lbl for lbl in args.drop_generated_by if not dropped_labels[lbl]]
    if never_hit:
        print(f"\nNOTE  --drop-generated-by label(s) {never_hit} matched no row in "
              "any input. Harmless if those generators are simply absent here, but "
              "check for a\n      typo — a misspelled label drops nothing and says "
              "nothing.")

    # --- coverage across inputs --------------------------------------------
    table = coverage_table(frames)
    print("\nColumn coverage")
    print_coverage(table, collapse_emb=True)
    partial = table[table["rows_set"] < table["rows_total"]]
    if not partial.empty:
        message = (f"{len(partial)} column(s) are not set for every row; those "
                   "cells become NaN. A baseline or feature block reading one of "
                   "them\n  silently trains on a subset of the corpus — check the "
                   "'missing from' column above before using it.")
        if args.require_full_coverage:
            raise SystemExit(f"\n--require-full-coverage: {message}")
        print(f"\n  NOTE  {message}")

    # --- union --------------------------------------------------------------
    columns: list[str] = []
    for frame in frames.values():
        for col in frame.columns:
            if col not in columns:
                columns.append(col)
    combined = pd.concat(frames.values(), ignore_index=True)
    combined = combined.reindex(columns=order_columns(columns, args.extra_cols))

    # --- row identity -------------------------------------------------------
    # A repeated ``id`` is expected: it is the pair key, shared by the human and
    # machine answers to one source prompt. A repeated (id, generated_by) is not
    # -- that is the same row twice, and the uid would collide.
    key = [ID_COL, TARGET_COL]
    duplicated = combined.duplicated(subset=key, keep=False)
    if duplicated.any():
        sample = combined.loc[duplicated, UID_COL].head(5).tolist()
        if not args.drop_duplicate_rows:
            raise SystemExit(
                f"\n{int(duplicated.sum())} row(s) share a ({ID_COL}, {TARGET_COL}) "
                f"key across the inputs, e.g. {sample}. That is the same row twice, "
                f"not a pair,\nand it would give two rows the same {UID_COL}: check "
                "the inputs are not variants of the same corpus, or pass "
                "--drop-duplicate-rows to keep the first of each."
            )
        before = len(combined)
        combined = combined.drop_duplicates(subset=key, keep="first")
        combined = combined.reset_index(drop=True)
        print(f"\n--drop-duplicate-rows: {before} -> {len(combined)} rows")

    # --- statistics ---------------------------------------------------------
    print("\n" + "=" * 88)
    print(f"STATISTICS  —  {len(combined)} rows, {len(combined.columns)} columns")
    print("=" * 88)
    print_funnel(funnel)
    print_class_table(combined)
    print_pair_report(combined)
    print_length_stats(combined)
    print_score_stats(combined)
    print_dim_check(combined)

    if args.dry_run:
        print(f"\nColumns: {list(combined.columns)}")
        print("\nDry run — nothing written.\n")
        return

    print()
    save_parquet(combined, args.out)
    print()


if __name__ == "__main__":
    main()
