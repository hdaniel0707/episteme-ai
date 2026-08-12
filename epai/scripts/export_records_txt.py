"""Write a corpus parquet out as one readable .txt per record, for eyeballing.

A pair is unreadable in a dataframe: the two texts are 200-word strings in one
cell each, the prompt that produced one of them is longer still, and
``df.head()`` shows none of it. This script turns each ``id`` — the pair key,
shared by a paper's human row and every generated row derived from it — into a
single file holding

    the original (human) text, the prompt, then each AI text,

separated by labelled rules and wrapped to a readable width, so that judging
whether a generated abstract is any good is a matter of opening a file.

Nothing here is science_data-specific: the column names are all arguments, and a
parquet with no ``id`` column is exported one file per row. The defaults match
the corpora this repo writes (``id`` / ``text`` / ``prompt`` / ``generated_by``,
human rows labelled ``human``).

The output directory is only ever written to, never cleaned: an existing file of
the same name is overwritten, but files from a previous export of a different
corpus are left where they are, and reported.

Run:
    uv run python epai/scripts/export_records_txt.py data/parquet/science_v1_gpt56luna_0807A.parquet
    uv run python epai/scripts/export_records_txt.py corpus.parquet --sample 25 --seed 0
    uv run python epai/scripts/export_records_txt.py corpus.parquet --where domain=biology --limit 10
    uv run python epai/scripts/export_records_txt.py corpus.parquet -o /tmp/look --wrap 0
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent

DEFAULT_OUT_ROOT = BASE_DIR / "data" / "inspect"

# Same column names the ai_detection and science_data corpora use. They are
# arguments rather than constants because this script is also the quickest way
# to look at a parquet that predates those conventions.
DEFAULT_ID_COL = "id"
DEFAULT_TEXT_COL = "text"
DEFAULT_PROMPT_COL = "prompt"
DEFAULT_LABEL_COL = "generated_by"
DEFAULT_HUMAN_LABEL = "human"

# Metadata printed in each file's header, in this order, when the column is
# present and the value is not null. Deliberately short: the header is there to
# identify the paper, and anything longer pushes the actual text off the screen.
HEADER_FIELDS = [
    "uid",
    "domain",
    "title",
    "year",
    "venue",
    "doi",
    "language",
    "citation_count",
    "fwci",
    "query",
]

WIDTH = 80
RULE = "=" * WIDTH
THIN = "-" * WIDTH


def is_blank(value: object) -> bool:
    """True for null, and for a string that is empty or only whitespace."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        # Arrays and lists (e.g. `fields_of_study`) raise rather than answer;
        # they are not blank.
        return False
    return isinstance(value, str) and not value.strip()


def as_text(value: object) -> str:
    """A cell as a plain string, with the null cases collapsed to ``""``."""
    return "" if is_blank(value) else str(value)


def wrap(text: str, width: int) -> str:
    """Break lines longer than *width*; ``width <= 0`` leaves the text alone.

    Line by line, NOT paragraph by paragraph. The two things this prints want
    opposite treatment and this is the rule that suits both: an abstract is one
    very long line and gets reflowed, while a prompt is a structured block whose
    own newlines and indentation are most of its readability — reflowing it into
    paragraphs would run its labelled lines together.
    """
    text = text.strip()
    if width <= 0 or not text:
        return text
    out: list[str] = []
    for line in text.splitlines():
        line = line.rstrip()
        if len(line) <= width:
            out.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        out.extend(
            textwrap.fill(
                line.strip(),
                width=width,
                initial_indent=indent,
                subsequent_indent=indent,
                # A DOI or a long token is a thing to copy, so it is better
                # over-long than split in half.
                break_long_words=False,
                break_on_hyphens=False,
            ).splitlines()
        )
    return "\n".join(out)


def words(text: str) -> int:
    return len(text.split())


def section(title: str, detail: str = "") -> str:
    """A labelled rule: ``--- ORIGINAL (human) ------------------- 213 words``."""
    left = f"{THIN[:3]} {title} "
    if detail:
        right = f" {detail} {THIN[:3]}"
        pad = max(WIDTH - len(left) - len(right), 0)
        return f"{left}{'-' * pad}{right}"
    return f"{left}{'-' * max(WIDTH - len(left), 0)}"


def slugify(value: str, max_len: int = 60) -> str:
    """A filesystem-safe fragment of an id.

    The ids in this repo look like ``science:biology:W2143845670``. Colons are
    legal in a POSIX filename but confuse shell completion and scp-style
    ``host:path`` parsing, and Windows entirely — the same reasoning that turns
    the ``:`` in a ``generated_by`` label into ``_`` in a filename.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug[:max_len] or "record"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def parse_where(pairs: list[str]) -> dict[str, str]:
    """``["domain=biology", "year=2019"]`` -> ``{"domain": "biology", ...}``."""
    conditions: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--where expects COLUMN=VALUE, got {pair!r}")
        col, value = pair.split("=", 1)
        col = col.strip()
        if not col:
            raise ValueError(f"--where expects COLUMN=VALUE, got {pair!r}")
        conditions[col] = value
    return conditions


def apply_where(df: pd.DataFrame, conditions: dict[str, str]) -> pd.DataFrame:
    """Keep the rows matching every condition, comparing as strings.

    String comparison so ``--where year=2019`` matches an int column, the same
    rule ``corpus_utils`` applies to a ``where:`` block in a config.
    """
    for col, value in conditions.items():
        if col not in df.columns:
            raise ValueError(
                f"--where column {col!r} is not in the parquet. "
                f"Available: {list(df.columns)}"
            )
        keep = df[col].astype("string") == value
        if not keep.any():
            present = sorted(df[col].astype("string").dropna().unique())[:10]
            raise ValueError(
                f"--where {col}={value!r} matches no rows. "
                f"{col} values include: {present}"
            )
        df = df[keep.fillna(False)]
    return df


def select_groups(keys: list, args: argparse.Namespace) -> list:
    """Cut the group keys down to the ones asked for, in a stable order.

    ``--sample`` is drawn from the groups that survived ``--where`` / ``--ids``,
    not from the file, so sampling within a domain is ``--where domain=x
    --sample 20`` and means what it looks like.
    """
    if args.sample:
        if args.sample < len(keys):
            rng = pd.Series(keys).sample(n=args.sample, random_state=args.seed)
            # Back to corpus order: a sample is a subset to read, and reading it
            # in the file's own order keeps neighbouring papers together.
            chosen = set(rng)
            keys = [k for k in keys if k in chosen]
    if args.limit:
        keys = keys[: args.limit]
    return keys


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_header(key: object, rows: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    lines = [RULE, f"  {key}", RULE]
    first = rows.iloc[0]
    for field in HEADER_FIELDS:
        if field == "uid" and len(rows) > 1:
            # One uid per row, so it identifies a row and not the group; it is
            # printed per section instead.
            continue
        if field in rows.columns and not is_blank(first[field]):
            lines.append(f"{field:<16}{as_text(first[field])}")
    labels = (
        [as_text(v) for v in rows[args.label_column]]
        if args.label_column in rows.columns
        else []
    )
    if labels:
        lines.append(f"{'rows':<16}{', '.join(labels)}")
    lines.append("")
    return lines


def render_text_block(
    title: str, text: str, args: argparse.Namespace, note: str = ""
) -> list[str]:
    """A labelled rule, an optional one-line note, then the text."""
    head = [section(title, f"{words(text)} words" if text else "empty")]
    if note:
        head.append(f"# {note}")
    body = wrap(text, args.wrap) if text else "(no text in this row)"
    return head + ["", body, ""]


def render_group(key: object, rows: pd.DataFrame, args: argparse.Namespace) -> str:
    """One record as text: original, prompt, then each generated text."""
    label_col, text_col, prompt_col = args.label_column, args.text_column, args.prompt_column
    has_labels = label_col in rows.columns

    if has_labels:
        # fillna(False): a row with no label is not the original, and pandas
        # refuses to mask with a boolean column that still holds NA anyway.
        is_human = (rows[label_col].astype("string") == args.human_label).fillna(False)
        human_rows, ai_rows = rows[is_human], rows[~is_human]
    else:
        # No label column: nothing distinguishes the rows, so they are all just
        # "text" and the file is a dump of the group rather than a pair.
        human_rows, ai_rows = rows.iloc[0:0], rows

    lines = render_header(key, rows, args)

    if len(human_rows):
        for _, row in human_rows.iterrows():
            lines += render_text_block(
                f"ORIGINAL ({as_text(row[label_col]) if has_labels else 'row'})",
                as_text(row[text_col]),
                args,
                note=as_text(row["uid"]) if "uid" in rows.columns else "",
            )
    elif has_labels:
        lines += [section("ORIGINAL", "missing"), "", "(no human row for this id)", ""]

    # The prompt is per-paper in this pipeline — every generator sees the same
    # one — so it is printed once, between the original and the generations it
    # produced. If a corpus ever varies it per generator, the block below is
    # skipped and each AI section carries its own.
    ai_prompts = (
        [as_text(v) for v in ai_rows[prompt_col]] if prompt_col in rows.columns else []
    )
    shared_prompt = ""
    distinct = {p for p in ai_prompts if p}
    if len(distinct) == 1:
        shared_prompt = distinct.pop()
        lines += render_text_block("PROMPT", shared_prompt, args)
    elif prompt_col in rows.columns and not distinct and len(ai_rows):
        lines += [section("PROMPT", "missing"), "", "(no prompt recorded)", ""]

    for _, row in ai_rows.iterrows():
        title = f"AI  {as_text(row[label_col])}" if has_labels else "TEXT"
        if not shared_prompt and prompt_col in rows.columns:
            prompt = as_text(row[prompt_col])
            if prompt:
                lines += render_text_block(f"PROMPT for {title}", prompt, args)
        uid = as_text(row["uid"]) if "uid" in rows.columns else ""
        lines += render_text_block(title, as_text(row[text_col]), args, note=uid)

    lines.append(RULE)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a corpus parquet as one readable .txt per record: the "
            "original text, the prompt, and each AI text, separated."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("parquet_file", help="Path to the Parquet file.")
    parser.add_argument(
        "-o",
        "--out-dir",
        default=None,
        help="Directory for the .txt files "
        f"(default: {DEFAULT_OUT_ROOT.relative_to(BASE_DIR)}/<parquet stem>/).",
    )

    g = parser.add_argument_group("which records")
    g.add_argument("--ids", nargs="+", metavar="ID", default=None,
                   help="Export only these id values.")
    g.add_argument("--where", nargs="+", metavar="COL=VALUE", default=None,
                   help="Keep only rows matching every condition, e.g. domain=biology.")
    g.add_argument("--sample", type=int, default=0,
                   help="Export a random sample of this many records (0 = all).")
    g.add_argument("--seed", type=int, default=0,
                   help="Seed for --sample.")
    g.add_argument("--limit", type=int, default=0,
                   help="Export at most this many records, after any sampling (0 = all).")

    g = parser.add_argument_group("columns")
    g.add_argument("--id-column", default=DEFAULT_ID_COL,
                   help="Pair key: rows sharing it become one file.")
    g.add_argument("--text-column", default=DEFAULT_TEXT_COL)
    g.add_argument("--prompt-column", default=DEFAULT_PROMPT_COL)
    g.add_argument("--label-column", default=DEFAULT_LABEL_COL)
    g.add_argument("--human-label", default=DEFAULT_HUMAN_LABEL,
                   help="Value of the label column marking the original text.")

    g = parser.add_argument_group("formatting")
    g.add_argument("--wrap", type=int, default=WIDTH,
                   help="Wrap text to this many columns (0 = leave single-line).")
    g.add_argument("--no-index", action="store_true",
                   help="Do not write the _index.tsv summary.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    path = Path(args.parquet_file)
    if not path.is_file():
        print(f"Parquet file not found: {path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(path)
    if args.text_column not in df.columns:
        print(
            f"Text column {args.text_column!r} not found in {path}.\n"
            f"Available columns: {list(df.columns)}",
            file=sys.stderr,
        )
        return 2
    print(f"Loaded {len(df):,} rows from {path}")

    if args.prompt_column not in df.columns:
        print(
            f"NOTE: no {args.prompt_column!r} column in this parquet, so the "
            "files carry no prompt. A corpus generated before generate_ai.py "
            "started writing it can be rebuilt from its "
            "<dataset_id>_prompts_<prompt_config>.parquet — the generate stage "
            "is checkpointed, "
            "so re-running it costs no API calls."
        )

    try:
        if args.where:
            df = apply_where(df, parse_where(args.where))
            print(f"  --where kept {len(df):,} rows")
        if args.ids:
            if args.id_column not in df.columns:
                raise ValueError(f"--ids needs an {args.id_column!r} column")
            wanted = set(args.ids)
            missing = wanted - set(df[args.id_column].astype("string"))
            if missing:
                raise ValueError(f"--ids not present in the parquet: {sorted(missing)}")
            df = df[df[args.id_column].astype("string").isin(wanted)]
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if df.empty:
        print("No rows selected. Nothing written.")
        return 0

    if args.id_column in df.columns:
        blank_ids = df[args.id_column].isna().sum()
        if blank_ids:
            # groupby would drop these silently, and a row that vanishes between
            # the parquet and the directory is the one thing an inspection tool
            # must not do quietly.
            print(f"NOTE: {blank_ids:,} row(s) have no {args.id_column} and are skipped.")
            df = df[df[args.id_column].notna()]
        # sort=False: groups come out in the order the corpus presents them, so
        # a --limit reads the head of the file rather than an alphabetical slice
        # of it.
        groups = {key: rows for key, rows in df.groupby(args.id_column, sort=False)}
    else:
        print(
            f"NOTE: no {args.id_column!r} column, so each ROW becomes a file "
            "rather than each pair."
        )
        groups = {i: df.loc[[i]] for i in df.index}

    keys = select_groups(list(groups), args)
    if not keys:
        print("No records selected. Nothing written.")
        return 0

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_ROOT / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in out_dir.glob("*.txt")}

    index_rows: list[str] = []
    written: list[Path] = []
    for n, key in enumerate(keys, start=1):
        rows = groups[key]
        domain = as_text(rows.iloc[0]["domain"]) if "domain" in rows.columns else ""
        name = f"{n:04d}_{slugify(domain) + '_' if domain else ''}{slugify(str(key))}.txt"
        target = out_dir / name
        target.write_text(render_group(key, rows, args), encoding="utf-8")
        written.append(target)
        index_rows.append(f"{name}\t{key}\t{domain}\t{len(rows)}")

    if not args.no_index:
        index = out_dir / "_index.tsv"
        index.write_text(
            "file\tid\tdomain\trows\n" + "\n".join(index_rows) + "\n", encoding="utf-8"
        )

    stale = existing - {p.name for p in written}
    print(f"\nWrote {len(written):,} file(s) to {out_dir}")
    print(f"  {len(df):,} rows across {len(groups):,} record(s), {len(keys):,} exported")
    if not args.no_index:
        print(f"  index: {out_dir / '_index.tsv'}")
    if stale:
        # Not deleted: this may be a second export into a shared directory on
        # purpose. Said out loud because a stale file from another corpus reads
        # exactly like a current one.
        print(
            f"  NOTE: {len(stale)} .txt file(s) already in this directory were "
            f"not written by this run, e.g. {sorted(stale)[:3]}"
        )
    if written:
        print(f"\nFirst file: {written[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
