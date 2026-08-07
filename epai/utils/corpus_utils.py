r"""One way to name a corpus, and one way to read it.

Every training and analysis script used to say "a corpus is a parquet file", and
each one re-implemented the same three steps: resolve the path, drop the
unwanted generators, count what was lost. Since
``epai/ai_detection/prepare/combine_domains.py`` folded essay/reuter/wp into one
file with a ``domain`` column, that definition no longer holds -- a corpus is a
SLICE of a file -- and the duplication became a place for the scripts to drift
apart. This module is the single definition:

    a corpus = (parquet file, optional domain, optional dropped generators)

expressed as a :class:`CorpusSpec`, read by :func:`read_corpus`, and checked by
:func:`check_corpus_files` before anything is loaded for real.

Config shape, in ``data``:

    parquet_path: "data/parquet/ghostbuster_gpt35t.parquet"   # default for all
    datasets:
      - name: "essay"                                  # key used in the matrix
        label: "essay (gpt-3.5)"                       # optional, for figures
        where: {domain: "essay"}                       # optional; omit = whole file
        drop_classes: []                               # optional generators to skip

``parquet_path``, ``drop_classes`` and ``where`` may be given once under ``data``
and left out of the entries, or repeated per entry, in which case the entry wins.
The first form is the one to use when every corpus is a slice of one file -- the
path repeated three times says nothing and drifts when the file is renamed; the
second is for a matrix spanning several files.

WHICH SLICE: ``where`` is a mapping of column to the value (or list of values) to
keep, ANDed across columns. ``domain`` is only the most common column to cut on,
not a special one, so a matrix can just as well run across generators, sources or
years -- and the corpora of one run need not be cut on the same column at all::

    - {name: essay_gpt,    where: {domain: essay, generated_by: [human, gpt]}}
    - {name: essay_claude, where: {domain: essay, generated_by: [human, claude]}}

``domain: essay`` is kept as shorthand for ``where: {domain: essay}``. Both are
optional, so a config that predates the combined parquet keeps working unchanged:
no conditions means the whole file, which is exactly what a per-domain parquet
holds.

Values are compared as strings (``year: 2024`` matches an int column), and
:func:`check_corpus_files` verifies every value against the file before a row is
loaded -- a typo is "essay is not one of essai/reuter/wp", not an empty corpus
twenty minutes into a run. ``drop_classes`` is the complement of ``where``: the
generators to remove rather than the values to keep.

ORDER OF OPERATIONS, and why it matters: the domain filter and ``drop_classes``
are applied HERE, before any caller drops rows for missing values. The
missing-value drop is group-aware (a human document with no embedding takes its
machine rewrites with it, so the classes stay balanced), so a row that the run
was never going to use must be gone before it can take a group down with it.
Doing it the other way round -- as ``run_clf_cross_domain.py`` used to -- lets a
dropped generator's Null embedding delete rows the run actually wanted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from epai.utils.parquet_utils import load_parquet
from epai.utils.train_utils import CLF_TARGET_COL

BASE_DIR = Path(__file__).parent.parent.parent

# Written by combine_domains.py. ``uid`` is unique per row; ``id`` is the PAIR
# key -- a source prompt shared by its human and machine answers -- and stays
# the grouping column for every split.
DOMAIN_COL = "domain"
ID_COL = "id"
UID_COL = "uid"


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


#: One normalised ``where`` condition: a column, and the values kept in it.
Condition = tuple[str, tuple[str, ...]]


def normalise_where(raw, domain=None) -> tuple[Condition, ...]:
    """A ``where`` mapping (plus the ``domain`` shorthand) as ordered conditions.

    ``{domain: essay, generated_by: [human, gpt]}`` becomes
    ``(("domain", ("essay",)), ("generated_by", ("human", "gpt")))``: a scalar is
    a one-value condition, a list is a membership test, and several conditions
    are ANDed. Config order is kept so messages read the way the file does.

    Values are normalised to strings, and :func:`read_corpus` compares the column
    as a string too -- the same thing the ``domain`` filter has always done, now
    applied to every column. It means ``year: 2024`` matches an int column, and
    it means the preflight's "available: [...]" list is the authority on what a
    value has to look like.
    """
    conds: dict[str, tuple[str, ...]] = {}
    if isinstance(raw, dict):
        for col, value in raw.items():
            values = value if isinstance(value, (list, tuple)) else [value]
            conds[str(col)] = tuple(str(v) for v in values)
    # The shorthand loses to an explicit condition on the same column; setting
    # both is contradictory and check_dataset_specs reports it.
    if domain is not None and DOMAIN_COL not in conds:
        conds[DOMAIN_COL] = (str(domain),)
    return tuple(conds.items())


@dataclass(frozen=True)
class CorpusSpec:
    """One entry of ``data.datasets``, resolved against the project root."""

    name: str
    path: Path
    where: tuple[Condition, ...] = ()
    label: str = ""
    drop_classes: tuple[str, ...] = ()

    @classmethod
    def from_dict(
        cls,
        spec: dict,
        base_dir: Path = BASE_DIR,
        default_path: str | None = None,
        default_drop_classes: list[str] | None = None,
        default_where=None,
    ) -> CorpusSpec:
        """One dataset entry, with the ``data``-level defaults filled in.

        ``parquet_path``, ``drop_classes`` and ``where`` may be set once under
        ``data`` and omitted per dataset -- the common case now that three
        corpora live in one file, where repeating the same path per entry says
        nothing and drifts when the file is renamed. An entry that names any of
        them wins, so a matrix mixing two combined files stays expressible.
        """
        raw = spec.get("parquet_path") or default_path
        if not raw:
            raise KeyError(
                f"dataset {spec.get('name')!r}: no 'parquet_path', and "
                f"'data.parquet_path' is not set either."
            )
        raw_path = Path(str(raw))
        drop = spec.get("drop_classes")
        if drop is None:
            drop = default_drop_classes or ()
        raw_where = spec.get("where")
        if raw_where is None and spec.get("domain") is None:
            raw_where = default_where
        return cls(
            name=str(spec["name"]),
            path=raw_path if raw_path.is_absolute() else base_dir / raw_path,
            where=normalise_where(raw_where, spec.get("domain")),
            label=str(spec.get("label") or spec["name"]),
            drop_classes=tuple(drop),
        )

    @property
    def filter_columns(self) -> list[str]:
        """Columns ``where`` reads, so the pruned parquet read can include them."""
        return [col for col, _ in self.where]

    @property
    def domain(self) -> str | None:
        """The single ``domain`` value this corpus is narrowed to, if any.

        A convenience for the code and messages that still think in domains --
        ``domain`` is just one column ``where`` can filter on, but it is the one
        ``combine_domains.py`` writes and the one most corpora are sliced by.
        """
        for col, values in self.where:
            if col == DOMAIN_COL and len(values) == 1:
                return values[0]
        return None

    @property
    def where_str(self) -> str:
        """The conditions as ``domain=essay, generated_by=human|gpt``."""
        return ", ".join(f"{c}={'|'.join(v)}" for c, v in self.where)

    @property
    def source(self) -> str:
        """How this corpus is identified in a message: file, and slice of it."""
        return f"{self.path.name} [{self.where_str}]" if self.where else self.path.name


def single_corpus_spec(cfg: dict, base_dir: Path = BASE_DIR) -> CorpusSpec:
    """The ONE corpus of an in-domain config, where ``data`` *is* the corpus.

    A cross-domain config carries a ``datasets`` list because the point is the
    matrix between them. An in-domain run has nothing to put in a matrix, so it
    would be a list of one -- a shape that reads as though a second entry were
    coming. Instead ``data`` holds the corpus directly: the same
    ``parquet_path`` / ``where`` / ``drop_classes`` keys, one level up.

    ``name`` and ``label`` default to the file stem, narrowed by the conditions,
    since there is no sibling corpus to tell it apart from.
    """
    data = cfg["data"]
    where = normalise_where(data.get("where"), data.get("domain"))
    stem = Path(str(data.get("parquet_path", "corpus"))).stem
    name = data.get("name") or (
        "+".join([stem] + [v for _, values in where for v in values]) if where else stem
    )
    return CorpusSpec.from_dict(
        {
            "name": str(name),
            "label": str(data.get("label") or name),
            "parquet_path": data.get("parquet_path"),
            "where": data.get("where"),
            "domain": data.get("domain"),
            "drop_classes": data.get("drop_classes"),
        },
        base_dir,
    )


def dataset_specs(cfg: dict, base_dir: Path = BASE_DIR) -> list[CorpusSpec]:
    """Every dataset of a cross-domain config, in config order."""
    data = cfg["data"]
    default_path = data.get("parquet_path")
    default_drop = data.get("drop_classes")
    default_where = data.get("where")
    return [
        CorpusSpec.from_dict(spec, base_dir, default_path, default_drop, default_where)
        for spec in data["datasets"]
    ]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


@dataclass
class CorpusFrame:
    """The rows of one corpus, and what was filtered out on the way in."""

    spec: CorpusSpec
    df: pd.DataFrame
    n_file: int  # rows in the parquet
    n_filtered: int  # rows the `where` conditions excluded
    n_class_dropped: int  # rows removed by drop_classes

    @property
    def n(self) -> int:
        return len(self.df)

    def summary(self) -> str:
        parts = [f"{self.n_file} in file"]
        if self.n_filtered:
            parts.append(f"{self.n_filtered} outside {self.spec.where_str}")
        if self.n_class_dropped:
            parts.append(f"{self.n_class_dropped} dropped generators")
        return f"{self.n} rows ({', '.join(parts)})"


def read_corpus(
    spec: CorpusSpec, columns: list[str] | None = None, quiet: bool = False
) -> CorpusFrame:
    """Read one corpus: the file, narrowed by ``where``, minus its drop_classes.

    ``columns`` prunes the read, which is not optional in practice -- the
    combined parquets carry fifteen embeddings and run to hundreds of MB, so a
    full read per (feature set, dataset) pair would move gigabytes to use one
    column. The columns needed to APPLY the filters are added automatically, so
    a caller only has to name what it wants to train on.

    Every ``where`` condition is a membership test read as strings, so a config
    filters on any column -- ``domain`` is only the most common one -- and
    several conditions narrow the corpus in the order the config lists them.
    """
    if columns is not None:
        needed = list(columns) + spec.filter_columns
        if spec.drop_classes:
            needed.append(CLF_TARGET_COL)
        df = pd.read_parquet(spec.path, columns=list(dict.fromkeys(needed)))
    else:
        df = load_parquet(spec.path) if not quiet else pd.read_parquet(spec.path)

    n_file = len(df)

    n_filtered = 0
    for col, values in spec.where:
        if col not in df.columns:
            hint = (
                f" Build it with epai/ai_detection/prepare/combine_domains.py, or "
                f"drop the condition to use the whole file."
                if col == DOMAIN_COL
                else ""
            )
            raise KeyError(
                f"{spec.name}: {spec.path.name} has no {col!r} column, but the "
                f"config filters on {col}={'|'.join(values)}.{hint}"
            )
        keep = df[col].astype(str).isin(values)
        n_filtered += int((~keep).sum())
        df = df.loc[keep].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"{spec.name}: no rows left with {col}={'|'.join(values)} in "
                f"{spec.path.name}. Available: {sorted(column_values(spec.path, col))}"
            )

    n_class_dropped = 0
    if spec.drop_classes:
        keep = ~df[CLF_TARGET_COL].isin(list(spec.drop_classes))
        n_class_dropped = int((~keep).sum())
        df = df.loc[keep].reset_index(drop=True)

    return CorpusFrame(
        spec=spec,
        df=df.reset_index(drop=True),
        n_file=n_file,
        n_filtered=n_filtered,
        n_class_dropped=n_class_dropped,
    )


# ---------------------------------------------------------------------------
# Inspection (footer / single column only -- these run in preflight)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def parquet_schema(path: Path) -> tuple[tuple[str, ...], int]:
    """(column names, row count) from the file footer. No row is read."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    return tuple(pf.schema_arrow.names), int(pf.metadata.num_rows)


@lru_cache(maxsize=256)
def column_counts(path: Path, column: str) -> dict[str, int]:
    """How many rows hold each value of one column, or {} if it is not there.

    Reads that one column and nothing else, so it stays cheap enough to run in
    preflight on a 400 MB file -- and cached, since a config with six datasets
    filtering on the same column would otherwise re-read it six times.
    """
    columns, _ = parquet_schema(path)
    if column not in columns:
        return {}
    values = pd.read_parquet(path, columns=[column])[column].astype(str)
    return {str(k): int(v) for k, v in values.value_counts().items()}


def column_values(path: Path, column: str) -> tuple[str, ...]:
    """The distinct values of one column, sorted. () if the column is absent."""
    return tuple(sorted(column_counts(path, column)))


def available_domains(path: Path) -> tuple[str, ...]:
    """Domains present in a file, or () if it has no domain column."""
    return column_values(path, DOMAIN_COL)


def domain_row_counts(path: Path) -> dict[str, int]:
    return column_counts(path, DOMAIN_COL)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def check_corpus_files(
    specs: list[CorpusSpec],
    needed: list[str],
    report,
    emb_cols: tuple[str, ...] | list[str] = (),
    rule: int = 88,
) -> None:
    """Every corpus exists, opens, holds its columns, and has its ``where`` values.

    The ``where`` check is the one worth having: a typo in a filter value selects
    zero rows, and without this it would surface as an empty-dataset error
    somewhere inside the first fit instead of as "essay is not one of
    essai/reuter/wp". Every condition is checked, not just the first, so one pass
    reports every typo in the file.
    """
    print(f"{'dataset':<22}{'rows':>9}{'cols':>7}  source")
    print("-" * rule)

    for spec in specs:
        if not spec.path.is_file():
            report.error(f"{spec.name}: parquet not found: {spec.path}")
            print(f"{spec.name:<22}{'-':>9}{'-':>7}  {spec.path}  MISSING")
            continue
        try:
            columns, n_rows = parquet_schema(spec.path)
        except Exception as exc:
            report.error(
                f"{spec.name}: cannot read {spec.path}: {type(exc).__name__}: {exc}"
            )
            print(f"{spec.name:<22}{'-':>9}{'-':>7}  {spec.path}  UNREADABLE")
            continue

        n_shown, note, ok = n_rows, "", True
        for col, values in spec.where:
            if col not in columns:
                hint = (
                    " — build it with prepare/combine_domains.py or drop the key"
                    if col == DOMAIN_COL
                    else f" — columns are {sorted(columns)[:8]}…"
                )
                report.error(
                    f"{spec.name}: config filters on {col}={'|'.join(values)} but "
                    f"{spec.path.name} has no {col!r} column{hint}"
                )
                note, ok = f"  NO {col.upper()} COLUMN", False
                continue
            counts = column_counts(spec.path, col)
            unknown = [v for v in values if v not in counts]
            if unknown:
                report.error(
                    f"{spec.name}: {col}={unknown} not found in {spec.path.name}; "
                    f"available: {sorted(counts)}"
                )
                note, ok = f"  UNKNOWN {col.upper()}", False
            elif ok:
                # Rows matching THIS condition. Exact for a single condition (the
                # usual `domain: essay`); an upper bound once several narrow each
                # other, which is why stage 3 prints the real post-filter count.
                n_shown = min(n_shown, sum(counts[v] for v in values))

        if ok and spec.where:
            note = f"  [{spec.where_str}]" + ("  (≤)" if len(spec.where) > 1 else "")

        print(f"{spec.name:<22}{n_shown:>9}{len(columns):>7}  "
              f"{spec.path}{note}")

        missing = [c for c in needed if c not in columns]
        if missing:
            # A missing embedding is a config/parquet mismatch with an obvious
            # fix, so it is named separately from a structural one.
            missing_emb = [c for c in missing if c in tuple(emb_cols)]
            missing_other = [c for c in missing if c not in tuple(emb_cols)]
            if missing_other:
                report.error(f"{spec.name}: missing column(s) {missing_other}")
            if missing_emb:
                report.error(
                    f"{spec.name}: missing embedding column(s) {missing_emb} — run "
                    f"epai/ai_detection/embed/add_embeddings.py on this parquet, or "
                    f"drop them from 'embeddings' in the config"
                )


def check_where(raw, context: str, report) -> None:
    """Validate one ``where`` block's shape. Values are checked against the file
    later, by :func:`check_corpus_files`, which can list what is actually there.
    """
    if raw is None:
        return
    if not isinstance(raw, dict):
        report.error(
            f"{context} 'where' must be a mapping of column -> value or list of "
            f"values, got {type(raw).__name__}"
        )
        return
    if not raw:
        report.error(f"{context} 'where' is empty — omit it to use the whole file")
        return
    for col, value in raw.items():
        if not isinstance(col, str) or not col.strip():
            report.error(f"{context} 'where' has a non-string column key {col!r}")
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        if not values:
            report.error(
                f"{context} 'where.{col}' is an empty list, which keeps no rows"
            )
        if any(isinstance(v, (list, tuple, dict)) for v in values):
            report.error(
                f"{context} 'where.{col}' must hold scalars (a value, or a list of "
                f"values to keep), got {value!r}"
            )


def check_single_corpus(cfg: dict, report) -> None:
    """Validate ``data`` when it *is* the corpus -- the in-domain config shape.

    The counterpart of :func:`check_dataset_specs`. A ``datasets`` list here is
    almost certainly a cross-domain config pointed at the wrong runner, so it is
    named as such rather than ignored.
    """
    data = cfg.get("data")
    if not isinstance(data, dict):
        report.error("'data' must be a mapping")
        return

    for key in ("human_label", "group_col", "parquet_path"):
        if key not in data:
            report.error(f"data is missing {key!r}")

    if "datasets" in data:
        report.error(
            "data has a 'datasets' list, which is the cross-domain shape: an "
            "in-domain run trains on ONE corpus, so 'data' names it directly "
            "(parquet_path / where / drop_classes). Run the matrix with "
            "epai/ai_detection/train/run_clf_cross_domain.py instead"
        )

    domain = data.get("domain")
    if domain is not None and (not isinstance(domain, str) or not domain.strip()):
        report.error(f"data '{DOMAIN_COL}' must be a non-empty string when given")
    if domain is not None and isinstance(data.get("where"), dict) and (
        DOMAIN_COL in data["where"]
    ):
        report.error(
            f"data sets both '{DOMAIN_COL}: {domain}' and "
            f"'where.{DOMAIN_COL}: {data['where'][DOMAIN_COL]}' — 'domain' is "
            f"shorthand for the same condition, so keep only one"
        )
    check_where(data.get("where"), "data", report)

    drop = data.get("drop_classes") or []
    if not isinstance(drop, (list, tuple)):
        report.error("data 'drop_classes' must be a list")


def check_dataset_specs(cfg: dict, report, min_datasets: int = 2) -> None:
    """Validate the shape of ``data.datasets`` -- keys, types, unique names.

    Shared by every cross-domain runner: they used to carry a copy each, which
    is how one of them ended up accepting a key the others rejected.
    """
    data = cfg.get("data")
    if not isinstance(data, dict):
        report.error("'data' must be a mapping")
        return

    for key in ("human_label", "group_col", "datasets"):
        if key not in data:
            report.error(f"data is missing {key!r}")

    specs = data.get("datasets") or []
    if not isinstance(specs, (list, tuple)) or len(specs) < min_datasets:
        report.error(
            f"cross-domain needs at least {min_datasets} datasets, got {len(specs)}"
        )
        return

    # `parquet_path`, `drop_classes` and `where` may be set once under `data` and
    # left out of the entries -- the usual shape when every corpus is a slice of
    # one combined file.
    default_path = data.get("parquet_path")
    default_drop = data.get("drop_classes")
    if default_drop is not None and not isinstance(default_drop, (list, tuple)):
        report.error("data 'drop_classes' must be a list when given")
    check_where(data.get("where"), "data", report)

    names: list[str] = []
    for i, spec in enumerate(specs):
        if not isinstance(spec, dict):
            report.error(f"datasets[{i}] must be a mapping")
            continue
        if "name" not in spec:
            report.error(f"datasets[{i}] is missing 'name'")
        if not spec.get("parquet_path") and not default_path:
            report.error(
                f"datasets[{i}] has no 'parquet_path' and 'data.parquet_path' "
                f"is not set — one of the two must name a file"
            )
        domain = spec.get("domain")
        if domain is not None and (not isinstance(domain, str) or not domain.strip()):
            report.error(
                f"datasets[{i}] '{DOMAIN_COL}' must be a non-empty string when given"
            )
        check_where(spec.get("where"), f"datasets[{i}]", report)
        if domain is not None and isinstance(spec.get("where"), dict) and (
            DOMAIN_COL in spec["where"]
        ):
            # Both name the same column, and they would not agree. Silently
            # preferring one is how a config comes to describe a run it did not do.
            report.error(
                f"datasets[{i}] sets both '{DOMAIN_COL}: {domain}' and "
                f"'where.{DOMAIN_COL}: {spec['where'][DOMAIN_COL]}' — 'domain' is "
                f"shorthand for the same condition, so keep only one"
            )
        drop = spec.get("drop_classes") or []
        if not isinstance(drop, (list, tuple)):
            report.error(f"datasets[{i}] 'drop_classes' must be a list")
        names.append(spec.get("name"))

    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        report.error(f"dataset names must be unique, repeated: {dupes}")


# ---------------------------------------------------------------------------
# Analysis-side helpers
# ---------------------------------------------------------------------------


@dataclass
class CorpusSelection:
    """Corpora chosen on the command line rather than by a config.

    The analysis scripts take parquets as positional arguments and used to label
    each corpus by file stem. With a combined file that label is wrong -- three
    corpora share one stem -- so they take ``--domains`` instead and this turns
    the pair into the same CorpusSpec list the training side uses.
    """

    specs: list[CorpusSpec] = field(default_factory=list)

    def names(self) -> list[str]:
        return [s.name for s in self.specs]


def specs_from_paths(
    paths: list[Path],
    domains: list[str] | None = None,
    drop_classes: list[str] | None = None,
    base_dir: Path = BASE_DIR,
) -> list[CorpusSpec]:
    """Build specs from CLI arguments: N parquets x the domains asked for.

    With no ``--domains``, a file that HAS a domain column is split into one
    corpus per domain it contains (the useful default for a combined parquet),
    and a file without one becomes a single corpus named after its stem.
    """
    specs: list[CorpusSpec] = []
    for raw in paths:
        path = Path(raw)
        path = path if path.is_absolute() else base_dir / path
        if not path.is_file():
            raise SystemExit(f"Parquet not found: {path}")

        present = available_domains(path)
        wanted = list(domains) if domains else list(present)

        if not wanted:  # no domain column and none requested: the whole file
            specs.append(CorpusSpec(
                name=path.stem, path=path, label=path.stem,
                drop_classes=tuple(drop_classes or ()),
            ))
            continue

        unknown = [d for d in wanted if d not in present]
        if unknown:
            raise SystemExit(
                f"{path.name}: unknown {DOMAIN_COL}(s) {unknown}; available: "
                f"{list(present)}"
            )
        multi_file = len(paths) > 1
        for domain in wanted:
            # With several files in play the domain alone is ambiguous (essay
            # from which corpus?), so the stem stays in the name.
            name = f"{path.stem}:{domain}" if multi_file else domain
            specs.append(CorpusSpec(
                name=name, path=path, where=((DOMAIN_COL, (domain,)),), label=name,
                drop_classes=tuple(drop_classes or ()),
            ))
    return specs
