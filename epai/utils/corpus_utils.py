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
        domain: "essay"                                # optional; omit = whole file
        drop_classes: []                               # optional generators to skip

``parquet_path`` and ``drop_classes`` may be given once under ``data`` and left
out of the entries, or repeated per entry, in which case the entry wins. The
first form is the one to use when every corpus is a domain of one file -- the
path repeated three times says nothing and drifts when the file is renamed; the
second is for a matrix spanning several files.

``domain`` is optional, so a config that predates the combined parquet keeps
working unchanged: no domain means the whole file, which is exactly what a
per-domain parquet holds.

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


@dataclass(frozen=True)
class CorpusSpec:
    """One entry of ``data.datasets``, resolved against the project root."""

    name: str
    path: Path
    domain: str | None = None
    label: str = ""
    drop_classes: tuple[str, ...] = ()

    @classmethod
    def from_dict(
        cls,
        spec: dict,
        base_dir: Path = BASE_DIR,
        default_path: str | None = None,
        default_drop_classes: list[str] | None = None,
    ) -> CorpusSpec:
        """One dataset entry, with the ``data``-level defaults filled in.

        ``parquet_path`` and ``drop_classes`` may be set once under ``data`` and
        omitted per dataset -- the common case now that three corpora live in one
        file, where repeating the same path per entry says nothing and drifts
        when the file is renamed. An entry that names either one wins, so a
        matrix mixing two combined files stays expressible.
        """
        raw = spec.get("parquet_path") or default_path
        if not raw:
            raise KeyError(
                f"dataset {spec.get('name')!r}: no 'parquet_path', and "
                f"'data.parquet_path' is not set either."
            )
        raw_path = Path(str(raw))
        domain = spec.get("domain")
        drop = spec.get("drop_classes")
        if drop is None:
            drop = default_drop_classes or ()
        return cls(
            name=str(spec["name"]),
            path=raw_path if raw_path.is_absolute() else base_dir / raw_path,
            domain=str(domain) if domain else None,
            label=str(spec.get("label") or spec["name"]),
            drop_classes=tuple(drop),
        )

    @property
    def source(self) -> str:
        """How this corpus is identified in a message: file, and slice of it."""
        where = f"{self.path.name}"
        return f"{where} [{DOMAIN_COL}={self.domain}]" if self.domain else where


def dataset_specs(cfg: dict, base_dir: Path = BASE_DIR) -> list[CorpusSpec]:
    """Every dataset of a cross-domain config, in config order."""
    data = cfg["data"]
    default_path = data.get("parquet_path")
    default_drop = data.get("drop_classes")
    return [
        CorpusSpec.from_dict(spec, base_dir, default_path, default_drop)
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
    n_domain_dropped: int  # rows belonging to another domain
    n_class_dropped: int  # rows removed by drop_classes

    @property
    def n(self) -> int:
        return len(self.df)

    def summary(self) -> str:
        parts = [f"{self.n_file} in file"]
        if self.n_domain_dropped:
            parts.append(f"{self.n_domain_dropped} other {DOMAIN_COL}s")
        if self.n_class_dropped:
            parts.append(f"{self.n_class_dropped} dropped generators")
        return f"{self.n} rows ({', '.join(parts)})"


def read_corpus(
    spec: CorpusSpec, columns: list[str] | None = None, quiet: bool = False
) -> CorpusFrame:
    """Read one corpus: the file, narrowed to its domain, minus its drop_classes.

    ``columns`` prunes the read, which is not optional in practice -- the
    combined parquets carry fifteen embeddings and run to hundreds of MB, so a
    full read per (embedding, dataset) pair would move gigabytes to use one
    column. The columns needed to APPLY the filters are added automatically, so
    a caller only has to name what it wants to train on.
    """
    if columns is not None:
        needed = list(columns)
        if spec.domain:
            needed.append(DOMAIN_COL)
        if spec.drop_classes:
            needed.append(CLF_TARGET_COL)
        df = pd.read_parquet(spec.path, columns=list(dict.fromkeys(needed)))
    else:
        df = load_parquet(spec.path) if not quiet else pd.read_parquet(spec.path)

    n_file = len(df)

    n_domain_dropped = 0
    if spec.domain:
        if DOMAIN_COL not in df.columns:
            raise KeyError(
                f"{spec.name}: {spec.path.name} has no {DOMAIN_COL!r} column, but the "
                f"config asks for {DOMAIN_COL}={spec.domain!r}. Build it with "
                f"epai/ai_detection/prepare/combine_domains.py, or drop the "
                f"'{DOMAIN_COL}' key to use the whole file."
            )
        keep = df[DOMAIN_COL].astype(str) == spec.domain
        n_domain_dropped = int((~keep).sum())
        df = df.loc[keep].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"{spec.name}: no rows with {DOMAIN_COL}={spec.domain!r} in "
                f"{spec.path.name}. Available: {sorted(available_domains(spec.path))}"
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
        n_domain_dropped=n_domain_dropped,
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


@lru_cache(maxsize=64)
def available_domains(path: Path) -> tuple[str, ...]:
    """Domains present in a file, or () if it has no domain column.

    Reads that one column and nothing else, so it stays cheap enough to run in
    preflight on a 400 MB file.
    """
    columns, _ = parquet_schema(path)
    if DOMAIN_COL not in columns:
        return ()
    values = pd.read_parquet(path, columns=[DOMAIN_COL])[DOMAIN_COL]
    return tuple(sorted(str(v) for v in values.dropna().unique()))


@lru_cache(maxsize=64)
def domain_row_counts(path: Path) -> dict[str, int]:
    columns, _ = parquet_schema(path)
    if DOMAIN_COL not in columns:
        return {}
    values = pd.read_parquet(path, columns=[DOMAIN_COL])[DOMAIN_COL].astype(str)
    return {str(k): int(v) for k, v in values.value_counts().items()}


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
    """Every corpus exists, opens, holds its columns, and has its domain.

    The domain check is the one worth having: a typo in ``domain:`` selects zero
    rows, and without this it would surface as an empty-dataset error somewhere
    inside the first fit instead of as "essay is not one of essai/reuter/wp".
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

        n_shown = n_rows
        domain_note = ""
        if spec.domain:
            if DOMAIN_COL not in columns:
                report.error(
                    f"{spec.name}: config asks for {DOMAIN_COL}={spec.domain!r} but "
                    f"{spec.path.name} has no {DOMAIN_COL!r} column — build it with "
                    f"prepare/combine_domains.py or drop the key"
                )
                domain_note = f"  NO {DOMAIN_COL.upper()} COLUMN"
            else:
                counts = domain_row_counts(spec.path)
                if spec.domain not in counts:
                    report.error(
                        f"{spec.name}: {DOMAIN_COL}={spec.domain!r} is not in "
                        f"{spec.path.name}; available: {sorted(counts)}"
                    )
                    domain_note = "  UNKNOWN DOMAIN"
                else:
                    n_shown = counts[spec.domain]
                    domain_note = f"  [{DOMAIN_COL}={spec.domain}]"

        print(f"{spec.name:<22}{n_shown:>9}{len(columns):>7}  "
              f"{spec.path}{domain_note}")

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

    # `parquet_path` and `drop_classes` may be set once under `data` and left
    # out of the entries -- the usual shape when every corpus is a domain of one
    # combined file.
    default_path = data.get("parquet_path")
    default_drop = data.get("drop_classes")
    if default_drop is not None and not isinstance(default_drop, (list, tuple)):
        report.error("data 'drop_classes' must be a list when given")

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
                name=name, path=path, domain=domain, label=name,
                drop_classes=tuple(drop_classes or ()),
            ))
    return specs
