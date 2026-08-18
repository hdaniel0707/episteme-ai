from pathlib import Path
import pandas as pd
import random
from tqdm import tqdm

# Three escapes, defined here rather than imported: config_utils carries only
# YELLOW and RESET, and preflight_utils sets the precedent of a module naming
# the ones it needs.
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


def coerce_dtypes(df: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Force *dtypes* onto whichever of those columns are present.

    Guards a specific parquet failure: a column that is entirely null gets
    written with Arrow type ``null`` rather than the type it would have had
    with data in it. Reading one such file alone is fine, but reading it
    *beside* a file where the same column has values raises
    ``ArrowNotImplementedError: Unsupported cast from double to null`` — which
    is exactly what happens with a per-domain shard directory, or when an
    optional column is populated by one data source and not the other.

    Fixing it at write time rather than at read time keeps the files
    self-describing: every shard of a run then carries the same schema whether
    or not that particular slice had any values.

    Missing columns are skipped, so this doubles as a no-op on frames that
    never carried the column.
    """
    present = {c: t for c, t in dtypes.items() if c in df.columns}
    if not present:
        return df
    out = df.copy()
    for col, dtype in present.items():
        # to_numeric first: an object column of None cannot go straight to a
        # numeric dtype, and a genuine non-numeric value should become null
        # rather than abort the write.
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(dtype)
    return out


def save_parquet(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    tqdm.write(f"Saved {len(df)} rows → {p}")


def load_parquet(p: Path) -> pd.DataFrame:
    if not p.is_file():
        raise FileNotFoundError(f"Parquet file not found: {p}")
    df = pd.read_parquet(p)
    tqdm.write(f"  Loaded {len(df)} rows ← {p}")
    return df


# The column score_parquets.py fills. Spelled out here rather than imported
# from that script's SCORE_COL: it lives in the Binoculars fork, which runs in
# its own venv on an incompatible transformers pin and cannot be imported from
# this side at all. Same string as cd_core.DEFAULT_BINOX_COL, which is not
# imported either — that module pulls in sklearn, and this is used by pipeline
# runners that otherwise import nothing heavy.
BINOX_COLUMN = "binoculars_score"


def column_fill(p: Path, column: str) -> tuple[bool, int, int]:
    """``(present, n_filled, n_rows)`` for one column, without reading the rest.

    Only that column's pages are touched, so this stays cheap on a corpus whose
    ``emb_*`` columns run to hundreds of megabytes — which is the point: it is
    meant to be called as a postcondition after a stage that fills one column,
    where loading the whole frame to check would cost more than the check is
    worth.

    NaN counts as unfilled, matching how ``score_parquets.py`` and
    ``add_embeddings.py`` both read a cell that holds no usable value: a float
    column round-tripped through pandas can carry either a null or a NaN for the
    same missing value, depending on the dtype it passed through. A column that
    is absent entirely returns ``(False, 0, n_rows)`` rather than raising:
    absent and present-but-empty are different findings, and the caller is
    usually reporting on the difference.

    For a scalar column. Counting in Python rather than through a compute
    kernel keeps the NaN rule identical to the one the fillers use and costs
    nothing at one column of a corpus; pointed at an ``emb_*`` column it would
    materialise every vector.
    """
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(p)
    n_rows = handle.metadata.num_rows
    if column not in handle.schema_arrow.names:
        return False, 0, n_rows

    values = pq.read_table(p, columns=[column]).column(0).to_pylist()
    # v != v is the NaN test, and the only one that needs no import.
    missing = sum(1 for v in values if v is None or v != v)
    return True, len(values) - missing, n_rows


def check_column_filled(
    p: Path,
    column: str,
    limit: int | None = None,
) -> tuple[bool, str]:
    """Did the stage that was supposed to fill *column* actually fill it?

    Returns ``(ok, message)`` for a pipeline stage to print and turn into an
    exit code. Lives here, and not in either runner, so the two pipelines cannot
    drift into different ideas of what a filled column looks like.

    Why a stage needs this at all: an exit code says the subprocess ran, not
    that it did the work. ``score_parquets.py`` deliberately catches a per-file
    failure so one bad file cannot lose the rest of a multi-file run, and it
    used to report the skip and still return 0 — which is how a corpus reached
    the embed stage, and the end of a "successful" pipeline, with no
    ``binoculars_score`` column at all. The fork now exits non-zero for that,
    but the postcondition is worth checking on the file itself either way: it is
    cheap, and it is the only thing that actually looks at the result.

    Three findings, because they need different responses:

      * column absent — the stage wrote nothing at all. Not ok.
      * column present, every value Null — it wrote the column and filled
        nothing. Not ok.
      * some values Null — ok with a warning. Rows whose text is blank
        legitimately keep a Null score, so this is normal, and how many is
        exactly what an operator wants told rather than guessed.

    ``limit`` mirrors the ``--limit N`` that both fillers take: under one, only
    the first N rows were meant to be touched, so the rest being Null is the
    intended outcome and not worth a warning.
    """
    present, filled, n_rows = column_fill(p, column)
    expected = n_rows if limit is None else min(limit, n_rows)

    if not present:
        return False, (
            f"{_RED}{p.name} has no {column!r} column.{_RESET}\n"
            f"  The stage reported success but wrote nothing. Read its output "
            f"above for what it skipped."
        )
    if filled == 0:
        return False, (
            f"{_RED}{column!r} is present in {p.name} but every one of "
            f"{n_rows} row(s) is Null.{_RESET}\n"
            f"  The column was written and nothing was filled in."
        )
    if filled < expected:
        return True, (
            f"{_YELLOW}NOTE: {column!r} is filled for {filled} of {expected} "
            f"row(s); {expected - filled} are Null.{_RESET}\n"
            f"  Expected when some rows hold blank text — those are never "
            f"scored or embedded. Unexpected otherwise."
        )
    return True, f"{_GREEN}{column!r} filled for {filled} row(s).{_RESET}"


def pick_label(q: str, query_mode: str = "random"):
    if q is None or len(q) == 0:
        return "unknown"
    if query_mode == "random":
        return random.choice(q)
    if query_mode == "concat":
        return " + ".join(q)
    return q[0]
