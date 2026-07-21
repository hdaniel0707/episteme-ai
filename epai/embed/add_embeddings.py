"""Add several embedding columns to a Parquet in one pass.

Reads a Parquet, works out which of the embedders in ``EMBEDDERS`` are already
present in it, reports the findings, asks whether the ones that already have
values should be recomputed, and then fills in whatever is left.

"Already present" is decided per row, not per column: a column that exists but is
half Null counts as *partial*, and the missing rows are filled even when the user
declines to recompute. As in the prepare_* scripts for text-only corpora, the
full text is passed to add_embedding as the *title* with an empty *abstract*, so
the vectors are directly comparable to the ``emb_specter2`` column those scripts
produce. Rows whose text is empty (None/NaN/blank) are never sent to a model and
keep a Null embedding.

Two progress bars are shown: the outer one walks the selected embedders, the
inner one (from ``add_embedding``) walks the rows of the current embedder.

The Parquet is written after every embedder finishes, so a crash or an
interrupt half-way through a long run does not throw away the models that
already completed. A model that cannot run here (missing optional dependency, no
API key) is dropped from the plan up front; one that fails mid-run (out of
memory, ...) is reported and skipped -- the remaining models still run.

Batch size is the global ``BATCH_SIZE``, except for the models in
``BATCH_SIZE_OVERRIDE`` -- qwen3 and LUAR want a smaller one, openai a much
larger one. ``--batch-size`` overrides all of them.

``--limit N`` embeds only the first N rows of the file, for debugging a model or
timing a run before committing to the whole corpus. Rows past the limit keep
whatever they already have; nothing is dropped from the output.

``--gpu`` and ``--cpu-threads`` pin the run to one GPU and cap the CPU pools, so
several corpora can be embedded side by side on a multi-GPU box without the runs
fighting each other for threads. Both are applied before torch is imported,
which is why the command line is parsed at module level.

Where an embedding is recomputed on top of an existing one, the summary reports
cosine(old, new) with the same buckets as
``epai/prepare/recompute_embedding.py`` -- identical / drifted / changed -- since
re-running a model on the same text rarely reproduces bit-identical floats.

Run:
    uv run python epai/embed/add_embeddings.py data/parquet/essay_gpt54mini_binox0.parquet --cpu-threads 16 --gpu 0
    uv run python epai/embed/add_embeddings.py data/parquet/reuter_gpt54mini_binox0.parquet --cpu-threads 16 --gpu 0
    uv run python epai/embed/add_embeddings.py data/parquet/wp_gpt54mini_binox0.parquet --cpu-threads 16 --gpu 0

    uv run python epai/embed/add_embeddings.py data/parquet/essay_binox0.parquet --cpu-threads 16 --gpu 0
    uv run python epai/embed/add_embeddings.py data/parquet/reuter_binox0.parquet --cpu-threads 16 --gpu 0
    uv run python epai/embed/add_embeddings.py data/parquet/wp_binox0.parquet --cpu-threads 16 --gpu 0

    # debug: 20 rows only, into a scratch copy, recomputing what is already there
    uv run python epai/embed/add_embeddings.py data/parquet/essay_binox0.parquet \
        --limit 20 -o data/parquet/essay_debug.parquet --recompute --yes
"""

from __future__ import annotations

# Standard library only at this point: --gpu and --cpu-threads have to become
# environment variables before torch is imported, so the command line is parsed
# and applied further down, and the heavy imports come after it.
import argparse
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent

# The embedders to put into the Parquet. Each one becomes an ``emb_{name}``
# column. Trim this list to run a subset; every name must be in EMB_TYPES.
EMBEDDERS = [
    # content-heavy
    "specter2",
    "scincl",
    "scibert",
    # general semantic
    "mpnet",
    "minilm",
    "e5",
    "bge",
    "qwen3",
    # raw LM baseline
    "roberta",
    # style / authorship
    "styledistance",
    "style_wegmann",
    "luar",
    "luar_crud",
    # api (billed per row)
    "openai_small",
    "openai_large",
]

# Texts per forward pass. Sized for a 40GB A100 running inference only (no
# gradients): the BERT-base sized encoders sit around 2-4GB at this batch with
# 512-token inputs, so there is plenty of headroom left. Lower it if a model runs
# out of memory, or if the GPU is shared.
BATCH_SIZE = 128

# Models that do not want the global default. qwen3 is 0.6B rather than 110M, so
# its weights and activations are several times larger; LUAR pads every input to
# the full 512 tokens instead of to the longest text in the batch, so its cost is
# always worst-case. openai is network-bound rather than GPU-bound and wants a
# large batch to cut round-trips (the endpoint accepts up to 2048 inputs).
# A --batch-size given on the command line overrides all of these.
BATCH_SIZE_OVERRIDE = {
    "qwen3": 32,
    "luar": 64,
    "luar_crud": 64,
    "openai_small": 256,
    "openai_large": 256,
}

DEFAULT_TEXT_COL = "text"

# Optional third-party package each embedder needs on top of transformers. Used
# for an up-front check so a long run does not die on the fifth model.
_REQUIRED_PACKAGE = {
    "specter2": "adapters",
    "mpnet": "sentence_transformers",
    "minilm": "sentence_transformers",
    "e5": "sentence_transformers",
    "bge": "sentence_transformers",
    "qwen3": "sentence_transformers",
    "styledistance": "sentence_transformers",
    "style_wegmann": "sentence_transformers",
    "openai_small": "openai",
    "openai_large": "openai",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parquet", type=Path, help="Parquet to add embeddings to.")
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output path (default: overwrite the input in place).",
    )
    ap.add_argument(
        "--text-col",
        default=DEFAULT_TEXT_COL,
        help=f"Column holding the text to embed (default: {DEFAULT_TEXT_COL!r}).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Texts per forward pass, applied to every model. Default: "
        f"{BATCH_SIZE}, except {BATCH_SIZE_OVERRIDE} which have per-model values.",
    )
    ap.add_argument(
        "--gpu",
        default=None,
        help="Value for CUDA_VISIBLE_DEVICES (which GPU(s) to use, e.g. '0' or "
        "'0,1'). Default: leave unset so all visible GPUs are used.",
    )
    ap.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Limit CPU threads used for inference (sets OMP_NUM_THREADS / "
        "MKL_NUM_THREADS and torch's intra-op pool). Default: leave unset.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debug option: only embed the first N rows of the file. Later rows "
        "keep whatever they already have, they are not dropped from the output.",
    )
    ap.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute embedders that already have values, without asking.",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompts (implies keeping existing values "
        "unless --recompute is also given).",
    )
    return ap.parse_args()


args = parse_args()

# CUDA_VISIBLE_DEVICES is read when the CUDA driver initialises, and the thread
# limits when the OpenMP runtime loads -- both of which happen inside torch.
# Setting them after torch is imported is too late, so they go here, before the
# imports below pull torch in transitively through embedding_utils.
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.cpu_threads is not None:
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)

import importlib.util  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from tqdm import tqdm  # noqa: E402

from epai.prepare.recompute_embedding import DRIFT_COS, IDENTICAL_COS  # noqa: E402
from epai.utils.embedding_utils import EMB_TYPES, add_embedding  # noqa: E402
from epai.utils.parquet_utils import load_parquet, save_parquet  # noqa: E402

load_dotenv()

if args.cpu_threads is not None:
    # The env vars above cover the OpenMP/MKL pools; this covers torch's own
    # intra-op pool, which does not always follow them.
    torch.set_num_threads(args.cpu_threads)


def describe_device() -> str:
    """One line naming what the models will actually run on."""
    if not torch.cuda.is_available():
        return "cpu"
    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    total = torch.cuda.get_device_properties(index).total_memory / 1024**3
    # After CUDA_VISIBLE_DEVICES the chosen GPU is always renumbered to 0, so
    # report the original id too or parallel runs are impossible to tell apart.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    suffix = f", CUDA_VISIBLE_DEVICES={visible}" if visible else ""
    return f"cuda:{index} — {name}, {total:.0f} GiB{suffix}"


@dataclass
class ColumnStatus:
    """What a single embedder currently looks like in the Parquet."""

    emb_type: str
    col: str
    exists: bool
    n_filled: int  # rows holding a usable vector
    n_missing: int  # rows with a non-empty text but no vector
    dims: set[int] = field(default_factory=set)
    blocked: str | None = None  # why it cannot run here, None if it can

    @property
    def state(self) -> str:
        if not self.exists:
            return "absent"
        if self.n_filled == 0:
            return "empty"
        if self.n_missing > 0:
            return "partial"
        return "complete"

    @property
    def dim_label(self) -> str:
        if not self.dims:
            return "-"
        if len(self.dims) == 1:
            return str(next(iter(self.dims)))
        return "MIXED " + "/".join(str(d) for d in sorted(self.dims))


def _is_empty_text(value) -> bool:
    """True for None/NaN or a string that is empty or only whitespace."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def _as_vector(value) -> np.ndarray | None:
    """Coerce a stored embedding cell to a float vector, or None if it has none."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(value, dtype=float)
        return arr if arr.size else None
    return None


def batch_size_for(emb_type: str, cli_batch_size: int | None) -> int:
    """Batch size for one embedder.

    An explicit --batch-size wins for every model; otherwise the per-model
    override applies, falling back to the global BATCH_SIZE.
    """
    if cli_batch_size is not None:
        return cli_batch_size
    return BATCH_SIZE_OVERRIDE.get(emb_type, BATCH_SIZE)


def blocking_reason(emb_type: str) -> str | None:
    """Why this embedder cannot run here, or None if it can.

    Checked up front so a fourteen-model run does not die on the twelfth.
    """
    package = _REQUIRED_PACKAGE.get(emb_type)
    if package is not None and importlib.util.find_spec(package) is None:
        return f"needs {package}"
    if emb_type.startswith("openai") and not os.environ.get("OPENAI_API_KEY"):
        return "needs OPENAI_API_KEY"
    return None


def apply_limit(embeddable: np.ndarray, limit: int | None) -> np.ndarray:
    """Restrict the embeddable mask to the first ``limit`` rows of the file.

    Rows past the limit are left untouched, they are not dropped -- the output
    keeps every row, only fewer of them get an embedding.
    """
    if limit is None:
        return embeddable
    if limit < 1:
        raise ValueError(f"--limit must be >= 1, got {limit}")
    limited = embeddable.copy()
    limited[limit:] = False
    if not limited.any():
        raise ValueError(
            f"--limit {limit}: none of the first {limit} row(s) has a non-empty text."
        )
    return limited


def cosine_summary(old_values: list, new_values: list) -> dict[str, object]:
    """Classify each recomputed row by how far the new vector moved.

    Same thresholds as epai/prepare/recompute_embedding.py: re-running a model on
    the same text rarely reproduces bit-identical floats, so tiny drift is
    separated from real change instead of testing for equality.
    """
    stats: dict[str, object] = {
        "compared": 0,  # rows with an old and a new vector of equal dimension
        "new": 0,  # old value was Null (nothing to compare against)
        "dim_mismatch": 0,  # old and new dimensions differ
        "identical": 0,  # cos >= IDENTICAL_COS
        "drifted": 0,  # DRIFT_COS <= cos < IDENTICAL_COS
        "changed": 0,  # cos <  DRIFT_COS
    }
    cosines: list[float] = []

    for old_raw, new_raw in zip(old_values, new_values):
        old = _as_vector(old_raw)
        new = _as_vector(new_raw)
        if new is None:
            continue
        if old is None:
            stats["new"] += 1
            continue
        if old.shape != new.shape:
            stats["dim_mismatch"] += 1
            continue

        denom = float(np.linalg.norm(old) * np.linalg.norm(new))
        cos = float(np.dot(old, new) / denom) if denom else 0.0
        cosines.append(cos)
        stats["compared"] += 1

        if cos >= IDENTICAL_COS:
            stats["identical"] += 1
        elif cos >= DRIFT_COS:
            stats["drifted"] += 1
        else:
            stats["changed"] += 1

    stats["cosines"] = cosines
    return stats


def check_parquet(df: pd.DataFrame, text_col: str, path: Path) -> np.ndarray:
    """Validate the Parquet and return the boolean mask of embeddable rows.

    Raises if the file is unusable: no rows, no text column, or no row with a
    non-empty text. Returns a mask that is True where the text can be embedded.
    """
    if len(df) == 0:
        raise ValueError(f"{path} has no rows.")
    if text_col not in df.columns:
        raise KeyError(
            f"{path} is missing the text column {text_col!r}. "
            f"Available columns: {list(df.columns)}"
        )

    embeddable = ~df[text_col].apply(_is_empty_text).to_numpy()
    if not embeddable.any():
        raise ValueError(f"{path}: every value in {text_col!r} is empty — nothing to do.")
    return embeddable


def check_embedders(embedders: list[str]) -> None:
    """Abort on any embedder name add_embedding would not accept."""
    unknown = [e for e in embedders if e not in EMB_TYPES]
    if unknown:
        raise ValueError(
            f"Unknown embedder(s) in EMBEDDERS: {unknown}. "
            f"Choose from: {', '.join(EMB_TYPES)}"
        )


def inspect(
    df: pd.DataFrame, embedders: list[str], embeddable: np.ndarray
) -> list[ColumnStatus]:
    """Report, per embedder, which rows already hold a usable vector."""
    statuses = []
    for emb_type in embedders:
        col = f"emb_{emb_type}"
        blocked = blocking_reason(emb_type)

        if col not in df.columns:
            statuses.append(
                ColumnStatus(
                    emb_type=emb_type,
                    col=col,
                    exists=False,
                    n_filled=0,
                    n_missing=int(embeddable.sum()),
                    blocked=blocked,
                )
            )
            continue

        # A column can exist and still be Null/empty per row, so look at values.
        vectors = [_as_vector(v) for v in df[col]]
        filled = np.array([v is not None for v in vectors])
        dims = {len(v) for v in vectors if v is not None}
        statuses.append(
            ColumnStatus(
                emb_type=emb_type,
                col=col,
                exists=True,
                n_filled=int(filled.sum()),
                n_missing=int((embeddable & ~filled).sum()),
                dims=dims,
                blocked=blocked,
            )
        )
    return statuses


def print_findings(
    path: Path,
    df: pd.DataFrame,
    text_col: str,
    embeddable: np.ndarray,
    statuses: list[ColumnStatus],
) -> None:
    n_empty = len(df) - int(embeddable.sum())

    print("=" * 78)
    print("Parquet")
    print(f"  file          : {path}")
    print(f"  rows          : {len(df)}")
    print(f"  text column   : {text_col!r}")
    print(f"  embeddable    : {int(embeddable.sum())}  ({n_empty} empty -> Null)")
    print(f"  device        : {describe_device()}")
    print(f"  cpu threads   : {torch.get_num_threads()}")
    print("=" * 78)
    print(f"{'column':<22}{'state':<10}{'filled':>8}{'missing':>9}{'dim':>10}  note")
    print("-" * 78)
    for st in statuses:
        note = ""
        if st.blocked:
            note = st.blocked
        elif len(st.dims) > 1:
            note = "inconsistent dimensions"
        print(
            f"{st.col:<22}{st.state:<10}{st.n_filled:>8}{st.n_missing:>9}"
            f"{st.dim_label:>10}  {note}"
        )
    print("-" * 78)


def ask_yes_no(question: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    answer = input(question + suffix).strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def plan(
    statuses: list[ColumnStatus], recompute_existing: bool
) -> list[ColumnStatus]:
    """The embedders that actually need work under the chosen policy."""
    if recompute_existing:
        return [st for st in statuses if st.n_filled > 0 or st.n_missing > 0]
    return [st for st in statuses if st.n_missing > 0]


def positions_to_embed(
    df: pd.DataFrame, st: ColumnStatus, embeddable: np.ndarray, recompute_existing: bool
) -> np.ndarray:
    """Integer row positions to send to the model for one embedder."""
    if recompute_existing or not st.exists:
        return np.flatnonzero(embeddable)
    filled = np.array([_as_vector(v) is not None for v in df[st.col]])
    return np.flatnonzero(embeddable & ~filled)


def embed_column(
    df: pd.DataFrame,
    text_col: str,
    emb_type: str,
    positions: np.ndarray,
    batch_size: int,
) -> tuple[pd.DataFrame, list, list]:
    """Compute one embedder for the given row positions and write the column.

    Returns (df, old_values, new_values), the two value lists aligned to
    ``positions`` so the caller can diff what the recomputation changed.
    """
    col = f"emb_{emb_type}"
    values: list = list(df[col]) if col in df.columns else [None] * len(df)
    old_values = [values[pos] for pos in positions]

    # Same call path as the prepare_* scripts: full text as title, no abstract.
    tmp = pd.DataFrame(
        {"title": df.iloc[positions][text_col].astype(str).to_numpy(), "abstract": ""}
    )
    tmp = add_embedding(tmp, emb_type=emb_type, batch_size=batch_size)
    new_values = list(tmp[col])

    for pos, vec in zip(positions, new_values):
        values[pos] = vec
    df[col] = pd.Series(values, index=df.index, dtype="object")
    return df, old_values, new_values


def print_summary(
    out_path: Path,
    done: list[tuple[str, int, float, dict]],
    failed: list[tuple[str, str]],
) -> None:
    print()
    print("=" * 78)
    print("Embeddings written")
    print(f"  output : {out_path}")
    print("=" * 78)
    for emb_type, n_rows, seconds, stats in done:
        rate = n_rows / seconds if seconds else float("inf")
        print(
            f"  emb_{emb_type:<20} {n_rows:>7} rows  "
            f"{seconds:>8.1f}s  ({rate:.1f} rows/s)"
        )
        if stats["new"]:
            print(f"      new (no previous value)          : {stats['new']}")
        if stats["dim_mismatch"]:
            print(f"      dimension mismatch (old vs new)  : {stats['dim_mismatch']}")

        # Only meaningful where a previous vector existed to compare against.
        cosines = stats["cosines"]
        if cosines:
            arr = np.asarray(cosines)
            print(f"      compared with the old embedding  : {stats['compared']}")
            print(f"        identical (cos >= {IDENTICAL_COS})   : {stats['identical']}")
            print(f"        drifted   (cos >= {DRIFT_COS})       : {stats['drifted']}")
            print(f"        changed   (cos <  {DRIFT_COS})       : {stats['changed']}")
            print(
                f"      cosine(old, new): min {arr.min():.6f}  "
                f"median {np.median(arr):.6f}  mean {arr.mean():.6f}"
            )
    if failed:
        print()
        print("  skipped / failed:")
        for emb_type, message in failed:
            print(f"    emb_{emb_type:<20} {message}")
    if not done and not failed:
        print("  nothing computed.")
    print()


def main() -> None:
    # args is parsed at import time, before torch, so --gpu/--cpu-threads apply.
    out_path = args.out or args.parquet

    check_embedders(EMBEDDERS)

    df = load_parquet(args.parquet)
    embeddable = check_parquet(df, args.text_col, args.parquet)
    n_embeddable = int(embeddable.sum())
    embeddable = apply_limit(embeddable, args.limit)

    # Counts below describe the limited window, so the plan matches the work.
    statuses = inspect(df, EMBEDDERS, embeddable)
    print_findings(args.parquet, df, args.text_col, embeddable, statuses)
    if args.limit is not None:
        print(
            f"NOTE  --limit {args.limit}: only the first {args.limit} row(s) of "
            f"{len(df)} are considered ({int(embeddable.sum())} of {n_embeddable} "
            "embeddable rows); the counts above are within that window."
        )

    # Decide what to do with the embedders that already hold values.
    n_present = sum(1 for st in statuses if st.n_filled > 0)
    recompute_existing = args.recompute
    if n_present and not recompute_existing and not args.yes:
        recompute_existing = ask_yes_no(
            f"{n_present} embedder(s) already have values. Recompute them?",
            default=False,
        )

    wanted = plan(statuses, recompute_existing)

    # Anything unrunnable here is dropped now rather than failing mid-run.
    blocked = [st for st in wanted if st.blocked]
    todo = [st for st in wanted if not st.blocked]
    if not wanted:
        print("Everything is already embedded — nothing to do.")
        return

    print()
    print(f"Will compute {len(todo)} embedder(s) into {out_path}:")
    for st in todo:
        positions = positions_to_embed(df, st, embeddable, recompute_existing)
        batch = batch_size_for(st.emb_type, args.batch_size)
        print(
            f"  {st.col:<22} {len(positions):>7} rows  "
            f"batch {batch:<5} ({st.state})"
        )

    if blocked:
        print()
        print("Skipped — cannot run here:")
        for st in blocked:
            print(f"  {st.col:<22} {st.blocked}")

    if not todo:
        print()
        print("Nothing runnable left — nothing to do.")
        return

    api_models = [st for st in todo if st.emb_type.startswith("openai")]
    if api_models:
        print()
        for st in api_models:
            n_api = len(positions_to_embed(df, st, embeddable, recompute_existing))
            print(f"NOTE  {st.col} is a billed API call: {n_api} row(s) will be sent.")

    if not args.yes:
        answer = input(f"Proceed and write to {out_path}? Type 'yes': ").strip().lower()
        if answer != "yes":
            print("Aborted — no changes made.")
            return

    done: list[tuple[str, int, float, dict]] = []
    failed: list[tuple[str, str]] = [(st.emb_type, st.blocked) for st in blocked]

    # Outer bar: embedders. Inner bar: rows, from add_embedding.
    for st in tqdm(todo, desc="Embedders", unit="model"):
        positions = positions_to_embed(df, st, embeddable, recompute_existing)
        if len(positions) == 0:
            continue

        batch = batch_size_for(st.emb_type, args.batch_size)
        tqdm.write(f"→ {st.col}  ({len(positions)} rows, batch {batch})")
        started = time.perf_counter()
        try:
            df, old_values, new_values = embed_column(
                df, args.text_col, st.emb_type, positions, batch
            )
        except Exception as exc:  # keep going: one bad model must not lose the rest
            failed.append((st.emb_type, f"{type(exc).__name__}: {exc}"))
            tqdm.write(f"  failed: {type(exc).__name__}: {exc}")
            continue

        elapsed = time.perf_counter() - started
        done.append(
            (st.emb_type, len(positions), elapsed, cosine_summary(old_values, new_values))
        )

        # Save after every model so an interrupt does not lose finished work.
        save_parquet(df, out_path)

    print_summary(out_path, done, failed)


if __name__ == "__main__":
    main()
