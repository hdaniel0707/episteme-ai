r"""Build an eval-only corpus end to end: restructure, standardise, embed, score.

    uv run python epai/ai_detection/prepare/run_holdout_pipeline.py --yes

Takes ``data/parquet/denario_papers.parquet`` -- the citation_impact schema, no
detector columns at all -- and leaves ``data/parquet/denario_detect.parquet``
carrying every column a training config reads, so that ``data.eval_only`` can be
switched on. Nothing here trains anything or reports a result; it fills columns.

WHY A PIPELINE RATHER THAN A LIST IN A README
----------------------------------------------
Because the ORDER is load-bearing in two places and neither failure is loud:

  * ``standardize_text.py`` must run before anything reads ``text``. It refuses
    to run once scores or embeddings exist, so getting this wrong does not
    corrupt anything -- it just stops, halfway through, after the expensive
    stages. Putting it in the chain is how it stays first.
  * ``add_cue_features.py`` reads the rationales ``add_llm_judge_reasons.py``
    writes, which read the scores ``add_llm_judge_scores.py`` writes. Three
    stages, one column, and running them out of order produces an empty cue
    vector rather than an error.

The rest is one gotcha per script, each of which costs a GPU hour to discover by
hand and is a default in the stage list below:

  * ``binoculars_score`` (falcon-7b, the paper's pair and the only calibrated
    threshold in the config) is NOT in ``run_binox_pairs.py``'s DEFAULT_PAIRS --
    it has to be named, or that baseline row has no column.
  * ``falcon3-7b``'s base and instruct tokenizers are "extended" (same vocab
    plus added control tokens), which the strict tokenizer check rejects at
    startup -- a fast exit 1 with an empty column, before a single row is
    scored. It runs as its own stage carrying ``--allow-token-mismatch``, so
    that flag does not also switch the check off for the four pairs that pass
    it honestly.
  * ``add_detector_scores.py`` defaults to ``radar pangram_roberta`` and the
    Pangram repo is gated, so the default set fails on access. Only radar is
    asked for.
  * The judge is one process per judge. Three judges in the config, three runs.

WHAT IT PRODUCES, AND HOW IT KNOWS
-----------------------------------
Every stage declares the columns it is supposed to fill, and the runner counts
the filled rows of each afterwards (``parquet_utils.column_fill``). An exit code
says the subprocess ran; it does not say every row got a value -- several of
these scripts catch per-row failures and finish successfully having scored some,
none, or most of the corpus.

A holdout has no blank-text rows (``prepare_denario_detect.py`` drops empty
abstracts), so **any** Null in a produced column is a real gap, not an expected
consequence of a blank document. When a stage leaves one, the runner re-runs it
-- the scoring scripts skip rows already done, so each pass retries only the
failures -- up to ``--fill-retries`` times with a ``--fill-cooldown`` pause (for
a rate-limited endpoint to drain), then fails the pipeline. ``--fill-retries 0``
fails on the first gap. If the corpus DOES carry blank text, the runner falls
back to the lenient reading and neither retries nor fails on those rows.

A full column is checked once more: a numeric score column
(``llm_judge_score_*``, ``binoculars_score*``, ``fastdetect_crit_*``,
``detector_score_*``) that came out constant or with 99% of its rows on one
value is a scorer that fell back to a default -- the run fails on that too,
without retrying (the same prompts give the same non-answer).

NOT JUST FOR HOLDOUTS. ``--from embed`` on a corpus that is already standardised
runs the detector suite over anything -- a training corpus included. Pair it
with ``carry_suite_columns.py``, which copies the columns a sibling corpus
already scored, so only the new rows are computed.

**Stages are separate processes**, exactly as ``epai/science_data/run_pipeline.py``
runs its own: each is a CLI with its own argument parsing, its own progress bars
and its own exit codes, and a stage that dies takes nothing with it.

**Every stage is critical here.** Unlike the science_data pipeline there are no
diagnostic stages to carry on past: each column is a row of a results table, and
the training preflight refuses to start while one is missing. A partial corpus
is not a partial result, it is a run that will not start.

**Nothing is skipped because its output exists.** The scoring scripts skip
filled rows from the inside, which is the right place for that decision -- a
runner guessing from the file is how a corpus ends up half-built from two
different model pairs. Use ``--only`` / ``--from`` to resume deliberately.

RESUMING

    --from binox             skip the stages before Binoculars
    --only judge-gpt56luna    run exactly this one (retries its Null rows too)
    --skip-existing          skip stages whose columns are already filled
    --fill-retries 4         a flaky endpoint: give a stage more attempts
    --fill-retries 0         fail on the first unfilled row instead of retrying

COST. 82 rows. Every GPU stage here is dominated by model loading rather than by
scoring -- nine models to score 82 documents each -- and the three judges are
246 calls in total. Budget an afternoon, most of it spent on downloads.

Run:
    uv run python epai/ai_detection/prepare/run_holdout_pipeline.py --dry-run
    uv run python epai/ai_detection/prepare/run_holdout_pipeline.py --yes --gpu 0
    uv run python epai/ai_detection/prepare/run_holdout_pipeline.py --from binox --gpu 0
    uv run python epai/ai_detection/prepare/run_holdout_pipeline.py \
        --in data/parquet/denario_papers.parquet \
        -o data/parquet/denario_detect.parquet --yes

Then turn the block on and check before training anything:
    # cfg/ai_detection/train/clf_id/clf_id_domains_all.yaml
    #   data.eval_only.enabled: true
    uv run python epai/ai_detection/train/run_clf_in_domain.py \
        --cfg cfg/ai_detection/train/clf_id/clf_id_domains_all.yaml --dry-run

Example
    Dry run:
        uv run python epai/ai_detection/prepare/run_holdout_pipeline.py --dry-run --cpu-threads 16 --gpu 0

    Run:
        uv run python epai/ai_detection/prepare/run_holdout_pipeline.py --yes --cpu-threads 16 --gpu 0 

"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from epai.utils import notify_utils
from epai.utils.parquet_utils import column_fill

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

INTERRUPTED = 130

BASE_DIR = Path(__file__).resolve().parents[3]

DEFAULT_IN = BASE_DIR / "data" / "parquet" / "denario_papers.parquet"
DEFAULT_OUT = BASE_DIR / "data" / "parquet" / "denario_detect.parquet"

PREPARE_DIR = "epai/ai_detection/prepare"
EMBED_DIR = "epai/ai_detection/embed"
SCORE_DIR = "epai/ai_detection/score"

#: The canonical model set, shared with every other dataset — see
#: cfg/detector_suite.md. The DEFAULT_* below come from it; the literals in this
#: file are the fallback for a checkout that does not have the YAML.
SUITE_PATH = BASE_DIR / "cfg" / "detector_suite.yaml"


def _load_suite() -> dict:
    try:
        return yaml.safe_load(SUITE_PATH.read_text()) or {}
    except Exception:
        return {}


_SUITE = _load_suite()


def _suite_keys(group: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    """The ``key`` (or bare string) of each entry in one suite group."""
    items = _SUITE.get(group)
    if not items:
        return fallback
    return tuple(i["key"] if isinstance(i, dict) else str(i) for i in items)


def _suite_judge_kinds() -> dict[str, set[str]]:
    """``{kind: {judge alias, ...}}`` from the suite's ``judges`` block, so the
    runner can cap concurrency for the ``genai4science`` ones. Fallback maps the
    two known GenAI4Science judges."""
    out: dict[str, set[str]] = {}
    for entry in _SUITE.get("judges") or []:
        if isinstance(entry, dict) and "kind" in entry:
            out.setdefault(entry["kind"], set()).add(entry["key"])
    return out or {"genai4science": {"gptoss-120b", "llama33-70b"}}


#: Embedders to score. From cfg/detector_suite.yaml `embedders` — the full set,
#: the same one cfg/science_data/*.yaml carries, so a holdout matches the
#: training corpora column-for-column. The training configs still only read
#: emb_luar_crud; the rest are there so the feature axis can widen without a
#: re-score. Override with --embedders for a quick run.
DEFAULT_EMBEDDERS = _suite_keys("embedders", ("luar_crud",))

#: Binoculars pairs. `falcon-7b` FIRST and explicitly: it writes the legacy
#: `binoculars_score` column, it is the only pair with a calibrated threshold in
#: the configs, and it is absent from run_binox_pairs.py's own defaults.
DEFAULT_BINOX_PAIRS = _suite_keys(
    "binoculars_pairs",
    ("falcon-7b", "qwen25-7b", "falcon3-7b", "mistral-v03", "llama31-8b"),
)

#: Pairs whose base and instruct tokenizers are "extended" -- same vocabulary
#: plus added control tokens -- which score_parquets.py's strict tokenizer check
#: rejects at startup, before it scores a single row (a fast exit 1 with an
#: empty column, not a scoring failure). They need --allow-token-mismatch.
#:
#: They get their OWN stage rather than that flag being added to the whole run.
#: run_binox_pairs.py takes the flag once and passes it to every pair of that
#: invocation, so a single stage would also switch the check off for the pairs
#: that pass it honestly -- and the check is what stands between a genuinely
#: re-mapped token ("INCOMPATIBLE", not "extended") and a column full of
#: numbers that mean nothing. Each pair is already its own subprocess, so
#: splitting the stage costs one extra parent process and no extra model loads.
#:
#: Check a pair before adding it here, and read the verdict rather than assuming:
#:   uv run --project external/Binoculars python external/Binoculars/check_pairs.py
BINOX_EXTENDED_TOKENIZER = ("falcon3-7b", "falcon3-1b")

#: Fast-DetectGPT pairs -- kept identical to DEFAULT_BINOX_PAIRS by the suite, so
#: every binoculars__<pair> row has a fastdetect__<pair> twin (see
#: cfg/detector_suite.md). All but falcon-7b carry a placeholder threshold.
DEFAULT_FASTDETECT_PAIRS = _suite_keys(
    "fastdetect_pairs", ("falcon-7b", "qwen25-7b", "falcon3-7b", "mistral-v03"),
)

#: Pretrained detectors. NOT the script's default (`radar pangram_roberta`):
#: the Pangram repo is gated on the Hub and access is still pending, so asking
#: for it fails the stage on an access error rather than a missing model.
DEFAULT_DETECTORS = _suite_keys("detectors", ("radar",))

#: One process per judge, in the order the `llm_judge_baseline` block lists them.
#: gpt56luna leads because the cue-vector chain hangs off it.
DEFAULT_JUDGES = _suite_keys("judges", ("gpt56luna", "qwen3-14b", "sonnet5"))

#: The judge whose rationales become the cue vector. The column name carries the
#: judge, so this choice is not cosmetic: `llm_judge_cuevec_gpt56luna_v3` is the
#: column the configs' `cuevec` feature set names.
CUE_JUDGE = _SUITE.get("cue_judge", "gpt56luna")

#: Prompt version, in every judge column name. Changing it changes the columns.
PROMPT_VERSION = _SUITE.get("judge_prompt_version", "v3")

#: Rows to explain for the cue vector, with ``--select all``. Comfortably above
#: the 82 Denario papers: the reasons script caps at this per band, and a row it
#: never asked about has no cue vector, which makes the training loader drop it
#: and its whole group. Raise it for a larger holdout.
DEFAULT_PER_BAND = 5000


def rel(path: Path) -> str:
    """A path relative to the repo root where it is under it, else as given.

    ``Path.relative_to`` raises rather than falling back, and a path passed with
    ``-o /scratch/...`` is a legitimate thing to do to this script.
    """
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def judge_stem(judge: str) -> str:
    """``qwen3-14b`` -> ``qwen3_14b``: the column-name form of a judge alias.

    The registry aliases carry hyphens and the columns carry underscores, which
    is the one transformation standing between "the stage ran" and "the config
    can find what it wrote".
    """
    return judge.replace("-", "_").replace(".", "_")


@dataclass
class Stage:
    key: str
    script: str
    args: list[str] = field(default_factory=list)
    #: Columns this stage must leave filled. Checked afterwards, because an exit
    #: code says the subprocess ran and not that it did the work.
    produces: list[str] = field(default_factory=list)
    #: Whether the stage reads the parquet's `text` column. Only used for the
    #: ordering assertion below.
    reads_text: bool = True

    @property
    def path(self) -> Path:
        return BASE_DIR / self.script


def build_stages(args: argparse.Namespace) -> list[Stage]:
    """The pipeline, in order, with every stage's arguments resolved."""
    out = str(args.out)
    gpu = ["--gpu", str(args.gpu)] if args.gpu is not None else []
    yes = ["--yes"] if args.yes else []
    # Passed only to the six stages that take it. The three that do not --
    # restructure, standardize and cue-features -- run no model at all: they are
    # pandas over 82 rows, so there is no thread pool to cap and the flag would
    # be an unrecognised argument rather than a no-op.
    cpu = ["--cpu-threads", str(args.cpu_threads)] if args.cpu_threads else []

    # `--select all` explains at most `--per-band` rows; below the corpus size it
    # leaves rows with no cue vector, which the training loader drops together
    # with their whole group. Raise it to the row count when the file is already
    # there (always true for `--from embed`).
    per_band = args.per_band
    if Path(out).is_file():
        try:
            import pyarrow.parquet as _pq
            per_band = max(per_band, _pq.ParquetFile(out).metadata.num_rows)
        except Exception:
            pass

    judge_cols = [
        f"llm_judge_score_{judge_stem(j)}_{PROMPT_VERSION}" for j in args.judges
    ]

    stages = [
        # 1. Restructure. The only stage that reads the input file; everything
        #    after it works on `out` in place.
        Stage(
            "restructure",
            f"{PREPARE_DIR}/prepare_denario_detect.py",
            ["--in", str(args.src), "-o", out, "--report-lengths"],
            produces=["id", "uid", "domain", "text", "generated_by"],
            reads_text=False,
        ),

        # 2. Standardise. MUST precede every stage below: they all read `text`,
        #    and a score computed on the pre-standardised text describes a
        #    document that is no longer in the corpus. The training corpora went
        #    through this same stage, so skipping it would make standardisation
        #    itself a difference between them and this one.
        Stage(
            "standardize",
            f"{PREPARE_DIR}/standardize_text.py",
            [out],
            produces=["text", "text_raw"],
            reads_text=False,
        ),

        # 3. The feature columns. The suite's embedders; see DEFAULT_EMBEDDERS.
        Stage(
            "embed",
            f"{EMBED_DIR}/add_embeddings.py",
            [out, "--embedders", *args.embedders, *gpu, *cpu, *yes],
            produces=[f"emb_{e}" for e in args.embedders],
        ),

    ]

    # 4. Binoculars. Split in two by tokenizer verdict, not for convenience:
    #    see BINOX_EXTENDED_TOKENIZER for why the permissive flag must not cover
    #    the pairs that do not need it. Either stage is omitted when no pair
    #    asked for is on its side.
    strict = [p for p in args.binox_pairs if p not in BINOX_EXTENDED_TOKENIZER]
    extended = [p for p in args.binox_pairs if p in BINOX_EXTENDED_TOKENIZER]
    if strict:
        stages.append(Stage(
            "binox",
            f"{SCORE_DIR}/run_binox_pairs.py",
            [out, "--pairs", *strict, *gpu, *cpu, *yes],
            produces=[_binox_column(p) for p in strict],
        ))
    if extended:
        stages.append(Stage(
            "binox-extended",
            f"{SCORE_DIR}/run_binox_pairs.py",
            [out, "--pairs", *extended, "--allow-token-mismatch",
             *gpu, *cpu, *yes],
            produces=[_binox_column(p) for p in extended],
        ))

    stages += [
        # 5. Fast-DetectGPT over the same model pairs, one statistic swapped.
        #    Only the `_crit_` columns are ever read: the `_prob_` ones are Null
        #    for every pair upstream did not calibrate, and non-monotone where
        #    they exist.
        Stage(
            "fastdetect",
            f"{SCORE_DIR}/run_fastdetect_pairs.py",
            [out, "--pairs", *args.fastdetect_pairs, *gpu, *cpu, *yes],
            produces=[
                f"fastdetect_crit_{p.replace('-', '_')}" for p in args.fastdetect_pairs
            ],
        ),

        # 6. The pretrained supervised detector. See DEFAULT_DETECTORS for why
        #    the script's own default set is not used.
        Stage(
            "detectors",
            f"{SCORE_DIR}/add_detector_scores.py",
            [out, "--detectors", *args.detectors, *gpu, *cpu, *yes],
            produces=[f"detector_score_{d}" for d in args.detectors],
        ),
    ]

    # 7. One stage per judge. Separate processes because they are separate
    #    models on separate providers -- two of these are API calls and one is
    #    a local 14B on the GPU -- and because a judge that fails should not
    #    take the other two with it.
    # API judges are threaded; --workers is how many rows are in flight at once
    # (a local judge ignores it and batches). The GenAI4Science endpoint sheds
    # load under concurrency, so those get a lower cap.
    workers = ["--workers", str(args.api_workers)]
    g4s_workers = ["--workers", str(min(args.api_workers, args.g4s_workers))]
    g4s_judges = _suite_judge_kinds().get("genai4science", set())
    for judge, column in zip(args.judges, judge_cols):
        w = g4s_workers if judge in g4s_judges else workers
        stages.append(
            Stage(
                f"judge-{judge}",
                f"{SCORE_DIR}/add_llm_judge_scores.py",
                [out, "--model", judge, "--prompt", PROMPT_VERSION, *w, *cpu, *yes],
                produces=[column],
            )
        )

    # 8. The cue-vector chain. Rationales first (they need the score column to
    #    pick which rows to explain), then the eleven bits built out of the cue
    #    lists. Two stages, one feature column, and out of order it produces an
    #    empty vector rather than an error.
    #
    #    `--select all` IS NOT OPTIONAL, and it is the expensive mistake this
    #    stage exists to prevent. add_llm_judge_reasons.py defaults to sampling
    #    three bands of 100 rows, because a rationale costs output tokens on
    #    every row of a 6000-row corpus. A row nobody asked about gets no cue
    #    vector, and the training loader then drops it AND its whole group --
    #    silently, as far as the config is concerned. On 82 rows the saving is
    #    meaningless and the partial coverage would be fatal, so the whole
    #    corpus is explained.
    cue_stem = f"{judge_stem(args.cue_judge)}_{PROMPT_VERSION}"
    stages += [
        Stage(
            "judge-reasons",
            f"{SCORE_DIR}/add_llm_judge_reasons.py",
            [out, "--model", args.cue_judge, "--prompt", PROMPT_VERSION,
             "--select", "all", "--per-band", str(per_band),
             *(g4s_workers if args.cue_judge in g4s_judges else workers),
             *cpu, *yes],
            # The RATIONALE column, not the cue lists. Coverage is decided by
            # whether the judge answered at all: an explained row whose two cue
            # lists are empty is a real observation (the judge found nothing)
            # and becomes an all-zeros vector, so checking the cue lists for
            # fullness would fail a perfectly good run. See the "WHAT IS NOT
            # MISSING" section of add_cue_features.py.
            produces=[f"llm_judge_reason_{cue_stem}"],
        ),
        Stage(
            "cue-features",
            f"{SCORE_DIR}/add_cue_features.py",
            [out, "--judge", cue_stem],
            produces=[f"llm_judge_cuevec_{cue_stem}"],
        ),
    ]
    return stages


def _binox_column(pair: str) -> str:
    """The column one Binoculars pair writes.

    ``falcon-7b`` is the exception and the reason this is a function: it writes
    the unsuffixed legacy name, because every score that existed before the
    other pairs did is that pair's. See ``external/Binoculars/model_pairs.py``.
    """
    return (
        "binoculars_score" if pair == "falcon-7b"
        else f"binoculars_score_{pair.replace('-', '_')}"
    )


def select(stages: list[Stage], args: argparse.Namespace) -> list[Stage]:
    """Apply ``--from`` / ``--only``. Pipeline order is preserved either way."""
    keys = [s.key for s in stages]

    if args.only:
        unknown = [k for k in args.only if k not in keys]
        if unknown:
            raise SystemExit(f"unknown stage(s) {unknown}. Choose from: {keys}")
        # Pipeline order, not the order they were typed: these stages form a
        # chain, and running them out of order feeds one its predecessor's
        # stale output instead of the one this run produced.
        return [s for s in stages if s.key in args.only]

    if args.start_from:
        if args.start_from not in keys:
            raise SystemExit(
                f"unknown stage {args.start_from!r}. Choose from: {keys}"
            )
        stages = stages[keys.index(args.start_from):]
        # The one ordering rule that survives a resume: standardize rewrites
        # `text`, and the scoring scripts read it. Restarting after them would
        # be refused by the stage itself, but saying so here costs nothing and
        # saves reading a traceback for it.
        if any(s.key == "standardize" for s in stages) and stages[0].key != "restructure":
            print(
                f"{YELLOW}NOTE: resuming into 'standardize'. It refuses to run "
                f"once scores or embeddings exist, which is correct — a score of "
                f"the pre-standardised text is a score of a document that is no "
                f"longer in the corpus.{RESET}"
            )
    return stages


#: The text column every scoring/embedding stage reads. A row whose text is
#: blank is never scored and legitimately keeps a Null — see :func:`blank_text_count`.
TEXT_COL = "text"


def blank_text_count(out: Path) -> int:
    """Rows with no usable text — None, empty, or whitespace only.

    On an eval-only holdout this is normally 0 (``prepare_denario_detect.py``
    drops empty abstracts), which is what lets :func:`verify` treat ANY Null
    score as a real gap. When it is >0 the pipeline falls back to the lenient
    reading: those rows are expected to be Null and are not its job to fill.
    """
    import pyarrow.parquet as pq

    if not out.is_file() or TEXT_COL not in pq.ParquetFile(out).schema_arrow.names:
        return 0
    values = pq.read_table(out, columns=[TEXT_COL]).column(0).to_pylist()
    return sum(1 for v in values if v is None or not str(v).strip())


#: Numeric score columns whose values carry a per-row meaning — a constant or
#: near-constant one is a scorer that fell back to a default for every row (the
#: GenAI4Science-judge failure mode: the column is "filled", but with noise).
#: Not emb_* (a constant embedding is caught as all-zero, separately) and not
#: the string/list produced columns.
DEGENERATE_CHECK_PREFIXES = (
    "binoculars_score", "fastdetect_crit_", "detector_score_", "llm_judge_score_",
)

#: Share of a score column's rows on one value above which the run FAILS. Set
#: high on purpose — no real judge or detector clumps this hard, so a false
#: positive is not a concern. The softer "worth a look" band (80%) lives in
#: epai/scripts/check_parquet_completeness.py, which is the place for judgement
#: calls; this one only catches the unambiguous "wrote a default everywhere".
DEGENERATE_FRAC = 0.97


def degenerate_score_column(out: Path, column: str) -> str | None:
    """A message if ``column`` came out (near-)constant, else None."""
    import pyarrow.parquet as pq
    from collections import Counter

    vals = pq.read_table(out, columns=[column]).column(0).to_pylist()
    finite = [v for v in vals if isinstance(v, (int, float)) and v == v]
    if len(finite) < 10:
        return None
    counts = Counter(finite)
    top = counts.most_common(1)[0][1]
    frac = top / len(finite)
    if len(counts) == 1:
        return (f"{column}: CONSTANT — every one of {len(finite)} rows is "
                f"{finite[0]:g}. The scorer wrote one value into all of them.")
    if frac >= DEGENERATE_FRAC:
        return (f"{column}: {frac:.0%} of {len(finite)} rows on a single value "
                f"({len(counts)} distinct) — the scorer is not producing real "
                f"scores.")
    return None


def already_filled(stage: Stage, out: Path) -> bool:
    """True when every column this stage produces is already present and filled."""
    if not out.is_file() or not stage.produces:
        return False
    for column in stage.produces:
        present, filled, rows = column_fill(out, column)
        if not present or filled < rows:
            return False
    return True


def verify(stage: Stage, out: Path, n_blank_text: int) -> tuple[bool, list[str], int]:
    """Did the stage fill what it said it would? ``(complete, messages, gap)``.

    The check an exit code cannot make: several of these scripts catch per-row
    failures and exit 0 having scored nothing.

    ``complete``  every produced column is filled — or the corpus carries
                  blank-text rows, which keep a Null score by design and are not
                  this pipeline's to chase (``n_blank_text > 0``, the lenient path).
    ``gap``       rows still unfilled with no blank text to explain them. > 0
                  means re-running the stage is worth a try — the scorers skip
                  rows already done — before the pipeline gives up. ``gap == 0``
                  with ``complete`` False means the failure is not a missing row
                  (the file is absent, or a score column came out degenerate):
                  nothing a retry would fix, so fail now.
    """
    if not stage.produces:
        return True, [], 0
    if not out.is_file():
        return False, [f"{out.name} was not written"], 0

    messages: list[str] = []
    gap = 0
    for column in stage.produces:
        present, filled, rows = column_fill(out, column)
        if present and filled == rows:
            continue
        missing = rows - filled
        messages.append(
            f"{column}: not written" if not present
            else f"{column}: {filled}/{rows} filled ({missing} Null)"
        )
        gap += missing

    if messages:
        if n_blank_text > 0:
            return True, [f"{m}  (corpus has {n_blank_text} blank-text row(s) — "
                          f"tolerated)" for m in messages], 0
        return False, messages, gap

    # Every produced column is FULL. Now the check a null count misses: a score
    # column that got filled with one value for every row. A retry re-runs the
    # same prompts and gets the same result, so this fails immediately (gap 0).
    degenerate = [
        m for column in stage.produces
        if column.startswith(DEGENERATE_CHECK_PREFIXES)
        and (m := degenerate_score_column(out, column))
    ]
    if degenerate:
        return False, degenerate, 0
    return True, [], 0


def run(stage: Stage, dry_run: bool) -> tuple[int, float]:
    """Run one stage; return ``(returncode, seconds)``."""
    cmd = [sys.executable, str(stage.path), *stage.args]

    print(f"\n{BOLD}{'━' * 74}{RESET}")
    print(f"{BOLD}  {stage.key}{RESET}{DIM}   {stage.script}{RESET}")
    if stage.produces:
        print(f"{DIM}  → {', '.join(stage.produces)}{RESET}")
    print(f"{BOLD}{'━' * 74}{RESET}")
    print(f"{DIM}$ {' '.join(cmd[1:])}{RESET}\n", flush=True)

    if dry_run:
        return 0, 0.0

    start = time.monotonic()
    # No capture: stdout and stderr are inherited, so progress bars and the
    # confirmation prompts behave exactly as they do run by hand.
    proc = subprocess.run(cmd, cwd=BASE_DIR)
    return proc.returncode, time.monotonic() - start


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in", dest="src", type=Path, default=DEFAULT_IN,
                   help="Denario parquet in the citation_impact schema.")
    p.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT,
                   help="The eval-only corpus this builds. Written by stage 1, "
                        "then filled in place by every stage after it.")
    p.add_argument("--gpu", default=None, metavar="ID",
                   help="GPU to pass to the embedding and scoring stages.")
    p.add_argument("--cpu-threads", type=int, default=None, metavar="N",
                   help="Cap the CPU pools (OMP/MKL and torch's intra-op pool) "
                        "of every stage that runs a model. Same flag, same "
                        "meaning as on those scripts run by hand; the three "
                        "stages that run no model do not take it and are not "
                        "given it. Worth setting on a shared box, where an "
                        "uncapped tokenizer pool will take every core it can "
                        "see.")
    p.add_argument("--embedders", nargs="+", default=list(DEFAULT_EMBEDDERS),
                   metavar="NAME",
                   help="Embedders to compute (default: the cfg/detector_suite.yaml "
                        "set). Pass a short list for a quick run.")
    p.add_argument("--binox-pairs", nargs="+", default=list(DEFAULT_BINOX_PAIRS),
                   metavar="PAIR",
                   help="Binoculars model pairs. Keep 'falcon-7b': it writes "
                        "`binoculars_score`, the only calibrated threshold in "
                        "the configs, and it is not one of that script's own "
                        "defaults.")
    p.add_argument("--fastdetect-pairs", nargs="+",
                   default=list(DEFAULT_FASTDETECT_PAIRS), metavar="PAIR",
                   help="Fast-DetectGPT model pairs.")
    p.add_argument("--detectors", nargs="+", default=list(DEFAULT_DETECTORS),
                   metavar="NAME",
                   help="Pretrained detectors. 'pangram_roberta' is gated on "
                        "the Hub, so it is not asked for by default.")
    p.add_argument("--judges", nargs="+", default=list(DEFAULT_JUDGES),
                   metavar="ALIAS",
                   help="LLM judges, one process each. Must cover every column "
                        "the config's llm_judge_baseline block lists.")
    p.add_argument("--cue-judge", default=CUE_JUDGE, metavar="ALIAS",
                   help="Judge whose rationales become the cue vector. The "
                        "column name carries it, so this is not cosmetic.")
    p.add_argument("--per-band", type=int, default=DEFAULT_PER_BAND,
                   metavar="N",
                   help="Floor on the rows the cue-reasons stage explains "
                        "(--select all). Automatically raised to the corpus size "
                        "when the file is present, so every row gets a cue "
                        "vector — an unexplained row is dropped, with its whole "
                        "group, by the training loader.")
    p.add_argument("--api-workers", type=int, default=8, metavar="N",
                   help="Concurrent requests for every API-based stage — the "
                        "OpenAI/Anthropic judge scores AND the cue-reasons pass "
                        "(default: 8; 24 is fine for OpenAI/Anthropic). The "
                        "local judge (qwen3-14b) ignores this and batches on the "
                        "GPU; GenAI4Science judges are further capped by "
                        "--g4s-workers.")
    p.add_argument("--g4s-workers", type=int, default=6, metavar="N",
                   help="Concurrency cap for the GenAI4Science judges only "
                        "(default: 6). That endpoint sheds load under "
                        "concurrency and is a shared cluster, so more in-flight "
                        "requests do not buy throughput. Effective value: "
                        "min(this, --api-workers).")
    p.add_argument("--from", dest="start_from", metavar="STAGE",
                   help="Start here instead of at the beginning, e.g. --from binox")
    p.add_argument("--only", nargs="+", metavar="STAGE",
                   help="Run only these stages (always in pipeline order).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a stage whose columns are already present and "
                        "fully filled. Off by default: the scoring scripts "
                        "already skip filled rows from the inside, which is a "
                        "better place for that decision than a runner guessing "
                        "from the file.")
    p.add_argument("--yes", action="store_true",
                   help="Answer yes to the scoring stages' confirmation "
                        "prompts. Needed for an unattended run.")
    p.add_argument("--fill-retries", type=int, default=2, metavar="N",
                   help="When a stage exits 0 but leaves score/embedding rows "
                        "Null (and the corpus has no blank text to explain "
                        "them), re-run it up to N times — the scorers skip rows "
                        "already done, so each pass only retries the failures. "
                        "0 fails on the first gap. Default: 2.")
    p.add_argument("--fill-cooldown", type=float, default=60.0, metavar="SECONDS",
                   help="Pause before a fill retry (default: 60), long enough "
                        "for a queued API/endpoint to drain rather than be hit "
                        "again immediately.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands without running anything.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    stages = select(build_stages(args), args)
    if not stages:
        raise SystemExit("no stages selected")

    if not args.src.is_file() and any(s.key == "restructure" for s in stages):
        raise SystemExit(f"input parquet not found: {args.src}")

    print(f"{BOLD}eval-only corpus pipeline{RESET}")
    print(f"  input        {rel(args.src)}")
    print(f"  output       {rel(args.out)}")
    print(f"  gpu          {args.gpu if args.gpu is not None else 'not set'}")
    print(f"  cpu threads  {args.cpu_threads if args.cpu_threads else 'not capped'}")
    print(f"  stages       {' → '.join(s.key for s in stages)}")
    columns = [c for s in stages for c in s.produces]
    print(f"  columns      {len(columns)} — {', '.join(columns)}")
    if args.dry_run:
        print(f"\n{YELLOW}--dry-run: nothing will be executed.{RESET}")
    elif not args.yes:
        print(
            f"\n{YELLOW}NOTE: --yes not given, so the scoring stages will stop "
            f"to ask before writing.{RESET}"
        )

    # Phone pushes at every stage boundary, for the hours nobody watches this.
    # Silent unless NTFY_PIPELINE_TOPIC is set; see epai/utils/notify_utils.py.
    notify = notify_utils.configure(
        "holdout", args.out.stem, [s.key for s in stages],
        enabled=not args.dry_run,
    )
    notify.started(f"{len(stages)} stages")

    results: list[tuple[Stage, int, float, str]] = []
    failed = None
    n_blank_text: int | None = None  # computed once, after the file exists

    for stage in stages:
        if args.skip_existing and not args.dry_run and already_filled(stage, args.out):
            print(
                f"\n{DIM}  {stage.key}: {', '.join(stage.produces)} already "
                f"filled — skipped (--skip-existing){RESET}"
            )
            results.append((stage, 0, 0.0, "skipped"))
            continue

        code, note, stage_seconds = 0, "", 0.0
        for attempt in range(args.fill_retries + 1):
            try:
                code, seconds = run(stage, args.dry_run)
            except KeyboardInterrupt:
                print(f"\n{YELLOW}Interrupted during {stage.key}.{RESET}")
                code, seconds = INTERRUPTED, 0.0
            stage_seconds += seconds

            if code != 0 or args.dry_run:
                break

            # An exit code says the subprocess ran. This says it did the work.
            if n_blank_text is None:
                n_blank_text = blank_text_count(args.out)
            complete, messages, gap = verify(stage, args.out, n_blank_text)
            if complete:
                note = "; ".join(messages)
                break
            if gap == 0 or attempt >= args.fill_retries:
                code = 1
                note = "; ".join(messages) or f"{gap} row(s) unfilled"
                headline = (
                    f"exited 0 but its column(s) are not usable" if gap == 0
                    else f"could not fill every row after {args.fill_retries} retr"
                         f"{'y' if args.fill_retries == 1 else 'ies'}"
                )
                print(
                    f"\n{RED}{stage.key} {headline}:{RESET}\n    "
                    + "\n    ".join(messages)
                )
                break
            print(
                f"\n{YELLOW}{stage.key}: {gap} row(s) still Null — "
                f"{'; '.join(messages)}\n  retry {attempt + 1}/{args.fill_retries} "
                f"in {args.fill_cooldown:.0f}s (only the Null rows are re-scored)"
                f"{RESET}",
                flush=True,
            )
            notify.stage(f"{stage.key} retry {attempt + 1}/{args.fill_retries}",
                         0, seconds, critical=False)
            time.sleep(args.fill_cooldown)

        results.append((stage, code, stage_seconds, note))
        notify.stage(stage.key, code, stage_seconds, critical=True)

        if code != 0:
            failed = stage
            break

    # ------------------------------------------------------------- summary --
    print(f"\n{BOLD}{'━' * 74}{RESET}")
    print(f"{BOLD}  summary{RESET}")
    print(f"{BOLD}{'━' * 74}{RESET}")
    for stage, code, seconds, note in results:
        if note == "skipped":
            mark, colour = "skipped (already filled)", DIM
        elif code == 0:
            mark, colour = "ok", GREEN
        elif code == INTERRUPTED:
            mark, colour = "interrupted", YELLOW
        else:
            mark, colour = f"FAILED ({code})", RED
        print(f"  {stage.key:<20} {colour}{mark:<26}{RESET} {seconds:>7.1f}s")

    ran = {s.key for s, _, _, _ in results}
    for stage in stages:
        if stage.key not in ran:
            print(f"  {stage.key:<20} {DIM}{'not reached':<26}{RESET}")

    total = sum(s for _, _, s, _ in results)
    print(f"\n  total {total:.1f}s")

    if failed is not None:
        resume = (
            f"uv run python epai/ai_detection/prepare/run_holdout_pipeline.py "
            f"--from {failed.key}"
            + (f" --gpu {args.gpu}" if args.gpu is not None else "")
            + (f" --cpu-threads {args.cpu_threads}" if args.cpu_threads else "")
        )
        notify.failed(failed.key, results[-1][1], results[-1][2], resume=resume)
        print(
            f"\n{RED}Pipeline stopped at '{failed.key}'.{RESET}\n"
            f"  Every column is a row of a results table and the training "
            f"preflight refuses to start while one is missing, so there is\n"
            f"  nothing useful to do with a partial corpus. Fix the cause, then:"
            f"\n    {resume}"
        )
        sys.exit(1)

    if args.dry_run:
        print(f"\n{YELLOW}Dry run — nothing was executed.{RESET}")
        return

    notify.finished(f"corpus: {rel(args.out)}", total)
    print(
        f"\n{GREEN}Done.{RESET} {rel(args.out)} carries every "
        f"column the configs read.\n"
        f"  Turn the block on:\n"
        f"    cfg/ai_detection/train/clf_id/clf_id_domains_all.yaml"
        f"  →  data.eval_only.enabled: true\n"
        f"  Then check it before training anything:\n"
        f"    uv run python epai/ai_detection/train/run_clf_in_domain.py \\\n"
        f"        --cfg cfg/ai_detection/train/clf_id/clf_id_domains_all.yaml "
        f"--dry-run"
    )


if __name__ == "__main__":
    main()
