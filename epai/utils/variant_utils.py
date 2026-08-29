r"""The two axes a classifier config sweeps: feature sets and algorithms.

Both training runs -- the cross-domain matrix
(``epai/ai_detection/train/run_clf_cross_domain.py``) and the in-domain one
(``run_clf_in_domain.py``) -- ask the same thing of a config before they diverge:
*what should be trained?* The answer is a cross product of two lists, and one
element of it is a :class:`Variant`::

    feature_sets: [emb_luar, emb_bge]      one design matrix per entry
    algorithms:   [logistic, simple_mlp]   the registry in epai/algs

That question has nothing to do with how the variant is then evaluated -- a
source x target matrix, or a single held-out split -- so it lives here, next to
the validation that rejects a malformed axis before any data is read.
:mod:`epai.utils.cd_core` holds the other half: the corpus loading, split and
metrics the two runs share once the variants are known.

Keeping it in one place is not tidiness for its own sake. Three scripts now read
these keys (the two runners and ``run_sweep_cross_domain.py``), and the last time
this codebase let each runner own its config parsing, they drifted into accepting
slightly different files -- which is what a preflight exists to prevent.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

import numpy as np

from epai.algs import ALGORITHMS, build_algorithm
from epai.utils.cd_core import (
    DEFAULT_BINOX_COL,
    EMB_PREFIX,
    baseline_columns,
    binary_text_columns,
    binary_text_features,
    check_score_baselines,
    raw_feature_col,
)
from epai.utils.corpus_utils import CorpusSpec, check_corpus_files, check_dataset_specs
from epai.utils.preflight_utils import (
    RULE,
    Report,
    check_identity,
    check_model_block,
)
from epai.utils.train_utils import CLF_TARGET_COL


# ---------------------------------------------------------------------------
# The two axes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSet:
    """One design matrix a variant can be trained on.

    ``columns`` is the *full* stacked list -- the axis entry plus any
    ``extra_features`` and binarised text features -- so it is what both the
    loader and ``stack_features`` are handed, in that order.
    """

    id: str
    label: str
    columns: tuple[str, ...]


@dataclass
class Variant:
    """One (feature set, algorithm) pair: one matrix, one output directory."""

    id: str  # output-folder name
    label: str  # shown in the ranking and on the figures
    fs_id: str  # the FeatureSet this was built from
    features: list[str]  # the design matrix, in stacking order
    alg_name: str  # registry key in epai/algs
    params: dict = field(default_factory=dict)
    aux_col: str | None = None  # a hybrid's second input, kept out of `features`

    @property
    def hybrid(self) -> bool:
        return self.aux_col is not None


def short_name(column: str) -> str:
    """A column's display name: ``emb_luar_crud`` reads as ``luar_crud``.

    Only the prefix is dropped, and only for labels -- ids stay the literal
    column name so a results directory can be traced back to the parquet column
    without guessing.
    """
    return column[len(EMB_PREFIX):] if column.startswith(EMB_PREFIX) else column


def entry_columns(entry) -> tuple[list[str], str | None, str | None]:
    """One ``feature_sets`` entry as ``(columns, id, label)``. Unusable -> ([], ...).

    Three forms, in ascending order of how much they say:

        - emb_luar_crud                              a one-column set
        - [emb_luar_crud, binoculars_score]          a set of several columns
        - {columns: [...], id: ..., label: ...}      the same, named explicitly

    The mapping form mirrors an ``algorithms`` entry, so both axes are written
    the same way and ``id`` means the same thing on each: the output folder.
    """
    if isinstance(entry, str):
        return ([entry] if entry.strip() else []), None, None
    if isinstance(entry, (list, tuple)):
        return [c for c in entry if isinstance(c, str) and c.strip()], None, None
    if isinstance(entry, dict):
        raw = entry.get("columns")
        cols = [raw] if isinstance(raw, str) else list(raw or [])
        cols = [c for c in cols if isinstance(c, str) and c.strip()]
        ident = entry.get("id") if isinstance(entry.get("id"), str) else None
        label = entry.get("label") if isinstance(entry.get("label"), str) else None
        return cols, ident, label
    return [], None, None


def build_feature_sets(cfg: dict) -> list[FeatureSet]:
    """The feature axis: one design matrix per ``feature_sets`` entry.

    One entry is always one variant, whether it names one column or five. That is
    the whole point of the key: the two it replaced (``features``, which combined
    its list into one matrix, and ``embeddings``, which made one matrix per item)
    took the same syntax and meant opposite things by it.

    ``extra_features`` and the derived ``extra_binary_text`` features are appended
    to every set, so a comparison across sets shifts them all equally and stays
    fair. Column names are taken literally -- nothing is prefixed for you, since a
    set may mix an embedding with a scalar score.
    """
    extra = [c for c in (cfg.get("extra_features") or []) if isinstance(c, str)]
    tail = extra + binary_text_features(cfg)

    sets: list[FeatureSet] = []
    for entry in cfg.get("feature_sets") or []:
        cols, ident, label = entry_columns(entry)
        if not cols:
            continue
        short = [short_name(c) for c in cols]
        sets.append(
            FeatureSet(
                # The id names a directory, so it stays the literal column(s);
                # the label is what a human reads off a figure.
                id=ident or "+".join(cols),
                label=label or " + ".join(short),
                columns=tuple(dict.fromkeys(cols + tail)),
            )
        )
    return sets


def spec_aux_col(entry: dict) -> str | None:
    """The auxiliary column one ``algorithms`` entry asks for, or None.

    ``aux: <column>`` names it and ``aux: true`` takes the default Binoculars
    column; an absent key means the entry is a plain algorithm on the feature
    set alone (writing ``false`` is rejected by the preflight rather than read
    as "no" -- drop the key instead). A non-None value makes the entry a
    *hybrid*: it sees that column on top of the shared feature set, which the
    ranking and the figures mark.

    ``binoculars`` is the older spelling of the same key and still works, since
    every existing config and every sweep leaderboard entry is written with it.
    It reads wrong on the algorithms this key now mostly serves -- ``fusion_mlp``
    and ``gated_mlp`` take a judge's cue vector, not a score -- so ``aux`` is
    what new configs should say. When both appear, ``aux`` wins and the
    preflight rejects the entry if they disagree, rather than silently picking.
    """
    value = entry.get("aux")
    if value is None:
        value = entry.get("binoculars")
    if value is True:
        return DEFAULT_BINOX_COL
    if isinstance(value, str) and value.strip():
        return value
    return None


#: The old name of :func:`spec_aux_col`, kept so an import outside this package
#: does not break on the rename.
spec_binox_col = spec_aux_col


def spec_feature_sets(entry: dict) -> tuple[str, ...] | None:
    """The feature-set ids one ``algorithms`` entry restricts itself to, or None.

    ``None`` -- the key is absent -- means "every feature set", which is the
    cross product this config's two axes have always produced and what every
    entry without the key keeps doing.

    The key exists for an algorithm that is only *meaningful* on some of them.
    ``fusion_mlp`` and ``gated_mlp`` name their second input on the entry itself
    (``aux``), so the grid would otherwise hand them a design matrix holding the
    very column they were already given as ``Z`` -- a cell that trains, lands in
    the ranking, and is marked as a hybrid while having seen nothing extra. A
    restriction says which ``X`` the entry is valid for, so the entry owns both
    of its inputs rather than half of one.

    An unusable value (``[]``, a mapping, a list of non-strings) returns an empty
    tuple, which produces no variants at all and which
    :func:`check_algorithms` reports rather than letting the entry vanish.
    """
    value = entry.get("feature_sets")
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(v for v in value if isinstance(v, str) and v.strip())


def algorithm_specs(cfg: dict) -> list[dict]:
    """Normalise ``algorithms`` to ``{id, name, label, params, aux, feature_sets}``.

    An entry may be a bare string (``"logistic"``) or a mapping with ``name``
    and optional ``id`` (output-folder name, so two entries of the same
    algorithm with different params do not collide), ``label`` (shown on figures),
    ``params`` (hyper-parameters passed to the algorithm), ``aux`` (an
    auxiliary column handed to the algorithm beside the design matrix -- only
    algorithms whose class sets ``uses_aux`` accept one; spelled ``binoculars``
    in older configs, see :func:`spec_aux_col`) and ``feature_sets`` (the ids
    this entry restricts itself to; see :func:`spec_feature_sets`).
    """
    specs: list[dict] = []
    for entry in cfg.get("algorithms", []) or []:
        if isinstance(entry, str):
            specs.append(
                {
                    "id": entry,
                    "name": entry,
                    "label": entry,
                    "params": {},
                    "aux": None,
                    "feature_sets": None,
                }
            )
        elif isinstance(entry, dict) and isinstance(entry.get("name"), str):
            name = entry["name"]
            specs.append(
                {
                    "id": entry.get("id", name),
                    "name": name,
                    "label": entry.get("label", name),
                    "params": entry.get("params") or {},
                    "aux": spec_aux_col(entry),
                    "feature_sets": spec_feature_sets(entry),
                }
            )
    return specs


def variant_pairs(cfg: dict) -> list[tuple[FeatureSet, dict]]:
    """Every (feature set, algorithm) pair this config asks for, in run order.

    The cross product of the two axes, minus the cells an entry's
    ``feature_sets`` restriction excludes (see :func:`spec_feature_sets`). A grid
    with a restriction in it is ragged, which the figures show as a missing cell
    -- honest, since that cell was never trained.

    A restriction changes which cells exist, never what any of them is trained
    on: the missing-value filter is still built per feature set from
    :func:`variant_columns`, over every auxiliary column in the config. Two cells
    in one ranking must stand on the same rows whether or not they happen to use
    the same inputs.
    """
    algs = algorithm_specs(cfg)
    return [
        (fs, alg)
        for fs in build_feature_sets(cfg)
        for alg in algs
        if alg["feature_sets"] is None or fs.id in alg["feature_sets"]
    ]


def build_variants(cfg: dict) -> list[Variant]:
    """The pairs of :func:`variant_pairs`, named by whichever axis actually varies.

    A pure embedding sweep keeps writing ``<alg_id>/emb_luar/`` and a pure
    algorithm sweep ``<alg_id>/logistic/`` -- the axis that is held fixed adds
    nothing to a name, and pinning it there would only make every existing
    results path longer. Only a genuine 2-D run needs both halves.

    "Varies" is counted over the pairs that survive the restrictions rather than
    over the two lists, so a config whose entries are all pinned to one feature
    set is named like the one-feature-set config it effectively is. With no
    restriction anywhere the two counts coincide, so no existing config's paths
    move.
    """
    pairs = variant_pairs(cfg)
    n_fs = len({fs.id for fs, _ in pairs})
    n_alg = len({alg["id"] for _, alg in pairs})
    both = n_fs > 1 and n_alg > 1

    variants: list[Variant] = []
    for fs, alg in pairs:
        if both:
            vid, label = f"{fs.id}__{alg['id']}", f"{fs.label} — {alg['label']}"
        elif n_fs > 1:
            vid, label = fs.id, fs.label
        else:
            vid, label = alg["id"], alg["label"]
        variants.append(
            Variant(
                id=vid,
                label=label,
                fs_id=fs.id,
                features=list(fs.columns),
                alg_name=alg["name"],
                params=alg["params"],
                aux_col=alg["aux"],
            )
        )
    return variants


def comparison_noun(cfg: dict) -> str:
    """What the varying axis is called, for titles and figure filenames.

    A run that swept algorithms should say so on its figures rather than call
    them "variants"; only a 2-D run has no better word.
    """
    pairs = variant_pairs(cfg)
    n_fs = len({fs.id for fs, _ in pairs})
    n_alg = len({alg["id"] for _, alg in pairs})
    if n_fs > 1 and n_alg > 1:
        return "variant"
    return "feature set" if n_fs > 1 else "algorithm"


def aux_columns(cfg: dict) -> list[str]:
    """Every auxiliary column any algorithm in this config asks for, deduplicated.

    These are *not* part of any feature set: they never enter a design matrix, so
    the plain algorithms are unaffected by their presence. They do join the
    missing-value filter, though -- see ``variant_columns`` -- so that every
    algorithm in the config trains and is scored on an identical row set.
    """
    cols = [s["aux"] for s in algorithm_specs(cfg) if s["aux"]]
    return list(dict.fromkeys(cols))


def variant_columns(fs: FeatureSet, cfg: dict) -> list[str]:
    """Columns one feature set is read and missing-filtered on: features + auxiliary.

    One list, used by the preflight and by the run alike, so the rows a hybrid
    algorithm trains on are exactly the rows the plain ones train on. Filtering
    per algorithm instead would score them on different subsets and quietly make
    the comparison table meaningless.

    Deliberately not narrowed by an entry's ``feature_sets`` restriction either.
    A feature set that no hybrid runs on is still filtered on every auxiliary
    column in the config, so its cells stand on the same rows as the cells that
    do use one -- one row set per config, whichever cells exist.
    """
    return list(dict.fromkeys(list(fs.columns) + aux_columns(cfg)))


def load_columns(cfg: dict) -> list[str]:
    """Every column any variant of this config is filtered on, deduplicated.

    Used by the file check and by ``run_sweep_cross_domain.py``, whose base
    config always carries a single ``features`` set.
    """
    cols: list[str] = []
    for fs in build_feature_sets(cfg):
        cols.extend(variant_columns(fs, cfg))
    return list(dict.fromkeys(cols))


def report_columns(cfg: dict) -> list[str]:
    """Columns a run describes its split by, from ``report_by``.

    Read and carried beside the design matrix, never into it: ``domain`` tells
    you whether the held-out half looks like the training half, and would be a
    leak if the model could see it. ``GroupShuffleSplit`` groups but does not
    stratify, so without this a pooled run says nothing about its own balance.
    """
    return [c for c in (cfg.get("report_by") or []) if isinstance(c, str) and c.strip()]


def resolved_params(variant: Variant, seed: int) -> dict:
    """Every hyper-parameter one variant will actually train with, defaults included.

    A config carries only what it overrides, so ``variant.params`` alone does not
    say which model ran -- and the interesting settings now come from
    ``run_sweep_cross_domain.py``, transcribed by hand. Asking the algorithm to
    resolve them makes that visible: it is printed before each matrix and stored
    in the payload as ``resolved_params``, so a results directory records the
    model that was fitted rather than the diff against defaults.

    Constructing the algorithm is cheap (no network is built until ``fit``), and
    it never raises here: the preflight has already rejected anything unbuildable,
    and a run that got this far must not die over a display string.
    """
    try:
        return build_algorithm(
            variant.alg_name, seed, **variant.params
        ).resolved_params()
    except Exception:
        return dict(variant.params)


def check_feature_sets(cfg: dict, report: Report) -> None:
    """Validate the ``feature_sets`` list: entry shapes, columns, unique ids."""
    raw = cfg["feature_sets"]
    if not isinstance(raw, (list, tuple)) or not raw:
        report.error(
            "'feature_sets' must be a non-empty list; every entry is one design "
            "matrix (a column name, a list of column names, or a mapping with "
            "'columns' and an optional 'id' / 'label')"
        )
        return

    for i, entry in enumerate(raw):
        cols, ident, _ = entry_columns(entry)
        if not isinstance(entry, (str, list, tuple, dict)):
            report.error(
                f"feature_sets[{i}] must be a column name, a list of column names, "
                f"or a mapping with 'columns', got {type(entry).__name__}"
            )
            continue
        if isinstance(entry, dict) and "columns" not in entry:
            report.error(f"feature_sets[{i}] is a mapping but has no 'columns'")
            continue
        if not cols:
            report.error(f"feature_sets[{i}] names no usable column")
            continue
        if ident is not None and not ident.strip():
            report.error(f"feature_sets[{i}]: 'id' must be a non-empty string")
        dupes = sorted({c for c in cols if cols.count(c) > 1})
        if dupes:
            # Stacking a column twice doubles its weight in a distance or a
            # penalty, silently, so it is a mistake rather than a shorthand.
            report.error(f"feature_sets[{i}] repeats column(s) {dupes}")

    ids = [fs.id for fs in build_feature_sets(cfg)]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        report.error(
            f"feature_sets ids must be unique (set an 'id' on an entry to "
            f"disambiguate): {dupes}"
        )


def check_algorithms(cfg: dict, report: Report) -> None:
    """Validate the ``algorithms`` list: names, params, and the hybrid contract."""
    raw = cfg["algorithms"]
    specs = algorithm_specs(cfg)
    if not isinstance(raw, (list, tuple)) or not raw:
        report.error(
            f"'algorithms' must be a non-empty list. Choose from: {sorted(ALGORITHMS)}"
        )
        return
    if len(specs) != len(raw):
        report.error(
            "every 'algorithms' entry must be a name string or a mapping with a "
            "string 'name'"
        )
        return

    for entry, s in zip(raw, specs):
        if s["name"] not in ALGORITHMS:
            report.error(
                f"unknown algorithm {s['name']!r}. Choose from: {sorted(ALGORITHMS)}"
            )
            continue
        if not isinstance(s["params"], dict):
            report.error(f"{s['id']}: 'params' must be a mapping")
            continue

        # An auxiliary column and an algorithm that takes one must agree, in
        # both directions: an 'aux' key on an algorithm that ignores it would
        # silently not be a hybrid, and a hybrid without one cannot fit.
        uses_aux = bool(getattr(ALGORITHMS[s["name"]], "uses_aux", False))
        keys = ["aux", "binoculars"] if isinstance(entry, dict) else []
        given = {k: entry[k] for k in keys if entry.get(k) is not None}
        if len(given) == 2 and given["aux"] != given["binoculars"]:
            # One entry naming two different columns has no reading that is
            # obviously right, and picking 'aux' silently would train on a column
            # the config also names elsewhere.
            report.error(
                f"{s['id']}: 'aux' and 'binoculars' are the same key and disagree "
                f"({given['aux']!r} vs {given['binoculars']!r}); keep 'aux' only"
            )
        elif given and not s["aux"]:
            key = "aux" if "aux" in given else "binoculars"
            report.error(
                f"{s['id']}: {key!r} must be true or a column name, got "
                f"{given[key]!r}"
            )
        elif s["aux"] and not uses_aux:
            report.error(
                f"{s['id']}: {s['name']!r} does not take an auxiliary column, so "
                f"'aux' would be ignored — drop it, or use an algorithm "
                f"that sets uses_aux"
            )
        elif uses_aux and not s["aux"]:
            report.error(
                f"{s['id']}: {s['name']!r} needs an auxiliary column — set "
                f"'aux: <column>' on this entry (e.g. "
                f"'aux: {DEFAULT_BINOX_COL}')"
            )

        # A restriction must name feature sets this config declares: a typo that
        # matched nothing would silently drop every cell of this entry, which
        # looks exactly like an algorithm that was never configured.
        restrict = s["feature_sets"]
        if restrict is not None:
            known = [fs.id for fs in build_feature_sets(cfg)]
            if not restrict:
                report.error(
                    f"{s['id']}: 'feature_sets' names no usable feature-set id "
                    f"(it takes a string or a list of them, from {known}); drop "
                    f"the key to run on every feature set"
                )
            for fs_id in restrict:
                if fs_id in known:
                    continue
                close = difflib.get_close_matches(fs_id, known, n=1)
                hint = f" (did you mean {close[0]!r}?)" if close else ""
                report.error(
                    f"{s['id']}: 'feature_sets' names {fs_id!r}, which is not a "
                    f"feature set of this config{hint}. Declared: {known}"
                )

        # Build it and check its hyper-parameter *values* here, so a name the
        # algorithm does not read (a mistranscribed sweep winner) or an
        # out-of-range value fails now rather than inside the first fit --
        # or, worse, trains the default model while the config claims
        # otherwise. See ClfAlgorithm's docstring.
        try:
            alg = build_algorithm(s["name"], cfg.get("seed", 0), **s["params"])
            alg.validate_params()
        except Exception as exc:  # a known name that still cannot be built
            report.error(f"{s['id']}: cannot build {s['name']!r}: {exc}")

    # A feature set every entry has restricted itself away from is loaded, row-
    # filtered and then trained on by nothing. That is legal -- it still widens
    # the missing-value filter for the sets that do run -- but it is far more
    # often a leftover entry or a restriction that named the wrong id, so say so
    # rather than let a config quietly describe an axis it does not run.
    used = {fs.id for fs, _ in variant_pairs(cfg)}
    idle = [fs.id for fs in build_feature_sets(cfg) if fs.id not in used]
    if idle and used:
        report.warn(
            f"feature set(s) {idle} have no algorithm: every entry's "
            f"'feature_sets' restriction excludes them, so nothing will be "
            f"trained on them (their columns still join the row filter)"
        )


def fit_alg(alg, X: np.ndarray, y: np.ndarray, Z: np.ndarray | None):
    """Fit, passing the auxiliary matrix only to algorithms that declare it.

    Keeps the plain ``fit(X, y)`` signature intact for every algorithm that does
    not set ``uses_aux``; ``Z`` is None for those and never reaches them.
    """
    return alg.fit(X, y, Z) if alg.uses_aux else alg.fit(X, y)


def predict_alg(alg, X: np.ndarray, Z: np.ndarray | None) -> np.ndarray:
    """Predict, mirroring ``fit_alg``'s two signatures."""
    return alg.predict(X, Z) if alg.uses_aux else alg.predict(X)


def score_alg(alg, X: np.ndarray, Z: np.ndarray | None) -> np.ndarray | None:
    """The continuous machine-ness score, or None if this algorithm has none.

    Mirrors ``predict_alg``'s two signatures. AUROC and TPR@FPR need a ranking,
    which a label vector cannot supply, so every runner asks for one here and
    falls back to the label-only metrics when the answer is None. Returning None
    rather than raising is deliberate: a scoreless algorithm should cost its
    variant the score metrics, not the whole matrix.
    """
    try:
        return alg.decision_score(X, Z) if alg.uses_aux else alg.decision_score(X)
    except NotImplementedError:
        return None


# ---------------------------------------------------------------------------
# Preflight: the config checks both runners share
# ---------------------------------------------------------------------------


def check_config(cfg: dict, report: Report, data_check=check_dataset_specs) -> None:
    """Validate one config's shape and values, without touching any file.

    Shared by both runners: the axes, the identity keys, the model block and the
    baseline blocks mean the same thing whether the config will produce a
    transfer matrix or a single held-out split. ``data`` is the one part that
    differs -- a list of corpora to put in a matrix, or one corpus to train on --
    so its check is passed in (``check_dataset_specs`` for a matrix run,
    ``check_in_domain_corpora`` for the in-domain one).
    """
    if not isinstance(cfg, dict):
        report.error(f"config must be a YAML mapping, got {type(cfg).__name__}")
        return

    # The algorithm used to live here, one per config. Reading it silently would
    # train whatever `algorithms` happens to say while the file claims otherwise.
    if isinstance(cfg.get("model"), dict) and "ml_alg" in cfg["model"]:
        report.error(
            "'model.ml_alg' has moved: give the algorithm as an entry in the "
            "top-level 'algorithms' list instead (e.g. `algorithms: [logistic]`), "
            "so one config can compare several"
        )

    # 'features' combined its list into one matrix; 'embeddings' made one matrix
    # per item. Same syntax, opposite meaning -- which is why they are now one
    # key whose entries are each exactly one matrix. Reading either silently
    # would run a different experiment from the one the file describes.
    for old, hint in (
        ("features", "wrap the list in one entry: `feature_sets: [[a, b]]`"),
        ("embeddings", "the list becomes the entries: `feature_sets: [a, b]`"),
    ):
        if old in cfg:
            report.error(
                f"{old!r} has been replaced by 'feature_sets', where every entry is "
                f"one design matrix (a column, a list of columns, or a mapping with "
                f"'columns'/'id'/'label') — {hint}"
            )

    for key in ("alg_id", "alg_label", "seed", "feature_sets", "algorithms", "data", "model"):
        if key not in cfg:
            report.error(f"missing top-level key {key!r}")
    if report.errors:
        return  # everything below indexes into these keys

    check_identity(cfg, report)
    n_errors = len(report.errors)
    check_feature_sets(cfg, report)

    extra = cfg.get("extra_features") or []
    if not isinstance(extra, (list, tuple)) or not all(
        isinstance(c, str) for c in extra
    ):
        report.error("'extra_features' must be a list of column names")

    extra_bin = cfg.get("extra_binary_text") or []
    if not isinstance(extra_bin, (list, tuple)) or not all(
        isinstance(c, str) and c.strip() for c in extra_bin
    ):
        report.error("'extra_binary_text' must be a list of column names")

    report_by = cfg.get("report_by") or []
    if not isinstance(report_by, (list, tuple)) or not all(
        isinstance(c, str) and c.strip() for c in report_by
    ):
        report.error("'report_by' must be a list of column names")

    check_algorithms(cfg, report)
    check_score_baselines(cfg, report)
    data_check(cfg, report)
    check_model_block(cfg, report)

    # Each variant names an output directory; a repeat would overwrite one. Only
    # meaningful once both axes are known to be well-formed: enumerating the
    # variants of a malformed axis reports collisions that are artifacts of the
    # earlier error rather than anything to fix.
    if len(report.errors) == n_errors:
        ids = [v.id for v in build_variants(cfg)]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            report.error(
                f"variant ids must be unique (set an 'id' on an algorithms entry "
                f"to disambiguate): {dupes}"
            )


def check_files(cfg: dict, specs: list[CorpusSpec], report: Report) -> None:
    """Every corpus exists, opens, holds its columns, and has its `where` values.

    ``specs`` is passed in rather than derived: a cross-domain run checks the
    ``datasets`` list, an in-domain run its single corpus, and the columns they
    need are otherwise identical.
    """
    group_col = cfg["data"]["group_col"]
    # A missing `emb_*` column has an obvious fix (run add_embeddings.py), so it
    # is reported separately from a structurally missing one.
    emb_cols = [c for c in load_columns(cfg) if c.startswith(EMB_PREFIX)]
    # A `<col>__bin` feature is derived from the raw text column of the same stem.
    needed = list(
        dict.fromkeys(
            [group_col, CLF_TARGET_COL]
            + [raw_feature_col(c) for c in load_columns(cfg)]
            + binary_text_columns(cfg)
            + baseline_columns(cfg)
            + report_columns(cfg)
        )
    )
    check_corpus_files(specs, needed, report, emb_cols=emb_cols, rule=RULE)
