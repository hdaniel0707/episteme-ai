r"""Canonical column list for the 4 core AI-detection parquets -- see
``cfg/ai_detection/core_parquet_columns.yaml`` for the write-up.

``canonical_columns()`` is the exact, ordered, stem-agnostic list every one
of the 4 parquets must carry identically (meta_common + the detector-suite
block, derived from ``cfg/detector_suite.yaml`` membership + this file's own
order lists, + the literal ``judge_columns_canonical`` list). A column
legitimately specific to one corpus lives in ``per_stem_extra`` instead --
allowed there, not required anywhere else, never flagged.
``check_core_columns.py`` is the only consumer.
"""

from __future__ import annotations

from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parents[3]
CORE_CFG_PATH = BASE_DIR / "cfg" / "ai_detection" / "core_parquet_columns.yaml"
DETECTOR_SUITE_PATH = BASE_DIR / "cfg" / "detector_suite.yaml"

CORE_CFG = yaml.safe_load(CORE_CFG_PATH.read_text())
DETECTOR_SUITE = yaml.safe_load(DETECTOR_SUITE_PATH.read_text())

CORE_STEMS: list[str] = CORE_CFG["stems"]
#: opt-in only -- pass via check_core_columns.py --only, never swept by a
#: bare (no --only) run, same convention as score_logjudge_ord2.py's
#: REPHRASE_STEMS. Checked against the SAME canonical_columns() as the 4
#: core stems (MG-Reph inherits the full core schema plus its own extras,
#: per_stem_extra below).
REPHRASE_STEMS: list[str] = CORE_CFG.get("rephrase_stems", [])
ALL_STEMS: list[str] = CORE_STEMS + REPHRASE_STEMS
PRIMARY_KEY: str = CORE_CFG["primary_key"]

#: Columns that are null/empty on every row on purpose (not a capture bug --
#: see core_parquet_columns.yaml's "allow_null" comment for why each one is
#: there). check_core_columns.py downgrades an otherwise-ERROR "entirely
#: null" finding to INFO for these.
ALLOW_NULL: set[str] = set(CORE_CFG.get("allow_null", []))


def _key(pair: dict) -> str:
    return pair["key"].replace("-", "_")


def suite_membership() -> dict[str, set[str]]:
    """The *set* of names cfg/detector_suite.yaml says should exist per
    group, normalized to the underscored form column names use. Compared
    against the ``*_order`` lists in core_parquet_columns.yaml by
    :func:`suite_order_drift` -- the order lists are what actually drive
    column naming (see :func:`suite_columns`), so a model added to one file
    and not the other would otherwise just silently go uncovered."""
    return {
        "embedders": set(DETECTOR_SUITE["embedders"]),
        "model_pairs": {_key(p) for p in DETECTOR_SUITE["binoculars_pairs"]}
        | {_key(p) for p in DETECTOR_SUITE["fastdetect_pairs"]},
        "detectors": set(DETECTOR_SUITE["detectors"]),
    }


def suite_order_drift() -> dict[str, tuple[set[str], set[str]]]:
    """For each suite group, ``(missing_from_order, extra_in_order)`` --
    non-empty only when core_parquet_columns.yaml's order list and
    detector_suite.yaml's membership have drifted apart."""
    membership = suite_membership()
    order = {
        "embedders": set(CORE_CFG["embedders_order"]),
        "model_pairs": set(CORE_CFG["model_pair_order"]),
        "detectors": set(CORE_CFG["detectors_order"]),
    }
    return {
        g: (membership[g] - order[g], order[g] - membership[g])
        for g in membership
        if membership[g] != order[g]
    }


def suite_columns() -> list[str]:
    """Canonical embedders -> binoculars_pairs -> fastdetect_pairs -> detectors
    column names, in the order set by core_parquet_columns.yaml's
    ``embedders_order`` / ``model_pair_order`` / ``detectors_order``. Unlike
    judges, every combination here is legitimately exact -- there's no
    partial-coverage concept for a Binoculars/Fast-DetectGPT pair or an
    embedder -- so generating this list is safe and isn't the
    shape-matching problem judge_columns_canonical exists to avoid."""
    cols: list[str] = [f"emb_{name}" for name in CORE_CFG["embedders_order"]]

    pairs = CORE_CFG["model_pair_order"]
    cols += ["binoculars_score"] + [f"binoculars_score_{k}" for k in pairs[1:]]

    for k in pairs:
        cols += [f"fastdetect_{stat}_{k}" for stat in CORE_CFG["fastdetect_stat_order"]]

    for name in CORE_CFG["detectors_order"]:
        cols += [f"detector_{stat}_{name}" for stat in CORE_CFG["detector_stat_order"]]

    return cols


def canonical_columns() -> list[str]:
    """The exact, ordered column list required in every one of the 4
    parquets, identically: meta_common -> detector-suite block ->
    judge_columns_canonical (all three literal or exactly-generated, see
    core_parquet_columns.yaml's header comment for why nothing here is
    shape/pattern-matched)."""
    return list(CORE_CFG["meta_common"]) + suite_columns() + list(CORE_CFG["judge_columns_canonical"])


def allowed_extra(stem: str) -> set[str]:
    """Columns allowed on ``stem`` in addition to canonical_columns() --
    not required there, not required anywhere else, never flagged as extra
    on the stem(s) that list them."""
    return set(CORE_CFG.get("per_stem_extra", {}).get(stem, []))


def expected_order(stem: str, actual_cols: list[str]) -> list[str]:
    """Full expected column order for ``stem``: canonical_columns()
    restricted to columns actually present, in canonical order, then every
    other actual column (per_stem_extra or genuinely unrecognized) appended
    in whatever order it already has -- only the canonical block's order is
    enforced, see the yaml's header comment."""
    canon = canonical_columns()
    core = [c for c in canon if c in actual_cols]
    canon_set = set(canon)
    rest = [c for c in actual_cols if c not in canon_set]
    return core + rest
