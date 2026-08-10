"""Validating an embedder selection, for the pipelines that carry one in config.

Both full pipelines let a config name which embedders their embed stage runs, and
both hand the list to ``epai/ai_detection/embed/add_embeddings.py --embedders``.
That script validates the names itself, but only once it is running — which on a
pipeline is after the corpus has been built and scored. These two functions let a
runner reject a typo in its preflight instead, hours earlier.

The split is deliberate:

  - :func:`validate_embedders` checks the *shape* (a non-empty list of distinct
    non-empty strings) with no imports at all, so a config module can call it;
  - :func:`check_embedder_names` checks the names against the canonical
    ``EMB_TYPES``, which means importing ``embedding_utils`` and therefore torch.
    Call it from a preflight, not at module import.
"""

from __future__ import annotations


def validate_embedders(value: object, where: str) -> list[str] | None:
    """Shape-check a configured embedder list; returns it, or None if unset.

    ``None`` means "not configured", which every caller reads as "use the
    script's own default list". An empty list is an error rather than a synonym
    for that: writing ``embedders: []`` reads like "no embedders", and silently
    running thirteen of them instead is the opposite of what it says.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(
            f"{where} must be a list of embedder names, or null to use the "
            f"default list in add_embeddings.py"
        )
    if not value:
        raise ValueError(
            f"{where} is an empty list. Remove the key (or set it to null) to run "
            f"the default list; an empty list would mean embedding nothing."
        )
    bad = [v for v in value if not isinstance(v, str) or not v.strip()]
    if bad:
        raise ValueError(f"{where} contains non-string or empty entries: {bad}")
    names = [v.strip() for v in value]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"{where} contains duplicates: {duplicates}")
    return names


def check_embedder_names(names: list[str] | None) -> None:
    """Reject names ``add_embedding`` would not accept. Raises ``ValueError``.

    Imports ``embedding_utils`` lazily: it pulls in torch, which is fine in a
    preflight and much too heavy for a config module that every stage loads.
    """
    if not names:
        return
    from epai.utils.embedding_utils import EMB_TYPES

    unknown = [n for n in names if n not in EMB_TYPES]
    if unknown:
        raise ValueError(
            f"unknown embedder(s) {unknown}. Choose from: {', '.join(EMB_TYPES)}"
        )
