"""Loading the multi parquet and cutting it into human-vs-one-generator slices.

The only input trainv2 accepts is a ``*_multi_*`` parquet: ``human`` plus two or
more generators, one row per ``(source document, generator)``, every row of a
document sharing one ``group_col`` value. :func:`load_multi` refuses anything
else — a single-generator ("pairwise") file cannot answer "which generator is
more detectable".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .core import POOLED

GEN_COL = "generated_by"


def _is_missing(value) -> bool:
    """True if an embedding cell is null, empty, or holds a non-finite entry."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return True
        return arr.size == 0 or not np.isfinite(arr).all()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


class MultiCorpus:
    """A loaded multi parquet, ready to slice.

    ``generators`` is every non-human label, sorted. ``domains`` is every domain
    plus :data:`~epai.ai_detection.trainv2.core.POOLED`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        group_col: str,
        human_label: str,
        domain_col: str,
        source: str,
    ):
        self.df = df
        self.group_col = group_col
        self.human_label = human_label
        self.domain_col = domain_col
        self.source = source
        self.generators = sorted(set(df[GEN_COL].unique()) - {human_label})
        self.domains = sorted(df[domain_col].unique().tolist()) + [POOLED]

    def slice(
        self, generator: str, domain: str, embedding: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """One binary training set: human vs ``generator``, within ``domain``.

        Returns ``(X, y, groups, info)`` where ``y`` is 0 human / 1 machine,
        ``groups`` is the group id per row, and ``info`` records how many rows
        and groups survived the missing-embedding drop.

        A group is dropped whole if any of its rows has a missing embedding, so
        the human original and its rewrite are never separated.
        """
        m = self.df[GEN_COL].isin([self.human_label, generator])
        if domain != POOLED:
            m &= self.df[self.domain_col] == domain
        sub = self.df.loc[m, [self.group_col, GEN_COL, embedding]]

        bad = sub[embedding].map(_is_missing)
        n_missing = int(bad.sum())
        if n_missing:
            bad_groups = sub.loc[bad, self.group_col].unique()
            sub = sub[~sub[self.group_col].isin(bad_groups)]
        sub = sub.reset_index(drop=True)

        X = np.asarray(np.stack(sub[embedding].to_numpy()), dtype=np.float32)
        y = (sub[GEN_COL].to_numpy() != self.human_label).astype(int)
        groups = sub[self.group_col].to_numpy()
        info = {
            "n_rows": int(len(sub)),
            "n_groups": int(len(np.unique(groups))),
            "n_missing_rows": n_missing,
        }
        return X, y, groups, info


def load_multi(
    parquet_path: str | Path,
    *,
    embeddings: list[str],
    group_col: str,
    human_label: str,
    domain_col: str,
) -> MultiCorpus:
    """Read the columns trainv2 needs and verify the file is a multi parquet."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"parquet not found: {path}")

    need = list(dict.fromkeys([group_col, GEN_COL, domain_col, *embeddings]))
    try:
        df = pd.read_parquet(path, columns=need)
    except Exception as exc:  # missing column, unreadable file
        raise ValueError(f"cannot read {path} with columns {need}: {exc}") from exc

    labels = sorted(df[GEN_COL].unique().tolist())
    generators = [g for g in labels if g != human_label]
    if human_label not in labels:
        raise ValueError(
            f"{path.name}: no {human_label!r} rows in {GEN_COL!r} (labels: {labels})"
        )
    if len(generators) < 2:
        raise ValueError(
            f"{path.name}: only {len(generators)} generator(s) ({generators}). "
            f"trainv2 needs a *_multi_* parquet with human + >=2 generators."
        )

    return MultiCorpus(
        df,
        group_col=group_col,
        human_label=human_label,
        domain_col=domain_col,
        source=path.name,
    )
