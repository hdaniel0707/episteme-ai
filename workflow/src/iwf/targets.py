"""Impact targets + 3-class labels (plan sections 4.1 / 4.4).

Each target is a plain, stateless function ``f(df, min_refset) -> pd.Series`` that returns a
value in [0, 1] aligned to ``df.index`` (NaN where impossible). They are looked up in the
``TARGETS`` registry; ``main_prepare`` computes and stores all of them and ``TARGET_KIND``
selects the active one. (Targets are stateless, so they are functions - not classes.)

  field_year_percentile  (C, default)  CP-IN/CP-EX field-&-year citation percentile.
  fwci_percentile        (A)           OpenAlex FWCI percentile (sparse, but kept intact).
  normalized_percentile  (B)           relative citation rate cf=c/c0 (MNCS) -> [0,1] CDF.
  bayesian_impact                      Poisson-Gamma field-&-time Bayesian score -> [0,1] CDF.

Field key = the ~100%-covered ``domains`` column (subfield/domain are fallbacks); year is the
second bucket dimension; sparse (field x year) cells fall back to field-level then global once
below ``min_refset`` (standard CWTS practice).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COL = "target"
CLASS_NAMES = ["Weak", "Average", "Good"]

# Every selectable target kind; main_prepare computes and stores all of these.
ALL_TARGET_KINDS = [
    "field_year_percentile",   # C - CP-IN/CP-EX field-&-year citation percentile (default)
    "fwci_percentile",         # A - OpenAlex FWCI percentile
    "normalized_percentile",   # B - relative citation rate cf=c/c0 (MNCS) -> [0,1] CDF
    "bayesian_impact",         # Bayesian field-&-time-normalized score -> [0,1] CDF
]


# --------------------------------------------------------------- field / bucket helpers

def field_key(df: pd.DataFrame) -> pd.Series:
    """Per-row field label from ``domains`` (~100% covered); falls back to subfield/domain."""
    def first(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return str(v[0]) if len(v) else None
        if v is None:
            return None
        s = str(v)
        return s if s and s.lower() != "nan" else None

    if "domains" in df.columns:
        f = df["domains"].map(first)
    else:
        f = pd.Series([None] * len(df), index=df.index, dtype="object")
    for alt in ("subfield", "domain"):
        if alt in df.columns:
            f = f.where(f.notna(), df[alt])
    return f.fillna("unknown").astype(str)


def _year_key(df: pd.DataFrame) -> pd.Series:
    if "year" not in df.columns:
        return pd.Series(["0"] * len(df), index=df.index)
    return pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int).astype(str)


def _levels(df: pd.DataFrame) -> list[pd.Series]:
    """Bucket granularities, finest -> coarsest: field|year -> field -> global."""
    field, year = field_key(df), _year_key(df)
    return [field + "|" + year, field + "|*", pd.Series(["*|*"] * len(df), index=df.index)]


def hazen_percentile(values: pd.Series) -> pd.Series:
    """Continuous percentile rank in [0, 1] with the Hazen continuity correction.

    Ascending rank (least-cited = rank 1), ties take the average rank; PR=(r-0.5)/N - the
    midpoint of the CP-IN/CP-EX pair (Bornmann & Williams 2020), so the median lands on 0.50.
    """
    v = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    valid = v.dropna()
    n = len(valid)
    if n:
        out.loc[valid.index] = ((valid.rank(method="average", ascending=True) - 0.5) / n).to_numpy()
    return out


def empirical_cdf(values: pd.Series) -> pd.Series:
    """Map an unbounded score to [0, 1] via its global empirical CDF (percentile rank)."""
    return pd.to_numeric(values, errors="coerce").rank(method="average", pct=True).astype("float64")


def _bucketed(df: pd.DataFrame, value_col: str, min_refset: int, how: str) -> pd.Series:
    """Field-&-year statistic of ``value_col`` with small-cell fallback.

    ``how='percentile'`` -> Hazen percentile within the bucket (already in [0,1]);
    ``how='cf'``         -> value / bucket-mean (relative citation rate cf=c/c0, unbounded).
    Each row uses the finest bucket (field|year -> field -> global) with >= ``min_refset`` rows.
    """
    if value_col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    vals = pd.to_numeric(df[value_col], errors="coerce")
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    remaining = pd.Series(True, index=df.index)
    levels = _levels(df)
    for i, lvl in enumerate(levels):
        if not remaining.any():
            break
        sizes = lvl.map(lvl.value_counts())
        assign = remaining & ((sizes >= min_refset) | (i == len(levels) - 1))
        if not assign.any():
            continue
        if how == "percentile":
            full = pd.Series(np.nan, index=df.index, dtype="float64")
            for _, idx in df.groupby(lvl).groups.items():
                full.loc[idx] = hazen_percentile(vals.loc[idx])
            out.loc[assign] = full.loc[assign]
        else:  # cf = value / bucket mean
            mean = vals.groupby(lvl).transform("mean").replace(0, np.nan)
            out.loc[assign] = (vals / mean).loc[assign]
        remaining &= ~assign
    return out


def _percentile_within(value: pd.Series, buckets: pd.Series) -> pd.Series:
    """Plain percentile rank of ``value`` within each bucket (used by the FWCI fallback)."""
    out = pd.Series(np.nan, index=value.index, dtype="float64")
    v = pd.to_numeric(value, errors="coerce")
    for _, grp in v.groupby(buckets):
        valid = grp.dropna()
        if len(valid):
            out.loc[valid.index] = valid.rank(method="average", pct=True).to_numpy()
    return out


# --------------------------------------------------------------------------- the targets

def field_year_percentile(df: pd.DataFrame, min_refset: int = 30) -> pd.Series:
    """Option C - CP-IN/CP-EX percentile of citation_count within (field x year). Native [0,1]."""
    return _bucketed(df, "citation_count", min_refset, "percentile")


def normalized_percentile(df: pd.DataFrame, min_refset: int = 30) -> pd.Series:
    """Option B - relative citation rate cf=c/c0 (MNCS; Radicchi 2008 / Waltman 2011) -> [0,1] CDF."""
    return empirical_cdf(_bucketed(df, "citation_count", min_refset, "cf"))


def fwci_percentile(df: pd.DataFrame, min_refset: int = 30) -> pd.Series:
    """Option A - OpenAlex field-normalized percentile; FWCI-rank fallback where missing.

    Uses ``citation_normalized_percentile`` directly (already 0..1, field+year+type normalized).
    Where absent but ``fwci`` exists, ranks FWCI within the (field x year) bucket. FWCI logic
    is preserved exactly.
    """
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    if "citation_normalized_percentile" in df.columns:
        cnp = pd.to_numeric(df["citation_normalized_percentile"], errors="coerce")
        out = out.where(cnp.isna(), cnp)
    if "fwci" in df.columns and out.isna().any():
        buckets = field_key(df) + "|" + _year_key(df)
        out = out.where(out.notna(), _percentile_within(df["fwci"], buckets))
    return out


def _gamma_moments(c: pd.Series) -> tuple[float, float] | None:
    """Method-of-moments (alpha, beta) for the Gamma prior; None if not overdispersed."""
    c = pd.to_numeric(c, errors="coerce").dropna()
    if len(c) < 2:
        return None
    mu, s2 = float(c.mean()), float(c.var(ddof=1))
    if mu <= 0 or s2 <= mu:                     # need overdispersion (s2 > mu) for a valid prior
        return None
    beta = mu / (s2 - mu)
    alpha = mu * beta
    return (alpha, beta) if alpha > 0 and beta > 0 else None


def bayesian_impact(df: pd.DataFrame, min_refset: int = 30) -> pd.Series:
    """Poisson-Gamma field-&-time Bayesian score (Gomez-Deniz & Dorta-Gonzalez 2024) -> [0,1] CDF.

    F*(c,t) = (beta/alpha) * (alpha + c) / (beta + t), omega=1 (so F*(0,0)=1); alpha,beta per
    field by method of moments, global fallback. Mapped to [0,1] by its empirical CDF.
    """
    if "citation_count" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    cites = pd.to_numeric(df["citation_count"], errors="coerce")
    if "year" in df.columns:
        year = pd.to_numeric(df["year"], errors="coerce")
        age = (year.max() - year).clip(lower=0).fillna(0.0)    # t in years (>= 0)
    else:
        age = pd.Series(0.0, index=df.index)
    field = field_key(df)
    glob = _gamma_moments(cites) or (1.0, 1.0)
    score = pd.Series(np.nan, index=df.index, dtype="float64")
    for _, idx in df.groupby(field).groups.items():
        c = cites.loc[idx]
        est = _gamma_moments(c) if len(idx) >= min_refset else None
        alpha, beta = est if est is not None else glob
        score.loc[idx] = (beta / alpha) * (alpha + c) / (beta + age.loc[idx])   # omega = 1
    return empirical_cdf(score)


TARGETS = {
    "field_year_percentile": field_year_percentile,
    "fwci_percentile": fwci_percentile,
    "normalized_percentile": normalized_percentile,
    "bayesian_impact": bayesian_impact,
}


def compute_target(kind: str, df: pd.DataFrame, min_refset: int = 30) -> pd.Series:
    """Compute one impact target by name; returns a float64 Series in [0, 1]."""
    kind = (kind or "field_year_percentile").lower()
    if kind not in TARGETS:
        raise ValueError(f"Unknown target_kind {kind!r}. Choose from {list(TARGETS)}.")
    return TARGETS[kind](df, min_refset).astype("float64")


# ----------------------------------------------------------------- ordinal class labels

def default_class_names(n_classes: int) -> list[str]:
    """Ordinal class names, ascending by impact (index 0 = lowest impact).

    3 classes -> Weak/Average/Good (back-compat). Any other count (2..25, see
    ``iwf.config.MIN_CLASSES/MAX_CLASSES``) -> quantile labels Q{n}..Q1, generalising the
    original 4-class quartile scheme (Q4..Q1) to any class count, e.g. n=9 -> Q9..Q1."""
    if n_classes == 3:
        return ["Weak", "Average", "Good"]
    return [f"Q{i}" for i in range(n_classes, 0, -1)]


def _by_thresholds(values: pd.Series, thresholds: list[float],
                   names: list[str] | None = None) -> pd.Series:
    """Bin the [0,1] target by ascending thresholds into len(thresholds)+1 ordinal classes."""
    thr = sorted(float(t) for t in thresholds)
    n_classes = len(thr) + 1
    names = list(names) if names else default_class_names(n_classes)
    if len(names) != n_classes:                       # ignore a mismatched override
        names = default_class_names(n_classes)
    v = pd.to_numeric(values, errors="coerce")
    idx = np.digitize(v.to_numpy(dtype="float64"), thr, right=False)   # 0 .. len(thr)
    out = pd.Series([names[int(i)] for i in idx], index=values.index, dtype=object)
    out[v.isna()] = None
    return out


def css_boundaries(values: np.ndarray, n_classes: int = 3) -> list[float]:
    """Characteristic Scores and Scales (Glanzel & Schubert 1988): iterated conditional means."""
    v = np.sort(values[~np.isnan(values)])
    if v.size == 0:
        return [0.5, 0.9]
    cuts: list[float] = []
    current = v
    for _ in range(n_classes - 1):
        if current.size == 0:
            cuts.append(float(cuts[-1] if cuts else np.mean(v)))
            continue
        m = float(np.mean(current))
        cuts.append(m)
        current = current[current >= m]
    return cuts


def to_classes(values: pd.Series, scheme: str = "percentile",
               thresholds: list[float] | None = None,
               names: list[str] | None = None) -> pd.Series:
    """Map the continuous [0,1] impact target to ordinal classes.

    The number of classes follows the thresholds (N thresholds -> N+1 classes), so this works
    for the default 3-class scheme and for the 4-quartile (Q4..Q1) scheme alike. ``names`` are
    the ascending-impact labels (defaults derived from the class count)."""
    values = pd.to_numeric(values, errors="coerce")
    n_classes = len(names) if names else (len(thresholds) + 1 if thresholds else 3)
    if (scheme or "percentile").lower() == "css":
        cuts = css_boundaries(values.to_numpy(dtype="float64"), n_classes=n_classes)
        return _by_thresholds(values, cuts, names)
    return _by_thresholds(values, thresholds or [0.50, 0.90], names)
