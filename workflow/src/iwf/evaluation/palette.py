"""Unified color system for all evaluation plots — palette constants and helpers.

Every figure uses apply_style() at entry; colors from this module ensure consistency
across classification/regression tasks, workflow types, and split/domain annotations.
"""

from __future__ import annotations

import re
from typing import Sequence

import matplotlib
import matplotlib.colors
import numpy as np

# ========================================================================= CONSTANTS

OLIVE = "#556B2F"          # primary: emb_clf, train set, true labels
SAGE = "#8FA35E"           # secondary: emb_reg, val set
SAND = "#D3DEB8"           # tertiary: test set
INK = "#3D4A1F"            # darkest: edges, outlines
TERRACOTTA = "#C4756B"     # warm accent: wrong/error, external models
DUSK_BLUE = "#7189A6"      # cool accent: tfm_clf, high-value items
MAUVE = "#9A8AA8"          # secondary accent: wrong/inaccurate in clf plots
GOLD = "#D9B96E"           # warm highlight: metadata, top metrics
GRAY = "#8C8C8C"           # neutral: target/unknown/external/unlabeled
ERROR_BAR_GRAY = "#4D4D4D" # neutral dark gray: seed-spread error bars (never pure black)

# Blue family (for heatmaps, correlation, confusion, cross-domain, year×domain)
BLUE_DARK = "#1F3A5F"      # darkest navy: strong positive / high count
BLUE_MID = "#3E6D9C"       # mid blue
BLUE_LIGHT = "#9DC3E6"     # pale blue: weak / low count
# Red family (for the "rest" completeness-field group + secondary ramps)
RED_DARK = "#8C2F2A"
RED_MID = "#C0504D"
RED_LIGHT = "#E6A6A2"

# Workflow type → bar color (emb_clf=OLIVE, emb_reg=SAGE, tfm_clf=DUSK_BLUE, etc.)
CATEGORY_COLORS = {
    "emb_clf": OLIVE,
    "emb_reg": SAGE,
    "tfm_clf": DUSK_BLUE,
    "tfm_reg": "#93A9C4",  # lighter than DUSK_BLUE
    "meta": GOLD,          # every emb_meta_* / emb_all_meta_* workflow
    "external": TERRACOTTA,  # downloaded/outsider scorers (naip, specter2_knn, etc.)
    "target": GRAY,
}

# Impact ramp: low→high (for ordinal scales, heatmaps, error bars)
ORDINAL_RAMP = [SAND, "#B9C98F", SAGE, "#6E8440", OLIVE, INK]

# Train / val / test bar shades: one olive hue, darkest → palest
SPLIT_SHADES = {"train": OLIVE, "val": SAGE, "test": SAND}

# Correct / wrong colour pair used across correctness + error plots.
# UMAP correctness plots use a strong yellow (correct) / purple (wrong) pair.
CORRECT_COLOR = "#F2C230"    # strong yellow: accurate / high-impact
WRONG_COLOR = "#6A3D9A"      # strong purple: inaccurate / low-impact

# Plotly/matplotlib error scales (yellow = accurate → purple = off)
ERROR_COLORSCALE_PLOTLY = [[0, CORRECT_COLOR], [1, WRONG_COLOR]]
ERROR_CMAP_MATPLOTLIB = "plasma_r"

# Diverging colormap for correlation matrices — a blue diverging map (light blue for
# negative, white at zero, dark navy for positive), registered lazily by name below.
DIVERGING_CMAP = "blue_div"

# Four categorical colours for the impact-class UMAP scatter — colourblind-safe
# (Okabe–Ito subset) and legible together. Cycled if more than four classes.
CLASS_COLORS_4 = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]

# ---- completeness-field grouping (used by the completeness plots) ----
# Group order is fixed so the "field completeness" and "completeness by year" figures
# list fields in the same sequence: olive core, blue identifiers, then red "rest".
FIELD_GROUP_OLIVE = ("title", "abstract", "year")
FIELD_GROUP_BLUE = ("domains", "subfield", "doi", "oa_id")
FIELD_GROUP_LABELS = {"olive": "core text/metadata", "blue": "identifiers / linkage",
                      "red": "citation & venue metrics"}
_OLIVE_SHADES = [OLIVE, SAGE, "#6E8440", INK]
_BLUE_SHADES = [BLUE_DARK, BLUE_MID, DUSK_BLUE, BLUE_LIGHT]
_RED_SHADES = [RED_DARK, RED_MID, TERRACOTTA, RED_LIGHT, "#D98880"]


def field_group_of(field: str) -> str:
    """Return the completeness colour-group ('olive' | 'blue' | 'red') for a field."""
    f = str(field).strip().lower()
    if f in FIELD_GROUP_OLIVE:
        return "olive"
    if f in FIELD_GROUP_BLUE:
        return "blue"
    return "red"


def order_fields(fields: Sequence[str]) -> list[str]:
    """Reorder ``fields`` as olive group, then blue group, then the rest ('red'),
    each group keeping the canonical order above / original order for the rest."""
    fields = list(fields)
    olive = [f for f in FIELD_GROUP_OLIVE if f in fields]
    blue = [f for f in FIELD_GROUP_BLUE if f in fields]
    used = set(olive) | set(blue)
    red = [f for f in fields if f not in used]
    return olive + blue + red


def field_color_map(fields: Sequence[str]) -> dict:
    """Map each field to a distinct shade of its group colour (olive/blue/red)."""
    buckets = {"olive": [], "blue": [], "red": []}
    for f in order_fields(fields):
        buckets[field_group_of(f)].append(f)
    ramps = {"olive": _OLIVE_SHADES, "blue": _BLUE_SHADES, "red": _RED_SHADES}
    out: dict = {}
    for grp, members in buckets.items():
        ramp = ramps[grp]
        for i, f in enumerate(members):
            out[f] = ramp[i] if i < len(ramp) else ramp[-1]
    return out


def class_colors(n: int) -> list[str]:
    """n categorical colours for impact-class scatter plots (cycles CLASS_COLORS_4)."""
    return [CLASS_COLORS_4[i % len(CLASS_COLORS_4)] for i in range(max(0, n))]


# ========================================================================= REGISTRY HELPERS

# Hardcoded external scorer names — do NOT import external_impact (would create cycle)
EXTERNAL_SCORER_NAMES = frozenset({
    "naip", "naip_local", "specter2_knn", "llm_judge",
    "venue_ais", "forecite", "cited_potential"
})


def category_of(name: str) -> str:
    """Map a workflow/scorer name to a color category.

    Algorithm:
      1. "target" → "target"
      2. "ours" → "target" (standard naming for the trained run)
      3. known external scorer → "external"
      4. strip trailing _<digits> (e.g., "emb_clf_3" → "emb_clf")
      5. emb_meta* / emb_all_meta* → "meta"
      6. in CATEGORY_COLORS → that key
      7. else → "external"
    """
    n = str(name).strip().lower()

    if n == "target":
        return "target"
    if n == "ours":
        return "target"

    if n in EXTERNAL_SCORER_NAMES:
        return "external"

    # Strip trailing _<digits>
    base = re.sub(r"_\d+$", "", n)

    if base.startswith("emb_meta") or base.startswith("emb_all_meta"):
        return "meta"

    if base in CATEGORY_COLORS:
        return base

    return "external"


def color_for(name: str) -> str:
    """Retrieve the hex color for a workflow/scorer name."""
    return CATEGORY_COLORS.get(category_of(name), GRAY)


def ordinal_colors(n: int) -> list[str]:
    """Return n evenly-spaced colors along the ORDINAL_RAMP.

    For n <= len(ORDINAL_RAMP): sample via index rounding.
    For n > len(ORDINAL_RAMP): interpolate through LinearSegmentedColormap.
    """
    if n <= len(ORDINAL_RAMP):
        indices = np.linspace(0, len(ORDINAL_RAMP) - 1, n).round().astype(int)
        return [ORDINAL_RAMP[i] for i in indices]
    else:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("_ordinal_tmp", ORDINAL_RAMP)
        positions = np.linspace(0, 1, n)
        colors = [matplotlib.colors.to_hex(cmap(p)) for p in positions]
        return colors


def register_olive_seq() -> str:
    """Register and return the name of the olive sequential colormap.

    Creates LinearSegmentedColormap from [SAND → OLIVE → INK] if not already
    registered. Idempotent.
    """
    cmap_name = "olive_seq"
    if cmap_name not in matplotlib.colormaps():
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            cmap_name, [SAND, OLIVE, INK]
        )
        matplotlib.colormaps.register(cmap)
    return cmap_name


def register_red_seq() -> str:
    """Register and return the name of the red sequential colormap.

    Pale red (low) → terracotta → dark red (high). The training-subset analysis plots use
    this as the red counterpart of ``olive_seq`` (which colours the full-corpus plots), so
    the two suites read as "full dataset = olive, training data = red". Idempotent.
    """
    cmap_name = "red_seq"
    if cmap_name not in matplotlib.colormaps():
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            cmap_name, ["#F7E4E2", RED_LIGHT, RED_MID, RED_DARK]
        )
        matplotlib.colormaps.register(cmap)
    return cmap_name


def register_blue_seq() -> str:
    """Register and return the name of the blue sequential colormap.

    Pale blue (low) → mid blue → dark navy (high). Used for count/percentage heatmaps
    (confusion matrix, year×domain, completeness-by-year, cross-domain). Idempotent.
    """
    cmap_name = "blue_seq"
    if cmap_name not in matplotlib.colormaps():
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            cmap_name, ["#F2F6FA", BLUE_LIGHT, BLUE_MID, BLUE_DARK]
        )
        matplotlib.colormaps.register(cmap)
    return cmap_name


def register_red_div() -> str:
    """Register and return the name of the red diverging colormap for the training-subset
    correlation heatmap. Light red (−1) → white (0) → dark red (+1): sign shows as
    light-vs-dark red so the whole map stays 'shades of red'. Idempotent."""
    cmap_name = "red_div"
    if cmap_name not in matplotlib.colormaps():
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            cmap_name, ["#E6A6A2", "#F2D0CE", "#FFFFFF", RED_MID, RED_DARK]
        )
        matplotlib.colormaps.register(cmap)
    return cmap_name


def register_blue_div() -> str:
    """Register and return the name of the blue diverging colormap for correlations.

    Light blue (−1) → white (0) → dark navy (+1): sign shows as light-vs-dark blue so the
    whole map stays 'shades of blue'. Idempotent.
    """
    cmap_name = "blue_div"
    if cmap_name not in matplotlib.colormaps():
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            cmap_name, ["#6BAED6", "#C6DBEF", "#FFFFFF", BLUE_MID, BLUE_DARK]
        )
        matplotlib.colormaps.register(cmap)
    return cmap_name


def apply_style() -> None:
    """Apply unified matplotlib style to the current figure/axes.

    Call at the beginning of every figure-generation function.
    """
    matplotlib.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.frameon": False,
    })


# Register the named colormaps referenced by string (DIVERGING_CMAP) at import time so
# imshow(cmap="blue_div"/"blue_seq") resolves without an explicit register_* call.
register_blue_seq()
register_blue_div()
register_olive_seq()
register_red_seq()
register_red_div()
