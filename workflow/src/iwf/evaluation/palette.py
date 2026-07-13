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

# Correct / wrong colour pair used across correctness + error plots
CORRECT_COLOR = GOLD         # accurate / high-impact
WRONG_COLOR = MAUVE          # inaccurate / low-impact

# Plotly/matplotlib error scales
ERROR_COLORSCALE_PLOTLY = [[0, GOLD], [1, MAUVE]]
ERROR_CMAP_MATPLOTLIB = "plasma_r"

# Diverging colormap for correlation matrices
DIVERGING_CMAP = "BrBG"


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
