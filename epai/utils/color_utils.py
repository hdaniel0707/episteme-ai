"""Shared color mapping for source / LLM families.

Colors live in the external config cfg/colors.yaml so that every plotting script
(epai/utils/visualization_utils.py, epai/citation_impact/analyse/analyse_naip.py, ...) stays consistent.
Labels are matched to a family by longest matching prefix, so specific variants
such as ``gpt_prompt1`` win over the generic ``gpt`` entry.
"""

from functools import lru_cache
from pathlib import Path

import yaml

CFG_PATH = Path(__file__).parent.parent.parent / "cfg" / "colors.yaml"


@lru_cache(maxsize=None)
def _load() -> tuple[dict[str, tuple[str, str]], str]:
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    colors = {str(k): (str(v[0]), str(v[1])) for k, v in cfg["colors"].items()}
    default = str(cfg.get("default", "#aaaaaa"))
    return colors, default


def _match_key(label: str) -> str | None:
    """Return the longest family key that ``label`` starts with, or None."""
    colors, _ = _load()
    lower = label.lower()
    candidates = [k for k in colors if lower.startswith(k)]
    return max(candidates, key=len) if candidates else None


def llm_color(label: str) -> str | None:
    """Color for a known LLM/source label; darker shade when 'aggressive'.

    Returns None for unknown labels so callers can supply their own fallback.
    """
    colors, _ = _load()
    key = _match_key(label)
    if key is None:
        return None
    normal, aggressive = colors[key]
    return aggressive if "aggressive" in label.lower() else normal


def group_color(group: str) -> str:
    """Color for a group label, falling back to the configured default color."""
    color = llm_color(group)
    return color if color is not None else _load()[1]
