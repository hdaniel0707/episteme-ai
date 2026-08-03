"""Shared reporting helpers for the science_data analysis scripts.

The four analysis scripts all do the same three things — echo a report to the
terminal and a log file, drop CSVs next to it, and save headless matplotlib
figures — so that plumbing lives here once rather than being copied into each.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "science_data"

_WORD_RE = re.compile(r"[a-z0-9]+")


class Report:
    """Collects printed lines, CSVs and figures under one output directory."""

    def __init__(self, stage: str, dataset_id: str) -> None:
        self.dir = RESULTS_DIR / stage / dataset_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self._lines: list[str] = []
        self.line(f"{stage} report for {dataset_id}")
        self.line(f"generated {datetime.now():%Y-%m-%d %H:%M:%S}")
        self.line("")

    def line(self, text: str = "") -> None:
        print(text)
        self._lines.append(text)

    def header(self, text: str) -> None:
        self.line("")
        self.line(text)
        self.line("-" * len(text))

    def table(self, df: pd.DataFrame, name: str, show: bool = True) -> None:
        df.to_csv(self.dir / f"{name}.csv", index=False)
        if show:
            self.line(df.to_string(index=False))

    def figure(self, fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(self.dir / f"{name}.png", dpi=150)
        plt.close(fig)

    def close(self) -> None:
        (self.dir / f"{self.stage}_report.log").write_text(
            "\n".join(self._lines) + "\n", encoding="utf-8"
        )
        print(f"\nReport → {self.dir}")


def describe(series: pd.Series, label: str) -> dict:
    """Summary statistics in a fixed shape, for stacking into a table."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"group": label, "n": 0}
    return {
        "group": label,
        "n": len(s),
        "mean": round(float(s.mean()), 2),
        "std": round(float(s.std()), 2),
        "min": round(float(s.min()), 2),
        "p25": round(float(s.quantile(0.25)), 2),
        "median": round(float(s.median()), 2),
        "p75": round(float(s.quantile(0.75)), 2),
        "max": round(float(s.max()), 2),
    }


def ngrams(text: str, n: int) -> Counter:
    """Lowercased word n-grams, punctuation stripped."""
    words = _WORD_RE.findall((text or "").lower())
    if len(words) < n:
        return Counter()
    return Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def overlap_fraction(a: str, b: str, n: int = 5) -> float:
    """Fraction of *b*'s n-grams that also appear in *a*.

    The leakage measure used across stages 3 and 4: how much of a generated
    text is verbatim reuse of the human source. Asymmetric on purpose — the
    question is always how much of the *derived* text came from the original.
    """
    a_grams, b_grams = ngrams(a, n), ngrams(b, n)
    if not b_grams:
        return 0.0
    shared = sum(count for gram, count in b_grams.items() if gram in a_grams)
    return shared / sum(b_grams.values())


def longest_common_run(a: str, b: str) -> int:
    """Length in words of the longest verbatim span shared by *a* and *b*.

    A single long quoted span is a leak even when the overall n-gram overlap
    looks low, so this catches what the averaged measure hides.
    """
    a_words = _WORD_RE.findall((a or "").lower())
    b_words = _WORD_RE.findall((b or "").lower())
    if not a_words or not b_words:
        return 0

    # Rolling DP over one row: O(len(a) * len(b)) time, O(len(b)) memory.
    prev = [0] * (len(b_words) + 1)
    best = 0
    for i in range(1, len(a_words) + 1):
        cur = [0] * (len(b_words) + 1)
        ai = a_words[i - 1]
        for j in range(1, len(b_words) + 1):
            if ai == b_words[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best
