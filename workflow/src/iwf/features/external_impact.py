"""Pluggable external abstract-based impact models (plan section 4.5c).

Each model implements ``score(title, abstract) -> float | None`` and is FROZEN. Scores are
cached per paper_id (resume-safe). A model that is not configured/available returns None,
which the feature_builder turns into a neutral 0 + mask bit (so the fusion never breaks).

  * naip            - NAIP/NAID via the HF Gradio Space (same as epai/fetch_naip.py).
  * forecite        - ForeCite (ref [18]); wire an endpoint via FORECITE_URL to enable.
  * cited_potential - Cited-Potential predictor (ref [19]); wire CITED_POTENTIAL_URL.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from workflow.src.iwf import config
from workflow.src.iwf.data.identity import PAPER_ID


class ExternalImpactModel(ABC):
    name: str = "external"

    @abstractmethod
    def score(self, title: str, abstract: str) -> float | None:
        ...

    def score_dataframe(self, df: pd.DataFrame, cache_path: Path) -> pd.Series:
        """Score every row, caching by paper_id; returns a Series named ``self.name``."""
        cache: dict[str, float | None] = {}
        cache_path = Path(cache_path)
        if cache_path.is_file():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                cache = {}
        out: list[float | None] = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{self.name}"):
            pid = str(row[PAPER_ID])
            if pid in cache:
                out.append(cache[pid])
                continue
            val = self.score(str(row.get("title") or ""), str(row.get("abstract") or ""))
            cache[pid] = val
            out.append(val)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
        return pd.Series(out, index=df.index, name=self.name, dtype="float64")


class NAIPModel(ExternalImpactModel):
    name = "naip"

    def __init__(self, model_version: str = "v2.1"):
        self.model_version = model_version
        self._client = None

    def _get_client(self):
        if self._client is None:
            from gradio_client import Client

            self._client = Client(
                "ssocean/Newborn_Article_Impact_Predict", token=config.secret("HF_TOKEN")
            )
        return self._client

    def score(self, title: str, abstract: str) -> float | None:
        if not title and not abstract:
            return None
        try:
            result = self._get_client().predict(
                title=title, abstract=abstract,
                model_version=self.model_version, api_name="/predict",
            )
            data = result.get("data", []) if isinstance(result, dict) else result
            return float(data[1][1]) if len(data) > 1 else None  # normalized TNCSI_sp
        except Exception as e:  # pragma: no cover - network dependent
            tqdm.write(f"  [naip] {e}")
            return None


class _EndpointModel(ExternalImpactModel):
    """Generic REST scorer: POST {title, abstract} -> {'score': float}."""

    def __init__(self, name: str, url_env: str):
        self.name = name
        self.url = config.secret(url_env)

    def score(self, title: str, abstract: str) -> float | None:
        if not self.url:
            return None
        try:
            r = requests.post(self.url, json={"title": title, "abstract": abstract}, timeout=30)
            return float(r.json().get("score")) if r.ok else None
        except Exception:
            return None


class ForeCiteModel(_EndpointModel):
    def __init__(self):
        super().__init__("forecite", "FORECITE_URL")


class CitedPotentialModel(_EndpointModel):
    def __init__(self):
        super().__init__("cited_potential", "CITED_POTENTIAL_URL")


_REGISTRY = {
    "naip": NAIPModel,
    "forecite": ForeCiteModel,
    "cited_potential": CitedPotentialModel,
}


def get_external_models(names: list[str]) -> list[ExternalImpactModel]:
    models = []
    for n in names:
        cls = _REGISTRY.get(n.lower())
        if cls is None:
            raise ValueError(f"Unknown external impact model {n!r}. Have {list(_REGISTRY)}.")
        models.append(cls())
    return models
