"""Pluggable external abstract-based impact models (plan section 4.5c).

Each model implements ``score(title, abstract) -> float | None`` and is FROZEN. Scores are
cached per paper_id (resume-safe). A model that is not configured/available returns None,
which the feature_builder turns into a neutral 0 + mask bit (so the fusion never breaks).

  * naip            - NAIP/NAID v2.1 via the HF Gradio Space (same as epai add_naip_scores).
  * specter2_knn    - frozen SPECTER2 embedding + kNN target-percentile transfer from TRAIN.
  * llm_judge       - pinned zero-shot LLM 0-100 impact rating (temperature 0, prompt v1).
  * venue_ais       - UEFISCDI/JCR Article Influence Score of the venue (per-year percentile).
  * forecite        - ForeCite (ref [18]); wire an endpoint via FORECITE_URL to enable.
  * cited_potential - Cited-Potential predictor (ref [19]); wire CITED_POTENTIAL_URL.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
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
            out.append(val)
            if val is not None:   # never cache failures — a fixed key/network must retry
                cache[pid] = val
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
        return pd.Series(out, index=df.index, name=self.name, dtype="float64")


class NAIPModel(ExternalImpactModel):
    name = "naip"

    def __init__(self, model_version: str = "v2.1"):
        self.model_version = model_version
        self._client = None
        self._warned = False

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
            # Explicit bounds checking for result schema
            if not isinstance(data, (list, tuple)) or len(data) < 2:
                if not self._warned:
                    tqdm.write(f"  [naip] unexpected response format: {type(data)}")
                    self._warned = True
                return None
            if not isinstance(data[1], (list, tuple)) or len(data[1]) < 2:
                if not self._warned:
                    tqdm.write(f"  [naip] unexpected response format at [1]: {type(data[1])}")
                    self._warned = True
                return None
            return float(data[1][1])  # normalized TNCSI_sp
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


class Specter2KnnBaseline(ExternalImpactModel):
    """Frozen SPECTER2 embedding + kNN percentile transfer (retrieval baseline).

    score(row) = mean ``target`` of the k most similar TRAIN papers. The embedding is
    frozen and no head is trained; only TRAIN labels are read, so val/test stay clean.
    Deterministic and offline — the cache file is written only for the record."""

    name = "specter2_knn"
    EMB_COL = "emb_specter2"

    def __init__(self, k: int | None = None):
        self.k = int(k or config.get_int("BASELINE_KNN_K", 15))

    def score(self, title: str, abstract: str) -> float | None:  # pragma: no cover
        return None  # this baseline scores whole dataframes, not single texts

    def score_dataframe(self, df: pd.DataFrame, cache_path: Path) -> pd.Series:
        from sklearn.neighbors import NearestNeighbors

        if self.EMB_COL not in df.columns or "split" not in df.columns \
                or "target" not in df.columns:
            print(f"[{self.name}] needs {self.EMB_COL}/split/target columns - returning None")
            return pd.Series([None] * len(df), index=df.index, name=self.name,
                             dtype="float64")
        X = np.array(df[self.EMB_COL].tolist(), dtype=np.float32)
        y = pd.to_numeric(df["target"], errors="coerce").to_numpy(dtype=np.float64)
        train = (df["split"].astype(str) == "train").to_numpy() & np.isfinite(y)
        if train.sum() < 2:
            return pd.Series([None] * len(df), index=df.index, name=self.name,
                             dtype="float64")
        k = int(min(self.k, train.sum() - 1))
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1).fit(X[train])
        dist, idx = nn.kneighbors(X)
        y_train = y[train]
        train_pos = np.where(train)[0]
        out = np.empty(len(df), dtype=np.float64)
        for i in range(len(df)):
            neigh = idx[i]
            if train[i]:  # drop self for train rows (it is always the 0-distance hit)
                neigh = [j for j in neigh if train_pos[j] != i]
            out[i] = float(np.mean(y_train[list(neigh)[:k]]))
        try:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(json.dumps(
                dict(zip(df[PAPER_ID].astype(str), out.tolist()))), encoding="utf-8")
        except OSError:
            pass
        return pd.Series(out, index=df.index, name=self.name, dtype="float64")


JUDGE_PROMPT_VERSION = "v1"
JUDGE_PROMPT = (
    "You are an experienced journal editor. Rate the expected scholarly impact of the "
    "following paper on a 0-100 scale, where 0 = negligible impact and 100 = "
    "field-defining. Judge only the title and abstract. Reply with ONLY the integer.\n\n"
    "Title: {title}\n\nAbstract: {abstract}"
)


class LLMJudgeBaseline(ExternalImpactModel):
    """Zero-shot LLM impact rating, pinned model + prompt v1, temperature 0.

    Provider from LLM_JUDGE_PROVIDER (anthropic | openai | ollama), model from
    LLM_JUDGE_MODEL. Returns score/100 in [0,1]; None when unconfigured/unparseable."""

    name = "llm_judge"

    def __init__(self):
        self.provider = (config.get_str("LLM_JUDGE_PROVIDER", "anthropic")).lower()
        defaults = {"anthropic": "claude-haiku-4-5", "openai": "gpt-4o-mini",
                    "ollama": "llama3"}
        self.model = config.get_str("LLM_JUDGE_MODEL", defaults.get(self.provider, ""))

    def _complete(self, prompt: str) -> str | None:
        try:
            if self.provider == "anthropic":
                key = config.secret("ANTHROPIC_API_KEY")
                if not key:
                    return None
                r = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                             "content-type": "application/json"},
                    json={"model": self.model, "max_tokens": 8, "temperature": 0,
                          "messages": [{"role": "user", "content": prompt}]},
                    timeout=60)
                return r.json()["content"][0]["text"] if r.ok else None
            if self.provider == "openai":
                key = config.secret("OPENAI_API_KEY")
                if not key:
                    return None
                r = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": self.model, "max_tokens": 8, "temperature": 0,
                          "messages": [{"role": "user", "content": prompt}]},
                    timeout=60)
                return r.json()["choices"][0]["message"]["content"] if r.ok else None
            if self.provider == "ollama":
                base = config.get_str("OLLAMA_URL", "http://localhost:11434")
                r = requests.post(f"{base}/api/generate",
                                  json={"model": self.model, "prompt": prompt,
                                        "stream": False, "options": {"temperature": 0}},
                                  timeout=120)
                return r.json().get("response") if r.ok else None
        except (requests.RequestException, KeyError, ValueError, IndexError):
            return None
        return None

    def score(self, title: str, abstract: str) -> float | None:
        if not title and not abstract:
            return None
        text = self._complete(JUDGE_PROMPT.format(title=title, abstract=abstract))
        if not text:
            return None
        m = re.search(r"\d{1,3}", text)
        if not m:
            return None
        val = int(m.group(0))
        return min(100, max(0, val)) / 100.0


class VenueAISBaseline(ExternalImpactModel):
    """Journal-level Article Influence Score of the paper's venue (UEFISCDI lists).

    The venue-only null baseline: no content is read at all. AIS values are joined by
    normalized journal name (nearest available JCR year), then percentile-ranked within
    publication year so the output is comparable to the other [0,1] scores."""

    name = "venue_ais"

    def score(self, title: str, abstract: str) -> float | None:  # pragma: no cover
        return None  # venue-based; scores whole dataframes

    def score_dataframe(self, df: pd.DataFrame, cache_path: Path) -> pd.Series:
        from workflow.src.iwf.fetchers.uefiscdi import load_ais_table, normalize_journal

        try:
            ais = load_ais_table()
        except FileNotFoundError as e:
            print(f"[{self.name}] {e} - returning None")
            return pd.Series([None] * len(df), index=df.index, name=self.name,
                             dtype="float64")
        if "venue" not in df.columns:
            return pd.Series([None] * len(df), index=df.index, name=self.name,
                             dtype="float64")

        # journal -> {jcr_year -> ais}
        by_name: dict[str, dict[int, float]] = {}
        for _, r in ais.iterrows():
            by_name.setdefault(r["journal_norm"], {})[int(r["year"])] = float(r["ais"])

        if "year" in df.columns:
            years = pd.to_numeric(df["year"], errors="coerce")
        else:
            years = pd.Series([float("nan")] * len(df), index=df.index)
        raw = np.full(len(df), np.nan)
        for i, (venue, yr) in enumerate(zip(df["venue"], years)):
            table = by_name.get(normalize_journal(venue))
            if not table:
                continue
            if yr == yr:  # nearest available JCR year to the publication year
                year_key = min(table, key=lambda k: abs(k - int(yr)))
            else:
                year_key = max(table)
            raw[i] = table[year_key]

        # percentile within publication year (fallback: global) -> [0,1]
        out = np.full(len(df), np.nan)
        yr_key = years.fillna(-1).astype(int)
        for y in yr_key.unique():
            m = (yr_key == y).to_numpy() & np.isfinite(raw)
            if m.sum() >= 5:
                ranks = pd.Series(raw[m]).rank(pct=True).to_numpy()
                out[m] = ranks
        rest = np.isfinite(raw) & ~np.isfinite(out)
        if rest.any():
            out[rest] = pd.Series(raw[rest]).rank(pct=True).to_numpy()
        return pd.Series([v if np.isfinite(v) else None for v in out],
                         index=df.index, name=self.name, dtype="float64")


_REGISTRY = {
    "naip": NAIPModel,
    "specter2_knn": Specter2KnnBaseline,
    "llm_judge": LLMJudgeBaseline,
    "venue_ais": VenueAISBaseline,
    "forecite": ForeCiteModel,
    "cited_potential": CitedPotentialModel,
}


def known_external_models() -> list[str]:
    return sorted(_REGISTRY)


def get_external_models(names: list[str]) -> list[ExternalImpactModel]:
    models = []
    for n in names:
        cls = _REGISTRY.get(n.lower())
        if cls is None:
            raise ValueError(f"Unknown external impact model {n!r}. Have {list(_REGISTRY)}.")
        models.append(cls())
    return models
