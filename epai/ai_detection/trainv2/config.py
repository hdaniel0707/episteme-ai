"""The trainv2 config: a small YAML file, validated before anything is trained.

No preflight report, no staged checks — one function that either returns a
:class:`Config` or raises with a message naming the fix.

Example (``cfg/ai_detection/trainv2/gen_effectiveness_science_v3f.yaml``)::

    experiment: generator_effectiveness
    parquet: data/parquet/scribe_sci.parquet
    group_col: id
    human_label: human
    domain_col: domain

    embeddings: [emb_luar_crud, emb_specter2, emb_mpnet]
    algorithms:
      - {name: logistic, params: {C: 1.0, max_iter: 1000}}
      - {name: simple_mlp, label: "Simple MLP", params: {hidden_dims: [256], epochs: 120}}

    seeds: [0, 1, 2]
    test_size: 0.30
    fpr: 0.01
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from epai.algs import ALGORITHMS


@dataclass
class Config:
    name: str
    parquet: Path
    group_col: str
    human_label: str
    domain_col: str
    embeddings: list[str]
    algorithms: list[dict]
    seeds: list[int]
    test_size: float
    fpr: float
    raw: dict = field(default_factory=dict)

    @property
    def fpr_pct(self) -> str:
        return f"{self.fpr * 100:g}"


def _norm_algorithms(raw) -> list[dict]:
    """Accept ``"logistic"`` or ``{name, label?, params?}``; return the mapping form."""
    out: list[dict] = []
    for entry in raw or []:
        if isinstance(entry, str):
            out.append({"name": entry, "label": entry, "params": {}})
        elif isinstance(entry, dict) and isinstance(entry.get("name"), str):
            out.append(
                {
                    "name": entry["name"],
                    "label": entry.get("label", entry["name"]),
                    "params": entry.get("params") or {},
                }
            )
        else:
            raise ValueError(
                f"algorithms entry must be a name or a mapping with 'name', got {entry!r}"
            )
    return out


def load_config(path: str | Path, base_dir: Path) -> Config:
    """Read and validate one config. Paths in the file are relative to ``base_dir``."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")

    missing = [k for k in ("parquet", "embeddings", "algorithms") if k not in raw]
    if missing:
        raise ValueError(f"{path}: missing key(s) {missing}")

    embeddings = list(raw["embeddings"] or [])
    if not embeddings:
        raise ValueError(f"{path}: 'embeddings' is empty")
    bad_emb = [c for c in embeddings if not isinstance(c, str)]
    if bad_emb:
        raise ValueError(f"{path}: 'embeddings' must be column names, got {bad_emb}")

    algorithms = _norm_algorithms(raw["algorithms"])
    if not algorithms:
        raise ValueError(f"{path}: 'algorithms' is empty")
    for alg in algorithms:
        if alg["name"] not in ALGORITHMS:
            raise ValueError(
                f"{path}: unknown algorithm {alg['name']!r}. "
                f"Choose from {sorted(ALGORITHMS)}"
            )
        if getattr(ALGORITHMS[alg["name"]], "uses_aux", False):
            raise ValueError(
                f"{path}: {alg['name']!r} needs an auxiliary column — trainv2's "
                f"generator experiment is plain embedding x algorithm only"
            )

    seeds = list(raw.get("seeds", [0, 1, 2]))
    if not seeds or not all(isinstance(s, int) for s in seeds):
        raise ValueError(f"{path}: 'seeds' must be a non-empty list of ints")

    test_size = float(raw.get("test_size", 0.30))
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"{path}: 'test_size' must be in (0, 1), got {test_size}")

    fpr = float(raw.get("fpr", 0.01))
    if not 0.0 < fpr < 1.0:
        raise ValueError(f"{path}: 'fpr' must be in (0, 1), got {fpr}")

    return Config(
        name=raw.get("name", path.stem),
        parquet=(base_dir / raw["parquet"]).resolve(),
        group_col=raw.get("group_col", "id"),
        human_label=raw.get("human_label", "human"),
        domain_col=raw.get("domain_col", "domain"),
        embeddings=embeddings,
        algorithms=algorithms,
        seeds=seeds,
        test_size=test_size,
        fpr=fpr,
        raw=raw,
    )
