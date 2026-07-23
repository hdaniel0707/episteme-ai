"""Two-level module registry (plan section 1).

The system is a two-level hierarchy:

    workflow type          which pipeline topology to use (config.yaml `workflow:`)
      MODULE_SLOT          a swappable component within that topology
        MODULE_IMPL        the concrete implementation chosen for the slot

Every module implementation registers itself for a ``(slot, name)`` pair so a workflow can look
it up *by name* without importing the concrete class - swapping an implementation never changes
the workflow function's body. The five slots are:

  * ``embedder``          title+abstract -> [B, embed_dim]            (EMBEDDERS / EMBED_FOR_ENCODER)
  * ``encoder``           [B, embed_dim] -> [B, latent_dim]          (ENCODER_KIND)
  * ``metadata_fuser``    z + scalar features -> fused matrix         (W3/W4, phase 8)
  * ``impact_predictor``  [B, in_dim] -> [B, out_dim]                (MODEL_ARCH)
  * ``text_transformer``  raw (title, abstract) -> [B, out_dim]      (TEXT_TRANSFORMER_ARCH, W5/W6)

Module contract (duck-typed):

    class MyModule:
        def __init__(self, cfg: Settings): ...
        def fit(self, data) -> None: ...              # no-op if stateless
        def transform(self, data) -> torch.Tensor: ...
        # OR, for end-to-end trainable modules (text_transformer + head fused):
        def forward(self, batch: dict) -> torch.Tensor: ...
        def save(self, path: Path) -> None: ...
        def load(self, path: Path) -> None: ...
"""

from __future__ import annotations

from typing import Any, Callable

# slot -> {name -> implementation (class or factory callable)}
_MODULES: dict[str, dict[str, Any]] = {}

SLOTS = ("embedder", "encoder", "metadata_fuser", "impact_predictor", "text_transformer")

# Slots whose implementations live in a separate module and self-register on import. We import
# that module lazily the first time the slot is queried, so registry import stays torch-free.
_LAZY_IMPORTS = {
    "text_transformer": "workflow.src.iwf.models.text_transformers",
}
_LOADED_SLOTS: set[str] = set()


def register_module(slot: str, name: str) -> Callable[[Any], Any]:
    """Decorator: register an implementation for ``(slot, name)`` (case-insensitive name)."""
    if slot not in SLOTS:
        raise ValueError(f"Unknown module slot {slot!r}; valid slots: {list(SLOTS)}.")

    def deco(impl: Any) -> Any:
        _MODULES.setdefault(slot, {})[name.lower()] = impl
        return impl

    return deco


def _ensure_loaded(slot: str) -> None:
    """Import a lazily-registered slot's module so its implementations are present."""
    if slot in _LOADED_SLOTS:
        return
    _LOADED_SLOTS.add(slot)          # mark first to avoid a re-entrant import loop
    mod = _LAZY_IMPORTS.get(slot)
    if mod is not None:
        __import__(mod)
    _register_builtin_slots()


def get_module(slot: str, name: str) -> Any:
    """Return the registered implementation for ``(slot, name)`` or raise a clear error."""
    if slot not in SLOTS:
        raise ValueError(f"Unknown module slot {slot!r}; valid slots: {list(SLOTS)}.")
    _ensure_loaded(slot)
    impls = _MODULES.get(slot, {})
    key = (name or "").lower()
    if key not in impls:
        raise ValueError(
            f"No '{name}' implementation registered for slot {slot!r}; "
            f"available: {sorted(impls)}."
        )
    return impls[key]


def has_module(slot: str, name: str) -> bool:
    _ensure_loaded(slot)
    return (name or "").lower() in _MODULES.get(slot, {})


def available_modules(slot: str) -> list[str]:
    _ensure_loaded(slot)
    return sorted(_MODULES.get(slot, {}))


def all_modules() -> dict[str, list[str]]:
    for s in SLOTS:
        _ensure_loaded(s)
    return {s: sorted(_MODULES.get(s, {})) for s in SLOTS}


# --------------------------------------------------------------------------- builtin slots
# The embedder / encoder / impact_predictor / metadata_fuser slots are already implemented as
# functions elsewhere; we register thin lazy factories here so the registry reflects the real
# set of selectable names (used for listing and validation). Each factory imports on call, so
# registering them never pulls torch into a plain ``import registry``.

_BUILTIN_REGISTERED = False


def _register_builtin_slots() -> None:
    global _BUILTIN_REGISTERED
    if _BUILTIN_REGISTERED:
        return
    _BUILTIN_REGISTERED = True

    def _encoder_factory(kind: str):
        def make(latent_dim: int, **kw):
            from workflow.src.iwf.encoder.base import get_encoder
            return get_encoder(kind, latent_dim=latent_dim, **kw)
        return make

    for kind in ("autoencoder", "epai_autoencoder", "pca", "umap", "identity"):
        _MODULES.setdefault("encoder", {}).setdefault(kind, _encoder_factory(kind))

    def _embedder_factory(name: str):
        def add(df):
            from workflow.src.iwf.embedding.embedders import add_embedding
            return add_embedding(df, name)
        return add

    for name in ("specter2", "specter2_base", "specter2_adhoc",
                 "scibert", "scincl", "e5", "openai"):
        _MODULES.setdefault("embedder", {}).setdefault(name, _embedder_factory(name))

    def _predictor_factory(arch: str):
        def build(in_dim: int, out_dim: int, *, task: str, hidden, dropout, activation):
            from workflow.src.iwf.models.architectures import build_block
            return build_block(arch, in_dim, out_dim, task=task, hidden=hidden,
                               dropout=dropout, activation=activation)
        return build

    try:
        from workflow.src.iwf.models.architectures import available_archs
        for arch in available_archs():
            _MODULES.setdefault("impact_predictor", {}).setdefault(arch, _predictor_factory(arch))
    except Exception:
        for arch in ("mlp", "resmlp", "ft_transformer"):
            _MODULES.setdefault("impact_predictor", {}).setdefault(arch, _predictor_factory(arch))

    def _fuse(scalar_cols):
        from workflow.src.iwf.features.feature_builder import scalar_feature_columns
        return scalar_feature_columns

    _MODULES.setdefault("metadata_fuser", {}).setdefault("concat", _fuse)
