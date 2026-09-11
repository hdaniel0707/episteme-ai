"""Shared training machinery for the PyTorch classifiers in this package.

:mod:`simple_mlp`, :mod:`binox_mlp`, :mod:`fusion_mlp` and :mod:`gated_mlp`
differ only in what their network does with the inputs. Everything around the
network -- how the features are normalised, how an auxiliary matrix is cleaned,
how a validation split is taken, when training stops, what the loss does to the
labels -- is identical, and lives here so that a regularizer added once applies
to all of them rather than being implemented four times and drifting.

Every knob below defaults to what the two algorithms did before this module
existed, so a config written against the old code trains the same model:
``normalize: standard``, no validation split (early stopping on the *training*
loss), no input dropout, no label smoothing, no mixup.

Why these knobs, for a cross-domain run
---------------------------------------
The failure mode this package is chasing is a classifier that is excellent on
its own corpus and mediocre on the next one. Two of the options here address
that directly and are worth more attention than the usual dropout/weight-decay
sweep:

``normalize``
    The default ``standard`` fits a :class:`StandardScaler` on the *source*
    corpus and applies it unchanged to every target. If the embedding's mean and
    scale shift between corpora -- and between corpora built from different
    source documents they do -- that scaler feeds the domain shift straight into
    the first layer. ``l2`` normalises each row to unit length instead, which is
    a per-row operation and therefore carries no corpus statistics across the
    boundary at all.

``val_frac``
    With no validation split, "early stopping" watches the training loss, so it
    fires when the model stops fitting the training set better -- which is after
    it has overfit, not before. It is not regularising anything. Set
    ``val_frac`` and a real held-out split is taken from the training rows
    (grouped, when the runner supplies groups) and the best-scoring weights are
    restored at the end.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

#: Normalisation steps ``FeatureNormalizer`` understands, applied left to right.
NORMALIZE_STEPS = {"none", "standard", "l2"}

#: Quantities ``val_metric`` may early-stop on, and whether bigger is better.
#:
#: ``auroc`` is the one to reach for when the run is reported at a low-FPR
#: operating point. ``accuracy`` stops on the argmax at 0.5, which is blind to
#: how the two classes are *ordered* either side of it -- two epochs with the
#: same accuracy can have very different tails, and the tail is exactly what
#: TPR@1%FPR reads.
VAL_METRICS = {"loss": False, "accuracy": True, "auroc": True}


def logit_margin(logits: torch.Tensor) -> np.ndarray:
    """Machine-minus-human logit, as float64. A ranking score, not a probability.

    Deliberately not ``softmax(...)[:, 1]``. The metrics that read this are
    evaluated in the tail of the human distribution -- TPR at 1% FPR is set by a
    handful of the most machine-looking human rows -- and in float32 a confident
    model saturates softmax to exactly 1.0 for a large share of rows. Those ties
    are indistinguishable to a threshold, so the tail resolution the metric needs
    is destroyed by the very transform that makes the number look like a
    probability. The margin is monotone in that probability and keeps every
    distinct ordering the network actually produced.
    """
    return (logits[:, 1] - logits[:, 0]).to(torch.float64).cpu().numpy()


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------


@dataclass
class TrainHParams:
    """The hyper-parameters shared by every torch algorithm in this package.

    Read out of a config entry's ``params`` by :func:`read_hparams`, which
    ignores any key it does not recognise -- the algorithm-specific ones
    (``hidden_dims``, ``pre_dims``, ...) stay in the same dict and are read by
    the algorithm itself.
    """

    # --- optimisation ---
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 0.0  # 0 disables

    # --- early stopping ---
    patience: int = 10  # 0 disables; always runs `epochs`
    val_frac: float = 0.0  # 0 => stop on training loss (legacy behaviour)
    val_metric: str = "loss"

    # --- regularisation ---
    dropout: float = 0.1  # between hidden layers
    input_dropout: float = 0.0  # on the raw feature vector
    norm_layer: str = "none"  # none | layernorm
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0  # 0 disables

    # --- preprocessing ---
    normalize: str = "standard"

    # --- tail-aware loss / domain-invariance (epai/algs/robust_mlp.py) ---
    # See epai/ai_detection/trainv2/improving_sota_ideas.md for the method and
    # citations. Every default here reproduces plain cross-entropy training
    # with no domain penalty, so no existing algorithm/config is affected.
    loss_kind: str = "ce"  # ce | toppush
    push_frac: float = 0.05  # fraction of a batch's human rows to push against
    push_weight: float = 1.0  # weight of the push term when loss_kind='toppush'
    push_margin: float = 1.0  # hinge margin, in logit-margin units
    domain_penalty: str = "none"  # none | coral | vrex
    domain_penalty_weight: float = 0.0

    def __post_init__(self) -> None:
        # Coerce before validating: a YAML ``1e-3`` parses as a string in some
        # writings, and a swept value arrives as whatever json.loads made of it.
        # The pre-refactor code did the same with int()/float() at every read.
        for name in ("epochs", "batch_size", "patience"):
            setattr(self, name, int(getattr(self, name)))
        for name in (
            "lr",
            "weight_decay",
            "grad_clip",
            "val_frac",
            "dropout",
            "input_dropout",
            "label_smoothing",
            "mixup_alpha",
            "push_frac",
            "push_weight",
            "push_margin",
            "domain_penalty_weight",
        ):
            setattr(self, name, float(getattr(self, name)))

        if self.loss_kind not in ("ce", "toppush"):
            raise ValueError(f"loss_kind must be 'ce' or 'toppush', got {self.loss_kind!r}")
        if not 0.0 < self.push_frac <= 1.0:
            raise ValueError(f"push_frac must be in (0, 1], got {self.push_frac}")
        if self.push_weight < 0:
            raise ValueError(f"push_weight must be >= 0, got {self.push_weight}")
        if self.domain_penalty not in ("none", "coral", "vrex"):
            raise ValueError(
                f"domain_penalty must be 'none', 'coral' or 'vrex', got "
                f"{self.domain_penalty!r}"
            )
        if self.domain_penalty_weight < 0:
            raise ValueError(
                f"domain_penalty_weight must be >= 0, got {self.domain_penalty_weight}"
            )
        if self.mixup_alpha > 0 and (self.loss_kind == "toppush" or self.domain_penalty != "none"):
            raise ValueError(
                "mixup_alpha > 0 is not supported together with loss_kind="
                "'toppush' or domain_penalty != 'none'"
            )

        if self.val_metric not in VAL_METRICS:
            raise ValueError(
                f"val_metric must be one of {sorted(VAL_METRICS)}, "
                f"got {self.val_metric!r}"
            )
        if self.norm_layer not in {"none", "layernorm"}:
            raise ValueError(
                f"norm_layer must be 'none' or 'layernorm', got {self.norm_layer!r}"
            )
        if not 0.0 <= self.val_frac < 1.0:
            raise ValueError(f"val_frac must be in [0, 1), got {self.val_frac}")
        for name in ("dropout", "input_dropout"):
            p = getattr(self, name)
            if not 0.0 <= p < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {p}")
        # Ranges for the rest. Every one of these is reachable from a swept value
        # or a hand-written config, and each failure mode is silent rather than
        # loud: epochs=0 trains nothing and still reports a matrix, lr<=0 leaves
        # the weights at their initialisation. Checked here so a config is
        # rejected by the preflight instead of producing a plausible-looking run.
        for name in ("epochs", "batch_size"):
            v = getattr(self, name)
            if v < 1:
                raise ValueError(f"{name} must be >= 1, got {v}")
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0 (0 disables), got {self.patience}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        for name in ("weight_decay", "grad_clip", "mixup_alpha"):
            v = getattr(self, name)
            if v < 0:
                raise ValueError(f"{name} must be >= 0 (0 disables), got {v}")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {self.label_smoothing}"
            )
        # Parsed here rather than in the normalizer so a bad value fails at
        # config-read time, before anything is loaded or trained.
        parse_normalize(self.normalize)


#: The shared hyper-parameter names, for an algorithm's ``PARAMS`` declaration.
#: Derived from the dataclass so the two can never drift: a knob added to
#: :class:`TrainHParams` is accepted by every torch algorithm's config entry the
#: moment it exists, and the sweep can search it without touching this file.
TRAIN_HPARAM_NAMES: tuple[str, ...] = tuple(f.name for f in fields(TrainHParams))


def read_hparams(params: dict, **overrides) -> TrainHParams:
    """Build :class:`TrainHParams` from a config entry's ``params``.

    Keys the dataclass does not declare are ignored *here*, so an algorithm's own
    hyper-parameters (``hidden_dims``, ``pre_dims``, ...) can share the one dict.
    That leniency is why every algorithm declares ``PARAMS``: a key nothing reads
    is a typo, and ``ClfAlgorithm`` rejects it before this function sees it rather
    than letting it be dropped in silence.

    ``overrides`` lets an algorithm change a default it disagrees with before the
    config is applied.
    """
    known = set(TRAIN_HPARAM_NAMES)
    kwargs = {k: v for k, v in overrides.items() if k in known}
    kwargs.update({k: v for k, v in params.items() if k in known})
    return TrainHParams(**kwargs)


def hparams_dict(hp: TrainHParams) -> dict:
    """The shared hyper-parameters as a plain dict, defaults resolved."""
    return asdict(hp)


def parse_dims(value, name: str) -> list[int]:
    """Normalise a layer-width spec to a list of positive ints.

    Accepts a list (``[512, 128]``), a bare int for a single layer (``256``) and
    an empty list, which is the meaningful degenerate case -- a linear probe for
    ``hidden_dims``, early fusion for ``pre_dims``. Anything else raises, so a
    ``hidden_dims: 256.5`` or a stray string is a preflight error rather than a
    ``TypeError`` several minutes into the first fit.
    """
    if isinstance(value, bool):  # bool is an int; almost certainly a mistake
        raise ValueError(f"{name} must be a list of widths or an int, got {value!r}")
    if isinstance(value, int):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{name} must be a list of layer widths (or a single int), got "
            f"{type(value).__name__} {value!r}"
        )
    out: list[int] = []
    for v in value:
        try:
            width = int(v)
        except (TypeError, ValueError):
            raise ValueError(f"{name} holds a non-integer width: {v!r}") from None
        if width < 1:
            raise ValueError(f"{name} widths must be >= 1, got {v!r}")
        out.append(width)
    return out


# ---------------------------------------------------------------------------
# Feature normalisation
# ---------------------------------------------------------------------------


def parse_normalize(mode: str) -> list[str]:
    """``"l2+standard"`` -> ``["l2", "standard"]``, validated.

    ``"none"`` yields an empty pipeline. Steps are applied in the order written.
    """
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError(f"normalize must be a non-empty string, got {mode!r}")
    steps = [s.strip() for s in mode.split("+") if s.strip()]
    bad = [s for s in steps if s not in NORMALIZE_STEPS]
    if bad:
        raise ValueError(
            f"unknown normalize step(s) {bad}; choose from {sorted(NORMALIZE_STEPS)} "
            f"joined with '+', e.g. 'l2+standard'"
        )
    return [s for s in steps if s != "none"]


class FeatureNormalizer:
    """A small fit/transform pipeline over the normalisation steps in a config.

    ``standard`` is :class:`StandardScaler`: it learns the *source* corpus's mean
    and standard deviation and carries them onto every target. ``l2`` scales each
    row to unit length, which learns nothing and so transfers unchanged. The two
    compose, and which combination wins is exactly the kind of thing the sweep is
    for.
    """

    #: Rows with (near) zero norm would divide by ~0; floor the divisor instead.
    EPS = 1e-12

    def __init__(self, mode: str = "standard"):
        self.mode = mode
        self.steps = parse_normalize(mode)
        self.scalers_: dict[int, StandardScaler] = {}

    def fit(self, X: np.ndarray) -> "FeatureNormalizer":
        self.fit_transform(X)
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        self._check_width(X)
        self.scalers_ = {}
        for i, step in enumerate(self.steps):
            if step == "l2":
                X = self._l2(X)
            else:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                self.scalers_[i] = scaler
        return X.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        for i, step in enumerate(self.steps):
            X = self._l2(X) if step == "l2" else self.scalers_[i].transform(X)
        return X.astype(np.float32)

    def _l2(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, self.EPS)

    def _check_width(self, X: np.ndarray) -> None:
        # L2-normalising a single column maps every non-zero value to +/-1, which
        # destroys the column outright. Silently doing that to a Binoculars score
        # in the middle of a sweep would be very hard to notice in the results.
        if "l2" in self.steps and X.shape[1] < 2:
            raise ValueError(
                f"normalize={self.mode!r} includes 'l2', but the matrix has "
                f"{X.shape[1]} column(s): per-row L2 on a single column collapses "
                f"it to its sign. Use 'standard' or 'none' for scalar inputs."
            )


# ---------------------------------------------------------------------------
# The auxiliary matrix
# ---------------------------------------------------------------------------


#: How a non-finite auxiliary entry is replaced. See :func:`aux_fill_values`.
AUX_FILL_MODES = ("zero", "median")


def parse_aux_fill(mode: str) -> str:
    """Validate an ``aux_fill`` setting. Returns it unchanged."""
    if mode not in AUX_FILL_MODES:
        raise ValueError(
            f"aux_fill must be one of {sorted(AUX_FILL_MODES)}, got {mode!r}"
        )
    return mode


def as_aux_matrix(Z) -> np.ndarray:
    """The auxiliary input as a 2-D float64 matrix, whatever width it has.

    A scalar column (the Binoculars score) arrives as ``(n,)`` and a vector one
    (a judge's cue vector) as ``(n, k)``; both become ``(n, k>=1)`` here so every
    caller downstream is written once, for a matrix.
    """
    return np.asarray(Z, dtype=np.float64).reshape(len(Z), -1)


def aux_fill_values(Z, mode: str) -> np.ndarray:
    """Per-column stand-in for a non-finite auxiliary entry, fitted on train.

    Which stand-in is right depends on what the column *is*, so it is a config
    choice rather than a constant:

    ``median``
        For a continuous score. A failed Binoculars computation leaves a NaN or
        an inf, and a single one of those would take a StandardScaler's mean and
        std with it; the training median is the least committal replacement.

    ``zero``
        For a multi-hot indicator vector, where the training median is a
        fraction that no real row can hold and that a downstream ReLU basis
        would read as a half-cited cue. 0.0 is the vector's own word for "this
        cue was not cited", which is the honest reading of a missing entry --
        the rows that were never *looked at* are dropped upstream by the
        loader's missing-value filter and never reach this function.

    The value is computed once on the training split and reused unchanged on
    every target corpus, exactly like the normaliser's statistics.
    """
    Z = as_aux_matrix(Z)
    if parse_aux_fill(mode) == "zero":
        return np.zeros(Z.shape[1], dtype=np.float64)
    finite = np.where(np.isfinite(Z), Z, np.nan)
    return np.nan_to_num(np.nanmedian(finite, axis=0), nan=0.0)


def clean_aux(Z, fill: np.ndarray) -> np.ndarray:
    """Replace every non-finite auxiliary entry with its column's ``fill`` value."""
    Z = as_aux_matrix(Z)
    bad = ~np.isfinite(Z)
    if bad.any():
        Z = np.where(bad, fill, Z)
    return Z


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------


def hidden_block(in_dim: int, out_dim: int, dropout: float, norm_layer: str) -> list:
    """One ``Linear -> [LayerNorm] -> ReLU -> Dropout`` stack.

    LayerNorm rather than BatchNorm on purpose: BatchNorm's running statistics
    are the *source* corpus's, so a BatchNorm model carries the training corpus's
    mean and variance into every transfer evaluation -- the same problem the
    ``standard`` normaliser has, moved inside the network. LayerNorm normalises
    per sample and has no such state.
    """
    layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
    if norm_layer == "layernorm":
        layers.append(nn.LayerNorm(out_dim))
    layers += [nn.ReLU(), nn.Dropout(dropout)]
    return layers


def mlp_layers(in_dim: int, hidden_dims: list[int], hp: TrainHParams) -> tuple[list, int]:
    """The hidden stack for ``hidden_dims``, plus the width it ends at.

    An empty ``hidden_dims`` yields just the input dropout, i.e. a linear probe
    once the caller appends its logit head. That degenerate case is worth keeping
    reachable: on embedding features a linear probe often transfers better than
    anything deeper, and it is the control the deeper models should be read
    against.
    """
    layers: list[nn.Module] = []
    if hp.input_dropout:
        layers.append(nn.Dropout(hp.input_dropout))
    prev = in_dim
    for h in hidden_dims:
        layers += hidden_block(prev, h, hp.dropout, hp.norm_layer)
        prev = h
    return layers, prev


def aux_layers(in_dim: int, dims: list[int], dropout: float) -> tuple[list, int]:
    """The hidden stack for an auxiliary branch, plus the width it ends at.

    Two deliberate differences from :func:`mlp_layers`, both of which exist to
    keep a *small* input intact where the embedding path is free to be squeezed:

    * no input dropout. Zeroing one of eleven cue bits removes an eleventh of
      everything the branch knows, where zeroing one of 512 embedding dimensions
      removes something the other 511 largely still carry.
    * no LayerNorm, whatever ``norm_layer`` says. LayerNorm rescales each sample
      to unit variance across the layer's features, which throws away the branch
      activation's overall magnitude -- and for a multi-hot cue vector that
      magnitude is *how many cues were cited*, one of the few unambiguous
      signals eleven bits carry.

    An empty ``dims`` yields an empty stack, i.e. the raw auxiliary vector is
    passed through unchanged.
    """
    layers: list[nn.Module] = []
    prev = in_dim
    for h in dims:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        prev = h
    return layers, prev


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainHistory:
    """What the training loop did, recorded for the sweep's leaderboard."""

    epochs_run: int = 0
    best_epoch: int = 0
    best_score: float = float("nan")
    stopped_early: bool = False
    used_val: bool = False
    n_train: int = 0
    n_val: int = 0
    val_fallback: str = ""  # why a requested val split was not used
    train_loss: list[float] = field(default_factory=list)
    val_score: list[float] = field(default_factory=list)


def split_train_val(
    n: int,
    hp: TrainHParams,
    seed: int,
    groups: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Carve a validation split out of the training rows.

    Grouped when ``groups`` is given, which it must be for these corpora: a human
    document and its LLM rewrites share an id, so a random split would put a
    rewrite of the same source text on both sides and the validation score --
    and therefore the epoch early stopping picks -- would be optimistic.

    Returns ``(train_idx, val_idx, fallback_reason)``. The reason is non-empty
    when a validation split was asked for but could not be used, in which case
    ``val_idx`` is empty and the caller falls back to the training loss.
    """
    if hp.val_frac <= 0:
        return np.arange(n), np.empty(0, dtype=int), ""

    splitter_cls = GroupShuffleSplit if groups is not None else ShuffleSplit
    splitter = splitter_cls(n_splits=1, test_size=hp.val_frac, random_state=seed)
    try:
        train_idx, val_idx = next(splitter.split(np.zeros((n, 1)), groups=groups))
    except ValueError as exc:  # too few rows / groups to split at this fraction
        return np.arange(n), np.empty(0, dtype=int), f"split failed: {exc}"

    if len(val_idx) == 0 or len(train_idx) == 0:
        return np.arange(n), np.empty(0, dtype=int), "split left one side empty"
    # A single-class validation set makes accuracy meaningless and the loss a
    # poor stopping signal; better to say so and fall back than to stop on noise.
    if y is not None and len(np.unique(np.asarray(y)[val_idx])) < 2:
        return np.arange(n), np.empty(0, dtype=int), "validation split is single-class"
    return train_idx, val_idx, ""


def _mixup(xs: list[torch.Tensor], alpha: float, rng: np.random.Generator):
    """Convex-combine each batch with a shuffled copy of itself.

    Returns ``(mixed_inputs, permutation, lam)``. The same lambda and permutation
    are used for every input tensor so that a hybrid's embedding and its score
    stay paired -- mixing them independently would build rows whose score belongs
    to a different document than the embedding.
    """
    lam = float(rng.beta(alpha, alpha))
    perm = torch.randperm(xs[0].shape[0], device=xs[0].device)
    return [lam * x + (1.0 - lam) * x[perm] for x in xs], perm, lam


# ---------------------------------------------------------------------------
# Tail-aware loss (TopPush) + domain-invariance penalties (CORAL / V-REx)
#
# See epai/ai_detection/trainv2/improving_sota_ideas.md (#1, #2) for the
# method and citations. Only ``robust_mlp.RobustMLPAlg`` sets ``loss_kind`` /
# ``domain_penalty`` away from their defaults, so these branches are dead code
# for every other algorithm in this package.
# ---------------------------------------------------------------------------


def _margin(logits: torch.Tensor) -> torch.Tensor:
    """Differentiable machine-minus-human logit margin (keeps grad; contrast
    with :func:`logit_margin`, which detaches to numpy)."""
    return logits[:, 1] - logits[:, 0]


def _push_loss(logits: torch.Tensor, y: torch.Tensor, push_frac: float,
               margin: float) -> torch.Tensor:
    """TopPush: a pairwise hinge between every machine row and only the
    *hardest* (highest-scoring) ``push_frac`` fraction of the batch's human
    rows -- the surrogate for TPR@low-FPR, not for full AUROC. ``topk`` keeps
    gradient through the selected human scores."""
    s = _margin(logits)
    hum, mach = s[y == 0], s[y == 1]
    if hum.numel() == 0 or mach.numel() == 0:
        return torch.zeros((), device=logits.device)
    k = min(hum.numel(), max(1, int(np.ceil(push_frac * hum.numel()))))
    hard_hum = torch.topk(hum, k).values
    diff = mach.unsqueeze(1) - hard_hum.unsqueeze(0)  # (n_mach, k)
    return F.softplus(margin - diff).mean()


def _domain_codes(domains: np.ndarray) -> np.ndarray:
    """String/mixed domain labels -> stable small-int codes."""
    _, codes = np.unique(np.asarray(domains), return_inverse=True)
    return codes.astype(np.int64)


def _coral_penalty(features: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
    """Deep CORAL: mean pairwise squared Frobenius distance between domains'
    feature covariance matrices, computed over whatever domains happen to be
    present in this batch. A domain with < 2 rows in the batch is skipped (no
    covariance); < 2 domains present -> zero penalty for this batch."""
    uniq = torch.unique(domains)
    covs = []
    for d in uniq:
        f = features[domains == d]
        if f.shape[0] < 2:
            continue
        f = f - f.mean(dim=0, keepdim=True)
        covs.append((f.T @ f) / (f.shape[0] - 1))
    if len(covs) < 2:
        return torch.zeros((), device=features.device)
    d2 = features.shape[1] ** 2
    terms = [((covs[i] - covs[j]) ** 2).sum() / (4 * d2)
             for i in range(len(covs)) for j in range(i + 1, len(covs))]
    return torch.stack(terms).mean()


def _vrex_penalty(per_row_loss: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
    """V-REx: variance of the per-domain mean loss within this batch. A domain
    with 0 rows in the batch is skipped; < 2 domains present -> zero."""
    uniq = torch.unique(domains)
    means = [per_row_loss[domains == d].mean() for d in uniq
             if (domains == d).any()]
    if len(means) < 2:
        return torch.zeros((), device=per_row_loss.device)
    return torch.stack(means).var(unbiased=False)


def train_torch_model(
    model: nn.Module,
    inputs: list[np.ndarray],
    y: np.ndarray,
    hp: TrainHParams,
    device: torch.device,
    seed: int,
    groups: np.ndarray | None = None,
    domains: np.ndarray | None = None,
) -> TrainHistory:
    """Fit ``model`` under ``hp``. Shared by every torch algorithm here.

    ``inputs`` is the list of already-normalised design matrices the model's
    ``forward`` takes, in order (one for the plain MLP, two for the hybrid), so
    this loop never needs to know which architecture it is training.

    With ``val_frac > 0`` the loop early-stops on the held-out split and restores
    the best weights before returning. With ``val_frac == 0`` it early-stops on
    the training loss, which is what this package did before and is kept as the
    default so old configs reproduce -- see the module docstring for why it is
    not really regularisation.

    ``domains``, when given AND ``hp.domain_penalty != "none"``, is one
    environment label per row of ``y`` (see
    ``epai/ai_detection/trainv2/improving_sota_ideas.md`` #2). In that case
    ``model.forward`` is called with ``return_features=True`` inside the batch
    loop (and only there -- inference elsewhere still gets plain logits), so a
    model that wants the domain penalty must accept that keyword and return
    ``(logits, features)``. Every model in this package except
    ``robust_mlp._RobustMLP`` is unaffected, since nothing else sets
    ``domain_penalty``.
    """
    n = len(y)
    train_idx, val_idx, fallback = split_train_val(n, hp, seed, groups, y)
    use_val = len(val_idx) > 0

    y_arr = np.asarray(y, dtype=np.int64)
    tensors = [torch.from_numpy(np.ascontiguousarray(x)).to(device) for x in inputs]
    y_t = torch.from_numpy(y_arr).to(device)

    has_dom = domains is not None and hp.domain_penalty != "none"
    if has_dom:
        dom_t = torch.from_numpy(_domain_codes(domains)).to(device)

    tr = torch.from_numpy(train_idx).to(device)
    train_tensors = [t[tr] for t in tensors]
    y_train = y_t[tr]
    dom_train = dom_t[tr] if has_dom else None
    if use_val:
        va = torch.from_numpy(val_idx).to(device)
        val_tensors = [t[va] for t in tensors]
        y_val = y_t[va]

    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=hp.label_smoothing)
    use_push = hp.loss_kind == "toppush"
    dataset_tensors = [*train_tensors, y_train] + ([dom_train] if has_dom else [])
    loader = DataLoader(
        TensorDataset(*dataset_tensors),
        batch_size=hp.batch_size,
        shuffle=True,
    )
    rng = np.random.default_rng(seed)

    higher_is_better = VAL_METRICS[hp.val_metric]
    best = -float("inf") if (use_val and higher_is_better) else float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    hist = TrainHistory(
        used_val=use_val,
        n_train=len(train_idx),
        n_val=len(val_idx),
        val_fallback=fallback,
    )

    for epoch in range(hp.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            if has_dom:
                xb, yb, db = list(batch[:-2]), batch[-2], batch[-1]
            else:
                xb, yb, db = list(batch[:-1]), batch[-1], None
            opt.zero_grad()
            if hp.mixup_alpha > 0 and len(yb) > 1:
                # Guaranteed not combined with has_dom/use_push -- see
                # TrainHParams.__post_init__.
                xb, perm, lam = _mixup(xb, hp.mixup_alpha, rng)
                out = model(*xb)
                loss = lam * loss_fn(out, yb) + (1.0 - lam) * loss_fn(out, yb[perm])
            else:
                out, feats = model(*xb, return_features=True) if has_dom else (model(*xb), None)
                per_row = None
                if has_dom and hp.domain_penalty == "vrex":
                    # V-REx reads the per-domain mean of the SAME per-row CE
                    # this batch's base loss uses, so compute it un-reduced.
                    per_row = F.cross_entropy(out, yb, reduction="none",
                                              label_smoothing=hp.label_smoothing)
                    loss = per_row.mean()
                else:
                    loss = loss_fn(out, yb)
                if use_push:
                    loss = loss + hp.push_weight * _push_loss(
                        out, yb, hp.push_frac, hp.push_margin)
                if has_dom:
                    pen = (_coral_penalty(feats, db) if hp.domain_penalty == "coral"
                          else _vrex_penalty(per_row, db))
                    loss = loss + hp.domain_penalty_weight * pen
            loss.backward()
            if hp.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= max(len(loader), 1)
        hist.train_loss.append(epoch_loss)
        hist.epochs_run = epoch + 1

        if use_val:
            score = _val_score(model, val_tensors, y_val, hp, loss_fn)
            hist.val_score.append(score)
            improved = (score > best + 1e-5) if higher_is_better else (score < best - 1e-5)
        else:
            score = epoch_loss
            improved = score < best - 1e-4

        if improved:
            best = score
            best_epoch = epoch + 1
            no_improve = 0
            if use_val:
                # Only worth the copy when there is a real held-out score to
                # restore to; on the training loss the last epoch is the best one.
                best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if hp.patience and no_improve >= hp.patience:
                hist.stopped_early = True
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    hist.best_epoch = best_epoch
    hist.best_score = float(best)
    return hist


@torch.no_grad()
def _val_score(
    model: nn.Module,
    val_tensors: list[torch.Tensor],
    y_val: torch.Tensor,
    hp: TrainHParams,
    loss_fn: nn.Module,
) -> float:
    """One validation pass. Evaluated in one shot -- these splits are small."""
    model.eval()
    logits = model(*val_tensors)
    if hp.val_metric == "accuracy":
        return float((logits.argmax(dim=1) == y_val).float().mean().item())
    if hp.val_metric == "auroc":
        # split_train_val guarantees both classes are present here, so this is
        # never the degenerate single-class case roc_auc_score rejects.
        return float(
            roc_auc_score(y_val.cpu().numpy(), logit_margin(logits))
        )
    return float(loss_fn(logits, y_val).item())
