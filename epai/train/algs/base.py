"""Abstract base for the pluggable cross-domain classifiers."""

from __future__ import annotations

import difflib
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np


class ClfAlgorithm(ABC):
    """A binary human-vs-machine classifier for the cross-domain comparison.

    One instance is fit on a single corpus's training split and then asked to
    predict on that corpus's held-out split (in-domain) and on every other
    corpus (transfer). Implementations own their own preprocessing -- feature
    scaling, for example -- so the runner only ever hands them the raw stacked
    design matrix and calls ``fit`` / ``predict``.

    To add an algorithm:
      * subclass this and set the class attribute ``name`` (the key a config
        selects it by and the folder its results land in);
      * declare ``PARAMS``, the hyper-parameter names it reads (see below);
      * implement ``fit(X, y)`` and ``predict(X)``;
      * register it in ``epai/train/algs/__init__.py``;
      * optionally override ``save`` if the default joblib pickle is unsuitable.

    ``params`` captures whatever keyword arguments the config passed, so an
    algorithm is fully described by ``(name, seed, params)`` and each subclass
    reads its own hyper-parameters out of ``self.params`` with sensible
    defaults.

    The hyper-parameter contract
    ----------------------------
    Because every subclass reads ``self.params`` with ``.get(name, default)``, a
    key nothing reads has no effect at all and no error either. That is a bad
    trade in this codebase: hyper-parameters are found by the sweep
    (``sweep_clf_cross_domain.py``) and then transcribed by hand into a
    ``clf_cd_algs`` config, and a mistranscribed name would quietly train the
    *default* model while the results directory still recorded the params that
    were asked for. So each subclass declares ``PARAMS`` -- the names it actually
    reads -- and an entry naming anything else is rejected at construction, which
    the runner's preflight turns into an error before a single row is loaded.

    Two hooks make a setting inspectable rather than merely accepted:
    :meth:`validate_params` checks *values* (ranges, enums, layer widths) at
    preflight time instead of inside ``fit``, and :meth:`resolved_params` reports
    every hyper-parameter that will actually be used with the defaults filled in,
    which the runner prints and records beside each matrix.

    An algorithm that wants a *second* input beside the design matrix -- the
    Binoculars score, say, kept out of ``X`` so it can enter the network late --
    sets ``uses_aux = True`` and takes a third argument ``Z`` on ``fit`` and a
    second on ``predict``. The runner passes ``Z`` only to those algorithms, so
    every other subclass keeps the plain two-argument signature, and a config
    that names an auxiliary column for an algorithm which does not declare
    ``uses_aux`` is a preflight error rather than a silently ignored key.
    """

    #: Registry key / output-folder name. Overridden by every concrete subclass.
    name: str = "base"

    #: Does ``fit`` / ``predict`` take the auxiliary matrix ``Z``? See above.
    uses_aux: bool = False

    #: Hyper-parameter names a config entry may set on this algorithm, i.e. the
    #: ones ``fit`` reads. An empty tuple means "accept anything", kept as the
    #: default only so a subclass outside this package is not broken by the check.
    PARAMS: tuple[str, ...] = ()

    def __init__(self, seed: int = 0, **params):
        self.seed = int(seed)
        self.params = dict(params)
        self.fit_groups_: np.ndarray | None = None
        self._check_param_names()

    @classmethod
    def accepted_params(cls) -> list[str]:
        """The hyper-parameter names a config may set, sorted. For messages."""
        return sorted(cls.PARAMS)

    def _check_param_names(self) -> None:
        """Reject params this algorithm does not read. See the class docstring."""
        if not self.PARAMS:
            return
        accepted = set(self.PARAMS)
        unknown = sorted(k for k in self.params if k not in accepted)
        if not unknown:
            return
        detail = []
        for key in unknown:
            # The realistic failure is a near-miss ('val_fraction', 'hidden_dim'),
            # so name the intended key rather than only the full accepted list.
            close = difflib.get_close_matches(key, sorted(accepted), n=1)
            detail.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise ValueError(
            f"{self.name}: unknown hyper-parameter(s) {', '.join(detail)}. "
            f"Accepted: {self.accepted_params()}"
        )

    def validate_params(self) -> None:
        """Check hyper-parameter *values*, without touching any data.

        Called by the runner's preflight so a bad range, enum or layer width is an
        error before the corpora are read, rather than an exception inside the
        first ``fit``. Default: nothing to check.
        """

    def resolved_params(self) -> dict:
        """Every hyper-parameter that will be used, defaults filled in.

        The config only carries what was overridden, so this is what tells you
        which model actually ran. Default: the config entry as given.
        """
        return dict(self.params)

    def set_fit_groups(self, groups: np.ndarray | None) -> None:
        """Tell the algorithm which group each training row belongs to.

        Optional context, not data: an algorithm that carves its own validation
        split out of the training rows needs it to keep a document and its
        rewrites on the same side of that split, exactly as the runner's outer
        train/test split does. Algorithms that take no validation split ignore
        it, and the attribute stays None when the caller does not set it, so
        every existing algorithm is unaffected.
        """
        self.fit_groups_ = None if groups is None else np.asarray(groups)

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClfAlgorithm":
        """Fit on one corpus's training split. Returns ``self``."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted class labels (0 = human, 1 = machine), one per row of ``X``."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Persist the fitted algorithm. Default: joblib-pickle the whole object.

        Best-effort and overridable; the runner wraps the call so a
        serialization failure never aborts a long run.
        """
        joblib.dump(self, path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(seed={self.seed}, params={self.params})"
