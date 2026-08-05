r"""One preflight: collect every problem with a run before anything is fitted.

The cross-domain runners are slow -- fifteen embedders x nine matrix cells x
three seeds -- so the failures worth catching are the ones that would otherwise
surface twenty minutes in: a column missing from the third corpus, a typo in a
``domain``, an ``ml_alg`` that does not exist, a split that leaves one class
empty. All three runners used to carry their own copy of this scaffolding, which
is how they drifted into accepting slightly different configs.

The rule everywhere: run every check to the end, record what failed, print it
once, and exit before training. A run this long should not be restarted once per
typo.

Two pieces live here -- :class:`Report`, which accumulates errors and warnings
against the config file they belong to, and the config-shape checks every runner
shares. What each runner still owns is what only it understands: the
``embeddings`` list, the ``algorithms`` list, its baseline blocks. Dataset
checks live next to the loader, in :mod:`epai.utils.corpus_utils`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from epai.utils.config_utils import RESET, YELLOW

RED = "\033[91m"
GREEN = "\033[92m"
RULE = 78

# Classifiers train_utils.build_clf_model knows about. Checked up front so a
# typo does not survive until the first fit.
ML_ALGS = ("gradient_boosting", "logistic", "svm", "mlp")


@dataclass
class Report:
    """Problems found by the preflight check.

    Errors abort the run before any model is fitted; warnings are printed and
    the run continues. ``context`` is the config file every message is about --
    several configs are checked in one pass, so an unattributed message would
    not say which file to fix.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    context: str = ""

    def _tag(self, message: str) -> str:
        return f"[{self.context}] {message}" if self.context else message

    def error(self, message: str) -> None:
        self.errors.append(self._tag(message))

    def warn(self, message: str) -> None:
        self.warnings.append(self._tag(message))

    @property
    def ok(self) -> bool:
        return not self.errors


def print_report(report: Report, subject: str = "every config, dataset and split") -> None:
    print()
    print("=" * RULE)
    if report.warnings:
        print(f"{YELLOW}Warnings ({len(report.warnings)}){RESET}")
        for message in report.warnings:
            print(f"  {YELLOW}⚠️  {message}{RESET}")
        print()
    if report.errors:
        print(f"{RED}Preflight FAILED — {len(report.errors)} problem(s){RESET}")
        for message in report.errors:
            print(f"  {RED}✗ {message}{RESET}")
        print(f"{RED}Nothing was trained; fix the above and re-run.{RESET}")
    else:
        print(f"{GREEN}Preflight OK — {subject} checks out.{RESET}")
    print("=" * RULE)


def load_yaml(path: Path, report: Report) -> dict | None:
    """Parse one config, turning a YAML error into a report entry."""
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        report.error(f"cannot parse YAML: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(cfg, dict):
        report.error(f"config must be a YAML mapping, got {type(cfg).__name__}")
        return None
    return cfg


# ---------------------------------------------------------------------------
# Config-shape checks shared by every runner
# ---------------------------------------------------------------------------


def check_identity(cfg: dict, report: Report) -> None:
    """``alg_id`` / ``alg_label`` / ``seed`` -- what names the output directory."""
    for key in ("alg_id", "alg_label", "seed"):
        if key not in cfg:
            report.error(f"missing top-level key {key!r}")
    if "alg_id" in cfg and (not isinstance(cfg["alg_id"], str) or not cfg["alg_id"].strip()):
        report.error("'alg_id' must be a non-empty string")
    if "seed" in cfg and not isinstance(cfg["seed"], int):
        report.error(f"'seed' must be an int, got {type(cfg['seed']).__name__}")


def check_model_block(cfg: dict, report: Report, check_alg: bool = True) -> None:
    """``model.ml_alg`` / ``n_runs`` / ``test_size``, plus the run-size guards."""
    model = cfg.get("model")
    if not isinstance(model, dict):
        report.error("'model' must be a mapping")
        return

    if check_alg:
        ml_alg = model.get("ml_alg")
        if ml_alg not in ML_ALGS:
            report.error(f"unknown ml_alg {ml_alg!r}. Choose from: {list(ML_ALGS)}")
        else:
            from epai.utils.train_utils import build_clf_model  # noqa: PLC0415

            try:
                build_clf_model(ml_alg, cfg.get("seed", 0))
            except Exception as exc:  # a known name that still cannot be built
                report.error(f"ml_alg {ml_alg!r} cannot be built: {exc}")

    n_runs = model.get("n_runs")
    if not isinstance(n_runs, int) or n_runs < 1:
        report.error(f"'n_runs' must be an int >= 1, got {n_runs!r}")

    test_size = model.get("test_size", 0.30)
    if not isinstance(test_size, (int, float)) or not 0.0 < float(test_size) < 1.0:
        report.error(f"'test_size' must be in (0, 1), got {test_size!r}")

    min_rows = cfg.get("min_rows", 50)
    if not isinstance(min_rows, int) or min_rows < 1:
        report.error(f"'min_rows' must be an int >= 1, got {min_rows!r}")

    min_share = cfg.get("min_class_share", 0.10)
    if not isinstance(min_share, (int, float)) or not 0.0 <= float(min_share) < 0.5:
        report.error(f"'min_class_share' must be in [0, 0.5), got {min_share!r}")
