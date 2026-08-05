"""Hyper-parameter search over one algorithm's cross-domain transfer.

``run_clf_cross_domain_algs.py`` compares a handful of algorithms you wrote out
by hand. This drives the same machinery over hundreds of sampled
hyper-parameter settings instead, so a long regularisation search is one command
on a server rather than a folder of hand-edited YAML.

It reuses the runner rather than reimplementing it: the corpora are loaded and
stacked once, then every trial becomes one ``algorithms`` entry handed to
``run_algorithm``. So a trial is scored by exactly the same grouped split,
binarised target and group-exclusion rule as a normal run, and the winner's
numbers are directly comparable to every matrix already in ``results/``.

What it is *not*: an LLM-in-the-loop that rewrites code between trials. For ten
or so numeric knobs, random search over a log-uniform prior is at least as good
and costs nothing per trial. The place an LLM earns its keep is one level up --
deciding which regulariser should exist next round, after reading a leaderboard
-- so :func:`propose` is the seam where such a proposer would go, taking the
history and returning the next batch of parameter dicts.

Selection bias
--------------
The point of the run is transfer, so the obvious thing is to rank trials on
mean off-diagonal accuracy. With three corpora that is six numbers, and a few
hundred trials will reliably find a setting that looks several points better on
those six purely by fitting their noise. ``holdout`` is the defence: the named
corpora are removed from the search entirely -- not just from the objective, so
they cost no compute either -- and the winner meets them for the first time in
``--final``. That final number is the one to believe. An empty ``holdout`` is
allowed and warned about loudly; then the winning score is an optimistic
estimate and the honest reading of a sweep gain is "hypothesis", not "result".

Run:
    # search (resumable: re-running continues where it stopped)
    python -m epai.ai_detection.train.sweep_clf_cross_domain --cfg cfg/ai_detection/train/sweep/luar_mlp_reg.yaml

    # look at what it found so far, from another shell, while it runs
    python -m epai.ai_detection.train.sweep_clf_cross_domain --cfg ... --report

    # print the winner as a clf_cd_algs 'algorithms' entry, ready to paste
    python -m epai.ai_detection.train.sweep_clf_cross_domain --cfg ... --params

    # re-run the winner on the full matrix, holdout included, with baselines and figures
    python -m epai.ai_detection.train.sweep_clf_cross_domain --cfg ... --final
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from epai.ai_detection.train.run_clf_cross_domain_algs import (
    BASE_DIR,
    load_columns,
    preflight_config,
    print_report,
    run_algorithm,
    run_config,
)
from epai.ai_detection.train.run_clf_cross_domain_embeddings import (
    Report,
    load_prepared,
    summarise_embedding,
)
from epai.ai_detection.train.plot_sweep import (
    draw_plots,
    read_trials_file,
)
from epai.utils.config_utils import RESET, YELLOW, rel
from epai.utils.corpus_utils import dataset_specs
from epai.utils.train_utils import stack_features

DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "sweep_clf_cd"
DEFAULT_CFG_DIR = BASE_DIR / "cfg" / "ai_detection" / "train" / "sweep"

GREEN = "\033[92m"
RED = "\033[91m"
DIM = "\033[2m"
RULE = 100

#: Metrics of ``summarise_embedding`` a sweep may rank on.
OBJECTIVE_METRICS = {
    "transfer_mean",
    "transfer_min",
    "transfer_f1_mean",
    "in_domain_mean",
    "gap",
}

#: One trial's record in trials.jsonl. Every field is written even when the
#: trial is rejected, so a resumed sweep never re-runs a known-bad setting.
TRIAL_FIELDS = ("trial_id", "stage", "params", "objective", "rejected")


# ---------------------------------------------------------------------------
# Sweep config
# ---------------------------------------------------------------------------


def load_sweep_config(path: Path) -> dict:
    """Read and validate a sweep config. Raises with a pointed message on error."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: not a YAML mapping")

    missing = {"sweep_id", "base", "search", "algorithm"} - cfg.keys()
    if missing:
        raise ValueError(f"{path}: missing top-level keys: {sorted(missing)}")

    if not isinstance(cfg["search"], dict) or not cfg["search"]:
        raise ValueError(f"{path}: 'search' must be a non-empty mapping")
    if not isinstance(cfg["algorithm"], dict) or "name" not in cfg["algorithm"]:
        raise ValueError(f"{path}: 'algorithm' must be a mapping with a 'name'")

    obj = cfg.setdefault("objective", {})
    obj.setdefault("metric", "transfer_min")
    obj.setdefault("direction", "max")
    obj.setdefault("min_in_domain", 0.0)
    if obj["metric"] not in OBJECTIVE_METRICS:
        raise ValueError(
            f"{path}: objective.metric must be one of {sorted(OBJECTIVE_METRICS)}, "
            f"got {obj['metric']!r}"
        )
    if obj["direction"] not in {"max", "min"}:
        raise ValueError(f"{path}: objective.direction must be 'max' or 'min'")

    halving = cfg.setdefault("halving", {})
    halving.setdefault("enabled", True)
    halving.setdefault("stage1_n_runs", 1)
    halving.setdefault("keep_frac", 0.25)
    halving.setdefault("stage2_n_runs", 3)
    if not 0 < halving["keep_frac"] <= 1:
        raise ValueError(f"{path}: halving.keep_frac must be in (0, 1]")

    cfg.setdefault("strategy", "random")
    if cfg["strategy"] not in {"random", "grid"}:
        raise ValueError(f"{path}: strategy must be 'random' or 'grid'")
    cfg.setdefault("n_trials", 100)
    cfg.setdefault("seed", 1234)
    cfg.setdefault("holdout", [])
    if not isinstance(cfg["holdout"], list):
        raise ValueError(f"{path}: 'holdout' must be a list of dataset names")

    # Validate every distribution now, so a typo in the space fails before the
    # corpora are read rather than several hundred trials in.
    for name, spec in cfg["search"].items():
        parse_distribution(name, spec)

    return cfg


def resolve_base_config(sweep_cfg: dict, sweep_path: Path) -> tuple[Path, dict]:
    """Load the cross-domain config the sweep varies.

    Read the same way the runner reads it -- plain YAML, then the runner's own
    ``preflight_config``. Deliberately *not* ``config_utils.load_cd_config``:
    that validator requires ``model.ml_alg``, which a clf_cd_algs config does not
    carry (the algorithm lives in the ``algorithms`` list instead), so it rejects
    exactly the configs this sweep exists to search over.
    """
    base_path = BASE_DIR / sweep_cfg["base"]
    if not base_path.is_file():
        raise FileNotFoundError(
            f"{sweep_path}: base config '{sweep_cfg['base']}' not found at {base_path}"
        )
    with open(base_path) as f:
        return base_path, yaml.safe_load(f)


def split_datasets(base_cfg: dict, holdout: list[str]) -> tuple[list[dict], list[dict]]:
    """Partition the base config's datasets into (dev pool, holdout).

    The holdout corpora are dropped from the search entirely rather than merely
    excluded from the objective: a corpus the search never trains on and never
    evaluates on also costs nothing to keep out, and only a corpus that has
    influenced nothing gives an honest final number.
    """
    all_specs = base_cfg["data"]["datasets"]
    known = {spec["name"] for spec in all_specs}
    unknown = [h for h in holdout if h not in known]
    if unknown:
        raise ValueError(
            f"holdout names {unknown} are not datasets in the base config "
            f"(have: {sorted(known)})"
        )
    dev = [s for s in all_specs if s["name"] not in set(holdout)]
    held = [s for s in all_specs if s["name"] in set(holdout)]
    if len(dev) < 2:
        raise ValueError(
            f"the search needs at least 2 dev datasets to have any off-diagonal "
            f"cell to optimise; holdout={holdout} leaves {len(dev)}"
        )
    return dev, held


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------


def parse_distribution(name: str, spec) -> dict:
    """Normalise one search-space entry to ``{kind, ...}``.

    Shorthands: a bare list is ``choice``, any other bare scalar is ``const``.
    """
    if isinstance(spec, list):
        return {"kind": "choice", "values": spec}
    if not isinstance(spec, dict):
        return {"kind": "const", "value": spec}

    kind = spec.get("kind")
    if kind == "const":
        if "value" not in spec:
            raise ValueError(f"search.{name}: 'const' needs a 'value'")
        return {"kind": "const", "value": spec["value"]}
    if kind == "choice":
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"search.{name}: 'choice' needs a non-empty 'values' list")
        return {"kind": "choice", "values": values}
    if kind in {"uniform", "loguniform", "int_uniform", "int_loguniform"}:
        if "low" not in spec or "high" not in spec:
            raise ValueError(f"search.{name}: '{kind}' needs 'low' and 'high'")
        low, high = float(spec["low"]), float(spec["high"])
        if high < low:
            raise ValueError(f"search.{name}: high ({high}) < low ({low})")
        if kind.endswith("loguniform") and low <= 0:
            raise ValueError(f"search.{name}: '{kind}' needs low > 0, got {low}")
        return {"kind": kind, "low": low, "high": high}
    raise ValueError(
        f"search.{name}: unknown kind {kind!r}; use const, choice, uniform, "
        f"loguniform, int_uniform or int_loguniform"
    )


def sample_one(dist: dict, rng: np.random.Generator):
    """Draw a single value from a parsed distribution."""
    kind = dist["kind"]
    if kind == "const":
        return dist["value"]
    if kind == "choice":
        # Index rather than choice(values): the values may be lists (hidden_dims),
        # which numpy would otherwise try to broadcast into an array.
        return dist["values"][int(rng.integers(len(dist["values"])))]
    low, high = dist["low"], dist["high"]
    if kind == "uniform":
        return float(rng.uniform(low, high))
    if kind == "loguniform":
        return float(math.exp(rng.uniform(math.log(low), math.log(high))))
    if kind == "int_uniform":
        return int(rng.integers(int(low), int(high) + 1))
    # int_loguniform
    return int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))


def propose(space: dict, n: int, rng: np.random.Generator, history: list[dict]) -> list[dict]:
    """Return the next ``n`` parameter dicts to try.

    Random search: each draw is independent, so ``history`` is unused. It is in
    the signature because this is the seam a smarter proposer plugs into -- a
    Bayesian sampler, or the LLM-reads-the-leaderboard loop the module docstring
    argues against doing per trial. Anything obeying this signature drops in.

    Duplicates are rejected against ``history`` and within the batch, since a
    space with mostly discrete axes will otherwise re-propose settings the sweep
    has already paid for.
    """
    dists = {name: parse_distribution(name, spec) for name, spec in space.items()}
    seen = {rec["trial_id"] for rec in history}
    out: list[dict] = []
    # Bounded: a small discrete space legitimately runs out of distinct settings,
    # and the sweep should say so rather than spin.
    for _ in range(n * 50):
        if len(out) >= n:
            break
        params = {name: sample_one(d, rng) for name, d in dists.items()}
        tid = params_id(params)
        if tid in seen:
            continue
        seen.add(tid)
        out.append(params)
    return out


def grid_points(space: dict) -> list[dict]:
    """Every combination of a fully discrete space, in config order."""
    dists = {name: parse_distribution(name, spec) for name, spec in space.items()}
    continuous = [
        n for n, d in dists.items() if d["kind"] not in {"const", "choice"}
    ]
    if continuous:
        raise ValueError(
            f"strategy: grid needs every axis discrete, but {continuous} are "
            f"continuous. Use strategy: random, or give those axes 'choice' values."
        )
    axes = {
        n: (d["values"] if d["kind"] == "choice" else [d["value"]])
        for n, d in dists.items()
    }
    names = list(axes)
    return [dict(zip(names, combo)) for combo in itertools.product(*axes.values())]


def params_id(params: dict) -> str:
    """A stable short id for a parameter setting, used to resume and deduplicate."""
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def varying_keys(space: dict) -> list[str]:
    """The axes that actually vary, i.e. the ones worth a leaderboard column."""
    out = []
    for name, spec in space.items():
        dist = parse_distribution(name, spec)
        if dist["kind"] == "const":
            continue
        if dist["kind"] == "choice" and len(dist["values"]) < 2:
            continue
        out.append(name)
    return out


# ---------------------------------------------------------------------------
# Trial bookkeeping
# ---------------------------------------------------------------------------


#: Load trials.jsonl, skipping any line that will not parse. Defined in
#: ``plot_sweep`` so the folder-reading half of a sweep can be used without
#: importing the training stack, and aliased here because this is where the
#: resume logic that depends on it lives.
read_trials = read_trials_file


def append_trial(path: Path, record: dict) -> None:
    """Append one finished trial and flush, so a kill -9 loses at most one trial."""
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()


def done_keys(records: list[dict]) -> set[tuple[str, int]]:
    """(trial_id, stage) pairs already on disk."""
    return {(r["trial_id"], int(r.get("stage", 1))) for r in records}


# ---------------------------------------------------------------------------
# Running one trial
# ---------------------------------------------------------------------------


def score_trial(payload: dict, objective: dict) -> tuple[float | None, dict, str]:
    """Collapse a trial's matrix to its objective value.

    Returns ``(objective, metrics, rejected_reason)``. A rejected trial keeps its
    metrics -- knowing *how* it failed is worth as much as the score -- but
    carries no objective and never enters the ranking.
    """
    metrics = summarise_embedding(payload)
    value = float(metrics[objective["metric"]])

    floor = float(objective.get("min_in_domain", 0.0))
    if metrics["in_domain_mean"] < floor:
        # Guards against a degenerate winner: a model that has collapsed to one
        # class scores ~the majority share everywhere, which on a balanced corpus
        # can beat a real classifier's *worst* transfer cell.
        return (
            None,
            metrics,
            f"in-domain {metrics['in_domain_mean']:.3f} below floor {floor:.3f}",
        )
    if not math.isfinite(value):
        return None, metrics, f"objective is not finite ({value})"
    return value, metrics, ""


def run_trial(
    params: dict,
    stage: int,
    n_runs: int,
    sweep_cfg: dict,
    cfg_dev: dict,
    X_all: dict,
    data: dict,
    names: list[str],
    labels: dict,
    features: list[str],
    out_dir: Path,
) -> dict:
    """Fit and score one hyper-parameter setting. Never raises.

    A bad combination of parameters is a data point, not a crash: it is recorded
    as rejected with its error so a resumed sweep skips it and the leaderboard
    can show how many settings the space is wasting.
    """
    trial_id = params_id(params)
    alg_cfg = sweep_cfg["algorithm"]
    spec = {
        "id": trial_id,
        "name": alg_cfg["name"],
        "label": f"trial {trial_id}",
        "params": dict(params),
        "binoculars": alg_cfg.get("binoculars") or None,
    }
    cfg_trial = dict(cfg_dev)
    cfg_trial["model"] = {**cfg_dev["model"], "n_runs": n_runs}
    cfg_trial["alg_id"] = f"{sweep_cfg['sweep_id']}_s{stage}"

    record = {
        "trial_id": trial_id,
        "stage": stage,
        "n_runs": n_runs,
        "params": params,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    started = time.time()
    trial_dir = out_dir / "trials" / f"s{stage}_{trial_id}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = run_algorithm(
            spec,
            cfg_trial,
            X_all,
            data,
            names,
            labels,
            features,
            trial_dir,
            save_models=False,
        )
        objective, metrics, rejected = score_trial(payload, sweep_cfg["objective"])
        record.update(
            objective=objective,
            rejected=rejected or None,
            metrics={
                k: metrics[k]
                for k in (
                    "in_domain_mean",
                    "in_domain_min",
                    "transfer_mean",
                    "transfer_std",
                    "transfer_min",
                    "transfer_f1_mean",
                    "gap",
                )
            },
            accuracy=metrics["accuracy"],
            dataset_names=names,
        )
    except Exception as exc:  # a bad setting must not end the sweep
        record.update(
            objective=None,
            rejected=f"{type(exc).__name__}: {exc}",
            metrics=None,
        )
    record["seconds"] = round(time.time() - started, 2)
    return record


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def rank_trials(records: list[dict], objective: dict, stage: int | None = None) -> list[dict]:
    """Scored trials, best first. Rejected trials are dropped."""
    rows = [r for r in records if r.get("objective") is not None]
    if stage is not None:
        rows = [r for r in rows if int(r.get("stage", 1)) == stage]
    return sorted(
        rows, key=lambda r: r["objective"], reverse=objective["direction"] == "max"
    )


def best_trial(records: list[dict], objective: dict) -> dict | None:
    """The winner: the best trial at the deepest stage that produced any."""
    for stage in sorted({int(r.get("stage", 1)) for r in records}, reverse=True):
        ranked = rank_trials(records, objective, stage=stage)
        if ranked:
            return ranked[0]
    return None


def fmt_value(v) -> str:
    """Compact rendering of one parameter value for the leaderboard."""
    if isinstance(v, list):
        return "[" + ",".join(str(x) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def print_leaderboard(
    records: list[dict], sweep_cfg: dict, limit: int = 25
) -> None:
    """The table you read after leaving this on a server overnight."""
    objective = sweep_cfg["objective"]
    keys = varying_keys(sweep_cfg["search"])
    stages = sorted({int(r.get("stage", 1)) for r in records})

    print()
    print("=" * RULE)
    print(f"  SWEEP {sweep_cfg['sweep_id']} — {len(records)} trials on disk")
    print("=" * RULE)

    n_rejected = sum(1 for r in records if r.get("objective") is None)
    if n_rejected:
        reasons: dict[str, int] = {}
        for r in records:
            if r.get("objective") is None:
                # Collapse "in-domain 0.61 below floor" to one bucket per kind.
                reason = str(r.get("rejected", "?")).split(":")[0].split(" below ")[0]
                key = "below in-domain floor" if "in-domain" in reason else reason
                reasons[key] = reasons.get(key, 0) + 1
        detail = ", ".join(f"{k} ({n})" for k, n in sorted(reasons.items()))
        print(f"  {n_rejected} rejected: {detail}")

    for stage in stages:
        ranked = rank_trials(records, objective, stage=stage)
        if not ranked:
            continue
        n_runs = ranked[0].get("n_runs", "?")
        print()
        print(
            f"  stage {stage} — {len(ranked)} scored, {n_runs} seed(s) each, "
            f"ranked by {objective['direction']} {objective['metric']}"
        )
        header = (
            f"  {'rank':>4}  {'trial':<11}{'obj':>8}{'in-dom':>8}{'transf':>8}"
            f"{'t-min':>8}{'gap':>7}  "
        ) + "".join(f"{k[:11]:>13}" for k in keys)
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for i, r in enumerate(ranked[:limit], 1):
            m = r["metrics"]
            row = (
                f"  {i:>4}  {r['trial_id']:<11}{r['objective']:>8.4f}"
                f"{m['in_domain_mean']:>8.3f}{m['transfer_mean']:>8.3f}"
                f"{m['transfer_min']:>8.3f}{m['gap']:>7.3f}  "
            )
            row += "".join(f"{fmt_value(r['params'].get(k, '')):>13}" for k in keys)
            print(row)
        if len(ranked) > limit:
            print(f"  … {len(ranked) - limit} more")

    best = best_trial(records, objective)
    if best:
        print()
        print(f"  BEST: {best['trial_id']}  (stage {best['stage']})")
        for k, v in sorted(best["params"].items()):
            print(f"    {k:<20} {fmt_value(v)}")
    if not sweep_cfg["holdout"]:
        print()
        print(
            f"{YELLOW}  ⚠️  holdout is empty: these scores are the ones the search "
            f"optimised,{RESET}"
        )
        print(
            f"{YELLOW}      so the winner's margin is optimistic. Treat a gain as a "
            f"hypothesis.{RESET}"
        )
    else:
        print()
        print(
            f"  holdout {sweep_cfg['holdout']} took no part in the search; "
            f"--final scores the winner on it."
        )
    print("=" * RULE)


def print_best_params(sweep_cfg: dict, records: list[dict]) -> None:
    """Print the winner as an ``algorithms`` entry, ready to paste into a config.

    The alternative is transcribing a dozen 17-digit floats out of
    ``sweep_summary.json`` by hand, and a mistyped *value* is invisible where a
    mistyped *name* is now caught (the algorithms declare which params they read,
    so the runner's preflight rejects anything else). This closes the other half:
    the block is emitted in the exact shape ``clf_cd_algs`` configs take.
    """
    best = best_trial(records, sweep_cfg["objective"])
    if best is None:
        print(f"{YELLOW}No scored trial yet. Run the search first.{RESET}")
        return

    alg_cfg = sweep_cfg["algorithm"]
    entry: dict = {
        "name": alg_cfg["name"],
        "id": f"{alg_cfg['name']}_swept",
        "label": f"{alg_cfg['name']} (swept)",
    }
    if alg_cfg.get("binoculars"):
        entry["binoculars"] = alg_cfg["binoculars"]
    entry["params"] = dict(best["params"])

    metrics = best.get("metrics") or {}
    print(
        f"# Best trial of the {sweep_cfg['sweep_id']} sweep ({best['trial_id']}, "
        f"stage {best['stage']} / {best.get('n_runs', '?')} seeds), ranked on "
        f"{sweep_cfg['objective']['direction']} {sweep_cfg['objective']['metric']}."
    )
    if metrics:
        print(
            f"# Dev-pool scores: transfer {metrics.get('transfer_mean', float('nan')):.3f}, "
            f"in-domain {metrics.get('in_domain_mean', float('nan')):.3f}, "
            f"worst transfer cell {metrics.get('transfer_min', float('nan')):.3f}."
        )
    print(
        f"# Searched on {best.get('dataset_names') or 'the dev pool'}; "
        f"holdout {sweep_cfg['holdout'] or 'NONE — the margin is optimistic'}."
    )
    print(yaml.safe_dump([entry], sort_keys=False, default_flow_style=False).rstrip())


# ---------------------------------------------------------------------------
# The final, honest run
# ---------------------------------------------------------------------------


def run_final(
    sweep_cfg: dict, base_cfg: dict, records: list[dict], out_dir: Path
) -> None:
    """Re-run the winner on the FULL matrix -- holdout included -- with baselines.

    Goes through ``run_config`` rather than ``run_algorithm`` so the winner lands
    in the usual layout with the usual figures, ranked against the Binoculars and
    LLM-judge baselines exactly as a hand-written config would be.
    """
    best = best_trial(records, sweep_cfg["objective"])
    if best is None:
        print(f"{YELLOW}No scored trial to finalise. Run the search first.{RESET}")
        return

    alg_cfg = sweep_cfg["algorithm"]
    final_cfg = dict(base_cfg)
    final_cfg["alg_id"] = f"{sweep_cfg['sweep_id']}_best"
    final_cfg["alg_label"] = (
        f"{sweep_cfg['sweep_id']} — best trial {best['trial_id']}"
    )
    entry = {
        "name": alg_cfg["name"],
        "id": f"best_{best['trial_id']}",
        "label": f"{alg_cfg['name']} (swept)",
        "params": best["params"],
    }
    if alg_cfg.get("binoculars"):
        entry["binoculars"] = alg_cfg["binoculars"]

    # The tuned model beside the config's own entries: the untuned defaults are
    # the only honest reference for what the search actually bought.
    final_cfg["algorithms"] = [entry] + list(base_cfg.get("algorithms") or [])

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = final_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(final_cfg, f, sort_keys=False)

    print()
    print("=" * RULE)
    print(f"  FINAL RUN — winner {best['trial_id']} on all "
          f"{len(final_cfg['data']['datasets'])} corpora")
    if sweep_cfg["holdout"]:
        print(f"  holdout {sweep_cfg['holdout']} is scored here for the first time.")
    print("=" * RULE)

    run_config(cfg_path, final_cfg, final_dir / final_cfg["alg_id"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def resolve_cfg_path(arg: str | None, cfg_dir: Path = DEFAULT_CFG_DIR) -> Path:
    """The sweep config to use: the one named, or the only one in ``cfg_dir``."""
    if arg:
        path = Path(arg)
        if not path.is_absolute():
            path = BASE_DIR / path
        if not path.is_file():
            print(f"{YELLOW}ERROR: sweep config not found: {path}{RESET}", file=sys.stderr)
            sys.exit(1)
        return path

    candidates = sorted(cfg_dir.glob("*.yaml")) + sorted(cfg_dir.glob("*.yml"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        print(f"{YELLOW}ERROR: no sweep configs in {cfg_dir}{RESET}", file=sys.stderr)
    else:
        print(
            f"{YELLOW}ERROR: {len(candidates)} sweep configs in {cfg_dir}; name one "
            f"with --cfg:{RESET}\n  "
            + "\n  ".join(str(p.relative_to(BASE_DIR)) for p in candidates),
            file=sys.stderr,
        )
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cfg", help="sweep config YAML (default: the only one in --cfg-dir)")
    p.add_argument("--cfg-dir", type=Path, default=DEFAULT_CFG_DIR,
                   help=f"directory searched when --cfg is omitted "
                        f"(default: {rel(DEFAULT_CFG_DIR)})")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                   help=f"where sweep directories are created "
                        f"(default: {rel(DEFAULT_RESULTS_DIR)})")
    p.add_argument("--out", help="sweep directory (default: <results-dir>/<sweep_id>)")
    p.add_argument("--n-trials", type=int, help="override the config's n_trials")
    p.add_argument(
        "--report", action="store_true", help="print the leaderboard and exit"
    )
    p.add_argument(
        "--params",
        action="store_true",
        help="print the winner as a clf_cd_algs 'algorithms' entry and exit",
    )
    p.add_argument(
        "--final",
        action="store_true",
        help="re-run the best trial on the full matrix (holdout included) and exit",
    )
    p.add_argument(
        "--plots",
        action="store_true",
        help="(re)draw the sweep figures from what is on disk and exit",
    )
    p.add_argument(
        "--limit", type=int, default=25, help="leaderboard rows per stage (default 25)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = resolve_cfg_path(args.cfg, args.cfg_dir)
    sweep_cfg = load_sweep_config(cfg_path)
    base_path, base_cfg = resolve_base_config(sweep_cfg, cfg_path)

    out_dir = (
        Path(args.out)
        if args.out
        else args.results_dir / sweep_cfg["sweep_id"]
    )
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.jsonl"
    records = read_trials(trials_path)

    if args.report:
        print_leaderboard(records, sweep_cfg, limit=args.limit)
        return
    if args.params:
        print_best_params(sweep_cfg, records)
        return
    if args.plots:
        draw_plots(out_dir, records, sweep_cfg)
        return
    if args.final:
        run_final(sweep_cfg, base_cfg, records, out_dir)
        # The holdout figure only becomes drawable once this run exists, so it is
        # the natural place to refresh the whole set.
        draw_plots(out_dir, records, sweep_cfg)
        return

    # ---- set up the search ------------------------------------------------
    dev_specs, held_specs = split_datasets(base_cfg, sweep_cfg["holdout"])
    n_trials = args.n_trials or int(sweep_cfg["n_trials"])
    halving = sweep_cfg["halving"]
    objective = sweep_cfg["objective"]

    cfg_dev = dict(base_cfg)
    cfg_dev["data"] = {**base_cfg["data"], "datasets": dev_specs}
    # load_columns reads `features` plus whatever auxiliary column the algorithms
    # ask for, so give it the sweep's algorithm entry rather than the base's list.
    alg_entry = {"name": sweep_cfg["algorithm"]["name"]}
    if sweep_cfg["algorithm"].get("binoculars"):
        alg_entry["binoculars"] = sweep_cfg["algorithm"]["binoculars"]
    cfg_dev["algorithms"] = [alg_entry]

    features = list(base_cfg["features"])
    columns = load_columns(cfg_dev)
    group_col = base_cfg["data"]["group_col"]
    human_label = base_cfg["data"]["human_label"]
    dev_corpora = dataset_specs(cfg_dev, BASE_DIR)
    names = [s.name for s in dev_corpora]
    labels = {s.name: s.label for s in dev_corpora}

    print("=" * RULE)
    print(f"  SWEEP      : {sweep_cfg['sweep_id']}  ({cfg_path.relative_to(BASE_DIR)})")
    print(f"  BASE       : {base_path.relative_to(BASE_DIR)}")
    print(f"  ALGORITHM  : {sweep_cfg['algorithm']['name']}")
    print(f"  DEV POOL   : {names}")
    if held_specs:
        print(f"  HOLDOUT    : {[s['name'] for s in held_specs]}  (not touched here)")
    else:
        print(
            f"{YELLOW}  HOLDOUT    : none — the winning score will be optimistic, "
            f"see the module docstring{RESET}"
        )
    print(f"  OBJECTIVE  : {objective['direction']} {objective['metric']}"
          f"   (in-domain floor {objective['min_in_domain']})")
    print(f"  SEARCH     : {sweep_cfg['strategy']}, {n_trials} trials over "
          f"{len(sweep_cfg['search'])} axes")
    if halving["enabled"]:
        print(f"  HALVING    : {halving['stage1_n_runs']} seed(s) → top "
              f"{halving['keep_frac']:.0%} at {halving['stage2_n_runs']} seeds")
    print(f"  OUTPUT     : {out_dir}")
    if records:
        print(f"  RESUMING   : {len(records)} trial(s) already on disk")
    print("=" * RULE)

    # ---- preflight --------------------------------------------------------
    # The runner's own checks, on the dev config the search will actually use, so
    # a missing feature column or a single-class corpus fails in the first minute
    # rather than after a night of trials. It also validates the sweep's algorithm
    # entry -- notably that a hybrid was given its auxiliary column. Costs one
    # extra read of the corpora, which is worth it at this stake.
    report = Report()
    preflight_config(cfg_dev, report)
    print_report(report)
    if not report.ok:
        sys.exit(1)

    # ---- load the corpora once -------------------------------------------
    data = {}
    for spec in dev_corpora:
        prep = load_prepared(spec, columns, group_col, human_label)
        print(f"  {spec.name:<24} {len(prep.df):>6} rows  {prep.class_counts}")
        data[spec.name] = prep.df
    X_all = {name: stack_features(df, features)[0] for name, df in data.items()}

    with open(out_dir / "sweep_config.yaml", "w") as f:
        yaml.safe_dump(sweep_cfg, f, sort_keys=False)

    # ---- stage 1 ----------------------------------------------------------
    rng = np.random.default_rng(sweep_cfg["seed"])
    # n_trials is a budget for the whole sweep, not for one invocation, so a
    # resumed run must ask only for what is left. propose() already rejects
    # everything in `records`, so handing it the full n_trials again returns
    # n_trials settings that are all new -- stacking another full budget on top
    # of what is on disk instead of finishing the original one.
    n_done1 = sum(1 for r in records if int(r.get("stage", 1)) == 1)
    if sweep_cfg["strategy"] == "grid":
        candidates = grid_points(sweep_cfg["search"])
        budget = len(candidates)
        print(f"\n  grid has {budget} points; running all of them")
    else:
        budget = n_trials
        remaining = max(0, n_trials - n_done1)
        candidates = propose(sweep_cfg["search"], remaining, rng, records)
        if len(candidates) < remaining:
            print(
                f"{YELLOW}  space exhausted: {len(candidates)} distinct settings "
                f"left, not {remaining}{RESET}"
            )

    stage1_runs = (
        int(halving["stage1_n_runs"]) if halving["enabled"] else int(base_cfg["model"]["n_runs"])
    )
    done = done_keys(records)
    # Redundant for the random strategy (propose already deduplicated) but load
    # bearing for grid, whose points are enumerated without consulting history.
    todo = [p for p in candidates if (params_id(p), 1) not in done]
    print(f"\n  stage 1: {len(todo)} trial(s) to run "
          f"({n_done1} of {budget} already done)\n")

    for params in tqdm(todo, desc="Stage 1", unit="trial"):
        record = run_trial(
            params, 1, stage1_runs, sweep_cfg, cfg_dev, X_all, data,
            names, labels, features, out_dir,
        )
        append_trial(trials_path, record)
        records.append(record)
        if record["objective"] is None:
            tqdm.write(f"  {DIM}{record['trial_id']}  rejected: {record['rejected']}{RESET}")
        else:
            tqdm.write(
                f"  {record['trial_id']}  {objective['metric']}="
                f"{record['objective']:.4f}  "
                f"in-dom={record['metrics']['in_domain_mean']:.3f}  "
                f"({record['seconds']:.0f}s)"
            )

    # ---- stage 2: re-run the survivors on more seeds -----------------------
    if halving["enabled"]:
        ranked = rank_trials(records, objective, stage=1)
        keep = max(1, int(round(len(ranked) * float(halving["keep_frac"]))))
        survivors = ranked[:keep]
        stage2_runs = int(halving["stage2_n_runs"])
        done2 = done_keys(records)
        todo2 = [r["params"] for r in survivors if (r["trial_id"], 2) not in done2]
        print(
            f"\n  stage 2: top {keep} of {len(ranked)} re-run at {stage2_runs} "
            f"seeds — {len(todo2)} to do\n"
        )
        for params in tqdm(todo2, desc="Stage 2", unit="trial"):
            record = run_trial(
                params, 2, stage2_runs, sweep_cfg, cfg_dev, X_all, data,
                names, labels, features, out_dir,
            )
            append_trial(trials_path, record)
            records.append(record)
            if record["objective"] is not None:
                tqdm.write(
                    f"  {record['trial_id']}  {objective['metric']}="
                    f"{record['objective']:.4f}  ({record['seconds']:.0f}s)"
                )

    print_leaderboard(records, sweep_cfg, limit=args.limit)
    draw_plots(out_dir, records, sweep_cfg)

    best = best_trial(records, objective)
    if best:
        with open(out_dir / "sweep_summary.json", "w") as f:
            json.dump(
                {
                    "sweep_id": sweep_cfg["sweep_id"],
                    "base": sweep_cfg["base"],
                    "objective": objective,
                    "holdout": sweep_cfg["holdout"],
                    "dev_pool": names,
                    "n_trials": len(records),
                    "best": best,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\nDone. {len(records)} trials in {trials_path}")
        print(
            f"Next: python -m epai.ai_detection.train.sweep_clf_cross_domain "
            f"--cfg {cfg_path.relative_to(BASE_DIR)} --final"
        )


if __name__ == "__main__":
    main()
