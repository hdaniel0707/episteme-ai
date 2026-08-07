"""Report and figures for a finished (or running) hyper-parameter sweep folder.

``run_sweep_cross_domain.py`` draws its figures as a side effect of searching,
which means getting them back later costs you the sweep config, the base config
it points at, and an import of the whole training stack -- torch included -- to
render four PNGs. This script is the other direction: point it at a results
folder and it reads what is in there.

That works because the folder is self-describing. The search writes
``sweep_config.yaml`` beside ``trials.jsonl``, so the objective, the search space
and the holdout all travel with the results; the ``--final`` run leaves its
``algorithm_comparison.json`` under ``final/``. Nothing here needs the base
config, the corpora, or a GPU, so a sweep that ran on a server can be reported on
a laptop by copying one directory.

It is also where the folder-reading helpers live, and the sweep script imports
them from here rather than the reverse -- this module pulls in matplotlib and
nothing heavier, so depending on it costs the sweep nothing.

Run (``--no-sync`` because ``uv run`` otherwise syncs the environment to the lock
file, which a reporting script has no business doing):
    # one folder, by path or by sweep id -- either resolves from any directory
    uv run python -m epai.ai_detection.train.replot_sweep results/sweep_clf_cd/luar_mlp_reg
    uv run python -m epai.ai_detection.train.replot_sweep luar_mlp_reg

    # several at once, and the per-trial table as well as the digest
    uv run python -m epai.ai_detection.train.replot_sweep luar_mlp_reg other --leaderboard

    # what is available
    uv run python -m epai.ai_detection.train.replot_sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from epai.utils.config_utils import RESET, YELLOW
from epai.utils.visualization_utils import plot_sweep_results, sweep_axis_effects

BASE_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "sweep_clf_cd"

RULE = 100
GREEN = "\033[92m"
DIM = "\033[2m"

#: Axes listed in the digest. The tail of a ten-axis search is noise at this
#: sample size, and printing all of it buries the two or three that moved.
TOP_AXES = 6


# ---------------------------------------------------------------------------
# Reading a sweep folder
# ---------------------------------------------------------------------------


def read_trials_file(path: Path) -> list[dict]:
    """Load trials.jsonl, skipping any line that will not parse.

    A sweep killed mid-write leaves a truncated last line, and a sweep killed
    twice can leave a partial record with a whole one appended after it on the
    same line. Both are dropped: the trial they belonged to did not finish, so it
    will be proposed again, and a resumed sweep must not choke on its own crash.
    """
    if not path.is_file():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_final_comparison(out_dir: Path) -> dict | None:
    """The ``--final`` run's algorithm comparison, if that run has happened.

    Globbed rather than built from ``alg_id``: the winner's directory is named
    after the sweep, but a sweep re-finalised under a changed id would leave two,
    and the newest is the one the figures should describe.
    """
    candidates = sorted(
        (out_dir / "final").glob("*/algorithm_comparison.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        return None
    with open(candidates[-1]) as f:
        return json.load(f)


def draw_plots(out_dir: Path, records: list[dict], sweep_cfg: dict) -> list[Path]:
    """Write the sweep figures, reporting what was and was not drawable.

    Never raises: this also runs at the end of a search that may have taken all
    night, and a figure that cannot be drawn must not lose the trials that
    produced it.
    """
    comparison = load_final_comparison(out_dir)
    try:
        written = plot_sweep_results(out_dir, records, sweep_cfg, comparison)
    except Exception as exc:
        print(f"{YELLOW}Could not draw the sweep figures: "
              f"{type(exc).__name__}: {exc}{RESET}")
        return []

    if not written:
        print(f"{YELLOW}No sweep figures drawn — nothing scored yet?{RESET}")
        return []
    print(f"\n  figures ({len(written)}):")
    for path in written:
        rel = path.relative_to(BASE_DIR) if BASE_DIR in path.parents else path
        print(f"    {rel}")
    if comparison is None:
        print(
            f"  {YELLOW}sweep_holdout.png needs the final run; "
            f"re-run the sweep with --final to get it.{RESET}"
        )
    return written


def available_sweeps() -> list[Path]:
    """Every directory under ``results/sweep_clf_cd`` that holds trials."""
    if not RESULTS_DIR.is_dir():
        return []
    return sorted(p.parent for p in RESULTS_DIR.glob("*/trials.jsonl"))


def resolve_sweep_dir(arg: str) -> Path:
    """A sweep folder from a path or a bare sweep id.

    Accepting the id is not just convenience: the id is what the config calls the
    run and what the leaderboard prints, so it is the name you have in hand when
    you come back to a sweep days later.
    """
    for candidate in (Path(arg), BASE_DIR / arg, RESULTS_DIR / arg):
        if candidate.is_dir():
            return candidate
    known = available_sweeps()
    hint = (
        "\n  ".join(p.name for p in known) if known
        else f"(none under {RESULTS_DIR.relative_to(BASE_DIR)})"
    )
    print(
        f"{YELLOW}ERROR: no sweep folder at {arg!r}. Known sweeps:{RESET}\n  {hint}",
        file=sys.stderr,
    )
    sys.exit(1)


def load_sweep_config_from_dir(out_dir: Path, cfg_override: Path | None) -> dict:
    """The sweep config that produced this folder.

    Normally the copy the search dropped in the folder, which is the one that
    actually ran -- reading the config in ``cfg/`` instead would describe whatever
    it has been edited into since. ``cfg_override`` covers the folder that predates
    the copy, or one assembled by hand.
    """
    if cfg_override is not None:
        with open(cfg_override) as f:
            return yaml.safe_load(f)

    path = out_dir / "sweep_config.yaml"
    if not path.is_file():
        print(
            f"{YELLOW}ERROR: {path} not found, so the objective, search space and "
            f"holdout are unknown.{RESET}\n"
            f"Pass --cfg <sweep config> to supply them.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        print(f"{YELLOW}ERROR: {path} is not a YAML mapping{RESET}", file=sys.stderr)
        sys.exit(1)
    return cfg


# ---------------------------------------------------------------------------
# The digest
# ---------------------------------------------------------------------------


def fmt_value(v) -> str:
    """Compact rendering of one parameter value. Matches the leaderboard's format."""
    if isinstance(v, list):
        return "[" + ",".join(str(x) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def summarise_sweep(out_dir: Path, records: list[dict], sweep_cfg: dict) -> None:
    """The numbers behind the figures, for a terminal.

    Deliberately not the per-trial leaderboard (``--leaderboard`` prints that):
    this is the reading of the sweep as a whole -- how much was searched, what
    moved the objective, and how much of the margin survives a second look.
    """
    scored = [r for r in records if r.get("objective") is not None and r.get("metrics")]
    stages = sorted({int(r.get("stage", 1)) for r in records})
    objective = sweep_cfg.get("objective", {}) or {}
    metric = objective.get("metric", "objective")
    maximise = objective.get("direction", "max") == "max"
    holdout = sweep_cfg.get("holdout") or []

    print("=" * RULE)
    print(f"  SWEEP    : {sweep_cfg.get('sweep_id', out_dir.name)}")
    print(f"  FOLDER   : {out_dir.relative_to(BASE_DIR) if BASE_DIR in out_dir.parents else out_dir}")
    print(f"  OBJECTIVE: {objective.get('direction', 'max')} {metric}"
          f"   (in-domain floor {objective.get('min_in_domain', 0.0)})")
    if scored:
        print(f"  DEV POOL : {scored[0].get('dataset_names')}")
    print(f"  HOLDOUT  : {holdout if holdout else 'none'}")
    print("=" * RULE)

    counts = {s: sum(1 for r in records if int(r.get("stage", 1)) == s) for s in stages}
    print(f"\n  {len(records)} trial(s): "
          + ", ".join(f"stage {s} × {n}" for s, n in counts.items()))
    rejected = [r for r in records if r.get("objective") is None]
    if rejected:
        reasons: dict[str, int] = {}
        for r in rejected:
            reason = str(r.get("rejected", "?")).split(":")[0].split(" below ")[0]
            key = "below in-domain floor" if "in-domain" in reason else reason
            reasons[key] = reasons.get(key, 0) + 1
        print(f"  {len(rejected)} rejected: "
              + ", ".join(f"{k} ({n})" for k, n in sorted(reasons.items())))
    if not scored:
        print(f"\n{YELLOW}  Nothing scored yet — no figures and no digest.{RESET}")
        return

    # --- the winner, at the deepest stage that produced one -----------------
    best = None
    for stage in sorted({int(r.get("stage", 1)) for r in scored}, reverse=True):
        ranked = [r for r in scored if int(r.get("stage", 1)) == stage]
        if ranked:
            best = sorted(ranked, key=lambda r: r["objective"], reverse=maximise)[0]
            break
    if best:
        m = best["metrics"]
        print(f"\n  BEST  {best['trial_id']}  (stage {best['stage']}, "
              f"{best.get('n_runs', '?')} seed(s))")
        print(f"    {metric:<18} {best['objective']:.4f}")
        print(f"    {'in-domain':<18} {m['in_domain_mean']:.4f}")
        print(f"    {'transfer (worst)':<18} {m['transfer_min']:.4f}")
        print(f"    {'gap':<18} {m['gap']:.4f}")
        for k, v in sorted(best["params"].items()):
            print(f"      {k:<20} {fmt_value(v)}")

    # --- what moved the objective ------------------------------------------
    effects = sweep_axis_effects(records)
    if effects:
        print(f"\n  AXES BY MARGINAL EFFECT on {metric} "
              f"(spread of group means, stage 1):")
        for key, effect in effects[:TOP_AXES]:
            bar = "█" * max(1, int(round(effect * 100 * 2)))
            print(f"    {key:<20} {effect*100:>5.2f} pts  {DIM}{bar}{RESET}")
        if len(effects) > TOP_AXES:
            tail = ", ".join(k for k, _ in effects[TOP_AXES:])
            print(f"    {DIM}below {effects[TOP_AXES][1]*100:.2f} pts: {tail}{RESET}")
        print(f"    {DIM}marginals over a random sample: suggestive, "
              f"not an importance ranking{RESET}")

    # --- how much of the ranking is noise ----------------------------------
    s1 = {r["trial_id"]: r for r in scored if int(r.get("stage", 1)) == 1}
    s2 = {r["trial_id"]: r for r in scored if int(r.get("stage", 1)) == 2}
    shared = [tid for tid in s1 if tid in s2]
    if shared:
        a = np.array([s1[t]["objective"] for t in shared])
        b = np.array([s2[t]["objective"] for t in shared])
        fell = int((b < a).sum())
        print(f"\n  SEED NOISE: re-running {len(shared)} survivor(s) on more seeds "
              f"moved {metric} by {np.mean(b - a):+.4f} on average; "
              f"{fell} of {len(shared)} fell.")
        print(f"    {DIM}that is the part of a stage-1 margin that was one "
              f"seed's luck{RESET}")
    elif 2 in counts:
        print(f"\n  {YELLOW}stage 2 trials exist but none share an id with stage 1 "
              f"— cannot measure seed noise.{RESET}")
    else:
        print(f"\n  {YELLOW}No stage-2 trials yet, so the seed noise in the "
              f"ranking is unmeasured.{RESET}")

    # --- the two caveats a reader has to be handed --------------------------
    in_dom = np.array([r["metrics"]["in_domain_mean"] for r in scored])
    floor = float(objective.get("min_in_domain", 0.0) or 0.0)
    if floor > 0 and in_dom.min() > floor:
        print(f"\n  {YELLOW}The in-domain floor ({floor:g}) never bound: the lowest "
              f"in-domain of any trial was {in_dom.min():.3f}. It is not currently "
              f"guarding against a collapsed model.{RESET}")
    if not holdout:
        print(f"\n{YELLOW}  ⚠️  holdout is empty: these scores are the ones the "
              f"search optimised, so the winner's margin is optimistic. Treat a "
              f"gain as a hypothesis, not a result.{RESET}")
    else:
        n_dev = len(scored[0].get("dataset_names") or [])
        n_cells = max(0, n_dev * (n_dev - 1))
        print(f"\n  Held out {holdout} — no part in the search. The objective was "
              f"the mean of {n_cells} transfer cell(s) over {n_dev} dev corpora;")
        print(f"  sweep_holdout.png is where that ranking meets a corpus it "
              f"never saw.")
    print("=" * RULE)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "sweep", nargs="*",
        help="sweep folder(s), by path or by sweep id; omit to list what is available",
    )
    p.add_argument(
        "--cfg",
        help="sweep config to use instead of the folder's own sweep_config.yaml",
    )
    p.add_argument(
        "--no-figures", action="store_true", help="print the digest only",
    )
    p.add_argument(
        "--leaderboard", action="store_true",
        help="also print the per-trial table (imports the training stack)",
    )
    p.add_argument(
        "--limit", type=int, default=25, help="leaderboard rows per stage (default 25)",
    )
    return p.parse_args()


def report_sweep(out_dir: Path, args: argparse.Namespace) -> bool:
    """Digest, figures and optional leaderboard for one folder. False if unusable."""
    cfg_override = None
    if args.cfg:
        cfg_override = Path(args.cfg)
        if not cfg_override.is_absolute():
            cfg_override = BASE_DIR / cfg_override
        if not cfg_override.is_file():
            print(f"{YELLOW}ERROR: sweep config not found: {cfg_override}{RESET}",
                  file=sys.stderr)
            sys.exit(1)

    sweep_cfg = load_sweep_config_from_dir(out_dir, cfg_override)
    records = read_trials_file(out_dir / "trials.jsonl")
    if not records:
        print(f"{YELLOW}No trials in {out_dir / 'trials.jsonl'} — nothing to "
              f"report.{RESET}")
        return False

    summarise_sweep(out_dir, records, sweep_cfg)
    if not args.no_figures:
        draw_plots(out_dir, records, sweep_cfg)
    if args.leaderboard:
        # Imported here, not at module scope: it lives in the sweep script, which
        # pulls in the algorithms and therefore torch. A folder report should not
        # need a training stack unless it is asked for this table.
        from epai.ai_detection.train.run_sweep_cross_domain import print_leaderboard

        print_leaderboard(records, sweep_cfg, limit=args.limit)
    return True


def main() -> None:
    args = parse_args()
    if not args.sweep:
        known = available_sweeps()
        if not known:
            print(f"No sweeps under {RESULTS_DIR}")
            return
        print(f"Sweeps under {RESULTS_DIR.relative_to(BASE_DIR)}:")
        for path in known:
            n = len(read_trials_file(path / "trials.jsonl"))
            final = "final ✓" if load_final_comparison(path) else "no final run"
            print(f"  {path.name:<28} {n:>5} trial(s)  {final}")
        print("\nName one to report on it.")
        return

    ok = 0
    for arg in args.sweep:
        out_dir = resolve_sweep_dir(arg)
        if len(args.sweep) > 1:
            print(f"\n{'#' * RULE}\n# {out_dir.name}\n{'#' * RULE}")
        ok += bool(report_sweep(out_dir, args))
    if len(args.sweep) > 1:
        print(f"\n{GREEN}Reported on {ok} of {len(args.sweep)} sweep(s).{RESET}")


if __name__ == "__main__":
    main()
