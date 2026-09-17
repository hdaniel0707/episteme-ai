r"""Shared machinery for the four ``calibrate_*_baseline.py`` scripts.

Each of those scripts calibrates a threshold per score column of one detector
family (Binoculars / Fast-DetectGPT / pretrained detector / LLM judge) and
writes it as a ``*_baseline`` block that
``epai.utils.cd_core.baseline_entries()`` reads.

They differ only in four things -- a column prefix, the machine-is-high
orientation, the YAML block key, and how a column name becomes a short label
-- so everything else lives here.

TWO operating points per (column, corpus):

* ``threshold_fpr``    -- the fixed 1%-FPR cut. The deployable point; written
  to the YAML as ``threshold`` (what cd_core reads).
* ``threshold_youden`` -- Youden's J (max TPR - FPR), the balance-invariant
  point a trained variant is calibrated against. Written to the YAML as
  ``threshold_youden`` (recorded, not yet consumed by cd_core) and reported
  in the CSV and on the plots.

Every run writes ``<...>_thresholds.csv`` with one row per (column, corpus).
The plots are drawn FROM that CSV -- the threshold lines and the panel stats
are read back off disk -- so a PNG can never disagree with the table.

Two modes, both through :func:`run`:

* single dataset (the historical behaviour): one positional parquet, pooled
  over every domain -> table + CSV + YAML.
* ``--corpora`` : calibrate on each named corpus AND on all of them pooled
  ('all'), print a per-corpus table, write the CSV, drop one distribution PNG
  per column, and write the YAML from the pooled 'all' row.

``--corpora all`` (and a bare run) uses the four core splits, in the fixed
left-to-right order used everywhere: ``gb_orig`` (GB-orig), ``mg_gb`` (MG-GB),
``mg_sci`` (MG-Sci), ``mg_aisci`` (MG-AI-Sci); pooled is ``all`` (MG-All), built
from ``splits.POOL_SPLITS`` (currently all four -- see ``splits.py``), always
last. The MG-Reph family (``mg_reph`` / ``mg_reph_sci`` / ``mg_reph_gb``) and
GB-orig-stand are registered but opt-in only -- pass either by name
(``--corpora mg_reph_sci`` / ``gb_orig_stand`` …).

``--generators MODEL ...`` restricts the machine class to rows whose
``generated_by`` model (the part before ``:``) is named -- e.g.
``--generators gpt56luna`` calibrates human vs. gpt-5.6-luna only, the
single-generator comparison the rephrase experiments make. Such a run writes
everything (CSV, PNGs, YAML) under
``results/trainv2/baseline_calibration_[flag=<models>]/`` and never touches the
shared ``cfg/ai_detection/trainv2/*_baseline_calibrated.yaml``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from epai.utils.cd_core import (
    auroc_score,
    oriented_scores,
    rates_at_threshold,
    roc_point_at_fpr,
    youden_threshold,
)

from .splits import POOL_SPLITS, POOLED, SPLITS
from .splits import display as _display
from .splits import resolve as _resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT_DIR = BASE_DIR / "cfg" / "ai_detection" / "trainv2"
DEFAULT_FIG_DIR = BASE_DIR / "results" / "trainv2" / "baseline_calibration"

GEN_COL = "generated_by"

#: CSV column order. Scalars only -- the raw score arrays never go here.
CSV_FIELDS = [
    "column", "label", "corpus", "n", "n_human", "n_machine", "auroc",
    "threshold_fpr", "tpr_at_fpr", "fpr_at_fpr",
    "threshold_youden", "tpr_at_youden", "fpr_at_youden", "youden_j",
    "human_p1", "human_p50", "human_p99",
    "machine_p1", "machine_p50", "machine_p99",
    "machine_is_high", "fpr_target",
]

_HUMAN_COLOR = "#1c6fb3"  # human = blue
_MACHINE_COLOR = "#d92d20"  # AI = red
_FPR_COLOR = "#7048e8"  # threshold: fixed-FPR cut = violet
_YOUDEN_COLOR = "#0ca678"  # threshold: Youden's J cut = teal


def resolve_corpora(names: list[str]) -> list[tuple[str, Path]]:
    """``["all"]`` or a subset of MANUGRAM split keys -> ``[(key, path), ...]``."""
    return _resolve_splits(names, PARQUET_DIR)


def _row(
    column: str,
    label: str,
    corpus: str,
    raw: np.ndarray,
    y: np.ndarray,
    machine_is_high: bool,
    fpr_target: float,
) -> dict | None:
    """One column on one set of rows: AUROC, BOTH operating points (the fixed
    ``fpr_target`` cut and Youden's J), the rates each realises, and the human
    / machine score percentiles that say where they sit. ``None`` when there is
    no ROC to fit (single class or n < 10)."""
    if np.unique(y).size < 2 or len(y) < 10:
        return None
    oriented = oriented_scores(raw, machine_is_high)

    tpr_f, thr_or_f = roc_point_at_fpr(y, oriented, fpr_target)
    _, fpr_f = rates_at_threshold(y, oriented, thr_or_f)

    thr_or_j = youden_threshold(y, oriented)
    tpr_j, fpr_j = rates_at_threshold(y, oriented, thr_or_j)

    # oriented = raw when machine_is_high, else -raw: the raw-unit threshold a
    # caller applies (raw >= t / raw <= t) is thr_or resp. -thr_or.
    def raw_thr(thr_or: float) -> float:
        return float(thr_or) if machine_is_high else -float(thr_or)

    h, m = raw[y == 0], raw[y == 1]
    return {
        "column": column,
        "label": label,
        "corpus": corpus,
        "n": int(len(y)),
        "n_human": int((y == 0).sum()),
        "n_machine": int((y == 1).sum()),
        "auroc": float(auroc_score(y, oriented)),
        "threshold_fpr": raw_thr(thr_or_f),
        "tpr_at_fpr": float(tpr_f),
        "fpr_at_fpr": float(fpr_f),
        "threshold_youden": raw_thr(thr_or_j),
        "tpr_at_youden": float(tpr_j),
        "fpr_at_youden": float(fpr_j),
        "youden_j": float(tpr_j - fpr_j),
        "human_p1": float(np.percentile(h, 1)),
        "human_p50": float(np.median(h)),
        "human_p99": float(np.percentile(h, 99)),
        "machine_p1": float(np.percentile(m, 1)),
        "machine_p50": float(np.median(m)),
        "machine_p99": float(np.percentile(m, 99)),
        "machine_is_high": bool(machine_is_high),
        "fpr_target": float(fpr_target),
    }


def _fmt(x: float, spec: str = ".3g") -> str:
    return "--" if not np.isfinite(x) else format(x, spec)


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(path, index=False)
    print(f"wrote {path}")


def _plot_column(
    column: str,
    csv_rows: list[dict],
    raw_by_corpus: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
    bins: int,
    xlabel: str,
) -> None:
    """One PNG per score column: a panel per corpus plus 'all' pooled, each with
    the human vs. machine score histogram and BOTH thresholds marked. Threshold
    values and stats come from ``csv_rows`` (read back from the CSV), the
    histogram shape from ``raw_by_corpus``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = [r for r in csv_rows if r["corpus"] != POOLED] + [
        r for r in csv_rows if r["corpus"] == POOLED
    ]
    n = len(ordered)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    axes = axes[0]

    all_raw = np.concatenate([raw_by_corpus[r["corpus"]][0] for r in ordered])
    lo, hi = np.percentile(all_raw, [0.5, 99.5])
    thrs = [
        r[k] for r in ordered for k in ("threshold_fpr", "threshold_youden")
        if np.isfinite(r[k])
    ]
    if thrs:
        lo, hi = min(lo, *thrs), max(hi, *thrs)
    pad = 0.03 * (hi - lo) or 0.01
    lo, hi = lo - pad, hi + pad
    edges = np.linspace(lo, hi, bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2

    for ax, r in zip(axes, ordered):
        raw, y = raw_by_corpus[r["corpus"]]
        h = np.histogram(raw[y == 0], bins=edges, density=True)[0]
        m = np.histogram(raw[y == 1], bins=edges, density=True)[0]
        ax.step(mids, h, where="mid", color=_HUMAN_COLOR, lw=1.6, label="human")
        ax.fill_between(mids, h, step="mid", color=_HUMAN_COLOR, alpha=0.15)
        ax.step(mids, m, where="mid", color=_MACHINE_COLOR, lw=1.6, label="AI")
        ax.fill_between(mids, m, step="mid", color=_MACHINE_COLOR, alpha=0.15)

        ytop = ax.get_ylim()[1]
        for key, color, ls, lbl, ytxt in (
            ("threshold_fpr", _FPR_COLOR, "--", f"{r['fpr_target'] * 100:g}% FPR", 0.98),
            ("threshold_youden", _YOUDEN_COLOR, "-.", "Youden J", 0.90),
        ):
            t = r[key]
            if not np.isfinite(t):
                continue
            ax.axvline(t, color=color, lw=1.6, ls=ls, label=lbl)
            ax.text(
                t, ytop * ytxt, f" {t:.3g}", color=color,
                fontsize=8, va="top",
                ha="left" if t < lo + 0.85 * (hi - lo) else "right",
            )
        ax.set_title(
            f"{_display(r['corpus'])}  (n={r['n']})\n"
            f"AUROC {r['auroc']:.3f}   "
            f"TPR@fpr {r['tpr_at_fpr']:.2f}  |  J {r['youden_j']:.2f}",
            fontsize=9,
        )
        ax.set_xlabel(xlabel)
        ax.set_xlim(lo, hi)
        ax.grid(True, lw=0.4, alpha=0.4)
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def _filter_generators(
    df: pd.DataFrame, human_label: str, generators: list[str] | None
) -> pd.DataFrame:
    """Keep the human rows and, of the machine rows, only those whose
    ``generated_by`` model (the part before ``:``) is in ``generators``.
    ``None`` / empty -> unchanged."""
    if not generators:
        return df
    g = df[GEN_COL].astype(str)
    keep = g == human_label
    for spec in generators:
        keep |= (g == spec) | g.str.startswith(f"{spec}:")
    return df[keep]


def _load(
    path: Path, prefix: str, cols_arg: list[str] | None, exclude: frozenset[str],
    human_label: str, generators: list[str] | None,
) -> pd.DataFrame:
    """Read ``generated_by`` and the score columns only -- never the embedding
    columns, which are gigabytes and unused here. ``generators`` restricts the
    machine class (see :func:`_filter_generators`)."""
    import pyarrow.parquet as pq

    names = pq.read_schema(path).names
    if GEN_COL not in names:
        raise SystemExit(f"{path}: no {GEN_COL!r} column")
    if cols_arg:
        want = set(cols_arg)
    else:
        want = {c for c in names if c.startswith(prefix)} - exclude
    need = [GEN_COL] + sorted(want & set(names))
    return _filter_generators(pd.read_parquet(path, columns=need),
                              human_label, generators)


def _yaml_entries(
    rows_by_col: dict[str, dict], label_fn: Callable[[str], str]
) -> tuple[list[dict], list[str]]:
    """One YAML column entry per column whose 1%-FPR cut is finite. Carries both
    thresholds. Returns ``(entries, dropped_columns)``."""
    entries, dropped = [], []
    for col, r in rows_by_col.items():
        if not np.isfinite(r["threshold_fpr"]):
            dropped.append(col)
            continue
        entry = {"column": col, "threshold": round(float(r["threshold_fpr"]), 6)}
        if np.isfinite(r["threshold_youden"]):
            entry["threshold_youden"] = round(float(r["threshold_youden"]), 6)
        entry["label"] = label_fn(col)
        entries.append(entry)
    return entries, dropped


def _write_yaml(
    entries: list[dict],
    block_key: str,
    out: Path,
    script: str,
    source_desc: str,
    fpr_target: float,
) -> None:
    cfg = {block_key: {"columns": entries}}
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Calibrated by {script} on {source_desc}\n"
        f"# 'threshold' = {fpr_target:g}-FPR cut, pooled over every domain ('all').\n"
        f"# 'threshold_youden' = Youden's J cut (recorded; cd_core reads 'threshold').\n"
        f"# Read by cd_core.baseline_entries().\n"
    )
    out.write_text(header + yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out}")


def _table_header() -> str:
    return (
        f"{'corpus':<18}{'n_hum':>7}{'n_mac':>7}{'auroc':>8}"
        f"{'thr@fpr':>11}{'tpr':>6}{'fpr':>7}"
        f"{'thr_J':>11}{'tpr':>6}{'fpr':>6}{'J':>6}"
    )


def _table_row(r: dict) -> str:
    return (
        f"{r['corpus']:<18}{r['n_human']:>7}{r['n_machine']:>7}{r['auroc']:>8.3f}"
        f"{_fmt(r['threshold_fpr'], '.4g'):>11}{r['tpr_at_fpr']:>6.2f}"
        f"{r['fpr_at_fpr']:>7.3f}"
        f"{_fmt(r['threshold_youden'], '.4g'):>11}{r['tpr_at_youden']:>6.2f}"
        f"{r['fpr_at_youden']:>6.2f}{r['youden_j']:>6.2f}"
    )


def _single(
    dataset: Path,
    *,
    prefix: str,
    machine_is_high: bool,
    block_key: str,
    label_fn: Callable[[str], str],
    cols_arg: list[str] | None,
    exclude: frozenset[str],
    fpr_target: float,
    human_label: str,
    generators: list[str] | None,
    out: Path,
    csv_path: Path,
    script: str,
) -> None:
    """Historical mode: one parquet, pooled over domains, one row per column."""
    df = _load(dataset, prefix, cols_arg, exclude, human_label, generators)
    cols = cols_arg or sorted(c for c in df.columns if c.startswith(prefix))
    if not cols:
        raise SystemExit(f"no {prefix}* column in {dataset}")

    corpus = dataset.stem
    if generators:
        print(f"machine class restricted to: {', '.join(generators)}")
    print(f"calibrating on {dataset}  ({len(df)} rows, target FPR {fpr_target:g})\n")
    print(_table_header())

    rows: list[dict] = []
    rows_by_col: dict[str, dict] = {}
    for col in cols:
        sub = df[[col, GEN_COL]].dropna(subset=[col])
        raw = sub[col].to_numpy(dtype=float)
        y = (sub[GEN_COL].astype(str) != human_label).to_numpy().astype(int)
        r = _row(col, label_fn(col), corpus, raw, y, machine_is_high, fpr_target)
        if r is None:
            print(f"{col}  -- skipped: single class or n<10 --")
            continue
        print(f"{col}\n{_table_row(r)}")
        rows.append(r)
        rows_by_col[col] = r

    if not rows:
        raise SystemExit("\nnothing calibrated -- every column was skipped")

    _write_csv(rows, csv_path)
    entries, dropped = _yaml_entries(rows_by_col, label_fn)
    if dropped:
        print(f"not written ({fpr_target:g}-FPR unreachable): {dropped}")
    if not entries:
        raise SystemExit("nothing to write -- no finite threshold")
    src_desc = dataset.name
    if generators:
        src_desc += f" [machine: {'+'.join(generators)}]"
    _write_yaml(entries, block_key, out, script, src_desc, fpr_target)


def _multi(
    corpora: list[tuple[str, Path]],
    *,
    prefix: str,
    machine_is_high: bool,
    block_key: str,
    label_fn: Callable[[str], str],
    cols_arg: list[str] | None,
    exclude: frozenset[str],
    fpr_target: float,
    human_label: str,
    generators: list[str] | None,
    out: Path,
    csv_path: Path,
    fig_dir: Path,
    bins: int,
    no_plots: bool,
    xlabel: str,
    script: str,
) -> None:
    """``--corpora`` mode: per-corpus + pooled 'all' rows, CSV, one PNG per
    column drawn from the CSV, YAML from the pooled 'all' row."""
    frames = {name: _load(path, prefix, cols_arg, exclude, human_label, generators)
              for name, path in corpora}
    pool_names = [n for n in POOL_SPLITS if n in frames]
    pooled = (pd.concat([frames[n] for n in pool_names], ignore_index=True)
              if pool_names else next(iter(frames.values())).iloc[0:0])

    if cols_arg:
        cols = cols_arg
    else:
        cols = sorted({c for df in frames.values() for c in df.columns
                       if c.startswith(prefix)})
    if not cols:
        raise SystemExit(f"no {prefix}* column in any of {list(frames)}")

    orient = "machine = HIGH score" if machine_is_high else "machine = LOW score"
    if generators:
        print(f"machine   : restricted to {', '.join(generators)}")
    print(f"corpora   : {', '.join(f'{n} ({len(df)})' for n, df in frames.items())}")
    print(f"pooled    : {POOLED} <- {'+'.join(pool_names) or '(none)'} "
          f"({len(pooled)} rows)")
    print(f"columns   : {cols}")
    print(f"target FPR: {fpr_target:g}   orientation: {orient}\n")

    rows: list[dict] = []
    raw_by: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    pooled_by_col: dict[str, dict] = {}

    for col in cols:
        print(f"{col}   ({label_fn(col)})")
        print(_table_header())
        for name, df in list(frames.items()) + [(POOLED, pooled)]:
            if col not in df.columns:
                print(f"{name:<18}  -- column absent --")
                continue
            sub = df[[col, GEN_COL]].dropna(subset=[col])
            raw = sub[col].to_numpy(dtype=float)
            y = (sub[GEN_COL].astype(str) != human_label).to_numpy().astype(int)
            r = _row(col, label_fn(col), name, raw, y, machine_is_high, fpr_target)
            if r is None:
                print(f"{name:<18}  -- skipped: single class or n<10 --")
                continue
            print(_table_row(r))
            rows.append(r)
            raw_by[(col, name)] = (raw, y)
            if name == POOLED:
                pooled_by_col[col] = r
        print()

    if not rows:
        raise SystemExit("nothing calibrated -- every (column, corpus) was skipped")

    _write_csv(rows, csv_path)

    if not no_plots:
        from_csv = pd.read_csv(csv_path)
        for col in from_csv["column"].unique():
            csv_rows = from_csv[from_csv["column"] == col].to_dict("records")
            raw_for_col = {
                r["corpus"]: raw_by[(col, r["corpus"])]
                for r in csv_rows
                if (col, r["corpus"]) in raw_by
            }
            csv_rows = [r for r in csv_rows if r["corpus"] in raw_for_col]
            if csv_rows:
                _plot_column(col, csv_rows, raw_for_col,
                             fig_dir / f"{col}.png", bins, xlabel)

    if not pooled_by_col:
        raise SystemExit("nothing to write -- no pooled 'all' row")
    entries, dropped = _yaml_entries(pooled_by_col, label_fn)
    if dropped:
        print(f"not written ({fpr_target:g}-FPR unreachable on pooled 'all'): {dropped}")
    if not entries:
        raise SystemExit("nothing to write -- no finite pooled threshold")
    pool_src = "+".join(pool_names)
    desc = "MG-All" if pool_names == list(POOL_SPLITS) else pool_src
    src_desc = f"{desc} ({pool_src}), pooled ('all')"
    if generators:
        src_desc += f" [machine: {'+'.join(generators)}]"
    _write_yaml(entries, block_key, out, script, src_desc, fpr_target)


def run(
    *,
    script: str,
    doc: str,
    prefix: str,
    machine_is_high: bool,
    block_key: str,
    label_fn: Callable[[str], str],
    family: str,
    xlabel: str,
    default_dataset: Path,
    default_out: Path,
    exclude_cols: frozenset[str] = frozenset(),
    argv: list[str] | None = None,
) -> None:
    """The shared entry point. Each ``calibrate_*_baseline.py`` fills in the
    family-specific pieces and calls this. ``family`` is unused now but kept in
    the signature so the wrappers do not all have to change at once."""
    ap = argparse.ArgumentParser(
        prog=script, description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("dataset", type=Path, nargs="?", default=default_dataset,
                    help=f"single calibration source, pooled over domains "
                         f"(default: {default_dataset.relative_to(BASE_DIR)})")
    ap.add_argument("--corpora", nargs="+", default=None, metavar="NAME",
                    help=f"calibrate per-corpus AND pooled 'all'; show the "
                         f"distributions. Names: {list(SPLITS)} or 'all'. "
                         f"When given, the positional dataset is ignored.")
    ap.add_argument("--cols", nargs="+", default=None,
                    help=f"columns to calibrate (default: every {prefix}* column "
                         f"except {sorted(exclude_cols) or 'none'})")
    ap.add_argument("--fpr-target", type=float, default=0.01)
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--generators", nargs="+", default=None, metavar="MODEL",
                    help="restrict the machine class to these generator model(s), "
                         "matched on the part before ':' (e.g. 'gpt56luna'); human "
                         "rows are kept. Output then goes to "
                         "results/trainv2/baseline_calibration_[flag=<models>]/ and "
                         "the YAML is written there, NOT to cfg/.")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help=f"YAML path (default: {default_out.relative_to(BASE_DIR)}, "
                         f"or inside the flag dir when --generators is given)")
    ap.add_argument("--csv", type=Path, default=None,
                    help=f"thresholds CSV path "
                         f"(default: {DEFAULT_FIG_DIR}/<script>/…_thresholds.csv)")
    ap.add_argument("--fig-dir", type=Path, default=None,
                    help=f"where --corpora writes one PNG per column "
                         f"(default: {DEFAULT_FIG_DIR}/<script>/)")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--no-plots", action="store_true",
                    help="--corpora mode: skip the PNGs (CSV is still written)")
    args = ap.parse_args(argv)

    fig_root = DEFAULT_FIG_DIR
    if args.generators:
        fig_root = DEFAULT_FIG_DIR.with_name(
            f"{DEFAULT_FIG_DIR.name}_[flag={'+'.join(args.generators)}]"
        )
    out_dir = fig_root / script.removesuffix(".py")

    yaml_out = args.out
    if yaml_out is None:
        yaml_out = (out_dir / default_out.name) if args.generators else default_out

    common = dict(
        prefix=prefix, machine_is_high=machine_is_high, block_key=block_key,
        label_fn=label_fn, cols_arg=args.cols,
        exclude=exclude_cols,  # ignored when --cols is given (explicit wins)
        fpr_target=args.fpr_target, human_label=args.human_label,
        generators=args.generators, out=yaml_out,
        script=script,
    )
    if args.corpora:
        _multi(
            resolve_corpora(args.corpora),
            csv_path=args.csv or out_dir / "thresholds.csv",
            fig_dir=args.fig_dir or out_dir,
            bins=args.bins, no_plots=args.no_plots, xlabel=xlabel, **common,
        )
    else:
        _single(
            args.dataset,
            csv_path=args.csv or out_dir / f"{args.dataset.stem}_thresholds.csv",
            **common,
        )
