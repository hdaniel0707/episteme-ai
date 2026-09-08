r"""Judge x generator matrix: AUROC and TPR@1%FPR (oracle), one dataset.

    uv run python epai/ai_detection/trainv2/run_judge_by_generator.py \
        data/parquet/scribe_sci.parquet

For every (judge, generator) pair in ONE dataset -- human rows vs. that ONE
generator's rows, every other generator's rows excluded from the cell -- this
reports AUROC (threshold-free) and the ORACLE TPR@1%FPR: the threshold read
off THAT CELL's own ROC curve (``cd_core.roc_point_at_fpr``), not a threshold
carried in from anywhere else. That is a deliberate difference from
``run_judge_performance.py``, which applies one FIXED, pre-calibrated cutoff
across several datasets -- this is a within-dataset comparison of what each
judge CAN tell apart, generator by generator, not what one deployed cutoff
would catch.

Cells where the judge and the generator share a model family are marked
``*`` -- whether a judge is better or worse at catching text from its own
family is the reason this matrix exists, not a footnote.

Output:
    <out>_auroc.csv               judge x generator, AUROC
    <out>_tpr_at_<L>fpr.csv       judge x generator, oracle TPR@L FPR
    <out>_auroc.png               heatmap
    <out>_tpr_at_<L>fpr.png       heatmap

Run:
    uv run python epai/ai_detection/trainv2/run_judge_by_generator.py
    uv run python epai/ai_detection/trainv2/run_judge_by_generator.py \
        data/parquet/scribe_sci.parquet --fpr-target 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.cd_core import auroc_score, oriented_scores, roc_point_at_fpr

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = BASE_DIR / "data" / "parquet" / "scribe_sci.parquet"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "judge_by_generator"

JUDGE_PREFIX = "llm_judge_score_"
GEN_COL = "generated_by"
# The judge writes P(human): a LOW score means machine. Fixed, not a flag --
# matches cd_core's llm_judge BaselineKind.
MACHINE_IS_HIGH = False


def _judge_key(column: str) -> str:
    return column.replace(JUDGE_PREFIX, "").replace("_v3", "")


def _norm(s: str) -> str:
    """Strip separators and case so 'gptoss_120b' and 'gptoss120b' compare equal."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def is_self(judge: str, generator: str) -> bool:
    """Same model family judging its own generations (the prompt/version id
    after ':' is the generator's, not the model's, so it is stripped first)."""
    return _norm(judge) == _norm(generator.split(":", 1)[0])


def cell(df: pd.DataFrame, column: str, generator: str, human_label: str,
         fpr_target: float) -> dict | None:
    sub = df[df[GEN_COL].isin([human_label, generator])][[column, GEN_COL]]
    sub = sub.dropna(subset=[column])
    raw = sub[column].to_numpy(dtype=float)
    y = (sub[GEN_COL].astype(str) != human_label).to_numpy().astype(int)
    if np.unique(y).size < 2 or len(y) < 10:
        return None
    oriented = oriented_scores(raw, MACHINE_IS_HIGH)
    tpr, _ = roc_point_at_fpr(y, oriented, fpr_target)
    return {"n": int(len(y)), "auroc": auroc_score(y, oriented), "tpr_at_fpr": tpr}


def plot_heatmap(matrix: pd.DataFrame, judges: list[str], generators: list[str],
                  title: str, path: Path, cmap: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = matrix.loc[judges, generators].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.3 * len(generators) + 2.4, 0.9 * len(judges) + 1.6))
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(generators)))
    ax.set_xticklabels(generators, rotation=30, ha="right")
    ax.set_yticks(range(len(judges)))
    ax.set_yticklabels(judges)
    for i, j in enumerate(judges):
        for k, g in enumerate(generators):
            v = data[i, k]
            text = "-" if np.isnan(v) else f"{v:.2f}"
            if not np.isnan(v) and is_self(j, g):
                text += "*"
            ax.text(k, i, text, ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(f"{title}\n(* = judge and generator share a model family)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_DATASET)
    ap.add_argument("--cols", nargs="+", default=None,
                     help="judge columns (default: every llm_judge_score_* column)")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--fpr-target", type=float, default=0.01)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    if GEN_COL not in df.columns:
        raise SystemExit(f"{args.dataset}: no {GEN_COL!r} column")
    cols = args.cols or sorted(c for c in df.columns if c.startswith(JUDGE_PREFIX))
    if not cols:
        raise SystemExit(f"no {JUDGE_PREFIX}* column in {args.dataset}")
    generators = sorted(g for g in df[GEN_COL].astype(str).unique() if g != args.human_label)
    if not generators:
        raise SystemExit(f"{args.dataset}: no non-human {GEN_COL!r} values")
    judges = [_judge_key(c) for c in cols]

    print(f"dataset    : {args.dataset}  ({len(df)} rows)")
    print(f"judges     : {judges}")
    print(f"generators : {generators}\n")

    auroc_mat = pd.DataFrame(index=judges, columns=generators, dtype=float)
    tpr_mat = pd.DataFrame(index=judges, columns=generators, dtype=float)

    for col, judge in zip(cols, judges):
        for gen in generators:
            r = cell(df, col, gen, args.human_label, args.fpr_target)
            if r is None:
                print(f"  skipped: {judge} x {gen} -- n<10 or single class")
                continue
            auroc_mat.loc[judge, gen] = r["auroc"]
            tpr_mat.loc[judge, gen] = r["tpr_at_fpr"]

    def _fmt(mat: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=mat.index, columns=mat.columns, dtype=object)
        for j in mat.index:
            for g in mat.columns:
                v = mat.loc[j, g]
                if pd.isna(v):
                    out.loc[j, g] = ""
                else:
                    out.loc[j, g] = f"{v:.3f}" + ("*" if is_self(j, g) else "")
        return out

    print(f"\nAUROC  (machine = positive; * = own model family)")
    print(_fmt(auroc_mat).to_string())
    print(f"\nTPR@{args.fpr_target * 100:g}%FPR (oracle; * = own model family)")
    print(_fmt(tpr_mat).to_string())

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.dataset.stem
    auroc_csv = out_dir / f"{stem}_auroc.csv"
    tpr_csv = out_dir / f"{stem}_tpr_at_{args.fpr_target:g}fpr.csv"
    auroc_mat.to_csv(auroc_csv)
    tpr_mat.to_csv(tpr_csv)
    print(f"\nwrote {auroc_csv}\nwrote {tpr_csv}")

    if not args.no_plots:
        plot_heatmap(auroc_mat, judges, generators, "AUROC by judge x generator",
                     out_dir / f"{stem}_auroc.png", "coolwarm")
        plot_heatmap(tpr_mat, judges, generators,
                     f"TPR@{args.fpr_target * 100:g}%FPR (oracle) by judge x generator",
                     out_dir / f"{stem}_tpr_at_{args.fpr_target:g}fpr.png", "viridis")


if __name__ == "__main__":
    main()
