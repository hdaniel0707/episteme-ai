"""Per-workflow result folders + cross-seed aggregation.

Every workflow run writes to an auto-incremented folder ``results/<wf_type>_N/`` with:
  * ``run_config.json``      snapshot of the exact Settings used,
  * ``seed_<i>/``            per-seed model + ``result.json`` (written by
                             training.experiment or training.text_trainer — same format),
  * ``aggregate/``           mean ± std metrics summary + the full evaluation plot suite
                             + an auto-generated model description.

``cross_workflow_comparison`` scans the filesystem for every folder that has an
``aggregate/metrics_summary.json`` and produces the final comparison json + figure.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from workflow.src.iwf import paths  # noqa: E402
from workflow.src.iwf.config import WORKFLOWS  # noqa: E402

PRIMARY_METRIC = {"reg": "spearman", "clf": "macro_f1"}


def task_of(wf_type: str) -> str:
    return WORKFLOWS.get((wf_type or "").lower(), {}).get("task", "reg")


# ===================================================================== folder naming

def next_workflow_dir(wf_type: str) -> Path:
    """Return the next auto-incremented folder ``results/<wf_type>_N/`` (first run -> _1)."""
    pattern = re.compile(rf"^{re.escape(wf_type)}_(\d+)$")
    existing: list[int] = []
    if paths.RESULTS_DIR.is_dir():
        for p in paths.RESULTS_DIR.iterdir():
            if p.is_dir():
                m = pattern.match(p.name)
                if m:
                    existing.append(int(m.group(1)))
    n = max(existing, default=0) + 1
    return paths.RESULTS_DIR / f"{wf_type}_{n}"


def latest_model_dir(task: str = "reg", features: str = "z") -> Path | None:
    """Newest results folder of the given task/feature-set that has a saved joblib model.

    Used by main_predict_generated to load the most recent trained regressor. The
    default ``features="z"`` matters: generated papers have no metadata, so only a
    z-only model's input dimension matches what the scorer builds.
    """
    if not paths.RESULTS_DIR.is_dir():
        return None
    candidates = []
    for d in paths.RESULTS_DIR.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"^([a-z_]+)_(\d+)$", d.name)
        if not m:
            continue
        spec = WORKFLOWS.get(m.group(1))
        if not spec or spec["task"] != task or spec["features"] != features:
            continue
        if any((sd / "model" / "model.joblib").is_file() for sd in d.glob("seed_*")):
            candidates.append((d.stat().st_mtime, d))
    return max(candidates)[1] if candidates else None


# ===================================================================== per-workflow output

def write_run_config(wf_type: str, settings, wf_dir: Path) -> None:
    wf_dir.mkdir(parents=True, exist_ok=True)
    snap = {"workflow_type": wf_type, "folder": wf_dir.name, **settings.to_dict()}
    (wf_dir / "run_config.json").write_text(json.dumps(snap, indent=2, default=str),
                                            encoding="utf-8")


def seed_results(wf_dir: Path) -> list[dict]:
    """Load every ``seed_<i>/result.json`` in the folder (tagged with its seed dir)."""
    out = []
    for sd in sorted(Path(wf_dir).glob("seed_*")):
        rj = sd / "result.json"
        if rj.is_file():
            r = json.loads(rj.read_text(encoding="utf-8"))
            r["_seed_dir"] = str(sd)
            r["_seed_name"] = sd.name
            out.append(r)
    return out


def aggregate(wf_type: str, wf_dir: Path, settings=None) -> dict | None:
    """Write aggregate/metrics_summary.json (mean ± std across seeds) and the seed figures."""
    task = task_of(wf_type)
    primary = PRIMARY_METRIC[task]
    seeds = seed_results(wf_dir)
    if not seeds:
        return None

    scalar_keys = sorted({k for s in seeds for k, v in s.items()
                          if isinstance(v, (int, float)) and not k.startswith("_")})
    metrics_summary: dict = {}
    for k in scalar_keys:
        vals = [float(s[k]) for s in seeds if isinstance(s.get(k), (int, float))]
        if vals:
            metrics_summary[k] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals)), "n_seeds": len(vals),
            }

    summary = {
        "workflow_type": wf_type, "folder": wf_dir.name,
        "task": task, "primary_metric": primary,
        "n_seeds": len(seeds), "metrics": metrics_summary,
    }
    pvals = [float(s[primary]) for s in seeds if isinstance(s.get(primary), (int, float))]
    if pvals:
        summary[f"best_{primary}"] = float(np.max(pvals))
        summary[f"mean_{primary}"] = float(np.mean(pvals))
        summary[f"std_{primary}"] = float(np.std(pvals))

    agg = wf_dir / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    (agg / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        from workflow.src.iwf.evaluation.plots import plot_seed_spread_figures
        plot_seed_spread_figures(wf_type, task, seeds, agg,
                                 list(getattr(settings, "class_names", []) or []),
                                 list(getattr(settings, "class_thresholds", []) or []))
    except Exception as e:
        print(f"[results] {wf_type} aggregate figures skipped ({e})")
    return summary


# ===================================================================== model description

def _fmt_value(v) -> str:
    if v is None or v == "":
        return "(none)"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) if v else "(none)"
    return str(v).replace("|", "\\|")


def _md_table(headers: list[str], rows: list[tuple]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt_value(c) for c in row) + " |")
    return "\n".join(lines)


def _actual_network_shape(wf_dir: Path) -> list[int] | None:
    """Best-effort: load the first available seed's saved model and read its real
    Linear-layer sizes, so the description reflects the network actually trained."""
    for seed_dir in sorted(wf_dir.glob("seed_*")):
        candidate = seed_dir / "model" / "model.joblib"
        if not candidate.is_file():
            continue
        try:
            import joblib
            from workflow.src.iwf.evaluation.plots import extract_linear_weights
            model = joblib.load(candidate)
            net = getattr(model, "net_", None)
            if net is None:
                return None
            weights = [w for w in extract_linear_weights(net)
                       if w.shape[0] > 0 and w.shape[1] > 0]
            if not weights:
                return None
            return [weights[0].shape[1]] + [w.shape[0] for w in weights]
        except Exception:
            return None
    return None


def _encoder_meta(dataset_id: str | None) -> dict | None:
    if not dataset_id:
        return None
    mp = paths.encoder_meta_path(dataset_id)
    if not mp.is_file():
        return None
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_model_description(wf_type: str, task: str, settings, wf_dir: Path, agg: Path,
                            summary: dict | None) -> None:
    """Write ``aggregate/description_<wf_type>.md`` — an auto-generated record of exactly
    which model/config produced this run, most-important-first. Sections that do not apply
    to the workflow type are skipped."""
    s = settings.to_dict() if hasattr(settings, "to_dict") else dict(settings or {})
    is_tfm = wf_type.startswith("tfm_")
    is_meta = "meta" in wf_type
    primary = (summary or {}).get("primary_metric", PRIMARY_METRIC[task])
    best = (summary or {}).get(f"best_{primary}")
    mean = (summary or {}).get(f"mean_{primary}")
    std = (summary or {}).get(f"std_{primary}")
    n_seeds = (summary or {}).get("n_seeds", s.get("n_runs"))

    what = "text-transformer fine-tune" if is_tfm else "frozen embedding + trained head"
    if is_meta:
        what += " + fused metadata features"
    mean_std = (f"{mean:.4g} ± {std:.4g}"
                if isinstance(mean, (int, float)) and isinstance(std, (int, float)) else "n/a")

    lines = [
        f"# Model Description — {wf_type}",
        "",
        f"Auto-generated from this run's `config.yaml` parameters and "
        f"`aggregate/metrics_summary.json` — do not hand-edit, it is regenerated every run. "
        f"This is a **{task.upper()}** workflow ({what}) trained on dataset "
        f"`{s.get('dataset_id')}` against target `{s.get('target_kind')}`. Primary metric "
        f"**{primary}**: best={_fmt_value(best)}, mean±std={mean_std}, across {n_seeds} seed(s). "
        f"See `../../mathematical_documentation.md` for what every metric/target means.",
        "",
        "## 1. Run Summary",
    ]
    rows = [
        ("Workflow type", wf_type),
        ("Task", task),
        ("Dataset ID", s.get("dataset_id")),
        ("Target kind", s.get("target_kind")),
        ("Target min. refset", s.get("target_min_refset")),
        ("Primary metric", primary),
        (f"Best {primary}", best),
        (f"Mean {primary}", mean),
        (f"Std {primary}", std),
        ("Seeds run", n_seeds),
        ("Model seed", s.get("model_seed")),
    ]
    if task == "clf":
        class_names = s.get("class_names") or []
        rows += [
            ("Number of classes", len(class_names)),
            ("Class names (low→high impact)", class_names),
            ("Class scheme", s.get("class_scheme")),
            ("Class thresholds", s.get("class_thresholds")),
        ]
    lines.append(_md_table(["Parameter", "Value"], rows))
    lines.append("")

    lines.append("## 2. Neural Network Architecture")
    if is_tfm:
        arch = s.get("text_transformer_arch") or ""
        rows = [
            ("Backbone", arch),
            ("Max sequence length", s.get("text_transformer_max_len")),
            ("Frozen encoder layers", s.get("text_transformer_freeze_layers")),
        ]
        if "lora" in arch:
            rows.append(("LoRA r / alpha / dropout",
                        f"{s.get('text_transformer_lora_r')} / "
                        f"{s.get('text_transformer_lora_alpha')} / "
                        f"{s.get('text_transformer_lora_dropout')}"))
        else:
            rows.append(("LoRA", "(full fine-tune, no LoRA)"))
    else:
        rows = [
            ("Model architecture", s.get("model_arch")),
            ("Hidden layer sizes (configured)", s.get("hidden")),
            ("Hidden activation", s.get("activation")),
            ("Dropout", s.get("dropout")),
        ]
        actual = _actual_network_shape(wf_dir)
        if actual:
            rows.append(("Actual trained layer sizes (in→...→out)",
                        " → ".join(map(str, actual))))
    lines.append(_md_table(["Parameter", "Value"], rows))
    lines.append("")

    if not is_tfm:
        lines.append("## 3. Encoder / Dimensionality Reduction")
        rows = [
            ("Embedder(s)", s.get("embedders")),
            ("Embedder feeding the encoder", s.get("embed_for_encoder")),
            ("Encoder kind", s.get("encoder_kind")),
            ("Candidate latent-dim grid", s.get("dim_grid")),
            ("Reconstruction R² target", s.get("dim_recon_target")),
            ("Dim-selection tolerance", s.get("dim_tolerance")),
            ("Encoder max epochs", s.get("encoder_max_epochs")),
            ("Encoder patience", s.get("encoder_patience")),
        ]
        enc_meta = _encoder_meta(s.get("dataset_id"))
        if enc_meta:
            rows.append(("Selected latent dim k*", enc_meta.get("k_star")))
            rows.append(("Input dim (pre-encoder)", enc_meta.get("input_dim")))
            recon = enc_meta.get("reconstruction") or {}
            if recon:
                rows.append(("Reconstruction MSE", recon.get("reconstruction_mse")))
                rows.append(("Reconstruction R²", recon.get("reconstruction_r2")))
        lines.append(_md_table(["Parameter", "Value"], rows))
        lines.append("")

    lines.append("## 4. Training Hyperparameters")
    if is_tfm:
        rows = [
            ("Learning rate", s.get("text_transformer_lr")),
            ("Epochs", s.get("text_transformer_epochs")),
            ("Warmup ratio", s.get("text_transformer_warmup_ratio")),
            ("Batch size (micro)", s.get("text_transformer_batch_size")),
            ("Seeds (n_runs)", s.get("text_transformer_n_runs")),
        ]
    else:
        rows = [
            ("Learning rate", s.get("lr")),
            ("Weight decay", s.get("weight_decay")),
            ("Batch size", s.get("batch_size")),
            ("Max epochs", s.get("max_epochs")),
            ("Early-stop patience", s.get("patience")),
            ("Seeds (n_runs)", s.get("n_runs")),
        ]
        if task == "clf":
            rows += [
                ("Loss function", s.get("loss")),
                ("Focal gamma", s.get("focal_gamma")),
                ("Class-imbalance weighting", s.get("class_weight")),
                ("Effective-number beta", s.get("effective_beta")),
                ("Label smoothing", s.get("label_smoothing")),
            ]
    lines.append(_md_table(["Parameter", "Value"], rows))
    lines.append("")

    lines.append("## 5. Data / Split Configuration")
    rows = [
        ("Split seed", s.get("split_seed")),
        ("Split ratios (train/val/test)", s.get("split_ratios")),
        ("Year range", f"{s.get('year_from') or 'any'} .. {s.get('year_to') or 'any'}"),
        ("Max papers (cap)", s.get("max_papers") or "(no cap)"),
    ]
    lines.append(_md_table(["Parameter", "Value"], rows))
    lines.append("")

    if is_meta:
        lines.append("## 6. Feature Configuration")
        rows = [
            ("Use metadata features", s.get("use_metadata")),
            ("Metadata fields", s.get("metadata_fields")),
            ("Use novelty (embedding-neighbourhood)", s.get("use_novelty")),
            ("Use feasibility (embedding-neighbourhood)", s.get("use_feasibility")),
            ("External impact model(s)", s.get("external_impact")),
            ("Bypass leakage guard", s.get("bypass_leakage_guard")),
        ]
        lines.append(_md_table(["Parameter", "Value"], rows))
        lines.append("")

    lines.append("## 7. Full Parameter Snapshot (reproducibility)")
    lines.append("Every configuration field resolved for this run, least-curated but complete:")
    lines.append("")
    lines.append(_md_table(["Field", "Value"], sorted(s.items(), key=lambda kv: kv[0])))
    lines.append("")

    out_path = agg / f"description_{wf_type}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[results] {wf_type}: model description -> {out_path.name}")


def finalize_workflow(wf_type: str, settings, wf_dir: Path) -> dict | None:
    """Aggregate the seeds of one finished workflow run and produce all plots + the
    model description. Never raises for plotting problems."""
    write_run_config(wf_type, settings, wf_dir)
    summary = aggregate(wf_type, wf_dir, settings=settings)

    task = task_of(wf_type)
    try:
        from workflow.src.iwf.evaluation.plots import generate_workflow_plots
        generate_workflow_plots(wf_type, task, wf_dir, settings)
    except Exception as e:
        print(f"[results] {wf_type}: evaluation plots failed ({e})")

    if summary is not None:
        try:
            write_model_description(wf_type, task, settings, wf_dir,
                                    wf_dir / "aggregate", summary)
        except Exception as e:
            print(f"[results] {wf_type}: model description failed ({e})")

    return summary


# ===================================================================== cross-workflow summary

def cross_workflow_comparison() -> dict:
    """Scan the filesystem for every ``aggregate/metrics_summary.json`` and produce the
    cross-workflow comparison json + bar chart."""
    comparison: dict = {}
    if not paths.RESULTS_DIR.is_dir():
        return comparison
    for d in sorted(paths.RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        agg = d / "aggregate" / "metrics_summary.json"
        if agg.is_file():
            try:
                comparison[d.name] = json.loads(agg.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

    out_json = paths.RESULTS_DIR / "cross_workflow_comparison.json"
    out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    if comparison:
        try:
            _plot_cross_workflow(comparison)
        except Exception as e:
            print(f"[results] cross-workflow figure skipped ({e})")
    return comparison


def _panel(ax, wf_names, comparison, metric, title):
    labels, means, stds = [], [], []
    for wf in wf_names:
        c = comparison.get(wf)
        if not c:
            continue
        m = c.get("metrics", {}).get(metric)
        if not m:
            continue
        labels.append(wf)
        means.append(m["mean"])
        stds.append(m["std"])
    if labels:
        y = np.arange(len(labels))
        ax.barh(y, means, xerr=stds, color="tab:green", alpha=0.8, capsize=4)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=max(6, 10 - len(labels) // 4))
        ax.set_xlabel(metric)
    ax.set_title(title)


def _plot_cross_workflow(comparison: dict) -> None:
    reg_wfs = sorted(k for k, v in comparison.items() if v.get("task") == "reg")
    clf_wfs = sorted(k for k, v in comparison.items() if v.get("task") == "clf")
    width = max(12, (len(reg_wfs) + len(clf_wfs)) * 0.8)
    height = max(4.5, max(len(reg_wfs), len(clf_wfs), 1) * 0.4 + 1.5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height))
    _panel(ax1, reg_wfs, comparison, "spearman", "Regression (Spearman ρ)")
    _panel(ax2, clf_wfs, comparison, "macro_f1", "Classification (macro-F1)")
    fig.suptitle("Cross-workflow comparison (mean ± std across seeds)")
    fig.tight_layout()
    fig.savefig(paths.RESULTS_DIR / "cross_workflow_comparison.png", dpi=120)
    plt.close(fig)
