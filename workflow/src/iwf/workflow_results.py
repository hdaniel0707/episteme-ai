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

import itertools
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from workflow.src.iwf import paths  # noqa: E402
from workflow.src.iwf.config import WORKFLOWS  # noqa: E402
from workflow.src.iwf.evaluation import palette  # noqa: E402
from workflow.src.iwf.evaluation.metrics import test_correctness  # noqa: E402

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
        # Split on last underscore to handle workflow names with digits
        parts = d.name.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        wf_type, run_num = parts
        spec = WORKFLOWS.get(wf_type)
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
                                 list(getattr(settings, "class_thresholds", []) or []),
                                 target_label=getattr(settings, "target_kind",
                                                      "field_year_percentile"),
                                 seed_outlier_enabled=bool(getattr(
                                     settings, "seed_outlier_enabled", True)),
                                 seed_outlier_min_n=int(getattr(
                                     settings, "seed_outlier_min_n", 5)),
                                 seed_outlier_gap_pct=float(getattr(
                                     settings, "seed_outlier_gap_pct", 15.0)))
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


def _embedding_rows(s: dict, task: str) -> list[tuple]:
    """Describe the embedding actually used, including WHICH SPECTER2 adapter produced it.

    Without this a description could not distinguish specter2 (proximity adapter) from
    specter2_clf (classification adapter) — they differ only in the adapter, and that is
    exactly the variable these runs compare."""
    try:
        from workflow.src.iwf.embedding.embedders import (
            SPECTER2_ADAPTER_PURPOSE, SPECTER2_VARIANTS, TASK_MATCHED_EMBEDDER,
        )
    except Exception:
        return [("Embedder(s)", s.get("embedders"))]

    name = s.get("embed_for_encoder") or (s.get("embedders") or [None])[0]
    rows: list[tuple] = [("Embedder(s)", s.get("embedders")),
                        ("Embedder feeding the encoder", name)]
    if name in SPECTER2_VARIANTS:
        base_model, adapter = SPECTER2_VARIANTS[name]
        rows += [
            ("Base weights", base_model),
            ("SPECTER2 adapter", adapter or "(none — plain base weights)"),
            ("Adapter is designed for", SPECTER2_ADAPTER_PURPOSE.get(name, "(unknown)")),
        ]
        want = TASK_MATCHED_EMBEDDER.get(task)
        if want and want in SPECTER2_VARIANTS:
            match = ("yes" if name == want else
                    f"NO — this is a **{task}** run; the task-matched adapter is `{want}` "
                    f"({SPECTER2_VARIANTS[want][1]})")
            rows.append(("Adapter matches the task format?", match))
    return rows


def _embedder_dim(s: dict):
    """Width of the embedding the head sees when no encoder compresses it."""
    try:
        from workflow.src.iwf.embedding.embedders import DIMS
    except Exception:
        return "(unknown)"
    name = s.get("embed_for_encoder") or (s.get("embedders") or [None])[0]
    return DIMS.get(name, "(unknown)")


def _split_rows(s: dict) -> list[tuple]:
    """Report the split that was ACTUALLY applied. When the fixed-holdout knobs are set they
    replace split_ratios entirely, so printing the ratios alone (as this file used to)
    described a split the run never performed."""
    val_n, test_n = s.get("holdout_val_n") or 0, s.get("holdout_test_n") or 0
    rows: list[tuple] = [("Max papers (pool cap)", s.get("max_papers") or "(no cap)")]
    if val_n and test_n:
        rows += [
            ("Split mode", "fixed holdout (split_ratios ignored)"),
            ("Train rows (train_n)", s.get("train_n") or "(all remaining pool rows)"),
            ("Validation rows", val_n),
            ("Test rows", test_n),
            ("Holdout seed (fixes WHICH rows are val/test)", s.get("holdout_seed")),
            ("Train-selection seed (nested subset order)", s.get("train_select_seed")),
        ]
    else:
        rows += [
            ("Split mode", "ratio split"),
            ("Split ratios (train/val/test)", s.get("split_ratios")),
            ("Split seed", s.get("split_seed")),
        ]
    rows.append(("Year range", f"{s.get('year_from') or 'any'} .. {s.get('year_to') or 'any'}"))
    rows.append(("Training sources", s.get("training_sources")))
    return rows


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
    is_judge = wf_type.startswith("llm_")
    is_meta = "meta" in wf_type
    primary = (summary or {}).get("primary_metric", PRIMARY_METRIC[task])
    best = (summary or {}).get(f"best_{primary}")
    mean = (summary or {}).get(f"mean_{primary}")
    std = (summary or {}).get(f"std_{primary}")
    n_seeds = (summary or {}).get("n_seeds", s.get("n_runs"))

    if is_judge:
        what = "LLM-as-judge (RAG store of precedents + evolving notes.md, no trained network)"
    elif is_tfm:
        what = "text-transformer fine-tune"
    else:
        what = "frozen embedding + trained head"
    if is_meta:
        what += " + fused metadata features"
    mean_std = (f"{mean:.4g} ± {std:.4g}"
                if isinstance(mean, (int, float)) and isinstance(std, (int, float)) else "n/a")

    _n = itertools.count(1)

    def sec(title: str) -> str:
        """Number sections as they are emitted, so skipped sections leave no gap."""
        return f"## {next(_n)}. {title}"

    # headline identity of the run: model + embedding + how much data, in one line
    emb_name = s.get("embed_for_encoder") or (s.get("embedders") or [None])[0]
    arch = s.get("text_transformer_arch") if is_tfm else s.get("model_arch")
    trained_on = (f"{s.get('train_n')} train / {s.get('holdout_val_n')} val / "
                 f"{s.get('holdout_test_n')} test rows"
                 if (s.get("holdout_val_n") and s.get("holdout_test_n"))
                 else f"{s.get('max_papers') or 'all'} papers, ratio split")

    lines = [
        f"# {wf_type} — `{arch}` on `{emb_name}`",
        "",
        f"> **{primary} = {mean_std}** (best {_fmt_value(best)}) over {n_seeds} seed(s) · "
        f"target `{s.get('target_kind')}` · {trained_on}",
        "",
        f"Auto-generated from this run's `config.yaml` parameters and "
        f"`aggregate/metrics_summary.json` — do not hand-edit, it is regenerated every run. "
        f"This is a **{task.upper()}** workflow ({what}) on dataset `{s.get('dataset_id')}`. "
        f"See `../../mathematical_documentation.md` for what every metric/target means.",
        "",
        "**At a glance**",
        "",
    ]
    glance = [
        ("Model", f"`{arch}`" + (f" {s.get('hidden')}" if s.get("hidden") and not is_tfm
                                 and not is_judge else "")),
        ("Embedding", f"`{emb_name}`"),
        ("Trained on", trained_on),
        ("Result", f"{primary} = {mean_std}"),
    ]
    lines.append(_md_table(["", ""], glance))
    lines.append("")

    lines.append(sec("Run Summary"))
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

    lines.append(sec("Model"))
    if is_judge:
        from workflow.src.iwf import config as _config
        from workflow.src.main_train_judge import RETRIEVAL_TOP_K
        rows = [
            ("Kind", "RAG store built first from TRAIN's true values, then ONE LLM call "
                    "per record (train/val/test alike), fixed rubric (no notes rewriting, "
                    "no trained network)"),
            ("Retrieval embedding", s.get("embed_for_encoder") or "specter2"),
            ("Max TRAIN papers used to build the RAG", s.get("judge_max_train_papers") or
             "(all of TRAIN)"),
            ("Retrieved precedents per call (top-k)", RETRIEVAL_TOP_K),
            ("Exact-abstract-match precedents excluded", True),
            ("RAG token budget (.env.workflow)",
             _config.get_int("JUDGE_RAG_TOKEN_BUDGET", 3000)),
            ("RAG chunk token budget (.env.workflow)",
             _config.get_int("JUDGE_RAG_CHUNK_TOKENS", 200)),
            ("LLM context window (.env.workflow, 0=unknown)",
             _config.get_int("LLM_JUDGE_MAX_CONTEXT_TOKENS", 0)),
        ]
    elif is_tfm:
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
    if is_judge:
        lines.append(
            "Each `seed_<i>/model/` holds the artifact this run actually produced: "
            "`notes.md` (the fixed judging rubric, identical for every paper in the run), "
            "`rag_chunks.parquet` (every RAG-selected TRAIN abstract + its true impact, for "
            "audit), and `rag_index.joblib` (the same chunks as embeddings, for fast "
            "retrieval). The RAG store is built FIRST, deterministically, from TRAIN's true "
            "values (no LLM call). TRAIN/VAL/TEST predictions are then each exactly one blind "
            "LLM call against that one complete store — a TRAIN paper's own exact-abstract "
            "match is excluded from its own retrieved precedents, so TRAIN/VAL/TEST are judged "
            "on equal footing. No label ever leaks forward, and nothing is rewritten or "
            "fine-tuned during the run.")
        lines.append("")

    if not is_tfm and not is_judge:
        lines.append(sec("Embedding / Encoder"))
        rows = [*_embedding_rows(s, task), ("Encoder kind", s.get("encoder_kind"))]
        if (s.get("encoder_kind") or "").lower() != "identity":
            # the encoder search knobs only mean something when an encoder is actually trained
            rows += [
                ("Candidate latent-dim grid", s.get("dim_grid")),
                ("Reconstruction R² target", s.get("dim_recon_target")),
                ("Dim-selection tolerance", s.get("dim_tolerance")),
                ("Encoder max epochs", s.get("encoder_max_epochs")),
                ("Encoder patience", s.get("encoder_patience")),
            ]
        # models/<dataset_id>_encoder.json is keyed by DATASET only, so it belongs to
        # whichever queue entry last trained an encoder on this dataset. Report it only when
        # it actually describes THIS run — otherwise an identity run silently inherits the
        # autoencoder entry's k*/reconstruction numbers and misreports its own input width.
        enc_meta = _encoder_meta(s.get("dataset_id")) or {}
        kind = (s.get("encoder_kind") or "").lower()
        describes_this_run = (
            (enc_meta.get("kind") or "").lower() == kind
            and enc_meta.get("embed_for_encoder") == s.get("embed_for_encoder")
        )
        if kind == "identity":
            rows.append(("Dimensionality reduction",
                        "none — the head trains on the full embedding"))
            rows.append(("Vector width fed to the head", _embedder_dim(s)))
        elif describes_this_run:
            rows.append(("Selected latent dim k*", enc_meta.get("k_star")))
            rows.append(("Input dim (pre-encoder)", enc_meta.get("input_dim")))
            recon = enc_meta.get("reconstruction") or {}
            if recon:
                rows.append(("Reconstruction MSE", recon.get("reconstruction_mse")))
                rows.append(("Reconstruction R²", recon.get("reconstruction_r2")))
        else:
            rows.append(("Encoder metadata",
                        "(not recorded for this run — the cached encoder on this dataset "
                        "was trained by a different queue entry)"))
        lines.append(_md_table(["Parameter", "Value"], rows))
        lines.append("")

    lines.append(sec("Training Hyperparameters"))
    if is_judge:
        from workflow.src.iwf import config as _config
        rows = [
            ("Seeds (n_runs)", s.get("n_runs")),
            ("LLM provider", _config.get_str("LLM_JUDGE_PROVIDER", "(not configured)")),
            ("LLM model", _config.get_str("LLM_JUDGE_MODEL", "(not configured)")),
            ("Order TRAIN papers are seen", "shuffled per seed (model_seed + seed index)"),
        ]
    elif is_tfm:
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

    lines.append(sec("Data / Split Configuration"))
    lines.append(_md_table(["Parameter", "Value"], _split_rows(s)))
    lines.append("")

    if is_meta:
        lines.append(sec("Feature Configuration"))
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

    lines.append(sec("Full Parameter Snapshot (reproducibility)"))
    lines.append("<details><summary>Every configuration field resolved for this run — "
                "least-curated but complete (click to expand)</summary>")
    lines.append("")
    lines.append(_md_table(["Field", "Value"], sorted(s.items(), key=lambda kv: kv[0])))
    lines.append("")
    lines.append("</details>")
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

    try:
        from workflow.src.iwf.evaluation.baseline_compare import generate_baseline_plots
        generate_baseline_plots(wf_type, task, wf_dir, settings)
    except Exception as e:
        print(f"[results] {wf_type}: baseline comparison skipped ({e})")

    try:
        from workflow.src.iwf.evaluation.nf_analysis import generate_nf_analysis
        generate_nf_analysis(wf_type, wf_dir, settings)
    except Exception as e:
        print(f"[results] {wf_type}: novelty/feasibility analysis skipped ({e})")

    try:
        from workflow.src.iwf.evaluation.cross_domain import generate_cross_domain
        generate_cross_domain(wf_type, task, wf_dir, settings)
    except Exception as e:
        print(f"[results] {wf_type}: cross-domain analysis skipped ({e})")

    try:
        from workflow.src.iwf.evaluation.benchmarks import run_benchmarks
        run_benchmarks(wf_dir, settings)
    except Exception as e:
        print(f"[results] {wf_type}: benchmark evaluation skipped ({e})")

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

    summary_dir = paths.RESULTS_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_json = summary_dir / "cross_workflow_comparison.json"
    out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    if comparison:
        try:
            _write_cross_workflow_csv(comparison, summary_dir)
        except Exception as e:
            print(f"[results] cross-workflow CSV skipped ({e})")
        try:
            _plot_cross_workflow(comparison, summary_dir)
        except Exception as e:
            print(f"[results] cross-workflow figure skipped ({e})")
        try:
            _plot_algorithm_comparison(comparison, summary_dir)
        except Exception as e:
            print(f"[results] algorithm comparison figure skipped ({e})")
        try:
            _plot_correctness_comparisons(summary_dir)
        except Exception as e:
            print(f"[results] correctness comparison figures skipped ({e})")
    return comparison


def _write_cross_workflow_csv(comparison: dict, summary_dir: Path) -> None:
    """Tidy per-workflow metric table behind the cross-workflow figure: one row per
    (folder, task, primary_metric) with best/mean/std of the primary metric plus the
    mean±std of every scalar metric aggregated across seeds."""
    import csv as _csv

    rows = []
    for folder, c in comparison.items():
        primary = c.get("primary_metric", "")
        base = {
            "folder": folder,
            "workflow_type": c.get("workflow_type", ""),
            "task": c.get("task", ""),
            "primary_metric": primary,
            "n_seeds": c.get("n_seeds", ""),
            f"best_{primary}": c.get(f"best_{primary}", ""),
            f"mean_{primary}": c.get(f"mean_{primary}", ""),
            f"std_{primary}": c.get(f"std_{primary}", ""),
        }
        for mname, mstats in (c.get("metrics", {}) or {}).items():
            base[f"{mname}_mean"] = mstats.get("mean", "")
            base[f"{mname}_std"] = mstats.get("std", "")
        rows.append(base)
    if not rows:
        return
    header: list[str] = []
    for r in rows:
        for k in r:
            if k not in header:
                header.append(k)
    with (summary_dir / "cross_workflow_comparison.csv").open("w", newline="",
                                                              encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


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
        # Fixed bar thickness so a 1–2 run panel doesn't balloon into giant bars.
        ax.barh(y, means, xerr=stds, height=min(0.6, 0.9 / max(1, len(labels))),
                color=palette.OLIVE, alpha=0.85, edgecolor=palette.INK, linewidth=0.5,
                capsize=4)
        for yi, (mn, sd) in enumerate(zip(means, stds)):
            ax.text(mn + sd, yi, f" {mn:.3f}", va="center", ha="left", fontsize=8,
                    color=palette.INK)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=max(6, 10 - len(labels) // 4))
        ax.set_ylim(-0.6, len(labels) - 0.4)           # vertical breathing room
        # x headroom so bars + labels never touch the right spine (metric is 0..1-ish)
        hi = max((m + s) for m, s in zip(means, stds)) if means else 1.0
        ax.set_xlim(min(0.0, *means, 0.0), min(1.05, hi * 1.18 + 0.02))
        ax.set_xlabel(metric)
    else:
        ax.text(0.5, 0.5, "(no runs)", ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=palette.GRAY)
    ax.set_title(title)


def _panel_train_vs_test(ax, wf_names, comparison, metric, title):
    """Grouped horizontal bars: train vs test mean+-std of one metric per workflow."""
    labels, train_m, train_s, test_m, test_s = [], [], [], [], []
    for wf in wf_names:
        c = comparison.get(wf)
        if not c:
            continue
        te = c.get("metrics", {}).get(metric)
        if not te:
            continue
        tr = c.get("metrics", {}).get(f"train_{metric}")
        labels.append(wf)
        train_m.append(tr["mean"] if tr else np.nan)
        train_s.append(tr["std"] if tr else 0.0)
        test_m.append(te["mean"])
        test_s.append(te["std"])
    if not labels:
        ax.text(0.5, 0.5, "(no runs)", ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=palette.GRAY)
        ax.set_title(title)
        return
    y = np.arange(len(labels))
    h = 0.35
    ax.barh(y - h / 2, train_m, xerr=train_s, height=h, color=palette.DUSK_BLUE,
           alpha=0.85, label="train", capsize=3)
    ax.barh(y + h / 2, test_m, xerr=test_s, height=h, color=palette.OLIVE,
           alpha=0.85, label="test", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=max(6, 10 - len(labels) // 4))
    ax.set_ylim(-0.6, len(labels) - 0.4)
    finite = [v for v in (*train_m, *test_m) if np.isfinite(v)]
    hi = max(finite) if finite else 1.0
    ax.set_xlim(min(0.0, *finite, 0.0) if finite else 0.0, min(1.05, hi * 1.18 + 0.05))
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=8)


def _plot_algorithm_comparison(comparison: dict, summary_dir: Path) -> None:
    """Train-vs-test figure across every trained algorithm (all model_arch/workflow types),
    mean+-std across seeds — the 'all algorithms, train+test' comparison view."""
    reg_wfs = sorted(k for k, v in comparison.items() if v.get("task") == "reg")
    clf_wfs = sorted(k for k, v in comparison.items() if v.get("task") == "clf")
    n_max = max(len(reg_wfs), len(clf_wfs), 1)
    height = min(11.0, max(2.6, n_max * 0.6 + 1.6))
    palette.apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, height))
    _panel_train_vs_test(ax1, reg_wfs, comparison, "spearman", "Regression (Spearman rho)")
    _panel_train_vs_test(ax2, clf_wfs, comparison, "macro_f1", "Classification (macro-F1)")
    fig.suptitle("All-algorithm comparison — train vs test (mean +- std across seeds)")
    fig.tight_layout()
    fig.savefig(summary_dir / "algorithm_comparison_train_test.png", dpi=120)
    plt.close(fig)
    print(f"[results] algorithm comparison -> "
          f"{summary_dir / 'algorithm_comparison_train_test.png'}")


def _plot_cross_workflow(comparison: dict, summary_dir: Path | None = None) -> None:
    if summary_dir is None:
        summary_dir = paths.RESULTS_DIR / "summary"
    reg_wfs = sorted(k for k, v in comparison.items() if v.get("task") == "reg")
    clf_wfs = sorted(k for k, v in comparison.items() if v.get("task") == "clf")
    n_max = max(len(reg_wfs), len(clf_wfs), 1)
    # Height tracks the busiest panel's bar count (compact for few runs, taller for many).
    height = min(11.0, max(2.6, n_max * 0.6 + 1.6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, height))
    _panel(ax1, reg_wfs, comparison, "spearman", "Regression (Spearman ρ)")
    _panel(ax2, clf_wfs, comparison, "macro_f1", "Classification (macro-F1)")
    fig.suptitle("Cross-workflow comparison (mean ± std across seeds)")
    fig.tight_layout()
    fig.savefig(summary_dir / "cross_workflow_comparison.png", dpi=120)
    plt.close(fig)


# =============================================================== unified correctness bars
# One "correctness" number per run, comparable across EVERY algorithm regardless of task:
# classifiers report it directly (test accuracy); regressors/LLM-judge-reg report the
# fraction of TEST predictions that land in the correct impact class once binned by that
# SAME run's own resolved class_thresholds (see targets.resolve_settings_class_thresholds) -
# so a regression model is only "100% correct" here if its predictions truly rank papers on
# the right side of that run's actual, data-derived cut line, not by coincidence of scale.

def _collect_correctness(results_dir: Path) -> dict[str, dict]:
    """Per-run folder: mean+-std TEST and TRAIN correctness across ACCEPTED seeds (see
    metrics.accepted_seeds - a catastrophically-failed seed is dropped the same way here as
    in the per-run seed-spread plots), plus enough of run_config.json to categorize it
    (task/workflow_type/embedder/encoder)."""
    from workflow.src.iwf.evaluation.metrics import accepted_seeds
    out: dict[str, dict] = {}
    if not results_dir.is_dir():
        return out
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        rc_path = d / "run_config.json"
        if not rc_path.is_file():
            continue
        try:
            rc = json.loads(rc_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        wf_type = str(rc.get("workflow_type") or "")
        task = WORKFLOWS.get(wf_type, {}).get("task")
        if task is None:
            continue
        thresholds = rc.get("class_thresholds") or []
        results = []
        for sd in sorted(d.glob("seed_*")):
            rp = sd / "result.json"
            if not rp.is_file():
                continue
            try:
                results.append(json.loads(rp.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        if not results:
            continue
        accepted, excluded = accepted_seeds(
            results, task, thresholds,
            enabled=bool(rc.get("seed_outlier_enabled", True)),
            min_n=int(rc.get("seed_outlier_min_n", 5)),
            gap_pct=float(rc.get("seed_outlier_gap_pct", 15.0)))
        if not accepted:
            accepted, excluded = results, []
        test_vals = [v for v in (test_correctness(r, task, thresholds) for r in accepted)
                    if v is not None]
        train_vals = [v for v in (test_correctness(r, task, thresholds, split="train")
                                  for r in accepted) if v is not None]
        if not test_vals:
            continue
        out[d.name] = {
            "correctness_mean": float(np.mean(test_vals)),
            "correctness_std": float(np.std(test_vals)),
            "train_correctness_mean": float(np.mean(train_vals)) if train_vals else None,
            "train_correctness_std": float(np.std(train_vals)) if train_vals else None,
            "n_seeds": len(test_vals), "n_excluded": len(excluded),
            "n_classes": len(thresholds) + 1, "task": task, "wf_type": wf_type,
            "run_config": rc,
        }
    return out


def _algo_category(wf_type: str, task: str) -> str:
    """One of classifier/regressor/text_transformer/llm_judge for a workflow type."""
    if wf_type.startswith("llm_"):
        return "llm_judge"
    if wf_type.startswith("tfm_"):
        return "text_transformer"
    return "classifier" if task == "clf" else "regressor"


_ALGO_CATEGORY_COLORS = {
    "classifier": palette.OLIVE, "regressor": palette.SAGE,
    "text_transformer": palette.DUSK_BLUE, "llm_judge": palette.GOLD,
}


def _embedder_category(wf_type: str, rc: dict) -> str:
    """Embedder/encoding group for a run — LLM judge and text-transformers have no fixed
    embedder, so each gets its own category; embedding workflows group by
    embedder + (identity | encoder_kind + latent size)."""
    if wf_type.startswith("llm_"):
        return "llm_judge (no embedder)"
    if wf_type.startswith("tfm_"):
        arch = rc.get("text_transformer_arch") or "text_transformer"
        return f"text_transformer ({arch})"
    embedders = rc.get("embedders") or []
    embedder = rc.get("embed_for_encoder") or (embedders[0] if embedders else "specter2")
    encoder_kind = str(rc.get("encoder_kind") or "identity").lower()
    if encoder_kind == "identity":
        return str(embedder)
    dim_grid = rc.get("dim_grid") or []
    size = dim_grid[0] if len(dim_grid) == 1 else "auto"
    return f"{embedder}+{encoder_kind}{size}"


_EMBEDDER_PALETTE = [palette.OLIVE, palette.SAGE, palette.GOLD, palette.DUSK_BLUE,
                     palette.TERRACOTTA, palette.MAUVE, palette.BLUE_MID, palette.INK,
                     "#93A9C4", palette.GRAY]


def _embedder_color_map(categories: list[str]) -> dict[str, str]:
    """Stable color per embedder/encoding category, LLM judge always first (fixed color)."""
    ordered = sorted(set(categories), key=lambda c: (not c.startswith("llm_judge"), c))
    return {c: _EMBEDDER_PALETTE[i % len(_EMBEDDER_PALETTE)] for i, c in enumerate(ordered)}


def _plot_correctness_bars(entries: dict[str, dict], categories: dict[str, str],
                           color_map: dict[str, str], title: str, out_path: Path) -> None:
    """Horizontal bars: TRAIN (faded) + TEST (solid) correctness (%) per run, mean+-std
    across ACCEPTED seeds (see metrics.accepted_seeds), colored by ``categories``.
    Classifiers/regressors/text-transformers/LLM-judge all land on the same 0-100% scale,
    so every trained algorithm is directly comparable in one figure."""
    if not entries:
        return
    names = sorted(entries, key=lambda k: entries[k]["correctness_mean"])
    test_m = [entries[k]["correctness_mean"] * 100 for k in names]
    test_s = [entries[k]["correctness_std"] * 100 for k in names]
    train_m = [(entries[k]["train_correctness_mean"] or 0.0) * 100 for k in names]
    train_s = [(entries[k]["train_correctness_std"] or 0.0) * 100 for k in names]
    colors = [color_map.get(categories[k], palette.GRAY) for k in names]
    n_classes = next((entries[k]["n_classes"] for k in names), 2) or 2

    palette.apply_style()
    fig, ax = plt.subplots(figsize=(9.5, max(3.0, 0.6 * len(names) + 1.6)))
    y = np.arange(len(names))
    h = 0.35
    ax.barh(y - h / 2, train_m, xerr=train_s, height=h, color=colors, alpha=0.4,
           edgecolor=palette.INK, linewidth=0.5, capsize=3)
    ax.barh(y + h / 2, test_m, xerr=test_s, height=h, color=colors, alpha=0.9,
           edgecolor=palette.INK, linewidth=0.5, capsize=3)
    for yi, (m, sd) in enumerate(zip(test_m, test_s)):
        ax.text(m + sd + 1.0, yi + h / 2, f"{m:.0f}%", va="center", ha="left", fontsize=8,
                color=palette.INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlim(0, 108)
    chance = 100.0 / max(2, n_classes)
    ax.axvline(chance, color=palette.GRAY, ls="--", lw=1, alpha=0.6,
              label=f"chance ({n_classes} classes)")
    ax.set_xlabel("correctness (%) — solid = test, faded = train; accuracy for classifiers, "
                 "regression predictions binned by this run's own class cut-lines otherwise")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    present = sorted({categories[k] for k in names})
    handles = [Patch(facecolor=color_map.get(c, palette.GRAY), edgecolor=palette.INK,
                     linewidth=0.5, label=c) for c in present]
    handles.append(Patch(facecolor=palette.GRAY, alpha=0.9, edgecolor=palette.INK,
                         linewidth=0.5, label="test"))
    handles.append(Patch(facecolor=palette.GRAY, alpha=0.4, edgecolor=palette.INK,
                         linewidth=0.5, label="train"))
    handles.append(plt.Line2D([0], [0], color=palette.GRAY, ls="--", lw=1,
                              label=f"chance ({n_classes} classes)"))
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[results] correctness comparison -> {out_path}")


def _plot_correctness_comparisons(summary_dir: Path) -> None:
    """Both correctness-bar figures (by algorithm category, by embedder/encoding), same
    underlying values, different coloring - see _plot_correctness_bars."""
    entries = _collect_correctness(paths.RESULTS_DIR)
    if not entries:
        return

    algo_cats = {k: _algo_category(v["wf_type"], v["task"]) for k, v in entries.items()}
    _plot_correctness_bars(
        entries, algo_cats, _ALGO_CATEGORY_COLORS,
        "All-algorithm correctness — by category (classifier / regressor / "
        "text-transformer / LLM judge)",
        summary_dir / "correctness_by_category.png")

    emb_cats = {k: _embedder_category(v["wf_type"], v["run_config"]) for k, v in entries.items()}
    emb_colors = _embedder_color_map(list(emb_cats.values()))
    _plot_correctness_bars(
        entries, emb_cats, emb_colors,
        "All-algorithm correctness — by embedder / encoding",
        summary_dir / "correctness_by_embedder.png")
