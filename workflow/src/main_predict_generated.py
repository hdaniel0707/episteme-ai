"""Standalone: score LLM-generated papers (MIRAI / ADP) with a trained regressor.

Generated abstracts get embedded + reduced with the SAME frozen embedder + encoder, then
scored by the most recent trained regression workflow. They have no citation target; their
predicted-impact distribution is compared to the human TEST split.

    python -m workflow.src.main_predict_generated

Set GENERATED_FILE in .env.workflow to a local export (.parquet/.csv/.jsonl); otherwise
the MIRAI API is attempted (and skipped gracefully if not configured).
"""

from __future__ import annotations

import json

import joblib

from workflow.src.iwf import config, paths, workflow_results
from workflow.src.iwf.config import Settings, load_settings
from workflow.src.iwf.data import generated_sources
from workflow.src.iwf.data.dataset import load_parquet, save_parquet, split_masks
from workflow.src.iwf.embedding.embedders import embed_dataframe, emb_matrix
from workflow.src.iwf.encoder.base import load_saved_encoder
from workflow.src.iwf.evaluation.metrics import generated_vs_human
from workflow.src.iwf.evaluation.plots import plot_generated_distribution
from workflow.src.iwf.features.feature_builder import build_xy


def _load_reg_model(wf_dir):
    """Load (model, scaler) from the first seed of a trained regression workflow."""
    for sd in sorted(wf_dir.glob("seed_*")):
        mp = sd / "model" / "model.joblib"
        sp = sd / "model" / "scaler.joblib"
        if mp.is_file() and sp.is_file():
            return joblib.load(mp), joblib.load(sp)
    raise FileNotFoundError(f"No seed with a saved model/scaler found in {wf_dir}.")


def _predict(df, model, scaler):
    X, _, _, _ = build_xy(df, task="reg", scalar_cols=[])
    return model.predict(scaler.transform(X))


def main(settings: Settings | None = None) -> None:
    s = settings or load_settings()

    # 1) load generated papers
    gen_file = config.get_str("GENERATED_FILE", "")
    source = config.get_str("GENERATED_SOURCE", "mirai")
    if gen_file:
        gen = generated_sources.load_from_file(gen_file, source=source)
    else:
        gen = generated_sources.load_from_api(source=source)
    if len(gen) == 0:
        print("[predict_generated] no generated papers available - nothing to score.")
        return

    # 2) embed + reduce with the SAME frozen embedder + encoder
    enc, meta = load_saved_encoder(s.dataset_id)
    gen = embed_dataframe(gen, [meta["embed_for_encoder"]])
    Xg = emb_matrix(gen, meta["embed_for_encoder"])
    gen["z"] = [row.tolist() for row in enc.transform(Xg)]

    # 3) score with the most recently trained regression workflow
    reg_dir = workflow_results.latest_model_dir(task="reg")
    if reg_dir is None:
        raise FileNotFoundError(
            "No trained regression workflow found under workflow/results - run an "
            "emb_reg (or *_meta_reg) workflow first."
        )
    print(f"[predict_generated] scoring with the model from {reg_dir.name}")
    model, scaler = _load_reg_model(reg_dir)
    gen["pred_impact"] = _predict(gen, model, scaler)
    save_parquet(gen, paths.generated_parquet(s.dataset_id))

    # 4) compare to the human TEST split (predicted with the same model)
    human = load_parquet(paths.reduced_parquet(s.dataset_id))
    test = human[split_masks(human)["test"]].reset_index(drop=True)
    human_pred = _predict(test, model, scaler)

    by_source = {src: gen.loc[gen["source"] == src, "pred_impact"].to_numpy()
                 for src in sorted(gen["source"].unique())}
    summary = generated_vs_human(human_pred, by_source)
    summary["model_workflow"] = reg_dir.name

    out_dir = paths.GENERATED_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_generated_distribution(human_pred, by_source, out_dir / "generated_vs_human.png")
    print(f"[predict_generated] scored {len(gen)} generated papers -> {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
