"""Phase 3 - encoder: select k* and fit the final encoder (plan sections 4.3, 4.6).

Latent-dim selection uses the regression target as the downstream probe signal: the
encoder is fit UNSUPERVISED on train+val, the probe trains on train z and is scored on
val z. The chosen smallest-good k* then defines the final encoder, which is fit on
train+val and saved (frozen) for Phase 4.

    python -m workflow.src.main_encoder
"""

from __future__ import annotations

import json

import numpy as np

from workflow.src.iwf import paths
from workflow.src.iwf.config import Settings, load_settings
from workflow.src.iwf.data.dataset import load_parquet, split_masks
from workflow.src.iwf.embedding.embedders import emb_matrix
from workflow.src.iwf.encoder.base import get_encoder
from workflow.src.iwf.encoder.dim_selector import select_dim
from workflow.src.iwf.evaluation.plots import plot_dim_curve

# ===== CONFIG =====
PROBE_TASK = "reg"   # use the main regression target to drive dim selection
# ==================


def main(settings: Settings | None = None) -> None:
    s = settings or load_settings()
    df = load_parquet(paths.embedded_parquet(s.dataset_id))
    masks = split_masks(df)

    X = emb_matrix(df, s.embed_for_encoder)
    y = df["target"].to_numpy(dtype=np.float32)

    if s.encoder_kind == "identity":
        report = {"k_star": X.shape[1], "curve": [], "baseline_metric": None,
                  "anchors": {}, "note": "identity encoder - no reduction"}
    else:
        report = select_dim(
            X, masks["train"], masks["val"], y,
            candidate_dims=s.dim_grid, task=PROBE_TASK,
            encoder_kind=s.encoder_kind, tolerance=s.dim_tolerance,
            max_epochs=s.encoder_max_epochs, seed=s.split_seed, activation=s.activation,
            patience=s.encoder_patience, recon_target=s.dim_recon_target,
        )
    k_star = int(report["k_star"])
    print(f"[encoder] embedding={s.embed_for_encoder}  k*={k_star}  "
          f"criterion={report.get('criterion')}  baseline={report.get('baseline_metric')}")

    # Fit FINAL encoder on train+val (unsupervised) with k*.
    fit_mask = masks["train"] | masks["val"]
    enc = get_encoder(s.encoder_kind, latent_dim=k_star, input_dim=X.shape[1],
                      max_epochs=s.encoder_max_epochs, seed=s.split_seed,
                      activation=s.activation, patience=s.encoder_patience)
    enc.fit(X[fit_mask])
    enc.save(paths.encoder_path(s.dataset_id))

    # Encoder "accuracy": how well encode->decode reconstructs the embedding (autoencoder).
    recon = enc.reconstruction_stats(X) if hasattr(enc, "reconstruction_stats") else {}
    if recon:
        print(f"[encoder] reconstruction over all rows: "
              f"MSE={recon['reconstruction_mse']:.5f}  R^2={recon['reconstruction_r2']:.3f}")

    meta = {"kind": s.encoder_kind, "k_star": k_star, "input_dim": int(X.shape[1]),
            "embed_for_encoder": s.embed_for_encoder, "activation": s.activation,
            "reconstruction": recon, "report": report}
    paths.encoder_meta_path(s.dataset_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if report.get("curve"):
        plot_dim_curve(report["curve"], report["baseline_metric"], k_star,
                       paths.MODELS_DIR / f"{s.dataset_id}_dim_curve.png")


if __name__ == "__main__":
    main()
