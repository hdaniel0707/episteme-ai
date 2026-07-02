"""Phase 4 - reduce: apply the frozen encoder to produce the latent z (plan section 4.6).

The encoder was fit on train+val; here it is frozen and applied to ALL rows (incl. test)
to add the ``z`` column. This is inference, not training, so it is leakage-free.

    python -m workflow.src.main_reduce
"""

from __future__ import annotations

from workflow.src.iwf import paths
from workflow.src.iwf.config import Settings, load_settings
from workflow.src.iwf.data.dataset import load_parquet, save_parquet
from workflow.src.iwf.embedding.embedders import emb_matrix
from workflow.src.iwf.encoder.base import load_saved_encoder


def main(settings: Settings | None = None) -> None:
    s = settings or load_settings()
    df = load_parquet(paths.embedded_parquet(s.dataset_id))

    # Bug fix §9.2: with ENCODER_KIND=identity, phases 3/4 are effectively no-ops - z is just the
    # raw embedding. Produce it directly so phase 5 always has a valid z, even if no encoder
    # artifact was written (e.g. the encoder phase was skipped for this workflow).
    if (s.encoder_kind or "").lower() == "identity":
        X = emb_matrix(df, s.embed_for_encoder)
        df["z"] = [row.tolist() for row in X]
        print(f"[reduce] identity encoder - z = raw {s.embed_for_encoder} embedding "
              f"(dim {X.shape[1]}) for {len(df)} papers")
        save_parquet(df, paths.reduced_parquet(s.dataset_id))
        return

    enc, meta = load_saved_encoder(s.dataset_id)
    X = emb_matrix(df, meta["embed_for_encoder"])
    Z = enc.transform(X)
    df["z"] = [row.tolist() for row in Z]
    print(f"[reduce] z dim = {Z.shape[1]} for {len(df)} papers")
    save_parquet(df, paths.reduced_parquet(s.dataset_id))


if __name__ == "__main__":
    main()
