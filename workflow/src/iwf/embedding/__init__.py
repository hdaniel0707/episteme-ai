"""Embedding stage (frozen): wraps epai embedders, adds SciNCL / E5."""

from workflow.src.iwf.embedding.embedders import add_embedding, embed_dataframe, emb_matrix

__all__ = ["add_embedding", "embed_dataframe", "emb_matrix"]
