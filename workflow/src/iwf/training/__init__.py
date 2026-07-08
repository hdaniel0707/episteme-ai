"""Generic torch training loop + the multi-seed experiment runner."""

from workflow.src.iwf.training.trainer import set_seed, train_torch_model

__all__ = ["set_seed", "train_torch_model"]
