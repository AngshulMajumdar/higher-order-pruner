"""Model-agnostic higher-order pruning for PyTorch linear layers."""

from .core import PruneConfig, PruneReport, discover_linear_groups, prune
from .pruner import HigherOrderPruner

def __getattr__(name):
    if name in {"causal_lm_loss", "load_causal_lm", "prepare_texts", "prune_hf_model"}:
        from importlib import import_module
        return getattr(import_module(".hf", __name__), name)
    raise AttributeError(name)

__all__ = [
    "HigherOrderPruner",
    "PruneConfig",
    "PruneReport",
    "discover_linear_groups",
    "prune",
    "prune_hf_model",
    "load_causal_lm",
    "prepare_texts",
    "causal_lm_loss",
]
