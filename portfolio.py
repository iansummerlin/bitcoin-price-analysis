"""Backward-compatibility shim — use evaluation.cost_model instead."""
from evaluation.cost_model import CostSimulator as Portfolio  # noqa: F401

__all__ = ["Portfolio"]
