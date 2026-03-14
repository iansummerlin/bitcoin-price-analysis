"""Backward-compatibility shim — use evaluation.signal_rules instead."""
from evaluation.signal_rules import (  # noqa: F401
    Signal,
    SignalRule as Strategy,
    ThresholdRule as ThresholdStrategy,
    ATRThresholdRule as ATRThresholdStrategy,
    MultiFactorRule as MultiFactorStrategy,
)

__all__ = [
    "Signal",
    "Strategy",
    "ThresholdStrategy",
    "ATRThresholdStrategy",
    "MultiFactorStrategy",
]
