"""Naive baselines used for honest model comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_baseline_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Attach all canonical baseline outputs to a dataset."""
    baseline = df.copy()
    baseline["baseline_persistence_price"] = baseline["close"]
    baseline["baseline_zero_return"] = 0.0
    baseline["baseline_momentum_return"] = baseline["close"].pct_change().fillna(0.0)
    baseline["baseline_mean_reversion_return"] = -baseline["baseline_momentum_return"]
    baseline["baseline_persistence_direction"] = (baseline["baseline_momentum_return"] > 0).astype(int)
    baseline["baseline_zero_direction"] = 0
    baseline["baseline_momentum_direction"] = (baseline["baseline_momentum_return"] > 0).astype(int)
    baseline["baseline_mean_reversion_direction"] = (baseline["baseline_mean_reversion_return"] > 0).astype(int)
    baseline["buy_and_hold_return"] = baseline["target_simple_return_1"]
    return baseline


def baseline_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact baseline prediction frame aligned to the input index."""
    baseline = add_baseline_predictions(df)
    return pd.DataFrame(
        {
            "timestamp": baseline.index,
            "persistence_price": baseline["baseline_persistence_price"],
            "zero_return": baseline["baseline_zero_return"],
            "momentum_return": baseline["baseline_momentum_return"],
            "mean_reversion_return": baseline["baseline_mean_reversion_return"],
            "momentum_direction": baseline["baseline_momentum_direction"],
            "mean_reversion_direction": baseline["baseline_mean_reversion_direction"],
        }
    ).set_index("timestamp")

