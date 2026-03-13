"""Trading-aligned target builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DEFAULT_ACTIONABLE_THRESHOLD, DEFAULT_COST_BUFFER_PCT


def add_targets(
    df: pd.DataFrame,
    horizon: int = 1,
    actionable_threshold: float = DEFAULT_ACTIONABLE_THRESHOLD,
    cost_buffer: float = DEFAULT_COST_BUFFER_PCT,
) -> pd.DataFrame:
    """Add price, return, and direction targets for a forward horizon."""
    target_df = df.copy()
    next_close = target_df["close"].shift(-horizon)
    simple_return = next_close / target_df["close"] - 1
    log_return = np.log(next_close / target_df["close"])

    target_df["target_close_next"] = next_close
    target_df["target_simple_return_1"] = simple_return
    target_df["target_log_return_1"] = log_return
    target_df["target_direction_1"] = (simple_return > 0).astype(int)
    target_df["target_direction_cost_adj"] = (simple_return > actionable_threshold).astype(int)
    target_df["target_actionable_move"] = (
        simple_return.abs() > max(actionable_threshold, cost_buffer)
    ).astype(int)
    return target_df

