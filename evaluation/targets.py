"""Trading-aligned target builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DEFAULT_ACTIONABLE_THRESHOLD,
    DEFAULT_COST_BUFFER_PCT,
    HORIZON_CONFIGS,
)


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


def add_horizon_targets(
    df: pd.DataFrame,
    horizon: int,
    cost_buffer: float | None = None,
    actionable_threshold: float | None = None,
) -> pd.DataFrame:
    """Add targets for a specific horizon with horizon-labeled column names.

    Unlike ``add_targets``, columns include the horizon suffix (e.g.
    ``target_direction_cost_adj_4h``) so multiple horizons can coexist.
    """
    cfg = HORIZON_CONFIGS.get(horizon, {})
    if cost_buffer is None:
        cost_buffer = cfg.get("cost_buffer", DEFAULT_COST_BUFFER_PCT)
    if actionable_threshold is None:
        actionable_threshold = cfg.get("actionable_threshold", DEFAULT_ACTIONABLE_THRESHOLD)

    target_df = df.copy()
    next_close = target_df["close"].shift(-horizon)
    simple_return = next_close / target_df["close"] - 1
    log_return = np.log(next_close / target_df["close"])

    suffix = f"_{horizon}h"
    target_df[f"target_close_next{suffix}"] = next_close
    target_df[f"target_simple_return{suffix}"] = simple_return
    target_df[f"target_log_return{suffix}"] = log_return
    target_df[f"target_direction{suffix}"] = (simple_return > 0).astype(int)
    target_df[f"target_direction_cost_adj{suffix}"] = (
        simple_return > actionable_threshold
    ).astype(int)
    target_df[f"target_actionable_move{suffix}"] = (
        simple_return.abs() > max(actionable_threshold, cost_buffer)
    ).astype(int)
    return target_df


def horizon_target_column(horizon: int) -> str:
    """Return the cost-adjusted direction target column name for a horizon."""
    return f"target_direction_cost_adj_{horizon}h"
