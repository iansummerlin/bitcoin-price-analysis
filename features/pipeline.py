"""Canonical feature pipeline shared by train, evaluation, and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ATR_WINDOW,
    CROSSASSET_COLUMNS,
    EXOG_COLUMNS,
    LAGGED_CLOSE_PERIODS,
    MOVING_AVERAGE_PERIODS,
    REGIME_WINDOWS,
    VOLATILITY_WINDOW,
    VOLUME_ZSCORE_WINDOW,
)
from features.averages import calculate_average_true_range, calculate_moving_averages
from features.price import calculate_lagged_close
from features.technical import calculate_rsi
from features.volatility import (
    calculate_ewma_volatility,
    calculate_parkinson_volatility,
    calculate_rolling_volatility,
)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    return ((series - rolling_mean) / rolling_std.replace(0, np.nan)).fillna(0.0)


def apply_feature_pipeline(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Apply the full deterministic feature set."""
    features = df.copy()

    for lag in LAGGED_CLOSE_PERIODS:
        calculate_lagged_close(features, lag)

    for period in MOVING_AVERAGE_PERIODS:
        calculate_moving_averages(features, period)

    calculate_average_true_range(features, ATR_WINDOW)
    calculate_rsi(features)
    calculate_rolling_volatility(features, VOLATILITY_WINDOW)
    calculate_ewma_volatility(features, VOLATILITY_WINDOW)
    calculate_parkinson_volatility(features)

    features["return_1"] = features["close"].pct_change(1)
    features["return_24"] = features["close"].pct_change(24)
    features["log_return_1"] = np.log(features["close"]).diff()
    features["ma_spread_7_24"] = features["MA_7"] / features["MA_24"] - 1
    features["ma_spread_24_72"] = features["MA_24"] / features["MA_72"] - 1
    features["atr_pct"] = features["atr_24"] / features["close"]
    features["volume_btc_zscore_24"] = _zscore(features["Volume BTC"], VOLUME_ZSCORE_WINDOW)
    features["volume_usd_zscore_24"] = _zscore(features["Volume USD"], VOLUME_ZSCORE_WINDOW)

    for window in REGIME_WINDOWS:
        features[f"trend_regime_{window}"] = np.sign(features["close"].diff(window))

    if dropna:
        # Drop NaN on base features; cross-asset features may still have NaN
        # at the start due to rolling windows — those are handled in build_dataset
        drop_cols = [c for c in EXOG_COLUMNS if c in features.columns]
        features = features.dropna(subset=drop_cols)

    return features


def active_feature_columns() -> list[str]:
    """Return the full ordered list of active feature columns.

    Since Phase 12B, cross-asset features are included in EXOG_COLUMNS.
    """
    return list(EXOG_COLUMNS)


def feature_columns() -> list[str]:
    """Return the canonical ordered feature columns."""
    return list(EXOG_COLUMNS)
