"""Microstructure feature computation — funding rate only (Phase 12E).

Computes 3 features from Binance BTCUSDT funding rates (published every 8h).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


MICROSTRUCTURE_FEATURE_COLUMNS = [
    "funding_rate_8h",
    "funding_rate_zscore_7d",
    "funding_rate_cumulative_24h",
]


def _zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=max(1, window // 2)).mean()
    rolling_std = series.rolling(window=window, min_periods=max(1, window // 2)).std()
    return ((series - rolling_mean) / rolling_std.replace(0, np.nan)).fillna(0.0)


def compute_microstructure_features(
    hourly_df: pd.DataFrame,
    funding_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge funding rate data into an hourly DataFrame and compute features.

    Parameters
    ----------
    hourly_df : pd.DataFrame
        Hourly BTC price data with UTC DatetimeIndex.
    funding_df : pd.DataFrame
        Funding rate data with UTC DatetimeIndex and ``funding_rate`` column.
        Published every 8 hours.

    Returns
    -------
    pd.DataFrame
        Copy of *hourly_df* with funding rate feature columns added.
    """
    result = hourly_df.copy()

    if funding_df.empty:
        for col in MICROSTRUCTURE_FEATURE_COLUMNS:
            result[col] = float("nan")
        return result

    funding = funding_df.copy()
    funding.index = pd.to_datetime(funding.index, utc=True)

    # Reindex to hourly and forward-fill (8h data → hourly)
    hourly_idx = result.index
    funding_hourly = funding.reindex(hourly_idx, method="ffill")

    # Raw funding rate (forward-filled from 8h to hourly)
    result["funding_rate_8h"] = funding_hourly["funding_rate"]

    # Z-score over 7-day window (168 hourly periods)
    result["funding_rate_zscore_7d"] = _zscore(result["funding_rate_8h"], 168)

    # Cumulative funding over 24h (3 funding periods at 8h each)
    # This represents the annualized carry cost signal
    result["funding_rate_cumulative_24h"] = (
        result["funding_rate_8h"].rolling(24, min_periods=1).sum()
    )

    # Fill NaN with 0 (neutral) — funding rate of 0 means balanced longs/shorts.
    # This allows rows before the funding rate data begins (~2019) to remain
    # in the dataset rather than being dropped by the pipeline's dropna.
    for col in MICROSTRUCTURE_FEATURE_COLUMNS:
        result[col] = result[col].fillna(0.0)

    return result
