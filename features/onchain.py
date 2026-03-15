"""On-chain feature computation (Phase 12C).

Computes 6 features from daily on-chain data (hash rate, difficulty,
transaction count, transaction volume). Features capture network health
and miner behaviour as supply/demand proxies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ONCHAIN_FEATURE_COLUMNS = [
    "hashrate_change_24h",
    "hashrate_change_7d",
    "difficulty_change",
    "tx_count_zscore_7d",
    "tx_volume_zscore_7d",
    "hashrate_price_divergence",
]


def _zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=max(1, window // 2)).mean()
    rolling_std = series.rolling(window=window, min_periods=max(1, window // 2)).std()
    return ((series - rolling_mean) / rolling_std.replace(0, np.nan)).fillna(0.0)


def compute_onchain_features(
    hourly_df: pd.DataFrame,
    daily_onchain: pd.DataFrame,
) -> pd.DataFrame:
    """Merge on-chain daily data into an hourly DataFrame and compute features.

    Parameters
    ----------
    hourly_df : pd.DataFrame
        Hourly BTC price data with ``close`` column and UTC DatetimeIndex.
    daily_onchain : pd.DataFrame
        Daily on-chain data with columns ``hashrate``, ``difficulty``,
        ``tx_count``, ``tx_volume_usd`` and UTC DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Copy of *hourly_df* with on-chain feature columns added.
    """
    result = hourly_df.copy()

    if daily_onchain.empty:
        for col in ONCHAIN_FEATURE_COLUMNS:
            result[col] = float("nan")
        return result

    daily = daily_onchain.copy()
    daily.index = pd.to_datetime(daily.index, utc=True)

    # Align daily to hourly by date merge
    result["_merge_date"] = result.index.normalize()
    daily["_merge_date"] = daily.index.normalize()

    merged = result.merge(
        daily, on="_merge_date", how="left", suffixes=("", "_onchain")
    )
    merged.index = result.index

    # Forward-fill daily values across hourly rows
    for col in ["hashrate", "difficulty", "tx_count", "tx_volume_usd"]:
        if col in merged.columns:
            merged[col] = merged[col].ffill()

    # Compute features
    # Hash rate momentum (24h = 24 hourly periods, 7d = 168)
    merged["hashrate_change_24h"] = merged["hashrate"].pct_change(24, fill_method=None)
    merged["hashrate_change_7d"] = merged["hashrate"].pct_change(168, fill_method=None)

    # Difficulty change (since daily data is forward-filled, this captures
    # the latest difficulty adjustment)
    merged["difficulty_change"] = merged["difficulty"].pct_change(24, fill_method=None)

    # Transaction activity z-scores (7-day window = 168 hours)
    merged["tx_count_zscore_7d"] = _zscore(merged["tx_count"], 168)
    merged["tx_volume_zscore_7d"] = _zscore(merged["tx_volume_usd"], 168)

    # Hash rate / price divergence: z-score of ratio
    hp_ratio = merged["hashrate"] / merged["close"]
    merged["hashrate_price_divergence"] = _zscore(hp_ratio, 720)  # 30-day window

    # Clean up
    result = merged.drop(
        columns=["_merge_date"] + [c for c in ["hashrate", "difficulty", "tx_count", "tx_volume_usd"] if c in merged.columns],
        errors="ignore",
    )

    return result
