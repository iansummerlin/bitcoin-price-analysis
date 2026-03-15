"""Cross-asset feature computation (Phase 12B).

Computes 10 features from daily cross-asset data (DXY, S&P 500, VIX,
Gold, ETH). Features are designed to capture macro regime context and
relative value signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Canonical cross-asset feature column names.
CROSSASSET_FEATURE_COLUMNS = [
    "dxy_return_1d",
    "dxy_return_5d",
    "sp500_return_1d",
    "btc_sp500_corr_30d",
    "vix_level",
    "vix_change_1d",
    "gold_btc_ratio",
    "gold_btc_ratio_zscore_30d",
    "eth_btc_ratio",
    "eth_btc_ratio_change_7d",
]


def compute_cross_asset_features(
    hourly_df: pd.DataFrame,
    daily_cross: pd.DataFrame,
) -> pd.DataFrame:
    """Merge cross-asset daily data into an hourly DataFrame and compute features.

    Parameters
    ----------
    hourly_df : pd.DataFrame
        Hourly BTC price data with ``close`` column and UTC DatetimeIndex.
    daily_cross : pd.DataFrame
        Daily cross-asset data with columns ``dxy``, ``sp500``, ``vix``,
        ``gold``, ``eth`` and UTC DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Copy of *hourly_df* with cross-asset feature columns added.
    """
    result = hourly_df.copy()

    if daily_cross.empty:
        # Return all feature columns as NaN if no data available
        for col in CROSSASSET_FEATURE_COLUMNS:
            result[col] = float("nan")
        return result

    # Align daily data to hourly by date, then forward-fill
    daily = daily_cross.copy()
    daily.index = pd.to_datetime(daily.index, utc=True)

    # Create a date column for merging
    result["_merge_date"] = result.index.normalize()
    daily["_merge_date"] = daily.index.normalize()

    # Merge on date
    merged = result.merge(
        daily, on="_merge_date", how="left", suffixes=("", "_cross")
    )
    merged.index = result.index

    # Forward-fill to propagate daily values across hourly rows
    for col in ["dxy", "sp500", "vix", "gold", "eth"]:
        if col in merged.columns:
            merged[col] = merged[col].ffill()

    # Compute features
    # DXY momentum
    merged["dxy_return_1d"] = merged["dxy"].pct_change(24, fill_method=None)  # 24 hourly periods = 1 day
    merged["dxy_return_5d"] = merged["dxy"].pct_change(24 * 5, fill_method=None)

    # Equity market
    merged["sp500_return_1d"] = merged["sp500"].pct_change(24, fill_method=None)

    # BTC/S&P correlation (rolling 30-day = 720 hours)
    btc_ret = merged["close"].pct_change(fill_method=None)
    sp_ret = merged["sp500"].pct_change(fill_method=None)
    merged["btc_sp500_corr_30d"] = btc_ret.rolling(720, min_periods=168).corr(sp_ret)

    # VIX
    merged["vix_level"] = merged["vix"]
    merged["vix_change_1d"] = merged["vix"].diff(24)

    # Gold/BTC ratio
    merged["gold_btc_ratio"] = merged["gold"] / merged["close"]
    gold_btc = merged["gold_btc_ratio"]
    rolling_mean = gold_btc.rolling(720, min_periods=168).mean()
    rolling_std = gold_btc.rolling(720, min_periods=168).std()
    merged["gold_btc_ratio_zscore_30d"] = (
        (gold_btc - rolling_mean) / rolling_std.replace(0, np.nan)
    ).fillna(0.0)

    # ETH/BTC ratio
    merged["eth_btc_ratio"] = merged["eth"] / merged["close"]
    merged["eth_btc_ratio_change_7d"] = merged["eth_btc_ratio"].pct_change(24 * 7, fill_method=None)

    # Clean up merge column
    result = merged.drop(
        columns=["_merge_date"] + [c for c in ["dxy", "sp500", "vix", "gold", "eth"] if c in merged.columns],
        errors="ignore",
    )

    return result
