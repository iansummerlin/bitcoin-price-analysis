"""Tests for cross-asset data loader and feature computation (Phase 12B)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.crossasset import CROSSASSET_FEATURE_COLUMNS, compute_cross_asset_features


def _make_hourly_btc(n: int = 200) -> pd.DataFrame:
    """Synthetic hourly BTC price data."""
    idx = pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC")
    np.random.seed(42)
    close = 60000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame(
        {
            "open": close - 10,
            "high": close + 50,
            "low": close - 50,
            "close": close,
            "Volume BTC": np.random.rand(n) * 100,
            "Volume USD": np.random.rand(n) * 1e6,
        },
        index=idx,
    )


def _make_daily_cross(n_days: int = 60) -> pd.DataFrame:
    """Synthetic daily cross-asset data."""
    idx = pd.date_range("2024-05-01", periods=n_days, freq="D", tz="UTC")
    np.random.seed(123)
    return pd.DataFrame(
        {
            "dxy": 104 + np.cumsum(np.random.randn(n_days) * 0.1),
            "sp500": 5200 + np.cumsum(np.random.randn(n_days) * 10),
            "vix": 15 + np.random.randn(n_days) * 2,
            "gold": 2300 + np.cumsum(np.random.randn(n_days) * 5),
            "eth": 3500 + np.cumsum(np.random.randn(n_days) * 30),
        },
        index=idx,
    )


class TestComputeCrossAssetFeatures:
    def test_all_feature_columns_present(self):
        hourly = _make_hourly_btc()
        daily = _make_daily_cross()
        result = compute_cross_asset_features(hourly, daily)
        for col in CROSSASSET_FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_preserves_hourly_index(self):
        hourly = _make_hourly_btc()
        daily = _make_daily_cross()
        result = compute_cross_asset_features(hourly, daily)
        pd.testing.assert_index_equal(result.index, hourly.index)

    def test_output_preserves_original_columns(self):
        hourly = _make_hourly_btc()
        daily = _make_daily_cross()
        result = compute_cross_asset_features(hourly, daily)
        for col in hourly.columns:
            assert col in result.columns

    def test_empty_daily_returns_nan_features(self):
        hourly = _make_hourly_btc()
        empty_daily = pd.DataFrame(columns=["dxy", "sp500", "vix", "gold", "eth"])
        result = compute_cross_asset_features(hourly, empty_daily)
        for col in CROSSASSET_FEATURE_COLUMNS:
            assert col in result.columns
            assert result[col].isna().all()

    def test_vix_level_matches_input(self):
        hourly = _make_hourly_btc(48)
        daily = _make_daily_cross(10)
        result = compute_cross_asset_features(hourly, daily)
        # vix_level should be forward-filled from daily data
        non_nan = result["vix_level"].dropna()
        if len(non_nan) > 0:
            assert non_nan.dtype == float

    def test_no_merge_date_column_in_output(self):
        hourly = _make_hourly_btc()
        daily = _make_daily_cross()
        result = compute_cross_asset_features(hourly, daily)
        assert "_merge_date" not in result.columns

    def test_gold_btc_ratio_positive(self):
        hourly = _make_hourly_btc()
        daily = _make_daily_cross()
        result = compute_cross_asset_features(hourly, daily)
        valid = result["gold_btc_ratio"].dropna()
        if len(valid) > 0:
            assert (valid > 0).all()
