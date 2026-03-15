"""Tests for on-chain data loader and feature computation (Phase 12C)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.onchain import ONCHAIN_FEATURE_COLUMNS, compute_onchain_features


def _make_hourly_btc(n: int = 200) -> pd.DataFrame:
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


def _make_daily_onchain(n_days: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-05-01", periods=n_days, freq="D", tz="UTC")
    np.random.seed(456)
    return pd.DataFrame(
        {
            "hashrate": 5e17 + np.cumsum(np.random.randn(n_days) * 1e15),
            "difficulty": 8e13 + np.cumsum(np.random.randn(n_days) * 1e11),
            "tx_count": 300000 + np.random.randint(-5000, 5000, n_days),
            "tx_volume_usd": 1e10 + np.random.randn(n_days) * 1e8,
        },
        index=idx,
    )


class TestComputeOnchainFeatures:
    def test_all_feature_columns_present(self):
        hourly = _make_hourly_btc()
        daily = _make_daily_onchain()
        result = compute_onchain_features(hourly, daily)
        for col in ONCHAIN_FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_preserves_hourly_index(self):
        hourly = _make_hourly_btc()
        daily = _make_daily_onchain()
        result = compute_onchain_features(hourly, daily)
        pd.testing.assert_index_equal(result.index, hourly.index)

    def test_empty_daily_returns_nan_features(self):
        hourly = _make_hourly_btc()
        empty_daily = pd.DataFrame(columns=["hashrate", "difficulty", "tx_count", "tx_volume_usd"])
        result = compute_onchain_features(hourly, empty_daily)
        for col in ONCHAIN_FEATURE_COLUMNS:
            assert col in result.columns
            assert result[col].isna().all()

    def test_no_merge_date_column_in_output(self):
        hourly = _make_hourly_btc()
        daily = _make_daily_onchain()
        result = compute_onchain_features(hourly, daily)
        assert "_merge_date" not in result.columns

    def test_preserves_original_columns(self):
        hourly = _make_hourly_btc()
        daily = _make_daily_onchain()
        result = compute_onchain_features(hourly, daily)
        for col in hourly.columns:
            assert col in result.columns
