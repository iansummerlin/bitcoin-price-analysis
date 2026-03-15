"""Tests for microstructure feature computation (Phase 12E)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.microstructure import MICROSTRUCTURE_FEATURE_COLUMNS, compute_microstructure_features


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


def _make_funding_rate(n_periods: int = 30) -> pd.DataFrame:
    """Synthetic 8h funding rate data."""
    idx = pd.date_range("2024-05-25", periods=n_periods, freq="8h", tz="UTC")
    np.random.seed(789)
    return pd.DataFrame(
        {"funding_rate": np.random.randn(n_periods) * 0.001},
        index=idx,
    )


class TestComputeMicrostructureFeatures:
    def test_all_feature_columns_present(self):
        hourly = _make_hourly_btc()
        funding = _make_funding_rate()
        result = compute_microstructure_features(hourly, funding)
        for col in MICROSTRUCTURE_FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_preserves_hourly_index(self):
        hourly = _make_hourly_btc()
        funding = _make_funding_rate()
        result = compute_microstructure_features(hourly, funding)
        pd.testing.assert_index_equal(result.index, hourly.index)

    def test_empty_funding_returns_nan_features(self):
        hourly = _make_hourly_btc()
        empty = pd.DataFrame(columns=["funding_rate"])
        result = compute_microstructure_features(hourly, empty)
        for col in MICROSTRUCTURE_FEATURE_COLUMNS:
            assert col in result.columns
            assert result[col].isna().all()

    def test_preserves_original_columns(self):
        hourly = _make_hourly_btc()
        funding = _make_funding_rate()
        result = compute_microstructure_features(hourly, funding)
        for col in hourly.columns:
            assert col in result.columns
