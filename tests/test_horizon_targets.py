"""Tests for multi-horizon target construction (Phase 12A)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.targets import add_horizon_targets, add_targets, horizon_target_column


def _make_price_df(n: int = 100) -> pd.DataFrame:
    """Synthetic hourly price data."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    np.random.seed(42)
    close = 40000 + np.cumsum(np.random.randn(n) * 100)
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


class TestAddHorizonTargets:
    def test_column_names_include_horizon(self):
        df = _make_price_df()
        result = add_horizon_targets(df, horizon=4)
        assert "target_direction_cost_adj_4h" in result.columns
        assert "target_close_next_4h" in result.columns
        assert "target_simple_return_4h" in result.columns
        assert "target_log_return_4h" in result.columns
        assert "target_direction_4h" in result.columns
        assert "target_actionable_move_4h" in result.columns

    def test_24h_horizon_columns(self):
        df = _make_price_df()
        result = add_horizon_targets(df, horizon=24)
        assert "target_direction_cost_adj_24h" in result.columns
        assert "target_close_next_24h" in result.columns

    def test_1h_horizon_matches_legacy(self):
        """1h horizon targets should match the legacy add_targets output."""
        df = _make_price_df()
        legacy = add_targets(df, horizon=1)
        horizon_result = add_horizon_targets(df, horizon=1)

        np.testing.assert_array_equal(
            legacy["target_direction_cost_adj"].values,
            horizon_result["target_direction_cost_adj_1h"].values,
        )
        np.testing.assert_array_almost_equal(
            legacy["target_close_next"].values,
            horizon_result["target_close_next_1h"].values,
        )

    def test_longer_horizon_shifts_correctly(self):
        df = _make_price_df(50)
        result = add_horizon_targets(df, horizon=4)
        # The target at position i should use close at position i+4
        for i in range(len(df) - 4):
            expected = df["close"].iloc[i + 4]
            actual = result["target_close_next_4h"].iloc[i]
            assert actual == pytest.approx(expected), f"Mismatch at row {i}"

    def test_nan_at_tail(self):
        df = _make_price_df(50)
        result_4h = add_horizon_targets(df, horizon=4)
        result_24h = add_horizon_targets(df, horizon=24)
        # Last 4 rows should be NaN for 4h target
        assert result_4h["target_close_next_4h"].iloc[-4:].isna().all()
        # Last 24 rows should be NaN for 24h target
        assert result_24h["target_close_next_24h"].iloc[-24:].isna().all()

    def test_multiple_horizons_coexist(self):
        df = _make_price_df()
        result = add_horizon_targets(df, horizon=1)
        result = add_horizon_targets(result, horizon=4)
        result = add_horizon_targets(result, horizon=24)
        assert "target_direction_cost_adj_1h" in result.columns
        assert "target_direction_cost_adj_4h" in result.columns
        assert "target_direction_cost_adj_24h" in result.columns

    def test_custom_cost_buffer(self):
        df = _make_price_df()
        result = add_horizon_targets(df, horizon=4, cost_buffer=0.01)
        # With higher cost buffer, fewer actionable moves
        default = add_horizon_targets(df, horizon=4)
        assert result["target_actionable_move_4h"].sum() <= default["target_actionable_move_4h"].sum()


class TestHorizonTargetColumn:
    def test_returns_correct_name(self):
        assert horizon_target_column(1) == "target_direction_cost_adj_1h"
        assert horizon_target_column(4) == "target_direction_cost_adj_4h"
        assert horizon_target_column(24) == "target_direction_cost_adj_24h"
