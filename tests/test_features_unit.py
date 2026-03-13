"""
Unit tests for all feature calculation modules.
"""
import unittest
import numpy as np
import pandas as pd

from features.price import calculate_lagged_close
from features.averages import calculate_moving_averages, calculate_average_true_range
from features.technical import calculate_rsi
from features.volatility import (
    calculate_rolling_volatility,
    calculate_ewma_volatility,
    calculate_parkinson_volatility,
)


def _make_df(n=200, base_price=100.0, seed=42):
    """Helper to create a realistic OHLCV DataFrame."""
    rng = np.random.RandomState(seed)
    close = base_price + np.cumsum(rng.randn(n) * 2)
    close = np.maximum(close, 1.0)  # Ensure positive prices
    high = close + rng.uniform(0.5, 3.0, n)
    low = close - rng.uniform(0.5, 3.0, n)
    low = np.maximum(low, 0.01)
    open_ = close + rng.randn(n) * 0.5

    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'Volume BTC': rng.uniform(1, 100, n),
        'Volume USD': rng.uniform(1000, 100000, n),
    })


class TestPriceFeatures(unittest.TestCase):
    def setUp(self):
        self.df = _make_df()

    def test_lagged_close_creates_column(self):
        calculate_lagged_close(self.df, 90)
        self.assertIn('close_lag_90', self.df.columns)

    def test_lagged_close_values_correct(self):
        calculate_lagged_close(self.df, 5)
        # Row 5 should have the value from row 0
        self.assertAlmostEqual(
            self.df['close_lag_5'].iloc[5],
            self.df['close'].iloc[0],
            places=5,
        )

    def test_lagged_close_first_rows_are_nan(self):
        calculate_lagged_close(self.df, 10)
        # First 10 rows should be NaN
        self.assertTrue(self.df['close_lag_10'].iloc[:10].isna().all())
        # Row 10 should NOT be NaN
        self.assertFalse(pd.isna(self.df['close_lag_10'].iloc[10]))

    def test_lagged_close_different_lags(self):
        for lag in [1, 5, 50, 120]:
            calculate_lagged_close(self.df, lag)
            self.assertIn(f'close_lag_{lag}', self.df.columns)


class TestAverageFeatures(unittest.TestCase):
    def setUp(self):
        self.df = _make_df()

    def test_moving_average_creates_column(self):
        calculate_moving_averages(self.df, 7)
        self.assertIn('MA_7', self.df.columns)

    def test_moving_average_correct_value(self):
        calculate_moving_averages(self.df, 7)
        # MA_7 at row 6 (0-indexed) should equal mean of rows 0-6
        expected = self.df['close'].iloc[:7].mean()
        self.assertAlmostEqual(self.df['MA_7'].iloc[6], expected, places=5)

    def test_moving_average_first_rows_nan(self):
        calculate_moving_averages(self.df, 24)
        self.assertTrue(self.df['MA_24'].iloc[:23].isna().all())

    def test_atr_creates_column(self):
        calculate_average_true_range(self.df)
        self.assertIn('atr_24', self.df.columns)

    def test_atr_positive_values(self):
        calculate_average_true_range(self.df)
        valid = self.df['atr_24'].dropna()
        self.assertTrue((valid >= 0).all())

    def test_atr_custom_window(self):
        calculate_average_true_range(self.df, window_length=10)
        self.assertIn('atr_10', self.df.columns)


class TestRSI(unittest.TestCase):
    def setUp(self):
        self.df = _make_df()

    def test_rsi_creates_column(self):
        calculate_rsi(self.df)
        self.assertIn('rsi', self.df.columns)

    def test_rsi_range(self):
        calculate_rsi(self.df)
        valid = self.df['rsi'].dropna()
        self.assertTrue((valid >= 0).all(), "RSI should be >= 0")
        self.assertTrue((valid <= 100).all(), "RSI should be <= 100")

    def test_rsi_all_gains_equals_100(self):
        """When price only goes up, RSI should be 100."""
        df = pd.DataFrame({'close': list(range(1, 50))})
        calculate_rsi(df, window_length=5)
        # After warmup period, RSI should be 100 (no losses)
        valid = df['rsi'].dropna()
        last_values = valid.iloc[-5:]
        for val in last_values:
            self.assertAlmostEqual(val, 100.0, places=1)

    def test_rsi_all_losses(self):
        """When price only goes down, RSI should be close to 0."""
        df = pd.DataFrame({'close': list(range(100, 50, -1))})
        calculate_rsi(df, window_length=5)
        valid = df['rsi'].dropna()
        last_values = valid.iloc[-5:]
        for val in last_values:
            self.assertLessEqual(val, 5.0)

    def test_rsi_flat_price(self):
        """When price is flat, gain and loss are both 0. Should not crash."""
        df = pd.DataFrame({'close': [100.0] * 30})
        calculate_rsi(df, window_length=5)
        # Should not raise, and should produce valid values
        self.assertIn('rsi', df.columns)

    def test_rsi_custom_window(self):
        calculate_rsi(self.df, window_length=7)
        self.assertIn('rsi', self.df.columns)


class TestVolatility(unittest.TestCase):
    def setUp(self):
        self.df = _make_df()

    def test_rolling_volatility_creates_column(self):
        calculate_rolling_volatility(self.df)
        self.assertIn('volatility_24', self.df.columns)

    def test_rolling_volatility_non_negative(self):
        calculate_rolling_volatility(self.df)
        valid = self.df['volatility_24'].dropna()
        self.assertTrue((valid >= 0).all())

    def test_ewma_volatility_creates_column(self):
        calculate_ewma_volatility(self.df)
        self.assertIn('volatility_ewma_24', self.df.columns)

    def test_ewma_volatility_non_negative(self):
        calculate_ewma_volatility(self.df)
        valid = self.df['volatility_ewma_24'].dropna()
        self.assertTrue((valid >= 0).all())

    def test_parkinson_volatility_creates_column(self):
        calculate_parkinson_volatility(self.df)
        self.assertIn('parkinson_volatility', self.df.columns)

    def test_parkinson_volatility_non_negative(self):
        calculate_parkinson_volatility(self.df)
        valid = self.df['parkinson_volatility'].dropna()
        self.assertTrue((valid >= 0).all())

    def test_parkinson_handles_zero_close(self):
        """Parkinson volatility should not crash with zero previous close."""
        df = _make_df(n=50)
        df.loc[0, 'close'] = 0  # Force a zero
        calculate_parkinson_volatility(df)
        # Should produce NaN for row 1 (where prev close was 0) but not crash
        self.assertIn('parkinson_volatility', df.columns)

    def test_rolling_volatility_custom_window(self):
        calculate_rolling_volatility(self.df, window_length=12)
        self.assertIn('volatility_12', self.df.columns)

    def test_ewma_volatility_custom_span(self):
        calculate_ewma_volatility(self.df, span=12)
        self.assertIn('volatility_ewma_12', self.df.columns)


if __name__ == '__main__':
    unittest.main()
