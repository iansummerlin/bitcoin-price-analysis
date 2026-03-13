"""Tests for the live analytics preparation path."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from collect import handle_is_enough_data, prepare_exog_data, validate_row
from features.pipeline import apply_feature_pipeline


def _make_df(n: int = 220) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": 50_000 + rng.randn(n) * 100,
            "high": 50_200 + rng.randn(n) * 100,
            "low": 49_800 + rng.randn(n) * 100,
            "close": 50_000 + rng.randn(n) * 100,
            "Volume BTC": rng.uniform(1, 100, n),
            "Volume USD": rng.uniform(50_000, 500_000, n),
            "fng_value": rng.randint(10, 90, n).astype(float),
        },
        index=dates,
    )
    df.index.name = "date"
    return apply_feature_pipeline(df, dropna=False)


class TestValidateRow(unittest.TestCase):
    def test_valid_row(self):
        self.assertTrue(validate_row({"close": 50_000, "high": 50_100, "low": 49_900, "rsi": 55}))

    def test_invalid_row(self):
        self.assertFalse(validate_row({"close": -1, "high": 50_100, "low": 49_900}))


class TestHandleIsEnoughData(unittest.TestCase):
    def test_threshold_logic(self):
        self.assertTrue(handle_is_enough_data(pd.DataFrame({"close": range(120)})))
        self.assertFalse(handle_is_enough_data(pd.DataFrame({"close": range(119)})))


class TestPrepareExogData(unittest.TestCase):
    def test_appends_and_recomputes_features(self):
        df = _make_df()
        payload = {
            "k": {
                "t": 1704888000000,
                "o": "50100",
                "h": "50300",
                "l": "49900",
                "c": "50250",
                "v": "10.5",
                "q": "527625",
            }
        }
        result = prepare_exog_data(df, payload, 55.0)
        self.assertGreater(len(result), len(df))
        self.assertIn("atr_24", result.columns)
        self.assertIn("trend_regime_72", result.columns)


if __name__ == "__main__":
    unittest.main()
