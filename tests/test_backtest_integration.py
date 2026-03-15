"""Integration tests for the research/evaluation pipeline."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import apply_features, generate_naive_predictions, run_backtest
from config import CROSSASSET_COLUMNS, DEFAULT_DIRECTION_TARGET_COLUMN, MICROSTRUCTURE_COLUMNS, ONCHAIN_COLUMNS
from evaluation.targets import add_targets
from evaluation.walk_forward import iter_walk_forward_slices, walk_forward_evaluate
from signals.export import export_latest_signal, validate_signal_artifact
from evaluation.signal_rules import MultiFactorRule, ThresholdRule


def _make_historical_df(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 50_000 + np.cumsum(rng.randn(n) * 90)
    high = close + rng.uniform(50, 200, n)
    low = close - rng.uniform(50, 200, n)
    open_ = close + rng.randn(n) * 35
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "Volume BTC": rng.uniform(1, 100, n),
            "Volume USD": rng.uniform(50_000, 500_000, n),
            "fng_value": rng.randint(10, 90, n).astype(float),
        },
        index=dates,
    )
    featured = apply_features(df)
    rng2 = np.random.RandomState(seed + 1)
    for col in CROSSASSET_COLUMNS + ONCHAIN_COLUMNS + MICROSTRUCTURE_COLUMNS:
        featured[col] = rng2.randn(len(featured))
    return add_targets(featured)


class TestBacktestCompatibilityHelpers(unittest.TestCase):
    def test_naive_predictions_and_signal_consumer(self):
        df = _make_historical_df(350)
        naive = generate_naive_predictions(df)
        self.assertIn("predicted_price", naive.columns)
        portfolio, buy_hold = run_backtest(naive, ThresholdRule())
        self.assertEqual(len(portfolio.equity_curve), len(naive))
        self.assertEqual(len(buy_hold), len(naive))


class TestWalkForwardEvaluation(unittest.TestCase):
    def test_slices_are_strictly_past_to_future(self):
        df = _make_historical_df()
        slices = list(iter_walk_forward_slices(df, train_window=240, test_window=120, min_train_rows=120))
        self.assertGreater(len(slices), 0)
        train_df, test_df = slices[0]
        self.assertLess(train_df.index.max(), test_df.index.min())

    def test_walk_forward_produces_predictions_and_summary(self):
        df = _make_historical_df()
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = walk_forward_evaluate(
                df,
                model_name="xgboost_direction",
                target_column=DEFAULT_DIRECTION_TARGET_COLUMN,
                train_window=240,
                test_window=120,
                output_dir=tmp_dir,
            )
            self.assertGreater(result.windows_evaluated, 0)
            self.assertIn("directional_accuracy", result.scores)
            self.assertTrue(Path(result.prediction_path).exists())

    def test_signal_export_contract(self):
        df = _make_historical_df()
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = walk_forward_evaluate(
                df,
                model_name="xgboost_direction",
                target_column=DEFAULT_DIRECTION_TARGET_COLUMN,
                train_window=240,
                test_window=120,
                output_dir=tmp_dir,
            )
            predictions = pd.read_csv(result.prediction_path, index_col=0, parse_dates=True)
            signal_path = Path(tmp_dir) / "latest_signal.json"
            artifact = export_latest_signal(
                predictions,
                signal_path,
                instrument="BTCUSD",
                model_version="xgboost_direction_test",
            )
            payload = validate_signal_artifact(signal_path)
            self.assertEqual(payload["instrument"], "BTCUSD")
            self.assertFalse(payload["is_stale"])
            self.assertEqual(artifact.signal_schema_version, payload["signal_schema_version"])


if __name__ == "__main__":
    unittest.main()
