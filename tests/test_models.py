"""Model interface tests."""

import unittest

import numpy as np
import pandas as pd

from config import DEFAULT_DIRECTION_TARGET_COLUMN, DEFAULT_RETURN_TARGET_COLUMN, EXOG_COLUMNS
from evaluation.targets import add_targets
from features.pipeline import apply_feature_pipeline
from models.arima_model import ImprovedARIMAModel
from models.base import BaseModel
from models.xgboost_model import XGBoostDirectionModel, XGBoostPriceModel


def _make_train_df(n: int = 260, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 50_000 + np.cumsum(rng.randn(n) * 80)
    high = close + rng.uniform(25, 150, n)
    low = close - rng.uniform(25, 150, n)
    open_ = close + rng.randn(n) * 30
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
    engineered = apply_feature_pipeline(df, dropna=False)
    targeted = add_targets(engineered)
    return targeted.dropna()


class TestBaseModel(unittest.TestCase):
    def test_default_feature_columns(self):
        model = BaseModel()
        self.assertEqual(model.feature_columns, EXOG_COLUMNS)

    def test_prepare_features_raises_on_missing(self):
        model = BaseModel(feature_columns=["missing"])
        with self.assertRaises(ValueError):
            model._prepare_features(pd.DataFrame({"close": [1.0]}))

    def test_scaler_round_trip(self):
        model = BaseModel(feature_columns=["rsi"])
        frame = pd.DataFrame({"rsi": [10.0, 20.0, 30.0]})
        scaled = model._fit_scaler(frame)
        restored = model._transform_scaler(frame)
        self.assertAlmostEqual(float(scaled["rsi"].mean()), 0.0, places=5)
        self.assertEqual(restored.shape, frame.shape)


class TestXGBoostPriceModel(unittest.TestCase):
    def test_fit_and_predict(self):
        df = _make_train_df()
        model = XGBoostPriceModel(n_estimators=20, max_depth=2)
        metrics = model.fit(df, target_column=DEFAULT_RETURN_TARGET_COLUMN)
        self.assertIn("rmse", metrics)
        predictions = model.predict(df)
        self.assertEqual(len(predictions), len(df))

    def test_feature_importance(self):
        df = _make_train_df()
        model = XGBoostPriceModel(n_estimators=10, max_depth=2)
        model.fit(df, target_column=DEFAULT_RETURN_TARGET_COLUMN)
        importance = model.feature_importance()
        self.assertTrue(set(importance).issubset(set(EXOG_COLUMNS)))


class TestXGBoostDirectionModel(unittest.TestCase):
    def test_fit_predict_and_proba(self):
        df = _make_train_df()
        model = XGBoostDirectionModel(n_estimators=20, max_depth=2)
        metrics = model.fit(df, target_column=DEFAULT_DIRECTION_TARGET_COLUMN)
        self.assertIn("accuracy", metrics)
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        self.assertEqual(len(predictions), len(df))
        self.assertEqual(len(probabilities), len(df))
        self.assertTrue(set(np.unique(predictions)).issubset({0, 1}))


class TestImprovedARIMAModel(unittest.TestCase):
    def test_fit_and_predict(self):
        df = _make_train_df()
        model = ImprovedARIMAModel(p_range=range(0, 2), d_range=range(0, 2), q_range=range(0, 2))
        metrics = model.fit(df, target_column=DEFAULT_RETURN_TARGET_COLUMN)
        self.assertIn("aic", metrics)
        predictions = model.predict(df.iloc[-20:])
        self.assertEqual(len(predictions), 20)


if __name__ == "__main__":
    unittest.main()
