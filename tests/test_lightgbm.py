"""Tests for LightGBM model (Phase 12D)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import CROSSASSET_COLUMNS, DEFAULT_DIRECTION_TARGET_COLUMN, EXOG_COLUMNS, MICROSTRUCTURE_COLUMNS, ONCHAIN_COLUMNS
from evaluation.targets import add_targets
from features.pipeline import apply_feature_pipeline
from models.lightgbm_model import LightGBMDirectionModel


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
    for col in CROSSASSET_COLUMNS + ONCHAIN_COLUMNS + MICROSTRUCTURE_COLUMNS:
        engineered[col] = rng.randn(n)
    targeted = add_targets(engineered)
    return targeted.dropna()


class TestLightGBMDirectionModel:
    def test_fit_predict_and_proba(self):
        df = _make_train_df()
        model = LightGBMDirectionModel(n_estimators=20, max_depth=2)
        metrics = model.fit(df, target_column=DEFAULT_DIRECTION_TARGET_COLUMN)
        assert "accuracy" in metrics
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        assert len(predictions) == len(df)
        assert len(probabilities) == len(df)
        assert set(np.unique(predictions)).issubset({0, 1})
        assert (probabilities >= 0).all() and (probabilities <= 1).all()

    def test_feature_importance(self):
        df = _make_train_df()
        model = LightGBMDirectionModel(n_estimators=10, max_depth=2)
        model.fit(df, target_column=DEFAULT_DIRECTION_TARGET_COLUMN)
        importance = model.feature_importance()
        assert len(importance) == len(EXOG_COLUMNS)

    def test_predict_before_fit_raises(self):
        model = LightGBMDirectionModel()
        with pytest.raises(RuntimeError):
            model.predict(pd.DataFrame())

    def test_model_name(self):
        model = LightGBMDirectionModel()
        assert model.model_name == "lightgbm_direction"
