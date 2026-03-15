"""LightGBM direction classifier (Phase 12D).

Implements the BaseModel interface for gradient-boosted classification
using LightGBM as an alternative to XGBoost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from models.base import BaseModel

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class LightGBMDirectionModel(BaseModel):
    """LightGBM classifier for cost-aware directional targets."""

    model_name = "lightgbm_direction"

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        use_scaling: bool = True,
    ):
        super().__init__(feature_columns=feature_columns, use_scaling=use_scaling)
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is required for LightGBMDirectionModel")
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            objective="binary",
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )

    def fit(self, train_df: pd.DataFrame, target_column: str) -> dict:
        self.target_column = target_column
        X = self._fit_scaler(self._prepare_features(train_df))
        y = train_df[target_column].astype(int).values
        self.model.fit(X, y)
        self.is_fitted = True
        prediction = self.model.predict(X)
        return {
            "accuracy": float(accuracy_score(y, prediction)),
            "positive_rate": float(np.mean(y)),
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X = self._transform_scaler(self._prepare_features(df))
        return np.asarray(self.model.predict(X)).astype(int)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X = self._transform_scaler(self._prepare_features(df))
        return np.asarray(self.model.predict_proba(X))[:, 1]

    def feature_importance(self) -> dict[str, float]:
        if not self.is_fitted:
            return {}
        return dict(zip(self.feature_columns, self.model.feature_importances_))
