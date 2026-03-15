"""Tree-based regression and classification models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from models.base import BaseModel

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class XGBoostPriceModel(BaseModel):
    """Gradient-boosted regressor for return or price targets."""

    model_name = "xgboost_regressor"

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        use_scaling: bool = True,
    ):
        super().__init__(feature_columns=feature_columns, use_scaling=use_scaling)
        if HAS_XGBOOST:
            self.model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
            )

    def fit(self, train_df: pd.DataFrame, target_column: str) -> dict:
        self.target_column = target_column
        X = self._fit_scaler(self._prepare_features(train_df))
        y = train_df[target_column].values
        self.model.fit(X, y)
        self.is_fitted = True
        prediction = self.model.predict(X)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "mae": float(mean_absolute_error(y, prediction)),
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X = self._transform_scaler(self._prepare_features(df))
        return np.asarray(self.model.predict(X))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def feature_importance(self) -> dict[str, float]:
        if not self.is_fitted or not hasattr(self.model, "feature_importances_"):
            return {}
        return dict(zip(self.feature_columns, self.model.feature_importances_))


class XGBoostDirectionModel(BaseModel):
    """Gradient-boosted classifier for cost-aware directional targets."""

    model_name = "xgboost_direction"

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        use_scaling: bool = True,
        decision_threshold: float = 0.5,
    ):
        super().__init__(
            feature_columns=feature_columns,
            use_scaling=use_scaling,
            decision_threshold=decision_threshold,
        )
        if HAS_XGBOOST:
            self.model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                objective="binary:logistic",
                random_state=42,
                n_jobs=1,
                eval_metric="logloss",
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
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
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X))[:, 1]
        scores = np.asarray(self.model.decision_function(X))
        return 1 / (1 + np.exp(-scores))

    def feature_importance(self) -> dict[str, float]:
        if not self.is_fitted or not hasattr(self.model, "feature_importances_"):
            return {}
        return dict(zip(self.feature_columns, self.model.feature_importances_))
