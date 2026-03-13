"""ARIMA/SARIMAX-style baseline regression model."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from models.base import BaseModel


class ImprovedARIMAModel(BaseModel):
    """ARIMA regressor using canonical features as exogenous inputs."""

    model_name = "arima_regressor"

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        use_scaling: bool = True,
        p_range: range = range(0, 3),
        d_range: range = range(0, 2),
        q_range: range = range(0, 3),
    ):
        super().__init__(feature_columns=feature_columns, use_scaling=use_scaling)
        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range
        self.model = None
        self.best_order = (1, 0, 0)

    def _grid_search(self, y: pd.Series, exog: pd.DataFrame) -> tuple[tuple[int, int, int], float]:
        best_aic = np.inf
        best_order = self.best_order
        for order in product(self.p_range, self.d_range, self.q_range):
            try:
                fit = ARIMA(y, order=order, exog=exog).fit()
            except Exception:
                continue
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_order = order
        return best_order, float(best_aic)

    def fit(self, train_df: pd.DataFrame, target_column: str) -> dict:
        self.target_column = target_column
        X = self._fit_scaler(self._prepare_features(train_df))
        y = train_df[target_column]
        self.best_order, best_aic = self._grid_search(y, X)
        self.model = ARIMA(y, order=self.best_order, exog=X).fit()
        self.is_fitted = True

        fitted = pd.Series(self.model.fittedvalues, index=y.index).reindex(y.index).dropna()
        truth = y.loc[fitted.index]
        return {
            "rmse": float(np.sqrt(mean_squared_error(truth, fitted))),
            "mae": float(mean_absolute_error(truth, fitted)),
            "aic": best_aic,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        X = self._transform_scaler(self._prepare_features(df))
        forecast = self.model.forecast(steps=len(df), exog=X)
        return np.asarray(forecast)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
