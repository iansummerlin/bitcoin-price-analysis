import unittest

import numpy as np
import pandas as pd

from models.base import BaseModel


class DummyProbabilisticModel(BaseModel):
    def fit(self, train_df: pd.DataFrame, target_column: str) -> dict:
        self.is_fitted = True
        self.target_column = target_column
        return {}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([0, 1, 1])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([0.4, 0.6, 0.8])


class TestDecisionThreshold(unittest.TestCase):
    def test_predict_frame_uses_configured_decision_threshold(self):
        model = DummyProbabilisticModel(feature_columns=["close_lag_1"], decision_threshold=0.7)
        model.is_fitted = True
        df = pd.DataFrame({"close_lag_1": [1.0, 2.0, 3.0]})

        result = model.predict_frame(df)

        self.assertListEqual(result["prediction"].tolist(), [0, 0, 1])
        self.assertListEqual(result["actionable"].tolist(), [False, False, True])
        self.assertListEqual(result["probability"].round(1).tolist(), [0.4, 0.6, 0.8])


if __name__ == "__main__":
    unittest.main()
