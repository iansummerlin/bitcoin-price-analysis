"""Common model interface for signal research models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import EXOG_COLUMNS


@dataclass
class ModelMetadata:
    model_name: str
    target_column: str
    feature_columns: list[str]
    trained_at: str


class BaseModel:
    """Abstract model wrapper with shared schema handling."""

    model_name = "base_model"

    def __init__(self, feature_columns: list[str] | None = None, use_scaling: bool = True):
        self.feature_columns = feature_columns or EXOG_COLUMNS
        self.use_scaling = use_scaling
        self.scaler: StandardScaler | None = None
        self.is_fitted = False
        self.target_column: str | None = None

    def fit(self, train_df: pd.DataFrame, target_column: str) -> dict:
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return canonical timestamp-aligned predictions."""
        prediction = self.predict(df)
        frame = pd.DataFrame({"prediction": prediction}, index=df.index)
        try:
            probability = self.predict_proba(df)
        except NotImplementedError:
            probability = None
        if probability is not None:
            frame["probability"] = probability
        frame["actionable"] = frame["prediction"] > 0
        return frame

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.feature_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return df[self.feature_columns].copy()

    def _fit_scaler(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.use_scaling:
            return X
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(scaled, columns=X.columns, index=X.index)

    def _transform_scaler(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.use_scaling or self.scaler is None:
            return X
        scaled = self.scaler.transform(X)
        return pd.DataFrame(scaled, columns=X.columns, index=X.index)

    def save(self, path: str | Path) -> None:
        """Persist model and sidecar metadata."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        metadata = ModelMetadata(
            model_name=self.model_name,
            target_column=self.target_column or "",
            feature_columns=list(self.feature_columns),
            trained_at=datetime.now(timezone.utc).isoformat(),
        )
        output_path.with_suffix(".metadata.json").write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "BaseModel":
        return joblib.load(path)
