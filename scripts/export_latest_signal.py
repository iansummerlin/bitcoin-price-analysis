"""Export the latest prediction to the downstream signal contract."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signals.export import export_latest_signal


if __name__ == "__main__":
    predictions = pd.read_csv("artifacts/xgboost_direction_predictions.csv", index_col=0, parse_dates=True)
    artifact = export_latest_signal(
        predictions,
        "artifacts/latest_signal.json",
        instrument="BTCUSD",
        model_version="xgboost_direction",
    )
    print(artifact)
