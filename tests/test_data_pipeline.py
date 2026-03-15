"""Tests for canonical dataset loading and validation."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.loaders import load_price_history, load_sentiment_history
from data.pipeline import build_dataset
from data.validation import validate_ohlcv_frame


PRICE_CSV = """https://example.com/source
date,open,high,low,close,Volume BTC,Volume USD
2024-01-01 00:00:00,100,110,90,105,10,1050
2024-01-01 01:00:00,105,115,95,110,11,1210
2024-01-01 02:00:00,110,120,100,115,12,1380
"""

SENTIMENT_CSV = """date,fng_value
2024-01-01,45
"""


class TestLoadersAndValidation(unittest.TestCase):
    def test_load_price_history_handles_url_header(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "price.csv"
            path.write_text(PRICE_CSV, encoding="utf-8")
            frame = load_price_history(path)
            self.assertEqual(len(frame), 3)
            self.assertEqual(frame.index.tz.zone if hasattr(frame.index.tz, "zone") else "UTC", "UTC")

    def test_validation_flags_bad_ohlc(self):
        frame = pd.DataFrame(
            {
                "open": [100],
                "high": [90],
                "low": [95],
                "close": [92],
                "Volume BTC": [10],
                "Volume USD": [920],
            },
            index=pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC"),
        )
        errors = validate_ohlcv_frame(frame)
        self.assertTrue(any("high below low" in error for error in errors))

    def test_build_dataset_merges_features_and_targets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            price_path = Path(tmp_dir) / "price.csv"
            sentiment_path = Path(tmp_dir) / "sentiment.csv"
            long_price = "https://example.com/source\n" + "date,open,high,low,close,Volume BTC,Volume USD\n"
            rows = []
            for i in range(260):
                rows.append(f"2024-01-{1 + i // 24:02d} {i % 24:02d}:00:00,{100+i},{105+i},{95+i},{100+i},10,1000")
            price_path.write_text(long_price + "\n".join(rows) + "\n", encoding="utf-8")
            sentiment_path.write_text(SENTIMENT_CSV, encoding="utf-8")
            dataset, metadata = build_dataset(
                price_path=price_path,
                sentiment_path=sentiment_path,
                include_crossasset=False,
                include_onchain=False,
                include_microstructure=False,
            )
            self.assertGreater(len(dataset), 50)
            self.assertIn("target_direction_cost_adj", dataset.columns)
            self.assertEqual(metadata.market_source, "gemini_btcusd_spot_1h")


if __name__ == "__main__":
    unittest.main()
