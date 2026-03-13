"""Configuration contract tests."""

import unittest

import config


class TestConfigConstants(unittest.TestCase):
    def test_feature_columns_are_defined(self):
        self.assertIsInstance(config.EXOG_COLUMNS, list)
        self.assertGreater(len(config.EXOG_COLUMNS), 10)
        self.assertIn("return_1", config.EXOG_COLUMNS)
        self.assertIn("volume_btc_zscore_24", config.EXOG_COLUMNS)

    def test_schema_versions_are_strings(self):
        self.assertIsInstance(config.DATA_SCHEMA_VERSION, str)
        self.assertIsInstance(config.FEATURE_SCHEMA_VERSION, str)
        self.assertIsInstance(config.SIGNAL_SCHEMA_VERSION, str)

    def test_market_policy_is_explicit(self):
        self.assertEqual(config.MODELING_MARKET_SOURCE, "gemini_btcusd_spot_1h")
        self.assertEqual(config.LIVE_MARKET_SOURCE, "binance_btcusdt_spot_1h")

    def test_cost_threshold_exceeds_raw_fees(self):
        self.assertGreater(config.DEFAULT_ACTIONABLE_THRESHOLD, config.DEFAULT_FEE_PCT + config.DEFAULT_SLIPPAGE_PCT)

    def test_trading_csv_columns_contain_core_fields(self):
        self.assertIn("date", config.TRADING_CSV_COLUMNS)
        for column in config.OHLCV_COLUMNS:
            self.assertIn(column, config.TRADING_CSV_COLUMNS)


if __name__ == "__main__":
    unittest.main()
