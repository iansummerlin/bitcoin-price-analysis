"""Shared configuration for the Bitcoin signal research repository."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATASET_DIR = ROOT_DIR / "data"
EVALUATION_DIR = ROOT_DIR / "evaluation"
SIGNAL_DIR = ROOT_DIR / "signals"

# Source policy: use Gemini hourly spot candles for research/training and
# Binance hourly spot candles only for optional live monitoring.
MODELING_MARKET_SOURCE = "gemini_btcusd_spot_1h"
LIVE_MARKET_SOURCE = "binance_btcusdt_spot_1h"
DATA_TIMEFRAME = "1h"
TIMEZONE = "UTC"
DATA_SCHEMA_VERSION = "2026-03-13"
FEATURE_SCHEMA_VERSION = "2026-03-13"
SIGNAL_SCHEMA_VERSION = "2026-03-13"

# Local artifact paths.
BTC_PRICEDATA_1H_URL = "https://www.cryptodatadownload.com/cdd/Gemini_BTCUSD_1h.csv"
BTC_PRICEDATA_PATH = str(ROOT_DIR / "BTCUSD_1H.csv")
BTC_SENTIMENTDATA_URL = "https://api.alternative.me/fng/?limit=250000&format=csv&date_format=kr"
BTC_SENTIMENTDATA_PATH = str(ROOT_DIR / "BTC_sentiment.csv")
TRADING_DATA_PATH = str(ROOT_DIR / "BTCUSD_trading.csv")
TRADING_ERROR_LOG = str(ROOT_DIR / "BTCUSD_trading_errors")
BTCUSD_STREAM_URL = "wss://stream.binance.com:9443/ws/btcusdt@kline_1h"
BTC_SENTIMENTDATA_LIVE_URL = "https://api.alternative.me/fng/?limit=1&date_format=kr"

# Legacy artifact names retained for compatibility, but all should be treated as
# ephemeral outputs rather than evidence of current model quality.
BTC_CLOSING_PRICE_CHART_PATH = str(ROOT_DIR / "bitcoin-closing-price.png")
BTC_ACTUAL_VS_PREDICTED_CHART_PATH = str(ROOT_DIR / "bitcoin-actual-vs-predicted-price.png")
ARIMA_CLOSING_PRICE_MODEL = str(ROOT_DIR / "arima-btc-closing-price.pkl")

# Canonical dataset columns.
OHLCV_COLUMNS = ["open", "high", "low", "close", "Volume BTC", "Volume USD"]
SENTIMENT_COLUMNS = ["fng_value"]

# Feature parameters.
LAGGED_CLOSE_PERIODS = [1, 6, 24, 90, 120]
MOVING_AVERAGE_PERIODS = [7, 24, 72]
RSI_WINDOW = 14
VOLATILITY_WINDOW = 24
ATR_WINDOW = 24
VOLUME_ZSCORE_WINDOW = 24
REGIME_WINDOWS = [24, 72]

# Canonical model features.
EXOG_COLUMNS = [
    "close_lag_1",
    "close_lag_6",
    "close_lag_24",
    "close_lag_90",
    "close_lag_120",
    "return_1",
    "return_24",
    "log_return_1",
    "MA_7",
    "MA_24",
    "MA_72",
    "ma_spread_7_24",
    "ma_spread_24_72",
    "rsi",
    "volatility_24",
    "volatility_ewma_24",
    "parkinson_volatility",
    "atr_24",
    "atr_pct",
    "volume_btc_zscore_24",
    "volume_usd_zscore_24",
    "trend_regime_24",
    "trend_regime_72",
    "fng_value",
]

TRADING_CSV_COLUMNS = ["date", *OHLCV_COLUMNS, *EXOG_COLUMNS]

# Targets and cost assumptions.
DEFAULT_TARGET_COLUMN = "target_direction_cost_adj"
DEFAULT_PRICE_TARGET_COLUMN = "target_close_next"
DEFAULT_RETURN_TARGET_COLUMN = "target_log_return_1"
DEFAULT_DIRECTION_TARGET_COLUMN = "target_direction_1"
DEFAULT_ACTIONABLE_THRESHOLD = 0.0035
DEFAULT_FEE_PCT = 0.001
DEFAULT_SLIPPAGE_PCT = 0.0005
DEFAULT_COST_BUFFER_PCT = DEFAULT_FEE_PCT + DEFAULT_SLIPPAGE_PCT + 0.002

# Walk-forward defaults.
DEFAULT_INITIAL_CAPITAL = 10_000.0
DEFAULT_RISK_FREE_RATE = 0.04
DEFAULT_BUY_THRESHOLD = 1.001
DEFAULT_SELL_THRESHOLD = 0.999
DEFAULT_MAX_RISK_PER_TRADE = 0.02
DEFAULT_MAX_DRAWDOWN_PCT = 0.20
DEFAULT_TRAIN_WINDOW = 24 * 180
DEFAULT_TEST_WINDOW = 24 * 30
DEFAULT_MIN_TRAIN_ROWS = 24 * 60
DEFAULT_EVALUATION_MAX_ROWS = 24 * 365

# Modeling defaults.
DEFAULT_REGRESSION_MODEL = "xgboost_regressor"
DEFAULT_CLASSIFICATION_MODEL = "xgboost_direction"
MODEL_STALE_AFTER_DAYS = 30
