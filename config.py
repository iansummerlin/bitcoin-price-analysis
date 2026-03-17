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
    # Cross-asset features (Phase 12B).
    "dxy_return_1d",
    "dxy_return_5d",
    "sp500_return_1d",
    "btc_sp500_corr_30d",
    "vix_level",
    "vix_change_1d",
    "gold_btc_ratio",
    "gold_btc_ratio_zscore_30d",
    "eth_btc_ratio",
    "eth_btc_ratio_change_7d",
    # On-chain features (Phase 12C).
    "hashrate_change_24h",
    "hashrate_change_7d",
    "difficulty_change",
    "tx_count_zscore_7d",
    "tx_volume_zscore_7d",
    "hashrate_price_divergence",
    # Microstructure features (Phase 12E).
    "funding_rate_8h",
    "funding_rate_zscore_7d",
    "funding_rate_cumulative_24h",
]

# Cross-asset feature columns (Phase 12B) — also listed separately for
# ablation and conditional pipeline control.
CROSSASSET_COLUMNS = [
    "dxy_return_1d",
    "dxy_return_5d",
    "sp500_return_1d",
    "btc_sp500_corr_30d",
    "vix_level",
    "vix_change_1d",
    "gold_btc_ratio",
    "gold_btc_ratio_zscore_30d",
    "eth_btc_ratio",
    "eth_btc_ratio_change_7d",
]

# On-chain feature columns (Phase 12C).
ONCHAIN_COLUMNS = [
    "hashrate_change_24h",
    "hashrate_change_7d",
    "difficulty_change",
    "tx_count_zscore_7d",
    "tx_volume_zscore_7d",
    "hashrate_price_divergence",
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

# Multi-horizon configuration for Phase 12A.
# Per-trade cost (fee + slippage) is constant regardless of holding period.
# The buffer component is the same across horizons — the evaluation will show
# whether longer horizons naturally produce larger moves that clear the bar.
HORIZON_CONFIGS = {
    1: {"cost_buffer": DEFAULT_COST_BUFFER_PCT, "actionable_threshold": DEFAULT_ACTIONABLE_THRESHOLD},
    4: {"cost_buffer": DEFAULT_COST_BUFFER_PCT, "actionable_threshold": DEFAULT_ACTIONABLE_THRESHOLD},
    24: {"cost_buffer": DEFAULT_COST_BUFFER_PCT, "actionable_threshold": DEFAULT_ACTIONABLE_THRESHOLD},
}

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

# Microstructure feature columns (Phase 12E).
MICROSTRUCTURE_COLUMNS = [
    "funding_rate_8h",
    "funding_rate_zscore_7d",
    "funding_rate_cumulative_24h",
]

# Cache TTLs (seconds) for external data sources.
CACHE_TTL_CROSSASSET = 6 * 3600       # 6 hours — daily data, markets close at different times
CACHE_TTL_ONCHAIN = 24 * 3600         # 24 hours — daily resolution
CACHE_TTL_MICROSTRUCTURE = 1 * 3600   # 1 hour — funding rates update every 8h

# Phase 13: Held-out validation split.
# Reserve the most recent 3 months of data as a final validation set.
# The experiment loop must NEVER train on or evaluate against this period.
# It is only used once, at the very end, to check if the best surviving
# configuration generalizes.
HELD_OUT_MONTHS = 3
HELD_OUT_HOURS = HELD_OUT_MONTHS * 30 * 24  # ~2160 hours

# Phase 13: Experiment loop budget.
EXPERIMENT_BUDGET_MAX = 100
EXPERIMENT_BUDGET_HOURS = 12
EXPERIMENT_MIN_IMPROVEMENT = 0.005  # minimum metric improvement to keep a change
EXPERIMENT_MAX_ADDED_LINES = 20  # max lines added without proportional gain

# Modeling defaults.
DEFAULT_REGRESSION_MODEL = "xgboost_regressor"
DEFAULT_CLASSIFICATION_MODEL = "xgboost_direction"
MODEL_STALE_AFTER_DAYS = 30

# Reviewed baseline used by the canonical train/backtest/export paths.
# This is intentionally conservative and must only change after explicit review.
REVIEWED_BASELINE_CONFIG = {
    "description": "reviewed baseline: xgb default repo config",
    "model_name": "xgboost_direction",
    "model_kwargs": {
        "n_estimators": 100,
        "max_depth": 3,
    },
    "feature_columns": list(EXOG_COLUMNS),
}

# Best surviving configuration from the March 15, 2026 Phase 13 autoresearch run.
# This is a research candidate baseline for future experiment loops, not a
# reviewed repo-wide default.
PHASE13_CANDIDATE_BASELINE_CONFIG = {
    "description": "phase13 candidate baseline: xgb n_est=300, max_d=5, lr=0.1",
    "model_name": "xgboost_direction",
    "model_kwargs": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.1,
        "decision_threshold": 0.5,
    },
    "feature_columns": list(EXOG_COLUMNS),
}

# The experiment loop should start from the best known research candidate rather
# than the reviewed production-style default, but that candidate is not promoted
# automatically into the canonical train/backtest/export path.
EXPERIMENT_LOOP_BASELINE_CONFIG = PHASE13_CANDIDATE_BASELINE_CONFIG

# Phase 13 precision-first search settings.
EXPERIMENT_DECISION_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
EXPERIMENT_ACTIONABLE_THRESHOLDS = [0.0035, 0.0045, 0.0055, 0.0065, 0.0075]
EXPERIMENT_COST_BUFFER_PCTS = [DEFAULT_COST_BUFFER_PCT, 0.0045, 0.0055, 0.0065]
EXPERIMENT_RECALL_FLOOR = 0.15
