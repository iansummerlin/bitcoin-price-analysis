"""Signal-consumption rules used to evaluate downstream strategy assumptions."""
from enum import Enum
from config import (
    DEFAULT_BUY_THRESHOLD,
    DEFAULT_SELL_THRESHOLD,
    DEFAULT_MAX_RISK_PER_TRADE,
    DEFAULT_MAX_DRAWDOWN_PCT,
)


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Strategy:
    """Base interface for consuming prediction streams."""

    def on_candle(self, row, portfolio) -> Signal:
        raise NotImplementedError

    def get_position_size_usd(self, row, portfolio) -> float:
        """How much USD to use for a buy order. Override for custom sizing."""
        return min(100.0, portfolio.cash)


class ThresholdStrategy(Strategy):
    """
    Simple threshold strategy used as a baseline signal consumer.
    Buys if predicted price > current * buy_threshold.
    Sells if predicted price < current * sell_threshold.
    Fixed position sizing.
    """

    def __init__(
        self,
        buy_threshold: float = DEFAULT_BUY_THRESHOLD,
        sell_threshold: float = DEFAULT_SELL_THRESHOLD,
    ):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def on_candle(self, row, portfolio) -> Signal:
        predicted = row.get('predicted_price')
        current = row['close']

        if predicted is None:
            return Signal.HOLD

        if predicted > current * self.buy_threshold:
            return Signal.BUY
        elif predicted < current * self.sell_threshold:
            return Signal.SELL
        return Signal.HOLD


class ATRThresholdStrategy(Strategy):
    """
    Dynamic threshold strategy using ATR for volatility-adaptive thresholds.
    Thresholds widen in volatile markets, tighten in quiet markets.
    """

    def __init__(self, atr_multiplier: float = 0.5):
        self.atr_multiplier = atr_multiplier

    def on_candle(self, row, portfolio) -> Signal:
        predicted = row.get('predicted_price')
        current = row['close']
        atr = row.get('atr_24', 0)

        if predicted is None or atr == 0 or current == 0:
            return Signal.HOLD

        dynamic_threshold = self.atr_multiplier * atr / current
        buy_threshold = 1 + dynamic_threshold
        sell_threshold = 1 - dynamic_threshold

        if predicted > current * buy_threshold:
            return Signal.BUY
        elif predicted < current * sell_threshold:
            return Signal.SELL
        return Signal.HOLD

    def get_position_size_usd(self, row, portfolio) -> float:
        # Inverse volatility sizing: trade smaller when volatile
        atr = row.get('atr_24', 0)
        current = row['close']
        if atr == 0 or current == 0:
            return min(100.0, portfolio.cash)
        volatility_ratio = atr / current
        # Base size of 5% of portfolio, scaled down by volatility
        base_size = portfolio.get_portfolio_value(current) * 0.05
        size = base_size / (1 + volatility_ratio * 100)
        return min(size, portfolio.cash)


class MultiFactorStrategy(Strategy):
    """
    Multi-factor confirmation strategy.
    Requires multiple signals to align before trading:
    - Model prediction direction
    - RSI not overbought/oversold
    - MA crossover trend confirmation
    - Fear & Greed sentiment filter
    """

    def __init__(
        self,
        atr_multiplier: float = 0.5,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        fng_extreme_greed: float = 80.0,
        fng_extreme_fear: float = 20.0,
        max_risk_per_trade: float = DEFAULT_MAX_RISK_PER_TRADE,
        max_drawdown_pct: float = DEFAULT_MAX_DRAWDOWN_PCT,
    ):
        self.atr_multiplier = atr_multiplier
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.fng_extreme_greed = fng_extreme_greed
        self.fng_extreme_fear = fng_extreme_fear
        self.max_risk_per_trade = max_risk_per_trade
        self.max_drawdown_pct = max_drawdown_pct

    def _check_drawdown_breaker(self, portfolio) -> bool:
        """Returns True if drawdown circuit breaker is triggered."""
        if len(portfolio.equity_curve) < 2:
            return False
        peak = max(portfolio.equity_curve)
        current = portfolio.equity_curve[-1]
        if peak == 0:
            return False
        drawdown = (peak - current) / peak
        return drawdown >= self.max_drawdown_pct

    def on_candle(self, row, portfolio) -> Signal:
        # Circuit breaker: stop trading if drawdown too large
        if self._check_drawdown_breaker(portfolio):
            # Still allow sells to reduce exposure
            if portfolio.has_position():
                return Signal.SELL
            return Signal.HOLD

        predicted = row.get('predicted_price')
        current = row['close']
        rsi = row.get('rsi', 50)
        ma_7 = row.get('MA_7', current)
        ma_24 = row.get('MA_24', current)
        atr = row.get('atr_24', 0)
        fng = row.get('fng_value', 50)

        if predicted is None or current == 0:
            return Signal.HOLD

        # Dynamic threshold from ATR
        if atr > 0:
            dynamic_threshold = self.atr_multiplier * atr / current
        else:
            dynamic_threshold = 0.001  # Fallback to 0.1%

        # --- BUY conditions ---
        price_signal_buy = predicted > current * (1 + dynamic_threshold)
        rsi_ok_buy = rsi < self.rsi_overbought  # Not overbought
        trend_ok_buy = ma_7 > ma_24  # Uptrend
        fng_ok_buy = fng < self.fng_extreme_greed  # Not extreme greed

        if price_signal_buy and rsi_ok_buy and trend_ok_buy and fng_ok_buy:
            if not portfolio.has_position():
                return Signal.BUY

        # --- SELL conditions ---
        price_signal_sell = predicted < current * (1 - dynamic_threshold)
        rsi_ok_sell = rsi > self.rsi_oversold  # Not oversold
        trend_ok_sell = ma_7 < ma_24  # Downtrend
        fng_ok_sell = fng > self.fng_extreme_fear  # Not extreme fear

        if price_signal_sell and rsi_ok_sell and trend_ok_sell and fng_ok_sell:
            if portfolio.has_position():
                return Signal.SELL

        # --- Stop-loss check ---
        if portfolio.has_position() and portfolio._open_position:
            entry = portfolio._open_position['entry_price']
            stop_loss = entry * (1 - self.max_risk_per_trade * 5)  # 10% stop-loss
            if current <= stop_loss:
                return Signal.SELL

        # --- Take-profit check ---
        if portfolio.has_position() and portfolio._open_position:
            entry = portfolio._open_position['entry_price']
            take_profit = entry * (1 + self.max_risk_per_trade * 7.5)  # 15% take-profit
            if current >= take_profit:
                return Signal.SELL

        return Signal.HOLD

    def get_position_size_usd(self, row, portfolio) -> float:
        """Risk-based position sizing."""
        current = row['close']
        atr = row.get('atr_24', 0)

        portfolio_value = portfolio.get_portfolio_value(current)
        risk_amount = portfolio_value * self.max_risk_per_trade

        if atr > 0 and current > 0:
            # Position size = risk / (ATR as % of price)
            atr_pct = atr / current
            if atr_pct > 0:
                size = risk_amount / atr_pct
            else:
                size = risk_amount
        else:
            size = risk_amount

        # Cap at available cash and 20% of portfolio
        max_size = portfolio_value * 0.20
        return min(size, max_size, portfolio.cash)
