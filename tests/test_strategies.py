"""
Unit tests for signal rules (evaluation/signal_rules.py).
"""
import unittest
from evaluation.signal_rules import (
    Signal,
    ThresholdRule,
    ATRThresholdRule,
    MultiFactorRule,
)
from evaluation.cost_model import CostSimulator


def _make_row(**overrides):
    """Create a mock data row for strategy testing."""
    defaults = {
        'close': 50_000,
        'predicted_price': 50_000,
        'atr_24': 500,
        'rsi': 50,
        'MA_7': 50_000,
        'MA_24': 49_500,
        'fng_value': 50,
    }
    defaults.update(overrides)
    return defaults


class TestThresholdRule(unittest.TestCase):
    def test_buy_signal(self):
        s = ThresholdRule(buy_threshold=1.001, sell_threshold=0.999)
        row = _make_row(close=50_000, predicted_price=50_100)
        signal = s.on_candle(row, CostSimulator())
        self.assertEqual(signal, Signal.BUY)

    def test_sell_signal(self):
        s = ThresholdRule(buy_threshold=1.001, sell_threshold=0.999)
        row = _make_row(close=50_000, predicted_price=49_900)
        signal = s.on_candle(row, CostSimulator())
        self.assertEqual(signal, Signal.SELL)

    def test_hold_signal(self):
        s = ThresholdRule(buy_threshold=1.001, sell_threshold=0.999)
        row = _make_row(close=50_000, predicted_price=50_020)
        signal = s.on_candle(row, CostSimulator())
        self.assertEqual(signal, Signal.HOLD)

    def test_no_prediction(self):
        s = ThresholdRule()
        row = _make_row(predicted_price=None)
        signal = s.on_candle(row, CostSimulator())
        self.assertEqual(signal, Signal.HOLD)

    def test_exact_threshold_is_hold(self):
        s = ThresholdRule(buy_threshold=1.001, sell_threshold=0.999)
        # predicted = 50050 = close * 1.001, not strictly greater -> HOLD
        row = _make_row(close=50_000, predicted_price=50_049.99)
        signal = s.on_candle(row, CostSimulator())
        self.assertEqual(signal, Signal.HOLD)


class TestATRThresholdRule(unittest.TestCase):
    def test_buy_with_low_volatility(self):
        s = ATRThresholdRule(atr_multiplier=0.5)
        # ATR = 100, threshold = 0.5 * 100/50000 = 0.001, need > 50050
        row = _make_row(close=50_000, atr_24=100, predicted_price=50_100)
        signal = s.on_candle(row, CostSimulator())
        self.assertEqual(signal, Signal.BUY)

    def test_hold_with_high_volatility(self):
        s = ATRThresholdRule(atr_multiplier=0.5)
        # ATR = 2000, threshold = 0.5 * 2000/50000 = 0.02, need > 51000
        row = _make_row(close=50_000, atr_24=2000, predicted_price=50_500)
        signal = s.on_candle(row, CostSimulator())
        self.assertEqual(signal, Signal.HOLD)

    def test_sell_signal(self):
        s = ATRThresholdRule(atr_multiplier=0.5)
        row = _make_row(close=50_000, atr_24=100, predicted_price=49_900)
        signal = s.on_candle(row, CostSimulator())
        self.assertEqual(signal, Signal.SELL)

    def test_zero_atr_holds(self):
        s = ATRThresholdRule()
        row = _make_row(close=50_000, atr_24=0, predicted_price=50_100)
        signal = s.on_candle(row, CostSimulator())
        self.assertEqual(signal, Signal.HOLD)

    def test_position_sizing_inverse_volatility(self):
        s = ATRThresholdRule(atr_multiplier=0.5)
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)

        low_vol_row = _make_row(close=50_000, atr_24=100)
        high_vol_row = _make_row(close=50_000, atr_24=2000)

        low_vol_size = s.get_position_size_usd(low_vol_row, p)
        high_vol_size = s.get_position_size_usd(high_vol_row, p)

        self.assertGreater(low_vol_size, high_vol_size)


class TestMultiFactorRule(unittest.TestCase):
    def test_buy_all_conditions_met(self):
        s = MultiFactorRule(atr_multiplier=0.5)
        row = _make_row(
            close=50_000,
            predicted_price=50_200,
            atr_24=100,
            rsi=45,           # Not overbought
            MA_7=50_100,      # Above MA_24 (uptrend)
            MA_24=49_900,
            fng_value=50,     # Not extreme greed
        )
        p = CostSimulator(initial_capital=10_000)
        signal = s.on_candle(row, p)
        self.assertEqual(signal, Signal.BUY)

    def test_buy_blocked_by_rsi_overbought(self):
        s = MultiFactorRule()
        row = _make_row(
            close=50_000,
            predicted_price=50_200,
            atr_24=100,
            rsi=75,           # Overbought!
            MA_7=50_100,
            MA_24=49_900,
            fng_value=50,
        )
        p = CostSimulator(initial_capital=10_000)
        signal = s.on_candle(row, p)
        self.assertEqual(signal, Signal.HOLD)

    def test_buy_blocked_by_downtrend(self):
        s = MultiFactorRule()
        row = _make_row(
            close=50_000,
            predicted_price=50_200,
            atr_24=100,
            rsi=45,
            MA_7=49_800,      # Below MA_24 (downtrend)
            MA_24=50_100,
            fng_value=50,
        )
        p = CostSimulator(initial_capital=10_000)
        signal = s.on_candle(row, p)
        self.assertEqual(signal, Signal.HOLD)

    def test_buy_blocked_by_extreme_greed(self):
        s = MultiFactorRule()
        row = _make_row(
            close=50_000,
            predicted_price=50_200,
            atr_24=100,
            rsi=45,
            MA_7=50_100,
            MA_24=49_900,
            fng_value=85,     # Extreme greed!
        )
        p = CostSimulator(initial_capital=10_000)
        signal = s.on_candle(row, p)
        self.assertEqual(signal, Signal.HOLD)

    def test_sell_all_conditions_met(self):
        s = MultiFactorRule(atr_multiplier=0.5)
        row = _make_row(
            close=50_000,
            predicted_price=49_800,
            atr_24=100,
            rsi=55,           # Not oversold
            MA_7=49_800,      # Below MA_24 (downtrend)
            MA_24=50_100,
            fng_value=50,     # Not extreme fear
        )
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        p.execute_buy(50_000, 5_000)  # Need a position to sell
        signal = s.on_candle(row, p)
        self.assertEqual(signal, Signal.SELL)

    def test_sell_blocked_by_rsi_oversold(self):
        s = MultiFactorRule()
        row = _make_row(
            close=50_000,
            predicted_price=49_800,
            atr_24=100,
            rsi=25,           # Oversold!
            MA_7=49_800,
            MA_24=50_100,
            fng_value=50,
        )
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        p.execute_buy(50_000, 5_000)
        signal = s.on_candle(row, p)
        # RSI says oversold, so sell is blocked by rsi_ok_sell condition
        self.assertEqual(signal, Signal.HOLD)

    def test_drawdown_circuit_breaker(self):
        s = MultiFactorRule(max_drawdown_pct=0.10)
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        # Simulate equity dropping 15%
        p.equity_curve = [10_000, 9_500, 9_000, 8_500]
        row = _make_row(close=50_000, predicted_price=50_200)
        signal = s.on_candle(row, p)
        self.assertEqual(signal, Signal.HOLD)

    def test_drawdown_breaker_allows_sell(self):
        s = MultiFactorRule(max_drawdown_pct=0.10)
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        p.execute_buy(50_000, 5_000)
        p.equity_curve = [10_000, 9_500, 9_000, 8_500]  # >10% drawdown
        row = _make_row(close=50_000, predicted_price=50_200)
        signal = s.on_candle(row, p)
        # Should allow sell to reduce exposure even during circuit breaker
        self.assertEqual(signal, Signal.SELL)

    def test_stop_loss(self):
        s = MultiFactorRule(max_risk_per_trade=0.02)
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        p.execute_buy(50_000, 5_000)
        # Stop loss = entry * (1 - 0.02 * 5) = entry * 0.9 = 45000
        row = _make_row(close=44_000, predicted_price=44_100)  # Below stop loss
        signal = s.on_candle(row, p)
        self.assertEqual(signal, Signal.SELL)

    def test_take_profit(self):
        s = MultiFactorRule(max_risk_per_trade=0.02)
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        p.execute_buy(50_000, 5_000)
        # Take profit = entry * (1 + 0.02 * 7.5) = entry * 1.15 = 57500
        row = _make_row(close=58_000, predicted_price=57_500)
        signal = s.on_candle(row, p)
        self.assertEqual(signal, Signal.SELL)

    def test_position_sizing_risk_based(self):
        s = MultiFactorRule(max_risk_per_trade=0.02)
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        row = _make_row(close=50_000, atr_24=500)
        size = s.get_position_size_usd(row, p)
        # Should be reasonable fraction of portfolio
        self.assertGreater(size, 0)
        self.assertLessEqual(size, p.cash)

    def test_no_prediction_holds(self):
        s = MultiFactorRule()
        row = _make_row(predicted_price=None)
        signal = s.on_candle(row, CostSimulator())
        self.assertEqual(signal, Signal.HOLD)


class TestSignalEnum(unittest.TestCase):
    def test_signal_values(self):
        self.assertEqual(Signal.BUY.value, "BUY")
        self.assertEqual(Signal.SELL.value, "SELL")
        self.assertEqual(Signal.HOLD.value, "HOLD")


if __name__ == '__main__':
    unittest.main()
