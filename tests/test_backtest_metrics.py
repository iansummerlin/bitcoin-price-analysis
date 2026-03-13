"""
Unit tests for backtesting metrics.
"""
import unittest
import math
from backtest_metrics import (
    total_return,
    annualized_return,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
    profit_factor,
    average_win_loss,
    calmar_ratio,
    generate_report,
)


class TestTotalReturn(unittest.TestCase):
    def test_positive_return(self):
        curve = [100, 110, 120, 130]
        self.assertAlmostEqual(total_return(curve), 30.0, places=1)

    def test_negative_return(self):
        curve = [100, 90, 80, 70]
        self.assertAlmostEqual(total_return(curve), -30.0, places=1)

    def test_flat_return(self):
        curve = [100, 100, 100, 100]
        self.assertAlmostEqual(total_return(curve), 0.0)

    def test_empty_curve(self):
        self.assertEqual(total_return([]), 0.0)

    def test_single_value(self):
        self.assertEqual(total_return([100]), 0.0)

    def test_zero_start(self):
        self.assertEqual(total_return([0, 100]), 0.0)


class TestAnnualizedReturn(unittest.TestCase):
    def test_positive_annual(self):
        # Doubling in 8760 periods (1 year of hourly)
        curve = [100, 200]
        # With only 1 period that results in 100% return,
        # annualized over 8760 periods will be astronomically high
        result = annualized_return(curve, periods_per_year=1)
        self.assertAlmostEqual(result, 100.0, places=1)

    def test_empty_curve(self):
        self.assertEqual(annualized_return([]), 0.0)


class TestMaxDrawdown(unittest.TestCase):
    def test_no_drawdown(self):
        curve = [100, 110, 120, 130]
        dd, duration = max_drawdown(curve)
        self.assertAlmostEqual(dd, 0.0)

    def test_simple_drawdown(self):
        curve = [100, 120, 90, 110]
        dd, duration = max_drawdown(curve)
        # Max DD from 120 to 90 = 25%
        self.assertAlmostEqual(dd, 25.0, places=1)

    def test_complete_loss(self):
        curve = [100, 50, 0.01]
        dd, _ = max_drawdown(curve)
        self.assertGreater(dd, 99)

    def test_empty_curve(self):
        dd, duration = max_drawdown([])
        self.assertEqual(dd, 0.0)
        self.assertEqual(duration, 0)

    def test_single_value(self):
        dd, duration = max_drawdown([100])
        self.assertEqual(dd, 0.0)


class TestSharpeRatio(unittest.TestCase):
    def test_positive_sharpe(self):
        # Steadily increasing equity
        curve = list(range(100, 200))
        result = sharpe_ratio(curve, risk_free_rate=0)
        self.assertGreater(result, 0)

    def test_negative_sharpe(self):
        # Steadily decreasing equity
        curve = list(range(200, 100, -1))
        result = sharpe_ratio(curve, risk_free_rate=0.04)
        self.assertLess(result, 0)

    def test_flat_equity(self):
        curve = [100.0] * 50
        result = sharpe_ratio(curve)
        self.assertEqual(result, 0.0)

    def test_empty_curve(self):
        self.assertEqual(sharpe_ratio([]), 0.0)


class TestSortinoRatio(unittest.TestCase):
    def test_all_positive_returns(self):
        curve = list(range(100, 200))
        result = sortino_ratio(curve, risk_free_rate=0)
        # No downside, but there might be small returns below rf
        self.assertGreaterEqual(result, 0)

    def test_empty_curve(self):
        self.assertEqual(sortino_ratio([]), 0.0)


class TestWinRate(unittest.TestCase):
    def test_all_winners(self):
        trades = [{'pnl': 100}, {'pnl': 50}, {'pnl': 200}]
        self.assertAlmostEqual(win_rate(trades), 100.0)

    def test_all_losers(self):
        trades = [{'pnl': -100}, {'pnl': -50}]
        self.assertAlmostEqual(win_rate(trades), 0.0)

    def test_mixed(self):
        trades = [{'pnl': 100}, {'pnl': -50}, {'pnl': 200}, {'pnl': -30}]
        self.assertAlmostEqual(win_rate(trades), 50.0)

    def test_empty_trades(self):
        self.assertAlmostEqual(win_rate([]), 0.0)

    def test_trades_without_pnl(self):
        trades = [{'pnl': None}, {'pnl': 100}]
        self.assertAlmostEqual(win_rate(trades), 100.0)


class TestProfitFactor(unittest.TestCase):
    def test_basic(self):
        trades = [{'pnl': 200}, {'pnl': -100}]
        self.assertAlmostEqual(profit_factor(trades), 2.0)

    def test_no_losses(self):
        trades = [{'pnl': 100}, {'pnl': 200}]
        self.assertEqual(profit_factor(trades), float('inf'))

    def test_no_wins(self):
        trades = [{'pnl': -100}]
        self.assertAlmostEqual(profit_factor(trades), 0.0)

    def test_empty(self):
        self.assertAlmostEqual(profit_factor([]), 0.0)


class TestAverageWinLoss(unittest.TestCase):
    def test_basic(self):
        trades = [{'pnl': 100}, {'pnl': 200}, {'pnl': -50}, {'pnl': -150}]
        avg_win, avg_loss = average_win_loss(trades)
        self.assertAlmostEqual(avg_win, 150.0)
        self.assertAlmostEqual(avg_loss, -100.0)

    def test_no_trades(self):
        avg_win, avg_loss = average_win_loss([])
        self.assertEqual(avg_win, 0.0)
        self.assertEqual(avg_loss, 0.0)


class TestCalmarRatio(unittest.TestCase):
    def test_basic(self):
        # Simple increasing curve with a dip
        curve = [100, 110, 90, 105, 115]
        result = calmar_ratio(curve, periods_per_year=4)
        # Should be positive (positive return, has drawdown)
        self.assertGreater(result, 0)

    def test_no_drawdown(self):
        curve = [100, 110, 120]
        result = calmar_ratio(curve, periods_per_year=2)
        self.assertEqual(result, 0.0)  # 0 drawdown -> 0 calmar


class TestGenerateReport(unittest.TestCase):
    def test_report_keys(self):
        equity = [100, 110, 105, 115, 120]
        trades = [{'pnl': 10}, {'pnl': -5}, {'pnl': 15}]
        report = generate_report(equity, trades)
        expected_keys = [
            'total_return_pct', 'annualized_return_pct', 'max_drawdown_pct',
            'max_drawdown_duration', 'sharpe_ratio', 'sortino_ratio',
            'win_rate_pct', 'profit_factor', 'avg_win', 'avg_loss',
            'total_trades', 'calmar_ratio',
        ]
        for key in expected_keys:
            self.assertIn(key, report, f"Missing key: {key}")

    def test_report_with_buy_and_hold(self):
        equity = [100, 110, 120]
        bh = [100, 105, 108]
        trades = [{'pnl': 20}]
        report = generate_report(equity, trades, buy_and_hold_curve=bh)
        self.assertIn('buy_and_hold_return_pct', report)
        self.assertIn('excess_return_pct', report)

    def test_report_empty(self):
        report = generate_report([100], [], None)
        self.assertEqual(report['total_trades'], 0)


if __name__ == '__main__':
    unittest.main()
