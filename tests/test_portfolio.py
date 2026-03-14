"""
Unit tests for the cost simulator (evaluation/cost_model.py).
"""
import unittest
from evaluation.cost_model import CostSimulator


class TestCostSimulatorInit(unittest.TestCase):
    def test_default_values(self):
        p = CostSimulator()
        self.assertEqual(p.cash, 10_000.0)
        self.assertEqual(p.btc_holdings, 0.0)
        self.assertEqual(p.initial_capital, 10_000.0)
        self.assertEqual(len(p.trades), 0)
        self.assertEqual(len(p.equity_curve), 1)

    def test_custom_capital(self):
        p = CostSimulator(initial_capital=50_000.0)
        self.assertEqual(p.cash, 50_000.0)

    def test_custom_fees(self):
        p = CostSimulator(fee_pct=0.005, slippage_pct=0.001)
        self.assertEqual(p.fee_pct, 0.005)
        self.assertEqual(p.slippage_pct, 0.001)


class TestCostSimulatorBuy(unittest.TestCase):
    def test_basic_buy(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        result = p.execute_buy(50_000, 1_000)
        self.assertTrue(result)
        self.assertEqual(p.cash, 9_000)
        self.assertAlmostEqual(p.btc_holdings, 1_000 / 50_000, places=8)

    def test_buy_with_fees(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0.001, slippage_pct=0)
        p.execute_buy(50_000, 1_000)
        # Fee = 1000 * 0.001 = 1.0, net = 999, BTC = 999/50000
        self.assertAlmostEqual(p.cash, 9_000, places=2)
        self.assertAlmostEqual(p.btc_holdings, 999 / 50_000, places=8)

    def test_buy_with_slippage(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0.001)
        p.execute_buy(50_000, 1_000)
        # Effective price = 50000 * 1.001 = 50050
        # BTC = 1000 / 50050
        expected_btc = 1_000 / 50_050
        self.assertAlmostEqual(p.btc_holdings, expected_btc, places=8)

    def test_buy_insufficient_funds(self):
        p = CostSimulator(initial_capital=100)
        result = p.execute_buy(50_000, 200)
        self.assertFalse(result)
        self.assertEqual(p.cash, 100)
        self.assertEqual(p.btc_holdings, 0)

    def test_buy_records_trade(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        p.execute_buy(50_000, 500, timestamp="2024-01-01")
        self.assertEqual(len(p.trades), 1)
        self.assertEqual(p.trades[0]['type'], 'ENTRY')
        self.assertEqual(p.trades[0]['timestamp'], "2024-01-01")
        self.assertIsNone(p.trades[0]['pnl'])  # P&L calculated on sell

    def test_buy_all_cash(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        result = p.execute_buy(50_000, 10_000)
        self.assertTrue(result)
        self.assertAlmostEqual(p.cash, 0, places=2)


class TestCostSimulatorSell(unittest.TestCase):
    def test_basic_sell(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        p.execute_buy(50_000, 5_000)
        btc = p.btc_holdings
        result = p.execute_sell(55_000)  # Sell all at higher price
        self.assertTrue(result)
        self.assertAlmostEqual(p.btc_holdings, 0, places=8)
        self.assertGreater(p.cash, 5_000)  # Should have initial cash + profit

    def test_sell_with_profit(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        p.execute_buy(50_000, 5_000)  # Buy 0.1 BTC
        p.execute_sell(60_000)  # Sell at 60k
        # P&L = 0.1 * 60000 - 5000 = 1000
        last_trade = p.trades[-1]
        self.assertAlmostEqual(last_trade['pnl'], 1_000, places=2)

    def test_sell_with_loss(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        p.execute_buy(50_000, 5_000)  # Buy 0.1 BTC
        p.execute_sell(40_000)  # Sell at 40k
        # P&L = 0.1 * 40000 - 5000 = -1000
        last_trade = p.trades[-1]
        self.assertAlmostEqual(last_trade['pnl'], -1_000, places=2)

    def test_sell_no_holdings(self):
        p = CostSimulator(initial_capital=10_000)
        result = p.execute_sell(50_000)
        self.assertFalse(result)

    def test_sell_partial(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        p.execute_buy(50_000, 5_000)  # Buy 0.1 BTC
        btc = p.btc_holdings
        result = p.execute_sell(50_000, btc_amount=btc / 2)
        self.assertTrue(result)
        self.assertAlmostEqual(p.btc_holdings, btc / 2, places=8)

    def test_sell_with_fees(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0.001, slippage_pct=0)
        p.execute_buy(50_000, 5_000)
        p.execute_sell(50_000)
        # Should have less than initial due to fees on both sides
        self.assertLess(p.cash, 10_000)

    def test_sell_records_trade(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        p.execute_buy(50_000, 5_000)
        p.execute_sell(50_000, timestamp="2024-01-02")
        self.assertEqual(len(p.trades), 2)
        self.assertEqual(p.trades[1]['type'], 'EXIT')
        self.assertIsNotNone(p.trades[1]['pnl'])


class TestCostSimulatorHelpers(unittest.TestCase):
    def test_portfolio_value(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        p.execute_buy(50_000, 5_000)
        # 5000 cash + 0.1 BTC * 50000 = 10000
        self.assertAlmostEqual(p.get_portfolio_value(50_000), 10_000, places=2)

    def test_portfolio_value_price_increase(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        p.execute_buy(50_000, 5_000)
        # 5000 cash + 0.1 BTC * 60000 = 11000
        self.assertAlmostEqual(p.get_portfolio_value(60_000), 11_000, places=2)

    def test_has_position(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        self.assertFalse(p.has_position())
        p.execute_buy(50_000, 5_000)
        self.assertTrue(p.has_position())
        p.execute_sell(50_000)
        self.assertFalse(p.has_position())

    def test_record_equity(self):
        p = CostSimulator(initial_capital=10_000)
        self.assertEqual(len(p.equity_curve), 1)
        p.record_equity(50_000)
        self.assertEqual(len(p.equity_curve), 2)

    def test_unrealized_pnl(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        p.execute_buy(50_000, 5_000)
        # At 60k: unrealized = 0.1 * 60000 - 5000 = 1000
        self.assertAlmostEqual(p.get_unrealized_pnl(60_000), 1_000, places=2)

    def test_unrealized_pnl_no_position(self):
        p = CostSimulator(initial_capital=10_000)
        self.assertEqual(p.get_unrealized_pnl(50_000), 0.0)

    def test_summary(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        p.execute_buy(50_000, 5_000)
        summary = p.summary(50_000)
        self.assertIn('cash', summary)
        self.assertIn('btc_holdings', summary)
        self.assertIn('portfolio_value', summary)
        self.assertIn('total_trades', summary)
        self.assertIn('pnl', summary)
        self.assertEqual(summary['total_trades'], 1)


class TestCostSimulatorRoundTrips(unittest.TestCase):
    """Test complete buy-sell cycles."""

    def test_round_trip_no_fees_no_slippage(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        p.execute_buy(50_000, 5_000)
        p.execute_sell(50_000)
        # Should be exactly back to initial
        self.assertAlmostEqual(p.cash, 10_000, places=2)

    def test_round_trip_with_fees_loses_money(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0.001, slippage_pct=0)
        p.execute_buy(50_000, 5_000)
        p.execute_sell(50_000)
        # Should have lost money to fees
        self.assertLess(p.cash, 10_000)

    def test_multiple_round_trips(self):
        p = CostSimulator(initial_capital=10_000, fee_pct=0, slippage_pct=0)
        for _ in range(5):
            p.execute_buy(50_000, 1_000)
            p.execute_sell(55_000)
        # Should have made profit each time
        self.assertGreater(p.cash, 10_000)
        self.assertEqual(len(p.trades), 10)


if __name__ == '__main__':
    unittest.main()
