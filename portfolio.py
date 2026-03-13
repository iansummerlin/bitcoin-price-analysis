"""Portfolio accounting used only for signal-usefulness evaluation."""
from config import DEFAULT_INITIAL_CAPITAL, DEFAULT_FEE_PCT, DEFAULT_SLIPPAGE_PCT


class Portfolio:
    """
    Tracks cash, BTC holdings, trades, and equity over time.
    """

    def __init__(
        self,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        fee_pct: float = DEFAULT_FEE_PCT,
        slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.btc_holdings = 0.0
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.trades: list[dict] = []
        self.equity_curve: list[float] = [initial_capital]
        self._open_position: dict | None = None

    def get_portfolio_value(self, current_price: float) -> float:
        """Total portfolio value in USD."""
        return self.cash + self.btc_holdings * current_price

    def execute_buy(self, price: float, amount_usd: float, timestamp=None) -> bool:
        """
        Buy BTC with the given USD amount.
        Applies slippage and fees.

        Returns:
            True if the buy was executed, False if insufficient funds.
        """
        if amount_usd > self.cash:
            return False

        # Apply slippage (price moves against us)
        effective_price = price * (1 + self.slippage_pct)
        fee = amount_usd * self.fee_pct
        net_usd = amount_usd - fee
        btc_bought = net_usd / effective_price

        self.cash -= amount_usd
        self.btc_holdings += btc_bought

        self._open_position = {
            'entry_price': effective_price,
            'entry_time': timestamp,
            'btc_amount': btc_bought,
            'cost_basis': amount_usd,
        }

        self.trades.append({
            'type': 'BUY',
            'price': effective_price,
            'btc_amount': btc_bought,
            'usd_amount': amount_usd,
            'fee': fee,
            'timestamp': timestamp,
            'pnl': None,  # P&L calculated on sell
        })

        return True

    def execute_sell(self, price: float, btc_amount: float | None = None, timestamp=None) -> bool:
        """
        Sell BTC at the given price.
        If btc_amount is None, sells entire position.
        Applies slippage and fees.

        Returns:
            True if the sell was executed, False if insufficient holdings.
        """
        if btc_amount is None:
            btc_amount = self.btc_holdings

        if btc_amount <= 0 or btc_amount > self.btc_holdings:
            return False

        # Apply slippage (price moves against us)
        effective_price = price * (1 - self.slippage_pct)
        gross_usd = btc_amount * effective_price
        fee = gross_usd * self.fee_pct
        net_usd = gross_usd - fee

        self.btc_holdings -= btc_amount
        self.cash += net_usd

        # Calculate P&L if we have an open position
        pnl = None
        if self._open_position:
            pnl = net_usd - self._open_position['cost_basis']
            self._open_position = None

        self.trades.append({
            'type': 'SELL',
            'price': effective_price,
            'btc_amount': btc_amount,
            'usd_amount': net_usd,
            'fee': fee,
            'timestamp': timestamp,
            'pnl': pnl,
        })

        return True

    def record_equity(self, current_price: float) -> None:
        """Record current portfolio value to the equity curve."""
        self.equity_curve.append(self.get_portfolio_value(current_price))

    def has_position(self) -> bool:
        """Check if we have an open BTC position."""
        return self.btc_holdings > 0

    def get_position_value(self, current_price: float) -> float:
        """Get the value of current BTC holdings."""
        return self.btc_holdings * current_price

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Get unrealized P&L on current position."""
        if not self._open_position:
            return 0.0
        current_value = self.btc_holdings * current_price
        return current_value - self._open_position['cost_basis']

    def summary(self, current_price: float) -> dict:
        """Get portfolio summary."""
        return {
            'cash': self.cash,
            'btc_holdings': self.btc_holdings,
            'portfolio_value': self.get_portfolio_value(current_price),
            'total_trades': len(self.trades),
            'pnl': self.get_portfolio_value(current_price) - self.initial_capital,
        }
