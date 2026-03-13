"""
Backtesting performance metrics for evaluating trading strategies.
"""
import numpy as np
import pandas as pd
from config import DEFAULT_RISK_FREE_RATE


def total_return(equity_curve: list[float]) -> float:
    """Calculate total return as a percentage."""
    if not equity_curve or equity_curve[0] == 0:
        return 0.0
    return ((equity_curve[-1] - equity_curve[0]) / equity_curve[0]) * 100


def annualized_return(equity_curve: list[float], periods_per_year: int = 8760) -> float:
    """
    Calculate annualized return.
    Default assumes hourly data (8760 hours/year).
    """
    if not equity_curve or len(equity_curve) < 2 or equity_curve[0] == 0:
        return 0.0
    total_ret = equity_curve[-1] / equity_curve[0]
    if total_ret <= 0:
        return -100.0
    n_periods = len(equity_curve) - 1
    try:
        return (total_ret ** (periods_per_year / n_periods) - 1) * 100
    except OverflowError:
        return float('inf') if total_ret > 1 else float('-inf')


def max_drawdown(equity_curve: list[float]) -> tuple[float, int]:
    """
    Calculate maximum drawdown percentage and its duration in periods.

    Returns:
        (max_drawdown_pct, duration_periods)
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0, 0

    peak = equity_curve[0]
    max_dd = 0.0
    max_dd_duration = 0
    current_dd_start = 0

    for i, value in enumerate(equity_curve):
        if value >= peak:
            peak = value
            current_dd_start = i
        dd = (peak - value) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_duration = i - current_dd_start

    return max_dd * 100, max_dd_duration


def sharpe_ratio(
    equity_curve: list[float],
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = 8760,
) -> float:
    """
    Calculate annualized Sharpe ratio.
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0

    returns = pd.Series(equity_curve).pct_change().dropna()
    if returns.std() == 0:
        return 0.0

    # Convert annual risk-free rate to per-period rate
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_per_period
    return float((excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year))


def sortino_ratio(
    equity_curve: list[float],
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = 8760,
) -> float:
    """
    Calculate annualized Sortino ratio (penalises downside volatility only).
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0

    returns = pd.Series(equity_curve).pct_change().dropna()
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_per_period
    downside = excess_returns[excess_returns < 0]

    if len(downside) == 0 or downside.std() == 0:
        return 0.0

    return float((excess_returns.mean() / downside.std()) * np.sqrt(periods_per_year))


def win_rate(trades: list[dict]) -> float:
    """Calculate percentage of profitable trades."""
    closed = [t for t in trades if t.get('pnl') is not None]
    if not closed:
        return 0.0
    winners = sum(1 for t in closed if t['pnl'] > 0)
    return (winners / len(closed)) * 100


def profit_factor(trades: list[dict]) -> float:
    """Calculate gross profit / gross loss."""
    closed = [t for t in trades if t.get('pnl') is not None]
    gross_profit = sum(t['pnl'] for t in closed if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in closed if t['pnl'] < 0))
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def average_win_loss(trades: list[dict]) -> tuple[float, float]:
    """Calculate average winning trade and average losing trade P&L."""
    closed = [t for t in trades if t.get('pnl') is not None]
    winners = [t['pnl'] for t in closed if t['pnl'] > 0]
    losers = [t['pnl'] for t in closed if t['pnl'] < 0]
    avg_win = np.mean(winners) if winners else 0.0
    avg_loss = np.mean(losers) if losers else 0.0
    return float(avg_win), float(avg_loss)


def calmar_ratio(
    equity_curve: list[float],
    periods_per_year: int = 8760,
) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown)."""
    ann_ret = annualized_return(equity_curve, periods_per_year)
    max_dd, _ = max_drawdown(equity_curve)
    if max_dd == 0:
        return 0.0
    return ann_ret / max_dd


def generate_report(
    equity_curve: list[float],
    trades: list[dict],
    buy_and_hold_curve: list[float] | None = None,
    periods_per_year: int = 8760,
) -> dict:
    """
    Generate a full performance report.

    Returns:
        Dictionary with all KPIs.
    """
    max_dd, max_dd_duration = max_drawdown(equity_curve)
    avg_win, avg_loss = average_win_loss(trades)
    closed_trades = [t for t in trades if t.get('pnl') is not None]

    report = {
        'total_return_pct': total_return(equity_curve),
        'annualized_return_pct': annualized_return(equity_curve, periods_per_year),
        'max_drawdown_pct': max_dd,
        'max_drawdown_duration': max_dd_duration,
        'sharpe_ratio': sharpe_ratio(equity_curve, periods_per_year=periods_per_year),
        'sortino_ratio': sortino_ratio(equity_curve, periods_per_year=periods_per_year),
        'win_rate_pct': win_rate(trades),
        'profit_factor': profit_factor(trades),
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_trades': len(closed_trades),
        'calmar_ratio': calmar_ratio(equity_curve, periods_per_year),
    }

    if buy_and_hold_curve:
        report['buy_and_hold_return_pct'] = total_return(buy_and_hold_curve)
        report['excess_return_pct'] = report['total_return_pct'] - report['buy_and_hold_return_pct']

    return report


def print_report(report: dict) -> None:
    """Print a formatted performance report."""
    print("\n" + "=" * 60)
    print("  BACKTEST PERFORMANCE REPORT")
    print("=" * 60)
    print(f"  Total Return:          {report['total_return_pct']:>10.2f}%")
    print(f"  Annualized Return:     {report['annualized_return_pct']:>10.2f}%")
    print(f"  Max Drawdown:          {report['max_drawdown_pct']:>10.2f}%")
    print(f"  Max DD Duration:       {report['max_drawdown_duration']:>10d} periods")
    print(f"  Sharpe Ratio:          {report['sharpe_ratio']:>10.3f}")
    print(f"  Sortino Ratio:         {report['sortino_ratio']:>10.3f}")
    print(f"  Calmar Ratio:          {report['calmar_ratio']:>10.3f}")
    print(f"  Win Rate:              {report['win_rate_pct']:>10.2f}%")
    print(f"  Profit Factor:         {report['profit_factor']:>10.3f}")
    print(f"  Avg Win:              ${report['avg_win']:>10.2f}")
    print(f"  Avg Loss:             ${report['avg_loss']:>10.2f}")
    print(f"  Total Trades:          {report['total_trades']:>10d}")

    if 'buy_and_hold_return_pct' in report:
        print(f"  Buy & Hold Return:     {report['buy_and_hold_return_pct']:>10.2f}%")
        print(f"  Excess Return:         {report['excess_return_pct']:>10.2f}%")

    print("=" * 60)
