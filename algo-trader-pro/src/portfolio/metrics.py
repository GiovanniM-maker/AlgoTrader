"""
src/portfolio/metrics.py

Portfolio Performance Metrics.

All metric functions are pure and stateless: they accept pandas Series /
lists and return floats or dicts.  No side effects, no DB access.

Metric definitions
------------------
Sharpe Ratio  : annualised excess return per unit of total volatility.
Sortino Ratio : annualised excess return per unit of downside volatility.
Max Drawdown  : largest peak-to-trough decline as a percentage.
Calmar Ratio  : CAGR / max drawdown (risk-adjusted return over deep losses).
Profit Factor : gross profit / gross loss over all trades.
Win Rate      : fraction of trades with positive net P&L.
Avg Win / Loss: average P&L percentage for winning / losing trades.

Annualisation
-------------
All ratio calculations default to 8,760 periods per year, matching an
hourly trading cadence (24 h × 365 days).  Pass a different value for
other timeframes:
    * 4h bars : periods_per_year = 2,190
    * Daily   : periods_per_year = 365
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Risk-adjusted return ratios
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 8_760,
) -> float:
    """
    Compute the annualised Sharpe Ratio.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns (not cumulative).  Must be non-empty.
        Values should be expressed as decimals (e.g. 0.01 for 1%).
    risk_free_rate : float
        Annual risk-free rate (default 2%).  Converted to per-period
        internally.
    periods_per_year : int
        Number of return observations per year (default 8,760 for hourly).

    Returns
    -------
    float
        Annualised Sharpe Ratio.  Returns 0.0 if returns are constant
        (zero standard deviation).
    """
    if returns is None or len(returns) == 0:
        return 0.0

    returns = returns.dropna().astype(float)
    if len(returns) == 0:
        return 0.0

    # Per-period risk-free rate
    rf_per_period: float = risk_free_rate / periods_per_year
    excess: pd.Series = returns - rf_per_period

    std: float = float(excess.std(ddof=1))
    if std == 0.0 or math.isnan(std) or std <= 1e-12:
        return 0.0

    raw_sharpe = float((excess.mean() / std) * math.sqrt(periods_per_year))
    # Clamp extreme values (e.g. from near-zero std with few data points)
    if math.isnan(raw_sharpe) or math.isinf(raw_sharpe) or abs(raw_sharpe) > 100:
        return 0.0
    return raw_sharpe


def sortino_ratio(
    returns: pd.Series,
    target: float = 0.0,
    periods_per_year: int = 8_760,
) -> float:
    """
    Compute the annualised Sortino Ratio.

    Unlike Sharpe, only downside deviations (returns below *target*) are
    penalised.  This avoids punishing upside volatility.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns as decimals.
    target : float
        Minimum acceptable return per period (default 0.0).
    periods_per_year : int
        Number of periods per year (default 8,760 for hourly).

    Returns
    -------
    float
        Annualised Sortino Ratio.  Returns 0.0 if downside deviation is zero.
    """
    if returns is None or len(returns) == 0:
        return 0.0

    returns = returns.dropna().astype(float)
    if len(returns) == 0:
        return 0.0

    # Downside deviations: only returns below the target threshold.
    downside: pd.Series = returns[returns < target] - target

    if len(downside) == 0:
        # No losing periods → perfect Sortino (conceptually infinite).
        # Return a large finite number to avoid inf in reports.
        return float("inf") if returns.mean() > target else 0.0

    # Downside deviation = RMS of downside excess returns.
    downside_std: float = math.sqrt(float((downside ** 2).mean()))

    if downside_std == 0.0 or math.isnan(downside_std):
        return 0.0

    return float(
        (returns.mean() - target) / downside_std * math.sqrt(periods_per_year)
    )


# ---------------------------------------------------------------------------
# Drawdown and risk metrics
# ---------------------------------------------------------------------------

def max_drawdown(equity_series: pd.Series) -> float:
    """
    Compute the maximum drawdown as a positive percentage.

    The maximum drawdown is the largest percentage decline from any equity
    peak to the subsequent trough before a new peak is reached.

    Parameters
    ----------
    equity_series : pd.Series
        Equity curve values (absolute, not returns).  Must be monotonically
        indexed (i.e. time-ordered).

    Returns
    -------
    float
        Maximum drawdown as a positive percentage (e.g. 15.3 for 15.3%).
        Returns 0.0 for a series of length < 2 or a flat equity curve.
    """
    if equity_series is None or len(equity_series) < 2:
        return 0.0

    equity = equity_series.dropna().astype(float)
    if len(equity) < 2:
        return 0.0

    rolling_max: pd.Series = equity.cummax()
    # Avoid division by zero if rolling_max ever hits 0.
    safe_rolling_max = rolling_max.replace(0.0, float("nan"))
    drawdown: pd.Series = (equity - safe_rolling_max) / safe_rolling_max

    mdd = float(abs(drawdown.min()))
    return mdd * 100.0 if not math.isnan(mdd) else 0.0


def calmar_ratio(
    equity_series: pd.Series,
    periods_per_year: int = 8_760,
) -> float:
    """
    Compute the Calmar Ratio (CAGR / Max Drawdown).

    A higher Calmar indicates better risk-adjusted returns relative to the
    worst observed drawdown.

    Parameters
    ----------
    equity_series : pd.Series
        Equity curve (absolute values, time-ordered).
    periods_per_year : int
        Number of observations per year (default 8,760 for hourly).

    Returns
    -------
    float
        Calmar Ratio.  Returns 0.0 if max drawdown is zero (no losses).
    """
    if equity_series is None or len(equity_series) < 2:
        return 0.0

    equity = equity_series.dropna().astype(float)
    if len(equity) < 2:
        return 0.0

    initial: float = float(equity.iloc[0])
    final: float = float(equity.iloc[-1])

    if initial <= 0.0:
        return 0.0

    total_return: float = (final / initial) - 1.0
    n_periods: int = len(equity)

    # CAGR: (1 + total_return)^(periods_per_year / n_periods) - 1
    try:
        cagr: float = (1.0 + total_return) ** (periods_per_year / n_periods) - 1.0
    except (OverflowError, ZeroDivisionError):
        cagr = 0.0

    mdd: float = max_drawdown(equity) / 100.0  # convert back to decimal

    if mdd <= 0.0:
        return 0.0  # no drawdown — ratio is undefined, return 0 for safety

    return float(cagr / mdd)


# ---------------------------------------------------------------------------
# Trade-level metrics
# ---------------------------------------------------------------------------

def profit_factor(trades: List[dict]) -> float:
    """
    Compute the Profit Factor: gross_profit / gross_loss.

    Parameters
    ----------
    trades : list of dict
        Each dict must have a ``net_pnl`` key (float, USDT).

    Returns
    -------
    float
        Profit Factor.  Returns ``float('inf')`` if there are no losing
        trades, and 0.0 if there are no winning trades.
    """
    if not trades:
        return 0.0

    gross_profit: float = sum(
        float(t.get("net_pnl", 0.0))
        for t in trades
        if float(t.get("net_pnl", 0.0)) > 0.0
    )
    gross_loss: float = abs(sum(
        float(t.get("net_pnl", 0.0))
        for t in trades
        if float(t.get("net_pnl", 0.0)) < 0.0
    ))

    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0.0 else 0.0

    return float(gross_profit / gross_loss)


def win_rate(trades: List[dict]) -> float:
    """
    Compute the win rate: fraction of trades with positive net P&L.

    Parameters
    ----------
    trades : list of dict
        Each dict must have a ``net_pnl`` key.

    Returns
    -------
    float
        Win rate in [0.0, 1.0].  Returns 0.0 for an empty list.
    """
    if not trades:
        return 0.0

    wins: int = sum(
        1 for t in trades if float(t.get("net_pnl", 0.0)) > 0.0
    )
    return float(wins / len(trades))


def avg_win_loss(trades: List[dict]) -> dict:
    """
    Compute average percentage return for winning and losing trades.

    Parameters
    ----------
    trades : list of dict
        Each dict must have a ``pnl_pct`` key (percentage, e.g. 3.5 for 3.5%).

    Returns
    -------
    dict with keys:
        avg_win_pct  : float (positive, or 0.0 if no wins)
        avg_loss_pct : float (negative, or 0.0 if no losses)
    """
    if not trades:
        return {"avg_win_pct": 0.0, "avg_loss_pct": 0.0}

    win_pcts = [
        float(t.get("pnl_pct", 0.0))
        for t in trades
        if float(t.get("pnl_pct", 0.0)) > 0.0
    ]
    loss_pcts = [
        float(t.get("pnl_pct", 0.0))
        for t in trades
        if float(t.get("pnl_pct", 0.0)) <= 0.0
    ]

    avg_win: float = float(np.mean(win_pcts)) if win_pcts else 0.0
    avg_loss: float = float(np.mean(loss_pcts)) if loss_pcts else 0.0

    return {
        "avg_win_pct": round(avg_win, 4),
        "avg_loss_pct": round(avg_loss, 4),
    }


# ---------------------------------------------------------------------------
# Composite metrics computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    equity_series: pd.Series,
    trades: List[dict],
    periods_per_year: int = 8_760,
    risk_free_rate: float = 0.02,
) -> dict:
    """
    Compute the full suite of performance metrics from an equity curve and
    a list of closed trade records.

    Parameters
    ----------
    equity_series : pd.Series
        Time-ordered equity curve (absolute USDT values).
        Typically produced by sampling ``PaperExecutor.get_portfolio_value``
        at regular intervals.
    trades : list of dict
        Closed trade records.  Each dict must have at minimum:
        - ``net_pnl`` (float): net profit/loss in USDT.
        - ``pnl_pct`` (float): net P&L as a percentage.
    periods_per_year : int
        Observations per year for annualisation.  Default 8,760 (hourly).
    risk_free_rate : float
        Annual risk-free rate for Sharpe / Sortino.  Default 0.02 (2%).

    Returns
    -------
    dict with keys:
        sharpe_ratio, sortino_ratio, max_drawdown_pct, calmar_ratio,
        profit_factor, win_rate, avg_win_pct, avg_loss_pct,
        total_trades, total_return_pct, initial_equity, final_equity
    """
    # ---- Equity-curve derived metrics ----
    equity = equity_series.dropna().astype(float)

    initial_equity: float = float(equity.iloc[0]) if len(equity) > 0 else 0.0
    final_equity: float = float(equity.iloc[-1]) if len(equity) > 0 else 0.0

    total_return_pct: float = (
        ((final_equity - initial_equity) / initial_equity * 100.0)
        if initial_equity > 0.0 else 0.0
    )

    # Compute per-period returns from the equity curve.
    # pct_change() drops the first NaN row automatically.
    hourly_returns: pd.Series = equity.pct_change().dropna()

    sharpe: float = sharpe_ratio(
        hourly_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    sortino: float = sortino_ratio(
        hourly_returns,
        target=0.0,
        periods_per_year=periods_per_year,
    )
    mdd: float = max_drawdown(equity)
    calmar: float = calmar_ratio(equity, periods_per_year=periods_per_year)

    # ---- Trade-level metrics ----
    closed = [t for t in trades if t.get("status", "closed") in ("closed", "CLOSED")]
    if not closed:
        closed = trades  # caller may have pre-filtered

    pf: float = profit_factor(closed)
    wr: float = win_rate(closed)
    wl: dict = avg_win_loss(closed)
    total_trades: int = len(closed)

    # ---- Sanitise ratios (avoid JSON-inf and extreme values) ----
    def _safe_ratio(v: float) -> float:
        if v is None or math.isnan(v) or math.isinf(v) or abs(v) > 100:
            return 0.0
        return round(v, 4)

    sharpe = _safe_ratio(sharpe)
    sortino = _safe_ratio(sortino) if sortino != float("inf") else 0.0
    pf = _safe_ratio(pf) if pf != float("inf") else 0.0

    # ---- Assemble result ----
    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": round(mdd, 4),
        "calmar_ratio": round(calmar, 4),
        "profit_factor": pf,
        "win_rate": round(wr, 4),
        "avg_win_pct": wl["avg_win_pct"],
        "avg_loss_pct": wl["avg_loss_pct"],
        "total_trades": total_trades,
        "total_return_pct": round(total_return_pct, 4),
        "initial_equity": round(initial_equity, 4),
        "final_equity": round(final_equity, 4),
    }
