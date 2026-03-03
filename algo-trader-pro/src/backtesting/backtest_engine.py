"""
AlgoTrader Pro - Event-Driven Backtesting Engine
=================================================
Simulates trading strategy execution on historical OHLCV data using the same
strategy/signal codebase that runs in live paper trading.

Architecture
------------
- Iterates through OHLCV candles chronologically
- On each candle: builds a rolling window, computes all signals, evaluates strategy
- Simulates order fills with slippage + fees (Gaussian noise model)
- Checks intracandle stop-loss and take-profit using candle high/low
- Tracks equity curve, open positions, and completed trades
- Returns a BacktestResult dataclass with full performance metrics

Usage
-----
    engine = BacktestEngine(config, strategy, risk_manager, initial_capital=10000.0)
    result = await engine.run(
        symbol="BTCUSDT",
        timeframe="1h",
        start_date="2023-01-01",
        end_date="2023-12-31",
        ohlcv_df=df,
    )
    print(result.metrics)
"""

from __future__ import annotations

import asyncio
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# BacktestResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """
    Full result of a single backtest run.

    Attributes
    ----------
    equity_curve : list of dict
        Chronological equity snapshots:
        ``[{"timestamp": str, "equity": float, "drawdown_pct": float}, ...]``
    trades : list of dict
        Every completed trade with entry/exit details and P&L.
    metrics : dict
        Computed performance metrics (Sharpe, Sortino, CAGR, …).
    start_equity : float
        Capital at the start of the backtest.
    end_equity : float
        Capital at the end of the backtest.
    start_date : str
        ISO-8601 start date of the tested period.
    end_date : str
        ISO-8601 end date of the tested period.
    symbol : str
        Trading pair, e.g. ``"BTCUSDT"``.
    timeframe : str
        Candle granularity, e.g. ``"1h"``.
    """

    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    start_equity: float = 10_000.0
    end_equity: float = 10_000.0
    start_date: str = ""
    end_date: str = ""
    symbol: str = ""
    timeframe: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "start_equity": self.start_equity,
            "end_equity": self.end_equity,
            "metrics": self.metrics,
            "equity_curve": self.equity_curve,
            "trades": self.trades,
        }


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Parameters
    ----------
    config : Any
        Application configuration object.  Relevant attributes:
          - ``config.risk.fee_rate``      (float, default 0.001 = 0.1 %)
          - ``config.risk.slippage_std``  (float, default 0.0005 = 0.05 %)
          - ``config.strategy.lookback_candles`` (int, default 200)
    strategy : Any
        Strategy object with an ``evaluate(ohlcv_df, sentiment_df)`` async method
        that returns a dict like::

            {
                "direction": "LONG" | "SHORT" | "HOLD",
                "confidence": 0.75,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "size_usd": 500.0,
                "signals": [...],
            }

    risk_manager : Any
        Risk manager with an optional ``approve(signal_dict, equity)`` method.
        If ``approve`` returns False the trade is skipped.
    initial_capital : float
        Starting capital in quote currency (default 10 000 USD).
    """

    # Default constants (overridden by config when present)
    _DEFAULT_FEE_RATE: float = 0.001       # 0.10 % taker fee
    _DEFAULT_SLIPPAGE_STD: float = 0.0005  # 0.05 % std-dev Gaussian slip
    _DEFAULT_LOOKBACK: int = 200           # rolling window size

    def __init__(
        self,
        config: Any,
        strategy: Any,
        risk_manager: Any,
        initial_capital: float = 10_000.0,
    ) -> None:
        self._config = config
        self._strategy = strategy
        self._risk_manager = risk_manager
        self._initial_capital = float(initial_capital)

        # Resolve tunable parameters from config with safe fallbacks
        self._fee_rate = self._safe_cfg("risk.fee_rate", self._DEFAULT_FEE_RATE)
        self._slippage_std = self._safe_cfg("risk.slippage_std", self._DEFAULT_SLIPPAGE_STD)
        self._lookback = int(self._safe_cfg("strategy.lookback_candles", self._DEFAULT_LOOKBACK))

        logger.info(
            "BacktestEngine created",
            extra={
                "initial_capital": initial_capital,
                "fee_rate": self._fee_rate,
                "slippage_std": self._slippage_std,
                "lookback": self._lookback,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        ohlcv_df: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run the backtest over the supplied OHLCV data.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"BTCUSDT"``.
        timeframe : str
            Candle timeframe, e.g. ``"1h"``.
        start_date : str
            ISO-8601 start date (inclusive), e.g. ``"2023-01-01"``.
        end_date : str
            ISO-8601 end date (inclusive), e.g. ``"2023-12-31"``.
        ohlcv_df : pd.DataFrame
            OHLCV data.  Required columns: ``open``, ``high``, ``low``,
            ``close``, ``volume``.  Index should be a ``DatetimeIndex`` or
            convertible to one.
        sentiment_df : pd.DataFrame, optional
            Aligned sentiment data (same index as *ohlcv_df*).

        Returns
        -------
        BacktestResult
        """
        logger.info(
            "Backtest starting",
            extra={"symbol": symbol, "timeframe": timeframe,
                   "start": start_date, "end": end_date,
                   "candles": len(ohlcv_df)},
        )

        # -- Validate & clean input ----------------------------------------
        df = self._prepare_df(ohlcv_df)
        if df.empty:
            logger.warning("Empty OHLCV DataFrame — returning empty result.")
            return BacktestResult(
                start_equity=self._initial_capital,
                end_equity=self._initial_capital,
                start_date=start_date,
                end_date=end_date,
                symbol=symbol,
                timeframe=timeframe,
            )

        # -- State variables -----------------------------------------------
        equity: float = self._initial_capital
        cash: float = self._initial_capital
        peak_equity: float = self._initial_capital

        # Open position (one position at a time for simplicity)
        position: Optional[Dict[str, Any]] = None

        equity_curve: List[Dict[str, Any]] = []
        trades: List[Dict[str, Any]] = []

        total_candles = len(df)

        # -- Main loop --------------------------------------------------------
        for i in range(self._lookback, total_candles):
            candle = df.iloc[i]
            window_df = df.iloc[max(0, i - self._lookback): i + 1].copy()

            ts_raw = candle.name
            timestamp_str = (
                ts_raw.isoformat() if hasattr(ts_raw, "isoformat") else str(ts_raw)
            )
            close_price: float = float(candle["close"])
            high_price: float = float(candle["high"])
            low_price: float = float(candle["low"])

            # -- 1. Check intracandle exits (stop-loss / take-profit) ------
            if position is not None:
                exit_result = self._check_intracandle_exit(
                    position, high_price, low_price, close_price, timestamp_str
                )
                if exit_result is not None:
                    trade_record, pnl = exit_result
                    cash += position["size_usd"] + pnl
                    equity = cash  # no other open positions
                    trades.append(trade_record)
                    logger.debug(
                        "Intracandle exit",
                        extra={"type": trade_record["exit_reason"], "pnl": pnl},
                    )
                    position = None

            # -- 2. Mark-to-market equity ----------------------------------
            if position is not None:
                unreal = self._unrealized_pnl(position, close_price)
                equity = cash + position["size_usd"] + unreal
            else:
                equity = cash

            # Update peak
            if equity > peak_equity:
                peak_equity = equity
            drawdown_pct = (
                0.0 if peak_equity == 0 else (equity - peak_equity) / peak_equity * 100.0
            )

            equity_curve.append(
                {
                    "timestamp": timestamp_str,
                    "equity": round(equity, 4),
                    "drawdown_pct": round(drawdown_pct, 4),
                }
            )

            # -- 3. Get sentiment slice for this candle --------------------
            sentiment_slice: Optional[pd.DataFrame] = None
            if sentiment_df is not None and not sentiment_df.empty:
                sentiment_slice = self._get_sentiment_slice(
                    sentiment_df, ts_raw, self._lookback
                )

            # -- 4. Evaluate strategy --------------------------------------
            try:
                signal = await self._evaluate_strategy(window_df, sentiment_slice)
            except Exception as exc:
                logger.warning(
                    "Strategy evaluation failed on candle %d: %s", i, exc
                )
                signal = {"direction": "HOLD"}

            direction = signal.get("direction", "HOLD")

            # -- 5. Entry logic -------------------------------------------
            if position is None and direction in ("LONG", "SHORT"):
                # Ask risk manager for approval
                approved = self._risk_approve(signal, equity)
                if approved:
                    size_usd = float(signal.get("size_usd", equity * 0.05))
                    size_usd = min(size_usd, cash * 0.95)  # never over-invest

                    if size_usd > 10.0:
                        fill_price, qty, fee = self._simulate_fill(
                            price=close_price,
                            direction=direction,
                            size_usd=size_usd,
                            fee_rate=self._fee_rate,
                        )
                        cash -= size_usd + fee

                        stop_loss_pct = float(signal.get("stop_loss_pct", 0.02))
                        take_profit_pct = float(signal.get("take_profit_pct", 0.04))

                        position = {
                            "id": str(uuid.uuid4()),
                            "symbol": symbol,
                            "direction": direction,
                            "entry_price": fill_price,
                            "quantity": qty,
                            "size_usd": size_usd,
                            "entry_fee": fee,
                            "entry_time": timestamp_str,
                            "stop_loss_price": (
                                fill_price * (1 - stop_loss_pct)
                                if direction == "LONG"
                                else fill_price * (1 + stop_loss_pct)
                            ),
                            "take_profit_price": (
                                fill_price * (1 + take_profit_pct)
                                if direction == "LONG"
                                else fill_price * (1 - take_profit_pct)
                            ),
                            "signals": signal.get("signals", []),
                            "confidence": signal.get("confidence", 0.0),
                        }
                        logger.debug(
                            "Position opened",
                            extra={
                                "direction": direction,
                                "entry": fill_price,
                                "size_usd": size_usd,
                            },
                        )

            # -- 6. Exit on opposite signal --------------------------------
            elif position is not None and self._is_opposite_signal(
                position["direction"], direction
            ):
                fill_price, qty, fee = self._simulate_fill(
                    price=close_price,
                    direction="SELL" if position["direction"] == "LONG" else "BUY",
                    size_usd=position["size_usd"],
                    fee_rate=self._fee_rate,
                )
                pnl = self._compute_pnl(position, fill_price, fee)
                cash += position["size_usd"] + pnl

                trade_record = self._build_trade_record(
                    position=position,
                    exit_price=fill_price,
                    exit_time=timestamp_str,
                    exit_fee=fee,
                    pnl=pnl,
                    exit_reason="signal_reversal",
                )
                trades.append(trade_record)
                position = None

        # -- End of loop: close any open position at last close ------------
        if position is not None:
            last_candle = df.iloc[-1]
            last_close = float(last_candle["close"])
            last_ts = (
                last_candle.name.isoformat()
                if hasattr(last_candle.name, "isoformat")
                else str(last_candle.name)
            )
            fill_price, qty, fee = self._simulate_fill(
                price=last_close,
                direction="SELL" if position["direction"] == "LONG" else "BUY",
                size_usd=position["size_usd"],
                fee_rate=self._fee_rate,
            )
            pnl = self._compute_pnl(position, fill_price, fee)
            cash += position["size_usd"] + pnl
            equity = cash

            trade_record = self._build_trade_record(
                position=position,
                exit_price=fill_price,
                exit_time=last_ts,
                exit_fee=fee,
                pnl=pnl,
                exit_reason="end_of_data",
            )
            trades.append(trade_record)

        # -- Build result --------------------------------------------------
        metrics = self._compute_metrics(
            equity_curve=equity_curve,
            trades=trades,
            initial_capital=self._initial_capital,
        )

        result = BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            start_equity=self._initial_capital,
            end_equity=round(equity, 4),
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            timeframe=timeframe,
        )

        logger.info(
            "Backtest complete",
            extra={
                "trades": len(trades),
                "end_equity": result.end_equity,
                "total_return_pct": metrics.get("total_return_pct"),
                "sharpe": metrics.get("sharpe"),
            },
        )
        return result

    # ------------------------------------------------------------------
    # Order simulation
    # ------------------------------------------------------------------

    def _simulate_fill(
        self,
        price: float,
        direction: str,
        size_usd: float,
        fee_rate: float = 0.001,
    ) -> Tuple[float, float, float]:
        """
        Simulate an order fill with Gaussian slippage and taker fees.

        Parameters
        ----------
        price : float
            Current market price.
        direction : str
            "LONG" or "BUY" → fills slightly above; "SHORT" or "SELL" → slightly below.
        size_usd : float
            Notional value of the order in USD.
        fee_rate : float
            Taker fee fraction (default 0.001 = 0.1 %).

        Returns
        -------
        (fill_price, quantity, fee) : Tuple[float, float, float]
            fill_price – actual execution price after slippage
            quantity   – units of base asset purchased/sold
            fee        – total fee paid in USD
        """
        if price <= 0 or size_usd <= 0:
            return price, 0.0, 0.0

        # Gaussian slippage: std = 0.05 % of price
        slip = random.gauss(0.0, self._slippage_std)

        # Slip direction: buying pays more, selling receives less
        if direction in ("LONG", "BUY"):
            fill_price = price * (1.0 + abs(slip))
        else:
            fill_price = price * (1.0 - abs(slip))

        fill_price = max(fill_price, 1e-12)  # guard against zero/negative
        quantity = size_usd / fill_price
        fee = size_usd * fee_rate

        return round(fill_price, 8), round(quantity, 8), round(fee, 6)

    # ------------------------------------------------------------------
    # Intracandle exit detection
    # ------------------------------------------------------------------

    def _check_intracandle_exit(
        self,
        position: Dict[str, Any],
        high: float,
        low: float,
        close: float,
        timestamp: str,
    ) -> Optional[Tuple[Dict[str, Any], float]]:
        """
        Check whether the candle's high/low triggered a stop-loss or take-profit.

        Returns ``(trade_record, pnl)`` or ``None`` if no exit was triggered.
        """
        direction = position["direction"]
        sl = position["stop_loss_price"]
        tp = position["take_profit_price"]

        exit_price: Optional[float] = None
        exit_reason: Optional[str] = None

        if direction == "LONG":
            if low <= sl:
                exit_price = sl
                exit_reason = "stop_loss"
            elif high >= tp:
                exit_price = tp
                exit_reason = "take_profit"
        else:  # SHORT
            if high >= sl:
                exit_price = sl
                exit_reason = "stop_loss"
            elif low <= tp:
                exit_price = tp
                exit_reason = "take_profit"

        if exit_price is None:
            return None

        # Simulate fill at the triggered price
        fill_dir = "SELL" if direction == "LONG" else "BUY"
        fill_price, _, fee = self._simulate_fill(
            price=exit_price,
            direction=fill_dir,
            size_usd=position["size_usd"],
            fee_rate=self._fee_rate,
        )
        pnl = self._compute_pnl(position, fill_price, fee)
        trade_record = self._build_trade_record(
            position=position,
            exit_price=fill_price,
            exit_time=timestamp,
            exit_fee=fee,
            pnl=pnl,
            exit_reason=exit_reason,
        )
        return trade_record, pnl

    # ------------------------------------------------------------------
    # P&L helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_pnl(position: Dict[str, Any], exit_price: float, exit_fee: float) -> float:
        """Compute realised P&L for a position (long or short)."""
        direction = position["direction"]
        entry_price = position["entry_price"]
        quantity = position["quantity"]
        entry_fee = position["entry_fee"]

        if direction == "LONG":
            gross = (exit_price - entry_price) * quantity
        else:
            gross = (entry_price - exit_price) * quantity

        return gross - exit_fee - entry_fee

    @staticmethod
    def _unrealized_pnl(position: Dict[str, Any], current_price: float) -> float:
        """Mark-to-market unrealised P&L (fees already paid on entry)."""
        direction = position["direction"]
        entry_price = position["entry_price"]
        quantity = position["quantity"]

        if direction == "LONG":
            return (current_price - entry_price) * quantity
        else:
            return (entry_price - current_price) * quantity

    # ------------------------------------------------------------------
    # Trade record builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_trade_record(
        position: Dict[str, Any],
        exit_price: float,
        exit_time: str,
        exit_fee: float,
        pnl: float,
        exit_reason: str,
    ) -> Dict[str, Any]:
        size_usd = position["size_usd"]
        pnl_pct = (pnl / size_usd * 100.0) if size_usd > 0 else 0.0

        # Duration in hours
        try:
            entry_dt = datetime.fromisoformat(position["entry_time"].replace("Z", "+00:00"))
            exit_dt = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
            duration_hours = (exit_dt - entry_dt).total_seconds() / 3600.0
        except Exception:
            duration_hours = 0.0

        return {
            "id": position["id"],
            "symbol": position["symbol"],
            "direction": position["direction"],
            "entry_price": round(position["entry_price"], 6),
            "exit_price": round(exit_price, 6),
            "quantity": round(position["quantity"], 8),
            "size_usd": round(size_usd, 4),
            "entry_time": position["entry_time"],
            "exit_time": exit_time,
            "exit_reason": exit_reason,
            "pnl": round(pnl, 4),
            "pnl_pct": round(pnl_pct, 4),
            "entry_fee": round(position["entry_fee"], 6),
            "exit_fee": round(exit_fee, 6),
            "duration_hours": round(duration_hours, 2),
            "stop_loss_price": round(position["stop_loss_price"], 6),
            "take_profit_price": round(position["take_profit_price"], 6),
            "confidence": position.get("confidence", 0.0),
            "status": "CLOSED",
        }

    # ------------------------------------------------------------------
    # Strategy / risk helpers
    # ------------------------------------------------------------------

    async def _evaluate_strategy(
        self,
        window_df: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """Call strategy.evaluate() — supports both async and sync strategies."""
        try:
            if asyncio.iscoroutinefunction(self._strategy.evaluate):
                result = await self._strategy.evaluate(window_df, sentiment_df)
            else:
                result = self._strategy.evaluate(window_df, sentiment_df)
            return result if isinstance(result, dict) else {"direction": "HOLD"}
        except Exception as exc:
            logger.debug("Strategy evaluate error: %s", exc)
            return {"direction": "HOLD"}

    def _risk_approve(self, signal: Dict[str, Any], equity: float) -> bool:
        """Ask risk manager to approve a trade.  Falls back to True if not available."""
        try:
            if hasattr(self._risk_manager, "approve"):
                return bool(self._risk_manager.approve(signal, equity))
        except Exception as exc:
            logger.debug("Risk manager approval error: %s", exc)
        return True

    @staticmethod
    def _is_opposite_signal(current_direction: str, new_direction: str) -> bool:
        if current_direction == "LONG" and new_direction == "SHORT":
            return True
        if current_direction == "SHORT" and new_direction == "LONG":
            return True
        return False

    # ------------------------------------------------------------------
    # Sentiment alignment helper
    # ------------------------------------------------------------------

    @staticmethod
    def _get_sentiment_slice(
        sentiment_df: pd.DataFrame,
        ts: Any,
        lookback: int,
    ) -> Optional[pd.DataFrame]:
        try:
            loc = sentiment_df.index.get_indexer([ts], method="nearest")[0]
            start = max(0, loc - lookback)
            return sentiment_df.iloc[start: loc + 1]
        except Exception:
            return None

    # ------------------------------------------------------------------
    # DataFrame preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalise the OHLCV DataFrame."""
        required_cols = {"open", "high", "low", "close", "volume"}
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"OHLCV DataFrame missing columns: {missing}")

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, utc=True)
            except Exception:
                pass

        df.sort_index(inplace=True)
        df.dropna(subset=["close"], inplace=True)

        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        return df

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        equity_curve: List[Dict[str, Any]],
        trades: List[Dict[str, Any]],
        initial_capital: float,
    ) -> Dict[str, Any]:
        """Compute comprehensive performance metrics from equity curve and trades."""

        if not equity_curve:
            return self._empty_metrics()

        equities = [p["equity"] for p in equity_curve]
        final_equity = equities[-1]
        total_return_pct = (final_equity - initial_capital) / initial_capital * 100.0

        # --- Returns series (percentage changes) --------------------------
        equity_series = np.array(equities, dtype=float)
        returns = np.diff(equity_series) / equity_series[:-1]

        # --- Sharpe ratio (annualised, assumes hourly candles by default) -
        if len(returns) > 1 and returns.std() > 0:
            periods_per_year = 8_760  # hourly default; overridden below
            sharpe = float(returns.mean() / returns.std() * math.sqrt(periods_per_year))
        else:
            sharpe = 0.0

        # --- Sortino ratio ------------------------------------------------
        downside = returns[returns < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = float(returns.mean() / downside.std() * math.sqrt(8_760))
        else:
            sortino = 0.0

        # --- Maximum drawdown ---------------------------------------------
        peak = equity_series[0]
        max_dd = 0.0
        for eq in equity_series:
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak * 100.0 if peak > 0 else 0.0
            if dd < max_dd:
                max_dd = dd

        # --- CAGR ---------------------------------------------------------
        n_years = max(len(equity_curve) / 8_760, 1 / 365)
        cagr = (
            ((final_equity / initial_capital) ** (1.0 / n_years) - 1.0) * 100.0
            if initial_capital > 0
            else 0.0
        )

        # --- Calmar -------------------------------------------------------
        calmar = abs(cagr / max_dd) if max_dd != 0.0 else 0.0

        # --- Trade-level metrics ------------------------------------------
        closed_trades = [t for t in trades if t.get("status") == "CLOSED"]
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in closed_trades if t.get("pnl", 0) <= 0]
        win_rate = len(winning_trades) / total_trades * 100.0 if total_trades > 0 else 0.0

        gross_profit = sum(t["pnl"] for t in winning_trades)
        gross_loss = abs(sum(t["pnl"] for t in losing_trades))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        )

        avg_duration = (
            sum(t.get("duration_hours", 0) for t in closed_trades) / total_trades
            if total_trades > 0
            else 0.0
        )

        return {
            "total_return_pct": round(total_return_pct, 4),
            "cagr": round(cagr, 4),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "calmar": round(calmar, 4),
            "max_drawdown": round(max_dd, 4),
            "profit_factor": round(profit_factor, 4),
            "win_rate": round(win_rate, 4),
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "avg_duration_hours": round(avg_duration, 2),
            "gross_profit": round(gross_profit, 4),
            "gross_loss": round(gross_loss, 4),
            "final_equity": round(final_equity, 4),
            "initial_capital": round(initial_capital, 4),
        }

    @staticmethod
    def _empty_metrics() -> Dict[str, Any]:
        keys = [
            "total_return_pct", "cagr", "sharpe", "sortino", "calmar",
            "max_drawdown", "profit_factor", "win_rate", "total_trades",
            "winning_trades", "losing_trades", "avg_duration_hours",
            "gross_profit", "gross_loss", "final_equity", "initial_capital",
        ]
        return {k: 0.0 for k in keys}

    # ------------------------------------------------------------------
    # Config helper
    # ------------------------------------------------------------------

    def _safe_cfg(self, dotted_key: str, default: Any) -> Any:
        """Safely traverse config attributes using dot notation."""
        obj = self._config
        for part in dotted_key.split("."):
            try:
                obj = getattr(obj, part)
            except AttributeError:
                return default
        return obj
