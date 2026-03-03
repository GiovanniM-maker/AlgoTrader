"""
src/execution/paper_executor.py

Paper Trading Executor.

Simulates trade execution with realistic slippage (Gaussian) and fee
deduction. Maintains an in-memory ledger of open positions and closed
trades, provides stop / take-profit checking, trailing stop management,
and portfolio-value queries.

All monetary values are in USDT.

Slippage model
--------------
Fill price is sampled from a Gaussian centred on the requested price:

    LONG  entry : fill = entry_price * (1 + N(0, slippage_std))
    SHORT entry : fill = entry_price * (1 - N(0, slippage_std))
    LONG  exit  : fill = exit_price  * (1 - N(0, slippage_std))   (adverse)
    SHORT exit  : fill = exit_price  * (1 + N(0, slippage_std))   (adverse)

Fee model
---------
    fee = quantity * fill_price * fee_rate
"""

from __future__ import annotations

import uuid
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any


# ---------------------------------------------------------------------------
# Trade dataclass
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """
    Full lifecycle record for a single paper trade.

    Fields follow the DB schema in database/schema.sql so that records can
    be persisted by PortfolioManager without any transformation.
    """
    # Identity
    trade_id: str
    symbol: str
    direction: str          # 'LONG' or 'SHORT'
    status: str             # 'OPEN' or 'CLOSED'

    # Entry
    entry_time: datetime
    entry_price: float
    entry_slippage: float   # absolute price difference (fill - requested)
    entry_fee: float        # USDT fee paid at entry

    # Sizing
    quantity: float
    notional_value: float   # entry_price * quantity (pre-fill, pre-fee)

    # Risk levels
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float]  # None if not activated yet

    # Exit
    exit_time: Optional[datetime]
    exit_price: Optional[float]
    exit_slippage: float    # absolute price difference
    exit_fee: float         # USDT fee paid at exit
    exit_reason: Optional[str]

    # P&L
    gross_pnl: float        # before fees
    net_pnl: float          # after all fees
    pnl_pct: float          # net_pnl / (entry_price * quantity) * 100
    duration_minutes: int   # 0 while open

    # Signal metadata (stored for analytics)
    confidence_score: float
    signal_breakdown: dict
    kelly_fraction: float
    risk_amount: float
    leverage: float = 1.0       # 1x=spot, 2-10x simulated

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a flat dict suitable for DB insertion."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "direction": self.direction.lower(),
            "status": self.status.lower(),
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "entry_slippage": self.entry_slippage,
            "entry_fee": self.entry_fee,
            "quantity": self.quantity,
            "notional_value": self.notional_value,
            "stop_loss_price": self.stop_loss,
            "take_profit_price": self.take_profit,
            "trailing_stop": self.trailing_stop,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "exit_slippage": self.exit_slippage,
            "exit_fee": self.exit_fee,
            "exit_reason": self.exit_reason.lower() if self.exit_reason else None,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "pnl_pct": self.pnl_pct,
            "duration_minutes": self.duration_minutes,
            "confidence_score": self.confidence_score,
            "signal_breakdown": self.signal_breakdown,
            "kelly_fraction": self.kelly_fraction,
            "risk_amount": self.risk_amount,
            "leverage": self.leverage,
        }

    def __repr__(self) -> str:
        return (
            f"Trade(id={self.trade_id[:8]} {self.symbol} {self.direction} "
            f"status={self.status} entry={self.entry_price:.4f} "
            f"qty={self.quantity:.6f} pnl={self.net_pnl:.2f})"
        )


# ---------------------------------------------------------------------------
# PaperExecutor
# ---------------------------------------------------------------------------

class PaperExecutor:
    """
    Paper-trading execution engine.

    Parameters
    ----------
    initial_capital : float
        Starting USDT balance (default $10,000).
    fee_rate : float
        Per-trade fee fraction, e.g. 0.001 for 0.1% taker fee.
    slippage_std : float
        Standard deviation of the Gaussian slippage model as a fraction
        of the requested price (e.g. 0.0005 = 0.05%).
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        fee_rate: float = 0.001,
        slippage_std: float = 0.0005,
    ) -> None:
        self.initial_capital: float = initial_capital
        self.cash: float = initial_capital
        self.fee_rate: float = fee_rate
        self.slippage_std: float = slippage_std

        # Active positions: trade_id → Trade
        self.positions: Dict[str, Trade] = {}

        # Historical closed trades
        self.closed_trades: List[Trade] = []

        # High-water mark for drawdown calculation
        self.peak_equity: float = initial_capital

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "PaperExecutor initialised: capital=%.2f fee_rate=%.4f slippage_std=%.4f",
            initial_capital, fee_rate, slippage_std,
        )

    # ------------------------------------------------------------------
    # Recovery (restore open positions from DB after restart)
    # ------------------------------------------------------------------

    def restore_open_positions(
        self,
        trades_from_db: List[Dict[str, Any]],
        total_realized_pnl: float = 0.0,
    ) -> None:
        """
        Restore open positions from DB into memory (used after bot restart).

        Cash is computed as: initial_capital + total_realized_pnl - total_cost_of_open.
        This correctly accounts for profits from closed trades before the restart.
        """
        if not trades_from_db:
            return
        total_cost = 0.0
        for row in trades_from_db:
            try:
                from datetime import datetime
                entry_time = row.get("entry_time")
                if isinstance(entry_time, str) and "T" in entry_time:
                    dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                    entry_time = dt.replace(tzinfo=None) if dt.tzinfo else dt
                elif entry_time is None:
                    entry_time = datetime.utcnow().replace(tzinfo=None)

                cost = float(row.get("entry_price", 0)) * float(row.get("quantity", 0))
                cost += float(row.get("entry_fee", 0))
                total_cost += cost

                sl = row.get("stop_loss_price") or row.get("stop_loss") or 0
                tp = row.get("take_profit_price") or row.get("take_profit") or 0

                trade = Trade(
                    trade_id=str(row.get("trade_id", "")),
                    symbol=str(row.get("symbol", "")),
                    direction=(row.get("direction", "long") or "long").upper(),
                    status="OPEN",
                    entry_time=entry_time,
                    entry_price=float(row.get("entry_price", 0)),
                    entry_slippage=float(row.get("entry_slippage", 0)),
                    entry_fee=float(row.get("entry_fee", 0)),
                    quantity=float(row.get("quantity", 0)),
                    notional_value=float(row.get("notional_value", 0)),
                    stop_loss=float(sl or 0),
                    take_profit=float(tp or 0),
                    trailing_stop=None,
                    exit_time=None,
                    exit_price=None,
                    exit_slippage=0.0,
                    exit_fee=0.0,
                    exit_reason=None,
                    gross_pnl=0.0,
                    net_pnl=0.0,
                    pnl_pct=0.0,
                    duration_minutes=0,
                    confidence_score=float(row.get("confidence_score", 0)),
                    signal_breakdown=row.get("signal_breakdown") or {},
                    kelly_fraction=float(row.get("kelly_fraction", 0)),
                    risk_amount=float(row.get("risk_amount", 0)),
                    leverage=float(row.get("leverage", 1.0)),
                )
                trade.signal_breakdown = (
                    trade.signal_breakdown
                    if isinstance(trade.signal_breakdown, dict)
                    else {}
                )
                self.positions[trade.trade_id] = trade
            except Exception as exc:
                self.logger.warning("restore_open_positions: skip row %s: %s", row.get("trade_id"), exc)

        # Cash = initial + realized from closed trades - cost of open positions.
        # Do NOT clamp to 0: negative cash indicates over-leverage (bug elsewhere).
        self.cash = self.initial_capital + total_realized_pnl - total_cost
        if self.cash < 0:
            self.logger.warning(
                "restore_open_positions: cash=%.2f (negative). "
                "Positions cost %.2f exceeds available %.2f. "
                "Equity will use mark-to-market of positions.",
                self.cash, total_cost, self.initial_capital + total_realized_pnl,
            )
        self.logger.info(
            "Restored %d open position(s) from DB. cash=%.2f (realized=%.2f)",
            len(self.positions), self.cash, total_realized_pnl,
        )

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def execute_buy(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        position_size_usd: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        confidence_score: float,
        signal_breakdown: dict,
        kelly_fraction: float = 0.05,
        risk_amount: float = 0.0,
        leverage: float = 1.0,
    ) -> Optional[Trade]:
        """
        Simulate opening a position.

        Applies Gaussian slippage and deducts fees from ``self.cash``.
        Returns None and logs a warning if there is insufficient cash.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. "BTCUSDT".
        direction : str
            'LONG' or 'SHORT'.
        entry_price : float
            Requested fill price.
        position_size_usd : float
            Intended notional value before slippage.
        quantity : float
            Number of base-asset units.
        stop_loss : float
            Stop-loss price.
        take_profit : float
            Take-profit price.
        confidence_score : float
            Signal confidence [0, 100].
        signal_breakdown : dict
            Per-layer signal details for analytics.
        kelly_fraction : float
            Kelly fraction used for sizing (stored for audit).
        risk_amount : float
            USDT amount at risk (stored for portfolio risk tracking).

        Returns
        -------
        Trade or None
        """
        direction = direction.upper()

        # ---- Slippage simulation ----
        slippage_pct: float = random.gauss(0.0, self.slippage_std)
        if direction == "LONG":
            # Long entries are filled slightly above the quoted price (adverse).
            fill_price: float = entry_price * (1.0 + abs(slippage_pct))
        else:
            # Short entries are filled slightly below the quoted price (adverse).
            fill_price = entry_price * (1.0 - abs(slippage_pct))

        entry_slippage: float = abs(fill_price - entry_price)

        # ---- Re-compute quantity at fill price ----
        # Use the provided quantity but recalculate cost at the actual fill price.
        actual_quantity: float = quantity  # honour the requested quantity

        # ---- Fee ----
        entry_fee: float = actual_quantity * fill_price * self.fee_rate
        total_cost: float = actual_quantity * fill_price + entry_fee

        # ---- Cash check (allow 5% tolerance for fee rounding) ----
        if self.cash < 0:
            self.logger.warning(
                "[%s] Cash is negative (%.2f). Trade rejected.",
                symbol, self.cash,
            )
            return None
        if self.cash < total_cost * 0.95:
            self.logger.warning(
                "[%s] Insufficient cash: have %.2f, need %.2f. Trade rejected.",
                symbol, self.cash, total_cost,
            )
            return None

        # ---- Create trade ----
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            symbol=symbol,
            direction=direction,
            status="OPEN",
            entry_time=datetime.utcnow(),
            entry_price=fill_price,
            entry_slippage=entry_slippage,
            entry_fee=entry_fee,
            quantity=actual_quantity,
            notional_value=entry_price * actual_quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=None,
            exit_time=None,
            exit_price=None,
            exit_slippage=0.0,
            exit_fee=0.0,
            exit_reason=None,
            gross_pnl=0.0,
            net_pnl=0.0,
            pnl_pct=0.0,
            duration_minutes=0,
            confidence_score=confidence_score,
            signal_breakdown=signal_breakdown,
            kelly_fraction=kelly_fraction,
            risk_amount=risk_amount,
            leverage=leverage,
        )

        # ---- Deduct cash ----
        self.cash -= total_cost

        # ---- Register ----
        self.positions[trade.trade_id] = trade

        self.logger.info(
            "TRADE OPENED | %s | id=%s dir=%s fill=%.4f qty=%.6f "
            "sl=%.4f tp=%.4f fee=%.4f slip=%.6f cash_remaining=%.2f",
            symbol, trade.trade_id[:8], direction, fill_price, actual_quantity,
            stop_loss, take_profit, entry_fee, entry_slippage, self.cash,
        )
        return trade

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def execute_sell(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
    ) -> Optional[Trade]:
        """
        Simulate closing an open position.

        Applies adverse slippage in the exit direction, computes P&L,
        returns cash, and moves the trade to ``closed_trades``.

        Parameters
        ----------
        trade_id : str
            UUID of the open trade to close.
        exit_price : float
            Requested exit price (e.g. current market price or stop level).
        exit_reason : str
            Human-readable reason (STOP_LOSS / TAKE_PROFIT / TRAILING_STOP /
            TIME_EXIT / SIGNAL_REVERSAL / MANUAL).

        Returns
        -------
        Trade with all exit fields populated, or None if trade_id not found.
        """
        trade = self.positions.get(trade_id)
        if trade is None:
            self.logger.error(
                "execute_sell: trade_id %s not found in open positions.", trade_id[:8]
            )
            return None

        # ---- Slippage (adverse direction at exit) ----
        slippage_pct: float = random.gauss(0.0, self.slippage_std)
        if trade.direction == "LONG":
            # LONG exits are filled slightly below the quoted price (adverse).
            fill_price: float = exit_price * (1.0 - abs(slippage_pct))
        else:
            # SHORT exits are filled slightly above the quoted price (adverse).
            fill_price = exit_price * (1.0 + abs(slippage_pct))

        exit_slippage: float = abs(fill_price - exit_price)

        # ---- Fees ----
        exit_fee: float = trade.quantity * fill_price * self.fee_rate

        # ---- P&L ----
        if trade.direction == "LONG":
            gross_pnl: float = (fill_price - trade.entry_price) * trade.quantity
        else:  # SHORT
            gross_pnl = (trade.entry_price - fill_price) * trade.quantity

        net_pnl: float = gross_pnl - trade.entry_fee - exit_fee

        # P&L percentage relative to the initial notional (entry_price × qty).
        initial_notional: float = trade.entry_price * trade.quantity
        pnl_pct: float = (net_pnl / initial_notional * 100.0) if initial_notional > 0 else 0.0

        # ---- Duration ----
        now = datetime.utcnow()
        duration_minutes: int = max(0, int((now - trade.entry_time).total_seconds() // 60))

        # ---- Mutate trade record ----
        trade.exit_time = now
        trade.exit_price = fill_price
        trade.exit_slippage = exit_slippage
        trade.exit_fee = exit_fee
        trade.exit_reason = exit_reason
        trade.gross_pnl = round(gross_pnl, 6)
        trade.net_pnl = round(net_pnl, 6)
        trade.pnl_pct = round(pnl_pct, 4)
        trade.duration_minutes = duration_minutes
        trade.status = "CLOSED"

        # ---- Return cash (proceeds minus exit fee) ----
        # Cash returned = quantity * fill_price (the sale proceeds) - exit_fee
        cash_back: float = trade.quantity * fill_price - exit_fee
        self.cash += cash_back

        # ---- Move to closed ----
        del self.positions[trade_id]
        self.closed_trades.append(trade)

        # ---- Update peak equity ----
        equity_estimate: float = self.cash  # simplified: positions are now cash
        self.peak_equity = max(self.peak_equity, equity_estimate)

        self.logger.info(
            "TRADE CLOSED | %s | id=%s dir=%s fill=%.4f qty=%.6f "
            "gross_pnl=%+.4f net_pnl=%+.4f (%.2f%%) reason=%s duration=%dm",
            trade.symbol, trade.trade_id[:8], trade.direction,
            fill_price, trade.quantity,
            gross_pnl, net_pnl, pnl_pct,
            exit_reason, duration_minutes,
        )
        return trade

    # ------------------------------------------------------------------
    # Stop / take-profit checking
    # ------------------------------------------------------------------

    def check_stops(
        self,
        current_prices: Dict[str, float],
        candle_extremes: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> List[Tuple[str, float, str]]:
        """
        Evaluate all open positions against current market prices and
        identify any that have hit their stop-loss, take-profit, or
        trailing-stop levels.

        Parameters
        ----------
        current_prices : dict
            Mapping of symbol → current price.
        candle_extremes : dict, optional
            symbol → (low, high) per candela chiusa. Se fornito, usa low per
            stop/trailing e high per take-profit (cattura il minimo della candela).

        Returns
        -------
        list of (trade_id, exit_price, exit_reason) tuples.
        """
        to_close: List[Tuple[str, float, str]] = []

        for trade_id, trade in self.positions.items():
            price = current_prices.get(trade.symbol)
            if price is None:
                continue

            # Per candela chiusa: usa low per stop, high per TP (dato mancante!)
            price_for_stop = price
            price_for_tp = price
            if candle_extremes and trade.symbol in candle_extremes:
                low, high = candle_extremes[trade.symbol]
                price_for_stop = low   # stop scatta se low <= stop
                price_for_tp = high   # TP scatta se high >= tp

            if trade.direction == "LONG":
                # Stop-loss hit (usa low della candela)
                if price_for_stop <= trade.stop_loss:
                    to_close.append((trade_id, trade.stop_loss, "STOP_LOSS"))
                    continue
                # Take-profit hit (usa high della candela)
                if price_for_tp >= trade.take_profit:
                    to_close.append((trade_id, trade.take_profit, "TAKE_PROFIT"))
                    continue
                # Trailing stop hit (usa low)
                if trade.trailing_stop is not None and price_for_stop <= trade.trailing_stop:
                    to_close.append((trade_id, trade.trailing_stop, "TRAILING_STOP"))
                    continue

            else:  # SHORT
                # Stop-loss hit (usa high per SHORT)
                if price_for_tp >= trade.stop_loss:
                    to_close.append((trade_id, trade.stop_loss, "STOP_LOSS"))
                    continue
                # Take-profit hit (usa low per SHORT)
                if price_for_stop <= trade.take_profit:
                    to_close.append((trade_id, trade.take_profit, "TAKE_PROFIT"))
                    continue
                # Trailing stop hit (usa high)
                if trade.trailing_stop is not None and price_for_tp >= trade.trailing_stop:
                    to_close.append((trade_id, trade.trailing_stop, "TRAILING_STOP"))
                    continue

        if to_close:
            self.logger.debug(
                "check_stops: %d position(s) triggered: %s",
                len(to_close),
                [(tid[:8], reason) for tid, _, reason in to_close],
            )

        return to_close

    # ------------------------------------------------------------------
    # Trailing stop management
    # ------------------------------------------------------------------

    def update_trailing_stops(
        self,
        current_prices: Dict[str, float],
        atr_values: Optional[Dict[str, float]] = None,
        trail_multiplier: float = 1.5,
    ) -> None:
        """
        Ratchet trailing stops upward (LONG) or downward (SHORT) as the
        trade moves in our favour.

        For LONG:
            new_trail = current_price - (ATR * trail_multiplier)
            Activate / update only if new_trail > current trailing_stop.

        For SHORT:
            new_trail = current_price + (ATR * trail_multiplier)
            Activate / update only if new_trail < current trailing_stop.

        If ATR is not provided for a symbol, falls back to 2% of price.

        Parameters
        ----------
        current_prices : dict
            symbol → current price.
        atr_values : dict, optional
            symbol → current ATR value.
        trail_multiplier : float
            ATR multiplier for trail distance (default 1.5).
        """
        if atr_values is None:
            atr_values = {}

        for trade_id, trade in self.positions.items():
            price = current_prices.get(trade.symbol)
            if price is None:
                continue

            atr = atr_values.get(trade.symbol, price * 0.02)
            trail_distance = atr * trail_multiplier

            if trade.direction == "LONG":
                # Only update if the trade is currently in profit.
                if price <= trade.entry_price:
                    continue

                new_trail = price - trail_distance

                if trade.trailing_stop is None:
                    trade.trailing_stop = new_trail
                    self.logger.debug(
                        "[%s] Trailing stop activated at %.6f", trade.symbol, new_trail
                    )
                elif new_trail > trade.trailing_stop:
                    old = trade.trailing_stop
                    trade.trailing_stop = new_trail
                    self.logger.debug(
                        "[%s] Trailing stop raised: %.6f → %.6f",
                        trade.symbol, old, new_trail,
                    )

            else:  # SHORT
                # Only update if the trade is currently in profit.
                if price >= trade.entry_price:
                    continue

                new_trail = price + trail_distance

                if trade.trailing_stop is None:
                    trade.trailing_stop = new_trail
                    self.logger.debug(
                        "[%s] Trailing stop activated at %.6f (SHORT)", trade.symbol, new_trail
                    )
                elif new_trail < trade.trailing_stop:
                    old = trade.trailing_stop
                    trade.trailing_stop = new_trail
                    self.logger.debug(
                        "[%s] Trailing stop lowered: %.6f → %.6f (SHORT)",
                        trade.symbol, old, new_trail,
                    )

    def get_open_positions(self) -> list:
        """Return a list of all currently open Trade objects."""
        return list(self.positions.values())

    # ------------------------------------------------------------------
    # Portfolio valuation
    # ------------------------------------------------------------------

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Total portfolio value = cash + mark-to-market value of all open positions.

        For LONG positions: mark-to-market = quantity × current_price.
        For SHORT positions: mark-to-market = entry_notional + unrealised_pnl.

        Parameters
        ----------
        current_prices : dict
            symbol → current price.

        Returns
        -------
        float
            Total equity in USDT.
        """
        positions_value: float = 0.0

        for trade in self.positions.values():
            price = current_prices.get(trade.symbol)
            if price is None:
                # Fall back to entry price if no current quote is available.
                price = trade.entry_price

            if trade.direction == "LONG":
                positions_value += trade.quantity * price
            else:  # SHORT
                # Value = initial notional + unrealised P&L
                unrealised = (trade.entry_price - price) * trade.quantity
                positions_value += trade.entry_price * trade.quantity + unrealised

        return self.cash + positions_value

    def get_equity_snapshot(self, current_prices: Dict[str, float]) -> dict:
        """
        Return a structured snapshot of current portfolio state.

        Parameters
        ----------
        current_prices : dict
            symbol → current price.

        Returns
        -------
        dict with keys:
            equity, cash, positions_value, drawdown_pct,
            open_trades, initial_capital, total_return_pct
        """
        equity: float = self.get_portfolio_value(current_prices)
        positions_value: float = equity - self.cash
        self.peak_equity = max(self.peak_equity, equity)
        drawdown_pct: float = (
            (self.peak_equity - equity) / self.peak_equity * 100.0
            if self.peak_equity > 0 else 0.0
        )
        total_return_pct: float = (
            (equity - self.initial_capital) / self.initial_capital * 100.0
            if self.initial_capital > 0 else 0.0
        )

        return {
            "equity": round(equity, 4),
            "cash": round(self.cash, 4),
            "positions_value": round(positions_value, 4),
            "drawdown_pct": round(drawdown_pct, 4),
            "open_trades": len(self.positions),
            "initial_capital": self.initial_capital,
            "total_return_pct": round(total_return_pct, 4),
        }

    # ------------------------------------------------------------------
    # Trade statistics
    # ------------------------------------------------------------------

    def get_trade_stats(self) -> dict:
        """
        Compute summary statistics over all closed trades.

        Returns
        -------
        dict with keys:
            total_trades, winning_trades, losing_trades, win_rate,
            avg_win_pct, avg_loss_pct, total_pnl, profit_factor,
            best_trade_pnl, worst_trade_pnl, avg_duration_minutes
        """
        if not self.closed_trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "total_pnl": 0.0,
                "profit_factor": 0.0,
                "best_trade_pnl": 0.0,
                "worst_trade_pnl": 0.0,
                "avg_duration_minutes": 0,
            }

        wins = [t for t in self.closed_trades if t.net_pnl > 0]
        losses = [t for t in self.closed_trades if t.net_pnl <= 0]

        total_trades: int = len(self.closed_trades)
        winning_trades: int = len(wins)
        losing_trades: int = len(losses)
        win_rate: float = winning_trades / total_trades if total_trades > 0 else 0.0

        avg_win_pct: float = (
            sum(t.pnl_pct for t in wins) / len(wins) if wins else 0.0
        )
        avg_loss_pct: float = (
            sum(t.pnl_pct for t in losses) / len(losses) if losses else 0.0
        )

        total_pnl: float = sum(t.net_pnl for t in self.closed_trades)

        gross_profit: float = sum(t.net_pnl for t in wins)
        gross_loss: float = abs(sum(t.net_pnl for t in losses))
        profit_factor: float = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        best_trade_pnl: float = max(t.net_pnl for t in self.closed_trades)
        worst_trade_pnl: float = min(t.net_pnl for t in self.closed_trades)

        avg_duration: float = (
            sum(t.duration_minutes for t in self.closed_trades) / total_trades
        )

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 4),
            "avg_win_pct": round(avg_win_pct, 4),
            "avg_loss_pct": round(avg_loss_pct, 4),
            "total_pnl": round(total_pnl, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else None,
            "best_trade_pnl": round(best_trade_pnl, 4),
            "worst_trade_pnl": round(worst_trade_pnl, 4),
            "avg_duration_minutes": round(avg_duration, 1),
        }
