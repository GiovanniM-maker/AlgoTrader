"""
src/core/state_manager.py

Thread-safe global state manager for the algorithmic trading bot.

Exposes a singleton StateManager that keeps track of:
  - Bot lifecycle status (BotStatus enum)
  - Current equity and initial capital
  - Open positions
  - Last signal results per symbol
  - Running performance statistics
  - A snapshot method for the REST API

All mutations go through public methods protected by a threading.RLock,
making the manager safe for concurrent reads and writes from multiple
threads (e.g. the strategy loop, the Flask API, and the event handler).
"""

from __future__ import annotations

import logging
import threading
from copy import deepcopy
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BotStatus
# ---------------------------------------------------------------------------

class BotStatus(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------

class StateManager:
    """
    Singleton thread-safe store for all mutable bot state.

    Do NOT instantiate directly – always use StateManager.get_instance().

    Attributes (read via get_snapshot())
    -------------------------------------
    bot_status       : BotStatus
    current_equity   : float
    initial_capital  : float
    open_positions   : dict  {trade_id: position_dict}
    last_signals     : dict  {symbol: signal_result_dict}
    started_at       : Optional[datetime]
    stats            : dict  (running performance statistics)
    """

    _instance: Optional["StateManager"] = None
    _instance_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()

        self._bot_status: BotStatus = BotStatus.STOPPED
        self._current_equity: float = 0.0
        self._initial_capital: float = 0.0
        self._open_positions: Dict[str, dict] = {}
        self._last_signals: Dict[str, dict] = {}
        self._started_at: Optional[datetime] = None

        # Running stats – updated externally by the strategy and portfolio layers.
        self._stats: Dict[str, Any] = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl_usd": 0.0,
            "total_pnl_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": None,
            "win_rate": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,
            "current_streak": 0,      # positive = winning streak, negative = losing
        }

        logger.debug("StateManager initialised.")

    @classmethod
    def get_instance(cls) -> "StateManager":
        """Return the process-wide singleton."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton (for tests)."""
        with cls._instance_lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Properties (read-only, locked)
    # ------------------------------------------------------------------

    @property
    def bot_status(self) -> BotStatus:
        with self._lock:
            return self._bot_status

    @property
    def current_equity(self) -> float:
        with self._lock:
            return self._current_equity

    @property
    def initial_capital(self) -> float:
        with self._lock:
            return self._initial_capital

    @property
    def open_positions(self) -> Dict[str, dict]:
        """Returns a shallow copy – mutating the returned dict is safe."""
        with self._lock:
            return dict(self._open_positions)

    @property
    def last_signals(self) -> Dict[str, dict]:
        with self._lock:
            return dict(self._last_signals)

    @property
    def started_at(self) -> Optional[datetime]:
        with self._lock:
            return self._started_at

    @property
    def stats(self) -> dict:
        with self._lock:
            return deepcopy(self._stats)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_status(self, status: BotStatus) -> None:
        """
        Transition the bot to *status*.

        Also records started_at when transitioning to RUNNING for the
        first time in a session.

        Parameters
        ----------
        status : BotStatus
        """
        if not isinstance(status, BotStatus):
            raise TypeError(f"Expected BotStatus, got {type(status)}")

        with self._lock:
            previous = self._bot_status
            self._bot_status = status
            if status == BotStatus.RUNNING and self._started_at is None:
                self._started_at = datetime.utcnow()
            logger.info("BotStatus: %s -> %s", previous.value, status.value)

    def initialise_capital(self, initial_capital: float) -> None:
        """
        Set the initial capital (call once at startup).

        Sets both initial_capital and current_equity to *initial_capital*.
        """
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")
        with self._lock:
            self._initial_capital = float(initial_capital)
            self._current_equity = float(initial_capital)
            logger.info("Capital initialised: %.2f USD", initial_capital)

    # ------------------------------------------------------------------
    # Equity
    # ------------------------------------------------------------------

    def update_equity(self, value: float) -> None:
        """
        Set current_equity to *value*.

        Parameters
        ----------
        value : float
            New total portfolio equity in USD.
        """
        if value < 0:
            raise ValueError(f"Equity cannot be negative, got {value}")
        with self._lock:
            self._current_equity = float(value)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def update_position(self, trade_id: str, position_dict: dict) -> None:
        """
        Insert or replace the position identified by *trade_id*.

        Parameters
        ----------
        trade_id     : str
        position_dict: dict  – arbitrary position payload (from Position.to_dict())
        """
        if not trade_id:
            raise ValueError("trade_id must not be empty")
        if not isinstance(position_dict, dict):
            raise TypeError("position_dict must be a dict")

        with self._lock:
            self._open_positions[trade_id] = position_dict
            logger.debug("Position updated: trade_id=%s", trade_id)

    def close_position(self, trade_id: str) -> Optional[dict]:
        """
        Remove the position identified by *trade_id* from open_positions.

        Parameters
        ----------
        trade_id : str

        Returns
        -------
        dict or None
            The removed position dict, or None if it was not found.
        """
        with self._lock:
            removed = self._open_positions.pop(trade_id, None)
            if removed is not None:
                logger.debug("Position removed from state: trade_id=%s", trade_id)
            else:
                logger.warning(
                    "close_position called for unknown trade_id=%s", trade_id
                )
            return removed

    def position_count(self) -> int:
        """Return the number of currently open positions."""
        with self._lock:
            return len(self._open_positions)

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def update_signal(self, symbol: str, signal_result: dict) -> None:
        """
        Store the latest signal result for *symbol*.

        Parameters
        ----------
        symbol        : str
        signal_result : dict – e.g. {"direction": "LONG", "confidence": 0.72, ...}
        """
        if not symbol:
            raise ValueError("symbol must not be empty")
        with self._lock:
            self._last_signals[symbol] = {
                **signal_result,
                "_recorded_at": datetime.utcnow().isoformat(),
            }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def update_stats(self, partial_stats: Dict[str, Any]) -> None:
        """
        Merge *partial_stats* into the internal stats dict.

        Only keys that already exist in the stats schema are updated;
        unknown keys raise a ValueError to catch typos early.

        Parameters
        ----------
        partial_stats : dict
        """
        with self._lock:
            for key, value in partial_stats.items():
                if key not in self._stats:
                    raise ValueError(
                        f"Unknown stats key: {key!r}. "
                        f"Valid keys: {list(self._stats.keys())}"
                    )
                self._stats[key] = value

    def record_trade_result(self, pnl_pct: float, pnl_usd: float) -> None:
        """
        Convenience method to update win/loss counters and P&L accumulators
        after a trade closes.

        Parameters
        ----------
        pnl_pct : float  – realised P&L as percentage of position size
        pnl_usd : float  – realised P&L in USD
        """
        with self._lock:
            self._stats["total_trades"] += 1
            self._stats["total_pnl_usd"] += pnl_usd
            # Running total_pnl_pct is a simple average over all trades
            n = self._stats["total_trades"]
            prev_avg = self._stats["total_pnl_pct"]
            self._stats["total_pnl_pct"] = prev_avg + (pnl_pct - prev_avg) / n

            if pnl_pct > 0:
                self._stats["winning_trades"] += 1
                wins = self._stats["winning_trades"]
                prev_avg_win = self._stats["avg_win_pct"]
                self._stats["avg_win_pct"] = (
                    prev_avg_win + (pnl_pct - prev_avg_win) / wins
                )
                streak = self._stats["current_streak"]
                self._stats["current_streak"] = streak + 1 if streak >= 0 else 1
            else:
                self._stats["losing_trades"] += 1
                losses = self._stats["losing_trades"]
                prev_avg_loss = self._stats["avg_loss_pct"]
                self._stats["avg_loss_pct"] = (
                    prev_avg_loss + (abs(pnl_pct) - prev_avg_loss) / losses
                )
                streak = self._stats["current_streak"]
                self._stats["current_streak"] = streak - 1 if streak <= 0 else -1

            # Recompute win_rate
            total = self._stats["total_trades"]
            self._stats["win_rate"] = (
                self._stats["winning_trades"] / total if total > 0 else 0.0
            )

            # Recompute profit_factor
            gross_profit = (
                self._stats["avg_win_pct"] * self._stats["winning_trades"]
            )
            gross_loss = (
                self._stats["avg_loss_pct"] * self._stats["losing_trades"]
            )
            self._stats["profit_factor"] = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        """
        Return a serialisable snapshot of all state for the REST API.

        The snapshot is a deep copy so callers cannot mutate internal state
        through the returned dict.

        Returns
        -------
        dict
        """
        with self._lock:
            uptime_seconds: Optional[float] = None
            if self._started_at is not None:
                uptime_seconds = (
                    datetime.utcnow() - self._started_at
                ).total_seconds()

            return {
                "bot_status": self._bot_status.value,
                "current_equity": self._current_equity,
                "initial_capital": self._initial_capital,
                "equity_change_pct": (
                    (self._current_equity - self._initial_capital)
                    / self._initial_capital
                    * 100
                    if self._initial_capital > 0
                    else 0.0
                ),
                "open_positions_count": len(self._open_positions),
                "open_positions": deepcopy(self._open_positions),
                "last_signals": deepcopy(self._last_signals),
                "started_at": (
                    self._started_at.isoformat() if self._started_at else None
                ),
                "uptime_seconds": uptime_seconds,
                "stats": deepcopy(self._stats),
            }

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"StateManager(status={self._bot_status.value}, "
                f"equity={self._current_equity:.2f}, "
                f"open_positions={len(self._open_positions)})"
            )
