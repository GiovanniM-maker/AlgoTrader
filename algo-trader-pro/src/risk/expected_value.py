"""
src/risk/expected_value.py

Expected Value (EV) calculator with an optional auto-pause guard.

The EV of a trading system is defined as:

    EV = (win_rate * avg_win) - (loss_rate * avg_loss)

A positive EV means the system is profitable on average.  A negative EV
means the system loses money on average and trading should be paused until
performance recovers.

The calculator maintains a rolling window of the last N closed-trade P&L
percentages and recomputes statistics on every call to ``calculate()``.

Usage
-----
::

    ev_calc = ExpectedValueCalculator(rolling_window=30, auto_pause_if_negative=True)

    # After each closed trade:
    ev_calc.update(pnl_pct=0.023)   # 2.3% gain

    result = ev_calc.calculate()
    print(result.ev)                # e.g. 0.012
    print(result.win_rate)          # e.g. 0.60
    print(result.is_positive)       # True

    if ev_calc.should_pause():
        # Halt new trades until EV recovers
        ...
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, List

logger = logging.getLogger(__name__)

# Minimum number of samples required before the auto-pause logic will fire.
# Below this threshold we assume "no data" rather than "bad system".
_MIN_SAMPLES_FOR_PAUSE: int = 10


@dataclass
class EVResult:
    """
    Snapshot of Expected Value statistics from the rolling trade window.

    Attributes
    ----------
    ev : float
        Expected value per trade as a percentage.
        EV = (win_rate * avg_win) - (loss_rate * avg_loss).
    win_rate : float
        Fraction of winning trades in the rolling window [0.0, 1.0].
    avg_win : float
        Mean percentage return on winning trades (positive).
    avg_loss : float
        Mean percentage loss on losing trades (positive, i.e. absolute value).
    profit_factor : float
        Gross profit / gross loss.  Values > 1.0 are profitable.
        Returns 0.0 if there are no losing trades.
        Returns float('inf') if there are wins but no losses.
    sample_size : int
        Number of trades in the current rolling window.
    is_positive : bool
        True when EV > 0.
    """

    ev: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sample_size: int
    is_positive: bool

    def to_dict(self) -> dict:
        return {
            "ev": round(self.ev, 6),
            "win_rate": round(self.win_rate, 6),
            "avg_win": round(self.avg_win, 6),
            "avg_loss": round(self.avg_loss, 6),
            "profit_factor": round(self.profit_factor, 6) if self.profit_factor != float("inf") else "inf",
            "sample_size": self.sample_size,
            "is_positive": self.is_positive,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EVResult(ev={self.ev:+.4f}%, win_rate={self.win_rate:.2%}, "
            f"avg_win={self.avg_win:.4f}%, avg_loss={self.avg_loss:.4f}%, "
            f"PF={self.profit_factor:.3f}, n={self.sample_size}, "
            f"positive={self.is_positive})"
        )


class ExpectedValueCalculator:
    """
    Rolling Expected Value calculator for a live trading system.

    Parameters
    ----------
    rolling_window : int
        Maximum number of recent trades to keep in the rolling buffer.
        Older trades are evicted as new ones arrive.  Default: 30.
    auto_pause_if_negative : bool
        When True, ``should_pause()`` returns True whenever the rolling EV
        is negative and the sample size is at least ``min_samples_to_pause``.
        Default: True.
    min_samples_to_pause : int
        Minimum number of trades required before the auto-pause can trigger.
        Prevents pausing during the warm-up period.  Default: 10.
    """

    def __init__(
        self,
        rolling_window: int = 30,
        auto_pause_if_negative: bool = True,
        min_samples_to_pause: int = _MIN_SAMPLES_FOR_PAUSE,
    ) -> None:
        if rolling_window < 1:
            raise ValueError(f"rolling_window must be >= 1, got {rolling_window}")
        if min_samples_to_pause < 1:
            raise ValueError(f"min_samples_to_pause must be >= 1, got {min_samples_to_pause}")

        self.rolling_window: int = rolling_window
        self.auto_pause_if_negative: bool = auto_pause_if_negative
        self.min_samples_to_pause: int = min_samples_to_pause

        # deque automatically enforces the rolling window size
        self._pnl_buffer: Deque[float] = deque(maxlen=rolling_window)

        logger.debug(
            "ExpectedValueCalculator initialised: window=%d, auto_pause=%s, min_samples=%d",
            rolling_window,
            auto_pause_if_negative,
            min_samples_to_pause,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, pnl_pct: float) -> None:
        """
        Add a single closed-trade P&L percentage to the rolling window.

        Parameters
        ----------
        pnl_pct : float
            Realised P&L as a percentage of the trade's notional value.
            Positive for winning trades, negative for losing trades.
            E.g. 2.5 represents a 2.5 % gain; -1.3 represents a 1.3 % loss.
        """
        self._pnl_buffer.append(float(pnl_pct))
        logger.debug(
            "EV update: pnl=%.4f%% | buffer size=%d / %d",
            pnl_pct,
            len(self._pnl_buffer),
            self.rolling_window,
        )

    def calculate(self) -> EVResult:
        """
        Compute Expected Value statistics from the current rolling window.

        Returns
        -------
        EVResult
            All metrics are zero / False when the buffer is empty.

        Notes
        -----
        Trades with pnl_pct == 0.0 are counted as losses (breakeven is not
        a win — fees make them net negative).
        """
        data: List[float] = list(self._pnl_buffer)
        n = len(data)

        if n == 0:
            return EVResult(
                ev=0.0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                sample_size=0,
                is_positive=False,
            )

        wins = [p for p in data if p > 0.0]
        losses = [abs(p) for p in data if p <= 0.0]

        win_count = len(wins)
        loss_count = len(losses)

        win_rate: float = win_count / n
        loss_rate: float = 1.0 - win_rate

        avg_win: float = sum(wins) / win_count if win_count > 0 else 0.0
        avg_loss: float = sum(losses) / loss_count if loss_count > 0 else 0.0

        ev: float = (win_rate * avg_win) - (loss_rate * avg_loss)

        # Profit factor: total gross profit / total gross loss
        gross_profit: float = sum(wins)
        gross_loss: float = sum(losses)

        if gross_loss == 0.0:
            profit_factor = float("inf") if gross_profit > 0.0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss

        result = EVResult(
            ev=round(ev, 6),
            win_rate=round(win_rate, 6),
            avg_win=round(avg_win, 6),
            avg_loss=round(avg_loss, 6),
            profit_factor=profit_factor,
            sample_size=n,
            is_positive=ev > 0.0,
        )

        logger.debug(
            "EV calculated: %s",
            result,
        )

        return result

    def should_pause(self) -> bool:
        """
        Return True if new trading should be paused.

        Pause conditions (ALL must be met):
        1. ``auto_pause_if_negative`` is True.
        2. Rolling EV is strictly negative.
        3. Sample size >= ``min_samples_to_pause``.

        Returns
        -------
        bool
        """
        if not self.auto_pause_if_negative:
            return False

        result = self.calculate()

        if result.sample_size < self.min_samples_to_pause:
            logger.debug(
                "EV auto-pause: insufficient samples (%d < %d). No pause.",
                result.sample_size,
                self.min_samples_to_pause,
            )
            return False

        should = not result.is_positive
        if should:
            logger.warning(
                "EV auto-pause triggered: EV=%.4f%% (negative) | "
                "win_rate=%.2f%% | sample_size=%d",
                result.ev,
                result.win_rate * 100.0,
                result.sample_size,
            )
        return should

    def to_dict(self) -> dict:
        """
        Return a JSON-serialisable representation of the calculator's
        current state (statistics + configuration).

        Returns
        -------
        dict
        """
        stats = self.calculate()
        return {
            "stats": stats.to_dict(),
            "config": {
                "rolling_window": self.rolling_window,
                "auto_pause_if_negative": self.auto_pause_if_negative,
                "min_samples_to_pause": self.min_samples_to_pause,
            },
            "buffer_size": len(self._pnl_buffer),
        }

    # ------------------------------------------------------------------
    # Additional helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the rolling buffer (useful after a strategy reconfiguration)."""
        self._pnl_buffer.clear()
        logger.info("ExpectedValueCalculator buffer reset.")

    def get_recent_pnls(self) -> List[float]:
        """Return a copy of the current rolling P&L buffer (oldest first)."""
        return list(self._pnl_buffer)

    @property
    def sample_size(self) -> int:
        """Current number of trades in the rolling window."""
        return len(self._pnl_buffer)
