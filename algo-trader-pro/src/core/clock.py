"""
src/core/clock.py

Unified time abstraction used throughout the engine.

Both LiveClock and BacktestClock implement BaseClock, which exposes:
    now()           -> datetime  (UTC)
    timestamp_ms()  -> int       (Unix milliseconds)

Module-level helpers
--------------------
get_clock()                      -> BaseClock (current singleton)
set_live_mode()                  -> switch to / return LiveClock
set_backtest_mode(start_time)    -> switch to BacktestClock, set start time
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseClock(ABC):
    """Interface that every clock implementation must satisfy."""

    @abstractmethod
    def now(self) -> datetime:
        """Return the current (or simulated) UTC datetime."""

    def timestamp_ms(self) -> int:
        """Return the current time as Unix milliseconds (UTC)."""
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        now = self.now()
        # Normalise to UTC-aware if naive (assume UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        delta = now - epoch
        return int(delta.total_seconds() * 1000)


# ---------------------------------------------------------------------------
# LiveClock
# ---------------------------------------------------------------------------

class LiveClock(BaseClock):
    """
    Returns the real wall-clock UTC time.

    This is the default clock used in production mode.  It is stateless –
    every call to now() queries the system clock.
    """

    def now(self) -> datetime:
        """Return datetime.utcnow() (naive UTC)."""
        return datetime.utcnow()

    def __repr__(self) -> str:
        return "LiveClock()"


# ---------------------------------------------------------------------------
# BacktestClock
# ---------------------------------------------------------------------------

class BacktestClock(BaseClock):
    """
    Simulated clock used during back-testing and replay.

    The clock starts at the time supplied to the constructor (or via
    set_time) and advances only when set_time() or advance() are called
    explicitly.  All reads are thread-safe via an internal lock.

    Parameters
    ----------
    start_time : datetime
        Initial simulated UTC time (naive assumed UTC, or tz-aware).
    """

    def __init__(self, start_time: datetime) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._current: datetime = self._normalise(start_time)

    # ------------------------------------------------------------------
    # BaseClock interface
    # ------------------------------------------------------------------

    def now(self) -> datetime:
        """Return the current simulated UTC time (naive)."""
        with self._lock:
            return self._current

    # ------------------------------------------------------------------
    # Backtest-specific controls
    # ------------------------------------------------------------------

    def set_time(self, ts: datetime) -> None:
        """
        Jump the simulated clock to *ts*.

        Parameters
        ----------
        ts : datetime
            New simulated time; may be naive (assumed UTC) or tz-aware.

        Raises
        ------
        ValueError
            If *ts* is earlier than the current simulated time (clocks do
            not go backwards).
        """
        normalised = self._normalise(ts)
        with self._lock:
            if normalised < self._current:
                raise ValueError(
                    f"BacktestClock cannot go backwards: "
                    f"current={self._current.isoformat()}, "
                    f"requested={normalised.isoformat()}"
                )
            self._current = normalised

    def advance(self, delta: timedelta) -> datetime:
        """
        Move the clock forward by *delta*.

        Parameters
        ----------
        delta : timedelta
            Must be positive.

        Returns
        -------
        datetime
            The new simulated time after advancing.

        Raises
        ------
        ValueError
            If *delta* is negative or zero.
        """
        if delta <= timedelta(0):
            raise ValueError(f"delta must be positive, got {delta!r}")
        with self._lock:
            self._current += delta
            return self._current

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(ts: datetime) -> datetime:
        """Strip timezone info, converting to UTC naive if necessary."""
        if ts.tzinfo is not None:
            # Convert to UTC then strip tzinfo so we always store naive UTC.
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        return ts

    def __repr__(self) -> str:
        with self._lock:
            return f"BacktestClock(current={self._current.isoformat()})"


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------

_clock_lock: threading.Lock = threading.Lock()
_current_clock: Optional[BaseClock] = None


def get_clock() -> BaseClock:
    """
    Return the process-wide clock singleton.

    If no clock has been set yet, defaults to LiveClock.
    """
    global _current_clock
    if _current_clock is None:
        with _clock_lock:
            if _current_clock is None:
                _current_clock = LiveClock()
    return _current_clock


def set_live_mode() -> LiveClock:
    """
    Switch the singleton to a LiveClock and return it.

    Safe to call multiple times – returns the same LiveClock instance if
    the current clock is already a LiveClock.
    """
    global _current_clock
    with _clock_lock:
        if not isinstance(_current_clock, LiveClock):
            _current_clock = LiveClock()
        return _current_clock  # type: ignore[return-value]


def set_backtest_mode(start_time: datetime) -> BacktestClock:
    """
    Switch the singleton to a BacktestClock starting at *start_time*.

    A new BacktestClock is always created so that the simulated time
    resets to *start_time* on each call.

    Parameters
    ----------
    start_time : datetime
        The initial simulated UTC time for the back-test run.

    Returns
    -------
    BacktestClock
        The newly created and registered clock instance.
    """
    global _current_clock
    clock = BacktestClock(start_time)
    with _clock_lock:
        _current_clock = clock
    return clock


def _reset_clock() -> None:
    """
    Reset the singleton to None (for use in tests only).

    The next call to get_clock() will re-create a fresh LiveClock.
    """
    global _current_clock
    with _clock_lock:
        _current_clock = None
