"""
src/risk/time_exit.py

Time-based exit logic.

A position is flagged for time-based exit when:
    (current_time − entry_time) > max_hold_hours
    AND current P&L percentage < min_profit_pct

This prevents capital from being tied up indefinitely in stale trades
that are not meeting their profit target.

Usage
-----
    from src.risk.time_exit import should_exit

    exit_now = should_exit(
        entry_time=position.entry_time,
        current_time=clock.now(),
        current_pnl_pct=position.current_pnl_pct(current_price),
        max_hold_hours=48,
        min_profit_pct=0.5,
    )
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def should_exit(
    entry_time: datetime,
    current_time: datetime,
    current_pnl_pct: float,
    max_hold_hours: int,
    min_profit_pct: float,
) -> bool:
    """
    Determine whether a position should be exited on time-based grounds.

    Exit conditions (BOTH must be true):
        1. ``(current_time - entry_time).total_hours() > max_hold_hours``
        2. ``current_pnl_pct < min_profit_pct``

    Rationale
    ---------
    A position that has been open longer than expected AND has not reached
    the minimum profit threshold has likely failed to develop as anticipated.
    Exiting frees capital for better opportunities and avoids runaway losses
    from stale trades that may reverse against the original thesis.

    Parameters
    ----------
    entry_time     : datetime
        UTC time the position was opened (naive or tz-aware).
    current_time   : datetime
        Current UTC time from the clock abstraction (naive or tz-aware).
    current_pnl_pct: float
        Current unrealised P&L as a percentage of the position size.
        A value of 1.5 means the position is up 1.5 %;
        a value of -2.0 means it is down 2.0 %.
    max_hold_hours : int
        Maximum number of hours a position is allowed to be open before
        time-based rules apply (must be >= 1).
    min_profit_pct : float
        Minimum P&L percentage the position must show *at* or *after*
        max_hold_hours to avoid the time-based exit.
        A value of 0.5 means the position must be at least +0.5 % in profit.
        Set to a negative number to exit only deeply losing stale positions.

    Returns
    -------
    bool
        ``True``  – close the position on time grounds.
        ``False`` – no time-based exit warranted at this moment.

    Raises
    ------
    ValueError
        If max_hold_hours is less than 1.
    TypeError
        If entry_time or current_time are not datetime instances.
    """
    # --- Input validation ---------------------------------------------------
    if not isinstance(entry_time, datetime):
        raise TypeError(f"entry_time must be datetime, got {type(entry_time)}")
    if not isinstance(current_time, datetime):
        raise TypeError(f"current_time must be datetime, got {type(current_time)}")
    if max_hold_hours < 1:
        raise ValueError(f"max_hold_hours must be >= 1, got {max_hold_hours}")

    # --- Normalise timezone awareness ---------------------------------------
    # Strip tz info from both if necessary to allow comparison.
    # We assume both are UTC when naive.
    if entry_time.tzinfo is not None and current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=entry_time.tzinfo)
    elif entry_time.tzinfo is None and current_time.tzinfo is not None:
        entry_time = entry_time.replace(tzinfo=current_time.tzinfo)

    # --- Age calculation ----------------------------------------------------
    age_delta = current_time - entry_time
    age_hours: float = age_delta.total_seconds() / 3600.0

    if age_hours < 0:
        # Clock skew or bad data – do not exit.
        logger.warning(
            "should_exit: negative position age (%.2f h); ignoring time exit.",
            age_hours,
        )
        return False

    # --- Decision -----------------------------------------------------------
    over_max_hold: bool = age_hours > max_hold_hours
    below_min_profit: bool = current_pnl_pct < min_profit_pct

    exit_flag: bool = over_max_hold and below_min_profit

    logger.debug(
        "Time-exit check: age=%.2fh max_hold=%dh pnl=%.3f%% min_profit=%.3f%% "
        "over_hold=%s below_profit=%s -> exit=%s",
        age_hours,
        max_hold_hours,
        current_pnl_pct,
        min_profit_pct,
        over_max_hold,
        below_min_profit,
        exit_flag,
    )

    return exit_flag


def position_age_hours(entry_time: datetime, current_time: datetime) -> float:
    """
    Convenience function: return how many hours a position has been open.

    Parameters
    ----------
    entry_time   : datetime
    current_time : datetime

    Returns
    -------
    float
        Hours since entry_time.  Always >= 0.
    """
    if not isinstance(entry_time, datetime) or not isinstance(current_time, datetime):
        raise TypeError("Both entry_time and current_time must be datetime instances.")

    # Normalise tz awareness (same as should_exit)
    if entry_time.tzinfo is not None and current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=entry_time.tzinfo)
    elif entry_time.tzinfo is None and current_time.tzinfo is not None:
        entry_time = entry_time.replace(tzinfo=current_time.tzinfo)

    delta = current_time - entry_time
    return max(0.0, delta.total_seconds() / 3600.0)
