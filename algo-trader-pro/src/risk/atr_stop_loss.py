"""
src/risk/atr_stop_loss.py

ATR-based dynamic stop-loss and take-profit calculator.

Uses pandas_ta for ATR computation.  All price inputs/outputs are in the
same unit as the OHLCV data (e.g. USDT for crypto pairs).

Public API
----------
    calculate_atr(df, period=14)                                -> pd.Series
    calculate_stop(entry, atr, direction, multiplier, high_vol) -> float
    calculate_take_profit(entry, stop, direction, rr=2.5)       -> float
    is_high_volatility(df, threshold_pct=5.0)                   -> bool
    get_multiplier(is_high_vol, base=2.0, high_vol=3.0)         -> float
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import pandas_ta as ta  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_ATR_PERIOD: int = 14
_DEFAULT_RISK_REWARD_RATIO: float = 2.5
_DEFAULT_HIGH_VOL_THRESHOLD_PCT: float = 5.0
_DEFAULT_BASE_MULTIPLIER: float = 2.0
_DEFAULT_HIGH_VOL_MULTIPLIER: float = 3.0


# ---------------------------------------------------------------------------
# ATR calculation
# ---------------------------------------------------------------------------

def calculate_atr(df: pd.DataFrame, period: int = _DEFAULT_ATR_PERIOD) -> pd.Series:
    """
    Compute the Average True Range for *df* using pandas_ta.

    Parameters
    ----------
    df     : pd.DataFrame
        Must contain columns ``high``, ``low``, ``close`` (case-insensitive).
    period : int
        Lookback period for ATR (default 14).

    Returns
    -------
    pd.Series
        ATR values indexed like *df*.  The first ``period - 1`` values will
        be NaN (insufficient data for the calculation).

    Raises
    ------
    KeyError
        If required columns are missing from *df*.
    ValueError
        If *period* is less than 2.
    """
    if period < 2:
        raise ValueError(f"ATR period must be >= 2, got {period}")

    # Normalise column names to lower case
    df_norm = df.rename(columns=str.lower)
    required = {"high", "low", "close"}
    missing = required - set(df_norm.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {missing}")

    atr_series: pd.Series = ta.atr(
        high=df_norm["high"],
        low=df_norm["low"],
        close=df_norm["close"],
        length=period,
    )

    if atr_series is None:
        # pandas_ta returns None when it cannot compute – return all-NaN series
        logger.warning("pandas_ta.atr returned None for period=%d", period)
        return pd.Series([float("nan")] * len(df), index=df.index, name="ATR")

    return atr_series


# ---------------------------------------------------------------------------
# Volatility detection
# ---------------------------------------------------------------------------

def is_high_volatility(
    df: pd.DataFrame,
    threshold_pct: float = _DEFAULT_HIGH_VOL_THRESHOLD_PCT,
) -> bool:
    """
    Determine whether the most recent candle exhibits high volatility.

    High volatility is defined as the candle's high-low range (as a
    percentage of the low) exceeding *threshold_pct*.

    Parameters
    ----------
    df            : pd.DataFrame
        OHLCV data; uses the *last* row.
    threshold_pct : float
        Percentage threshold for the high-low daily range (default 5.0 %).

    Returns
    -------
    bool
    """
    if df.empty:
        return False

    df_norm = df.rename(columns=str.lower)
    last = df_norm.iloc[-1]

    high: float = float(last["high"])
    low: float = float(last["low"])

    if low <= 0:
        logger.warning("is_high_volatility: low price is zero or negative, skipping.")
        return False

    range_pct: float = (high - low) / low * 100.0

    is_high: bool = range_pct > threshold_pct
    logger.debug(
        "Volatility check: high=%.4f low=%.4f range_pct=%.2f%% threshold=%.2f%% -> %s",
        high,
        low,
        range_pct,
        threshold_pct,
        "HIGH" if is_high else "NORMAL",
    )
    return is_high


# ---------------------------------------------------------------------------
# Multiplier selection
# ---------------------------------------------------------------------------

def get_multiplier(
    is_high_vol: bool,
    base_mult: float = _DEFAULT_BASE_MULTIPLIER,
    high_vol_mult: float = _DEFAULT_HIGH_VOL_MULTIPLIER,
) -> float:
    """
    Return the ATR multiplier appropriate for the current volatility regime.

    In high-volatility conditions a larger multiplier gives the position
    more room to breathe and avoids premature stop-outs.

    Parameters
    ----------
    is_high_vol   : bool
    base_mult     : float   ATR multiplier in normal conditions (default 2.0)
    high_vol_mult : float   ATR multiplier in high-vol conditions  (default 3.0)

    Returns
    -------
    float
    """
    multiplier = high_vol_mult if is_high_vol else base_mult
    logger.debug(
        "ATR multiplier selected: %.1f (high_vol=%s)", multiplier, is_high_vol
    )
    return multiplier


# ---------------------------------------------------------------------------
# Stop-loss calculation
# ---------------------------------------------------------------------------

def calculate_stop(
    entry_price: float,
    atr: float,
    direction: str,
    multiplier: float,
    is_high_volatility_flag: bool = False,
) -> float:
    """
    Calculate the initial stop-loss price based on ATR.

    Stop = entry ± (ATR × multiplier)

    For LONG  positions the stop is *below* entry.
    For SHORT positions the stop is *above* entry.

    Parameters
    ----------
    entry_price          : float  – approximate fill price
    atr                  : float  – current ATR value
    direction            : str    – ``'LONG'`` or ``'SHORT'`` (case-insensitive)
    multiplier           : float  – ATR multiplier (use get_multiplier())
    is_high_volatility_flag : bool – unused but accepted for signature parity;
                                     callers should pass get_multiplier() result

    Returns
    -------
    float
        Stop-loss price.

    Raises
    ------
    ValueError
        If entry_price or atr is non-positive, or direction is invalid.
    """
    if entry_price <= 0:
        raise ValueError(f"entry_price must be > 0, got {entry_price}")
    if atr <= 0:
        raise ValueError(f"atr must be > 0, got {atr}")
    if multiplier <= 0:
        raise ValueError(f"multiplier must be > 0, got {multiplier}")

    direction_upper = direction.upper()
    if direction_upper not in {"LONG", "SHORT"}:
        raise ValueError(f"direction must be 'LONG' or 'SHORT', got {direction!r}")

    offset = atr * multiplier

    if direction_upper == "LONG":
        stop_price = entry_price - offset
    else:  # SHORT
        stop_price = entry_price + offset

    logger.debug(
        "Stop-loss: entry=%.4f atr=%.4f mult=%.2f dir=%s stop=%.4f",
        entry_price,
        atr,
        multiplier,
        direction_upper,
        stop_price,
    )
    return stop_price


# ---------------------------------------------------------------------------
# Take-profit calculation
# ---------------------------------------------------------------------------

def calculate_take_profit(
    entry_price: float,
    stop_price: float,
    direction: str,
    risk_reward_ratio: float = _DEFAULT_RISK_REWARD_RATIO,
) -> float:
    """
    Calculate the take-profit price from the risk/reward ratio.

    TP is placed such that the potential profit is exactly
    ``risk_reward_ratio`` times the initial risk distance.

        risk_distance = |entry_price − stop_price|
        TP (LONG)  = entry + risk_distance × risk_reward_ratio
        TP (SHORT) = entry − risk_distance × risk_reward_ratio

    Parameters
    ----------
    entry_price       : float
    stop_price        : float
    direction         : str    – ``'LONG'`` or ``'SHORT'`` (case-insensitive)
    risk_reward_ratio : float  – target R:R (default 2.5)

    Returns
    -------
    float
        Take-profit price.

    Raises
    ------
    ValueError
        If inputs are invalid or the stop is on the wrong side of entry.
    """
    if entry_price <= 0:
        raise ValueError(f"entry_price must be > 0, got {entry_price}")
    if risk_reward_ratio <= 0:
        raise ValueError(f"risk_reward_ratio must be > 0, got {risk_reward_ratio}")

    direction_upper = direction.upper()
    if direction_upper not in {"LONG", "SHORT"}:
        raise ValueError(f"direction must be 'LONG' or 'SHORT', got {direction!r}")

    risk_distance = abs(entry_price - stop_price)
    if risk_distance == 0:
        raise ValueError("entry_price and stop_price cannot be identical.")

    # Sanity: stop should be on the right side of entry
    if direction_upper == "LONG" and stop_price >= entry_price:
        raise ValueError(
            f"For LONG, stop_price ({stop_price}) must be < entry_price ({entry_price})."
        )
    if direction_upper == "SHORT" and stop_price <= entry_price:
        raise ValueError(
            f"For SHORT, stop_price ({stop_price}) must be > entry_price ({entry_price})."
        )

    reward_distance = risk_distance * risk_reward_ratio

    if direction_upper == "LONG":
        tp_price = entry_price + reward_distance
    else:  # SHORT
        tp_price = entry_price - reward_distance

    logger.debug(
        "Take-profit: entry=%.4f stop=%.4f rr=%.2f dir=%s tp=%.4f",
        entry_price,
        stop_price,
        risk_reward_ratio,
        direction_upper,
        tp_price,
    )
    return tp_price
