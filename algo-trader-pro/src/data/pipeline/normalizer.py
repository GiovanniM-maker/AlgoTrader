"""
Data normalization pipeline for OHLCV and sentiment data.

Responsibilities
----------------
* ``normalize_ohlcv``    – dtype enforcement, dedup, sort, gap-fill, validation,
                           derived column creation.
* ``normalize_sentiment`` – scale different sentiment sources to a common [0, 100]
                            range.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Maximum number of consecutive NaN / missing candles to forward-fill
MAX_FFILL_GAPS = 3

# Supported sentiment sources and their native ranges
SentimentSource = Literal["fear_greed", "cryptopanic", "google_trends"]

OHLCV_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time"]

FLOAT_COLS = ["open", "high", "low", "close", "volume"]
INT_COLS   = ["open_time", "close_time"]


# ---------------------------------------------------------------------------
# OHLCV normalisation
# ---------------------------------------------------------------------------


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a clean, fully normalised OHLCV DataFrame.

    Steps (in order)
    ----------------
    1. Enforce required columns exist.
    2. Cast dtypes (float64 for price/volume, int64 for timestamps).
    3. Remove duplicate rows on ``open_time``, keeping the last occurrence.
    4. Sort ascending by ``open_time``.
    5. Forward-fill up to ``MAX_FFILL_GAPS`` consecutive missing candles
       (gaps are detected by finding missing multiples of the modal interval).
    6. Validate OHLCV invariants (high ≥ low, volume ≥ 0, open/close within
       [low, high]).  Rows violating these constraints are logged and dropped.
    7. Add derived columns:
       - ``returns``       – percentage change of close (pct_change)
       - ``log_returns``   – log of (close_t / close_{t-1})
       - ``typical_price`` – (high + low + close) / 3

    Parameters
    ----------
    df:
        Raw OHLCV DataFrame.  Must contain at minimum the columns listed in
        ``OHLCV_COLUMNS``.

    Returns
    -------
    pd.DataFrame
        Normalised DataFrame with original + derived columns.  Empty DataFrame
        (correct dtypes) if input is empty.

    Raises
    ------
    ValueError
        If required columns are missing from *df*.
    """
    _check_required_columns(df, OHLCV_COLUMNS)

    if df.empty:
        return _empty_normalized_df()

    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Dtype enforcement
    # ------------------------------------------------------------------
    for col in FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in INT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")  # nullable int

    # Drop rows where open_time is NaN (cannot sort or deduplicate them)
    df = df.dropna(subset=["open_time"])
    df["open_time"]  = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].fillna(df["open_time"]).astype("int64")

    # ------------------------------------------------------------------
    # 2. Dedup & sort
    # ------------------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=["open_time"], keep="last")
    after = len(df)
    if before != after:
        logger.debug("normalize_ohlcv: dropped %d duplicate rows.", before - after)

    df = df.sort_values("open_time").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Gap filling (forward-fill ≤ MAX_FFILL_GAPS consecutive gaps)
    # ------------------------------------------------------------------
    df = _fill_gaps(df)

    # ------------------------------------------------------------------
    # 4. Validation – drop invalid rows
    # ------------------------------------------------------------------
    df = _validate_and_clean(df)

    # ------------------------------------------------------------------
    # 5. Derived columns
    # ------------------------------------------------------------------
    df["returns"]       = df["close"].pct_change()
    df["log_returns"]   = np.log(df["close"] / df["close"].shift(1))
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sentiment normalisation
# ---------------------------------------------------------------------------


def normalize_sentiment(value: float, source: SentimentSource) -> float:
    """
    Normalise a raw sentiment value to the common [0, 100] scale.

    Source rules
    ------------
    * ``"fear_greed"``    – raw range [0, 100] → pass-through, clamp to [0,100].
    * ``"cryptopanic"``   – raw range [-1, +1] → mapped linearly to [0, 100].
    * ``"google_trends"`` – raw range [0, 100] → pass-through, clamp to [0,100].

    Parameters
    ----------
    value:
        Raw sentiment value from the source.
    source:
        One of ``"fear_greed"``, ``"cryptopanic"``, ``"google_trends"``.

    Returns
    -------
    float
        Normalised score in [0.0, 100.0].

    Raises
    ------
    ValueError
        If *source* is not one of the supported sources.
    """
    if source == "fear_greed":
        return float(max(0.0, min(100.0, value)))

    if source == "cryptopanic":
        # Linear map: -1 → 0, 0 → 50, +1 → 100
        clamped = max(-1.0, min(1.0, value))
        return (clamped + 1.0) / 2.0 * 100.0

    if source == "google_trends":
        return float(max(0.0, min(100.0, value)))

    raise ValueError(
        f"Unknown sentiment source '{source}'. "
        f"Supported: 'fear_greed', 'cryptopanic', 'google_trends'."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_required_columns(df: pd.DataFrame, required: list) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}"
        )


def _empty_normalized_df() -> pd.DataFrame:
    """Return an empty DataFrame with normalized OHLCV + derived column dtypes."""
    cols = OHLCV_COLUMNS + ["returns", "log_returns", "typical_price"]
    dtypes = {
        "open_time":     "int64",
        "open":          "float64",
        "high":          "float64",
        "low":           "float64",
        "close":         "float64",
        "volume":        "float64",
        "close_time":    "int64",
        "returns":       "float64",
        "log_returns":   "float64",
        "typical_price": "float64",
    }
    df = pd.DataFrame(columns=cols)
    for col, dtype in dtypes.items():
        df[col] = df[col].astype(dtype)
    return df


def _fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and forward-fill gaps in the candle series.

    The dominant candle interval is inferred from the median time delta
    between consecutive ``open_time`` values.  Missing rows (multiples of
    that interval) are synthesised by forward-filling OHLCV values for up
    to ``MAX_FFILL_GAPS`` consecutive missing candles.
    """
    if len(df) < 2:
        return df

    # Compute the most common interval (median of diffs)
    diffs = df["open_time"].diff().dropna()
    if diffs.empty:
        return df

    interval_ms = int(diffs.median())
    if interval_ms <= 0:
        return df

    expected_times = pd.RangeIndex(
        start=int(df["open_time"].iloc[0]),
        stop=int(df["open_time"].iloc[-1]) + interval_ms,
        step=interval_ms,
    )

    # Reindex to the expected grid
    df = df.set_index("open_time")
    df = df.reindex(expected_times)

    # Count consecutive NaN blocks and only fill ≤ MAX_FFILL_GAPS
    is_missing = df["close"].isna()
    # Build a group ID that resets at every non-missing row
    group_ids = (~is_missing).cumsum()
    # Count missing within each group
    gap_count  = is_missing.groupby(group_ids).cumsum()

    # Forward-fill only where the gap is within the allowed limit
    df_ffill = df.ffill()
    mask = (gap_count <= MAX_FFILL_GAPS) & is_missing
    df = df.where(~mask, other=df_ffill)

    df.index.name = "open_time"
    df = df.reset_index()
    df["open_time"] = df["open_time"].astype("int64")

    # Drop rows that still have NaN (gaps larger than MAX_FFILL_GAPS)
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    # Recompute close_time for synthesised rows
    df["close_time"] = df["open_time"] + interval_ms - 1

    return df


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply OHLCV sanity checks and drop or warn on invalid rows.

    Invariants checked
    ------------------
    * high  ≥ low
    * high  ≥ open
    * high  ≥ close
    * low   ≤ open
    * low   ≤ close
    * volume ≥ 0
    """
    invalid_mask = (
        (df["high"]   <  df["low"])
        | (df["high"] <  df["open"])
        | (df["high"] <  df["close"])
        | (df["low"]  >  df["open"])
        | (df["low"]  >  df["close"])
        | (df["volume"] < 0)
    )

    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        logger.warning(
            "normalize_ohlcv: dropping %d rows that violate OHLCV invariants.",
            n_invalid,
        )
        df = df[~invalid_mask].reset_index(drop=True)

    return df
