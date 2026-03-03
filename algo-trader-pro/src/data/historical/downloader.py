"""
src/data/historical/downloader.py

Historical OHLCV downloader with intelligent cache management.

The downloader coordinates between:
  - BybitRESTProvider  (network: fetches raw candles from Bybit)
  - OHLCVStore         (disk: Parquet cache)

Strategy
--------
1. Check whether a cache file already exists for (symbol, timeframe).
2. If **no cache** → download from start_date to end_date in full.
3. If **cache exists**:
   a. Compare the latest cached timestamp to the requested end_date.
   b. If the cache covers the full requested range → return cached data.
   c. Otherwise do an **incremental update** from (latest_cached_ts + 1 ms)
      to end_date, merge with the existing cache, and persist.
4. Save the result to the Parquet store.
5. Return the complete DataFrame filtered to [start_date, end_date].
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from src.data.cache.ohlcv_store import OHLCVStore
from src.data.providers.bybit_rest import BybitRESTProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Candle-count estimation: seconds per timeframe
# ---------------------------------------------------------------------------

TIMEFRAME_SECONDS: Dict[str, int] = {
    "1m":  60,
    "5m":  300,
    "15m": 900,
    "1h":  3_600,
    "4h":  14_400,
    "1d":  86_400,
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_date(date_str: str) -> datetime:
    """
    Parse a date string in ``YYYY-MM-DD`` or ``YYYY-MM-DDTHH:MM:SS`` format
    into a timezone-aware UTC datetime.

    Parameters
    ----------
    date_str : str

    Returns
    -------
    datetime (UTC-aware)

    Raises
    ------
    ValueError
        If the string cannot be parsed.
    """
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse date string '{date_str}'. "
        "Expected format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS"
    )


def _dt_to_ms(dt: datetime) -> int:
    """Convert a timezone-aware or naive UTC datetime to Unix milliseconds."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# ---------------------------------------------------------------------------
# HistoricalDownloader
# ---------------------------------------------------------------------------

class HistoricalDownloader:
    """
    Downloads and caches historical OHLCV data using Bybit REST + local Parquet.

    Parameters
    ----------
    provider : BybitRESTProvider
        Configured Bybit REST client.
    store    : OHLCVStore
        Parquet-backed cache store.
    config   : object (optional)
        Application config object.  Not used directly by the downloader
        but stored for future extension (e.g. reading timeouts, symbols).
    """

    def __init__(
        self,
        provider: BybitRESTProvider,
        store: OHLCVStore,
        config=None,
    ) -> None:
        self._provider = provider
        self._store    = store
        self._config   = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_symbol(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Ensure the cache is complete for (symbol, timeframe, [start, end]).

        Logic
        -----
        1. If no cache → full download.
        2. If cache exists and already covers end_date → return cached slice.
        3. If cache exists but is stale → incremental update from
           (latest_cached_ts + 1 ms) to end_date.

        Parameters
        ----------
        symbol     : str
            Bybit symbol or CoinGecko ID, e.g. ``"BTCUSDT"`` or ``"bitcoin"``.
        timeframe  : str
            Candle interval, e.g. ``"1h"``.
        start_date : str
            Inclusive start date in ``YYYY-MM-DD`` format.
        end_date   : str
            Inclusive end date in ``YYYY-MM-DD`` format.

        Returns
        -------
        pd.DataFrame
            Complete OHLCV data for the requested range, sorted ascending
            by open_time.  May contain slightly more rows than estimated if
            the cache already held extra data.
        """
        start_dt = _parse_date(start_date)
        end_dt   = _parse_date(end_date)
        since_ms = _dt_to_ms(start_dt)
        until_ms = _dt_to_ms(end_dt)

        estimated = self.estimate_candles(start_date, end_date, timeframe)
        logger.info(
            "download_symbol: %s/%s  %s → %s  (~%d candles estimated)",
            symbol, timeframe, start_date, end_date, estimated,
        )

        # --- Step 1: inspect cache ---
        latest_cached_ts = self._store.get_latest_timestamp(symbol, timeframe)

        if latest_cached_ts is None:
            # No cache exists → full download
            logger.info(
                "No cache for %s/%s – performing full download.", symbol, timeframe
            )
            new_df = self._fetch_range(symbol, timeframe, since_ms, until_ms)
            if not new_df.empty:
                self._store.save(new_df, symbol, timeframe)
        else:
            # --- Step 2: check coverage ---
            if latest_cached_ts >= until_ms:
                logger.info(
                    "Cache for %s/%s already covers end_date – no download needed.",
                    symbol, timeframe,
                )
                # Return from cache only; nothing to fetch
                return self._store.load(symbol, timeframe, since_ms, until_ms)

            # --- Step 3: incremental update ---
            update_since_ms = latest_cached_ts + 1  # exclusive of last cached candle
            logger.info(
                "Cache for %s/%s is stale. Incremental update from %d to %d.",
                symbol, timeframe, update_since_ms, until_ms,
            )
            new_df = self._fetch_range(symbol, timeframe, update_since_ms, until_ms)
            if not new_df.empty:
                self._store.save(new_df, symbol, timeframe)
            else:
                logger.info(
                    "Incremental update returned no new candles for %s/%s.",
                    symbol, timeframe,
                )

        # Load and return the filtered range from the (now up-to-date) cache
        return self._store.load(symbol, timeframe, since_ms, until_ms)

    def download_all(
        self,
        symbols: List[str],
        timeframes: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Download all combinations of (symbol, timeframe).

        Prints progress to the logger and returns a nested dict.

        Parameters
        ----------
        symbols    : list of str
        timeframes : list of str
        start_date : str
        end_date   : str

        Returns
        -------
        dict
            ``{symbol: {timeframe: DataFrame}}``
        """
        total = len(symbols) * len(timeframes)
        done  = 0
        results: Dict[str, Dict[str, pd.DataFrame]] = {}

        for symbol in symbols:
            results[symbol] = {}
            for timeframe in timeframes:
                done += 1
                estimated = self.estimate_candles(start_date, end_date, timeframe)
                logger.info(
                    "[%d/%d] Downloading %s/%s  (%s → %s, ~%d candles) …",
                    done, total, symbol, timeframe,
                    start_date, end_date, estimated,
                )
                print(
                    f"[{done}/{total}] {symbol}/{timeframe}  "
                    f"{start_date} → {end_date}  (~{estimated:,} candles)",
                    flush=True,
                )
                try:
                    df = self.download_symbol(symbol, timeframe, start_date, end_date)
                    results[symbol][timeframe] = df
                    print(
                        f"       OK – {len(df):,} candles loaded.",
                        flush=True,
                    )
                    logger.info(
                        "Completed %s/%s: %d candles.", symbol, timeframe, len(df)
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to download %s/%s: %s", symbol, timeframe, exc
                    )
                    print(f"       ERROR: {exc}", flush=True)
                    # Store empty DataFrame so the caller can check without KeyError
                    results[symbol][timeframe] = pd.DataFrame(
                        columns=[
                            "open_time", "open", "high", "low",
                            "close", "volume", "close_time",
                        ]
                    )

        logger.info("download_all complete: %d combinations processed.", total)
        print(f"\ndownload_all complete: {total} combinations processed.", flush=True)
        return results

    def estimate_candles(
        self,
        start_date: str,
        end_date: str,
        timeframe: str,
    ) -> int:
        """
        Estimate the number of candles in the requested range.

        Parameters
        ----------
        start_date : str    ``YYYY-MM-DD``
        end_date   : str    ``YYYY-MM-DD``
        timeframe  : str    e.g. ``"1h"``

        Returns
        -------
        int
            Approximate candle count.  Returns 0 if the timeframe is
            not recognised or the date range is invalid.
        """
        interval_s = TIMEFRAME_SECONDS.get(timeframe)
        if interval_s is None:
            logger.warning(
                "estimate_candles: unknown timeframe '%s' – returning 0.", timeframe
            )
            return 0

        try:
            start_dt = _parse_date(start_date)
            end_dt   = _parse_date(end_date)
        except ValueError:
            return 0

        duration_s = (end_dt - start_dt).total_seconds()
        if duration_s <= 0:
            return 0

        return max(0, int(duration_s / interval_s))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_range(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        until_ms: int,
    ) -> pd.DataFrame:
        """
        Fetch candles from the provider in a single call.

        The BybitRESTProvider.fetch_ohlcv() handles pagination internally
        (200-candle chunks), so one call here is sufficient regardless of
        range size.

        Parameters
        ----------
        symbol    : str
        timeframe : str
        since_ms  : int  Unix ms, inclusive
        until_ms  : int  Unix ms, inclusive

        Returns
        -------
        pd.DataFrame
            Raw OHLCV data; may be empty if no candles exist in the range.
        """
        try:
            df = self._provider.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since_ms=since_ms,
                until_ms=until_ms,
            )
            return df
        except Exception as exc:
            logger.error(
                "_fetch_range failed for %s/%s [%d, %d]: %s",
                symbol, timeframe, since_ms, until_ms, exc,
            )
            raise

    def get_cache_summary(self) -> None:
        """
        Print a summary of all data currently held in the cache to stdout.
        """
        available = self._store.list_available()
        if not available:
            print("Cache is empty.", flush=True)
            return

        print("\n--- Cache Summary ---", flush=True)
        for symbol, timeframes in sorted(available.items()):
            for tf in sorted(timeframes):
                stats = self._store.get_cache_stats(symbol, tf)
                min_ts = stats.get("min_open_time")
                max_ts = stats.get("max_open_time")
                min_dt = (
                    datetime.utcfromtimestamp(min_ts / 1000).strftime("%Y-%m-%d")
                    if min_ts else "N/A"
                )
                max_dt = (
                    datetime.utcfromtimestamp(max_ts / 1000).strftime("%Y-%m-%d")
                    if max_ts else "N/A"
                )
                print(
                    f"  {symbol:10s} {tf:5s}  {stats['rows']:>8,} rows  "
                    f"{min_dt} → {max_dt}  ({stats['file_size_kb']:.1f} KB)",
                    flush=True,
                )
        print("--- End of Cache Summary ---\n", flush=True)

    def __repr__(self) -> str:
        return (
            f"HistoricalDownloader("
            f"provider={self._provider!r}, "
            f"store={self._store!r})"
        )
