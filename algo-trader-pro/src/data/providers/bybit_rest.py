"""
src/data/providers/bybit_rest.py

Synchronous Bybit V5 REST API client for historical OHLCV data and
current price queries.

Bybit V5 kline endpoint
-----------------------
GET https://api.bybit.com/v5/market/kline
  category  = "spot"
  symbol    = "BTCUSDT"
  interval  = "1" | "5" | "15" | "60" | "240" | "D"
  start     = Unix ms (inclusive)
  end       = Unix ms (inclusive)
  limit     = 1..200 (max per request)

Response (result.list) is an array of arrays, NEWEST FIRST:
  [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]

All seven fields are strings – cast to the appropriate numeric type.

Column contract (matches BaseProvider / CoinGecko provider)
------------------------------------------------------------
open_time   int64   Unix ms
open        float64
high        float64
low         float64
close       float64
volume      float64
close_time  int64   Unix ms
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.bybit.com"
KLINE_ENDPOINT = "/v5/market/kline"
TICKER_ENDPOINT = "/v5/market/tickers"

MAX_LIMIT_PER_REQUEST = 200   # Bybit hard cap per request
RATE_LIMIT_SLEEP_S = 0.2      # 200 ms between requests (≈5 req/s, well under limits)
MAX_RETRIES = 3
BASE_BACKOFF_S = 1.0          # seconds; doubles on each retry

# Timeframe string (used by the rest of the codebase) → Bybit interval code
INTERVAL_MAP: Dict[str, str] = {
    "1m":  "1",
    "5m":  "5",
    "15m": "15",
    "1h":  "60",
    "4h":  "240",
    "1d":  "D",
}

# Candle duration in milliseconds, used to compute close_time and paginate
INTERVAL_MS: Dict[str, int] = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}

# CoinGecko ID → Bybit symbol (used when callers pass CoinGecko IDs directly)
COINGECKO_TO_BYBIT: Dict[str, str] = {
    "bitcoin":     "BTCUSDT",
    "ethereum":    "ETHUSDT",
    "solana":      "SOLUSDT",
    "binancecoin": "BNBUSDT",
}


# ---------------------------------------------------------------------------
# Helper: empty canonical DataFrame
# ---------------------------------------------------------------------------

def _empty_ohlcv_df() -> pd.DataFrame:
    """Return an empty DataFrame with the correct OHLCV schema."""
    return pd.DataFrame(
        columns=["open_time", "open", "high", "low", "close", "volume", "close_time"]
    ).astype(
        {
            "open_time":  "int64",
            "open":       "float64",
            "high":       "float64",
            "low":        "float64",
            "close":      "float64",
            "volume":     "float64",
            "close_time": "int64",
        }
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class BybitRESTProvider:
    """
    Synchronous Bybit V5 REST client for historical OHLCV and price data.

    Authentication
    --------------
    Market-data endpoints (kline, tickers) do not require an API key.
    BYBIT_API_KEY and BYBIT_API_SECRET are read from the environment and
    stored for use when signed endpoints are eventually needed.

    Parameters
    ----------
    base_url : str
        Override the base URL (e.g. for testnet
        ``"https://api-testnet.bybit.com"``).
    timeout  : int
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: int = 30,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

        # Read credentials from environment (market data endpoints don't need them)
        self._api_key: str = os.environ.get("BYBIT_API_KEY", "")
        self._api_secret: str = os.environ.get("BYBIT_API_SECRET", "")

        if self._api_key:
            logger.debug("BybitRESTProvider: API key loaded from environment.")
        else:
            logger.debug("BybitRESTProvider: No API key found – public endpoints only.")

        # Build a requests.Session with automatic low-level retries for
        # connection-level failures (not application-level HTTP errors).
        self._session = self._build_session()

        # Track the last outgoing request time for rate limiting
        self._last_request_ts: float = 0.0

    # ------------------------------------------------------------------
    # Session setup
    # ------------------------------------------------------------------

    def _build_session(self) -> requests.Session:
        """Create a requests.Session with a retry-enabled HTTPAdapter."""
        session = requests.Session()
        retry_strategy = Retry(
            total=0,           # Application-level retries handled separately
            backoff_factor=0,
            status_forcelist=[],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        return session

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """Block until at least RATE_LIMIT_SLEEP_S has passed since the last request."""
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < RATE_LIMIT_SLEEP_S:
            time.sleep(RATE_LIMIT_SLEEP_S - elapsed)

    def _get(self, endpoint: str, params: dict) -> dict:
        """
        Perform a GET request with exponential-backoff retry.

        Parameters
        ----------
        endpoint : str
            Path relative to base_url, e.g. ``"/v5/market/kline"``.
        params   : dict
            Query parameters.

        Returns
        -------
        dict
            Parsed JSON response body.

        Raises
        ------
        RuntimeError
            If the request fails after MAX_RETRIES attempts or the API
            returns a non-zero retCode.
        """
        url = f"{self._base_url}{endpoint}"
        last_error: Exception = RuntimeError("Unknown error")

        for attempt in range(1, MAX_RETRIES + 1):
            self._throttle()
            try:
                logger.debug(
                    "GET %s params=%s (attempt %d/%d)",
                    url, params, attempt, MAX_RETRIES
                )
                response = self._session.get(url, params=params, timeout=self._timeout)
                self._last_request_ts = time.monotonic()

                # Handle HTTP-level errors
                if response.status_code == 429:
                    retry_after = float(
                        response.headers.get("Retry-After", BASE_BACKOFF_S * (2 ** attempt))
                    )
                    logger.warning(
                        "Rate limited by Bybit (HTTP 429). Sleeping %.1fs.", retry_after
                    )
                    time.sleep(retry_after)
                    continue

                if response.status_code != 200:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:300]}"
                    )

                data = response.json()

                # Bybit application-level error
                ret_code = data.get("retCode", -1)
                if ret_code != 0:
                    ret_msg = data.get("retMsg", "unknown")
                    raise RuntimeError(
                        f"Bybit API error retCode={ret_code}: {ret_msg}"
                    )

                return data

            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as exc:
                last_error = exc
                backoff = BASE_BACKOFF_S * (2 ** (attempt - 1))
                logger.warning(
                    "Request failed (%s). Retrying in %.1fs (attempt %d/%d).",
                    exc, backoff, attempt, MAX_RETRIES,
                )
                time.sleep(backoff)
            except RuntimeError as exc:
                last_error = exc
                backoff = BASE_BACKOFF_S * (2 ** (attempt - 1))
                logger.warning(
                    "API error (%s). Retrying in %.1fs (attempt %d/%d).",
                    exc, backoff, attempt, MAX_RETRIES,
                )
                time.sleep(backoff)

        raise RuntimeError(
            f"Bybit request to {url} failed after {MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    def _resolve_symbol(self, symbol: str) -> str:
        """
        Resolve a symbol string to a Bybit trading pair.

        Accepts either:
        - A CoinGecko ID   (e.g. ``"bitcoin"``) → mapped to ``"BTCUSDT"``
        - A Bybit symbol   (e.g. ``"BTCUSDT"``) → returned as-is (uppercased)
        """
        lower = symbol.lower()
        if lower in COINGECKO_TO_BYBIT:
            return COINGECKO_TO_BYBIT[lower]
        return symbol.upper()

    def _parse_kline_row(
        self,
        row: List[str],
        interval_ms: int,
    ) -> dict:
        """
        Parse a single Bybit kline array into a canonical dict.

        Bybit row format (all strings):
          [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]

        Parameters
        ----------
        row         : list of str
        interval_ms : candle duration in milliseconds (used to compute close_time)

        Returns
        -------
        dict with keys: open_time, open, high, low, close, volume, close_time
        """
        start_time_ms = int(row[0])
        return {
            "open_time":  start_time_ms,
            "open":       float(row[1]),
            "high":       float(row[2]),
            "low":        float(row[3]),
            "close":      float(row[4]),
            "volume":     float(row[5]),
            "close_time": start_time_ms + interval_ms - 1,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 2,
    ) -> pd.DataFrame:
        """
        Fetch the latest N kline bars (for polling). Returns newest-first.

        Parameters
        ----------
        symbol : str
        interval : str
            Bybit interval code ("1", "5", "15", "60", "240", "D") or
            string like "1h" (mapped to "60").
        limit : int
            Number of bars to fetch (default 2: latest closed + forming).
        """
        tf = str(interval)
        # Mappa codici Bybit → stringhe: 60→1h, 240→4h, D→1d
        bybit_to_str = {"60": "1h", "240": "4h", "D": "1d", "1": "1m", "5": "5m", "15": "15m"}
        tf = bybit_to_str.get(tf, tf)
        if tf not in INTERVAL_MAP:
            tf = "1h"
        interval_ms = INTERVAL_MS.get(tf, INTERVAL_MS["1h"])
        until_ms = int(time.time() * 1000)
        since_ms = until_ms - (limit * interval_ms)
        df = self.fetch_ohlcv(symbol, tf, since_ms, until_ms)
        if df is None or df.empty:
            return pd.DataFrame()
        return df.tail(limit).iloc[::-1].reset_index(drop=True)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        until_ms: int,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV candles for a given symbol and timeframe.

        Paginates automatically in 200-candle chunks, sleeping 200 ms between
        requests.  The Bybit API returns candles in reverse-chronological order
        (newest first); this method returns them sorted ascending by open_time.

        Parameters
        ----------
        symbol    : str
            Bybit symbol (e.g. ``"BTCUSDT"``) or CoinGecko ID (``"bitcoin"``).
        timeframe : str
            Candle interval.  Supported values: ``"1m"``, ``"5m"``, ``"15m"``,
            ``"1h"``, ``"4h"``, ``"1d"``.
        since_ms  : int
            Start of the requested range (Unix milliseconds, inclusive).
        until_ms  : int
            End of the requested range (Unix milliseconds, inclusive).

        Returns
        -------
        pd.DataFrame
            Columns: open_time, open, high, low, close, volume, close_time.
            Sorted ascending by open_time.  Empty DataFrame if no data found.

        Raises
        ------
        ValueError
            If *timeframe* is not supported.
        RuntimeError
            If the API request fails after all retry attempts.
        """
        bybit_symbol = self._resolve_symbol(symbol)
        interval_code = INTERVAL_MAP.get(timeframe)
        if interval_code is None:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported: {list(INTERVAL_MAP.keys())}"
            )
        interval_ms = INTERVAL_MS[timeframe]

        all_rows: List[dict] = []
        # Bybit returns candles NEWEST FIRST. Paginate backwards: request
        # [since_ms, current_end], get the 200 most recent; then set
        # current_end = oldest_in_batch - 1 to fetch older data.
        current_end = until_ms

        logger.info(
            "Fetching OHLCV: symbol=%s timeframe=%s since=%d until=%d",
            bybit_symbol, timeframe, since_ms, until_ms,
        )

        while current_end >= since_ms:
            params = {
                "category": "spot",
                "symbol":   bybit_symbol,
                "interval": interval_code,
                "start":    since_ms,
                "end":      current_end,
                "limit":    MAX_LIMIT_PER_REQUEST,
            }

            data = self._get(KLINE_ENDPOINT, params)
            raw_list: List[List[str]] = data.get("result", {}).get("list", [])

            if not raw_list:
                logger.debug("Empty response – pagination complete.")
                break

            # raw_list is NEWEST FIRST – reverse to process oldest first
            raw_list_asc = list(reversed(raw_list))

            batch_rows = [
                self._parse_kline_row(row, interval_ms) for row in raw_list_asc
            ]

            # Filter to requested range (Bybit may return candles slightly
            # outside the requested window)
            batch_rows = [
                r for r in batch_rows
                if since_ms <= r["open_time"] <= until_ms
            ]

            if not batch_rows:
                logger.debug("All batch rows outside requested range – stopping.")
                break

            all_rows.extend(batch_rows)

            # Oldest candle in this batch – next request fetches data before it
            oldest_open_time = min(r["open_time"] for r in batch_rows)
            next_end = oldest_open_time - interval_ms

            logger.debug(
                "Batch: %d candles fetched (oldest=%d, newest=%d). Next end=%d",
                len(batch_rows),
                oldest_open_time,
                max(r["open_time"] for r in batch_rows),
                next_end,
            )

            if next_end < since_ms:
                break

            # If Bybit returned fewer candles than the limit, no more data
            if len(raw_list) < MAX_LIMIT_PER_REQUEST:
                logger.debug(
                    "Received %d < %d candles – no more data available.",
                    len(raw_list), MAX_LIMIT_PER_REQUEST,
                )
                break

            current_end = next_end

        if not all_rows:
            logger.info("No candles returned for %s %s.", bybit_symbol, timeframe)
            return _empty_ohlcv_df()

        df = pd.DataFrame(all_rows)
        df = (
            df.sort_values("open_time")
            .drop_duplicates(subset="open_time")
            .reset_index(drop=True)
        )

        # Ensure correct dtypes
        df["open_time"]  = df["open_time"].astype("int64")
        df["close_time"] = df["close_time"].astype("int64")
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype("float64")

        logger.info(
            "fetch_ohlcv complete: %d candles for %s/%s",
            len(df), bybit_symbol, timeframe,
        )
        return df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]

    def fetch_current_price(self, symbol: str) -> float:
        """
        Return the latest traded price for *symbol*.

        Uses the Bybit V5 ``/v5/market/tickers`` endpoint (spot category).

        Parameters
        ----------
        symbol : str
            Bybit symbol (e.g. ``"BTCUSDT"``) or CoinGecko ID (``"bitcoin"``).

        Returns
        -------
        float
            Last traded price in USDT.

        Raises
        ------
        RuntimeError
            If the API request fails or the symbol is not found in the response.
        """
        bybit_symbol = self._resolve_symbol(symbol)
        params = {
            "category": "spot",
            "symbol":   bybit_symbol,
        }

        data = self._get(TICKER_ENDPOINT, params)

        try:
            tickers: List[dict] = data["result"]["list"]
            if not tickers:
                raise RuntimeError(
                    f"No ticker data returned for symbol '{bybit_symbol}'."
                )
            last_price = float(tickers[0]["lastPrice"])
            logger.debug(
                "Current price for %s: %.8f", bybit_symbol, last_price
            )
            return last_price
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to parse ticker response for '{bybit_symbol}': {data}"
            ) from exc

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()
        logger.debug("BybitRESTProvider: session closed.")

    def __enter__(self) -> "BybitRESTProvider":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"BybitRESTProvider(base_url={self._base_url!r}, "
            f"auth={'yes' if self._api_key else 'no'})"
        )
