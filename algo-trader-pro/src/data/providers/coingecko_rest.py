"""
Async CoinGecko REST data provider.

Uses the free CoinGecko public API (no API key required for the endpoints used).
Implements exponential-backoff retry logic and an in-memory request cache to
stay within the free-tier rate limit (~10-30 calls/minute).

Supported endpoints
-------------------
* /coins/{id}/ohlc                – short-range OHLCV (≤ 90 days for hourly)
* /coins/{id}/market_chart        – longer-range market data (hourly / daily)
* /simple/price                   – current spot price

Column contract (matches BaseProvider)
---------------------------------------
open_time   int64   Unix ms
open        float64
high        float64
low         float64
close       float64
volume      float64
close_time  int64   Unix ms
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.coingecko.com/api/v3"

# Maps unified symbol → CoinGecko coin ID
SYMBOL_TO_ID: Dict[str, str] = {
    "BTCUSDT":  "bitcoin",
    "ETHUSDT":  "ethereum",
    "SOLUSDT":  "solana",
    "BNBUSDT":  "binancecoin",
    "XRPUSDT":  "ripple",
    "ADAUSDT":  "cardano",
    "DOGEUSDT": "dogecoin",
    "AVAXUSDT": "avalanche-2",
    "DOTUSDT":  "polkadot",
    "MATICUSDT":"matic-network",
}

# Maps timeframe → (days_per_page, ohlc_days_param)
# CoinGecko /ohlc supports: 1, 7, 14, 30, 90, 180, 365 days
# For hourly OHLC data the cap is 90 days per request.
TIMEFRAME_CONFIG: Dict[str, Dict[str, Any]] = {
    "1m":  {"days_per_page": 1,   "granularity": "minutely"},
    "5m":  {"days_per_page": 1,   "granularity": "minutely"},
    "15m": {"days_per_page": 1,   "granularity": "minutely"},
    "30m": {"days_per_page": 7,   "granularity": "minutely"},
    "1h":  {"days_per_page": 90,  "granularity": "hourly"},
    "4h":  {"days_per_page": 90,  "granularity": "hourly"},
    "1d":  {"days_per_page": 365, "granularity": "daily"},
    "1w":  {"days_per_page": 365, "granularity": "daily"},
}

# Seconds per candle for building close_time and pagination
TIMEFRAME_SECONDS: Dict[str, int] = {
    "1m":  60,
    "5m":  300,
    "15m": 900,
    "30m": 1_800,
    "1h":  3_600,
    "4h":  14_400,
    "1d":  86_400,
    "1w":  604_800,
}

# Max retries and back-off settings
MAX_RETRIES = 3
BASE_BACKOFF_S = 2.0        # seconds
RATE_LIMIT_SLEEP_S = 6.5    # ~9 req/min to stay well under 30 req/min
CACHE_TTL_S = 300           # cache GET responses for 5 minutes (OHLCV, market_chart)
PRICE_CACHE_TTL_S = 30      # cache prezzi spot solo 30s per dashboard real-time


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class CoinGeckoRestProvider(BaseProvider):
    """Async CoinGecko REST provider with caching, retry and rate-limiting."""

    name = "CoinGecko REST"

    def __init__(self, vs_currency: str = "usd") -> None:
        """
        Parameters
        ----------
        vs_currency:
            Quote currency used for price queries (default ``"usd"``).
        """
        self.vs_currency = vs_currency
        self._session: Optional[aiohttp.ClientSession] = None
        # Simple dict cache: url → (response_data, expiry_unix_s)
        self._cache: Dict[str, Tuple[Any, float]] = {}
        # Timestamp of last outgoing request (for rate limiting)
        self._last_request_ts: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {"Accept": "application/json"}
            # Senza base_url per evitare problemi aiohttp su alcune versioni
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout,
            )
            logger.info("CoinGeckoRestProvider: HTTP session opened.")

    async def disconnect(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("CoinGeckoRestProvider: HTTP session closed.")
        self._session = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _symbol_to_id(self, symbol: str) -> str:
        coin_id = SYMBOL_TO_ID.get(symbol.upper())
        if coin_id is None:
            raise ValueError(
                f"Symbol '{symbol}' is not mapped. "
                f"Supported: {list(SYMBOL_TO_ID.keys())}"
            )
        return coin_id

    def _get_cache(self, url: str) -> Optional[Any]:
        entry = self._cache.get(url)
        if entry is None:
            return None
        data, expiry = entry
        if time.monotonic() < expiry:
            return data
        del self._cache[url]
        return None

    def _set_cache(self, url: str, data: Any, ttl: float = CACHE_TTL_S) -> None:
        self._cache[url] = (data, time.monotonic() + ttl)

    async def _throttle(self) -> None:
        """Ensure at least RATE_LIMIT_SLEEP_S seconds between requests."""
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < RATE_LIMIT_SLEEP_S:
            await asyncio.sleep(RATE_LIMIT_SLEEP_S - elapsed)

    async def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        cache_ttl: Optional[float] = None,
    ) -> Any:
        """
        Perform a GET request with retry + exponential back-off.

        Returns parsed JSON (list or dict).
        cache_ttl: override default cache TTL (seconds). None = use CACHE_TTL_S.
        """
        if self._session is None:
            raise RuntimeError("Provider is not connected. Call await connect() first.")

        # Build full URL string for cache key
        param_str = "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
        cache_key = f"{path}?{param_str}"
        ttl = cache_ttl if cache_ttl is not None else CACHE_TTL_S

        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.debug("Cache HIT: %s", cache_key)
            return cached

        # URL completo (evita problemi base_url su aiohttp)
        url = f"{BASE_URL.rstrip('/')}/{path.lstrip('/')}" if not path.startswith("http") else path

        last_error: Exception = RuntimeError("Unknown error")
        for attempt in range(1, MAX_RETRIES + 1):
            await self._throttle()
            try:
                logger.debug("GET %s %s (attempt %d)", url, params, attempt)
                async with self._session.get(url, params=params) as resp:
                    self._last_request_ts = time.monotonic()

                    if resp.status == 429:
                        retry_after = float(resp.headers.get("Retry-After", BASE_BACKOFF_S * (2 ** attempt)))
                        logger.warning("Rate limited. Sleeping %.1fs before retry.", retry_after)
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        self._set_cache(cache_key, data, ttl=ttl)
                        return data

                    body = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")

            except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as exc:
                last_error = exc
                backoff = BASE_BACKOFF_S * (2 ** (attempt - 1))
                logger.warning("Request failed (%s). Retrying in %.1fs.", exc, backoff)
                await asyncio.sleep(backoff)

        raise RuntimeError(
            f"CoinGecko request failed after {MAX_RETRIES} attempts: {last_error}"
        )

    # ------------------------------------------------------------------
    # OHLCV parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ohlc_response_to_df(raw: List[List[float]], timeframe: str) -> pd.DataFrame:
        """
        Convert CoinGecko /ohlc response to canonical DataFrame.

        CoinGecko format: [[timestamp_ms, open, high, low, close], ...]
        Note: CoinGecko OHLC endpoint does NOT return volume.
        """
        if not raw:
            return _empty_ohlcv_df()

        df = pd.DataFrame(raw, columns=["open_time", "open", "high", "low", "close"])
        df["open_time"] = df["open_time"].astype("int64")
        interval_ms = TIMEFRAME_SECONDS.get(timeframe, 3600) * 1000
        df["close_time"] = df["open_time"] + interval_ms - 1
        df["volume"] = 0.0  # OHLC endpoint doesn't provide volume

        df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
        df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
        return df

    @staticmethod
    def _market_chart_to_df(data: Dict[str, Any], timeframe: str) -> pd.DataFrame:
        """
        Convert CoinGecko /market_chart response to canonical DataFrame.

        data keys: prices [[ts, price]], market_caps, total_volumes [[ts, vol]]
        """
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])

        if not prices:
            return _empty_ohlcv_df()

        price_df = pd.DataFrame(prices, columns=["open_time", "close"])
        vol_df   = pd.DataFrame(volumes, columns=["open_time", "volume"])

        df = price_df.merge(vol_df, on="open_time", how="left")
        df["open_time"] = df["open_time"].astype("int64")
        df = df.sort_values("open_time").reset_index(drop=True)

        # Resample to the requested timeframe to build proper OHLCV
        interval_s = TIMEFRAME_SECONDS.get(timeframe, 3600)
        interval_ms = interval_s * 1000

        df["bucket"] = (df["open_time"] // interval_ms) * interval_ms

        ohlcv = (
            df.groupby("bucket")
            .agg(
                open=("close", "first"),
                high=("close", "max"),
                low=("close", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
            .reset_index()
            .rename(columns={"bucket": "open_time"})
        )
        ohlcv["close_time"] = ohlcv["open_time"] + interval_ms - 1
        ohlcv = ohlcv[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
        ohlcv = ohlcv.sort_values("open_time").reset_index(drop=True)
        return ohlcv

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles.

        For short windows (≤ 90 days for hourly, ≤ 365 days for daily) uses
        the /ohlc endpoint. Falls back to /market_chart for longer periods.
        """
        coin_id = self._symbol_to_id(symbol)
        config  = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["1h"])
        interval_s = TIMEFRAME_SECONDS.get(timeframe, 3600)
        interval_ms = interval_s * 1000

        # Determine date range from `since` and `limit`
        now_ms = int(time.time() * 1000)
        if since is not None:
            start_ms = since
        else:
            start_ms = now_ms - limit * interval_ms

        needed_days = max(1, int((now_ms - start_ms) / 86_400_000) + 1)

        # Choose endpoint based on needed range and granularity
        if needed_days <= config["days_per_page"] and config["granularity"] in ("hourly", "daily"):
            # /ohlc endpoint – best for short, accurate OHLCV (no volume though)
            days_param = _nearest_ohlc_days(needed_days)
            raw = await self._get(
                f"coins/{coin_id}/ohlc",
                params={"vs_currency": self.vs_currency, "days": days_param},
            )
            df = self._ohlc_response_to_df(raw, timeframe)
        else:
            # /market_chart with from/to – supports longer periods
            from_ts = start_ms // 1000
            to_ts   = now_ms // 1000
            raw = await self._get(
                f"coins/{coin_id}/market_chart/range",
                params={
                    "vs_currency": self.vs_currency,
                    "from": from_ts,
                    "to": to_ts,
                },
            )
            df = self._market_chart_to_df(raw, timeframe)

        # Apply `since` filter and row limit
        if since is not None:
            df = df[df["open_time"] >= since]
        if len(df) > limit:
            df = df.tail(limit)

        return df.reset_index(drop=True)

    async def fetch_current_price(self, symbol: str) -> float:
        """
        Return latest price from /simple/price.
        Cache 30s per dashboard real-time (non 5 min).
        """
        coin_id = self._symbol_to_id(symbol)
        data = await self._get(
            "simple/price",
            params={"ids": coin_id, "vs_currencies": self.vs_currency},
            cache_ttl=PRICE_CACHE_TTL_S,
        )
        try:
            return float(data[coin_id][self.vs_currency])
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected /simple/price response for {coin_id}: {data}"
            ) from exc

    async def fetch_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Download up to ~2 years of historical OHLCV data by paginating
        /market_chart/range in chunks that stay within CoinGecko limits.

        Parameters
        ----------
        symbol:     Unified symbol, e.g. "BTCUSDT".
        timeframe:  Candle interval, e.g. "1h", "4h", "1d".
        start_date: Inclusive start (timezone-aware or naive UTC).
        end_date:   Inclusive end.

        Returns
        -------
        pd.DataFrame with canonical OHLCV columns, sorted ascending.
        """
        coin_id = self._symbol_to_id(symbol)
        config  = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["1h"])
        chunk_days = config["days_per_page"]  # Max days per API call

        # Normalise to UTC unix seconds
        start_ts = _dt_to_unix(start_date)
        end_ts   = _dt_to_unix(end_date)

        all_frames: List[pd.DataFrame] = []
        cursor = start_ts
        chunk_s = chunk_days * 86_400

        logger.info(
            "Fetching historical OHLCV for %s/%s from %s to %s (chunk=%dd)",
            symbol, timeframe,
            datetime.utcfromtimestamp(start_ts).strftime("%Y-%m-%d"),
            datetime.utcfromtimestamp(end_ts).strftime("%Y-%m-%d"),
            chunk_days,
        )

        while cursor < end_ts:
            chunk_end = min(cursor + chunk_s, end_ts)
            try:
                raw = await self._get(
                    f"/coins/{coin_id}/market_chart/range",
                    params={
                        "vs_currency": self.vs_currency,
                        "from": cursor,
                        "to": chunk_end,
                    },
                )
                chunk_df = self._market_chart_to_df(raw, timeframe)
                if not chunk_df.empty:
                    all_frames.append(chunk_df)
            except Exception as exc:
                logger.error("Chunk [%d, %d] failed: %s", cursor, chunk_end, exc)
            cursor = chunk_end

        if not all_frames:
            return _empty_ohlcv_df()

        df = pd.concat(all_frames, ignore_index=True)
        df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

        # Filter precisely to requested range
        start_ms = start_ts * 1000
        end_ms   = end_ts   * 1000
        df = df[(df["open_time"] >= start_ms) & (df["open_time"] <= end_ms)]

        logger.info("Historical fetch complete: %d candles returned.", len(df))
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _empty_ohlcv_df() -> pd.DataFrame:
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


def _nearest_ohlc_days(needed: int) -> int:
    """Return the smallest valid CoinGecko /ohlc 'days' value >= needed."""
    valid = [1, 7, 14, 30, 90, 180, 365]
    for v in valid:
        if v >= needed:
            return v
    return 365


def _dt_to_unix(dt: datetime) -> int:
    """Convert datetime to Unix seconds (UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())
