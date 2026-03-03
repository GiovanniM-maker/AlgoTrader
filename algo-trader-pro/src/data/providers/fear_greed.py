"""
Async Fear & Greed Index provider.

Data source: https://api.alternative.me/fng/
The index is published once per day. Values range from 0 (Extreme Fear) to
100 (Extreme Greed). Results are cached for 1 hour to avoid redundant requests.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Tuple

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FNG_API_URL = "https://api.alternative.me/fng/"

CACHE_TTL_S = 3_600          # 1 hour
REQUEST_TIMEOUT_S = 15
MAX_RETRIES = 3
BASE_BACKOFF_S = 2.0

# Classification bands (inclusive lower bound)
CLASSIFICATION_BANDS = [
    (0,  25, "Extreme Fear"),
    (25, 45, "Fear"),
    (45, 55, "Neutral"),
    (55, 75, "Greed"),
    (75, 101, "Extreme Greed"),
]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class FearGreedProvider:
    """
    Async provider for the Alternative.me Fear & Greed Index.

    Usage (standalone)::

        async with FearGreedProvider() as fng:
            current = await fng.fetch_current()
            hist_df = await fng.fetch_historical(limit=365)
    """

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None
        # cache: key → (data, expiry_monotonic)
        self._cache: Dict[str, Tuple[Any, float]] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
            self._session = aiohttp.ClientSession(timeout=timeout)
            logger.info("FearGreedProvider: HTTP session opened.")

    async def disconnect(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("FearGreedProvider: HTTP session closed.")
        self._session = None

    async def __aenter__(self) -> "FearGreedProvider":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cache(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        data, expiry = entry
        if time.monotonic() < expiry:
            return data
        del self._cache[key]
        return None

    def _set_cache(self, key: str, data: Any, ttl: float = CACHE_TTL_S) -> None:
        self._cache[key] = (data, time.monotonic() + ttl)

    # ------------------------------------------------------------------
    # Internal HTTP
    # ------------------------------------------------------------------

    async def _get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a GET to FNG_API_URL with retry + exponential back-off.
        """
        if self._session is None:
            raise RuntimeError(
                "FearGreedProvider is not connected. Call await connect() first."
            )

        cache_key = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.debug("FNG cache HIT for params: %s", params)
            return cached

        last_exc: Exception = RuntimeError("Unknown error")
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.debug("GET FNG (attempt %d) params=%s", attempt, params)
                async with self._session.get(FNG_API_URL, params=params) as resp:
                    if resp.status == 429:
                        backoff = BASE_BACKOFF_S * (2 ** attempt)
                        logger.warning("FNG rate limited. Retrying in %.1fs.", backoff)
                        await asyncio.sleep(backoff)
                        continue
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")
                    data = await resp.json(content_type=None)
                    self._set_cache(cache_key, data)
                    return data

            except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as exc:
                last_exc = exc
                backoff = BASE_BACKOFF_S * (2 ** (attempt - 1))
                logger.warning("FNG request error (%s). Retry in %.1fs.", exc, backoff)
                await asyncio.sleep(backoff)

        raise RuntimeError(
            f"FNG request failed after {MAX_RETRIES} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch(self) -> Dict[str, Any]:
        """Alias for fetch_current() — used by TradingEngine."""
        await self.connect()
        return await self.fetch_current()

    async def fetch_current(self) -> Dict[str, Any]:
        """
        Fetch the most recent Fear & Greed reading.

        Returns
        -------
        dict
            {
              "value":          int,   # 0-100
              "classification": str,   # e.g. "Extreme Fear"
              "timestamp":      int,   # Unix seconds
            }
        """
        data = await self._get({"limit": 1, "format": "json"})
        return _parse_single(data["data"][0])

    async def fetch_historical(self, limit: int = 365) -> pd.DataFrame:
        """
        Fetch the last *limit* daily Fear & Greed readings.

        Parameters
        ----------
        limit:
            Number of historical data points (days). Max ~2000 on free tier.

        Returns
        -------
        pd.DataFrame
            Columns: timestamp (int64, Unix s), value (int64), classification (str).
            Sorted ascending by timestamp.
        """
        data = await self._get({"limit": limit, "format": "json"})
        records = [_parse_single(entry) for entry in data.get("data", [])]

        if not records:
            return pd.DataFrame(columns=["timestamp", "value", "classification"])

        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["timestamp"]      = df["timestamp"].astype("int64")
        df["value"]          = df["value"].astype("int64")
        df["classification"] = df["classification"].astype(str)
        return df

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def classify(value: int) -> str:
        """
        Return the classification string for a raw 0-100 value.

        This mirrors Alternative.me's own classification but is computed
        locally so callers can classify arbitrary values without an API call.
        """
        for lo, hi, label in CLASSIFICATION_BANDS:
            if lo <= value < hi:
                return label
        return "Unknown"


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _parse_single(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a single raw API data entry into the canonical dict format.

    The API returns value as a string and timestamp as a string.
    """
    raw_value = entry.get("value", "0")
    raw_ts    = entry.get("timestamp", "0")
    value     = int(raw_value)
    timestamp = int(raw_ts)

    # Prefer the API's own classification; fall back to local computation.
    classification = entry.get("value_classification") or _classify_value(value)

    return {
        "value":          value,
        "classification": classification,
        "timestamp":      timestamp,
    }


def _classify_value(value: int) -> str:
    for lo, hi, label in CLASSIFICATION_BANDS:
        if lo <= value < hi:
            return label
    return "Unknown"
