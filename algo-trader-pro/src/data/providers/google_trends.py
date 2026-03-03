"""
Google Trends provider using the pytrends unofficial API.

Returns interest-over-time data for crypto-related keywords. Results are
cached for 24 hours to avoid repeated 429 errors from Google.

NOTE: pytrends uses Google's unofficial API. Google may throttle or block
requests without warning. The provider handles 429 responses gracefully and
returns a neutral score (50) rather than raising to the caller.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import pytrends; make provider optional if not installed
# ---------------------------------------------------------------------------

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    logger.warning(
        "pytrends is not installed. GoogleTrendsProvider will return neutral scores. "
        "Install with: pip install pytrends"
    )

# ---------------------------------------------------------------------------
# urllib3 >= 2.0 compatibility shim
# pytrends internally calls Retry(method_whitelist=...) which was removed in
# urllib3 2.0. Apply a permanent module-level patch so it works regardless
# of when Retry() is instantiated (during TrendReq init AND during requests).
# ---------------------------------------------------------------------------
try:
    from urllib3.util.retry import Retry as _Retry

    _orig_retry_init = _Retry.__init__

    def _patched_retry_init(self_r, *args, **kwargs):
        if "method_whitelist" in kwargs:
            kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
        return _orig_retry_init(self_r, *args, **kwargs)

    _Retry.__init__ = _patched_retry_init
    logger.debug("Applied urllib3 Retry.method_whitelist compatibility patch.")
except Exception as _patch_exc:
    logger.debug("Could not apply urllib3 patch: %s", _patch_exc)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cache results for 24 hours (Google Trends data is low-frequency)
CACHE_TTL_S = 24 * 3_600

# How long to sleep before retrying on 429
RETRY_SLEEP_S = 60.0
MAX_RETRIES = 3

# Keywords to query per unified symbol
SYMBOL_KEYWORDS: Dict[str, List[str]] = {
    "BTCUSDT":   ["bitcoin",  "buy bitcoin"],
    "ETHUSDT":   ["ethereum", "buy ethereum"],
    "SOLUSDT":   ["solana",   "buy solana"],
    "BNBUSDT":   ["bnb",      "binance coin"],
    "XRPUSDT":   ["xrp",      "buy xrp"],
    "ADAUSDT":   ["cardano",  "buy cardano"],
    "DOGEUSDT":  ["dogecoin", "buy dogecoin"],
    "AVAXUSDT":  ["avalanche crypto", "buy avax"],
    "DOTUSDT":   ["polkadot", "buy polkadot"],
    "MATICUSDT": ["polygon crypto", "buy matic"],
}

NEUTRAL_SCORE = 50.0  # Returned when data is unavailable


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class GoogleTrendsProvider:
    """
    Google Trends interest-over-time provider (via pytrends).

    This is a **synchronous-under-the-hood** provider because pytrends does
    not support async. All blocking calls are wrapped to be safe in async
    event loops when run via ``asyncio.get_event_loop().run_in_executor``.

    For simplicity in the trading pipeline the public methods are regular
    (sync) methods. Wrap them in ``asyncio.to_thread`` if you need async::

        score = await asyncio.to_thread(provider.get_trend_signal, "BTCUSDT")

    Usage::

        provider = GoogleTrendsProvider()
        score = provider.get_trend_signal("BTCUSDT")   # float 0–100
    """

    def __init__(
        self,
        hl: str = "en-US",
        tz: int = 0,
        timeout: Tuple[int, int] = (10, 25),
        proxy: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        hl:      Host language for Google Trends (default ``"en-US"``).
        tz:      Timezone offset in minutes from UTC (0 = UTC).
        timeout: (connect_timeout, read_timeout) in seconds for requests.
        proxy:   Optional proxy URL to route requests through, e.g.
                 ``"https://user:pass@host:port"``.
        """
        self._hl = hl
        self._tz = tz
        self._timeout = timeout
        self._proxies = {"https": proxy} if proxy else {}
        self._pytrends: Optional[Any] = None  # Lazy initialised
        # cache: key → (data, expiry_monotonic)
        self._cache: Dict[str, Tuple[Any, float]] = {}

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
    # Internal pytrends helper
    # ------------------------------------------------------------------

    def _get_pytrends(self) -> Any:
        """Lazy-initialise and return the TrendReq instance."""
        if not PYTRENDS_AVAILABLE:
            return None
        if self._pytrends is None:
            # urllib3 >= 2.0 patch is applied permanently at module level above.
            try:
                self._pytrends = TrendReq(
                    hl=self._hl,
                    tz=self._tz,
                    timeout=self._timeout,
                    proxies=self._proxies,
                    retries=2,
                    backoff_factor=0.5,
                )
            except Exception as exc:
                logger.warning(
                    "TrendReq initialisation failed (%s). "
                    "Google Trends will return neutral scores.",
                    exc,
                )
                self._pytrends = None
        return self._pytrends

    def _fetch_interest_raw(
        self, keywords: List[str], timeframe: str
    ) -> Optional[pd.DataFrame]:
        """
        Call pytrends and return the interest-over-time DataFrame.

        Returns ``None`` on any error (including 429).
        """
        pt = self._get_pytrends()
        if pt is None:
            return None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                pt.build_payload(keywords, timeframe=timeframe)
                df = pt.interest_over_time()
                return df
            except Exception as exc:
                exc_str = str(exc).lower()
                if "429" in exc_str or "too many" in exc_str or "response error" in exc_str:
                    logger.warning(
                        "Google Trends rate limit (attempt %d/%d). Sleeping %.0fs.",
                        attempt, MAX_RETRIES, RETRY_SLEEP_S,
                    )
                    time.sleep(RETRY_SLEEP_S)
                else:
                    logger.error("Google Trends error (attempt %d): %s", attempt, exc)
                    return None

        logger.error("Google Trends: exhausted retries for keywords %s.", keywords)
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_interest(
        self,
        keywords: List[str],
        timeframe: str = "today 3-m",
    ) -> pd.DataFrame:
        """
        Return interest-over-time data for the given *keywords*.

        Parameters
        ----------
        keywords:
            List of search terms, e.g. ``["bitcoin", "buy bitcoin"]``.
        timeframe:
            pytrends timeframe string, e.g. ``"today 3-m"``, ``"today 12-m"``,
            ``"2023-01-01 2023-12-31"``.

        Returns
        -------
        pd.DataFrame
            Index: datetime, columns: each keyword + 'isPartial'.
            Empty DataFrame on error.
        """
        cache_key = f"{','.join(sorted(keywords))}|{timeframe}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.debug("Google Trends cache HIT: %s", cache_key)
            return cached

        df = self._fetch_interest_raw(keywords, timeframe)
        if df is None or df.empty:
            empty = pd.DataFrame()
            return empty

        self._set_cache(cache_key, df)
        return df

    def get_trend_signal(
        self,
        symbol: str,
        timeframe: str = "today 3-m",
    ) -> float:
        """
        Return a normalised trend score (0-100) for *symbol*.

        Method
        ------
        1. Fetch the interest-over-time for the symbol's keywords over
           *timeframe* (default last 3 months).
        2. Take the mean of all keyword columns at each timestamp (composite).
        3. Compare the mean of the last 7 days vs the mean of the 7 days
           before that. The ratio is mapped to 0–100.

            score = (current_week_avg / (current_week_avg + prev_week_avg)) * 100

           where scores close to 50 indicate no change, >50 means rising
           interest, <50 means declining interest.
        4. If either week has no data, return the overall latest value (0–100).

        Returns
        -------
        float
            Trend score in [0, 100]. Returns ``NEUTRAL_SCORE`` (50.0) on
            any error or missing data.
        """
        keywords = SYMBOL_KEYWORDS.get(symbol.upper())
        if not keywords:
            logger.warning("No keywords mapped for symbol '%s'.", symbol)
            return NEUTRAL_SCORE

        df = self.fetch_interest(keywords, timeframe)

        if df is None or df.empty:
            return NEUTRAL_SCORE

        # Keep only numeric keyword columns (drop 'isPartial')
        numeric_cols = [c for c in df.columns if c in keywords]
        if not numeric_cols:
            return NEUTRAL_SCORE

        # Composite: mean across all keyword columns at each timestamp
        df = df.copy()
        df["_composite"] = df[numeric_cols].mean(axis=1)

        series = df["_composite"].dropna()
        if len(series) < 2:
            return float(series.iloc[-1]) if len(series) == 1 else NEUTRAL_SCORE

        # Split into current week and previous week (last 14 rows ≈ 14 days)
        n = len(series)
        if n >= 14:
            current_week = series.iloc[-7:].mean()
            prev_week    = series.iloc[-14:-7].mean()
        elif n >= 2:
            mid = n // 2
            current_week = series.iloc[mid:].mean()
            prev_week    = series.iloc[:mid].mean()
        else:
            return float(series.iloc[-1])

        total = current_week + prev_week
        if total == 0:
            return NEUTRAL_SCORE

        score = (current_week / total) * 100.0
        return float(max(0.0, min(100.0, score)))

    async def fetch(self) -> Dict[str, Any]:
        """Async fetch for TradingEngine — returns dict with google_trends_score."""
        import asyncio
        loop = asyncio.get_event_loop()
        score = await loop.run_in_executor(None, lambda: self.get_trend_signal("BTCUSDT"))
        return {"google_trends_score": score, "google_trends_scores": [score]}

    def get_all_signals(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "today 3-m",
    ) -> Dict[str, float]:
        """
        Return trend scores for multiple symbols.

        Parameters
        ----------
        symbols:
            List of unified trading symbols. Defaults to all mapped symbols.
        timeframe:
            pytrends timeframe string.

        Returns
        -------
        dict
            {symbol: score} where score ∈ [0, 100].
        """
        if symbols is None:
            symbols = list(SYMBOL_KEYWORDS.keys())

        results: Dict[str, float] = {}
        for sym in symbols:
            try:
                results[sym] = self.get_trend_signal(sym, timeframe)
            except Exception as exc:
                logger.error("get_trend_signal failed for %s: %s", sym, exc)
                results[sym] = NEUTRAL_SCORE

        return results
