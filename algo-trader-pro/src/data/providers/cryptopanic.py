"""
Async CryptoPanic news-sentiment provider.

Docs: https://cryptopanic.com/developers/api/
Base URL: https://cryptopanic.com/api/developer/v2

API key is loaded from the environment variable CRYPTOPANIC_API_KEY.
On free tier the rate limit is ~5 requests per minute.

Sentiment scoring
-----------------
Each CryptoPanic post has community vote counts:
    positive, negative, important, liked, disliked, lol, toxic, saved, comments

The sentiment score for a batch of posts is computed as:

    raw_score = positive / (positive + negative)   ∈ [0, 1]
    score     = raw_score * 2 - 1                  ∈ [-1, +1]

Posts are weighted by recency: posts from the last 4 hours receive a weight
of 1.0; older posts decay with a half-life of 4 hours.

On any API failure the provider returns 0.0 (neutral sentiment) rather than
raising so that downstream models degrade gracefully.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://cryptopanic.com/api/developer/v2"

# Free-tier: ~5 req/min  →  one request every 12 s
RATE_LIMIT_SLEEP_S = 12.5
MAX_RETRIES = 3
BASE_BACKOFF_S = 3.0
REQUEST_TIMEOUT_S = 20

# Recency half-life in seconds (4 hours)
RECENCY_HALF_LIFE_S = 4 * 3_600

# Maps unified trading symbols → CryptoPanic currency codes
SYMBOL_TO_CURRENCY: Dict[str, str] = {
    "BTCUSDT":  "BTC",
    "ETHUSDT":  "ETH",
    "SOLUSDT":  "SOL",
    "BNBUSDT":  "BNB",
    "XRPUSDT":  "XRP",
    "ADAUSDT":  "ADA",
    "DOGEUSDT": "DOGE",
    "AVAXUSDT": "AVAX",
    "DOTUSDT":  "DOT",
    "MATICUSDT":"MATIC",
}

VALID_KINDS    = {"news", "media", "all"}
VALID_FILTERS  = {"rising", "hot", "bullish", "bearish", "important", "saved", "lol"}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class CryptoPanicProvider:
    """
    Async provider for CryptoPanic news and sentiment data.

    Usage::

        async with CryptoPanicProvider() as cp:
            score = await cp.get_sentiment_score("BTCUSDT")
            df    = await cp.fetch_and_store_sentiment(["BTCUSDT", "ETHUSDT"])
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        api_key:
            CryptoPanic auth token. If ``None``, read from env var
            ``CRYPTOPANIC_API_KEY``. If the key is missing or empty the
            provider will still work but every call will return neutral
            sentiment or empty data.
        """
        self._api_key: str = api_key or os.getenv("CRYPTOPANIC_API_KEY", "")
        if not self._api_key:
            logger.warning(
                "CRYPTOPANIC_API_KEY is not set. "
                "All requests will return neutral/empty results."
            )
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_ts: float = 0.0
        # Cache: cache_key → (data, expiry_monotonic)
        self._cache: Dict[str, Tuple[Any, float]] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
            self._session = aiohttp.ClientSession(
                headers={"Accept": "application/json"},
                timeout=timeout,
            )
            logger.info("CryptoPanicProvider: HTTP session opened.")

    async def disconnect(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("CryptoPanicProvider: HTTP session closed.")
        self._session = None

    async def __aenter__(self) -> "CryptoPanicProvider":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Internal helpers
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

    def _set_cache(self, key: str, data: Any, ttl: float = 60.0) -> None:
        self._cache[key] = (data, time.monotonic() + ttl)

    async def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < RATE_LIMIT_SLEEP_S:
            await asyncio.sleep(RATE_LIMIT_SLEEP_S - elapsed)

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """GET request with retry, back-off, and caching."""
        if not self._api_key:
            return {}

        if self._session is None:
            raise RuntimeError("CryptoPanicProvider not connected.")

        params = dict(params or {})
        params["auth_token"] = self._api_key

        cache_key = path + "&".join(f"{k}={v}" for k, v in sorted(params.items()) if k != "auth_token")
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        last_exc: Exception = RuntimeError("Unknown")
        for attempt in range(1, MAX_RETRIES + 1):
            await self._throttle()
            url = f"{BASE_URL}{path}"
            try:
                logger.debug("GET %s params=%s (attempt %d)", url, {k: v for k, v in params.items() if k != "auth_token"}, attempt)
                async with self._session.get(url, params=params) as resp:
                    self._last_request_ts = time.monotonic()

                    if resp.status == 429:
                        backoff = BASE_BACKOFF_S * (2 ** attempt)
                        logger.warning("CryptoPanic rate limited. Sleeping %.1fs.", backoff)
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
                logger.warning("CryptoPanic request failed (%s). Retry in %.1fs.", exc, backoff)
                await asyncio.sleep(backoff)

        logger.error("CryptoPanic request failed after %d attempts: %s", MAX_RETRIES, last_exc)
        return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_news(
        self,
        currencies: Optional[List[str]] = None,
        kind: str = "news",
        filter: str = "hot",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Fetch news / media posts from CryptoPanic.

        Parameters
        ----------
        currencies:
            List of currency codes, e.g. ``["BTC", "ETH"]``.
            ``None`` means all currencies.
        kind:
            Post type: ``"news"``, ``"media"``, or ``"all"``.
        filter:
            ``"rising"``, ``"hot"``, ``"bullish"``, ``"bearish"``,
            ``"important"``, ``"saved"``, or ``"lol"``.
        limit:
            Maximum number of posts to return (≤ 100 per page).

        Returns
        -------
        list of dicts
            Each dict contains: title, url, published_at, currencies,
            votes (dict with positive, negative, important, liked, disliked).
        """
        if kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {VALID_KINDS}")
        if filter not in VALID_FILTERS:
            raise ValueError(f"filter must be one of {VALID_FILTERS}")

        params: Dict[str, Any] = {
            "kind":    kind,
            "filter":  filter,
        }
        if currencies:
            params["currencies"] = ",".join(currencies)

        posts_collected: List[Dict[str, Any]] = []
        next_url: Optional[str] = "/posts/"

        while next_url and len(posts_collected) < limit:
            # Strip base URL if it was returned as absolute
            if next_url.startswith("http"):
                path = next_url.replace(BASE_URL, "")
                data = await self._get(path)
            else:
                data = await self._get(next_url, params)

            results = data.get("results", [])
            for post in results:
                posts_collected.append(_parse_post(post))
                if len(posts_collected) >= limit:
                    break

            # CryptoPanic paginates via 'next'
            next_url = data.get("next")
            if not results:
                break

        return posts_collected

    async def get_sentiment_score(self, symbol: str) -> float:
        """
        Aggregate recent news into a sentiment score for *symbol*.

        The score is computed as:
            1. For each post: raw = positive / (positive + negative),  ∈ [0, 1]
            2. Mapped to [-1, +1]: score_i = raw * 2 - 1
            3. Weighted by recency (half-life = 4 h)
            4. Weighted average of all score_i values

        Returns
        -------
        float
            Sentiment in [-1.0, +1.0]. Returns 0.0 on any error or if no
            data is available.
        """
        currency = SYMBOL_TO_CURRENCY.get(symbol.upper())
        if currency is None:
            logger.warning("Symbol %s not mapped in CryptoPanic provider.", symbol)
            return 0.0

        try:
            posts = await self.fetch_news(currencies=[currency], kind="news", filter="hot", limit=50)
        except Exception as exc:
            logger.error("get_sentiment_score failed for %s: %s", symbol, exc)
            return 0.0

        if not posts:
            return 0.0

        now_ts = time.time()
        weighted_sum = 0.0
        weight_total = 0.0

        for post in posts:
            votes    = post.get("votes", {})
            pos      = float(votes.get("positive",  0))
            neg      = float(votes.get("negative",  0))
            total    = pos + neg
            if total == 0:
                continue  # Skip posts with no votes

            raw_score  = pos / total          # 0..1
            score_i    = raw_score * 2 - 1    # -1..+1

            # Recency weight
            pub_ts = post.get("published_at_unix", now_ts)
            age_s  = max(0.0, now_ts - pub_ts)
            weight = math.exp(-age_s * math.log(2) / RECENCY_HALF_LIFE_S)

            weighted_sum  += score_i * weight
            weight_total  += weight

        if weight_total == 0:
            return 0.0

        final = weighted_sum / weight_total
        return float(max(-1.0, min(1.0, final)))

    async def fetch_sentiment(
        self, symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fetch aggregate sentiment for the engine. Returns dict with
        cryptopanic_score (float) and cryptopanic_article_count (int).
        """
        symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        await self.connect()
        df = await self.fetch_and_store_sentiment(symbols)
        if df is None or df.empty:
            return {"cryptopanic_score": 0.0, "cryptopanic_article_count": 0}
        score = float(df["sentiment_score"].mean())
        return {"cryptopanic_score": score, "cryptopanic_article_count": len(df)}

    async def fetch_and_store_sentiment(
        self, symbols: List[str]
    ) -> pd.DataFrame:
        """
        Fetch sentiment scores for multiple *symbols* and return as DataFrame.

        Parameters
        ----------
        symbols:
            List of unified trading symbols, e.g. ``["BTCUSDT", "ETHUSDT"]``.

        Returns
        -------
        pd.DataFrame
            Columns: symbol (str), currency (str), sentiment_score (float),
                     fetched_at (int, Unix seconds).
        """
        rows = []
        for symbol in symbols:
            score    = await self.get_sentiment_score(symbol)
            currency = SYMBOL_TO_CURRENCY.get(symbol.upper(), "UNKNOWN")
            rows.append(
                {
                    "symbol":          symbol,
                    "currency":        currency,
                    "sentiment_score": score,
                    "fetched_at":      int(time.time()),
                }
            )

        df = pd.DataFrame(rows)
        df["sentiment_score"] = df["sentiment_score"].astype("float64")
        df["fetched_at"]      = df["fetched_at"].astype("int64")
        return df


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_post(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a raw CryptoPanic API post dict."""
    votes = raw.get("votes") or {}

    # Parse published_at to Unix timestamp
    pub_at_str = raw.get("published_at", "")
    pub_ts = _iso_to_unix(pub_at_str)

    currencies = [
        c.get("code", "") for c in (raw.get("currencies") or [])
    ]

    return {
        "title":            raw.get("title", ""),
        "url":              raw.get("url", ""),
        "published_at":     pub_at_str,
        "published_at_unix": pub_ts,
        "currencies":       currencies,
        "votes": {
            "positive":  int(votes.get("positive",  0)),
            "negative":  int(votes.get("negative",  0)),
            "important": int(votes.get("important", 0)),
            "liked":     int(votes.get("liked",     0)),
            "disliked":  int(votes.get("disliked",  0)),
            "lol":       int(votes.get("lol",       0)),
            "toxic":     int(votes.get("toxic",     0)),
            "saved":     int(votes.get("saved",     0)),
            "comments":  int(votes.get("comments",  0)),
        },
    }


def _iso_to_unix(iso_str: str) -> float:
    """Convert ISO 8601 string to Unix seconds. Returns current time on failure."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return time.time()
