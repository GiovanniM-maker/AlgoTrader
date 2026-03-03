"""
Abstract base class for all data providers in the algorithmic trading system.

All concrete providers must implement this interface to ensure consistent
behaviour across different data sources (REST, WebSocket, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable
import pandas as pd


class BaseProvider(ABC):
    """
    Abstract base for every market-data provider.

    Concrete implementations are expected to:
    - Handle their own authentication / session lifecycle in connect/disconnect.
    - Return DataFrames whose columns always match the contract described in
      fetch_ohlcv (open_time, open, high, low, close, volume, close_time).
    - Be safe to use as async context managers via __aenter__ / __aexit__.
    """

    # Subclasses must set a human-readable name, e.g. "CoinGecko REST".
    name: str = "base"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """
        Initialise the provider (open HTTP session, authenticate, etc.).
        Must be called before any data-fetching method.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Clean up resources (close HTTP session, unsubscribe, etc.).
        Should be idempotent – safe to call even if connect() was never called.
        """
        ...

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles for *symbol* on the given *timeframe*.

        Parameters
        ----------
        symbol:
            Trading pair in unified format, e.g. "BTCUSDT", "ETHUSDT".
        timeframe:
            Candle interval string, e.g. "1m", "5m", "1h", "4h", "1d".
        since:
            Unix timestamp in **milliseconds** of the earliest candle to fetch.
            ``None`` means fetch the most recent *limit* candles.
        limit:
            Maximum number of candles to return in a single call.

        Returns
        -------
        pd.DataFrame
            Columns: open_time (int64, ms), open (float64), high (float64),
                     low (float64), close (float64), volume (float64),
                     close_time (int64, ms).
            Sorted ascending by open_time, no duplicate open_time values.
        """
        ...

    @abstractmethod
    async def fetch_current_price(self, symbol: str) -> float:
        """
        Return the latest traded price for *symbol* as a plain float.

        Parameters
        ----------
        symbol:
            Trading pair in unified format, e.g. "BTCUSDT".

        Returns
        -------
        float
            Current market price in quote currency (e.g. USD).

        Raises
        ------
        ValueError
            If *symbol* is not supported by this provider.
        RuntimeError
            If the provider is not connected or the request fails after retries.
        """
        ...

    # ------------------------------------------------------------------
    # Optional streaming interface
    # ------------------------------------------------------------------

    def subscribe(self, callback: Callable[[pd.DataFrame], None]) -> None:
        """
        Register *callback* to be invoked whenever new candle data arrives.

        This is an **optional** extension point for WebSocket-based providers.
        REST-only providers raise ``NotImplementedError`` (the default here).

        Parameters
        ----------
        callback:
            Async or sync callable that accepts a single-row DataFrame whose
            schema matches the return value of ``fetch_ohlcv``.

        Raises
        ------
        NotImplementedError
            For providers that do not support streaming.
        """
        raise NotImplementedError(
            f"Provider '{self.name}' does not support streaming subscriptions."
        )

    # ------------------------------------------------------------------
    # Async context-manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "BaseProvider":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Helpers available to all sub-classes
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        """
        Lightweight structural validation of an OHLCV DataFrame.

        Raises
        ------
        ValueError
            If required columns are missing or data invariants are violated.
        """
        required = {"open_time", "open", "high", "low", "close", "volume", "close_time"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame is missing required OHLCV columns: {missing}"
            )

        if df.empty:
            return  # Nothing to validate on empty result

        if (df["high"] < df["low"]).any():
            raise ValueError("Invariant violated: high < low found in OHLCV data.")
        if (df["volume"] < 0).any():
            raise ValueError("Invariant violated: negative volume found in OHLCV data.")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
