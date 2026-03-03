"""
src/data/providers/bybit_ws.py

Bybit V5 WebSocket client for real-time OHLCV candle data.

Connection
----------
wss://stream.bybit.com/v5/public/spot

Subscription topic format
--------------------------
  "kline.{interval}.{symbol}"   e.g. "kline.60.BTCUSDT"

Bybit WebSocket interval codes
--------------------------------
  1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M

Message format (simplified)
-----------------------------
{
  "topic": "kline.60.BTCUSDT",
  "type":  "snapshot",
  "ts":    1699000000000,
  "data": [
    {
      "start":   1699000000000,
      "end":     1699003600000,
      "interval": "60",
      "open":    "34000",
      "close":   "34500",
      "high":    "35000",
      "low":     "33900",
      "volume":  "123.45",
      "turnover": "4234567.89",
      "confirm": true,          <-- True = candle is closed, False = forming
      "timestamp": 1699003600000
    }
  ]
}

Only candles where ``confirm == True`` trigger the user callback.

Callback signature
-------------------
  callback(symbol: str, ohlcv_row: dict) -> None (or coroutine)

ohlcv_row keys: open_time, open, high, low, close, volume, close_time
  (matches the canonical DataFrame column schema used throughout the project)

Auto-reconnect
--------------
On any disconnect the client waits an exponentially increasing delay
(1s, 2s, 4s, 8s, 16s, …, capped at 60s) then re-connects and re-subscribes
to all previously registered topics.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Callable, Dict, List, Optional, Set

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WS_URL = "wss://stream.bybit.com/v5/public/spot"

# Maps human-readable timeframe strings to Bybit WebSocket interval codes
TIMEFRAME_TO_WS_INTERVAL: Dict[str, str] = {
    "1m":   "1",
    "3m":   "3",
    "5m":   "5",
    "15m":  "15",
    "30m":  "30",
    "1h":   "60",
    "2h":   "120",
    "4h":   "240",
    "6h":   "360",
    "12h":  "720",
    "1d":   "D",
    "1w":   "W",
    "1M":   "M",
}

# Candle duration in milliseconds (for computing close_time)
INTERVAL_MS: Dict[str, int] = {
    "1":   60_000,
    "3":   180_000,
    "5":   300_000,
    "15":  900_000,
    "30":  1_800_000,
    "60":  3_600_000,
    "120": 7_200_000,
    "240": 14_400_000,
    "360": 21_600_000,
    "720": 43_200_000,
    "D":   86_400_000,
    "W":   604_800_000,
    "M":   2_592_000_000,  # ~30 days
}

# Reconnect configuration
MIN_RECONNECT_DELAY_S = 1.0
MAX_RECONNECT_DELAY_S = 60.0
RECONNECT_MULTIPLIER  = 2.0

# Bybit heartbeat: ping every 20s to keep the connection alive
PING_INTERVAL_S = 20.0
PING_TIMEOUT_S  = 10.0


# ---------------------------------------------------------------------------
# WebSocket Provider
# ---------------------------------------------------------------------------

class BybitWebSocketProvider:
    """
    Async Bybit V5 WebSocket client for real-time kline (OHLCV) data.

    Usage (inside an async context)
    --------------------------------
    .. code-block:: python

        provider = BybitWebSocketProvider()
        provider.subscribe(
            symbols=["BTCUSDT", "ETHUSDT"],
            timeframe="1h",
            callback=my_candle_handler,
        )
        await provider.connect()   # blocks; reconnects automatically

    Parameters
    ----------
    ws_url : str
        Override the WebSocket URL (e.g. for testnet).
    """

    def __init__(self, ws_url: str = WS_URL) -> None:
        self._ws_url = ws_url

        # Set of topics to subscribe to, populated via subscribe()
        self._topics: Set[str] = set()

        # Registered callback (single global callback – sufficient for the engine)
        self._callback: Optional[Callable] = None

        # Internal state
        self._running: bool = False
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._reconnect_delay: float = MIN_RECONNECT_DELAY_S

        # Maps topic → (symbol, ws_interval) for fast message dispatch
        self._topic_meta: Dict[str, tuple] = {}

        # Lock to protect _topics / _topic_meta mutations
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(
        self,
        symbols: List[str],
        timeframe: str,
        callback: Callable,
    ) -> None:
        """
        Register symbols and a timeframe to subscribe to, plus the callback
        that will be invoked on each closed candle.

        This method is synchronous and safe to call before ``connect()``.
        It may also be called while connected – the new subscriptions will
        be sent to Bybit on the next reconnect (or immediately if an
        ``asyncio.run_coroutine_threadsafe`` context is available).

        Parameters
        ----------
        symbols   : list of str
            Bybit trading pairs, e.g. ``["BTCUSDT", "ETHUSDT"]``.
        timeframe : str
            Human-readable timeframe, e.g. ``"1h"``.  Must be a key in
            ``TIMEFRAME_TO_WS_INTERVAL``.
        callback  : callable
            Invoked as ``callback(symbol, ohlcv_row)`` when a candle closes.
            May be a plain function or a coroutine function.

        Raises
        ------
        ValueError
            If *timeframe* is not supported.
        """
        ws_interval = TIMEFRAME_TO_WS_INTERVAL.get(timeframe)
        if ws_interval is None:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported: {list(TIMEFRAME_TO_WS_INTERVAL.keys())}"
            )

        self._callback = callback

        for sym in symbols:
            upper_sym = sym.upper()
            topic = f"kline.{ws_interval}.{upper_sym}"
            self._topics.add(topic)
            self._topic_meta[topic] = (upper_sym, ws_interval)
            logger.info("Registered subscription: topic=%s", topic)

    async def connect(self) -> None:
        """
        Establish the WebSocket connection and start the receive loop.

        Blocks indefinitely; reconnects automatically on any disconnect.
        Cancel the enclosing task (or call ``disconnect()``) to stop.
        """
        self._running = True
        logger.info("BybitWebSocketProvider: starting connection loop.")

        while self._running:
            try:
                await self._connect_and_run()
                # _connect_and_run returned cleanly → stop
                break
            except asyncio.CancelledError:
                logger.info("BybitWebSocketProvider: cancelled.")
                break
            except Exception as exc:
                if not self._running:
                    break
                logger.warning(
                    "WebSocket connection lost (%s). "
                    "Reconnecting in %.1fs…",
                    exc, self._reconnect_delay,
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * RECONNECT_MULTIPLIER,
                    MAX_RECONNECT_DELAY_S,
                )

        logger.info("BybitWebSocketProvider: connection loop exited.")

    async def disconnect(self) -> None:
        """
        Signal the receive loop to stop and close the WebSocket.

        Safe to call even if the connection is not open.
        """
        self._running = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        logger.info("BybitWebSocketProvider: disconnected.")

    # ------------------------------------------------------------------
    # Internal connection lifecycle
    # ------------------------------------------------------------------

    async def _connect_and_run(self) -> None:
        """
        Open one WebSocket connection, subscribe, then receive messages
        until the connection closes or _running is set to False.
        """
        logger.info("Connecting to %s …", self._ws_url)

        async with websockets.connect(
            self._ws_url,
            ping_interval=PING_INTERVAL_S,
            ping_timeout=PING_TIMEOUT_S,
            open_timeout=15,
            close_timeout=5,
        ) as ws:
            self._ws = ws
            self._reconnect_delay = MIN_RECONNECT_DELAY_S  # Reset on success
            logger.info("WebSocket connected to %s", self._ws_url)

            # Subscribe to all registered topics
            await self._send_subscriptions(ws)

            # Start receiving messages
            await self._receive_loop(ws)

    async def _send_subscriptions(
        self,
        ws: websockets.WebSocketClientProtocol,
    ) -> None:
        """Send a subscription request for every registered topic."""
        async with self._lock:
            topics_snapshot = list(self._topics)

        if not topics_snapshot:
            logger.warning("No topics registered – nothing to subscribe to.")
            return

        # Bybit accepts up to 10 topics per subscription message.
        # Chunk into batches of 10 to stay within limits.
        batch_size = 10
        for i in range(0, len(topics_snapshot), batch_size):
            batch = topics_snapshot[i : i + batch_size]
            msg = {
                "req_id": f"sub_{i}",
                "op": "subscribe",
                "args": batch,
            }
            await ws.send(json.dumps(msg))
            logger.debug("Sent subscription request: %s", batch)
            await asyncio.sleep(0.05)  # Small pause between subscription batches

    async def _receive_loop(
        self,
        ws: websockets.WebSocketClientProtocol,
    ) -> None:
        """
        Main receive loop.  Processes each incoming message from Bybit.
        Exits when the connection is closed or _running is set to False.
        """
        async for raw_message in ws:
            if not self._running:
                break
            try:
                await self._handle_message(raw_message)
            except Exception as exc:
                logger.exception(
                    "Error handling WebSocket message: %s\nRaw: %s",
                    exc, raw_message[:500] if isinstance(raw_message, str) else raw_message,
                )

    # ------------------------------------------------------------------
    # Message parsing
    # ------------------------------------------------------------------

    async def _handle_message(self, raw: str) -> None:
        """
        Parse a raw WebSocket message and dispatch to the callback if
        the message contains a closed candle.

        Parameters
        ----------
        raw : str
            Raw JSON text received from the WebSocket.
        """
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Received non-JSON message: %s", raw[:200])
            return

        # Handle subscription confirmation messages
        op = msg.get("op")
        if op == "subscribe":
            success = msg.get("success", False)
            ret_msg = msg.get("ret_msg", "")
            if success:
                logger.info("Subscription confirmed: %s", msg.get("req_id", ""))
            else:
                logger.error(
                    "Subscription failed: req_id=%s msg=%s",
                    msg.get("req_id", ""), ret_msg,
                )
            return

        # Handle pong messages (response to our ping)
        if op == "pong":
            logger.debug("Received pong from Bybit.")
            return

        # Handle kline data messages
        topic: str = msg.get("topic", "")
        if not topic.startswith("kline."):
            logger.debug("Ignoring non-kline message: topic=%s", topic)
            return

        meta = self._topic_meta.get(topic)
        if meta is None:
            logger.debug("Received data for unregistered topic: %s", topic)
            return

        symbol, ws_interval = meta
        data_list: list = msg.get("data", [])

        if not data_list:
            logger.debug("Empty data list for topic %s", topic)
            return

        # Bybit sends one entry per kline update
        candle_data: dict = data_list[0]

        # Only process closed candles
        if not candle_data.get("confirm", False):
            return  # Candle still forming – ignore

        ohlcv_row = self._parse_candle(candle_data, ws_interval)

        logger.debug(
            "Closed candle: symbol=%s open_time=%d close=%.8f",
            symbol, ohlcv_row["open_time"], ohlcv_row["close"],
        )

        await self._dispatch_callback(symbol, ohlcv_row)

    def _parse_candle(self, candle_data: dict, ws_interval: str) -> dict:
        """
        Convert a Bybit kline data dict into the canonical ohlcv_row format.

        Parameters
        ----------
        candle_data : dict
            Raw kline object from the WebSocket message.
        ws_interval : str
            Bybit WebSocket interval code (e.g. ``"60"``).

        Returns
        -------
        dict
            Keys: open_time, open, high, low, close, volume, close_time
        """
        start_ms   = int(candle_data["start"])
        interval_duration = INTERVAL_MS.get(ws_interval, 3_600_000)
        close_time = start_ms + interval_duration - 1

        return {
            "open_time":  start_ms,
            "open":       float(candle_data["open"]),
            "high":       float(candle_data["high"]),
            "low":        float(candle_data["low"]),
            "close":      float(candle_data["close"]),
            "volume":     float(candle_data["volume"]),
            "close_time": close_time,
        }

    async def _dispatch_callback(self, symbol: str, ohlcv_row: dict) -> None:
        """
        Invoke the registered callback with the closed candle data.

        Handles both regular functions and coroutine functions transparently.

        Parameters
        ----------
        symbol    : str   Bybit trading pair, e.g. ``"BTCUSDT"``
        ohlcv_row : dict  Canonical OHLCV row
        """
        if self._callback is None:
            return

        try:
            import asyncio as _asyncio
            import inspect as _inspect
            if _inspect.iscoroutinefunction(self._callback):
                await self._callback(symbol, ohlcv_row)
            else:
                self._callback(symbol, ohlcv_row)
        except Exception as exc:
            logger.exception(
                "Callback raised an exception for symbol=%s: %s", symbol, exc
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """True if the WebSocket connection is currently open."""
        return (
            self._ws is not None
            and not self._ws.closed
            and self._running
        )

    @property
    def subscribed_topics(self) -> List[str]:
        """Return a copy of the currently registered topics."""
        return list(self._topics)

    def __repr__(self) -> str:
        return (
            f"BybitWebSocketProvider("
            f"connected={self.is_connected}, "
            f"topics={len(self._topics)})"
        )
