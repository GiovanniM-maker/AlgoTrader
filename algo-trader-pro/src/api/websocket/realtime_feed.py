"""
src/api/websocket/realtime_feed.py
====================================
WebSocket real-time broadcast hub for AlgoTrader Pro.

All live-trading and market data events are pushed through this class to every
connected browser client.  The server.py module creates a single module-level
instance (``realtime_feed``) which is shared across all route handlers via the
``get_feed()`` helper.

Protocol
--------
Every message sent to clients is a JSON object with the following envelope::

    {
        "event":     "<event_type>",   # string constant, e.g. "trade_opened"
        "data":      { ... },          # payload dict, event-specific
        "timestamp": "2024-01-15T..."  # ISO-8601 UTC timestamp added by broadcast()
    }

Event types
-----------
connected         - Sent immediately after handshake to confirm connection.
ping              - Keep-alive heartbeat every 30 seconds.
equity_update     - Periodic portfolio equity snapshot.
trade_opened      - New position opened.
trade_closed      - Position closed (stop, TP, manual, etc.).
signal            - Signal evaluation result (whether acted on or not).
market_tick       - Latest price tick for a tracked symbol.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Set

from fastapi import WebSocket, WebSocketDisconnect


class RealtimeFeed:
    """
    Hub that holds all active WebSocket connections and broadcasts events.

    Thread-safety note
    ------------------
    All ``broadcast*`` methods are coroutines and must be awaited from an
    async context.  The underlying ``self.connections`` set is not
    thread-safe; callers must call broadcast methods from the same event loop
    that the FastAPI application is running on (which is the normal case for
    ``asyncio``-based code).

    Usage example (from a route or background task)::

        from src.api.server import get_feed

        feed = get_feed()
        await feed.broadcast_trade_opened(trade_dict)
    """

    def __init__(self) -> None:
        self.connections: Set[WebSocket] = set()
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # WebSocket lifecycle
    # ------------------------------------------------------------------

    async def websocket_endpoint(self, websocket: WebSocket) -> None:
        """
        Accept a new WebSocket connection, register it, and maintain it.

        This coroutine is the direct handler for the ``/ws/realtime``
        endpoint in ``server.py``.  It blocks (via the keep-alive loop)
        for the lifetime of the connection.

        Args:
            websocket: The incoming ``WebSocket`` instance provided by FastAPI.
        """
        await websocket.accept()
        self.connections.add(websocket)
        client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
        self.logger.info(
            "WS client connected. client=%s total_connections=%d",
            client_info,
            len(self.connections),
        )

        try:
            # Send initial handshake confirmation.
            await websocket.send_json(
                {
                    "event": "connected",
                    "data": {
                        "message": "AlgoTrader Pro connected",
                        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    },
                }
            )

            # Keep-alive loop: ping every 30 seconds so the browser does not
            # close an idle connection, and to detect dropped clients.
            while True:
                await asyncio.sleep(30)
                await websocket.send_json(
                    {
                        "event": "ping",
                        "data": {
                            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                        },
                    }
                )

        except WebSocketDisconnect:
            self.connections.discard(websocket)
            self.logger.info(
                "WS client disconnected. client=%s total_connections=%d",
                client_info,
                len(self.connections),
            )
        except Exception as exc:
            # Catch any unexpected error (e.g. broken pipe, cancelled task)
            # and clean up gracefully so we do not leave a dead socket in the set.
            self.connections.discard(websocket)
            self.logger.warning(
                "WS connection error for client=%s: %s",
                client_info,
                exc,
            )

    # ------------------------------------------------------------------
    # Core broadcast primitive
    # ------------------------------------------------------------------

    async def broadcast(self, event_type: str, data: dict) -> None:
        """
        Send a JSON message to every connected WebSocket client.

        Clients that have disconnected without triggering a clean
        ``WebSocketDisconnect`` are detected here (their ``send_text`` will
        raise an exception) and silently removed from the connection set.

        Args:
            event_type: Logical event name (e.g. ``"trade_opened"``).
            data:       Payload dictionary serialised into the ``"data"`` key.
        """
        message = json.dumps(
            {
                "event": event_type,
                "data": data,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            },
            default=str,  # Serialize datetime/Decimal objects safely
        )

        dead_connections: Set[WebSocket] = set()

        for ws in list(self.connections):  # iterate a snapshot to avoid set-size-change errors
            try:
                await ws.send_text(message)
            except Exception as exc:
                self.logger.debug(
                    "Removing dead WS connection during broadcast of '%s': %s",
                    event_type,
                    exc,
                )
                dead_connections.add(ws)

        self.connections -= dead_connections

        if dead_connections:
            self.logger.info(
                "Cleaned up %d dead WS connection(s). Remaining: %d",
                len(dead_connections),
                len(self.connections),
            )

    # ------------------------------------------------------------------
    # Typed broadcast helpers
    # ------------------------------------------------------------------

    async def broadcast_equity_update(self, snapshot: dict) -> None:
        """
        Broadcast a portfolio equity snapshot.

        Args:
            snapshot: Dict with at minimum ``equity``, ``cash``,
                      ``drawdown_pct``, and ``timestamp`` keys.
                      Produced by ``PaperExecutor`` or the engine loop.
        """
        await self.broadcast("equity_update", snapshot)

    async def broadcast_trade_opened(self, trade_dict: dict) -> None:
        """
        Broadcast a newly opened trade to all connected clients.

        Args:
            trade_dict: Full trade record dict (same shape as the ``trades``
                        DB row, typically from ``Trade.to_dict()``).
        """
        await self.broadcast("trade_opened", trade_dict)

    async def broadcast_trade_closed(self, trade_dict: dict) -> None:
        """
        Broadcast a closed trade event.

        Args:
            trade_dict: Full trade record dict including ``exit_price``,
                        ``net_pnl``, and ``exit_reason``.
        """
        await self.broadcast("trade_closed", trade_dict)

    async def broadcast_signal(self, signal_dict: dict) -> None:
        """
        Broadcast the result of a signal evaluation cycle.

        Args:
            signal_dict: Dict containing ``symbol``, ``confidence_score``,
                         ``direction``, layer scores, ``action_taken``, etc.
        """
        await self.broadcast("signal", signal_dict)

    async def broadcast_market_tick(self, tick_data: dict) -> None:
        """
        Broadcast a real-time market price tick.

        Args:
            tick_data: Dict with at minimum ``symbol`` and ``price`` keys.
                       May also include ``change_24h_pct``, ``volume_24h``.
        """
        await self.broadcast("market_tick", tick_data)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def connection_count(self) -> int:
        """Return the current number of active WebSocket connections."""
        return len(self.connections)

    def __repr__(self) -> str:
        return f"<RealtimeFeed connections={self.connection_count}>"
