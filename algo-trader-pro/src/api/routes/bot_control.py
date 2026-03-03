"""
src/api/routes/bot_control.py
==============================
Bot lifecycle control endpoints.

POST /api/v1/bot/start    – Start the trading engine (if initialised).
POST /api/v1/bot/pause    – Pause new trade entries (existing positions kept).
POST /api/v1/bot/stop     – Stop the trading engine gracefully.
GET  /api/v1/bot/status   – Return current engine status.

Design notes
------------
The TradingEngine is NOT created by this module.  It is constructed in
``scripts/start_bot.py`` and injected via ``server.set_engine()``.
These endpoints merely control the already-injected instance.

Concurrency
-----------
``/bot/start`` awaits ``engine.start()`` directly inside the route handler
(FastAPI runs async routes on the event loop, so awaiting is safe).  This
means the HTTP response is sent only after ``start()`` completes – which is
intentional, as the caller gets confirmation that the engine is running.

For long-running operations in the future, use ``BackgroundTasks`` to
decouple the HTTP response from the work.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

import src.api.server as _server_module  # late import – avoids circular dep

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_start_time: Optional[datetime] = None   # set when /bot/start succeeds
_bot_task: Optional[asyncio.Task] = None  # optional background task handle

# ---------------------------------------------------------------------------
# POST /bot/start
# ---------------------------------------------------------------------------


@router.post(
    "/bot/start",
    summary="Start the trading engine.",
    tags=["bot"],
)
async def start_bot() -> JSONResponse:
    """
    Start the TradingEngine.

    The engine must have been injected via ``server.set_engine()`` before
    calling this endpoint (which is done by ``scripts/start_bot.py``).

    Returns
    -------
    - ``{"status": "error", "message": "Engine not initialized ..."}``
      if the engine has not been injected yet.
    - ``{"status": "already_running"}`` if the engine reports it is already
      in RUNNING state.
    - ``{"status": "started", "message": "Bot started successfully"}``
      on success.
    """
    global _start_time

    engine = _server_module.engine_instance

    if engine is None:
        logger.warning("POST /bot/start called but no engine is injected.")
        return JSONResponse(
            status_code=409,
            content={
                "status": "error",
                "message": (
                    "Engine not initialized. "
                    "Start via scripts/start_bot.py which constructs the engine "
                    "and injects it before the API server starts accepting requests."
                ),
            },
        )

    # Check current status to avoid double-starting.
    try:
        current_status = engine.get_status()  # type: ignore[attr-defined]
        engine_state: str = current_status.get("status", "STOPPED")
    except Exception as exc:
        logger.error("Could not read engine status: %s", exc)
        engine_state = "UNKNOWN"

    if engine_state == "RUNNING":
        logger.info("POST /bot/start: engine is already running.")
        return JSONResponse(
            content={
                "status": "already_running",
                "message": "Engine is already in RUNNING state.",
            }
        )

    # Start the engine.
    try:
        await engine.start()  # type: ignore[attr-defined]
        _start_time = datetime.now(tz=timezone.utc)
        logger.info("TradingEngine started via API at %s.", _start_time.isoformat())
    except Exception as exc:
        logger.error("Engine start failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start engine: {exc}",
        )

    return JSONResponse(
        content={
            "status": "started",
            "message": "Bot started successfully.",
            "started_at": _start_time.isoformat(),
        }
    )


# ---------------------------------------------------------------------------
# POST /bot/pause
# ---------------------------------------------------------------------------


@router.post(
    "/bot/pause",
    summary="Pause the trading engine (stops new entries, keeps positions open).",
    tags=["bot"],
)
async def pause_bot() -> JSONResponse:
    """
    Pause the TradingEngine.

    In PAUSED state the engine continues monitoring open positions
    (stop-loss and take-profit checks remain active) but will not open
    any new trades until resumed.

    The pause is implemented by setting ``engine._state`` to ``"PAUSED"``
    (a convention used by TradingEngine's ``get_status()``).  If the engine
    exposes a dedicated ``pause()`` method it is called instead.

    Returns
    -------
    - ``{"status": "error", "message": ...}`` if engine is not initialised.
    - ``{"status": "paused"}`` on success.
    """
    engine = _server_module.engine_instance

    if engine is None:
        return JSONResponse(
            status_code=409,
            content={
                "status": "error",
                "message": "Engine not initialized. Cannot pause a non-existent engine.",
            },
        )

    try:
        # If the engine has a dedicated pause() method, prefer it.
        if hasattr(engine, "pause") and callable(engine.pause):  # type: ignore[attr-defined]
            result = engine.pause()  # type: ignore[attr-defined]
            if asyncio.iscoroutine(result):
                await result
        elif hasattr(engine, "_state"):
            # Direct attribute mutation as fallback.
            engine._state = "PAUSED"  # type: ignore[attr-defined]
            logger.info("TradingEngine paused via direct state mutation.")
        else:
            logger.warning("Engine has no pause() method and no _state attribute. Pause is a no-op.")

    except Exception as exc:
        logger.error("Error pausing engine: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to pause engine: {exc}",
        )

    return JSONResponse(
        content={
            "status": "paused",
            "message": "Engine paused. Open positions are still monitored.",
            "paused_at": datetime.now(tz=timezone.utc).isoformat(),
        }
    )


# ---------------------------------------------------------------------------
# POST /bot/stop
# ---------------------------------------------------------------------------


@router.post(
    "/bot/stop",
    summary="Stop the trading engine gracefully.",
    tags=["bot"],
)
async def stop_bot() -> JSONResponse:
    """
    Stop the TradingEngine.

    Calls ``engine.stop()``, which:
    - Cancels the REST-polling or WebSocket data loops.
    - Stops the APScheduler sentiment jobs.
    - Logs a shutdown event to ``bot_events``.

    Returns
    -------
    - ``{"status": "not_running"}`` if the engine is already stopped.
    - ``{"status": "stopped"}`` on success.
    """
    engine = _server_module.engine_instance

    if engine is None:
        return JSONResponse(
            content={
                "status": "not_running",
                "message": "Engine is not initialized — nothing to stop.",
            }
        )

    try:
        current_status = engine.get_status()  # type: ignore[attr-defined]
        engine_state = current_status.get("status", "STOPPED")
    except Exception:
        engine_state = "UNKNOWN"

    if engine_state == "STOPPED":
        return JSONResponse(
            content={
                "status": "not_running",
                "message": "Engine is already in STOPPED state.",
            }
        )

    try:
        await engine.stop()  # type: ignore[attr-defined]
        logger.info("TradingEngine stopped via API.")
    except Exception as exc:
        logger.error("Error stopping engine: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop engine: {exc}",
        )

    return JSONResponse(
        content={
            "status": "stopped",
            "message": "Engine stopped successfully.",
            "stopped_at": datetime.now(tz=timezone.utc).isoformat(),
        }
    )


# ---------------------------------------------------------------------------
# GET /bot/status
# ---------------------------------------------------------------------------


@router.get(
    "/bot/status",
    summary="Current trading engine status and session statistics.",
    tags=["bot"],
)
async def get_bot_status() -> JSONResponse:
    """
    Return the current operational status of the TradingEngine.

    Response fields
    ---------------
    status        : str   – 'RUNNING', 'PAUSED', 'STOPPED', or 'ERROR'.
    started_at    : str | null – ISO-8601 UTC start time.
    uptime_seconds: float | null – Seconds the engine has been running.
    symbols       : list[str] – Symbols currently being traded.
    mode          : str   – 'paper' or 'live'.
    total_trades  : int   – Total positions opened in this session.
    current_prices: dict  – Latest cached price per symbol.
    message       : str   – Human-readable description (only when stopped).
    """
    engine = _server_module.engine_instance

    if engine is None:
        return JSONResponse(
            content={
                "status": "STOPPED",
                "started_at": None,
                "uptime_seconds": None,
                "symbols": [],
                "mode": "paper",
                "total_trades": 0,
                "message": "Engine not initialized. Start via scripts/start_bot.py.",
            }
        )

    try:
        status_dict: Dict[str, Any] = engine.get_status()  # type: ignore[attr-defined]
    except Exception as exc:
        logger.error("Error fetching engine status: %s", exc, exc_info=True)
        return JSONResponse(
            content={
                "status": "ERROR",
                "message": f"Could not retrieve engine status: {exc}",
            }
        )

    # Compute uptime if started_at is present in the status dict.
    started_at_str: Optional[str] = status_dict.get("started_at")
    uptime_seconds: Optional[float] = None
    if started_at_str:
        try:
            # Handles both timezone-aware and naive ISO strings.
            started_at_dt = datetime.fromisoformat(started_at_str)
            if started_at_dt.tzinfo is None:
                started_at_dt = started_at_dt.replace(tzinfo=timezone.utc)
            uptime_seconds = (datetime.now(tz=timezone.utc) - started_at_dt).total_seconds()
        except Exception:
            pass

    return JSONResponse(
        content={
            "status": status_dict.get("status", "UNKNOWN"),
            "started_at": started_at_str,
            "uptime_seconds": round(uptime_seconds, 1) if uptime_seconds is not None else None,
            "symbols": status_dict.get("symbols", []),
            "mode": status_dict.get("mode", "paper"),
            "total_trades": status_dict.get("total_trades", 0),
            "current_prices": status_dict.get("current_prices", {}),
            "open_positions": status_dict.get("open_positions", 0),
        }
    )
