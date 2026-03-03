"""
src/api/server.py
==================
AlgoTrader Pro – FastAPI application factory and lifecycle manager.

Responsibilities
----------------
* Create and configure the FastAPI application instance.
* Register all route modules with their URL prefixes.
* Handle application startup (DB init) and shutdown (engine teardown).
* Mount the static dashboard files.
* Expose the single WebSocket endpoint for real-time market data.
* Provide module-level globals (``engine_instance``, ``realtime_feed``) shared
  across the entire API layer.

Circular-import safety
----------------------
Routes import ``engine_instance`` and ``get_feed`` from this module.
This module imports the route *modules* (not their symbols) and registers
their routers; because Python evaluates ``import`` statements lazily at the
module level, and our route modules only reference ``engine_instance`` inside
function bodies (not at import time), there is no circular-import issue.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Internal imports — routes and utilities
# ---------------------------------------------------------------------------

# Routes are imported here so their ``router`` objects can be registered.
# The actual engine / feed globals must be defined BEFORE any route module
# tries to import them, which is satisfied by Python's module-load order:
# this file is fully parsed (all top-level assignments made) before the
# route modules' function bodies execute.
from src.api.routes import dashboard, portfolio, trades, signals, bot_control, backtests
from src.api.websocket.realtime_feed import RealtimeFeed
from src.utils.db import init_db

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared globals
# ---------------------------------------------------------------------------

# ``engine_instance`` is set by the application entry-point script
# (scripts/start_bot.py) after the TradingEngine is fully constructed.
# Route handlers and the lifespan function reference this name.
engine_instance: Optional[object] = None  # type: Optional[TradingEngine]

# Single WebSocket broadcast hub shared by all route handlers.
realtime_feed: RealtimeFeed = RealtimeFeed()

# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Startup
    -------
    1. Initialise the SQLite database (run schema.sql idempotently).
    2. Log a confirmation message.

    Shutdown
    --------
    1. If the trading engine is running, stop it gracefully so all open
       positions and pending orders are handled cleanly before the process
       exits.
    """
    # ----- Startup -----
    logger.info("AlgoTrader Pro API – startup sequence initiated.")

    try:
        init_db()
        logger.info("Database initialised successfully.")
    except Exception as exc:
        logger.error("Database initialisation failed: %s", exc, exc_info=True)
        raise

    # Avvia engine nello stesso event loop (WebSocket e broadcast funzionano)
    if engine_instance is not None:
        try:
            logger.info("Starting TradingEngine ...")
            await engine_instance.start()  # type: ignore[attr-defined]
            logger.info("TradingEngine started. Bot ATTIVO.")
        except Exception as exc:
            logger.error("TradingEngine start failed: %s", exc, exc_info=True)
            raise

    logger.info("API server started. Listening for requests.")

    yield

    # ----- Shutdown -----
    logger.info("AlgoTrader Pro API – shutdown sequence initiated.")

    if engine_instance is not None:
        try:
            logger.info("Stopping TradingEngine …")
            await engine_instance.stop()  # type: ignore[attr-defined]
            logger.info("TradingEngine stopped cleanly.")
        except Exception as exc:
            logger.error("Error stopping TradingEngine: %s", exc, exc_info=True)
    else:
        logger.info("No TradingEngine to stop.")

    logger.info("API server shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app: FastAPI = FastAPI(
    title="AlgoTrader Pro API",
    version="1.0.0",
    description=(
        "REST + WebSocket API for the AlgoTrader Pro algorithmic trading bot. "
        "Provides portfolio management, trade history, live signals, "
        "bot control, and backtesting endpoints."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # In production, restrict to your frontend origin.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

app.include_router(
    dashboard.router,
    prefix="/api/v1",
    tags=["dashboard"],
)

app.include_router(
    portfolio.router,
    prefix="/api/v1",
    tags=["portfolio"],
)

app.include_router(
    trades.router,
    prefix="/api/v1",
    tags=["trades"],
)

app.include_router(
    signals.router,
    prefix="/api/v1",
    tags=["signals"],
)

app.include_router(
    bot_control.router,
    prefix="/api/v1",
    tags=["bot"],
)

app.include_router(
    backtests.router,
    prefix="/api/v1",
    tags=["backtests"],
)

# ---------------------------------------------------------------------------
# Static file serving (built dashboard SPA)
# ---------------------------------------------------------------------------

# The dashboard directory is relative to the project root (where uvicorn is
# launched from).  StaticFiles will raise an error at startup if the directory
# does not exist, so we guard with a check.
_DASHBOARD_DIR = "dashboard"
if os.path.isdir(_DASHBOARD_DIR):
    # Route esplicita PRIMA del mount: index con no-cache (evita STOPPED stale)
    _index_path = Path(_DASHBOARD_DIR) / "index.html"
    if _index_path.exists():

        @app.get("/dashboard/index.html", include_in_schema=False)
        async def _serve_dashboard_index() -> FileResponse:
            return FileResponse(
                _index_path,
                media_type="text/html",
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )

    app.mount(
        "/dashboard",
        StaticFiles(directory=_DASHBOARD_DIR, html=True),
        name="dashboard",
    )
    logger.debug("Static dashboard mounted from '%s'.", _DASHBOARD_DIR)
else:
    logger.warning(
        "Dashboard directory '%s' not found – static files not mounted. "
        "Run 'npm run build' inside the dashboard/ folder to generate assets.",
        _DASHBOARD_DIR,
    )

# ---------------------------------------------------------------------------
# Core HTTP endpoints
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Redirect browser visits to the root URL to the dashboard index page."""
    import time
    return RedirectResponse(url=f"/dashboard/index.html?t={int(time.time())}")


@app.get(
    "/health",
    tags=["system"],
    summary="Health check",
    response_description="Service health and version information.",
)
async def health_check() -> JSONResponse:
    """
    Lightweight health probe endpoint.

    Returns the service status, API version, and current trading mode.
    Used by load-balancers, Docker HEALTHCHECK instructions, and monitoring
    dashboards (e.g. Grafana).
    """
    mode = "paper"
    if engine_instance is not None:
        try:
            status = engine_instance.get_status()  # type: ignore[attr-defined]
            mode = status.get("mode", "paper")
        except Exception:
            pass

    return JSONResponse(
        content={
            "status": "ok",
            "version": "1.0.0",
            "mode": mode,
        }
    )


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket) -> None:
    """
    Real-time event stream via WebSocket.

    Connect to ``ws://<host>/ws/realtime`` to receive a stream of JSON
    messages.  Each message has the envelope::

        {"event": "<type>", "data": {...}, "timestamp": "..."}

    Event types: ``connected``, ``ping``, ``equity_update``, ``trade_opened``,
    ``trade_closed``, ``signal``, ``market_tick``.
    """
    await realtime_feed.websocket_endpoint(websocket)


# ---------------------------------------------------------------------------
# Engine injection helpers
# ---------------------------------------------------------------------------


def set_engine(eng: object) -> None:
    """
    Inject the TradingEngine instance into this module's global scope.

    This is called by ``scripts/start_bot.py`` after constructing the
    ``TradingEngine`` so that route handlers can access it without having
    to import the engine directly (which would create a complex dependency
    graph).

    Args:
        eng: An initialised ``TradingEngine`` instance (or any object that
             exposes ``start()``, ``stop()``, and ``get_status()`` methods).
    """
    global engine_instance
    engine_instance = eng
    logger.info(
        "TradingEngine injected into API server. type=%s",
        type(eng).__name__,
    )


def get_feed() -> RealtimeFeed:
    """
    Return the module-level :class:`RealtimeFeed` instance.

    Route handlers and background tasks call this to broadcast live events
    to connected WebSocket clients::

        from src.api.server import get_feed

        feed = get_feed()
        await feed.broadcast_trade_opened(trade.to_dict())

    Returns:
        The singleton :class:`RealtimeFeed` instance.
    """
    return realtime_feed
