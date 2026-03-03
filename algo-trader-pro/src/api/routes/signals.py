"""
src/api/routes/signals.py
==========================
Signal evaluation endpoints.

GET /api/v1/signals/live
    Returns the most-recent signal evaluation for each tracked symbol.
    Uses live engine data if available, otherwise falls back to the DB.

GET /api/v1/signals/history
    Returns a paginated log of all past signal evaluations, with optional
    symbol filter.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

import src.api.server as _server_module  # late import – avoids circular dep

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _get_db_path() -> str:
    env_url = os.getenv("DATABASE_URL", "")
    if env_url.startswith("sqlite:///"):
        return env_url.replace("sqlite:///", "", 1)
    if env_url.startswith("sqlite://./"):
        return env_url.replace("sqlite://./", "", 1)
    base = os.path.dirname(__file__)
    return os.path.normpath(
        os.path.join(base, "..", "..", "..", "database", "algotrader.db")
    )


def _safe_connect() -> Optional[sqlite3.Connection]:
    try:
        db_path = _get_db_path()
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as exc:
        logger.warning("Cannot open DB: %s", exc)
        return None


def _signal_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a signals_log DB row to a clean dict."""
    d: Dict[str, Any] = {}
    for key in row.keys():
        val = row[key]
        if key == "raw_signals" and val is not None:
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                val = None
        d[key] = val
    return d


# ---------------------------------------------------------------------------
# GET /signals/live
# ---------------------------------------------------------------------------


@router.get(
    "/signals/live",
    summary="Most-recent signal evaluation per symbol.",
    tags=["signals"],
)
async def get_live_signals() -> JSONResponse:
    """
    Return the latest signal evaluation for each tracked symbol.

    Data priority
    -------------
    1. Engine's in-memory last-evaluated signals (if the engine is running).
    2. Most recent row per symbol in ``signals_log`` (DB fallback).

    Response fields (per symbol)
    ----------------------------
    symbol, confidence_score, direction, layer1_score, layer2_score,
    layer3_score, ml_score, action_taken, timestamp.
    """
    engine = _server_module.engine_instance
    live_signals: List[Dict[str, Any]] = []

    # ---- 1. Try live engine ----
    if engine is not None:
        try:
            # TradingEngine may expose last_signals as a dict: symbol -> signal_dict
            last_signals = getattr(engine, "last_signals", None)
            if last_signals and isinstance(last_signals, dict):
                for symbol, sig in last_signals.items():
                    live_signals.append(
                        {
                            "symbol": symbol,
                            "confidence_score": float(sig.get("confidence_score", 0)),
                            "direction": sig.get("direction", "neutral"),
                            "layer1_score": float(sig.get("layer1_score", 0)),
                            "layer2_score": float(sig.get("layer2_score", 0)),
                            "layer3_score": float(sig.get("layer3_score", 0)),
                            "ml_score": float(sig.get("ml_score", 0)),
                            "action_taken": sig.get("action_taken", "skipped_threshold"),
                            "timestamp": sig.get(
                                "timestamp", datetime.now(tz=timezone.utc).isoformat()
                            ),
                        }
                    )
        except Exception as exc:
            logger.warning("Could not read live signals from engine: %s", exc)

    if live_signals:
        return JSONResponse(content=live_signals)

    # ---- 2. DB fallback: latest row per symbol ----
    conn = _safe_connect()
    if conn is None:
        return JSONResponse(content=[])

    try:
        # Use a window function approach: for each symbol get the row with
        # the highest (latest) timestamp.
        rows = conn.execute(
            """
            SELECT
                s.symbol,
                s.confidence_score,
                s.direction,
                s.layer1_score,
                s.layer2_score,
                s.layer3_score,
                s.ml_score,
                s.action_taken,
                s.timestamp
            FROM signals_log s
            INNER JOIN (
                SELECT symbol, MAX(timestamp) AS max_ts
                FROM signals_log
                GROUP BY symbol
            ) latest
            ON s.symbol = latest.symbol AND s.timestamp = latest.max_ts
            ORDER BY s.timestamp DESC
            """
        ).fetchall()

        live_signals = [
            {
                "symbol": r["symbol"],
                "confidence_score": round(float(r["confidence_score"]), 2),
                "direction": r["direction"],
                "layer1_score": round(float(r["layer1_score"]), 2),
                "layer2_score": round(float(r["layer2_score"]), 2),
                "layer3_score": round(float(r["layer3_score"]), 2),
                "ml_score": round(float(r["ml_score"]), 2),
                "action_taken": r["action_taken"],
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error("signals/live DB error: %s", exc, exc_info=True)
    finally:
        conn.close()

    return JSONResponse(content=live_signals)


# ---------------------------------------------------------------------------
# GET /signals/history
# ---------------------------------------------------------------------------


@router.get(
    "/signals/history",
    summary="Historical log of all signal evaluations.",
    tags=["signals"],
)
async def get_signal_history(
    symbol: Optional[str] = Query(
        default=None,
        description="Filter by trading pair, e.g. 'BTCUSDT'. Returns all symbols if omitted.",
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of records to return (max 500).",
    ),
) -> JSONResponse:
    """
    Return the most recent signal evaluations from ``signals_log``.

    Results are ordered by timestamp descending (newest first).
    Use the ``symbol`` query parameter to narrow results to a single pair.

    Response fields (per entry)
    ---------------------------
    id, timestamp, symbol, timeframe, confidence_score, direction,
    layer1_score, layer2_score, layer3_score, ml_score,
    raw_signals (parsed JSON), action_taken, trade_id.
    """
    conn = _safe_connect()
    if conn is None:
        return JSONResponse(content=[])

    try:
        conditions: List[str] = []
        params: List[Any] = []

        if symbol is not None:
            conditions.append("symbol = ?")
            params.append(symbol.upper())

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        rows = conn.execute(
            f"""
            SELECT
                id, timestamp, symbol, timeframe,
                confidence_score, direction,
                layer1_score, layer2_score, layer3_score, ml_score,
                raw_signals, action_taken, trade_id
            FROM signals_log
            {where}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()

        result = [_signal_row_to_dict(r) for r in rows]

    except Exception as exc:
        logger.error("signals/history DB error: %s", exc, exc_info=True)
        result = []
    finally:
        conn.close()

    return JSONResponse(content=result)
